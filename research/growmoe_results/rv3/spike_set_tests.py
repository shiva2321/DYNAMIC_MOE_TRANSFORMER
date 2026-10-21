"""THROWAWAY spike on DEV seeds (100-102): which SET-level novelty statistic is calibrated AND powerful?   (gatekeeper research, entry E-023)

Why: at centre cosine >= 0.6 the per-sample novelty test is blind (rejects 0.8 % of foreign inputs at 0.85) and my first set-level attempt (Fisher combination of the
per-sample p-values, threshold 1e-4) over-fired inside the live pool (72-131 experts for 10 regimes). A set-level test must (a) hold its false-alarm rate on in-regime
batches and (b) detect foreign batches. Four statistics, each against the expert's own stored reference (its <= 256 prototypes):

  fisher_owner  -2 sum ln p_i of the OWNER expert's conformal p-values, chi-square null (the naive one)
  fisher_pool   the same on the pool's max-over-experts p-values (what the first attempt used; live pool only)
  meandist      ||mean(batch) - mean(reference)||, exact permutation null over the pooled points
  mmd           squared MMD with an RBF kernel (median heuristic), exact permutation null
  scoresum      mean KDE score of the batch against b-sums bootstrapped from the expert's own leave-one-out calibration scores

A1 (clean): per regime a signature on its first-visit data; in-regime batches vs foreign batches (calibration and power, no pool involved).
A2 (live shadow): the real pool is trained as usual; before every batch the statistics are computed against the batch's owner expert and only LOGGED (the pool is not changed).
Labels in A2: onset = first batch of a first-visit regime; known = batch index >= 3 of a first-visit segment, or any batch of a revisit.

    python research/growmoe_results/rv3/spike_set_tests.py
"""

import json
import os
import sys
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402

from growmoe.config import PhyloCfg                                                     # noqa: E402
from growmoe.conformal import benjamini_hochberg                                        # noqa: E402
from growmoe.pools.phylo import PhyloMoE                                                # noqa: E402
from growmoe.signature import VmfSignature                                              # noqa: E402
from growmoe.streams_m1 import function_regimes                                         # noqa: E402

torch.set_num_threads(3)
RHOS, SEEDS, N_PERM, BS = (0.0, 0.3, 0.6, 0.85), (100, 101, 102), 999, 64
STATS = ("fisher_owner", "meandist", "mmd", "scoresum")
ALPHAS = (0.05, 0.01, 0.001)


# ------------------------------------------------------------------------------------------------ the statistics
def fisher_p(pvals: torch.Tensor) -> float:
    stat = -2.0 * torch.log(pvals.double().clamp_min(1e-12)).sum()
    return float(torch.special.gammaincc(torch.tensor(float(len(pvals)), dtype=torch.float64), stat / 2.0))


def _perm_idx(P: int, b: int, g) -> torch.Tensor:
    return torch.rand(N_PERM, P, generator=g).argsort(1)[:, :b]


def meandist_p(ref: torch.Tensor, batch: torch.Tensor, g) -> float:
    Z = torch.cat([ref, batch])
    P, b = len(Z), len(batch)
    n = P - b
    obs = (batch.mean(0) - ref.mean(0)).norm()
    bm = Z[_perm_idx(P, b, g)].mean(1)
    rm = (Z.sum(0) - bm * b) / n
    T = (bm - rm).norm(dim=1)
    return (1 + int((T >= obs).sum())) / (1 + N_PERM)


def mmd_p(ref: torch.Tensor, batch: torch.Tensor, g) -> float:
    Z = torch.cat([ref, batch])
    P, b = len(Z), len(batch)
    n = P - b
    d2 = (2.0 - 2.0 * Z @ Z.T).clamp_min(0.0)
    K = torch.exp(-d2 / d2[d2 > 0].median())
    perm = _perm_idx(P, b, g)
    W = torch.zeros(P, N_PERM + 1)
    W[n:, 0] = 1.0                                                                      # column 0 = the real split (batch = last b rows)
    W[:, 1:].scatter_(0, perm.T, 1.0)
    s = K.sum(1)
    KW = K @ W
    BB = (W * KW).sum(0)
    ws = (W * s[:, None]).sum(0)
    BR = ws - BB
    RR = float(s.sum()) - 2.0 * ws + BB
    mmd = BB / b ** 2 - 2.0 * BR / (b * n) + RR / n ** 2
    return (1 + int((mmd[1:] >= mmd[0]).sum())) / (1 + N_PERM)


def scoresum_p(sig: VmfSignature, batch: torch.Tensor, g) -> float:
    cal = sig.cal_scores
    obs = sig.score(batch).mean()
    null = cal[torch.randint(0, len(cal), (N_PERM, len(batch)), generator=g)].mean(1)
    return (1 + int((null <= obs).sum())) / (1 + N_PERM)


def all_stats(sig: VmfSignature, batch: torch.Tensor, g) -> dict:
    ref = sig.protos
    return {"fisher_owner": fisher_p(sig.pvalue(batch)), "meandist": meandist_p(ref, batch, g), "mmd": mmd_p(ref, batch, g), "scoresum": scoresum_p(sig, batch, g),
            "persample_flagged": float(benjamini_hochberg(sig.pvalue(batch), 0.02).float().mean())}


def auroc(neg: list, pos: list, higher_is_pos: bool = False) -> float:
    """P(score of a positive (novel) batch is more extreme than that of a negative one); p-values: smaller = more novel."""
    a, b = np.asarray(neg, float), np.asarray(pos, float)
    diff = (a[:, None] - b[None, :]) if not higher_is_pos else (b[:, None] - a[None, :])
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


# ------------------------------------------------------------------------------------------------ A1: clean calibration and power
def run_a1(rho: float, seed: int) -> dict:
    st = function_regimes(seed, n_regimes=10, overlap=rho)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    sigs = {}
    for r in range(10):
        seg = next(s for s in st.segments if s.regime == r and s.first_visit)
        sig = VmfSignature(st.d, 256, seed=r)
        sig.update(seg.x)
        sig.fit()
        sigs[r] = sig
    own, foreign = {k: [] for k in STATS + ("persample_flagged",)}, {k: [] for k in STATS + ("persample_flagged",)}
    for r, sig in sigs.items():
        xs = st.regimes[r].x_te
        for lo in range(0, len(xs) - BS + 1, BS):
            for k, v in all_stats(sig, xs[lo:lo + BS], g).items():
                own[k].append(v)
        for r2 in rng.choice([q for q in range(10) if q != r], size=3, replace=False):
            xs2 = st.regimes[int(r2)].x_te
            for lo in range(0, len(xs2) - BS + 1, BS):
                for k, v in all_stats(sig, xs2[lo:lo + BS], g).items():
                    foreign[k].append(v)
    out = {"rho": rho, "seed": seed, "n_own": len(own["mmd"]), "n_foreign": len(foreign["mmd"])}
    for k in STATS:
        out[k] = {"fa": {str(a): float(np.mean(np.asarray(own[k]) <= a)) for a in ALPHAS}, "det": {str(a): float(np.mean(np.asarray(foreign[k]) <= a)) for a in ALPHAS},
                  "auroc": auroc(own[k], foreign[k])}
    out["persample_flagged"] = {"own": float(np.mean(own["persample_flagged"])), "foreign": float(np.mean(foreign["persample_flagged"])),
                                "auroc": auroc(own["persample_flagged"], foreign["persample_flagged"], higher_is_pos=True)}
    return out


# ------------------------------------------------------------------------------------------------ A2: live shadow
class ShadowPool(PhyloMoE):
    """The ordinary pool; before each batch it LOGS the set-level statistics against the batch's owner expert. It changes nothing."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.shadow_rows, self._g = [], torch.Generator().manual_seed(0)
        self.label, self.meta = None, (None, None, None)

    def observe(self, x, y=None, ctx=None):
        if self.experts and self.label is not None:
            xn = self._normalize(x.float())
            eids = list(self.experts)
            P, LogF = self._gate_matrices(xn, eids)
            owner = eids[int(LogF.sum(0).argmax())]
            sig = self.experts[owner].meta["sig"]
            if sig.n >= 16:
                row = all_stats(sig, xn, self._g)
                pn = P.max(1).values
                row["fisher_pool"] = fisher_p(pn)
                row["persample_pool_flagged"] = float(benjamini_hochberg(pn, self.cfg.q).float().mean())
                rec = self.experts[owner]
                row.update(label=self.label, n_experts=len(eids), owner_n=sig.n, owner=owner, owner_state=rec.state, owner_age=int(self.step - rec.birth_step),
                           mean_p_owner=float(sig.pvalue(xn).mean()), regime=self.meta[0], within=self.meta[1], seg=self.meta[2])
                self.shadow_rows.append(row)
        return super().observe(x, y, ctx)


class ShadowPoolRecent(ShadowPool):
    """Also tests each batch against the owner's RECENT PAST (the last five batches it trained on) instead of its stored reservoir: a change detector."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.recent = {}

    def observe(self, x, y=None, ctx=None):
        extra, n0 = None, len(self.shadow_rows)
        if self.experts and self.label is not None:
            xn = self._normalize(x.float())
            eids = list(self.experts)
            _, LogF = self._gate_matrices(xn, eids)
            owner = eids[int(LogF.sum(0).argmax())]
            if owner in self.recent and sum(len(t) for t in self.recent[owner]) >= 48:
                ref = torch.cat(list(self.recent[owner]))
                extra = {"meandist_recent": meandist_p(ref, xn, self._g), "mmd_recent": mmd_p(ref, xn, self._g), "recent_n": len(ref)}
        out = super().observe(x, y, ctx)
        if extra is not None and len(self.shadow_rows) > n0:
            self.shadow_rows[-1].update(extra)
        xnorm = self._normalize(x.float())
        for eid in torch.unique(out.assigned).tolist():
            if eid >= 0:
                self.recent.setdefault(eid, deque(maxlen=5)).append(xnorm[out.assigned == eid])
        return out


POOL_CLS = ShadowPool


def summarize_recent(a2: list) -> None:
    print("\n=== A2 live shadow, reference = the owner's RECENT PAST (rows where the owner has >= 48 recent samples) ===")
    print("rho  statistic        false alarm @0.05 / @0.01 / @0.001     detection @0.05 / @0.01 / @0.001     AUROC     n onset / known")
    for rho in RHOS:
        rows = [r for a in a2 if a["rho"] == rho for r in a["rows"] if "meandist_recent" in r]
        for k in ("meandist_recent", "mmd_recent"):
            on = [r[k] for r in rows if r["label"] == "onset"]
            kn = [r[k] for r in rows if r["label"] == "known"]
            if on and kn:
                fa = [np.mean(np.asarray(kn) <= a) for a in ALPHAS]
                de = [np.mean(np.asarray(on) <= a) for a in ALPHAS]
                print(f"{rho:4.2f} {k:16s} {fa[0]:8.3f} {fa[1]:8.3f} {fa[2]:8.3f}          {de[0]:8.3f} {de[1]:8.3f} {de[2]:8.3f}          {auroc(kn, on):6.3f}    {len(on)} / {len(kn)}")


def run_a2(rho: float, seed: int) -> dict:
    st = function_regimes(seed, n_regimes=10, overlap=rho)
    pool = POOL_CLS(PhyloCfg(router=False, spawn_rule="probe", growth_rule="probe", seed=seed, d=st.d, n_classes=st.n_classes, inner_steps=5, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf"))
    walk = list(st._walk(BS, seed))
    last_seg, within = None, 0
    for (k, seg, _), (x, y, ctx) in zip(walk, st.algo_batches(BS, seed)):
        within = within + 1 if k == last_seg else 0
        last_seg = k
        if seg.first_visit and within == 0 and k > 0:
            pool.label = "onset"
        elif (seg.first_visit and within >= 3) or not seg.first_visit:
            pool.label = "known"
        else:
            pool.label = None
        pool.meta = (seg.regime, within, k)
        pool.observe(x, y, ctx)
    rows = pool.shadow_rows
    out = {"rows": rows, "rho": rho, "seed": seed, "n_onset": sum(r["label"] == "onset" for r in rows), "n_known": sum(r["label"] == "known" for r in rows), "experts_end": len(pool.experts)}
    for k in STATS + ("fisher_pool",):
        on = [r[k] for r in rows if r["label"] == "onset"]
        kn = [r[k] for r in rows if r["label"] == "known"]
        out[k] = {"fa": {str(a): float(np.mean(np.asarray(kn) <= a)) for a in ALPHAS}, "det": {str(a): float(np.mean(np.asarray(on) <= a)) for a in ALPHAS}, "auroc": auroc(kn, on)}
    for k in ("persample_flagged", "persample_pool_flagged"):
        on = [r[k] for r in rows if r["label"] == "onset"]
        kn = [r[k] for r in rows if r["label"] == "known"]
        out[k] = {"onset": float(np.mean(on)), "known": float(np.mean(kn)), "auroc": auroc(kn, on, higher_is_pos=True)}
    return out


def summarize(rows: list, keys: tuple, title: str) -> None:
    print(f"\n=== {title} (mean over seeds {SEEDS}) ===")
    print("rho  statistic       false alarm @0.05 / @0.01 / @0.001     detection @0.05 / @0.01 / @0.001     AUROC")
    for rho in RHOS:
        rr = [r for r in rows if r["rho"] == rho]
        for k in keys:
            fa = [np.mean([r[k]["fa"][str(a)] for r in rr]) for a in ALPHAS]
            de = [np.mean([r[k]["det"][str(a)] for r in rr]) for a in ALPHAS]
            print(f"{rho:4.2f} {k:15s} {fa[0]:8.3f} {fa[1]:8.3f} {fa[2]:8.3f}          {de[0]:8.3f} {de[1]:8.3f} {de[2]:8.3f}          {np.mean([r[k]['auroc'] for r in rr]):6.3f}")
        ps = [r["persample_flagged"] for r in rr]
        print(f"{rho:4.2f} per-sample BH    flagged fraction own/known vs foreign/onset: "
              + (f"{np.mean([p['own'] for p in ps]):.3f} vs {np.mean([p['foreign'] for p in ps]):.3f}" if "own" in ps[0] else f"{np.mean([p['known'] for p in ps]):.3f} vs {np.mean([p['onset'] for p in ps]):.3f}")
              + f"   AUROC {np.mean([p['auroc'] for p in ps]):.3f}")


def conditional(a2: list) -> None:
    """False-alarm rate at alpha = 0.05 on KNOWN batches, split by properties of the owner expert (this is where the miscalibration comes from)."""
    print("\n=== A2 conditional false alarms (known batches, alpha 0.05): where do they come from? ===")
    for rho in RHOS:
        rows = [dict(r, rho=rho, seed=a["seed"]) for a in a2 if a["rho"] == rho for r in a["rows"] if r["label"] == "known"]
        dom = {}
        for a in [a for a in a2 if a["rho"] == rho]:
            for reg in {r["regime"] for r in a["rows"]}:
                owners = [r["owner"] for r in a["rows"] if r["regime"] == reg and r["label"] == "known"]
                if owners:
                    dom[(a["seed"], reg)] = max(set(owners), key=owners.count)
        for r in rows:
            r["is_dom"] = dom.get((r["seed"], r["regime"])) == r["owner"]
        def fa(sel, k):
            s = [r for r in rows if sel(r)]
            return (f"{np.mean([r[k] <= 0.05 for r in s]):.2f} (n={len(s)})" if s else "-")
        print(f"rho {rho}: mean per-sample p of the owner on known batches {np.mean([r['mean_p_owner'] for r in rows]):.3f} (0.5 if calibrated)")
        for name, sel in (("owner is the regime's usual owner", lambda r: r["is_dom"]), ("owner is NOT the usual owner", lambda r: not r["is_dom"]),
                          ("owner has < 64 prototypes", lambda r: r["owner_n"] < 64), ("owner has 64-255 prototypes", lambda r: 64 <= r["owner_n"] < 256), ("owner has 256 prototypes", lambda r: r["owner_n"] >= 256),
                          ("owner is an infant", lambda r: r["owner_state"] == "infant"), ("owner age < 15 steps", lambda r: r["owner_age"] < 15), ("owner age >= 15 steps", lambda r: r["owner_age"] >= 15)):
            print(f"   {name:36s} meandist {fa(sel, 'meandist')}   mmd {fa(sel, 'mmd')}   scoresum {fa(sel, 'scoresum')}   fisher_owner {fa(sel, 'fisher_owner')}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"                    # all | a1 | a2 | a2recent
    only_a1, only_a2 = mode == "a1", mode in ("a2", "a2recent")
    if mode == "a2recent":
        globals()["POOL_CLS"] = ShadowPoolRecent
    a1 = []
    for rho in ([] if only_a2 else RHOS):
        for seed in SEEDS:
            a1.append(run_a1(rho, seed))
            print("A1", rho, seed, "done", flush=True)
    if not only_a2:
        (HERE / "spike_set_tests_a1.json").write_text(json.dumps(a1, indent=1), encoding="utf-8")
        summarize(a1, STATS, "A1 clean: in-regime vs foreign batches against the expert's own reference")
    if only_a1:
        sys.exit(0)
    a2 = []
    for rho in RHOS:
        for seed in SEEDS:
            a2.append(run_a2(rho, seed))
            print("A2", rho, seed, "done; experts at end", a2[-1]["experts_end"], "onsets", a2[-1]["n_onset"], "known", a2[-1]["n_known"], flush=True)
    (HERE / ("spike_set_tests_a2recent.json" if POOL_CLS is ShadowPoolRecent else "spike_set_tests_a2.json")).write_text(json.dumps(a2, indent=1), encoding="utf-8")
    summarize(a2, STATS + ("fisher_pool",), "A2 live shadow: onset batches vs known batches inside the running pool")
    conditional(a2)
    if POOL_CLS is ShadowPoolRecent:
        summarize_recent(a2)
