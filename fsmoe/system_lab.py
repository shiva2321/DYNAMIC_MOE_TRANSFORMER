"""Layer S: the system on a continual stream over the three domains.

* S1 teaching: the pool is taught the taught skills of arithmetic, then code, then language (labelled batches, no regime ids), and compared with equal-memory baselines (kNN, replay net, streaming
  LDA) and the offline joint network (a ceiling). Read on rows the pool never trained on: taught skills in teaching wordings and in HELD-OUT wordings.
* S3 teams: at answer time every expert gives its blended (network + closed-form) answer and the input's team (smallest set of experts holding 99 % of the routing weight, at most 3) mixes them;
  the weights come from each expert's own closed-form statistics (Mahalanobis distance to its nearest taught class), so no extra model is needed.
* S2 deployment: an unlabelled stream of known skills (all wordings), withheld skills (never taught) and far texts. A budget of VERIFIED outcomes per batch is spent by a policy (none, random,
  novelty-ranked, uncertainty-ranked, oracle); verified items become labelled training events (the pool's own spawn/extend rule decides what a new skill is). Measured on rows the stream never
  contained: how many withheld skills the pool ends up answering, and what happens to known skills in held-out wordings.
The pool is ``unimoe.systems2.make_system2('ULDA')`` (the validated pool with the closed-form head in every expert); nothing in growmoe/ or unimoe/ is edited.
``python -m fsmoe.system_lab tune`` (dev seed) then ``python -m fsmoe.system_lab run --seeds ...``.
"""

import argparse
import copy
import itertools
import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from growmoe.accounting import pool_report
from unimoe.density import DensityHead
from unimoe.pools import blend
from unimoe.systems import make_system
from unimoe.systems2 import make_system2

from .bed import ROOT, Bed, build_bed
from .detectors import Maha
from .domains import NAMES
from .frontend import Frontend
from .recog_lab import _parity
from .team import choose_team, mix, team_size

OUT = ROOT / "results" / "fsmoe" / "S"
TUNE_SEED = 700
BATCH = 64
FS_GRID = [dict(inner_steps=s, replay_size=r, lda_blend=lam) for s, r, lam in itertools.product((1, 5), (256, 4000), (0.5, 0.75))]
BASE_GRID = [dict(inner_steps=s, lr=lr) for s, lr in itertools.product((1, 5), (1e-3, 3e-3))]
POLICIES = (("none", 0.0), ("random", 0.05), ("novelty", 0.05), ("uncertainty", 0.05), ("oracle", 0.05), ("random", 0.10), ("novelty", 0.10))
N_NOVEL, N_FAR, VERIFY_CHUNK = 800, 600, 16


class Prep:
    """The frontend fitted on teaching rows, every row embedded, and the test / calibration halves of the evaluation rows (the stream uses the test half; the final reading uses the other half)."""

    def __init__(self, bed: Bed) -> None:
        self.bed = bed
        tr = bed.teaching()
        self.fe = Frontend(128, "pca", 0).fit(bed.raw[tr], np.ones(int(tr.sum()), dtype=bool))
        self.X = self.fe(bed.raw)
        self.y = torch.from_numpy(bed.y)
        self.d = self.X.shape[1]
        self.n_classes = len(bed.skills)
        ev = np.flatnonzero(bed.mask(split="eval"))
        test = _parity(bed, ev)
        self.test_rows, self.cal_rows = ev[test], ev[~test]
        self.far_X = torch.cat([self.fe(v[1]) for k, v in sorted(bed.far.items())])

    def rows(self, which: str, domain: Optional[str] = None, kind: str = "known_tw") -> np.ndarray:
        r = self.test_rows if which == "test" else self.cal_rows
        b = self.bed
        m = np.ones(len(r), dtype=bool) if domain is None else b.domain[r] == domain
        if kind == "known_tw":
            m &= b.taught[r] & ~b.heldout[r]
        elif kind == "known_ho":
            m &= b.taught[r] & b.heldout[r]
        elif kind == "novel":
            m &= ~b.taught[r]
        return r[m]


def accuracy(pool, prep: Prep, rows: np.ndarray) -> float:
    if not len(rows):
        return float("nan")
    return float((pool.predict(prep.X[rows]).probs.argmax(1) == prep.y[rows]).float().mean())


# ---------------------------------------------------------------------------------------------------------------------------------- S3: teams
def _density_head(lda) -> DensityHead:
    h = DensityHead(lda.d, lda.n_classes, lda.shrink)
    h.n, h.s, h.S, h.total = lda.n, lda.s, lda.S, lda.total
    return h


@torch.no_grad()
def team_predict(pool, X: torch.Tensor, tau: float = 0.3, mass: float = 0.99, cap: int = 3):
    """(probs (B, C), team sizes (B,)). Experts without a closed-form head are left out of teams."""
    lam = pool.cfg.lda_blend
    ids = [e for e, r in pool.experts.items() if r.meta.get("lda") is not None and r.meta["lda"].total > 0]
    probs, dist = [], []
    for e in ids:
        rec = pool.experts[e]
        rec.module.eval()
        probs.append(blend(rec.module.proba(X), rec.meta["lda"].proba(X), lam))
        dist.append(_density_head(rec.meta["lda"]).distance(X))
    P, D = torch.stack(probs), torch.stack(dist, 1)                                   # (E, B, C), (B, E)
    ratio = D / D.min(1, keepdim=True).values.clamp_min(1e-6)
    members, w = choose_team(torch.softmax(-(ratio - 1.0) / tau, dim=1), mass, cap)
    return mix(P, members, w), team_size(members)


# ---------------------------------------------------------------------------------------------------------------------------------- S1: teaching
def make_pool(name: str, prep: Prep, seed: int, recipe: dict, mem_bytes: Optional[int] = None, total: Optional[int] = None):
    if name == "FS":
        return make_system2("ULDA", prep.n_classes, seed, recipe, d=prep.d)
    return make_system(name, prep.n_classes, seed, recipe, mem_bytes=mem_bytes, total_samples=total, d=prep.d)


def teach(pool, prep: Prep, seed: int) -> dict:
    """Teach the three domains in turn; read every domain's known skills (teaching wording, test rows) after each phase and everything at the end."""
    bed = prep.bed
    hist, t0 = [], time.perf_counter()
    for k, dom in enumerate(NAMES):
        idx = np.flatnonzero(bed.teaching(dom))
        idx = idx[np.random.default_rng([seed, 0x5445, k]).permutation(len(idx))]
        for lo in range(0, len(idx), BATCH):
            b = idx[lo:lo + BATCH]
            pool.observe(prep.X[b], prep.y[b])
        hist.append({d: accuracy(pool, prep, prep.rows("test", d, "known_tw")) for d in NAMES[:k + 1]})
    final = {d: {"tw": accuracy(pool, prep, prep.rows("test", d, "known_tw")), "ho": accuracy(pool, prep, prep.rows("test", d, "known_ho"))} for d in NAMES}
    forgetting = {d: hist[NAMES.index(d)][d] - final[d]["tw"] for d in NAMES}
    rep = pool_report(pool, prep.d)
    return {"history": hist, "final": final, "forgetting": forgetting, "mean_tw": float(np.mean([final[d]["tw"] for d in NAMES])), "mean_ho": float(np.mean([final[d]["ho"] for d in NAMES])),
            "n_experts": int(pool.summary().get("n_experts", 1)), "params": int(rep.params), "stored_vectors": int(rep.stored_vectors), "memory_bytes": int(rep.memory_bytes),
            "events": dict(Counter(e.kind for e in getattr(pool, "events", []))), "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------------------------------------------------------------------------- S2: deployment
def build_stream(prep: Prep, seed: int):
    rng = np.random.default_rng([seed, 0x53545245])
    bed = prep.bed
    known = prep.rows("test", None, "known_tw")
    known = np.concatenate([known, prep.rows("test", None, "known_ho")])
    nov_all = prep.rows("test", None, "novel")
    novel = rng.choice(nov_all, min(N_NOVEL, len(nov_all)), replace=False)
    far_i = rng.choice(len(prep.far_X), min(N_FAR, len(prep.far_X)), replace=False)
    X = torch.cat([prep.X[known], prep.X[novel], prep.far_X[far_i]])
    y = torch.cat([prep.y[known], prep.y[novel], torch.full((len(far_i),), -1, dtype=torch.long)])
    kind = np.asarray(["known"] * len(known) + ["novel"] * len(novel) + ["far"] * len(far_i))
    order = rng.permutation(len(y))
    return X[order], y[order], kind[order]


def deploy(pool, prep: Prep, seed: int, policy: str, budget: float) -> dict:
    X, y, kind = build_stream(prep, seed)
    rng = np.random.default_rng([seed, 0x504F4C, zlib_policy(policy, budget)])
    teach_rows = np.flatnonzero(prep.bed.teaching())
    scorer = Maha().fit(prep.X[teach_rows], prep.y[teach_rows]) if policy in ("novelty", "diverse") else None
    buf_x, buf_y = [], []
    ver_total = ver_novel = ver_known = ver_far = 0
    ver_skills = set()
    online = {"novel": [], "known": []}
    n = len(y)
    half = n // 2
    for lo in range(0, n, BATCH):
        sl = slice(lo, min(lo + BATCH, n))
        x, yy, kk = X[sl], y[sl], kind[sl]
        probs = pool.predict(x).probs
        pred = probs.argmax(1)
        for c in ("novel", "known"):
            m = torch.from_numpy(kk == c)
            if lo >= half and bool(m.any()):
                online[c].append((pred[m] == yy[m]).float())
        k = int(round(budget * len(x)))
        if policy != "none" and k > 0:
            if policy == "random":
                pick = rng.permutation(len(x))[:k]
            elif policy == "novelty":
                pick = np.argsort(-scorer.score(x))[:k]
            elif policy == "diverse":
                pick = diverse_pick(scorer.score(x), x, k)
            elif policy == "uncertainty":
                pick = np.argsort(-(1.0 - probs.max(1).values).numpy())[:k]
            elif policy == "oracle":
                nov = np.flatnonzero(kk == "novel")
                pick = nov[:k]
            else:
                raise ValueError(policy)
            for j in pick:
                ver_total += 1
                if kk[j] == "far":
                    ver_far += 1                                                          # verified as "no skill": nothing to learn from it
                    continue
                ver_novel += int(kk[j] == "novel")
                ver_known += int(kk[j] == "known")
                ver_skills.add(int(yy[j])) if kk[j] == "novel" else None
                buf_x.append(x[j])
                buf_y.append(yy[j])
            if len(buf_y) >= VERIFY_CHUNK:
                pool.observe(torch.stack(buf_x), torch.stack(buf_y))
                buf_x, buf_y = [], []
    if buf_y:
        pool.observe(torch.stack(buf_x), torch.stack(buf_y))
    withheld = sorted({int(s) for s in prep.y[prep.rows("cal", None, "novel")].tolist()})
    per_skill = {}
    cal_nov = prep.rows("cal", None, "novel")
    for s in withheld:
        r = cal_nov[prep.y[cal_nov].numpy() == s]
        per_skill[s] = accuracy(pool, prep, r)
    mean_online = lambda c: float(torch.cat(online[c]).mean()) if online[c] else float("nan")
    return {"policy": policy, "budget": budget, "verified": ver_total, "verified_novel": ver_novel, "verified_known": ver_known, "verified_far": ver_far,
            "precision_novel": ver_novel / ver_total if ver_total else float("nan"), "novel_skills_verified": len(ver_skills), "n_withheld": len(withheld),
            "learned_fraction": float(np.mean([a >= 0.5 for a in per_skill.values()])), "novel_acc_final": float(np.mean(list(per_skill.values()))),
            "novel_acc_online": mean_online("novel"), "known_acc_online": mean_online("known"),
            "known_ho_final": accuracy(pool, prep, prep.rows("cal", None, "known_ho")), "known_tw_final": accuracy(pool, prep, prep.rows("cal", None, "known_tw")),
            "n_experts": int(pool.summary().get("n_experts", 1))}


def diverse_pick(score: np.ndarray, x: torch.Tensor, k: int, pool_factor: int = 4) -> np.ndarray:
    """Novelty-ranked but spread out: take the ``pool_factor * k`` most novel inputs, start from the most novel and add, k times, the candidate farthest (in cosine distance) from those already picked.
    Plain novelty ranking picks the same few skills again and again; this spends the budget over many skills."""
    cand = np.argsort(-score)[:min(len(score), pool_factor * k)]
    if len(cand) <= k:
        return cand
    xc = torch.nn.functional.normalize(x[cand].float(), dim=1)
    picked = [0]
    dist = 1.0 - xc @ xc[0]
    for _ in range(k - 1):
        j = int(dist.argmax())
        picked.append(j)
        dist = torch.minimum(dist, 1.0 - xc @ xc[j])
    return cand[picked]


def zlib_policy(policy: str, budget: float) -> int:
    import zlib
    return zlib.crc32(f"{policy}{budget}".encode())


# ---------------------------------------------------------------------------------------------------------------------------------- the runs
def _score(res: dict) -> float:
    return 0.5 * (res["mean_tw"] + res["mean_ho"])


def tune(seed: int = TUNE_SEED) -> dict:
    """Dev seed only: every arm gets its own training recipe (F-57). Writes results/fsmoe/S/tuning.json."""
    OUT.mkdir(parents=True, exist_ok=True)
    bed = build_bed(seed)
    prep = Prep(bed)
    out = {"seed": seed, "FS": [], "baselines": {}}
    best, best_s, best_mem = None, -1.0, None
    for rec in FS_GRID:
        res = teach(make_pool("FS", prep, seed, rec), prep, seed)
        out["FS"].append({"recipe": rec, "mean_tw": res["mean_tw"], "mean_ho": res["mean_ho"], "n_experts": res["n_experts"], "memory_bytes": res["memory_bytes"]})
        print("FS", rec, f"tw {res['mean_tw']:.3f} ho {res['mean_ho']:.3f} experts {res['n_experts']}", flush=True)
        if _score(res) > best_s:
            best, best_s, best_mem = rec, _score(res), res["memory_bytes"]
    out["FS_best"], out["mem_bytes"] = best, best_mem
    total = int(sum(bed.teaching(d).sum() for d in NAMES))
    for name in ("replay_eqmem", "knn_eqmem", "lda", "joint"):
        grid = BASE_GRID if name in ("replay_eqmem", "joint") else [{}]
        rows = []
        for rec in grid:
            try:
                res = teach(make_pool(name, prep, seed, rec, best_mem, total), prep, seed)
            except ValueError as err:
                print(name, rec, "skipped:", err, flush=True)
                continue
            rows.append({"recipe": rec, "mean_tw": res["mean_tw"], "mean_ho": res["mean_ho"]})
            print(name, rec, f"tw {res['mean_tw']:.3f} ho {res['mean_ho']:.3f}", flush=True)
        if rows:
            pick = max(rows, key=lambda r: r["mean_tw"] + r["mean_ho"])
            out["baselines"][name] = {"recipe": pick["recipe"], "grid": rows}
    # team temperature on the dev seed
    pool = make_pool("FS", prep, seed, best)
    teach(pool, prep, seed)
    rows = prep.rows("test", None, "known_ho")
    out["team_tau"] = {}
    for tau in (0.1, 0.3, 1.0):
        probs, size = team_predict(pool, prep.X[rows], tau)
        out["team_tau"][str(tau)] = {"acc": float((probs.argmax(1) == prep.y[rows]).float().mean()), "mean_size": float(size.float().mean())}
        print("team tau", tau, out["team_tau"][str(tau)], flush=True)
    top = max(out["team_tau"][str(t)]["acc"] for t in (0.1, 0.3, 1.0))
    out["team_tau_best"] = min(t for t in (0.1, 0.3, 1.0) if out["team_tau"][str(t)]["acc"] >= top - 0.002)      # a tie within 0.002 goes to the smaller team
    (OUT / "tuning.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def run_seed(seed: int, tuning: dict) -> dict:
    path = OUT / f"S_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    bed = build_bed(seed)
    prep = Prep(bed)
    total = int(sum(bed.teaching(d).sum() for d in NAMES))
    res: dict = {"seed": seed, "teaching": {}, "team": {}, "deploy": []}
    fs_pool = make_pool("FS", prep, seed, tuning["FS_best"])
    res["teaching"]["FS"] = teach(fs_pool, prep, seed)
    mem = res["teaching"]["FS"]["memory_bytes"]
    for name, spec in tuning["baselines"].items():
        res["teaching"][name] = teach(make_pool(name, prep, seed, spec["recipe"], mem, total), prep, seed)
        print(f"seed {seed} {name} done", flush=True)
    rows = prep.rows("test", None, "known_ho")
    top1 = (fs_pool.predict(prep.X[rows]).probs.argmax(1) == prep.y[rows]).float().mean()
    probs, size = team_predict(fs_pool, prep.X[rows], tuning["team_tau_best"])
    res["team"] = {"tau": tuning["team_tau_best"], "top1_ho": float(top1), "team_ho": float((probs.argmax(1) == prep.y[rows]).float().mean()), "mean_size": float(size.float().mean()),
                   "size_hist": {str(k): int(v) for k, v in zip(*np.unique(size.numpy(), return_counts=True))}}
    for policy, budget in POLICIES:
        pool = copy.deepcopy(fs_pool)
        before = accuracy(pool, prep, prep.rows("cal", None, "known_ho"))
        r = deploy(pool, prep, seed, policy, budget)
        r["known_ho_before"] = before
        res["deploy"].append(r)
        print(f"seed {seed} deploy {policy} {budget}: learned {r['learned_fraction']:.2f} precision {r['precision_novel']:.2f}", flush=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(path)
    return res


EXTRA_POLICIES = (("oracle", 0.10), ("oracle", 0.20), ("oracle", 0.40), ("diverse", 0.05), ("diverse", 0.10))


def run_extra(seed: int, tuning: dict) -> dict:
    """Added AFTER the main readout showed that novelty ranking concentrates the budget on a few skills and that the oracle's 5 % is too small to learn 28 skills: the oracle's budget curve and a diversity-aware
    novelty policy. The pool is re-taught with the same seed (the recorded teaching numbers must reproduce) and the extra policies run from copies of it. Writes S_extra_s<seed>.json."""
    path = OUT / f"S_extra_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    main = json.loads((OUT / f"S_s{seed}.json").read_text(encoding="utf-8"))
    bed = build_bed(seed)
    prep = Prep(bed)
    fs_pool = make_pool("FS", prep, seed, tuning["FS_best"])
    t = teach(fs_pool, prep, seed)
    reproduced = abs(t["mean_tw"] - main["teaching"]["FS"]["mean_tw"]) < 1e-9 and abs(t["mean_ho"] - main["teaching"]["FS"]["mean_ho"]) < 1e-9
    res = {"seed": seed, "teaching_reproduced": bool(reproduced), "deploy": []}
    for policy, budget in EXTRA_POLICIES:
        pool = copy.deepcopy(fs_pool)
        before = accuracy(pool, prep, prep.rows("cal", None, "known_ho"))
        r = deploy(pool, prep, seed, policy, budget)
        r["known_ho_before"] = before
        res["deploy"].append(r)
        print(f"seed {seed} extra {policy} {budget}: learned {r['learned_fraction']:.2f} skills met {r['novel_skills_verified']}", flush=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(path)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "run", "extra"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[TUNE_SEED])
    a = ap.parse_args()
    if a.cmd == "tune":
        tune()
    else:
        tuning = json.loads((OUT / "tuning.json").read_text(encoding="utf-8"))
        for s in a.seeds:
            (run_seed if a.cmd == "run" else run_extra)(s, tuning)
