"""Exploratory, THROWAWAY diagnostics on DEV seeds only (100-102); nothing here is pre-registered or feeds a verdict.

D3a  Where do A's training samples actually go? For every first-visit segment of F-15: share of samples sent to the expert that
     ends up owning the regime, to other experts (contamination), or to nobody (-1 = held in probation), by batch index; plus
     the event timeline of the first regimes.
D3b  Does A's expert RECIPE explain the gap to oracle-routed experts? Oracle gating (true regime id) with pieces of A's
     recipe added one at a time: per-expert replay mixing, infant learning-rate boost, and the training budget.

    python research/growmoe_results/m0/diagnose_routing_causes.py
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402

import run_m0 as M                                                                      # noqa: E402
from growmoe import bench                                                               # noqa: E402
from growmoe.config import CommonCfg, PhyloCfg                                          # noqa: E402
from growmoe.experts import MLPExpert                                                   # noqa: E402
from growmoe.fairfight import run_fast                                                  # noqa: E402
from growmoe.pools.base import BasePool                                                 # noqa: E402
from growmoe.pools.phylo import PhyloMoE                                                # noqa: E402
from growmoe.types import Prediction, StepRecord                                        # noqa: E402

torch.set_num_threads(3)
TUNING = json.loads((HERE / "tuning.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------------------- D3a
def d3a(seed: int) -> dict:
    st = M.get_stream("F-15", seed, False)
    P = M.params_for("A", "func", TUNING)
    pool = PhyloMoE(PhyloCfg(seed=seed, d=st.d, n_classes=st.n_classes, inner_steps=P["steps"], **M.A_KW))
    rows, seen_ev, cur, j = [], 0, -1, 0
    for k, seg, idx in st._walk(64, seed):
        j = 0 if k != cur else j + 1
        cur = k
        rec = pool.observe(seg.x[idx], y=seg.y[idx])
        evs = list(pool.events)[seen_ev:]
        seen_ev = len(pool.events)
        rows.append({"k": k, "regime": seg.regime, "first": bool(seg.first_visit), "j": j, "assigned": rec.assigned.tolist(),
                     "events": [(e.kind, e.fields.get("expert"), e.fields.get("parent")) for e in evs]})
    # owner of each regime = the expert that received most of its first-visit samples
    owner, mass = {}, {}
    for r in st.regimes:
        cnt, tot = Counter(), 0
        for row in rows:
            if row["regime"] == r and row["first"]:
                for a in row["assigned"]:
                    tot += 1
                    if a >= 0:
                        cnt[a] += 1
        owner[r] = cnt.most_common(1)[0][0] if cnt else -1
        mass[r] = (cnt[owner[r]] / tot) if tot else 0.0
    by_j = defaultdict(lambda: Counter())
    for row in rows:
        if row["first"]:
            o = owner[row["regime"]]
            for a in row["assigned"]:
                by_j[row["j"]]["own" if a == o else ("none" if a < 0 else "other")] += 1
    purity = {int(jj): {k: v / sum(c.values()) for k, v in c.items()} for jj, c in sorted(by_j.items())}
    # who is contaminated: experts that received samples of a regime that is not theirs
    contaminated = Counter()
    for row in rows:
        if row["first"]:
            o = owner[row["regime"]]
            for a in row["assigned"]:
                if a >= 0 and a != o:
                    contaminated[a] += 1
    owners = Counter(owner.values())
    # held-out accuracy per regime with the learned gate (batched predict)
    acc = {}
    for r, reg in st.regimes.items():
        pred = pool.predict(reg.x_te).probs.argmax(1)
        acc[r] = float((pred == reg.y_te).float().mean())
    corr = float(np.corrcoef([mass[r] for r in st.regimes], [acc[r] for r in st.regimes])[0, 1]) if len(st.regimes) > 2 else float("nan")
    timeline = [(row["k"], row["regime"], row["j"], row["events"]) for row in rows if row["events"] and row["k"] < 5]
    return {"seed": seed, "n_experts": len(pool.experts), "owner": owner, "mass_to_owner": mass, "regimes_sharing_an_expert": sum(1 for v in owners.values() if v > 1),
            "purity_by_batch_index": purity, "samples_into_non_owner_experts": int(sum(contaminated.values())),
            "mean_mass_to_owner": float(np.mean(list(mass.values()))), "acc_by_regime": acc, "mean_acc": float(np.mean(list(acc.values()))),
            "corr_mass_vs_acc": corr, "timeline_first_regimes": timeline}


# ------------------------------------------------------------------------------------------------------------- D3b
class OracleGated(BasePool):
    """One expert per regime, regime id supplied by the benchmark (like B4), but trained with A's own ``train_expert``
    (``replay`` = mix half of every step's batch from the expert's own replay reservoir) and optionally A's infant learning rate."""
    name = "oracle_gated"

    def __init__(self, cfg: CommonCfg, regime_of, replay: bool, infant: bool) -> None:
        super().__init__(cfg)
        self.regime_of, self.replay, self.infant = regime_of, replay, infant
        self.by_regime, self.eval_regime, self.calls = {}, None, {}

    def set_eval_regime(self, r: int) -> None:
        self.eval_regime = r

    def observe(self, x, y=None, ctx=None) -> StepRecord:
        c = self.cfg
        r = int(self.regime_of(self.step))
        self.step += 1
        if r not in self.by_regime:
            mod = MLPExpert(c.d, c.hidden, c.depth, c.n_classes, seed=c.seed * 1000 + len(self.by_regime))
            rec = self.add_expert(mod, state="infant" if self.infant else "active")
            if self.infant:
                self.set_lr_mult(rec.id, 2.0)
            self.by_regime[r], self.calls[rec.id] = rec.id, 0
        eid = self.by_regime[r]
        loss = self.train_expert(eid, x, y, replay=self.replay)
        self.calls[eid] += 1
        if self.infant and self.calls[eid] >= 5 and self.experts[eid].state == "infant":
            self.set_state(eid, "active")
            self.set_lr_mult(eid, 1.0)
        b = x.shape[0]
        return StepRecord(novel=torch.zeros(b, dtype=torch.bool), p_nov=torch.ones(b), assigned=torch.full((b,), eid, dtype=torch.long),
                          loss=loss, spawned=[], n_experts=len(self.experts))

    def predict(self, x, ctx=None) -> Prediction:
        eid = self.by_regime.get(self.eval_regime)
        b = x.shape[0]
        if eid is None:
            return Prediction(probs=torch.full((b, self.cfg.n_classes), 1.0 / self.cfg.n_classes), expert=torch.full((b,), -1), p_nov=torch.ones(b),
                              unknown=torch.zeros(b, dtype=torch.bool))
        with torch.no_grad():
            return Prediction(probs=self.experts[eid].module.proba(x), expert=torch.full((b,), eid), p_nov=torch.ones(b), unknown=torch.zeros(b, dtype=torch.bool))

    def summary(self) -> dict:
        return {"n_experts": len(self.experts)}


def d3b(seeds=(100, 101, 102)) -> list:
    out = []
    variants = [("B4 as run in M0 (5 steps, no replay mixing)", dict(kind="b4", steps=5)),
                ("oracle gating, plain training, 5 steps", dict(kind="og", steps=5, replay=False, infant=False)),
                ("+ A's replay mixing, 5 steps", dict(kind="og", steps=5, replay=True, infant=False)),
                ("+ A's replay mixing + infant lr, 5 steps", dict(kind="og", steps=5, replay=True, infant=True)),
                ("oracle gating, plain training, 50 steps", dict(kind="og", steps=50, replay=False, infant=False)),
                ("+ A's replay mixing, 50 steps", dict(kind="og", steps=50, replay=True, infant=False)),
                ("+ A's replay mixing + infant lr, 50 steps", dict(kind="og", steps=50, replay=True, infant=True)),
                ("+ A's replay mixing, 20 steps", dict(kind="og", steps=20, replay=True, infant=False))]
    for label, v in variants:
        accs = []
        for sd in seeds:
            st = M.get_stream("F-15", sd, False)
            reg_of = bench.make_regime_of(st, 64)
            cfg = CommonCfg(seed=sd, d=st.d, n_classes=st.n_classes, inner_steps=v["steps"])
            if v["kind"] == "b4":
                from growmoe.baselines import OracleRouted
                make = lambda: OracleRouted(reg_of, cfg)                                 # noqa: E731
            else:
                make = lambda: OracleGated(cfg, reg_of, v["replay"], v["infant"])        # noqa: E731
            accs.append(run_fast(make, st, sd, eval_every="final").acc)
        out.append({"variant": label, "dev_acc": float(np.mean(accs)), "per_seed": accs})
        print(f"{label:52s} {100 * np.mean(accs):5.1f}   per seed {[round(100 * a, 1) for a in accs]}", flush=True)
    return out


if __name__ == "__main__":
    res = {"d3a": [], "d3b": []}
    for sd in (100, 101):
        r = d3a(sd)
        res["d3a"].append(r)
        print(f"\nD3a seed {sd}: experts {r['n_experts']}, mean accuracy {100 * r['mean_acc']:.1f}, regimes sharing an expert {r['regimes_sharing_an_expert']}, "
              f"mean share of a regime's samples reaching its owner {100 * r['mean_mass_to_owner']:.1f}%, samples sent to non-owner experts {r['samples_into_non_owner_experts']}, "
              f"corr(mass to owner, accuracy) {r['corr_mass_vs_acc']:.2f}", flush=True)
        for jj, p in r["purity_by_batch_index"].items():
            print(f"   batch {jj:2d}: " + "  ".join(f"{k} {100 * v:5.1f}%" for k, v in sorted(p.items())), flush=True)
    print("\nD3b dev accuracy % on F-15 (seeds 100-102):", flush=True)
    res["d3b"] = d3b()
    (HERE / "diagnostics_D3.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
