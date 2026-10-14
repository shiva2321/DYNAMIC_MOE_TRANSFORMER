"""DEV, exploratory, not pre-registered; tuned and scored on the same dev seeds (100-102), so optimistic for the tuned arms.

M5-C: does the amended stage-3 gate (set_gate=True, persistence set_gate_confirm=2) repair the real-text failure of M5-B?
The M5-B numbers for A_today, B2 (replay network), B4 (oracle-routed experts), B1 (joint ceiling) and kNN are read from m5b.json (same streams, same seeds).

    python research/growmoe_results/m5/run_m5c_stage3.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(3)

from growmoe.config import PhyloCfg  # noqa: E402
from growmoe.fairfight import run_fast, tune_steps  # noqa: E402
from growmoe.pools.phylo import PhyloMoE  # noqa: E402
from growmoe.streams_real import banking_incremental  # noqa: E402
from research.growmoe_results.m0.run_m0 import HybridFrontend  # noqa: E402

SEEDS = (100, 101, 102)
LABEL = "DEV, exploratory, not pre-registered; tuned and scored on the same dev seeds, so optimistic for the tuned arms"


def cfg_for(st, seed, steps, set_gate):
    return PhyloCfg(d=st.d, n_classes=st.n_classes, seed=seed, inner_steps=steps, route_rule="accept_then_logf", q=0.02, alpha_spawn=0.01, m_min=32, set_gate=set_gate)


def main():
    print(LABEL)
    fe = HybridFrontend(d=128, seed=0)
    streams = {sd: banking_incremental(seed=sd, frontend=fe) for sd in SEEDS}
    old = json.loads((HERE / "m5b.json").read_text(encoding="utf-8"))

    def run(sd, steps, set_gate):
        holder = []

        def make():
            holder.append(PhyloMoE(cfg_for(streams[sd], sd, steps, set_gate)))
            return holder[0]
        res = run_fast(make, streams[sd], sd, eval_every="final")
        return res, holder[0]

    out = {"label": LABEL, "seeds": list(SEEDS)}
    tuned = {}
    for name, set_gate in (("A_set", True),):
        print(f"tuning {name} ...", flush=True)

        def score(steps, set_gate=set_gate):
            return float(np.mean([run(sd, steps, set_gate)[0].acc for sd in SEEDS]))
        r = tune_steps(score, grid=(1, 5, 20, 50), factor=2.5, cap=200)
        tuned[name] = r.best
        out[f"{name}_tuning"] = {"best_steps": r.best, "at_edge": r.at_edge, "table": {str(k): float(v) for k, v in r.table.items()}}
        print(f"  {name}: tuned steps {r.best}, table {({k: round(v * 100, 1) for k, v in r.table.items()})}", flush=True)
    rows = {}
    for name, steps, set_gate in (("A_set (tuned steps)", tuned["A_set"], True), ("A_set (steps 20, like A_today)", 20, True), ("A_today (steps 20, re-run)", 20, False)):
        accs, experts, stats = [], [], []
        for sd in SEEDS:
            t0 = time.time()
            res, pool = run(sd, steps, set_gate)
            accs.append(res.acc)
            experts.append(res.report.n_experts)
            stats.append(dict(pool.set_stats) if set_gate else {})
            print(f"  {name:<32s} seed {sd}: acc {res.acc * 100:5.1f}%  experts {res.report.n_experts}  stored {res.report.stored_vectors}  bytes {res.report.memory_bytes / 1e6:.2f} MB  {time.time() - t0:.0f}s"
                  + (f"  set tests {stats[-1].get('tests')} novel {stats[-1].get('novel')} held {stats[-1].get('held')} known {stats[-1].get('known')}" if set_gate else ""), flush=True)
        rows[name] = {"acc_by_seed": accs, "mean": float(np.mean(accs)), "experts": experts, "set_stats": stats}
    out["amended_runs"] = rows
    out["m5b_reference"] = {a: float(np.mean([old["final_runs"][a][str(sd)]["acc"] for sd in SEEDS])) for a in old["final_runs"]}
    print("\nReference from m5b.json (same streams and seeds):", {a: round(100 * v, 1) for a, v in out["m5b_reference"].items()})
    print("Amended stage 3:", {k: round(100 * v["mean"], 1) for k, v in rows.items()})
    (HERE / "m5c_stage3.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
