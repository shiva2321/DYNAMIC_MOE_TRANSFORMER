"""Spike J: is the benchmark budget-limited? Vary inner_steps (optimiser steps per routed batch) for A and for the
replay baseline B2 on the benchmark's own S1/S2 configs. Everything else unchanged. Throwaway diligence script."""
import sys
import time

from dil_common import *  # noqa

from growmoe.baselines import ReplayNet
from growmoe.bench import run_stream
from growmoe.config import CommonCfg, PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams import s1_types, s2_permuted

SEEDS = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 5))
STEPS = [1, 5, 20]
t0 = time.time()
out = {}
for sd in SEEDS:
    streams = {"s2": s2_permuted(seed=sd, n_regimes=5, schedule=[0, 1, 2, 3, 4, 1, 3], n_train=400, n_test=100),
               "s1": s1_types(seed=sd, n_train=400, n_test=150)}
    for sname, st in streams.items():
        for k in STEPS:
            res = run_stream(lambda: PhyloMoE(PhyloCfg(router=False, spawn_rule="probe", growth_rule="probe", seed=sd, q=0.02, alpha_spawn=0.01, m_min=32,
                                                       route_rule="accept_then_logf", inner_steps=k)), st, seed=sd)
            out.setdefault(("A", sname, k), []).append((res.accuracy_stats["avg"], res.accuracy_stats["forgetting"],
                                                        res.pool_summary["n_experts"]))
            res = run_stream(lambda: ReplayNet(CommonCfg(seed=sd, n_classes=st.n_classes, inner_steps=k)), st, seed=sd)
            out.setdefault(("B2", sname, k), []).append((res.accuracy_stats["avg"], res.accuracy_stats["forgetting"], 1))
    print(f"seed {sd} done ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== Accuracy (%) vs optimiser steps per routed batch, {len(SEEDS)} seeds, mean [95% CI] ===")
for sname in ("s2", "s1"):
    print(f"\n-- {sname}   (kNN with all stored samples: " + ("95.6" if sname == "s2" else "77.95") + ")")
    print(f"{'method':6s} {'inner_steps':>11s} {'accuracy':>22s} {'forgetting':>20s} {'experts':>8s}")
    for m in ("A", "B2"):
        for k in STEPS:
            rows = out[(m, sname, k)]
            print(f"{m:6s} {k:11d} {fmt([r[0] for r in rows]):>22s} {fmt([r[1] for r in rows]):>20s} {np.mean([r[2] for r in rows]):8.1f}")
