"""Spike K: does the picture change as the number of regimes grows? Permuted digits (the S2 builder, domain-incremental),
n_regimes in {5, 15, 30}, 400 train / 100 test per regime. A vs replay baseline B2 (each at 1 and 20 optimiser steps per
batch) vs kNN. Throwaway diligence script; S5/S6 are deliberately NOT touched (their pre-registration is being rewritten)."""
import sys
import time

from dil_common import *  # noqa

from growmoe.baselines import ReplayNet
from growmoe.bench import run_stream
from growmoe.config import CommonCfg, PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams import S2_SCHEDULE, s2_permuted

SEEDS = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
SIZES = [5, 15, 30]
D, BF = 128, 4


def schedule(n):
    if n == 5:
        return [0, 1, 2, 3, 4, 1, 3]
    if n == 15:
        return list(range(15)) + [3, 7, 0, 12]
    return S2_SCHEDULE


def a_mem(pool):
    vecs = sum(r.meta["sig"].n + (r.replay.n if r.replay is not None else 0) for r in pool.experts.values())
    params = sum(p.numel() for r in pool.experts.values() for p in r.module.parameters())
    return (params * BF + vecs * D * BF) / 1024.0


out = {}
t0 = time.time()
for sd in SEEDS:
    for n in SIZES:
        st = s2_permuted(seed=sd, n_regimes=n, schedule=schedule(n), n_train=400, n_test=100)
        a_kb = None
        for k in (1, 20):
            hold = []

            def mk():
                p = PhyloMoE(PhyloCfg(router=False, spawn_rule="probe", growth_rule="probe", seed=sd, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf", inner_steps=k))
                hold.append(p)
                return p

            res = run_stream(mk, st, seed=sd)
            kb = a_mem(hold[-1])
            if k == 20:
                a_kb = kb
            out.setdefault((n, f"A_steps{k}"), []).append((res.accuracy_stats["avg"], len(hold[-1].experts), kb, res.wall_clock))
            res = run_stream(lambda: ReplayNet(CommonCfg(seed=sd, n_classes=10, inner_steps=k)), st, seed=sd)
            out.setdefault((n, f"B2_steps{k}"), []).append((res.accuracy_stats["avg"], 1, 1160.0, res.wall_clock))
        res = run_stream(lambda: KnnPool(10, D, k=5, capacity=None, seed=sd), st, seed=sd)
        out.setdefault((n, "kNN_all_samples"), []).append((res.accuracy_stats["avg"], 1, 400 * n * D * BF / 1024.0, res.wall_clock))
        cap = max(int(a_kb * 1024 / (D * BF)), 16)
        res = run_stream(lambda: KnnPool(10, D, k=5, capacity=cap, seed=sd), st, seed=sd)
        out.setdefault((n, "kNN_same_bytes_as_A20"), []).append((res.accuracy_stats["avg"], 1, cap * D * BF / 1024.0, res.wall_clock))
        print(f"seed {sd} n_regimes {n} done ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== Permuted digits: final average accuracy (%) vs number of regimes; {len(SEEDS)} seeds, mean [95% CI] ===")
print(f"{'regimes':>7s} {'method':24s} {'accuracy':>22s} {'experts':>8s} {'memory KB':>10s} {'sec/run':>8s}")
for n in SIZES:
    for key in ("A_steps1", "A_steps20", "B2_steps1", "B2_steps20", "kNN_all_samples", "kNN_same_bytes_as_A20"):
        rows = out[(n, key)]
        print(f"{n:7d} {key:24s} {fmt([r[0] for r in rows]):>22s} {np.mean([r[1] for r in rows]):8.1f} "
              f"{np.mean([r[2] for r in rows]):10.0f} {np.mean([r[3] for r in rows]):8.1f}")
    print()
