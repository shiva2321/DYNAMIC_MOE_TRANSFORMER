"""Spike O (PILOT, not pre-registered): a continual stream of FUNCTION regimes. Regime r lives in its own region of the sphere
(inputs = unit(8 * centre_r + noise)); its label is a hidden random teacher network applied to the within-region deviation, so
neighbours in input space do NOT share labels (memorising fails) but each regime is learnable. Methods: PhyloMoE (1 and 20 steps),
replay network (1 and 20 steps), oracle-routed experts (20 steps), kNN. Throwaway diligence script."""
import sys
import time

from dil_common import *  # noqa

from growmoe.baselines import OracleRouted, ReplayNet
from growmoe.bench import make_regime_of, run_stream
from growmoe.config import CommonCfg, PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams import Regime, Stream, _balanced_teacher, _long, _rng, _segment, _tensor, _unit

SEEDS = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2").split(",")]
SIZES = [5, 15, 30]
D, C, BF = 128, 8, 4


def function_stream(seed, n_regimes, n_train=800, n_test=400, center_scale=8.0, tw=8, td=1):
    rng = _rng(seed, 0xF5)
    centers = _unit(rng.standard_normal((n_regimes, D)))
    regimes, pools = {}, {}
    for r in range(n_regimes):
        teacher = _balanced_teacher(tw, td, D, C, seed=seed * 1000 + r)

        def make(n):
            z = rng.standard_normal((n, D))
            x = _unit(center_scale * centers[r] + z)
            with torch.no_grad():
                y = teacher(_tensor(_unit(z))).argmax(1).numpy()
            return x, y

        x_tr, y_tr = make(2 * n_train)
        x_te, y_te = make(n_test)
        regimes[r] = Regime(r, f"fn{r}", _tensor(x_te), _long(y_te), _long(np.arange(n_test)), meta={"type": "function"})
        pools[r] = (x_tr, y_tr)
    schedule = list(range(n_regimes)) + [0, n_regimes // 2]
    segments, visits = [], {}
    for r in schedule:
        v = visits.get(r, 0)
        visits[r] = v + 1
        x_tr, y_tr = pools[r]
        sl = slice(v * n_train, (v + 1) * n_train)
        segments.append(_segment(r, v == 0, _tensor(x_tr[sl]), _long(y_tr[sl]), np.arange(sl.start, sl.stop)))
    return Stream("function_regimes", D, C, regimes, segments, {"schedule": schedule})


def a_mem(pool):
    vecs = sum(r.meta["sig"].n + (r.replay.n if r.replay is not None else 0) for r in pool.experts.values())
    params = sum(p.numel() for r in pool.experts.values() for p in r.module.parameters())
    return (params * BF + vecs * D * BF) / 1024.0


out = {}
t0 = time.time()
for sd in SEEDS:
    for n in SIZES:
        st = function_stream(sd, n)
        a_kb = None
        for k in (1, 20):
            hold = []

            def mk():
                p = PhyloMoE(PhyloCfg(seed=sd, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf", inner_steps=k))
                hold.append(p)
                return p

            res = run_stream(mk, st, seed=sd)
            kb = a_mem(hold[-1])
            if k == 20:
                a_kb = kb
            out.setdefault((n, f"PhyloMoE {k} step(s)"), []).append((res.accuracy_stats["avg"], len(hold[-1].experts), kb))
            res = run_stream(lambda: ReplayNet(CommonCfg(seed=sd, n_classes=C, inner_steps=k)), st, seed=sd)
            out.setdefault((n, f"Replay net {k} step(s)"), []).append((res.accuracy_stats["avg"], 1, 1160.0))
        res = run_stream(lambda: OracleRouted(make_regime_of(st, 64), CommonCfg(seed=sd, n_classes=C, inner_steps=20)), st, seed=sd)
        out.setdefault((n, "Oracle: one expert per regime"), []).append((res.accuracy_stats["avg"], n, n * 167.2  # 41,792 float32 parameters per expert = 167.2 KB (the original printout mis-scaled this column)))
        res = run_stream(lambda: KnnPool(C, D, k=5, capacity=None, seed=sd), st, seed=sd)
        out.setdefault((n, "kNN stores everything"), []).append((res.accuracy_stats["avg"], 1, (n + 2) * 800 * D * BF / 1024.0))
        cap = max(int(a_kb * 1024 / (D * BF)), 16)
        res = run_stream(lambda: KnnPool(C, D, k=5, capacity=cap, seed=sd), st, seed=sd)
        out.setdefault((n, "kNN same bytes as PhyloMoE"), []).append((res.accuracy_stats["avg"], 1, cap * D * BF / 1024.0))
        print(f"seed {sd} regimes {n} done ({time.time()-t0:.0f}s)", flush=True)

import json
with open(f"function_stream_raw_seeds_{'_'.join(map(str, SEEDS))}.json", "w") as fh:
    json.dump({f"{n}|{k}": v for (n, k), v in out.items()}, fh)

print(f"\n=== Function regimes (8 classes, chance 12.5%): final average accuracy %, {len(SEEDS)} seeds, mean [95% CI] ===")
print(f"{'regimes':>7s} {'method':32s} {'accuracy':>22s} {'experts':>8s} {'memory KB':>10s}")
for n in SIZES:
    for key in ("kNN stores everything", "kNN same bytes as PhyloMoE", "Replay net 1 step(s)", "Replay net 20 step(s)",
                "PhyloMoE 1 step(s)", "PhyloMoE 20 step(s)", "Oracle: one expert per regime"):
        rows = out[(n, key)]
        print(f"{n:7d} {key:32s} {fmt([r[0] for r in rows]):>22s} {np.mean([r[1] for r in rows]):8.1f} {np.mean([r[2] for r in rows]):10.0f}")
    print()
