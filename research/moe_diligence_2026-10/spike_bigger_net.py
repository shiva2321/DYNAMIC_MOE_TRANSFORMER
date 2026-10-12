"""Spike P (PILOT): is the pool's advantage just 'more capacity'? Same function-regime stream; replay network with width
128 / 512 / 2048 and depth 1 / 2 (20 optimiser steps per batch). Throwaway diligence script."""
import sys
import time

from dil_common import *  # noqa

from growmoe.baselines import ReplayNet
from growmoe.bench import run_stream
from growmoe.config import CommonCfg
from growmoe.streams import Regime, Stream, _balanced_teacher, _long, _rng, _segment, _tensor, _unit

SEEDS = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2").split(",")]
SIZES = [15, 30]
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


out = {}
t0 = time.time()
for sd in SEEDS:
    for n in SIZES:
        st = function_stream(sd, n)
        for hidden, depth, buf in ((128, 1, 2048), (512, 1, 2048), (2048, 1, 2048), (512, 2, 2048), (512, 1, 8192)):
            hold = []

            def mk():
                p = ReplayNet(CommonCfg(seed=sd, n_classes=C, inner_steps=20, hidden=hidden, depth=depth), replay_size=buf)
                hold.append(p)
                return p

            res = run_stream(mk, st, seed=sd)
            pb = hold[-1]
            params = sum(p.numel() for p in pb.net.parameters())
            kb = (params * BF + pb.buffer.n * D * BF) / 1024.0
            macs = sum(p.numel() for p in pb.net.parameters() if p.dim() == 2)
            out.setdefault((n, hidden, depth, buf), []).append((res.accuracy_stats["avg"], kb, macs))
        print(f"seed {sd} regimes {n} done ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== Function regimes: a single replay network, bigger and deeper (20 steps), {len(SEEDS)} seeds, mean [95% CI] ===")
print(f"{'regimes':>7s} {'width':>6s} {'depth':>5s} {'buffer':>6s} {'accuracy %':>22s} {'memory KB':>10s} {'mult-adds/sample':>17s}")
for (n, h, dep, buf), rows in sorted(out.items()):
    print(f"{n:7d} {h:6d} {dep:5d} {buf:6d} {fmt([r[0] for r in rows]):>22s} {np.mean([r[1] for r in rows]):10.0f} {int(np.mean([r[2] for r in rows])):17,d}")
