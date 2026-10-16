# throwaway diagnosis (see the module docstring below): written 2026-10-16 for the merge-repair note
"""Why does a false spawn cost accuracy even when regimes are long?  Per regime: accuracy of the original N alone, the duplicate D alone, the pool's routed answer, the average of N and D, and the best-of-two oracle."""
import sys
from pathlib import Path

MS = Path(__file__).resolve().parent
sys.path.insert(0, str(MS))
sys.path.insert(0, str(MS.parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import healing2_study as H  # noqa: E402
from growmoe.streams_m1 import function_regimes  # noqa: E402

torch.set_num_threads(2)
n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 640
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
INJ = int(sys.argv[3]) if len(sys.argv) > 3 else 3
st = function_regimes(seed, n_regimes=H.REGIMES, d=H.D, n_classes=H.C, n_train=n_train, n_test=200)
pool = H.make_pool("none", seed, None)
nat, inj, last_k, within, seg_nat = [], [], None, 0, 0
for k, seg, idx in st._walk(64, seed):
    if k != last_k:
        last_k, within, seg_nat = k, 0, 0
    within += 1
    before = set(pool.experts)
    pool.observe(seg.x[idx], seg.y[idx])
    new = sorted(set(pool.experts) - before)
    nat += new
    for e in new:
        pool.regime_of[e] = seg.regime
    seg_nat += len(new)
    if seg.first_visit and within == INJ and seg_nat:
        new = H.inject(pool, seg.x[idx], seg.y[idx])
        inj += new
        for e in new:
            pool.regime_of[e] = seg.regime
tests = st.test_sets()
rows = []
for D in inj:
    reg = pool.regime_of[D]
    N = next(e for e in nat if pool.regime_of.get(e) == reg)
    xt, yt = tests[reg]
    xn = pool._normalize(xt.float())
    with torch.no_grad():
        pN, pD = pool.experts[N].module.proba(xn), pool.experts[D].module.proba(xn)
        routed = pool.predict(xt)
    y = yt.cpu()
    aN, aD = float((pN.argmax(1).cpu() == y).float().mean()), float((pD.argmax(1).cpu() == y).float().mean())
    aR = float((routed.probs.argmax(1).cpu() == y).float().mean())
    aE = float(((pN + pD).argmax(1).cpu() == y).float().mean())
    aO = float((((pN.argmax(1) == yt) | (pD.argmax(1) == yt)).float()).mean())
    share = float((routed.expert == D).float().mean())
    rows.append((reg, N, D, aN, aD, aR, aE, aO, share, pool.experts[N].n_assigned, pool.experts[D].n_assigned))
print(f"n_train {n_train} seed {seed}: per regime  N alone | D alone | routed | N+D average | oracle(either right) | share routed to D | samples assigned N / D")
for r in rows:
    print(f"regime {r[0]:2d} N={r[1]:2d} D={r[2]:2d}  {100*r[3]:5.1f} | {100*r[4]:5.1f} | {100*r[5]:5.1f} | {100*r[6]:5.1f} | {100*r[7]:5.1f} | {100*r[8]:4.0f} % | {r[9]} / {r[10]}")
a = np.array([r[3:9] for r in rows])
print("mean       ", " | ".join(f"{100*v:5.1f}" for v in a.mean(0)[:5]), f"| {100*a.mean(0)[5]:4.0f} %")
