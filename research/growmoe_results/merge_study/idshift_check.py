"""Is the accuracy loss of a pool with a false spawn caused by the false spawn or by the bookkeeping (expert ids)?  Arms: clean, id_shift_only (the ids advance as if a spawn had happened, nothing is founded), inject_then_remove,
inject_keep.  usage: python idshift_check.py N_TRAIN INJECT_BATCH SEEDS [POOL_SEED_OFFSET]   (offset 0: the head start of the function-regime bench is present; offset 7 (default): it is not)"""
import sys
from pathlib import Path
MS = Path(__file__).resolve().parent
sys.path.insert(0, str(MS)); sys.path.insert(0, str(MS.parents[2]))
import numpy as np, torch
import healing2_study as H
from growmoe.streams_m1 import function_regimes
torch.set_num_threads(2)
n_train, INJ = int(sys.argv[1]), int(sys.argv[2])
seeds = [int(s) for s in sys.argv[3].split(",")]
H.POOL_SEED_OFFSET = int(sys.argv[4]) if len(sys.argv) > 4 else 7        # 0 reproduces the aliased numbers of the first corrected study (clean 76.2 / id shift 70.8 / injection removed 70.8 / injection kept 70.0)

def build(variant, seed):
    st = function_regimes(seed, n_regimes=H.REGIMES, d=H.D, n_classes=H.C, n_train=n_train, n_test=200)
    pool = H.make_pool("none" if variant != "clean" else "clean", seed, None)
    nat, last_k, within, seg_nat = [], None, 0, 0
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
        if variant != "clean" and seg.first_visit and within == INJ and seg_nat:
            if variant == "id_shift_only":
                pool._next_id += 1
                continue
            new = H.inject(pool, seg.x[idx], seg.y[idx])
            if variant == "inject_then_remove":
                for e in new:
                    pool.remove_expert(e)
    tests = st.test_sets()
    own = []
    routed = []
    for e in nat:
        reg = pool.regime_of[e]
        xt, yt = tests[reg]
        with torch.no_grad():
            p = pool.experts[e].module.proba(pool._normalize(xt.float()))
        own.append(float((p.argmax(1).cpu() == yt.cpu()).float().mean()))
    for r, (xt, yt) in tests.items():
        with torch.no_grad():
            routed.append(float((pool.predict(xt).probs.argmax(1).cpu() == yt.cpu()).float().mean()))
    return np.mean(own), np.mean(routed), len(pool.experts)

res = {}
for v in ("clean", "id_shift_only", "inject_then_remove", "inject_keep"):
    res[v] = [build(v, s) for s in seeds]
    own = [100 * r[0] for r in res[v]]; routed = [100 * r[1] for r in res[v]]
    print(f"n_train {n_train} inject at {INJ} | {v:20s} own expert alone {np.mean(own):5.1f} (+-{np.std(own, ddof=1)/np.sqrt(len(own)) if len(own)>1 else 0:.1f})   whole pool routed {np.mean(routed):5.1f} (+-{np.std(routed, ddof=1)/np.sqrt(len(routed)) if len(routed)>1 else 0:.1f})   experts {np.mean([r[2] for r in res[v]]):.0f}", flush=True)
