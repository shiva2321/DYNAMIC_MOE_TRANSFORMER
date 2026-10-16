import sys
from pathlib import Path
MS = Path(__file__).resolve().parent
sys.path.insert(0, str(MS)); sys.path.insert(0, str(MS.parents[2]))
import numpy as np, torch
import healing2_study as H
H.POOL_SEED_OFFSET = 0          # this script chooses the pool seed itself: stream seed (aliased) or stream seed + 7 (not aliased)
from growmoe.experts import MLPExpert
from growmoe.streams import _balanced_teacher
from growmoe.streams_m1 import function_regimes
torch.set_num_threads(2)

# 1. is the first layer of the expert with seed S the teacher's first layer?
for S in (0, 1, 5):
    t = _balanced_teacher(8, 1, 32, 4, seed=S)
    e = MLPExpert(d_in=32, hidden=32, depth=1, n_classes=4, seed=S)
    same = torch.equal(t.blocks[0].w1.data, e.blocks[0].w1.data[:8])
    e2 = MLPExpert(d_in=32, hidden=32, depth=1, n_classes=4, seed=S + 777)
    print(f"seed {S}: expert's first 8 hidden units == teacher's hidden units: {same}   (an expert with another seed: {torch.equal(t.blocks[0].w1.data, e2.blocks[0].w1.data[:8])})")

# 2. does it matter for a clean pool?  stream seed 0; pool seed 0 (ids alias the teachers) versus pool seed 7 (they do not)
def clean(stream_seed, pool_seed, n_train):
    st = function_regimes(stream_seed, n_regimes=H.REGIMES, d=H.D, n_classes=H.C, n_train=n_train, n_test=200)
    pool = H.make_pool("clean", pool_seed, None)
    for k, seg, idx in st._walk(64, stream_seed):
        pool.observe(seg.x[idx], seg.y[idx])
    accs = []
    for r, (xt, yt) in st.test_sets().items():
        with torch.no_grad():
            accs.append(float((pool.predict(xt).probs.argmax(1).cpu() == yt.cpu()).float().mean()))
    return 100 * np.mean(accs), len(pool.experts)

for n_train in (640, 1280):
    for stream_seed in (0, 1, 2):
        a = clean(stream_seed, stream_seed, n_train)
        b = clean(stream_seed, stream_seed + 7, n_train)
        print(f"n_train {n_train} stream seed {stream_seed}: pool seed = stream seed (aliased) {a[0]:.1f} ({a[1]} experts) | pool seed = stream seed + 7 (not aliased) {b[0]:.1f} ({b[1]} experts)", flush=True)
