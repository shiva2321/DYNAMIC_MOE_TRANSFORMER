"""Stage-4 spike C2: which "gap" is a robust capacity signal?  (throwaway; no product file touched)

Spike C found that the gap between the honest prequential loss and the loss on replayed stored samples rises monotonically with the width of the expert (0.03 at 16 units ... 0.4 at 512).
That gap depends on how many samples the reservoir holds. Candidates that need no big reservoir: (a) the same replay gap with a reservoir of only 256 samples, (b) the within-batch fit gap
(honest loss on a new batch before its 5 steps against the loss on the same batch after them), (c) the recent-window gap (honest loss on the new batch against the loss on the previous 8
batches, 512 samples, which the model saw 1 to 8 batches ago). All are measured over the last 100 of 900 batches, for 5 widths, reservoirs of 256 and 20,000, 3 seeds.

    python spike_gap_definition.py --seed 0
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent / "diagnosis"
sys.path.insert(0, str(DIAG))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import diag_lib as D  # noqa: E402
from diag_lib import A, L  # noqa: E402
from growmoe.config import CommonCfg  # noqa: E402
from growmoe.pools.base import BasePool  # noqa: E402

OUT = HERE / "gap_definition"
WIDTHS = (16, 32, 64, 128, 512)
BATCHES = 900


def loss_of(net, x, y):
    with torch.no_grad():
        return float(F.cross_entropy(net(x), y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    tr = L.build_train("FULL", a.seed, 8000)
    X, Y = torch.from_numpy(D.unit(tr.x())), torch.from_numpy(tr.y).long()
    order = np.random.default_rng([a.seed, 0x5F1]).permutation(len(X))
    for res_size in (256, 20000):
        for w in WIDTHS:
            stem = OUT / f"w{w}_r{res_size}_s{a.seed}"
            if D.valid_json(stem.with_suffix(".json")):
                continue
            t0 = time.time()
            pool = BasePool(CommonCfg(d=A.D_MODEL, n_classes=A.N_CLASS, lr=3e-3, inner_steps=5, replay_size=res_size, seed=a.seed))
            D.ARCHS["_tmp"] = ("expert", w, 1, True)
            eid = pool.add_expert(D.build_net("_tmp", a.seed), parent=None, state="active").id
            rec = pool.experts[eid]
            fifo = deque(maxlen=8)
            rows = []
            for k in range(BATCHES):
                b = torch.from_numpy(order[k * 64:(k + 1) * 64])
                xb, yb = X[b], Y[b]
                pre = loss_of(rec.module, xb, yb)
                rep = loss_of(rec.module, *rec.replay.sample(64)) if rec.replay.n >= 64 else float("nan")
                rcn = loss_of(rec.module, torch.cat([f[0] for f in fifo]), torch.cat([f[1] for f in fifo])) if len(fifo) == 8 else float("nan")
                pool.train_expert(eid, xb, yb, replay=True)
                post = loss_of(rec.module, xb, yb)
                fifo.append((xb, yb))
                rows.append((pre, rep, rcn, post))
            r = np.array(rows[-100:])
            pre, rep, rcn, post = r.mean(0)
            out = {"width": w, "reservoir": res_size, "seed": a.seed, "prequential": float(pre), "replay": float(rep), "recent": float(rcn), "post_batch": float(post),
                   "gap_replay": float((pre - rep) / pre), "gap_recent": float((pre - rcn) / pre), "gap_batch": float((pre - post) / pre), "seconds": time.time() - t0}
            D.atomic_write_text(stem.with_suffix(".json"), json.dumps(out))
            print(f"[w{w} r{res_size} s{a.seed}] preq {pre:.3f}  gap replay {out['gap_replay']:.3f}  recent {out['gap_recent']:.3f}  within-batch {out['gap_batch']:.3f}   {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
