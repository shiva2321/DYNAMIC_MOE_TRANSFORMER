"""Study 5b: would growing the expert have paid off? A direct test of the growth policy's horizon.

The growth policy judges a candidate by a 480-step probe on the replay reservoir. The question here is whether capacity added in the MIDDLE of the stream pays off over the rest of the stream,
which is what "grow as needed" ought to mean. Same stream, same recipe (the pool's: batches of 64 seen once, 5 steps per batch with an equal share of replay from a 20,000 reservoir, Adam 3e-3).
At batch n/2 the 128-wide expert is grown exactly as the pool does on a commit (function-preserving ``widen`` / ``deepen``, optimiser rebuilt), with the candidate sizes the policy would propose
(+64, +384, +896 hidden units, or one more block) and the stream goes on.  Controls: the 128-wide expert left alone, and 512-wide from the start.

    python probe_widen_midstream.py --seed 0
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import diag_lib as D  # noqa: E402
from diag_lib import A, L  # noqa: E402
from growmoe.config import CommonCfg  # noqa: E402
from growmoe.pools.base import BasePool  # noqa: E402

OUT = HERE / "widen_midstream"


def run_arm(arm, hidden0, grow, tr, test, seen, seed, per_cell):
    torch.set_num_threads(3)
    cfg = CommonCfg(d=A.D_MODEL, n_classes=A.N_CLASS, lr=3e-3, inner_steps=5, replay_size=20000, seed=seed)
    pool = BasePool(cfg)
    D.ARCHS["_tmp"] = ("expert", hidden0, 1, True)
    net = D.build_net("_tmp", seed)
    eid = pool.add_expert(net, parent=None, state="active").id
    x, y = D.unit(tr.x()), tr.y
    X, Y = torch.from_numpy(x), torch.from_numpy(y).long()
    order = np.random.default_rng([seed, 0x5F1]).permutation(len(x))
    nb = math.ceil(len(order) / 64)
    t0 = time.time()
    for k, i in enumerate(range(0, len(order), 64)):
        if grow and k == nb // 2:
            rec = pool.experts[eid]
            if grow[0] == "widen":
                rec.module.widen(grow[1])
            else:
                rec.module.deepen()
            rec.opt = pool.make_optimizer(rec.module, rec.lr_mult)             # what the pool does after a committed growth
        b = torch.from_numpy(order[i:i + 64])
        pool.train_expert(eid, X[b], Y[b], replay=True)
    learner = D.NetLearner()
    learner.net, learner.arch, learner.device = pool.experts[eid].module, "E128", torch.device("cpu")
    res, _ = L.evaluate(learner, test, per_cell, seen)
    s = D.summarise(res)
    n_params = sum(p.numel() for p in learner.net.parameters())
    print(f"[{arm} s{seed}] unseen {100 * s['mean_exact']:.1f}  seen {100 * s['seen_exact']:.1f}  params {n_params:,}  {time.time() - t0:.0f}s", flush=True)
    return {"unseen_exact": s["mean_exact"], "seen_exact": s["seen_exact"], "params": n_params, "exact_by_op": s["exact_by_op"], "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-cell", type=int, default=200)
    ap.add_argument("--start", type=int, default=128, help="hidden units of the starting expert (128 = the shipped expert, 16 = a capacity-starved one)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    stem = OUT / (f"s{a.seed}" if a.start == 128 else f"h{a.start}_s{a.seed}")
    if stem.with_suffix(".json").exists():
        return
    tr = L.build_train("FULL", a.seed, 8000)
    test = L.build_test(a.per_cell)
    seen = L.build_seen(tr, 500, a.seed)
    if a.start == 128:
        arms = [("E128 left alone", 128, None), ("E512 from the start", 512, None), ("E128, widen +64 at half", 128, ("widen", 64)), ("E128, widen +384 at half", 128, ("widen", 384)),
                ("E128, widen +896 at half", 128, ("widen", 896)), ("E128, one more block at half", 128, ("deepen", 0))]
    else:
        s0 = a.start
        arms = [(f"E{s0} left alone", s0, None), (f"E{s0} widened to 128 at half", s0, ("widen", 128 - s0)), (f"E{s0} widened to 512 at half", s0, ("widen", 512 - s0)),
                (f"E{s0}, one more block at half", s0, ("deepen", 0)), ("E128 from the start", 128, None), ("E512 from the start", 512, None)]
    out = {"seed": a.seed, "start_hidden": a.start, "arms": {}}
    for name, h0, g in arms:
        out["arms"][name] = run_arm(name, h0, g, tr, test, seen, a.seed, a.per_cell)
        D.atomic_write_text(stem.with_suffix(".partial.json"), json.dumps(out))
    D.atomic_write_text(stem.with_suffix(".json"), json.dumps(out))


if __name__ == "__main__":
    main()
