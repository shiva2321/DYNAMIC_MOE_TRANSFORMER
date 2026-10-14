"""Study 1: why does GrowMoE learn arithmetic worse than a dense network? Separate the ARCHITECTURE from the TRAINING RECIPE.

Same data (FULL lengths, mixed operators, 8,000 problems per operator = 298,000 examples), same held-out test grid, same exact-match scoring as the benchmark.
A grid of (network, recipe) cells, two seeds each:

  networks   E128    the shipped GrowMoE expert (35 k parameters)        E512   the tuned expert (134 k)
             S256x2  MLP-S of the benchmark (102 k)  S256x2n same on unit-length inputs   L1024x3  MLP-L (2.2 M)
  recipes    D40     the benchmark's dense recipe: 40 epochs, shuffled batches of 512, AdamW one-cycle schedule  (12 M example-gradients)
             D10     the same, 10 epochs (3 M example-gradients = the same compute as the pool's recipe)
             O5-r256       the pool's recipe as shipped: batches of 64 seen once, 5 steps per batch on that batch + 64 replayed from a reservoir of 256, Adam 3e-3 constant
             O5-r20000     the tuned pool recipe: the same with a reservoir of 20,000
             O1-r20000     one step per batch (no repeats) with the 20,000 reservoir
             O5-r20000-anneal   as O5-r20000 with the rate decayed linearly to zero over the stream

    python run_ladder.py --seeds 0,1 --shard 0/3
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import diag_lib as D  # noqa: E402
from diag_lib import A, L  # noqa: E402

OUT = HERE / "ladder"

RECIPES = {
    "D40": lambda arch: D.Offline(arch, 40),
    "D10": lambda arch: D.Offline(arch, 10),
    "O5-r256": lambda arch: D.Online(arch, 5, 256),
    "O5-r20000": lambda arch: D.Online(arch, 5, 20000),
    "O1-r20000": lambda arch: D.Online(arch, 1, 20000),
    "O5-r20000-anneal": lambda arch: D.Online(arch, 5, 20000, anneal=True),
    # the memory sweep and the equal-stored-examples control (second batch of cells)
    "O5-r2000": lambda arch: D.Online(arch, 5, 2000),
    "O5-r100000": lambda arch: D.Online(arch, 5, 100000),
    "O5-r300000": lambda arch: D.Online(arch, 5, 300000),
    "M2000-D40": lambda arch: D.OfflineSubset(arch, 2000, 160),              # SUPERSEDED (only 160 steps: under-trained); kept so old result files stay readable
    "M20000-D40": lambda arch: D.OfflineSubset(arch, 20000, 1600),           # SUPERSEDED (1,600 steps)
    "M2000-S3k": lambda arch: D.OfflineSubset(arch, 2000, 3000),
    "M2000-S23k": lambda arch: D.OfflineSubset(arch, 2000, 23000),
    "M20000-S3k": lambda arch: D.OfflineSubset(arch, 20000, 3000),
    "M20000-S23k": lambda arch: D.OfflineSubset(arch, 20000, 23000),
}

from ladder_cells import CELLS, CELLS_EXTRA, CELLS_SUBSET  # noqa: E402  (the cell lists live in a module without heavy imports; queue.py reads them too)


def run_cell(arch, recipe, seed, tr, test, seen, per_cell):
    stem = OUT / f"{arch}__{recipe}__s{seed}"
    if D.valid_json(stem.with_suffix(".json")):
        return
    learner = RECIPES[recipe](arch)
    t0 = time.time()
    info = learner.fit(tr, seed)
    res, _ = L.evaluate(learner, test, per_cell, seen)
    s = D.summarise(res)
    res.update(arch=arch, recipe=recipe, seed=seed, train=info, summary=s, total_seconds=time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    D.atomic_write_text(stem.with_suffix(".json"), json.dumps(res))
    print(f"[{arch:8s} {recipe:18s} s{seed}] unseen {100 * s['mean_exact']:5.1f}  seen {100 * s['seen_exact']:5.1f}  add {100 * s['exact_by_op']['add']:5.1f} sub {100 * s['exact_by_op']['sub']:5.1f} "
          f"mul {100 * s['exact_by_op']['mul']:5.1f} max {100 * s['exact_by_op']['max']:5.1f}  train-CE {info['train_loss']:.3f}  params {info['params']:,}  {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--n-per-op", type=int, default=8000)
    ap.add_argument("--per-cell", type=int, default=200)
    ap.add_argument("--seen-per-op", type=int, default=500)
    ap.add_argument("--extra", action="store_true", help="run the memory-sweep cells instead of the main grid")
    ap.add_argument("--all", action="store_true", help="run the main grid and then the memory-sweep cells")
    ap.add_argument("--subset", action="store_true", help="run the equal-stored-examples offline cells (M<N>-S<steps>)")
    a = ap.parse_args()
    k, n = (int(v) for v in a.shard.split("/"))
    test = L.build_test(a.per_cell)
    cells = CELLS_SUBSET if a.subset else (CELLS + CELLS_EXTRA if a.all else (CELLS_EXTRA if a.extra else CELLS))
    jobs = [(arch, recipe, int(s)) for s in a.seeds.split(",") for arch, recipe in cells]
    for j, (arch, recipe, seed) in enumerate(jobs):
        if j % n != k:
            continue
        tr = L.build_train("FULL", seed, a.n_per_op)
        seen = L.build_seen(tr, a.seen_per_op, seed)
        run_cell(arch, recipe, seed, tr, test, seen, a.per_cell)


if __name__ == "__main__":
    main()
