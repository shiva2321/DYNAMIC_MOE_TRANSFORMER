"""Harness check: does the runner reproduce the old unaliased shootout (N = 10, old development seeds 100-104)?

This is a check of the instrument, not an M1 result: it runs the legacy pool (A1L, the pool of 2026-10-14 before stage 4) and the replay network (A3) given the
memory the old replay record was sized from (the old shootout sized its replay networks from the pool of `runs/pool__<seed>.json`), and compares accuracy, bytes and
the replay buffer with the old result files (`runs_unaliased/pool__<seed>.json` and `runs/replay_eq_w2048_s20__<seed>.json`). The old shootout ran on 3 threads, which
is the default here and reproduces it exactly; the M1 runs use 2 pinned threads. The replay network is a big-matmul network whose trained accuracy depends on the thread
count at the level of a point or less (float summation order); `--thread-probe` shows that on the first seed.
Run: python -m m1.harness_check --seeds 100 101 102 103 104 --thread-probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from costlab.env import environment, fixed_threads
from m1.arms import Arm, build_arm
from m1.beds import Bed, build_bed
from m1.cell import save_cell_record

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "research" / "growmoe_results" / "shootout"
SPEC_BAR = 0.02               # a bar set by the stage-A spec (2 points, as a fraction): an input to the comparison, not a result
SAME_ACC = 1e-6               # two accuracies this close are the same result (the old file keeps float32 noise)
THREAD_PROBE = ((3, (0, 2, 4)), (2, (0, 2)), (2, (0, 2)), (1, (0,)))      # (threads, logical processors); the second 2-thread run shows a fixed setting repeats


def final_accuracy(arm: Arm, bed: Bed) -> float:
    """Train the arm on the whole stream, then return the mean over the regimes of the accuracy on each regime's test set."""
    for x, y, _ in bed:
        arm.observe(x, y)
    tests = bed.test_sets
    accs = [float((arm.predict(x).probs.argmax(dim=-1).cpu() == y.cpu()).float().mean().item()) for x, y in (tests[r] for r in range(bed.n_regimes))]
    return sum(accs) / len(accs)


def thread_probe(seed: int, n_regimes: int, budget_bytes: int, old_acc: float) -> List[Dict[str, Any]]:
    """The replay arm on one seed under several thread counts: does its accuracy depend on them, and does a fixed setting repeat?"""
    bed = build_bed("B1", n_regimes, seed)
    rows = []
    for threads, cpus in THREAD_PROBE:
        with fixed_threads(n=threads, cpus=list(cpus)) as pinning:
            acc = final_accuracy(build_arm("A3", bed, pool_memory_bytes=budget_bytes), bed)
        rows.append({"seed": seed, "threads": threads, "cpus": list(cpus), "blas_limited": pinning.get("blas_limited"), "acc": acc, "old_acc": old_acc,
                     "same_as_old": abs(acc - old_acc) < SAME_ACC})
        print(f"thread probe seed {seed}: {threads} thread(s) on {list(cpus)}: replay {acc * 100:.3f}% (old {old_acc * 100:.3f}%, same: {rows[-1]['same_as_old']})", flush=True)
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Compare the runner with the old unaliased shootout (harness check only).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[100])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="results/m1/harness")
    ap.add_argument("--cpus", default="0,2,4", help="logical processors, one per physical core (default: the old shootout's 3 threads)")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--thread-probe", action="store_true", help="also run the replay arm of the first seed at 3, 2, 2 and 1 threads")
    args = ap.parse_args(argv)

    def load(sub: str, name: str, seed: int) -> Dict[str, Any]:
        return json.loads((OLD / sub / f"N{args.n}" / f"{name}__{seed}.json").read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []
    with fixed_threads(n=args.threads, cpus=[int(c) for c in args.cpus.split(",")]) as pinning:
        for seed in args.seeds:
            old_pool, old_replay = load("runs_unaliased", "pool", seed), load("runs", "replay_eq_w2048_s20", seed)
            budget = int(load("runs", "pool", seed)["bytes"])        # the memory the old replay record was sized from: the pool of the first (head-start) shootout

            pool = build_arm("A1L", build_bed("B1", args.n, seed))
            bed = pool.bed
            pool_acc, pool_bytes = final_accuracy(pool, bed), pool.memory_bytes()

            replay = build_arm("A3", bed, pool_memory_bytes=budget)
            replay_acc, replay_bytes = final_accuracy(replay, bed), replay.memory_bytes()

            rows.append({
                "seed": seed,
                "pool": {"old_acc": old_pool["acc"], "new_acc": pool_acc, "old_bytes": old_pool["bytes"], "new_bytes": pool_bytes,
                         "acc_ok": abs(pool_acc - old_pool["acc"]) <= SPEC_BAR, "bytes_ok": pool_bytes == old_pool["bytes"]},
                "replay": {"old_acc": old_replay["acc"], "new_acc": replay_acc, "old_bytes": old_replay["bytes"], "new_bytes": replay_bytes,
                           "old_buffer": old_replay["buffer"], "new_buffer": replay.buffer, "budget_bytes": budget, "unaliased_pool_bytes": pool_bytes,
                           "acc_ok": abs(replay_acc - old_replay["acc"]) <= SPEC_BAR, "bytes_ok": replay_bytes == old_replay["bytes"]},
            })
            r = rows[-1]
            print(f"seed {seed}: pool {r['pool']['new_acc'] * 100:.3f}% (old {r['pool']['old_acc'] * 100:.3f}%), bytes {r['pool']['new_bytes']} (old {r['pool']['old_bytes']}) | "
                  f"replay {r['replay']['new_acc'] * 100:.3f}% (old {r['replay']['old_acc'] * 100:.3f}%), bytes {r['replay']['new_bytes']} (old {r['replay']['old_bytes']}), "
                  f"buffer {r['replay']['new_buffer']} (old {r['replay']['old_buffer']})", flush=True)

    probe = thread_probe(args.seeds[0], args.n, rows[0]["replay"]["budget_bytes"], rows[0]["replay"]["old_acc"]) if args.thread_probe else []

    result = {
        "what": "harness check against the old unaliased shootout; not an M1 result",
        "n_regimes": args.n, "tolerance_accuracy": SPEC_BAR, "pinning": pinning,
        "environment": environment(), "rows": rows, "replay_thread_probe": probe,
        "all_ok": all(r[k][c] for r in rows for k in ("pool", "replay") for c in ("acc_ok", "bytes_ok")),
    }
    out = Path(args.out) / f"harness_check_N{args.n}_seeds{'_'.join(str(s) for s in args.seeds)}.json"
    save_cell_record(out, result)
    print(f"all within tolerance (accuracy 2 points, pool bytes exact): {result['all_ok']}; saved {out}")


if __name__ == "__main__":
    main()
