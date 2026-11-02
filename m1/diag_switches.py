"""Development diagnostic: which switch makes the shipped pool (Baseline v1, arm A1) spawn far more experts than there are regimes on bed B1?

The first seed-800 cell of A1 ended with 67 experts for 10 regimes and 30.7 % accuracy, against 10 experts and 43.8 % for the pre-stage-4 pool (A1L). This script moves
from one pool to the other one switch at a time. Development seeds only (800-804); the result is a diagnosis that feeds the tuning of A1's operating point, not an M1 result.
Run: python -m m1.diag_switches --seeds 800 801 802 803 804 --extras-seed 800
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from costlab.env import environment, fixed_threads
from m1.arms import A1PoolArm
from m1.beds import build_bed
from m1.cell import save_cell_record
from m1.harness_check import final_accuracy

Config = Tuple[str, str, Dict[str, Any]]      # label, variant ("baseline_v1" | "pre_stage4"), overrides of the variant's PhyloCfg fields

CORE: List[Config] = [
    ("A1L pre-stage-4 pool (probe spawn, q 0.02, router off)", "pre_stage4", {}),
    ("A1 Baseline v1 (transfer spawn, q 0.05, router on), 5 steps", "baseline_v1", {}),
    ("A1 with q 0.02", "baseline_v1", {"q": 0.02}),
    ("A1 with the probe spawn rule", "baseline_v1", {"spawn_rule": "probe"}),
]
EXTRAS: List[Config] = [
    ("A1L with the router", "pre_stage4", {"router": True}),
    ("A1L with the transfer spawn rule", "pre_stage4", {"spawn_rule": "transfer"}),
    ("A1L with q 0.05", "pre_stage4", {"q": 0.05}),
    ("A1L with the shortlist from 64 experts", "pre_stage4", {"gate_shortlist_min_experts": 64}),
    ("A1 with 1 step per batch (the class default)", "baseline_v1", {"inner_steps": 1}),
    ("A1 with the router off", "baseline_v1", {"router": False}),
]


def run_config(n_regimes: int, seed: int, cfg: Config) -> Dict[str, Any]:
    """Train one pool on the stream of ``seed`` and return its final accuracy, expert count and memory."""
    label, variant, overrides = cfg
    t0 = time.time()
    bed = build_bed("B1", n_regimes, seed)
    arm = A1PoolArm(bed, variant=variant, **overrides)
    acc = final_accuracy(arm, bed)
    events = dict(Counter(e.kind for e in arm.model.events))              # what the pool did: how often it spawned, shared, merged, grew ...
    row = {"config": label, "variant": variant, "overrides": overrides, "seed": seed, "acc": acc, "experts": len(arm.model.experts),
           "memory_bytes": arm.memory_bytes(), "event_counts": events, "seconds": time.time() - t0}
    print(f"seed {seed} | {label:62s} acc {acc * 100:6.2f}%  experts {row['experts']:3d}  {row['memory_bytes'] / 1e6:6.2f} MB  {row['seconds']:5.1f}s  events {events}", flush=True)
    return row


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="One-switch-at-a-time diagnosis of the shipped pool on B1 (development seeds only).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[800, 801, 802, 803, 804])
    ap.add_argument("--extras-seed", type=int, default=800, help="the six single-switch variants run on this seed only")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="results/m1/dev/diag_switches_B1_n10.json")
    ap.add_argument("--cpus", default="0,2")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    if any(s < 800 or s > 804 for s in args.seeds + [args.extras_seed]):
        raise ValueError("this diagnostic uses the M1 development seeds 800-804 only")

    rows: List[Dict[str, Any]] = []
    with fixed_threads(n=args.threads, cpus=[int(c) for c in args.cpus.split(",")]) as pinning:
        for seed in args.seeds:
            rows += [run_config(args.n, seed, c) for c in CORE]
        rows += [run_config(args.n, args.extras_seed, c) for c in EXTRAS]

    save_cell_record(args.out, {"what": "which switch makes the shipped pool spawn too many experts on B1; development seeds only; not an M1 result",
                                "n_regimes": args.n, "pinning": pinning, "environment": environment(), "rows": rows})
    print(f"saved {Path(args.out)}")


if __name__ == "__main__":
    main()
