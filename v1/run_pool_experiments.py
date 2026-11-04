"""v1/run_pool_experiments.py: Run Part B pool removal experiments across removal sets and dev seeds.

Saves structured JSON results under results/v1/pool_r{rid}_s{seed}.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import torch

from costlab.env import fixed_threads
import m1.beds
import v0.system
from v1.pool_removal import ExposureTracker, compare_pool_removal, remove_labels, scan_traces

REMOVAL_SETS = {
    "R1": [0],
    "R2": [0, 1, 2, 3, 4],
    "R3": [101, 102, 103, 104, 105, 106],
    "R4": [77, 78],
}


def train_full_v0_arm(bed: Any) -> Tuple[Any, ExposureTracker, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Train full V0 pool arm on bed stream with ExposureTracker attached."""
    arm = v0.system.build_v0(bed)
    tracker = ExposureTracker(arm.model)
    stream_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for x_b, y_b, regimes, first_regimes, phase_idx in bed.walk_full(64):
        stream_batches.append((x_b, y_b))
        arm.observe(x_b, y_b)

    return arm, tracker, stream_batches


def run_pool_experiment(
    rid: str,
    seed: int,
    out_dir: Path,
    cached_full: Optional[Tuple[Any, ExposureTracker, List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Execute Part B pool removal evaluation for removal set rid on dev seed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"pool_{rid.lower()}_s{seed}.json"

    # Idempotence check
    if out_file.exists() and not overwrite:
        try:
            cached = json.loads(out_file.read_text(encoding="utf-8"))
            if cached.get("status") == "complete":
                print(f"[SKIP] {out_file.name} already complete.")
                return cached
        except (OSError, ValueError):
            pass

    t_start = time.perf_counter()
    target_classes = REMOVAL_SETS[rid]
    target_set = set(target_classes)

    bed = m1.beds.build_bed("B3seq", 22, seed)

    # 1. Train full pool (or use cached for this seed)
    if cached_full is not None:
        arm_full, tracker, stream_batches = cached_full
    else:
        print(f"Training full V0 pool for seed={seed}...")
        t0 = time.perf_counter()
        arm_full, tracker, stream_batches = train_full_v0_arm(bed)
        print(f"Full V0 pool trained in {time.perf_counter() - t0:.2f}s ({len(arm_full.model.experts)} experts)")

    # Extract removed data from stream batches
    rem_x_list = []
    rem_y_list = []
    for xb, yb in stream_batches:
        mask_rem = torch.isin(yb, torch.tensor(target_classes, dtype=torch.long))
        if bool(mask_rem.any()):
            rem_x_list.append(xb[mask_rem])
            rem_y_list.append(yb[mask_rem])
    x_rem = torch.cat(rem_x_list, dim=0) if rem_x_list else torch.empty((0, bed.d))
    y_rem = torch.cat(rem_y_list, dim=0) if rem_y_list else torch.empty((0,), dtype=torch.long)
    data_rem = (x_rem, y_rem)

    # 2. Build P_removed via certified remove_labels
    print(f"Applying remove_labels for {rid} ({len(target_classes)} classes)...")
    t0 = time.perf_counter()
    arm_removed = copy.deepcopy(arm_full)
    certificate = remove_labels(
        arm_removed.model,
        target_classes,
        data=data_rem,
        tracker=tracker,
    )
    t_removal = time.perf_counter() - t0
    print(f"Removal complete in {t_removal:.2f}s (retained experts: {len(arm_removed.model.experts)})")

    # 3. Train fresh never-trained pool (P_never) on stream without removed examples
    print(f"Training never-trained V0 pool for {rid} on seed={seed}...")
    t0 = time.perf_counter()
    arm_never = v0.system.build_v0(bed)
    for xb, yb, regimes, first_regimes, phase_idx in bed.walk_full(64):
        keep_mask = ~torch.isin(yb, torch.tensor(target_classes, dtype=torch.long))
        if bool(keep_mask.any()):
            arm_never.observe(xb[keep_mask], yb[keep_mask])
    t_never = time.perf_counter() - t0
    print(f"Never-trained pool trained in {t_never:.2f}s ({len(arm_never.model.experts)} experts)")

    # 4. Compare models
    print(f"Comparing P_full, P_removed, P_never on retained and removed test sets...")
    t0 = time.perf_counter()
    comparison = compare_pool_removal(
        p_full=arm_full,
        p_removed=arm_removed,
        p_never=arm_never,
        bed=bed,
        labels=target_classes,
        seed=seed,
    )
    t_comp = time.perf_counter() - t0

    t_total = time.perf_counter() - t_start

    results: Dict[str, Any] = {
        "status": "complete",
        "removal_set": rid,
        "target_classes": target_classes,
        "seed": seed,
        "timing_seconds": {
            "removal": t_removal,
            "never_train": t_never,
            "comparison": t_comp,
            "total": t_total,
        },
        "exposure": {
            "n_direct_exposed": certificate["exposure_summary"]["n_direct_exposed"],
            "n_indirect_exposed": certificate["exposure_summary"]["n_indirect_exposed"],
            "dropped_experts": certificate["exposure_summary"]["dropped_experts"],
            "retained_experts": certificate["exposure_summary"]["retained_experts"],
            "n_collateral_classes_lost": certificate["exposure_summary"]["n_collateral_classes_lost"],
            "collateral_classes_lost": certificate["exposure_summary"]["collateral_classes_lost"],
            "n_retained_in_exposed": certificate["exposure_summary"]["n_retained_in_exposed"],
        },
        "certificate": certificate,
        "comparison": comparison,
    }

    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote {out_file} in {t_total:.2f}s")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Part B pool removal experiments")
    parser.add_argument("--rid", choices=["R1", "R2", "R3", "R4", "all"], default="R1")
    parser.add_argument("--seed", type=int, default=800)
    parser.add_argument("--out-dir", type=str, default="results/v1")
    parser.add_argument("--overwrite", action="store_true", help="Force re-run even if JSON exists")
    args = parser.parse_args()

    with fixed_threads(n=2, cpus=[8, 10]):
        out_dir = Path(args.out_dir)
        bed = m1.beds.build_bed("B3seq", 22, args.seed)
        cached_full = train_full_v0_arm(bed)

        if args.rid == "all":
            for r in ["R1", "R2", "R3", "R4"]:
                run_pool_experiment(r, args.seed, out_dir, cached_full=cached_full, overwrite=args.overwrite)
        else:
            run_pool_experiment(args.rid, args.seed, out_dir, cached_full=cached_full, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
