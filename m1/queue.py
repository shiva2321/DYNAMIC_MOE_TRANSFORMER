"""CLI queue runner: executes benchmark cells idempotently."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from costlab.env import fixed_threads
from m1.beds import build_bed
from m1.cell import (
    cell_filename,
    execute_cell,
    is_valid_cell_record,
    load_cell_record,
    read_pool_budget,
    read_pool_macs,
    record_matches_config,
    save_cell_record,
)
from m1.config_m1 import H_CANDIDATES_ALL_BEDS, H_CANDIDATES_B3, PINNING_CPUS, PINNING_THREADS

ALL_ARM_ORDER = [
    "A1",
    "A1q",
    "A2",
    "A3",
    "A4",
    "A5a",
    "A5b",
    "A6",
    "A7",
    "A7all",
    "A8",
    "A3-2",
    "A4-2",
    "A7-2",
]
EQUAL_MEMORY_ARMS_R1 = ("A3", "A7")
EQUAL_MEMORY_ARMS_R2 = ("A3-2", "A7-2")
COMPUTE_MATCHED_ARMS_R1 = ("A4",)
COMPUTE_MATCHED_ARMS_R2 = ("A4-2",)


def h_arms_for(bed_name: str) -> List[str]:
    """Arm ids of the held-default pair cells for a bed (shorthand ``H`` on the command line)."""
    cands = list(H_CANDIDATES_ALL_BEDS) + (list(H_CANDIDATES_B3) if bed_name.startswith("B3") else [])
    return [f"H_{c}" for c in cands]


def parse_seeds(text: str) -> List[int]:
    """Parse comma-separated seed specifications and ranges (e.g. '800', '800-804')."""
    seeds: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_str, _, hi_str = part.partition("-")
            seeds.extend(range(int(lo_str), int(hi_str) + 1))
        else:
            seeds.append(int(part))
    return sorted(list(dict.fromkeys(seeds)))


def load_tuning_map(tuning_arg: Optional[str], bed_name: str) -> Dict[str, Any]:
    """Load chosen tuning configurations for bed from file or directory."""
    if not tuning_arg:
        return {}
    p = Path(tuning_arg)
    file_path = p / f"{bed_name}.json" if p.is_dir() else p
    if not file_path.exists():
        raise FileNotFoundError(f"--tuning was given but {file_path} does not exist: run m1.tune first (cells made without it would be untuned).")
    try:
        data = load_cell_record(file_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"The tuning file {file_path} cannot be read ({exc}); refusing to run untuned arms.") from exc
    return {a: info["chosen"] for a, info in data.get("arms", {}).items() if "chosen" in info}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run queue of benchmark cells idempotently.")
    parser.add_argument("--split", default="dev", help="Split name ('dev' only; 'report' refused).")
    parser.add_argument("--beds", default="B1", help="Comma-separated bed names (default: B1).")
    parser.add_argument("--ns", default="10", help="Comma-separated regime counts (default: 10).")
    parser.add_argument("--arms", default=",".join(ALL_ARM_ORDER), help="Comma-separated arms.")
    parser.add_argument("--seeds", default="800", help="Comma-separated seeds or ranges (default: 800).")
    parser.add_argument("--out", default="results/m1/dev", help="Output directory.")
    parser.add_argument("--tuning", default=None, help="Directory or file containing tuning results.")
    parser.add_argument("--profile", default="full", choices=["full", "light"], help="Profiling mode.")
    parser.add_argument("--cpus", default="0,2", help="Logical processors to pin, one per physical core.")
    parser.add_argument("--threads", type=int, default=2, help="Thread count (default: 2).")
    return parser.parse_args(argv)


def run_queue(
    split: str,
    beds: List[str],
    ns: List[int],
    arms: List[str],
    seeds: List[int],
    out: str | Path,
    tuning_dir: Optional[str] = None,
    profile: str = "full",
    cpus: Optional[List[int]] = None,
    threads: int = 2,
) -> None:
    """Execute queue of cells, skipping records that are complete, valid, and match current config."""
    if split == "report":
        raise ValueError("Stage B refuses --split report; report seeds are reserved.")

    cpus_list = cpus if cpus is not None else list(PINNING_CPUS)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enforce strict dependency ordering: A1, A1q, A2, baselines, then -2 baselines
    arm_rank = {a: i for i, a in enumerate(ALL_ARM_ORDER)}
    sorted_arms = sorted(arms, key=lambda a: arm_rank.get(a, 99))

    cells_total = len(beds) * len(ns) * len(seeds) * len(sorted_arms)
    cells_run = 0
    cells_skipped = 0

    print(f"Queue starting: {cells_total} total cells across seeds {seeds}")

    with fixed_threads(n=threads, cpus=cpus_list) as pinning:
        for bed_name in beds:
            tuning_map = load_tuning_map(tuning_dir, bed_name)
            for n in ns:
                for seed in seeds:
                    bed = build_bed(bed_name, n, seed)
                    arms_here = [a for a in sorted_arms if a != "H"] + (h_arms_for(bed_name) if "H" in sorted_arms else [])
                    pool_bytes_r1 = read_pool_budget(out_dir, bed, pool_arm="A1", tuning=tuning_map.get("A1"))
                    pool_macs_r1 = read_pool_macs(out_dir, bed, pool_arm="A1", tuning=tuning_map.get("A1"))
                    pool_bytes_r2 = read_pool_budget(out_dir, bed, pool_arm="A2", tuning=tuning_map.get("A2"))
                    pool_macs_r2 = read_pool_macs(out_dir, bed, pool_arm="A2", tuning=tuning_map.get("A2"))

                    for arm in arms_here:
                        target_file = out_dir / cell_filename(bed_name, n, arm, seed)
                        arm_tuning = tuning_map.get("A1q" if arm.startswith("H_") else arm)       # pair cells run at the base operating point of A1q
                        if tuning_dir and arm_tuning is None:
                            raise RuntimeError(f"{arm} is not in the tuning file of {bed_name}: tune it first (m1.tune) or run without --tuning.")

                        # Determine memory and compute budgets
                        mem_budget: Optional[int] = None
                        macs_budget: Optional[int] = None
                        if arm in EQUAL_MEMORY_ARMS_R1 or arm in COMPUTE_MATCHED_ARMS_R1:
                            mem_budget = pool_bytes_r1
                        elif arm in EQUAL_MEMORY_ARMS_R2 or arm in COMPUTE_MATCHED_ARMS_R2:
                            mem_budget = pool_bytes_r2

                        if arm in COMPUTE_MATCHED_ARMS_R1:
                            macs_budget = pool_macs_r1
                        elif arm in COMPUTE_MATCHED_ARMS_R2:
                            macs_budget = pool_macs_r2

                        needs_pool = {"A3": "A1", "A4": "A1", "A7": "A1", "A3-2": "A2", "A4-2": "A2", "A7-2": "A2"}.get(arm)
                        if needs_pool is not None and mem_budget is None:
                            raise RuntimeError(f"{arm} on {bed_name} N={n} seed {seed} needs a current {needs_pool} record in {out_dir} (its memory is the budget): run {needs_pool} first.")
                        if arm in COMPUTE_MATCHED_ARMS_R1 + COMPUTE_MATCHED_ARMS_R2 and macs_budget is None:
                            raise RuntimeError(f"{arm} on {bed_name} N={n} seed {seed} needs the counted multiply-adds of a current {needs_pool} full-profile record in {out_dir}.")

                        if target_file.exists():
                            try:
                                rec = load_cell_record(target_file)
                            except (OSError, ValueError):
                                rec = None
                                print(f"  [CORRUPT] {target_file.name} cannot be read as JSON")

                            if (
                                rec is not None
                                and is_valid_cell_record(rec)
                                and record_matches_config(
                                    rec,
                                    bed,
                                    arm,
                                    pool_memory_bytes=mem_budget,
                                    pool_macs_budget=macs_budget,
                                    tuning=arm_tuning,
                                    profile=profile,
                                )
                            ):
                                if arm == "A1":
                                    pool_bytes_r1 = int(rec["memory_bytes"])
                                    pool_macs_r1 = rec.get("update_cost", {}).get("counted_macs")
                                elif arm == "A2":
                                    pool_bytes_r2 = int(rec["memory_bytes"])
                                    pool_macs_r2 = rec.get("update_cost", {}).get("counted_macs")

                                print(f"  [SKIP] {target_file.name} (complete, valid, current configuration)")
                                cells_skipped += 1
                                continue

                            print(f"  [STALE] {target_file.name} is unreadable, invalid, or made by another configuration: rerunning")

                        print(f"  [RUN ] {bed_name} N={n} {arm} seed={seed} (profile={profile})...", flush=True)
                        t0 = time.time()
                        record = execute_cell(
                            bed=bed,
                            arm_name=arm,
                            pool_memory_bytes=mem_budget,
                            pool_macs_budget=macs_budget,
                            split=split,
                            pinning=pinning,
                            profile=profile,
                            tuning=arm_tuning,
                        )
                        save_cell_record(target_file, record)

                        if arm == "A1":
                            pool_bytes_r1 = int(record["memory_bytes"])
                            pool_macs_r1 = record.get("update_cost", {}).get("counted_macs")
                        elif arm == "A2":
                            pool_bytes_r2 = int(record["memory_bytes"])
                            pool_macs_r2 = record.get("update_cost", {}).get("counted_macs")

                        dt = time.time() - t0
                        print(
                            f"  [DONE] {target_file.name}: acc {record['accuracy'] * 100.0:.2f}% | "
                            f"{record['memory_bytes'] / 1e6:.2f} MB | {dt:.1f}s",
                            flush=True,
                        )
                        cells_run += 1

    print(f"Queue finished: {cells_run} cells executed, {cells_skipped} cells skipped.")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.split == "report":
        print("Error: --split report is refused; report seeds are reserved.", file=sys.stderr)
        sys.exit(1)

    beds = [b.strip() for b in args.beds.split(",") if b.strip()]
    ns = [int(n.strip()) for n in args.ns.split(",") if n.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = parse_seeds(args.seeds)
    cpus = [int(c.strip()) for c in args.cpus.split(",")] if args.cpus else list(PINNING_CPUS)

    run_queue(
        split=args.split,
        beds=beds,
        ns=ns,
        arms=arms,
        seeds=seeds,
        out=args.out,
        tuning_dir=args.tuning,
        profile=args.profile,
        cpus=cpus,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
