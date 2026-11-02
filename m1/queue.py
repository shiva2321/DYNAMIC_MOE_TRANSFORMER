"""CLI queue runner: executes benchmark cells idempotently."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from costlab.env import fixed_threads
from m1.beds import build_bed
from m1.cell import (
    cell_filename,
    execute_cell,
    is_valid_cell_record,
    load_cell_record,
    read_pool_budget,
    record_matches_config,
    save_cell_record,
)

EQUAL_MEMORY_ARMS = ("A3", "A7")


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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run queue of benchmark cells idempotently.")
    parser.add_argument("--split", default="dev", help="Split name ('dev' only; 'report' refused).")
    parser.add_argument("--beds", default="B1", help="Comma-separated bed names (default: B1).")
    parser.add_argument("--ns", default="10", help="Comma-separated regime counts (default: 10).")
    parser.add_argument("--arms", default="A1,A3,A7,A8", help="Comma-separated arms (default: A1,A3,A7,A8).")
    parser.add_argument("--seeds", default="800", help="Comma-separated seeds or ranges (default: 800).")
    parser.add_argument("--out", default="results/m1/dev", help="Output directory.")
    parser.add_argument("--cpus", default="0,2", help="Logical processors to pin, one per physical core (default: 0,2).")
    parser.add_argument("--threads", type=int, default=2, help="Thread count (default: 2).")
    return parser.parse_args(argv)


def run_queue(
    split: str,
    beds: List[str],
    ns: List[int],
    arms: List[str],
    seeds: List[int],
    out: str | Path,
    cpus: Optional[List[int]] = None,
    threads: int = 2,
) -> None:
    """Execute queue of cells, skipping records that are complete, valid and made by the arm as it is configured now."""
    if split == "report":
        raise ValueError("Stage A refuses --split report; report seeds are reserved.")

    cpus_list = cpus if cpus is not None else [0, 2]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A1 runs first per bed, n and seed: its final memory is the budget of the equal-memory arms
    arm_order = {"A1": 0, "A1L": 1, "A3": 2, "A7": 3, "A8": 4}
    sorted_arms = sorted(arms, key=lambda a: arm_order.get(a, 99))

    cells_total = len(beds) * len(ns) * len(seeds) * len(sorted_arms)
    cells_run = 0
    cells_skipped = 0

    print(f"Queue starting: {cells_total} total cells across seeds {seeds}")

    with fixed_threads(n=threads, cpus=cpus_list) as pinning:
        for bed_name in beds:
            for n in ns:
                for seed in seeds:
                    bed = build_bed(bed_name, n, seed)
                    pool_bytes = read_pool_budget(out_dir, bed)

                    for arm in sorted_arms:
                        target_file = out_dir / cell_filename(bed_name, n, arm, seed)
                        budget = pool_bytes if arm in EQUAL_MEMORY_ARMS else None
                        if target_file.exists():
                            try:
                                rec = load_cell_record(target_file)
                            except (OSError, ValueError):
                                rec = None
                            if rec is not None and is_valid_cell_record(rec) and record_matches_config(rec, bed, arm, budget):
                                if arm == "A1":
                                    pool_bytes = int(rec["memory_bytes"])
                                print(f"  [SKIP] {target_file.name} (complete, valid, current configuration)")
                                cells_skipped += 1
                                continue
                            print(f"  [STALE] {target_file.name} is unreadable, invalid or made by another configuration: rerunning")

                        print(f"  [RUN ] {bed_name} N={n} {arm} seed={seed}...", flush=True)
                        t0 = time.time()
                        record = execute_cell(bed=bed, arm_name=arm, pool_memory_bytes=budget, split=split, pinning=pinning)
                        save_cell_record(target_file, record)

                        if arm == "A1":
                            pool_bytes = int(record["memory_bytes"])

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
    cpus = [int(c.strip()) for c in args.cpus.split(",")] if args.cpus else [0, 2]

    run_queue(split=args.split, beds=beds, ns=ns, arms=arms, seeds=seeds, out=args.out, cpus=cpus, threads=args.threads)


if __name__ == "__main__":
    main()
