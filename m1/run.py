"""CLI entrypoint to run one benchmark cell."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from costlab.env import fixed_threads
from m1.beds import build_bed
from m1.cell import (
    cell_filename,
    execute_cell,
    read_pool_budget,
    read_pool_macs,
    save_cell_record,
)
from m1.config_m1 import PINNING_CPUS, PINNING_THREADS
from m1.queue import ALL_ARM_ORDER, load_tuning_map


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one M1 benchmark cell.")
    parser.add_argument("--bed", default="B1", help="Bed identifier (default: B1).")
    parser.add_argument("--n", type=int, required=True, help="Number of regimes.")
    parser.add_argument(
        "--arm",
        required=True,
        choices=ALL_ARM_ORDER + ["A1L"],
        help="Arm name; A1L is the legacy shootout pool and is for the harness check only.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Stream RNG seed.")
    parser.add_argument("--split", default="dev", help="Split name ('dev' only; 'report' is refused).")
    parser.add_argument("--out", default="results/m1/dev", help="Output directory for cell JSON.")
    parser.add_argument("--tuning", default=None, help="Directory or file containing tuning results.")
    parser.add_argument("--profile", default="full", choices=["full", "light"], help="Profiling mode.")
    parser.add_argument("--cpus", default="0,2", help="Logical processors to pin, one per physical core.")
    parser.add_argument("--threads", type=int, default=2, help="Thread count (default: 2).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.split == "report":
        print("Error: --split report is refused; report seeds are reserved.", file=sys.stderr)
        sys.exit(1)

    cpus = [int(c.strip()) for c in args.cpus.split(",")] if args.cpus else list(PINNING_CPUS)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tuning_map = load_tuning_map(args.tuning, args.bed)
    arm_tuning = tuning_map.get(args.arm)

    with fixed_threads(n=args.threads, cpus=cpus) as pinning:
        bed = build_bed(args.bed, args.n, args.seed)

        pool_bytes = None
        pool_macs = None
        if args.arm in ("A3", "A4", "A7", "A7all"):
            pool_bytes = read_pool_budget(out_dir, bed, pool_arm="A1")
            if args.arm == "A4":
                pool_macs = read_pool_macs(out_dir, bed, pool_arm="A1")
        elif args.arm in ("A3-2", "A4-2", "A7-2"):
            pool_bytes = read_pool_budget(out_dir, bed, pool_arm="A2")
            if args.arm == "A4-2":
                pool_macs = read_pool_macs(out_dir, bed, pool_arm="A2")

        record = execute_cell(
            bed=bed,
            arm_name=args.arm,
            pool_memory_bytes=pool_bytes,
            pool_macs_budget=pool_macs,
            split=args.split,
            pinning=pinning,
            profile=args.profile,
            tuning=arm_tuning,
        )

    out_file = out_dir / cell_filename(args.bed, args.n, args.arm, args.seed)
    save_cell_record(out_file, record)

    forgetting = "undefined" if record["forgetting"] is None else f"{record['forgetting'] * 100:.2f}%"
    print(f"Saved cell record to: {out_file.resolve()}")
    print(
        f"  Arm: {record['arm']} | Bed: {record['bed']} N={record['n_regimes']} Seed: {record['seed']} | "
        f"Acc: {record['accuracy'] * 100:.2f}% | Forgetting: {forgetting} | "
        f"Bytes: {record['memory_bytes'] / 1e6:.2f} MB | Time: {record['update_cost']['wall_clock_total']:.2f}s"
    )


if __name__ == "__main__":
    main()
