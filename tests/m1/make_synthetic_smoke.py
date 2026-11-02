"""Write SYNTHETIC cell records with the real schema 1.2.0 for a smoke run of the report: python -m tests.m1.make_synthetic_smoke [out_dir].

The numbers are invented (they only exercise the code) and every record carries ``split = "synthetic"``, so the report labels all its output SYNTHETIC.
Never write under results/m1/dev; the default folder is results/m1/smoke/synthetic_cells. Held-default pair cells have the arm ids H_<candidate> (as the real queue writes them).
"""

from __future__ import annotations

import sys
from pathlib import Path

from m1.cell import save_cell_record
from tests.m1.test_verdict import make_cell

BEDS = [("B1", 10), ("B1", 30), ("B1", 100), ("B2", 6), ("B3seq", 22), ("B3mix", 22), ("B3drift", 22)]
ARMS = ["A1", "A1q", "A2", "A3", "A4", "A5a", "A5b", "A6", "A7", "A7all", "A8", "A3-2", "A4-2", "A7-2"]
PAIR_CANDIDATES = ("shipped", "shipped+gap", "shipped+skip", "shipped+chance", "U-real", "ULDA")
SEEDS = (800, 801)


def main(out: str = "results/m1/smoke/synthetic_cells") -> int:
    out_dir = Path(out)
    if "dev" in out_dir.parts or "report" in out_dir.parts:
        raise ValueError("synthetic records must not be written next to real cells")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()
    count = 0
    for bed, n in BEDS:
        for arm in ARMS:
            for seed in SEEDS:
                pool = arm in ("A1", "A1q", "A2")
                rec = make_cell(
                    bed=bed, n_regimes=n, arm=arm, seed=seed, split="synthetic",
                    acc=0.35 + (0.01 if "2" in arm else 0.0) + (0.05 if arm == "A8" else 0.0),
                    up_wall=10.0 if pool else 5.0, lat_b1=0.005 if pool else 0.003,
                    pool_time_split={"observe": 10.0, "train_expert": 2.5, "router_refit": 1.2} if pool else None,
                    memory_bytes_by_regime={str(r): (r + 1) * 100_000 for r in range(n)}, experts=max(1, n // 2),
                    profile="full" if seed == SEEDS[0] else "det",              # timing only on the first seed, as in the real report phase
                )
                save_cell_record(out_dir / f"{bed}_n{n}_{arm}__seed{seed}.json", rec)
                count += 1
        for cand in PAIR_CANDIDATES:
            for seed in SEEDS:
                rec = make_cell(bed=bed, n_regimes=n, arm=f"H_{cand}", seed=seed, candidate=cand, acc=0.36, up_wall=8.0, lat_b1=0.004, split="synthetic", profile="full" if seed == SEEDS[0] else "det")
                save_cell_record(out_dir / f"{bed}_n{n}_H_{cand}__seed{seed}.json", rec)
                count += 1
    print(f"Wrote {count} SYNTHETIC cell records under {out_dir}")
    return count


if __name__ == "__main__":
    main(*sys.argv[1:2])
