"""Dev cells of the from-scratch reader arms (Task 11b): E1 on the dev seeds 100 to 104 and E2 on 100 to 129, written next to the cells of the other readers so that the freeze reads them the same way.

    PYTHONPATH=<repo root> python run_fs_dev.py

Operating points come from ``scratch_reader.HYPER`` (chosen on the tuning seeds 150 to 154). Refuses seeds outside 100 to 129 (the dev range) and the confirmatory pool.
"""

import argparse
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import run_e2 as R
import runlib
import scratch_reader as SR

HERE = Path(__file__).resolve().parent
E1_SEEDS = tuple(range(100, 105))
E2_SEEDS = tuple(range(100, 130))


def run(arms: Sequence[str] = SR.ARMS, e1_seeds: Sequence[int] = E1_SEEDS, e2_seeds: Sequence[int] = E2_SEEDS, out: Path = HERE / "results", stamp: Optional[str] = None,
        e1: Optional[Callable] = None, e2: Optional[Callable] = None, log: Callable[[str], None] = print) -> List[str]:
    bad = [s for s in list(e1_seeds) + list(e2_seeds) if not 100 <= s <= 129]
    if bad:
        raise ValueError(f"the dev cells of the from-scratch arms use seeds 100 to 129 only; refusing {sorted(set(bad))}")
    stamp = stamp or SR.stamp()
    e1, e2 = e1 or SR.run_e1_cell, e2 or SR.run_e2_cell
    done = []
    for arm in arms:
        for seed in e1_seeds:
            r = e1(arm, seed, "dev", Path(out) / "e1", stamp)
            log(f"{arm} E1 seed {seed}: B1a {r['b1a']:.3f} B1b {r['b1b']:.3f} B1c {r['b1c']:.3f}")
            done.append(f"e1 {arm} {seed}")
        for seed in e2_seeds:
            r = e2(arm, seed, "dev", Path(out) / "e2", stamp)
            s = r["machine"][f"q{R.HEADLINE_Q:g}"]["standard"]
            log(f"{arm} E2 seed {seed}: macro F1 {s['macro_f1']['macro_f1']:.3f} false-unseen {s['false_unseen_heldout']:.3f}")
            done.append(f"e2 {arm} {seed}")
    return done


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    runlib.set_threads(3)
    run()
