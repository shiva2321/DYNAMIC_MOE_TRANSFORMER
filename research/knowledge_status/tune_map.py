"""Tuning of the learned map (Task 9) on dev seeds that are NOT the E1 reporting seeds (105 to 109 against 100 to 104). Every setting is one small result file per seed,
so the whole command can be run again after a power cut.

    python tune_map.py --base R2_mpnet --seeds 105-107 --stage 1

Stage 1 changes one setting at a time around the default operating point; stage 2 combines what helped (the names are listed in STAGE2 below once stage 1 is read).
The score is the mean of the three centred headline numbers (B1a, B1b, B1c): it is only used to rank settings of the SAME map on the SAME base, never across bases.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import ceiling
import learned_map
import reps
import run_e1
import runlib

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "map_tuning"
CACHE = HERE / "_cache" / "reps"
STAMP_FILES = [HERE / "tune_map.py", HERE / "learned_map.py", HERE / "ceiling.py", HERE / "assignments.py", HERE / "generator.py", HERE / "reps.py"]
SIZES = ceiling.Sizes(assignments=6)                                    # fewer teaching assignments than the report (20): enough to rank, three times faster

STAGE1: Dict[str, dict] = {
    "default": {},
    "dim32": dict(out_dim=32), "dim128": dict(out_dim=128),
    "wd1e-1": dict(weight_decay=1e-1), "wd1e-3": dict(weight_decay=1e-3),
    "tw1": dict(twin_weight=1.0), "tw16": dict(twin_weight=16.0),
    "ep150": dict(epochs=150), "ep600": dict(epochs=600),
    "tau0.05": dict(tau=0.05), "tau0.2": dict(tau=0.2),
    "hid256": dict(hidden=256),
}
STAGE2: Dict[str, dict] = {}                                            # filled in after stage 1 is read, see the ledger for the choice and the reason


def seed_range(text: str) -> List[int]:
    return run_e1.seed_range(text)


def tune_cell(base: str, name: str, hyper: dict, seed: int, stamp: str) -> dict:
    path = OUT / f"{base}__{name}__s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    r = ceiling.run_ceiling(base, seed, "dev", sizes=SIZES, cache_dir=CACHE, mapper=learned_map.make_mapper(seed, **hyper))
    rec = {k: r[k] for k in ("b1a", "b1b", "b1c", "b1b_taught", "b1b_story", "b1c_raw", "secs")}
    rec.update(base=base, name=name, seed=seed, hyper={**learned_map.HYPER, **hyper}, code_stamp=stamp)
    OUT.mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(rec))
    return rec


def summarise(base: str, names: List[str], seeds: List[int], stamp: str) -> str:
    lines = [f"Map tuning on `{base}`, seeds {seeds[0]}-{seeds[-1]}, {SIZES.assignments} teaching assignments per seed. Mean over seeds; score = mean of B1a, B1b, B1c.", "",
             "| setting | B1a | B1b | B1b taught | B1b story | B1c | score |", "|---|---|---|---|---|---|---|"]
    rows = []
    for name in names:
        recs = [json.loads((OUT / f"{base}__{name}__s{s}.json").read_text(encoding="utf-8")) for s in seeds if runlib.cell_done(OUT / f"{base}__{name}__s{s}.json", stamp)]
        if len(recs) != len(seeds):
            continue
        m = {k: float(np.mean([r[k] for r in recs])) for k in ("b1a", "b1b", "b1b_taught", "b1b_story", "b1c")}
        rows.append((np.mean([m["b1a"], m["b1b"], m["b1c"]]), name, m))
    for score, name, m in sorted(rows, key=lambda t: -t[0]):
        lines.append(f"| {name} | {m['b1a']:.3f} | {m['b1b']:.3f} | {m['b1b_taught']:.3f} | {m['b1b_story']:.3f} | {m['b1c']:.3f} | {score:.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="R2_mpnet")
    ap.add_argument("--seeds", default="105-107")
    ap.add_argument("--stage", type=int, default=1)
    args = ap.parse_args()
    runlib.set_threads(3)
    seeds = seed_range(args.seeds)
    grid = STAGE1 if args.stage == 1 else STAGE2
    stamp = runlib.code_stamp(STAMP_FILES)
    for name, hyper in grid.items():
        for s in seeds:
            r = tune_cell(args.base, name, hyper, s, stamp)
            print(f"{args.base} {name:8s} seed {s}: B1a {r['b1a']:.3f}  B1b {r['b1b']:.3f} (taught {r['b1b_taught']:.3f} story {r['b1b_story']:.3f})  B1c {r['b1c']:.3f}  {r['secs']}s", flush=True)
        reps.clear_models()
    text = summarise(args.base, list(grid), seeds, stamp)
    runlib.atomic_write_text(OUT / f"TUNING_{args.base}_stage{args.stage}.md", text)
    print(text)
