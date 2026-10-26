"""Competence sweep (spec 5.6): every skill taught 5, 20, 100, 500 and 2,000 examples, scored on problems it has never seen. One small file per (skill, count, seed),
written atomically and skipped when it already exists for the same code, so the whole command can simply be run again after a power cut.

    python run_competence.py --pool dev --seeds 100-102
"""

import argparse
import json
from pathlib import Path

import numpy as np

import assignments
import runlib
import skills

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "competence"
STAMP_FILES = [HERE / f for f in ("assignments.py", "generator.py", "pools.py", "pool_dev.py", "skills.py", "runlib.py")] + [HERE.parents[1] / "growmoe" / "arith.py"]


def seed_range(text: str):
    lo, _, hi = text.partition("-")
    return range(int(lo), int(hi or lo) + 1)


def run_cell(pool: str, skill: str, count: int, seed: int, stamp: str) -> dict:
    path = OUT / f"{pool}__{skill}__n{count}__s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    rng = np.random.default_rng([seed, skills.SKILLS.index(skill), count])
    row = assignments.competence_table({skill: "solid"}, {"solid": count}, rng, pool)[skill]
    rec = {"pool": pool, "skill": skill, "count": count, "seed": seed, "accuracy": row["accuracy"], "chance": row["chance"], "status": row["status"], "code_stamp": stamp}
    OUT.mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(rec))
    return rec


def report(pool: str, seeds, stamp: str) -> str:
    lines = [f"Competence sweep, pool `{pool}`, seeds {seeds.start}-{seeds.stop - 1}: mean held-out accuracy (chance in brackets), status by the draft rule.", "",
             "| skill | " + " | ".join(f"n={c}" for c in assignments.COUNT_SWEEP) + " |", "|---|" + "---|" * len(assignments.COUNT_SWEEP)]
    for skill in skills.SKILLS:
        cells = []
        for c in assignments.COUNT_SWEEP:
            rows = [run_cell(pool, skill, c, s, stamp) for s in seeds]
            acc, ch = np.mean([r["accuracy"] for r in rows]), np.mean([r["chance"] for r in rows])
            states = "/".join(sorted({r["status"] for r in rows}))
            cells.append(f"{acc:.2f} ({ch:.2f}) {states}")
        lines.append(f"| {skill} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="dev")
    ap.add_argument("--seeds", default="100-102")
    args = ap.parse_args()
    seeds = seed_range(args.seeds)
    stamp = runlib.code_stamp(STAMP_FILES)
    for seed in seeds:
        for skill in skills.SKILLS:
            for count in assignments.COUNT_SWEEP:
                rec = run_cell(args.pool, skill, count, seed, stamp)
                print(f"{skill:4s} n={count:<5d} seed {seed}: acc {rec['accuracy']:.3f} chance {rec['chance']:.3f} {rec['status']}", flush=True)
    text = report(args.pool, seeds, stamp)
    runlib.atomic_write_text(HERE / "results" / "COMPETENCE_TABLE.md", text)
    print(text)
