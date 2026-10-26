"""E1 runner: one small result file per (representation, seed), safe to run again after a power cut (finished cells with the same code stamp are skipped, damaged ones
are redone), then the table results/E1_TABLE.md.

    python run_e1.py --pool dev --seeds 100-104 --reps all
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import ceiling
import reps
import runlib

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "e1"
CACHE = HERE / "_cache" / "reps"
STAMP_FILES = [HERE / f for f in ("ceiling.py", "run_e1.py", "reps.py", "assignments.py", "generator.py", "corpora.py", "pools.py", "pool_dev.py", "skills.py", "runlib.py")] \
    + [HERE.parents[1] / "growmoe" / "arith.py", HERE.parents[1] / "growmoe" / "frontend.py"]
COLUMNS = (("b1a", "B1a probe (whole message)"), ("b1a_span", "B1a probe (gold span)"), ("b1b", "B1b near-twin AUROC (centred)"), ("b1b_raw", "B1b raw"),
           ("b1c", "B1c taught-vs-untaught AUROC (centred)"), ("b1c_raw", "B1c raw"))


def seed_range(text: str) -> List[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def run_cell(rep_id: str, seed: int, pool: str, out_dir, stamp: str, sizes: ceiling.Sizes = ceiling.SIZES) -> dict:
    path = Path(out_dir) / f"{rep_id}__s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    res = ceiling.run_ceiling(rep_id, seed, pool, sizes=sizes, cache_dir=CACHE)
    res["code_stamp"] = stamp
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(res))
    return res


def b2_validity(r0: Dict[str, float], bars: Dict[str, float] = ceiling.BARS) -> Tuple[bool, str]:
    """B2: the test bed is valid only if the word-matcher control R0 fails B1a or B1b by at least 0.15 (otherwise it cannot tell a word-matcher from a reader, anti-goal A1)."""
    gaps = {k: bars[k] - r0[k] for k in ("b1a", "b1b")}
    ok = any(g >= 0.15 - 1e-9 for g in gaps.values())
    text = f"R0 is below the bar by {gaps['b1a']:+.2f} on B1a and {gaps['b1b']:+.2f} on B1b (a gap of 0.15 or more on either is required): " + ("test bed VALID" if ok else "test bed INVALID")
    return ok, text


def load_cells(out_dir, rep_id: str, seeds: Sequence[int], stamp: str) -> List[dict]:
    rows = []
    for s in seeds:
        path = Path(out_dir) / f"{rep_id}__s{s}.json"
        if runlib.cell_done(path, stamp):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def report_e1(out_dir, rep_ids: Sequence[str], seeds: Sequence[int], stamp: str, pool: str = "dev") -> str:
    lines = [f"# E1, the information-ceiling table (pool `{pool}`, seeds {seeds[0]}-{seeds[-1]})", "",
             "Mean over seeds, with [min - max]. Draft bars: B1a >= 0.95 (chance 0.14), B1b >= 0.95 (chance 0.50), B1c >= 0.90 (chance 0.50). Centred = the mean of the teaching "
             "embeddings is subtracted first (the primary value); raw = cosine on the vectors as they are.", "",
             "| representation | seeds | " + " | ".join(label for _, label in COLUMNS) + " |", "|---|---|" + "---|" * len(COLUMNS)]
    means: Dict[str, Dict[str, float]] = {}
    for rep_id in rep_ids:
        rows = load_cells(out_dir, rep_id, seeds, stamp)
        if not rows:
            lines.append(f"| {rep_id} | 0 | " + " | ".join("missing" for _ in COLUMNS) + " |")
            continue
        means[rep_id] = {k: float(np.mean([r[k] for r in rows])) for k, _ in COLUMNS}
        cells = []
        for k, _ in COLUMNS:
            v = [r[k] for r in rows]
            bar = ceiling.BARS.get(k.split("_")[0])
            mark = "" if bar is None or k.endswith("raw") or k == "b1a_span" else (" PASS" if np.mean(v) >= bar else " fail")
            cells.append(f"{np.mean(v):.2f} [{min(v):.2f} - {max(v):.2f}]{mark}")
        lines.append(f"| {rep_id} | {len(rows)} | " + " | ".join(cells) + " |")
    lines.append("")
    if "R0_hash" in means:
        lines.append("**B2 validity of the test bed:** " + b2_validity(means["R0_hash"])[1])
    else:
        lines.append("**B2 validity of the test bed:** R0_hash has no results yet.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="dev")
    ap.add_argument("--seeds", default="100-104")
    ap.add_argument("--reps", default="all", help="'all' or a comma-separated list of representation ids")
    args = ap.parse_args()
    runlib.set_threads(3)
    seeds = seed_range(args.seeds)
    rep_ids = list(reps.REP_IDS) if args.reps == "all" else args.reps.split(",")
    stamp = runlib.code_stamp(STAMP_FILES)
    for rep_id in rep_ids:
        for seed in seeds:
            r = run_cell(rep_id, seed, args.pool, OUT, stamp)
            print(f"{rep_id:13s} seed {seed}: B1a {r['b1a']:.3f} (span {r['b1a_span']:.3f})  B1b {r['b1b']:.3f} (raw {r['b1b_raw']:.3f})  "
                  f"B1c {r['b1c']:.3f} (raw {r['b1c_raw']:.3f}, worst {r['b1c_min']:.3f})  {r['secs']}s", flush=True)
        reps.clear_models()                                    # one model in memory at a time, the GPU is freed between representations
    text = report_e1(OUT, rep_ids, seeds, stamp, args.pool)
    runlib.atomic_write_text(HERE / "results" / "E1_TABLE.md", text)
    print(text)
