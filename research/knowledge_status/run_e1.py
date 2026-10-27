"""E1 runner: one small result file per (representation, seed), safe to run again after a power cut (finished cells with the same code stamp are skipped, damaged ones
are redone), then the table results/E1_TABLE.md.

    python run_e1.py --pool dev --seeds 100-104 --reps all
    python run_e1.py --pool dev --seeds 100-104 --reps all,R4_qwen_L22,R5_R2_mpnet

``R5_<id>`` is the learned map (Task 9) on top of the representation ``<id>``; ``R4_qwen_L<n>`` is any other layer of the language model.
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

import ceiling
import reps
import runlib

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "e1"
CACHE = HERE / "_cache" / "reps"
STAMP_FILES = [HERE / f for f in ("ceiling.py", "run_e1.py", "reps.py", "assignments.py", "generator.py", "corpora.py", "pools.py", "pool_dev.py", "skills.py", "runlib.py")] \
    + [HERE.parents[1] / "growmoe" / "arith.py", HERE.parents[1] / "growmoe" / "frontend.py"]
LEARNED_MAP = HERE / "learned_map.py"
COLUMNS = (("b1a", "B1a probe (whole message)"), ("b1a_span", "B1a probe (gold span)"), ("b1b", "B1b near-twin AUROC (centred)"), ("b1b_raw", "B1b raw"),
           ("b1b_taught", "B1b, partner in a taught wording"), ("b1b_story", "B1b, partner a story"),
           ("b1c", "B1c taught-vs-untaught AUROC (centred)"), ("b1c_raw", "B1c raw"))


def seed_range(text: str) -> List[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def stamp_for(rep_id: str) -> str:
    """The code stamp of one representation: an R5 result also depends on the learned map's code, a plain one does not (so changing the map never invalidates them)."""
    return runlib.code_stamp(STAMP_FILES + ([LEARNED_MAP] if rep_id.startswith("R5_") else []))


def run_cell(rep_id: str, seed: int, pool: str, out_dir, stamp: str, sizes: ceiling.Sizes = ceiling.SIZES) -> dict:
    path = Path(out_dir) / f"{rep_id}__s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    if rep_id.startswith("R5_"):
        import learned_map
        res = ceiling.run_ceiling(rep_id[3:], seed, pool, sizes=sizes, cache_dir=CACHE, mapper=learned_map.make_mapper(seed))
        res["rep"] = rep_id
    else:
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


def report_e1(out_dir, rep_ids: Sequence[str], seeds: Sequence[int], stamp: Union[str, Callable[[str], str]], pool: str = "dev") -> str:
    """``stamp`` is one code stamp for every representation, or a function from representation id to its stamp."""
    stamp_of = stamp if callable(stamp) else (lambda _rep_id: stamp)
    lines = [f"# E1, the information-ceiling table (pool `{pool}`, seeds {seeds[0]}-{seeds[-1]})", "",
             "Mean over seeds, with [min - max]. Draft bars: B1a >= 0.95 (chance 0.14), B1b >= 0.95 (chance 0.50), B1c >= 0.90 (chance 0.50). Centred = the mean of the teaching "
             "embeddings is subtracted first (the primary value); raw = cosine on the vectors as they are. The two split columns repeat B1b for the triples whose reworded "
             "partner is in a taught wording (S1 to S3) and for those whose partner is a story (S4, never taught). R5_<id> is the learned map on top of <id>.", "",
             "| representation | seeds | " + " | ".join(label for _, label in COLUMNS) + " |", "|---|---|" + "---|" * len(COLUMNS)]
    means: Dict[str, Dict[str, float]] = {}
    for rep_id in rep_ids:
        rows = load_cells(out_dir, rep_id, seeds, stamp_of(rep_id))
        if not rows:
            lines.append(f"| {rep_id} | 0 | " + " | ".join("missing" for _ in COLUMNS) + " |")
            continue
        means[rep_id] = {k: float(np.mean([r[k] for r in rows])) for k, _ in COLUMNS}
        cells = []
        for k, _ in COLUMNS:
            v = [r[k] for r in rows]
            bar = ceiling.BARS.get(k)                                  # only the three headline numbers have a bar; the raw, span and split columns are diagnostics
            mark = "" if bar is None else (" PASS" if np.mean(v) >= bar else " fail")
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
    ap.add_argument("--reps", default="all", help="comma-separated representation ids; 'all' stands for the eleven of the plan")
    args = ap.parse_args()
    runlib.set_threads(3)
    seeds = seed_range(args.seeds)
    rep_ids = [r for token in args.reps.split(",") for r in (reps.REP_IDS if token == "all" else (token,))]
    for rep_id in rep_ids:
        for seed in seeds:
            r = run_cell(rep_id, seed, args.pool, OUT, stamp_for(rep_id))
            print(f"{rep_id:16s} seed {seed}: B1a {r['b1a']:.3f} (span {r['b1a_span']:.3f})  B1b {r['b1b']:.3f} (raw {r['b1b_raw']:.3f}, taught {r['b1b_taught']:.3f}, story {r['b1b_story']:.3f})  "
                  f"B1c {r['b1c']:.3f} (raw {r['b1c_raw']:.3f}, worst {r['b1c_min']:.3f})  {r['secs']}s", flush=True)
        reps.clear_models()                                    # one model in memory at a time, the GPU is freed between representations
    text = report_e1(OUT, rep_ids, seeds, stamp_for, args.pool)
    runlib.atomic_write_text(HERE / "results" / "E1_TABLE.md", text)
    print(text)
