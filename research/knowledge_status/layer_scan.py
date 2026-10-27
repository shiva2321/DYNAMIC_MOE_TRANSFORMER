"""Which layer of the language model to put the learned map on (Task 9): the E1 numbers for several Qwen layers on the TUNING seeds (105 to 109), never the report seeds.

    python layer_scan.py --layers 20,22,24,26,28 --seeds 105-107
"""

import argparse
import json
from pathlib import Path

import numpy as np

import ceiling
import reps
import run_e1
import runlib

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "layer_scan"
CACHE = HERE / "_cache" / "reps"
STAMP_FILES = [HERE / "layer_scan.py", HERE / "ceiling.py", HERE / "reps.py", HERE / "assignments.py", HERE / "generator.py"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="20,22,24,26,28")
    ap.add_argument("--seeds", default="105-107")
    args = ap.parse_args()
    runlib.set_threads(3)
    seeds, stamp = run_e1.seed_range(args.seeds), runlib.code_stamp(STAMP_FILES)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [f"Qwen layer scan, plain representation (no map), tuning seeds {seeds[0]}-{seeds[-1]}, mean over seeds.", "", "| layer | B1a | B1b | B1b taught | B1b story | B1c |", "|---|---|---|---|---|---|"]
    for layer in [int(x) for x in args.layers.split(",")]:
        rep_id, recs = f"R4_qwen_L{layer}", []
        for s in seeds:
            path = OUT / f"{rep_id}__s{s}.json"
            if runlib.cell_done(path, stamp):
                recs.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            r = ceiling.run_ceiling(rep_id, s, "dev", cache_dir=CACHE)
            r["code_stamp"] = stamp
            runlib.atomic_write_text(path, json.dumps(r))
            recs.append(r)
            print(f"{rep_id} seed {s}: B1a {r['b1a']:.3f} B1b {r['b1b']:.3f} (taught {r['b1b_taught']:.3f} story {r['b1b_story']:.3f}) B1c {r['b1c']:.3f}", flush=True)
        reps.clear_models()
        m = {k: float(np.mean([r[k] for r in recs])) for k in ("b1a", "b1b", "b1b_taught", "b1b_story", "b1c")}
        lines.append(f"| {layer} | {m['b1a']:.3f} | {m['b1b']:.3f} | {m['b1b_taught']:.3f} | {m['b1b_story']:.3f} | {m['b1c']:.3f} |")
    text = "\n".join(lines) + "\n"
    runlib.atomic_write_text(OUT / "LAYER_SCAN.md", text)
    print(text)
