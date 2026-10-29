"""Operating points of the from-scratch reader arms (Task 11b), chosen on the TUNING seeds 150 to 154 only, before any dev or confirmatory seed is run (the same discipline as tune_baselines.py).

    PYTHONPATH=<repo root> python tune_scratch.py

For each arm a small grid (width of the hidden layer, epochs, learning rate) is run through the information-ceiling measurement (E1) on the five tuning seeds. Rule fixed beforehand: the setting with the
highest mean of B1a, B1b and B1c wins; ties go to the smaller network. The winning setting is written to results/FS_TUNING.md; its values are then copied by hand into ``scratch_reader.HYPER``.
Refuses to run on any seed outside 150 to 154.
"""

import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import runlib
import scratch_reader as SR

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "fs_tuning"
SEEDS = (150, 151, 152, 153, 154)
GRID = {"hidden": (256, 1024), "epochs": (300, 800), "lr": (1e-3, 3e-3)}


def settings() -> List[Dict[str, object]]:
    keys = sorted(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]


def score(rows: Sequence[dict]) -> float:
    return float(np.mean([np.mean([r["b1a"], r["b1b"], r["b1c"]]) for r in rows]))


def choose(results: Dict[str, float], sizes: Dict[str, int]) -> str:
    """The key with the highest score; ties (to 9 decimals) go to the smaller network."""
    best = max(round(v, 9) for v in results.values())
    return min((k for k, v in results.items() if round(v, 9) == best), key=lambda k: (sizes[k], k))


def run(seeds: Sequence[int] = SEEDS, out: Path = OUT, arms: Sequence[str] = SR.ARMS) -> dict:
    bad = [s for s in seeds if s not in SEEDS]
    if bad:
        raise ValueError(f"the from-scratch arms are tuned on seeds {SEEDS} only; refusing {bad}")
    summary = {}
    stamp = runlib.code_stamp([HERE / "tune_scratch.py", HERE / "scratch_reader.py", HERE / "learned_map.py"])
    for arm in arms:
        results, sizes = {}, {}
        for st in settings():
            key = json.dumps(st, sort_keys=True)
            rows = [SR.run_e1_cell(arm, s, "dev", Path(out) / "cells" / hashed(arm, key), stamp, **st) for s in seeds]
            results[key], sizes[key] = score(rows), int(st["hidden"])
            print(f"{arm} {key}: mean(B1a,B1b,B1c) {results[key]:.4f}", flush=True)
        win = choose(results, sizes)
        summary[arm] = {"winner": json.loads(win), "score": results[win], "all": results}
    return summary


def hashed(arm: str, key: str) -> str:
    import hashlib
    return f"{arm}_{hashlib.sha1(key.encode()).hexdigest()[:8]}"


def report(summary: dict) -> str:
    L = ["# From-scratch reader arms, operating points (tuning seeds 150 to 154, E1 ceiling)", "",
         "Rule fixed before the run: the setting with the highest mean of B1a, B1b and B1c; ties to the smaller network. Dev and confirmatory seeds were not touched.", ""]
    for arm, s in summary.items():
        L += [f"## {arm}", "", "| setting | mean(B1a, B1b, B1c) |", "|---|---|"]
        for key, v in sorted(s["all"].items(), key=lambda kv: -kv[1]):
            L.append(f"| {key} | {v:.4f}{' (winner)' if json.loads(key) == s['winner'] else ''} |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    runlib.set_threads(3)
    summary = run()
    text = report(summary)
    runlib.atomic_write_text(HERE / "results" / "FS_TUNING.md", text)
    print(text)
