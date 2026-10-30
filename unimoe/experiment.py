"""Experiment driver: tune every arm on the tuning seed, then run the evaluation grid, every cell cached and relaunchable.

    python -m unimoe.experiment tune                                  # seed 499, sequential schedule: one recipe per arm
    python -m unimoe.experiment run --seeds 500 501 502               # all schedules, all systems, tuned recipes
    python -m unimoe.experiment run --seeds 500 --schedules seq --systems U-real shipped

Cells live in results/unimoe/runs, trials in results/unimoe/tune, the chosen recipes in results/unimoe/tuning.json. A cell whose code stamp differs from the current one is recomputed.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import domains as DM
from . import runlib, suite as SU, systems as SY
from .frontends import SphereFrontend
from .runner import RunRecord, run_cell

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "unimoe"
TUNE_SEED = 499
EVAL_SEEDS = (500, 501, 502)
STEP_GRID = (1, 5, 20)
RECIPE_GRID = [{"inner_steps": s, "replay_size": r} for s in (5, 20, 50) for r in (256, 1024)]
REPLAY_GRID = [{"hidden": w, "inner_steps": s} for w in (256, 1024, 4096) for s in (1, 5, 20)]
_STREAMS: Dict[int, Dict[str, SU.UStream]] = {}


def stamp() -> str:
    files = sorted((ROOT / "unimoe").glob("*.py")) + sorted((ROOT / "growmoe").glob("*.py")) + sorted((ROOT / "growmoe" / "pools").glob("*.py"))
    return runlib.code_stamp(files)


def streams(seed: int) -> Dict[str, SU.UStream]:
    if seed not in _STREAMS:
        fe = SphereFrontend(d=128, cache_dir=OUT / "emb")
        doms = {n: DM.load_domain(n, fe, seed) for n in DM.ALL_DOMAINS}
        ood = DM.ood_sets(fe, seed)
        _STREAMS[seed] = {sch: SU.build_suite(sch, doms, seed, ood) for sch in SU.SCHEDULES}
    return _STREAMS[seed]


def final_mean(rec: RunRecord) -> float:
    return float(rec.R[-1].mean())


def cell(system: str, stream: SU.UStream, seed: int, recipe: dict, runs_dir: Path, mem_bytes: Optional[int] = None, tag: str = "") -> RunRecord:
    total = int(sum(len(p.y) for p in stream.phases))
    make = lambda: SY.make_system(system, stream.n_classes, seed, recipe, mem_bytes=mem_bytes, total_samples=total)
    return run_cell(runs_dir, tag or system, make, stream, seed, stamp())


def tune() -> dict:
    st = streams(TUNE_SEED)["seq"]
    tdir = OUT / "tune"
    chosen: Dict[str, dict] = {}
    log = {}

    def best(system, grid, mem=None, tagger=lambda r: "_".join(f"{k}{v}" for k, v in sorted(r.items()))):
        scores = []
        for r in grid:
            try:
                rec = cell(system, st, TUNE_SEED, r, tdir, mem_bytes=mem, tag=f"{system}__{tagger(r)}")
            except ValueError as e:
                print(f"  {system} {r}: skipped ({e})", flush=True)
                continue
            scores.append((final_mean(rec), r, rec))
            print(f"  {system:<14s} {r}: final mean {final_mean(rec):.3f} ({rec.seconds:.0f}s)", flush=True)
        score, recipe, rec = max(scores, key=lambda t: t[0])
        log[system] = [{"recipe": r, "final_mean": s} for s, r, _ in scores]
        return recipe, rec

    for name in ("shipped", "U-real", "U-deep"):
        chosen[name], rec = best(name, [{"inner_steps": s} for s in STEP_GRID])
        if name == "U-real":
            budget = int(rec.report["memory_bytes"])
    chosen["U-recipe"], _ = best("U-recipe", RECIPE_GRID)
    chosen["replay_eqmem"], _ = best("replay_eqmem", REPLAY_GRID, mem=budget)
    chosen["finetune"], _ = best("finetune", [{"inner_steps": s} for s in (5, 20)])
    chosen["joint"], _ = best("joint", [{"inner_steps": s} for s in (5, 20)])
    chosen["oracle_experts"] = dict(chosen["finetune"])
    chosen["knn_eqmem"], chosen["lda"] = {}, {}
    out = {"tune_seed": TUNE_SEED, "chosen": chosen, "grid_results": log, "budget_bytes_u_real": budget, "code_stamp": stamp()}
    runlib.atomic_write_text(OUT / "tuning.json", json.dumps(out, indent=1))
    return out


def load_tuning() -> dict:
    return json.loads((OUT / "tuning.json").read_text(encoding="utf-8"))["chosen"]


def run(seeds, schedules, systems) -> None:
    recipes = load_tuning()
    runs = OUT / "runs"
    order = [s for s in SY.SYSTEMS if s in systems]                                              # UniMoE variants first: the equal-memory baselines read U-real's memory
    for seed in seeds:
        for sch in schedules:
            st = streams(seed)[sch]
            mem = None
            for name in order:
                t0 = time.time()
                if name in ("replay_eqmem", "knn_eqmem") and mem is None:
                    ref = cell("U-real", st, seed, recipes["U-real"], runs)
                    mem = int(ref.report["memory_bytes"])
                recipe = dict(recipes[name])
                if name == "replay_eqmem":                                                           # the tuned width may not fit another seed's budget: take the next smaller that fits
                    for w in sorted({256, 1024, 4096}, reverse=True):
                        if w <= recipe["hidden"]:
                            try:
                                rec = cell(name, st, seed, {**recipe, "hidden": w}, runs, mem_bytes=mem)
                                break
                            except ValueError:
                                continue
                else:
                    rec = cell(name, st, seed, recipe, runs, mem_bytes=mem)
                if name == "U-real":
                    mem = int(rec.report["memory_bytes"])
                print(f"seed {seed} {sch:<5s} {name:<15s} final mean {final_mean(rec):.3f} experts {rec.per_phase[-1]['n_experts']:>3d} {time.time() - t0:5.0f}s", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tune", "run"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(EVAL_SEEDS))
    ap.add_argument("--schedules", nargs="+", default=list(SU.SCHEDULES))
    ap.add_argument("--systems", nargs="+", default=list(SY.SYSTEMS))
    a = ap.parse_args(argv)
    if a.cmd == "tune":
        print(json.dumps(tune()["chosen"], indent=1))
    else:
        run(a.seeds, a.schedules, a.systems)


if __name__ == "__main__":
    main()
