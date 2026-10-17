"""The function-regime (s3) family of the dev runs WITHOUT the seed-aliasing head start (E-035, F-30).

`rv2d_lib.run_s3` builds the pool with seed = stream seed, so the expert founded for regime r starts with regime r's teacher units. This driver runs the SAME cells (same streams, seeds, arms, constants)
with the pool seed shifted by 7, through the unchanged `rv2d_lib.run_cell`, by patching `run_rv2c.arm_cfg` in the process. It touches no stamped file: the results go to `cells_s3u/` and carry the code
stamp that was current when the cell started, so they are comparable with `cells/`.

Since amendment A4-2 (docs/superpowers/specs/2026-10-15-routing-v2-stage4-AMENDMENTS.md) `rv2d_lib.run_s3` applies the same offset itself (`rv2d_lib.pool_seed`); this driver is kept because the
records of `cells_s3u/` were produced before that change, and it still gives the same seed (the library's replace wins, with the same value).

    python run_s3_unaliased.py list                  # the cells
    python run_s3_unaliased.py run --workers 2       # run what is missing (safe to repeat after a power cut)
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rv2d_lib as R  # noqa: E402

OFFSET = 7
OUT = HERE / "cells_s3u"
SEEDS = (100, 101, 102)
CONST = json.loads((HERE / "dev_constants.json").read_text(encoding="utf-8")) if (HERE / "dev_constants.json").exists() else {"grow_gap": 0.17, "transfer_margin": 0.25}
PHASES = {"core": ("F-15", "OV-0.6", "OV-0.85", "S1", "S2-5"), "rest": ("F-30", "S2-30", "OV-0.0", "OV-0.3", "OV-0.3-MIXED")}


def cells(phase: str):
    g, t = CONST["grow_gap"], CONST["transfer_margin"]
    out = []
    for st in PHASES[phase]:
        for s in SEEDS:
            for arm in (("S3", "S3+T", "S3+G", "S3+T+G") if phase == "core" else ("S3",)):
                out.append(R.Cell("s3", st, arm, s, gap=g if "G" in arm else 0.0, tmargin=t if "T" in arm else 0.0))
    return out


def run_cell_here(cell: R.Cell) -> None:
    import torch
    torch.set_num_threads(2)
    path = OUT / f"{cell.cid}.json"
    stamp = R.code_stamp()
    if R.cell_done(path, stamp):
        return
    import run_rv2c as R3
    orig = R3.arm_cfg
    R3.arm_cfg = lambda arm, st, seed, steps: dataclasses.replace(orig(arm, st, seed, steps), seed=seed + OFFSET)       # the only change against rv2d_lib.run_s3
    R.CELLS = OUT                                                                                                    # checkpoints and results of this driver live apart from the aliased cells
    OUT.mkdir(exist_ok=True)
    cpu0 = time.process_time()
    rec = R.run_cell(cell)
    rec["cpu_seconds"] = time.process_time() - cpu0
    rec["code_stamp"] = stamp
    rec["pool_seed_offset"] = OFFSET
    R.atomic_write_text(path, json.dumps(rec))


def cmd_run(args) -> None:
    todo = [c for c in cells(args.phase) if not R.cell_done(OUT / f"{c.cid}.json", R.code_stamp())]
    print(f"{len(todo)} cells to run of {len(cells(args.phase))}", flush=True)
    running = []
    while todo or running:
        running = [p for p in running if p.poll() is None]
        while todo and len(running) < args.workers:
            c = todo.pop(0)
            running.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "one", "--cid", c.cid, "--phase", args.phase]))
            time.sleep(1)
        time.sleep(5)
    print("all cells finished", flush=True)


def cmd_one(args) -> None:
    cell = {c.cid: c for c in cells(args.phase)}[args.cid]
    run_cell_here(cell)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("list", "run", "one"))
    ap.add_argument("--phase", default="core", choices=tuple(PHASES))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cid")
    a = ap.parse_args()
    if a.cmd == "list":
        for c in cells(a.phase):
            print(c.cid)
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "one":
        cmd_one(a)
