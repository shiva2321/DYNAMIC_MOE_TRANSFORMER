"""The single-use confirmatory run (Task 15). It runs ONLY after the user's explicit go, and only once:

    KS_ALLOW_CONFIRM=1 PYTHONPATH=<repo root> python run_confirm.py

Written and tested BEFORE the freeze, so that the frozen code stamp covers it. Before it computes anything it checks, in this order, that
  1. the run is not closed (``CONFIRM_CLOSED.json`` does not exist: a finished run is never repeated),
  2. ``KS_ALLOW_CONFIRM`` is "1",
  3. what was frozen is still on disk: ``bars.json`` against ``bars.sha256``, ``pool_confirm.py`` against its recorded sha256, every code file against the recorded code stamp,
  4. a run that was already started was started under this same freeze.
A power cut is not a second run: the cells are written one file at a time, a finished cell is never redone, and the same command picks up where it stopped.
When every cell exists, ``CONFIRM_CLOSED.json`` is written and the runner refuses to start again. Results go to ``results/confirm/{e1,e2,e4}``; ``report.py`` turns them into the verdict.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence
from unittest import mock

import freeze_prereg as FZ
import runlib

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "confirm"
STARTED, CLOSED = HERE / "CONFIRM_STARTED.json", HERE / "CONFIRM_CLOSED.json"
POOL = "confirm"


class ConfirmRefused(RuntimeError):
    """The single-use run may not start (or restart) under the current conditions."""


def preflight(frozen_dir: Path = HERE, pool_file: Path = HERE / "pool_confirm.py", code_files: Optional[Sequence[Path]] = None, environ: Mapping[str, str] = os.environ,
              closed: Path = CLOSED, started: Path = STARTED) -> dict:
    """The frozen record (bars.json) if the run may go ahead; ConfirmRefused with the reason otherwise."""
    if Path(closed).exists():
        raise ConfirmRefused(f"single use: {Path(closed).name} exists, the confirmatory run is finished and is not repeated")
    if environ.get("KS_ALLOW_CONFIRM") != "1":
        raise ConfirmRefused("the confirmatory pool is single-use: set KS_ALLOW_CONFIRM=1 only on the user's explicit go")
    chk = FZ.check_frozen(frozen_dir, pool_file, code_files if code_files is not None else FZ.default_code_files())
    if not chk["ok"]:
        raise ConfirmRefused(f"what was frozen is not what is on disk: {chk['problem']}")
    rec = json.loads((Path(frozen_dir) / "bars.json").read_text(encoding="utf-8"))
    if Path(started).exists():
        was = json.loads(Path(started).read_text(encoding="utf-8"))
        if was.get("bars_sha256") != FZ.bars_hash(rec):
            raise ConfirmRefused("a confirmatory run was started under a different freeze; this is not a resume")
    return rec


def mark_started(rec: Mapping, e4_seeds: Sequence[int], started: Path = STARTED) -> dict:
    """Write the start record once (a resume reads it back, so the list of E4 seeds cannot change between launches)."""
    started = Path(started)
    if started.exists():
        return json.loads(started.read_text(encoding="utf-8"))
    rec_started = {"bars_sha256": FZ.bars_hash(rec), "code_stamp": rec["code_stamp"], "pool_confirm_sha256": rec["pool_confirm_sha256"], "e4_seeds": [int(s) for s in e4_seeds],
                   "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    runlib.atomic_write_text(started, json.dumps(rec_started, indent=1, sort_keys=True) + "\n")
    return rec_started


def jobs(rec: Mapping, e4_seeds: Sequence[int]) -> List[tuple]:
    """(experiment, representation or None, seed) for every cell of the run, in the order they are computed: the ceiling (E1), the status of parts and tangled messages (E2), the traces (E4)."""
    seeds, reps = rec["seeds"]["confirm"], rec["representations"]
    return ([("e1", r, s) for r in reps for s in seeds] + [("e2", r, s) for r in reps for s in seeds] + [("e4", None, s) for s in e4_seeds])


def cell_path(out: Path, job: tuple) -> Path:
    exp, rep, seed = job
    return Path(out) / exp / (f"{rep}__s{seed}.json" if rep else f"s{seed}.json")


def stamps() -> dict:
    return FZ.current_stamps()


def run_e4_cell(seed: int, out_dir: Path, stamp: str, n_per_kind: int, **kw) -> dict:
    """The dev E4 runner on the confirmatory pool. ``run_e3.run_seed`` names the dev pool inside its body and is part of the stamped code of the dev results, so it is not edited; the pool
    is swapped at the one place where the world is built, and ``tests/knowledge_status/test_ks_run_confirm.py`` pins that this is the confirmatory pool."""
    import run_e3 as E3
    build = E3.build_trace_world

    def on_confirm(seed_, pool_, sizes, competence_fn, n_per_kind_):
        assert pool_ == "dev", "run_e3.run_seed changed; this wrapper must be revisited"
        return build(seed_, POOL, sizes, competence_fn, n_per_kind_)

    with mock.patch.object(E3, "build_trace_world", on_confirm):
        return E3.run_seed(seed, out_dir, stamp, n_per_kind=n_per_kind, **kw)


def e4_seeds_for(rec: Mapping, competence_fn: Optional[Callable] = None) -> List[int]:
    """Every confirmatory seed in which addition and subtraction are really Solid (decided by the measured accuracy of their solvers on the confirmatory pool, no model involved)."""
    import run_e3 as E3
    return E3.qualifying_seeds(rec["seeds"]["confirm"], len(rec["seeds"]["confirm"]), POOL, competence_fn)


def run_all(rec: Mapping, runners: Mapping[str, Callable], stamp_of: Mapping, e4_seeds: Sequence[int], out: Path = OUT, max_cells: Optional[int] = None,
            log: Callable[[str], None] = print) -> dict:
    """Compute every missing cell (a finished one is left alone). ``runners[exp](rep, seed, out_dir, stamp)`` runs one cell. Returns {"done": n, "total": n, "ran": n}."""
    todo = jobs(rec, e4_seeds)
    ran = 0
    for job in todo:
        exp, rep, seed = job
        stamp = stamp_of[exp](rep) if callable(stamp_of[exp]) else stamp_of[exp]
        path = cell_path(out, job)
        if runlib.cell_done(path, stamp):
            continue
        if max_cells is not None and ran >= max_cells:
            break
        t0 = time.time()
        runners[exp](rep, seed, Path(out) / exp, stamp)
        ran += 1
        log(f"{exp} {rep or ''} seed {seed}: done in {time.time() - t0:.0f}s")
    done = sum(1 for job in todo if runlib.cell_done(cell_path(out, job), stamp_of[job[0]](job[1]) if callable(stamp_of[job[0]]) else stamp_of[job[0]]))
    return {"done": done, "total": len(todo), "ran": ran}


def close(rec: Mapping, progress: Mapping, closed: Path = CLOSED) -> None:
    if progress["done"] != progress["total"]:
        raise ConfirmRefused(f"cannot close: {progress['done']} of {progress['total']} cells exist")
    runlib.atomic_write_text(Path(closed), json.dumps({"closed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "bars_sha256": FZ.bars_hash(rec), "cells": progress["total"]}, indent=1, sort_keys=True) + "\n")


def real_runners(rec: Mapping) -> Dict[str, Callable]:
    import reps
    import run_e1
    import run_e2
    last = {"rep": None}

    def switch(rep):                                          # one model in memory at a time: free the GPU when the representation changes
        if last["rep"] is not None and rep != last["rep"]:
            reps.clear_models()
        last["rep"] = rep

    def e1(rep, seed, out_dir, stamp):
        switch(rep)
        return run_e1.run_cell(rep, seed, POOL, out_dir, stamp)

    def e2(rep, seed, out_dir, stamp):
        switch(rep)
        return run_e2.run_cell(rep, seed, POOL, out_dir, stamp)

    def e4(_rep, seed, out_dir, stamp):
        switch("e4")
        return run_e4_cell(seed, out_dir, stamp, rec.get("e4_n_per_kind", 40))

    return {"e1": e1, "e2": e2, "e4": e4}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cells", type=int, default=None, help="stop after this many NEW cells (for a short test of the plumbing; the run is not closed until every cell exists)")
    args = ap.parse_args(argv)
    runlib.set_threads(3)
    try:
        rec = preflight()
    except ConfirmRefused as e:
        print(f"REFUSED: {e}")
        return 2
    st = mark_started(rec, e4_seeds_for(rec) if not STARTED.exists() else [], STARTED)
    progress = run_all(rec, real_runners(rec), stamps(), st["e4_seeds"], OUT, args.max_cells)
    print(progress, flush=True)
    if progress["done"] == progress["total"]:
        close(rec, progress, CLOSED)
        print("all cells exist; CONFIRM_CLOSED.json written. Run report.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
