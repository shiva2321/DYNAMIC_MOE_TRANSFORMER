"""Checks of growmoe's ``skip_known`` flag (E-063) on the beds of the earlier rounds. One JSON per (bed, schedule, seed, arm) in results/fsmoe/K, written atomically, finished files skipped.

``python -m fsmoe.skip_check run --bed f10 --seeds 710 --arms off k0.1 k0.25``

Beds: ``useq`` / ``umix`` / ``udrift`` the seven real text domains of E-054 (UniMoE suite, ULDA pool, seeds 500-504); ``f10`` / ``f30`` growmoe's function-regime stream (the shipped pool, 128 units, seeds 710-714);
``t`` the FS-MoE three-domain teaching bed (real text embeddings); ``bank`` banking77 class-incremental (the current default pool, 128 units). Arms: ``off`` = flag None, ``k<c>`` = ``skip_known`` c.
Each cell also records how many batches took optimiser steps and how many were skipped, and the wall time spent inside ``train_expert`` and inside ``observe``."""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from growmoe.fairfight import run_fast
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams_m1 import function_regimes
from growmoe.streams_real import banking_incremental
from unimoe import experiment as E
from unimoe import p4 as P4
from unimoe import systems2 as S2
from unimoe.runner import run_cell
from unimoe.systems import phylo_cfg

from .bed import ROOT, build_bed
from .chance_check import recipes as u_recipes
from .nbpool_lab import F_RECIPE
from .system_lab import OUT as S_OUT, Prep, teach

OUT = ROOT / "results" / "fsmoe" / "K"
U_OUT = ROOT / "results" / "unimoe" / "p5_skip"
BEDS = ("useq", "umix", "udrift", "f10", "f30", "t", "bank")


def skip_of(arm: str) -> dict:
    if arm == "off":
        return {}
    if arm.startswith("k"):
        return {"skip_known": float(arm[1:])}
    raise ValueError(f"unknown arm {arm!r}")


def instrument(pool) -> dict:
    """Wall time spent inside train_expert and inside observe (predict is not counted)."""
    t = {"train": 0.0, "observe": 0.0}
    train, observe = pool.train_expert, pool.observe

    def timed_train(*a, **k):
        t0 = time.perf_counter()
        try:
            return train(*a, **k)
        finally:
            t["train"] += time.perf_counter() - t0

    def timed_observe(*a, **k):
        t0 = time.perf_counter()
        try:
            return observe(*a, **k)
        finally:
            t["observe"] += time.perf_counter() - t0
    pool.train_expert, pool.observe = timed_train, timed_observe
    return t


def _store(path: Path, out: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out), encoding="utf-8")
    tmp.replace(path)
    return out


def _line(out: dict) -> str:
    s = out["skip"]
    rows = s["trained_rows"] + s["skipped_rows"]
    return (f"{out['bed']} s{out['seed']} {out['arm']}: acc {out['acc']:.4f} experts {out['experts']} skipped {s['skipped']}/{s['trained'] + s['skipped']} batches "
            f"({(s['skipped_rows'] / rows if rows else 0.0):.1%} of rows) train {out['train_seconds']:.1f}s of observe {out['observe_seconds']:.1f}s")


def run_u(sch: str, seed: int, arm: str) -> dict:
    path = OUT / f"u{sch}_s{seed}_{arm}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    st = E.streams(seed)[sch]
    blocks = [tuple(v) for v in st.label_blocks.values()]
    name, recipe = u_recipes(sch)["ULDA-off"]
    timers = {}

    def make():
        pool = S2.make_system2(name, st.n_classes, seed, {**recipe, **skip_of(arm)}, blocks)
        timers["t"] = instrument(pool)                                           # the live dict: it keeps growing while the cell runs
        return pool
    rec = run_cell(U_OUT, arm, make, st, seed, P4.stamp2())
    ev = rec.per_phase[-1]["events"]
    if "t" not in timers:                                                        # a cell finished earlier (a power cut after the cell, before this file): its timings were not kept
        raise RuntimeError(f"cell {arm} s{seed} of {sch} was cached without timings: delete {U_OUT / f'unimoe-{sch}'}/{arm}_s{seed}.* and run again")
    return _store(path, {"bed": f"u{sch}", "seed": seed, "arm": arm, "acc": E.final_mean(rec), "experts": rec.per_phase[-1]["n_experts"], "events": ev, "skip": rec.summary["skip"],
                         "train_seconds": timers["t"]["train"], "observe_seconds": timers["t"]["observe"], "seconds": rec.seconds})


def run_f(n: int, seed: int, arm: str) -> dict:
    path = OUT / f"f{n}_s{seed}_{arm}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    st = function_regimes(seed, n)
    holder = {}

    def make():
        holder["pool"] = S2.make_system2("ULDA", st.n_classes, seed, {**F_RECIPE, "hidden": 128, **skip_of(arm)}, d=st.d)
        holder["t"] = instrument(holder["pool"])
        return holder["pool"]
    res = run_fast(make, st, seed, eval_every="final")
    pool, t = holder["pool"], holder["t"]
    return _store(path, {"bed": f"f{n}", "seed": seed, "arm": arm, "acc": res.acc, "min_regime_acc": float(np.min(res.per_regime)), "experts": res.report.n_experts, "events": dict(Counter(e.kind for e in pool.events)),
                         "skip": dict(pool.skip_stats), "train_seconds": t["train"], "observe_seconds": t["observe"], "seconds": res.wall_clock})


def run_t(seed: int, arm: str) -> dict:
    path = OUT / f"t_s{seed}_{arm}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    bed = build_bed(seed)
    prep = Prep(bed)
    recipe = json.loads((S_OUT / "tuning.json").read_text(encoding="utf-8"))["FS_best"]
    pool = S2.make_system2("ULDA", prep.n_classes, seed, {**recipe, "hidden": 128, **skip_of(arm)}, d=prep.d)
    t = instrument(pool)
    r = teach(pool, prep, seed)
    return _store(path, {"bed": "t", "seed": seed, "arm": arm, "acc": 0.5 * (r["mean_tw"] + r["mean_ho"]), "mean_tw": r["mean_tw"], "mean_ho": r["mean_ho"], "final": r["final"], "experts": r["n_experts"],
                         "events": r["events"], "skip": dict(pool.skip_stats), "train_seconds": t["train"], "observe_seconds": t["observe"], "seconds": r["seconds"]})


def run_bank(seed: int, arm: str) -> dict:
    path = OUT / f"bank_s{seed}_{arm}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT))
    from research.growmoe_results.m0.run_m0 import HybridFrontend
    st = banking_incremental(seed=seed, frontend=HybridFrontend(d=128, seed=0))
    holder = {}

    def make():
        holder["pool"] = PhyloMoE(phylo_cfg("U-real", st.n_classes, seed, {"inner_steps": 20, "q": 0.02, "alpha_spawn": 0.01, "m_min": 32, **skip_of(arm)}, st.d))
        holder["t"] = instrument(holder["pool"])
        return holder["pool"]
    res = run_fast(make, st, seed, eval_every="final")
    pool, t = holder["pool"], holder["t"]
    return _store(path, {"bed": "bank", "seed": seed, "arm": arm, "acc": res.acc, "experts": res.report.n_experts, "events": dict(Counter(e.kind for e in pool.events)), "skip": dict(pool.skip_stats),
                         "train_seconds": t["train"], "observe_seconds": t["observe"], "seconds": res.wall_clock})


def run(bed: str, seed: int, arm: str) -> dict:
    out = run_u(bed[1:], seed, arm) if bed.startswith("u") else (run_f(int(bed[1:]), seed, arm) if bed in ("f10", "f30") else (run_t(seed, arm) if bed == "t" else run_bank(seed, arm)))
    print(_line(out), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--bed", default="f10", choices=BEDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[700])
    ap.add_argument("--arms", nargs="+", default=["off", "k0.1"])
    a = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "2")))
    for s in a.seeds:
        for arm in a.arms:
            run(a.bed, s, arm)
