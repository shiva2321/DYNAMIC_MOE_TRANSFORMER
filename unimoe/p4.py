"""P4 drivers: the four improvements, tuned on seed 499 and evaluated on 500-502 (confirmation 503-504). Cells are cached like the main grid (stamp = result-defining code + the new pool modules).

    python -m unimoe.p4 capacity-widths        # I1a: does width matter on the capacity stream?
    python -m unimoe.p4 capacity-tune          # I1: growth threshold on seed 499, 16-unit start
    python -m unimoe.p4 capacity-eval          # I1: growth from 16 units on seeds 500-502
    python -m unimoe.p4 oracle-lda             # I4: LDA blend under oracle routing, all schedules
    python -m unimoe.p4 main-tune SCHEDULE     # domain power and LDA weight on seed 499
    python -m unimoe.p4 main-run [--seeds ...] # UG0 UG1 UDOMS ULDA UPLUS on every schedule
    python -m unimoe.p4 table GROUP            # final means of one group
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from . import capacity as CP
from . import experiment as E
from . import runlib
from . import systems2 as S2
from .frontends import SphereFrontend
from .runner import run_cell

TUNE_SEED, EVAL, CONFIRM = 499, (500, 501, 502), (503, 504)
GRID_THETA = (0.02, 0.05, 0.10, 0.20, 0.40)
GRID_BETA = (0.5, 1.0, 2.0)
GRID_LAMBDA = (0.25, 0.5, 0.75)
SCHEDULES = ("seq", "mix", "drift")
TUNING = E.OUT / "p4_tuning.json"
_CAP: Dict[tuple, object] = {}


def stamp2() -> str:
    files = [E.ROOT / "unimoe" / f"{n}.py" for n in ("pools", "heads", "systems2", "capacity")]
    return hashlib.sha1((E.stamp() + runlib.code_stamp(files)).encode()).hexdigest()[:16]


def cap_stream(seed: int, schedule: str):
    if (seed, schedule) not in _CAP:
        _CAP[(seed, schedule)] = CP.build_capacity_suite(schedule, seed, SphereFrontend(d=128, cache_dir=E.OUT / "emb"))
    return _CAP[(seed, schedule)]


def cell2(name: str, stream, seed: int, recipe: dict, group: str, tag: str):
    blocks = [tuple(v) for v in stream.label_blocks.values()]
    return run_cell(E.OUT / f"p4_{group}", tag, lambda: S2.make_system2(name, stream.n_classes, seed, recipe, blocks), stream, seed, stamp2())


def say(seed, sch, tag, rec) -> None:
    ev = rec.per_phase[-1]["events"]
    print(f"seed {seed} {sch:<5s} {tag:<26s} final mean {E.final_mean(rec):.3f} experts {rec.per_phase[-1]['n_experts']:>3d} grow {ev.get('grow', 0)} params {rec.report['params']:>8d} {rec.seconds:5.0f}s", flush=True)


def load_tuning() -> dict:
    return json.loads(TUNING.read_text(encoding="utf-8")) if TUNING.exists() else {}


def save_tuning(update: dict) -> None:
    t = load_tuning()
    t.update(update)
    runlib.atomic_write_text(TUNING, json.dumps(t, indent=1))


# ------------------------------------------------------------------------------------------------------------------------------ I1: capacity
def capacity_widths(seeds=(TUNE_SEED,) + EVAL) -> None:
    for seed in seeds:
        for sch in ("seq", "mix"):
            st = cap_stream(seed, sch)
            for w in (16, 128, 512):
                say(seed, sch, f"fixed-h{w}", cell2("UG0", st, seed, {"hidden": w, "grow": False, "inner_steps": 20}, "capacity", f"fixed-h{w}"))


def capacity_tune() -> None:
    st = cap_stream(TUNE_SEED, "seq")
    res = {}
    for th in GRID_THETA:
        rec = cell2("UG1", st, TUNE_SEED, {"hidden": 16, "inner_steps": 20, "grow_gap": th}, "capacity_tune", f"UG1-h16-theta{th}")
        say(TUNE_SEED, "seq", f"UG1-h16-theta{th}", rec)
        res[th] = E.final_mean(rec)
    best = max(res, key=res.get)
    save_tuning({"theta": best, "theta_grid": {str(k): v for k, v in res.items()}})
    print("chosen threshold", best)


def capacity_eval(seeds=EVAL) -> None:
    theta = load_tuning()["theta"]
    for seed in seeds:
        for sch in ("seq", "mix"):
            st = cap_stream(seed, sch)
            base = {"hidden": 16, "inner_steps": 20}
            say(seed, sch, "shipped-gap-fast-h16", cell2("UG0", st, seed, {**base, **S2.FAST, "growth_rule": "gap"}, "capacity_growth", "shipped-gap-fast-h16"))
            say(seed, sch, "UG1-h16", cell2("UG1", st, seed, {**base, "grow_gap": theta}, "capacity_growth", "UG1-h16"))
            say(seed, sch, "UG1-h16-s5", cell2("UG1", st, seed, {**base, "inner_steps": 5, "grow_gap": theta}, "capacity_growth", "UG1-h16-s5"))
            say(seed, sch, "fixed-h16-s5", cell2("UG0", st, seed, {**base, "inner_steps": 5, "grow": False}, "capacity_growth", "fixed-h16-s5"))


# ------------------------------------------------------------------------------------------------------------------------------ I4: LDA under oracle routing
def oracle_lda(seeds=(TUNE_SEED,) + EVAL) -> None:
    for seed in seeds:
        for sch in SCHEDULES:
            st = E.streams(seed)[sch]
            base = E.load_tuning(sch)["oracle_experts"]
            for lam in (0.0, 0.3, 0.5, 0.7, 1.0):
                say(seed, sch, f"oracle_lda_{lam}", cell2("oracle_lda", st, seed, {**base, "lda_blend": lam}, "oracle_lda", f"oracle_lda_{lam}"))


# ------------------------------------------------------------------------------------------------------------------------------ main suite
def main_tune(schedule: str) -> None:
    st = E.streams(TUNE_SEED)[schedule]
    base = dict(E.load_tuning(schedule)["U-real"])
    out = {"beta": {}, "lambda": {}}
    for b in GRID_BETA:
        rec = cell2("UDOMS", st, TUNE_SEED, {**base, "domain_beta": b}, "main_tune", f"UDOMS-{schedule}-beta{b}")
        say(TUNE_SEED, schedule, f"UDOMS beta {b}", rec)
        out["beta"][b] = E.final_mean(rec)
    rec = cell2("UDOM", st, TUNE_SEED, base, "main_tune", f"UDOM-{schedule}")
    say(TUNE_SEED, schedule, "UDOM (hard)", rec)
    out["hard"] = E.final_mean(rec)
    for lam in GRID_LAMBDA:
        rec = cell2("ULDA", st, TUNE_SEED, {**base, "lda_blend": lam}, "main_tune", f"ULDA-{schedule}-lam{lam}")
        say(TUNE_SEED, schedule, f"ULDA lambda {lam}", rec)
        out["lambda"][lam] = E.final_mean(rec)
    best_beta = max(out["beta"], key=out["beta"].get)
    dom = {"domain_head": "soft", "domain_beta": best_beta} if out["beta"][best_beta] >= out["hard"] else {"domain_head": "hard"}
    save_tuning({schedule: {"domain": dom, "lda_blend": max(out["lambda"], key=out["lambda"].get), "grid": {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in out.items()}}})
    print(schedule, "chosen", load_tuning()[schedule]["domain"], load_tuning()[schedule]["lda_blend"])


def main_run(seeds=EVAL, schedules=SCHEDULES, systems=("UG0", "UG1", "UDOM", "UDOMS", "ULDA", "UPLUS")) -> None:
    tun = load_tuning()
    for seed in seeds:
        for sch in schedules:
            st = E.streams(seed)[sch]
            base = dict(E.load_tuning(sch)["U-real"])
            t = tun[sch]
            recipes = {"UG0": base, "UDOM": base, "UDOMS": {**base, "domain_beta": t["domain"].get("domain_beta", 1.0)}, "ULDA": {**base, "lda_blend": t["lda_blend"]}}
            if "theta" in tun:                                                                           # the growth threshold comes from the capacity stream (seed 499)
                recipes["UG1"] = {**base, "grow_gap": tun["theta"]}
                recipes["UPLUS"] = {**base, "grow_gap": tun["theta"], "lda_blend": t["lda_blend"], **t["domain"]}
            for name in systems:
                if name not in recipes:
                    print(f"skipped {name}: no growth threshold tuned yet", flush=True)
                    continue
                say(seed, sch, name, cell2(name, st, seed, recipes[name], "main", name))


def table(group: str) -> List[str]:
    from . import p3
    return p3.table(group)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("capacity-widths", "capacity-tune", "capacity-eval", "oracle-lda", "main-tune", "main-run", "table"))
    ap.add_argument("name", nargs="?", default="")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    a = ap.parse_args(argv)
    kw = {"seeds": tuple(a.seeds)} if a.seeds else {}
    if a.cmd == "capacity-widths":
        capacity_widths(**kw)
    elif a.cmd == "capacity-tune":
        capacity_tune()
    elif a.cmd == "capacity-eval":
        capacity_eval(**kw)
    elif a.cmd == "oracle-lda":
        oracle_lda(**kw)
    elif a.cmd == "main-tune":
        main_tune(a.name)
    elif a.cmd == "main-run":
        main_run(**kw)
    else:
        print("\n".join(table(a.name)))


if __name__ == "__main__":
    main()
