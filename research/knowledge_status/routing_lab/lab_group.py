"""Lab B (group evidence for the function check). The parts the look-based machine calls unseen are taken in windows; a competent taught skill that explains MORE of a window's stated answers than a
shuffle of those answers would (permutation test, Holm over the skills) is present, and the parts it explains are taken back from "unseen". Ceilings with perfect solvers first, then the real ones.

    PYTHONPATH=<repo root> python research/knowledge_status/routing_lab/lab_group.py [first_seed last_seed]

Tuning seeds 150 to 154; one small result file per seed, written atomically. The base machine, the solvers and the test parts are those of lab_function.py (same seeds, so the rows are comparable
with results/FUNCTION_LAB.md and FUNCTION_CEILING.md).
"""

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
KS = HERE.parent
ROOT = KS.parents[1]
for _p in (str(ROOT), str(KS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import evaluate as EV  # noqa: E402
import function_check as FC  # noqa: E402
import group_check as GC  # noqa: E402
import lab_function as LF  # noqa: E402
import run_e2 as R  # noqa: E402
import runlib  # noqa: E402
import skills  # noqa: E402
import status as ST  # noqa: E402

OUT = KS / "results" / "routing_lab" / "group"
STAMP_FILES = [HERE / "group_check.py", HERE / "lab_group.py", KS / "lab_function.py", KS / "function_check.py", KS / "run_e2.py", KS / "status.py", KS / "evaluate.py", R.LEARNED_MAP]
SEEDS = list(range(150, 155))
SETTINGS: Tuple[Tuple[int, float], ...] = ((4, 0.05), (6, 0.01), (8, 0.01), (16, 0.01))      # (window size, threshold): a window of 4 cannot pass 0.01 (its smallest p is 1/24)
HELD = ("S4", "S5", "S6", "S7", "S8")


def apply(assessed: Sequence[ST.Assessment], changed: Dict[int, str], acc: Dict[str, float]) -> List[ST.Assessment]:
    out = list(assessed)
    for i, s in changed.items():
        out[i] = replace(assessed[i], status="solid", skill=s, competence=float(acc.get(s, 1.0)))
    return out


def windows_for(rows: List[int], cases, kind: str, g: int, rng: np.random.Generator) -> List[List[int]]:
    """'mixed': the flagged parts in a random order, cut into windows of g (what a stream gives); 'same_skill': sorted by true skill first (the best a grouping could do)."""
    rows = list(rows)
    if kind == "mixed":
        order = [rows[i] for i in rng.permutation(len(rows))]
    elif kind == "same_skill":
        order = sorted(rows, key=lambda r: (str(cases[r].skill), int(rng.integers(0, 1 << 30))))
    else:
        raise ValueError(kind)
    return [order[i:i + g] for i in range(0, len(order), g)]


def run_seed(seed: int, out_dir, stamp: str, settings: Sequence[Tuple[int, float]] = SETTINGS, n_perm: int = 2000) -> dict:
    path = Path(out_dir) / f"s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    world = R.build_world(seed, "dev", R.SIZES)
    P, index = R.prepare(world, "R5_R4_qwen_L26")
    V = lambda ts: P[[index[t] for t in ts]]
    groups = {s: ST.SkillGroup(s, V([e.span for e in world.teach[s]]), len(world.teach[s]), None, world.comp[s]["accuracy"]) for s in world.taught}
    other = ST.SkillGroup("other", V([i.text for i in world.teach_none]), len(world.teach_none), None, 1.0)
    cal_vec, cal_grp, _, _ = R.calibration_arrays(world, V)
    m = ST.StatusMachine(groups, other, cal_vec, cal_grp, q=R.HEADLINE_Q, seed=seed)
    cases, texts = world.test_cases, world.test_texts
    assessed = m.assess_many(V(texts))
    ex = LF.test_examples(world)
    answers = {i: ex[t].answer for i, (t, c) in enumerate(zip(texts, cases)) if c.kind == "arith" and t in ex}
    numbers = {i: FC.parse_numbers(texts[i]) for i in answers}
    real, acc, competent_real = LF.train_solvers(world)
    perfect = {s: GC.PerfectSolver(s) for s in skills.SKILLS}
    solid = [s for s in world.taught if world.assignment[s] == "solid"]
    variants = {"real": (real, competent_real), "perfect_solid": (perfect, solid), "perfect_taught": (perfect, list(world.taught))}
    accs = {"real": acc, "perfect_solid": {s: 1.0 for s in skills.SKILLS}, "perfect_taught": {s: 1.0 for s in skills.SKILLS}}
    flagged = [i for i, a in enumerate(assessed) if a.status == "unseen" and i in answers]
    nearest = {i: assessed[i].nearest for i in flagged}
    held = [i for i, c in enumerate(cases) if c.kind == "arith" and c.status in ("solid", "thin") and c.family in HELD]
    untaught = [i for i, c in enumerate(cases) if c.kind == "arith" and c.status == "unseen"]
    base = EV.status_scores(assessed, cases)
    rec = {"seed": seed, "code_stamp": stamp, "competent_real": competent_real, "solid": solid, "n_flagged": len(flagged), "n_held": len(held), "n_untaught": len(untaught),
           "base": {"false_new": base["false_unseen_heldout"], "found": base["unseen_recall"], "macro_f1": base["macro_f1"]["macro_f1"]}, "cells": {}}
    for vname, (solvers, comp) in variants.items():
        for kind in ("mixed", "same_skill"):
            for g, alpha in settings:
                rng = np.random.default_rng([seed, 0xB1, g, len(vname), 0 if kind == "mixed" else 1])
                wins = windows_for(flagged, cases, kind, g, rng)
                changed = GC.overturn(flagged, numbers, answers, solvers, comp, g, rng, nearest=nearest, alpha=alpha, n_perm=n_perm, grouping=wins)
                new = apply(assessed, changed, accs[vname])
                sc = EV.status_scores(new, cases)
                rec["cells"][f"{vname}|{kind}|{g}|{alpha}"] = {
                    "false_new": sc["false_unseen_heldout"], "found": sc["unseen_recall"], "macro_f1": sc["macro_f1"]["macro_f1"],
                    "coincidence": float(np.mean([i in changed for i in untaught])) if untaught else float("nan"),
                    "right_skill": float(np.mean([changed[i] == cases[i].skill for i in held if i in changed])) if any(i in changed for i in held) else float("nan"),
                    "rescued_share": float(np.mean([i in changed for i in held if assessed[i].status == "unseen"])) if any(assessed[i].status == "unseen" for i in held) else float("nan")}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(rec))
    return rec


def _mean(rows: List[dict], key: str, field: str) -> float:
    v = [r["cells"][key][field] for r in rows]
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def report(rows: List[dict], seeds: Sequence[int], settings: Sequence[Tuple[int, float]] = SETTINGS) -> str:
    b = lambda f: float(np.nanmean([r["base"][f] for r in rows]))
    L = [f"# Lab B, group evidence for the function check (tuning seeds {seeds[0]}-{seeds[-1]}, learned-map reader, status machine q = 0.05)", "",
         "Parts the machine called unseen by their look are taken in windows of g; a competent taught skill present in a window by a permutation test (Holm, threshold alpha) takes back the parts it explains.", "",
         f"No check: known skills in new wordings called unseen (bar 0.05) {b('false_new'):.3f}; never-taught skills found {b('found'):.3f}; macro F1 {b('macro_f1'):.3f}.",
         f"Flagged parts per seed (mean): {np.mean([r['n_flagged'] for r in rows]):.0f}; competent real skills per seed: " + "; ".join(f"{r['seed']}: {','.join(r['competent_real']) or 'none'}" for r in rows), "",
         "| solvers | windows | g, alpha | false-new | found | macro F1 | coincidences (untaught taken back) | right skill | flagged known parts rescued |", "|---|---|---|---|---|---|---|---|---|"]
    for v in ("real", "perfect_solid", "perfect_taught"):
        for kind in ("mixed", "same_skill"):
            for g, a in settings:
                k = f"{v}|{kind}|{g}|{a}"
                L.append(f"| {v} | {kind} | {g}, {a} | {_mean(rows, k, 'false_new'):.3f} | {_mean(rows, k, 'found'):.3f} | {_mean(rows, k, 'macro_f1'):.3f} | {_mean(rows, k, 'coincidence'):.3f} | "
                         f"{_mean(rows, k, 'right_skill'):.3f} | {_mean(rows, k, 'rescued_share'):.3f} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (SEEDS[0], SEEDS[-1])
    runlib.set_threads(3)
    stamp = runlib.code_stamp(STAMP_FILES)
    rows = []
    for seed in range(lo, hi + 1):
        r = run_seed(seed, OUT, stamp)
        rows.append(r)
        c = r["cells"]["perfect_taught|mixed|8|0.01"]
        print(f"seed {seed}: base false-new {r['base']['false_new']:.3f} found {r['base']['found']:.3f}; perfect-taught g=8: false-new {c['false_new']:.3f} found {c['found']:.3f} coincidence {c['coincidence']:.3f}", flush=True)
    text = report(rows, list(range(lo, hi + 1)))
    runlib.atomic_write_text(KS / "results" / f"ROUTING_GROUP_{lo}_{hi}.md", text)
    print(text)
