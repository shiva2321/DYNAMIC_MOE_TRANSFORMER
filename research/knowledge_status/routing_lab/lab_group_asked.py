"""Lab F (group evidence with windows made from the asked operation). Same parts, base machine, solvers and seeds as lab_group.py (Lab B); the flagged parts are grouped by the operation the
model named for them (read from the cache of lab A), so the windows are nearly one skill.

    PYTHONPATH=<repo root> python research/knowledge_status/routing_lab/lab_group_asked.py [first_seed last_seed]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
KS = HERE.parent
ROOT = KS.parents[1]
for _p in (str(ROOT), str(KS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import ask_reader as AR  # noqa: E402
import evaluate as EV  # noqa: E402
import function_check as FC  # noqa: E402
import group_check as GC  # noqa: E402
import lab_function as LF  # noqa: E402
import lab_group as LG  # noqa: E402
import run_e2 as R  # noqa: E402
import runlib  # noqa: E402
import skills  # noqa: E402
import status as ST  # noqa: E402

OUT = KS / "results" / "routing_lab" / "group_asked"
ASK_CACHE = KS / "results" / "routing_lab" / "ask"
STAMP_FILES = [HERE / "group_check.py", HERE / "lab_group.py", HERE / "lab_group_asked.py", HERE / "ask_reader.py", KS / "lab_function.py", KS / "function_check.py", KS / "run_e2.py",
               KS / "status.py", KS / "evaluate.py", R.LEARNED_MAP]
SEEDS = list(range(150, 155))
SETTINGS: Tuple[Tuple[int, float], ...] = ((6, 0.01), (8, 0.01), (16, 0.01))


def windows_by_name(rows: Sequence[int], names: Dict[int, str], g: int, rng: np.random.Generator) -> List[List[int]]:
    """The rows grouped by the name the model gave them (rows of a name in a random order), each group cut into windows of g; the groups are visited in a fixed order."""
    out: List[List[int]] = []
    for name in sorted({names[r] for r in rows}):
        grp = [r for r in rows if names[r] == name]
        grp = [grp[i] for i in rng.permutation(len(grp))]
        out += [grp[i:i + g] for i in range(0, len(grp), g)]
    return out


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
    never = lambda ts: (_ for _ in ()).throw(RuntimeError("the asked-operation answers are missing; run lab A for this seed first"))
    Pa = AR.probabilities(texts, ASK_CACHE / f"probs_s{seed}.json", never)
    asked = {i: AR.OPS[int(Pa[i].argmax())] for i in range(len(texts))}
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
    held = [i for i, c in enumerate(cases) if c.kind == "arith" and c.status in ("solid", "thin") and c.family in LG.HELD]
    untaught = [i for i, c in enumerate(cases) if c.kind == "arith" and c.status == "unseen"]
    base = EV.status_scores(assessed, cases)
    pure = float(np.mean([asked[i] == cases[i].skill for i in flagged if cases[i].skill is not None])) if flagged else float("nan")
    rec = {"seed": seed, "code_stamp": stamp, "n_flagged": len(flagged), "competent_real": competent_real, "name_right_among_flagged": pure,
           "base": {"false_new": base["false_unseen_heldout"], "found": base["unseen_recall"]}, "cells": {}}
    for vname, (solvers, comp) in variants.items():
        for g, alpha in settings:
            rng = np.random.default_rng([seed, 0xF1, g, len(vname)])
            wins = windows_by_name(flagged, asked, g, rng)
            changed = GC.overturn(flagged, numbers, answers, solvers, comp, g, rng, nearest=nearest, alpha=alpha, n_perm=n_perm, grouping=wins)
            sc = EV.status_scores(LG.apply(assessed, changed, accs[vname]), cases)
            rec["cells"][f"{vname}|{g}|{alpha}"] = {
                "false_new": sc["false_unseen_heldout"], "found": sc["unseen_recall"],
                "coincidence": float(np.mean([i in changed for i in untaught])) if untaught else float("nan"),
                "right_skill": float(np.mean([changed[i] == cases[i].skill for i in held if i in changed])) if any(i in changed for i in held) else float("nan")}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(R._native(rec)))
    return rec


def report(rows: List[dict], seeds: Sequence[int], settings: Sequence[Tuple[int, float]] = SETTINGS) -> str:
    mean = lambda f: float(np.nanmean([f(r) for r in rows]))
    L = [f"# Lab F, group evidence with windows made from the asked operation (tuning seeds {seeds[0]}-{seeds[-1]})", "",
         f"No check: known skills in new wordings called unseen {mean(lambda r: r['base']['false_new']):.3f}; never-taught skills found {mean(lambda r: r['base']['found']):.3f}. "
         f"Among the flagged parts the model's name is the true skill {mean(lambda r: r['name_right_among_flagged']):.3f} of the time (the windows are that pure).", "",
         "| solvers | g, alpha | false-new | found | coincidences | right skill |", "|---|---|---|---|---|---|"]
    for v in ("real", "perfect_solid", "perfect_taught"):
        for g, a in settings:
            k = f"{v}|{g}|{a}"
            L.append(f"| {v} | {g}, {a} | {mean(lambda r: r['cells'][k]['false_new']):.3f} | {mean(lambda r: r['cells'][k]['found']):.3f} | {mean(lambda r: r['cells'][k]['coincidence']):.3f} | "
                     f"{mean(lambda r: r['cells'][k]['right_skill']):.3f} |")
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
        c = r["cells"]["perfect_taught|16|0.01"]
        print(f"seed {seed}: base false-new {r['base']['false_new']:.3f}; perfect-taught g=16: false-new {c['false_new']:.3f} found {c['found']:.3f} coincidence {c['coincidence']:.3f}", flush=True)
    text = report(rows, list(range(lo, hi + 1)))
    runlib.atomic_write_text(KS / "results" / f"ROUTING_GROUP_ASKED_{lo}_{hi}.md", text)
    print(text)
