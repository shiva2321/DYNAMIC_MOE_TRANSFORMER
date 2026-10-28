"""Function-check lab (user decision 2026-10-28: option 2, the function check first). When the status machine calls a part "unseen" by its LOOK, does the answer that comes with the part overturn
that call, because a competent taught skill reproduces it from the numbers in the message? Tuning seeds 150 to 154; the plan, the predictions and the rule for going on are in PROGRESS.md
(2026-10-28), fixed before this was run. One small result file per seed, written atomically.

    PYTHONPATH=<repo root> python lab_function.py [first_seed last_seed]

Only SOLID skills with a solver measured at the bar (accuracy 0.90 on problems it never saw) may explain; a skill taught 20 examples, or whose solver is weak, cannot confirm anything, so its
parts keep the look-based call. The check only ever removes "unseen" calls, so the false-unseen guarantee on the taught wordings is kept.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import assignments
import evaluate as EV
import function_check as FC
import run_e2 as R
import runlib
import skills
import status as ST
from growmoe import arith

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "function_lab"
STAMP_FILES = R.STAMP_FILES + [HERE / "lab_function.py", HERE / "function_check.py", R.LEARNED_MAP]
SEEDS = list(range(150, 155))


def apply_function_check(assessed: Sequence[ST.Assessment], texts: Sequence[str], answers: Sequence[Optional[int]], solvers: Mapping[str, object], competent: Sequence[str],
                         acc: Mapping[str, float]) -> Tuple[List[ST.Assessment], List[Optional[str]]]:
    """Overturn the "unseen" calls that a competent skill explains: the part goes to that skill (the nearest one when several explain it), status solid. Returns the new assessments and,
    per part, the skill that overturned it (None if untouched). Parts without an answer, and every call other than "unseen", are left exactly as they were."""
    if not (len(assessed) == len(texts) == len(answers)):
        raise ValueError(f"assessed, texts and answers must have the same length, got {len(assessed)}, {len(texts)}, {len(answers)}")
    idx = [i for i, (a, ans) in enumerate(zip(assessed, answers)) if a.status == "unseen" and ans is not None]
    explained = FC.explain_many(solvers, competent, [texts[i] for i in idx], [int(answers[i]) for i in idx]) if idx else []
    out, which = list(assessed), [None] * len(assessed)
    for i, hits in zip(idx, explained):
        if hits:
            s = assessed[i].nearest if assessed[i].nearest in hits else hits[0]
            out[i] = replace(assessed[i], status="solid", skill=s, competence=float(acc[s]))
            which[i] = s
    return out, which


def test_examples(world: R.World) -> Dict[str, assignments.Example]:
    """The test parts of the world with their numbers and answers. ``build_world`` keeps only the texts, so the draws are repeated with the same streams and in the same order (they are
    deterministic) and the result is checked against the texts."""
    used = {R.fingerprint(e.text) for s in world.teach for e in world.teach[s]}
    for s in world.taught:
        R._fresh_taught_wording(R._stream(world.seed, 4, skills.SKILLS.index(s)), world.pool, s, world.sizes.cal, used)
    for s in world.taught:
        ex = assignments.draw_test(R._stream(world.seed, 5, skills.SKILLS.index(s)), world.pool, s, world.sizes.adapt_per_family * len(assignments.TEST_FAMILIES), frozenset(used))
        used |= {R.fingerprint(e.text) for e in ex}
    out: Dict[str, assignments.Example] = {}
    for s in skills.SKILLS:
        for e in assignments.draw_test(R._stream(world.seed, 6, skills.SKILLS.index(s)), world.pool, s, world.sizes.test_heldout, frozenset(used)):
            out.setdefault(e.span, e)
        if world.sizes.test_taught:
            for e in R._fresh_taught_wording(R._stream(world.seed, 7, skills.SKILLS.index(s)), world.pool, s, world.sizes.test_taught, used):
                out.setdefault(e.span, e)
    missing = [t for t, c in zip(world.test_texts, world.test_cases) if c.kind == "arith" and t not in out]
    if missing:
        raise RuntimeError(f"the answers of {len(missing)} test parts could not be reproduced, for example {missing[0]!r}")
    return out


def train_solvers(world: R.World, epochs: int = 1000) -> Tuple[Dict[str, assignments.Solver], Dict[str, float], List[str]]:
    """A solver per taught skill on its teaching examples, its accuracy on problems it never saw, and the COMPETENT skills: Solid state, count and accuracy at the Solid bar."""
    solvers: Dict[str, assignments.Solver] = {}
    acc: Dict[str, float] = {}
    lens = (1, 2)
    for s in world.taught:
        ex = world.teach[s]
        a, b, y = (np.array([getattr(e, k) for e in ex], dtype=np.int64) for k in ("a", "b", "answer"))
        solvers[s] = assignments.Solver(seed=world.seed * 10 + skills.SKILLS.index(s), epochs=epochs).fit(a, b, y)
        op = skills.ARITH_OP_ID[s]
        held = arith.sample_problems(np.random.default_rng([world.seed, 0xF1, skills.SKILLS.index(s)]), 500, (op,), lens, lens, "test")
        acc[s] = float((solvers[s].predict(held["a"], held["b"]) == arith.answers(held["op"], held["a"], held["b"])).mean())
    competent = [s for s in world.taught if world.assignment[s] == "solid" and ST.group_state(len(world.teach[s]), acc[s]) == "solid"]
    return solvers, acc, competent


def run_seed(seed: int, out_dir, stamp: str) -> dict:
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
    assessed = m.assess_many(V(world.test_texts))
    ex = test_examples(world)
    cases, texts = world.test_cases, world.test_texts
    answers = [ex[t].answer if c.kind == "arith" and t in ex else None for t, c in zip(texts, cases)]
    solvers, acc, competent = train_solvers(world)
    new, which = apply_function_check(assessed, texts, answers, solvers, competent, acc)
    before, after = EV.status_scores(assessed, cases), EV.status_scores(new, cases)
    held = [i for i, c in enumerate(cases) if c.kind == "arith" and c.status in ("solid", "thin") and c.family in assignments.TEST_FAMILIES]
    rate = lambda rows, a: float(np.mean([a[i].status == "unseen" for i in rows])) if rows else float("nan")
    comp_rows = [i for i in held if cases[i].skill in competent]
    other_rows = [i for i in held if cases[i].skill not in competent]
    untaught = [i for i, c in enumerate(cases) if c.kind == "arith" and c.status == "unseen"]
    fam_rows = lambda f: [i for i in held if cases[i].family == f]
    two = {f: float(np.mean([len(FC.parse_numbers(texts[i])) == 2 for i, c in enumerate(cases) if c.kind == "arith" and c.family == f])) for f in sorted({c.family for c in cases if c.kind == "arith"})}
    over_known = [i for i in held if which[i] is not None]
    rec = {"seed": seed, "code_stamp": stamp, "competent": competent, "solver_accuracy": acc, "assignment": world.assignment,
           "false_unseen": {"before": before["false_unseen_heldout"], "after": after["false_unseen_heldout"]},
           "found": {"before": before["unseen_recall"], "after": after["unseen_recall"]},
           "macro_f1": {"before": before["macro_f1"]["macro_f1"], "after": after["macro_f1"]["macro_f1"]},
           "false_unseen_competent": {"before": rate(comp_rows, assessed), "after": rate(comp_rows, new), "n": len(comp_rows)},
           "false_unseen_other_skills": {"before": rate(other_rows, assessed), "after": rate(other_rows, new), "n": len(other_rows)},
           "by_family": {f: {"before": rate(fam_rows(f), assessed), "after": rate(fam_rows(f), new), "n": len(fam_rows(f))} for f in ("S4", "S5", "S6", "S7", "S8")},
           "right_skill_when_overturned": float(np.mean([which[i] == cases[i].skill for i in over_known])) if over_known else float("nan"),
           "untaught_overturned": float(np.mean([which[i] is not None for i in untaught])) if untaught else float("nan"),
           "share_of_known_parts_competent": len(comp_rows) / max(len(held), 1), "two_numbers_found": two}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(rec))
    return rec


def report(rows: List[dict], seeds: Sequence[int]) -> str:
    mean = lambda f: float(np.nanmean([f(r) for r in rows]))
    L = [f"# Function-check lab (tuning seeds {seeds[0]}-{seeds[-1]}, learned-map reader, status machine q = 0.05)", "",
         "A part called unseen by its look is overturned when a COMPETENT taught skill (Solid state, solver at 0.90 on unseen problems) reproduces the stated answer from the numbers in the message.", "",
         "| | before | after |", "|---|---|---|",
         f"| known skills in new wordings called unseen (bar 0.05) | {mean(lambda r: r['false_unseen']['before']):.3f} | {mean(lambda r: r['false_unseen']['after']):.3f} |",
         f"| ... of competent skills (add, sub, max when Solid) | {mean(lambda r: r['false_unseen_competent']['before']):.3f} | {mean(lambda r: r['false_unseen_competent']['after']):.3f} |",
         f"| ... of the other taught skills (thin, weak solver) | {mean(lambda r: r['false_unseen_other_skills']['before']):.3f} | {mean(lambda r: r['false_unseen_other_skills']['after']):.3f} |",
         f"| never-taught skills found | {mean(lambda r: r['found']['before']):.3f} | {mean(lambda r: r['found']['after']):.3f} |",
         f"| macro F1 | {mean(lambda r: r['macro_f1']['before']):.3f} | {mean(lambda r: r['macro_f1']['after']):.3f} |", ""]
    for f in ("S4", "S5", "S6", "S7", "S8"):
        L.append(f"- wording {f}: called unseen {mean(lambda r: r['by_family'][f]['before']):.3f} -> {mean(lambda r: r['by_family'][f]['after']):.3f}")
    L += ["", f"- share of the known test parts that belong to competent skills (the most the check can rescue): {mean(lambda r: r['share_of_known_parts_competent']):.3f}",
          f"- when a known part is overturned, the explaining skill is the right one: {mean(lambda r: r['right_skill_when_overturned']):.3f}",
          f"- never-taught parts wrongly explained by a taught skill (coincidence): {mean(lambda r: r['untaught_overturned']):.3f}",
          "- two numbers found in the message: " + ", ".join(f"{f} {np.mean([r['two_numbers_found'].get(f, float('nan')) for r in rows]):.3f}" for f in sorted(rows[0]['two_numbers_found'])),
          "- competent skills per seed: " + "; ".join(f"{r['seed']}: {','.join(r['competent']) or 'none'}" for r in rows), ""]
    return "\n".join(L)


if __name__ == "__main__":
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (150, 154)
    runlib.set_threads(3)
    stamp = runlib.code_stamp(STAMP_FILES)
    rows = []
    for seed in range(lo, hi + 1):
        r = run_seed(seed, OUT, stamp)
        rows.append(r)
        print(f"seed {seed}: false alarm {r['false_unseen']['before']:.3f} -> {r['false_unseen']['after']:.3f}  found {r['found']['before']:.3f} -> {r['found']['after']:.3f}  competent {r['competent']}", flush=True)
    text = report(rows, list(range(lo, hi + 1)))
    runlib.atomic_write_text(HERE / "results" / "FUNCTION_LAB.md", text)
    print(text)
