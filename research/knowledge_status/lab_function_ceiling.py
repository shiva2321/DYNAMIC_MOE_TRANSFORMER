"""Ceiling of the function check (side check, not part of the suite): what would it give if the solvers were PERFECT? Same tuning seeds 150 to 154, same machine; nothing is trained.
(i) perfect solvers for the Solid-state skills only; (ii) perfect solvers for every taught skill, the 20-example ones too (a best case no learner has reached).

    PYTHONPATH=<repo root> python lab_function_ceiling.py > results/FUNCTION_CEILING.md
"""

import sys
from typing import Dict, List

import numpy as np

import assignments
import evaluate as EV
import lab_function as LF
import run_e2 as R
import skills
import status as ST


class PerfectSolver:
    """Always right; a zero divisor returns something that cannot match (``explain_many`` skips it anyway)."""

    def __init__(self, skill: str) -> None:
        self.skill = skill

    def predict(self, a, b):
        return np.array([skills.truth(self.skill, int(x), int(y)) if not (self.skill in ("div", "mod") and int(y) == 0) else -10 ** 9 for x, y in zip(a, b)], dtype=np.int64)


def main(seeds) -> None:
    rows: Dict[str, List[dict]] = {"i: perfect for Solid-state skills": [], "ii: perfect for every taught skill": []}
    base: List[dict] = []
    for seed in seeds:
        world = R.build_world(seed, "dev", R.SIZES)
        P, index = R.prepare(world, "R5_R4_qwen_L26")
        V = lambda ts: P[[index[t] for t in ts]]
        groups = {s: ST.SkillGroup(s, V([e.span for e in world.teach[s]]), len(world.teach[s]), None, world.comp[s]["accuracy"]) for s in world.taught}
        other = ST.SkillGroup("other", V([i.text for i in world.teach_none]), len(world.teach_none), None, 1.0)
        cal_vec, cal_grp, _, _ = R.calibration_arrays(world, V)
        m = ST.StatusMachine(groups, other, cal_vec, cal_grp, q=R.HEADLINE_Q, seed=seed)
        assessed = m.assess_many(V(world.test_texts))
        ex = LF.test_examples(world)
        cases, texts = world.test_cases, world.test_texts
        answers = [ex[t].answer if c.kind == "arith" and t in ex else None for t, c in zip(texts, cases)]
        solvers = {s: PerfectSolver(s) for s in world.taught}
        acc = {s: 1.0 for s in world.taught}
        held = [i for i, c in enumerate(cases) if c.kind == "arith" and c.status in ("solid", "thin") and c.family in assignments.TEST_FAMILIES]
        untaught = [i for i, c in enumerate(cases) if c.kind == "arith" and c.status == "unseen"]
        b = EV.status_scores(assessed, cases)
        base.append({"false": b["false_unseen_heldout"], "found": b["unseen_recall"], "f1": b["macro_f1"]["macro_f1"]})
        for name, competent in (("i: perfect for Solid-state skills", [s for s in world.taught if world.assignment[s] == "solid"]), ("ii: perfect for every taught skill", list(world.taught))):
            new, which = LF.apply_function_check(assessed, texts, answers, solvers, competent, acc)
            a = EV.status_scores(new, cases)
            over = [i for i in held if which[i] is not None]
            rows[name].append({"false": a["false_unseen_heldout"], "found": a["unseen_recall"], "f1": a["macro_f1"]["macro_f1"],
                               "right": float(np.mean([which[i] == cases[i].skill for i in over])) if over else float("nan"),
                               "coincidence": float(np.mean([which[i] is not None for i in untaught])),
                               "stories": float(np.mean([new[i].status == "unseen" for i in held if cases[i].family == "S4"])),
                               "noise": float(np.mean([new[i].status == "unseen" for i in held if cases[i].family == "S7"]))})
        print(f"seed {seed} done", file=sys.stderr, flush=True)
    mean = lambda rs, k: float(np.nanmean([r[k] for r in rs]))
    print(f"# Ceiling of the function check with perfect solvers (tuning seeds {min(seeds)}-{max(seeds)})\n")
    print("| | known skills called unseen (bar 0.05) | stories S4 | noisy text S7 | never-taught skills found | macro F1 | right skill when overturned | coincidences |\n|---|---|---|---|---|---|---|---|")
    print(f"| no function check | {mean(base, 'false'):.3f} | n/a | n/a | {mean(base, 'found'):.3f} | {mean(base, 'f1'):.3f} | n/a | n/a |")
    for name, rs in rows.items():
        print(f"| {name} | {mean(rs, 'false'):.3f} | {mean(rs, 'stories'):.3f} | {mean(rs, 'noise'):.3f} | {mean(rs, 'found'):.3f} | {mean(rs, 'f1'):.3f} | {mean(rs, 'right'):.3f} | {mean(rs, 'coincidence'):.3f} |")


if __name__ == "__main__":
    main(range(150, 155))
