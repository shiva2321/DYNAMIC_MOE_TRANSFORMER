"""Lab D (teams of experts), offline from the cached asked probabilities of the tuning seeds.

    PYTHONPATH=<repo root> python research/knowledge_status/routing_lab/lab_team.py [first_seed last_seed]

Known-skill parts in the new wordings S4-S8 (and the taught wordings S1-S3 for reference), the operands and the stated answer of each part from the generator (reading the numbers is a separate
part of the system and is taken as given here). Two settings of who may be in the team and who may verify: every taught skill (a ceiling: all experts competent), and only the Solid ones (the
usable experts). No model is run.
"""

import sys
from pathlib import Path
from typing import List, Sequence

HERE = Path(__file__).resolve().parent
KS = HERE.parent
ROOT = KS.parents[1]
for _p in (str(ROOT), str(KS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import ask_reader as AR  # noqa: E402
import lab_function as LF  # noqa: E402
import run_e2 as R  # noqa: E402
import runlib  # noqa: E402
import status as ST  # noqa: E402
import team as TM  # noqa: E402

CACHE_DIR = KS / "results" / "routing_lab" / "ask"
HELD = ("S4", "S5", "S6", "S7", "S8")
TAUGHT_W = ("S1", "S2", "S3")


def never(_texts):
    raise RuntimeError("the cache is missing answers; run lab_ask.py for this seed first")


def seed_rows(seed: int) -> dict:
    world = R.build_world(seed, "dev", R.SIZES)
    P = lambda ts: AR.probabilities(ts, CACHE_DIR / f"probs_s{seed}.json", never)
    states = {s: ST.group_state(world.comp[s]["count"], world.comp[s]["accuracy"]) for s in world.taught}
    machine = AR.AskMachine({**{s: P(world.cal[s]) for s in world.taught}, "none": P(world.cal["other"])}, states, {s: world.comp[s]["accuracy"] for s in world.taught},
                            {s: world.comp[s]["count"] for s in world.taught}, q=0.05)
    ex = LF.test_examples(world)
    Pt = P(world.test_texts)
    cov = machine.covering(Pt)
    out = {"held": [], "taught_wordings": []}
    for i, (t, c) in enumerate(zip(world.test_texts, world.test_cases)):
        if c.kind != "arith" or c.skill not in world.taught or t not in ex:
            continue
        e = ex[t]
        row = {"p_row": Pt[i], "covered_row": cov[i], "a": int(e.a), "b": int(e.b), "answer": int(e.answer), "skill": c.skill}
        if c.family in HELD:
            out["held"].append(row)
        elif c.family in TAUGHT_W:
            out["taught_wordings"].append(row)
    solid = [s for s in world.taught if states[s] == "solid"]
    return {"rows": out, "known": machine.known, "taught": list(world.taught), "solid": solid}


def report(per_seed: List[dict], seeds: Sequence[int]) -> str:
    L = [f"# Lab D, teams of experts (seeds {seeds[0]}-{seeds[-1]}, from the cached asked probabilities)", "",
         "Known skills in the new wordings S4-S8. 'right' = the team settles on the true skill. 'recall' = the true skill is in the team. 'size' = experts called per part. With an answer = the stated answer is "
         "checked against each member that may verify. Operands are taken as given (reading the numbers is a separate part).", "",
         "| experts allowed (team and verifiers) | policy | recall | size | right, no answer | right, with the stated answer |", "|---|---|---|---|---|---|"]
    for setting in ("every taught skill", "only the Solid skills"):
        for pol in TM.POLICIES:
            res = []
            for d in per_seed:
                allowed = d["taught"] if setting == "every taught skill" else d["solid"]
                if not allowed:
                    continue
                rows = d["rows"]["held"]
                rows = [r for r in rows if r["skill"] in allowed] if setting != "every taught skill" else rows
                if not rows:
                    continue
                res.append((TM.evaluate(pol, rows, d["known"], allowed, allowed, False), TM.evaluate(pol, rows, d["known"], allowed, allowed, True)))
            if res:
                m = lambda k, j: float(np.mean([r[j][k] for r in res]))
                L.append(f"| {setting} | {pol} | {m('recall', 0):.3f} | {m('size', 0):.2f} | {m('right', 0):.3f} | {m('right', 1):.3f} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (150, 154)
    per = [seed_rows(s) for s in range(lo, hi + 1)]
    text = report(per, list(range(lo, hi + 1)))
    runlib.atomic_write_text(KS / "results" / f"ROUTING_TEAM_{lo}_{hi}.md", text)
    print(text)
