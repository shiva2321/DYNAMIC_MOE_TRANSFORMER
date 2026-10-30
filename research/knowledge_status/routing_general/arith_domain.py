"""Arithmetic in everyday language as a domain world of the general routing lab.

The only place where the general lab meets the arithmetic test bed: the teacher's names for the skills, and the mapping of the test bed's parts onto the generic sets (known messages in
NEW wordings S4-S8 of the taught skills, never-taught skills, foreign sentences) plus an extra set of look-alike traps (sentences that use arithmetic words for something else).
Calibration is the test bed's own: held-out messages of the taught skills in the taught wordings S1-S3 only, so the known test is a real wording shift.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_KS = _HERE.parent
for _p in (str(_KS.parents[1]), str(_KS), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import assignments  # noqa: E402
import cards as CD  # noqa: E402
import domains as DM  # noqa: E402
import run_e2 as R  # noqa: E402

TEACHER_NAMES = {"add": "addition", "sub": "subtraction", "mul": "multiplication", "div": "division (the whole-number quotient)", "mod": "the remainder after a division",
                 "max": "the larger of two numbers"}
HELD = frozenset(assignments.TEST_FAMILIES)


def build(seed: int) -> Tuple[DM.DomainWorld, List[str], Dict[str, List[str]]]:
    """(the world, the foreign messages, the extra sets) of one seed."""
    w = R.build_world(seed, "dev", R.SIZES)
    taught = list(w.taught)
    names = [TEACHER_NAMES[s] for s in taught]
    teach = {names[i]: [e.span for e in w.teach[s]] for i, s in enumerate(taught)}
    known_cal = [(t, i) for i, s in enumerate(taught) for t in w.cal[s]]
    known, unknown, unknown_labels, foreign, traps = [], [], [], [], []
    for text, c in zip(w.test_texts, w.test_cases):
        if c.kind == "arith" and c.skill in taught and c.status in ("solid", "thin") and c.family in HELD:
            known.append((text, taught.index(c.skill)))
        elif c.kind == "arith" and c.status == "unseen":
            unknown.append(text)
            unknown_labels.append(c.skill)
        elif c.kind in ("other", "unseen_content"):
            foreign.append(text)
        elif c.kind == "trap":
            traps.append(text)
    cards = [CD.Card(n, tuple(teach[n][:DM.N_EXEMPLARS])) for n in names]
    states = ["solid" if w.assignment[s] == "solid" else "thin" for s in taught]
    return DM.DomainWorld("arithmetic", names, cards, teach, known_cal, known, unknown, unknown_labels, states), foreign, {"trap": traps}
