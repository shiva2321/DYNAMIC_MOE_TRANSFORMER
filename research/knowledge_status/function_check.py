"""Checking by FUNCTION: does a taught skill reproduce the answer that comes with a message, from the numbers in it?

How a message reads changes with its wording; the numbers in it and the answer do not. So when an example arrives together with its answer (while teaching, while adapting, or in a worked
trace), the question "which taught skill's solver turns these two numbers into this answer?" separates a known skill in an unfamiliar wording (explained) from a skill that was never taught
(not explained) without relying on how the message looks. Only skills whose solver is COMPETENT (measured on problems it never saw) may explain: a weak solver cannot confirm anything.

Declared simplifications (Phase 1, spec 6.2): a fixed number reader takes the two numbers out of the message; it never reads an operator word or symbol (anti-goal A1), so the skill is not
decided by it. Operand ORDER depends on the wording ("5 subtracted from 17"), so both orders are tried. A misspelt number word is not read. Messages with other than two numbers are not explained.
Task 12 (worked traces) reuses ``parse_numbers`` and ``explain_many`` for the steps of a trace.
"""

import re
from typing import Collection, Dict, List, Mapping, Sequence

import numpy as np

_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_NUMBER_WORDS = {**_UNITS, **_TENS, "hundred": 100, "thousand": 1000}


def _words_value(group: List[str]) -> int:
    total = cur = 0
    for w in group:
        if w == "hundred":
            cur = (cur or 1) * 100
        elif w == "thousand":
            total += (cur or 1) * 1000
            cur = 0
        else:
            cur += _NUMBER_WORDS[w]
    return total + cur


def parse_numbers(text: str) -> List[int]:
    """The numbers of the message in the order they appear: digit groups if there are any (the word "one" in "in one bag" is then not a number), otherwise number words
    ("twenty-one" is 21). No operator word or symbol is read."""
    digits = re.findall(r"\d+", text)
    if digits:
        return [int(d) for d in digits]
    out: List[int] = []
    group: List[str] = []
    joined = False                                                       # a hyphen straight after a number word ("twenty-one") does not end the number
    for w in re.findall(r"[a-z]+|[^a-z\s]", text.lower()) + [""]:      # any other word or symbol, an operator word or a "?" alike, ends it
        if w in _NUMBER_WORDS:
            group.append(w)
            joined = False
        elif w == "-" and group and not joined:
            joined = True
        else:
            if group:
                out.append(_words_value(group))
                group = []
            joined = False
    return out


def explain_many(solvers: Mapping[str, object], competent: Collection[str], texts: Sequence[str], answers: Sequence[int]) -> List[List[str]]:
    """For each message the competent skills (in the order given) whose solver gives the stated answer from the two numbers, in either order. ``solvers[s].predict(a, b)`` takes and
    returns integer arrays. A zero divisor never explains anything for div and mod."""
    if len(texts) != len(answers):
        raise ValueError("texts and answers must have the same length")
    nums = [parse_numbers(t) for t in texts]
    rows = [i for i, n in enumerate(nums) if len(n) == 2]
    out: List[List[str]] = [[] for _ in texts]
    if not rows:
        return out
    a = np.array([nums[i][0] for i in rows], dtype=np.int64)
    b = np.array([nums[i][1] for i in rows], dtype=np.int64)
    ans = np.array([answers[i] for i in rows], dtype=np.int64)
    for skill in competent:
        hit = np.zeros(len(rows), dtype=bool)
        for x, y in ((a, b), (b, a)):
            ok = (y != 0) if skill in ("div", "mod") else np.ones(len(rows), dtype=bool)
            if ok.any():
                pred = np.zeros(len(rows), dtype=np.int64)
                pred[ok] = np.asarray(solvers[skill].predict(x[ok], y[ok]), dtype=np.int64)
                hit |= ok & (pred == ans)
        for k in np.flatnonzero(hit):
            out[rows[int(k)]].append(skill)
    return out


def explain_one(solvers: Mapping[str, object], competent: Collection[str], text: str, answer: int) -> List[str]:
    return explain_many(solvers, competent, [text], [answer])[0]
