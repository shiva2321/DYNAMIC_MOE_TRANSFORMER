"""Worked traces and relation (spec 5.6, E4). "3 times 4: 4 + 4 = 8, 8 + 4 = 12, so 12" is a worked example whose steps are instances of a skill the system already knows (addition); a division
is repeated subtraction. If every step is a verified instance of a known skill and the steps chain to the stated answer, that skill is named as a PARENT of the traced skill.

What makes a step an instance of a known skill (spec 5.6): the machine assigns the step to skill s AS A SOLID SKILL, AND the taught solver of s reproduces the step's result. The numbers are read by
the fixed number reader of ``function_check`` (declared simplification); no operator symbol or word is read to decide the skill (anti-goal A1). The chain check is arithmetic on those numbers:
an ADDITIVE chain starts from the repeated number (b + b), adds b every time, takes a - 1 steps and ends at the stated answer; a SUBTRACTIVE chain starts from a, subtracts b every time, takes as
many steps as the stated answer and ends below b. The parent set is the set of skills with at least one verified step in a VALID chain; a chain that is not valid names no parent.

Decoys: "wrong_steps" (the chain does not reach the answer), "irrelevant_valid_steps" (a maximum request with additions that are correct and have nothing to do with the answer),
"unrelated_skill" (a remainder request whose steps are additions that do not compute it).
"""

import re
from dataclasses import dataclass, replace
from typing import Callable, FrozenSet, List, Optional, Tuple

import numpy as np

import function_check as FC
import generator as G
import status as ST

N_VARIANTS = 4
DECOYS = ("wrong_steps", "irrelevant_valid_steps", "unrelated_skill")
_OPERATORS = re.compile(r"\b(?:plus|minus)\b|\+|(?<=\s)-(?=\s)")


@dataclass(frozen=True)
class Step:
    text: str
    skill: Optional[str]                 # the skill the step really is (ground truth, never read by the verifier)
    a: int
    b: int
    result: int


@dataclass(frozen=True)
class Trace:
    text: str
    header_skill: str
    a: int
    b: int
    answer: int
    steps: Tuple[Step, ...]
    decoy: Optional[str]


@dataclass(frozen=True)
class TraceVerdict:
    parents: FrozenSet[str]
    chain_valid: bool
    verified_steps: int


def replace_step_text(step: Step, text: str) -> Step:
    return replace(step, text=text)


def mask_operator(text: str) -> str:
    """The operator word or symbol of a step replaced by a question mark (to show the verifier does not read it)."""
    return _OPERATORS.sub("?", text)


# ---------------------------------------------------------------------------------------------------------------------------------- making traces
def _num(n: int, variant: int) -> str:
    return G.number_to_words(n) if variant == 2 else str(n)


def _step_text(x: int, y: int, r: int, op: str, variant: int) -> str:
    word = {"add": "plus", "sub": "minus"}[op]
    sym = {"add": "+", "sub": "-"}[op]
    if variant == 0 or variant == 3:
        return f"{x} {sym} {y} = {r}"
    if variant == 1:
        return f"{x} {word} {y} is {r}"
    return f"{_num(x, 2)} {word} {_num(y, 2)} equals {_num(r, 2)}"


def _assemble(header: str, steps: List[str], tail: str, variant: int) -> str:
    if variant == 0 or variant == 2:
        return f"{header}: " + ", ".join(steps) + f", {tail}"
    if variant == 1:
        return f"Work out {header}. " + " ".join(("First " if i == 0 else "Then ") + s + "." for i, s in enumerate(steps)) + f" {tail[0].upper() + tail[1:]}."
    return f"{header}. Counting: " + "; ".join(steps) + f". {tail[0].upper() + tail[1:]}."


def _header(skill: str, a: int, b: int, variant: int) -> str:
    n = lambda v: _num(v, variant)
    if skill == "mul":
        return {0: f"{n(a)} times {n(b)}", 1: f"{n(a)} times {n(b)}", 2: f"{n(a)} times {n(b)}", 3: f"I buy {n(a)} packs with {n(b)} sweets each"}[variant]
    if skill == "div":
        return {0: f"{n(a)} divided by {n(b)}", 1: f"{n(a)} divided by {n(b)}", 2: f"{n(a)} divided by {n(b)}", 3: f"I share {n(a)} sweets, taking {n(b)} away each time"}[variant]
    if skill == "max":
        return {0: f"the larger of {n(a)} and {n(b)}", 1: f"the larger of {n(a)} and {n(b)}", 2: f"the larger of {n(a)} and {n(b)}", 3: f"Ann has {n(a)} sweets and Ben has {n(b)}"}[variant]
    if skill == "mod":
        return {0: f"{n(a)} mod {n(b)}", 1: f"{n(a)} mod {n(b)}", 2: f"{n(a)} mod {n(b)}", 3: f"I share {n(a)} sweets in groups of {n(b)} and keep the rest"}[variant]
    raise ValueError(f"no header for {skill!r}")


def make_trace(rng: np.random.Generator, skill: str, variant: int, decoy: Optional[str] = None) -> Trace:
    """A worked trace of ``skill`` ("mul": repeated addition, "div": repeated subtraction) in wording ``variant`` (0 to 3). With a decoy: "wrong_steps" for mul or div; "irrelevant_valid_steps"
    needs skill "max" and "unrelated_skill" needs skill "mod"."""
    if not 0 <= variant < N_VARIANTS:
        raise ValueError(f"variant must be between 0 and {N_VARIANTS - 1}, got {variant}")
    if decoy is not None and decoy not in DECOYS:
        raise ValueError(f"unknown decoy {decoy!r}; the decoys are {DECOYS}")
    allowed = {None: ("mul", "div"), "wrong_steps": ("mul", "div"), "irrelevant_valid_steps": ("max",), "unrelated_skill": ("mod",)}[decoy]
    if skill not in allowed:
        raise ValueError(f"skill {skill!r} is not allowed for decoy {decoy!r}; use one of {allowed}")
    n = lambda v: _num(v, variant)
    if skill == "mul":
        a, b = int(rng.integers(3, 6)), int(rng.integers(2, 13))
        rows, x = [], b
        for _ in range(a - 1):
            rows.append((x, b, x + b))
            x += b
        answer, op, tail = a * b, "add", f"so {n(a * b)}"
    elif skill == "div":
        b, q, r = int(rng.integers(2, 10)), int(rng.integers(2, 6)), 0
        r = int(rng.integers(0, b))
        a = q * b + r
        rows, x = [], a
        for _ in range(q):
            rows.append((x, b, x - b))
            x -= b
        answer, op, tail = q, "sub", f"{n(q)} times, so {n(q)}"
    elif skill == "max":
        a, b = int(rng.integers(2, 30)), int(rng.integers(2, 30))
        x1, y1, x2, y2 = (int(v) for v in rng.integers(2, 20, size=4))
        if x2 == x1 + y1:
            x2 += 1
        rows, answer, op, tail = [(x1, y1, x1 + y1), (x2, y2, x2 + y2)], max(a, b), "add", f"so {n(max(a, b))}"
    else:                                                                    # mod
        b = int(rng.integers(2, 10))
        a = int(rng.integers(b + 1, 40))
        rows, answer, op, tail = [(a, b, a + b), (a + b, b, a + 2 * b)], a % b, "add", f"so {n(a % b)}"
    if decoy == "wrong_steps":
        x, y, r = rows[-1]
        rows[-1] = (x, y, r + 1)
    steps = tuple(Step(_step_text(x, y, r, op, variant), op if decoy != "unrelated_skill" or True else None, x, y, r) for x, y, r in rows)
    text = _assemble(_header(skill, a, b, variant), [s.text for s in steps], tail, variant)
    return Trace(text, skill, a, b, answer, steps, decoy)


# ---------------------------------------------------------------------------------------------------------------------------------- verifying traces
def _solver_gives(solver, x: int, y: int, r: int) -> bool:
    return int(np.asarray(solver.predict(np.array([x]), np.array([y])))[0]) == r


def _chain_valid(rows: List[Optional[Tuple[int, int, int]]], ha: int, hb: int, stated: int) -> bool:
    if not rows or any(r is None for r in rows):
        return False
    if any(rows[k][0] != rows[k - 1][2] for k in range(1, len(rows))):
        return False
    if all(r == x + y for x, y, r in rows):                                      # additive: b + b, then + b ..., a - 1 steps, ends at the stated answer
        return rows[0][0] == hb and all(y == hb for _, y, _ in rows) and len(rows) == ha - 1 and rows[-1][2] == stated
    if all(r == x - y for x, y, r in rows):                                      # subtractive: from a, take b away each time, as many steps as the stated answer, end below b
        return rows[0][0] == ha and all(y == hb for _, y, _ in rows) and len(rows) == stated and 0 <= rows[-1][2] < hb
    return False


def verify_trace(trace: Trace, assess_step: Callable[[str], ST.Assessment], solvers) -> TraceVerdict:
    """``assess_step(text)`` is the reader plus the status machine on one step; ``solvers[s].predict(a, b)`` the taught solver of skill s. See the module docstring for the rules."""
    nums = FC.parse_numbers(trace.text)
    if not trace.steps or len(nums) < 3:
        return TraceVerdict(frozenset(), False, 0)
    ha, hb, stated = nums[0], nums[1], nums[-1]
    rows: List[Optional[Tuple[int, int, int]]] = []
    verified: List[Optional[str]] = []
    for st in trace.steps:
        n = FC.parse_numbers(st.text)
        if len(n) < 3:
            rows.append(None)
            verified.append(None)
            continue
        x, y, r = n[0], n[1], n[2]
        rows.append((x, y, r))
        a = assess_step(st.text)
        s = a.skill if a.status == "solid" else None
        verified.append(s if s is not None and s in solvers and _solver_gives(solvers[s], x, y, r) else None)
    valid = _chain_valid(rows, ha, hb, stated)
    parents = frozenset(s for s in verified if s is not None) if valid else frozenset()
    return TraceVerdict(parents, valid, sum(s is not None for s in verified))
