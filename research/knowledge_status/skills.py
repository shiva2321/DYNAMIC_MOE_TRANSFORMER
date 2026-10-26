"""The six skills of the knowledge-status test bed, their family tree and the exact ground truth (spec sections 5.1 and 5.6).

A "skill" is one arithmetic operation. ``PARENTS`` says which skills a skill is built from (repeated addition gives multiplication, and so on); it is the
ground truth for the relation test and is fixed here by the definition of the operations.
"""

from typing import Dict, FrozenSet

SKILLS = ("add", "sub", "mul", "div", "mod", "max")

# The operator ids of growmoe/arith.py (op == 3 is mod, op == 4 is div there), so the two code bases agree on the arithmetic.
ARITH_OP_ID: Dict[str, int] = {"add": 0, "sub": 1, "mul": 2, "mod": 3, "div": 4, "max": 5}

PARENTS: Dict[str, FrozenSet[str]] = {
    "add": frozenset(),
    "sub": frozenset(),
    "mul": frozenset({"add"}),          # repeated addition
    "div": frozenset({"sub"}),          # repeated subtraction
    "mod": frozenset({"div", "sub"}),   # what is left after the division
    "max": frozenset({"sub"}),          # compare by subtracting
}

COMMUTATIVE = frozenset({"add", "mul", "max"})


def truth(skill: str, a: int, b: int) -> int:
    """The exact answer. div is the integer quotient and mod the remainder; neither is defined for b == 0 (the generator never draws it)."""
    if skill == "add":
        return a + b
    if skill == "sub":
        return a - b
    if skill == "mul":
        return a * b
    if skill == "max":
        return max(a, b)
    if skill in ("div", "mod"):
        if b == 0:
            raise ValueError(f"{skill} is not defined for a zero divisor")
        return a // b if skill == "div" else a % b
    raise KeyError(f"unknown skill {skill!r}")
