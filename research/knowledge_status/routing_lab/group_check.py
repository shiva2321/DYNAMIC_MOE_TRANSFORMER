"""Evidence over a GROUP of parts for the function check (Lab B).

One example is weak evidence: with small numbers a taught skill reproduces the stated answer of a part of a skill nobody taught about one time in six, by coincidence. A known skill, though, is
explained again and again by the SAME taught skill. So the parts that the look-based machine called unseen are taken in windows of g, and for every competent taught skill s the number of parts
it explains, T_s, is compared with what chance gives: the stated answers of the window are shuffled over its parts (the numbers and the answers stay, the pairing is broken) and T_s is counted
again. A skill whose count is far above its shuffled counts is PRESENT in the window (Holm correction over the skills tried); the parts it explains are overturned. Mixed windows are allowed:
nothing needs the parts of a window to share a skill or a wording.

The explaining skill never reads an operator word or symbol: it gets the two numbers of the message (either order) and must give the stated answer (function_check.explain_many).
"""

import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

_KS = Path(__file__).resolve().parents[1]
for _p in (str(_KS.parents[1]), str(_KS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import skills  # noqa: E402


class PerfectSolver:
    """The exact skill, vectorised (a zero divisor gives 0 for div and mod and is excluded by the caller). Used for the ceilings, where nothing is trained."""

    def __init__(self, skill: str) -> None:
        if skill not in skills.SKILLS:
            raise KeyError(f"unknown skill {skill!r}")
        self.skill = skill

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
        if self.skill == "add":
            return a + b
        if self.skill == "sub":
            return a - b
        if self.skill == "mul":
            return a * b
        if self.skill == "max":
            return np.maximum(a, b)
        safe = np.where(b == 0, 1, b)
        return np.where(b == 0, 0, a // safe if self.skill == "div" else a % safe)


def hit_matrix(solver, numbers: Sequence[Sequence[int]], answers: Sequence[int], skill: str) -> np.ndarray:
    """H[i, j] = 1 if the solver turns the two numbers of part i into the stated answer of part j, in either order. Parts without exactly two numbers have an all-zero row."""
    g = len(numbers)
    H = np.zeros((g, g), dtype=np.int8)
    rows = [i for i, n in enumerate(numbers) if len(n) == 2]
    if not rows:
        return H
    a = np.array([numbers[i][0] for i in rows], dtype=np.int64)
    b = np.array([numbers[i][1] for i in rows], dtype=np.int64)
    ans = np.asarray(answers, dtype=np.int64)
    for x, y in ((a, b), (b, a)):
        ok = (y != 0) if skill in ("div", "mod") else np.ones(len(rows), dtype=bool)
        pred = np.zeros(len(rows), dtype=np.int64)
        if ok.any():
            pred[ok] = np.asarray(solver.predict(x[ok], y[ok]), dtype=np.int64)
        match = (pred[:, None] == ans[None, :]) & ok[:, None]
        H[rows] |= match.astype(np.int8)
    return H


def permutation_p(H: np.ndarray, rng: np.random.Generator, n_perm: int = 2000) -> float:
    """The probability, under shuffled answers, of at least as many explained parts as observed: (1 + #{T_perm >= T}) / (n_perm + 1). A window of one part has no shuffle and returns 1.
    The smallest p a window of g parts can give is about 1/g! (4 parts: 0.042), so a small window can only pass a loose threshold: the caller chooses window and threshold together."""
    g = len(H)
    if g < 2:
        return 1.0
    t = int(np.trace(H))
    perms = np.argsort(rng.random((n_perm, g)), axis=1)
    t_perm = H[np.arange(g)[None, :], perms].sum(axis=1)
    return (1 + int((t_perm >= t).sum())) / (n_perm + 1)


def explained_window(solvers: Mapping[str, object], competent: Sequence[str], numbers: Sequence[Sequence[int]], answers: Sequence[int], rng: np.random.Generator,
                     alpha: float = 0.01, n_perm: int = 2000) -> Dict[str, dict]:
    """For every competent skill: how many parts of the window it explains, its permutation p-value, whether it is present (Holm over the skills), and which parts it explains."""
    mats = {s: hit_matrix(solvers[s], numbers, answers, s) for s in competent}
    p = {s: permutation_p(mats[s], rng, n_perm) for s in competent}
    order = sorted(competent, key=lambda s: p[s])
    present: Dict[str, bool] = {}
    alive = True
    for rank, s in enumerate(order):
        alive = alive and p[s] <= alpha / (len(order) - rank)                                    # Holm: stop at the first skill that fails its threshold
        present[s] = alive
    return {s: {"count": int(np.trace(mats[s])), "p": p[s], "present": present[s], "parts": [int(i) for i in np.flatnonzero(np.diag(mats[s]))]} for s in competent}


def overturn(unseen_rows: Sequence[int], numbers: Mapping[int, Sequence[int]], answers: Mapping[int, int], solvers: Mapping[str, object], competent: Sequence[str],
             window: int, rng: np.random.Generator, nearest: Optional[Mapping[int, str]] = None, alpha: float = 0.01, n_perm: int = 2000, grouping: Optional[Sequence[Sequence[int]]] = None
             ) -> Dict[int, str]:
    """The parts (by row id) that the group test takes back from "unseen", with the skill that explains each. ``unseen_rows`` are cut into consecutive windows of ``window`` unless
    ``grouping`` gives the windows. When several present skills explain a part, the nearest (by look) wins if it is among them, otherwise the one with the smallest p-value."""
    wins: List[List[int]] = [list(w) for w in grouping] if grouping is not None else [list(unseen_rows[i:i + window]) for i in range(0, len(unseen_rows), window)]
    out: Dict[int, str] = {}
    for w in wins:
        if len(w) < 2 or not competent:
            continue
        res = explained_window(solvers, competent, [numbers[r] for r in w], [answers[r] for r in w], rng, alpha, n_perm)
        for k, r in enumerate(w):
            cands = [s for s in competent if res[s]["present"] and k in res[s]["parts"]]
            if cands:
                near = (nearest or {}).get(r)
                out[r] = near if near in cands else min(cands, key=lambda s: res[s]["p"])
    return out
