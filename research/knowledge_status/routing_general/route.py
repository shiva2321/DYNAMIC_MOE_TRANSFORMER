"""The domain-free decision on the probabilities a reader gives over the cards and "none of these".

claimed: the most probable option is a card (ties go to the card) and its probability is at least tau. A claimed input takes its card's state (solid or thin, the expert's measured
competence) and gets a TEAM: the cards in order of probability, added until they hold `mass` of the probability the reader puts on the cards, at most `cap`. Unclaimed means "not mine" and
nothing more: the router does not say whether it is a new skill, noise or something out of scope (that takes recurrence and feedback, and belongs to the pool).
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

STATES = ("solid", "thin")


@dataclass(frozen=True)
class Decision:
    status: str                      # "solid" | "thin" | "unclaimed"
    card: Optional[int]              # index of the claiming card, None when unclaimed
    p: float                         # probability of the best card (claimed or not)
    team: Tuple[int, ...]            # card indices, most probable first; empty when unclaimed


def _check(P: np.ndarray, states: Sequence[str], tau: float, mass: float, cap: int) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    k = len(states)
    if P.ndim != 2 or P.shape[1] != k + 1:
        raise ValueError(f"expected one column per card plus one for 'none of these' ({k + 1}), got shape {P.shape}")
    if any(s not in STATES for s in states):
        raise ValueError(f"card states must be in {STATES}, got {list(states)}")
    if not 0.0 < tau <= 1.0 or not 0.0 < mass <= 1.0 or cap < 1:
        raise ValueError(f"need 0 < tau <= 1, 0 < mass <= 1 and cap >= 1, got {tau}, {mass}, {cap}")
    if (P < -1e-9).any() or not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("every row must be a probability distribution")
    return P


def decide(P: np.ndarray, states: Sequence[str], tau: float = 0.5, mass: float = 0.99, cap: int = 3) -> List[Decision]:
    P = _check(P, states, tau, mass, cap)
    k = len(states)
    out: List[Decision] = []
    for row in P:
        best = int(row.argmax())
        top_card = int(row[:k].argmax())
        p = float(row[top_card])
        if best >= k or p < tau:
            out.append(Decision("unclaimed", None, p, ()))
            continue
        order = np.argsort(-row[:k], kind="stable")
        total, held, team = float(row[:k].sum()), 0.0, []
        for c in order[:cap]:
            team.append(int(c))
            held += float(row[c])
            if total <= 0.0 or held / total >= mass:
                break
        out.append(Decision(states[best], best, p, tuple(team)))
    return out


def unclaimed_score(P: np.ndarray) -> np.ndarray:
    """1 minus the probability of the best card: larger = less likely to be anyone's. The continuous score behind the claimed / unclaimed cut (for AUROC)."""
    P = np.asarray(P, dtype=np.float64)
    return 1.0 - P[:, :-1].max(axis=1)
