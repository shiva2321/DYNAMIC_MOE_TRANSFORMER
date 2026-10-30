"""Measures of the general routing lab: how a claim / unclaimed decision, or a novelty score, treats known, never-taught and foreign messages. No model, no data: plain numbers in, numbers out."""

import math
from typing import Dict, Sequence

import numpy as np

import route as RT


def auroc(neg: Sequence[float], pos: Sequence[float]) -> float:
    """The probability that a random positive scores higher than a random negative (ties count half)."""
    neg, pos = np.asarray(neg, dtype=np.float64), np.asarray(pos, dtype=np.float64)
    if neg.size == 0 or pos.size == 0:
        raise ValueError("auroc needs at least one negative and one positive score")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (neg.size * pos.size))


def flag_rate(scores: Sequence[float], threshold: float) -> float:
    s = np.asarray(scores, dtype=np.float64)
    return float((s > threshold).mean()) if s.size else float("nan")


def claim_rates(known: Sequence[RT.Decision], truth: Sequence[int], unknown: Sequence[RT.Decision], foreign: Sequence[RT.Decision]) -> Dict[str, float]:
    if len(known) != len(truth):
        raise ValueError(f"{len(known)} decisions for {len(truth)} true cards")
    claimed = [d.status != "unclaimed" for d in known]
    right = [d.card == t for d, t in zip(known, truth)]
    n_claimed = sum(claimed)
    return {
        "known_claimed": float(np.mean(claimed)),
        "known_right": float(np.mean(right)),
        "right_given_claimed": (sum(right) / n_claimed) if n_claimed else float("nan"),
        "unknown_unclaimed": float(np.mean([d.status == "unclaimed" for d in unknown])) if len(unknown) else float("nan"),
        "foreign_unclaimed": float(np.mean([d.status == "unclaimed" for d in foreign])) if len(foreign) else float("nan"),
    }


def conformal_cut(cal_scores: Sequence[float], q: float) -> float:
    """The novelty score above which a message is flagged: the ceil((n+1)(1-q))-th smallest held-out known score (+infinity when there are too few to ever flag at q)."""
    s = np.sort(np.asarray(cal_scores, dtype=np.float64))
    if not 0.0 < q < 1.0 or s.size == 0:
        raise ValueError("need 0 < q < 1 and at least one calibration score")
    k = math.ceil((s.size + 1) * (1.0 - q) - 1e-9)
    return math.inf if k > s.size else float(s[k - 1])


def operating_point(cal: Sequence[float], known: Sequence[float], unknown: Sequence[float], foreign: Sequence[float], q: float) -> Dict[str, float]:
    """Scores are novelty scores (larger = less like anything taught). At the cut that flags a share q of held-out known messages: how many known, never-taught and foreign messages are
    flagged, plus the threshold-free AUROC of known against never-taught and of known against foreign."""
    cut = conformal_cut(cal, q)
    return {"cut": cut, "known_flagged": flag_rate(known, cut), "unknown_found": flag_rate(unknown, cut), "foreign_found": flag_rate(foreign, cut),
            "auroc_unknown": auroc(known, unknown) if len(unknown) else float("nan"), "auroc_foreign": auroc(known, foreign) if len(foreign) else float("nan")}
