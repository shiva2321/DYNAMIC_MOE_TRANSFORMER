"""Rank AUROC calculation with exact tie-handling."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats


def rank_auroc(neg: Sequence[float] | np.ndarray, pos: Sequence[float] | np.ndarray) -> float:
    """Compute rank AUROC between negative (known) and positive (novel/OOD) scores.

    Ties are handled by assigning average ranks.

    Args:
        neg: Sequence of scores for negative class (known/in-distribution).
        pos: Sequence of scores for positive class (novel/out-of-distribution).

    Returns:
        1.0 for perfect separation (all positive scores > all negative scores).
        0.5 for identical score distributions.
        0.0 for inverted separation (all negative scores > all positive scores).

    Raises:
        ValueError: If either neg or pos is empty.
    """
    neg_arr = np.asarray(neg, dtype=np.float64).ravel()
    pos_arr = np.asarray(pos, dtype=np.float64).ravel()
    n_neg = len(neg_arr)
    n_pos = len(pos_arr)
    if n_neg == 0 or n_pos == 0:
        raise ValueError(f"AUROC requires non-empty neg and pos samples, got neg={n_neg}, pos={n_pos}")

    combined = np.concatenate([neg_arr, pos_arr])
    ranks = stats.rankdata(combined, method="average")
    pos_ranks = ranks[n_neg:]
    u_pos = float(pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0)
    return float(u_pos / (n_neg * n_pos))
