"""Calibration against wrongly verified messages (lab G4c): small pure helpers on top of ``outcome.py``. A verified sample is only as clean as the outcome check; a never-taught message that the check
vouches for scores like a novel one and, when the cut is the largest score, sets it. The helpers here trim the sample, size an injected contamination, and apply the second opinion."""

from typing import Dict, Mapping, Sequence

import numpy as np


def trim_top(cal: Mapping[str, Sequence[float]], k: int) -> Dict[str, np.ndarray]:
    """Every reader's calibration scores without its k largest (the ones a wrongly verified message would be). For the sample sizes of the lab the cut then becomes the (k+1)-th largest score,
    so one reader's false-flag rate is about (k+1)/(n+1)."""
    if k < 0:
        raise ValueError(f"cannot trim {k} scores")
    out: Dict[str, np.ndarray] = {}
    for r, v in cal.items():
        s = np.sort(np.asarray(v, dtype=np.float64))
        if k >= s.size:
            raise ValueError(f"reader {r!r}: trimming {k} of {s.size} scores leaves nothing to calibrate on")
        out[r] = s[: s.size - k]
    return out


def injected_count(n_verified: int, share: float) -> int:
    """How many wrongly verified messages to add to n verified ones so that they make the share `share` of the whole (the expected share of a random sample)."""
    if not 0.0 <= share < 1.0:
        raise ValueError(f"share must be in [0, 1), got {share}")
    return int(round(share / (1.0 - share) * n_verified)) if n_verified > 0 else 0


def owner_agrees(top: Sequence[int], owners: Sequence[frozenset]) -> np.ndarray:
    """True where the reader's top option (a card index, -1 for "none of these") is one of the experts the outcome names: a second opinion from a different mechanism, the reader reading the
    words and the outcome checking the answer."""
    if len(top) != len(owners):
        raise ValueError(f"{len(top)} top options for {len(owners)} owner sets")
    return np.array([int(t) in o and int(t) >= 0 for t, o in zip(top, owners)], dtype=bool)
