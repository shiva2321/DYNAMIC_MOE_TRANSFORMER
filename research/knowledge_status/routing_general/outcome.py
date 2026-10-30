"""Readers calibrated on OUTCOMES from the stream's own wording.

A stated answer or a label says, for some messages, that a taught expert owns them. Those verified known messages are scored by every reader and used as the calibration of its cut (and of
its p-values when readers are fused), in place of held-out messages in the teaching wording. The functions here are pure: splitting the stream's known messages into a pool the verified
sample is drawn from and an evaluation half, drawing the sample, and scoring a system calibrated on it.
"""

import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

import measures as MS  # noqa: E402


def split_pool_eval(owner: Sequence[int], rng: np.random.Generator) -> Tuple[List[int], List[int]]:
    """Indices of the known messages split per class into a pool (half, rounded down) and an evaluation half, so a message is never used to calibrate and to judge."""
    pool: List[int] = []
    ev: List[int] = []
    owner = np.asarray(owner)
    for c in sorted(set(owner.tolist())):
        idx = np.flatnonzero(owner == c)
        idx = idx[rng.permutation(len(idx))]
        k = len(idx) // 2
        pool += [int(i) for i in idx[:k]]
        ev += [int(i) for i in idx[k:]]
    return sorted(pool), sorted(ev)


def verified_order(candidates: Sequence[int], verified: Sequence[bool], rng: np.random.Generator) -> List[int]:
    """Every candidate whose ownership was verified, in a random order. A sample of n is the first n; two samples cut from the same order share their members (a paired comparison)."""
    cand = [int(i) for i in candidates if verified[i]]
    return [cand[int(j)] for j in rng.permutation(len(cand))]


def verified_sample(candidates: Sequence[int], verified: Sequence[bool], n: int, rng: np.random.Generator) -> List[int]:
    """Up to n of the candidates whose ownership was verified, in a random order (fewer when fewer are verified)."""
    return verified_order(candidates, verified, rng)[:n]


def system_metrics(cal: Mapping[str, Sequence[float]], ev: Mapping[str, Mapping[str, Sequence[float]]], sources: Sequence[str], rule: str, q: float) -> Dict[str, float]:
    """A system of one or more readers calibrated on ``cal`` (reader -> novelty scores of verified known messages) and judged on ``ev`` (set -> reader -> scores, sets known / unknown /
    foreign). The cut is set so that a share q of the calibration messages would be flagged; a calibration too small to flag anything at q flags nothing."""
    if not sources or any(r not in cal or len(cal[r]) == 0 for r in sources):
        raise ValueError("every reader of the system needs calibration scores")
    cal_map = {r: np.asarray(cal[r], dtype=np.float64) for r in sources}
    f = {k: MS.fused_scores(cal_map, {r: ev[k][r] for r in sources}, list(sources), rule) for k in ("known", "unknown", "foreign")}
    fused_cal = MS.fused_scores(cal_map, cal_map, list(sources), rule, leave_one_out=True)
    return MS.operating_point(fused_cal, f["known"], f["unknown"], f["foreign"], q)


def mean_over_draws(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in rows[0]:
        vals = [r[k] for r in rows if not np.isnan(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["draws"] = len(rows)
    return out
