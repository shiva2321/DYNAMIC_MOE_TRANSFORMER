"""Metrics for the benchmark. Convention: novelty scores are HIGH for novel samples and ``labels == 1`` marks the
truly novel ones. Spawn events are ``Event(kind="spawn", step, fields={"expert": id, ...})``; the benchmark supplies
``owner_of`` (expert -> the regime it came to serve, from ``regime_owner``) and ``onsets`` (regime -> first step)."""

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from scipy import stats

NAN = float("nan")


def _np(x) -> np.ndarray:
    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)


def auroc(scores, labels) -> float:
    """Probability that a random positive outscores a random negative (ties count one half)."""
    s, y = _np(scores).astype(np.float64).ravel(), _np(labels).astype(bool).ravel()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs samples of both classes")
    ranks = stats.rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def false_novelty_rate(novel, known_mask) -> float:
    """Share of truly known samples that were flagged novel."""
    n, k = _np(novel).astype(bool).ravel(), _np(known_mask).astype(bool).ravel()
    if not k.any():
        raise ValueError("no known samples")
    return float(n[k].mean())


def novel_detection_rate(novel, known_mask) -> float:
    """Share of truly novel samples that were flagged novel."""
    n, k = _np(novel).astype(bool).ravel(), _np(known_mask).astype(bool).ravel()
    if k.all():
        raise ValueError("no novel samples")
    return float(n[~k].mean())


def _confusion(assigned, regimes) -> Dict[int, Counter]:
    a, r = _np(assigned).astype(np.int64).ravel(), _np(regimes).astype(np.int64).ravel()
    counts = defaultdict(Counter)
    for e, g in zip(a.tolist(), r.tolist()):
        if e >= 0:
            counts[e][g] += 1
    return counts


def regime_owner(assigned, regimes) -> Dict[int, int]:
    """Expert -> the regime contributing most of its samples (ties go to the smaller regime id). Unassigned (-1) ignored."""
    return {e: min(c, key=lambda g: (-c[g], g)) for e, c in _confusion(assigned, regimes).items()}


def routing_purity(assigned, regimes) -> float:
    """Cluster purity of the routing: sum over experts of the largest single-regime share, over assigned samples."""
    counts = _confusion(assigned, regimes)
    total = sum(sum(c.values()) for c in counts.values())
    return float(sum(max(c.values()) for c in counts.values()) / total) if total else NAN


def spawn_stats(events: Iterable, owner_of: Dict[int, int], onsets: Dict[int, int]) -> dict:
    """Quality of the spawn decisions. The first spawn whose expert owns a regime, at or after that regime's onset,
    is a true positive; later spawns for an already-served regime, spawns that own no regime and spawns before the
    regime existed are false positives. ``ratio`` = spawned experts per true regime."""
    spawns = sorted(((e.step, int(e.fields["expert"])) for e in events if e.kind == "spawn"), key=lambda t: t[0])
    claimed, delays = set(), []
    for step, expert in spawns:
        regime = owner_of.get(expert)
        if regime is not None and regime in onsets and regime not in claimed and step >= onsets[regime]:
            claimed.add(regime)
            delays.append(step - onsets[regime])
    n_sp, n_tp, n_reg = len(spawns), len(delays), len(onsets)
    return {"n_spawns": n_sp, "n_true_positive": n_tp,
            "precision": n_tp / n_sp if n_sp else NAN,
            "recall": n_tp / n_reg if n_reg else NAN,
            "ratio": n_sp / n_reg if n_reg else NAN,
            "time_to_spawn": float(np.mean(delays)) if delays else NAN,
            "time_to_spawn_median": float(np.median(delays)) if delays else NAN}


def accuracy_stats(R, learned_at: Sequence[int] = None, chance: float = 0.0) -> dict:
    """Continual-learning summary of an accuracy matrix ``R[t, k]`` = accuracy on regime k after evaluation row t.

    ``learned_at[k]`` is the row right after regime k was first trained (default: row k, the square form).
    avg        mean final accuracy over all regimes
    forgetting mean over regimes learned before the last row of (best accuracy since learning, final excluded) - final
    backward   mean over the same regimes of final - accuracy right after learning (negative = forgetting)
    forward    mean over regimes not learned first of (accuracy on k one row before learning it) - chance
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("R must be a (rows, regimes) matrix")
    T, K = R.shape
    la = np.arange(K) if learned_at is None else np.asarray(learned_at, dtype=np.int64)
    if la.shape != (K,) or la.min() < 0 or la.max() >= T:
        raise ValueError("learned_at must give one valid row per regime")
    final = R[-1]
    old = [k for k in range(K) if la[k] < T - 1]
    forgetting = float(np.mean([R[la[k]:T - 1, k].max() - final[k] for k in old])) if old else NAN
    backward = float(np.mean([final[k] - R[la[k], k] for k in old])) if old else NAN
    later = [k for k in range(K) if la[k] >= 1]
    forward = float(np.mean([R[la[k] - 1, k] - chance for k in later])) if later else NAN
    return {"avg": float(final.mean()), "forgetting": forgetting, "backward": backward, "forward": forward}


def mean_ci(values, level: float = 0.95) -> Tuple[float, float, float]:
    """Mean and the Student-t confidence interval over independent runs (seeds)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        raise ValueError("no values")
    m = float(v.mean())
    if v.size == 1:
        return m, float("-inf"), float("inf")
    half = float(stats.t.ppf(0.5 + level / 2.0, v.size - 1) * v.std(ddof=1) / math.sqrt(v.size))
    return m, m - half, m + half


def paired_diff(a, b, level: float = 0.95) -> dict:
    """Paired comparison of two methods run on the same seeds: mean difference a - b, its t interval, and two-sided
    p-values from the paired t test and the Wilcoxon signed-rank test."""
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired samples must be non-empty and of equal length")
    d = a - b
    n = d.size
    m = float(d.mean())
    if n == 1:
        return {"n": 1, "mean_diff": m, "ci_lo": float("-inf"), "ci_hi": float("inf"), "p_t": NAN, "p_wilcoxon": NAN, "dz": NAN}
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    half = float(stats.t.ppf(0.5 + level / 2.0, n - 1) * se)
    if sd == 0.0:
        p_t = 1.0 if m == 0.0 else 0.0
    else:
        p_t = float(2.0 * stats.t.sf(abs(m / se), n - 1))
    p_w = 1.0 if not d.any() else float(stats.wilcoxon(d).pvalue)
    return {"n": n, "mean_diff": m, "ci_lo": m - half, "ci_hi": m + half, "p_t": p_t, "p_wilcoxon": p_w,
            "dz": m / sd if sd > 0 else NAN}
