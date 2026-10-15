"""Routing v2, stage 4c: grow an expert's width from the within-batch fit gap (spec docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md, section 6).

The shadow-trial policy of today never fires on a starved expert (E-030). Spike C (E-032) showed that a short paired trial cannot confirm the need for width, but that one cheap
number tracks it: the share of a fresh batch an expert can fit in its own few training steps, g = 1 - sum(L_post) / sum(L_pre) over the last ``window`` batches, with L_pre the honest
loss before the steps and L_post the loss on the same rows after them. g rises with width (16 / 32 / 64 / 128 / 512 units: 0.08 / 0.12 / 0.17 / 0.23 / 0.39 on the arithmetic stream),
so growth stops by itself near the knee of the accuracy curve. An expert is *mature* when it has trained long enough and its honest loss has stopped improving.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from torch import Tensor


class GapMonitor:
    """Per-expert bookkeeping of the fit gap, of progress, of age and of the cooldown. Pure: it holds numbers only."""

    def __init__(self, window: int) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        self.window = window
        self.pairs: Deque[tuple] = deque(maxlen=window)         # per batch: (sum of honest losses before the steps, sum of losses after them), the same rows
        self.pre: Deque[float] = deque(maxlen=2 * window)       # per batch: mean honest loss before the steps
        self.age = 0                                            # batches seen since birth or the last growth
        self.since_growth: Optional[int] = None                 # batches since the last growth (None = never grown)

    def add(self, pre: Tensor, post: Tensor) -> None:
        """One trained batch: per-row losses before and after the expert's own steps on the same rows."""
        pre, post = pre.double().flatten(), post.double().flatten()
        if pre.numel() == 0:
            return
        self.pairs.append((float(pre.sum()), float(post.sum())))
        self.pre.append(float(pre.mean()))
        self.age += 1
        if self.since_growth is not None:
            self.since_growth += 1

    def gap(self) -> Optional[float]:
        """1 - sum(post) / sum(pre) over the last ``window`` batches; None until the window is full or when there is no loss to share."""
        if len(self.pairs) < self.window:
            return None
        s_pre, s_post = sum(p for p, _ in self.pairs), sum(q for _, q in self.pairs)
        return None if s_pre <= 0.0 else 1.0 - s_post / s_pre

    def improvement(self) -> Optional[float]:
        """Mean honest loss of the last window over that of the window before (below 1 = still improving); None until two windows exist."""
        if len(self.pre) < 2 * self.window:
            return None
        h = list(self.pre)
        prev, last = sum(h[:self.window]) / self.window, sum(h[self.window:]) / self.window
        return None if prev <= 0.0 else last / prev

    def mature(self, min_age: int, plateau: float) -> bool:
        imp = self.improvement()
        return self.age >= min_age and imp is not None and imp >= plateau

    def cooled(self, cooldown: int) -> bool:
        return self.since_growth is None or self.since_growth >= cooldown

    def grew(self) -> None:
        """A growth (or a reverted growth) happened: new windows, age and cooldown start again."""
        self.pairs.clear()
        self.pre.clear()
        self.age = 0
        self.since_growth = 0

    def recent_pre_mean(self, n: int) -> float:
        """Mean honest loss per batch over the last ``n`` batches (all of them when fewer exist; nan when none)."""
        h = list(self.pre)[-n:]
        return sum(h) / len(h) if h else float("nan")


def widen_increment(hidden: int, factor: float, max_hidden: int, min_widen: int) -> int:
    """Hidden units to add to multiply the width by ``factor``, capped at ``max_hidden``; 0 when the cap leaves room for fewer than ``min_widen``."""
    k = min(int(round(hidden * factor)), max_hidden) - hidden
    return k if k >= min_widen else 0


def params_of_widen(d_in: int, depth: int, k: int) -> int:
    """Parameters added by widening every block of a residual MLP expert by k units: w1 (k x d), b1 (k), w2 (d x k)."""
    return depth * (2 * d_in * k + k)


def guard_reverts(before: float, after: float, eps: float) -> bool:
    """True when the honest loss after a growth is worse than before it by MORE than the share ``eps``."""
    return after > (1.0 + eps) * before


@dataclass
class GrowthGuard:
    """What is kept for ``grow_guard_window`` batches after a growth so that it can be undone: the expert and its optimiser as they were."""
    module: object
    opt: object
    before: float                                               # mean honest loss per batch over the window before the growth
    hidden_before: int
    losses: List[float] = field(default_factory=list)           # mean honest loss of each batch since the growth
