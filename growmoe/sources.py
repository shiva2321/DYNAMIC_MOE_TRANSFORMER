"""Per-expert memory of the expert's recent SOURCES (routing v2, stage 3a).

A merged or long-lived expert has been trained on several sources (regimes) in turn. Its stored prototypes are a uniform sample of everything it ever
saw, so a batch from any single one of those sources looks "foreign" to the mixture and a set-level test over-fires. A ``SourceWindows`` keeps one
window per recent source instead: a window holds the last ``max_batches`` batches one source sent to the expert, and a trained batch joins the newest
window only if it is compatible with it (a permutation test at ``split_alpha``), otherwise it opens a new window. An expert that learned A, then B, then
A again keeps a reference for each.

Memory is bounded: at most ``max_windows`` windows of ``max_batches`` batches of at most ``max_rows`` rows (a larger batch is stored as a random
subset). Windows are only a reference for tests; nothing here trains or routes.
"""

from collections import deque
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .settest import STATS, cap_rows, perm_reject


class _Window:
    __slots__ = ("batches", "n", "last", "_cat", "_mean")

    def __init__(self, max_batches: int) -> None:
        self.batches: deque = deque(maxlen=max_batches)
        self.n = 0                                                                      # rows currently held
        self.last = -1                                                                  # step of the most recent push (recency across experts)
        self._cat: Optional[Tensor] = None                                              # cached concatenation and mean, dropped on every push
        self._mean: Optional[Tensor] = None

    def push(self, xb: Tensor, step: int) -> None:
        if len(self.batches) == self.batches.maxlen:
            self.n -= self.batches[0].shape[0]
        self.batches.append(xb)
        self.n += xb.shape[0]
        self.last = step
        self._cat = self._mean = None

    def cat(self) -> Tensor:
        if self._cat is None:
            self._cat = torch.cat(list(self.batches))
        return self._cat

    def mean(self) -> Tensor:
        if self._mean is None:
            self._mean = self.cat().mean(0)
        return self._mean

    def copy(self) -> "_Window":
        w = _Window(self.batches.maxlen)
        w.batches.extend(self.batches)
        w.n, w.last = self.n, self.last
        return w


class SourceWindows:
    def __init__(self, max_windows: int = 4, max_batches: int = 5, min_ref: int = 48, split_alpha: float = 0.01, min_split: int = 8,
                 n_perm: int = 999, chunk: int = 16, stat: str = "meandist", max_rows: int = 512) -> None:
        if max_windows < 1 or max_batches < 1 or min_ref < 1 or min_split < 1 or max_rows < 1:
            raise ValueError("max_windows, max_batches, min_ref, min_split and max_rows must each be at least 1")
        if not 0.0 < split_alpha < 1.0:
            raise ValueError("split_alpha must lie in (0, 1)")
        if stat not in STATS:
            raise ValueError(f"stat must be one of {STATS}, got {stat!r}")
        self.max_windows, self.max_batches, self.min_ref, self.split_alpha = max_windows, max_batches, min_ref, split_alpha
        self.min_split, self.n_perm, self.chunk, self.stat, self.max_rows = min_split, n_perm, chunk, stat, max_rows
        self._wins: List[_Window] = []

    def __len__(self) -> int:
        return len(self._wins)

    def n_samples(self) -> int:
        """Rows held in all windows (what the memory accounting counts)."""
        return sum(w.n for w in self._wins)

    def update(self, xb: Tensor, gen: torch.Generator, step: int = 0) -> bool:
        """Add the samples one trained batch sent to the expert. Returns True iff the batch opened a new window."""
        if xb.shape[0] == 0:
            return False
        xb = cap_rows(xb.detach().to(torch.float32), self.max_rows, gen).clone()           # an own copy: callers may reuse their tensor
        if not self._wins:
            self._wins.append(self._fresh(xb, step))
            return False
        cur = self._wins[-1]
        if cur.n >= self.min_ref and xb.shape[0] >= self.min_split:
            ref = cap_rows(cur.cat(), self.max_rows, gen)
            if perm_reject(ref, xb, self.split_alpha, self.n_perm, gen, self.stat, self.chunk)[0]:
                self._wins.append(self._fresh(xb, step))
                del self._wins[:-self.max_windows]
                return True
        cur.push(xb, step)
        return False

    def refs(self) -> List[Tensor]:
        """One tensor per window that holds at least ``min_ref`` rows, oldest first."""
        return [w.cat() for w in self._wins if w.n >= self.min_ref]

    def ref_pairs(self) -> List[Tuple[Tensor, Tensor]]:
        """Like ``refs`` but each reference comes with its mean row; both are cached until the window changes, so a pool with hundreds of experts can
        rank its references by distance to a batch without touching the rows."""
        return [(w.cat(), w.mean()) for w in self._wins if w.n >= self.min_ref]

    def merge_from(self, other: Optional["SourceWindows"]) -> None:
        """Take the union of the two stores' windows, newest ``max_windows`` kept (recency = the step of a window's last push; ties keep this store's first)."""
        if other is None or other is self or not other._wins:
            return
        both = sorted(self._wins + [w.copy() for w in other._wins], key=lambda w: w.last)
        self._wins = both[-self.max_windows:]

    def _fresh(self, xb: Tensor, step: int) -> _Window:
        w = _Window(self.max_batches)
        w.push(xb, step)
        return w
