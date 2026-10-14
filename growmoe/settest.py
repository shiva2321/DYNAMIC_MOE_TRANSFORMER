"""Exact permutation tests of a batch against a reference sample (routing v2, stage 3).

The question a batch test answers: "does this group of inputs look like a random draw from the sample the expert was trained on?"
The null hypothesis is exchangeability of the pooled points; a permutation draws a random b-subset of the P + b pooled points and
recomputes the statistic of that subset against the rest. With ``c`` permutations at least as extreme as the observed statistic out of
``n``, the p-value is (1 + c) / (1 + n) and is valid at every level (Phipson and Smyth 2010). Everything is counted in integers.

``perm_reject`` stops as soon as the answer "not rejected" is certain (the early stop never changes the decision, only the cost).
Rows may have any length; nothing assumes unit norm (the pool's vectors happen to be unit length).
"""

import math
from typing import Iterator, Tuple

import numpy as np
import torch
from torch import Tensor

STATS = ("meandist", "mmd")


def max_exceedances(alpha: float, n_perm: int) -> int:
    """The largest count ``c`` of permutations at least as extreme as the observed statistic for which p = (1 + c) / (1 + n_perm) <= alpha
    (-1 when even c = 0 fails: the test could never reject). Integer arithmetic only; the 1e-9 absorbs float error in alpha * (1 + n_perm)."""
    return math.floor(alpha * (1 + n_perm) + 1e-9) - 1


def generator(*key: int) -> torch.Generator:
    """A generator whose state is a pure function of ``key`` (non-negative integers): equal keys give equal streams, so a decision never
    depends on how many other random draws happened before it."""
    if any(int(k) < 0 for k in key):
        raise ValueError(f"generator key parts must be non-negative, got {key}")
    seed = int(np.random.SeedSequence([int(k) for k in key]).generate_state(1, dtype=np.uint64)[0]) & 0x7FFFFFFFFFFFFFFF
    return torch.Generator().manual_seed(seed)


def cap_rows(Z: Tensor, k: int, gen: torch.Generator) -> Tensor:
    """At most ``k`` rows of ``Z``: all of them when there are no more than ``k``, otherwise a uniformly random subset in the original order."""
    if k < 1:
        raise ValueError("k must be at least 1")
    n = Z.shape[0]
    if n <= k:
        return Z
    return Z[torch.randperm(n, generator=gen)[:k].sort().values]


def _bandwidth(Z: Tensor) -> float:
    """Median of the positive squared distances of the pooled sample (1.0 if every point is identical)."""
    sq = (Z * Z).sum(1)
    d2 = (sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)).clamp_min(0.0)
    pos = d2[d2 > 1e-12]
    return float(pos.median()) if pos.numel() else 1.0


class _Stat:
    """The statistic of "rows ``idx`` against the rest" for many index sets at once."""

    def __init__(self, Z: Tensor, stat: str) -> None:
        if stat not in STATS:
            raise ValueError(f"stat must be one of {STATS}, got {stat!r}")
        self.Z, self.stat, self.P = Z, stat, Z.shape[0]
        if stat == "meandist":
            self.total = Z.sum(0)
        else:
            sq = (Z * Z).sum(1)
            d2 = (sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)).clamp_min(0.0)
            self.K = torch.exp(-d2 / _bandwidth(Z))
            self.s = self.K.sum(1)
            self.sum_all = float(self.s.sum())

    def __call__(self, idx: Tensor) -> Tensor:
        """idx: (m, b) long, rows of the pooled sample forming the batch; returns the (m,) statistics."""
        m, b = idx.shape
        n = self.P - b
        if self.stat == "meandist":
            bm = self.Z[idx].mean(1)
            rm = (self.total - bm * b) / n
            return (bm - rm).norm(dim=1)
        W = torch.zeros(self.P, m)
        W.scatter_(0, idx.T, 1.0)
        bb = (W * (self.K @ W)).sum(0)
        ws = (W * self.s[:, None]).sum(0)
        br = ws - bb
        rr = self.sum_all - 2.0 * ws + bb
        return bb / b ** 2 - 2.0 * br / (b * n) + rr / n ** 2


def _chunks(Z: Tensor, b: int, n_perm: int, gen: torch.Generator, stat: str, chunk: int) -> Tuple[float, Iterator[Tensor]]:
    """The observed statistic and an iterator over the permuted statistics, ``chunk`` permutations at a time, in a fixed order."""
    P = Z.shape[0]
    f = _Stat(Z, stat)
    obs = f(torch.arange(P - b, P)[None, :])[0]

    def draw() -> Iterator[Tensor]:
        used = 0
        while used < n_perm:
            m = min(chunk, n_perm - used)
            idx = torch.rand(m, P, generator=gen).argsort(1)[:, :b]
            yield f(idx)
            used += m

    return float(obs), draw()


def _check(ref: Tensor, batch: Tensor) -> Tensor:
    if ref.dim() != 2 or batch.dim() != 2 or ref.shape[1] != batch.shape[1]:
        raise ValueError("ref and batch must be 2-D tensors with the same number of columns")
    if ref.shape[0] < 1 or batch.shape[0] < 1:
        raise ValueError("ref and batch must each hold at least one row")
    return torch.cat([ref, batch]).float()


def perm_reject(ref: Tensor, batch: Tensor, alpha: float = 0.001, n_perm: int = 999, gen: torch.Generator = None,
                stat: str = "meandist", chunk: int = 16) -> Tuple[bool, int, int]:
    """Return ``(rejected, exceedances, permutations_used)``.

    ``rejected`` is True iff p = (1 + c) / (1 + n_perm) <= alpha, i.e. c <= alpha (1 + n_perm) - 1. As soon as c exceeds that bound the answer
    "not rejected" is final and the loop stops; the decision equals the one computed from all ``n_perm`` permutations drawn from the same
    generator state."""
    Z = _check(ref, batch)
    gen = gen if gen is not None else torch.Generator().manual_seed(0)
    allowed = max_exceedances(alpha, n_perm)                                           # reject iff c <= allowed
    if allowed < 0:
        raise ValueError(f"n_perm={n_perm} is too small for alpha={alpha}: the smallest attainable p-value 1/(1+n_perm) exceeds alpha, so the test could never reject")
    obs, draws = _chunks(Z, batch.shape[0], n_perm, gen, stat, chunk)
    c = used = 0
    for t in draws:
        c += int((t >= obs).sum())
        used += int(t.numel())
        if c > allowed:
            return False, c, used
    return c <= allowed, c, used


def perm_pvalue(ref: Tensor, batch: Tensor, n_perm: int = 999, gen: torch.Generator = None, stat: str = "meandist", chunk: int = 16) -> float:
    """The full permutation p-value (1 + c) / (1 + n_perm); for tests and diagnostics."""
    Z = _check(ref, batch)
    gen = gen if gen is not None else torch.Generator().manual_seed(0)
    obs, draws = _chunks(Z, batch.shape[0], n_perm, gen, stat, chunk)
    c = sum(int((t >= obs).sum()) for t in draws)
    return (1 + c) / (1 + n_perm)
