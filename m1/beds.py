"""Bed builders and stream adapters for continual learning evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple

import torch
from torch import Tensor

from growmoe.streams import Stream
from growmoe.streams_m1 import function_regimes

POOL_SEED_OFFSET = 7


@dataclass
class Bed:
    """Continual learning test bed exposing stream batches and test sets.

    Attributes:
        name: Name of the bed ('B1').
        n_regimes: Number of regimes in the stream.
        seed: Stream RNG seed.
        pool_seed: RNG seed for the pool (seed + 7).
        d: Feature dimension of input vectors.
        n_classes: Total number of target labels.
        stream: Underlying materialised Stream instance.
    """

    name: str
    n_regimes: int
    seed: int
    pool_seed: int
    d: int
    n_classes: int
    stream: Stream

    @property
    def test_sets(self) -> Dict[int, Tuple[Tensor, Tensor]]:
        """Per-regime test sets mapping regime_id to (x_test, y_test)."""
        return self.stream.test_sets()

    @property
    def total_samples(self) -> int:
        """Total number of stream training samples across all segments."""
        return int(sum(len(seg.x) for seg in self.stream.segments))

    def batches(self, batch_size: int = 64) -> Iterator[Tuple[Tensor, Tensor, int]]:
        """Ordered iterator of (x, y, regime_id) batches.

        Batches do not cross segment boundaries; sample ordering within
        each segment is deterministic in the stream seed.
        """
        for _, seg, idx in self.stream._walk(batch_size, self.seed):
            yield seg.x[idx], seg.y[idx], int(seg.regime)

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor, int]]:
        """Default iterator yielding batches with batch_size=64."""
        return self.batches(batch_size=64)


def build_bed(name: str, n_regimes: int, seed: int, **stream_kwargs: Any) -> Bed:
    """Build a continual learning bed by name.

    Calls existing stream generators directly without duplicating logic.

    Args:
        name: Bed identifier ('B1').
        n_regimes: Number of regimes.
        seed: Stream RNG seed.
        **stream_kwargs: Optional overrides for function_regimes generator.

    Returns:
        Bed instance exposing batches, test sets, and dimensions.
    """
    if name != "B1":
        raise ValueError(f"Bed {name!r} not supported in Stage A (only 'B1').")

    stream = function_regimes(seed=seed, n_regimes=n_regimes, **stream_kwargs)
    return Bed(
        name=name,
        n_regimes=n_regimes,
        seed=seed,
        pool_seed=seed + POOL_SEED_OFFSET,
        d=stream.d,
        n_classes=stream.n_classes,
        stream=stream,
    )
