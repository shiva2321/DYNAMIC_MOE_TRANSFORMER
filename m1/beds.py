"""Bed builders and stream adapters for continual learning evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from torch import Tensor

from growmoe.streams import Stream
from growmoe.streams_m1 import function_regimes
from m1.config_m1 import POOL_SEED_OFFSET

_FRONTEND_B2: Optional[Any] = None


def get_b2_frontend() -> Any:
    """Return process-level singleton instance of HybridFrontend for Bed B2."""
    global _FRONTEND_B2
    if _FRONTEND_B2 is None:
        from research.growmoe_results.m0.run_m0 import HybridFrontend

        _FRONTEND_B2 = HybridFrontend(d=128, seed=0)
    return _FRONTEND_B2


@dataclass
class Bed:
    """Continual learning test bed exposing stream batches, phases, and test sets.

    Attributes:
        name: Name of the bed ('B1', 'B2').
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

    @property
    def n_phases(self) -> int:
        """Number of evaluation phases in the stream (for B1 and B2, each segment is a phase)."""
        return len(self.stream.segments)

    @property
    def regime_info(self) -> Dict[int, Dict[str, Any]]:
        """Metadata for each regime in the stream."""
        return {
            r: {
                "id": reg.id,
                "name": reg.name,
                "parent": getattr(reg, "parent", None),
            }
            for r, reg in self.stream.regimes.items()
        }

    def walk(self, batch_size: int = 64) -> Iterator[Tuple[Tensor, Tensor, int, bool, int]]:
        """Ordered iterator yielding (x, y, regime_id, first_visit, phase_index).

        Batches do not cross phase (segment) boundaries; sample ordering within
        each segment is deterministic in the stream seed.
        """
        for phase_idx, seg, idx in self.stream._walk(batch_size, self.seed):
            yield seg.x[idx], seg.y[idx], int(seg.regime), bool(seg.first_visit), int(phase_idx)

    def batches(self, batch_size: int = 64) -> Iterator[Tuple[Tensor, Tensor, int]]:
        """Ordered iterator of (x, y, regime_id) batches for Stage A compatibility."""
        for _, seg, idx in self.stream._walk(batch_size, self.seed):
            yield seg.x[idx], seg.y[idx], int(seg.regime)

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor, int]]:
        """Default iterator yielding batches with batch_size=64."""
        return self.batches(batch_size=64)


def build_bed(name: str, n_regimes: int, seed: int, **stream_kwargs: Any) -> Bed:
    """Build a continual learning bed by name.

    Args:
        name: Bed identifier ('B1', 'B2').
        n_regimes: Number of regimes (must be 6 for 'B2').
        seed: Stream RNG seed.
        **stream_kwargs: Optional overrides for stream generators.

    Returns:
        Bed instance exposing batches, walk(), test sets, and dimensions.
    """
    if name == "B1":
        stream = function_regimes(seed=seed, n_regimes=n_regimes, **stream_kwargs)
    elif name == "B2":
        if n_regimes != 6:
            raise ValueError(f"Bed B2 requires n_regimes=6, got {n_regimes}")
        fe = stream_kwargs.pop("frontend", None)
        if fe is None:
            fe = get_b2_frontend()
        from growmoe.streams_real import banking_incremental

        stream = banking_incremental(seed=seed, frontend=fe, n_regimes=6, **stream_kwargs)
    else:
        raise ValueError(f"Bed {name!r} not supported in Job B-1 (supported: 'B1', 'B2').")

    return Bed(
        name=name,
        n_regimes=n_regimes,
        seed=seed,
        pool_seed=seed + POOL_SEED_OFFSET,
        d=stream.d,
        n_classes=stream.n_classes,
        stream=stream,
    )
