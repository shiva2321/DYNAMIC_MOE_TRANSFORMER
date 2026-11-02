"""Bed builders and stream adapters for continual learning evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
        name: Name of the bed ('B1', 'B2', 'B3seq', 'B3mix', 'B3drift').
        n_regimes: Number of regimes in the stream.
        seed: Stream RNG seed.
        pool_seed: RNG seed for the pool (seed + 7).
        d: Feature dimension of input vectors.
        n_classes: Total number of target labels.
        stream: Underlying materialised Stream or UStream instance.
        label_blocks: Optional domain label blocks for B3 beds.
    """

    name: str
    n_regimes: int
    seed: int
    pool_seed: int
    d: int
    n_classes: int
    stream: Any
    label_blocks: Optional[Dict[str, Tuple[int, int]]] = None

    @property
    def test_sets(self) -> Dict[int, Tuple[Tensor, Tensor]]:
        """Per-regime test sets mapping regime_id to (x_test, y_test)."""
        return self.stream.test_sets()

    @property
    def total_samples(self) -> int:
        """Total number of stream training samples across all segments or phases."""
        if hasattr(self.stream, "segments"):
            return int(sum(len(seg.x) for seg in self.stream.segments))
        if hasattr(self.stream, "phases"):
            return int(sum(len(p.y) for p in self.stream.phases))
        return 0

    @property
    def n_phases(self) -> int:
        """Number of evaluation phases in the stream."""
        if hasattr(self.stream, "segments"):
            return len(self.stream.segments)
        if hasattr(self.stream, "phases"):
            return len(self.stream.phases)
        return 0

    @property
    def regime_info(self) -> Dict[int, Dict[str, Any]]:
        """Metadata for each regime in the stream."""
        return {
            r: {
                "id": reg.id,
                "name": reg.name,
                "domain": getattr(reg, "domain", None),
                "parent": getattr(reg, "parent", None),
            }
            for r, reg in self.stream.regimes.items()
        }

    def walk(self, batch_size: int = 64) -> Iterator[Tuple[Tensor, Tensor, int, bool, int]]:
        """Ordered iterator yielding (x, y, regime_id, first_visit, phase_index).

        Batches do not cross phase (segment) boundaries; sample ordering within
        each segment/phase is deterministic in the stream seed.
        """
        if hasattr(self.stream, "_walk"):
            for phase_idx, seg, idx in self.stream._walk(batch_size, self.seed):
                yield seg.x[idx], seg.y[idx], int(seg.regime), bool(seg.first_visit), int(phase_idx)
        elif hasattr(self.stream, "batches"):
            for k in range(len(self.stream.phases)):
                for x, y, gt in self.stream.batches(k, batch_size, self.seed):
                    reg_tensor = gt["regime"]
                    first_tensor = gt["first"]
                    reg_id = int(reg_tensor[0].item()) if reg_tensor.numel() > 0 else 0
                    first_visit = bool(first_tensor.any().item()) if first_tensor.numel() > 0 else False
                    yield x, y, reg_id, first_visit, int(k)
        else:
            raise NotImplementedError(f"Stream type {type(self.stream)} does not support walk()")

    def walk_full(self, batch_size: int = 64) -> Iterator[Tuple[Tensor, Tensor, List[int], List[int], int]]:
        """Like walk(), but with ALL regimes present in the batch and the regimes whose samples in this batch are first visits: (x, y, regimes, first_regimes, phase_index).

        A phase of the mixed and drifting schedules of B3 holds samples of many regimes; walk() names only the first one, which is not enough to know when each regime was learned.
        """
        if hasattr(self.stream, "_walk"):
            for x, y, regime_id, first_visit, phase_idx in self.walk(batch_size):
                yield x, y, [regime_id], ([regime_id] if first_visit else []), phase_idx
        elif hasattr(self.stream, "batches"):
            for k in range(len(self.stream.phases)):
                for x, y, gt in self.stream.batches(k, batch_size, self.seed):
                    regimes = sorted({int(r) for r in gt["regime"].tolist()})
                    first = sorted({int(r) for r, f in zip(gt["regime"].tolist(), gt["first"].tolist()) if f})
                    yield x, y, regimes, first, int(k)
        else:
            raise NotImplementedError(f"Stream type {type(self.stream)} does not support walk_full()")

    def batches(self, batch_size: int = 64) -> Iterator[Tuple[Tensor, Tensor, int]]:
        """Ordered iterator of (x, y, regime_id) batches for Stage A compatibility."""
        for x, y, reg, _, _ in self.walk(batch_size):
            yield x, y, reg

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor, int]]:
        """Default iterator yielding batches with batch_size=64."""
        return self.batches(batch_size=64)


def build_bed(name: str, n_regimes: int, seed: int, **stream_kwargs: Any) -> Bed:
    """Build a continual learning bed by name.

    Args:
        name: Bed identifier ('B1', 'B2', 'B3seq', 'B3mix', 'B3drift').
        n_regimes: Number of regimes (must be 6 for 'B2', 22 for 'B3*').
        seed: Stream RNG seed.
        **stream_kwargs: Optional overrides for stream generators.

    Returns:
        Bed instance exposing batches, walk(), test sets, and dimensions.
    """
    label_blocks = None
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
    elif name in ("B3seq", "B3mix", "B3drift"):
        schedule = name[2:]  # "seq", "mix", or "drift"
        custom_stream = stream_kwargs.pop("stream", None)
        if custom_stream is not None:
            stream = custom_stream
            if hasattr(stream, "regimes") and n_regimes != len(stream.regimes):
                raise ValueError(f"Bed {name} requires n_regimes={len(stream.regimes)}, got {n_regimes}")
        else:
            if n_regimes != 22:
                raise ValueError(f"Bed {name} requires n_regimes=22, got {n_regimes}")
            from unimoe.experiment import streams

            stream = streams(seed)[schedule]
        label_blocks = dict(stream.label_blocks) if hasattr(stream, "label_blocks") else None
    else:
        raise ValueError(f"Bed {name!r} not supported (supported: 'B1', 'B2', 'B3seq', 'B3mix', 'B3drift').")

    return Bed(
        name=name,
        n_regimes=n_regimes,
        seed=seed,
        pool_seed=seed + POOL_SEED_OFFSET,
        d=stream.d,
        n_classes=stream.n_classes,
        stream=stream,
        label_blocks=label_blocks,
    )
