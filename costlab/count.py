"""Operation and MACs counting for PyTorch and manual workloads."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import torch
from torch.utils.flop_counter import FlopCounterMode


@dataclasses.dataclass(frozen=True)
class MacCount:
    """Result of counting MACs and FLOPs.

    Attributes:
        macs: Total multiply-accumulate operations (flops / 2).
        flops: Total floating-point operations recorded by FlopCounterMode.
        by_op: Mapping from operation name to its MAC count.
    """

    macs: int
    flops: int
    by_op: dict[str, int]


def count_macs(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> MacCount:
    """Count multiply-accumulate operations (MACs) using PyTorch FlopCounterMode.

    macs = flops / 2.

    Operations counted:
        - Matrix products (e.g. torch.mm, torch.bmm, torch.addmm, nn.Linear).
        - Convolutions (e.g. conv1d, conv2d, conv3d).
        - Scaled dot-product attention (when executed via tracked GEMMs).

    Operations NOT counted:
        - Element-wise operations (addition, multiplication, activations such as
          ReLU, GELU, Sigmoid, LayerNorm/BatchNorm arithmetic).
        - NumPy operations and pure Python computations.
        - Data movement, indexing, slicing, gather/scatter.

    Args:
        fn: Callable to execute and profile.
        *args: Positional arguments forwarded to fn.
        **kwargs: Keyword arguments forwarded to fn.

    Returns:
        MacCount: dataclass holding total macs, total flops, and per-op breakdown.
    """
    with FlopCounterMode(display=False) as fc:
        fn(*args, **kwargs)

    total_flops = int(fc.get_total_flops())
    total_macs = total_flops // 2

    by_op: dict[str, int] = {}
    counts = fc.get_flop_counts()
    if "Global" in counts:
        for op, cnt in counts["Global"].items():
            op_name = getattr(op, "name", str(op))
            by_op[str(op_name)] = int(cnt) // 2

    return MacCount(macs=total_macs, flops=total_flops, by_op=by_op)


class Counter:
    """Manual MACs counter for non-torch work or custom operation accounting."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def add(self, name: str, macs: int | float) -> None:
        """Add MACs for a named operation."""
        self._counts[name] = self._counts.get(name, 0) + int(macs)

    def total(self) -> int:
        """Return total MACs across all recorded operations."""
        return sum(self._counts.values())

    def by_name(self) -> dict[str, int]:
        """Return a copy of the breakdown of MACs by operation name."""
        return dict(self._counts)


def macs_matmul(m: int, k: int, n: int) -> int:
    """Calculate analytic MACs for a matrix multiplication (m x k) @ (k x n).

    Args:
        m: First dimension (rows of left matrix).
        k: Inner contraction dimension.
        n: Last dimension (columns of right matrix).

    Returns:
        int: Number of multiply-accumulate operations: m * k * n.
    """
    return int(m * k * n)
