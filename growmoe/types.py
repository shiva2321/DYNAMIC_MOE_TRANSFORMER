"""Shared result types and the ContinualPool protocol (spec section 3)."""

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor

from .telemetry import EventLog


@dataclass
class StepRecord:
    """Returned by ``pool.observe``. One entry per sample in the batch."""

    novel: Tensor                 # (B,) bool   flagged novel by the pool's own rule
    p_nov: Tensor                 # (B,) float  novelty p-value or [0,1] surrogate (low = novel)
    assigned: Tensor              # (B,) long   expert id trained on the sample, -1 if none
    loss: float                   # mean training loss over trained samples (nan if none)
    spawned: list = field(default_factory=list)   # expert ids born during this call
    n_experts: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class Prediction:
    """Returned by ``pool.predict``. Never learns."""

    probs: Tensor                 # (B, C) class probabilities (best guess even if unknown)
    expert: Tensor                # (B,) long   routed expert id, -1 if none exists
    p_nov: Tensor                 # (B,) float
    unknown: Tensor               # (B,) bool   abstain flag at the pool's configured level
    meta: dict = field(default_factory=dict)


@runtime_checkable
class ContinualPool(Protocol):
    """Every architecture (A, B, C, baselines, stacks, systems) implements this."""

    name: str
    events: EventLog

    def observe(self, x: Tensor, y: Optional[Tensor] = None, ctx: Optional[Tensor] = None) -> StepRecord: ...

    def predict(self, x: Tensor, ctx: Optional[Tensor] = None) -> Prediction: ...

    def summary(self) -> dict: ...
