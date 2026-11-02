"""Arm adapters for continual learning architectures and baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch.nn as nn
from torch import Tensor

from growmoe.accounting import pool_report
from growmoe.baselines import JointOracle, ReplayNet
from growmoe.config import CommonCfg, PhyloCfg
from growmoe.experts import MLPExpert
from growmoe.memory_baselines import KnnMemory
from growmoe.pools.phylo import PhyloMoE
from m1.beds import Bed

# A1 = Baseline v1 (docs/roadmap/BASELINE_V1.md): the class defaults of PhyloCfg, with route_rule="accept_then_logf" (which switches the router on, as every
# measured pool does: the transfer spawn rule and the batch vote are class defaults) and the bed's training recipe (5 inner steps on the function regimes).
BASELINE_V1_CFG: Dict[str, Any] = dict(inner_steps=5, route_rule="accept_then_logf")
# A1L = the pool of the unaliased shootout of 2026-10-14, i.e. before stage 4 (router off, probe spawn rule, q 0.02, shortlist from 2 experts).
# It exists only for the harness check (m1/harness_check.py) and never enters an M1 result.
LEGACY_SHOOTOUT_CFG: Dict[str, Any] = dict(
    inner_steps=5, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf",
    gate_shortlist_min_experts=2, router=False, spawn_rule="probe", growth_rule="probe",
)


class Arm(ABC):
    """Uniform adapter interface for continual learning architectures."""

    def __init__(self, arm_id: str, model: Any, bed: Bed) -> None:
        self.arm_id = arm_id
        self.model = model
        self.bed = bed

    def observe(self, x: Tensor, y: Tensor) -> Any:
        """Observe one training batch of input features x and integer labels y."""
        return self.model.observe(x, y=y)

    def predict(self, x: Tensor) -> Any:
        """Predict labels for input features x without updating weights."""
        return self.model.predict(x)

    def memory_bytes(self) -> int:
        """Memory footprint in bytes calculated via growmoe.accounting.pool_report."""
        return int(pool_report(self.model, self.bed.d).memory_bytes)

    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """The static configuration of the arm: everything that decides what it computes, nothing that changes while it runs."""
        ...

    def describe(self) -> Dict[str, Any]:
        """The configuration plus the state at this moment (overridden where the state is informative)."""
        return self.config()

    def get_torch_modules(self) -> Optional[nn.Module]:
        """Return PyTorch module(s) for active parameter tracking in costlab."""
        return getattr(self.model, "net", None)


class A1PoolArm(Arm):
    """Adapter for the pool (PhyloMoE): A1 = Baseline v1, A1L = the legacy shootout pool (harness check only)."""

    def __init__(self, bed: Bed, variant: str = "baseline_v1", **cfg_overrides: Any) -> None:
        if variant not in ("baseline_v1", "pre_stage4"):
            raise ValueError(f"variant must be 'baseline_v1' or 'pre_stage4', got {variant!r}")
        self.variant = variant
        base = BASELINE_V1_CFG if variant == "baseline_v1" else LEGACY_SHOOTOUT_CFG
        cfg_dict: Dict[str, Any] = dict(d=bed.d, n_classes=bed.n_classes, seed=bed.pool_seed, **base)
        cfg_dict.update(cfg_overrides)
        self.cfg = PhyloCfg(**cfg_dict)
        super().__init__("A1" if variant == "baseline_v1" else "A1L", PhyloMoE(self.cfg), bed)

    def config(self) -> Dict[str, Any]:
        c = self.cfg
        return {
            "arm": self.arm_id, "variant": self.variant, "name": "PhyloMoE pool", "seed": self.bed.pool_seed,
            "inner_steps": c.inner_steps, "route_rule": c.route_rule, "router": c.router, "spawn_rule": c.spawn_rule,
            "growth_rule": c.growth_rule, "q": c.q, "alpha_spawn": c.alpha_spawn, "m_min": c.m_min,
            "gate_shortlist_min_experts": c.gate_shortlist_min_experts,
        }

    def describe(self) -> Dict[str, Any]:
        return {**self.config(), "experts": len(self.model.experts)}

    def get_torch_modules(self) -> Optional[nn.Module]:
        return nn.ModuleList([r.module for r in self.model.experts.values()]) if self.model.experts else None


class A3ReplayArm(Arm):
    """Adapter for Arm A3: Single network with replay matched at pool's memory."""

    def __init__(self, bed: Bed, pool_memory_bytes: int, hidden: int = 2048, inner_steps: int = 20) -> None:
        self.hidden, self.inner_steps, self.pool_memory_bytes = hidden, inner_steps, pool_memory_bytes
        net_template = MLPExpert(bed.d, hidden, 1, bed.n_classes)
        params_bytes = 4 * sum(p.numel() for p in net_template.parameters())
        buffer = min(bed.total_samples, max(16, int((pool_memory_bytes - params_bytes) // (4 * bed.d))))
        self.buffer = buffer
        cfg = CommonCfg(seed=bed.seed, d=bed.d, n_classes=bed.n_classes, inner_steps=inner_steps, hidden=hidden)
        super().__init__("A3", ReplayNet(cfg, replay_size=buffer), bed)

    def config(self) -> Dict[str, Any]:
        return {
            "arm": "A3", "name": "ReplayNet at equal memory", "seed": self.bed.seed,
            "hidden": self.hidden, "inner_steps": self.inner_steps, "replay_size": self.buffer,
            "target_pool_bytes": self.pool_memory_bytes,
        }


class A7KnnArm(Arm):
    """Adapter for Arm A7: k-Nearest Neighbours matched at pool's memory."""

    def __init__(self, bed: Bed, pool_memory_bytes: int, k: int = 5) -> None:
        self.pool_memory_bytes, self.k = pool_memory_bytes, k
        self.capacity = max(int(pool_memory_bytes // (4 * bed.d)), 16)
        cfg = CommonCfg(seed=bed.seed, d=bed.d, n_classes=bed.n_classes, inner_steps=1)
        super().__init__("A7", KnnMemory(cfg, k=k, capacity=self.capacity), bed)

    def config(self) -> Dict[str, Any]:
        return {
            "arm": "A7", "name": "KnnMemory at equal memory", "seed": self.bed.seed,
            "k": self.k, "capacity": self.capacity, "target_pool_bytes": self.pool_memory_bytes,
        }

    def get_torch_modules(self) -> Optional[nn.Module]:
        return None


class A8JointArm(Arm):
    """Adapter for Arm A8: Offline Joint Network (reference upper bound)."""

    def __init__(self, bed: Bed, hidden: int = 4096, inner_steps: int = 5) -> None:
        self.hidden, self.inner_steps = hidden, inner_steps
        cfg = CommonCfg(seed=bed.seed, d=bed.d, n_classes=bed.n_classes, inner_steps=inner_steps, hidden=hidden)
        super().__init__("A8", JointOracle(cfg), bed)

    def config(self) -> Dict[str, Any]:
        return {
            "arm": "A8", "name": "JointOracle reference ceiling", "seed": self.bed.seed,
            "hidden": self.hidden, "inner_steps": self.inner_steps,
        }


def pool_budget_bytes(bed: Bed, variant: str = "baseline_v1") -> int:
    """Memory in bytes of the pool at the end of the stream: the budget the equal-memory arms (A3, A7) are given."""
    pool = A1PoolArm(bed, variant=variant)
    for x, y, _ in bed:
        pool.observe(x, y)
    return pool.memory_bytes()


def build_arm(arm_id: str, bed: Bed, pool_memory_bytes: Optional[int] = None, **kwargs: Any) -> Arm:
    """Build an Arm adapter by identifier: 'A1' (Baseline v1), 'A1L' (legacy shootout pool, harness check only), 'A3', 'A7', 'A8'."""
    if arm_id == "A1":
        return A1PoolArm(bed, variant="baseline_v1", **kwargs)
    if arm_id == "A1L":
        return A1PoolArm(bed, variant="pre_stage4", **kwargs)
    if arm_id in ("A3", "A7"):
        budget = pool_memory_bytes if pool_memory_bytes is not None else pool_budget_bytes(bed)
        return A3ReplayArm(bed, pool_memory_bytes=budget, **kwargs) if arm_id == "A3" else A7KnnArm(bed, pool_memory_bytes=budget, **kwargs)
    if arm_id == "A8":
        return A8JointArm(bed, **kwargs)
    raise ValueError(f"Unknown arm {arm_id!r}; supported arms are A1, A1L, A3, A7, A8.")
