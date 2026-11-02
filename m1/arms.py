"""Arm adapters for continual learning architectures and baselines."""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from costlab.count import count_macs
from growmoe.accounting import pool_report
from growmoe.baselines import JointOracle, ReplayNet
from growmoe.config import CommonCfg, PhyloCfg
from growmoe.experts import MLPExpert
from growmoe.memory_baselines import KnnMemory, StreamingLDA
from growmoe.pools.phylo import PhyloMoE
from m1.baselines_extra import MahalanobisProto, RandomFeatureRidge
from m1.beds import Bed
from m1.config_m1 import A2_CANDIDATES, ALL_ARMS, B3_UNIPOOL_CANDIDATES

ALL_ARM_ORDER = list(ALL_ARMS)

# A1 = Baseline v1 (docs/roadmap/BASELINE_V1.md): class defaults of PhyloCfg, with route_rule="accept_then_logf"
# (which switches the router on) and 5 inner steps by default.
BASELINE_V1_CFG: Dict[str, Any] = dict(inner_steps=5, route_rule="accept_then_logf")

# A1L = pool of the unaliased shootout of 2026-10-14 (pre-stage-4; harness check only).
LEGACY_SHOOTOUT_CFG: Dict[str, Any] = dict(
    inner_steps=5,
    q=0.02,
    alpha_spawn=0.01,
    m_min=32,
    route_rule="accept_then_logf",
    gate_shortlist_min_experts=2,
    router=False,
    spawn_rule="probe",
    growth_rule="probe",
)


def instrument_pool(model: Any) -> Dict[str, float]:
    """Instance-level wrappers around model.observe, model.train_expert, and model._refit_router.

    Adds perf_counter durations to counters and returns the dictionary tracking pool time split.
    Pattern follows instrument() in fsmoe/skip_check.py.
    """
    split = {"observe": 0.0, "train_expert": 0.0, "router_refit": 0.0}
    orig_obs = model.observe
    orig_train = model.train_expert
    orig_refit = getattr(model, "_refit_router", None)

    def timed_train(*a: Any, **k: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig_train(*a, **k)
        finally:
            split["train_expert"] += time.perf_counter() - t0

    def timed_obs(*a: Any, **k: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig_obs(*a, **k)
        finally:
            split["observe"] += time.perf_counter() - t0

    def timed_refit(*a: Any, **k: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig_refit(*a, **k)
        finally:
            split["router_refit"] += time.perf_counter() - t0

    model.observe = timed_obs
    model.train_expert = timed_train
    if orig_refit is not None:
        model._refit_router = timed_refit

    return split


class Arm(ABC):
    """Uniform adapter interface for continual learning architectures."""

    uncounted_note: bool = False

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
        """Static configuration of the arm: everything that decides what it computes."""
        ...

    def describe(self) -> Dict[str, Any]:
        """Configuration plus state at this moment."""
        return self.config()

    def get_torch_modules(self) -> Optional[nn.Module]:
        """Return PyTorch module(s) for active parameter tracking in costlab."""
        return getattr(self.model, "net", None)


class A1PoolArm(Arm):
    """Adapter for the pool (PhyloMoE): A1 (Baseline v1), A1q, A2, and A1L (legacy shootout pool)."""

    uncounted_note: bool = True

    def __init__(
        self,
        bed: Bed,
        variant: str = "baseline_v1",
        arm_id: Optional[str] = None,
        instrument: bool = True,
        candidate: Optional[str] = None,
        model: Optional[Any] = None,
        cfg: Optional[Any] = None,
        **cfg_overrides: Any,
    ) -> None:
        self.candidate = candidate
        if variant not in ("baseline_v1", "pre_stage4"):
            raise ValueError(f"variant must be 'baseline_v1' or 'pre_stage4', got {variant!r}")
        self.variant = variant
        assigned_id = arm_id if arm_id is not None else ("A1" if variant == "baseline_v1" else "A1L")
        if model is not None and cfg is not None:
            self.cfg = cfg
            self.model = model
        else:
            base = BASELINE_V1_CFG if variant == "baseline_v1" else LEGACY_SHOOTOUT_CFG
            cfg_dict: Dict[str, Any] = dict(d=bed.d, n_classes=bed.n_classes, seed=bed.pool_seed, **base)
            cfg_dict.update(cfg_overrides)
            self.cfg = PhyloCfg(**cfg_dict)
            self.model = PhyloMoE(self.cfg)
        self.time_split: Optional[Dict[str, float]] = None
        if instrument:
            self.time_split = instrument_pool(self.model)
        super().__init__(assigned_id, self.model, bed)

    def config(self) -> Dict[str, Any]:
        c = self.cfg
        d: Dict[str, Any] = {
            "arm": self.arm_id,
            "variant": self.variant,
            "name": "UniPool" if hasattr(c, "lda_blend") else "PhyloMoE pool",
            "seed": self.bed.pool_seed,
            "inner_steps": c.inner_steps,
            "route_rule": c.route_rule,
            "router": c.router,
            "spawn_rule": c.spawn_rule,
            "growth_rule": c.growth_rule,
            "q": c.q,
            "alpha_spawn": c.alpha_spawn,
            "m_min": c.m_min,
            "gate_shortlist_min_experts": c.gate_shortlist_min_experts,
        }
        d.update(
            {
                "skip_known": c.skip_known,
                "transfer_chance": c.transfer_chance,
                "transfer_chance_weak_only": c.transfer_chance_weak_only,
                "set_gate": c.set_gate,
                "set_gate_confirm": c.set_gate_confirm,
                "sig_dedupe": c.sig_dedupe,
                "infant_relax": c.infant_relax,
            }
        )
        if hasattr(c, "lda_blend"):
            d.update(
                {
                    "gap_mode": getattr(c, "gap_mode", "recipe"),
                    "domain_blocks": [list(b) for b in getattr(c, "domain_blocks", ())],
                    "domain_head": getattr(c, "domain_head", "off"),
                    "domain_beta": getattr(c, "domain_beta", 1.0),
                    "lda_blend": getattr(c, "lda_blend", 0.0),
                    "lda_shrink": getattr(c, "lda_shrink", 0.1),
                }
            )
        if self.candidate is not None:
            d["candidate"] = self.candidate
        return d

    def describe(self) -> Dict[str, Any]:
        return {**self.config(), "experts": len(self.model.experts)}

    def get_torch_modules(self) -> Optional[nn.Module]:
        return nn.ModuleList([r.module for r in self.model.experts.values()]) if self.model.experts else None


class A3ReplayArm(Arm):
    """Adapter for Arm A3 and A3-2: Single network with replay matched at pool's memory."""

    uncounted_note: bool = False

    def __init__(
        self,
        bed: Bed,
        pool_memory_bytes: int,
        hidden: int = 2048,
        inner_steps: int = 20,
        arm_id: str = "A3",
    ) -> None:
        self.hidden, self.inner_steps, self.pool_memory_bytes = hidden, inner_steps, pool_memory_bytes
        net_template = MLPExpert(bed.d, hidden, 1, bed.n_classes)
        params_bytes = 4 * sum(p.numel() for p in net_template.parameters())
        buffer = min(bed.total_samples, max(16, int((pool_memory_bytes - params_bytes) // (4 * bed.d))))
        self.buffer = buffer
        cfg = CommonCfg(seed=bed.seed, d=bed.d, n_classes=bed.n_classes, inner_steps=inner_steps, hidden=hidden)
        super().__init__(arm_id, ReplayNet(cfg, replay_size=buffer), bed)

    def config(self) -> Dict[str, Any]:
        return {
            "arm": self.arm_id,
            "name": "ReplayNet at equal memory",
            "seed": self.bed.seed,
            "hidden": self.hidden,
            "inner_steps": self.inner_steps,
            "replay_size": self.buffer,
            "target_pool_bytes": self.pool_memory_bytes,
        }


class A4ReplayArm(Arm):
    """Adapter for Arm A4 and A4-2: Compute-matched replay network.

    Training multiply-adds do not exceed the pool's whole-stream counted MACs.
    """

    uncounted_note: bool = False

    def __init__(
        self,
        bed: Bed,
        pool_macs_budget: int,
        pool_memory_bytes: int,
        hidden: int = 256,
        inner_steps: int = 1,
        arm_id: str = "A4",
    ) -> None:
        self.hidden = hidden
        self.inner_steps = inner_steps
        self.pool_macs_budget = pool_macs_budget
        self.pool_memory_bytes = pool_memory_bytes
        net_template = MLPExpert(bed.d, hidden, 1, bed.n_classes)
        params_bytes = 4 * sum(p.numel() for p in net_template.parameters())
        buffer = min(bed.total_samples, max(16, int((pool_memory_bytes - params_bytes) // (4 * bed.d))))
        self.buffer = buffer
        cfg = CommonCfg(seed=bed.seed, d=bed.d, n_classes=bed.n_classes, inner_steps=inner_steps, hidden=hidden)
        super().__init__(arm_id, ReplayNet(cfg, replay_size=buffer), bed)

    def config(self) -> Dict[str, Any]:
        return {
            "arm": self.arm_id,
            "name": "Compute-matched ReplayNet",
            "seed": self.bed.seed,
            "hidden": self.hidden,
            "inner_steps": self.inner_steps,
            "replay_size": self.buffer,
            "target_pool_macs": self.pool_macs_budget,
            "target_pool_bytes": self.pool_memory_bytes,
        }


class A5aStreamingLdaArm(Arm):
    """Adapter for Arm A5a: Streaming linear discriminant analysis."""

    uncounted_note: bool = True

    def __init__(self, bed: Bed, shrinkage: float = 1e-2, arm_id: str = "A5a") -> None:
        self.shrinkage = float(shrinkage)
        cfg = CommonCfg(seed=bed.seed, d=bed.d, n_classes=bed.n_classes)
        super().__init__(arm_id, StreamingLDA(cfg, shrinkage=self.shrinkage), bed)

    def config(self) -> Dict[str, Any]:
        return {
            "arm": self.arm_id,
            "name": "StreamingLDA",
            "seed": self.bed.seed,
            "shrinkage": self.shrinkage,
        }

    def get_torch_modules(self) -> Optional[nn.Module]:
        return None


class A5bMahalanobisArm(Arm):
    """Adapter for Arm A5b: Mahalanobis prototype classifier (FeCAM style)."""

    uncounted_note: bool = True

    def __init__(self, bed: Bed, shrinkage: float = 0.1, scale: float = 1.0, arm_id: str = "A5b") -> None:
        self.shrinkage = float(shrinkage)
        self.scale = float(scale)
        cfg = CommonCfg(seed=bed.seed, d=bed.d, n_classes=bed.n_classes)
        super().__init__(arm_id, MahalanobisProto(cfg, shrinkage=self.shrinkage, scale=self.scale), bed)

    def config(self) -> Dict[str, Any]:
        return {
            "arm": self.arm_id,
            "name": "MahalanobisProto",
            "seed": self.bed.seed,
            "shrinkage": self.shrinkage,
            "scale": self.scale,
        }

    def get_torch_modules(self) -> Optional[nn.Module]:
        return None


class A6RandomFeatureArm(Arm):
    """Adapter for Arm A6: Random feature ridge head."""

    uncounted_note: bool = True

    def __init__(
        self,
        bed: Bed,
        D: int = 1024,
        ridge: float = 0.1,
        scale: float = 1.0,
        arm_id: str = "A6",
    ) -> None:
        self.D = int(D)
        self.ridge = float(ridge)
        self.scale = float(scale)
        cfg = CommonCfg(seed=bed.seed, d=bed.d, n_classes=bed.n_classes)
        super().__init__(arm_id, RandomFeatureRidge(cfg, D=self.D, ridge=self.ridge, scale=self.scale), bed)

    def config(self) -> Dict[str, Any]:
        return {
            "arm": self.arm_id,
            "name": "RandomFeatureRidge",
            "seed": self.bed.seed,
            "D": self.D,
            "ridge": self.ridge,
            "scale": self.scale,
        }

    def get_torch_modules(self) -> Optional[nn.Module]:
        return None


class A7KnnArm(Arm):
    """Adapter for Arm A7 and A7-2: k-Nearest Neighbours matched at pool's memory."""

    uncounted_note: bool = False

    def __init__(
        self,
        bed: Bed,
        pool_memory_bytes: int,
        k: int = 5,
        arm_id: str = "A7",
    ) -> None:
        self.pool_memory_bytes, self.k = pool_memory_bytes, k
        self.capacity = max(int(pool_memory_bytes // (4 * bed.d)), 16)
        cfg = CommonCfg(seed=bed.seed, d=bed.d, n_classes=bed.n_classes, inner_steps=1)
        super().__init__(arm_id, KnnMemory(cfg, k=k, capacity=self.capacity), bed)

    def config(self) -> Dict[str, Any]:
        return {
            "arm": self.arm_id,
            "name": "KnnMemory at equal memory",
            "seed": self.bed.seed,
            "k": self.k,
            "capacity": self.capacity,
            "target_pool_bytes": self.pool_memory_bytes,
        }

    def get_torch_modules(self) -> Optional[nn.Module]:
        return None


class A7allKnnArm(Arm):
    """Adapter for Arm A7all: kNN storing all seen examples."""

    uncounted_note: bool = False

    def __init__(self, bed: Bed, k: int = 5, arm_id: str = "A7all") -> None:
        self.k = k
        cfg = CommonCfg(seed=bed.seed, d=bed.d, n_classes=bed.n_classes, inner_steps=1)
        super().__init__(arm_id, KnnMemory(cfg, k=k, capacity=None), bed)

    def config(self) -> Dict[str, Any]:
        return {
            "arm": self.arm_id,
            "name": "KnnMemory all stored",
            "seed": self.bed.seed,
            "k": self.k,
            "capacity": "all",
        }

    def get_torch_modules(self) -> Optional[nn.Module]:
        return None


class A8JointArm(Arm):
    """Adapter for Arm A8: Offline Joint Network (reference ceiling)."""

    uncounted_note: bool = False

    def __init__(self, bed: Bed, hidden: int = 4096, inner_steps: int = 5, arm_id: str = "A8") -> None:
        self.hidden, self.inner_steps = hidden, inner_steps
        cfg = CommonCfg(seed=bed.seed, d=bed.d, n_classes=bed.n_classes, inner_steps=inner_steps, hidden=hidden)
        super().__init__(arm_id, JointOracle(cfg), bed)

    def config(self) -> Dict[str, Any]:
        return {
            "arm": self.arm_id,
            "name": "JointOracle reference ceiling",
            "seed": self.bed.seed,
            "hidden": self.hidden,
            "inner_steps": self.inner_steps,
        }


def pool_budget_bytes(bed: Bed, variant: str = "baseline_v1", **kwargs: Any) -> int:
    """Memory in bytes of the pool at the end of the stream."""
    pool = A1PoolArm(bed, variant=variant, **kwargs)
    for x, y, _ in bed:
        pool.observe(x, y)
    return pool.memory_bytes()


def pool_budget_macs(bed: Bed, pool_arm: str = "A1", **kwargs: Any) -> int:
    """Counted whole-stream training multiply-adds of the pool arm."""

    def _run() -> None:
        pool = build_arm(pool_arm, bed, **kwargs)
        for x, y, _ in bed:
            pool.observe(x, y)

    return count_macs(_run).macs


def compute_a4_steps(bed: Bed, pool_macs_budget: int, pool_memory_bytes: int, hidden: int) -> int:
    """Compute number of inner steps s(w) for A4 ReplayNet such that training MACs <= pool_macs_budget.

    Returns the integer steps s >= 1, or 0 if even 1 step exceeds budget.
    """
    net_template = MLPExpert(bed.d, hidden, 1, bed.n_classes)
    params_bytes = 4 * sum(p.numel() for p in net_template.parameters())
    buffer = min(bed.total_samples, max(16, int((pool_memory_bytes - params_bytes) // (4 * bed.d))))

    def _count(steps: int) -> int:
        cfg = CommonCfg(seed=bed.seed, d=bed.d, n_classes=bed.n_classes, inner_steps=steps, hidden=hidden)
        model = ReplayNet(cfg, replay_size=buffer)

        def _run() -> None:
            for x, y, _ in bed:
                model.observe(x, y=y)

        return count_macs(_run).macs

    c1 = _count(1)
    if c1 > pool_macs_budget:
        return 0

    s = int(math.floor(pool_macs_budget / c1))
    s = max(1, s)
    while s >= 1:
        c_s = _count(s)
        if c_s <= pool_macs_budget:
            return s
        s -= 1
    return 0


def build_pool_candidate(
    cand_name: str,
    bed: Bed,
    inner_steps: int = 5,
    q: float = 0.05,
    arm_id: str = "A2",
    instrument: bool = True,
    **kwargs: Any,
) -> Arm:
    """Build a pool candidate arm (PhyloCfg override or UniPool system)."""
    if cand_name in B3_UNIPOOL_CANDIDATES:
        if not bed.name.startswith("B3"):
            raise ValueError(f"{arm_id}: candidate {cand_name!r} is a UniPool candidate supported on B3 only.")
        from unimoe.pools import UniPool
        from unimoe.systems2 import uni_cfg

        lb = getattr(bed, "label_blocks", None)
        blocks = tuple(lb.values()) if isinstance(lb, dict) else tuple(lb or ())
        recipe = {"inner_steps": inner_steps, "q": q}
        recipe.update(kwargs)
        cfg = uni_cfg(cand_name, n_classes=bed.n_classes, seed=bed.seed, recipe=recipe, blocks=blocks, d=bed.d)
        model = UniPool(cfg)
        return A1PoolArm(bed, variant="baseline_v1", arm_id=arm_id, instrument=instrument, candidate=cand_name, model=model, cfg=cfg)

    if cand_name not in A2_CANDIDATES:
        if arm_id == "A2":
            raise ValueError(f"Unknown A2 candidate {cand_name!r}; the candidates are {sorted(A2_CANDIDATES)}.")
        raise ValueError(f"{arm_id}: candidate {cand_name!r} is not a PhyloCfg candidate ({sorted(A2_CANDIDATES)}) nor a UniPool candidate ({sorted(B3_UNIPOOL_CANDIDATES)}).")

    cand_cfg = dict(A2_CANDIDATES[cand_name])
    cand_cfg.update(inner_steps=inner_steps, q=q)
    merged = {**cand_cfg, **kwargs}
    return A1PoolArm(bed, variant="baseline_v1", arm_id=arm_id, instrument=instrument, candidate=cand_name, **merged)


def build_arm(
    arm_id: str,
    bed: Bed,
    pool_memory_bytes: Optional[int] = None,
    pool_macs_budget: Optional[int] = None,
    tuning: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Arm:
    """Build an Arm adapter by identifier and optional tuning dictionary."""
    if arm_id == "A1":
        steps = kwargs.pop("inner_steps", tuning.get("inner_steps", 5) if tuning else 5)
        return A1PoolArm(bed, variant="baseline_v1", arm_id="A1", inner_steps=steps, **kwargs)

    if arm_id == "A1q":
        steps = kwargs.pop("inner_steps", tuning.get("inner_steps", 5) if tuning else 5)
        q = kwargs.pop("q", tuning.get("q", 0.05) if tuning else 0.05)
        return A1PoolArm(bed, variant="baseline_v1", arm_id="A1q", inner_steps=steps, q=q, **kwargs)

    if arm_id == "A2":
        steps = kwargs.pop("inner_steps", tuning.get("inner_steps", 5) if tuning else 5)
        q = kwargs.pop("q", tuning.get("q", 0.05) if tuning else 0.05)
        cand_name = tuning.get("candidate", "shipped") if tuning else "shipped"
        return build_pool_candidate(cand_name, bed, inner_steps=steps, q=q, arm_id="A2", **kwargs)

    if arm_id == "A1L":
        return A1PoolArm(bed, variant="pre_stage4", arm_id="A1L", **kwargs)

    if arm_id.startswith("H_"):
        # held-default pair cell: candidate from the arm id, base operating point (steps and q of A1q) from the tuning dictionary
        cand_name = arm_id[2:]
        steps = kwargs.pop("inner_steps", tuning.get("inner_steps", 5) if tuning else 5)
        q = kwargs.pop("q", tuning.get("q", 0.05) if tuning else 0.05)
        return build_pool_candidate(cand_name, bed, inner_steps=steps, q=q, arm_id=arm_id, **kwargs)

    if arm_id in ("A3-2", "A4", "A4-2", "A7-2") and pool_memory_bytes is None:
        raise ValueError(f"{arm_id} needs the final memory of its reference pool (pool_memory_bytes); it is not computed from an untuned pool.")
    if arm_id in ("A4", "A4-2") and pool_macs_budget is None and not (tuning and "inner_steps" in tuning):
        raise ValueError(f"{arm_id} needs the counted training multiply-adds of its reference pool (pool_macs_budget) or a resolved inner_steps.")

    if arm_id in ("A3", "A3-2"):
        budget = pool_memory_bytes if pool_memory_bytes is not None else pool_budget_bytes(bed)
        hidden = kwargs.pop("hidden", tuning.get("hidden", 2048) if tuning else 2048)
        steps = kwargs.pop("inner_steps", tuning.get("inner_steps", 20) if tuning else 20)
        return A3ReplayArm(bed, pool_memory_bytes=budget, hidden=hidden, inner_steps=steps, arm_id=arm_id, **kwargs)

    if arm_id in ("A4", "A4-2"):
        mem_budget = pool_memory_bytes if pool_memory_bytes is not None else pool_budget_bytes(bed)
        macs_budget = pool_macs_budget
        hidden = kwargs.pop("hidden", tuning.get("hidden", 256) if tuning else 256)
        if "inner_steps" in kwargs:
            steps = kwargs.pop("inner_steps")
        elif tuning and "inner_steps" in tuning:
            steps = int(tuning["inner_steps"])
        else:
            steps = max(1, compute_a4_steps(bed, macs_budget, mem_budget, hidden))
        return A4ReplayArm(
            bed,
            pool_macs_budget=macs_budget,
            pool_memory_bytes=mem_budget,
            hidden=hidden,
            inner_steps=steps,
            arm_id=arm_id,
            **kwargs,
        )

    if arm_id == "A5a":
        shrinkage = kwargs.pop("shrinkage", tuning.get("shrinkage", 1e-2) if tuning else 1e-2)
        return A5aStreamingLdaArm(bed, shrinkage=shrinkage, arm_id="A5a", **kwargs)

    if arm_id == "A5b":
        shrinkage = kwargs.pop("shrinkage", tuning.get("shrinkage", 0.1) if tuning else 0.1)
        scale = kwargs.pop("scale", 1.0)
        return A5bMahalanobisArm(bed, shrinkage=shrinkage, scale=scale, arm_id="A5b", **kwargs)

    if arm_id == "A6":
        D = kwargs.pop("D", tuning.get("D", 1024) if tuning else 1024)
        ridge = kwargs.pop("ridge", tuning.get("ridge", 0.1) if tuning else 0.1)
        scale = kwargs.pop("scale", 1.0)
        return A6RandomFeatureArm(bed, D=D, ridge=ridge, scale=scale, arm_id="A6", **kwargs)

    if arm_id in ("A7", "A7-2"):
        budget = pool_memory_bytes if pool_memory_bytes is not None else pool_budget_bytes(bed)
        k = kwargs.pop("k", tuning.get("k", 5) if tuning else 5)
        return A7KnnArm(bed, pool_memory_bytes=budget, k=k, arm_id=arm_id, **kwargs)

    if arm_id == "A7all":
        k = kwargs.pop("k", tuning.get("k", 5) if tuning else 5)
        return A7allKnnArm(bed, k=k, arm_id="A7all", **kwargs)

    if arm_id == "A8":
        hidden = kwargs.pop("hidden", tuning.get("hidden", 4096) if tuning else 4096)
        steps = kwargs.pop("inner_steps", tuning.get("inner_steps", 5) if tuning else 5)
        return A8JointArm(bed, hidden=hidden, inner_steps=steps, arm_id="A8", **kwargs)

    raise ValueError(f"Unknown arm {arm_id!r}; supported arms are A1, A1q, A2, A1L, A3, A4, A5a, A5b, A6, A7, A7all, A8, A3-2, A4-2, A7-2.")
