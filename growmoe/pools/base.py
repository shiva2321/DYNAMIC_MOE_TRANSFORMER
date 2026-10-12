"""Bookkeeping shared by the three architectures: expert registry, lineage tree, per-expert optimisers and replay.

An expert is a module that offers ``train_step(x, y, opt, replay=None, project=False)``, ``proba(x)``,
``mark_seen(y)``, ``num_params()`` and ``clone()`` (``MLPExpert`` does; so must the delta experts of approach C).
Ids are unique, monotone and never reused. The lineage (id -> parent id) is kept for removed experts too, so
``ancestors`` keeps working after a prune or a merge; ``set_parent`` refuses anything that would create a cycle.
Every expert owns exactly one optimiser, built over exactly its own parameters, so adding or removing an expert
never touches another expert's optimiser state.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..config import CommonCfg
from ..replay import Reservoir
from ..telemetry import EventLog

STATES = ("infant", "active", "dormant", "archived")
NAN = float("nan")


@dataclass
class ExpertRecord:
    id: int
    module: nn.Module
    parent: Optional[int]
    birth_step: int
    state: str = "active"
    n_assigned: int = 0
    last_active: int = 0
    lr_mult: float = 1.0
    replay: Reservoir = None
    opt: Optional[torch.optim.Optimizer] = None
    meta: dict = field(default_factory=dict)               # architecture-specific: signature, monitor, key, ...

    @property
    def replay_x(self) -> torch.Tensor:
        return self.replay.data[0]

    @property
    def replay_y(self) -> torch.Tensor:
        return self.replay.data[1]


class BasePool:
    name = "base"

    def __init__(self, cfg: CommonCfg = None) -> None:
        self.cfg = cfg if cfg is not None else CommonCfg()
        self.events = EventLog()
        self.step = 0
        self.experts: Dict[int, ExpertRecord] = {}
        self._parents: Dict[int, Optional[int]] = {}       # lineage of every expert ever created, removed ones included
        self._next_id = 0
        self._rng = np.random.default_rng([self.cfg.seed, 0x504F4F4C])

    # --------------------------------------------------------------------------------------- registry
    def tick(self) -> None:
        self.step += 1

    def make_optimizer(self, module: nn.Module, lr_mult: float = 1.0) -> torch.optim.Optimizer:
        return torch.optim.Adam(module.parameters(), lr=self.cfg.lr * lr_mult)

    def add_expert(self, module: nn.Module, parent: Optional[int] = None, state: str = "active",
                   meta: Optional[dict] = None) -> ExpertRecord:
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}")
        if parent is not None and parent not in self._parents:
            raise KeyError(f"unknown parent expert {parent}")
        eid = self._next_id
        self._next_id += 1
        rec = ExpertRecord(id=eid, module=module, parent=parent, birth_step=self.step, state=state,
                           last_active=self.step,
                           replay=Reservoir(self.cfg.replay_size, self.cfg.d, np.random.default_rng([self.cfg.seed, 0x52504C, eid])),
                           opt=self.make_optimizer(module), meta=dict(meta or {}))
        self.experts[eid] = rec
        self._parents[eid] = parent
        return rec

    def remove_expert(self, eid: int) -> ExpertRecord:
        """Drop an expert and its optimiser; its place in the lineage stays."""
        return self.experts.pop(eid)

    def set_state(self, eid: int, state: str) -> None:
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}")
        self.experts[eid].state = state

    def set_lr_mult(self, eid: int, mult: float) -> None:
        rec = self.experts[eid]
        rec.lr_mult = mult
        for g in rec.opt.param_groups:
            g["lr"] = self.cfg.lr * mult

    # ----------------------------------------------------------------------------------------- lineage
    def ancestors(self, eid: int) -> List[int]:
        """Parent, grandparent, ... up to the root (nearest first). Works for removed experts too."""
        if eid not in self._parents:
            raise KeyError(f"unknown expert {eid}")
        out, cur = [], self._parents[eid]
        while cur is not None:
            out.append(cur)
            cur = self._parents[cur]
        return out

    def set_parent(self, eid: int, parent: Optional[int]) -> None:
        if eid not in self._parents or (parent is not None and parent not in self._parents):
            raise KeyError("unknown expert")
        if parent is not None and (parent == eid or eid in self.ancestors(parent)):
            raise ValueError(f"making {parent} the parent of {eid} would create a cycle")
        self._parents[eid] = parent
        if eid in self.experts:
            self.experts[eid].parent = parent

    # ----------------------------------------------------------------------------------------- training
    def train_expert(self, eid: int, x: torch.Tensor, y: torch.Tensor, replay: bool = True, project: bool = False) -> float:
        """``inner_steps`` optimiser steps on (x, y) mixed with the expert's own replay; then the batch enters its reservoir."""
        rec, c = self.experts[eid], self.cfg
        share = min(c.replay_frac, 0.95)
        n_replay = int(round(x.shape[0] * share / (1.0 - share))) if replay and rec.replay.n else 0
        loss = NAN
        for _ in range(c.inner_steps):
            batch = rec.replay.sample(n_replay) if n_replay > 0 else None
            loss = rec.module.train_step(x, y, rec.opt, replay=batch, project=project)
        if replay:
            rec.replay.add(x, y)
        rec.n_assigned += x.shape[0]
        rec.last_active = self.step
        return loss

    # ------------------------------------------------------------------------------------------ summary
    def summary(self) -> dict:
        live = list(self.experts.values())
        n_params = sum(p.numel() for r in live for p in r.module.parameters())
        out = {"n_experts": len(live), "n_params": n_params,
                "active_params_per_sample": n_params / len(live) if live else 0,
                "lineage": {r.id: r.parent for r in live}, "states": {r.id: r.state for r in live},
                "state_counts": dict(Counter(r.state for r in live))}
        if hasattr(self, "events") and self.events is not None:
            out["n_spawn"] = len(self.events.of_kind("spawn"))
            out["n_extend"] = len(self.events.of_kind("extend"))
            out["n_grow"] = len(self.events.of_kind("grow"))
            out["n_probation_events"] = len(self.events.of_kind("abstain")) + out["n_spawn"]
        return out
