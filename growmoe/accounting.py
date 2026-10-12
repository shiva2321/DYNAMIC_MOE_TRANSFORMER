"""Honest resource accounting for continual-learning pools (M0).

Three quantities, in the same units for every method:

* parameters       counted once even when several experts share a module (Pool C's frozen base)
* stored vectors   unit vectors kept for later use (signature prototypes, replay reservoirs, gate keys, kNN memory)
* predict MACs     multiply-adds needed to answer ONE input: the gate scan (every stored signature vector, scanned once)
                   plus the weight matrices of the ONE expert that answers

Memory in bytes is ``4 * parameters + 4 * d * stored_vectors`` (float32; labels and bookkeeping are ignored on purpose, they are
tiny next to the vectors).
"""

from dataclasses import dataclass
from typing import Iterable

import torch.nn as nn

BYTES_PER_FLOAT = 4


def unique_param_count(modules: Iterable[nn.Module]) -> int:
    """Number of parameters across ``modules`` with each shared tensor counted exactly once."""
    seen = {}
    for m in modules:
        for p in m.parameters():
            seen[id(p)] = p.numel()
    return int(sum(seen.values()))


def weight_macs(module: nn.Module) -> int:
    """Multiply-adds of one forward pass through ``module``'s weight matrices (2-D parameters), per input."""
    return int(sum(p.numel() for p in module.parameters() if p.dim() == 2))


@dataclass
class PoolReport:
    n_experts: int
    params: int                    # each shared tensor counted once
    stored_vectors: int
    memory_bytes: int
    gate_macs: int
    expert_macs: int
    predict_macs: int
    naive_params: int = None       # the per-expert sum that repeats a shared module (what Pool C used to report); equals params otherwise

    def __post_init__(self) -> None:
        if self.naive_params is None:
            self.naive_params = self.params


def _report(n_experts: int, params: int, stored: int, d: int, gate: int, expert: int, naive: int = None) -> PoolReport:
    return PoolReport(n_experts=n_experts, params=params, stored_vectors=stored,
                      memory_bytes=BYTES_PER_FLOAT * params + BYTES_PER_FLOAT * d * stored,
                      gate_macs=gate, expert_macs=expert, predict_macs=gate + expert, naive_params=naive)


def pool_report(pool, d: int) -> PoolReport:
    """Resource report for any pool in this package (expert registries, single networks, memory baselines)."""
    if hasattr(pool, "lookup_vectors"):                                                # kNN / class means / streaming LDA
        return _report(1, 0, pool.stored_vectors(), d, pool.lookup_vectors() * d, 0)
    if isinstance(getattr(pool, "experts", None), dict):                               # PhyloMoE, KeyedDeltaMoE, BNPMoE
        recs = list(pool.experts.values())
        if not recs:
            return _report(0, 0, 0, d, 0, 0)
        stored = gate = 0
        for r in recs:
            sig = r.meta.get("sig")
            if sig is not None:
                stored += sig.n
                gate += sig.n * d
            if r.replay is not None:
                stored += r.replay.n
        if all(r.meta.get("sig") is None for r in recs) and hasattr(pool, "keys"):     # key-gated pool: one cosine per expert
            stored += len(pool.keys)
            gate += len(pool.keys) * d
        expert = int(round(sum(weight_macs(r.module) for r in recs) / len(recs)))
        naive = sum(p.numel() for r in recs for p in r.module.parameters())
        return _report(len(recs), unique_param_count([r.module for r in recs]), stored, d, gate, expert, naive)
    if hasattr(pool, "net"):                                                           # FineTune, JointOracle, ReplayNet
        if hasattr(pool, "buffer"):
            stored = pool.buffer.n
        else:
            stored = getattr(pool, "_n", 0)
        return _report(1, unique_param_count([pool.net]), stored, d, 0, weight_macs(pool.net))
    models = getattr(pool, "models", None)                                             # OracleRouted, DPMeansGate
    if models:
        mods = list(models.values()) if isinstance(models, dict) else list(models)
        stored = len(getattr(pool, "centres", [])) if hasattr(pool, "centres") else 0
        return _report(len(mods), unique_param_count(mods), stored, d, stored * d, weight_macs(mods[0]))
    s = pool.summary()
    return _report(int(s.get("n_experts", 0)), int(s.get("n_params", 0)), 0, d, 0, 0)
