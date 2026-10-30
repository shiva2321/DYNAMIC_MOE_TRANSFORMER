"""The systems under test: UniMoE variants (the growmoe pool with the learned switches) and baselines, all behind the pool protocol (``observe``, ``predict``, ``summary``, ``events``).

UniMoE variants (every one has the shipped defaults of 2026-10-30: batch vote, fingerprint shortlist, learned router, transfer spawn rule, competence monitor armed after infancy, sharing when full):
* ``shipped``   nothing else changed (shadow-trial growth policy, no set gate);
* ``U-real``    what the measurements recommend: batch-level set gate with persistence 2 (A-15), the founding-cluster twin fix with infant relaxation 1 (C-78, C-79), width by the fit-gap rule (C-86);
* ``U-deep``    U-real with the shadow-trial growth policy instead, which can also DEEPEN an expert (the only automatic vertical growth the pool has; C-89);
* ``U-recipe``  U-real with the training recipe passed in (steps per batch, replay size), tuned per arm by the driver.
Layers of representation experts (``growmoe.stack``) are not a variant: that module has the machinery to add a layer but nothing that decides when, so Stage 1 evaluates depth at the expert level
and lists the missing trigger as a finding (design amendment 1).

Baselines: ``finetune`` (one network, no memory), ``replay_eqmem`` (one network whose weights and replay buffer together use the pool's memory), ``knn_eqmem`` (lookup holding as many vectors as the
pool's memory), ``lda`` (streaming class-mean LDA), ``joint`` (re-trains on everything seen: a ceiling), ``oracle_experts`` (one expert per regime, TOLD the regime through ``ctx``: an upper bound for
routing, not a method). The seed rule of C-75 applies: a pool's seed is the stream's seed + 7.
"""

from typing import Callable, Dict, Optional

import numpy as np
import torch

from growmoe.baselines import FineTune, JointOracle, ReplayNet, _Base
from growmoe.config import CommonCfg, PhyloCfg
from growmoe.experts import MLPExpert
from growmoe.memory_baselines import KnnMemory, StreamingLDA
from growmoe.pools.phylo import PhyloMoE

UNIMOE = ("shipped", "U-real", "U-deep", "U-recipe")
BASELINES = ("finetune", "replay_eqmem", "knn_eqmem", "lda", "joint", "oracle_experts")
SYSTEMS = UNIMOE + BASELINES
VARIANT_SWITCHES: Dict[str, dict] = {
    "shipped": {},
    "U-real": dict(set_gate=True, set_gate_confirm=2, sig_dedupe=True, infant_relax=1.0, growth_rule="gap"),
    "U-deep": dict(set_gate=True, set_gate_confirm=2, sig_dedupe=True, infant_relax=1.0, growth_rule="probe"),
    "U-recipe": dict(set_gate=True, set_gate_confirm=2, sig_dedupe=True, infant_relax=1.0, growth_rule="gap"),
}
WIDTHS = (256, 512, 1024, 2048, 4096)


class OracleExperts(_Base):
    """One expert per regime and the regime is GIVEN (``ctx`` = regime id per sample): the routing ceiling. Never a method."""
    name = "oracle_experts(upper-bound)"
    uses_regime = True

    def __init__(self, cfg: CommonCfg) -> None:
        super().__init__(cfg)
        self.models: Dict[int, MLPExpert] = {}
        self.opts: Dict[int, torch.optim.Optimizer] = {}

    def observe(self, x, y=None, ctx=None):
        if ctx is None:
            raise ValueError("oracle_experts needs the regime of every sample in ctx")
        self.step += 1
        c, loss, spawned = self.cfg, float("nan"), []
        for r in ctx.unique().tolist():
            if r not in self.models:
                self.models[r] = MLPExpert(c.d, c.hidden, c.depth, c.n_classes, seed=c.seed * 1000 + len(self.models))
                self.opts[r] = torch.optim.Adam(self.models[r].parameters(), lr=c.lr)
                self.events.add("spawn", self.step - 1, expert=len(self.models) - 1, oracle=True)
                spawned.append(r)
            if y is not None:
                m = ctx == r
                for _ in range(c.inner_steps):
                    loss = self.models[r].train_step(x[m], y[m], self.opts[r])
        return self._record(x.shape[0], -1, loss=loss, spawned=spawned, n_experts=len(self.models))

    def predict(self, x, ctx=None):
        if ctx is None:
            raise ValueError("oracle_experts needs the regime of every sample in ctx")
        probs = torch.full((x.shape[0], self.cfg.n_classes), 1.0 / self.cfg.n_classes)
        for r in ctx.unique().tolist():
            if r in self.models:
                m = ctx == r
                self.models[r].eval()
                with torch.no_grad():
                    probs[m] = self.models[r].proba(x[m])
        return self._prediction(probs, ctx.long())

    def summary(self) -> dict:
        models = list(self.models.values())
        return self._summary(models, models[0].num_params() if models else 0)


def _common(n_classes: int, seed: int, d: int, recipe: dict) -> CommonCfg:
    ok = {k: v for k, v in recipe.items() if k in ("hidden", "depth", "lr", "batch_size", "inner_steps", "replay_size", "replay_frac")}
    return CommonCfg(d=d, n_classes=n_classes, seed=seed, **ok)


def phylo_cfg(name: str, n_classes: int, seed: int, recipe: Optional[dict] = None, d: int = 128) -> PhyloCfg:
    if name not in VARIANT_SWITCHES:
        raise ValueError(f"{name!r} is not a UniMoE variant; they are {list(VARIANT_SWITCHES)}")
    kw = dict(d=d, n_classes=n_classes, route_rule="accept_then_logf", seed=seed + 7, **VARIANT_SWITCHES[name])
    kw.update({k: v for k, v in (recipe or {}).items() if k in PhyloCfg.__dataclass_fields__})
    return PhyloCfg(**kw)


def make_system(name: str, n_classes: int, seed: int, recipe: Optional[dict] = None, mem_bytes: Optional[int] = None, total_samples: Optional[int] = None, d: int = 128):
    """A fresh pool. ``recipe`` carries the tuned training recipe of this arm; ``mem_bytes`` (the UniMoE pool's final memory) and ``total_samples`` size the equal-memory baselines."""
    recipe = dict(recipe or {})
    if name in VARIANT_SWITCHES:
        return PhyloMoE(phylo_cfg(name, n_classes, seed, recipe, d))
    cfg = _common(n_classes, seed, d, recipe)
    if name == "finetune":
        return FineTune(cfg)
    if name == "joint":
        return JointOracle(cfg)
    if name == "lda":
        return StreamingLDA(cfg)
    if name == "oracle_experts":
        return OracleExperts(cfg)
    if name == "knn_eqmem":
        if mem_bytes is None:
            raise ValueError("knn_eqmem needs the memory budget in mem_bytes")
        return KnnMemory(cfg, capacity=max(int(mem_bytes // (4 * d)), 16))
    if name == "replay_eqmem":
        if mem_bytes is None or total_samples is None:
            raise ValueError("replay_eqmem needs mem_bytes and total_samples")
        params = sum(p.numel() for p in MLPExpert(d, cfg.hidden, cfg.depth, n_classes).parameters())
        buffer = min(int(total_samples), int((mem_bytes - 4 * params) // (4 * d)))
        if buffer < 512:
            raise ValueError(f"no room for a replay buffer: width {cfg.hidden} leaves {buffer} vectors in {mem_bytes} bytes")
        return ReplayNet(cfg, replay_size=buffer)
    raise ValueError(f"unknown system {name!r}; the systems are {list(SYSTEMS)}")
