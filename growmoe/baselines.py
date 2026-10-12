"""Baselines B0-B5, all ``ContinualPool`` s on the same ``MLPExpert``, optimiser and step budget as the pools.

B0 ``FineTune``        one network, sequential training (the forgetting reference)
B1 ``JointOracle``     one network trained on EVERYTHING seen so far, i.i.d. (the ceiling; uses extra compute and memory)
B2 ``ReplayNet``       one network + a global reservoir replay buffer
B3 ``StaticRouterMoE`` fixed number of experts, learned softmax router, auxiliary-loss-free load-balancing bias
B4 ``OracleRouted``    one expert per regime, regime id supplied by the benchmark (the routing upper bound)
B5 ``DPMeansGate``     online DP-means over inputs with a FIXED distance threshold lambda; one expert per cluster

None of them reads ground truth except B4, which receives regime ids only through a constructor-injected callable.
"""

import math
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import CommonCfg
from .experts import MLPExpert
from .replay import Reservoir as _Reservoir          # noqa: F401  (kept under its old name for the tests)
from .telemetry import EventLog
from .types import Prediction, StepRecord

NAN = float("nan")


def _per_sample(value, b: int, dtype=torch.long) -> Tensor:
    t = torch.as_tensor(value, dtype=dtype)
    return t.expand(b).clone() if t.ndim == 0 else t


class _Base:
    name = ""

    def __init__(self, cfg: Optional[CommonCfg]) -> None:
        self.cfg = cfg if cfg is not None else CommonCfg()
        self.events = EventLog()
        self.step = 0                                                           # observe calls so far
        self._rng = np.random.default_rng([self.cfg.seed, 0x42415345])

    def _record(self, b: int, assigned, loss: float = NAN, novel=None, p_nov=None, spawned=(), n_experts: int = 0):
        return StepRecord(novel=torch.zeros(b, dtype=torch.bool) if novel is None else novel,
                          p_nov=torch.ones(b) if p_nov is None else p_nov,
                          assigned=_per_sample(assigned, b), loss=loss, spawned=list(spawned), n_experts=n_experts)

    @staticmethod
    def _prediction(probs: Tensor, expert, p_nov=None, unknown=None) -> Prediction:
        b = probs.shape[0]
        return Prediction(probs=probs, expert=_per_sample(expert, b),
                          p_nov=torch.ones(b) if p_nov is None else p_nov,
                          unknown=torch.zeros(b, dtype=torch.bool) if unknown is None else unknown)

    def _summary(self, models: List[nn.Module], active: int, n_experts: int = None, **extra) -> dict:
        out = {"n_experts": len(models) if n_experts is None else n_experts,
               "n_params": sum(p.numel() for m in models for p in m.parameters()),
               "active_params_per_sample": active, "lineage": {}, "states": {}}
        out.update(extra)
        return out


# --------------------------------------------------------------------------------- single networks
class FineTune(_Base):
    name = "B0_finetune"

    def __init__(self, cfg: CommonCfg = None) -> None:
        super().__init__(cfg)
        c = self.cfg
        self.net = MLPExpert(c.d, c.hidden, c.depth, c.n_classes, seed=c.seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=c.lr)

    def _train(self, x, y):
        loss = NAN
        for _ in range(self.cfg.inner_steps):
            loss = self.net.train_step(x, y, self.opt)
        return loss

    def observe(self, x, y=None, ctx=None) -> StepRecord:
        self.step += 1
        if y is None:
            return self._record(x.shape[0], -1, n_experts=1)
        return self._record(x.shape[0], 0, loss=self._train(x, y), n_experts=1)

    def predict(self, x, ctx=None) -> Prediction:
        self.net.eval()
        with torch.no_grad():
            return self._prediction(self.net.proba(x), 0)

    def summary(self) -> dict:
        return self._summary([self.net], self.net.num_params())


class JointOracle(FineTune):
    """Keeps every sample ever seen and, at each ``observe``, trains on batches drawn i.i.d. from all of it."""
    name = "B1_joint(ceiling)"

    def __init__(self, cfg: CommonCfg = None, joint_steps: int = None) -> None:
        super().__init__(cfg)
        self.joint_steps = joint_steps if joint_steps is not None else 4 * self.cfg.inner_steps
        self._x, self._y, self._n = torch.zeros(1024, self.cfg.d), torch.zeros(1024, dtype=torch.long), 0

    def _store(self, x: Tensor, y: Tensor) -> None:
        need = self._n + x.shape[0]
        if need > self._x.shape[0]:                                             # grow geometrically, never re-copy per call
            cap = max(need, 2 * self._x.shape[0])
            self._x = torch.cat([self._x[:self._n], torch.zeros(cap - self._n, self.cfg.d)])
            self._y = torch.cat([self._y[:self._n], torch.zeros(cap - self._n, dtype=torch.long)])
        self._x[self._n:need], self._y[self._n:need] = x.detach(), y.detach()
        self._n = need

    def observe(self, x, y=None, ctx=None) -> StepRecord:
        self.step += 1
        if y is None:
            return self._record(x.shape[0], -1, n_experts=1)
        self._store(x, y)
        allx, ally = self._x[:self._n], self._y[:self._n]
        loss = NAN
        for _ in range(self.joint_steps):
            idx = torch.from_numpy(self._rng.integers(0, allx.shape[0], size=self.cfg.batch_size))
            loss = self.net.train_step(allx[idx], ally[idx], self.opt)
        return self._record(x.shape[0], 0, loss=loss, n_experts=1)


class ReplayNet(FineTune):
    """One network; every step mixes in a batch from a global reservoir (default 8x the per-expert replay size)."""
    name = "B2_replay"

    def __init__(self, cfg: CommonCfg = None, replay_size: int = None) -> None:
        super().__init__(cfg)
        size = replay_size if replay_size is not None else 8 * self.cfg.replay_size
        self.buffer = _Reservoir(size, self.cfg.d, self._rng)

    def observe(self, x, y=None, ctx=None) -> StepRecord:
        self.step += 1
        if y is None:
            return self._record(x.shape[0], -1, n_experts=1)
        c = self.cfg
        n_replay = int(round(x.shape[0] * c.replay_frac / (1.0 - c.replay_frac)))
        loss = NAN
        for _ in range(c.inner_steps):
            replay = self.buffer.sample(n_replay) if self.buffer.n and n_replay > 0 else None
            loss = self.net.train_step(x, y, self.opt, replay=replay)
        self.buffer.add(x, y)
        return self._record(x.shape[0], 0, loss=loss, n_experts=1)


# ------------------------------------------------------------------------------------ B3 static MoE
class StaticRouterMoE(_Base):
    """Top-k sparse mixture with a learned linear router. Load balance by a selection-only bias that is nudged
    towards uniform load after every step (no auxiliary loss). All experts share one seen-class mask."""
    name = "B3_static_router_moe"

    def __init__(self, cfg: CommonCfg = None, n_experts: int = 8, top_k: int = 2, bias_rate: float = 0.05) -> None:
        super().__init__(cfg)
        c = self.cfg
        self.top_k, self.bias_rate = top_k, bias_rate
        self.experts = nn.ModuleList(MLPExpert(c.d, c.hidden, c.depth, c.n_classes, seed=c.seed * 1000 + e)
                                     for e in range(n_experts))
        g = torch.Generator().manual_seed(c.seed + 77)
        with torch.random.fork_rng(devices=[]):                      # nn.Linear's default init would advance the global RNG
            self.router = nn.Linear(c.d, n_experts, bias=False)
        with torch.no_grad():
            self.router.weight.copy_(torch.randn(n_experts, c.d, generator=g) / math.sqrt(c.d))
        self.sel_bias = torch.zeros(n_experts)                       # used for expert SELECTION only, never for the gate
        self.opt = torch.optim.Adam(list(self.experts.parameters()) + list(self.router.parameters()), lr=c.lr)

    def _mix(self, x: Tensor):
        scores = self.router(x)
        top = (scores.detach() + self.sel_bias).topk(self.top_k, dim=1).indices
        gate = F.softmax(scores, dim=1).gather(1, top)
        gate = gate / gate.sum(dim=1, keepdim=True)
        probs = torch.stack([F.softmax(e(x), dim=-1) for e in self.experts], dim=1)          # (B, E, C)
        chosen = probs[torch.arange(x.shape[0])[:, None], top]                               # (B, k, C)
        return (gate[:, :, None] * chosen).sum(dim=1), top

    def observe(self, x, y=None, ctx=None) -> StepRecord:
        self.step += 1
        n = len(self.experts)
        if y is None:
            return self._record(x.shape[0], -1, n_experts=n)
        for e in self.experts:
            e.mark_seen(y)
        loss_val, top = NAN, None
        for _ in range(self.cfg.inner_steps):
            probs, top = self._mix(x)
            loss = -torch.log(probs.gather(1, y.long()[:, None]).clamp_min(1e-9)).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            loss_val = float(loss.detach())
            load = torch.bincount(top.flatten(), minlength=n).float() / top.numel()
            self.sel_bias += self.bias_rate * torch.sign(1.0 / n - load)
        return self._record(x.shape[0], top[:, 0], loss=loss_val, n_experts=n)

    def predict(self, x, ctx=None) -> Prediction:
        with torch.no_grad():
            probs, top = self._mix(x)
        return self._prediction(probs, top[:, 0])

    def summary(self) -> dict:
        per = sum(p.numel() for p in self.experts[0].parameters())
        return self._summary(list(self.experts) + [self.router], self.top_k * per, n_experts=len(self.experts),
                             bias=self.sel_bias.tolist())


# ------------------------------------------------------------------------------------- B4 oracle
class OracleRouted(_Base):
    """One expert per regime. The regime of training batch ``i`` comes from ``regime_of(i)``, a callable the
    benchmark injects; for evaluation the benchmark calls ``set_eval_regime``. An upper bound, not a method."""
    name = "B4_oracle_routed(upper-bound)"

    def __init__(self, regime_of: Callable[[int], int], cfg: CommonCfg = None) -> None:
        super().__init__(cfg)
        self.regime_of = regime_of
        self.eval_regime: Optional[int] = None
        self.models: Dict[int, MLPExpert] = {}
        self.opts: Dict[int, torch.optim.Optimizer] = {}
        self.expert_id: Dict[int, int] = {}

    def set_eval_regime(self, regime: int) -> None:
        self.eval_regime = regime

    def observe(self, x, y=None, ctx=None) -> StepRecord:
        r = int(self.regime_of(self.step))
        self.step += 1
        spawned = []
        if r not in self.models:
            c = self.cfg
            self.expert_id[r] = len(self.models)
            self.models[r] = MLPExpert(c.d, c.hidden, c.depth, c.n_classes, seed=c.seed * 1000 + len(self.models))
            self.opts[r] = torch.optim.Adam(self.models[r].parameters(), lr=c.lr)
            self.events.add("spawn", self.step - 1, expert=self.expert_id[r], oracle=True)
            spawned.append(self.expert_id[r])
        loss = NAN
        if y is not None:
            for _ in range(self.cfg.inner_steps):
                loss = self.models[r].train_step(x, y, self.opts[r])
        return self._record(x.shape[0], self.expert_id[r], loss=loss, spawned=spawned, n_experts=len(self.models))

    def predict(self, x, ctx=None) -> Prediction:
        r = self.eval_regime
        if r is None:
            raise RuntimeError("OracleRouted.predict needs set_eval_regime(regime) from the benchmark")
        if r not in self.models:
            uniform = torch.full((x.shape[0], self.cfg.n_classes), 1.0 / self.cfg.n_classes)
            return self._prediction(uniform, -1)
        with torch.no_grad():
            return self._prediction(self.models[r].proba(x), self.expert_id[r])

    def summary(self) -> dict:
        models = list(self.models.values())
        return self._summary(models, models[0].num_params() if models else 0)


# --------------------------------------------------------------------------------------- B5 DP-means
class DPMeansGate(_Base):
    """Online DP-means (Kulis & Jordan, 2012) on the unit sphere: a sample farther than ``lam`` (squared Euclidean
    distance, 0..4) from every centre founds a new cluster; each cluster owns one expert. ``lam`` is FIXED: this is the
    hand-tuned-threshold straw man."""
    name = "B5_dpmeans"

    def __init__(self, cfg: CommonCfg = None, lam: float = 1.0) -> None:
        super().__init__(cfg)
        if not 0.0 < lam < 4.0:
            raise ValueError("lam is a squared distance between unit vectors and must lie in (0, 4)")
        self.lam = lam
        self.centres = torch.zeros(0, self.cfg.d)
        self._sums = torch.zeros(0, self.cfg.d)
        self.models: List[MLPExpert] = []
        self.opts: List[torch.optim.Optimizer] = []

    def _spawn(self, x: Tensor) -> int:
        c = self.cfg
        k = len(self.models)
        self.centres = torch.cat([self.centres, x[None]])
        self._sums = torch.cat([self._sums, x[None]])
        self.models.append(MLPExpert(c.d, c.hidden, c.depth, c.n_classes, seed=c.seed * 1000 + k))
        self.opts.append(torch.optim.Adam(self.models[-1].parameters(), lr=c.lr))
        return k

    def observe(self, x, y=None, ctx=None) -> StepRecord:
        b = x.shape[0]
        novel = torch.zeros(b, dtype=torch.bool)
        p_nov = torch.ones(b)
        assigned = torch.zeros(b, dtype=torch.long)
        spawned = []
        for i in range(b):
            d2 = (2.0 - 2.0 * (self.centres @ x[i])).min().item() if len(self.models) else float("inf")
            if d2 > self.lam:
                novel[i] = True
                p_nov[i] = 0.0 if math.isinf(d2) else self.lam / d2
                k = self._spawn(x[i])
                spawned.append(k)
                self.events.add("spawn", self.step, expert=k, lam=self.lam)
            else:
                k = int((self.centres @ x[i]).argmax())
                self._sums[k] += x[i]
                self.centres[k] = F.normalize(self._sums[k], dim=0)
            assigned[i] = k
        loss = NAN
        if y is not None:
            losses = []
            for k in assigned.unique().tolist():
                m = assigned == k
                for _ in range(self.cfg.inner_steps):
                    loss_k = self.models[k].train_step(x[m], y[m], self.opts[k])
                losses.append(loss_k)
            loss = float(np.mean(losses))
        self.step += 1
        return self._record(b, assigned, loss=loss, novel=novel, p_nov=p_nov, spawned=spawned,
                            n_experts=len(self.models))

    def predict(self, x, ctx=None) -> Prediction:
        b = x.shape[0]
        if not self.models:
            return self._prediction(torch.full((b, self.cfg.n_classes), 1.0 / self.cfg.n_classes), -1,
                                    p_nov=torch.zeros(b), unknown=torch.ones(b, dtype=torch.bool))
        sim, idx = (x @ self.centres.T).max(dim=1)
        d2 = 2.0 - 2.0 * sim
        probs = torch.zeros(b, self.cfg.n_classes)
        with torch.no_grad():
            for k in idx.unique().tolist():
                m = idx == k
                probs[m] = self.models[k].proba(x[m])
        p_nov = torch.where(d2 > self.lam, self.lam / d2.clamp_min(1e-12), torch.ones_like(d2))
        return self._prediction(probs, idx, p_nov=p_nov, unknown=d2 > self.lam)

    def summary(self) -> dict:
        return self._summary(self.models, self.models[0].num_params() if self.models else 0, lam=self.lam)
