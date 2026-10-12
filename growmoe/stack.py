"""StackMoE and vertical scaling (Task 18).

Provides:
- RepExpert: Residual R^d -> R^d representation expert, identity-initialized (zero output weights).
- RepLayer: A layer of representation experts with per-expert VmfSignature routing gates and infant/mature lifecycle.
- StackMoE: Multi-layer vertical ContinualPool with frozen mature representation experts and head ContinualPool.
- StackGrowthPolicy: Generalised GrowthPolicy supporting {"widen", "deepen", "add_layer"}.
"""

import copy
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import CommonCfg, GrowthCfg
from .experts import MLPExpert, ResBlock
from .growth import GrowthDecision, GrowthPolicy
from .signature import VmfSignature
from .telemetry import EventLog
from .types import ContinualPool, Prediction, StepRecord


# --------------------------------------------------------------------------- RepExpert
class RepExpert(nn.Module):
    """Residual representation expert R^d -> R^d, identity-initialised with zero output weights."""

    def __init__(self, d: int = 128, hidden: int = 128, depth: int = 1, seed: int = 0) -> None:
        super().__init__()
        self.d = d
        self.hidden_dim = hidden
        self.seed = seed
        gen = torch.Generator().manual_seed(seed)
        self.blocks = nn.ModuleList(ResBlock(d, hidden, gen) for _ in range(depth))

        # Identity initialization: zero output weights so h + 0 = h
        with torch.no_grad():
            for b in self.blocks:
                b.w2.zero_()
                b.b2.zero_()

        self._ops = 0
        self.frozen = False

    @property
    def hidden(self) -> int:
        return self.blocks[0].w1.shape[0] if len(self.blocks) > 0 else self.hidden_dim

    def forward(self, h: Tensor) -> Tensor:
        for b in self.blocks:
            h = b(h)
        return h

    def freeze(self) -> "RepExpert":
        self.frozen = True
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self

    def unfreeze(self) -> "RepExpert":
        self.frozen = False
        for p in self.parameters():
            p.requires_grad = True
        self.train()
        return self

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def clone(self) -> "RepExpert":
        return copy.deepcopy(self)

    def _next_rng(self) -> np.random.Generator:
        self._ops += 1
        return np.random.default_rng([self.seed, 0x535441434B, self._ops])

    @torch.no_grad()
    def widen(self, k: int, rng: Optional[np.random.Generator] = None) -> "RepExpert":
        if k <= 0:
            raise ValueError("widen needs k >= 1")
        rng = rng if rng is not None else self._next_rng()
        for b in self.blocks:
            d = b.w1.shape[1]
            fresh = torch.from_numpy(rng.standard_normal((k, d)) / math.sqrt(d)).to(b.w1)
            b.w1 = nn.Parameter(torch.cat([b.w1.data, fresh]))
            b.b1 = nn.Parameter(torch.cat([b.b1.data, torch.zeros(k).to(b.b1)]))
            b.w2 = nn.Parameter(torch.cat([b.w2.data, torch.zeros(d, k).to(b.w2)], dim=1))
        return self

    @torch.no_grad()
    def deepen(self, rng: Optional[np.random.Generator] = None) -> "RepExpert":
        rng = rng if rng is not None else self._next_rng()
        ref = self.blocks[-1]
        gen = torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))
        block = ResBlock(ref.w1.shape[1], ref.w1.shape[0], gen).to(ref.w1)
        block.w2.zero_()
        block.b2.zero_()
        self.blocks.append(block)
        return self


class ExpertDict(nn.ModuleDict):
    """ModuleDict supporting both integer and string key lookup."""

    def __getitem__(self, key: Union[int, str]) -> nn.Module:
        return super().__getitem__(str(key))

    def __contains__(self, key: Union[int, str]) -> bool:
        return super().__contains__(str(key))


# --------------------------------------------------------------------------- RepLayer
# --------------------------------------------------------------------------- RepLayer
class RepLayer(nn.Module):
    """A vertical layer of representation experts with per-expert VmfSignature routing."""

    def __init__(self, layer_id: int, d: int = 128, hidden: int = 128, seed: int = 0) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.d = d
        self.hidden = hidden
        self.seed = seed
        self.experts = ExpertDict()
        self.signatures: Dict[int, VmfSignature] = {}
        self.states: Dict[int, str] = {}  # "infant" | "mature"
        self.sig_frozen: Dict[int, bool] = {}
        self.optimizers: Dict[int, torch.optim.Optimizer] = {}
        self.samples_seen: Dict[int, int] = {}
        self.loss_hist: Dict[int, List[float]] = {}
        self._update_counts: Dict[int, int] = {}
        self._next_eid = 0

        # Initialize with one mature identity expert
        self.spawn_expert(state="mature", seed=seed)

    def spawn_expert(self, state: str = "mature", seed: Optional[int] = None, lr: float = 3e-3) -> int:
        eid = self._next_eid
        self._next_eid += 1
        s = seed if seed is not None else (self.seed * 1000 + eid)
        rep = RepExpert(d=self.d, hidden=self.hidden, depth=1, seed=s)
        if state == "mature":
            rep.freeze()
        self.experts[str(eid)] = rep
        self.signatures[eid] = VmfSignature(d=self.d, seed=s)
        self.states[eid] = state
        self.sig_frozen[eid] = (state == "mature")
        self.samples_seen[eid] = 0
        self.loss_hist[eid] = []
        self._update_counts[eid] = 0
        if state == "infant":
            self.optimizers[eid] = torch.optim.Adam(rep.parameters(), lr=lr)
        return eid

    def mark_mature(self, eid: int) -> None:
        if str(eid) in self.experts:
            self.experts[str(eid)].freeze()
            self.states[eid] = "mature"
            self.sig_frozen[eid] = True
            if self.signatures[eid].n > 0 and (self.signatures[eid].protos is None or len(self.signatures[eid].protos) == 0):
                self.signatures[eid].fit()
            self.optimizers.pop(eid, None)

    def route(self, h: Tensor, update_signatures: bool = False, fit_every: int = 5) -> Tuple[Tensor, List[int]]:
        """Route representation batch h to experts; returns (transformed_h, routed_eids)."""
        b = h.shape[0]
        eids = [int(k) for k in self.experts.keys()]
        if len(eids) == 1:
            eid = eids[0]
            routed_eids = [eid] * b
            out = self.experts[str(eid)](h)
        else:
            # Check which experts are scorable (have fitted prototypes)
            scorable_eids = [
                eid for eid in eids
                if self.signatures[eid].n > 0 and self.signatures[eid].protos is not None and len(self.signatures[eid].protos) > 0
            ]
            infant_unscored = [eid for eid in eids if self.states[eid] == "infant" and eid not in scorable_eids]

            if not scorable_eids:
                # No expert has fitted prototypes: route to the first infant (or first expert) to initialize
                target_eid = infant_unscored[0] if infant_unscored else eids[0]
                routed_eids = [target_eid] * b
            elif infant_unscored and update_signatures:
                # Give an unscored infant priority on incoming samples to bootstrap its prototypes
                target_eid = infant_unscored[0]
                routed_eids = [target_eid] * b
            else:
                # Multi-expert routing via signatures
                scores = []
                for eid in scorable_eids:
                    scores.append(self.signatures[eid].logpdf(h))
                scores_mat = torch.stack(scores, dim=1)  # (B, len(scorable_eids))
                best_idx = torch.argmax(scores_mat, dim=1).cpu().tolist()
                routed_eids = [scorable_eids[idx] for idx in best_idx]

            out = torch.zeros_like(h)
            for eid in eids:
                mask = [i for i, r in enumerate(routed_eids) if r == eid]
                if mask:
                    idx_t = torch.tensor(mask, dtype=torch.long, device=h.device)
                    out[idx_t] = self.experts[str(eid)](h[idx_t])

        # Update signatures if requested
        if update_signatures:
            for eid in eids:
                mask = [i for i, r in enumerate(routed_eids) if r == eid]
                if mask and not self.sig_frozen[eid]:
                    sub_h = h[mask].detach()
                    self.signatures[eid].update(sub_h)
                    self._update_counts[eid] += 1
                    if self.signatures[eid].n >= 8 and (
                        self._update_counts[eid] % fit_every == 0
                        or self.signatures[eid].protos is None
                        or len(self.signatures[eid].protos) == 0
                    ):
                        self.signatures[eid].fit()

        return out, routed_eids


# --------------------------------------------------------------------------- StackMoE
class StackMoE(nn.Module):
    """StackMoE: vertical representation layers with head ContinualPool."""

    def __init__(
        self,
        d: int = 128,
        head_pool: Optional[ContinualPool] = None,
        hidden: int = 128,
        seed: int = 0,
        min_infant_samples: int = 64,
        max_infant_samples: int = 300,
        plateau_window: int = 5,
        plateau_tol: float = 0.02,
    ) -> None:
        super().__init__()
        self.name = "StackMoE"
        self.d = d
        self.hidden = hidden
        self.seed = seed
        self.layers = nn.ModuleList()
        self.head_pool = head_pool
        self.events = EventLog()
        self.step = 0
        self.min_infant_samples = min_infant_samples
        self.max_infant_samples = max_infant_samples
        self.plateau_window = plateau_window
        self.plateau_tol = plateau_tol
        self.growth_policy = None

    def add_layer(self, position: int = -1) -> int:
        """Add an identity-initialized representation layer at specified position."""
        layer_idx = len(self.layers) if position == -1 or position >= len(self.layers) else position
        new_layer = RepLayer(
            layer_id=len(self.layers),
            d=self.d,
            hidden=self.hidden,
            seed=self.seed * 100 + len(self.layers),
        )
        if position == -1 or position >= len(self.layers):
            self.layers.append(new_layer)
            pos_idx = len(self.layers) - 1
        else:
            self.layers.insert(position, new_layer)
            pos_idx = position

        self.events.add(kind="add_layer", step=self.step, position=pos_idx)
        return pos_idx

    def mark_mature(self, layer_idx: int, expert_id: int) -> None:
        if 0 <= layer_idx < len(self.layers):
            self.layers[layer_idx].mark_mature(expert_id)

    def spawn_infant(self, layer_idx: int, seed: Optional[int] = None, lr: float = 3e-3) -> int:
        if 0 <= layer_idx < len(self.layers):
            eid = self.layers[layer_idx].spawn_expert(state="infant", seed=seed, lr=lr)
            self.events.add(kind="spawn_rep", step=self.step, layer=layer_idx, expert=eid)
            return eid
        raise IndexError(f"Layer index {layer_idx} out of range")

    def _normalize(self, x: Tensor) -> Tensor:
        norm = torch.clamp(x.norm(dim=-1, keepdim=True), min=1e-12)
        return x / norm

    def forward(self, x: Tensor) -> Tensor:
        x = self._normalize(x.float())
        h, _ = self._forward_layers(x, update_signatures=False)
        if self.head_pool is not None and hasattr(self.head_pool, "experts") and len(self.head_pool.experts) > 0:
            pred = self.head_pool.predict(h)
            b = x.shape[0]
            n_cls = getattr(self.head_pool.cfg, "n_classes", 10) if hasattr(self.head_pool, "cfg") else 10
            logits = torch.zeros(b, n_cls, device=x.device)
            for eid, rec in self.head_pool.experts.items():
                mask = (pred.expert == eid).nonzero(as_tuple=True)[0]
                if len(mask) > 0:
                    logits[mask] = rec.module(h[mask])
            return logits
        elif hasattr(self.head_pool, "head"):
            return self.head_pool.head(h)
        return h

    def clone(self) -> "StackMoE":
        return copy.deepcopy(self)

    @torch.no_grad()
    def widen(self, k: int = 16, rng: Optional[np.random.Generator] = None) -> "StackMoE":
        if len(self.layers) > 0:
            for eid, exp in self.layers[-1].experts.items():
                exp.widen(k, rng=rng)
        return self

    @torch.no_grad()
    def deepen(self, rng: Optional[np.random.Generator] = None) -> "StackMoE":
        if len(self.layers) > 0:
            for eid, exp in self.layers[-1].experts.items():
                exp.deepen(rng=rng)
        return self

    def train_infant_step(
        self,
        layer_idx: int,
        infant_eid: int,
        x: Tensor,
        y: Tensor,
        lr: float = 3e-3,
    ) -> float:
        """Train an infant representation expert while keeping mature experts frozen."""
        layer = self.layers[layer_idx]
        expert = layer.experts[str(infant_eid)]
        opt = layer.optimizers.get(infant_eid)
        if opt is None:
            opt = torch.optim.Adam(expert.parameters(), lr=lr)

        h = self._normalize(x.float())
        for l_i, lyr in enumerate(self.layers):
            if l_i == layer_idx:
                h = expert(h)
            else:
                with torch.no_grad():
                    h, _ = lyr.route(h)

        b = x.shape[0]
        n_cls = getattr(self.head_pool.cfg, "n_classes", 10) if hasattr(self.head_pool, "cfg") else 10
        logits = torch.zeros(b, n_cls, device=x.device)
        if hasattr(self.head_pool, "experts") and len(self.head_pool.experts) > 0:
            pred = self.head_pool.predict(h.detach())
            for eid_h, rec_h in self.head_pool.experts.items():
                mask = (pred.expert == eid_h).nonzero(as_tuple=True)[0]
                if len(mask) > 0:
                    logits[mask] = rec_h.module(h[mask])
        elif hasattr(self.head_pool, "head"):
            logits = self.head_pool.head(h)
        else:
            logits = h

        valid_y = (y >= 0) & (y < n_cls)
        loss = F.cross_entropy(logits[valid_y], y[valid_y].long())
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.item())

    def record_growth_event(
        self,
        kind: str,
        gain: float = 0.0,
        p: float = 1.0,
        reason: str = "committed",
        increment: int = 0,
        params_added: int = 0,
    ) -> None:
        self.events.add(
            "grow",
            self.step,
            kind=kind,
            gain=gain,
            p=p,
            reason=reason,
            increment=increment,
            params_added=params_added,
        )

    def _forward_layers(self, x: Tensor, update_signatures: bool = False) -> Tuple[Tensor, List[List[Tuple[int, int]]]]:
        """Pass through representation layers, tracking the exact path taken by each sample."""
        h = x
        b = x.shape[0]
        sample_paths: List[List[Tuple[int, int]]] = [[] for _ in range(b)]

        for l_idx, lyr in enumerate(self.layers):
            h, routed_e = lyr.route(h, update_signatures=update_signatures)
            for i in range(b):
                sample_paths[i].append((l_idx, routed_e[i]))

        return h, sample_paths

    def predict(self, x: Tensor, ctx: Optional[Tensor] = None) -> Prediction:
        """End-to-end prediction with consistent path tracking."""
        x = self._normalize(x.float())
        h, sample_paths = self._forward_layers(x, update_signatures=False)
        if self.head_pool is not None:
            pred = self.head_pool.predict(h, ctx=ctx)
        else:
            b = x.shape[0]
            pred = Prediction(
                probs=torch.full((b, 10), 0.1),
                expert=torch.zeros(b, dtype=torch.long),
                p_nov=torch.zeros(b),
                unknown=torch.zeros(b, dtype=torch.bool),
            )
        pred.meta["paths"] = sample_paths
        return pred

    def observe(self, x: Tensor, y: Optional[Tensor] = None, ctx: Optional[Tensor] = None) -> StepRecord:
        """End-to-end observe with signature learning, infant gradient backprop, and promotion."""
        x = self._normalize(x.float())
        self.step += 1
        b = x.shape[0]

        # 1. Forward through representation layers with signature learning
        h, sample_paths = self._forward_layers(x, update_signatures=True)

        # 2. Head pool trains itself in its own observe
        if self.head_pool is not None:
            rec = self.head_pool.observe(h.detach(), y=y, ctx=ctx)
        else:
            rec = StepRecord(
                novel=torch.zeros(b, dtype=torch.bool),
                p_nov=torch.ones(b),
                assigned=torch.zeros(b, dtype=torch.long),
                loss=0.0,
            )

        # 3. Backpropagate head cross-entropy through routed head experts down to infant representation layers
        step_loss = 0.0
        has_infants = any(len(lyr.optimizers) > 0 for lyr in self.layers)
        if y is not None and has_infants and self.head_pool is not None:
            # Routing decisions detached
            pred_head = self.head_pool.predict(h.detach(), ctx=ctx)
            routed_head = pred_head.expert

            n_cls = getattr(self.head_pool.cfg, "n_classes", 10) if hasattr(self.head_pool, "cfg") else 10
            has_head_experts = hasattr(self.head_pool, "experts") and len(self.head_pool.experts) > 0

            if has_head_experts:
                logits = torch.zeros(b, n_cls, device=x.device)
                head_saved_grad = []
                for p in self.head_pool.parameters() if isinstance(self.head_pool, nn.Module) else []:
                    head_saved_grad.append((p, p.requires_grad))
                    p.requires_grad = False

                try:
                    for eid_h, rec_h in self.head_pool.experts.items():
                        mask_h = (routed_head == eid_h).nonzero(as_tuple=True)[0]
                        if len(mask_h) > 0:
                            logits[mask_h] = rec_h.module(h[mask_h])

                    valid_y = (y >= 0) & (y < n_cls)
                    if valid_y.any():
                        loss = F.cross_entropy(logits[valid_y], y[valid_y].long())
                        step_loss = float(loss.item())

                        for lyr in self.layers:
                            for opt in lyr.optimizers.values():
                                opt.zero_grad()

                        loss.backward()

                        for lyr in self.layers:
                            for opt in lyr.optimizers.values():
                                opt.step()
                                opt.zero_grad()
                finally:
                    for p, flag in head_saved_grad:
                        p.requires_grad = flag

        # 4. Check infant -> mature promotion rule
        for l_idx, lyr in enumerate(self.layers):
            for eid in list(lyr.optimizers.keys()):
                count_routed = sum(
                    1 for p in sample_paths
                    if any(path_layer == l_idx and path_eid == eid for path_layer, path_eid in p)
                )
                lyr.samples_seen[eid] += count_routed
                if count_routed > 0 and step_loss > 0.0:
                    lyr.loss_hist[eid].append(step_loss)

                n_seen = lyr.samples_seen[eid]
                hist = lyr.loss_hist[eid]
                should_promote = False
                if n_seen >= self.min_infant_samples:
                    if n_seen >= self.max_infant_samples:
                        should_promote = True
                    elif len(hist) >= self.plateau_window:
                        w_start = hist[-self.plateau_window]
                        w_end = hist[-1]
                        rel_drop = (w_start - w_end) / max(abs(w_start), 1e-6)
                        if rel_drop < self.plateau_tol:
                            should_promote = True

                if should_promote:
                    lyr.mark_mature(eid)
                    self.events.add(
                        kind="promote_rep",
                        step=self.step,
                        layer=l_idx,
                        expert=eid,
                        samples=n_seen,
                        loss=hist[-1] if hist else 0.0,
                    )

        rec.meta["paths"] = sample_paths
        return rec

    def summary(self) -> dict:
        return {
            "n_layers": len(self.layers),
            "n_rep_experts": sum(len(l.experts) for l in self.layers),
            "head_summary": self.head_pool.summary() if self.head_pool is not None else {},
            "n_events": len(self.events),
        }


# --------------------------------------------------------------------------- StackGrowthPolicy
class StackGrowthPolicy(GrowthPolicy):
    """GrowthPolicy generalised to propose {"widen", "deepen", "add_layer"}."""

    def __init__(self, cfg: Optional[GrowthCfg] = None, enable_add_layer: bool = True) -> None:
        super().__init__(cfg)
        self.enable_add_layer = enable_add_layer

    def step(
        self,
        expert_or_stack: Union[MLPExpert, StackMoE],
        x_tr: Tensor,
        y_tr: Tensor,
        x_va: Tensor,
        y_va: Tensor,
        rng: np.random.Generator,
    ) -> GrowthDecision:
        cfg = self.cfg
        if self.cooldown > 0:
            self.cooldown -= 1
            return GrowthDecision(reason="backoff")

        y_tr, y_va = y_tr.long(), y_va.long()
        if hasattr(expert_or_stack, "seen"):
            seen = expert_or_stack.seen.clone()
            seen[y_tr] = True
            if not bool(seen[y_va].all()):
                raise ValueError("validation labels include classes the expert has not seen")

        seeds = rng.integers(0, 2**31 - 1, size=5)
        proposals = {}

        # Widen proposals
        hidden_val = expert_or_stack.hidden if hasattr(expert_or_stack, "hidden") else 128
        for k in self.widen_increments(hidden_val):
            proposals[f"widen+{k}"] = (
                lambda m, k=k, s=int(seeds[1]) + k: (
                    m.widen(k, np.random.default_rng(s)) if hasattr(m, "widen") else m
                ),
                k,
            )

        # Deepen proposal
        depth_val = (
            len(expert_or_stack.blocks)
            if hasattr(expert_or_stack, "blocks")
            else (len(expert_or_stack.layers) if hasattr(expert_or_stack, "layers") else 1)
        )
        if depth_val < cfg.max_depth:
            proposals["deepen"] = (
                lambda m, s=int(seeds[2]): (
                    m.deepen(np.random.default_rng(s)) if hasattr(m, "deepen") else m
                ),
                0,
            )

        # Add-layer proposal (for StackMoE or as depth expansion)
        if self.enable_add_layer and depth_val < cfg.max_depth:
            if hasattr(expert_or_stack, "add_layer"):
                proposals["add_layer"] = (lambda m: m.add_layer(), 0)
            else:
                # On MLPExpert, add_layer deepens with identity block
                proposals["add_layer"] = (
                    lambda m, s=int(seeds[4]): (
                        m.deepen(np.random.default_rng(s)) if hasattr(m, "deepen") else m
                    ),
                    0,
                )

        if not proposals:
            return GrowthDecision(reason="at capacity cap")

        data_rng = np.random.default_rng(int(seeds[0]))
        batches = [
            torch.from_numpy(data_rng.integers(0, x_tr.shape[0], size=cfg.batch_size))
            for _ in range(cfg.train_steps)
        ]

        def probe(model) -> np.ndarray:
            opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            for idx in batches:
                if hasattr(model, "train_step"):
                    model.train_step(x_tr[idx], y_tr[idx], opt)
                else:
                    opt.zero_grad()
                    out = model(x_tr[idx])
                    logits = out.probs if hasattr(out, "probs") else out
                    loss = F.cross_entropy(logits, y_tr[idx])
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                out = model(x_va)
                logits = out.probs if hasattr(out, "probs") else out
                return F.cross_entropy(logits, y_va, reduction="none").cpu().numpy().astype(np.float64)

        expert_or_stack.eval()
        with torch.no_grad():
            out_curr = expert_or_stack(x_va)
            logits_curr = out_curr.probs if hasattr(out_curr, "probs") else out_curr
            status = F.cross_entropy(logits_curr, y_va, reduction="none").cpu().numpy().astype(np.float64)

        control = probe(expert_or_stack.clone() if hasattr(expert_or_stack, "clone") else copy.deepcopy(expert_or_stack))
        base_params = sum(p.numel() for p in expert_or_stack.parameters())
        scored = {}

        for name, (grow_fn, _) in proposals.items():
            shadow = (
                expert_or_stack.clone()
                if hasattr(expert_or_stack, "clone")
                else copy.deepcopy(expert_or_stack)
            )
            grow_fn(shadow)
            after = probe(shadow)
            delta = control - after
            gain = float(delta.mean() / max(float(control.mean()), 1e-12))
            vs_status = float((status.mean() - after.mean()) / max(float(status.mean()), 1e-12))
            added_p = sum(p.numel() for p in shadow.parameters()) - base_params
            scored[name] = (
                gain,
                self._paired_bootstrap_p(delta, np.random.default_rng(int(seeds[3]))),
                added_p,
                vs_status,
            )

        tried = {
            n: {"gain": g, "p": p, "params_added": a, "vs_status": v}
            for n, (g, p, a, v) in scored.items()
        }

        pick = self.select(scored, cfg.min_gain, cfg.alpha / len(scored))
        if pick is None:
            self.cooldown = self.backoff
            self.backoff = min(2 * self.backoff, cfg.max_backoff)
            return GrowthDecision(
                gain=max(v[0] for v in scored.values()),
                p_value=min(v[1] for v in scored.values()),
                reason="rejected",
                tried=tried,
            )

        grow_fn, units = proposals[pick]
        before_p = sum(p.numel() for p in expert_or_stack.parameters())
        grow_fn(expert_or_stack)
        self.backoff, self.cooldown = cfg.backoff0, 0

        decision_kind = "add_layer" if pick == "add_layer" else ("deepen" if pick == "deepen" else "widen")
        return GrowthDecision(
            kind=decision_kind,
            gain=scored[pick][0],
            params_added=sum(p.numel() for p in expert_or_stack.parameters()) - before_p,
            p_value=scored[pick][1],
            increment=units,
            reason="committed",
            tried=tried,
        )
