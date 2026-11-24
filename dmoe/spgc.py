"""
Selective Precision-Gated Consolidation (SPGC):
Unified Brain-Inspired Continual & Efficient Training Algorithm for Dynamic MoE Transformers.

Synthesizes four mutually reinforcing mechanisms:
  1. Matrix-Geometry Fast Acquisition (`zeropower_via_newtonschulz5` / Muon on 2D hidden
     operators + decoupled vector AdamW on embeddings/norms/heads + optional Grokfast-EMA)
     for rapid compositional representation grokking at low spectral norm.
  2. Two-Stage Selective Compaction (`no_grad` screen -> drop familiar & quarantined items ->
     autograd forward/backward strictly on compact active surprise sub-batch + balanced replay).
  3. Latent-Keyed Multi-Source / Temporal Tentative Quarantine Gate (`LatentQuarantineGate`):
     separates inference completely from episodic memory (zero logit distortion / zero NLL harm)
     while gating weight writes against single-source adversarial bursts within <= 16 KB.
  4. Vectorized `torch._foreach_` Diagonal BONG Natural Gradient Consolidation (`MuonBONGOptimizer`):
     true additive Fisher information (Lambda_t = gamma * Lambda_{t-1} + g^2) and natural-gradient
     step (g / (Lambda_t + damping)) preserving consolidated trunk geometry while keeping
     fast adapters, context projections, readout heads, and unfrozen MoE experts plastic.
"""

from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Compute approximate polar factor U V^T of a 2D matrix G via quintic Newton-Schulz iteration
    (Jordan et al., 2024; Muon optimizer).
    """
    if G.ndim != 2 or min(G.shape) == 0:
        return G
    norm = G.norm()
    a, b, c = 3.4445, -4.7750, 2.0315
    orig_dtype = G.dtype
    use_bf16 = G.is_cuda and max(G.shape) >= 64 and torch.cuda.is_bf16_supported()
    if G.is_cuda:
        X = G.to(dtype=(torch.bfloat16 if use_bf16 else torch.float32))
        transposed = X.size(-2) > X.size(-1)
        if transposed:
            X = X.mT
        X = X / (X.norm() + 1e-7)
        for _ in range(int(steps)):
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if transposed:
            X = X.mT
        return torch.where(torch.isfinite(norm) & (norm > 1e-12), X.to(dtype=orig_dtype), G)

    if not torch.isfinite(norm) or float(norm.item()) < 1e-12:
        return G
    X = G.to(dtype=torch.float32)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm() + 1e-7)
    for _ in range(int(steps)):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(dtype=orig_dtype)


@torch.no_grad()
def zeropower_via_newtonschulz5_batched(G_list: List[torch.Tensor], steps: int = 5) -> List[torch.Tensor]:
    """
    Batched Newton-Schulz5 grouping equal-shape 2D matrices via `torch.bmm` (~15x faster on small CUDA layers).
    """
    if not G_list:
        return []
    groups: Dict[Tuple[Tuple[int, ...], torch.dtype, torch.device], List[int]] = {}
    for idx, g in enumerate(G_list):
        if g.ndim != 2 or min(g.shape) == 0:
            key = ((), g.dtype, g.device)
        else:
            key = (tuple(g.shape), g.dtype, g.device)
        groups.setdefault(key, []).append(idx)

    out: List[Optional[torch.Tensor]] = [None] * len(G_list)
    a, b, c = 3.4445, -4.7750, 2.0315
    for (shape, orig_dtype, dev), idxs in groups.items():
        if not shape or len(idxs) == 1:
            for idx in idxs:
                out[idx] = zeropower_via_newtonschulz5(G_list[idx], steps=steps)
            continue
        stacked = torch.stack([G_list[idx] for idx in idxs], dim=0)
        use_bf16 = dev.type == "cuda" and max(shape) >= 64 and torch.cuda.is_bf16_supported()
        X = stacked.to(dtype=(torch.bfloat16 if use_bf16 else torch.float32))
        transposed = X.size(-2) > X.size(-1)
        if transposed:
            X = X.mT
        norms = X.norm(dim=(-2, -1), keepdim=True)
        safe = norms > 1e-12
        X = torch.where(safe, X / (norms + 1e-7), X)
        for _ in range(int(steps)):
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if transposed:
            X = X.mT
        X = torch.where(safe, X.to(dtype=orig_dtype), stacked)
        for k, idx in enumerate(idxs):
            out[idx] = X[k]
    return [t for t in out if t is not None]


def classify_spgc_param_groups(
    model: nn.Module,
    fast_module_names: Sequence[str] = ("ctx_proj", "fast_adapter", "attr_head", "prop_head", "val_head", "out_head"),
    embed_module_names: Sequence[str] = ("tok_emb", "byte_embed", "pos_embed"),
    wd_trunk: float = 0.22,
    wd_fast: float = 0.01,
    lr_muon: float = 1.5e-2,
    lr_adam: float = 6e-3,
    lr_fast: float = 8e-3,
    non_muon_module_names: Sequence[str] = ("gate", "router", "salience"),
    min_muon_dim: int = 4,
) -> List[Dict[str, Any]]:
    """
    Partition trainable model parameters into mathematically aligned SPGC parameter groups:
      - `trunk_muon`: real-valued 2D hidden linear/attention weight matrices with `min(shape) >= min_muon_dim`
        (Muon NS5 in acquire, BONG in consolidate)
      - `trunk_vec`: embeddings, biases, LayerNorm weights, complex tensors, thin gates/routers
        (AdamW in acquire, protected BONG in consolidate)
      - `fast_head`: context projections, fast adapters, experts/readout heads (low prec floor, high plasticity)
    """
    trunk_muon: List[nn.Parameter] = []
    trunk_vec: List[nn.Parameter] = []
    fast_head: List[nn.Parameter] = []
    seen_ids = set()

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen_ids:
            continue
        seen_ids.add(id(p))
        lname = name.lower()
        is_fast = any(tok in lname for tok in fast_module_names)
        is_embed_or_head = (
            any(tok in lname for tok in embed_module_names)
            or any(tok in lname for tok in non_muon_module_names)
            or "decoder" in lname
            or "head" in lname
            or "norm" in lname
        )
        is_muon_eligible = (
            p.ndim == 2
            and (not p.is_complex())
            and min(p.shape) >= int(min_muon_dim)
            and (not is_embed_or_head)
        )
        if is_fast:
            fast_head.append(p)
        elif is_muon_eligible:
            trunk_muon.append(p)
        else:
            trunk_vec.append(p)

    groups: List[Dict[str, Any]] = []
    if trunk_muon:
        groups.append(
            {
                "name": "trunk_muon",
                "params": trunk_muon,
                "use_muon": True,
                "role": "trunk",
                "lr": float(lr_muon),
                "lr_consolidate": float(lr_fast),
                "weight_decay": float(wd_trunk),
                "prec_scale": 2.0,
            }
        )
    if trunk_vec:
        groups.append(
            {
                "name": "trunk_vec",
                "params": trunk_vec,
                "use_muon": False,
                "role": "trunk",
                "lr": float(lr_adam),
                "lr_consolidate": float(lr_fast),
                "weight_decay": float(wd_trunk),
                "prec_scale": 2.0,
            }
        )
    if fast_head:
        groups.append(
            {
                "name": "fast_head",
                "params": fast_head,
                "use_muon": False,
                "role": "fast",
                "lr": float(lr_adam),
                "lr_consolidate": float(lr_fast),
                "weight_decay": float(wd_fast),
                "prec_scale": 0.15,
            }
        )
    return groups


class MuonBONGOptimizer:
    """
    Vectorized dual-regime optimizer combining:
      - Phase `acquire`: Muon (Newton-Schulz5) on 2D hidden matrices + vectorized AdamW on 1D/embeddings/heads
        + optional Grokfast-EMA gradient amplification + diagonal empirical Fisher tracking.
      - Phase `consolidate`: Vectorized `torch._foreach_` true Diagonal BONG step:
        `Lambda_t = gamma * Lambda_{t-1} + g_t^2`, `theta <- theta - lr * g / (Lambda_t + damping)`
        with optional proximal anchor pull on protected trunk weights.
    """

    def __init__(
        self,
        param_groups: List[Dict[str, Any]],
        *,
        muon_momentum: float = 0.90,
        ns_steps: int = 5,
        grokfast_alpha: float = 0.0,
        grokfast_lamb: float = 0.0,
        adam_betas: Tuple[float, float] = (0.9, 0.98),
        adam_eps: float = 1e-8,
        bong_gamma: float = 0.996,
        bong_damping: float = 1e-2,
        bong_init_prec: float = 2e-2,
        bong_min_prec: float = 5e-3,
        bong_max_prec: float = 100.0,
        bong_prior_pull: float = 0.0,
        grad_clip: float = 2.0,
        mode: str = "acquire",
        track_fisher: bool = True,
        disable_muon: bool = False,
        disable_bong: bool = False,
    ):
        self.param_groups = [dict(g) for g in param_groups if len(g.get("params", [])) > 0]
        self.muon_momentum = float(muon_momentum)
        self.ns_steps = int(ns_steps)
        self.grokfast_alpha = float(grokfast_alpha)
        self.grokfast_lamb = float(grokfast_lamb)
        self.beta1, self.beta2 = float(adam_betas[0]), float(adam_betas[1])
        self.adam_eps = float(adam_eps)
        self.bong_gamma = float(bong_gamma)
        self.bong_damping = float(bong_damping)
        self.bong_init_prec = float(bong_init_prec)
        self.bong_min_prec = float(bong_min_prec)
        self.bong_max_prec = float(bong_max_prec)
        self.bong_prior_pull = float(bong_prior_pull)
        self.grad_clip = float(grad_clip)
        self.mode = str(mode)
        self.track_fisher = bool(track_fisher)
        self.disable_muon = bool(disable_muon)
        self.disable_bong = bool(disable_bong)
        self.step_count = 0

        self.exp_avg: Dict[int, torch.Tensor] = {}
        self.exp_avg_sq: Dict[int, torch.Tensor] = {}
        self.grok_ema: Dict[int, torch.Tensor] = {}
        self.precision: Dict[int, torch.Tensor] = {}
        self.anchor: Dict[int, torch.Tensor] = {}
        self._init_missing_state()

    def _init_missing_state(self) -> None:
        need_acquire_state = (self.mode == "acquire") or self.disable_bong
        for g in self.param_groups:
            p_scale = float(g.get("prec_scale", 1.0))
            init_p = max(self.bong_min_prec * p_scale, self.bong_init_prec * p_scale)
            for p in g["params"]:
                pid = id(p)
                real_dtype = p.real.dtype if p.is_complex() else p.dtype
                if need_acquire_state and pid not in self.exp_avg:
                    self.exp_avg[pid] = torch.zeros_like(p.data)
                    self.exp_avg_sq[pid] = torch.zeros(p.shape, dtype=real_dtype, device=p.device)
                    if self.grokfast_alpha > 0.0 and self.grokfast_lamb > 0.0:
                        self.grok_ema[pid] = torch.zeros_like(p.data)
                if (not self.disable_bong) and (pid not in self.precision):
                    self.precision[pid] = torch.full(p.shape, init_p, dtype=real_dtype, device=p.device)
                if (not self.disable_bong) and self.bong_prior_pull > 0.0 and (pid not in self.anchor):
                    self.anchor[pid] = p.data.detach().clone()

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    @torch.no_grad()
    def set_group_mode(
        self,
        group_name_or_role: str,
        mode: str,
        snapshot_anchor: bool = True,
    ) -> int:
        """Set `acquire` vs `consolidate` mode selectively for matching parameter groups."""
        mode_s = str(mode)
        matched = 0
        for g in self.param_groups:
            if g.get("name") == group_name_or_role or g.get("role") == group_name_or_role:
                g["mode"] = mode_s
                matched += 1
                if snapshot_anchor and (not self.disable_bong) and self.bong_prior_pull > 0.0:
                    for p in g["params"]:
                        if id(p) in self.anchor:
                            self.anchor[id(p)].copy_(p.data)
        return matched

    @torch.no_grad()
    def set_mode(
        self,
        mode: str,
        snapshot_anchor: bool = True,
        new_weight_decay: Optional[float] = None,
        post_lr_scale: Optional[float] = None,
    ) -> None:
        self.mode = str(mode)
        if self.mode == "consolidate" and (not self.disable_bong):
            # Free acquisition momentum buffers no longer needed during diagonal BONG consolidation
            self.exp_avg.clear()
            self.exp_avg_sq.clear()
            self.grok_ema.clear()
        if snapshot_anchor and not self.disable_bong and self.bong_prior_pull > 0.0:
            for g in self.param_groups:
                for p in g["params"]:
                    if id(p) in self.anchor:
                        self.anchor[id(p)].copy_(p.data)
        if new_weight_decay is not None:
            for g in self.param_groups:
                if g.get("role", "trunk") == "fast":
                    g["weight_decay"] = min(float(new_weight_decay), 0.01)
                else:
                    g["weight_decay"] = float(new_weight_decay)
        if post_lr_scale is not None:
            for g in self.param_groups:
                g["lr_consolidate"] = float(g.get("lr_consolidate", g.get("lr", 4e-3))) * float(post_lr_scale)

    @torch.no_grad()
    def accumulate_fisher_sample(self, grads_scale: float = 1.0) -> None:
        """Accumulate diagonal empirical Fisher from current `.grad` tensors without stepping weights."""
        if self.disable_bong:
            return
        for g in self.param_groups:
            p_scale = float(g.get("prec_scale", 1.0))
            params = [p for p in g["params"] if p.grad is not None and p.requires_grad]
            if not params:
                continue
            grads = [p.grad for p in params]
            precs = [self.precision[id(p)] for p in params]
            torch._foreach_mul_(precs, self.bong_gamma)
            torch._foreach_addcmul_(precs, grads, grads, value=float(grads_scale) * p_scale)
            for pr in precs:
                pr.clamp_(min=self.bong_min_prec * p_scale, max=self.bong_max_prec)

    @torch.no_grad()
    def temper_precision(self, roles: Sequence[str] = ("fast",), factor: float = 0.25) -> None:
        if self.disable_bong:
            return
        role_set = set(roles)
        for g in self.param_groups:
            if g.get("role", "trunk") not in role_set:
                continue
            p_scale = float(g.get("prec_scale", 1.0))
            precs = [self.precision[id(p)] for p in g["params"]]
            if precs:
                torch._foreach_mul_(precs, float(factor))
                for pr, p in zip(precs, g["params"]):
                    pr.clamp_(min=self.bong_min_prec * p_scale)
                    if id(p) in self.anchor:
                        self.anchor[id(p)].copy_(p.data)

    @torch.no_grad()
    def rebuild_preserving_state(self, new_groups: List[Dict[str, Any]]) -> None:
        self.param_groups = [dict(g) for g in new_groups if len(g.get("params", [])) > 0]
        live_ids = {id(p) for g in self.param_groups for p in g["params"]}
        for state_dict in (self.exp_avg, self.exp_avg_sq, self.grok_ema, self.precision, self.anchor):
            for dead_pid in [pid for pid in state_dict if pid not in live_ids]:
                del state_dict[dead_pid]
        self._init_missing_state()

    @torch.no_grad()
    def step(self, fisher_scale: float = 1.0) -> None:
        self.step_count += 1
        all_grads = [
            p.grad
            for g in self.param_groups
            for p in g["params"]
            if p.grad is not None and p.requires_grad
        ]
        if not all_grads:
            return

        if self.grad_clip > 0.0:
            norms = torch._foreach_norm(all_grads)
            total_norm = torch.linalg.vector_norm(torch.stack(norms))
            tn = float(total_norm.item())
            if tn > self.grad_clip:
                torch._foreach_mul_(all_grads, self.grad_clip / (tn + 1e-6))

        t = self.step_count
        bc1 = 1.0 - (self.beta1 ** t)
        bc2 = 1.0 - (self.beta2 ** t)
        use_gf = self.grokfast_alpha > 0.0 and self.grokfast_lamb > 0.0
        f_scale = max(0.0, float(fisher_scale))
        for g in self.param_groups:
            g_mode = str(g.get("mode", self.mode))
            if g_mode == "acquire" or self.disable_bong:
                self._step_acquire_group(
                    g, t=t, bc1=bc1, bc2=bc2, use_gf=use_gf,
                    use_consolidate_lr=(g_mode != "acquire"),
                    fisher_scale=f_scale,
                )
            else:
                self._step_consolidate_group(g, fisher_scale=f_scale)

    def _step_acquire_group(
        self,
        g: Dict[str, Any],
        *,
        t: int,
        bc1: float,
        bc2: float,
        use_gf: bool,
        use_consolidate_lr: bool,
        fisher_scale: float = 1.0,
    ) -> None:
        lr = float(g.get("lr_consolidate", g.get("lr", 4e-3)) if use_consolidate_lr else g.get("lr", 6e-3))
        wd = float(g.get("weight_decay", 0.0))
        use_muon = bool(g.get("use_muon", False)) and (not self.disable_muon) and (not use_consolidate_lr)

        params: List[torch.Tensor] = []
        grads: List[torch.Tensor] = []
        for p in g["params"]:
            if p.grad is not None and p.requires_grad:
                params.append(p)
                grads.append(p.grad)
        if not params:
            return

        if use_gf:
            g_emas = []
            for p in params:
                pid = id(p)
                if pid not in self.grok_ema:
                    self.grok_ema[pid] = torch.zeros_like(p.data)
                g_emas.append(self.grok_ema[pid])
            torch._foreach_lerp_(g_emas, grads, 1.0 - self.grokfast_alpha)
            grads = torch._foreach_add(grads, g_emas, alpha=self.grokfast_lamb)

        if self.track_fisher and (not self.disable_bong) and fisher_scale > 0.0:
            p_scale = float(g.get("prec_scale", 1.0)) * float(fisher_scale)
            g_norms = torch._foreach_norm(grads)
            g_mag = float(torch.linalg.vector_norm(torch.stack(g_norms)).item())
            if g_mag > 1e-3:
                has_cplx = any(p.is_complex() for p in params)
                if not has_cplx:
                    precs = [self.precision[id(p)] for p in params]
                    torch._foreach_mul_(precs, 0.999)
                    torch._foreach_addcmul_(precs, grads, grads, value=p_scale)
                    for pr in precs:
                        pr.clamp_(min=self.bong_min_prec * float(g.get("prec_scale", 1.0)), max=self.bong_max_prec)
                else:
                    for p, gr in zip(params, grads):
                        pr = self.precision[id(p)]
                        pr.mul_(0.999)
                        g2 = (gr.conj() * gr).real if gr.is_complex() else (gr * gr)
                        pr.add_(g2, alpha=p_scale)
                        pr.clamp_(min=self.bong_min_prec * float(g.get("prec_scale", 1.0)), max=self.bong_max_prec)

        if wd > 0.0:
            torch._foreach_mul_(params, 1.0 - lr * wd)

        muon_params: List[torch.Tensor] = []
        muon_grads: List[torch.Tensor] = []
        vec_params: List[torch.Tensor] = []
        vec_grads: List[torch.Tensor] = []
        if use_muon:
            for p, gr in zip(params, grads):
                if p.ndim == 2 and (not p.is_complex()) and min(p.shape) >= 2:
                    muon_params.append(p)
                    muon_grads.append(gr)
                else:
                    vec_params.append(p)
                    vec_grads.append(gr)
        else:
            vec_params = params
            vec_grads = grads

        if muon_params:
            bufs = [self.exp_avg[id(p)] for p in muon_params]
            torch._foreach_mul_(bufs, self.muon_momentum)
            torch._foreach_add_(bufs, muon_grads, alpha=1.0)
            updates = torch._foreach_add(muon_grads, bufs, alpha=self.muon_momentum)
            orthos = zeropower_via_newtonschulz5_batched(updates, steps=self.ns_steps)
            for p, ortho in zip(muon_params, orthos):
                rms_scale = max(1.0, float(p.size(-2)) / max(1.0, float(p.size(-1)))) ** 0.5
                p.data.add_(ortho, alpha=-lr * rms_scale)

        if vec_params:
            has_cplx = any(p.is_complex() for p in vec_params)
            step_size = -lr / bc1
            if not has_cplx:
                ms = [self.exp_avg[id(p)] for p in vec_params]
                vs = [self.exp_avg_sq[id(p)] for p in vec_params]
                torch._foreach_lerp_(ms, vec_grads, 1.0 - self.beta1)
                torch._foreach_mul_(vs, self.beta2)
                torch._foreach_addcmul_(vs, vec_grads, vec_grads, value=1.0 - self.beta2)
                denoms = torch._foreach_sqrt(vs)
                torch._foreach_div_(denoms, math.sqrt(bc2))
                torch._foreach_add_(denoms, self.adam_eps)
                steps = torch._foreach_div(ms, denoms)
                torch._foreach_add_(vec_params, steps, alpha=step_size)
            else:
                for p, gr in zip(vec_params, vec_grads):
                    m_t = self.exp_avg[id(p)]
                    v_t = self.exp_avg_sq[id(p)]
                    m_t.mul_(self.beta1).add_(gr, alpha=1.0 - self.beta1)
                    g2 = (gr.conj() * gr).real if gr.is_complex() else (gr * gr)
                    v_t.mul_(self.beta2).add_(g2, alpha=1.0 - self.beta2)
                    denom = (v_t.sqrt() / math.sqrt(bc2)).add_(self.adam_eps)
                    p.data.add_(m_t / denom, alpha=step_size)

    def _step_consolidate_group(self, g: Dict[str, Any], fisher_scale: float = 1.0) -> None:
        """Vectorized diagonal Bayesian Online Natural Gradient step for one param group."""
        lr = float(g.get("lr_consolidate", g.get("lr", 8e-3)))
        wd = float(g.get("weight_decay", 0.01))
        base_scale = float(g.get("prec_scale", 1.0))
        p_scale = base_scale * max(0.0, float(fisher_scale))
        role = g.get("role", "trunk")

        params: List[torch.Tensor] = []
        grads: List[torch.Tensor] = []
        for p in g["params"]:
            if p.grad is not None and p.requires_grad:
                params.append(p)
                grads.append(p.grad)
        if not params:
            return

        max_p = self.bong_max_prec * (0.20 if role == "fast" else 1.0)
        has_cplx = any(p.is_complex() for p in params)
        if not has_cplx:
            precs = [self.precision[id(p)] for p in params]
            if fisher_scale > 0.0:
                torch._foreach_mul_(precs, self.bong_gamma)
                torch._foreach_addcmul_(precs, grads, grads, value=p_scale)
                for pr in precs:
                    pr.clamp_(min=self.bong_min_prec * base_scale, max=max_p)

            denoms = torch._foreach_add(precs, self.bong_damping)
            if self.bong_prior_pull > 0.0 and role == "trunk":
                anchors = [self.anchor[id(p)] for p in params]
                diffs = torch._foreach_sub(params, anchors)
                eff_grads = torch._foreach_add(grads, diffs, alpha=self.bong_prior_pull)
            else:
                eff_grads = grads

            if wd > 0.0:
                torch._foreach_mul_(params, 1.0 - lr * wd)

            updates = torch._foreach_div(eff_grads, denoms)
            torch._foreach_add_(params, updates, alpha=-lr)
        else:
            for p, gr in zip(params, grads):
                pr = self.precision[id(p)]
                if fisher_scale > 0.0:
                    pr.mul_(self.bong_gamma)
                    g2 = (gr.conj() * gr).real if gr.is_complex() else (gr * gr)
                    pr.add_(g2, alpha=p_scale)
                    pr.clamp_(min=self.bong_min_prec * base_scale, max=max_p)
                eff_g = gr
                if self.bong_prior_pull > 0.0 and role == "trunk" and id(p) in self.anchor:
                    eff_g = gr + self.bong_prior_pull * (p.data - self.anchor[id(p)])
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(eff_g / (pr + self.bong_damping), alpha=-lr)

    def precision_bytes(self) -> int:
        prec_b = sum(t.numel() * t.element_size() for t in self.precision.values())
        anch_b = sum(t.numel() * t.element_size() for t in self.anchor.values())
        return int(prec_b + anch_b)


class LatentQuarantineGate:
    """
    Strictly <= `max_bytes` episodic gate & compact core reservoir:
      - Zero Voronoi logit modification at inference (predictions come purely from neural weights).
      - Keys concepts via frozen orthogonal SimHash of detached causal latent `z_latent` (+ optional `q_tag`).
      - Separates streaming observations into:
          (1) Familiar -> drop backward pass, track confirmation timestamps & reservoir.
          (2) Confident contradiction from unverified/single source -> quarantine in tentative ring
              (skip backward, blocking single-source adversarial burst overwrites).
          (3) Uncertain/novel OR multi-source confirmed revision -> compact active slice + release
              buffered evidence + balanced reservoir draw for a precision-gated step.
    """

    BYTES_PER_TENT = 24
    BYTES_PER_PRIOR = 52  # prior_counts (16B) + up to 2x last_confirm_step (24B) + active_rev_until (12B)

    def __init__(
        self,
        latent_dim: int,
        seed: int = 0,
        max_bytes: int = 16384,
        n_bits: int = 10,
        conf_established: float = 0.48,
        tau_skip: float = 0.12,
        min_flip_count: int = 6,
        flip_ratio: float = 0.35,
        min_sources: int = 2,
        rev_grace_steps: int = 24,
        bytes_per_sample: int = 32,
        store_samples: bool = True,
        max_queue_samples: Optional[int] = None,
        tent_ttl_steps: int = 128,
    ):
        self.latent_dim = int(latent_dim)
        self.max_bytes = int(max_bytes)
        self.n_bits = int(n_bits)
        self.conf_established = float(conf_established)
        self.tau_skip = float(tau_skip)
        self.min_flip_count = int(min_flip_count)
        self.flip_ratio = float(flip_ratio)
        self.min_sources = int(min_sources)
        self.rev_grace_steps = int(rev_grace_steps)
        self.bytes_per_sample = int(bytes_per_sample)
        self.store_samples = bool(store_samples)
        self.tent_ttl_steps = int(tent_ttl_steps)

        if self.store_samples:
            # Allocate strictly within `max_bytes`:
            self.max_tent_entries = max(8, int(0.12 * max_bytes) // self.BYTES_PER_TENT)
            self.max_prior_entries = max(16, int(0.21 * max_bytes) // self.BYTES_PER_PRIOR)
            self.max_tent_samples = max(8, int(0.18 * max_bytes) // max(1, self.bytes_per_sample))
            self.max_queue_samples = (
                int(max_queue_samples)
                if max_queue_samples is not None
                else max(8, int(0.08 * max_bytes) // max(1, self.bytes_per_sample))
            )
            meta_reserve = self.max_tent_entries * self.BYTES_PER_TENT + self.max_prior_entries * self.BYTES_PER_PRIOR
            sample_budget = max(0, max_bytes - meta_reserve - (self.max_tent_samples + self.max_queue_samples) * self.bytes_per_sample)
            self.max_core_samples = max(8, sample_budget // max(1, self.bytes_per_sample))
        else:
            q_cap = int(max_queue_samples) if max_queue_samples is not None else 0
            if q_cap > 0 and self.bytes_per_sample <= 0:
                self.bytes_per_sample = 32
            queue_reserve = min(max_bytes // 3, q_cap * max(0, self.bytes_per_sample))
            self.max_queue_samples = queue_reserve // max(1, self.bytes_per_sample) if q_cap > 0 else 0
            avail_meta = max(0, max_bytes - queue_reserve)
            self.max_tent_entries = max(8, int(0.38 * avail_meta) // self.BYTES_PER_TENT)
            self.max_prior_entries = max(16, max(0, avail_meta - self.max_tent_entries * self.BYTES_PER_TENT) // self.BYTES_PER_PRIOR)
            self.max_tent_samples = 0
            self.max_core_samples = 0

        g = torch.Generator().manual_seed(seed * 997 + 13)
        proj = torch.randn(latent_dim, n_bits, generator=g)
        if latent_dim >= n_bits:
            q, _ = torch.linalg.qr(proj)
            proj = q[:, :n_bits]
        self._proj_cpu = proj
        self._proj_dev: Optional[torch.Tensor] = None
        self._bit_weights_dev: Optional[torch.Tensor] = None

        self.prior_counts: OrderedDict[int, int] = OrderedDict()
        self.last_confirm_step: OrderedDict[Tuple[int, int], int] = OrderedDict()
        self.active_rev_until: OrderedDict[int, int] = OrderedDict()
        self.tent_entries: OrderedDict[Tuple[int, int], Dict[str, Any]] = OrderedDict()

        self.queue_x: List[torch.Tensor] = []
        self.queue_y: List[torch.Tensor] = []
        self.queue_m: List[torch.Tensor] = []
        self.core_x: List[torch.Tensor] = []
        self.core_y: List[torch.Tensor] = []
        self.core_sigs: List[int] = []
        self.core_seen = 0
        self.step_idx = 0
        self.rng = torch.Generator().manual_seed(seed * 733 + 5)

        self.total_screened = 0
        self.total_skipped_familiar = 0
        self.total_quarantined = 0
        self.flips_executed = 0

    def _compute_sigs(self, z: torch.Tensor, q_tag: Optional[torch.Tensor] = None) -> List[int]:
        if self._proj_dev is None or self._proj_dev.device != z.device:
            self._proj_dev = self._proj_cpu.to(device=z.device, dtype=torch.float32)
            self._bit_weights_dev = 1 << torch.arange(self.n_bits, device=z.device, dtype=torch.long)
        z_norm = F.normalize(z.detach().float(), dim=-1)
        bits = (z_norm @ self._proj_dev > 0).to(torch.long)
        hashes = (bits * self._bit_weights_dev.unsqueeze(0)).sum(dim=-1)
        if q_tag is not None:
            hashes = (hashes << 6) | (q_tag.to(torch.long) & 0x3F)
        return hashes.cpu().tolist()

    def _add_core(self, x_item: torch.Tensor, y_item: torch.Tensor, sig: int) -> None:
        if self.max_core_samples <= 0 or (not self.store_samples):
            return
        self.core_seen += 1
        xc = x_item.detach().cpu()
        yc = y_item.detach().cpu()
        if len(self.core_x) < self.max_core_samples:
            self.core_x.append(xc)
            self.core_y.append(yc)
            self.core_sigs.append(int(sig))
        else:
            idx = int(torch.randint(0, self.core_seen, (1,), generator=self.rng).item())
            if idx < self.max_core_samples:
                self.core_x[idx] = xc
                self.core_y[idx] = yc
                self.core_sigs[idx] = int(sig)

    def _prune_stale_core_for_flip(self, sig: int, new_y: int) -> None:
        """Prune core entries contradicted by a confirmed flip only when the old value isn't co-confirmed."""
        kept_x, kept_y, kept_s = [], [], []
        for cx, cy, cs in zip(self.core_x, self.core_y, self.core_sigs):
            old_y = int(cy.item()) if cy.ndim == 0 else -1
            if cs == sig and old_y != new_y:
                last_c = self.last_confirm_step.get((sig, old_y), -999)
                if (self.step_idx - last_c) > 5:
                    continue
            kept_x.append(cx)
            kept_y.append(cy)
            kept_s.append(cs)
        self.core_x = kept_x
        self.core_y = kept_y
        self.core_sigs = kept_s

    def _tent_sample_count(self) -> int:
        return sum(len(v["samples"]) for v in self.tent_entries.values())

    def _evict_oldest_tent_sample(self) -> None:
        for _, v in self.tent_entries.items():
            if v["samples"]:
                v["samples"].pop(0)
                return

    def _enforce_budget(self) -> None:
        while len(self.prior_counts) > self.max_prior_entries:
            self.prior_counts.popitem(last=False)
        while len(self.last_confirm_step) > self.max_prior_entries * 2:
            self.last_confirm_step.popitem(last=False)
        while len(self.active_rev_until) > self.max_prior_entries:
            self.active_rev_until.popitem(last=False)
        while len(self.tent_entries) > self.max_tent_entries:
            self.tent_entries.popitem(last=False)
        while self._tent_sample_count() > self.max_tent_samples:
            self._evict_oldest_tent_sample()
        while len(self.core_x) > self.max_core_samples:
            self.core_x.pop(0)
            self.core_y.pop(0)
            self.core_sigs.pop(0)
        while len(self.queue_x) > self.max_queue_samples:
            self.queue_x.pop(0)
            self.queue_y.pop(0)
            if self.queue_m:
                self.queue_m.pop(0)
        while self.storage_bytes() > self.max_bytes:
            if self.core_x:
                self.core_x.pop(0)
                self.core_y.pop(0)
                self.core_sigs.pop(0)
            elif self._tent_sample_count() > 0:
                self._evict_oldest_tent_sample()
            elif self.queue_x:
                self.queue_x.pop(0)
                self.queue_y.pop(0)
                if self.queue_m:
                    self.queue_m.pop(0)
            elif self.tent_entries:
                self.tent_entries.popitem(last=False)
            elif self.last_confirm_step:
                self.last_confirm_step.popitem(last=False)
            elif self.prior_counts:
                self.prior_counts.popitem(last=False)
            else:
                break

    @torch.no_grad()
    def triage_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        logits: torch.Tensor,
        z_latent: torch.Tensor,
        source_id: str = "default",
        q_tag: Optional[torch.Tensor] = None,
        allow_core_write: bool = True,
        skip_familiar: bool = True,
        enable_quarantine: bool = True,
        compact: bool = True,
        return_active_mask: bool = False,
    ) -> Any:
        """
        Screen incoming batch and return `(x_train, y_train, unlocked_flips)` (plus `active_mask [B]` bool
        when `return_active_mask=True`). Uses GPU-side filtering before CPU hash lookups.
        """
        self.step_idx += 1
        B = x.size(0)
        self.total_screened += B
        active_mask_out = torch.zeros(B, dtype=torch.bool, device=x.device)

        probs = F.softmax(logits.detach().float(), dim=-1)
        confs, preds = probs.max(dim=-1)
        ces = F.cross_entropy(logits.detach().float(), y, reduction="none")

        familiar_mask = (preds == y) & (ces <= self.tau_skip) & (confs >= 0.72)
        if skip_familiar and bool(familiar_mask.all().item()):
            self.total_skipped_familiar += B
            if self.step_idx % 2 == 0:
                stride = max(1, B // 16)
                sample_idx = torch.arange(0, B, stride, device=x.device)[:16]
                sigs_f = self._compute_sigs(
                    z_latent[sample_idx],
                    q_tag[sample_idx] if q_tag is not None else None,
                )
                for sig_f in sigs_f:
                    self.prior_counts[sig_f] = min(self.prior_counts.get(sig_f, 0) + 1, 20)
                    self.prior_counts.move_to_end(sig_f)
                if allow_core_write and (self.step_idx % 4 == 0) and sigs_f:
                    self._add_core(x[0], y[0], sigs_f[0])
                self._enforce_budget()
            return (None, None, 0, active_mask_out) if return_active_mask else (None, None, 0)

        # Identify which indices actually need Python triage when skip_familiar=True
        if skip_familiar:
            active_idx = (~familiar_mask).nonzero(as_tuple=True)[0]
            n_fam = int(B - active_idx.numel())
            self.total_skipped_familiar += n_fam
            if n_fam > 0 and (self.step_idx % 2 == 0):
                fam_all = familiar_mask.nonzero(as_tuple=True)[0]
                stride = max(1, int(fam_all.numel()) // 16)
                fam_sel = fam_all[::stride][:16]
                sigs_f = self._compute_sigs(
                    z_latent[fam_sel],
                    q_tag[fam_sel] if q_tag is not None else None,
                )
                for sig_f in sigs_f:
                    self.prior_counts[sig_f] = min(self.prior_counts.get(sig_f, 0) + 1, 20)
                    self.prior_counts.move_to_end(sig_f)
                if allow_core_write and (self.step_idx % 4 == 0) and sigs_f:
                    fam_i = int(fam_sel[0].item())
                    self._add_core(x[fam_i], y[fam_i], sigs_f[0])
            idx_iter = active_idx.cpu().tolist()
        else:
            idx_iter = list(range(B))

        if not idx_iter:
            self._enforce_budget()
            return (None, None, 0, active_mask_out) if return_active_mask else (None, None, 0)

        idx_t_sel = torch.tensor(idx_iter, dtype=torch.long, device=x.device)
        sigs_sel = self._compute_sigs(
            z_latent[idx_t_sel],
            q_tag[idx_t_sel] if q_tag is not None else None,
        )
        confs_l = confs[idx_t_sel].cpu().tolist()
        preds_l = preds[idx_t_sel].cpu().tolist()
        ces_l = ces[idx_t_sel].cpu().tolist()
        y_l = y[idx_t_sel].detach().cpu().tolist()

        keep_indices: List[int] = []
        flush_x: List[torch.Tensor] = []
        flush_y: List[torch.Tensor] = []
        flips_here = 0

        for pos_k, i in enumerate(idx_iter):
            sig = int(sigs_sel[pos_k])
            conf_i = float(confs_l[pos_k])
            pred_i = int(preds_l[pos_k])
            ce_i = float(ces_l[pos_k])
            tgt_i = int(y_l[pos_k])

            # 1. Prediction matches observed truth
            if pred_i == tgt_i:
                lkey = (sig, tgt_i)
                self.last_confirm_step[lkey] = self.step_idx
                self.last_confirm_step.move_to_end(lkey)
                if sig in self.prior_counts:
                    self.prior_counts[sig] = min(self.prior_counts[sig] + 1, 20)
                    self.prior_counts.move_to_end(sig)
                elif conf_i >= self.conf_established:
                    self.prior_counts[sig] = 1

                if allow_core_write and (pos_k % 8 == 0):
                    self._add_core(x[i], y[i], sig)

                if skip_familiar and ce_i <= self.tau_skip and conf_i >= 0.72:
                    self.total_skipped_familiar += 1
                    continue
                keep_indices.append(i)
                continue

            # 2. Contradiction of an established belief (not already inside an active confirmed revision window)
            prior_cnt = self.prior_counts.get(sig, 0)
            in_grace = self.step_idx <= self.active_rev_until.get(sig, -1)
            is_confident_prior = (conf_i >= self.conf_established) or (prior_cnt >= 3)
            if enable_quarantine and is_confident_prior and (not in_grace):
                tkey = (sig, tgt_i)
                rec = self.tent_entries.get(tkey)
                is_stale = (
                    rec is not None
                    and self.tent_ttl_steps > 0
                    and (self.step_idx - int(rec["last_step"])) > self.tent_ttl_steps
                )
                if rec is None or is_stale:
                    rec = {
                        "count": 0,
                        "sources": set(),
                        "first_step": self.step_idx,
                        "last_step": self.step_idx,
                        "samples": [],
                    }
                    self.tent_entries[tkey] = rec
                    self.tent_entries.move_to_end(tkey)
                else:
                    self.tent_entries.move_to_end(tkey)

                rec["count"] += 1
                rec["sources"].add(source_id)
                rec["last_step"] = self.step_idx
                if self.store_samples and len(rec["samples"]) < 6:
                    while self._tent_sample_count() >= self.max_tent_samples:
                        self._evict_oldest_tent_sample()
                    rec["samples"].append((x[i].detach().cpu(), y[i].detach().cpu()))

                req = max(self.min_flip_count, int(math.ceil(min(prior_cnt, 20) * self.flip_ratio)))
                if rec["count"] >= req and len(rec["sources"]) >= self.min_sources:
                    flips_here += 1
                    self.flips_executed += 1
                    self.active_rev_until[sig] = self.step_idx + self.rev_grace_steps
                    self.active_rev_until.move_to_end(sig)
                    self.prior_counts[sig] = 1
                    self.prior_counts.move_to_end(sig)
                    if allow_core_write:
                        self._prune_stale_core_for_flip(sig, tgt_i)
                    for fx, fy in rec["samples"]:
                        flush_x.append(fx)
                        flush_y.append(fy)
                        if allow_core_write:
                            self._add_core(fx, fy, sig)
                    del self.tent_entries[tkey]
                    keep_indices.append(i)
                else:
                    self.total_quarantined += 1
                continue

            # 3. Novel / uncertain / active grace revision -> pass to compact update
            keep_indices.append(i)
            if allow_core_write and (in_grace or conf_i < 0.35) and (pos_k % 8 == 0):
                self._add_core(x[i], y[i], sig)

        self._enforce_budget()
        if keep_indices:
            idx_t = torch.tensor(keep_indices, dtype=torch.long, device=x.device)
            active_mask_out[idx_t] = True

        if not keep_indices and not flush_x:
            return (None, None, flips_here, active_mask_out) if return_active_mask else (None, None, flips_here)

        if not compact:
            return (x, y, flips_here, active_mask_out) if return_active_mask else (x, y, flips_here)

        parts_x: List[torch.Tensor] = []
        parts_y: List[torch.Tensor] = []
        if keep_indices:
            parts_x.append(x[idx_t])
            parts_y.append(y[idx_t])
        if flush_x:
            parts_x.append(torch.stack(flush_x, dim=0).to(device=x.device))
            parts_y.append(torch.stack(flush_y, dim=0).to(device=x.device))

        out_x = torch.cat(parts_x, dim=0)
        out_y = torch.cat(parts_y, dim=0)
        return (out_x, out_y, flips_here, active_mask_out) if return_active_mask else (out_x, out_y, flips_here)

    def sample_core_replay(self, k: int, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.core_x or k <= 0:
            return None, None
        n = min(int(k), len(self.core_x))
        picks = torch.randint(0, len(self.core_x), (n,), generator=self.rng).tolist()
        rx = torch.stack([self.core_x[j] for j in picks], dim=0).to(device=device)
        ry = torch.stack([self.core_y[j] for j in picks], dim=0).to(device=device)
        return rx, ry

    def enqueue_or_flush(
        self,
        x_act: Optional[torch.Tensor],
        y_act: Optional[torch.Tensor],
        min_flush: int = 1,
        force_flush: bool = False,
        device: Optional[torch.device] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Accumulate tiny residual surprise slices until >= `min_flush` surprises are ready,
        avoiding kernel-launch overhead on 1-3 item residuals while counting queued bytes honestly.
        """
        if x_act is not None and y_act is not None and x_act.size(0) > 0:
            if min_flush <= 1 and not self.queue_x:
                return x_act, y_act
            dev = x_act.device
            if self.queue_x and (self.queue_x[0].shape != x_act.shape[1:] or self.queue_y[0].shape != y_act.shape[1:]):
                self.queue_x.clear()
                self.queue_y.clear()
                self.queue_m.clear()
            if (len(self.queue_x) + x_act.size(0) >= min_flush) or force_flush or self.max_queue_samples <= 0:
                if self.queue_x:
                    qx = torch.stack(self.queue_x, dim=0).to(dev)
                    qy = torch.stack(self.queue_y, dim=0).to(dev)
                    self.queue_x.clear()
                    self.queue_y.clear()
                    self.queue_m.clear()
                    return torch.cat([qx, x_act], dim=0), torch.cat([qy, y_act], dim=0)
                return x_act, y_act
            for j in range(x_act.size(0)):
                if len(self.queue_x) < self.max_queue_samples:
                    self.queue_x.append(x_act[j].detach().cpu())
                    self.queue_y.append(y_act[j].detach().cpu())
            self._enforce_budget()
            return None, None

        if force_flush and self.queue_x:
            dev = device if device is not None else self.queue_x[0].device
            qx = torch.stack(self.queue_x, dim=0).to(dev)
            qy = torch.stack(self.queue_y, dim=0).to(dev)
            self.queue_x.clear()
            self.queue_y.clear()
            self.queue_m.clear()
            return qx, qy
        return None, None

    def enqueue_or_flush_masked(
        self,
        x_act: Optional[torch.Tensor],
        y_act: Optional[torch.Tensor],
        m_act: Optional[torch.Tensor],
        min_flush: int = 1,
        force_flush: bool = False,
        device: Optional[torch.device] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Sequence-aware micro-queue matching `enqueue_or_flush`, preserving per-token active masks `m_act [N, T]`.
        """
        if x_act is not None and y_act is not None and m_act is not None and x_act.size(0) > 0:
            if min_flush <= 1 and not self.queue_x:
                return x_act, y_act, m_act
            dev = x_act.device
            same_shape = (
                bool(self.queue_x)
                and bool(self.queue_m)
                and self.queue_x[0].shape == x_act.shape[1:]
                and self.queue_y[0].shape == y_act.shape[1:]
                and self.queue_m[0].shape == m_act.shape[1:]
            )
            if self.queue_x and not same_shape:
                self.queue_x.clear()
                self.queue_y.clear()
                self.queue_m.clear()
            if (len(self.queue_x) + x_act.size(0) >= min_flush) or force_flush or self.max_queue_samples <= 0:
                if self.queue_x and self.queue_m:
                    qx = torch.stack(self.queue_x, dim=0).to(dev)
                    qy = torch.stack(self.queue_y, dim=0).to(dev)
                    qm = torch.stack(self.queue_m, dim=0).to(dev)
                    self.queue_x.clear()
                    self.queue_y.clear()
                    self.queue_m.clear()
                    return (
                        torch.cat([qx, x_act], dim=0),
                        torch.cat([qy, y_act], dim=0),
                        torch.cat([qm, m_act], dim=0),
                    )
                return x_act, y_act, m_act
            for j in range(x_act.size(0)):
                if len(self.queue_x) < self.max_queue_samples:
                    self.queue_x.append(x_act[j].detach().cpu())
                    self.queue_y.append(y_act[j].detach().cpu())
                    self.queue_m.append(m_act[j].detach().cpu())
            self._enforce_budget()
            return None, None, None

        if force_flush and self.queue_x and self.queue_m:
            dev = device if device is not None else self.queue_x[0].device
            qx = torch.stack(self.queue_x, dim=0).to(dev)
            qy = torch.stack(self.queue_y, dim=0).to(dev)
            qm = torch.stack(self.queue_m, dim=0).to(dev)
            self.queue_x.clear()
            self.queue_y.clear()
            self.queue_m.clear()
            return qx, qy, qm
        return None, None, None

    @torch.no_grad()
    def triage_continuous(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pred: torch.Tensor,
        z_latent: torch.Tensor,
        source_id: str = "default",
        skip_familiar: bool = True,
        enable_quarantine: bool = True,
        return_active_mask: bool = False,
    ) -> Any:
        """
        Continuous/regression MSE screening & multi-source contradiction quarantine on latent SimHash buckets.
        """
        self.step_idx += 1
        B = x.size(0)
        self.total_screened += B
        active_mask_out = torch.zeros(B, dtype=torch.bool, device=x.device)

        mses = (pred.detach().float() - y.detach().float()).pow(2).reshape(B, -1).mean(dim=-1)
        familiar_mask = mses <= self.tau_skip
        if skip_familiar and bool(familiar_mask.all().item()):
            self.total_skipped_familiar += B
            return (None, None, 0, active_mask_out) if return_active_mask else (None, None, 0)

        active_idx = (~familiar_mask).nonzero(as_tuple=True)[0] if skip_familiar else torch.arange(B, device=x.device)
        if skip_familiar:
            self.total_skipped_familiar += int(B - active_idx.numel())
        if active_idx.numel() == 0:
            self._enforce_budget()
            return (None, None, 0, active_mask_out) if return_active_mask else (None, None, 0)

        idx_iter = active_idx.cpu().tolist()
        sigs_sel = self._compute_sigs(z_latent[active_idx])
        mses_l = mses[active_idx].cpu().tolist()
        shock_thresh = max(4.0 * self.tau_skip, 0.5)

        keep_indices: List[int] = []
        flips_here = 0
        for pos_k, i in enumerate(idx_iter):
            sig = int(sigs_sel[pos_k])
            mse_i = float(mses_l[pos_k])
            if mse_i <= self.tau_skip:
                self.prior_counts[sig] = min(self.prior_counts.get(sig, 0) + 1, 20)
                self.prior_counts.move_to_end(sig)
                if not skip_familiar:
                    keep_indices.append(i)
                continue

            prior_cnt = self.prior_counts.get(sig, 0)
            in_grace = self.step_idx <= self.active_rev_until.get(sig, -1)
            if enable_quarantine and prior_cnt >= 3 and mse_i >= shock_thresh and (not in_grace):
                tkey = (sig, -1)
                rec = self.tent_entries.get(tkey)
                is_stale = (
                    rec is not None
                    and self.tent_ttl_steps > 0
                    and (self.step_idx - int(rec["last_step"])) > self.tent_ttl_steps
                )
                if rec is None or is_stale:
                    rec = {"count": 0, "sources": set(), "first_step": self.step_idx, "last_step": self.step_idx, "samples": []}
                    self.tent_entries[tkey] = rec
                    self.tent_entries.move_to_end(tkey)
                else:
                    self.tent_entries.move_to_end(tkey)
                rec["count"] += 1
                rec["sources"].add(source_id)
                rec["last_step"] = self.step_idx
                req = max(self.min_flip_count, int(math.ceil(min(prior_cnt, 20) * self.flip_ratio)))
                if rec["count"] >= req and len(rec["sources"]) >= self.min_sources:
                    flips_here += 1
                    self.flips_executed += 1
                    self.active_rev_until[sig] = self.step_idx + self.rev_grace_steps
                    self.active_rev_until.move_to_end(sig)
                    self.prior_counts[sig] = 1
                    del self.tent_entries[tkey]
                    keep_indices.append(i)
                else:
                    self.total_quarantined += 1
                continue

            self.prior_counts[sig] = min(prior_cnt + 1, 20)
            self.prior_counts.move_to_end(sig)
            keep_indices.append(i)

        self._enforce_budget()
        if not keep_indices:
            return (None, None, flips_here, active_mask_out) if return_active_mask else (None, None, flips_here)
        idx_t = torch.tensor(keep_indices, dtype=torch.long, device=x.device)
        active_mask_out[idx_t] = True
        return (x[idx_t], y[idx_t], flips_here, active_mask_out) if return_active_mask else (x[idx_t], y[idx_t], flips_here)

    def storage_bytes(self) -> int:
        tent_meta_b = len(self.tent_entries) * self.BYTES_PER_TENT
        prior_meta_b = (
            len(self.prior_counts) * 16
            + len(self.last_confirm_step) * 12
            + len(self.active_rev_until) * 12
        )
        tent_samp_b = self._tent_sample_count() * self.bytes_per_sample
        queue_samp_b = len(self.queue_x) * self.bytes_per_sample
        core_samp_b = len(self.core_x) * self.bytes_per_sample
        return int(tent_meta_b + prior_meta_b + tent_samp_b + queue_samp_b + core_samp_b)

    def stats(self) -> Dict[str, int]:
        return {
            "screened": int(self.total_screened),
            "skipped_familiar": int(self.total_skipped_familiar),
            "quarantined": int(self.total_quarantined),
            "flips": int(self.flips_executed),
            "bytes": self.storage_bytes(),
        }


def classify_v7_spgc_param_groups(
    model: nn.Module,
    *,
    shared_mode: str = "consolidate",
    specialist_mode: str = "acquire",
    lr_muon: float = 8e-3,
    lr_adam: float = 3e-3,
    lr_consolidate: float = 2e-3,
    wd_trunk: float = 0.02,
    wd_fast: float = 0.01,
    include_decoder: bool = False,
    trunk_prec_scale: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    v7 Lifecycle-Scoped SPGC Partitioning:
      - `specialist_muon` / `specialist_vec`: Plastic active specialists (`.moe.experts.*`,
        `*_adapters.*`, `need_head`, `spec_gate_scale`) run fast `Muon` acquisition even after
        initial shared domains have shifted to diagonal `BONG` consolidation!
      - `shared_trunk_muon` / `shared_trunk_vec`: Shared Layers `0..2`, `SharedFFN`, `W_q`, `W_ctx`,
        `WorldModel`, `Thinker` core, and `DualTimescaleMemory` respect GPM + diagonal `BONG`.
    """
    spec_muon: List[nn.Parameter] = []
    spec_vec: List[nn.Parameter] = []
    fast_proto: List[nn.Parameter] = []
    wm_ret_vec: List[nn.Parameter] = []
    trunk_muon: List[nn.Parameter] = []
    trunk_vec: List[nn.Parameter] = []
    dec_params: List[nn.Parameter] = []
    seen = set()

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        ln = name.lower()
        if "decoder" in ln or "detached_renderer" in ln:
            if include_decoder:
                dec_params.append(p)
            continue
        if "equiv_proto_bank" in ln or "wm_ret_gate" in ln:
            fast_proto.append(p)
            continue
        if ln.startswith("world_model.") or ln.startswith("wm_ret_proj.") or ln.startswith("wm_fut_mlp."):
            wm_ret_vec.append(p)
            continue

        is_specialist = (
            ".moe.experts." in ln
            or "stm_adapters." in ln
            or "attn_adapters." in ln
            or "expert_adapters." in ln
            or "need_head" in ln
            or "spec_gate_scale" in ln
            or "expr_proj." in ln
            or "equiv_proto_bank" in ln
            or "long_term.op_" in ln
        )
        is_vec_only = (
            p.ndim != 2
            or p.is_complex()
            or min(p.shape) < 4
            or "embed" in ln
            or "norm" in ln
            or "gate" in ln
            or "context_router.q_proj" in ln
            or "context_router.ctx_proj" in ln
            or "expr_proj." in ln
            or "equiv_proto_bank" in ln
            or "long_term.op_" in ln
        )
        if is_specialist:
            if not is_vec_only:
                spec_muon.append(p)
            else:
                spec_vec.append(p)
        else:
            if not is_vec_only:
                trunk_muon.append(p)
            else:
                trunk_vec.append(p)

    groups: List[Dict[str, Any]] = []
    if trunk_muon:
        groups.append(
            {
                "name": "v7_shared_muon",
                "params": trunk_muon,
                "use_muon": True,
                "role": "trunk",
                "mode": shared_mode,
                "lr": float(lr_muon),
                "lr_consolidate": float(lr_consolidate),
                "weight_decay": float(wd_trunk),
                "prec_scale": float(trunk_prec_scale),
            }
        )
    if trunk_vec:
        groups.append(
            {
                "name": "v7_shared_vec",
                "params": trunk_vec,
                "use_muon": False,
                "role": "trunk",
                "mode": shared_mode,
                "lr": float(lr_adam),
                "lr_consolidate": float(lr_consolidate),
                "weight_decay": float(wd_trunk),
                "prec_scale": float(trunk_prec_scale),
            }
        )
    if spec_muon:
        groups.append(
            {
                "name": "v7_specialist_muon",
                "params": spec_muon,
                "use_muon": True,
                "role": "fast",
                "mode": specialist_mode,
                "lr": float(lr_muon),
                "lr_consolidate": float(lr_consolidate * 1.5),
                "weight_decay": float(wd_fast),
                "prec_scale": 0.25,
            }
        )
    if spec_vec:
        groups.append(
            {
                "name": "v7_specialist_vec",
                "params": spec_vec,
                "use_muon": False,
                "role": "fast",
                "mode": specialist_mode,
                "lr": float(lr_adam),
                "lr_consolidate": float(lr_consolidate * 1.5),
                "weight_decay": float(wd_fast),
                "prec_scale": 0.25,
            }
        )
    if fast_proto:
        groups.append(
            {
                "name": "v8_equiv_proto_bank",
                "params": fast_proto,
                "use_muon": False,
                "role": "fast",
                "mode": "acquire",
                "lr": float(max(0.018, lr_adam * 5.0)),
                "lr_consolidate": float(max(0.015, lr_adam * 4.0)),
                "weight_decay": 0.0,
                "prec_scale": 0.02,
            }
        )
    if wm_ret_vec:
        groups.append(
            {
                "name": "v8_wm_ret",
                "params": wm_ret_vec,
                "use_muon": False,
                "role": "fast",
                "mode": "acquire",
                "lr": float(max(0.004, lr_adam)),
                "lr_consolidate": float(max(0.004, lr_adam)),
                "weight_decay": float(wd_fast),
                "prec_scale": 0.06,
            }
        )
    if dec_params:
        groups.append(
            {
                "name": "v7_detached_decoder",
                "params": dec_params,
                "use_muon": False,
                "role": "fast",
                "mode": "acquire",
                "lr": float(lr_adam),
                "lr_consolidate": float(lr_adam),
                "weight_decay": float(wd_fast),
                "prec_scale": 0.2,
            }
        )
    return groups
