"""
Muon: MomentUm Orthogonalized by Newton-Schulz (Keller Jordan, 2024/2025).
Spectral norm momentum orthogonalization for 2D neural network weight matrices,
paired with a hybrid coordinator (HybridMuonAdamW) for seamless multi-parameter group training.

Provides ~2x computational efficiency (reaches target loss in ~50% fewer steps)
and halves optimizer memory by tracking only 1 momentum buffer for 2D weights.
"""

from typing import List, Dict, Any, Optional, Union, Iterable
import math
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from ishmn.dynamic_optimizer import DynamicWarmupAdamW


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Approximates the polar factor of G by iteratively driving singular values toward 1.
    
    Coefficients: a = 3.4445, b = -4.7750, c = 2.0315 (Keller Jordan reference values).
    """
    assert G.ndim == 2, f"zeropower_via_newtonschulz5 expects a 2D matrix, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    X = X / (X.norm() + eps)
    
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(dtype=G.dtype)


class Muon(Optimizer):
    """
    Muon optimizer: steepest descent under the spectral norm via Newton-Schulz orthogonalization.
    Specifically designed for 2D hidden weight matrices in Transformers and MLPs.
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    # Fallback for unexpected non-2D tensor in Muon group: standard momentum update
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    p.data.add_(buf, alpha=-lr)
                    continue

                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update = g + momentum * buf
                else:
                    update = buf

                update_ortho = zeropower_via_newtonschulz5(update, steps=ns_steps)
                # Scale update by aspect ratio: max(1, M/N)^0.5
                scale = max(1.0, update.size(0) / update.size(1)) ** 0.5
                p.data.add_(update_ortho, alpha=-lr * scale)

        return loss


class HybridMuonAdamW:
    """
    Unified Hybrid Optimizer coordinating Muon (for 2D weights) and DynamicWarmupAdamW
    (for 1D weights, embeddings, RMSNorms, and hyperspace coordinates).

    Automatically classifies parameters:
    - 2D parameters with > 128 elements (excluding embeddings) -> Muon (lr ~ 0.02)
    - All other parameters (embeddings, biases, 1D norms, hyperspace coordinates) -> AdamW (lr ~ 1e-3)
    """
    def __init__(
        self,
        model: nn.Module,
        muon_lr: float = 0.02,
        adamw_lr: float = 1e-3,
        muon_momentum: float = 0.95,
        adamw_betas: tuple = (0.9, 0.95),
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        use_dynamic_warmup: bool = True,
    ):
        self.model = model
        self.muon_params = []
        self.adamw_params = []

        # Find embedding parameter IDs to keep them in AdamW
        embedding_param_ids = set()
        for m in model.modules():
            if isinstance(m, (nn.Embedding, nn.EmbeddingBag)):
                for p in m.parameters():
                    embedding_param_ids.add(id(p))

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_unembed = ("head_primary" in name) or ("mtp_heads" in name) or ("unembed" in name) or ("lm_head" in name)
            if p.ndim == 2 and id(p) not in embedding_param_ids and not is_unembed and p.numel() >= 64:
                self.muon_params.append(p)
            else:
                self.adamw_params.append(p)

        # Initialize Muon
        if len(self.muon_params) > 0:
            self.muon = Muon(
                self.muon_params,
                lr=muon_lr,
                momentum=muon_momentum,
                nesterov=True,
                ns_steps=ns_steps,
                weight_decay=weight_decay,
            )
        else:
            self.muon = None

        # Initialize AdamW / DynamicWarmupAdamW
        if len(self.adamw_params) > 0:
            if use_dynamic_warmup:
                self.adamw = DynamicWarmupAdamW(
                    self.adamw_params,
                    lr=adamw_lr,
                    betas=adamw_betas,
                    weight_decay=weight_decay,
                )
            else:
                self.adamw = AdamW(
                    self.adamw_params,
                    lr=adamw_lr,
                    betas=adamw_betas,
                    weight_decay=weight_decay,
                )
        else:
            self.adamw = None

    def zero_grad(self, set_to_none: bool = True):
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss_muon = self.muon.step(closure=closure) if self.muon is not None else None
        loss_adamw = self.adamw.step(closure=closure) if self.adamw is not None else None
        return loss_muon or loss_adamw

    def set_lr(self, muon_lr: float, adamw_lr: float):
        """Dynamically adjusts learning rate for Muon and AdamW parameter groups."""
        if self.muon is not None:
            for g in self.muon.param_groups:
                g["lr"] = muon_lr
        if self.adamw is not None:
            for g in self.adamw.param_groups:
                g["lr"] = adamw_lr

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        groups = []
        if self.muon is not None:
            groups.extend(self.muon.param_groups)
        if self.adamw is not None:
            groups.extend(self.adamw.param_groups)
        return groups

    def state_dict(self) -> Dict[str, Any]:
        return {
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if self.muon is not None and state_dict.get("muon") is not None:
            self.muon.load_state_dict(state_dict["muon"])
        if self.adamw is not None and state_dict.get("adamw") is not None:
            self.adamw.load_state_dict(state_dict["adamw"])
