"""
VSA Non-Parametric Episodic Memory with Surprise-Adaptive Hebbian Writes.
Adapted from Resonance X / Vector Symbolic Architectures (ICML 2026).

Core Architecture:
1. Buffer Storage: Key directions on S^{D-1}, radial shells on R+, and value payloads
   are stored as registered buffers (excluded from optimizer, zero catastrophic forgetting,
   persisted in checkpoint state_dict).
2. Differentiable Soft-Attention Read:
   Queries with (u_q in S^{D-1}, r_q in R+) using Riemannian geodesic kernel:
   S_i = (kappa * <u_q, u_i> - beta_r * (ln r_q - ln r_i)^2) / sqrt(D)
   weights = softmax(beta_read * S)
   echo = weights @ values
   Gradients flow back through queries during test-time training (TTT).
3. Non-Gradient Surprise-Adaptive Hebbian Write (runs under torch.no_grad()):
   Evaluates model's current memory recall:
   predicted_sim = cos_sim(read(u, r), v)
   surprise = max(0, 1 - predicted_sim)
   Vigilance Gate: If predicted_sim >= vigilance (default 0.85), skips write entirely.
   Effective write strength: alpha_eff = base_alpha * (0.5 + 1.5 * surprise).
4. Direct Post-Freeze Editability:
   External controllers can inspect or directly edit memory slots to steer frozen models.
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ishmn.geometry import project_to_sphere, geodesic_distance_sphere, EPS


class VSAEpisodicMemory(nn.Module):
    """
    Non-parametric, buffer-backed Episodic Memory on S^{D-1} x R+.
    """
    def __init__(
        self,
        hyperspace_dim: int = 8,
        payload_dim: int = 32,
        capacity: int = 512,
        base_alpha: float = 0.8,
        vigilance: float = 0.85,
        beta_read: float = 4.0,
        kappa: float = 4.0,
        beta_r: float = 1.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.hyperspace_dim = hyperspace_dim
        self.payload_dim = payload_dim
        self.capacity = capacity
        self.base_alpha = base_alpha
        self.vigilance = vigilance
        self.beta_read = beta_read
        self.kappa = kappa
        self.beta_r = beta_r

        # Registered non-parametric buffers (excluded from AdamW parameters)
        self.register_buffer("slot_u", torch.zeros(capacity, hyperspace_dim, device=device))
        self.register_buffer("slot_r", torch.ones(capacity, 1, device=device))
        self.register_buffer("slot_v", torch.zeros(capacity, payload_dim, device=device))
        self.register_buffer("occupied", torch.zeros(capacity, dtype=torch.bool, device=device))
        self.register_buffer("next_write_slot", torch.zeros((), dtype=torch.long, device=device))
        self.register_buffer("write_count", torch.zeros((), dtype=torch.long, device=device))

    def forward(
        self,
        u_query: torch.Tensor,
        r_query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable soft-attention read over occupied slots.

        Parameters
        ----------
        u_query : torch.Tensor (..., hyperspace_dim) on S^{D-1}
        r_query : torch.Tensor (..., 1) in [1, inf)

        Returns
        -------
        echo : torch.Tensor (..., payload_dim)
        confidence : torch.Tensor (..., 1) in [0, 1]
        """
        if not bool(self.occupied.any()):
            shape = list(u_query.shape[:-1]) + [self.payload_dim]
            conf_shape = list(u_query.shape[:-1]) + [1]
            return torch.zeros(shape, device=u_query.device, dtype=u_query.dtype), \
                   torch.zeros(conf_shape, device=u_query.device, dtype=u_query.dtype)

        return self._attend(u_query, r_query)

    def _attend(
        self,
        u_query: torch.Tensor,
        r_query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_shape = u_query.shape[:-1]
        D = self.hyperspace_dim
        u_flat = project_to_sphere(u_query.reshape(-1, D))
        r_flat = torch.clamp(r_query.reshape(-1, 1), min=1.0)

        active_u = self.slot_u[self.occupied]   # (K, D)
        active_r = self.slot_r[self.occupied]   # (K, 1)
        active_v = self.slot_v[self.occupied]   # (K, payload_dim)

        # Angular similarity: cos(theta) in [-1, 1]
        cos_sim = u_flat @ active_u.T           # (N, K)

        # Radial log-distance
        ln_rq = torch.log(r_flat)               # (N, 1)
        ln_rk = torch.log(active_r.T)           # (1, K)
        rad_dist_sq = (ln_rq - ln_rk) ** 2      # (N, K)

        # Riemannian similarity kernel
        geom_sim = (self.kappa * cos_sim - self.beta_r * rad_dist_sq) / math.sqrt(D)
        weights = F.softmax(self.beta_read * geom_sim, dim=-1)  # (N, K)

        echo = weights @ active_v                # (N, payload_dim)
        echo = torch.clamp(torch.nan_to_num(echo), min=-5.0, max=5.0)
        
        # Read confidence is average alignment to winning slots
        max_sim = cos_sim.max(dim=-1, keepdim=True).values
        confidence = torch.clamp((max_sim + 1.0) * 0.5, min=0.0, max=1.0)

        echo_reshaped = echo.reshape(*orig_shape, self.payload_dim)
        conf_reshaped = confidence.reshape(*orig_shape, 1)
        return echo_reshaped, conf_reshaped

    @torch.no_grad()
    def write(
        self,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        force_alpha: Optional[float] = None
    ) -> float:
        """
        Non-gradient Hebbian write path with surprise-adaptive scaling and vigilance gating.

        Parameters
        ----------
        u : torch.Tensor (hyperspace_dim,) or (1, hyperspace_dim)
        r : torch.Tensor (1,) or scalar float
        v : torch.Tensor (payload_dim,) or (1, payload_dim)
        force_alpha : Optional float to bypass vigilance and surprise logic.

        Returns
        -------
        alpha_eff : float (0.0 if skipped by vigilance gate)
        """
        u_unit = project_to_sphere(u.reshape(1, self.hyperspace_dim)).squeeze(0)
        r_val = torch.as_tensor(r, device=self.slot_r.device, dtype=torch.float32).reshape(1, 1)
        r_clamped = torch.clamp(r_val, min=1.0).squeeze(0)
        v_vec = torch.clamp(torch.nan_to_num(v.reshape(1, self.payload_dim).squeeze(0)), min=-5.0, max=5.0)

        if force_alpha is not None:
            alpha = float(force_alpha)
        else:
            if bool(self.occupied.any()):
                # Check current recall from memory
                echo, _ = self._attend(u_unit.view(1, 1, -1), r_clamped.view(1, 1, -1))
                echo = echo.view(-1)
                norm_echo = torch.norm(echo, p=2)
                norm_v = torch.norm(v_vec, p=2) + EPS

                if norm_echo > 1e-5:
                    predicted_sim = float(F.cosine_similarity(echo.unsqueeze(0), v_vec.unsqueeze(0)).item())
                else:
                    predicted_sim = 0.0

                # Vigilance Gate: skip write if memory already predicts the association
                if predicted_sim >= self.vigilance:
                    return 0.0
            else:
                predicted_sim = 0.0

            surprise = max(0.0, 1.0 - predicted_sim)
            alpha = self.base_alpha * (0.5 + 1.5 * surprise)
            alpha = min(max(alpha, 0.1), 2.0)

        idx = int(self.next_write_slot.item())
        self.slot_u[idx] = u_unit
        self.slot_r[idx] = r_clamped
        self.slot_v[idx] = v_vec * alpha
        self.occupied[idx] = True
        self.next_write_slot.fill_((idx + 1) % self.capacity)
        self.write_count.add_(1)

        return alpha

    @torch.no_grad()
    def direct_edit(
        self,
        slot_idx: int,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor
    ) -> None:
        """
        Explicitly injects or overrides an association at a specific slot index.
        Enables post-freeze inspection and behavioral steering.
        """
        assert 0 <= slot_idx < self.capacity, f"slot_idx {slot_idx} out of range"
        self.slot_u[slot_idx] = project_to_sphere(u.reshape(1, self.hyperspace_dim)).squeeze(0)
        self.slot_r[slot_idx] = torch.clamp(torch.as_tensor(r, device=self.slot_r.device, dtype=torch.float32).reshape(1), min=1.0)
        self.slot_v[slot_idx] = v.reshape(self.payload_dim)
        self.occupied[slot_idx] = True

    def get_occupancy_ratio(self) -> float:
        return float(self.occupied.float().mean().item())

    def clear(self) -> None:
        self.slot_u.zero_()
        self.slot_r.fill_(1.0)
        self.slot_v.zero_()
        self.occupied.zero_()
        self.next_write_slot.zero_()
        self.write_count.zero_()
