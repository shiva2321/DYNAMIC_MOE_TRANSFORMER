"""
SLM-Manifold Cross-Space Adapter Bridge.
Connects Small Language Model (Qwen2.5-1.5B, hidden_size=1536) hidden states at Layer 14
with the Universal Cognitive Engine (UCE, d_model=32, S^7 x R+).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from ishmn.geometry import project_to_sphere, geodesic_distance_sphere, EPS


class SLMManifoldAdapter(nn.Module):
    """
    Bidirectional nonlinear adapter between transformer hidden space (H=1536)
    and Riemannian hyperspherical latent space (d_model=32, S^7 x R+).
    """
    def __init__(
        self,
        hidden_size: int = 1536,
        d_model: int = 32,
        hyperspace_dim: int = 8,
        mlp_dim: int = 256,
        alpha: float = 0.22,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.hyperspace_dim = hyperspace_dim
        self.mlp_dim = mlp_dim
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

        # Normalization layer for transformer hidden states
        self.h_norm = nn.LayerNorm(hidden_size)

        # Down-projection: H (1536) -> mlp_dim (256) -> d_model (32)
        self.down_proj = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model)
        )

        # Up-projection: d_model (32) -> mlp_dim (256) -> H (1536)
        self.up_proj = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_size)
        )

        # Orthogonal / Kaiming initialization
        for m in self.down_proj:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.up_proj:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if device is not None:
            self.to(device)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Maps transformer hidden state h [..., hidden_size] to manifold latent z [..., d_model].
        """
        h_float = h.float()
        h_normed = self.h_norm(h_float)
        z = self.down_proj(h_normed)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps manifold latent z [..., d_model] to un-normalized hidden steering direction [..., hidden_size].
        """
        v = self.up_proj(z.float())
        return v

    def compute_steering_vector(
        self,
        z_settled: torch.Tensor,
        h_ref: torch.Tensor,
        gain: Optional[float] = None,
        coherence: Optional[float] = None,
        causal_error: Optional[float] = None
    ) -> torch.Tensor:
        """
        Computes the norm-scaled, coherence-modulated steering delta vector to add into h_ref.
        Delta h = gain_eff * ||h_ref|| * (v / ||v||)
        """
        s_gain = gain if gain is not None else self.alpha.item()
        if causal_error is not None:
            err_factor = 1.0 + 0.5 * math.tanh(causal_error)
            s_gain = s_gain * err_factor
        if coherence is not None:
            coh_factor = min(max(float(coherence), 0.5), 1.5)
            s_gain = s_gain * coh_factor

        v = self.decode(z_settled) # [..., hidden_size]
        v_unit = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-7)

        # Scale by reference hidden state norm
        h_norm_val = torch.norm(h_ref.float(), dim=-1, keepdim=True)
        delta_h = s_gain * h_norm_val * v_unit
        return delta_h.to(h_ref.dtype)

    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, device: Optional[torch.device] = None):
        target_device = device or next(self.parameters()).device
        state = torch.load(path, map_location=target_device, weights_only=False)
        self.load_state_dict(state)
        self.to(target_device)
        print(f"[+] Loaded SLM-Manifold Adapter weights from: {path}")
