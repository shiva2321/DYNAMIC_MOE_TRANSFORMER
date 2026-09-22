"""
Hippocampal Dentate Gyrus Pattern Separation & Toroidal Grid Encoders.
Harvested from E:\\universal_substrait\\hyperspace\\dentate_grid.py.

Modules:
1. DentateGyrusPatternSeparator:
   Biological Dentate Gyrus (DG) projection with hyper-sparse Winner-Take-All (k-WTA, 2%),
   driving input cosine overlap toward zero before memory storage to prevent catastrophic interference.
2. ToroidalGridEncoder:
   Multi-scale periodic harmonic grid manifold mapping continuous coordinates to torus T^D.
"""

from __future__ import annotations
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DentateGyrusPatternSeparator(nn.Module):
    """
    Biological Dentate Gyrus (DG) Pattern Separation.
    Projects dense inputs to a high-dimensional sparse layer with competitive lateral inhibition
    and hyper-sparse Winner-Take-All (k-WTA), driving pairwise similarity toward zero.
    """
    def __init__(self, d_in: int, d_sparse: int = 4096, sparsity_ratio: float = 0.02):
        super().__init__()
        self.d_in = d_in
        self.d_sparse = d_sparse
        self.sparsity_ratio = sparsity_ratio
        self.k_active = max(1, int(d_sparse * sparsity_ratio))

        # Perforant path projection
        self.proj = nn.Linear(d_in, d_sparse, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / math.sqrt(d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., d_in]
        Returns:
            sparse_acts: [..., d_sparse] (unit-normalized sparse vector with exactly k_active non-zeros)
        """
        raw_acts = self.proj(x)

        # Lateral Inhibition: Only the top k_active neurons survive
        topk_vals, _ = torch.topk(raw_acts, k=self.k_active, dim=-1)
        kth_val = topk_vals[..., -1:]

        mask = (raw_acts >= kth_val).float()
        sparse_acts = F.relu(raw_acts) * mask

        return F.normalize(sparse_acts, p=2, dim=-1)


class ToroidalGridEncoder(nn.Module):
    """
    Entorhinal Toroidal Grid Cell Encoder.
    Embeds continuous latent vectors into periodic, multi-scale harmonic phases:
    Phi_m(x) = [cos(omega_m * W x), sin(omega_m * W x)] across geometric scales omega_m.
    """
    def __init__(
        self,
        d_in: int,
        d_out: int = 512,
        num_scales: int = 4,
        base_scale: float = 1.0,
        scale_factor: float = 1.42
    ):
        super().__init__()
        assert d_out % (num_scales * 2) == 0, "d_out must be divisible by 2 * num_scales"
        self.d_in = d_in
        self.d_out = d_out
        self.num_scales = num_scales
        self.scale_dim = d_out // (num_scales * 2)

        for s in range(num_scales):
            omega = base_scale * (scale_factor ** s)
            proj = nn.Parameter(torch.randn(d_in, self.scale_dim) * omega)
            self.register_parameter(f"grid_proj_{s}", proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., d_in]
        Returns:
            grid_vec: [..., d_out] (periodic multi-scale grid embedding on torus)
        """
        harmonics = []
        for s in range(self.num_scales):
            proj = getattr(self, f"grid_proj_{s}")
            phases = torch.matmul(x, proj)
            harmonics.append(torch.cos(phases))
            harmonics.append(torch.sin(phases))

        grid_vec = torch.cat(harmonics, dim=-1)
        return F.normalize(grid_vec, p=2, dim=-1)
