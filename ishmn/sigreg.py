"""
Sliced Isotropic Gaussian Regularization (SIGReg)
Reference: Balestriero & LeCun (Nov 2025), "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"

Provides closed-form, parameter-free anti-collapse regularization via the
Cramér-Wold theorem and the Epps-Pulley empirical characteristic function test.
Enforces that representation embeddings follow the information-theoretically optimal
Standard Isotropic Gaussian N(0, I_d) without heuristic variance/covariance schedules.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def epps_pulley_statistic(y: torch.Tensor) -> torch.Tensor:
    """
    Computes the closed-form Epps-Pulley goodness-of-fit statistic against N(0, 1)
    for 1D projected sample batches.

    Args:
        y: Tensor of shape [B, K] containing B samples across K 1D slices.

    Returns:
        Scalar tensor representing the mean Epps-Pulley discrepancy across slices.
    """
    B, K = y.shape
    if B <= 1:
        return torch.tensor(0.0, device=y.device, dtype=y.dtype)

    # Term 1: Pairwise difference empirical characteristic integral
    # diff: [B, B, K]
    diff = y.unsqueeze(1) - y.unsqueeze(0)
    term1 = torch.exp(-0.5 * diff ** 2).mean(dim=(0, 1))  # [K]

    # Term 2: Interaction with standard Gaussian characteristic function
    term2 = math.sqrt(2.0) * torch.exp(-0.25 * y ** 2).mean(dim=0)  # [K]

    # Term 3: Constant integral of squared standard Gaussian characteristic function
    term3 = 1.0 / math.sqrt(3.0)

    # Statistic per slice: T(y_k) = term1 - term2 + term3
    t_stat = term1 - term2 + term3
    return torch.clamp(t_stat.mean(), min=0.0)


class SIGReg(nn.Module):
    """
    Sliced Isotropic Gaussian Regularizer (LeJEPA).

    Replaces heuristic VICReg variance/covariance terms with a single provably
    optimal distribution regularizer.
    """

    def __init__(
        self,
        num_slices: int = 64,
        lambda_reg: float = 1.0,
        fixed_projections: bool = False,
        dim: Optional[int] = None,
    ):
        """
        Args:
            num_slices: Number of random 1D Cramér-Wold slices to project onto.
            lambda_reg: Regularization strength multiplier.
            fixed_projections: If True, buffers a fixed set of projection vectors.
            dim: Feature dimension (required if fixed_projections=True).
        """
        super().__init__()
        self.num_slices = num_slices
        self.lambda_reg = lambda_reg
        self.fixed_projections = fixed_projections

        if fixed_projections and dim is not None:
            v = torch.randn(dim, num_slices)
            v = F.normalize(v, p=2, dim=0)
            self.register_buffer("fixed_v", v)
        else:
            self.fixed_v = None

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates the SIGReg loss and representation diagnostics for embeddings z.

        Args:
            z: Embedding tensor of shape [B, D] or [B, S, D].

        Returns:
            loss: Scaled scalar regularization loss.
            diagnostics: Dict containing mean, std, variance, and off-diagonal covariance.
        """
        if z.dim() == 3:
            B, S, D = z.shape
            z_flat = z.reshape(B * S, D)
        elif z.dim() == 2:
            z_flat = z
            B, D = z.shape
        else:
            raise ValueError(f"SIGReg expects 2D or 3D tensor, got shape {z.shape}")

        N = z_flat.shape[0]
        if N <= 1:
            zero_loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)
            return zero_loss, {"sigreg_loss": 0.0, "var": 0.0, "std": 0.0}

        # Sample or retrieve projection vectors on S^{D-1}
        if self.fixed_v is not None and self.fixed_v.shape[0] == D:
            v = self.fixed_v
        else:
            v = torch.randn(D, self.num_slices, device=z.device, dtype=z.dtype)
            v = F.normalize(v, p=2, dim=0)

        # Project representations to 1D slices
        y = z_flat @ v  # [N, num_slices]

        # Compute Epps-Pulley test statistic
        raw_loss = epps_pulley_statistic(y)
        loss = self.lambda_reg * raw_loss

        # Compute representation diagnostics
        with torch.no_grad():
            dim_vars = z_flat.var(dim=0)
            mean_var = dim_vars.mean().item()
            mean_std = torch.sqrt(dim_vars + 1e-6).mean().item()
            z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
            cov = (z_centered.T @ z_centered) / max(N - 1, 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            max_cov = off_diag.abs().max().item()

            diagnostics = {
                "sigreg_loss": raw_loss.item(),
                "mean_val": z_flat.mean().item(),
                "mean_var": mean_var,
                "mean_std": mean_std,
                "min_var": dim_vars.min().item(),
                "max_cov": max_cov,
            }

        return loss, diagnostics


class SphericalCircularVarianceLoss(nn.Module):
    """
    Spherical Directional Anti-Collapse Loss based on Circular Statistics.
    Reference: Mardia & Jupp (2000), Directional Statistics; Resonance X (2026).

    Quantifies directional dispersion on S^{D-1}:
    - Resultant Mean Vector: R_bar = || E[u] ||_2
    - Circular Variance: V = 1 - R_bar in [0, 1]
    - When R_bar -> 1.0, representations collapse into a narrow directional cone.
    - When R_bar -> 0.0, representations are uniformly dispersed across the sphere.

    Loss penalizes R_bar exceeding a maximum allowable cone ceiling tau_cone (e.g. 0.50).
    """

    def __init__(self, tau_cone: float = 0.50, lambda_cone: float = 1.0):
        super().__init__()
        self.tau_cone = tau_cone
        self.lambda_cone = lambda_cone

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            u: Unit directional vectors on S^{D-1}, shape (..., D)
        Returns:
            loss: scalar penalty for exceeding tau_cone
            diagnostics: dictionary with resultant length R_bar and circular variance V
        """
        D = u.shape[-1]
        u_flat = u.reshape(-1, D)
        N = u_flat.shape[0]

        if N <= 1:
            zero_loss = torch.tensor(0.0, device=u.device, dtype=u.dtype)
            return zero_loss, {"cone_norm": 1.0, "circular_variance": 0.0, "cone_loss": 0.0}

        # Normalize to ensure points lie strictly on S^{D-1}
        u_norm = F.normalize(u_flat, p=2, dim=-1)

        # Mean resultant vector
        mean_u = u_norm.mean(dim=0)
        r_bar = torch.norm(mean_u, p=2)

        # Circular variance
        circ_var = torch.clamp(1.0 - r_bar, min=0.0, max=1.0)

        # Quadratic penalty for cone collapse beyond tau_cone
        cone_excess = torch.clamp(r_bar - self.tau_cone, min=0.0)
        loss = self.lambda_cone * (cone_excess ** 2)

        diagnostics = {
            "cone_norm": float(r_bar.item()),
            "circular_variance": float(circ_var.item()),
            "cone_loss": float(loss.item()),
            "collapsed": bool(r_bar.item() > self.tau_cone),
        }

        return loss, diagnostics
