"""
Hyperspherical Geometry (S^{D-1}) with constant positive curvature K = +1.

Mathematical properties:
1. Space: S^{D-1} = { u in R^D : ||u|| = 1 }
2. Geodesic distance: d_S(u, v) = arccos(clamp(<u, v>, -1, 1))
3. Parallel transport has non-zero holonomy (proportional to enclosed area).
"""

import math
import torch
from geometry.base import Manifold

EPS = 1e-7


class Hypersphere(Manifold):
    def __init__(self, dim: int):
        super().__init__(dim)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Projects non-zero vectors onto the unit hypersphere S^{D-1}."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=EPS)
        return x / norm

    def distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes geodesic angle distance on S^{D-1} in radians [0, pi]."""
        u = self.project(u)
        v = self.project(v)
        ud = u.double() / u.double().norm(dim=-1, keepdim=True).clamp(min=EPS)
        vd = v.double() / v.double().norm(dim=-1, keepdim=True).clamp(min=EPS)
        cos_sim = torch.sum(ud * vd, dim=-1).clamp(-1.0, 1.0)
        dist = torch.acos(cos_sim).to(u.dtype)
        diff_sq = torch.sum((u - v) * (u - v), dim=-1)
        return torch.where(diff_sq < 1e-12, torch.zeros_like(dist), dist)


    def project_tangent(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Projects ambient vector v onto tangent space T_u S^{D-1}."""
        return v - torch.sum(u * v, dim=-1, keepdim=True) * u

    def exp_map(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Riemannian exponential map on S^{D-1}:
        exp_u(v) = cos(||v||) * u + sin(||v||) * (v / ||v||)
        """
        u = self.project(u)
        v_tan = self.project_tangent(u, v)
        v_norm = torch.norm(v_tan, p=2, dim=-1, keepdim=True)

        safe_norm = v_norm.clamp(min=EPS)
        sin_coeff = torch.where(v_norm < 1e-4, 1.0 - (v_norm ** 2) / 6.0, torch.sin(v_norm) / safe_norm)
        cos_coeff = torch.cos(v_norm)

        next_u = cos_coeff * u + sin_coeff * v_tan
        return self.project(next_u)

    def log_map(self, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Riemannian logarithmic map on S^{D-1}: log_u(w)."""
        u = self.project(u)
        w = self.project(w)
        cos_dist = torch.sum(u * w, dim=-1, keepdim=True).clamp(-1.0, 1.0)
        theta = torch.acos(cos_dist)

        diff = self.project_tangent(u, w)
        diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True).clamp(min=EPS)
        return theta * (diff / diff_norm)

    def parallel_transport(self, u: torch.Tensor, w: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Parallel transports tangent vector v from T_u to T_w along minimizing geodesic."""
        u = self.project(u)
        w = self.project(w)
        v_tan = self.project_tangent(u, v)

        uw_dot = torch.sum(u * w, dim=-1, keepdim=True).clamp(-1.0, 1.0)
        wv_dot = torch.sum(w * v_tan, dim=-1, keepdim=True)

        denom = (1.0 + uw_dot).clamp(min=EPS)
        transported = v_tan - (wv_dot / denom) * (u + w)
        return self.project_tangent(w, transported)

    def inner_product(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Cosine similarity on the unit sphere."""
        u = self.project(u)
        v = self.project(v)
        return torch.sum(u * v, dim=-1)
