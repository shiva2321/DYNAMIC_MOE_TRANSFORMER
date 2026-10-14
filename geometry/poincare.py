"""
Poincare Ball model of Hyperbolic Geometry (H^D) with constant negative curvature -c.

Mathematical properties:
1. Space: B_c^D = { x in R^D : c * ||x||^2 < 1 }
2. Conformal metric factor: lambda_x^c = 2 / (1 - c * ||x||^2)
3. Mobius addition:
       u (+) v = ((1 + 2c<u, v> + c||v||^2)u + (1 - c||u||^2)v) / (1 + 2c<u, v> + c^2||u||^2||v||^2)
4. Geodesic distance:
       d_c(u, v) = (2 / sqrt(c)) * artanh(sqrt(c) * ||(-u) (+) v||)
"""

import math
import torch
from geometry.base import Manifold

EPS = 1e-7


class PoincareBall(Manifold):
    def __init__(self, dim: int, c: float = 1.0):
        super().__init__(dim)
        if c <= 0:
            raise ValueError(f"Curvature parameter c must be positive, got {c}")
        self.c = float(c)
        self.sqrt_c = math.sqrt(self.c)

    def max_norm(self, eps: float = 1e-4) -> float:
        """Maximum allowable Euclidean radius to ensure points stay strictly inside the ball."""
        return (1.0 - eps) / self.sqrt_c

    def project(self, x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Projects points into the interior of the Poincare ball."""
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=EPS)
        max_r = self.max_norm(eps)
        cond = norm > max_r
        projected = torch.where(cond, x * (max_r / norm), x)
        return projected

    def lambda_x(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the conformal factor lambda_x = 2 / (1 - c * ||x||^2)."""
        x_sq = torch.sum(x * x, dim=-1, keepdim=True)
        denom = (1.0 - self.c * x_sq).clamp(min=EPS)
        return 2.0 / denom

    def mobius_add(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes Mobius addition u (+) v in the Poincare ball."""
        u_sq = self.c * torch.sum(u * u, dim=-1, keepdim=True)
        v_sq = self.c * torch.sum(v * v, dim=-1, keepdim=True)
        uv = self.c * torch.sum(u * v, dim=-1, keepdim=True)

        num = (1.0 + 2.0 * uv + v_sq) * u + (1.0 - u_sq) * v
        denom = 1.0 + 2.0 * uv + u_sq * v_sq
        res = num / denom.clamp(min=EPS)
        return self.project(res)

    def mobius_sub(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes Mobius subtraction u (-) v = u (+) (-v)."""
        return self.mobius_add(u, -v)

    def distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes exact hyperbolic geodesic distance d_c(u, v)."""
        u = self.project(u)
        v = self.project(v)
        u_sq = self.c * torch.sum(u * u, dim=-1, keepdim=True).clamp(max=1.0 - EPS)
        v_sq = self.c * torch.sum(v * v, dim=-1, keepdim=True).clamp(max=1.0 - EPS)
        diff_sq = self.c * torch.sum((u - v) * (u - v), dim=-1, keepdim=True)

        frac = 1.0 + 2.0 * diff_sq / ((1.0 - u_sq) * (1.0 - v_sq)).clamp(min=EPS)
        acosh_arg = frac.clamp(min=1.0)
        dist = torch.acosh(acosh_arg) / self.sqrt_c
        return torch.where(diff_sq < 1e-12, torch.zeros_like(dist), dist).squeeze(-1)

    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at the origin (x = 0): exp_0(v) = tanh(sqrt(c)*||v||) * v / (sqrt(c)*||v||)."""
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=EPS)
        scale = torch.tanh(self.sqrt_c * v_norm) / (self.sqrt_c * v_norm)
        return self.project(scale * v)

    def log_map_zero(self, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map at the origin (x = 0): log_0(y) = artanh(sqrt(c)*||y||) * y / (sqrt(c)*||y||)."""
        y = self.project(y)
        y_norm = torch.norm(y, dim=-1, keepdim=True).clamp(min=EPS)
        arg = (self.sqrt_c * y_norm).clamp(max=1.0 - EPS)
        scale = torch.atanh(arg) / (self.sqrt_c * y_norm)
        return scale * y

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """General Riemannian exponential map at point x: exp_x(v)."""
        x = self.project(x)
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=EPS)
        lam = self.lambda_x(x)
        arg = (self.sqrt_c * lam * v_norm / 2.0).clamp(max=15.0)
        tanh_val = torch.tanh(arg)
        direction = v / (self.sqrt_c * v_norm)
        v_mapped = tanh_val * direction
        return self.mobius_add(x, v_mapped)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """General Riemannian logarithmic map: log_x(y)."""
        x = self.project(x)
        y = self.project(y)
        diff = self.mobius_add(-x, y)
        diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=EPS)
        arg = (self.sqrt_c * diff_norm).clamp(max=1.0 - EPS)
        lam = self.lambda_x(x)
        factor = (2.0 / (self.sqrt_c * lam)) * torch.atanh(arg)
        return factor * (diff / diff_norm)
