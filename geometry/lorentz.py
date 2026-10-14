"""
Lorentz (hyperboloid) model of Hyperbolic Geometry (H^D) with constant negative curvature -c.

Mathematical properties:
1. Space: L_c^D = { x = (x0, x_s) in R^(D+1) : <x, x>_L = -1/c, x0 > 0 },  <x, y>_L = -x0*y0 + <x_s, y_s>.
2. The hyperboloid is the GRAPH of a function over R^D:  x0 = sqrt(1/c + ||x_s||^2).
   A point is therefore stored by its space-like part x_s alone ("chart"), every x_s in R^D is valid (there is no boundary
   to clamp at) and the time coordinate is derived. Vector addition of space-like parts is exactly the Lorentz residual
   connection of HyboNet (Chen et al. 2022).
3. Geodesic distance:      d(x, y)   = acosh(-c <x, y>_L) / sqrt(c)
   Squared Lorentz distance: d_L^2   = -2/c - 2 <x, y>_L        (linear in the inner product; what attention uses)
4. Poincare ball (radius 1/sqrt(c)) <-> hyperboloid:
       x0 = (1/sqrt(c)) (1 + c||u||^2) / (1 - c||u||^2),   x_s = 2u / (1 - c||u||^2),   u = x_s / (1 + sqrt(c) x0)
5. One linear functional <w, x>_L is three different neurons depending on the sign of <w, w>_L (see geometry/neuron.py).

Numerical note: <x, y>_L subtracts two numbers of size ~||x_s||^2 to get an O(1) answer, so half-precision inputs lose
~2^-p * R^2 absolute accuracy at radius R = ||x_s|| (measured in research/yggdrasil_poc/poc_c_attention_and_neuron.py).
"""

import math
import torch
from geometry.base import Manifold

EPS = 1e-7


class LorentzModel(Manifold):
    """Hyperboloid model. Points are full ambient vectors (x0, x_s) unless a method says 'chart' (x_s only)."""

    def __init__(self, dim: int, c: float = 1.0):
        super().__init__(dim)
        if c <= 0:
            raise ValueError(f"Curvature parameter c must be positive, got {c}")
        self.c = float(c)
        self.sqrt_c = math.sqrt(self.c)

    # ------------------------------------------------------------------------------------------ chart <-> ambient
    def time(self, x_s: torch.Tensor) -> torch.Tensor:
        """Time coordinate x0 = sqrt(1/c + ||x_s||^2) of the point whose space-like part is x_s."""
        return torch.sqrt(1.0 / self.c + (x_s * x_s).sum(dim=-1, keepdim=True))

    def lift(self, x_s: torch.Tensor) -> torch.Tensor:
        """Chart -> full ambient vector (x0, x_s)."""
        return torch.cat([self.time(x_s), x_s], dim=-1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Projects an ambient vector onto the hyperboloid by recomputing its time coordinate from its space part."""
        return self.lift(x[..., 1:])

    def origin(self, *batch: int, dtype=None, device=None) -> torch.Tensor:
        """The point (1/sqrt(c), 0, ..., 0)."""
        o = torch.zeros(*batch, self.dim + 1, dtype=dtype, device=device)
        o[..., 0] = 1.0 / self.sqrt_c
        return o

    # -------------------------------------------------------------------------------------------- inner product, distance
    @staticmethod
    def inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Minkowski inner product <x, y>_L of ambient vectors."""
        return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)

    def inner_chart(self, x_s: torch.Tensor, y_s: torch.Tensor) -> torch.Tensor:
        """<x, y>_L computed from the charts (no (D+1)-vector is formed)."""
        return -self.time(x_s).squeeze(-1) * self.time(y_s).squeeze(-1) + (x_s * y_s).sum(dim=-1)

    def sq_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Squared Lorentz distance -2/c - 2<x, y>_L (>= 0, equals 0 iff x == y)."""
        return (-2.0 / self.c - 2.0 * self.inner(x, y)).clamp(min=0.0)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic distance acosh(-c <x, y>_L) / sqrt(c); the argument is clamped to keep the gradient finite."""
        arg = (-self.c * self.inner(x, y)).clamp(min=1.0 + EPS)
        return torch.acosh(arg) / self.sqrt_c

    # -------------------------------------------------------------------------------------------- exp / log maps
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """exp_x(v) = cosh(sqrt(c)|v|) x + sinh(sqrt(c)|v|)/(sqrt(c)|v|) v, for ambient tangent v at x (<x, v>_L = 0)."""
        v_norm = torch.sqrt(self.inner(v, v).clamp(min=EPS * EPS)).unsqueeze(-1)
        a = (self.sqrt_c * v_norm).clamp(max=60.0)
        return self.project(torch.cosh(a) * x + torch.sinh(a) / a * v)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """log_x(y) = d(x, y) u / |u|_L with u = y + c <x, y>_L x (the part of y orthogonal to x). Since |u|_L = sinh(sqrt(c) d)/sqrt(c),
        d / |u|_L = asinh(sqrt(c)|u|_L) / (sqrt(c)|u|_L), which stays accurate (-> 1) when y is close to x."""
        u = y + self.c * self.inner(x, y).unsqueeze(-1) * x
        n = torch.sqrt(self.inner(u, u).clamp(min=EPS * EPS)).unsqueeze(-1)
        return torch.asinh(self.sqrt_c * n) / (self.sqrt_c * n) * u

    def exp_map_zero(self, v_s: torch.Tensor) -> torch.Tensor:
        """Exponential map at the origin of a space-like tangent vector v_s in R^D -> full point."""
        n = torch.norm(v_s, dim=-1, keepdim=True).clamp(min=EPS)
        a = (self.sqrt_c * n).clamp(max=60.0)
        return torch.cat([torch.cosh(a) / self.sqrt_c, torch.sinh(a) / (self.sqrt_c * n) * v_s], dim=-1)

    def log_map_zero(self, x: torch.Tensor) -> torch.Tensor:
        """Logarithmic map at the origin of a full point -> space-like tangent vector in R^D."""
        x_s = x[..., 1:]
        n = torch.norm(x_s, dim=-1, keepdim=True).clamp(min=EPS)
        return torch.asinh(self.sqrt_c * n) / (self.sqrt_c * n) * x_s            # sinh(sqrt(c) d) = sqrt(c) |x_s|: stable near the origin

    # -------------------------------------------------------------------------------------------- aggregation
    def centroid(self, weights: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Lorentz centroid of ambient points (Law et al. 2019): s / (sqrt(c) sqrt(-<s, s>_L)) with s = sum_i w_i x_i, w_i >= 0.

        weights (..., N), points (..., N, D+1) -> (..., D+1). The result lies exactly on the hyperboloid."""
        s = (weights.unsqueeze(-1) * points).sum(dim=-2)
        return s / (self.sqrt_c * torch.sqrt((-self.inner(s, s)).clamp(min=EPS)).unsqueeze(-1))

    def tangent_mean(self, weights: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Weighted mean taken in the tangent space at the origin, mapped back (the aggregation HexFormer 2026 uses because the
        Lorentz centroid loses precision at large radius). weights (..., N), points (..., N, D+1) -> (..., D+1)."""
        return self.exp_map_zero((weights.unsqueeze(-1) * self.log_map_zero(points)).sum(dim=-2))

    # -------------------------------------------------------------------------------------------- Poincare conversion
    def from_poincare(self, u: torch.Tensor) -> torch.Tensor:
        """Poincare-ball point (radius 1/sqrt(c)) -> hyperboloid point."""
        cu = self.c * (u * u).sum(dim=-1, keepdim=True)
        denom = (1.0 - cu).clamp(min=EPS)
        return torch.cat([(1.0 + cu) / (self.sqrt_c * denom), 2.0 * u / denom], dim=-1)

    def to_poincare(self, x: torch.Tensor) -> torch.Tensor:
        """Hyperboloid point -> Poincare-ball point."""
        return x[..., 1:] / (1.0 + self.sqrt_c * x[..., :1])
