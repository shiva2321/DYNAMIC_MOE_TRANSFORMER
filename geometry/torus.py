"""
Flat Clifford Torus (T^D = (S^1)^D) with constant zero curvature (K = 0).

Mathematical properties:
1. Space: T^D is isomorphic to D independent phase angles theta in [0, 2*pi)^D.
2. Flat metric: Curvature is identically zero everywhere.
3. Group binding (Abelian group U(1)^D):
       theta_bind = (theta_1 + theta_2) mod 2*pi
   Associative, commutative, exact isometry.
4. Exact inversion (no learned backwards model):
       theta_inv = (-theta) mod 2*pi
5. Zero holonomy:
       Moving along any closed loop results in exactly ZERO rotational drift.
"""

import math
import torch
from geometry.base import Manifold

TWO_PI = 2.0 * math.pi


class CliffordTorus(Manifold):
    def __init__(self, dim: int):
        super().__init__(dim)

    def wrap(self, theta: torch.Tensor) -> torch.Tensor:
        """Wraps phase angles strictly into the principal interval [0, 2*pi)."""
        return torch.remainder(theta, TWO_PI)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Projects continuous angles to the canonical torus coordinates [0, 2*pi)."""
        return self.wrap(x)

    def distance(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        """
        Geodesic distance on the flat torus:
        Shortest angular distance along each circle S^1, combined under the Euclidean metric:
            d_T(th1, th2) = sqrt( sum_j min(|th1_j - th2_j|, 2*pi - |th1_j - th2_j|)^2 )
        """
        diff = torch.abs(self.wrap(theta1) - self.wrap(theta2))
        wrap_diff = torch.minimum(diff, TWO_PI - diff)
        sq_sum = torch.sum(wrap_diff * wrap_diff, dim=-1)
        return torch.sqrt(torch.clamp(sq_sum, min=0.0))

    def bind(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        """
        Algebraic binding (group action):
        Adds phase angles modulo 2*pi: theta_res = (theta1 + theta2) mod 2*pi.
        """
        return self.wrap(theta1 + theta2)

    def invert(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Exact inverse operation (group inverse):
        theta_inv = (-theta) mod 2*pi.
        Equivalent to complex conjugation z* for unit phasors.
        """
        return self.wrap(-theta)

    def unbind(self, theta_compound: torch.Tensor, theta_key: torch.Tensor) -> torch.Tensor:
        """
        Unbinding operation: retrieves original value by subtracting key phases:
        theta_orig = (theta_compound - theta_key) mod 2*pi.
        """
        return self.wrap(theta_compound - theta_key)

    def exp_map(self, theta: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map on the flat torus:
        Because the torus is flat, the exponential map is simply translation wrapped by 2*pi:
            exp_theta(v) = (theta + v) mod 2*pi
        """
        return self.wrap(theta + v)

    def log_map(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map on the flat torus:
        Returns the shortest tangent vector v such that exp_theta1(v) = theta2.
        """
        diff = self.wrap(theta2) - self.wrap(theta1)
        return torch.remainder(diff + math.pi, TWO_PI) - math.pi

    def to_phasors(self, theta: torch.Tensor) -> torch.Tensor:
        """Converts phase angles theta into complex unit phasors z = e^{i*theta}."""
        wrapped = self.wrap(theta)
        return torch.polar(torch.ones_like(wrapped), wrapped)

    def from_phasors(self, z: torch.Tensor) -> torch.Tensor:
        """Converts complex phasors z into phase angles theta in [0, 2*pi)."""
        angle = torch.atan2(z.imag, z.real)
        return self.wrap(angle)

    def inner_product(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        """
        Normalized complex inner product (chordal alignment):
        (1 / D) * sum_j cos(theta1_j - theta2_j).
        Returns a value in [-1, 1], where 1 means identical phases and 0 means orthogonal.
        """
        diff = theta1 - theta2
        return torch.mean(torch.cos(diff), dim=-1)
