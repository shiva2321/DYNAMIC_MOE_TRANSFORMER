"""
The Unified Hybrid "Yggdrasil" Cognitive Manifold:
    M = H^{d_H} x T^{d_T} x R+   (a Riemannian PRODUCT: the factors act independently; see geometry/README.md section 0)

Combines:
1. Hyperbolic Base (H^{d_H}, Poincare Ball):
   - Encodes conceptual hierarchies, taxonomies, and abstraction depth.
   - Exponential volume expansion prevents concept crowding.
2. Flat Toroidal Fiber (T^{d_T}, Clifford Torus):
   - Encodes symbolic operations, relations, and analogical reasoning.
   - Zero curvature (K = 0) ensures ZERO holonomy and exact reversibility over reasoning chains.
3. Radial Shell (R+):
   - Encodes salience, intensity, scale, and emotional arousal.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch

from geometry.poincare import PoincareBall
from geometry.torus import CliffordTorus

EPS = 1e-7
R_FLOOR = 1e-5          # intensity floor inside the radial metric: the same value the GPU/CPU engine uses (its eps)


@dataclass
class YggdrasilState:
    """A point on the Yggdrasil cognitive manifold."""
    h: torch.Tensor      # [..., d_h] in Poincare ball (hierarchy / category)
    theta: torch.Tensor  # [..., d_t] in [0, 2*pi)^d_t (operational state / phase)
    r: torch.Tensor      # [..., 1] in [0, inf) (intensity / scale)

    def to(self, device: torch.device):
        return YggdrasilState(
            h=self.h.to(device),
            theta=self.theta.to(device),
            r=self.r.to(device)
        )


@dataclass
class YggdrasilAction:
    """An action / transformation on the Yggdrasil manifold."""
    delta_theta: torch.Tensor                 # [..., d_t] phase rotation on Torus fiber
    delta_h: Optional[torch.Tensor] = None     # [..., d_h] optional displacement in Hyperbolic base
    scale_factor: Optional[torch.Tensor] = None # [..., 1] optional multiplicative scale adjustment


class YggdrasilManifold:
    def __init__(self, d_h: int = 16, d_t: int = 16, c: float = 1.0):
        self.d_h = int(d_h)
        self.d_t = int(d_t)
        self.c = float(c)

        self.poincare = PoincareBall(self.d_h, c=self.c)
        self.torus = CliffordTorus(self.d_t)

    def project_state(self, h: torch.Tensor, theta: torch.Tensor, r: Optional[torch.Tensor] = None) -> YggdrasilState:
        """Projects continuous tensors into valid coordinates on the Yggdrasil manifold."""
        h_proj = self.poincare.project(h)
        theta_proj = self.torus.wrap(theta)
        if r is None:
            r = torch.ones(*h.shape[:-1], 1, dtype=h.dtype, device=h.device)
        else:
            r = torch.clamp(r, min=EPS)
        return YggdrasilState(h=h_proj, theta=theta_proj, r=r)

    def distance(
        self,
        s1: YggdrasilState,
        s2: YggdrasilState,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> torch.Tensor:
        """
        Computes joint geodesic product metric:
            d_M(s1, s2) = sqrt( w_h * d_H(h1, h2)^2 + w_t * d_T(th1, th2)^2 + w_r * (ln r1 - ln r2)^2 )
        """
        w_h, w_t, w_r = weights
        d_h = self.poincare.distance(s1.h, s2.h)
        d_t = self.torus.distance(s1.theta, s2.theta)

        log_r1 = torch.log(s1.r.squeeze(-1).clamp(min=R_FLOOR))
        log_r2 = torch.log(s2.r.squeeze(-1).clamp(min=R_FLOOR))
        d_r = torch.abs(log_r1 - log_r2)

        total_sq = w_h * (d_h ** 2) + w_t * (d_t ** 2) + w_r * (d_r ** 2)
        return torch.sqrt(torch.clamp(total_sq, min=0.0))

    def compose_action(self, state: YggdrasilState, action: YggdrasilAction) -> YggdrasilState:
        """
        Applies a relational transformation / action:
        - Operational logic is exact on the Torus fiber: (theta + delta_theta) mod 2*pi.
        - Hierarchical displacement uses Mobius left-addition: delta_h (+) state.h.
        - Salience scale updates multiplicatively: r * scale_factor.
        """
        next_theta = self.torus.bind(state.theta, action.delta_theta)

        if action.delta_h is not None:
            next_h = self.poincare.mobius_add(action.delta_h, state.h)
        else:
            next_h = state.h

        if action.scale_factor is not None:
            next_r = state.r * torch.clamp(action.scale_factor, min=EPS)
        else:
            next_r = state.r

        return YggdrasilState(h=next_h, theta=next_theta, r=next_r)

    def invert_action(self, state: YggdrasilState, action: YggdrasilAction) -> YggdrasilState:
        """
        Exact inverse operation (thinking backwards):
        Reverses the action without any learned backward neural network:
            - delta_theta on Torus fiber
            (-delta_h) (+) next_h on Hyperbolic base (exact left cancellation)
            1 / scale_factor on Radial shell
        """
        inv_theta = self.torus.unbind(state.theta, action.delta_theta)

        if action.delta_h is not None:
            inv_h = self.poincare.mobius_add(-action.delta_h, state.h)
        else:
            inv_h = state.h

        if action.scale_factor is not None:
            inv_r = state.r / torch.clamp(action.scale_factor, min=EPS)
        else:
            inv_r = state.r

        return YggdrasilState(h=inv_h, theta=inv_theta, r=inv_r)

    def extract_relation(self, s_a: YggdrasilState, s_b: YggdrasilState) -> YggdrasilAction:
        """
        Extracts the algebraic relational transformation from state A to state B:
        - delta_theta = (theta_B - theta_A) mod 2*pi
        - delta_h = b (-) a
        - scale_factor = r_B / r_A
        """
        d_th = self.torus.unbind(s_b.theta, s_a.theta)
        d_h = self.poincare.mobius_add(s_b.h, -s_a.h)
        s_fac = s_b.r / s_a.r.clamp(min=EPS)
        return YggdrasilAction(delta_theta=d_th, delta_h=d_h, scale_factor=s_fac)

    def analogy(self, s_a: YggdrasilState, s_b: YggdrasilState, s_c: YggdrasilState) -> YggdrasilState:
        """
        Solves A : B :: C : D (analogical transfer across disparate domains):
        Transports the operational transformation on the Torus fiber to base C with zero drift.
        """
        d_th = self.torus.unbind(s_b.theta, s_a.theta)
        th_d = self.torus.bind(s_c.theta, d_th)
        r_d = s_c.r * (s_b.r / s_a.r.clamp(min=EPS))
        h_d = s_c.h  # Target D resides in domain C
        return YggdrasilState(h=h_d, theta=th_d, r=r_d)
