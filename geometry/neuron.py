"""
The norm-gated neuron: a linear neuron plus ONE learned coefficient on the soft norm of its input.

    a_j(x) = <w_j, x> + b_j - c_j * rho(x),      rho(x) = sqrt(1/c + ||x||^2)

This is the Lorentz-linear neuron <w, x>_L + b written on the chart (space-like part) x_s = x, with the time weight w_0 = c_j and the
time coordinate derived (rho = x0). Which neuron it is depends only on the sign of <w, w>_L = ||w_j||^2 - c_j^2:

    ||w_j||^2 - c_j^2 > 0  space-like   a_j = ||w|| sinh(signed distance to a geodesic hyperplane)     (an ordinary half-space neuron)
    ||w_j||^2 - c_j^2 = 0  light-like   -a_j ~ exp(Busemann function of an ideal point)               (a horosphere / "plane wave" neuron)
    ||w_j||^2 - c_j^2 < 0  time-like    a_j = -sqrt(c_j^2 - ||w||^2) cosh(d(x, p))                    (a distance-to-a-prototype neuron)

(identities shown for curvature 1; the time-like one needs c_j > 0), so the prototype lookup of the old memory cortex, the usual hyperplane neuron and the horospherical neuron are one layer and the
model can learn which kind each unit should be. With c_j = 0 it is an ordinary linear neuron. Cost over nn.Linear: one extra
scalar feature (rho) per token and one extra weight per unit. Identities are checked in tests/test_lorentz_attention.py.
"""

import math
import torch
import torch.nn as nn


class NormGatedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, curvature: float = 1.0, bias: bool = True):
        super().__init__()
        self.curvature = float(curvature)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.time_weight = nn.Parameter(torch.zeros(out_features))          # c_j; zero = ordinary linear neuron at initialisation
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rho = torch.sqrt(1.0 / self.curvature + (x * x).sum(dim=-1, keepdim=True))
        out = x @ self.weight.t() - rho * self.time_weight
        return out if self.bias is None else out + self.bias

    @torch.no_grad()
    def causal_character(self) -> torch.Tensor:
        """||w_j||^2 - c_j^2 per unit: > 0 space-like (hyperplane), ~ 0 light-like (horosphere), < 0 time-like (prototype ball)."""
        return (self.weight ** 2).sum(dim=-1) - self.time_weight ** 2
