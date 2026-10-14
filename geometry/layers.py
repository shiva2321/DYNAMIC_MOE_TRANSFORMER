"""
Neural Network Layers for the Yggdrasil Manifold ("Geometric Sandwich Architecture").
Accelerated on NVIDIA GPUs via the compiled Rust kernel with exact analytic autodiff gradients.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from geometry.gpu_kernel import YggdrasilKernel


_BACKWARD_BLOCK_FLOATS = 1 << 24          # temporaries per block hold about this many floats (64 MB each)


def _block_grads(A, B, dist, grad_out, dim_h, dim_t, c, eps, w_h, w_t, w_r):
    """Gradient of sum(grad_out * dist) w.r.t. A (N, D) and the block B (m, D); dist and grad_out are (N, m)."""
    A_exp = A.unsqueeze(1)                    # (N, 1, D)
    B_exp = B.unsqueeze(0)                    # (1, m, D)

    # 1. Hyperbolic part: d_H = (2 / sqrt c) asinh(sqrt xi), xi = c |u - v|^2 / ((1 - c|u|^2)(1 - c|v|^2))
    u = A_exp[..., :dim_h]
    v = B_exp[..., :dim_h]
    diff_u_v = u - v
    diff_sq = (diff_u_v ** 2).sum(dim=-1, keepdim=True)
    u_sq = (u ** 2).sum(dim=-1, keepdim=True)
    v_sq = (v ** 2).sum(dim=-1, keepdim=True)
    alpha_u = torch.clamp(1.0 - c * u_sq, min=1e-7)
    alpha_v = torch.clamp(1.0 - c * v_sq, min=1e-7)
    xi = torch.clamp((c * diff_sq) / (alpha_u * alpha_v), min=0.0)
    d_h = (2.0 / math.sqrt(c)) * torch.asinh(torch.sqrt(xi + 1e-18))
    denom_xi = torch.clamp(torch.sqrt((xi + 1e-14) * (xi + 1.0)), min=1e-7)
    dd2_dxi = torch.where(xi < 1e-10, torch.full_like(xi, 4.0 / c), (2.0 * d_h) / (math.sqrt(c) * denom_xi))
    dxi_du = (2.0 * c / (alpha_u * alpha_v)) * (diff_u_v + (c * diff_sq / alpha_u) * u)
    dxi_dv = (2.0 * c / (alpha_u * alpha_v)) * (-diff_u_v + (c * diff_sq / alpha_v) * v)
    grad_d2_u = w_h * dd2_dxi * dxi_du
    grad_d2_v = w_h * dd2_dxi * dxi_dv

    # 2. Torus part: d_T^2 = sum_j wrap(theta_u - theta_v)^2
    t_u = A_exp[..., dim_h:dim_h + dim_t]
    t_v = B_exp[..., dim_h:dim_h + dim_t]
    two_pi = 2.0 * math.pi
    delta = (t_u - t_v) % two_pi
    delta = torch.where(delta >= math.pi, delta - two_pi, delta)
    grad_d2_tu = w_t * 2.0 * delta
    grad_d2_tv = -grad_d2_tu

    # 3. Radial part: d_R = |ln r_u - ln r_v| with r floored at eps (flat, hence zero gradient, below the floor)
    ru_raw, rv_raw = A_exp[..., -1:], B_exp[..., -1:]
    r_u, r_v = ru_raw.clamp(min=eps), rv_raw.clamp(min=eps)
    diff_r = torch.log(r_u) - torch.log(r_v)
    grad_d2_ru = w_r * 2.0 * diff_r / r_u * (ru_raw > eps).to(A.dtype)
    grad_d2_rv = -w_r * 2.0 * diff_r / r_v * (rv_raw > eps).to(A.dtype)

    # Chain rule through D = sqrt(D^2): dD = dD^2 / (2 D)
    inv_2dist = 1.0 / (2.0 * torch.clamp(dist.unsqueeze(-1), min=1e-7))
    g = grad_out.unsqueeze(-1) * inv_2dist
    grad_A = (g * torch.cat([grad_d2_u, grad_d2_tu, grad_d2_ru], dim=-1)).sum(dim=1)
    grad_B = (g * torch.cat([grad_d2_v, grad_d2_tv, grad_d2_rv], dim=-1)).sum(dim=0)
    return grad_A, grad_B


class YggdrasilDistanceFunction(torch.autograd.Function):
    """
    Differentiable pairwise distance on the Yggdrasil Manifold.
    Forward pass accelerated by Rust GPU kernel; backward pass computes exact analytic gradients.
    """

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        dim_h: int,
        dim_t: int,
        c: float = 1.0,
        eps: float = 1e-5,
        w_h: float = 1.0,
        w_t: float = 1.0,
        w_r: float = 1.0,
        kernel: Optional[YggdrasilKernel] = None,
    ):
        if kernel is None:
            kernel = YggdrasilKernel(device_ordinal=A.device.index or 0 if A.is_cuda else 0)

        # Call the high-throughput Rust GPU/CPU kernel
        dist = kernel.pairwise_distance(A, B, dim_h, dim_t, c, eps, w_h, w_t, w_r)

        ctx.save_for_backward(A, B, dist)
        ctx.dim_h = dim_h
        ctx.dim_t = dim_t
        ctx.c = c
        ctx.eps = eps
        ctx.w_h = w_h
        ctx.w_t = w_t
        ctx.w_r = w_r
        return dist

    @staticmethod
    def backward(ctx, grad_output):
        A, B, dist = ctx.saved_tensors
        n, m = dist.shape
        grad_A = torch.zeros_like(A)
        grad_B = torch.zeros_like(B)
        # B is processed in column blocks so the (N, block, D) temporaries stay near _BACKWARD_BLOCK_FLOATS floats;
        # the first version expanded to (N, M, D) at once (0.55 GB per temporary at N = M = 2048).
        block = max(1, min(m, _BACKWARD_BLOCK_FLOATS // max(1, n * A.shape[-1])))
        for lo in range(0, m, block):
            hi = min(m, lo + block)
            g_a, g_b = _block_grads(A, B[lo:hi], dist[:, lo:hi], grad_output[:, lo:hi],
                                    ctx.dim_h, ctx.dim_t, ctx.c, ctx.eps, ctx.w_h, ctx.w_t, ctx.w_r)
            grad_A += g_a
            grad_B[lo:hi] = g_b
        return grad_A, grad_B, None, None, None, None, None, None, None, None


class YggdrasilProjector(nn.Module):
    """
    Differentiable projection layer mapping Euclidean vectors to the Yggdrasil Manifold.
    Guarantees that outputs strictly satisfy manifold geometry bounds:
    - Hyperbolic: ||x|| < (1 - eps) / sqrt(c)
    - Torus: wrapped smoothly into [-pi, pi)
    - Radial: strictly positive intensity r > eps
    """

    def __init__(
        self,
        in_features: int,
        dim_h: int = 16,
        dim_t: int = 16,
        c: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_features = in_features
        self.dim_h = dim_h
        self.dim_t = dim_t
        self.c = c
        self.eps = eps
        self.total_dim = dim_h + dim_t + 1

        self.linear = nn.Linear(in_features, self.total_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)

        # 1. Hyperbolic projection via bounded tanh scaling
        max_r = (1.0 - self.eps) / math.sqrt(self.c)
        raw_h = out[..., :self.dim_h]
        norm_h = torch.norm(raw_h, dim=-1, keepdim=True).clamp(min=1e-8)
        scaled_h = (max_r * torch.tanh(norm_h)) * (raw_h / norm_h)

        # 2. Torus projection: atan2(sin, cos) for smooth continuous wrapping
        raw_t = out[..., self.dim_h:self.dim_h + self.dim_t]
        scaled_t = torch.atan2(torch.sin(raw_t), torch.cos(raw_t))

        # 3. Radial projection: positive scale via softplus
        raw_r = out[..., -1:]
        scaled_r = F.softplus(raw_r) + self.eps

        return torch.cat([scaled_h, scaled_t, scaled_r], dim=-1)


class YggdrasilMemoryCortex(nn.Module):
    """
    Central Cognitive Memory & Reasoning Cortex on the Yggdrasil Manifold.
    Maintains a bank of K memory prototypes / cognitive anchors in Yggdrasil space.
    Computes GPU-accelerated geodesic distances and generates attention weights
    and contextual memory readouts.
    """

    def __init__(
        self,
        num_prototypes: int,
        dim_h: int = 16,
        dim_t: int = 16,
        out_features: Optional[int] = None,
        c: float = 1.0,
        eps: float = 1e-5,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.dim_h = dim_h
        self.dim_t = dim_t
        self.total_dim = dim_h + dim_t + 1
        self.c = c
        self.eps = eps
        self.temperature = temperature
        self.out_features = out_features

        # Initialize trainable prototypes in manifold interior
        raw_prototypes = torch.randn(num_prototypes, self.total_dim) * 0.1
        # Radial positive
        raw_prototypes[:, -1] = raw_prototypes[:, -1].abs() + 0.5
        self.prototypes = nn.Parameter(raw_prototypes)

        if out_features is not None:
            self.value_matrix = nn.Parameter(
                torch.randn(num_prototypes, out_features) / math.sqrt(num_prototypes)
            )
        else:
            self.value_matrix = None

        self._kernel = None

    def _get_kernel(self, device):
        if self._kernel is None:
            ordinal = device.index or 0 if device.type == "cuda" else 0
            self._kernel = YggdrasilKernel(device_ordinal=ordinal)
        return self._kernel

    def clamp_prototypes(self):
        """Retracts prototypes strictly inside manifold boundaries after optimizer steps."""
        with torch.no_grad():
            max_r = (1.0 - self.eps) / math.sqrt(self.c)
            h = self.prototypes.data[:, :self.dim_h]
            h_norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(max_r / h_norm, max=1.0)
            self.prototypes.data[:, :self.dim_h] = h * scale

            # Torus wrap
            two_pi = 2.0 * math.pi
            t = self.prototypes.data[:, self.dim_h:self.dim_h + self.dim_t]
            t = (t + math.pi) % two_pi - math.pi
            self.prototypes.data[:, self.dim_h:self.dim_h + self.dim_t] = t

            # Radial clamp
            r = self.prototypes.data[:, -1:]
            self.prototypes.data[:, -1:] = torch.clamp(r, min=self.eps)

    def forward(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        queries: (Batch, Total_Dim) on Yggdrasil Manifold
        Returns:
            logits: (Batch, Num_Prototypes) negative scaled squared distances
            context: (Batch, Out_Features) if value_matrix is defined, else None
        """
        kernel = self._get_kernel(queries.device)

        # Pairwise distance via GPU/CPU autograd kernel: (Batch, Num_Prototypes)
        dist = YggdrasilDistanceFunction.apply(
            queries,
            self.prototypes,
            self.dim_h,
            self.dim_t,
            self.c,
            self.eps,
            1.0, 1.0, 1.0,
            kernel,
        )

        # Cognitive attention logits: -d^2 / temperature
        logits = -(dist ** 2) / self.temperature

        context = None
        if self.value_matrix is not None:
            attn = F.softmax(logits, dim=-1)
            context = torch.matmul(attn, self.value_matrix)

        return logits, context


class YggdrasilSandwichModel(nn.Module):
    """
    Complete "Geometric Sandwich" Neural Network:
    - Sensory Perception (Flat, fast Euclidean MLP)
    - Cognitive Memory Cortex (Yggdrasil Manifold + GPU Prototypes)
    - Readout Head (Decision / Classification)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_prototypes: int,
        num_classes: int,
        dim_h: int = 16,
        dim_t: int = 16,
        c: float = 1.0,
    ):
        super().__init__()
        # 1. Sensory Perception (Fast, flat Euclidean encoder)
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # 2. Projector into Yggdrasil Manifold
        self.projector = YggdrasilProjector(
            in_features=hidden_dim,
            dim_h=dim_h,
            dim_t=dim_t,
            c=c,
        )

        # 3. Central Cognitive Cortex (GPU accelerated prototypes)
        self.cortex = YggdrasilMemoryCortex(
            num_prototypes=num_prototypes,
            dim_h=dim_h,
            dim_t=dim_t,
            out_features=hidden_dim,
            c=c,
        )

        # 4. Readout Head
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flat sensory perception
        feat = self.encoder(x)

        # Projection into Yggdrasil Manifold
        state = self.projector(feat)

        # Cognitive cortex reasoning
        _, context = self.cortex(state)

        # Downstream classification
        out = self.head(context + feat)  # Residual connection from perception
        return out
