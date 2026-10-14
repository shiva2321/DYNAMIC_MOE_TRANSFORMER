"""
Yggdrasil attention: hyperbolic hierarchy + torus phase binding + salience bias as ONE fused dot product.

For every query i and key j the attention logit is

    logit_ij = (1/tau) <q_i, k_j>_L  +  kappa * sum_m cos(theta_q,i,m - theta_k,j,m)  +  beta_j

* <q, k>_L = -q0 k0 + <q_s, k_s> is the Minkowski inner product of two hyperboloid points (curvature -c). It is the
  negative of half the squared Lorentz distance up to the constant 1/c, so -d_L^2 / (2 tau) is the hyperbolic term.
* the cosine term is a von-Mises kernel on the flat torus (smooth where the wrapped distance has its cut locus; this is the
  RoPE / FHRR phase similarity, it depends only on the phase DIFFERENCE).
* beta_j is a per-key salience bias (the log-radius of the key).

All three are inner products, so they are written as one dot product of augmented vectors

    q~ = [ -q0/tau, q_s/tau, sqrt(kappa) cos(theta_q), sqrt(kappa) sin(theta_q), 1    ]
    k~ = [  k0,     k_s,     sqrt(kappa) cos(theta_k), sqrt(kappa) sin(theta_k), beta ]

and evaluated by torch.nn.functional.scaled_dot_product_attention (FlashAttention / memory-efficient kernels, O(T) memory,
causal masks, KV cache). The values are lifted to the hyperboloid and the weighted sum is renormalised back onto it (Lorentz
centroid), so the output is again a valid point; its space-like part goes back into the residual stream.

Numerics (measured, research/yggdrasil_poc/poc_c_attention_and_neuron.py): <q, k>_L loses ~2^-p * R^2 of absolute accuracy at
radius R = ||x_s|| in a p-bit mantissa. The augmented product is therefore evaluated in `compute_dtype` (float32 by default),
and `safe_radius` / `clip_radius` give the radius to stay under for a lower precision.
"""

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F

EPS = 1e-7
Scalar = Union[float, torch.Tensor]


def _time(x_s: torch.Tensor, c: Scalar) -> torch.Tensor:
    return torch.sqrt(1.0 / torch.as_tensor(c, dtype=x_s.dtype, device=x_s.device) + (x_s * x_s).sum(dim=-1, keepdim=True))


def _log0(x_s: torch.Tensor, c: Scalar) -> torch.Tensor:
    """Space-like part of the log map at the origin: asinh(sqrt(c)|x_s|) / (sqrt(c)|x_s|) * x_s (stable near the origin)."""
    sqc = torch.sqrt(torch.as_tensor(c, dtype=x_s.dtype, device=x_s.device))
    n = torch.norm(x_s, dim=-1, keepdim=True).clamp(min=EPS)
    return torch.asinh(sqc * n) / (sqc * n) * x_s


def _exp0(m: torch.Tensor, c: Scalar) -> torch.Tensor:
    """Space-like part of the exp map at the origin: sinh(sqrt(c)|m|) / (sqrt(c)|m|) * m."""
    sqc = torch.sqrt(torch.as_tensor(c, dtype=m.dtype, device=m.device))
    n = torch.norm(m, dim=-1, keepdim=True).clamp(min=EPS)
    return torch.sinh((sqc * n).clamp(max=60.0)) / (sqc * n) * m


def safe_radius(dtype: Union[torch.dtype, str]) -> float:
    """Largest ||x_s|| for which the attention logit error stays below ~0.05 nat (measured: error grows like R^2)."""
    table = {
        torch.bfloat16: 2.0, "bfloat16": 2.0, "bf16": 2.0,
        torch.float16: 6.0, "float16": 6.0, "fp16": 6.0,
        torch.float32: 50.0, "float32": 50.0, "fp32": 50.0,
        torch.float64: 1e6, "float64": 1e6, "fp64": 1e6,
    }
    return table.get(dtype, 6.0)


def clip_radius(x_s: torch.Tensor, r_max: float) -> torch.Tensor:
    """Smooth-enough feature clipping (Guo et al. 2022): x_s * min(1, r_max / ||x_s||). Do NOT use QK-norm instead: it would erase the
    radius, which is the depth / salience information."""
    n = torch.norm(x_s, dim=-1, keepdim=True).clamp(min=EPS)
    return x_s * torch.clamp(r_max / n, max=1.0)


def augment(q_s, k_s, v_s, *, c: Scalar = 1.0, tau: Scalar = 1.0, theta_q=None, theta_k=None, kappa: Scalar = 1.0, key_bias=None,
            pad_to: int = 8):
    """Returns (q~, k~, v~, d_v): augmented tensors with a common last dimension (multiple of `pad_to`) and the value width.

    q_s, k_s: (..., T, D) space-like parts; v_s: (..., T, Dv); theta_q/theta_k: (..., T, M) angles; key_bias: (..., T).
    c, tau and kappa may be floats or tensors broadcastable to (B, H, 1, 1), e.g. t.view(1, H, 1, 1) (per-head curvature, temperature, phase strength)."""
    dt = q_s.dtype
    tau_t = torch.as_tensor(tau, dtype=dt, device=q_s.device)
    q_parts = [-_time(q_s, c) / tau_t, q_s / tau_t]
    k_parts = [_time(k_s, c), k_s]
    if theta_q is not None:
        sk = torch.sqrt(torch.as_tensor(kappa, dtype=dt, device=q_s.device))
        q_parts += [sk * torch.cos(theta_q), sk * torch.sin(theta_q)]
        k_parts += [sk * torch.cos(theta_k), sk * torch.sin(theta_k)]
    if key_bias is not None:
        q_parts.append(torch.ones_like(q_s[..., :1]))
        k_parts.append(key_bias.unsqueeze(-1).to(dt))
    q_aug, k_aug = torch.cat(q_parts, dim=-1), torch.cat(k_parts, dim=-1)
    v_aug = torch.cat([_time(v_s, c), v_s], dim=-1)
    width = max(q_aug.shape[-1], v_aug.shape[-1])
    width += (-width) % pad_to
    pad = lambda t: F.pad(t, (0, width - t.shape[-1]))
    return pad(q_aug), pad(k_aug), pad(v_aug), v_s.shape[-1]


def yggdrasil_attention(q_s, k_s, v_s, *, c: Scalar = 1.0, tau: Scalar = 1.0, theta_q=None, theta_k=None, kappa: Scalar = 1.0,
                        key_bias=None, is_causal: bool = False, attn_mask: Optional[torch.Tensor] = None, aggregate: str = "lorentz",
                        compute_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Fused Yggdrasil attention. Returns the space-like part (..., T, Dv) of the aggregated value points.

    aggregate: 'lorentz'  weighted sum of lifted values, renormalised onto the hyperboloid (Lorentz centroid)
               'tangent'  weighted mean of log_0(values), mapped back with exp_0 (more precise at large radius)
               'euclid'   weighted mean of the space-like values (geometry only decides the weights)
    compute_dtype: dtype of the augmented product; default float32 (float64 inputs stay float64). Pass e.g. torch.float16 only together
    with clip_radius(., safe_radius(torch.float16)). The result is cast back to the input dtype. Pass attn_mask OR is_causal, not both."""
    out_dtype = q_s.dtype
    cd = compute_dtype or (torch.float64 if out_dtype == torch.float64 else torch.float32)
    q_s, k_s, v_s = q_s.to(cd), k_s.to(cd), v_s.to(cd)
    theta_q = None if theta_q is None else theta_q.to(cd)
    theta_k = None if theta_k is None else theta_k.to(cd)
    key_bias = None if key_bias is None else key_bias.to(cd)
    if aggregate == "tangent":
        v_for_sum = _log0(v_s, c)
    else:
        v_for_sum = v_s
    q_aug, k_aug, v_aug, dv = augment(q_s, k_s, v_for_sum, c=c, tau=tau, theta_q=theta_q, theta_k=theta_k, kappa=kappa, key_bias=key_bias)
    if aggregate == "euclid":
        v_aug = F.pad(v_s, (0, q_aug.shape[-1] - v_s.shape[-1]))
    elif aggregate == "tangent":
        v_aug = F.pad(v_for_sum, (0, q_aug.shape[-1] - v_for_sum.shape[-1]))
    out = F.scaled_dot_product_attention(q_aug, k_aug, v_aug, attn_mask=attn_mask, is_causal=is_causal, scale=1.0)
    if aggregate == "euclid":
        return out[..., :dv].to(out_dtype)
    if aggregate == "tangent":
        return _exp0(out[..., :dv], c).to(out_dtype)
    point = out[..., : dv + 1]                                           # sum_j a_j * (x0_j, x_s,j), a_j >= 0, sum a_j = 1
    norm = torch.sqrt(torch.as_tensor(c, dtype=cd, device=point.device) * (point[..., :1] ** 2 - (point[..., 1:] ** 2).sum(-1, keepdim=True)).clamp(min=EPS))
    return (point[..., 1:] / norm).to(out_dtype)


def reference_attention(q_s, k_s, v_s, *, c: Scalar = 1.0, tau: Scalar = 1.0, theta_q=None, theta_k=None, kappa: Scalar = 1.0, key_bias=None,
                        is_causal: bool = False, aggregate: str = "lorentz") -> torch.Tensor:
    """Pair-by-pair definition (O(T^2 D) memory), used only to test the fused version."""
    c_t = torch.as_tensor(c, dtype=q_s.dtype, device=q_s.device)
    q0, k0 = _time(q_s, c), _time(k_s, c)
    sq_space = ((q_s[..., :, None, :] - k_s[..., None, :, :]) ** 2).sum(-1)
    sq_time = (q0[..., :, None, 0] - k0[..., None, :, 0]) ** 2
    d2 = sq_space - sq_time                                             # squared Lorentz distance
    logit = -d2 / (2.0 * torch.as_tensor(tau, dtype=q_s.dtype, device=q_s.device))
    if theta_q is not None:
        logit = logit + kappa * torch.cos(theta_q[..., :, None, :] - theta_k[..., None, :, :]).sum(-1)
    if key_bias is not None:
        logit = logit + key_bias[..., None, :]
    if is_causal:
        t = q_s.shape[-2]
        logit = logit.masked_fill(~torch.tril(torch.ones(t, t, dtype=torch.bool, device=q_s.device)), float("-inf"))
    a = torch.softmax(logit, dim=-1)
    if aggregate == "euclid":
        return a @ v_s
    if aggregate == "tangent":
        return _exp0(a @ _log0(v_s, c), c)
    s = a @ torch.cat([_time(v_s, c), v_s], dim=-1)
    norm = torch.sqrt(c_t * (s[..., :1] ** 2 - (s[..., 1:] ** 2).sum(-1, keepdim=True)).clamp(min=EPS))
    return s[..., 1:] / norm
