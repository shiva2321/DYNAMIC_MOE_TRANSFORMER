"""
YggMultiHeadAttention: Multi-head attention with hyperbolic, toroidal phase, and salience bias options.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry.attention import clip_radius, safe_radius, yggdrasil_attention
from .config import YggCfg


def inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class YggMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention supporting both Euclidean and Yggdrasil geometries.
    """

    def __init__(self, cfg: YggCfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.geometry = cfg.head_geometry
        self.n_phase = cfg.n_phase
        self.r_max = cfg.r_max

        # Q, K, V projections
        self.qkv = nn.Linear(self.d_model, 3 * self.n_heads * self.head_dim, bias=False)
        self.out = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        # Scale factor to initialize step-0 median u ~ 0.35-0.45
        init_scale = 0.70 / math.sqrt(self.head_dim)
        if self.geometry in {"hyp", "hyp_phase", "hyp_phase_bias"}:
            self.q_scale = nn.Parameter(torch.full((1, self.n_heads, 1, 1), init_scale))
            self.k_scale = nn.Parameter(torch.full((1, self.n_heads, 1, 1), init_scale))
            c_init_val = inv_softplus(cfg.c_init)
            if cfg.learn_c:
                self.g_c = nn.Parameter(torch.full((self.n_heads,), c_init_val))
            else:
                self.register_buffer("g_c", torch.full((self.n_heads,), c_init_val))
        else:
            self.q_scale = None
            self.k_scale = None
            self.g_c = None

        if self.geometry in {"hyp", "hyp_phase", "hyp_phase_bias", "euclid_temp"}:
            tau_init_val = inv_softplus(cfg.tau_init)
            self.g_tau = nn.Parameter(torch.full((self.n_heads,), tau_init_val))
        else:
            self.g_tau = None

        if self.geometry in {"hyp_phase", "hyp_phase_bias"}:
            self.phase = nn.Linear(self.d_model, 2 * self.n_heads * self.n_phase, bias=False)
            self.g_kappa = nn.Parameter(torch.full((self.n_heads,), inv_softplus(1.0)))
        else:
            self.phase = None
            self.g_kappa = None

        if self.geometry == "hyp_phase_bias":
            self.bias_proj = nn.Linear(self.d_model, self.n_heads, bias=False)
        else:
            self.bias_proj = None

        if self.geometry == "euclid_gate":
            self.head_gate = nn.Parameter(torch.zeros(1, self.n_heads, 1, 1))
        else:
            self.head_gate = None

        if cfg.null_key:
            self.null_k = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_dim))
            self.null_v = nn.Parameter(torch.randn(1, self.n_heads, 1, self.head_dim) * 0.02)
            self.null_beta = nn.Parameter(torch.zeros(1, self.n_heads, 1))
        else:
            self.null_k = None
            self.null_v = None
            self.null_beta = None

        if hasattr(torch, cfg.qk_dtype):
            self.compute_dtype = getattr(torch, cfg.qk_dtype)
        else:
            self.compute_dtype = torch.float32
        self.is_tf32 = (cfg.qk_dtype == "tf32")

        # Cached last stats for operating_point report
        self.last_stats: Optional[Dict] = None

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = True,
        record_stats: bool = False,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H = self.n_heads
        DH = self.head_dim

        qkv = self.qkv(x).view(B, T, 3, H, DH).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, DH)

        if self.null_k is not None:
            # Prepend null key at origin
            k = torch.cat([self.null_k.expand(B, -1, 1, -1), k], dim=2)
            v = torch.cat([self.null_v.expand(B, -1, 1, -1), v], dim=2)
            # Custom causal mask allowing all queries to attend to null key (col 0)
            attn_mask = torch.ones(T, T + 1, dtype=torch.bool, device=x.device)
            attn_mask[:, 1:] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            attn_mask = attn_mask.view(1, 1, T, T + 1)
            effective_causal = False
        else:
            attn_mask = None
            effective_causal = is_causal

        if self.geometry in {"euclid", "euclid_gate", "euclid_temp"}:
            inv_tau = (1.0 / (F.softplus(self.g_tau) + 1e-3)).view(1, H, 1, 1) if self.geometry == "euclid_temp" else 1.0
            scale = (1.0 / math.sqrt(DH)) * inv_tau

            if record_stats:
                scores = (q @ k.transpose(-2, -1)) * scale
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, float("-inf"))
                elif effective_causal:
                    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
                    scores = scores.masked_fill(~causal, float("-inf"))
                attn_weights = F.softmax(scores, dim=-1)
                log_attn = F.log_softmax(scores, dim=-1)
                top1_mass = float(attn_weights.max(dim=-1).values.mean().item())
                entropy = float(-(attn_weights * log_attn).nan_to_num(0.0).sum(dim=-1).mean().item())
                self.last_stats = {
                    "geometry": self.geometry,
                    "top1_mass": top1_mass,
                    "entropy": entropy,
                    "attn_weights": attn_weights,
                }

            if self.geometry == "euclid_temp":
                q_scaled = q * inv_tau
                o = F.scaled_dot_product_attention(q_scaled, k, v, attn_mask=attn_mask, is_causal=effective_causal)
            else:
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=effective_causal)

            if self.geometry == "euclid_gate":
                o = torch.sigmoid(self.head_gate) * o
            return self.out(o.permute(0, 2, 1, 3).reshape(B, T, H * DH))

        # Hyperbolic / Yggdrasil attention
        c = (F.softplus(self.g_c) + 1e-3).view(1, H, 1, 1)
        tau = (F.softplus(self.g_tau) + 1e-3).view(1, H, 1, 1)

        # Apply per-head scale
        q = q * self.q_scale
        k = k * self.k_scale

        # Radius clipping
        if self.r_max is not None:
            q = clip_radius(q, self.r_max)
            k = clip_radius(k, self.r_max)
            v = clip_radius(v, self.r_max)

        # Torus phase
        if self.phase is not None:
            ph = self.phase(x).view(B, T, 2, H, self.n_phase).permute(2, 0, 3, 1, 4)
            th_q, th_k = ph[0], ph[1]
            if self.null_k is not None:
                th_k = torch.cat([torch.zeros(B, H, 1, self.n_phase, device=x.device), th_k], dim=2)
            kappa = F.softplus(self.g_kappa).view(1, H, 1, 1)
        else:
            th_q, th_k = None, None
            kappa = None

        # Key bias
        if self.bias_proj is not None:
            # (B, T, H) -> (B, H, T)
            key_bias = self.bias_proj(x).permute(0, 2, 1)
            if self.null_k is not None:
                key_bias = torch.cat([self.null_beta.expand(B, -1, 1), key_bias], dim=2)
        else:
            key_bias = None

        if record_stats:
            with torch.no_grad():
                sqc = torch.sqrt(c)  # (1, H, 1, 1)
                norm_q = q.norm(dim=-1, keepdim=True)
                norm_k = k.norm(dim=-1, keepdim=True)
                u_q = (sqc * torch.asinh(sqc * norm_q)).squeeze(-1)  # (B, H, T)
                u_k = (sqc * torch.asinh(sqc * norm_k)).squeeze(-1)  # (B, H, T)

                self.last_stats = {
                    "c": c.squeeze().tolist() if H > 1 else [c.squeeze().item()],
                    "u_q": u_q.cpu(),
                    "u_k": u_k.cpu(),
                    "q": q.cpu(),
                    "k": k.cpu(),
                    "th_q": th_q.cpu() if th_q is not None else None,
                    "th_k": th_k.cpu() if th_k is not None else None,
                    "key_bias": key_bias.cpu() if key_bias is not None else None,
                    "tau": tau.squeeze().tolist() if H > 1 else [tau.squeeze().item()],
                    "kappa": kappa.squeeze().tolist() if (kappa is not None and H > 1) else ([kappa.squeeze().item()] if kappa is not None else None),
                }

        orig_tf32 = None
        if self.is_tf32 and torch.cuda.is_available():
            orig_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True

        o = yggdrasil_attention(
            q,
            k,
            v,
            c=c,
            tau=tau,
            theta_q=th_q,
            theta_k=th_k,
            kappa=kappa,
            key_bias=key_bias,
            attn_mask=attn_mask,
            is_causal=effective_causal,
            aggregate=self.cfg.aggregate,
            compute_dtype=self.compute_dtype,
        )

        if orig_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = orig_tf32

        return self.out(o.permute(0, 2, 1, 3).reshape(B, T, H * DH))
