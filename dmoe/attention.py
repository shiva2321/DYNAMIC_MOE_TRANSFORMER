import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.complex_ops import ComplexLinear


class ComplexMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = ComplexLinear(d_model, d_model)
        self.k_proj = ComplexLinear(d_model, d_model)
        self.v_proj = ComplexLinear(d_model, d_model)
        self.out_proj = ComplexLinear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        # Learnable per-head logit log-temperature (initialized to 0 -> exp(0)=1, zero RNG used)
        self.log_temp = nn.Parameter(torch.zeros(num_heads, 1, 1))

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, S, _ = t.shape
        return t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def compute_attention_weights(
        self,
        z_r: torch.Tensor,
        z_i: torch.Tensor,
        causal: bool = False,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        qr, qi = self.q_proj(z_r, z_i)
        kr, ki = self.k_proj(z_r, z_i)
        qr, qi = self._split_heads(qr), self._split_heads(qi)
        kr, ki = self._split_heads(kr), self._split_heads(ki)

        if kv_cache is not None:
            if "kr" in kv_cache and "ki" in kv_cache:
                kr = torch.cat([kv_cache["kr"], kr], dim=2)
                ki = torch.cat([kv_cache["ki"], ki], dim=2)
            kv_cache["kr"] = kr
            kv_cache["ki"] = ki

        temp_scale = self.scale * torch.exp(self.log_temp.clamp(-3.0, 3.0))
        hermitian_real = (
            torch.matmul(qr, kr.transpose(-2, -1)) + torch.matmul(qi, ki.transpose(-2, -1))
        ) * temp_scale

        if causal:
            q_len, k_len = hermitian_real.size(-2), hermitian_real.size(-1)
            causal_mask = torch.ones((q_len, k_len), dtype=torch.bool, device=z_r.device).triu(
                diagonal=1 + (k_len - q_len)
            )
            hermitian_real = hermitian_real.masked_fill(causal_mask, float("-inf"))

        return F.softmax(hermitian_real, dim=-1)

    def forward(
        self,
        z_r: torch.Tensor,
        z_i: torch.Tensor,
        causal: bool = False,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = z_r.shape

        attn_weights = self.compute_attention_weights(z_r, z_i, causal=causal, kv_cache=kv_cache)
        vr, vi = self.v_proj(z_r, z_i)
        vr, vi = self._split_heads(vr), self._split_heads(vi)

        if kv_cache is not None:
            if "vr" in kv_cache and "vi" in kv_cache:
                vr = torch.cat([kv_cache["vr"], vr], dim=2)
                vi = torch.cat([kv_cache["vi"], vi], dim=2)
            kv_cache["vr"] = vr
            kv_cache["vi"] = vi

        out_r = torch.matmul(attn_weights, vr)
        out_i = torch.matmul(attn_weights, vi)

        def combine_heads(t: torch.Tensor) -> torch.Tensor:
            return t.transpose(1, 2).contiguous().view(B, S, D)

        out_r, out_i = combine_heads(out_r), combine_heads(out_i)
        return self.out_proj(out_r, out_i)

