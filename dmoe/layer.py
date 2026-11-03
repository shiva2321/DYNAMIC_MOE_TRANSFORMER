from typing import Tuple

import torch
import torch.nn as nn

from dmoe.complex_ops import ComplexLayerNorm
from dmoe.attention import ComplexMultiHeadAttention
from dmoe.experts import DynamicMorphicMoE


class DynamicMorphicTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, initial_experts: int = 4):
        super().__init__()
        self.d_model = d_model
        self.norm1 = ComplexLayerNorm(d_model)
        self.attn = ComplexMultiHeadAttention(d_model, num_heads)

        self.norm2 = ComplexLayerNorm(d_model)
        self.moe = DynamicMorphicMoE(d_model, d_ff, initial_experts=initial_experts)

        self.alpha_residual = nn.Parameter(torch.ones(1))
        self.register_buffer("layer_salience", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n1_r, n1_i = self.norm1(z_r, z_i)
        att_r, att_i = self.attn(n1_r, n1_i)
        mid_r = z_r + self.alpha_residual * att_r
        mid_i = z_i + self.alpha_residual * att_i

        n2_r, n2_i = self.norm2(mid_r, mid_i)
        moe_r, moe_i, entropy = self.moe(n2_r, n2_i)
        out_r = mid_r + self.alpha_residual * moe_r
        out_i = mid_i + self.alpha_residual * moe_i

        if self.training:
            with torch.no_grad():
                diff_norm = torch.sqrt((out_r - z_r) ** 2 + (out_i - z_i) ** 2 + 1e-8).mean()
                in_norm = torch.sqrt(z_r ** 2 + z_i ** 2 + 1e-8).mean()
                salience = diff_norm / (in_norm + 1e-5)
                self.layer_salience.mul_(0.95).add_(salience * 0.05)

        return out_r, out_i, entropy

