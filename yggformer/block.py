"""
YggBlock: Transformer block with pre-RMSNorm and clean residual stream.
"""

import torch
import torch.nn as nn

from .config import YggCfg
from .attention import YggMultiHeadAttention


class YggBlock(nn.Module):
    """
    Standard pre-norm transformer block:
    x = x + Attention(RMSNorm(x))
    x = x + MLP(RMSNorm(x))
    """

    def __init__(self, cfg: YggCfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        ffn_dim = int(cfg.d_model * cfg.ffn_mult)

        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = YggMultiHeadAttention(cfg)

        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, cfg.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor, is_causal: bool = True, record_stats: bool = False) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), is_causal=is_causal, record_stats=record_stats)
        x = x + self.mlp(self.norm2(x))
        return x
