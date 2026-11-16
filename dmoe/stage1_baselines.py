"""
Matched, strictly causal Stage 1 reference baselines for fair architecture comparisons:
  1. CausalDenseRoPEBaseline: Pre-norm Causal RoPE Transformer (length-generalizing).
  2. GRUBaseline: Multi-layer PyTorch nn.GRU causal sequence model.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.rope import RotaryEmbedding


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CausalRoPEBlock(nn.Module):
    """Pre-norm Transformer block with strictly causal Multi-Head RoPE Self-Attention and GELU FFN."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope_base: float = 10000.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv_proj(h).view(B, L, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)  # [B, H, L, dh]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        q, k = self.rope.apply_rotary(q, k)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.view(1, 1, L, L), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn_out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.out_proj(attn_out)

        h2 = self.norm2(x)
        x = x + self.ff2(self.act(self.ff1(h2)))
        return x


class CausalDenseRoPEBaseline(nn.Module):
    """Length-generalizing strictly causal RoPE Transformer baseline."""

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 128,
        num_layers: int = 2,
        vocab_size: int = 258,
        out_vocab: int = 256,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            CausalRoPEBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope_base=rope_base)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_vocab)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(self.final_norm(h))
        return {"output": logits, "latent": h}


class GRUBaseline(nn.Module):
    """Strictly causal multi-layer GRU baseline (native PyTorch recurrent baseline)."""

    def __init__(
        self,
        d_model: int = 64,
        num_layers: int = 2,
        vocab_size: int = 258,
        out_vocab: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_vocab)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = self.embed(x)
        h, _ = self.gru(emb)
        logits = self.head(self.final_norm(h))
        return {"output": logits, "latent": h}

