"""
Rotary Position Embeddings (RoPE) for length-generalizing causal attention
without polluting token content embeddings with additive positional offsets.
"""

from typing import Tuple
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last dimension by splitting into pairs: [-x2, x1]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Cached cos/sin RoPE table supporting dynamic arbitrary sequence lengths."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(device))  # [L, dh/2]
        emb = torch.cat([freqs, freqs], dim=-1)             # [L, dh]
        cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, L, dh]
        sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, L, dh]
        return cos, sin

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_offset: int = 0,
        k_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to q [B, H, Lq, dh] and k [B, H, Lk, dh].
        """
        cos_q, sin_q = self.get_cos_sin(q.size(2), q.device, q.dtype, offset=q_offset)
        cos_k, sin_k = self.get_cos_sin(k.size(2), k.device, k.dtype, offset=k_offset)
        q_rot = (q * cos_q) + (rotate_half(q) * sin_q)
        k_rot = (k * cos_k) + (rotate_half(k) * sin_k)
        return q_rot, k_rot

