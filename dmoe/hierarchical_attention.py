"""
Three-Level Hierarchical Causal Attention (Local + Mid/Chunk + Global/Memory).

1. Level 1 — Local Window Attention (with Causal RoPE):
   Strictly causal sliding-window attention (radius `local_window`), focusing on
   fine-grained immediate syntax and neighbor bindings without polluting residual embeddings
   with additive positional vectors.
2. Level 2 — Mid-Level Macro-Chunk Relational Attention (with Chunk-Level RoPE):
   Pools tokens into macro-chunks of size `chunk_stride`, runs strictly causal attention
   from each token to prior *completed* macro-chunks (zero future-token leakage).
3. Level 3 — Global Memory Cross-Attention (with Null-Key Sink):
   Cross-attends each token to external Long-Term Memory slots and/or Short-Term
   working-memory summaries, with a learnable null key (zero value) so unmatched queries
   safely suppress spurious memory injection.

An adaptive per-token softmax gate weights [Local, Mid, Global] dynamically.
"""

import functools
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.rope import RotaryEmbedding, rotate_half


@functools.lru_cache(maxsize=64)
def _get_local_causal_mask(seq_len: int, window: int, device_str: str) -> torch.Tensor:
    dev = torch.device(device_str)
    pos = torch.arange(seq_len, device=dev)
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    return ((diff >= 0) & (diff < window)).unsqueeze(0).unsqueeze(0)


@functools.lru_cache(maxsize=64)
def _get_mid_chunk_idx_and_mask(seq_len: int, stride: int, device_str: str) -> Tuple[torch.Tensor, torch.Tensor]:
    dev = torch.device(device_str)
    n_full = seq_len // stride
    tok_chunk = torch.arange(seq_len, device=dev) // stride
    key_idx = torch.arange(n_full + 1, device=dev)
    allowed = (key_idx.unsqueeze(0) <= tok_chunk.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    return tok_chunk, allowed


class ThreeLevelAttention(nn.Module):
    """Causal 3-level hierarchical attention with RoPE and per-token adaptive level gating."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        local_window: int = 4,
        chunk_stride: int = 4,
        use_rope: bool = True,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.local_window = max(1, int(local_window))
        self.chunk_stride = max(2, int(chunk_stride))
        self.scale = self.head_dim ** -0.5
        self.use_rope = bool(use_rope) and (self.head_dim % 2 == 0)
        self.rope = RotaryEmbedding(self.head_dim) if self.use_rope else None

        # Level 1: Local causal window QKV
        self.q_local = nn.Linear(d_model, d_model)
        self.k_local = nn.Linear(d_model, d_model)
        self.v_local = nn.Linear(d_model, d_model)

        # Level 2: Mid-level macro-chunk summary & causal relational QKV
        self.mid_pool_proj = nn.Linear(d_model, d_model)
        self.q_mid = nn.Linear(d_model, d_model)
        self.k_mid = nn.Linear(d_model, d_model)
        self.v_mid = nn.Linear(d_model, d_model)
        self.mid_bos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Level 3: Global memory cross-attention QKV + learnable null-key sink
        self.q_glob = nn.Linear(d_model, d_model)
        self.k_glob = nn.Linear(d_model, d_model)
        self.v_glob = nn.Linear(d_model, d_model)
        self.glob_null_key = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim) * 0.02)
        self.glob_fallback = nn.Parameter(torch.zeros(1, 1, d_model))

        # Per-token adaptive level gate: [Local, Mid, Global]
        self.level_gate = nn.Linear(d_model, 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        B, S, _ = t.shape
        return t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, t: torch.Tensor) -> torch.Tensor:
        B, _, S, _ = t.shape
        return t.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def _local_from_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._split(q)
        k = self._split(k)
        v = self._split(v)
        if self.rope is not None:
            q, k = self.rope.apply_rotary(q, k)
        mask = _get_local_causal_mask(seq_len, self.local_window, str(q.device))
        if key_mask is not None:
            # Causal key gate: suppress corrupted past keys (s < q) and any rejected antecedent key (s + 1 < q)
            # without altering self (s == q) or leaking future tokens into s.
            key_gate = key_mask[:, :seq_len].to(device=q.device, dtype=q.dtype).clamp(min=1e-4, max=1.0)
            log_gate = torch.log(key_gate).view(q.size(0), 1, 1, seq_len)
            log_next = F.pad(log_gate[..., 1:], (0, 1), value=0.0)
            past_only = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=-1)
            pair_past = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=-2)
            bias = torch.zeros(q.size(0), 1, seq_len, seq_len, device=q.device, dtype=q.dtype)
            bias = bias.masked_fill(~mask, -1e9)
            zeros = torch.zeros_like(log_gate)
            bias = bias + torch.where(past_only.view(1, 1, seq_len, seq_len), log_gate, zeros)
            bias = bias + torch.where(pair_past.view(1, 1, seq_len, seq_len), log_next, zeros)
            return self._merge(F.scaled_dot_product_attention(q, k, v, attn_mask=bias))
        return self._merge(F.scaled_dot_product_attention(q, k, v, attn_mask=mask))

    def _local_attention(
        self, x: torch.Tensor, key_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, D = x.shape
        w = torch.cat([self.q_local.weight, self.k_local.weight, self.v_local.weight], dim=0)
        b = torch.cat([self.q_local.bias, self.k_local.bias, self.v_local.bias], dim=0)
        q, k, v = F.linear(x, w, b).split(D, dim=-1)
        return self._local_from_qkv(q, k, v, L, key_mask=key_mask)

    def _mid_from_q(self, x: torch.Tensor, q_mid_proj: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        s = self.chunk_stride
        n_full = L // s
        if n_full == 0:
            return self.mid_bos.expand(B, L, D)

        chunks = x[:, : n_full * s, :].view(B, n_full, s, D)
        macro = self.mid_pool_proj(chunks.mean(dim=2))  # [B, n_full, D]
        macro_with_bos = torch.cat([self.mid_bos.expand(B, 1, D), macro], dim=1)  # [B, n_full + 1, D]

        w_kv = torch.cat([self.k_mid.weight, self.v_mid.weight], dim=0)
        b_kv = torch.cat([self.k_mid.bias, self.v_mid.bias], dim=0)
        k_raw, v_raw = F.linear(macro_with_bos, w_kv, b_kv).split(D, dim=-1)

        q = self._split(q_mid_proj)     # [B, H, L, dh]
        k = self._split(k_raw)          # [B, H, n_full + 1, dh]
        v = self._split(v_raw)

        tok_chunk, allowed = _get_mid_chunk_idx_and_mask(L, s, str(x.device))
        if self.rope is not None:
            # Rotate queries by their chunk index so relative chunk distance matches RoPE phase
            cos_all, sin_all = self.rope.get_cos_sin(n_full + 1, x.device, x.dtype)
            cos_q = cos_all[:, :, tok_chunk, :]
            sin_q = sin_all[:, :, tok_chunk, :]
            q = (q * cos_q) + (rotate_half(q) * sin_q)
            k = (k * cos_all) + (rotate_half(k) * sin_all)

        return self._merge(F.scaled_dot_product_attention(q, k, v, attn_mask=allowed))

    def _mid_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self._mid_from_q(x, self.q_mid(x))

    def _global_attention(
        self,
        x: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        if memory_context is None or memory_context.numel() == 0:
            return self.glob_fallback.expand(B, L, D)

        if memory_context.dim() == 2:
            mem = memory_context.unsqueeze(0).expand(B, -1, -1)
        else:
            mem = memory_context

        q = self._split(self.q_glob(x))
        k = self._split(self.k_glob(mem))
        v = self._split(self.v_glob(mem))

        # Concatenate a null key with zero value so unmatched queries abstain cleanly
        k_null = self.glob_null_key.expand(B, -1, 1, -1)
        v_null = v.new_zeros(B, self.num_heads, 1, self.head_dim)
        k_ext = torch.cat([k, k_null], dim=2)
        v_ext = torch.cat([v, v_null], dim=2)

        scores = (q @ k_ext.transpose(-2, -1)) * self.scale
        scores_null = scores[..., -1:] + math.log(max(mem.size(1), 1))
        scores = torch.cat([scores[..., :-1], scores_null], dim=-1)
        attn = F.softmax(scores, dim=-1)
        return self._merge(attn @ v_ext)

    def forward(
        self,
        x: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, L, d_model]
        memory_context: optional [M, d_model] or [B, M, d_model] global memory bank.
        key_mask: optional [B, L] causal key keep-gate in (0, 1] suppressing corrupted keys (s < t).
        Returns:
          out: [B, L, d_model]
          level_weights: [B, L, 3] (weights for [Local, Mid, Global])
        """
        B, L, D = x.shape
        w_all = torch.cat(
            [self.q_local.weight, self.k_local.weight, self.v_local.weight, self.q_mid.weight],
            dim=0,
        )
        b_all = torch.cat(
            [self.q_local.bias, self.k_local.bias, self.v_local.bias, self.q_mid.bias],
            dim=0,
        )
        ql, kl, vl, qm = F.linear(x, w_all, b_all).split(D, dim=-1)
        out_local = self._local_from_qkv(ql, kl, vl, L, key_mask=key_mask)
        x_mid = (x * key_mask.unsqueeze(-1).to(dtype=x.dtype)) if key_mask is not None else x
        out_mid = self._mid_from_q(x_mid, qm)

        gate_logits = self.level_gate(x)
        if memory_context is None or memory_context.numel() == 0:
            weights_2 = F.softmax(gate_logits[..., :2], dim=-1)
            fused = weights_2[..., 0:1] * out_local + weights_2[..., 1:2] * out_mid
            return self.out_proj(fused), F.pad(weights_2, (0, 1))

        out_glob = self._global_attention(x, memory_context=memory_context)
        weights = F.softmax(gate_logits, dim=-1)
        fused = (
            weights[..., 0:1] * out_local
            + weights[..., 1:2] * out_mid
            + weights[..., 2:3] * out_glob
        )
        return self.out_proj(fused), weights
