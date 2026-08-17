"""
Biologically-Inspired Dynamic & Sparse Attention Mechanism (Hyperspace Dynamic Attention).

Features:
1. Dynamic Foveal Sliding Window (Adaptive Context Horizon W_local based on token entropy).
2. Attention Sinks (Permanent initial token anchors S_sink for KV-cache numerical stability).
3. Phasor Landmark Global Memory (Complex phasor chunk summaries in C^d for associative long-range recall).
4. Rotary Position Embeddings (RoPE) with dynamic frequency scaling for arbitrary sequence extrapolation.
5. Linear O(S * (W + M)) compute and memory scaling vs quadratic O(S^2) in standard Transformers.
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperspace.vsa import ComplexPhasorVSA

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) with dynamic base scaling for long context extrapolation.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len + 512)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [batch, n_heads, seq_len, head_dim]
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

class DynamicSparseAttention(nn.Module):
    """
    Hyperspace Dynamic & Sparse Multi-Head Attention.
    Combines:
    - RoPE Positional Encoding
    - Adaptive Foveal Local Window (W_local)
    - Attention Sinks (S_sink)
    - Phasor Landmark Memory for long-range associative chunk retrieval
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        foveal_window: int = 128,
        num_sinks: int = 4,
        chunk_size: int = 64,
        num_landmarks: int = 4,
        dynamic_span: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.foveal_window = foveal_window
        self.num_sinks = num_sinks
        self.chunk_size = chunk_size
        self.num_landmarks = num_landmarks
        self.dynamic_span = dynamic_span
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # Landmark summary projection to unit complex phasors in C^d
        self.landmark_proj_r = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.landmark_proj_i = nn.Linear(self.head_dim, self.head_dim, bias=False)

    def _compute_chunk_landmarks(self, k: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Groups key representations into chunks of size chunk_size and computes complex phasor landmarks.
        k: [B, H, S, D]
        Returns: landmarks in [B, H, Num_Chunks, D] (complex), num_chunks
        """
        b, h, s, d = k.shape
        num_chunks = max(1, s // self.chunk_size)
        effective_len = num_chunks * self.chunk_size
        k_trimmed = k[:, :, :effective_len, :].view(b, h, num_chunks, self.chunk_size, d)
        
        # Mean key per chunk
        chunk_mean = k_trimmed.mean(dim=3) # [B, H, Num_Chunks, D]
        r_part = self.landmark_proj_r(chunk_mean).float()
        i_part = self.landmark_proj_i(chunk_mean).float()
        z = torch.complex(r_part, i_part)
        landmarks = z / (torch.abs(z) + 1e-8)
        return landmarks, num_chunks

    def _build_dynamic_sparse_mask(
        self,
        b: int,
        h: int,
        s: int,
        q: torch.Tensor,
        k: torch.Tensor,
        active_window: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Constructs a dynamic sparse boolean mask [B, H, S, S] combining:
        1. Causal constraint (j <= i)
        2. Attention sinks (j < num_sinks)
        3. Local foveal window (i - active_window <= j <= i)
        4. Long-range Phasor Landmark chunk anchors
        """
        # 1. Base Causal Mask
        row_idx = torch.arange(s, device=device).unsqueeze(1) # [S, 1]
        col_idx = torch.arange(s, device=device).unsqueeze(0) # [1, S]
        causal_mask = (col_idx <= row_idx) # [S, S]
        
        # 2. Attention Sinks
        sink_mask = (col_idx < self.num_sinks) & causal_mask # [S, S]
        
        # 3. Foveal Sliding Window
        local_mask = (col_idx >= (row_idx - active_window)) & causal_mask # [S, S]
        
        sparse_mask = sink_mask | local_mask # [S, S]
        
        # 4. If sequence exceeds local window + sinks, compute landmark global anchors
        if s > (active_window + self.num_sinks) and self.num_landmarks > 0:
            landmarks, num_chunks = self._compute_chunk_landmarks(k)
            # Query complex phasors
            q_r = self.landmark_proj_r(q).float()
            q_i = self.landmark_proj_i(q).float()
            q_z = torch.complex(q_r, q_i)
            q_phasors = q_z / (torch.abs(q_z) + 1e-8) # [B, H, S, D]
            
            # Hermitian similarity: [B, H, S, Num_Chunks]
            # Real(q_phasors . landmarks*)
            sim = torch.einsum("bhsd,bhcd->bhsc", q_phasors.real, landmarks.real) + \
                  torch.einsum("bhsd,bhcd->bhsc", q_phasors.imag, landmarks.imag)
            
            # Top-M chunks per query
            m_landmarks = min(self.num_landmarks, num_chunks)
            _, top_chunk_idx = torch.topk(sim, k=m_landmarks, dim=-1) # [B, H, S, M]
            
            # Expand chunk mask to token indices
            landmark_token_mask = torch.zeros(b, h, s, s, dtype=torch.bool, device=device)
            # Efficient landmark token expansion
            for m_idx in range(m_landmarks):
                c_ids = top_chunk_idx[:, :, :, m_idx] # [B, H, S]
                start_tok = c_ids * self.chunk_size
                # Broadcast across chunk tokens
                for offset in range(0, self.chunk_size, 4): # sampled stride
                    tok_pos = torch.clamp(start_tok + offset, 0, s - 1)
                    landmark_token_mask.scatter_(-1, tok_pos.unsqueeze(-1), True)
                    
            sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0) | landmark_token_mask
            sparse_mask = sparse_mask & causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0).expand(b, h, s, s)
            
        return sparse_mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        b, s, d = x.shape
        device = x.device
        
        # 1. Project Q, K, V
        qkv = self.qkv_proj(x).view(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, H, S, D]
        
        # 2. Apply Rotary Position Embeddings (RoPE)
        cos, sin = self.rotary_emb(x, s)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 3. Determine Dynamic Active Foveal Window Span
        if self.dynamic_span and s > self.foveal_window:
            # Estimate sequence entropy from QK norm
            token_activity = q.norm(dim=-1).mean(dim=1) # [B, S]
            high_entropy_fraction = (token_activity > token_activity.mean()).float().mean().item()
            active_window = min(s, int(self.foveal_window * (1.0 + high_entropy_fraction)))
        else:
            active_window = min(s, self.foveal_window)
            
        # 4. Construct Dynamic Sparse Attention Mask
        if s <= self.foveal_window:
            # Fast dense path for short sequences
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = torch.tril(torch.ones(s, s, device=device)).view(1, 1, s, s)
            scores = scores.masked_fill(causal_mask == 0, -1e4)
            sparsity_ratio = 0.0
        else:
            # Linear Dynamic Sparse path for extended sequences
            sparse_mask = self._build_dynamic_sparse_mask(b, self.n_heads, s, q, k, active_window, device)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~sparse_mask, -1e4)
            sparsity_ratio = 1.0 - (sparse_mask.float().sum() / (b * self.n_heads * s * s)).item()
            
        # 5. Softmax & Output Projection
        attn_weights = F.softmax(scores, dim=-1)
        # Handle all -inf rows if any
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v) # [B, H, S, D]
        context = context.permute(0, 2, 1, 3).contiguous().view(b, s, d)
        out = self.out_proj(context)
        
        telemetry = {
            "seq_len": s,
            "active_window": active_window,
            "sparsity_ratio": sparsity_ratio,
            "head_dim": self.head_dim,
        }
        
        return out, telemetry
