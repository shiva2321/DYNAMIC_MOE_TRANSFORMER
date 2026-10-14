"""
SphericalStream: Unit-hypersphere residual stream and nGPT transformer blocks.
Supports:
- nGPT (arXiv:2410.01131, 2608.01284): Unit-norm embeddings, hidden representations,
  and linear weight matrices; learned per-dimension step sizes (eigen learning rates)
  and feature scaling factors; post-step weight renormalization.
- ngpt_stream_only: Unit-norm residual stream with unconstrained weights.
- base: Standard pre-RMSNorm Transformer with RoPE and SwiGLU.
- base_qknorm: base with QK-normalization.
- base_scaled: Isolating control arm with identical learnable scales (alpha, s_qk, etc.)
  but no hypersphere normalizations.
- ngpt_salience: Spherical stream paired with an explicit scalar salience channel (Atlas P-1).
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Projects tensor onto the unit hypersphere along dimension `dim` in fp32 for precision."""
    orig_dtype = x.dtype
    x_fp32 = x.float()
    norm = torch.linalg.vector_norm(x_fp32, dim=dim, keepdim=True)
    return (x_fp32 / (norm + eps)).to(orig_dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for sequence length up to max_seq_len."""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: (B, H, T, d_head)
        T = q.shape[2]
        if T > self.cos_cached.shape[0]:
            self._build_cache(max(T, self.cos_cached.shape[0] * 2))
        cos = self.cos_cached[:T].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:T].to(dtype=q.dtype, device=q.device)
        # Apply rotation
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)


class SphericalAttention(nn.Module):
    """
    Attention block for nGPT / SphericalStream.
    Normalized Q, K, V with learned temperature s_qk, RoPE, and learned step size alpha_A.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 512,
        alpha_init: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

        # Learnable scaling for QK dot product (initialized to 1 / sqrt(d_k))
        self.s_qk = nn.Parameter(torch.tensor(1.0 / math.sqrt(self.head_dim)))
        # Learnable per-dimension step size on residual update
        self.alpha_A = nn.Parameter(torch.full((d_model,), alpha_init))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x is unit-norm on S^{d-1}: (B, T, d)
        B, T, D = x.shape
        q = self.w_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Unit-normalize along head dimension
        q = l2_norm(q, dim=-1)
        k = l2_norm(k, dim=-1)
        v = l2_norm(v, dim=-1)

        # RoPE (orthogonal, preserves unit norm)
        q, k = self.rope(q, k)

        # Scaled dot product
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.s_qk
        # Causal mask
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)

        # Value aggregation
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        h_A = l2_norm(self.w_o(out), dim=-1)

        # Spherical residual update: x + alpha * (h_A - x) projected to S^{d-1}
        delta = h_A - x
        x_next = l2_norm(x + self.alpha_A * delta, dim=-1)
        return x_next, h_A


class SphericalMLP(nn.Module):
    """
    MLP block for nGPT / SphericalStream.
    SwiGLU with normalized gates, learned scaling factors s_u, s_v, and learned step size alpha_M.
    """
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        alpha_init: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp

        self.w_gate = nn.Linear(d_model, d_mlp, bias=False)
        self.w_up = nn.Linear(d_model, d_mlp, bias=False)
        self.w_down = nn.Linear(d_mlp, d_model, bias=False)

        # Learnable scaling vectors for MLP branches
        self.s_u = nn.Parameter(torch.ones(d_mlp))
        self.s_v = nn.Parameter(torch.ones(d_mlp))
        # Learnable per-dimension step size
        self.alpha_M = nn.Parameter(torch.full((d_model,), alpha_init))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x is unit-norm on S^{d-1}
        u = l2_norm(self.w_gate(x), dim=-1) * self.s_u
        v = l2_norm(self.w_up(x), dim=-1) * self.s_v
        m = F.silu(u) * v

        h_M = l2_norm(self.w_down(m), dim=-1)
        delta = h_M - x
        x_next = l2_norm(x + self.alpha_M * delta, dim=-1)
        return x_next, h_M


class SphericalBlock(nn.Module):
    """Transformer block with spherical attention and spherical MLP."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        max_seq_len: int = 512,
        alpha_init: float = 0.05,
    ):
        super().__init__()
        self.attn = SphericalAttention(d_model, n_heads, max_seq_len=max_seq_len, alpha_init=alpha_init)
        self.mlp = SphericalMLP(d_model, d_mlp, alpha_init=alpha_init)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, h_A = self.attn(x)
        x, h_M = self.mlp(x)
        return x, h_A, h_M


class SphericalStreamLM(nn.Module):
    """
    Decoder-only Language Model with Spherical Residual Stream.
    Supports:
    - mode="ngpt": full nGPT (unit-norm weights, unit-norm stream, learned scales).
    - mode="ngpt_stream_only": unit-norm stream, unconstrained weights.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_mlp: int = 768,
        max_seq_len: int = 512,
        alpha_init: float = 0.05,
        mode: str = "ngpt",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.mode = mode

        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            SphericalBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_mlp=d_mlp,
                max_seq_len=max_seq_len,
                alpha_init=alpha_init,
            )
            for _ in range(n_layers)
        ])

        # Vocabulary projection
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Learnable logit scaling parameter
        self.s_z = nn.Parameter(torch.tensor(math.sqrt(d_model)))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.weight.shape[-1]))
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.weight.shape[-1]))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # Embed and normalize to S^{d-1}
        x = l2_norm(self.embed(idx), dim=-1)
        for block in self.blocks:
            x, _, _ = block(x)
        logits = self.head(x) * self.s_z
        return logits

    def forward_with_intermediates(self, idx: torch.Tensor) -> List[torch.Tensor]:
        """Returns stream states at input and after each transformer block."""
        states = []
        x = l2_norm(self.embed(idx), dim=-1)
        states.append(x)
        for block in self.blocks:
            x, _, _ = block(x)
            states.append(x)
        return states


# -------------------------------------------------------------------------------------------------
# Baseline Transformer (base, base_qknorm, base_scaled)
# -------------------------------------------------------------------------------------------------

class BaselineAttention(nn.Module):
    """Attention block for baseline Transformer, with support for qknorm and scaled modes."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 512,
        mode: str = "base",
        alpha_init: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.mode = mode

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

        if mode == "base_scaled":
            self.s_qk = nn.Parameter(torch.tensor(1.0 / math.sqrt(self.head_dim)))
            self.alpha_A = nn.Parameter(torch.full((d_model,), alpha_init))

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        B, T, D = x_norm.shape
        q = self.w_q(x_norm).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x_norm).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x_norm).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.mode == "base_qknorm":
            q = l2_norm(q, dim=-1)
            k = l2_norm(k, dim=-1)

        q, k = self.rope(q, k)

        if self.mode == "base_scaled":
            scale = self.s_qk
        else:
            scale = 1.0 / math.sqrt(self.head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x_norm.device), diagonal=1)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        h_A = self.w_o(out)
        return h_A


class BaselineMLP(nn.Module):
    """MLP block for baseline Transformer, with support for scaled mode."""
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        mode: str = "base",
        alpha_init: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.mode = mode

        self.w_gate = nn.Linear(d_model, d_mlp, bias=False)
        self.w_up = nn.Linear(d_model, d_mlp, bias=False)
        self.w_down = nn.Linear(d_mlp, d_model, bias=False)

        if mode == "base_scaled":
            self.s_u = nn.Parameter(torch.ones(d_mlp))
            self.s_v = nn.Parameter(torch.ones(d_mlp))
            self.alpha_M = nn.Parameter(torch.full((d_model,), alpha_init))

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        u = self.w_gate(x_norm)
        v = self.w_up(x_norm)
        if self.mode == "base_scaled":
            u = u * self.s_u
            v = v * self.s_v
        m = F.silu(u) * v
        h_M = self.w_down(m)
        return h_M


class BaselineBlock(nn.Module):
    """Standard Transformer block with pre-RMSNorm and SwiGLU MLP."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        max_seq_len: int = 512,
        mode: str = "base",
        alpha_init: float = 0.05,
    ):
        super().__init__()
        self.mode = mode
        self.norm_1 = RMSNorm(d_model)
        self.attn = BaselineAttention(d_model, n_heads, max_seq_len=max_seq_len, mode=mode, alpha_init=alpha_init)
        self.norm_2 = RMSNorm(d_model)
        self.mlp = BaselineMLP(d_model, d_mlp, mode=mode, alpha_init=alpha_init)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_A = self.attn(self.norm_1(x))
        if self.mode == "base_scaled":
            x = x + self.attn.alpha_A * h_A
        else:
            x = x + h_A

        h_M = self.mlp(self.norm_2(x))
        if self.mode == "base_scaled":
            x = x + self.mlp.alpha_M * h_M
        else:
            x = x + h_M
        return x, h_A, h_M


class BaselineLM(nn.Module):
    """
    Decoder-only Language Model baseline.
    Supports:
    - mode="base": Standard RoPE + pre-RMSNorm + SwiGLU.
    - mode="base_qknorm": base with QK-normalization.
    - mode="base_scaled": Isolating control with identical learnable scales (alpha, s_qk, etc.).
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_mlp: int = 768,
        max_seq_len: int = 512,
        alpha_init: float = 0.05,
        mode: str = "base",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.mode = mode

        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            BaselineBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_mlp=d_mlp,
                max_seq_len=max_seq_len,
                mode=mode,
                alpha_init=alpha_init,
            )
            for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if mode == "base_scaled":
            self.s_z = nn.Parameter(torch.tensor(math.sqrt(d_model)))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.weight.shape[-1]))
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.weight.shape[-1]))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for block in self.blocks:
            x, _, _ = block(x)
        x_norm = self.norm_out(x)
        logits = self.head(x_norm)
        if self.mode == "base_scaled":
            logits = logits * self.s_z
        return logits

    def forward_with_intermediates(self, idx: torch.Tensor) -> List[torch.Tensor]:
        states = []
        x = self.embed(idx)
        states.append(x)
        for block in self.blocks:
            x, _, _ = block(x)
            states.append(x)
        return states


# -------------------------------------------------------------------------------------------------
# Spherical Stream with Explicit Salience Channel (Atlas P-1)
# -------------------------------------------------------------------------------------------------

class SphericalStreamWithSalienceLM(nn.Module):
    """
    Spherical stream paired with an explicit scalar salience channel r in R+.
    Addresses Atlas P-1 (Sphere for state, scalar for importance).
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_mlp: int = 768,
        max_seq_len: int = 512,
        alpha_init: float = 0.05,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.salience_embed = nn.Embedding(vocab_size, 1)

        self.blocks = nn.ModuleList([
            SphericalBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_mlp=d_mlp,
                max_seq_len=max_seq_len,
                alpha_init=alpha_init,
            )
            for _ in range(n_layers)
        ])

        # Scalar update gates
        self.salience_gates = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model)) for _ in range(n_layers)
        ])

        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.s_z = nn.Parameter(torch.tensor(math.sqrt(d_model)))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_salience(idx)
        return logits

    def forward_with_salience(self, idx: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = l2_norm(self.embed(idx), dim=-1)
        r = F.softplus(self.salience_embed(idx)) + 1e-4

        for l_idx, block in enumerate(self.blocks):
            x, h_A, _ = block(x)
            # Salience update via scalar gate
            gate = torch.sigmoid((h_A * self.salience_gates[l_idx]).sum(dim=-1, keepdim=True))
            r = r * (0.5 + gate)

        logits = self.head(x) * self.s_z
        return logits, (x, r)


# -------------------------------------------------------------------------------------------------
# Weight Normalization & Optimization Hooks
# -------------------------------------------------------------------------------------------------

@torch.no_grad()
def normalize_weights_(model: nn.Module):
    """
    Normalizes every linear weight matrix row and embedding matrix row to unit L2 norm.
    Required by nGPT after each optimizer step.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            w = m.weight.data
            norm = torch.linalg.vector_norm(w.float(), dim=-1, keepdim=True).clamp_min_(1e-12).to(w.dtype)
            w.div_(norm)
        elif isinstance(m, nn.Embedding):
            w = m.weight.data
            norm = torch.linalg.vector_norm(w.float(), dim=-1, keepdim=True).clamp_min_(1e-12).to(w.dtype)
            w.div_(norm)


def compute_speedup(
    eval_steps: List[int],
    val_losses_base: List[float],
    val_losses_arm: List[float],
    step_times_base: List[float],
    step_times_arm: List[float],
    tokens_per_step: int = 8192,
) -> Dict[str, Any]:
    """
    Computes token and wall-clock speedup using a 3-point moving average
    over validation loss checkpoints.
    Speedup is defined as: tokens/time needed for arm to reach base final validation loss.
    """
    assert len(eval_steps) == len(val_losses_base) == len(val_losses_arm)
    assert len(eval_steps) == len(step_times_base) == len(step_times_arm)
    N = len(eval_steps)
    if N < 3:
        raise ValueError("At least 3 evaluation checkpoints required for smoothing")

    # 3-point moving average smoothing
    def smooth(vals: List[float]) -> List[float]:
        res = []
        for i in range(len(vals)):
            if i == 0:
                res.append(vals[0])
            elif i == 1:
                res.append((vals[0] + vals[1]) / 2.0)
            else:
                res.append((vals[i - 2] + vals[i - 1] + vals[i]) / 3.0)
        return res

    smoothed_base = smooth(val_losses_base)
    smoothed_arm = smooth(val_losses_arm)

    # Target loss: final smoothed validation loss of base
    target_loss = smoothed_base[-1]
    total_tokens_base = eval_steps[-1] * tokens_per_step
    total_time_base = step_times_base[-1]

    # Find earliest step where smoothed arm loss <= target_loss
    target_idx = None
    for idx, loss in enumerate(smoothed_arm):
        if loss <= target_loss:
            target_idx = idx
            break

    if target_idx is not None:
        target_step = eval_steps[target_idx]
        tokens_arm = target_step * tokens_per_step
        time_arm = step_times_arm[target_idx]
        reached_target = True
        token_speedup = float(total_tokens_base / tokens_arm)
        wallclock_speedup = float(total_time_base / time_arm) if time_arm > 0 else 1.0
    else:
        target_step = eval_steps[-1]
        tokens_arm = total_tokens_base
        time_arm = step_times_arm[-1]
        reached_target = False
        token_speedup = 1.0
        wallclock_speedup = 1.0

    return {
        "reached_target": reached_target,
        "target_step": target_step,
        "target_loss": float(target_loss),
        "tokens_base": total_tokens_base,
        "tokens_arm": tokens_arm,
        "time_base": total_time_base,
        "time_arm": time_arm,
        "token_speedup": token_speedup,
        "wallclock_speedup": wallclock_speedup,
    }
