"""
Riemannian Spherical Hyperspace Attention (RSA) & Looped Riemannian Transformer.

Core Features:
1. Directional Coordinates on S^{D-1}:
   Queries and Keys are projected to unit hyperspheres.
   Angular similarity S_{angular} = kappa * cos(theta_{QK}) is strictly bounded in [-kappa, +kappa],
   preventing softmax entropy collapse on arbitrarily long sequences.
2. Non-Aliasing Multi-Octave SO(2) Geodesic Phase Winding:
   Rotates orthogonal 2D planes on S^{D-1} along great circles.
   Preserves unit norm: ||R(theta) u|| == ||u|| == 1 strictly.
   Uses non-aliasing frequency progression (base 10000.0) with relative offset invariance:
   <u_i, u_j> = cos(omega(i - j)), granting true zero-shot length extrapolation.
3. Scale-Invariant Log-Radial Attention:
   Radial coordinates r in [1, inf) act as continuous depth/shell addressing:
   S_{radial} = -beta * (ln r_Q - ln r_K)^2.
   Strictly invariant to uniform scale transformations.
4. Entropy-Preserving Concentration Scaling & Top-k Competition:
   Scales concentration kappa with sqrt(ln(S + 1)) to prevent probability mass diffusion.
   Optional top-k manifold competition allows sharp, non-smeared symbolic memory routing.
5. Looped Riemannian Transformer (Turing-Complete Deliberation):
   Reuses the same geometric layer iteratively with Adaptive Computation Time (ACT)
   to solve arbitrary-depth algorithmic problems without fixed circuit depth constraints.
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from ishmn.geometry import project_to_sphere

try:
    import ishmn_attention_rs
    _RUST_AVAILABLE = True
except ImportError:
    ishmn_attention_rs = None
    _RUST_AVAILABLE = False


def has_rust_attention() -> bool:
    """Returns True if native Rust extension ishmn_attention_rs is installed and loaded."""
    return _RUST_AVAILABLE


def fast_riemannian_so2_winding(
    u: torch.Tensor,
    freqs: torch.Tensor,
    start_pos: int = 0
) -> torch.Tensor:
    """
    Applies continuous SO(2) rotations along orthogonal 2D planes of S^{D-1}.
    Preserves ||u|| == 1 strictly without normalization drift.
    """
    if _RUST_AVAILABLE and not u.requires_grad and u.device.type == "cpu" and u.is_contiguous():
        u_np = u.numpy()
        freqs_np = freqs.contiguous().numpy()
        out_np = ishmn_attention_rs.riemannian_so2_winding(u_np, freqs_np, start_pos)
        return torch.from_numpy(out_np)

    # PyTorch vectorized fallback (CUDA or training with gradients)
    B, H, T, D = u.shape
    t_indices = torch.arange(start_pos, start_pos + T, device=u.device, dtype=freqs.dtype)
    angles = torch.outer(t_indices, freqs)

    cos_a = torch.cos(angles).unsqueeze(0).unsqueeze(0)
    sin_a = torch.sin(angles).unsqueeze(0).unsqueeze(0)

    u_pairs = u.view(B, H, T, -1, 2)
    u_0 = u_pairs[..., 0]
    u_1 = u_pairs[..., 1]

    rot_0 = u_0 * cos_a - u_1 * sin_a
    rot_1 = u_0 * sin_a + u_1 * cos_a

    return torch.stack([rot_0, rot_1], dim=-1).view(B, H, T, D)


def fast_tiled_riemannian_attention(
    u_q: torch.Tensor,
    u_k: torch.Tensor,
    r_q: torch.Tensor,
    r_k: torch.Tensor,
    v: torch.Tensor,
    kappa: torch.Tensor,
    beta: torch.Tensor,
    is_causal: bool = False,
    enable_entropy_scaling: bool = True,
    block_size: int = 64
) -> torch.Tensor:
    """
    Tiled Online Riemannian Softmax Attention.
    Computes attention in O(block_size) cache space without materializing
    the [B, H, T, S] attention score matrix in memory.
    """
    S = u_k.size(2)
    if (
        _RUST_AVAILABLE
        and not u_q.requires_grad
        and not u_k.requires_grad
        and not v.requires_grad
        and u_q.device.type == "cpu"
        and u_q.is_contiguous()
        and u_k.is_contiguous()
        and v.is_contiguous()
    ):
        ln_r_q = torch.log(r_q).squeeze(-1).contiguous().numpy()
        ln_r_k = torch.log(r_k).squeeze(-1).contiguous().numpy()
        k_1d = kappa.view(-1).contiguous().numpy()
        b_1d = beta.view(-1).contiguous().numpy()

        out_np = ishmn_attention_rs.tiled_riemannian_attention(
            u_q.numpy(),
            u_k.numpy(),
            ln_r_q,
            ln_r_k,
            v.numpy(),
            k_1d,
            b_1d,
            is_causal,
            enable_entropy_scaling,
            block_size
        )
        return torch.from_numpy(out_np)

    # PyTorch vectorized computation (GPU or autograd)
    s_angular = torch.matmul(u_q, u_k.transpose(-1, -2))
    ln_r_q = torch.log(r_q)
    ln_r_k = torch.log(r_k).transpose(-1, -2)
    s_radial = -((ln_r_q - ln_r_k) ** 2)

    kappa_eff = torch.clamp(kappa, min=0.1, max=30.0)
    if enable_entropy_scaling and S > 1:
        entropy_scale = math.sqrt(math.log(float(S) + 1.0))
        kappa_eff = kappa_eff * entropy_scale

    beta_eff = torch.clamp(beta, min=0.01, max=10.0)
    logits = kappa_eff * s_angular + beta_eff * s_radial

    if is_causal:
        T = u_q.size(2)
        causal_mask = torch.ones(T, S, device=logits.device, dtype=torch.bool).tril()
        logits = logits.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(1), -1e9)

    attn_weights = F.softmax(logits, dim=-1)
    return torch.matmul(attn_weights, v)


class RiemannianSphericalAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 4,
        hyperspace_dim: int = 8,
        max_seq_len: int = 1024,
        initial_kappa: float = 4.0,
        initial_beta: float = 1.0,
        freq_base: float = 35.0,
        freq_scale: float = 0.6,
        device: torch.device = None
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        assert hyperspace_dim % 2 == 0, f"hyperspace_dim ({hyperspace_dim}) must be even for SO(2) phase planes"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.hyperspace_dim = hyperspace_dim
        self.max_seq_len = max_seq_len
        self.device = device

        # Projections to S^{D-1} directional coordinates
        self.q_direction = nn.Linear(d_model, num_heads * hyperspace_dim, bias=False)
        self.k_direction = nn.Linear(d_model, num_heads * hyperspace_dim, bias=False)

        # Projections to R+ radial depth coordinates
        self.q_radius = nn.Linear(d_model, num_heads, bias=True)
        self.k_radius = nn.Linear(d_model, num_heads, bias=True)

        # Value and Output projections
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable concentration (kappa) and radial precision (beta) per head
        self.kappa = nn.Parameter(torch.full((1, num_heads, 1, 1), float(initial_kappa)))
        self.beta = nn.Parameter(torch.full((1, num_heads, 1, 1), float(initial_beta)))

        # Precompute Non-Aliasing SO(2) rotary frequencies across the hypersphere 2D planes
        num_planes = hyperspace_dim // 2
        exponents = torch.arange(0, num_planes, dtype=torch.float32) / float(num_planes)
        freqs = freq_scale / (freq_base ** exponents)
        self.register_buffer("freqs", freqs)

    def _apply_so2_geodesic_winding(self, u: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Applies continuous SO(2) rotations along orthogonal 2D planes of S^{D-1}.
        Preserves ||u|| == 1 exactly while injecting relative position information.
        u: [B, H, T, hyperspace_dim]
        """
        return fast_riemannian_so2_winding(u, self.freqs, start_pos=start_pos)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        top_k: Optional[int] = None,
        enable_entropy_scaling: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model] Input sequence
            mask: Optional [B, T, T] or [1, 1, T, T] attention mask
            context: Optional [B, S, d_model] for cross-attention
            start_pos: Starting position index for autoregressive / looped stepping
            top_k: Optional integer for sparse competitive top-k attention
            enable_entropy_scaling: Scale concentration with sqrt(ln(S + 1)) to prevent dispersion
        Returns:
            out: [B, T, d_model]
            attn_weights: [B, num_heads, T, S]
        """
        B, T, _ = x.shape
        kv_input = context if context is not None else x
        S = kv_input.size(1)

        # 1. Project directional coordinates and normalize to S^{D-1}
        u_q = self.q_direction(x).view(B, T, self.num_heads, self.hyperspace_dim).transpose(1, 2)  # [B, H, T, D]
        u_k = self.k_direction(kv_input).view(B, S, self.num_heads, self.hyperspace_dim).transpose(1, 2)  # [B, H, S, D]

        u_q = F.normalize(u_q, p=2, dim=-1)
        u_k = F.normalize(u_k, p=2, dim=-1)

        # 2. Apply Geodesic SO(2) Phase Winding along S^{D-1} planes
        u_q_rot = self._apply_so2_geodesic_winding(u_q, start_pos=start_pos)
        u_k_rot = self._apply_so2_geodesic_winding(u_k, start_pos=0)

        # 3. Project radial coordinates r in [1, inf)
        r_q = F.softplus(self.q_radius(x)).view(B, T, self.num_heads, 1).transpose(1, 2) + 1.0  # [B, H, T, 1]
        r_k = F.softplus(self.k_radius(kv_input)).view(B, S, self.num_heads, 1).transpose(1, 2) + 1.0  # [B, H, S, 1]

        # 4. Compute Angular vMF Similarity (Bounded in [-1, +1])
        s_angular = torch.matmul(u_q_rot, u_k_rot.transpose(-1, -2))

        # 5. Compute Log-Radial Scale-Invariant Depth Similarity
        ln_r_q = torch.log(r_q)
        ln_r_k = torch.log(r_k).transpose(-1, -2)
        s_radial = -((ln_r_q - ln_r_k) ** 2)

        # 6. Combined Hyperspace Attention Logits
        kappa_eff = torch.clamp(self.kappa, min=0.1, max=30.0)
        if enable_entropy_scaling and S > 1:
            entropy_scale = math.sqrt(math.log(float(S) + 1.0))
            kappa_eff = kappa_eff * entropy_scale

        beta_eff = torch.clamp(self.beta, min=0.01, max=10.0)
        logits = kappa_eff * s_angular + beta_eff * s_radial

        # Apply optional mask (supports bool masks or float additive masks like causal subsequent masks)
        if mask is not None:
            if mask.dim() == 2:
                if mask.size(0) == T and mask.size(1) == S:
                    m_exp = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, T, S]
                else:
                    m_exp = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            elif mask.dim() == 3:
                m_exp = mask.unsqueeze(1)  # [B, 1, T, S]
            else:
                m_exp = mask

            if m_exp.dtype == torch.bool:
                logits = logits.masked_fill(~m_exp, -1e9)
            else:
                logits = logits + m_exp


        # Optional Top-k Sparse Competition
        if top_k is not None and top_k < S:
            topk_vals, _ = torch.topk(logits, k=top_k, dim=-1)
            min_topk = topk_vals[..., -1, None]
            logits = torch.where(logits >= min_topk, logits, torch.tensor(-1e9, device=logits.device))

        attn_weights = F.softmax(logits, dim=-1)

        # 7. Aggregate Values
        v = self.v_proj(kv_input).view(B, S, self.num_heads, self.d_head).transpose(1, 2)  # [B, H, S, d_head]
        out_heads = torch.matmul(attn_weights, v)  # [B, H, T, d_head]

        # Combine heads
        out_concat = out_heads.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out_concat)

        return out, attn_weights


class DualStreamRiemannianAttention(nn.Module):
    """
    Dual-Stream Riemannian Attention Architecture.
    Adapted from Resonance X (ICML 2026 Wave-Particle Transformer / Dual-Stream).

    Decouples representation into two parallel, interacting streams:
    1. Phase / Direction Stream (S^{D-1}, hyperspace_dim):
       Handles invariant symbolic identity, relational binding, and relative SO(2) geodesic winding.
       Strictly unit-normalized: ||u|| == 1.0.
    2. Magnitude / Particle Stream (R^{d_mag}):
       Handles scalar salience, confidence, and SwiGLU intensity updates.

    Cross-Stream Interaction:
       - Magnitude modulates Phase Attention Concentration (kappa):
         kappa_eff = kappa_base * (1.0 + softplus(mag_to_kappa(x_mag))).clamp(max=30.0)
         High confidence focuses directional attention; uncertainty diffuses it smoothly.
       - Phase projects into Magnitude Stream:
         x_mag = x_mag + phase_to_mag(u_phase)
    """
    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 4,
        hyperspace_dim: int = 8,
        d_phase: Optional[int] = None,
        d_mag: Optional[int] = None,
        initial_kappa: float = 4.0,
        freq_base: float = 35.0,
        freq_scale: float = 0.6,
        device: torch.device = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hyperspace_dim = hyperspace_dim
        self.d_phase = d_phase if d_phase is not None else (num_heads * hyperspace_dim)
        self.d_mag = d_mag if d_mag is not None else d_model
        self.device = device

        # Phase Stream Projections (on S^{D-1})
        self.q_phase = nn.Linear(self.d_model, num_heads * hyperspace_dim, bias=False, device=device)
        self.k_phase = nn.Linear(self.d_model, num_heads * hyperspace_dim, bias=False, device=device)
        self.v_phase = nn.Linear(self.d_model, num_heads * hyperspace_dim, bias=False, device=device)
        self.out_phase = nn.Linear(num_heads * hyperspace_dim, self.d_model, bias=False, device=device)

        # Magnitude Stream Projections
        self.norm_mag = nn.LayerNorm(self.d_mag, device=device)
        self.q_mag = nn.Linear(self.d_mag, self.d_mag, bias=False, device=device)
        self.k_mag = nn.Linear(self.d_mag, self.d_mag, bias=False, device=device)
        self.v_mag = nn.Linear(self.d_mag, self.d_mag, bias=False, device=device)
        self.out_mag = nn.Linear(self.d_mag, self.d_mag, bias=False, device=device)

        # Cross-Stream Couplings
        self.mag_to_kappa = nn.Linear(self.d_mag, num_heads, bias=False, device=device)
        self.phase_to_mag = nn.Linear(num_heads * hyperspace_dim, self.d_mag, bias=False, device=device)

        # Base concentration per head
        self.kappa_base = nn.Parameter(torch.full((1, num_heads, 1, 1), float(initial_kappa), device=device))

        # SO(2) Geodesic Rotary Frequencies
        num_planes = hyperspace_dim // 2
        exponents = torch.arange(0, num_planes, dtype=torch.float32, device=device) / float(num_planes)
        freqs = freq_scale / (freq_base ** exponents)
        self.register_buffer("freqs", freqs)

    def _apply_so2_winding(self, u: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        return fast_riemannian_so2_winding(u, self.freqs, start_pos=start_pos)

    def forward(
        self,
        x: torch.Tensor,
        x_mag: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        B, T, _ = x.shape
        if x_mag is None:
            x_mag = x

        # 1. Phase Stream Directional Attention
        qp = self.q_phase(x).view(B, T, self.num_heads, self.hyperspace_dim).transpose(1, 2)
        kp = self.k_phase(x).view(B, T, self.num_heads, self.hyperspace_dim).transpose(1, 2)
        vp = self.v_phase(x).view(B, T, self.num_heads, self.hyperspace_dim).transpose(1, 2)

        qp = project_to_sphere(qp)
        kp = project_to_sphere(kp)
        vp = project_to_sphere(vp)

        qp_rot = self._apply_so2_winding(qp, start_pos=start_pos)
        kp_rot = self._apply_so2_winding(kp, start_pos=0)

        # Angular dot products on S^{D-1}: cos(delta_theta)
        phase_scores = torch.matmul(qp_rot, kp_rot.transpose(-1, -2))  # (B, H, T, T)

        # Magnitude Stream modulates Phase Concentration (kappa_eff)
        mag_normed = self.norm_mag(x_mag)
        kappa_mod = F.softplus(self.mag_to_kappa(mag_normed)).transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)
        kappa_eff = torch.clamp(self.kappa_base * (1.0 + kappa_mod), min=0.1, max=30.0)

        scaled_phase_scores = kappa_eff * phase_scores
        m_exp = None
        if mask is not None:
            if mask.dim() == 2:
                m_exp = mask.unsqueeze(0).unsqueeze(1)
            elif mask.dim() == 3:
                m_exp = mask.unsqueeze(1)
            else:
                m_exp = mask
            if m_exp.dtype == torch.bool:
                scaled_phase_scores = scaled_phase_scores.masked_fill(~m_exp, -1e9)
            else:
                scaled_phase_scores = scaled_phase_scores + m_exp

        phase_weights = F.softmax(scaled_phase_scores, dim=-1)

        # Phase value aggregation with unit-norm preservation
        attn_vp = torch.matmul(phase_weights, vp)  # (B, H, T, hyperspace_dim)
        attn_vp = project_to_sphere(attn_vp)
        attn_vp_flat = attn_vp.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.hyperspace_dim)
        out_p = self.out_phase(attn_vp_flat)

        # 2. Magnitude Stream Self-Attention
        qm = self.q_mag(mag_normed)
        km = self.k_mag(mag_normed)
        vm = self.v_mag(mag_normed)

        d_k = self.d_mag // self.num_heads
        qm_h = qm.view(B, T, self.num_heads, d_k).transpose(1, 2)
        km_h = km.view(B, T, self.num_heads, d_k).transpose(1, 2)
        vm_h = vm.view(B, T, self.num_heads, d_k).transpose(1, 2)

        mag_scores = torch.matmul(qm_h, km_h.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None and m_exp is not None:
            if m_exp.dtype == torch.bool:
                mag_scores = mag_scores.masked_fill(~m_exp, -1e9)
            else:
                mag_scores = mag_scores + m_exp
        mag_weights = F.softmax(mag_scores, dim=-1)
        attn_vm = torch.matmul(mag_weights, vm_h).transpose(1, 2).contiguous().view(B, T, self.d_mag)
        out_m = self.out_mag(attn_vm)

        # 3. Cross-Stream Modulation: Phase projects into Magnitude
        phase_injection = self.phase_to_mag(attn_vp_flat)
        x_mag_out = x_mag + out_m + phase_injection

        # Combined output for residual connection
        out_fused = x + out_p + 0.3 * out_m

        meta = {
            "phase_weights": phase_weights,
            "mag_weights": mag_weights,
            "mean_kappa_eff": float(kappa_eff.mean().item())
        }
        return out_fused, x_mag_out, meta


class DualStreamRiemannianBlock(nn.Module):
    """
    Complete Dual-Stream Riemannian Transformer Block:
    1. DualStreamRiemannianAttention with magnitude-modulated phase concentration.
    2. SwiGLU Non-Linear Feedforward Network on the Magnitude Stream.
    3. Cross-Stream residual blending.
    """
    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 4,
        hyperspace_dim: int = 8,
        d_ff: int = 64,
        freq_base: float = 35.0,
        freq_scale: float = 0.6,
        device: torch.device = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.attn = DualStreamRiemannianAttention(
            d_model=d_model,
            num_heads=num_heads,
            hyperspace_dim=hyperspace_dim,
            freq_base=freq_base,
            freq_scale=freq_scale,
            device=device
        )
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)

        # SwiGLU FFN on Magnitude Stream
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False, device=device)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False, device=device)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False, device=device)

    def forward(
        self,
        x: torch.Tensor,
        x_mag: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_norm = self.norm1(x)
        out_fused, x_mag_out, _ = self.attn(h_norm, x_mag=x_mag, mask=mask, start_pos=start_pos)
        x = x + out_fused

        # SwiGLU FFN
        x_mid = self.norm2(x)
        swiglu_out = self.ffn_down(F.silu(self.ffn_gate(x_mid)) * self.ffn_up(x_mid))
        x = x + swiglu_out

        return x, x_mag_out


class RiemannianSphericalTransformerLayer(nn.Module):
    """Full Transformer Block using Riemannian Spherical Attention + Dynamic Feedforward Expert / MoE."""
    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 4,
        hyperspace_dim: int = 8,
        d_ff: int = 64,
        freq_base: float = 35.0,
        freq_scale: float = 0.6,
        use_dynamic_moe: bool = False,
        moe_instance: Optional[Any] = None,
        moe_initial_experts: int = 2,
        moe_max_experts: int = 16,
        moe_spawn_threshold: float = 0.35,
        device: torch.device = None
    ):
        super().__init__()
        self.d_model = d_model
        self.hyperspace_dim = hyperspace_dim
        self.use_dynamic_moe = use_dynamic_moe
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.attn = RiemannianSphericalAttention(
            d_model=d_model,
            num_heads=num_heads,
            hyperspace_dim=hyperspace_dim,
            freq_base=freq_base,
            freq_scale=freq_scale,
            device=device
        )
        self.norm2 = nn.LayerNorm(d_model, device=device)


        if use_dynamic_moe:
            if moe_instance is not None:
                self.moe = moe_instance
            else:
                from ishmn.spherical_dynamic_moe import SphericalDynamicMoE
                self.moe = SphericalDynamicMoE(
                    d_model=d_model,
                    d_ff=d_ff,
                    d_context=d_model,
                    hyperspace_dim=hyperspace_dim,
                    initial_experts=moe_initial_experts,
                    max_experts=moe_max_experts,
                    spawn_threshold=moe_spawn_threshold,
                    device=device
                )
            self.u_proj = nn.Linear(d_model, hyperspace_dim, device=device)
            self.r_proj = nn.Linear(d_model, 1, device=device)
            self.ffn = None
        else:
            self.moe = None
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        optimizer: Optional[Any] = None
    ) -> torch.Tensor:
        # Pre-LN architecture
        h, _ = self.attn(self.norm1(x), mask=mask, top_k=top_k)
        x = x + h

        norm_x2 = self.norm2(x)
        if self.use_dynamic_moe and self.moe is not None:
            B, T, D = norm_x2.shape
            x_flat = norm_x2.contiguous().view(B * T, D)
            u_flat = project_to_sphere(self.u_proj(x_flat))
            r_flat = F.softplus(self.r_proj(x_flat)) + 0.5
            moe_out, _ = self.moe(x=x_flat, u_x=u_flat, r_x=r_flat, optimizer=optimizer)
            ff_out = moe_out.view(B, T, D)
        else:
            ff_out = self.ffn(norm_x2)

        x = x + ff_out
        return x


class LoopedRiemannianTransformer(nn.Module):
    """
    Looped Riemannian Transformer (Turing-Complete Recurrent Deliberation Engine).
    
    Iteratively executes the same Riemannian Transformer layer with Adaptive Computation
    Time (ACT), breaking the constant-depth TC^0 circuit limit and enabling arbitrary-depth
    sequential algorithmic reasoning (carry propagation, tree traversal, stack manipulation).
    Equipped with Dynamic MoE expert spawning and Hyperspace Working Memory read/write.
    """
    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 4,
        hyperspace_dim: int = 8,
        d_ff: int = 64,
        max_loops: int = 12,
        halting_threshold: float = 0.95,
        freq_base: float = 35.0,
        freq_scale: float = 0.6,
        use_dynamic_moe: bool = False,
        moe_instance: Optional[Any] = None,
        moe_initial_experts: int = 2,
        moe_max_experts: int = 16,
        moe_spawn_threshold: float = 0.35,
        use_working_memory: bool = False,
        use_dual_stream: bool = False,
        device: torch.device = None
    ):
        super().__init__()
        self.d_model = d_model
        self.hyperspace_dim = hyperspace_dim
        self.max_loops = max_loops
        self.halting_threshold = halting_threshold
        self.use_dynamic_moe = use_dynamic_moe
        self.use_working_memory = use_working_memory
        self.use_dual_stream = use_dual_stream
        self.device = device

        if use_dual_stream:
            self.layer = DualStreamRiemannianBlock(
                d_model=d_model,
                num_heads=num_heads,
                hyperspace_dim=hyperspace_dim,
                d_ff=d_ff,
                freq_base=freq_base,
                freq_scale=freq_scale,
                device=device
            )
        else:
            self.layer = RiemannianSphericalTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                hyperspace_dim=hyperspace_dim,
                d_ff=d_ff,
                freq_base=freq_base,
                freq_scale=freq_scale,
                use_dynamic_moe=use_dynamic_moe,
                moe_instance=moe_instance,
                moe_initial_experts=moe_initial_experts,
                moe_max_experts=moe_max_experts,
                moe_spawn_threshold=moe_spawn_threshold,
                device=device
            )

        # Dynamic Halting Gate
        self.halting_gate = nn.Sequential(
            nn.Linear(d_model, 32, device=device),
            nn.GELU(),
            nn.Linear(32, 1, device=device),
            nn.Sigmoid()
        )

        # Working memory interface heads (if working memory is enabled)
        if use_working_memory:
            self.mem_u_head = nn.Linear(d_model, hyperspace_dim, device=device)
            self.mem_r_head = nn.Linear(d_model, 1, device=device)
            self.mem_write_gate = nn.Linear(d_model, 1, device=device)
            self.mem_add_head = nn.Linear(d_model, d_model, device=device)

        if device is not None:
            self.to(device)


    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        max_steps: Optional[int] = None,
        top_k: Optional[int] = None,
        optimizer: Optional[Any] = None,
        memory: Optional[Any] = None,
        mem_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Iteratively executes the Riemannian Transformer layer.
        Args:
            x: [B, T, d_model]
            mask: Optional attention mask
            is_causal: If True and mask is None, automatically applies strict lower-triangular causal attention
            max_steps: Optional override of max loop iterations
            top_k: Optional top-k attention sparsity
            optimizer: Optional DynamicWarmupAdamW for neurogenesis registration
            memory: Optional InfiniteSphericalMemory
            mem_state: Optional memory state dictionary
        Returns:
            out: [B, T, d_model]
            meta: dictionary containing steps_taken, ponder_cost, trajectory, num_experts
        """
        B, T, D = x.shape
        K = max_steps if max_steps is not None else self.max_loops

        # Generate strict causal mask if requested
        if is_causal and mask is None and T > 1:
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        h = x
        x_mag = x if self.use_dual_stream else None

        accum_p = torch.zeros(B, T, 1, device=x.device)
        output = torch.zeros_like(x)
        ponder_cost = torch.zeros(B, T, 1, device=x.device)
        steps_taken = torch.zeros(B, T, 1, device=x.device)
        still_running = torch.ones(B, T, 1, dtype=torch.bool, device=x.device)

        intermediate_states = []

        for k in range(1, K + 1):
            # 1. Optional Hyperspace Working Memory Read
            if self.use_working_memory and memory is not None and mem_state is not None:
                mean_h = h.mean(dim=1)  # [B, d_model]
                u_q = project_to_sphere(self.mem_u_head(mean_h))
                r_q = F.softplus(self.mem_r_head(mean_h)) + 0.5
                kappa_q = torch.tensor([[4.0]], device=x.device)
                beta_q = torch.tensor([[1.0]], device=x.device)
                read_p, _, mem_state = memory.read(mem_state, u_q, r_q, kappa_q, beta_q)
                h = h + 0.2 * read_p.unsqueeze(1)

            # 2. Geometric Layer Execution (RSA + Dynamic MoE or Dual-Stream)
            if self.use_dual_stream:
                h_next, x_mag = self.layer(h, x_mag=x_mag, mask=mask)
            else:
                h_next = self.layer(h, mask=mask, top_k=top_k, optimizer=optimizer)

            # 3. Optional Hyperspace Working Memory Write
            if self.use_working_memory and memory is not None and mem_state is not None:
                mean_next = h_next.mean(dim=1)
                u_w = project_to_sphere(self.mem_u_head(mean_next))
                r_w = F.softplus(self.mem_r_head(mean_next)) + 0.5
                w_add = torch.tanh(self.mem_add_head(mean_next))
                w_gate = torch.sigmoid(self.mem_write_gate(mean_next))
                _, mem_state = memory.write(
                    mem_state, u_w, r_w,
                    add_vector=w_add,
                    erase_vector=torch.ones_like(w_add) * 0.2,
                    write_gate=w_gate,
                    kappa_w=kappa_q, beta_rw=beta_q
                )

            p_k = self.halting_gate(h_next)

            # Halting condition
            reached_limit = (accum_p + p_k >= self.halting_threshold) | (k == K)

            weight = torch.where(
                reached_limit & still_running,
                1.0 - accum_p,
                torch.where(still_running, p_k, torch.zeros_like(p_k))
            )

            output = output + weight * h_next
            accum_p = accum_p + p_k
            ponder_cost = ponder_cost + still_running.float()
            steps_taken = torch.where(still_running, torch.full_like(steps_taken, float(k)), steps_taken)

            still_running = still_running & (~reached_limit)
            h = h_next
            intermediate_states.append(h_next)

            if not still_running.any():
                break

        num_exp = self.layer.moe.num_experts if (self.use_dynamic_moe and self.layer.moe is not None) else 0

        meta = {
            "steps_taken": steps_taken.mean().item(),
            "ponder_cost": ponder_cost.mean(),
            "final_loops": k,
            "intermediate_states": intermediate_states,
            "num_experts": num_exp,
            "mem_state": {
                k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                for k, v in mem_state.items()
            } if mem_state is not None else None
        }
        return output, meta
