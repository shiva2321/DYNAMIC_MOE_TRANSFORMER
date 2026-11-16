"""
Dual-Timescale Memory Architecture (Short-Term Working + Long-Term Episodic).

1. GatedDeltaWorkingMemory (Short-Term):
   Real-valued multi-head CausalConv + Gated Delta-Rule linear-RNN memory
   (Schlag et al. 2021; Yang et al. ICLR 2025; Grazzi et al. ICLR 2025, M19/M22):
     - Separate causal depthwise convolutions on k, q, v, gates so writes can bind
       lagged context keys (e.g. k_{t-1} -> v_t) while queries read content cleanly
       without absolute-position contamination.
     - State update:
         v_old = S_{t-1} @ k_t
         S_t = alpha_t * S_{t-1} + beta_t * outer(v_t - alpha_t * v_old, k_t)
       Equivalently:
         S_t = alpha_t * (I - beta_t k_t k_t^T) S_{t-1} + beta_t v_t k_t^T
       With signed_beta=True, beta_t in (0, 2) gives transition eigenvalues in (-1, 1)
       for parity/flip/non-abelian state tracking (M22) while remaining contraction-bounded.
     - Zero LayerNorm amplification of unretrieved states; uses SiLU output gating instead.

2. LongTermSlotMemory (Long-Term):
   Persistent bounded key-value slot bank [num_slots, d_model] with:
     - Learnable null-match sink logit so irrelevant/empty banks decay smoothly to 0
       instead of forcing softmax noise onto unrelated queries.
     - Explicit episode-scoped consolidation (`write`, `write_kv`, `reset`) with salience +
       age retention instead of blind cross-batch mini-batch bleeding.
"""

import functools
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@functools.lru_cache(maxsize=64)
def _get_scan_masks(
    chunk_size: int, device_str: str, dtype_str: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = torch.device(device_str)
    dtype = getattr(torch, dtype_str.split(".")[-1])
    tril_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=dev, dtype=torch.bool))
    tril_strict = torch.tril(torch.ones(chunk_size, chunk_size, device=dev, dtype=torch.bool), diagonal=-1)
    eye = torch.eye(chunk_size, device=dev, dtype=dtype)
    return tril_mask, tril_strict, eye


class CausalDepthwiseConv1d(nn.Module):
    """Strictly causal depthwise 1D convolution over [B, L, D]."""

    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = max(1, int(kernel_size))
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=self.kernel_size,
            groups=d_model,
            bias=True,
        )
        # Near-identity causal init (tap on current step) so gradients start stable
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, 0, -1] = 1.0
            if self.conv.bias is not None:
                self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return self.conv(x.transpose(1, 2)).transpose(1, 2)
        xt = x.transpose(1, 2)  # [B, D, L]
        padded = F.pad(xt, (self.kernel_size - 1, 0))
        return self.conv(padded).transpose(1, 2)


class GatedDeltaWorkingMemory(nn.Module):
    """Multi-head causal Gated Delta-Rule / DeltaProduct working memory with optional signed beta in (0, 2)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        conv_kernel: int = 4,
        signed_beta: bool = True,
        forget_bias: float = 4.5,
        delta_steps: int = 1,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if delta_steps not in (1, 2):
            raise ValueError(f"delta_steps must be 1 or 2, got {delta_steps}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.signed_beta = bool(signed_beta)
        self.delta_steps = int(delta_steps)

        # Projections and per-role causal convolutions (decouples lagged write-key from query)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_g = nn.Linear(d_model, d_model, bias=True)

        self.conv_k = CausalDepthwiseConv1d(d_model, kernel_size=conv_kernel)
        self.conv_q = CausalDepthwiseConv1d(d_model, kernel_size=conv_kernel)
        self.conv_v = CausalDepthwiseConv1d(d_model, kernel_size=conv_kernel)

        # Initialize w_k and conv_k cleanly so lag-1 and current-step keys are reachable immediately;
        # keep q and k projections aligned at init to prevent random key-query orthogonality traps.
        with torch.no_grad():
            self.w_q.weight.copy_(self.w_k.weight)
            if conv_kernel >= 2:
                self.conv_k.conv.weight.zero_()
                self.conv_k.conv.weight[:, 0, -2] = 0.5
                self.conv_k.conv.weight[:, 0, -1] = 0.5

        self.w_beta = nn.Linear(d_model, num_heads)
        self.w_alpha = nn.Linear(d_model, num_heads)
        self.o_norm_w = nn.Parameter(torch.ones(num_heads, self.head_dim))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        if self.delta_steps == 2:
            self.w_k2 = nn.Linear(d_model, d_model, bias=False)
            self.w_v2 = nn.Linear(d_model, d_model, bias=False)
            self.w_beta2 = nn.Linear(d_model, num_heads)
            with torch.no_grad():
                self.w_v2.weight.zero_()

        # High retention bias on both associative and reflection heads prevents OOD decay;
        # asymmetric beta keeps associative heads in (0, 1) and activates signed DeltaProduct
        # reflection heads in (1, 2) without a beta=1.0 zero-eigenvalue deadzone.
        with torch.no_grad():
            if num_heads == 4:
                self.w_alpha.bias.copy_(torch.tensor([6.0, 5.5, 6.0, 6.0]))
            elif num_heads >= 2:
                self.w_alpha.bias.copy_(torch.linspace(6.0, 5.5, num_heads))
            else:
                nn.init.constant_(self.w_alpha.bias, float(forget_bias))

            if self.signed_beta:
                if num_heads == 4:
                    self.w_beta.bias.copy_(torch.tensor([-2.0, -1.0, 1.0, 1.5]))
                elif num_heads >= 2:
                    self.w_beta.bias.copy_(torch.linspace(-2.0, 1.5, num_heads))
                else:
                    nn.init.constant_(self.w_beta.bias, -0.4)
                if self.delta_steps == 2:
                    if num_heads == 4:
                        self.w_beta2.bias.copy_(torch.tensor([-4.0, -3.5, 1.0, 1.5]))
                    elif num_heads >= 2:
                        self.w_beta2.bias.copy_(torch.linspace(-4.0, 1.5, num_heads))
                    else:
                        nn.init.constant_(self.w_beta2.bias, 0.0)
            else:
                nn.init.constant_(self.w_beta.bias, 0.0)
                if self.delta_steps == 2:
                    nn.init.constant_(self.w_beta2.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        init_state: Optional[torch.Tensor] = None,
        return_state: bool = True,
        write_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: [B, L, d_model]
        init_state: optional [B, H, head_dim, head_dim]
        Returns:
          out: [B, L, d_model]
          final_state: [B, H, head_dim, head_dim] or None when return_state=False
        """
        B, L, D = x.shape
        H, dh = self.num_heads, self.head_dim

        # Single fused linear projection for keys/queries/values, gates, and optional step-2 DeltaProduct terms.
        zero_bias = x.new_zeros(D)
        if self.delta_steps == 2:
            w_all = torch.cat(
                [
                    self.w_k.weight,
                    self.w_q.weight,
                    self.w_v.weight,
                    self.w_g.weight,
                    self.w_beta.weight,
                    self.w_alpha.weight,
                    self.w_k2.weight,
                    self.w_v2.weight,
                    self.w_beta2.weight,
                ],
                dim=0,
            )
            b_all = torch.cat(
                [
                    zero_bias,
                    zero_bias,
                    zero_bias,
                    self.w_g.bias,
                    self.w_beta.bias,
                    self.w_alpha.bias,
                    zero_bias,
                    zero_bias,
                    self.w_beta2.bias,
                ],
                dim=0,
            )
            kqv_in, out_gate_raw, beta_raw, alpha_raw, k2_raw, v2_raw, beta2_raw = F.linear(
                x, w_all, b_all
            ).split([3 * D, D, H, H, D, D, H], dim=-1)
        else:
            w_all = torch.cat(
                [
                    self.w_k.weight,
                    self.w_q.weight,
                    self.w_v.weight,
                    self.w_g.weight,
                    self.w_beta.weight,
                    self.w_alpha.weight,
                ],
                dim=0,
            )
            b_all = torch.cat(
                [
                    zero_bias,
                    zero_bias,
                    zero_bias,
                    self.w_g.bias,
                    self.w_beta.bias,
                    self.w_alpha.bias,
                ],
                dim=0,
            )
            kqv_in, out_gate_raw, beta_raw, alpha_raw = F.linear(
                x, w_all, b_all
            ).split([3 * D, D, H, H], dim=-1)

        # Single grouped causal depthwise 1D convolution across k, q, v channels.
        ks = self.conv_k.kernel_size
        cw = torch.cat([self.conv_k.conv.weight, self.conv_q.conv.weight, self.conv_v.conv.weight], dim=0)
        cb_parts = [
            b if b is not None else zero_bias
            for b in (self.conv_k.conv.bias, self.conv_q.conv.bias, self.conv_v.conv.bias)
        ]
        cb = torch.cat(cb_parts, dim=0)
        xt = kqv_in.transpose(1, 2)
        if ks > 1:
            xt = F.pad(xt, (ks - 1, 0))
        kr, qr, v_raw = F.silu(F.conv1d(xt, cw, cb, groups=3 * D).transpose(1, 2)).split(D, dim=-1)

        k = F.normalize(kr.view(B, L, H, dh), p=2, dim=-1, eps=1e-6)
        q = F.normalize(qr.view(B, L, H, dh), p=2, dim=-1, eps=1e-6)
        v = v_raw.view(B, L, H, dh)

        # Evaluate token-conditional retention/flip gates directly on x without conv smearing.
        beta_sig = torch.sigmoid(beta_raw)
        beta = 2.0 * beta_sig if self.signed_beta else beta_sig  # [B, L, H]
        if write_mask is not None:
            beta = beta * write_mask.to(device=beta.device, dtype=beta.dtype).unsqueeze(-1)
        alpha = torch.sigmoid(alpha_raw)                         # [B, L, H]
        out_gate = F.silu(out_gate_raw)                          # [B, L, D]

        if self.delta_steps == 2:
            k2 = F.normalize(k2_raw.view(B, L, H, dh), p=2, dim=-1, eps=1e-6)
            v2 = F.silu(v2_raw).view(B, L, H, dh)
            beta2_sig = torch.sigmoid(beta2_raw)
            beta2 = 2.0 * beta2_sig if self.signed_beta else beta2_sig
            if write_mask is not None:
                beta2 = beta2 * write_mask.to(device=beta2.device, dtype=beta2.dtype).unsqueeze(-1)
            alpha2 = torch.ones_like(alpha)

            k_scan = torch.stack([k, k2], dim=2).reshape(B, 2 * L, H, dh)
            q_scan = torch.stack([q, q], dim=2).reshape(B, 2 * L, H, dh)
            v_scan = torch.stack([v, v2], dim=2).reshape(B, 2 * L, H, dh)
            a_scan = torch.stack([alpha, alpha2], dim=2).reshape(B, 2 * L, H)
            b_scan = torch.stack([beta, beta2], dim=2).reshape(B, 2 * L, H)
        else:
            k_scan, q_scan, v_scan, a_scan, b_scan = k, q, v, alpha, beta

        if k_scan.size(1) <= 8:
            S0 = x.new_zeros(B, H, dh, dh) if init_state is None else init_state.to(device=x.device, dtype=x.dtype)
            out_scan, S = _sequential_gated_delta_scan(k_scan, q_scan, v_scan, a_scan, b_scan, init_state=S0)
            if not return_state:
                S = None
        else:
            init_arg = None if init_state is None else init_state.to(device=x.device, dtype=x.dtype)
            out_scan, S = _chunked_gated_delta_scan(
                k_scan,
                q_scan,
                v_scan,
                a_scan,
                b_scan,
                init_state=init_arg,
                chunk_size=64,
                return_state=return_state,
            )

        out_seq = out_scan[:, 1::2] if self.delta_steps == 2 else out_scan
        out_seq = F.rms_norm(out_seq, (dh,), eps=1e-5) * self.o_norm_w.view(1, 1, H, dh)

        out = self.out_proj(out_seq.reshape(B, L, D) * out_gate)
        return out, S


def _sequential_gated_delta_scan(
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    init_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference step-by-step Gated Delta-Rule recurrence over [B, L, H, dh]."""
    B, L, H, _ = k.shape
    S = init_state
    k_steps = k.unsqueeze(-1).unbind(dim=1)
    kt_steps = k.unsqueeze(-2).unbind(dim=1)
    q_steps = q.unsqueeze(-1).unbind(dim=1)
    v_steps = v.unbind(dim=1)
    b_steps = beta.view(B, L, H, 1, 1).unbind(dim=1)
    a_steps = alpha.view(B, L, H, 1, 1).unbind(dim=1)

    outputs = []
    for t in range(L):
        a_t = a_steps[t]
        v_old = torch.matmul(S, k_steps[t]).squeeze(-1)
        err = v_steps[t] - a_t.squeeze(-1) * v_old
        S = a_t * S + b_steps[t] * (err.unsqueeze(-1) * kt_steps[t])
        outputs.append(torch.matmul(S, q_steps[t]).squeeze(-1))
    return torch.stack(outputs, dim=1), S


def _chunked_gated_delta_scan(
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    init_state: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    return_state: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Exact chunkwise parallel WY scan for the Gated Delta Rule:
      S_t = alpha_t * (I - beta_t k_t k_t^T) S_{t-1} + beta_t v_t k_t^T,  o_t = S_t q_t
    Mathematically equivalent to the step-by-step recurrence up to floating-point associativity.
    """
    B, L, H, dh = k.shape
    C = min(int(chunk_size), max(8, 1 << (L - 1).bit_length()))
    pad = (C - (L % C)) % C
    if pad > 0:
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        alpha = F.pad(alpha, (0, 0, 0, pad), value=1.0)
        beta = F.pad(beta, (0, 0, 0, pad), value=0.0)

    Lp = k.size(1)
    nc = Lp // C
    BH = B * H
    kc = k.view(B, nc, C, H, dh).permute(0, 3, 1, 2, 4).reshape(BH * nc, C, dh)
    qc = q.view(B, nc, C, H, dh).permute(0, 3, 1, 2, 4).reshape(BH * nc, C, dh)
    vc = v.view(B, nc, C, H, dh).permute(0, 3, 1, 2, 4).reshape(BH * nc, C, dh)
    ac = alpha.view(B, nc, C, H).permute(0, 3, 1, 2).reshape(BH * nc, C)
    bc = beta.view(B, nc, C, H).permute(0, 3, 1, 2).reshape(BH * nc, C)

    tril_mask, tril_strict, eye = _get_scan_masks(C, str(k.device), str(k.dtype))
    g = torch.log(ac.clamp(min=1e-6)).cumsum(dim=-1)  # [BH*nc, C] <= 0
    decay_mat = torch.exp((g.unsqueeze(-1) - g.unsqueeze(-2)).masked_fill(~tril_mask, -float("inf")))
    kk = torch.bmm(kc, kc.transpose(1, 2))
    M = (bc.unsqueeze(-1) * decay_mat * kk).masked_fill(~tril_strict, 0.0)
    bc_u = bc.unsqueeze(-1)
    qk_decay = torch.bmm(qc, kc.transpose(1, 2)) * decay_mat

    if nc == 1 and init_state is None and not return_state:
        W_u = torch.linalg.solve_triangular(eye + M, bc_u * vc, upper=False, unitriangular=True)
        out = torch.bmm(qk_decay, W_u)
        return out.view(B, H, C, dh).transpose(1, 2)[:, :L], None

    exp_g = torch.exp(g).unsqueeze(-1)
    rhs = torch.cat([bc_u * vc, (bc_u * exp_g) * kc], dim=-1)
    sol = torch.linalg.solve_triangular(eye + M, rhs, upper=False, unitriangular=True)
    W_u, A_mod = sol.split(dh, dim=-1)

    if nc == 1:
        if init_state is None:
            W_0 = W_u
            out = torch.bmm(qk_decay, W_0)
            g_end = g[:, -1:]
            S = torch.bmm((torch.exp(g_end - g).unsqueeze(-1) * W_0).transpose(1, 2), kc)
        else:
            S0 = init_state.reshape(BH, dh, dh)
            S_T = S0.transpose(1, 2)
            W_0 = W_u - torch.bmm(A_mod, S_T)
            out = torch.bmm(qk_decay, W_0) + exp_g * torch.bmm(qc, S_T)
            if return_state:
                g_end = g[:, -1:]
                S = torch.exp(g_end).view(BH, 1, 1) * S0 + torch.bmm(
                    (torch.exp(g_end - g).unsqueeze(-1) * W_0).transpose(1, 2), kc
                )
            else:
                S = None
        return (
            out.view(B, H, C, dh).transpose(1, 2)[:, :L],
            S.view(B, H, dh, dh) if S is not None else None,
        )

    W_c = W_u.view(BH, nc, C, dh).unbind(1)
    A_c = A_mod.view(BH, nc, C, dh).unbind(1)
    qk_c = qk_decay.view(BH, nc, C, C).unbind(1)
    qc_c = qc.view(BH, nc, C, dh).unbind(1)
    kc_c = kc.view(BH, nc, C, dh).unbind(1)
    eg_c = exp_g.view(BH, nc, C, 1).unbind(1)
    g_all = g.view(BH, nc, C)
    g_end = g_all[:, :, -1:]
    de_c = torch.exp(g_end - g_all).unsqueeze(-1).unbind(1)
    ege_c = torch.exp(g_end).unsqueeze(-1).unbind(1)

    zero_s = init_state is None
    S = k.new_zeros(BH, dh, dh) if zero_s else init_state.reshape(BH, dh, dh)
    out_chunks = []
    for c in range(nc):
        is_last = c == nc - 1
        if zero_s and c == 0:
            W_i = W_c[0]
            out_chunks.append(torch.bmm(qk_c[0], W_i))
            if return_state or not is_last:
                S = torch.bmm((de_c[0] * W_i).transpose(1, 2), kc_c[0])
        else:
            S_T = S.transpose(1, 2)
            W_i = W_c[c] - torch.bmm(A_c[c], S_T)
            out_chunks.append(torch.bmm(qk_c[c], W_i) + eg_c[c] * torch.bmm(qc_c[c], S_T))
            if return_state or not is_last:
                S = ege_c[c] * S + torch.bmm((de_c[c] * W_i).transpose(1, 2), kc_c[c])

    out_seq = torch.stack(out_chunks, dim=1).view(B, H, Lp, dh).transpose(1, 2)[:, :L]
    return out_seq, (S.view(B, H, dh, dh) if return_state else None)


class LongTermSlotMemory(nn.Module):
    """
    Persistent, surprise-gated episodic/semantic slot memory bank with null-match sink.
    Stores [num_slots, d_model] keys and values across episodes without forcing soft-attention
    mass onto irrelevant entries when nothing matches.
    """

    def __init__(
        self,
        d_model: int,
        num_slots: int = 64,
        write_surprise_threshold: float = 0.25,
        ema_decay: float = 0.85,
        merge_cos_threshold: float = 0.92,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.write_surprise_threshold = float(write_surprise_threshold)
        self.ema_decay = float(ema_decay)
        self.merge_cos_threshold = float(merge_cos_threshold)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.salience_head = nn.Linear(d_model, 1)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable temperature and size-invariant null-sink score (logit + log(M));
        # initialize at 3.2 to preserve untrained random-slot rejection (mass_1 < 0.15)
        # while avoiding the 4.5 cold-start softmax gradient starvation on random keys.
        self.logit_scale = nn.Parameter(torch.tensor(12.0))
        self.null_logit = nn.Parameter(torch.tensor(3.2))

        self.register_buffer("slot_keys", torch.zeros(num_slots, d_model))
        self.register_buffer("slot_vals", torch.zeros(num_slots, d_model))
        self.register_buffer("slot_used", torch.zeros(num_slots, dtype=torch.bool))
        self.register_buffer("slot_age", torch.zeros(num_slots))
        self.register_buffer("slot_salience", torch.zeros(num_slots))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_writes", torch.tensor(0, dtype=torch.long))
        self._num_used_slots: int = 0

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._num_used_slots = int(self.slot_used.sum().item())

    @torch.no_grad()
    def reset(self) -> None:
        """Clear all stored episodic/semantic memory slots."""
        self.slot_keys.zero_()
        self.slot_vals.zero_()
        self.slot_used.zero_()
        self.slot_age.zero_()
        self.slot_salience.zero_()
        self.write_ptr.zero_()
        self.total_writes.zero_()
        self._num_used_slots = 0

    def encode_kv(
        self,
        key_latents: torch.Tensor,
        value_latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiably project raw key/value latents into the LTM key/value space."""
        k_norm = F.normalize(self.k_proj(key_latents), p=2, dim=-1, eps=1e-6)
        v_proj = self.v_proj(value_latents)
        return k_norm, v_proj

    def read(
        self,
        query_latents: torch.Tensor,
        ep_keys: Optional[torch.Tensor] = None,
        ep_vals: Optional[torch.Tensor] = None,
        ep_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant long-term memory for query_latents [B, L, d_model] from:
          - persistent slot bank (`slot_keys`, `slot_vals`), and/or
          - optional differentiable transient per-batch slots (`ep_keys`, `ep_vals`, `ep_mask`).
        Uses size-invariant null-sink normalization (`null_logit + log(M_active)`)
        so irrelevant slots decay to ~0 regardless of bank size.
        """
        B, L, D = query_latents.shape
        has_ep = ep_keys is not None and ep_vals is not None and ep_keys.size(1) > 0
        if self._num_used_slots == 0 and not has_ep:
            return query_latents.new_zeros(B, L, D), query_latents.new_zeros(B, L, 0)
        used_idx = (
            self.slot_used.nonzero(as_tuple=True)[0]
            if self._num_used_slots > 0
            else self.slot_used.new_empty(0, dtype=torch.long)
        )
        m_pers = used_idx.numel()

        q = F.normalize(self.q_proj(query_latents), p=2, dim=-1, eps=1e-6)  # [B, L, D]
        scale = self.logit_scale.clamp(1.0, 28.0)

        logit_parts = []
        valid_count = torch.full((B, 1, 1), float(m_pers), device=query_latents.device, dtype=query_latents.dtype)

        if m_pers > 0:
            pers_keys = F.normalize(self.slot_keys[used_idx], p=2, dim=-1, eps=1e-6)  # [M, D]
            pers_logits = torch.matmul(q, pers_keys.T) * scale                         # [B, L, M]
            logit_parts.append(pers_logits)

        if has_ep:
            ek = F.normalize(ep_keys, p=2, dim=-1, eps=1e-6)                           # [B, M_ep, D]
            ep_logits = torch.bmm(q, ek.transpose(-2, -1)) * scale                     # [B, L, M_ep]
            if ep_mask is not None:
                ep_logits = ep_logits.masked_fill(~ep_mask.unsqueeze(1), float("-inf"))
                valid_count = valid_count + ep_mask.sum(dim=-1).view(B, 1, 1).to(query_latents.dtype)
            else:
                valid_count = valid_count + float(ep_keys.size(1))
            logit_parts.append(ep_logits)

        null_col = self.null_logit.view(1, 1, 1) + torch.log(valid_count.clamp(min=1.0))
        logit_parts.append(null_col.expand(B, L, 1))

        all_logits = torch.cat(logit_parts, dim=-1)
        all_probs = F.softmax(all_logits, dim=-1)

        mem = query_latents.new_zeros(B, L, D)
        offset = 0
        pers_probs = query_latents.new_zeros(B, L, 0)
        if m_pers > 0:
            pers_probs = all_probs[..., :m_pers]
            pers_vals = self.slot_vals[used_idx]
            mem = mem + torch.matmul(pers_probs, pers_vals)
            offset = m_pers
        if has_ep:
            m_ep = ep_keys.size(1)
            ep_probs = all_probs[..., offset : offset + m_ep]
            mem = mem + torch.bmm(ep_probs, ep_vals)

        return self.out_proj(mem), pers_probs

    @torch.no_grad()
    def _insert_kv_vectors(
        self,
        k_cand: torch.Tensor,
        v_cand: torch.Tensor,
        saliences: torch.Tensor,
    ) -> int:
        self.slot_age[self.slot_used] += 1.0
        written = 0
        for i in range(k_cand.size(0)):
            ki = k_cand[i]
            vi = v_cand[i]
            si = float(saliences[i].item())

            if self.slot_used.any():
                used_idx = self.slot_used.nonzero(as_tuple=True)[0]
                sims = F.normalize(self.slot_keys[used_idx], p=2, dim=-1, eps=1e-6) @ ki
                best_sim, best_pos = sims.max(dim=0)
                if float(best_sim.item()) >= self.merge_cos_threshold:
                    slot_i = int(used_idx[best_pos].item())
                    self.slot_keys[slot_i] = F.normalize(
                        self.ema_decay * self.slot_keys[slot_i] + (1.0 - self.ema_decay) * ki,
                        p=2,
                        dim=-1,
                        eps=1e-6,
                    )
                    self.slot_vals[slot_i] = (
                        self.ema_decay * self.slot_vals[slot_i] + (1.0 - self.ema_decay) * vi
                    )
                    self.slot_age[slot_i] = 0.0
                    self.slot_salience[slot_i] = max(float(self.slot_salience[slot_i].item()), si)
                    written += 1
                    continue

            free = (~self.slot_used).nonzero(as_tuple=True)[0]
            if free.numel() > 0:
                slot_i = int(free[0].item())
            else:
                # Evict stalest / lowest-priority slot (age penalized by salience)
                evict_score = self.slot_age - 10.0 * self.slot_salience
                slot_i = int(torch.argmax(evict_score).item())

            self.slot_keys[slot_i] = ki
            self.slot_vals[slot_i] = vi
            self.slot_used[slot_i] = True
            self.slot_age[slot_i] = 0.0
            self.slot_salience[slot_i] = si
            self.write_ptr = (self.write_ptr + 1) % self.num_slots
            written += 1

        self.total_writes += written
        self._num_used_slots = int(self.slot_used.sum().item())
        return written

    @torch.no_grad()
    def write(
        self,
        latents: torch.Tensor,
        surprise: Optional[torch.Tensor] = None,
        force_write: bool = False,
        max_writes_per_call: int = 8,
    ) -> int:
        """
        Surprise-gated diary consolidation into long-term slots.
        latents: [B, L, d_model]
        surprise: optional [B, L] in [0, 1]
        """
        D = latents.size(-1)
        flat_x = latents.detach().reshape(-1, D)
        salience = torch.sigmoid(self.salience_head(flat_x)).squeeze(-1)

        if surprise is not None:
            flat_s = surprise.detach().reshape(-1).to(flat_x.device)
            gate_score = 0.6 * flat_s + 0.4 * salience
        else:
            gate_score = salience

        if force_write:
            write_mask = torch.ones_like(gate_score, dtype=torch.bool)
        else:
            write_mask = gate_score >= self.write_surprise_threshold

        if not self.slot_used.any() and flat_x.size(0) > 0 and not write_mask.any():
            top_i = int(torch.argmax(gate_score).item())
            write_mask[top_i] = True

        sel = flat_x[write_mask]
        sel_scores = gate_score[write_mask]
        if sel.size(0) == 0:
            return 0

        if sel.size(0) > max_writes_per_call:
            top_idx = torch.topk(sel_scores, k=max_writes_per_call).indices
            sel = sel[top_idx]
            sel_scores = sel_scores[top_idx]

        k_cand = F.normalize(self.k_proj(sel), p=2, dim=-1, eps=1e-6)
        v_cand = self.v_proj(sel)
        return self._insert_kv_vectors(k_cand, v_cand, sel_scores)

    @torch.no_grad()
    def write_kv(
        self,
        key_latents: torch.Tensor,
        value_latents: torch.Tensor,
        salience: float = 1.0,
    ) -> int:
        """
        Explicit episode-fact consolidation: store normalized (key -> value) latents cleanly.
        key_latents, value_latents: [N, d_model]
        """
        if key_latents.numel() == 0:
            return 0
        k_cand = F.normalize(self.k_proj(key_latents.detach().reshape(-1, self.d_model)), p=2, dim=-1, eps=1e-6)
        v_cand = self.v_proj(value_latents.detach().reshape(-1, self.d_model))
        saliences = torch.full((k_cand.size(0),), float(salience), device=k_cand.device)
        return self._insert_kv_vectors(k_cand, v_cand, saliences)


class DualTimescaleMemory(nn.Module):
    """
    Combines Short-Term GatedDeltaWorkingMemory with Long-Term Surprise-Gated Slot Memory.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        num_ltm_slots: int = 64,
        write_surprise_threshold: float = 0.25,
        conv_kernel: int = 4,
        signed_beta: bool = True,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(d_model)
        self.short_term = GatedDeltaWorkingMemory(
            d_model=d_model,
            num_heads=num_heads,
            conv_kernel=conv_kernel,
            signed_beta=signed_beta,
        )
        self.long_term = LongTermSlotMemory(
            d_model=d_model,
            num_slots=num_ltm_slots,
            write_surprise_threshold=write_surprise_threshold,
        )
        self.stm_gate = nn.Linear(d_model, 1)
        # Contextual LTM gate sees causal short-term state so prefix rules/tasks can suppress
        # episodic slot injection even when query tokens coincide with stored keys.
        self.ltm_gate = nn.Linear(2 * d_model, 1)
        # Zero-initialized contextual query refinement from short-term state so single-token
        # and multi-token cues can both index into long-term slots without hurting cold init.
        self.ltm_ctx_proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.ltm_ctx_proj.weight.zero_()

    def reset_long_term(self) -> None:
        self.long_term.reset()

    def build_ltm_query(self, x_norm: torch.Tensor, stm_out: torch.Tensor) -> torch.Tensor:
        return x_norm + self.ltm_ctx_proj(stm_out)

    def forward(
        self,
        x: torch.Tensor,
        init_stm_state: Optional[torch.Tensor] = None,
        update_ltm: bool = False,
        surprise: Optional[torch.Tensor] = None,
        ep_keys: Optional[torch.Tensor] = None,
        ep_vals: Optional[torch.Tensor] = None,
        ep_mask: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x_norm = self.in_norm(x)
        stm_out, stm_state = self.short_term(x_norm, init_state=init_stm_state, write_mask=write_mask)
        ltm_q = self.build_ltm_query(x_norm, stm_out)
        ltm_out, ltm_probs = self.long_term.read(
            ltm_q, ep_keys=ep_keys, ep_vals=ep_vals, ep_mask=ep_mask
        )

        g_stm = torch.sigmoid(self.stm_gate(x_norm))
        g_ltm = torch.sigmoid(self.ltm_gate(torch.cat([x_norm, stm_out], dim=-1)))
        fused = x + g_stm * stm_out + g_ltm * ltm_out

        writes = 0
        if update_ltm:
            writes = self.long_term.write(ltm_q, surprise=surprise)

        return {
            "fused": fused,
            "x_norm": x_norm,
            "ltm_q": ltm_q,
            "stm_out": stm_out,
            "stm_state": stm_state,
            "ltm_out": ltm_out,
            "ltm_probs": ltm_probs,
            "ltm_writes": torch.tensor(writes, device=x.device),
        }
