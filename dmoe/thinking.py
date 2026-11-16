"""
Surprise-Driven Dynamic Latent Thinking Algorithm (Phase A2 / Track R).

Implements an anchor-reinjected, surprise-gated latent reasoning loop:
1. Anchor + Working Memory State (h_0, z^(k)):
   - Preserves the initial backbone representation h_0 as a permanent anchor
     while iteratively refining the latent scratchpad z^(k).
2. Three-Part Surprise & Energy Evaluator S_t^(k):
   - Combines (a) normalized predictive entropy H_t^(k), (b) latent fixed-point
     residual R_t^(k), and (c) a learned error-predicting energy head E_t^(k).
3. Per-Token Dynamic Halting & Epistemic Verdict:
   - Skips thinking entirely when initial surprise is below skip_threshold.
   - Halts individual tokens early once their surprise drops below theta_halt
     or their latent change converges.
   - Flags tokens whose post-thinking surprise remains above abstain_threshold
     as epistemic UNKNOWN.
4. Deep-Supervised + Monotonic Progress + Surprise Calibration Training Objective:
   - Trains every thinking step k+1 to strictly reduce prediction error over step k
     using detached-carry deep supervision (TRM-style), a progress margin loss,
     and BCE calibration of the surprise head against actual token errors.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.rope import RotaryEmbedding


class LatentThinkingBlock(nn.Module):
    """
    Time-invariant causal-neighbor + relational latent operator applied at each thinking step.
    Re-injects the problem anchor h_0 and left-neighbor state z_{t-1}^(k) into the evolving
    working state z_t^(k), enabling both iterative multi-hop chain propagation and global
    relational attention while generalizing zero-shot to thinking steps k > K_train.
    """

    def __init__(self, d_model: int, num_heads: int = 2, d_ff: int = 128):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Time-invariant recurrent fusion of [z_t^(k), z_{t-1}^(k), h_{0,t}]
        self.recur_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm_recur = nn.LayerNorm(d_model)

        # Causal relational self-attention (with RoPE when head_dim is even) + FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope: Optional[RotaryEmbedding] = (
            RotaryEmbedding(self.head_dim) if (self.head_dim % 2 == 0) else None
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        z: torch.Tensor,
        h_anchor: torch.Tensor,
        step_idx: int = 0,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute candidate latent state z_cand from current working state z and anchor h_anchor.
        Strictly time-invariant (does not depend on step_idx) so the operator generalizes
        zero-shot to out-of-distribution thinking depths k > K_train.
        z, h_anchor: [B, L, d_model]
        """
        B, L, D = z.shape
        z_prev = torch.zeros_like(z)
        if L > 1:
            z_prev[:, 1:, :] = z[:, :-1, :]

        # 1. Causal RoPE relational attention first (with direct anchor injection)
        #    so non-adjacent antecedent states are gathered cleanly before MLP transitions
        h_attn = self.norm1(z + h_anchor)
        q = self.q_proj(h_attn).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_attn).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_attn).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope.apply_rotary(q, k)

        if key_mask is not None:
            causal = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
            past_only = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=-1)
            pair_past = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=-2)
            key_gate = key_mask[:, :L].to(device=q.device, dtype=q.dtype).clamp(min=1e-4, max=1.0)
            log_gate = torch.log(key_gate).view(B, 1, 1, L)
            log_next = F.pad(log_gate[..., 1:], (0, 1), value=0.0)
            bias = torch.zeros(B, 1, L, L, device=q.device, dtype=q.dtype)
            bias = bias.masked_fill(~causal.view(1, 1, L, L), -1e9)
            zeros = torch.zeros_like(log_gate)
            bias = bias + torch.where(past_only.view(1, 1, L, L), log_gate, zeros)
            bias = bias + torch.where(pair_past.view(1, 1, L, L), log_next, zeros)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        context = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        u = z + self.out_proj(context)

        # 2. Local recurrent state transition + Pre-LN FFN + bounded output normalization
        z_prev = torch.zeros_like(u)
        if L > 1:
            z_prev[:, 1:, :] = u[:, :-1, :]
        recur_delta = self.recur_mlp(torch.cat([u, z_prev, h_anchor], dim=-1))
        u2 = self.norm_recur(u + recur_delta)
        u3 = u2 + self.ffn(self.norm2(u2))
        z_cand = self.norm_out(u3)
        return z_cand


class SurpriseEnergyHead(nn.Module):
    """
    Learned surprise / error-prediction head E_theta(h_0, z^(k), delta_z).
    Trained via BCE to predict P(argmax y_hat^(k) != y_true), providing a calibrated
    internal surprise signal without access to ground-truth labels at test time.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, h_anchor: torch.Tensor, z: torch.Tensor, delta_z: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns surprise probability in (0, 1) of shape [B, L].
        """
        feat = torch.cat([h_anchor, z, delta_z], dim=-1)
        return torch.sigmoid(self.net(feat)).squeeze(-1)


class SurpriseDrivenLatentThinker(nn.Module):
    """
    Dynamic Latent-Space Thinking Controller driven by token-level surprise.
    """

    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 2,
        d_ff: int = 128,
        max_thinking_steps: int = 6,
        skip_threshold: float = 0.12,
        theta_halt: float = 0.18,
        min_surprise_drop: float = 0.005,
        abstain_threshold: float = 0.65,
        detached_carry_prob: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_thinking_steps = max_thinking_steps
        self.skip_threshold = skip_threshold
        self.theta_halt = theta_halt
        self.min_surprise_drop = min_surprise_drop
        self.abstain_threshold = abstain_threshold
        self.detached_carry_prob = detached_carry_prob

        self.think_block = LatentThinkingBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        self.energy_head = SurpriseEnergyHead(d_model=d_model)
        self.damping_logit = nn.Parameter(torch.tensor(2.2))  # sigmoid(2.2) ~ 0.90

        # Telemetry recorded on last forward pass
        self.last_steps_used: int = 0
        self.last_active_fraction: float = 0.0
        self.last_token_steps: Optional[torch.Tensor] = None
        self.last_surprise_trace: List[torch.Tensor] = []
        self.last_is_unknown: Optional[torch.Tensor] = None

    @staticmethod
    def _normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized Shannon entropy in [0, 1] from logits [..., V].
        """
        V = logits.size(-1)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        ent = -(probs * log_probs).sum(dim=-1)
        max_ent = float(torch.log(torch.tensor(float(V), device=logits.device)).item())
        # Collapse chunk dimension if present: [B, n_chunks, chunk_size, V] -> [B, n_chunks]
        if ent.dim() == 3:
            ent = ent.mean(dim=-1)
        return (ent / max(max_ent, 1e-6)).clamp(0.0, 1.0)

    def compute_surprise(
        self,
        h_anchor: torch.Tensor,
        z: torch.Tensor,
        z_cand: torch.Tensor,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute composite surprise S_t^(k) in [0, 1] and learned energy E_t^(k) in [0, 1].
        """
        delta_z = z_cand - z
        learned_energy = self.energy_head(h_anchor.detach(), z.detach(), delta_z.detach())
        pred_entropy = self._normalized_entropy(logits.detach())
        latent_residual = torch.tanh(
            delta_z.detach().norm(p=2, dim=-1) / (self.d_model ** 0.5)
        )
        surprise = (
            0.50 * learned_energy + 0.25 * pred_entropy + 0.25 * latent_residual
        ).clamp(0.0, 1.0)
        return surprise, learned_energy

    def forward(
        self,
        h_anchor: torch.Tensor,
        decoder_fn: Callable[[torch.Tensor], torch.Tensor],
        force_steps: Optional[int] = None,
        key_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Execute surprise-driven dynamic latent thinking starting from h_anchor.

        Args:
            h_anchor: [B, L, d_model] fast-pass representation from backbone.
            decoder_fn: callable mapping latent state z [B, L, d_model] -> logits [B, L, ..., V].
            force_steps: if provided, overrides adaptive halting and runs exactly force_steps.

        Returns dict with:
            - z_final: [B, L, d_model] refined latent state
            - logits_final: final decoded logits
            - step_logits: list of logits at steps k = 0 .. K
            - step_energies: list of learned energy predictions [B, L] at steps k = 0 .. K
            - surprise_trace: list of composite surprise tensors [B, L]
            - token_steps: [B, L] number of thinking steps taken per token
            - steps_used: int total loop iterations executed
            - active_fraction: float mean fraction of active tokens across steps
            - is_unknown: [B, L] bool tensor flagging unresolved high-surprise positions
        """
        B, L, D = h_anchor.shape
        max_k = self.max_thinking_steps if force_steps is None else int(force_steps)

        z = h_anchor
        logits_0 = decoder_fn(z)
        z_cand_0 = self.think_block(z, h_anchor, step_idx=0, key_mask=key_mask)
        surprise_0, energy_0 = self.compute_surprise(h_anchor, z, z_cand_0, logits_0)

        step_logits: List[torch.Tensor] = [logits_0]
        step_energies: List[torch.Tensor] = [energy_0]
        surprise_trace: List[torch.Tensor] = [surprise_0]
        token_steps = torch.zeros(B, L, device=h_anchor.device, dtype=torch.float32)

        # Fast-path skip at eval when force_steps is None and all tokens have low initial surprise
        if (
            not self.training
            and force_steps is None
            and max_k > 0
            and float(surprise_0.max().item()) <= self.skip_threshold
        ):
            self.last_steps_used = 0
            self.last_active_fraction = 0.0
            self.last_token_steps = token_steps
            self.last_surprise_trace = surprise_trace
            self.last_is_unknown = surprise_0 > self.abstain_threshold
            return {
                "z_final": z,
                "logits_final": logits_0,
                "step_logits": step_logits,
                "step_energies": step_energies,
                "surprise_trace": surprise_trace,
                "token_steps": token_steps,
                "steps_used": 0,
                "active_fraction": 0.0,
                "is_unknown": self.last_is_unknown,
            }

        damping = torch.sigmoid(self.damping_logit)
        active_mask = torch.ones(B, L, device=h_anchor.device, dtype=torch.bool)
        active_fractions: List[float] = []
        steps_used = 0
        prev_surprise = surprise_0

        for k in range(max_k):
            if k == 0:
                z_cand = z_cand_0
            else:
                # Detached-carry regularization during training (TRM-style) so every step
                # learns to improve from the current state without gradient explosion
                z_in = (
                    z.detach()
                    if (self.training and k > 0 and self.detached_carry_prob >= 1.0)
                    else (
                        0.5 * z + 0.5 * z.detach()
                        if (self.training and k > 0 and self.detached_carry_prob > 0.0)
                        else z
                    )
                )
                z_cand = self.think_block(z_in, h_anchor, step_idx=k, key_mask=key_mask)

            if not self.training and force_steps is None:
                # Per-token halting condition: halt tokens whose surprise is below theta_halt
                still_surprised = prev_surprise >= self.theta_halt
                active_mask = active_mask & still_surprised if k > 0 else still_surprised

                if not bool(active_mask.any().item()):
                    break

            mask_float = active_mask.to(dtype=z.dtype).unsqueeze(-1)
            active_fractions.append(float(active_mask.float().mean().item()))
            token_steps = token_steps + active_mask.float()

            # Active tokens advance via learned damped update; halted tokens remain frozen
            z = z + mask_float * damping * (z_cand - z)

            logits_k = decoder_fn(z)
            # Peek next candidate for energy/surprise estimation
            z_next_cand = self.think_block(z.detach(), h_anchor.detach(), step_idx=k + 1, key_mask=key_mask)
            surprise_k, energy_k = self.compute_surprise(h_anchor, z, z_next_cand, logits_k)

            step_logits.append(logits_k)
            step_energies.append(energy_k)
            surprise_trace.append(surprise_k)
            prev_surprise = surprise_k
            steps_used += 1

        final_surprise = surprise_trace[-1]
        is_unknown = final_surprise > self.abstain_threshold

        self.last_steps_used = steps_used
        self.last_active_fraction = (
            sum(active_fractions) / len(active_fractions) if active_fractions else 0.0
        )
        self.last_token_steps = token_steps
        self.last_surprise_trace = surprise_trace
        self.last_is_unknown = is_unknown

        return {
            "z_final": z,
            "logits_final": step_logits[-1],
            "step_logits": step_logits,
            "step_energies": step_energies,
            "surprise_trace": surprise_trace,
            "token_steps": token_steps,
            "steps_used": steps_used,
            "active_fraction": self.last_active_fraction,
            "is_unknown": is_unknown,
        }

    def compute_thinking_loss(
        self,
        think_out: Dict[str, Any],
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        progress_margin: float = 0.0,
        prog_weight: float = 0.25,
        cal_weight: float = 0.20,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the three-part training objective for Surprise-Driven Latent Thinking:
        1. Deep Supervision Loss (step-weighted cross-entropy across k = 0..K)
        2. Monotonic Progress Loss (penalizes step k if token NLL worsens vs step k-1)
        3. Surprise Calibration Loss (trains energy head to predict token error indicator)

        Args:
            think_out: dict returned by SurpriseDrivenLatentThinker.forward()
            targets: [B, L] ground-truth token indices
            valid_mask: optional [B, L] bool mask of positions to supervise
        """
        step_logits: List[torch.Tensor] = think_out["step_logits"]
        step_energies: List[torch.Tensor] = think_out["step_energies"]
        B, L = targets.shape
        num_steps = len(step_logits)

        if valid_mask is None:
            valid_mask = torch.ones(B, L, dtype=torch.bool, device=targets.device)
        mask_flat = valid_mask.reshape(-1)

        per_step_token_nll: List[torch.Tensor] = []
        per_step_ce: List[torch.Tensor] = []
        cal_losses: List[torch.Tensor] = []

        for k, logits_k in enumerate(step_logits):
            flat_logits = logits_k.reshape(B * L, -1)
            flat_targets = targets.reshape(B * L)
            tok_nll = F.cross_entropy(flat_logits, flat_targets, reduction="none")  # [B*L]
            per_step_token_nll.append(tok_nll)

            if bool(mask_flat.any()):
                ce_k = tok_nll[mask_flat].mean()
            else:
                ce_k = tok_nll.mean()
            per_step_ce.append(ce_k)

            # Calibration target: 1.0 where prediction is wrong, 0.0 where correct
            preds_k = flat_logits.detach().argmax(dim=-1)
            is_err = (preds_k != flat_targets).float()
            energy_flat = step_energies[k].reshape(B * L)
            if bool(mask_flat.any()):
                cal_k = F.binary_cross_entropy(
                    energy_flat[mask_flat].clamp(1e-5, 1.0 - 1e-5),
                    is_err[mask_flat],
                )
            else:
                cal_k = F.binary_cross_entropy(
                    energy_flat.clamp(1e-5, 1.0 - 1e-5),
                    is_err,
                )
            cal_losses.append(cal_k)

        # 1. Step-weighted deep supervision (later steps weighted higher: w_k = k + 1)
        weights = [float(k + 1) for k in range(num_steps)]
        w_sum = sum(weights)
        deep_ce_loss = sum(w * l for w, l in zip(weights, per_step_ce)) / w_sum

        # 2. Monotonic progress loss: step k NLL should be <= step k-1 NLL
        if num_steps > 1:
            prog_terms = []
            for k in range(1, num_steps):
                diff = per_step_token_nll[k] - per_step_token_nll[k - 1].detach() + progress_margin
                if bool(mask_flat.any()):
                    prog_terms.append(F.relu(diff[mask_flat]).mean())
                else:
                    prog_terms.append(F.relu(diff).mean())
            prog_loss = sum(prog_terms) / len(prog_terms)
        else:
            prog_loss = torch.tensor(0.0, device=targets.device)

        # 3. Surprise calibration loss
        cal_loss = sum(cal_losses) / len(cal_losses)

        total_loss = deep_ce_loss + prog_weight * prog_loss + cal_weight * cal_loss
        return {
            "loss": total_loss,
            "deep_ce_loss": deep_ce_loss,
            "prog_loss": prog_loss,
            "cal_loss": cal_loss,
            "step_0_ce": per_step_ce[0].detach(),
            "step_final_ce": per_step_ce[-1].detach(),
        }
