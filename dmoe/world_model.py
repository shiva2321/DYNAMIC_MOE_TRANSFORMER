"""
Internal Latent World Model & Multi-Step Imagination (JEPA + VICReg + Rollout).

Addresses the legacy M17 JEPA caveats and SKIP_landmines ('need world_model.imagine()'):
1. Real-Valued EMA Target Encoder:
   Maintains a stop-gradient EMA copy of the input encoder (`byte_embed`, `chunk_proj`,
   `pos_embed`) so latent targets are stable and non-trivial.
2. Causal Latent Transition Operator & `imagine(z_start, horizon)`:
   Predicts next latent states `z_{t+1} = norm(z_t + delta(z_t, mem_ctx))` and
   supports multi-step latent rollouts without decoding to bytes.
3. Anti-Collapse World-Model Training Loss:
   Combines multi-step causal latent prediction (SmoothL1 + Cosine) with VICReg-style
   variance and off-diagonal covariance regularization to prevent latent collapse.
"""

import copy
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentWorldModel(nn.Module):
    """
    Causal latent transition world model with EMA target encoder and multi-step `imagine()`.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 128,
        ema_decay: float = 0.99,
        max_rollout_horizon: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.ema_decay = float(ema_decay)
        self.max_rollout_horizon = max(1, int(max_rollout_horizon))

        # Transition operator conditioned on [z_t, mem_ctx]
        self.trans_net = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.trans_gate = nn.Linear(d_model * 2, d_model)
        self.out_norm = nn.LayerNorm(d_model)

        # Learned uncertainty / expected surprise head on imagined states
        self.uncertainty_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Lazily initialized EMA target encoder components (mirroring RealSelfStabilizingDMoE.encode)
        self.target_byte_embed: Optional[nn.Embedding] = None
        self.target_chunk_proj: Optional[nn.Linear] = None
        self.target_pos_embed: Optional[nn.Parameter] = None
        self.chunk_size: int = 1

    def bind_encoder(
        self,
        byte_embed: nn.Embedding,
        chunk_proj: nn.Linear,
        pos_embed: nn.Parameter,
        chunk_size: int,
    ) -> None:
        """Create frozen EMA target copies of the encoder modules."""
        self.target_byte_embed = copy.deepcopy(byte_embed)
        self.target_chunk_proj = copy.deepcopy(chunk_proj)
        self.target_pos_embed = nn.Parameter(pos_embed.detach().clone(), requires_grad=False)
        for p in self.target_byte_embed.parameters():
            p.requires_grad_(False)
        for p in self.target_chunk_proj.parameters():
            p.requires_grad_(False)
        self.chunk_size = chunk_size

    @torch.no_grad()
    def update_target_encoder(
        self,
        byte_embed: nn.Embedding,
        chunk_proj: nn.Linear,
        pos_embed: nn.Parameter,
    ) -> None:
        """EMA update of frozen target encoder from online encoder."""
        if self.target_byte_embed is None:
            return
        d = self.ema_decay
        for tp, op in zip(self.target_byte_embed.parameters(), byte_embed.parameters()):
            tp.mul_(d).add_(op.detach() * (1.0 - d))
        for tp, op in zip(self.target_chunk_proj.parameters(), chunk_proj.parameters()):
            tp.mul_(d).add_(op.detach() * (1.0 - d))
        self.target_pos_embed.data.mul_(d).add_(pos_embed.detach() * (1.0 - d))

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """Compute stop-gradient EMA target latents [B, n_chunks, d_model]."""
        if self.target_byte_embed is None or self.target_chunk_proj is None or self.target_pos_embed is None:
            raise RuntimeError("LatentWorldModel.bind_encoder() must be called before encode_target().")
        B, L = x.shape
        embedded = self.target_byte_embed(x)
        n_chunks = L // self.chunk_size
        chunked = embedded.view(B, n_chunks, self.chunk_size * self.d_model)
        h = self.target_chunk_proj(chunked)
        if n_chunks <= self.target_pos_embed.size(1):
            h = h + self.target_pos_embed[:, :n_chunks, :]
        return h

    def predict_next_latent(
        self,
        z_t: torch.Tensor,
        mem_ctx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step causal latent transition.
        z_t: [..., d_model]
        mem_ctx: optional [..., d_model] retrieved memory context
        Returns:
          z_next: [..., d_model]
          uncertainty: [..., 1] in [0, 1]
        """
        if mem_ctx is None:
            mem = torch.zeros_like(z_t)
        else:
            mem = mem_ctx
        inp = torch.cat([z_t, mem], dim=-1)
        delta = self.trans_net(inp)
        gate = torch.sigmoid(self.trans_gate(inp))
        z_next = self.out_norm(z_t + gate * delta)
        unc_feat = torch.cat([z_next.detach(), (z_next - z_t).detach()], dim=-1)
        uncertainty = torch.sigmoid(self.uncertainty_head(unc_feat))
        return z_next, uncertainty

    def imagine(
        self,
        z_start: torch.Tensor,
        horizon: int = 3,
        mem_ctx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-step internal imagination rollout in latent space.
        z_start: [B, L, d_model] or [B, d_model]
        Returns:
          imagined_latents: [B, ..., horizon, d_model]
          imagined_uncertainty: [B, ..., horizon]
        """
        cur = z_start
        latents = []
        uncertainties = []
        for _ in range(horizon):
            cur, unc = self.predict_next_latent(cur, mem_ctx=mem_ctx)
            latents.append(cur)
            uncertainties.append(unc.squeeze(-1))
        return {
            "imagined_latents": torch.stack(latents, dim=-2),
            "imagined_uncertainty": torch.stack(uncertainties, dim=-1),
        }

    @torch.no_grad()
    def compute_predictive_coding_error(
        self,
        wm_imagine: Dict[str, torch.Tensor],
        target_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute causal step-by-step predictive-coding discrepancy in [0, 1].
        At t=0 uses calibrated 1-step prior uncertainty; at t>=1 compares the
        model's imagined transition from t-1 against the observed target at t.
        Strictly causal: token t depends only on inputs <= t.
        """
        imagined = wm_imagine["imagined_latents"]  # [B, L, H, D]
        unc_0 = wm_imagine["imagined_uncertainty"][:, :1, 0]  # [B, 1]
        B, L, _, D = imagined.shape
        if L < 2:
            return unc_0
        pred_prev = imagined[:, :-1, 0, :]
        obs_curr = F.layer_norm(target_latents[:, 1:, :].detach(), (D,))
        pc_rest = (1.0 - F.cosine_similarity(pred_prev, obs_curr, dim=-1)).clamp(0.0, 1.0)
        return torch.cat([unc_0, pc_rest], dim=1)

    def compute_world_model_loss(
        self,
        z_seq: torch.Tensor,
        target_latents: torch.Tensor,
        mem_ctx: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Train the world model on causal multi-step latent prediction + VICReg anti-collapse.
        z_seq: [B, L, d_model] online model latents
        target_latents: [B, L, d_model] stop-gradient EMA target latents
        """
        B, L, D = z_seq.shape
        H = min(self.max_rollout_horizon if horizon is None else horizon, max(1, L - 1))
        if L < 2:
            zero = z_seq.sum() * 0.0
            return {
                "loss": zero,
                "pred_loss": zero,
                "rollout_loss": zero,
                "var_loss": zero,
                "cov_loss": zero,
                "unc_loss": zero,
                "mean_unc": zero,
                "cos_sim_1step": zero,
            }

        cur_z = z_seq
        cur_mem = mem_ctx
        step_losses = []
        unc_losses = []
        cos_1step = z_seq.sum() * 0.0
        mean_unc_1step = z_seq.sum() * 0.0
        pred_1step = z_seq

        for k in range(1, H + 1):
            cur_z, cur_unc = self.predict_next_latent(cur_z, mem_ctx=cur_mem)
            if k == 1:
                pred_1step = cur_z
            pred_slice = cur_z[:, :-k, :]
            unc_slice = cur_unc[:, :-k, 0]
            tgt_slice = target_latents[:, k:, :].detach()
            tgt_norm = F.layer_norm(tgt_slice, (D,))

            l1 = F.smooth_l1_loss(pred_slice, tgt_norm)
            cos_tok = F.cosine_similarity(pred_slice, tgt_norm, dim=-1)
            cos = cos_tok.mean()
            loss_k = l1 + 0.5 * (1.0 - cos)
            step_losses.append(loss_k)

            # Calibrate uncertainty head against actual 1 - cosine prediction error in [0, 1].
            err_target = (1.0 - cos_tok.detach()).clamp(0.0, 1.0)
            unc_k = F.binary_cross_entropy(unc_slice.clamp(1e-5, 1.0 - 1e-5), err_target)
            unc_losses.append(unc_k)

            if k == 1:
                cos_1step = cos.detach()
                mean_unc_1step = unc_slice.mean().detach()

        pred_loss = step_losses[0]
        rollout_loss = torch.stack(step_losses[1:]).mean() if len(step_losses) > 1 else (pred_loss * 0.0)
        unc_loss = torch.stack(unc_losses).mean()

        # VICReg-style anti-collapse regularization on 1-step predictions
        flat_pred = pred_1step.reshape(-1, D)
        std = torch.sqrt(flat_pred.var(dim=0, unbiased=False) + 1e-4)
        var_loss = F.relu(1.0 - std).mean()

        centered = flat_pred - flat_pred.mean(dim=0, keepdim=True)
        normed = centered / std
        corr = (normed.T @ normed) / max(1, flat_pred.size(0))
        off_diag = corr.pow(2) * (1.0 - torch.eye(D, device=corr.device, dtype=corr.dtype))
        cov_loss = off_diag.sum() / max(D * (D - 1), 1)

        total = pred_loss + 0.5 * rollout_loss + 0.1 * var_loss + 0.04 * cov_loss + 0.35 * unc_loss
        return {
            "loss": total,
            "pred_loss": pred_loss.detach(),
            "rollout_loss": rollout_loss.detach(),
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach(),
            "unc_loss": unc_loss.detach(),
            "mean_unc": mean_unc_1step,
            "cos_sim_1step": cos_1step,
        }

