"""
CognitiveDMoE — Unified Consolidated Architecture (v11).

Consolidates every empirically validated mechanism from v1–v10 into a single,
coherent, zero-dead-module end-to-end cognitive Dynamic MoE architecture:

1. Shift-Resilient Tokenizer-Free Sensory Front-End (`MultiScaleByteChunker`):
   - Raw UTF-8 byte stream (no BPE vocabulary), causal multi-scale 1D convs
     (kernels 1, 2, 4, 8), strictly causal BlurPool anti-aliasing before stride-4
     downsampling, and sparse nonnegative multi-scale k-gram descriptors.
2. Dual-Timescale Episodic & Working Memory (`DualTimescaleMemory`):
   - Fast short-term working memory (`GatedDeltaWorkingMemory`, signed-beta DeltaNet
     recurrence) at entry and lower trunk layers.
   - Persistent surprise-gated long-term slot bank with size-invariant null-sink
     rejection (`null_logit + log M`).
   - Dual-Head Closed-Form Ridge Episodic Operator (`LongTermSlotMemory.read_dual_operator`):
     high-rank nonlinear head (`op_rank=48`) for compositional permutation binding +
     low-rank linear-key / MLP-value group head (`op_rank_low=12`) for continuous
     cyclic-shift rule induction on strictly unseen meanings — zero weight updates,
     applied strictly post-Thinker.
3. Hierarchical Hybrid Dynamic-MoE Backbone (6 layers):
   - Shared Lower Trunk (Layers 0..2): DeltaNet STM + Causal RoPE Attention + Shared FFN,
     protected across continual domains by GPM null-space projection (`NullSpaceProjector`).
   - Mid-Depth Continual-Safe Semantic Router (Layer 3): `ThreeLevelAttention` ->
     Surface Null-Space Whitened Orthogonal-Residual `CausalContextTokenRouter` with
     frozen-exempt load balancing, content-masked causal EMA, and strict `k in {0, 1}`
     sparse dispatch (`mean_k <= 1.0`, no wasteful nuance escalation).
   - Upper Shared-Core + Sparse Specialist MoE (Layers 3..5): Always-on shared FFN +
     autonomously spawned/frozen/pruned specialists (`RealExpert`) and `RoutedAdapterBank`
     low-rank adapters sharing ONE sorted scatter-add dispatch plan per forward pass.
4. Internal Latent World Model (`LatentWorldModel`):
   - Stop-gradient EMA deep target encoder (JEPA), causal multi-step rollout `imagine()`,
     VICReg variance/covariance anti-collapse, future InfoNCE, calibrated uncertainty,
     and causal predictive-coding discrepancy.
5. Iterative Latent Reasoner & Epistemic Safety (`SurpriseDrivenLatentThinker` + `abstain_head`):
   - Time-invariant anchor-reinjected recurrent+attention reasoning block with learned
     `SurpriseEnergyHead`, per-token halting, and near/far-OOD epistemic abstention.
6. Non-Token Cognitive Core & Dual Meaning Factorization (`0.0` Decoder CE Gradient into Core):
   - `expr_head` (`z_expr`: direct content-masked pooling + multi-scale k-grams, cross-batch
     equivalence queue, SIGReg isotropy, hard-negative repulsion — no `rel_pool_proj`).
   - `ctx_head` (`z_ctx`: order-sensitive contextual/belief meaning + contradiction & IIT).
   - `fsp_head` (content-masked Future Summary Prediction over hashed byte bigrams).
7. Decoupled Generative Byte Renderer (`RealDecoder` + `generate()`):
   - Shared base + per-expert LoRA residual byte-chunk renderer strictly behind `.detach()`.
8. Always-Plastic Continual Learning Stack (`CognitiveTrainer`):
   - `MuonBONGOptimizer` (Newton-Schulz5 + Grokfast-EMA in `acquire`, vectorized Diagonal BONG
     in `consolidate`) + `NullSpaceProjector` (GPM, readout-head exempt) +
     `LatentQuarantineGate` (`<= 16 KB` SimHash multi-source burst quarantine) +
     `AutonomousExpertLifecycleManager` with automatic state-preserving optimizer rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.belief_training import sigreg_loss
from dmoe.chunker import BYTE_PAD_ID, MultiScaleByteChunker
from dmoe.lifecycle import AutonomousExpertLifecycleManager
from dmoe.memory import DualTimescaleMemory
from dmoe.novelty import NuanceClusterBuffer, UnifiedNoveltyDetector
from dmoe.null_space import NullSpaceProjector
from dmoe.real_model import RealDecoder, RealDynamicMorphicLayer
from dmoe.routing import (
    CausalContextTokenRouter,
    build_token_dispatch_plan,
    build_vectorized_dispatch_plan,
    compute_chunk_surface_features,
)
from dmoe.spgc import (
    LatentQuarantineGate,
    MuonBONGOptimizer,
    classify_v7_spgc_param_groups,
    stablemax_cross_entropy,
)
from dmoe.thinking import SurpriseDrivenLatentThinker
from dmoe.world_model import LatentWorldModel


@dataclass
class CognitiveDMoEConfig:
    """Clean, consolidated configuration for `CognitiveDMoE` (zero legacy bloat)."""

    # Core backbone dimensions
    d_model: int = 192
    num_heads: int = 6
    num_layers: int = 6
    router_layer: int = 3
    d_ff_shared: int = 384
    d_ff_spec: int = 256
    chunk_size: int = 4
    shift_invariant_pool: bool = False
    enable_hnet_boundary: bool = False

    # Attention & working memory
    local_window: int = 4
    chunk_stride: int = 4
    layer_stm_delta_steps: int = 2
    num_ltm_slots: int = 64
    ltm_null_logit: float = 3.2
    auto_write_ltm: bool = True

    # Dual-Head Closed-Form Episodic Induction (High-Rank Ridge + optional Multi-Plane SO(2)^p Procrustes)
    enable_dual_ridge_ep: bool = True
    dual_ridge_op_rank: int = 48
    dual_ridge_op_rank_low: int = 12
    dual_ridge_v_mlp: bool = True
    enable_procrustes_op: bool = False
    op_procrustes_planes: int = 8

    # Continual-safe semantic routing & autonomous lifecycle
    d_route: int = 96
    router_num_keys: int = 8
    max_experts: int = 6
    initial_experts: int = 1
    enable_surface_whitening: bool = True
    frozen_exempt_balance: bool = True
    residual_ortho_router: bool = True
    dual_granularity_router: bool = False
    use_vectorized_dispatch: bool = True
    use_infonce_cris: bool = False
    router_anchor_weight: float = 0.0
    use_loss_free_balance: bool = True
    enable_cris_loss: bool = True
    specialist_init_scale: float = 0.25
    shared_dropout_prob: float = 0.15
    adapter_rank: int = 16
    min_mature_steps: int = 100
    nuance_cluster_min: int = 12
    nuance_cluster_cos: float = 0.65
    z_nuance_thresh: float = 1.6
    d_sem_nuance_thresh: float = 1.6
    max_coactive_experts: int = 4

    # World model & latent thinking
    wm_rollout_horizon: int = 3
    wm_ema_decay: float = 0.94
    max_thinking_steps: int = 3
    skip_threshold: float = 0.12
    theta_halt: float = 0.18
    abstain_threshold: float = 0.65

    # Decoupled byte renderer
    detach_decoder: bool = True
    lora_decoder_rank: int = 16

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CognitiveDMoE(nn.Module):
    """
    Unified Consolidated Cognitive Dynamic MoE Architecture (v11).

    Every module registered on this class is actively wired into the live forward
    and training data paths (zero dead legacy modules).
    """

    def __init__(self, config: Optional[CognitiveDMoEConfig] = None):
        super().__init__()
        self.cfg = config if config is not None else CognitiveDMoEConfig()
        c = self.cfg

        self.d_model = int(c.d_model)
        self.num_heads = int(c.num_heads)
        self.num_layers = int(c.num_layers)
        self.router_layer = min(max(0, int(c.router_layer)), max(0, self.num_layers - 1))
        self._moe_ref_layer_idx: int = self.router_layer
        self.chunk_size = int(c.chunk_size)
        self.detach_decoder = bool(c.detach_decoder)
        self.auto_write_ltm = bool(c.auto_write_ltm)
        self.enable_dual_ridge_ep = bool(c.enable_dual_ridge_ep)
        self.enable_surface_whitening = bool(c.enable_surface_whitening)
        self.enable_cris_loss = bool(c.enable_cris_loss)
        self.use_vectorized_dispatch = bool(c.use_vectorized_dispatch)
        self.use_infonce_cris = bool(c.use_infonce_cris)
        self.router_anchor_weight = float(c.router_anchor_weight)
        self.z_nuance_thresh = float(c.z_nuance_thresh)
        self.d_sem_nuance_thresh = float(c.d_sem_nuance_thresh)
        self.nuance_cluster_min = int(c.nuance_cluster_min)
        self.nuance_cluster_cos = float(c.nuance_cluster_cos)
        self.abstain_threshold = float(c.abstain_threshold)
        self.use_v7_core = True
        self.use_v8_core = True
        self.null_space_projector: Optional[NullSpaceProjector] = None

        # 1. Tokenizer-Free Shift-Resilient Multi-Scale Byte Chunker (+ optional H-Net boundary salience)
        self.ms_chunker = MultiScaleByteChunker(
            d_model=self.d_model,
            chunk_size=self.chunk_size,
            shift_invariant_pool=c.shift_invariant_pool,
            enable_hnet_boundary=c.enable_hnet_boundary,
        )

        # 2. Dual-Timescale Working + Persistent LTM + Dual-Head Ridge / SO(2)^p Procrustes Operator
        self.memory = DualTimescaleMemory(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_ltm_slots=c.num_ltm_slots,
            signed_beta=True,
            ltm_null_logit=c.ltm_null_logit,
            enable_dual_ridge_op=c.enable_dual_ridge_ep,
            op_rank=c.dual_ridge_op_rank,
            op_rank_low=c.dual_ridge_op_rank_low,
            op_v_low_mlp=c.dual_ridge_v_mlp,
            enable_procrustes_op=c.enable_procrustes_op,
            op_procrustes_planes=c.op_procrustes_planes,
        )

        # 3. Hierarchical Hybrid Backbone (Lower Shared Trunk + Mid Router + Upper Sparse MoE)
        self.layers = nn.ModuleList(
            [
                RealDynamicMorphicLayer(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=c.d_ff_spec,
                    initial_experts=c.initial_experts,
                    use_three_level_attn=(l_idx == self.router_layer or l_idx == self.num_layers - 1),
                    local_window=c.local_window,
                    chunk_stride=c.chunk_stride,
                    causal_attn=True,
                    use_rope=True,
                    enable_layer_stm=(l_idx < self.router_layer),
                    signed_beta=True,
                    layer_stm_delta_steps=(c.layer_stm_delta_steps if l_idx == 0 else 1),
                    enable_full_stack_adapters=(l_idx >= self.router_layer),
                    adapter_rank=c.adapter_rank,
                    use_shared_expert=True,
                    d_ff_shared=c.d_ff_shared,
                    v7_trunk_only=(l_idx < self.router_layer),
                    init_spec_small=(l_idx >= self.router_layer),
                    specialist_init_scale=c.specialist_init_scale,
                    shared_dropout_prob=(c.shared_dropout_prob if l_idx >= self.router_layer else 0.0),
                )
                for l_idx in range(self.num_layers)
            ]
        )

        # 4. Continual-Safe Surface-Whitened Orthogonal-Residual Router + Autonomous Lifecycle
        # grad_scale=0.0 isolates trunk representations (Banking77, R@5, subtle edit nuance)
        # from router/CRIS gradients while letting q_proj/ctx_proj refine routing cleanly.
        self.context_router = CausalContextTokenRouter(
            d_model=self.d_model,
            d_route=c.d_route,
            max_experts=c.max_experts,
            initial_experts=c.initial_experts,
            num_keys=c.router_num_keys,
            grad_scale=0.0,
            ema_decay=0.95,
            tau_skip=0.38,
            tau_top2=1.05,  # Strict k in {0, 1} when escalation is disabled -> mean_k <= 1.00
            delta_mix=0.0,
            scale=16.0,
            use_loss_free_balance=c.use_loss_free_balance,
            bias_lr=0.02,
            enable_surface_whitening=c.enable_surface_whitening,
            surface_dim=6,
            frozen_exempt_balance=c.frozen_exempt_balance,
            residual_ortho_router=c.residual_ortho_router,
            dual_granularity=c.dual_granularity_router,
        )
        self.nuance_buffer = NuanceClusterBuffer(cap=128, d_route=c.d_route)
        self.context_nuance_detector = UnifiedNoveltyDetector(d_model=self.d_model)
        self.lifecycle = AutonomousExpertLifecycleManager(
            d_model=self.d_model,
            max_experts=c.max_experts,
            min_mature_steps=c.min_mature_steps,
            max_coactive_experts=c.max_coactive_experts,
        )

        # 5. Internal Latent World Model (EMA Deep Target + Multi-Step Imagination + Retrieval)
        # Keep WM and Thinker compact (<= 384) even when shared FFN scales to 512 for fast convergence & speed.
        d_ff_cog = min(int(c.d_ff_shared), 384)
        self.world_model = LatentWorldModel(
            d_model=self.d_model,
            d_ff=d_ff_cog,
            ema_decay=c.wm_ema_decay,
            max_rollout_horizon=c.wm_rollout_horizon,
        )
        self.world_model.bind_deep_trunk(self.ms_chunker, [self.layers[0]], None)
        self.wm_imagine_proj = nn.Linear(self.d_model, self.d_model)
        self.wm_imagine_gate = nn.Parameter(torch.tensor(0.25))
        self.wm_ret_proj = nn.Linear(self.d_model, self.d_model)
        self.wm_fut_mlp = nn.Sequential(
            nn.Linear(3 * self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.wm_ret_gate = nn.Parameter(torch.tensor(-1.0))

        # 6. Iterative Latent Thinker & Epistemic Abstention
        self.thinker = SurpriseDrivenLatentThinker(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=d_ff_cog,
            max_thinking_steps=c.max_thinking_steps,
            skip_threshold=c.skip_threshold,
            theta_halt=c.theta_halt,
            abstain_threshold=c.abstain_threshold,
            enable_expert_adapters=True,
            adapter_rank=c.adapter_rank,
            initial_experts=c.initial_experts,
            halt_use_energy_only=True,
            skip_final_lookahead=True,
        )
        self.abstain_head = nn.Linear(self.d_model, 1)

        # 7. Dual Meaning Factorization & Non-Token Cognitive Heads (Direct Pooling, NO rel_pool_proj!)
        self.expr_head = nn.Linear(self.d_model, self.d_model)
        self.ctx_head = nn.Linear(self.d_model, self.d_model)
        with torch.no_grad():
            self.expr_head.weight.mul_(0.35).add_(0.65 * torch.eye(self.d_model))
            self.ctx_head.weight.mul_(0.35).add_(0.65 * torch.eye(self.d_model))
        self.expr_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.equiv_proto_bank = nn.Embedding(256, self.d_model)
        self.fsp_head = nn.Linear(self.d_model, 128)
        self.i7_struct_head = nn.Linear(self.d_model, 3)

        # Cross-batch contrastive queues
        self.register_buffer("equiv_q_raw", torch.zeros(256, self.d_model))
        self.register_buffer("equiv_q_proj", torch.zeros(256, self.d_model))
        self.register_buffer("equiv_q_lbl", torch.full((256,), -1, dtype=torch.long))
        self.register_buffer("equiv_q_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("wm_ret_q", torch.zeros(128, self.d_model))
        self.register_buffer("wm_ret_q_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("wm_ret_q_full", torch.zeros(1, dtype=torch.long))
        self._step_ctr: int = 0
        self._equiv_q_ptr_i: int = 0
        self._wm_ret_q_ptr_i: int = 0
        self._wm_ret_q_full_i: int = 0

        # 8. Decoupled Generative Byte Renderer (strictly behind stop_gradient when detach_decoder=True)
        self.decoder = RealDecoder(
            d_model=self.d_model,
            chunk_size=self.chunk_size,
            vocab_size=256,
            enable_lora_residual=True,
            lora_rank=c.lora_decoder_rank,
        )

        # Telemetry
        self.last_assigned_experts: Optional[torch.Tensor] = None
        self.last_routed_expert: int = 0
        self.last_lifecycle_events: List[str] = []

    def attach_null_space_projector(self, projector: Optional[NullSpaceProjector]) -> None:
        self.null_space_projector = projector

    @property
    def byte_embed(self) -> nn.Embedding:
        """Expose chunker byte embedding cleanly (zero duplicate parameters) for eval utilities."""
        return self.ms_chunker.byte_embed

    def reset_ltm(self) -> None:
        self.memory.reset_long_term()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.ms_chunker(x)

    def encode_text(
        self,
        texts: Union[str, bytes, Sequence[Union[str, bytes]]],
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Convert raw UTF-8 strings/bytes into a padded `[B, L]` byte ID tensor on model device."""
        dev = self.ms_chunker.byte_embed.weight.device
        ids = MultiScaleByteChunker.text_to_bytes(texts, device=dev)
        if seq_len is not None:
            L = int(seq_len)
            if ids.size(1) < L:
                ids = F.pad(ids, (0, L - ids.size(1)), value=BYTE_PAD_ID)
            else:
                ids = ids[:, :L]
        else:
            rem = ids.size(1) % self.chunk_size
            if rem != 0:
                ids = F.pad(ids, (0, self.chunk_size - rem), value=BYTE_PAD_ID)
        return ids

    def _compute_chunk_masks(self, x: torch.Tensor, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = x.shape
        C = max(1, self.chunk_size)
        need_L = T * C
        xc = F.pad(x, (0, need_L - L), value=BYTE_PAD_ID) if L < need_L else x[:, :need_L]
        xc = xc.view(B, T, C)
        pad_mask = (xc != BYTE_PAD_ID).any(dim=-1)
        pad_mask = torch.where(pad_mask.any(dim=-1, keepdim=True), pad_mask, torch.ones_like(pad_mask))
        content_mask = ((xc != BYTE_PAD_ID) & (xc != 32) & (xc != 9) & (xc != 10) & (xc != 13)).any(dim=-1)
        content_mask = torch.where(content_mask.any(dim=-1, keepdim=True), content_mask, pad_mask)
        return pad_mask, content_mask

    @staticmethod
    def _direct_content_pool(
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Direct content-masked pooling (0.65 * mean + 0.35 * last_valid) — no rel_pool_proj bottleneck."""
        B, T, D = z.shape
        z_f = z.float()
        if mask is None:
            mask = torch.ones((B, T), dtype=torch.bool, device=z.device)
        w = mask.unsqueeze(-1).float()
        u_mean = (z_f * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        idx_grid = torch.arange(T, device=z.device).unsqueeze(0).expand(B, T)
        last_idx = torch.where(mask, idx_grid, torch.zeros_like(idx_grid)).max(dim=1).values
        u_last = torch.gather(z_f, 1, last_idx.view(B, 1, 1).expand(B, 1, D)).squeeze(1)
        raw = 0.65 * u_mean + 0.35 * u_last
        return raw.to(dtype=z.dtype), F.normalize(raw, dim=-1).to(dtype=z.dtype), (u_last - u_mean).to(dtype=z.dtype)

    @staticmethod
    def _pool_pos_chunks(pos_chunks: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pos_f = pos_chunks.float()
        if mask is not None and mask.size(1) == pos_f.size(1):
            m = mask.unsqueeze(-1).float()
            pos_f = pos_f * m
            pos_mean = pos_f.sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        else:
            pos_mean = pos_f.mean(dim=1)
        pos_max = pos_f.max(dim=1).values
        return F.layer_norm(0.45 * pos_mean + 0.55 * pos_max, (pos_f.size(-1),))

    def project_wm_retrieval(
        self,
        z_pred_raw: torch.Tensor,
        h_ctx: Optional[torch.Tensor] = None,
        pos_pre: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base = z_pred_raw.float() + 0.5 * self.wm_ret_proj(z_pred_raw).float()
        if h_ctx is not None and pos_pre is not None:
            alpha = torch.sigmoid(self.wm_ret_gate.float())
            mlp_in = torch.cat([base, h_ctx.float(), pos_pre.float()], dim=-1).to(
                dtype=self.wm_fut_mlp[0].weight.dtype
            )
            delta = F.normalize(self.wm_fut_mlp(mlp_in).float(), dim=-1)
            return F.normalize(alpha * F.normalize(pos_pre.float(), dim=-1) + (1.0 - alpha) * delta, dim=-1)
        return base.to(dtype=z_pred_raw.dtype)

    def _build_dispatch(self, top_ids: torch.Tensor, top_w: torch.Tensor, num_experts: int) -> Any:
        if self.use_vectorized_dispatch and num_experts <= 8:
            return build_vectorized_dispatch_plan(top_ids, top_w, num_experts=num_experts)
        return build_token_dispatch_plan(top_ids, top_w, num_experts=num_experts)

    def _forward_trunk_to_post_thinker(
        self,
        x: torch.Tensor,
        h_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Strictly causal deep forward through memory, backbone layers, WM, and Thinker (no stats)."""
        h = self.encode(x) if h_enc is None else h_enc
        byte_summary = self.ms_chunker.last_causal_byte_summary if h_enc is None else None
        T = h.size(1)
        _, content_mask = self._compute_chunk_masks(x, T)
        h = self.memory(h, update_ltm=False, return_stm_state=False)["fused"]
        for l_idx in range(self.router_layer):
            h, _ = self.layers[l_idx](h, update_stats=False)
        r_layer = self.layers[self.router_layer]
        h_attn3 = r_layer.forward_context(h)
        surface_feats = (
            compute_chunk_surface_features(x, chunk_size=self.chunk_size, num_chunks=T, format_only=True)
            if self.enable_surface_whitening
            else None
        )
        expert_states = list(self.lifecycle.expert_states)
        q_state = self.context_router.compute_queries(
            h_attn3,
            expert_states=expert_states,
            surface_feats=surface_feats,
            update_surface_stats=False,
            content_mask=content_mask,
            byte_summary=byte_summary,
        )
        route_res = self.context_router.select_experts(
            q_state,
            nuance_mask=None,
            content_mask=content_mask,
        )
        dispatch_plan = self._build_dispatch(
            route_res["top_ids"], route_res["top_w"], num_experts=len(r_layer.moe.experts)
        )
        h, _ = r_layer.forward_moe(h_attn3, update_stats=False, dispatch_plan=dispatch_plan)
        for layer in self.layers[self.router_layer + 1 :]:
            h, _ = layer(h, update_stats=False, dispatch_plan=dispatch_plan)
        wm_im = self.world_model.imagine(h, horizon=1)
        z_im = wm_im["imagined_latents"][:, :, 0, :]
        conf = (1.0 - wm_im["imagined_uncertainty"][:, :, :1]).detach()
        h = h + torch.tanh(self.wm_imagine_gate) * conf * self.wm_imagine_proj(z_im)
        t_out = self.thinker(
            h,
            decoder_fn=None,
            force_steps=None,
            nuance_mask=None,
            dispatch_plan=dispatch_plan,
        )
        return t_out["z_final"]

    def extract_router_queries(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lightweight lower-trunk + router query extraction `(rq_pool, q_tok)` for perturbation invariance.

        Runs lower shared trunk under `no_grad` (matching `grad_scale=0.0`) so surface-perturbation
        invariance optimizes `context_router` cleanly with zero trunk corruption and ~60s faster Track 2.
        """
        with torch.no_grad():
            h = self.encode(x)
            byte_summary = self.ms_chunker.last_causal_byte_summary
            T = h.size(1)
            _, content_mask = self._compute_chunk_masks(x, T)
            h = self.memory(h, update_ltm=False, return_stm_state=False)["fused"]
            for l_idx in range(self.router_layer):
                h, _ = self.layers[l_idx](h, update_stats=False)
            h_attn3 = self.layers[self.router_layer].forward_context(h)
        surface_feats = (
            compute_chunk_surface_features(x, chunk_size=self.chunk_size, num_chunks=T, format_only=True)
            if self.enable_surface_whitening
            else None
        )
        q_state = self.context_router.compute_queries(
            h_attn3,
            expert_states=list(self.lifecycle.expert_states),
            surface_feats=surface_feats,
            update_surface_stats=False,
            content_mask=content_mask,
            byte_summary=byte_summary,
        )
        q_tok = q_state["q_tok"]
        w_cm = content_mask.unsqueeze(-1).float()
        rq_pool = F.normalize((q_tok.float() * w_cm).sum(dim=1) / w_cm.sum(dim=1).clamp_min(1.0), dim=-1)
        return rq_pool, q_tok

    def extract_episode_kv(
        self,
        x_ep: torch.Tensor,
        top_k: int = 6,
        min_nll: float = 0.0,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract strictly causal episodic key-value pairs AND Dual-Head Closed-Form Ridge
        operator pairs `(op_keys, op_vals, op_mask)` from support episode byte tensors `x_ep [B, L]`.
        Zero future leakage: key at `t = last_content - 1` depends strictly on chunks `<= t`.
        """
        if x_ep.size(1) < max(2, 2 * self.chunk_size):
            return None
        B = x_ep.size(0)
        h_ep_enc = self.encode(x_ep)
        T_full = h_ep_enc.size(1)
        T_m = T_full - 1
        if T_m < 1:
            return None
        h_ep_ctx, _ = self.layers[0](h_ep_enc, update_stats=False)
        _, content_mask = self._compute_chunk_masks(x_ep, T_full)
        k = min(max(1, int(top_k)), T_m)
        top_idx = torch.arange(T_m - k, T_m, device=x_ep.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        ep_mask = content_mask[:, 1 : T_m + 1].gather(1, top_idx)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, h_ep_enc.size(-1))
        raw_k = torch.gather(h_ep_ctx[:, :-1, :], 1, gather_idx)
        raw_v = torch.gather(h_ep_enc[:, 1:, :], 1, gather_idx)
        ep_keys, ep_vals = self.memory.long_term.encode_kv(raw_k, raw_v, normalize_k=False)
        res: Dict[str, torch.Tensor] = {
            "ep_keys": ep_keys,
            "ep_vals": ep_vals,
            "ep_mask": ep_mask,
            "ep_salience": torch.ones_like(ep_mask, dtype=torch.float32),
        }
        if self.enable_dual_ridge_ep and self.memory.long_term.enable_dual_ridge_op:
            with torch.no_grad():
                h_deep = self._forward_trunk_to_post_thinker(x_ep, h_enc=h_ep_enc.detach())
            t_grid = torch.arange(T_full, device=x_ep.device).unsqueeze(0).expand(B, T_full)
            last_c = torch.where(content_mask, t_grid, torch.zeros_like(t_grid)).max(dim=1).values.clamp(min=1)
            key_idx = (last_c - 1).view(B, 1, 1).expand(B, 1, h_deep.size(-1))
            val_idx = last_c.view(B, 1, 1).expand(B, 1, h_ep_enc.size(-1))
            raw_op_k = torch.gather(h_deep, 1, key_idx)
            # Detach raw support answer chunks so per-episode random labels never distort ms_chunker
            raw_op_v = torch.gather(h_ep_enc.detach(), 1, val_idx)
            op_keys, op_vals = self.memory.long_term.encode_op_kv(raw_op_k, raw_op_v)
            op_mask = content_mask.gather(1, last_c.unsqueeze(1))
            res["op_keys"] = op_keys
            res["op_vals"] = op_vals
            res["op_mask"] = op_mask
        return res

    @torch.no_grad()
    def induce_from_episode(
        self,
        support_bytes: torch.Tensor,
        query_bytes: torch.Tensor,
        top_k: int = 6,
    ) -> Dict[str, Any]:
        """
        One-call zero-weight-update episodic rule induction:
          - `support_bytes`: `[E, n_shots, L_shot]` or `[n_shots, L_shot]`
          - `query_bytes`: `[E, L_q]` or `[1, L_q]`
        """
        was_training = self.training
        self.eval()
        try:
            if support_bytes.dim() == 2:
                support_bytes = support_bytes.unsqueeze(0)
            if query_bytes.dim() == 1:
                query_bytes = query_bytes.unsqueeze(0)
            e_sz, n_sh, l_sh = support_bytes.shape
            kv_flat = self.extract_episode_kv(support_bytes.reshape(e_sz * n_sh, l_sh), top_k=top_k, min_nll=0.0)
            if kv_flat is None:
                return self.forward(query_bytes, update_stats=False)
            k_sh = kv_flat["ep_keys"].size(1)
            d_kv = kv_flat["ep_keys"].size(-1)
            ep_mem: Dict[str, torch.Tensor] = {
                "ep_keys": kv_flat["ep_keys"].view(e_sz, n_sh * k_sh, d_kv),
                "ep_vals": kv_flat["ep_vals"].view(e_sz, n_sh * k_sh, d_kv),
                "ep_mask": kv_flat["ep_mask"].view(e_sz, n_sh * k_sh),
            }
            if "op_keys" in kv_flat and "op_vals" in kv_flat:
                ok_sh = kv_flat["op_keys"].size(1)
                ep_mem["op_keys"] = kv_flat["op_keys"].view(e_sz, n_sh * ok_sh, kv_flat["op_keys"].size(-1))
                ep_mem["op_vals"] = kv_flat["op_vals"].view(e_sz, n_sh * ok_sh, kv_flat["op_vals"].size(-1))
                ep_mem["op_mask"] = kv_flat["op_mask"].view(e_sz, n_sh * ok_sh)
            return self.forward(query_bytes, update_stats=False, episode_memory=ep_mem)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def consolidate_ltm(self, x_ep: torch.Tensor, top_k: int = 6) -> int:
        """Write episodic transitions into persistent LongTermSlotMemory."""
        was_training = self.training
        self.eval()
        try:
            kv = self.extract_episode_kv(x_ep, top_k=top_k, min_nll=0.0)
            if kv is None:
                return 0
            mask = kv["ep_mask"].reshape(-1)
            if not bool(mask.any().item()):
                return 0
            k_cand = F.normalize(kv["ep_keys"].reshape(-1, self.d_model)[mask], p=2, dim=-1, eps=1e-6)
            v_cand = kv["ep_vals"].reshape(-1, self.d_model)[mask]
            sal = kv["ep_salience"].reshape(-1)[mask]
            return self.memory.long_term._insert_kv_vectors(k_cand, v_cand, sal)
        finally:
            self.train(was_training)

    def forward(
        self,
        x: torch.Tensor,
        update_stats: bool = True,
        force_thinking_steps: Optional[int] = None,
        episode_memory: Optional[Dict[str, torch.Tensor]] = None,
        belief_conflict: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Full end-to-end forward pass through the consolidated cognitive core and detached renderer."""
        B = x.size(0)
        if self.training:
            self._step_ctr += 1

        # 1. Sensory lifting
        h = self.encode(x)
        h_embed_seq = h
        pos_chunks_seq = self.ms_chunker.last_pos_chunks
        byte_summary_seq = self.ms_chunker.last_causal_byte_summary
        T = h.size(1)
        pad_mask, content_mask = self._compute_chunk_masks(x, T)

        # 2. Dual-timescale memory entry (skip Layer-0 ep KV when post-Thinker Dual Ridge handles episodes)
        use_dual_op_active = bool(
            self.enable_dual_ridge_ep
            and episode_memory is not None
            and episode_memory.get("op_keys") is not None
            and episode_memory.get("op_vals") is not None
        )
        ep_k = None if use_dual_op_active else (episode_memory.get("ep_keys") if episode_memory is not None else None)
        ep_v = None if use_dual_op_active else (episode_memory.get("ep_vals") if episode_memory is not None else None)
        ep_m = None if use_dual_op_active else (episode_memory.get("ep_mask") if episode_memory is not None else None)

        do_ltm_w = bool(self.training and self.auto_write_ltm and update_stats and (self._step_ctr % 4 == 0))
        mem_out = self.memory(
            h,
            update_ltm=do_ltm_w,
            ep_keys=ep_k,
            ep_vals=ep_v,
            ep_mask=ep_m,
            return_stm_state=False,
        )
        h = mem_out["fused"]
        ctx_parts = []
        if self.memory.long_term._num_used_slots > 0:
            used_idx = self.memory.long_term.slot_used.nonzero(as_tuple=True)[0]
            if used_idx.numel() > 0:
                ctx_parts.append(self.memory.long_term.slot_vals[used_idx].unsqueeze(0).expand(B, -1, -1))
        if ep_v is not None and ep_v.size(1) > 0:
            ctx_parts.append(ep_v * ep_m.unsqueeze(-1).to(ep_v.dtype) if ep_m is not None else ep_v)
        memory_context = torch.cat(ctx_parts, dim=1) if ctx_parts else None
        mem_ctx_tok = mem_out["ltm_out"] + mem_out["stm_out"]

        # 3. Stop-gradient EMA deep target for World Model JEPA
        tgt_lat = self.world_model.encode_target(x)

        # 4. Shared lower trunk (Layers 0 .. router_layer - 1)
        for l_idx in range(self.router_layer):
            h, _ = self.layers[l_idx](h, update_stats=update_stats, memory_context=memory_context)
            if (
                l_idx == 0
                and (not use_dual_op_active)
                and episode_memory is not None
                and episode_memory.get("ep_keys") is not None
                and episode_memory.get("ep_vals") is not None
            ):
                ep_l0, _ = self.memory.long_term.read(
                    h,
                    ep_keys=episode_memory["ep_keys"],
                    ep_vals=episode_memory["ep_vals"],
                    ep_mask=episode_memory.get("ep_mask"),
                    include_null_for_ep=True,
                )
                h = h + 0.65 * ep_l0.to(h.dtype)

        # 5. Mid-depth ThreeLevelAttention -> Continual-Safe Semantic Router -> Sparse Dispatch
        r_layer = self.layers[self.router_layer]
        h_attn3 = r_layer.forward_context(h, memory_context=memory_context)
        wm_mid = self.world_model.imagine(h_attn3, horizon=1, mem_ctx=mem_ctx_tok)
        pc_err_mid = self.world_model.compute_predictive_coding_error(wm_mid, tgt_lat)
        wm_unc_mid = wm_mid["imagined_uncertainty"][:, :, 0]

        expert_states = list(self.lifecycle.expert_states)
        surface_feats = (
            compute_chunk_surface_features(x, chunk_size=self.chunk_size, num_chunks=T, format_only=True)
            if self.enable_surface_whitening
            else None
        )
        q_state = self.context_router.compute_queries(
            h_attn3,
            expert_states=expert_states,
            surface_feats=surface_feats,
            update_surface_stats=update_stats,
            content_mask=content_mask,
            byte_summary=byte_summary_seq,
        )
        nuance_info = self.context_nuance_detector.evaluate_contextual_chunks(
            d_ctx=q_state["d_ctx"],
            primary_experts=q_state["probs"].argmax(dim=-1),
            pc_err=pc_err_mid,
            wm_unc=wm_unc_mid,
            belief_conflict=belief_conflict,
            z_nuance_thresh=self.z_nuance_thresh,
            d_sem_thresh=self.d_sem_nuance_thresh,
            update_stats=update_stats,
        )
        # Decoupled lifecycle novelty detection without wasteful k=2 escalation!
        lifecycle_nuance_mask = nuance_info["nuance_mask"] & content_mask
        nuance_mask = torch.zeros(B, T, dtype=torch.bool, device=h.device)

        route_res = self.context_router.select_experts(
            q_state,
            nuance_mask=nuance_mask,
            content_mask=content_mask,
        )
        top_ids = route_res["top_ids"]
        top_w = route_res["top_w"]
        k_used = route_res["k_used"]
        moe_ref = r_layer.moe
        dispatch_plan = self._build_dispatch(top_ids, top_w, num_experts=len(moe_ref.experts))

        # 6. Upper dynamic shared-core + sparse specialist MoE layers
        h, _ = r_layer.forward_moe(h_attn3, update_stats=update_stats, dispatch_plan=dispatch_plan)
        for layer in self.layers[self.router_layer + 1 :]:
            h, _ = layer(
                h,
                update_stats=update_stats,
                memory_context=memory_context,
                dispatch_plan=dispatch_plan,
            )

        # 7. Post-MoE World Model multi-step imagination & EMA deep target update
        cm_f = content_mask.unsqueeze(-1).to(h.dtype)
        cum_h = torch.cumsum(h * cm_f, dim=1) / torch.cumsum(cm_f, dim=1).clamp_min(1.0)
        wm_mem = 0.5 * mem_ctx_tok + 0.5 * cum_h
        wm_imagine = self.world_model.imagine(
            h,
            horizon=self.world_model.max_rollout_horizon,
            mem_ctx=wm_mem,
        )
        pc_err_pre = self.world_model.compute_predictive_coding_error(wm_imagine, tgt_lat)
        wm_imagine["predictive_coding_error"] = pc_err_pre
        h_wm = 0.2 * h + 0.8 * h.detach()
        wm_loss_dict = self.world_model.compute_world_model_loss(h_wm, tgt_lat, mem_ctx=wm_mem)
        if self.training and update_stats and (self._step_ctr % 2 == 0):
            self.world_model.update_deep_trunk(self.ms_chunker, [self.layers[0]], None)
        z_imag_1 = wm_imagine["imagined_latents"][:, :, 0, :]
        conf_1 = (1.0 - wm_imagine["imagined_uncertainty"][:, :, :1]).detach()
        h = h + torch.tanh(self.wm_imagine_gate) * conf_1 * self.wm_imagine_proj(z_imag_1)

        # 8. Iterative Latent Thinker (energy-only halting, zero decoder calls inside loop)
        think_out = self.thinker(
            h,
            decoder_fn=None,
            force_steps=force_thinking_steps,
            nuance_mask=nuance_mask,
            dispatch_plan=dispatch_plan,
        )
        h = think_out["z_final"]

        # 9. Post-Thinker Dual-Head Closed-Form Ridge Episodic Operator (zero weight updates)
        if use_dual_op_active:
            h_for_op = h.detach() if self.training else h
            op_delta = self.memory.long_term.read_dual_operator(
                h_for_op,
                op_keys=episode_memory["op_keys"],
                op_vals=episode_memory["op_vals"],
                op_mask=episode_memory.get("op_mask"),
            )
            g_ep = torch.sigmoid(self.memory.long_term.op_override_logit).to(h.dtype)
            h = (1.0 - g_ep) * h_for_op + op_delta.to(h.dtype)

        # 10. Dual Meaning Factorization (Direct content-masked pooling, NO rel_pool_proj!)
        z_expr = self.expr_head(h)
        z_ctx = self.ctx_head(h)
        fsp_logits = self.fsp_head(h)

        h_emb_pool, _, _ = self._direct_content_pool(h_embed_seq, mask=content_mask)
        h_raw_pool, _, _ = self._direct_content_pool(h, mask=content_mask)
        ze_raw_head, _, _ = self._direct_content_pool(z_expr, mask=content_mask)
        if pos_chunks_seq is not None and pos_chunks_seq.size(1) == T:
            cm_p = content_mask.unsqueeze(-1).float()
            pos_c_f = pos_chunks_seq.float() * cm_p
            pos_mean = pos_c_f.sum(dim=1) / cm_p.sum(dim=1).clamp_min(1.0)
            pos_max = pos_c_f.max(dim=1).values
            pos_ngram_pool = (0.5 * pos_mean + 0.5 * pos_max).to(dtype=h_raw_pool.dtype)
            z_expr_raw_pool = 0.35 * pos_ngram_pool + 0.15 * h_emb_pool + 0.30 * h_raw_pool + 0.20 * ze_raw_head
        else:
            z_expr_raw_pool = 0.35 * h_emb_pool + 0.40 * h_raw_pool + 0.25 * ze_raw_head
        z_expr_pool = F.normalize(z_expr_raw_pool, dim=-1)
        z_expr_contrast = F.normalize(self.expr_proj(z_expr_raw_pool), dim=-1)

        z_ctx_raw_pool, z_ctx_pool, rel_dir_vec = self._direct_content_pool(z_ctx, mask=content_mask)
        i7_struct_logits = self.i7_struct_head(z_ctx_raw_pool)
        abstain_logits = self.abstain_head(h)

        # 11. Decoupled Byte Renderer strictly behind stop_gradient
        logits = self.decoder(h.detach() if self.detach_decoder else h, dispatch_plan=dispatch_plan)

        # 12. Router key updates & autonomous specialist lifecycle
        lifecycle_events: List[str] = []
        if update_stats:
            self.context_router.update_keys(route_res["q_tok"], top_ids, top_w, expert_states=expert_states)
            _, lifecycle_events = self.lifecycle.step_contextual_nuance_lifecycle(
                self,
                nuance_mask=lifecycle_nuance_mask,
                q_tok=route_res["q_tok"],
                nuance_score=nuance_info["nuance_score"],
                top_ids=top_ids,
                top_w=top_w,
                nuance_buffer=self.nuance_buffer,
                is_training=self.training,
                cluster_min_count=self.nuance_cluster_min,
                cluster_min_cos=self.nuance_cluster_cos,
            )

        # Modal expert per sequence strictly over active content chunks
        n_exp_cur = len(moe_ref.experts)
        c_active = content_mask & (k_used > 0)
        eff_mask = torch.where(c_active.any(dim=1, keepdim=True), c_active, content_mask)
        oh_e = F.one_hot(top_ids[..., 0].clamp(0, max(0, n_exp_cur - 1)), num_classes=max(1, n_exp_cur)).float()
        seq_modal = (oh_e * eff_mask.unsqueeze(-1).float()).sum(dim=1).argmax(dim=-1)
        self.last_assigned_experts = seq_modal
        self.last_routed_expert = int(seq_modal[0].item()) if seq_modal.numel() > 0 else 0
        self.last_lifecycle_events = lifecycle_events

        return {
            "output": logits,
            "latent": h,
            "z_expr": z_expr,
            "z_ctx": z_ctx,
            "z_expr_raw_pool": z_expr_raw_pool,
            "z_expr_pool": z_expr_pool,
            "z_expr_contrast": z_expr_contrast,
            "z_ctx_raw_pool": z_ctx_raw_pool,
            "z_ctx_pool": z_ctx_pool,
            "rel_dir_vec": rel_dir_vec,
            "chunk_pad_mask": pad_mask,
            "chunk_content_mask": content_mask,
            "i7_struct_logits": i7_struct_logits,
            "fsp_logits": fsp_logits,
            "tgt_lat": tgt_lat.detach(),
            "abstain_logits": abstain_logits,
            "router_q": route_res["q_tok"],
            "router_probs": route_res["probs"],
            "need": route_res["need"],
            "top_ids": top_ids,
            "top_w": top_w,
            "k_used": k_used,
            "d_ctx": route_res["d_ctx"],
            "d_sem": nuance_info["d_sem"],
            "z_ctx_nuance": nuance_info["z_ctx"],
            "nuance_mask": nuance_mask,
            "lifecycle_nuance_mask": lifecycle_nuance_mask,
            "nuance_score": nuance_info["nuance_score"],
            "router_balance_loss": route_res["balance_loss"],
            "router_erc_loss": route_res["erc_loss"],
            "assigned_experts": seq_modal,
            "thinking_out": think_out,
            "thinker_z_trace": think_out.get("z_trace"),
            "thinking_steps_used": think_out["steps_used"],
            "active_token_fraction": think_out["active_fraction"],
            "is_unknown": think_out["is_unknown"],
            "lifecycle_events": lifecycle_events,
            "num_experts": len(moe_ref.experts),
            "expert_states": list(self.lifecycle.expert_states),
            "memory_out": mem_out,
            "level_weights": [l.last_level_weights for l in self.layers if l.last_level_weights is not None],
            "world_model_imagine": wm_imagine,
            "world_model_loss": wm_loss_dict,
            "pre_pc_error": pc_err_pre,
            "pos_chunks": pos_chunks_seq,
        }

    def forward_interchange(
        self,
        x_base: torch.Tensor,
        x_donor: torch.Tensor,
        *,
        intervene_chunks: Optional[Tuple[int, int]] = None,
        dim_slice: Optional[Tuple[int, int]] = None,
        force_thinking_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Causal Abstraction Interchange-Intervention forward pass at `router_layer`."""
        hb = self.encode(x_base)
        hd = self.encode(x_donor)
        T = hb.size(1)
        _, content_mask = self._compute_chunk_masks(x_base, T)
        hb = self.memory(hb, update_ltm=False)["fused"]
        hd = self.memory(hd, update_ltm=False)["fused"]
        for l_idx in range(self.router_layer):
            hb, _ = self.layers[l_idx](hb, update_stats=False)
            hd, _ = self.layers[l_idx](hd, update_stats=False)
        r_layer = self.layers[self.router_layer]
        hb_ctx = r_layer.forward_context(hb)
        hd_ctx = r_layer.forward_context(hd)

        h_int = hb_ctx.clone()
        if intervene_chunks is not None:
            c0, c1 = max(0, int(intervene_chunks[0])), min(T, int(intervene_chunks[1]))
            if c1 > c0:
                if dim_slice is not None:
                    d0, d1 = max(0, int(dim_slice[0])), min(self.d_model, int(dim_slice[1]))
                    h_int[:, c0:c1, d0:d1] = hd_ctx[:, c0:c1, d0:d1]
                else:
                    h_int[:, c0:c1, :] = hd_ctx[:, c0:c1, :]
        elif dim_slice is not None:
            d0, d1 = max(0, int(dim_slice[0])), min(self.d_model, int(dim_slice[1]))
            if d1 > d0:
                h_int[:, :, d0:d1] = hd_ctx[:, :, d0:d1]

        q_state = self.context_router.compute_queries(
            h_int,
            expert_states=list(self.lifecycle.expert_states),
            content_mask=content_mask,
        )
        route_res = self.context_router.select_experts(
            q_state,
            nuance_mask=None,
            content_mask=content_mask,
        )
        dispatch_plan = self._build_dispatch(
            route_res["top_ids"], route_res["top_w"], num_experts=len(r_layer.moe.experts)
        )
        h, _ = r_layer.forward_moe(h_int, update_stats=False, dispatch_plan=dispatch_plan)
        for layer in self.layers[self.router_layer + 1 :]:
            h, _ = layer(h, update_stats=False, dispatch_plan=dispatch_plan)
        wm_im = self.world_model.imagine(h, horizon=1)
        z_im = wm_im["imagined_latents"][:, :, 0, :]
        conf = (1.0 - wm_im["imagined_uncertainty"][:, :, :1]).detach()
        h = h + torch.tanh(self.wm_imagine_gate) * conf * self.wm_imagine_proj(z_im)
        t_out = self.thinker(
            h,
            decoder_fn=None,
            force_steps=force_thinking_steps,
            nuance_mask=None,
            dispatch_plan=dispatch_plan,
        )
        h = t_out["z_final"]
        _, ze_p, _ = self._direct_content_pool(self.expr_head(h), mask=content_mask)
        _, zc_p, _ = self._direct_content_pool(self.ctx_head(h), mask=content_mask)
        return {
            "latent": h,
            "z_expr_pool": ze_p,
            "z_ctx_pool": zc_p,
            "top_ids": route_res["top_ids"],
            "k_used": route_res["k_used"],
        }

    def render_text_detached(self, out: Dict[str, Any]) -> torch.Tensor:
        """Render byte-chunk logits strictly behind stop_gradient from converged cognitive state."""
        h_fin = out["latent"].detach()
        dispatch_plan = self._build_dispatch(
            out["top_ids"].detach(),
            out["top_w"].detach(),
            num_experts=int(out.get("num_experts", 1)),
        )
        return self.decoder(h_fin, dispatch_plan=dispatch_plan)

    def compute_detached_decoder_loss(self, out: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
        """
        Train ONLY the decoupled byte renderer `self.decoder` on next-chunk byte prediction.
        Guaranteed 0.0 gradient into the cognitive trunk (`out['latent'].detach()`).
        """
        logits = self.render_text_detached(out)  # [B, T, chunk_size, 256]
        B, T, C, V = logits.shape
        if T < 2:
            return logits.sum() * 0.0
        need_L = T * C
        xc = F.pad(x.long(), (0, need_L - x.size(1)), value=BYTE_PAD_ID) if x.size(1) < need_L else x[:, :need_L].long()
        xc = xc.view(B, T, C)
        tgt = xc[:, 1:, :]  # next chunk bytes [B, T-1, C]
        pred = logits[:, :-1, :, :]  # [B, T-1, C, 256]
        valid = (tgt >= 0) & (tgt < 256)
        if not bool(valid.any().item()):
            return logits.sum() * 0.0
        return F.cross_entropy(pred[valid].float(), tgt[valid])

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, bytes, Sequence[Union[str, bytes]], torch.Tensor],
        max_new_chunks: int = 8,
        temperature: float = 0.8,
        greedy: bool = True,
        episode_memory: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Autoregressive byte-chunk generation from the cognitive core + decoupled renderer.
        Returns generated UTF-8 strings, full byte tensor, and per-step epistemic abstention trace.
        """
        was_training = self.training
        self.eval()
        try:
            if isinstance(prompt, torch.Tensor):
                cur = prompt.long().to(self.ms_chunker.byte_embed.weight.device)
                if cur.dim() == 1:
                    cur = cur.unsqueeze(0)
                rem = cur.size(1) % self.chunk_size
                if rem != 0:
                    cur = F.pad(cur, (0, self.chunk_size - rem), value=BYTE_PAD_ID)
            else:
                cur = self.encode_text(prompt)

            abstain_trace: List[torch.Tensor] = []
            expert_trace: List[torch.Tensor] = []
            for _ in range(max(0, int(max_new_chunks))):
                out = self.forward(cur, update_stats=False, episode_memory=episode_memory)
                next_logits = out["output"][:, -1, :, :]  # [B, chunk_size, 256]
                if greedy or temperature <= 1e-6:
                    next_bytes = next_logits.argmax(dim=-1)
                else:
                    probs = F.softmax(next_logits.float() / float(temperature), dim=-1)
                    B_g, C_g, V_g = probs.shape
                    next_bytes = torch.multinomial(probs.view(B_g * C_g, V_g), num_samples=1).view(B_g, C_g)
                cur = torch.cat([cur, next_bytes], dim=1)
                abstain_trace.append(torch.sigmoid(out["abstain_logits"][:, -1, 0]))
                expert_trace.append(out["assigned_experts"])

            return {
                "byte_ids": cur,
                "texts": MultiScaleByteChunker.bytes_to_text(cur),
                "abstain_probs": torch.stack(abstain_trace, dim=1) if abstain_trace else cur.new_zeros((cur.size(0), 0), dtype=torch.float32),
                "assigned_experts_trace": torch.stack(expert_trace, dim=1) if expert_trace else cur.new_zeros((cur.size(0), 0), dtype=torch.long),
            }
        finally:
            self.train(was_training)

    @staticmethod
    def _compute_future_bigram_summary(
        x: torch.Tensor,
        chunk_size: int = 4,
        num_bins: int = 128,
        horizon_chunks: int = 4,
    ) -> torch.Tensor:
        B, L = x.shape
        T = L // chunk_size
        dev = x.device
        xc = x[:, : T * chunk_size].view(B, T, chunk_size).long()
        hist = torch.full((B, T, num_bins), 1e-4, device=dev, dtype=torch.float32)
        hist.scatter_add_(-1, xc % num_bins, torch.ones((B, T, chunk_size), device=dev, dtype=torch.float32))
        if chunk_size >= 2:
            bg = (xc[..., :-1] * 31 + xc[..., 1:]) % num_bins
            hist.scatter_add_(-1, bg, torch.ones((B, T, chunk_size - 1), device=dev, dtype=torch.float32))
        if T < 2:
            return hist / hist.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        fut = torch.zeros((B, T - 1, num_bins), device=dev, dtype=torch.float32)
        for h_i in range(1, min(horizon_chunks, T) + 1):
            span = T - h_i
            if span <= 0:
                break
            fut[:, :span, :] += hist[:, h_i : h_i + span, :]
        return fut / fut.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _semantic_state_loss(
        self,
        ze: torch.Tensor,
        zc: torch.Tensor,
        *,
        equiv_labels: Optional[torch.Tensor] = None,
        contra_pair_idx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hard_neg_pair_idx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        content_mask: Optional[torch.Tensor] = None,
        sigreg_weight: float = 0.25,
        equiv_weight: float = 1.0,
        contra_weight: float = 1.0,
        ze_pool_override: Optional[torch.Tensor] = None,
        update_queue: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        T = zc.size(1)
        if ze_pool_override is not None:
            ze_raw = ze_pool_override.float()
            z_ep = F.normalize(ze_raw, dim=-1)
        else:
            ze_raw, z_ep, _ = self._direct_content_pool(ze.float(), mask=content_mask)
        z_ep_c = F.normalize(self.expr_proj(ze_raw.to(dtype=ze.dtype)).float(), dim=-1)
        _, z_cp, _ = self._direct_content_pool(zc.float(), mask=content_mask)
        tail_m = content_mask[:, max(0, T // 2) :] if content_mask is not None else None
        if tail_m is not None:
            tail_m = torch.where(tail_m.any(dim=-1, keepdim=True), tail_m, torch.ones_like(tail_m))
        _, z_c_tail, _ = self._direct_content_pool(zc[:, max(0, T // 2) :].float(), mask=tail_m)

        dev = z_ep.device
        zero = z_ep.sum() * 0.0
        l_sigreg = (
            sigreg_loss(F.layer_norm(ze_raw, (ze_raw.size(-1),))).to(ze.dtype)
            if z_ep.size(0) >= 4
            else zero
        )

        l_equiv = zero
        if equiv_labels is not None and z_ep.size(0) >= 2:
            y = equiv_labels.to(dev)
            B_e = y.size(0)
            same = ((y.unsqueeze(1) == y.unsqueeze(0)) & (y.unsqueeze(1) >= 0)).float()
            same.fill_diagonal_(0.0)
            if bool((same.sum(dim=-1) > 0.5).any().item()):
                valid_q = self.equiv_q_lbl >= 0
                if bool(valid_q.any().item()):
                    all_proj = torch.cat([z_ep_c, self.equiv_q_proj[valid_q].float().detach()], dim=0)
                    all_raw = torch.cat([z_ep, self.equiv_q_raw[valid_q].float().detach()], dim=0)
                    all_lbl = torch.cat([y, self.equiv_q_lbl[valid_q]], dim=0)
                else:
                    all_proj, all_raw, all_lbl = z_ep_c, z_ep, y
                same_full = ((y.unsqueeze(1) == all_lbl.unsqueeze(0)) & (y.unsqueeze(1) >= 0)).float()
                eye_mask = torch.zeros_like(same_full, dtype=torch.bool)
                eye_mask[:, :B_e].fill_diagonal_(True)
                same_full = same_full.masked_fill(eye_mask, 0.0)
                pos_c_full = same_full.sum(dim=-1)
                has_pos_f = (pos_c_full > 0.5) & (y >= 0)
                if bool(has_pos_f.any().item()):
                    logits_eq = ((z_ep_c @ all_proj.t()) / 0.20).masked_fill(eye_mask, -1e4)
                    log_prob = logits_eq - torch.logsumexp(logits_eq, dim=-1, keepdim=True)
                    nce_proj = (-(same_full * log_prob).sum(dim=-1) / pos_c_full.clamp_min(1.0))[has_pos_f].mean()

                    logits_lin = ((z_ep @ all_raw.t()) / 0.28).masked_fill(eye_mask, -1e4)
                    log_prob_l = logits_lin - torch.logsumexp(logits_lin, dim=-1, keepdim=True)
                    nce_lin = (-(same_full * log_prob_l).sum(dim=-1) / pos_c_full.clamp_min(1.0))[has_pos_f].mean()
                else:
                    nce_proj = zero
                    nce_lin = zero
                pos_ctx_l = ((1.0 - (z_cp @ z_cp.t())) * same).sum() / same.sum().clamp_min(1.0)
                v_pm = y >= 0
                l_proto = zero
                if bool(v_pm.any().item()):
                    y_mod = (y[v_pm] % self.equiv_proto_bank.num_embeddings).long()
                    p_all = F.normalize(self.equiv_proto_bank.weight.float(), dim=-1)
                    l_proto = F.cross_entropy((z_ep[v_pm] @ p_all.t()) * 10.0, y_mod)
                l_equiv = 0.70 * nce_proj + 0.45 * nce_lin + 0.85 * l_proto + 0.30 * pos_ctx_l

                if self.training and update_queue and bool(v_pm.any().item()):
                    with torch.no_grad():
                        q_sz = self.equiv_q_lbl.size(0)
                        ptr = self._equiv_q_ptr_i % q_sz
                        src_r = z_ep[v_pm].detach().to(self.equiv_q_raw.dtype)[:q_sz]
                        src_p = z_ep_c[v_pm].detach().to(self.equiv_q_proj.dtype)[:q_sz]
                        src_l = y[v_pm].detach()[:q_sz]
                        k_add = src_l.size(0)
                        if ptr + k_add <= q_sz:
                            self.equiv_q_raw[ptr : ptr + k_add] = src_r
                            self.equiv_q_proj[ptr : ptr + k_add] = src_p
                            self.equiv_q_lbl[ptr : ptr + k_add] = src_l
                        else:
                            f_sp = q_sz - ptr
                            self.equiv_q_raw[ptr:] = src_r[:f_sp]
                            self.equiv_q_proj[ptr:] = src_p[:f_sp]
                            self.equiv_q_lbl[ptr:] = src_l[:f_sp]
                            rem = k_add - f_sp
                            self.equiv_q_raw[:rem] = src_r[f_sp:]
                            self.equiv_q_proj[:rem] = src_p[f_sp:]
                            self.equiv_q_lbl[:rem] = src_l[f_sp:]
                        self._equiv_q_ptr_i = (ptr + k_add) % q_sz
                        self.equiv_q_ptr.fill_(self._equiv_q_ptr_i)

        if hard_neg_pair_idx is not None:
            ha, hb = hard_neg_pair_idx[0].to(dev), hard_neg_pair_idx[1].to(dev)
            if ha.numel() > 0:
                hn_cos_p = (z_ep_c[ha] * z_ep_c[hb]).sum(dim=-1)
                l_equiv = l_equiv + 0.85 * F.relu(hn_cos_p + 0.15).pow(2).mean()

        l_contra = zero
        if contra_pair_idx is not None:
            idx_a, idx_b = contra_pair_idx[0].to(dev), contra_pair_idx[1].to(dev)
            if idx_a.numel() > 0:
                cos_ab = (z_cp[idx_a] * z_cp[idx_b]).sum(dim=-1)
                cos_ab_tail = (z_c_tail[idx_a] * z_c_tail[idx_b]).sum(dim=-1)
                l_contra = 0.45 * (cos_ab + 0.35).clamp_min(0.0).pow(2).mean() + 0.65 * (
                    cos_ab_tail + 0.40
                ).clamp_min(0.0).pow(2).mean()

        sem_tot = sigreg_weight * l_sigreg + equiv_weight * l_equiv + contra_weight * l_contra
        return sem_tot, l_sigreg, l_equiv, l_contra

    def compute_cognitive_loss(
        self,
        out: Dict[str, Any],
        *,
        x: Optional[torch.Tensor] = None,
        equiv_labels: Optional[torch.Tensor] = None,
        contra_pair_idx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hard_neg_pair_idx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        nuance_targets: Optional[torch.Tensor] = None,
        struct_targets: Optional[torch.Tensor] = None,
        wm_weight: float = 1.0,
        fsp_weight: float = 0.35,
        sigreg_weight: float = 0.25,
        equiv_weight: float = 1.0,
        contra_weight: float = 1.0,
        think_weight: float = 0.35,
        erc_weight: float = 0.20,
        balance_weight: float = 0.5,
        struct_weight: float = 0.20,
        router_q_perturbed: Optional[torch.Tensor] = None,
        router_q_pert_idx: Optional[torch.Tensor] = None,
        cris_weight: float = 0.35,
        exp_orth_weight: float = 0.15,
    ) -> Dict[str, torch.Tensor]:
        """Pure Non-Token Cognitive Training Objective (0.0 next-byte CE gradient into core)."""
        ze_fin = out["z_expr"]
        zc_fin = out["z_ctx"]
        z_ep = out["z_expr_pool"]
        c_mask = out.get("chunk_content_mask")
        dev = z_ep.device
        T = zc_fin.size(1)
        zero = z_ep.sum() * 0.0

        wm_loss = out["world_model_loss"]["loss"] if out.get("world_model_loss") is not None else zero

        # Content-masked Future Summary Prediction (FSP) + bag-of-ngram concept preservation
        l_fsp = zero
        if x is not None and out.get("fsp_logits") is not None and T >= 2:
            fsp_lg = out["fsp_logits"][:, :-1, :].float()
            p_fut = self._compute_future_bigram_summary(
                x, chunk_size=self.chunk_size, num_bins=fsp_lg.size(-1)
            )[:, : fsp_lg.size(1), :]
            ce_fsp = -(p_fut * F.log_softmax(fsp_lg, dim=-1)).sum(dim=-1)
            if c_mask is not None and c_mask.size(1) > fsp_lg.size(1):
                vm_f = c_mask[:, 1 : 1 + fsp_lg.size(1)].float()
                l_fsp = ((ce_fsp * vm_f).sum() / vm_f.sum().clamp_min(1.0)).to(ze_fin.dtype)
            else:
                l_fsp = ce_fsp.mean().to(ze_fin.dtype)
            nbins = self.fsp_head.out_features
            ids_c = x[:, : T * self.chunk_size].view(x.size(0), T, self.chunk_size).long()
            vm_b = (ids_c != BYTE_PAD_ID).float()
            h_doc = torch.full((x.size(0), nbins), 1e-4, device=dev, dtype=torch.float32)
            h_doc.scatter_add_(-1, (ids_c % nbins).view(x.size(0), -1), vm_b.view(x.size(0), -1))
            if self.chunk_size >= 2:
                bg_c = (ids_c[..., :-1] * 31 + ids_c[..., 1:]) % nbins
                vm_bg = ((ids_c[..., :-1] != BYTE_PAD_ID) & (ids_c[..., 1:] != BYTE_PAD_ID)).float()
                h_doc.scatter_add_(-1, bg_c.view(x.size(0), -1), vm_bg.view(x.size(0), -1))
            p_doc = h_doc / h_doc.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            ze_lg = F.log_softmax(self.fsp_head(out["z_expr_raw_pool"]).float(), dim=-1)
            l_fsp = l_fsp + 0.55 * (-(p_doc * ze_lg).sum(dim=-1).mean()).to(ze_fin.dtype)

        sem_tot, l_sigreg, l_equiv, l_contra_vec = self._semantic_state_loss(
            ze_fin,
            zc_fin,
            equiv_labels=equiv_labels,
            contra_pair_idx=contra_pair_idx,
            hard_neg_pair_idx=hard_neg_pair_idx,
            content_mask=c_mask,
            sigreg_weight=sigreg_weight,
            equiv_weight=equiv_weight,
            contra_weight=contra_weight,
            ze_pool_override=out.get("z_expr_raw_pool"),
        )

        # Selective d_ctx margin + need calibration
        l_contra_extra = zero
        if contra_pair_idx is not None and "d_ctx" in out:
            idx_a, idx_b = contra_pair_idx[0].to(dev), contra_pair_idx[1].to(dev)
            if idx_a.numel() > 0:
                d_ctx_f = out["d_ctx"].float()
                tail_s = max(0, T // 2)
                d_tail_slice = d_ctx_f[:, tail_s:]
                if c_mask is not None:
                    tm_d = c_mask[:, tail_s:]
                    w_d = tm_d.float()
                    d_mean_t = (d_tail_slice * w_d).sum(dim=-1) / w_d.sum(dim=-1).clamp_min(1.0)
                    d_max_t = torch.where(tm_d, d_tail_slice, torch.zeros_like(d_tail_slice)).max(dim=-1).values
                    d_tail = 0.5 * d_max_t + 0.5 * d_mean_t
                else:
                    d_tail = 0.5 * d_tail_slice.max(dim=-1).values + 0.5 * d_tail_slice.mean(dim=-1)
                if equiv_labels is not None:
                    non_c = torch.ones(d_tail.size(0), dtype=torch.bool, device=dev)
                    non_c[idx_b] = False
                    if bool(non_c.any().item()):
                        l_contra_extra = l_contra_extra + 0.35 * F.relu(d_tail[non_c] - 0.25).pow(2).mean()
                if nuance_targets is not None:
                    nt = nuance_targets.to(dev).float().reshape(-1)
                    if nt.numel() == idx_b.numel():
                        is_flip = nt > 0.7
                        is_clean = nt < 0.3
                        if bool(is_flip.any().item()):
                            l_contra_extra = l_contra_extra + 0.90 * F.relu(0.75 - d_tail[idx_b[is_flip]]).pow(2).mean()
                            l_contra_extra = l_contra_extra + 0.45 * F.relu(d_tail[idx_a[is_flip]] - 0.25).pow(2).mean()
                        if bool(is_clean.any().item()):
                            l_contra_extra = l_contra_extra + 0.45 * F.relu(d_tail[idx_b[is_clean]] - 0.25).pow(2).mean()
                    need_seq = out["need"].float().mean(dim=-1)
                    need_pred = need_seq[idx_b] if (nt.numel() == idx_b.numel() and nt.numel() != need_seq.numel()) else need_seq[: nt.numel()]
                    red_tgt = nt
                    if out.get("pre_pc_error") is not None and nt.numel() == idx_b.numel():
                        pc_seq = out["pre_pc_error"].detach().float().mean(dim=-1)
                        red_tgt = (0.75 * nt + 0.25 * pc_seq[idx_b]).clamp(0.05, 0.95)
                    with torch.amp.autocast(device_type=dev.type, enabled=False):
                        l_contra_extra = l_contra_extra + 0.25 * F.binary_cross_entropy(
                            need_pred.float().clamp(1e-4, 1.0 - 1e-4),
                            red_tgt.float(),
                        )

        # Thinker stepwise latent progress & energy/abstain calibration
        l_think = zero
        z_trace = out.get("thinker_z_trace")
        if z_trace is not None and len(z_trace) >= 2:
            z0 = z_trace[0].detach()
            with torch.no_grad():
                base_sem, _, _, _ = self._semantic_state_loss(
                    self.expr_head(z0),
                    self.ctx_head(z0),
                    equiv_labels=equiv_labels,
                    contra_pair_idx=contra_pair_idx,
                    hard_neg_pair_idx=hard_neg_pair_idx,
                    content_mask=c_mask,
                    sigreg_weight=0.0,
                    equiv_weight=equiv_weight,
                    contra_weight=contra_weight,
                    update_queue=False,
                )
            step_sems = []
            for zk in z_trace[1:-1]:
                s_k, _, _, _ = self._semantic_state_loss(
                    self.expr_head(zk),
                    self.ctx_head(zk),
                    equiv_labels=equiv_labels,
                    contra_pair_idx=contra_pair_idx,
                    hard_neg_pair_idx=hard_neg_pair_idx,
                    content_mask=c_mask,
                    sigreg_weight=0.0,
                    equiv_weight=equiv_weight,
                    contra_weight=contra_weight,
                    update_queue=False,
                )
                step_sems.append(s_k)
            fin_sem = equiv_weight * l_equiv + contra_weight * l_contra_vec
            l_think = F.relu(fin_sem - base_sem.detach() + 0.03) + (
                0.25 * torch.stack(step_sems).mean() if step_sems else zero
            )
        think_out = out.get("thinking_out")
        if think_out is not None and think_out.get("step_energies"):
            e0 = think_out["step_energies"][0].float()
            ef = think_out["step_energies"][-1].float()
            tgt_e = out["pre_pc_error"].detach().float().clamp(0.0, 1.0)[:, : e0.size(1)]
            l_think = l_think + 0.05 * (F.mse_loss(e0, tgt_e) + F.mse_loss(ef, tgt_e * 0.7))
        ab_lg = out["abstain_logits"].float().squeeze(-1)
        ab_tgt = (out["d_ctx"].detach().float() > 0.6).float()
        l_think = l_think + 0.05 * F.binary_cross_entropy_with_logits(ab_lg, ab_tgt)

        # Structural polarity classification
        l_struct = zero
        s_logits = out.get("i7_struct_logits")
        if s_logits is not None and struct_weight > 0.0:
            if struct_targets is not None:
                l_struct = F.cross_entropy(s_logits.float(), struct_targets.to(dev))
            elif equiv_labels is not None:
                st_t = torch.zeros(s_logits.size(0), dtype=torch.long, device=dev)
                same_m = ((equiv_labels.unsqueeze(1) == equiv_labels.unsqueeze(0)) & (equiv_labels.unsqueeze(1) >= 0)).float()
                same_m.fill_diagonal_(0.0)
                st_t = torch.where(same_m.sum(dim=-1) > 0.5, torch.ones_like(st_t), st_t)
                if contra_pair_idx is not None and contra_pair_idx[1].numel() > 0:
                    ib_s = contra_pair_idx[1].to(dev)
                    if nuance_targets is not None and nuance_targets.numel() == ib_s.numel():
                        st_t[ib_s[nuance_targets.to(dev).float() > 0.7]] = 2
                    else:
                        st_t[ib_s] = 2
                l_struct = F.cross_entropy(s_logits.float(), st_t)

        # Prefix-to-Continuation Concept Retrieval InfoNCE
        l_wm_ret = zero
        if x is not None and x.size(1) >= 16 and out.get("world_model_imagine") is not None and z_ep.size(0) >= 2:
            wm_imag = out["world_model_imagine"]["imagined_latents"]
            half_b = x.size(1) // 2
            t_cut = max(1, (half_b // self.chunk_size) - 1)
            h_k = min(3, wm_imag.size(2))
            if t_cut < wm_imag.size(1):
                fut_slice = x[:, half_b:]
                ret_idx = (
                    ((fut_slice != BYTE_PAD_ID).float().mean(dim=-1) >= 0.75)
                    & ((fut_slice != 32).float().mean(dim=-1) >= 0.35)
                ).nonzero(as_tuple=False).squeeze(-1)
                if ret_idx.numel() >= 2 and out.get("pos_chunks") is not None:
                    z_fut_in_raw = wm_imag[ret_idx, t_cut, :h_k, :].float().mean(dim=1)
                    cm_pre = c_mask[:, : t_cut + 1] if c_mask is not None else None
                    pos_pre_all = self._pool_pos_chunks(out["pos_chunks"][:, : t_cut + 1, :], mask=cm_pre)
                    h_pre_seq = out["latent"][:, : t_cut + 1, :].float()
                    if cm_pre is not None:
                        cm_pf = cm_pre.unsqueeze(-1).float()
                        h_pre_mean = (h_pre_seq * cm_pf).sum(dim=1) / cm_pf.sum(dim=1).clamp_min(1.0)
                    else:
                        h_pre_mean = h_pre_seq.mean(dim=1)
                    z_fut_in = 0.20 * z_fut_in_raw + 0.80 * z_fut_in_raw.detach()
                    h_ret_in = (0.15 * h_pre_mean + 0.85 * h_pre_mean.detach())[ret_idx]
                    pos_ret = pos_pre_all[ret_idx]
                    zp_n = self.project_wm_retrieval(z_fut_in.to(dtype=ze_fin.dtype), h_ctx=h_ret_in, pos_pre=pos_ret).float()
                    has_neg = contra_pair_idx is not None and contra_pair_idx[1].numel() > 0
                    with torch.no_grad():
                        if has_neg:
                            neg_idx = contra_pair_idx[1].to(dev)
                            both_tgt = self.world_model.encode_target_summary(
                                torch.cat([x[ret_idx, half_b:], x[neg_idx, half_b:]], dim=0)
                            )
                            zt_n, zn_f = both_tgt[: ret_idx.numel()], both_tgt[ret_idx.numel() :]
                        else:
                            zt_n = self.world_model.encode_target_summary(x[ret_idx, half_b:])
                            zn_f = None
                    cands_t = zt_n
                    if self._wm_ret_q_full_i > 0:
                        cands_t = torch.cat([cands_t, self.wm_ret_q[: self._wm_ret_q_full_i].float().detach()], dim=0)
                    if zn_f is not None:
                        cands_t = torch.cat([cands_t, zn_f], dim=0)
                    lbl_f = torch.arange(ret_idx.numel(), device=dev)
                    sim_q = (zp_n @ cands_t.t()) / 0.13
                    sim_b = (zp_n @ zt_n.t()) / 0.13
                    sim_s = (F.normalize(pos_ret, dim=-1) @ cands_t.t()) / 0.14
                    l_wm_ret = 0.82 * (0.5 * F.cross_entropy(sim_q, lbl_f) + 0.5 * F.cross_entropy(sim_b.t(), lbl_f)) + 0.18 * F.cross_entropy(sim_s, lbl_f)
                    if self.training:
                        with torch.no_grad():
                            q_sz = self.wm_ret_q.size(0)
                            ptr = self._wm_ret_q_ptr_i % q_sz
                            src_q = zt_n[:q_sz].detach().to(dtype=self.wm_ret_q.dtype)
                            k_q = src_q.size(0)
                            if ptr + k_q <= q_sz:
                                self.wm_ret_q[ptr : ptr + k_q] = src_q
                            else:
                                sp = q_sz - ptr
                                self.wm_ret_q[ptr:] = src_q[:sp]
                                self.wm_ret_q[: k_q - sp] = src_q[sp:]
                            self._wm_ret_q_ptr_i = (ptr + k_q) % q_sz
                            self._wm_ret_q_full_i = min(q_sz, self._wm_ret_q_full_i + k_q)
                            self.wm_ret_q_ptr.fill_(self._wm_ret_q_ptr_i)
                            self.wm_ret_q_full.fill_(self._wm_ret_q_full_i)

        # Clean CRIS (Surface-Perturbation Invariance + Equivalence + Hard-Neg Repulsion + Expert Orthogonality)
        l_cris_route = zero
        l_exp_orth = zero
        if self.enable_cris_loss and (cris_weight > 0.0 or exp_orth_weight > 0.0):
            rq_tok = out["router_q"].float()
            if rq_tok.size(0) >= 2:
                w_rq = c_mask.unsqueeze(-1).float() if (c_mask is not None and c_mask.size(1) == rq_tok.size(1)) else torch.ones_like(rq_tok[..., :1])
                rq_pool = F.normalize((rq_tok * w_rq).sum(dim=1) / w_rq.sum(dim=1).clamp_min(1.0), dim=-1)

                # 1. Equivalence alignment & labeled distinct-intent separation (strictly y >= 0; zero noise on unlabeled -1!)
                if equiv_labels is not None:
                    y_eq = equiv_labels.to(dev)
                    cos_rq = rq_pool @ rq_pool.t()
                    same_eq = (y_eq.unsqueeze(1) == y_eq.unsqueeze(0)) & (y_eq.unsqueeze(1) >= 0)
                    same_eq.fill_diagonal_(False)
                    if bool(same_eq.any().item()):
                        l_cris_route = l_cris_route + ((1.0 - cos_rq) * same_eq.float()).sum() / same_eq.float().sum().clamp_min(1.0)
                    diff_labeled = (y_eq.unsqueeze(1) != y_eq.unsqueeze(0)) & (y_eq.unsqueeze(1) >= 0) & (y_eq.unsqueeze(0) >= 0)
                    diff_labeled.fill_diagonal_(False)
                    if self.use_infonce_cris and c_mask is not None and c_mask.size(0) == y_eq.size(0):
                        # Length-matched distinct-intent repulsion prevents length-shortcut routing!
                        c_lens = c_mask.float().sum(dim=-1)
                        same_len_band = (c_lens.unsqueeze(1) - c_lens.unsqueeze(0)).abs() <= 2.5
                        diff_labeled = diff_labeled & same_len_band
                    if bool(diff_labeled.any().item()):
                        l_cris_route = l_cris_route + 0.30 * (F.relu(cos_rq - 0.30).pow(2) * diff_labeled.float()).sum() / diff_labeled.float().sum().clamp_min(1.0)

                if hard_neg_pair_idx is not None:
                    ha_r, hb_r = hard_neg_pair_idx[0].to(dev), hard_neg_pair_idx[1].to(dev)
                    if ha_r.numel() > 0:
                        l_cris_route = l_cris_route + 0.45 * F.relu((rq_pool[ha_r] * rq_pool[hb_r]).sum(dim=-1) - 0.15).pow(2).mean()

                # 2. Surface formatting perturbation invariance (SimCLR InfoNCE + SIGReg when use_infonce_cris=True)
                if router_q_perturbed is not None:
                    rq_pert = F.normalize(router_q_perturbed.to(dev).float(), dim=-1)
                    rq_ref = rq_pool[router_q_pert_idx.to(dev)] if router_q_pert_idx is not None else rq_pool[: rq_pert.size(0)]
                    if rq_ref.size(0) == rq_pert.size(0) and rq_pert.size(0) > 0:
                        if self.use_infonce_cris and rq_pert.size(0) >= 4:
                            # Symmetric Perturbation InfoNCE + strong positive alignment + SIGReg dispersion
                            sim_cp = (rq_ref @ rq_pert.t()) / 0.25
                            lbl_p = torch.arange(rq_pert.size(0), device=dev)
                            l_nce_p = 0.5 * (F.cross_entropy(sim_cp, lbl_p) + F.cross_entropy(sim_cp.t(), lbl_p))
                            l_pos_p = (1.0 - (rq_ref * rq_pert).sum(dim=-1)).mean()
                            l_sig_r = sigreg_loss(F.layer_norm(rq_pool, (rq_pool.size(-1),))) if rq_pool.size(0) >= 4 else zero
                            l_cris_route = l_cris_route + 0.50 * l_nce_p + 2.50 * l_pos_p + 0.20 * l_sig_r
                        else:
                            l_cris_route = l_cris_route + 2.5 * (1.0 - (rq_ref * rq_pert).sum(dim=-1)).mean()
                        if self.context_router.num_experts > 1:
                            E_cur = self.context_router.num_experts
                            k_e = F.normalize(self.context_router.expert_keys[:E_cur].float(), dim=-1)
                            s_clean = torch.einsum("bd,emd->bem", rq_ref, k_e).max(dim=-1).values * self.context_router.scale
                            s_pert = torch.einsum("bd,emd->bem", rq_pert, k_e).max(dim=-1).values * self.context_router.scale
                            p_c = F.softmax(s_clean.detach(), dim=-1)
                            p_p = F.softmax(s_pert.detach(), dim=-1)
                            sym_kl = 0.5 * (
                                (p_c * (p_c.clamp_min(1e-6).log() - F.log_softmax(s_pert, dim=-1))).sum(dim=-1)
                                + (p_p * (p_p.clamp_min(1e-6).log() - F.log_softmax(s_clean, dim=-1))).sum(dim=-1)
                            ).mean()
                            l_cris_route = l_cris_route + 1.40 * sym_kl

                if self.router_anchor_weight > 0.0:
                    l_cris_route = l_cris_route + self.router_anchor_weight * self.context_router.router_anchor_loss()

            # 3. Permutation-invariant expert output subspace orthogonality (every 4th step to save wall-clock)
            if exp_orth_weight > 0.0 and (self._step_ctr % 4 == 0):
                orth_terms: List[torch.Tensor] = []
                for layer in self.layers[self.router_layer :]:
                    exps = layer.moe.experts
                    n_e = len(exps)
                    if n_e >= 2:
                        grams = []
                        trainable = []
                        for exp in exps:
                            w2 = exp.linear2.weight.float()
                            g_e = w2 @ w2.t()
                            grams.append(g_e / g_e.norm().clamp_min(1e-6))
                            trainable.append(bool(exp.linear2.weight.requires_grad))
                        for i in range(n_e):
                            for j in range(i + 1, n_e):
                                if trainable[i] or trainable[j]:
                                    ov = (grams[i] * grams[j]).sum()
                                    orth_terms.append(ov * ov)
                if orth_terms:
                    l_exp_orth = torch.stack(orth_terms).mean()

        l_bal = out["router_balance_loss"]
        l_erc = out["router_erc_loss"]
        total = (
            wm_weight * (0.40 * wm_loss + 3.20 * l_wm_ret)
            + fsp_weight * l_fsp
            + sem_tot
            + contra_weight * l_contra_extra
            + think_weight * l_think
            + struct_weight * l_struct
            + erc_weight * l_erc
            + balance_weight * l_bal
            + cris_weight * l_cris_route
            + exp_orth_weight * l_exp_orth
        )
        return {
            "loss": total,
            "wm_loss": wm_loss.detach(),
            "wm_ret_loss": l_wm_ret.detach(),
            "fsp_loss": l_fsp.detach(),
            "sigreg_loss": l_sigreg.detach(),
            "equiv_loss": l_equiv.detach(),
            "contra_loss": (l_contra_vec + l_contra_extra).detach(),
            "think_loss": l_think.detach(),
            "struct_loss": l_struct.detach(),
            "erc_loss": l_erc.detach(),
            "balance_loss": l_bal.detach(),
            "cris_route_loss": l_cris_route.detach(),
            "exp_orth_loss": l_exp_orth.detach(),
        }

    # Alias for compatibility with evaluation harnesses
    compute_v7_non_token_loss = compute_cognitive_loss
    compute_v8_non_token_loss = compute_cognitive_loss

    def architecture_summary(self) -> Dict[str, Any]:
        """Return exact parameter counts broken down by subsystem."""
        buckets = {
            "sensory_chunker": 0,
            "dual_memory_and_ridge": 0,
            "shared_trunk_layers": 0,
            "router_and_lifecycle": 0,
            "specialist_moe_and_adapters": 0,
            "world_model_and_retrieval": 0,
            "thinker_and_abstain": 0,
            "meaning_and_fsp_heads": 0,
            "detached_decoder": 0,
        }
        for name, p in self.named_parameters():
            n = p.numel()
            if name.startswith("ms_chunker."):
                buckets["sensory_chunker"] += n
            elif name.startswith("memory."):
                buckets["dual_memory_and_ridge"] += n
            elif name.startswith("decoder."):
                buckets["detached_decoder"] += n
            elif name.startswith("context_router."):
                buckets["router_and_lifecycle"] += n
            elif name.startswith("world_model.") or name.startswith("wm_"):
                buckets["world_model_and_retrieval"] += n
            elif name.startswith("thinker.") or name.startswith("abstain_head."):
                buckets["thinker_and_abstain"] += n
            elif name.startswith(("expr_", "ctx_head.", "equiv_proto_bank.", "fsp_head.", "i7_struct_head.")):
                buckets["meaning_and_fsp_heads"] += n
            elif ".moe.experts." in name or "_adapters." in name:
                buckets["specialist_moe_and_adapters"] += n
            else:
                buckets["shared_trunk_layers"] += n
        total = sum(buckets.values())
        core_total = total - buckets["detached_decoder"]
        n_exp = len(self.layers[self.router_layer].moe.experts)
        spec_per_exp = buckets["specialist_moe_and_adapters"] // max(1, n_exp)
        active_k1 = core_total - buckets["specialist_moe_and_adapters"] + spec_per_exp
        return {
            "total_params": total,
            "cognitive_core_params": core_total,
            "active_core_params_k1": active_k1,
            "num_experts": n_exp,
            "subsystems": buckets,
        }


class CognitiveTrainer:
    """
    Unified Always-Plastic Continual Training Orchestrator for `CognitiveDMoE`:
      - `MuonBONGOptimizer` (Newton-Schulz5 + optional Grokfast-EMA in `acquire`,
        vectorized Diagonal BONG in `consolidate`)
      - `NullSpaceProjector` (GPM/AlphaEdit with readout-head exemption)
      - `LatentQuarantineGate` (`<= 16 KB` SimHash multi-source burst immunity)
      - Separate detached AdamW optimizer for `model.decoder` (zero core gradient)
      - Automatic state-preserving optimizer rebuild on autonomous expert spawns
    """

    def __init__(
        self,
        model: CognitiveDMoE,
        *,
        seed: int = 0,
        lr_muon: float = 8e-3,
        lr_adam: float = 3e-3,
        lr_consolidate: float = 2e-3,
        wd_trunk: float = 0.02,
        wd_fast: float = 0.01,
        trunk_prec_scale: float = 0.85,
        grokfast_alpha: float = 0.0,
        grokfast_lamb: float = 0.0,
        ortho_grad: bool = False,
        bong_prior_pull: float = 0.02,
        bong_max_prec: float = 100.0,
        grad_clip: float = 2.5,
        enable_gpm: bool = True,
        exempt_readout_heads: bool = True,
        decoder_lr: float = 3e-3,
        extra_param_groups: Optional[List[Dict[str, Any]]] = None,
    ):
        self.model = model
        self.lr_muon = float(lr_muon)
        self.lr_adam = float(lr_adam)
        self.lr_consolidate = float(lr_consolidate)
        self.wd_trunk = float(wd_trunk)
        self.wd_fast = float(wd_fast)
        self.trunk_prec_scale = float(trunk_prec_scale)
        self.grad_clip = float(grad_clip)
        self.shared_mode: str = "acquire"
        self.extra_param_groups: List[Dict[str, Any]] = list(extra_param_groups or [])
        self.total_spawns: int = 0

        self.gpm: Optional[NullSpaceProjector] = (
            NullSpaceProjector(
                model,
                energy_threshold=0.97,
                max_rank_ratio=0.85,
                max_step_rank_ratio=0.24,
                step_rank_decay=0.84,
                defer_cov_commit=True,
                track_embeddings=True,
                freeze_shared_norms_on_update=True,
                freeze_unprojected_shared=True,
                exempt_readout_heads=exempt_readout_heads,
            )
            if enable_gpm
            else None
        )
        if self.gpm is not None:
            model.attach_null_space_projector(self.gpm)

        groups = self._build_groups()
        self.opt = MuonBONGOptimizer(
            groups,
            mode="acquire",
            grokfast_alpha=grokfast_alpha,
            grokfast_lamb=grokfast_lamb,
            ortho_grad=ortho_grad,
            bong_prior_pull=bong_prior_pull,
            bong_max_prec=bong_max_prec,
            grad_clip=max(4.5, grad_clip),
        )
        self.gate = LatentQuarantineGate(
            latent_dim=model.d_model,
            seed=seed,
            max_bytes=16384,
            store_samples=False,
        )
        self.decoder_opt = torch.optim.AdamW(model.decoder.parameters(), lr=decoder_lr, weight_decay=1e-2)

    def _build_groups(self) -> List[Dict[str, Any]]:
        groups = classify_v7_spgc_param_groups(
            self.model,
            shared_mode=self.shared_mode,
            specialist_mode="acquire",
            lr_muon=self.lr_muon,
            lr_adam=self.lr_adam,
            lr_consolidate=self.lr_consolidate,
            wd_trunk=self.wd_trunk,
            wd_fast=self.wd_fast,
            trunk_prec_scale=self.trunk_prec_scale,
        )
        for eg in self.extra_param_groups:
            groups.append(dict(eg))
        return groups

    def maybe_rebuild_on_spawn(self, lifecycle_events: Sequence[str]) -> bool:
        if any(ev.startswith("spawned_") for ev in lifecycle_events):
            self.total_spawns += 1
            self.opt.rebuild_preserving_state(self._build_groups())
            self.decoder_opt = torch.optim.AdamW(
                [p for p in self.model.decoder.parameters() if p.requires_grad],
                lr=self.decoder_opt.param_groups[0]["lr"],
                weight_decay=1e-2,
            )
            return True
        return False

    def consolidate_domain(self, domain_id: int) -> None:
        """Commit GPM null-space basis for completed domain, snapshot router anchor, and shift trunk to BONG."""
        if self.gpm is not None:
            self.gpm.update_basis_for_domain(domain_id)
        if domain_id == 0 and self.model.router_anchor_weight > 0.0:
            self.model.context_router.snapshot_router_anchor()
        self.shared_mode = "consolidate"
        self.opt.set_group_mode("trunk", "consolidate")

    def step_loss(
        self,
        loss: torch.Tensor,
        *,
        lifecycle_events: Sequence[str] = (),
        saved_weights: Optional[Dict[str, torch.Tensor]] = None,
        fisher_scale: float = 1.0,
        extra_clip_params: Optional[Sequence[nn.Parameter]] = None,
    ) -> None:
        """Execute GPM-projected backward + MuonBONG step + automatic spawn rebuild."""
        self.maybe_rebuild_on_spawn(lifecycle_events)
        self.opt.zero_grad()
        loss.backward()
        if self.gpm is not None and self.gpm.p_null:
            self.gpm.project_gradients()
        # Clip context_router gradients independently from trunk/specialist gradients so
        # CRIS/InfoNCE routing gradients NEVER attenuate cognitive trunk updates via global norm clipping!
        router_params: List[nn.Parameter] = []
        trunk_params: List[nn.Parameter] = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("context_router."):
                router_params.append(p)
            else:
                trunk_params.append(p)
        if extra_clip_params:
            trunk_params.extend(list(extra_clip_params))
        if trunk_params:
            nn.utils.clip_grad_norm_(trunk_params, self.grad_clip)
        if router_params:
            nn.utils.clip_grad_norm_(router_params, self.grad_clip)
        self.opt.step(fisher_scale=fisher_scale)
        if saved_weights is not None and self.gpm is not None:
            self.gpm.project_parameter_deltas(saved_weights)
        if self.gpm is not None and self.gpm.defer_cov_commit:
            self.gpm.commit_pending()

    def step_decoder(self, out: Dict[str, Any], x: torch.Tensor) -> float:
        """Train ONLY the detached byte renderer on next-chunk byte prediction (0.0 core grad)."""
        dec_loss = self.model.compute_detached_decoder_loss(out, x)
        self.decoder_opt.zero_grad()
        dec_loss.backward()
        nn.utils.clip_grad_norm_(self.model.decoder.parameters(), 2.0)
        self.decoder_opt.step()
        return float(dec_loss.detach().item())
