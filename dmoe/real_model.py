"""
Real-Valued Dynamic Morphic MoE (FP32 Core).
Replaces complex-valued arithmetic with standard real-valued FP32 linear and attention layers:
- Uses native real matrix multiplications (2x to 4x faster).
- Fully parameter-matched.
- Compatible with Evidence-Accumulating Upkeeper and modular per-domain output heads.
"""

import functools
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.adapters import RoutedAdapterBank
from dmoe.belief_training import SparseConceptQuantizer, sigreg_loss
from dmoe.chunker import MultiScaleByteChunker
from dmoe.hierarchical_attention import ThreeLevelAttention
from dmoe.lifecycle import AutonomousExpertLifecycleManager, STATE_DORMANT
from dmoe.memory import DualTimescaleMemory, GatedDeltaWorkingMemory
from dmoe.novelty import NuanceClusterBuffer, UnifiedNoveltyDetector
from dmoe.rope import RotaryEmbedding
from dmoe.routing import (
    CausalContextTokenRouter,
    CausalPrototypeRouter,
    SequenceContentRouter,
    build_token_dispatch_plan,
    compute_chunk_surface_features,
    sparse_weighted_expert_dispatch,
)
from dmoe.thinking import SurpriseDrivenLatentThinker
from dmoe.upkeeper import EvidenceAccumulatingUpkeeper
from dmoe.world_model import LatentWorldModel


@functools.lru_cache(maxsize=64)
def _get_layer_causal_masks(
    seq_len: int, device_str: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = torch.device(device_str)
    ones = torch.ones(seq_len, seq_len, device=dev, dtype=torch.bool)
    causal = torch.tril(ones).view(1, 1, seq_len, seq_len)
    past_only = torch.tril(ones, diagonal=-1).view(1, 1, seq_len, seq_len)
    pair_past = torch.tril(ones, diagonal=-2).view(1, 1, seq_len, seq_len)
    return causal, past_only, pair_past


class RealExpert(nn.Module):
    """Real-valued feed-forward expert with rolling sequence exemplar buffer."""

    def __init__(self, d_model: int, d_ff: int, max_exemplars: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_exemplars = max_exemplars
        self.linear1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.register_buffer("exemplars", torch.zeros(0, d_model, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))

    @torch.no_grad()
    def update_exemplars(self, latents: torch.Tensor) -> None:
        flat = latents.detach().reshape(-1, self.d_model).float()
        if self.exemplars.numel() == 0:
            self.exemplars = flat[: self.max_exemplars].clone()
        else:
            combined = torch.cat([self.exemplars, flat], dim=0)
            if combined.size(0) > self.max_exemplars:
                combined = combined[-self.max_exemplars :]
            self.exemplars = combined.clone()

    def clone_with_mutation(self, mutation_std: float = 0.015) -> "RealExpert":
        new_exp = RealExpert(self.d_model, self.d_ff, max_exemplars=self.max_exemplars).to(
            self.linear1.weight.device
        )
        with torch.no_grad():
            new_exp.linear1.weight.copy_(
                self.linear1.weight + torch.randn_like(self.linear1.weight) * mutation_std
            )
            new_exp.linear1.bias.copy_(
                self.linear1.bias + torch.randn_like(self.linear1.bias) * mutation_std
            )
            new_exp.linear2.weight.copy_(
                self.linear2.weight + torch.randn_like(self.linear2.weight) * mutation_std
            )
            new_exp.linear2.bias.copy_(
                self.linear2.bias + torch.randn_like(self.linear2.bias) * mutation_std
            )
        return new_exp


class RealDynamicMorphicMoE(nn.Module):
    """Dynamic Morphic MoE layer with real-valued feed-forward experts and optional v7 shared core."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        initial_experts: int = 1,
        use_shared_expert: bool = False,
        d_ff_shared: Optional[int] = None,
        v7_trunk_only: bool = False,
        init_spec_small: bool = False,
        specialist_init_scale: float = 0.05,
        shared_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.v7_trunk_only = bool(v7_trunk_only)
        self.init_spec_small = bool(init_spec_small)
        self.specialist_init_scale = float(specialist_init_scale)
        self.shared_dropout_prob = float(shared_dropout_prob)
        self.use_shared_expert = bool(use_shared_expert or v7_trunk_only)
        sh_dff = int(d_ff_shared) if d_ff_shared is not None else d_ff
        self.shared_expert: Optional[RealExpert] = (
            RealExpert(d_model, sh_dff) if self.use_shared_expert else None
        )

        if self.v7_trunk_only:
            self.experts = nn.ModuleList([])
            self.register_buffer("expert_usage", torch.zeros(0))
            self.register_buffer("expert_age", torch.zeros(0))
            self.register_buffer("expert_frozen", torch.zeros(0, dtype=torch.bool))
            self._expert_frozen_flags: List[bool] = []
        else:
            experts = [RealExpert(d_model, d_ff) for _ in range(initial_experts)]
            if self.init_spec_small:
                with torch.no_grad():
                    for exp in experts:
                        exp.linear2.weight.mul_(self.specialist_init_scale)
                        exp.linear2.bias.zero_()
            self.experts = nn.ModuleList(experts)
            self.register_buffer("expert_usage", torch.ones(initial_experts) / max(1, initial_experts))
            self.register_buffer("expert_age", torch.zeros(initial_experts))
            self.register_buffer("expert_frozen", torch.zeros(initial_experts, dtype=torch.bool))
            self._expert_frozen_flags: List[bool] = [False] * initial_experts

        self.assigned_experts: Optional[torch.Tensor] = None
        self.force_routed_expert: Optional[int] = None
        self.skip_uniform_check: bool = False
        self.register_buffer("_zero_entropy", torch.tensor(0.0), persistent=False)

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
        self._expert_frozen_flags = [bool(f.item()) for f in self.expert_frozen]

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def spawn_expert(self, source_idx: Optional[int] = None, fresh_init: bool = False) -> int:
        if self.v7_trunk_only:
            return 0
        device = self.expert_frozen.device
        if source_idx is None:
            source_idx = int(torch.argmax(self.expert_usage).item()) if len(self.experts) > 0 else 0

        if fresh_init or len(self.experts) == 0:
            new_expert = RealExpert(self.d_model, self.d_ff).to(device)
            if self.init_spec_small:
                with torch.no_grad():
                    new_expert.linear2.weight.mul_(self.specialist_init_scale)
                    new_expert.linear2.bias.zero_()
        else:
            new_expert = self.experts[source_idx].clone_with_mutation(mutation_std=0.015).to(device)
        new_expert.exemplars = torch.zeros(0, self.d_model, dtype=torch.float32, device=device)
        self.experts.append(new_expert)
        new_idx = len(self.experts) - 1

        self.expert_usage = torch.cat([self.expert_usage, torch.tensor([0.1], device=device)])
        self.expert_age = torch.cat([self.expert_age, torch.zeros(1, device=device)])
        self.expert_frozen = torch.cat([self.expert_frozen, torch.zeros(1, dtype=torch.bool, device=device)])
        self._expert_frozen_flags.append(False)
        return new_idx

    def prune_expert(self, expert_idx: int) -> bool:
        """Cleanly remove an expert and its usage/age/frozen entries (M08-safe)."""
        if self.v7_trunk_only or len(self.experts) <= 1 or not (0 <= expert_idx < len(self.experts)):
            return False
        del self.experts[expert_idx]
        keep = [i for i in range(self.expert_usage.numel()) if i != expert_idx]
        idx_t = torch.tensor(keep, dtype=torch.long, device=self.expert_usage.device)
        self.expert_usage = self.expert_usage[idx_t]
        self.expert_age = self.expert_age[idx_t]
        self.expert_frozen = self.expert_frozen[idx_t]
        self._expert_frozen_flags = [self._expert_frozen_flags[i] for i in keep]
        return True

    def freeze_expert(self, expert_idx: int) -> None:
        if self.v7_trunk_only:
            return
        if 0 <= expert_idx < len(self.experts):
            for p in self.experts[expert_idx].parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                self.expert_frozen[expert_idx] = True
            self._expert_frozen_flags[expert_idx] = True

    def forward_sparse(
        self,
        x: torch.Tensor,
        dispatch_plan: Optional[Any] = None,
        update_stats: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run shared always-on FFN + per-token sparse specialist dispatch `(k in 0..2)`."""
        base = self.shared_expert(x) if self.shared_expert is not None else torch.zeros_like(x)
        if self.v7_trunk_only or dispatch_plan is None or len(self.experts) == 0:
            return base, self._zero_entropy
        E = len(self.experts)
        if isinstance(dispatch_plan, dict) and dispatch_plan.get("mode") == "dense_mask":
            one_hot_w = dispatch_plan["one_hot_w"][:, :E].to(dtype=x.dtype)  # [N, E]
            x_flat = x.reshape(-1, self.d_model)
            if E == 1:
                spec = self.experts[0](x_flat) * one_hot_w[:, :1]
            else:
                w1 = torch.cat([e.linear1.weight for e in self.experts], dim=0)  # [E*d_ff, D]
                b1 = torch.cat([e.linear1.bias for e in self.experts], dim=0)    # [E*d_ff]
                h_all = self.experts[0].act(F.linear(x_flat, w1, b1)).view(-1, E, self.d_ff)
                h_masked = (h_all * one_hot_w.unsqueeze(-1)).reshape(-1, E * self.d_ff)
                w2 = torch.cat([e.linear2.weight for e in self.experts], dim=1)  # [D, E*d_ff]
                b2 = torch.stack([e.linear2.bias for e in self.experts], dim=0)  # [E, D]
                spec = F.linear(h_masked, w2) + torch.matmul(one_hot_w, b2)
            spec_3d = spec.to(dtype=x.dtype).view_as(x)
        else:
            spec = sparse_weighted_expert_dispatch(
                x.reshape(-1, self.d_model),
                self.experts,
                dispatch_plan,
                out_dim=self.d_model,
            )
            if spec is None:
                return base, self._zero_entropy
            spec_3d = spec.to(dtype=x.dtype).view_as(x)
        if self.training and self.shared_dropout_prob > 0.0 and len(self.experts) >= 2 and self.shared_expert is not None:
            has_spec = spec_3d.detach().abs().sum(dim=-1, keepdim=True) > 1e-8
            drop_mask = (torch.rand(has_spec.shape, device=x.device) < self.shared_dropout_prob) & has_spec
            base = torch.where(drop_mask, base * 0.35, base)
        return (base + spec_3d), self._zero_entropy

    def forward(self, x: torch.Tensor, update_stats: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, L, d_model]
        Dispatches sequences to assigned experts.
        """
        if self.v7_trunk_only and self.shared_expert is not None:
            return self.shared_expert(x), self._zero_entropy
        if len(self.experts) == 1:
            return self.experts[0](x), self._zero_entropy
        if self.force_routed_expert is not None:
            chosen = min(max(int(self.force_routed_expert), 0), len(self.experts) - 1)
            return self.experts[chosen](x), self._zero_entropy
        B = x.shape[0]
        if self.assigned_experts is not None and self.assigned_experts.numel() == B:
            if not self.skip_uniform_check:
                e0 = int(self.assigned_experts[0].item())
                if bool((self.assigned_experts == e0).all().item()):
                    return self.experts[min(e0, len(self.experts) - 1)](x), self._zero_entropy
            was_2d = x.dim() == 2
            x3 = x.unsqueeze(1) if was_2d else x
            ids = self.assigned_experts.clamp(0, len(self.experts) - 1)
            w1 = torch.stack([e.linear1.weight for e in self.experts], dim=0)[ids]
            b1 = torch.stack([e.linear1.bias for e in self.experts], dim=0)[ids]
            w2 = torch.stack([e.linear2.weight for e in self.experts], dim=0)[ids]
            b2 = torch.stack([e.linear2.bias for e in self.experts], dim=0)[ids]
            h = self.experts[0].act(torch.bmm(x3, w1.transpose(1, 2)) + b1.unsqueeze(1))
            out = torch.bmm(h, w2.transpose(1, 2)) + b2.unsqueeze(1)
            return (out.squeeze(1) if was_2d else out), self._zero_entropy

        return self.experts[0](x), self._zero_entropy


class RealDynamicMorphicLayer(nn.Module):
    """Layer combining Causal (or Hierarchical) Multi-Head Attention, optional STM, and Real Dynamic MoE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        initial_experts: int = 1,
        use_three_level_attn: bool = False,
        local_window: int = 4,
        chunk_stride: int = 4,
        causal_attn: bool = True,
        use_rope: bool = False,
        enable_layer_stm: bool = False,
        signed_beta: bool = True,
        layer_stm_delta_steps: int = 2,
        enable_full_stack_adapters: bool = False,
        adapter_rank: int = 24,
        use_shared_expert: bool = False,
        d_ff_shared: Optional[int] = None,
        v7_trunk_only: bool = False,
        init_spec_small: bool = False,
        specialist_init_scale: float = 0.05,
        shared_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_three_level_attn = use_three_level_attn
        self.causal_attn = bool(causal_attn)
        self.use_rope = bool(use_rope) and (self.head_dim % 2 == 0)

        if enable_layer_stm:
            self.stm_norm: Optional[nn.LayerNorm] = nn.LayerNorm(d_model)
            self.layer_stm: Optional[GatedDeltaWorkingMemory] = GatedDeltaWorkingMemory(
                d_model=d_model,
                num_heads=num_heads,
                signed_beta=signed_beta,
                delta_steps=layer_stm_delta_steps,
            )
            self.stm_gate: Optional[nn.Linear] = nn.Linear(d_model, 1)
        else:
            self.stm_norm = None
            self.layer_stm = None
            self.stm_gate = None

        self.stm_adapters: Optional[RoutedAdapterBank] = (
            RoutedAdapterBank(d_model, rank=adapter_rank, initial_experts=initial_experts)
            if (enable_full_stack_adapters and enable_layer_stm)
            else None
        )
        self.attn_adapters: Optional[RoutedAdapterBank] = (
            RoutedAdapterBank(d_model, rank=adapter_rank, initial_experts=initial_experts)
            if enable_full_stack_adapters
            else None
        )

        if use_three_level_attn:
            self.hier_attn: Optional[ThreeLevelAttention] = ThreeLevelAttention(
                d_model=d_model,
                num_heads=num_heads,
                local_window=local_window,
                chunk_stride=chunk_stride,
                use_rope=self.use_rope,
            )
            self.rope = None
            self.q_proj = self.k_proj = self.v_proj = self.out_proj = None
        else:
            self.hier_attn = None
            self.rope = RotaryEmbedding(self.head_dim) if self.use_rope else None
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

        self.last_level_weights: Optional[torch.Tensor] = None
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = RealDynamicMorphicMoE(
            d_model,
            d_ff,
            initial_experts=initial_experts,
            use_shared_expert=use_shared_expert,
            d_ff_shared=d_ff_shared,
            v7_trunk_only=v7_trunk_only,
            init_spec_small=init_spec_small,
            specialist_init_scale=specialist_init_scale,
            shared_dropout_prob=shared_dropout_prob,
        )

    def spawn_expert_adapters(self, source_idx: Optional[int] = None, fresh_init: bool = False) -> None:
        if self.stm_adapters is not None:
            self.stm_adapters.spawn(source_idx=source_idx, fresh_init=fresh_init)
        if self.attn_adapters is not None:
            self.attn_adapters.spawn(source_idx=source_idx, fresh_init=fresh_init)

    def freeze_expert_adapters(self, idx: int) -> None:
        if self.stm_adapters is not None:
            self.stm_adapters.freeze(idx)
        if self.attn_adapters is not None:
            self.attn_adapters.freeze(idx)

    def prune_expert_adapters(self, idx: int) -> None:
        if self.stm_adapters is not None:
            self.stm_adapters.prune(idx)
        if self.attn_adapters is not None:
            self.attn_adapters.prune(idx)

    def forward_context(
        self,
        x: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
        assigned_experts: Optional[torch.Tensor] = None,
        force_expert: Optional[int] = None,
        skip_uniform_check: bool = False,
        dispatch_plan: Optional[Tuple[torch.Tensor, torch.Tensor, List[int]]] = None,
    ) -> torch.Tensor:
        """Run strictly causal working memory (STM) and attention before MoE dispatch."""
        if self.layer_stm is not None and self.stm_norm is not None and self.stm_gate is not None:
            h_stm = self.stm_norm(x)
            stm_out, _ = self.layer_stm(h_stm, return_state=False, write_mask=write_mask)
            if self.stm_adapters is not None:
                stm_out = stm_out + self.stm_adapters(
                    h_stm,
                    assigned_experts=assigned_experts,
                    force_idx=force_expert,
                    skip_uniform_check=skip_uniform_check,
                    dispatch_plan=dispatch_plan,
                )
            x = x + torch.sigmoid(self.stm_gate(h_stm)) * stm_out

        # Multi-head attention (3-level hierarchical or flat)
        h = self.norm1(x)
        if self.use_three_level_attn and self.hier_attn is not None:
            attn_out, level_weights = self.hier_attn(h, memory_context=memory_context, key_mask=write_mask)
            self.last_level_weights = level_weights
            if self.attn_adapters is not None:
                attn_out = attn_out + self.attn_adapters(
                    h,
                    assigned_experts=assigned_experts,
                    force_idx=force_expert,
                    skip_uniform_check=skip_uniform_check,
                    dispatch_plan=dispatch_plan,
                )
            x = x + attn_out
        else:
            B, L, D = h.shape
            head_dim = self.head_dim

            q = self.q_proj(h).view(B, L, self.num_heads, head_dim).transpose(1, 2)
            k = self.k_proj(h).view(B, L, self.num_heads, head_dim).transpose(1, 2)
            v = self.v_proj(h).view(B, L, self.num_heads, head_dim).transpose(1, 2)
            if self.rope is not None:
                q, k = self.rope.apply_rotary(q, k)

            if write_mask is not None and self.causal_attn:
                causal, past_only, pair_past = _get_layer_causal_masks(L, str(q.device))
                key_gate = write_mask[:, :L].to(device=q.device, dtype=q.dtype).clamp(min=1e-4, max=1.0)
                log_gate = torch.log(key_gate).view(B, 1, 1, L)
                log_next = F.pad(log_gate[..., 1:], (0, 1), value=0.0)
                bias = torch.zeros(B, 1, L, L, device=q.device, dtype=q.dtype)
                bias = bias.masked_fill(~causal, -1e9)
                zeros = torch.zeros_like(log_gate)
                bias = bias + torch.where(past_only, log_gate, zeros)
                bias = bias + torch.where(pair_past, log_next, zeros)
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            else:
                attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal_attn)
            context = attn_out.transpose(1, 2).contiguous().view(B, L, D)
            attn_res = self.out_proj(context)
            if self.attn_adapters is not None:
                attn_res = attn_res + self.attn_adapters(
                    h,
                    assigned_experts=assigned_experts,
                    force_idx=force_expert,
                    skip_uniform_check=skip_uniform_check,
                    dispatch_plan=dispatch_plan,
                )
            x = x + attn_res
        return x

    def forward_moe(
        self,
        x: torch.Tensor,
        update_stats: bool = True,
        dispatch_plan: Optional[Tuple[torch.Tensor, torch.Tensor, List[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run MoE FFN dispatch on contextualized hidden states."""
        if dispatch_plan is not None or self.moe.v7_trunk_only or self.moe.use_shared_expert:
            moe_out, entropy = self.moe.forward_sparse(self.norm2(x), dispatch_plan=dispatch_plan, update_stats=update_stats)
        else:
            moe_out, entropy = self.moe(self.norm2(x), update_stats=update_stats)
        return x + moe_out, entropy

    def forward(
        self,
        x: torch.Tensor,
        update_stats: bool = True,
        memory_context: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
        assigned_experts: Optional[torch.Tensor] = None,
        force_expert: Optional[int] = None,
        skip_uniform_check: bool = False,
        dispatch_plan: Optional[Tuple[torch.Tensor, torch.Tensor, List[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_ctx = self.forward_context(
            x,
            memory_context=memory_context,
            write_mask=write_mask,
            assigned_experts=assigned_experts,
            force_expert=force_expert,
            skip_uniform_check=skip_uniform_check,
            dispatch_plan=dispatch_plan,
        )
        return self.forward_moe(x_ctx, update_stats=update_stats, dispatch_plan=dispatch_plan)


class RealDecoder(nn.Module):
    """Modular output head decoder: supports per-domain heads or shared base + rank-r expert LoRA residuals."""

    def __init__(
        self,
        d_model: int,
        chunk_size: int = 4,
        vocab_size: int = 256,
        enable_lora_residual: bool = False,
        lora_rank: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.vocab_size = vocab_size
        self.enable_lora_residual = bool(enable_lora_residual)
        self.lora_rank = max(4, int(lora_rank))
        self.final_norm = nn.LayerNorm(d_model)
        if self.enable_lora_residual:
            self.shared_base: Optional[nn.Linear] = nn.Linear(d_model, chunk_size * vocab_size)
            self.lora_a = nn.ModuleList([nn.Linear(d_model, self.lora_rank, bias=False)])
            self.lora_b = nn.ModuleList([nn.Linear(self.lora_rank, chunk_size * vocab_size, bias=False)])
            with torch.no_grad():
                self.lora_b[0].weight.zero_()
            self.expert_heads = nn.ModuleList([])
        else:
            self.shared_base = None
            self.lora_a = nn.ModuleList([])
            self.lora_b = nn.ModuleList([])
            self.expert_heads = nn.ModuleList([
                nn.Linear(d_model, chunk_size * vocab_size)
            ])

    def spawn_expert_head(self) -> int:
        device = next(self.parameters()).device
        if self.enable_lora_residual:
            la = nn.Linear(self.d_model, self.lora_rank, bias=False).to(device)
            lb = nn.Linear(self.lora_rank, self.chunk_size * self.vocab_size, bias=False).to(device)
            with torch.no_grad():
                lb.weight.zero_()
            self.lora_a.append(la)
            self.lora_b.append(lb)
            return len(self.lora_a) - 1
        new_head = nn.Linear(self.d_model, self.chunk_size * self.vocab_size).to(device)
        self.expert_heads.append(new_head)
        return len(self.expert_heads) - 1

    def freeze_expert_head(self, idx: int) -> None:
        if self.enable_lora_residual:
            if 0 <= idx < len(self.lora_a):
                for p in self.lora_a[idx].parameters():
                    p.requires_grad_(False)
                for p in self.lora_b[idx].parameters():
                    p.requires_grad_(False)
            return
        if 0 <= idx < len(self.expert_heads):
            for p in self.expert_heads[idx].parameters():
                p.requires_grad_(False)

    def prune_expert_head(self, idx: int) -> bool:
        """Cleanly remove a decoder head matching a pruned expert."""
        if self.enable_lora_residual:
            if len(self.lora_a) <= 1 or not (0 <= idx < len(self.lora_a)):
                return False
            del self.lora_a[idx]
            del self.lora_b[idx]
            return True
        if len(self.expert_heads) <= 1 or not (0 <= idx < len(self.expert_heads)):
            return False
        del self.expert_heads[idx]
        return True

    def forward(
        self,
        h: torch.Tensor,
        assigned_experts: Optional[torch.Tensor] = None,
        force_expert: Optional[int] = None,
        skip_uniform_check: bool = False,
        dispatch_plan: Optional[Tuple[torch.Tensor, torch.Tensor, List[int]]] = None,
    ) -> torch.Tensor:
        B, seq_chunks, D = h.shape
        h_norm = self.final_norm(h)
        if self.enable_lora_residual and self.shared_base is not None:
            base = self.shared_base(h_norm)  # [B, seq_chunks, V_out]
            if dispatch_plan is not None:
                if isinstance(dispatch_plan, dict) and dispatch_plan.get("mode") == "dense_mask":
                    E = len(self.lora_a)
                    if E > 0:
                        w = dispatch_plan["one_hot_w"][:, :E].to(dtype=h_norm.dtype)
                        hf = h_norm.reshape(-1, D)
                        if E == 1:
                            res = self.lora_b[0](self.lora_a[0](hf)) * w[:, :1]
                        else:
                            wa = torch.cat([la.weight for la in self.lora_a], dim=0)  # [E*R, D]
                            wb = torch.cat([lb.weight for lb in self.lora_b], dim=1)  # [V_out, E*R]
                            ha = F.linear(hf, wa).view(-1, E, self.lora_rank)
                            hm = (ha * w.unsqueeze(-1)).reshape(-1, E * self.lora_rank)
                            res = F.linear(hm, wb)
                        base = base + res.to(dtype=base.dtype).view_as(base)
                else:
                    res = sparse_weighted_expert_dispatch(
                        h_norm.reshape(-1, D),
                        self.lora_a,
                        dispatch_plan,
                        fn_call=lambda la, c, idx: self.lora_b[idx](la(c)),
                        out_dim=base.size(-1),
                    )
                    if res is not None:
                        base = base + res.to(dtype=base.dtype).view_as(base)
            return base.view(B, seq_chunks, self.chunk_size, self.vocab_size)

        if len(self.expert_heads) == 1:
            return self.expert_heads[0](h_norm).view(B, seq_chunks, self.chunk_size, self.vocab_size)
        if force_expert is not None:
            idx = force_expert if 0 <= force_expert < len(self.expert_heads) else 0
            return self.expert_heads[idx](h_norm).view(B, seq_chunks, self.chunk_size, self.vocab_size)

        if assigned_experts is not None and len(self.expert_heads) > 1:
            if not skip_uniform_check:
                e0 = int(assigned_experts[0].item())
                if assigned_experts.numel() == B and bool((assigned_experts == e0).all().item()):
                    idx = e0 if 0 <= e0 < len(self.expert_heads) else 0
                    return self.expert_heads[idx](h_norm).view(B, seq_chunks, self.chunk_size, self.vocab_size)
            if assigned_experts.numel() == B:
                valid = (assigned_experts >= 0) & (assigned_experts < len(self.expert_heads))
                ids = torch.where(valid, assigned_experts, torch.zeros_like(assigned_experts))
                hw = torch.stack([hd.weight for hd in self.expert_heads], dim=0)[ids]
                hb = torch.stack([hd.bias for hd in self.expert_heads], dim=0)[ids]
                out = torch.bmm(h_norm, hw.transpose(1, 2)) + hb.unsqueeze(1)
                return out.view(B, seq_chunks, self.chunk_size, self.vocab_size)
            logits = torch.zeros(B, seq_chunks, self.chunk_size, self.vocab_size, device=h.device)
            handled = torch.zeros(B, dtype=torch.bool, device=h.device)
            for exp_idx, head in enumerate(self.expert_heads):
                mask = (assigned_experts == exp_idx).nonzero(as_tuple=True)[0]
                if len(mask) > 0:
                    handled[mask] = True
                    logits[mask] = head(h_norm[mask]).view(len(mask), seq_chunks, self.chunk_size, self.vocab_size)
            unhandled = (~handled).nonzero(as_tuple=True)[0]
            if len(unhandled) > 0:
                logits[unhandled] = self.expert_heads[0](h_norm[unhandled]).view(
                    len(unhandled), seq_chunks, self.chunk_size, self.vocab_size
                )
            return logits

        active_head = self.expert_heads[-1]
        return active_head(h_norm).view(B, seq_chunks, self.chunk_size, self.vocab_size)


class RealSelfStabilizingDMoE(nn.Module):
    """
    Complete Real-Valued Dynamic Morphic MoE (FP32) with optional
    Surprise-Driven Dynamic Latent Thinking (Phase A2 / Track R).
    """

    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 2,
        d_ff: int = 128,
        initial_layers: int = 2,
        initial_experts: int = 1,
        chunk_size: int = 4,
        enable_thinking: bool = False,
        max_thinking_steps: int = 6,
        skip_threshold: float = 0.12,
        theta_halt: float = 0.18,
        abstain_threshold: float = 0.65,
        auto_lifecycle: bool = False,
        max_experts: int = 8,
        min_mature_steps: int = 25,
        novel_confirm_steps: int = 3,
        novel_surprise_threshold: float = 0.45,
        dormant_age: int = 60,
        prune_age: int = 150,
        min_usage_threshold: float = 0.02,
        revival_sim_threshold: float = 0.80,
        enable_memory: bool = False,
        num_ltm_slots: int = 64,
        enable_three_level_attn: bool = False,
        local_window: int = 4,
        chunk_stride: int = 4,
        enable_world_model: bool = False,
        wm_rollout_horizon: int = 3,
        causal_attn: bool = True,
        use_rope: bool = False,
        enable_layer_stm: bool = False,
        signed_beta: bool = True,
        auto_write_ltm: bool = False,
        layer_stm_delta_steps: int = 2,
        router_mode: str = "content",
        num_router_protos: int = 32,
        router_lda_reg: float = 0.05,
        router_sim_threshold: float = 0.55,
        enable_pc_rectification: bool = False,
        pc_thresh: float = 0.25,
        pc_suppress_gain: float = 1.8,
        pc_span_window: int = 2,
        route_during_train: bool = False,
        hybrid_mode: bool = False,
        router_tail_window: int = 0,
        use_trunk_routing: bool = False,
        router_context_blend: float = 0.20,
        extend_loss_margin: float = 0.25,
        align_wm_target_with_encoder: bool = False,
        enable_full_stack_adapters: bool = False,
        adapter_rank: int = 24,
        router_use_idf: bool = False,
        router_center_multi: bool = False,
        train_revisit_conf_margin: float = 0.0,
        preserve_input_novel_on_extend: bool = False,
        defer_on_novel_streak: bool = False,
        enable_mixed_split: bool = False,
        mixed_split_gap: float = 0.12,
        max_coactive_experts: int = 4,
        halt_use_energy_only: bool = False,
        use_marginal_drop_halt: bool = False,
        min_surprise_drop: float = 0.01,
        skip_final_lookahead: bool = False,
        pc_mode: str = "legacy",
        ltm_null_logit: float = 3.2,
        layer_stm_every: int = 1,
        use_v7_core: bool = False,
        use_v8_core: bool = False,
        use_multiscale_chunker: bool = False,
        router_layer: Optional[int] = None,
        d_ff_shared: Optional[int] = None,
        d_ff_spec: Optional[int] = None,
        d_route: int = 96,
        router_num_keys: int = 8,
        detach_decoder: bool = True,
        lora_decoder_rank: int = 16,
        z_nuance_thresh: float = 2.0,
        d_sem_nuance_thresh: float = 2.0,
        nuance_cluster_min: int = 20,
        nuance_cluster_cos: float = 0.72,
        use_rel_pool: bool = True,
        enable_nuance_escalation: bool = True,
        enable_surface_whitening: bool = False,
        enable_cris_loss: bool = False,
        enable_dual_ridge_ep: bool = False,
        dual_ridge_op_rank: int = 48,
        decouple_nuance_lifecycle: bool = False,
        specialist_init_scale: float = 0.05,
        shared_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.chunk_size = chunk_size
        self.enable_thinking = enable_thinking
        self.auto_lifecycle = auto_lifecycle
        self.enable_memory = enable_memory
        self.enable_three_level_attn = enable_three_level_attn
        self.enable_world_model = enable_world_model
        self.causal_attn = bool(causal_attn)
        self.use_rope = bool(use_rope)
        self.auto_write_ltm = bool(auto_write_ltm)
        self.router_mode = str(router_mode)
        self.enable_pc_rectification = bool(enable_pc_rectification)
        self.pc_thresh = float(pc_thresh)
        self.pc_suppress_gain = float(pc_suppress_gain)
        self.pc_span_window = max(1, int(pc_span_window))
        self.pc_mode = str(pc_mode)
        self.layer_stm_every = max(1, int(layer_stm_every))
        self.route_during_train = bool(route_during_train)
        self.hybrid_mode = bool(hybrid_mode)
        self.router_tail_window = max(0, int(router_tail_window))
        self.use_trunk_routing = bool(use_trunk_routing)
        self.align_wm_target_with_encoder = bool(align_wm_target_with_encoder)
        self.enable_full_stack_adapters = bool(enable_full_stack_adapters)
        self.adapter_rank = int(adapter_rank)
        self.train_revisit_conf_margin = float(train_revisit_conf_margin)
        self.router_context_blend = float(router_context_blend)
        self.extend_loss_margin = float(extend_loss_margin)
        self.use_v8_core = bool(use_v8_core)
        self.use_v7_core = bool(use_v7_core or self.use_v8_core or self.router_mode == "causal_context")
        self.use_multiscale_chunker = bool(use_multiscale_chunker)
        self.router_layer = (
            min(3, max(0, initial_layers - 1))
            if router_layer is None
            else min(max(0, int(router_layer)), max(0, initial_layers - 1))
        )
        self._moe_ref_layer_idx: int = self.router_layer if self.use_v7_core else 0
        self.detach_decoder = bool(detach_decoder)
        self.use_rel_pool = bool(use_rel_pool)
        self.enable_nuance_escalation = bool(enable_nuance_escalation)
        self.enable_surface_whitening = bool(enable_surface_whitening)
        self.enable_cris_loss = bool(enable_cris_loss)
        self.enable_dual_ridge_ep = bool(enable_dual_ridge_ep)
        self.dual_ridge_op_rank = int(dual_ridge_op_rank)
        self.decouple_nuance_lifecycle = bool(decouple_nuance_lifecycle)
        self.specialist_init_scale = float(specialist_init_scale)
        self.shared_dropout_prob = float(shared_dropout_prob)
        if (not self.enable_nuance_escalation) and (not self.decouple_nuance_lifecycle):
            self.z_nuance_thresh = 1e9
            self.d_sem_nuance_thresh = 1e9
        else:
            self.z_nuance_thresh = float(z_nuance_thresh)
            self.d_sem_nuance_thresh = float(d_sem_nuance_thresh)
        self.nuance_cluster_min = int(nuance_cluster_min)
        self.nuance_cluster_cos = float(nuance_cluster_cos)
        self.null_space_projector: Optional[Any] = None

        self.byte_embed = nn.Embedding(258, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        self.chunk_proj = nn.Linear(d_model * chunk_size, d_model)
        self.ms_chunker: Optional[MultiScaleByteChunker] = (
            MultiScaleByteChunker(d_model=d_model, chunk_size=chunk_size)
            if self.use_multiscale_chunker
            else None
        )
        if self.ms_chunker is not None:
            self.byte_embed.weight.requires_grad_(False)
            self.pos_embed.requires_grad_(False)
            for p in self.chunk_proj.parameters():
                p.requires_grad_(False)
        self.sequence_router = SequenceContentRouter(d_model=d_model, sim_threshold=0.5)
        self.causal_router: Optional[CausalPrototypeRouter] = (
            CausalPrototypeRouter(
                d_model=d_model,
                max_protos=num_router_protos,
                lda_reg=router_lda_reg,
                sim_threshold=router_sim_threshold,
                tail_window=self.router_tail_window,
                use_idf_weights=router_use_idf,
                center_multi_expert=router_center_multi,
                revisit_sim_margin=train_revisit_conf_margin,
            )
            if self.router_mode == "causal_proto"
            else None
        )

        self.context_router: Optional[CausalContextTokenRouter] = (
            CausalContextTokenRouter(
                d_model=d_model,
                d_route=d_route,
                max_experts=max_experts,
                initial_experts=initial_experts,
                num_keys=router_num_keys,
                enable_surface_whitening=self.enable_surface_whitening,
                surface_dim=(6 if self.enable_surface_whitening else 12),
                delta_mix=(0.0 if (self.decouple_nuance_lifecycle and not self.enable_nuance_escalation) else 0.05),
                tau_top2=(1.05 if (self.decouple_nuance_lifecycle and not self.enable_nuance_escalation) else 0.82),
                scale=(16.0 if self.enable_cris_loss else 12.0),
                ema_decay=(0.92 if self.enable_cris_loss else 0.85),
                grad_scale=(0.35 if self.enable_cris_loss else 0.25),
                use_loss_free_balance=True,
                bias_lr=(0.02 if self.enable_cris_loss else 0.05),
            )
            if self.use_v7_core
            else None
        )
        self.expr_head: Optional[nn.Linear] = nn.Linear(d_model, d_model) if self.use_v7_core else None
        self.expr_proj: Optional[nn.Sequential] = (
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if self.use_v8_core
            else None
        )
        self.ctx_head: Optional[nn.Linear] = nn.Linear(d_model, d_model) if self.use_v7_core else None
        self.abstain_head: Optional[nn.Linear] = nn.Linear(d_model, 1) if self.use_v7_core else None
        self.fsp_head: Optional[nn.Linear] = nn.Linear(d_model, 128) if self.use_v7_core else None
        self.rel_pool_proj: Optional[nn.Linear] = (
            nn.Linear(3 * d_model, d_model) if (self.use_v8_core and self.use_rel_pool) else None
        )
        self.wm_ret_proj: Optional[nn.Linear] = nn.Linear(d_model, d_model) if self.use_v8_core else None
        self.wm_fut_mlp: Optional[nn.Sequential] = (
            nn.Sequential(
                nn.Linear(3 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if self.use_v8_core
            else None
        )
        self.wm_ret_gate: Optional[nn.Parameter] = (
            nn.Parameter(torch.tensor(-1.0)) if self.use_v8_core else None
        )
        self.i7_struct_head: Optional[nn.Linear] = nn.Linear(d_model, 3) if self.use_v8_core else None
        if self.use_v8_core:
            with torch.no_grad():
                if self.expr_head is not None:
                    self.expr_head.weight.mul_(0.35).add_(0.65 * torch.eye(d_model))
                if self.ctx_head is not None:
                    self.ctx_head.weight.mul_(0.35).add_(0.65 * torch.eye(d_model))
            self.equiv_proto_bank: Optional[nn.Embedding] = nn.Embedding(256, d_model)
            self.register_buffer("equiv_q_raw", torch.zeros(256, d_model))
            self.register_buffer("equiv_q_proj", torch.zeros(256, d_model))
            self.register_buffer("equiv_q_lbl", torch.full((256,), -1, dtype=torch.long))
            self.register_buffer("equiv_q_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("wm_ret_q", torch.zeros(128, d_model))
            self.register_buffer("wm_ret_q_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("wm_ret_q_full", torch.zeros(1, dtype=torch.long))
            self._v8_step_ctr: int = 0
            self._equiv_q_ptr_i: int = 0
            self._wm_ret_q_ptr_i: int = 0
            self._wm_ret_q_full_i: int = 0
        else:
            self.equiv_proto_bank = None
        self.nuance_buffer: Optional[NuanceClusterBuffer] = (
            NuanceClusterBuffer(cap=128, d_route=d_route) if self.use_v7_core else None
        )
        self.context_nuance_detector: Optional[UnifiedNoveltyDetector] = (
            UnifiedNoveltyDetector(d_model=d_model) if self.use_v7_core else None
        )
        self.concept_quantizer: Optional[SparseConceptQuantizer] = (
            SparseConceptQuantizer(d_model=d_model, num_atoms=48) if self.use_v7_core else None
        )

        sh_dff_val = int(d_ff_shared) if d_ff_shared is not None else d_ff
        sp_dff_val = int(d_ff_spec) if d_ff_spec is not None else d_ff
        self.layers = nn.ModuleList([
            RealDynamicMorphicLayer(
                d_model,
                num_heads,
                (sp_dff_val if self.use_v7_core else d_ff),
                initial_experts=initial_experts,
                use_three_level_attn=(
                    (
                        l_idx == initial_layers - 1
                        or l_idx % 4 == 3
                        or (self.use_v7_core and l_idx == self.router_layer)
                    )
                    if (self.hybrid_mode and enable_three_level_attn and initial_layers > 1)
                    else enable_three_level_attn
                ),
                local_window=local_window,
                chunk_stride=chunk_stride,
                causal_attn=self.causal_attn,
                use_rope=self.use_rope,
                enable_layer_stm=(
                    (
                        (l_idx < initial_layers - 1 and l_idx % 4 != 3 and not (self.use_v7_core and l_idx == self.router_layer))
                        if (self.hybrid_mode and enable_layer_stm and initial_layers > 1)
                        else enable_layer_stm
                    )
                    and (l_idx % self.layer_stm_every == 0)
                ),
                signed_beta=signed_beta,
                layer_stm_delta_steps=(
                    1 if (self.hybrid_mode and l_idx > 0) else layer_stm_delta_steps
                ),
                enable_full_stack_adapters=(
                    self.enable_full_stack_adapters
                    and (l_idx >= self.router_layer if self.use_v7_core else (l_idx > 0 or initial_layers == 1))
                ),
                adapter_rank=self.adapter_rank,
                use_shared_expert=(self.use_v7_core and l_idx >= self.router_layer),
                d_ff_shared=(sh_dff_val if l_idx < self.router_layer else (d_ff_shared or d_ff)),
                v7_trunk_only=(self.use_v7_core and l_idx < self.router_layer),
                init_spec_small=(self.use_v7_core and l_idx >= self.router_layer),
                specialist_init_scale=self.specialist_init_scale,
                shared_dropout_prob=(self.shared_dropout_prob if l_idx >= self.router_layer else 0.0),
            )
            for l_idx in range(initial_layers)
        ])
        self.decoder = RealDecoder(
            d_model,
            chunk_size=chunk_size,
            vocab_size=256,
            enable_lora_residual=self.use_v7_core,
            lora_rank=lora_decoder_rank,
        )
        self.upkeeper = EvidenceAccumulatingUpkeeper()

        if enable_memory:
            self.memory: Optional[DualTimescaleMemory] = DualTimescaleMemory(
                d_model=d_model,
                num_heads=num_heads,
                num_ltm_slots=num_ltm_slots,
                signed_beta=signed_beta,
                ltm_null_logit=ltm_null_logit,
                enable_dual_ridge_op=self.enable_dual_ridge_ep,
                op_rank=self.dual_ridge_op_rank,
            )
        else:
            self.memory = None

        self.abstain_threshold = float(abstain_threshold)
        if enable_world_model:
            self.world_model: Optional[LatentWorldModel] = LatentWorldModel(
                d_model=d_model,
                d_ff=d_ff,
                ema_decay=0.94 if self.use_v8_core else 0.99,
                max_rollout_horizon=wm_rollout_horizon,
            )
            self.world_model.bind_encoder(
                self.byte_embed,
                self.chunk_proj,
                self.pos_embed,
                chunk_size,
                add_pos=not (self.use_rope and self.align_wm_target_with_encoder),
            )
            if self.use_v7_core and self.router_layer > 0:
                trunk_t = [self.layers[0]] if self.use_v8_core else list(self.layers[: self.router_layer])
                self.world_model.bind_deep_trunk(
                    self.ms_chunker,
                    trunk_t,
                    None if self.use_v8_core else self.expr_head,
                )
            elif self.ms_chunker is not None:
                self.world_model.bind_chunker(self.ms_chunker)
            self.wm_imagine_proj = nn.Linear(d_model, d_model)
            self.wm_imagine_gate = nn.Parameter(torch.tensor(0.25))
        else:
            self.world_model = None
            self.wm_imagine_proj = None
            self.wm_imagine_gate = None

        if enable_thinking:
            self.thinker: Optional[SurpriseDrivenLatentThinker] = SurpriseDrivenLatentThinker(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_thinking_steps=max_thinking_steps,
                skip_threshold=skip_threshold,
                theta_halt=theta_halt,
                abstain_threshold=abstain_threshold,
                enable_expert_adapters=self.enable_full_stack_adapters,
                adapter_rank=self.adapter_rank,
                initial_experts=initial_experts,
                use_marginal_drop_halt=use_marginal_drop_halt,
                min_surprise_drop=min_surprise_drop,
                halt_use_energy_only=halt_use_energy_only,
                skip_final_lookahead=skip_final_lookahead,
            )
        else:
            self.thinker = None

        if auto_lifecycle:
            self.lifecycle: Optional[AutonomousExpertLifecycleManager] = AutonomousExpertLifecycleManager(
                d_model=d_model,
                max_experts=max_experts,
                min_mature_steps=min_mature_steps,
                novel_confirm_steps=novel_confirm_steps,
                novel_surprise_threshold=novel_surprise_threshold,
                dormant_age=dormant_age,
                prune_age=prune_age,
                min_usage_threshold=min_usage_threshold,
                revival_sim_threshold=revival_sim_threshold,
                extend_loss_margin=self.extend_loss_margin,
                preserve_input_novel_on_extend=preserve_input_novel_on_extend,
                defer_on_novel_streak=defer_on_novel_streak,
                enable_mixed_split=enable_mixed_split,
                mixed_split_gap=mixed_split_gap,
                max_coactive_experts=max_coactive_experts,
            )
        else:
            self.lifecycle = None

        self.last_assigned_experts: Optional[torch.Tensor] = None
        self.last_routed_expert: int = 0
        self.last_lifecycle_events: List[str] = []

    def attach_null_space_projector(self, projector: Any) -> None:
        """Attach a NullSpaceProjector so autonomous expert spawns automatically consolidate prior subspaces."""
        self.null_space_projector = projector

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode byte sequences into chunk latents."""
        if self.ms_chunker is not None:
            h = self.ms_chunker(x)
            n_chunks = h.size(1)
            if not self.use_rope and n_chunks <= self.pos_embed.size(1):
                h = h + self.pos_embed[:, :n_chunks, :]
            return h
        B, L = x.shape
        embedded = self.byte_embed(x)  # [B, L, d_model]
        n_chunks = L // self.chunk_size
        chunked = embedded.view(B, n_chunks, self.chunk_size * self.d_model)
        h = self.chunk_proj(chunked)  # [B, n_chunks, d_model]
        if not self.use_rope and n_chunks <= self.pos_embed.size(1):
            h = h + self.pos_embed[:, :n_chunks, :]
        return h

    def reset_ltm(self) -> None:
        """Clear persistent long-term memory slots between independent episodes."""
        if self.memory is not None:
            self.memory.reset_long_term()

    def _pool_pos_chunks(self, pos_chunks: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool sparse nonnegative multi-scale k-gram chunks via valid-masked mean+max and LayerNorm."""
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
        """Causal prefix-to-continuation retrieval projection combining WM rollout, prefix latent, & multi-scale k-grams."""
        if self.use_v8_core and self.wm_ret_proj is not None:
            base = z_pred_raw.float() + 0.5 * self.wm_ret_proj(z_pred_raw).float()
            if self.wm_fut_mlp is not None and self.wm_ret_gate is not None and h_ctx is not None and pos_pre is not None:
                alpha = torch.sigmoid(self.wm_ret_gate.float())
                mlp_in = torch.cat([base, h_ctx.float(), pos_pre.float()], dim=-1).to(
                    dtype=self.wm_fut_mlp[0].weight.dtype
                )
                delta = F.normalize(self.wm_fut_mlp(mlp_in).float(), dim=-1)
                return F.normalize(alpha * F.normalize(pos_pre.float(), dim=-1) + (1.0 - alpha) * delta, dim=-1)
            return base.to(dtype=z_pred_raw.dtype)
        return z_pred_raw

    def _forward_v7_trunk_to_post_moe(
        self,
        x: torch.Tensor,
        h_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Strictly causal forward through memory, all MoE layers, and Thinker (no Decoder, no stat writes)."""
        h = self.encode(x) if h_enc is None else h_enc
        T = h.size(1)
        _, content_mask = self._compute_chunk_masks(x, T)
        if self.enable_memory and self.memory is not None:
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
        expert_states = list(self.lifecycle.expert_states) if self.lifecycle is not None else None
        q_state = self.context_router.compute_queries(
            h_attn3,
            expert_states=expert_states,
            surface_feats=surface_feats,
            update_surface_stats=False,
        )
        route_res = self.context_router.select_experts(
            q_state,
            nuance_mask=None,
            content_mask=(content_mask if self.decouple_nuance_lifecycle else None),
        )
        dispatch_plan = build_token_dispatch_plan(
            route_res["top_ids"], route_res["top_w"], num_experts=len(r_layer.moe.experts)
        )
        h, _ = r_layer.forward_moe(h_attn3, update_stats=False, dispatch_plan=dispatch_plan)
        for layer in self.layers[self.router_layer + 1 :]:
            h, _ = layer(h, update_stats=False, dispatch_plan=dispatch_plan)
        if self.enable_world_model and self.world_model is not None and self.wm_imagine_proj is not None and self.wm_imagine_gate is not None:
            wm_im = self.world_model.imagine(h, horizon=1)
            z_im = wm_im["imagined_latents"][:, :, 0, :]
            conf = (1.0 - wm_im["imagined_uncertainty"][:, :, :1]).detach()
            h = h + torch.tanh(self.wm_imagine_gate) * conf * self.wm_imagine_proj(z_im)
        if self.enable_thinking and self.thinker is not None:
            t_out = self.thinker(
                h,
                decoder_fn=None,
                force_steps=None,
                nuance_mask=None,
                dispatch_plan=dispatch_plan,
            )
            h = t_out["z_final"]
        return h

    def extract_router_queries(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lightweight lower-trunk + router query extraction `(rq_pool [B, d_route], q_tok [B, T, d_route])`
        for self-supervised surface-perturbation invariance (CRIS) without running upper MoE/WM/Thinker.
        """
        h = self.encode(x)
        T = h.size(1)
        _, content_mask = self._compute_chunk_masks(x, T)
        if self.enable_memory and self.memory is not None:
            h = self.memory(h, update_ltm=False, return_stm_state=False)["fused"]
        for l_idx in range(self.router_layer):
            h, _ = self.layers[l_idx](h, update_stats=False)
        h_attn3 = self.layers[self.router_layer].forward_context(h)
        surface_feats = (
            compute_chunk_surface_features(x, chunk_size=self.chunk_size, num_chunks=T, format_only=True)
            if self.enable_surface_whitening
            else None
        )
        expert_states = list(self.lifecycle.expert_states) if self.lifecycle is not None else None
        q_state = self.context_router.compute_queries(
            h_attn3,
            expert_states=expert_states,
            surface_feats=surface_feats,
            update_surface_stats=False,
        )
        q_tok = q_state["q_tok"]
        w_cm = content_mask.unsqueeze(-1).float()
        rq_pool = F.normalize((q_tok.float() * w_cm).sum(dim=1) / w_cm.sum(dim=1).clamp_min(1.0), dim=-1)
        return rq_pool, q_tok

    def extract_episode_kv(
        self,
        x_ep: torch.Tensor,
        top_k: int = 8,
        min_nll: float = 0.7,
        cached_out: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Differentiably extract key->value episodic memory candidates from an episode x_ep [B, L]
        by finding high-surprise transitions (`chunk_size == 1` via NLL, or `chunk_size > 1` via
        world-model / latent surprise). Uses NO target labels outside x_ep itself.
        """
        if not self.enable_memory or self.memory is None or x_ep.size(1) < max(2, 2 * self.chunk_size):
            return None

        B, L = x_ep.shape
        was_training = self.training
        # Fast v8 contextual path when all transitions are kept (`min_nll <= 0.0`):
        # avoids a full 6-layer+Thinker forward pass and builds causal Layer-0 keys directly.
        if self.use_v8_core and cached_out is None and float(min_nll) <= 0.0:
            h_ep_enc = self.encode(x_ep)
            T_full = h_ep_enc.size(1)
            T_m = T_full - 1
            if T_m < 1:
                return None
            # Match `_forward_v7` Layer 0 representation so episodic keys & queries live in the same space
            h_ep_ctx, _ = self.layers[0](h_ep_enc, update_stats=False)
            _, content_mask = self._compute_chunk_masks(x_ep, T_full)
            k = min(max(1, int(top_k)), T_m)
            top_idx = torch.arange(T_m - k, T_m, device=x_ep.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            ep_mask = content_mask[:, 1 : T_m + 1].gather(1, top_idx)
            salience = torch.ones_like(ep_mask, dtype=torch.float32)
            gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, h_ep_enc.size(-1))
            raw_k = torch.gather(h_ep_ctx[:, :-1, :], 1, gather_idx)
            raw_v = torch.gather(h_ep_enc[:, 1:, :], 1, gather_idx)
            ep_keys, ep_vals = self.memory.long_term.encode_kv(raw_k, raw_v, normalize_k=False)
            res: Dict[str, torch.Tensor] = {
                "ep_keys": ep_keys,
                "ep_vals": ep_vals,
                "ep_mask": ep_mask,
                "ep_salience": salience,
            }
            if self.enable_dual_ridge_ep and getattr(self.memory.long_term, "enable_dual_ridge_op", False):
                # Strictly causal post-MoE deep latent at last prompt chunk (t = last_content - 1)
                # mapped to the final answer chunk (t + 1 = last_content), with zero future leakage.
                # Extract trunk support states under no_grad (3x faster, preserves grokking stability)
                # while keeping full gradients through `encode_op_kv` and `read_dual_operator`.
                with torch.no_grad():
                    h_deep = self._forward_v7_trunk_to_post_moe(x_ep, h_enc=h_ep_enc.detach())
                t_grid = torch.arange(T_full, device=x_ep.device).unsqueeze(0).expand(B, T_full)
                last_c = torch.where(content_mask, t_grid, torch.zeros_like(t_grid)).max(dim=1).values.clamp(min=1)
                key_idx = (last_c - 1).view(B, 1, 1).expand(B, 1, h_deep.size(-1))
                val_idx = last_c.view(B, 1, 1).expand(B, 1, h_ep_enc.size(-1))
                raw_op_k = torch.gather(h_deep, 1, key_idx)    # [B, 1, D]
                raw_op_v = torch.gather(h_ep_enc, 1, val_idx)  # [B, 1, D]
                op_keys, op_vals = self.memory.long_term.encode_op_kv(raw_op_k, raw_op_v)
                op_mask = content_mask.gather(1, last_c.unsqueeze(1))
                res["op_keys"] = op_keys
                res["op_vals"] = op_vals
                res["op_mask"] = op_mask
            return res

        # Reuse episode forward output when provided, else run cleanly without touching stats
        out = cached_out if cached_out is not None else self.forward(x_ep, update_stats=False)
        mem_out = out.get("memory_out")
        if mem_out is None:
            return None

        ltm_q = mem_out["ltm_q"][:, :-1, :]
        x_norm_next = mem_out["x_norm"][:, 1:, :]
        T_m = ltm_q.size(1)
        if T_m < 1:
            return None

        if self.chunk_size == 1:
            logits = out["output"].reshape(B, L, -1)
            V = logits.size(-1)
            with torch.no_grad():
                nll = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, V),
                    x_ep[:, 1:].reshape(-1),
                    reduction="none",
                ).view(B, L - 1)
                k = min(max(1, int(top_k)), T_m)
                top_vals, top_idx = torch.topk(nll[:, :T_m], k=k, dim=-1)
                ep_mask = top_vals >= float(min_nll)
                max_ent = math.log(float(V))
                salience = (top_vals / max(max_ent, 1e-6)).clamp(0.0, 1.0)
                if self.enable_pc_rectification and out.get("world_model_imagine") is not None:
                    pc_err_full = out["world_model_imagine"].get("predictive_coding_error")
                    if pc_err_full is not None and pc_err_full.size(1) >= L:
                        pc_next = torch.gather(pc_err_full[:, 1:].to(salience.device), 1, top_idx)
                        salience = salience * (1.0 - 0.85 * pc_next).clamp(0.05, 1.0)
                        ep_mask = ep_mask & (pc_next < (self.pc_thresh + 0.20))
        else:
            with torch.no_grad():
                pc_full = out.get("pre_pc_error")
                if pc_full is not None and pc_full.size(1) > T_m:
                    surp = pc_full[:, 1 : T_m + 1].detach().float()
                else:
                    fused = mem_out["fused"].detach().float()
                    surp = (1.0 - F.cosine_similarity(fused[:, :-1, :], fused[:, 1:, :], dim=-1)).clamp(0.0, 1.0)
                if "chunk_content_mask" in out and out["chunk_content_mask"] is not None:
                    vm = out["chunk_content_mask"][:, 1 : T_m + 1]
                    surp = torch.where(vm, surp, torch.full_like(surp, -1.0))
                k = min(max(1, int(top_k)), T_m)
                top_vals, top_idx = torch.topk(surp, k=k, dim=-1)
                thr_chunk = min(float(min_nll), 0.12)
                ep_mask = top_vals >= thr_chunk
                salience = top_vals.clamp(0.0, 1.0)

        D = ltm_q.size(-1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, D)
        if self.use_v8_core:
            # Contextualize episode chunks through Layer 0 so each position `t` encodes prefix `(0..t)`
            # while value remains incoming transition `t+1` (zero future leakage!).
            h_ep_enc = self.encode(x_ep)
            h_ep_ctx = self.layers[0].forward_context(h_ep_enc)
            k_src = h_ep_ctx[:, :-1, :]
            v_src = h_ep_enc[:, 1:, :]
        else:
            k_src = ltm_q
            v_src = x_norm_next
        raw_k = torch.gather(k_src, 1, gather_idx)
        raw_v = torch.gather(v_src, 1, gather_idx)
        ep_keys, ep_vals = self.memory.long_term.encode_kv(raw_k, raw_v)
        if was_training != self.training:
            self.train(was_training)
        return {
            "ep_keys": ep_keys,
            "ep_vals": ep_vals,
            "ep_mask": ep_mask,
            "ep_salience": salience,
        }

    @torch.no_grad()
    def consolidate_ltm(
        self,
        x_ep: torch.Tensor,
        top_k: int = 8,
        min_nll: float = 0.7,
    ) -> int:
        """Consolidate high-surprise transitions from episode x_ep into persistent LTM slots."""
        if not self.enable_memory or self.memory is None:
            return 0
        was_training = self.training
        self.eval()
        try:
            kv = self.extract_episode_kv(x_ep, top_k=top_k, min_nll=min_nll)
            if kv is None:
                return 0
            mask = kv["ep_mask"].reshape(-1)
            if not bool(mask.any().item()):
                return 0
            k_cand = kv["ep_keys"].reshape(-1, self.d_model)[mask]
            v_cand = kv["ep_vals"].reshape(-1, self.d_model)[mask]
            saliences = kv["ep_salience"].reshape(-1)[mask]
            return self.memory.long_term._insert_kv_vectors(k_cand, v_cand, saliences)
        finally:
            self.train(was_training)

    def _maybe_update_expert_prototypes(
        self,
        moe0: RealDynamicMorphicMoE,
        chosen_training_expert: int,
        assigned_experts: torch.Tensor,
        seq_sig: torch.Tensor,
        proto_sig: Optional[torch.Tensor],
        is_novel: object,
        lifecycle_events: List[str],
        unfrozen: List[int],
        update_stats: bool,
    ) -> None:
        """Update active expert exemplars and causal prototypes unless novelty deferral applies."""
        if not (self.training and unfrozen and update_stats and len(moe0.experts) > 0):
            return
        cur_idx = min(chosen_training_expert, len(moe0.experts) - 1)
        has_existing_ex = moe0.experts[cur_idx].exemplars.numel() > 0
        if has_existing_ex and self.auto_lifecycle and self.lifecycle is not None:
            v_last = self.lifecycle.last_novelty_verdict
            v_novel = v_last is not None and v_last.novelty_type != "none"
            is_converged = (
                cur_idx < len(self.lifecycle.expert_converged)
                and self.lifecycle.expert_converged[cur_idx]
            )
            input_novel = (
                isinstance(is_novel, torch.Tensor)
                and is_novel.numel() > 0
                and float(is_novel.float().mean().item()) >= 0.5
            )
            still_calibrating = (
                (not is_converged)
                and self.router_mode == "causal_proto"
                and self.causal_router is not None
                and cur_idx < len(self.causal_router._sim_counts)
                and self.causal_router._sim_counts[cur_idx] < 5
            )
            # Once converged, freeze prototypes immediately on novel/burst ticks so bursts
            # cannot contaminate routing; allow initial causal_proto calibration to finish before deferring.
            defer_proto = (
                (self.lifecycle.novel_streak >= 1 or v_novel or input_novel)
                if is_converged
                else ((not still_calibrating) and (self.lifecycle.novel_streak >= 2 or v_novel))
            )
            if defer_proto or any(
                ev.startswith("spawned_expert_") or ev.startswith("mixed_split_")
                for ev in lifecycle_events
            ):
                return
        if len(unfrozen) > 1:
            for ui in unfrozen:
                u_mask = assigned_experts == ui
                if not moe0._expert_frozen_flags[ui] and bool(u_mask.any().item()):
                    u_seq = seq_sig[u_mask]
                    for layer in self.layers:
                        layer.moe.experts[ui].update_exemplars(u_seq)
                    if proto_sig is not None and self.causal_router is not None:
                        self.causal_router.update_prototypes(ui, proto_sig[u_mask])
            return
        active_mask = assigned_experts == cur_idx
        if not moe0._expert_frozen_flags[cur_idx] and bool(active_mask.any().item()):
            active_seq = seq_sig[active_mask]
            for layer in self.layers:
                layer.moe.experts[cur_idx].update_exemplars(active_seq)
            if proto_sig is not None and self.causal_router is not None:
                self.causal_router.update_prototypes(cur_idx, proto_sig[active_mask])

    def _compute_chunk_masks(self, x: torch.Tensor, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute `(pad_mask [B, T], content_mask [B, T])` masking trailing BYTE_PAD_ID=257 & blank trailing spaces."""
        B, L = x.shape
        C = max(1, int(self.chunk_size))
        need_L = T * C
        if L < need_L:
            xc = F.pad(x, (0, need_L - L), value=257)
        else:
            xc = x[:, :need_L]
        xc = xc.view(B, T, C)
        pad_mask = (xc != 257).any(dim=-1)
        has_p = pad_mask.any(dim=-1, keepdim=True)
        pad_mask = torch.where(has_p, pad_mask, torch.ones_like(pad_mask))
        content_mask = ((xc != 257) & (xc != 32)).any(dim=-1)
        has_c = content_mask.any(dim=-1, keepdim=True)
        content_mask = torch.where(has_c, content_mask, pad_mask)
        return pad_mask, content_mask

    def _valid_relational_pool(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_rel: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute order-sensitive & padding-free pooled representations `(raw_pool, unit_pool, rel_vec)`.
        Preserves directional order (`u_second_half - u_first_half`) so swapped clauses differ cleanly.
        """
        B, T, D = z.shape
        z_f = z.float()
        if mask is None:
            mask = torch.ones((B, T), dtype=torch.bool, device=z.device)
        w = mask.unsqueeze(-1).float()
        denom = w.sum(dim=1).clamp_min(1.0)
        u_mean = (z_f * w).sum(dim=1) / denom

        idx_grid = torch.arange(T, device=z.device).unsqueeze(0).expand(B, T)
        last_idx = torch.where(mask, idx_grid, torch.zeros_like(idx_grid)).max(dim=1).values
        u_last = torch.gather(z_f, 1, last_idx.view(B, 1, 1).expand(B, 1, D)).squeeze(1)

        if not (use_rel and self.rel_pool_proj is not None and getattr(self, "use_rel_pool", True)):
            raw = (0.65 * u_mean + 0.35 * u_last) if self.use_v8_core else u_mean
            raw_d = raw.to(dtype=z.dtype)
            return raw_d, F.normalize(raw, dim=-1).to(dtype=z.dtype), (u_last - u_mean).to(dtype=z.dtype)

        valid_cnt = mask.long().sum(dim=1, keepdim=True)
        cum = mask.long().cumsum(dim=1)
        first_half = mask & (cum <= ((valid_cnt + 1) // 2))
        second_half = mask & (~first_half)
        u_first = (z_f * first_half.unsqueeze(-1).float()).sum(dim=1) / first_half.sum(dim=1, keepdim=True).clamp_min(1).float()
        has_sec = second_half.any(dim=1, keepdim=True)
        u_sec_raw = (z_f * second_half.unsqueeze(-1).float()).sum(dim=1) / second_half.sum(dim=1, keepdim=True).clamp_min(1).float()
        u_sec = torch.where(has_sec, u_sec_raw, u_last)
        u_dir = u_sec - u_first

        rel_in = torch.cat([u_mean, u_dir, u_last], dim=-1).to(dtype=self.rel_pool_proj.weight.dtype)
        rel_delta = torch.tanh(self.rel_pool_proj(rel_in)).float()
        ctx_raw = u_mean + 0.65 * rel_delta
        return ctx_raw.to(dtype=z.dtype), F.normalize(ctx_raw, dim=-1).to(dtype=z.dtype), u_dir.to(dtype=z.dtype)

    def _forward_v7(
        self,
        x: torch.Tensor,
        update_stats: bool = True,
        force_thinking_steps: Optional[int] = None,
        episode_memory: Optional[Dict[str, torch.Tensor]] = None,
        belief_conflict: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        v7/v8 Non-Token Concept-Chunk Cognitive Core Forward Pass:
          1. Lift sensory inputs into causal concept chunks (`MultiScaleByteChunker`), tracking padding/content masks.
          2. Run shared GPM trunk (`Layers 0 .. router_layer-1`).
          3. Run Layer `router_layer` ThreeLevelAttention -> Dual Meaning factorization (`z_expr`, `z_ctx`).
          4. Detect contextual + world-model semantic nuance (`nuance_mask`) and compute
             per-chunk dynamic `k in {0, 1, 2}` routing ONCE for `Layers >= router_layer` & Thinker.
          5. Escalate `nuance_mask` chunks into `k=2` + `SurpriseDrivenLatentThinker`.
          6. Apply optional order-sensitive Relational Pool + optional `RealDecoder` strictly behind `stop_gradient`.
        """
        B, _ = x.shape
        if self.training and self.use_v8_core:
            self._v8_step_ctr += 1
        h = self.encode(x)
        h_embed_seq = h
        pos_chunks_seq = (
            getattr(self.ms_chunker, "last_pos_chunks", None)
            if (self.use_v8_core and self.ms_chunker is not None)
            else None
        )
        T = h.size(1)
        pad_mask, content_mask = self._compute_chunk_masks(x, T)

        mem_out: Optional[Dict[str, torch.Tensor]] = None
        memory_context: Optional[torch.Tensor] = None
        use_dual_op_active = bool(
            self.enable_dual_ridge_ep
            and episode_memory is not None
            and episode_memory.get("op_keys") is not None
            and episode_memory.get("op_vals") is not None
        )
        if self.enable_memory and self.memory is not None:
            ep_k = None if use_dual_op_active else (episode_memory.get("ep_keys") if episode_memory is not None else None)
            ep_v = None if use_dual_op_active else (episode_memory.get("ep_vals") if episode_memory is not None else None)
            ep_m = None if use_dual_op_active else (episode_memory.get("ep_mask") if episode_memory is not None else None)
            do_ltm_w = bool(
                self.training
                and self.auto_write_ltm
                and update_stats
                and ((not self.use_v8_core) or (self._v8_step_ctr % 4 == 0))
            )
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
            if ctx_parts:
                memory_context = torch.cat(ctx_parts, dim=1)
        mem_ctx_tok = mem_out["ltm_out"] + mem_out["stm_out"] if mem_out is not None else None

        tgt_lat: Optional[torch.Tensor] = None
        if self.enable_world_model and self.world_model is not None:
            tgt_lat = self.world_model.encode_target(x)

        # Shared lower trunk (Layers 0 .. router_layer - 1)
        for l_idx in range(self.router_layer):
            h, _ = self.layers[l_idx](h, update_stats=update_stats, memory_context=memory_context)
            if (
                l_idx == 0
                and self.use_v8_core
                and (not use_dual_op_active)
                and episode_memory is not None
                and self.enable_memory
                and self.memory is not None
                and episode_memory.get("ep_keys") is not None
                and episode_memory.get("ep_vals") is not None
            ):
                ep_l0, _ = self.memory.long_term.read(
                    h,
                    ep_keys=episode_memory["ep_keys"],
                    ep_vals=episode_memory["ep_vals"],
                    ep_mask=episode_memory.get("ep_mask"),
                    include_null_for_ep=bool(self.enable_dual_ridge_ep),
                )
                h = h + 0.65 * ep_l0.to(h.dtype)

        # Layer router_layer attention/STM -> contextual routing & nuance detection
        r_layer = self.layers[self.router_layer]
        h_attn3 = r_layer.forward_context(h, memory_context=memory_context)

        pc_err_mid: Optional[torch.Tensor] = None
        wm_unc_mid: Optional[torch.Tensor] = None
        if self.enable_world_model and self.world_model is not None and tgt_lat is not None:
            wm_mid = self.world_model.imagine(h_attn3, horizon=1, mem_ctx=mem_ctx_tok)
            pc_err_mid = self.world_model.compute_predictive_coding_error(wm_mid, tgt_lat)
            wm_unc_mid = wm_mid["imagined_uncertainty"][:, :, 0]

        expert_states = list(self.lifecycle.expert_states) if self.lifecycle is not None else None
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
        )
        det = self.context_nuance_detector
        nuance_info = (
            det.evaluate_contextual_chunks(
                d_ctx=q_state["d_ctx"],
                primary_experts=q_state["probs"].argmax(dim=-1),
                pc_err=pc_err_mid,
                wm_unc=wm_unc_mid,
                belief_conflict=belief_conflict,
                z_nuance_thresh=self.z_nuance_thresh,
                d_sem_thresh=self.d_sem_nuance_thresh,
                update_stats=update_stats,
            )
            if det is not None
            else {
                "nuance_mask": torch.zeros(B, T, dtype=torch.bool, device=h.device),
                "nuance_score": torch.zeros(B, T, device=h.device),
                "z_ctx": torch.zeros(B, T, device=h.device),
                "d_sem": torch.zeros(B, T, device=h.device),
                "z_belief": torch.zeros(B, T, device=h.device),
            }
        )
        nuance_detect_mask = nuance_info["nuance_mask"] & (
            content_mask if self.use_v8_core else torch.ones_like(content_mask)
        )
        if not getattr(self, "enable_nuance_escalation", True):
            nuance_mask = torch.zeros(B, T, dtype=torch.bool, device=h.device)
        else:
            nuance_mask = nuance_detect_mask
        lifecycle_nuance_mask = nuance_detect_mask if self.decouple_nuance_lifecycle else nuance_mask

        route_res = self.context_router.select_experts(
            q_state,
            nuance_mask=nuance_mask,
            content_mask=(content_mask if self.decouple_nuance_lifecycle else None),
        )
        top_ids = route_res["top_ids"]
        top_w = route_res["top_w"]
        k_used = route_res["k_used"]
        moe_ref = r_layer.moe
        dispatch_plan = build_token_dispatch_plan(top_ids, top_w, num_experts=len(moe_ref.experts))

        # Complete upper dynamic MoE layers reusing ONE cached dispatch_plan
        h, _ = r_layer.forward_moe(h_attn3, update_stats=update_stats, dispatch_plan=dispatch_plan)
        for layer in self.layers[self.router_layer + 1 :]:
            h, _ = layer(
                h,
                update_stats=update_stats,
                memory_context=memory_context,
                dispatch_plan=dispatch_plan,
            )

        # Post-MoE concept-chunk World Model rollout & training loss
        wm_imagine: Optional[Dict[str, torch.Tensor]] = None
        wm_loss_dict: Optional[Dict[str, torch.Tensor]] = None
        pc_err_pre: Optional[torch.Tensor] = None
        if self.enable_world_model and self.world_model is not None and tgt_lat is not None:
            wm_mem = mem_ctx_tok
            if self.use_v8_core:
                cm_f = content_mask.unsqueeze(-1).to(h.dtype)
                cum_h = torch.cumsum(h * cm_f, dim=1) / torch.cumsum(cm_f, dim=1).clamp_min(1.0)
                wm_mem = (0.5 * mem_ctx_tok + 0.5 * cum_h) if mem_ctx_tok is not None else cum_h
            wm_imagine = self.world_model.imagine(
                h,
                horizon=self.world_model.max_rollout_horizon,
                mem_ctx=wm_mem,
            )
            pc_err_pre = self.world_model.compute_predictive_coding_error(wm_imagine, tgt_lat)
            wm_imagine["predictive_coding_error"] = pc_err_pre
            h_wm = 0.2 * h + 0.8 * h.detach()
            wm_loss_dict = self.world_model.compute_world_model_loss(h_wm, tgt_lat, mem_ctx=wm_mem)
            if self.training and update_stats and ((not self.use_v8_core) or (self._v8_step_ctr % 2 == 0)):
                if self.world_model.target_deep_layers is not None:
                    trunk_t = [self.layers[0]] if self.use_v8_core else list(self.layers[: self.router_layer])
                    self.world_model.update_deep_trunk(
                        self.ms_chunker,
                        trunk_t,
                        None if self.use_v8_core else self.expr_head,
                    )
                elif self.ms_chunker is not None:
                    self.world_model.update_target_chunker(self.ms_chunker)
                else:
                    self.world_model.update_target_encoder(self.byte_embed, self.chunk_proj, self.pos_embed)
            if self.wm_imagine_proj is not None and self.wm_imagine_gate is not None:
                z_imag_1 = wm_imagine["imagined_latents"][:, :, 0, :]
                conf_1 = (1.0 - wm_imagine["imagined_uncertainty"][:, :, :1]).detach()
                h = h + torch.tanh(self.wm_imagine_gate) * conf_1 * self.wm_imagine_proj(z_imag_1)

        # Thinker latent refinement escalated by nuance_mask (skip wasted decoder dispatches when v8 energy-halting & detached)
        think_out: Optional[Dict[str, Any]] = None
        if self.enable_thinking and self.thinker is not None:
            skip_dec_in_think = (
                self.use_v8_core
                and self.detach_decoder
                and bool(getattr(self.thinker, "halt_use_energy_only", False))
            )
            decode_fn = (
                None
                if skip_dec_in_think
                else (
                    lambda z_s: self.decoder(
                        z_s.detach() if self.detach_decoder else z_s,
                        dispatch_plan=dispatch_plan,
                    )
                )
            )
            think_out = self.thinker(
                h,
                decoder_fn=decode_fn,
                force_steps=force_thinking_steps,
                nuance_mask=nuance_mask,
                dispatch_plan=dispatch_plan,
            )
            h = think_out["z_final"]

        # v10 Closed-Form Dual Ridge In-Context Operator Readout strictly post-Thinker
        # so contradictory per-episode shifts never corrupt Layers 0..5, WM, or Thinker!
        if (
            self.enable_dual_ridge_ep
            and episode_memory is not None
            and self.enable_memory
            and self.memory is not None
            and episode_memory.get("op_keys") is not None
            and episode_memory.get("op_vals") is not None
        ):
            h_for_op = h.detach() if self.training else h
            op_delta = self.memory.long_term.read_dual_operator(
                h_for_op,
                op_keys=episode_memory["op_keys"],
                op_vals=episode_memory["op_vals"],
                op_mask=episode_memory.get("op_mask"),
            )
            override_logit = getattr(self.memory.long_term, "op_override_logit", None)
            if override_logit is not None:
                g_ep = torch.sigmoid(override_logit).to(h.dtype)
                h = (1.0 - g_ep) * h_for_op + op_delta.to(h.dtype)
            else:
                h = h_for_op + op_delta.to(h.dtype)

        # Post-MoE / Post-Thinker dual meaning factorization & order-sensitive pooling
        z_expr = self.expr_head(h) if self.expr_head is not None else h
        z_ctx = self.ctx_head(h) if self.ctx_head is not None else h
        fsp_logits = self.fsp_head(h) if self.fsp_head is not None else None

        if self.use_v8_core:
            h_emb_pool, _, _ = self._valid_relational_pool(h_embed_seq, mask=content_mask, use_rel=False)
            h_raw_pool = self._valid_relational_pool(h, mask=content_mask, use_rel=False)[0]
            ze_raw_head, _, _ = self._valid_relational_pool(z_expr, mask=content_mask, use_rel=False)
            if pos_chunks_seq is not None and pos_chunks_seq.size(1) == T:
                cm_f = content_mask.unsqueeze(-1).float()
                pos_c_f = pos_chunks_seq.float() * cm_f
                pos_mean = pos_c_f.sum(dim=1) / cm_f.sum(dim=1).clamp_min(1.0)
                pos_max = pos_c_f.max(dim=1).values
                pos_ngram_pool = (0.5 * pos_mean + 0.5 * pos_max).to(dtype=h_raw_pool.dtype)
                # Blend sparse nonnegative multi-scale k-grams cleanly into z_expr WITHOUT polluting Layers 0..5 order sensitivity!
                z_expr_raw_pool = 0.35 * pos_ngram_pool + 0.15 * h_emb_pool + 0.30 * h_raw_pool + 0.20 * ze_raw_head
            else:
                z_expr_raw_pool = 0.35 * h_emb_pool + 0.40 * h_raw_pool + 0.25 * ze_raw_head
            z_expr_pool = F.normalize(z_expr_raw_pool, dim=-1)
            z_expr_contrast = (
                F.normalize(self.expr_proj(z_expr_raw_pool), dim=-1)
                if self.expr_proj is not None
                else z_expr_pool
            )
            z_ctx_raw_pool, z_ctx_pool, rel_dir_vec = self._valid_relational_pool(z_ctx, mask=content_mask, use_rel=True)
            i7_struct_logits = self.i7_struct_head(z_ctx_raw_pool) if self.i7_struct_head is not None else None
        else:
            z_expr_raw_pool = z_expr.mean(dim=1)
            z_expr_pool = F.normalize(z_expr_raw_pool, dim=-1)
            z_expr_contrast = z_expr_pool
            z_ctx_raw_pool = z_ctx.mean(dim=1)
            z_ctx_pool = F.normalize(z_ctx_raw_pool, dim=-1)
            rel_dir_vec = z_ctx_raw_pool - z_ctx[:, 0, :]
            i7_struct_logits = None

        # Optional text renderer runs strictly behind stop_gradient when detach_decoder=True
        logits = self.decoder(h.detach() if self.detach_decoder else h, dispatch_plan=dispatch_plan)
        abstain_logits = self.abstain_head(h) if self.abstain_head is not None else None

        lifecycle_events: List[str] = []
        if update_stats:
            self.context_router.update_keys(route_res["q_tok"], top_ids, top_w, expert_states=expert_states)
            if self.auto_lifecycle and self.lifecycle is not None:
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

        # Modal expert per sequence for routing NMI / switch-rate diagnostics
        if self.decouple_nuance_lifecycle:
            # Compute modal expert strictly over active content chunks (never trailing BYTE_PAD_ID=257!)
            n_exp_cur = len(moe_ref.experts)
            c_active = content_mask & (k_used > 0)
            has_act = c_active.any(dim=1, keepdim=True)
            eff_mask = torch.where(has_act, c_active, content_mask)
            oh_e = F.one_hot(top_ids[..., 0].clamp(0, max(0, n_exp_cur - 1)), num_classes=max(1, n_exp_cur)).float()
            counts_seq = (oh_e * eff_mask.unsqueeze(-1).float()).sum(dim=1)
            seq_modal = counts_seq.argmax(dim=-1)
        else:
            masked_ids = torch.where(k_used > 0, top_ids[..., 0], torch.zeros_like(top_ids[..., 0]))
            seq_modal = torch.mode(masked_ids, dim=1).values
        self.last_assigned_experts = seq_modal
        self.last_routed_expert = int(seq_modal[0].item()) if seq_modal.numel() > 0 else 0
        self.last_lifecycle_events = lifecycle_events

        is_unknown = (
            think_out["is_unknown"]
            if think_out is not None
            else (nuance_info["nuance_score"] > 1.25)
        )
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
            "tgt_lat": tgt_lat.detach() if tgt_lat is not None else None,
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
            "nuance_score": nuance_info["nuance_score"],
            "router_balance_loss": route_res["balance_loss"],
            "router_erc_loss": route_res.get("erc_loss", h.new_tensor(0.0)),
            "assigned_experts": seq_modal,
            "thinking_out": think_out,
            "thinker_z_trace": think_out.get("z_trace") if think_out is not None else None,
            "thinking_steps_used": think_out["steps_used"] if think_out is not None else 0,
            "active_token_fraction": think_out["active_fraction"] if think_out is not None else 0.0,
            "is_unknown": is_unknown,
            "lifecycle_events": lifecycle_events,
            "num_experts": len(moe_ref.experts),
            "expert_states": list(self.lifecycle.expert_states) if self.lifecycle is not None else [],
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
        """
        I5 Causal Abstraction Interchange-Intervention forward pass:
          Runs lower shared trunk on `x_base` and `x_donor`, transplants `x_donor` internal
          subspaces (`dim_slice`) and/or causal chunk positions (`intervene_chunks`) into
          `x_base` mid-forward at `router_layer`, re-routes upper MoE layers & Thinker, and
          exposes counterfactual latent states for IIT training/verification.
        """
        B, _ = x_base.shape
        hb = self.encode(x_base)
        hd = self.encode(x_donor)
        T = hb.size(1)
        pad_mask, content_mask = self._compute_chunk_masks(x_base, T)

        if self.enable_memory and self.memory is not None:
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

        expert_states = list(self.lifecycle.expert_states) if self.lifecycle is not None else None
        q_state = self.context_router.compute_queries(h_int, expert_states=expert_states)
        det = self.context_nuance_detector
        nuance_mask = (
            det.evaluate_contextual_chunks(
                d_ctx=q_state["d_ctx"],
                primary_experts=q_state["probs"].argmax(dim=-1),
                z_nuance_thresh=self.z_nuance_thresh,
                d_sem_thresh=self.d_sem_nuance_thresh,
                update_stats=False,
            )["nuance_mask"]
            if det is not None
            else torch.zeros(B, T, dtype=torch.bool, device=h_int.device)
        )
        route_res = self.context_router.select_experts(q_state, nuance_mask=nuance_mask)
        moe_ref = r_layer.moe
        dispatch_plan = build_token_dispatch_plan(route_res["top_ids"], route_res["top_w"], num_experts=len(moe_ref.experts))
        h, _ = r_layer.forward_moe(h_int, update_stats=False, dispatch_plan=dispatch_plan)
        for layer in self.layers[self.router_layer + 1 :]:
            h, _ = layer(h, update_stats=False, dispatch_plan=dispatch_plan)

        if self.enable_world_model and self.world_model is not None and self.wm_imagine_proj is not None and self.wm_imagine_gate is not None:
            wm_im = self.world_model.imagine(h, horizon=1)
            z_im = wm_im["imagined_latents"][:, :, 0, :]
            conf = (1.0 - wm_im["imagined_uncertainty"][:, :, :1]).detach()
            h = h + torch.tanh(self.wm_imagine_gate) * conf * self.wm_imagine_proj(z_im)

        if self.enable_thinking and self.thinker is not None:
            t_out = self.thinker(
                h,
                decoder_fn=None,
                force_steps=force_thinking_steps,
                nuance_mask=nuance_mask,
                dispatch_plan=dispatch_plan,
            )
            h = t_out["z_final"]

        ze = self.expr_head(h) if self.expr_head is not None else h
        zc = self.ctx_head(h) if self.ctx_head is not None else h
        _, ze_p, _ = self._valid_relational_pool(ze, mask=content_mask, use_rel=False)
        _, zc_p, _ = self._valid_relational_pool(zc, mask=content_mask, use_rel=True)
        return {
            "latent": h,
            "z_expr_pool": ze_p,
            "z_ctx_pool": zc_p,
            "top_ids": route_res["top_ids"],
            "k_used": route_res["k_used"],
        }

    def render_text_detached(
        self,
        out: Dict[str, Any],
    ) -> torch.Tensor:
        """Render decoded byte logits strictly behind `stop_gradient` from converged cognitive state."""
        h_fin = out["latent"].detach()
        dispatch_plan = build_token_dispatch_plan(
            out["top_ids"].detach(),
            out["top_w"].detach(),
            num_experts=int(out.get("num_experts", 1)),
        )
        return self.decoder(h_fin, dispatch_plan=dispatch_plan)

    @staticmethod
    def _compute_future_bigram_summary(
        x: torch.Tensor,
        chunk_size: int = 4,
        num_bins: int = 128,
        horizon_chunks: int = 4,
    ) -> torch.Tensor:
        """Vectorized long-range hashed byte-gram summary P_future[:, t, bins] over chunks t+1..t+H."""
        B, L = x.shape
        T = L // chunk_size
        dev = x.device
        xc = x[:, : T * chunk_size].view(B, T, chunk_size).long()
        hist = torch.full((B, T, num_bins), 1e-4, device=dev, dtype=torch.float32)
        uni_ones = torch.ones((B, T, chunk_size), device=dev, dtype=torch.float32)
        hist.scatter_add_(-1, xc % num_bins, uni_ones)
        if chunk_size >= 2:
            bg = (xc[..., :-1] * 31 + xc[..., 1:]) % num_bins
            bg_ones = torch.ones((B, T, chunk_size - 1), device=dev, dtype=torch.float32)
            hist.scatter_add_(-1, bg, bg_ones)
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
        if self.use_v8_core:
            if ze_pool_override is not None:
                ze_raw = ze_pool_override.float()
                z_ep = F.normalize(ze_raw, dim=-1)
            else:
                ze_raw, z_ep, _ = self._valid_relational_pool(ze.float(), mask=content_mask, use_rel=False)
            if self.expr_proj is not None:
                z_ep_c = F.normalize(self.expr_proj(ze_raw.to(dtype=ze.dtype)).float(), dim=-1)
            else:
                z_ep_c = z_ep
            _, z_cp, _ = self._valid_relational_pool(zc.float(), mask=content_mask, use_rel=True)
            tail_m = content_mask[:, max(0, T // 2) :] if content_mask is not None else None
            if tail_m is not None:
                has_tm = tail_m.any(dim=-1, keepdim=True)
                tail_m = torch.where(has_tm, tail_m, torch.ones_like(tail_m))
            _, z_c_tail, _ = self._valid_relational_pool(zc[:, max(0, T // 2) :].float(), mask=tail_m, use_rel=True)
        else:
            ze_raw = ze.float().mean(dim=1)
            z_ep = F.normalize(ze_raw, dim=-1)
            z_ep_c = z_ep
            z_cp = F.normalize(zc.float().mean(dim=1), dim=-1)
            z_c_tail = F.normalize(zc[:, max(0, T // 2) :].float().mean(dim=1), dim=-1)
        dev = z_ep.device
        zero = z_ep.sum() * 0.0

        if z_ep.size(0) >= 4:
            l_sigreg = (
                sigreg_loss(F.layer_norm(ze_raw, (ze_raw.size(-1),))).to(ze.dtype)
                if self.use_v8_core
                else sigreg_loss(z_ep).to(ze.dtype)
            )
        else:
            l_sigreg = zero

        l_equiv = zero
        if equiv_labels is not None and z_ep.size(0) >= 2:
            y = equiv_labels.to(dev)
            B_e = y.size(0)
            same = (y.unsqueeze(1) == y.unsqueeze(0)).float()
            same.fill_diagonal_(0.0)
            pos_counts = same.sum(dim=-1)
            has_pos = pos_counts > 0.5
            if bool(has_pos.any().item()):
                # Cross-batch equivalence queue (`256` recent labeled vectors) gives dense 77-class contrast!
                if self.use_v8_core and hasattr(self, "equiv_q_lbl"):
                    valid_q = self.equiv_q_lbl >= 0
                    if bool(valid_q.any().item()):
                        q_proj_v = self.equiv_q_proj[valid_q].float().detach()
                        q_raw_v = self.equiv_q_raw[valid_q].float().detach()
                        q_lbl_v = self.equiv_q_lbl[valid_q]
                        all_proj = torch.cat([z_ep_c, q_proj_v], dim=0)
                        all_raw = torch.cat([z_ep, q_raw_v], dim=0)
                        all_lbl = torch.cat([y, q_lbl_v], dim=0)
                    else:
                        all_proj, all_raw, all_lbl = z_ep_c, z_ep, y
                    same_full = (y.unsqueeze(1) == all_lbl.unsqueeze(0)).float()
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
                    n_pos = same.sum()
                    cos_ctx_same = z_cp @ z_cp.t()
                    pos_ctx_l = ((1.0 - cos_ctx_same) * same).sum() / n_pos
                    l_proto = zero
                    if self.equiv_proto_bank is not None:
                        v_pm = y >= 0
                        if bool(v_pm.any().item()):
                            y_mod = (y[v_pm] % self.equiv_proto_bank.num_embeddings).long()
                            p_all = F.normalize(self.equiv_proto_bank.weight.float(), dim=-1)
                            logits_p = (z_ep[v_pm] @ p_all.t()) * 10.0
                            l_proto = F.cross_entropy(logits_p, y_mod)
                    l_equiv = 0.70 * nce_proj + 0.45 * nce_lin + 0.85 * l_proto + 0.30 * pos_ctx_l
                    if self.training and update_queue:
                        v_m = y >= 0
                        if bool(v_m.any().item()):
                            with torch.no_grad():
                                q_sz = self.equiv_q_lbl.size(0)
                                ptr = self._equiv_q_ptr_i % q_sz
                                src_r = z_ep[v_m].detach().to(self.equiv_q_raw.dtype)[:q_sz]
                                src_p = z_ep_c[v_m].detach().to(self.equiv_q_proj.dtype)[:q_sz]
                                src_l = y[v_m].detach()[:q_sz]
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
                else:
                    cos_proj = z_ep_c @ z_ep_c.t()
                    logits_eq = (cos_proj / 0.15).masked_fill(torch.eye(B_e, dtype=torch.bool, device=dev), -1e4)
                    log_prob = logits_eq - torch.logsumexp(logits_eq, dim=-1, keepdim=True)
                    nce_proj = (-(same * log_prob).sum(dim=-1) / pos_counts.clamp_min(1.0))[has_pos].mean()
                    n_pos = same.sum()
                    cos_ctx_same = z_cp @ z_cp.t()
                    pos_ctx_l = ((1.0 - cos_ctx_same) * same).sum() / n_pos
                    l_equiv = nce_proj + 0.35 * pos_ctx_l
        if hard_neg_pair_idx is not None:
            ha, hb = hard_neg_pair_idx[0].to(dev), hard_neg_pair_idx[1].to(dev)
            if ha.numel() > 0:
                hn_cos_p = (z_ep_c[ha] * z_ep_c[hb]).sum(dim=-1)
                if self.use_v8_core and self.expr_proj is not None:
                    l_equiv = l_equiv + 0.85 * F.relu(hn_cos_p + 0.15).pow(2).mean()
                else:
                    hn_cos_r = (z_ep[ha] * z_ep[hb]).sum(dim=-1)
                    l_equiv = l_equiv + 0.65 * F.relu(hn_cos_p + 0.15).pow(2).mean() + 0.30 * F.relu(hn_cos_r + 0.10).pow(2).mean()

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

    def compute_v7_non_token_loss(
        self,
        out: Dict[str, Any],
        *,
        x: Optional[torch.Tensor] = None,
        equiv_labels: Optional[torch.Tensor] = None,
        contra_pair_idx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hard_neg_pair_idx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        nuance_targets: Optional[torch.Tensor] = None,
        teacher_gram: Optional[torch.Tensor] = None,
        gram_mask: Optional[torch.Tensor] = None,
        struct_targets: Optional[torch.Tensor] = None,
        wm_weight: float = 1.0,
        fsp_weight: float = 0.35,
        sigreg_weight: float = 0.25,
        equiv_weight: float = 1.0,
        contra_weight: float = 1.0,
        think_weight: float = 0.35,
        erc_weight: float = 0.20,
        balance_weight: float = 0.5,
        teacher_weight: float = 0.0,
        struct_weight: float = 0.20,
        router_q_perturbed: Optional[torch.Tensor] = None,
        router_q_pert_idx: Optional[torch.Tensor] = None,
        cris_weight: float = 0.35,
        exp_orth_weight: float = 0.15,
    ) -> Dict[str, torch.Tensor]:
        """
        Pure Non-Token `v7`/`v8` Training Loss (`Zero Next-Byte Cross-Entropy`):
          - `L_jepa`: Deep-target WorldModel multi-step prediction + VICReg + future InfoNCE (unpadded windows)
          - `L_fsp`: Content-masked Future Summary Prediction over hashed byte bigrams
          - `L_sigreg`: LeJEPA Epps-Pulley isotropic regularization on `z_expr_pool`
          - `L_equiv`: Same-meaning equivalence (`z_expr` + `z_ctx`) & hard-neg repulsion + linear margin
          - `L_contra`: Order-sensitive Relational Pool + suffix context separation + selective `d_ctx` & `need`
          - `L_think`: Stepwise latent progress & RLP-style reward over `Thinker` steps vs `Think0`
          - `L_erc` / `L_bal`: Expert-Router Coupling & sparsity rate balance
          - `L_struct`: Structural role classification (Dyck/order/flip vs equiv vs raw)
          - `L_gram` (`I9`, strictly optional): Offline teacher Gram similarity structure matching
        """
        ze_fin = out["z_expr"]
        zc_fin = out["z_ctx"]
        z_ep = out["z_expr_pool"]
        c_mask = out.get("chunk_content_mask") if self.use_v8_core else None
        dev = z_ep.device
        T = zc_fin.size(1)
        zero = z_ep.sum() * 0.0

        wm_loss = (
            out["world_model_loss"]["loss"]
            if out.get("world_model_loss") is not None
            else zero
        )

        # Future Summary Prediction (I3), content-masked when v8 active
        l_fsp = zero
        if x is not None and out.get("fsp_logits") is not None and T >= 2:
            fsp_lg = out["fsp_logits"][:, :-1, :].float()
            p_fut = self._compute_future_bigram_summary(
                x, chunk_size=self.chunk_size, num_bins=fsp_lg.size(-1)
            )[:, : fsp_lg.size(1), :]
            log_q = F.log_softmax(fsp_lg, dim=-1)
            ce_fsp = -(p_fut * log_q).sum(dim=-1)
            if c_mask is not None and c_mask.size(1) > fsp_lg.size(1):
                vm_f = c_mask[:, 1 : 1 + fsp_lg.size(1)].float()
                l_fsp = ((ce_fsp * vm_f).sum() / vm_f.sum().clamp_min(1.0)).to(ze_fin.dtype)
            else:
                l_fsp = ce_fsp.mean().to(ze_fin.dtype)
            # Content-masked intra-sequence bag-of-ngram concept preservation on pre-MLP `z_expr_raw_pool`
            # (zero byte-order/next-token CE; preserves linear multi-class probe capacity under InfoNCE)
            if self.use_v8_core and self.fsp_head is not None and out.get("z_expr_raw_pool") is not None:
                nbins = self.fsp_head.out_features
                ids_c = x[:, : T * self.chunk_size].view(x.size(0), T, self.chunk_size).long()
                vm_b = (ids_c != 257).float()
                h_doc = torch.full((x.size(0), nbins), 1e-4, device=dev, dtype=torch.float32)
                h_doc.scatter_add_(-1, (ids_c % nbins).view(x.size(0), -1), vm_b.view(x.size(0), -1))
                if self.chunk_size >= 2:
                    bg_c = (ids_c[..., :-1] * 31 + ids_c[..., 1:]) % nbins
                    vm_bg = ((ids_c[..., :-1] != 257) & (ids_c[..., 1:] != 257)).float()
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
            ze_pool_override=out.get("z_expr_raw_pool") if self.use_v8_core else None,
        )

        # Selective d_ctx margin + reducible-error need calibration + abstain_head supervision
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
                if nuance_targets is not None and "need" in out:
                    need_seq = out["need"].float().mean(dim=-1)
                    nt = nuance_targets.to(dev).float().reshape(-1)
                    if nt.numel() == idx_b.numel() and nt.numel() != need_seq.numel():
                        need_pred = need_seq[idx_b]
                    else:
                        need_pred = need_seq[: nt.numel()]
                    red_tgt = nt
                    if out.get("pre_pc_error") is not None:
                        pc_seq = out["pre_pc_error"].detach().float().mean(dim=-1)
                        if nt.numel() == idx_b.numel():
                            red_tgt = (0.75 * nt + 0.25 * pc_seq[idx_b]).clamp(0.05, 0.95)
                    with torch.amp.autocast(device_type=dev.type, enabled=False):
                        bce = F.binary_cross_entropy(
                            need_pred.float().clamp(1e-4, 1.0 - 1e-4),
                            red_tgt.float(),
                        )
                    l_contra_extra = l_contra_extra + 0.25 * bce
                    clean_mask_b = torch.ones(need_seq.size(0), dtype=torch.bool, device=dev)
                    clean_mask_b[idx_b] = False
                    if bool(clean_mask_b.any().item()):
                        pc_clean = (
                            out["pre_pc_error"].detach().float().mean(dim=-1)[clean_mask_b]
                            if out.get("pre_pc_error") is not None
                            else out["d_ctx"].detach().float().mean(dim=-1)[clean_mask_b]
                        )
                        low_tgt = (0.18 + 0.35 * pc_clean).clamp(0.15, 0.45)
                        with torch.amp.autocast(device_type=dev.type, enabled=False):
                            l_contra_extra = l_contra_extra + 0.15 * F.binary_cross_entropy(
                                need_seq[clean_mask_b].float().clamp(1e-4, 1.0 - 1e-4),
                                low_tgt.float(),
                            )

        # I2 Thinker Latent Stepwise + RLP Progress Loss & energy_head / abstain_head calibration
        l_think = zero
        z_trace = out.get("thinker_z_trace")
        if z_trace is not None and len(z_trace) >= 2 and self.expr_head is not None and self.ctx_head is not None:
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
            rlp_prog = F.relu(fin_sem - base_sem.detach() + 0.03)
            l_think = rlp_prog + (0.25 * torch.stack(step_sems).mean() if step_sems else zero)

        think_out = out.get("thinking_out")
        if think_out is not None and think_out.get("step_energies"):
            e0 = think_out["step_energies"][0].float()
            ef = think_out["step_energies"][-1].float()
            tgt_e = (
                out["pre_pc_error"].detach().float().clamp(0.0, 1.0)[:, : e0.size(1)]
                if out.get("pre_pc_error") is not None
                else out["nuance_score"].detach().float().clamp(0.0, 1.0)[:, : e0.size(1)]
            )
            l_think = l_think + 0.05 * (F.mse_loss(e0, tgt_e) + F.mse_loss(ef, tgt_e * 0.7))

        if out.get("abstain_logits") is not None:
            ab_lg = out["abstain_logits"].float().squeeze(-1)
            ab_tgt = (out["d_ctx"].detach().float() > 0.6).float()
            l_think = l_think + 0.05 * F.binary_cross_entropy_with_logits(ab_lg, ab_tgt)

        # Structural polarity / I7 loss
        l_struct = zero
        s_logits = out.get("i7_struct_logits")
        if s_logits is not None and struct_weight > 0.0:
            if struct_targets is not None:
                l_struct = F.cross_entropy(s_logits.float(), struct_targets.to(dev))
            elif equiv_labels is not None:
                st_t = torch.zeros(s_logits.size(0), dtype=torch.long, device=dev)
                same_m = (equiv_labels.unsqueeze(1) == equiv_labels.unsqueeze(0)).float()
                same_m.fill_diagonal_(0.0)
                st_t = torch.where(same_m.sum(dim=-1) > 0.5, torch.ones_like(st_t), st_t)
                if contra_pair_idx is not None:
                    _, ib_s = contra_pair_idx[0].to(dev), contra_pair_idx[1].to(dev)
                    if ib_s.numel() > 0:
                        if nuance_targets is not None and nuance_targets.numel() == ib_s.numel():
                            flp_s = nuance_targets.to(dev).float() > 0.7
                            st_t[ib_s[flp_s]] = 2
                        else:
                            st_t[ib_s] = 2
                l_struct = F.cross_entropy(s_logits.float(), st_t)

        l_gram = zero
        if teacher_gram is not None and teacher_weight > 0.0 and z_ep.size(0) >= 2:
            z_g_src = out.get("z_expr_contrast", z_ep).float()
            tg = teacher_gram.to(dev).float()
            sg = z_g_src @ z_g_src.t()
            B_g = tg.size(0)
            g_mask = ~torch.eye(B_g, dtype=torch.bool, device=dev)
            if gram_mask is not None:
                g_mask = g_mask & gram_mask.to(device=dev, dtype=torch.bool)
            else:
                if contra_pair_idx is not None:
                    ia_g, ib_g = contra_pair_idx[0].to(dev), contra_pair_idx[1].to(dev)
                    g_mask[ia_g, ib_g] = False
                    g_mask[ib_g, ia_g] = False
                if (not self.use_v8_core) and B_g >= 12:
                    g_mask[8:, :] = False
                    g_mask[:, 8:] = False
            if bool(g_mask.any().item()):
                l_gram = ((sg - tg).pow(2) * g_mask.float()).sum() / g_mask.float().sum().clamp_min(1.0)
                row_has = g_mask.any(dim=-1)
                if bool(row_has.any().item()):
                    t_logits = tg[row_has].masked_fill(~g_mask[row_has], -1e4) / 0.15
                    s_logits_g = sg[row_has].masked_fill(~g_mask[row_has], -1e4) / 0.15
                    l_gram = l_gram + 0.25 * (-(F.softmax(t_logits, dim=-1) * F.log_softmax(s_logits_g, dim=-1)).sum(dim=-1)).mean()

        # Prefix-to-Continuation Concept Retrieval InfoNCE strictly on unpadded real continuations
        l_wm_ret = zero
        if (
            x is not None
            and x.size(1) >= 16
            and self.world_model is not None
            and out.get("world_model_imagine") is not None
            and z_ep.size(0) >= 2
        ):
            wm_imag = out["world_model_imagine"]["imagined_latents"]  # [B, T, H, D]
            half_b = x.size(1) // 2
            t_cut = max(1, (half_b // self.chunk_size) - 1)
            h_k = min(3, wm_imag.size(2))
            if t_cut < wm_imag.size(1):
                if self.use_v8_core:
                    fut_slice = x[:, half_b:]
                    non_pad_frac = (fut_slice != 257).float().mean(dim=-1)
                    non_spc_frac = (fut_slice != 32).float().mean(dim=-1)
                    ret_idx = ((non_pad_frac >= 0.75) & (non_spc_frac >= 0.35)).nonzero(as_tuple=False).squeeze(-1)
                else:
                    ret_idx = torch.arange(x.size(0), device=dev) if z_ep.size(0) >= 4 else torch.zeros(0, dtype=torch.long, device=dev)
                if ret_idx.numel() >= 2:
                    z_fut_in_raw = wm_imag[ret_idx, t_cut, :h_k, :].float().mean(dim=1)
                    if self.use_v8_core and out.get("pos_chunks") is not None and self.wm_fut_mlp is not None:
                        cm_pre = c_mask[:, : t_cut + 1] if c_mask is not None else None
                        pos_pre_all = self._pool_pos_chunks(out["pos_chunks"][:, : t_cut + 1, :], mask=cm_pre)
                        h_pre_seq = out["latent"][:, : t_cut + 1, :].float()
                        if cm_pre is not None:
                            cm_pf = cm_pre.unsqueeze(-1).float()
                            h_pre_mean = (h_pre_seq * cm_pf).sum(dim=1) / cm_pf.sum(dim=1).clamp_min(1.0)
                        else:
                            h_pre_mean = h_pre_seq.mean(dim=1)
                        # Throttle Layers 0..5 backprop cleanly so topic retrieval never overwrites clause-order syntax!
                        z_fut_in = 0.20 * z_fut_in_raw + 0.80 * z_fut_in_raw.detach()
                        h_ret_in = (0.15 * h_pre_mean + 0.85 * h_pre_mean.detach())[ret_idx]
                        pos_ret = pos_pre_all[ret_idx]
                        zp_n = self.project_wm_retrieval(
                            z_fut_in.to(dtype=ze_fin.dtype), h_ctx=h_ret_in, pos_pre=pos_ret
                        ).float()
                        has_neg = contra_pair_idx is not None and contra_pair_idx[1].numel() > 0
                        with torch.no_grad():
                            if has_neg:
                                neg_idx = contra_pair_idx[1].to(dev)
                                both_tgt = self.world_model.encode_target_summary(
                                    torch.cat([x[ret_idx, half_b:], x[neg_idx, half_b:]], dim=0)
                                )
                                n_ret = ret_idx.numel()
                                zt_n, zn_f = both_tgt[:n_ret], both_tgt[n_ret:]
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
                        z_pre_n = F.normalize(pos_ret, dim=-1)
                        sim_s = (z_pre_n @ cands_t.t()) / 0.14
                        l_wm_ret = 0.82 * (0.5 * F.cross_entropy(sim_q, lbl_f) + 0.5 * F.cross_entropy(sim_b.t(), lbl_f)) + 0.18 * F.cross_entropy(sim_s, lbl_f)
                        if self.training and hasattr(self, "wm_ret_q"):
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
                    else:
                        z_pred_fut = self.project_wm_retrieval(z_fut_in_raw.to(dtype=ze_fin.dtype)).float()
                        with torch.no_grad():
                            if self.use_v8_core:
                                zt_n = self.world_model.encode_target_summary(x[ret_idx, half_b:])
                            else:
                                tgt_64 = self.world_model.encode_target(x[ret_idx, half_b:])
                                zt_n = F.normalize(F.layer_norm(tgt_64.float().detach(), (tgt_64.size(-1),)).mean(dim=1), dim=-1)
                        zp_n = F.normalize(z_pred_fut, dim=-1)
                        cands_t = zt_n
                        if self.use_v8_core and contra_pair_idx is not None and contra_pair_idx[1].numel() > 0:
                            neg_idx = contra_pair_idx[1].to(dev)
                            with torch.no_grad():
                                zn_f = self.world_model.encode_target_summary(x[neg_idx, half_b:])
                            cands_t = torch.cat([cands_t, zn_f], dim=0)
                        sim_f = (zp_n @ cands_t.t()) / 0.15
                        lbl_f = torch.arange(ret_idx.numel(), device=dev)
                        sim_inb = (zp_n @ zt_n.t()) / 0.15
                        l_wm_ret = 0.5 * (F.cross_entropy(sim_f, lbl_f) + F.cross_entropy(sim_inb.t(), lbl_f)) + 0.25 * (1.0 - (zp_n * zt_n).sum(dim=-1)).mean()

        l_bal = out.get("router_balance_loss", zero)
        l_erc = out.get("router_erc_loss", zero)

        # v10 Causal Role-Intervention Specialization (CRIS) + Expert Subspace Orthogonality
        l_cris_route = zero
        l_exp_orth = zero
        if self.enable_cris_loss:
            rq_tok = out.get("router_q")
            if rq_tok is not None and rq_tok.size(0) >= 2:
                rq_f = rq_tok.float()
                if c_mask is not None and c_mask.size(1) == rq_f.size(1):
                    w_rq = c_mask.unsqueeze(-1).float()
                    rq_pool = F.normalize((rq_f * w_rq).sum(dim=1) / w_rq.sum(dim=1).clamp_min(1.0), dim=-1)
                else:
                    rq_pool = F.normalize(rq_f.mean(dim=1), dim=-1)

                # 1. Semantic equivalence invariance for router queries
                if equiv_labels is not None:
                    y_eq = equiv_labels.to(dev)
                    same_eq = (y_eq.unsqueeze(1) == y_eq.unsqueeze(0)) & (y_eq.unsqueeze(1) >= 0)
                    same_eq.fill_diagonal_(False)
                    if bool(same_eq.any().item()):
                        cos_rq = rq_pool @ rq_pool.t()
                        l_cris_route = l_cris_route + ((1.0 - cos_rq) * same_eq.float()).sum() / same_eq.float().sum().clamp_min(1.0)

                # 2. Separate distinct-domain / hard-negative / contradiction queries without collapsing raw domains
                if hard_neg_pair_idx is not None:
                    ha_r, hb_r = hard_neg_pair_idx[0].to(dev), hard_neg_pair_idx[1].to(dev)
                    if ha_r.numel() > 0:
                        cos_hn = (rq_pool[ha_r] * rq_pool[hb_r]).sum(dim=-1)
                        l_cris_route = l_cris_route + 0.45 * F.relu(cos_hn - 0.15).pow(2).mean()
                # Repel unrelated sequences across batch (excluding same-meaning pairs) so router maintains sharp spread
                if equiv_labels is not None:
                    y_eq = equiv_labels.to(dev)
                    cos_rq = rq_pool @ rq_pool.t()
                    diff_m = (y_eq.unsqueeze(1) != y_eq.unsqueeze(0))
                    diff_m.fill_diagonal_(False)
                    if bool(diff_m.any().item()):
                        l_cris_route = l_cris_route + 0.30 * (F.relu(cos_rq - 0.35).pow(2) * diff_m.float()).sum() / diff_m.float().sum().clamp_min(1.0)

                # 3. Self-supervised surface formatting perturbation invariance (both query cosine & routing probs!)
                if router_q_perturbed is not None:
                    rq_pert = F.normalize(router_q_perturbed.to(dev).float(), dim=-1)
                    if router_q_pert_idx is not None:
                        rq_ref = rq_pool[router_q_pert_idx.to(dev)]
                    else:
                        rq_ref = rq_pool[: rq_pert.size(0)]
                    if rq_ref.size(0) == rq_pert.size(0) and rq_pert.size(0) > 0:
                        l_cris_route = l_cris_route + 2.0 * (1.0 - (rq_ref * rq_pert).sum(dim=-1)).mean()
                        # Directly align expert assignment distributions under surface formatting perturbation
                        if self.context_router is not None and self.context_router.num_experts > 1:
                            E_cur = self.context_router.num_experts
                            k_e = F.normalize(self.context_router.expert_keys[:E_cur].float(), dim=-1)
                            s_clean = torch.einsum("bd,emd->bem", rq_ref, k_e).max(dim=-1).values * self.context_router.scale
                            s_pert = torch.einsum("bd,emd->bem", rq_pert, k_e).max(dim=-1).values * self.context_router.scale
                            p_clean = F.softmax(s_clean.detach(), dim=-1)
                            log_p_pert = F.log_softmax(s_pert, dim=-1)
                            p_pert_d = F.softmax(s_pert.detach(), dim=-1)
                            log_p_clean = F.log_softmax(s_clean, dim=-1)
                            sym_kl = 0.5 * (
                                (p_clean * (p_clean.clamp_min(1e-6).log() - log_p_pert)).sum(dim=-1)
                                + (p_pert_d * (p_pert_d.clamp_min(1e-6).log() - log_p_clean)).sum(dim=-1)
                            ).mean()
                            l_cris_route = l_cris_route + 0.50 * sym_kl

            # 4. Permutation-invariant expert output subspace orthogonality: <G_e, G_{e'}>^2 where G_e = W2_e W2_e^T / ||.||_F
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
                        g_n = g_e / g_e.norm().clamp_min(1e-6)
                        grams.append(g_n)
                        trainable.append(bool(exp.linear2.weight.requires_grad))
                    for i in range(n_e):
                        for j in range(i + 1, n_e):
                            if trainable[i] or trainable[j]:
                                ov = (grams[i] * grams[j]).sum()
                                orth_terms.append(ov * ov)
            if orth_terms:
                l_exp_orth = torch.stack(orth_terms).mean()

        total = (
            wm_weight * (0.40 * wm_loss + 3.20 * l_wm_ret)
            + fsp_weight * l_fsp
            + sem_tot
            + contra_weight * l_contra_extra
            + think_weight * l_think
            + struct_weight * l_struct
            + erc_weight * l_erc
            + balance_weight * l_bal
            + teacher_weight * l_gram
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
            "gram_loss": l_gram.detach(),
            "cris_route_loss": l_cris_route.detach(),
            "exp_orth_loss": l_exp_orth.detach(),
        }

    compute_v8_non_token_loss = compute_v7_non_token_loss

    def forward(
        self,
        x: torch.Tensor,
        update_stats: bool = True,
        output_head: str = "byte_chunk",
        force_thinking_steps: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        episode_memory: Optional[Dict[str, torch.Tensor]] = None,
        force_expert: Optional[int] = None,
        belief_conflict: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if self.use_v7_core:
            return self._forward_v7(
                x,
                update_stats=update_stats,
                force_thinking_steps=force_thinking_steps,
                episode_memory=episode_memory,
                belief_conflict=belief_conflict,
            )
        B, L = x.shape
        h = self.encode(x)  # [B, n_chunks, d_model]
        h_embed = h

        ep_keys_saved: Optional[torch.Tensor] = None
        ep_vals_saved: Optional[torch.Tensor] = None
        ep_mask_saved: Optional[torch.Tensor] = None
        mem_out: Optional[Dict[str, torch.Tensor]] = None
        memory_context: Optional[torch.Tensor] = None
        if self.enable_memory and self.memory is not None:
            ep_keys_saved = episode_memory.get("ep_keys") if episode_memory is not None else None
            ep_vals_saved = episode_memory.get("ep_vals") if episode_memory is not None else None
            ep_mask_saved = episode_memory.get("ep_mask") if episode_memory is not None else None
            mem_out = self.memory(
                h,
                update_ltm=False,
                ep_keys=ep_keys_saved,
                ep_vals=ep_vals_saved,
                ep_mask=ep_mask_saved,
                return_stm_state=False,
            )
            h = mem_out["fused"]
            ctx_parts = []
            if self.memory.long_term._num_used_slots > 0:
                used_idx = self.memory.long_term.slot_used.nonzero(as_tuple=True)[0]
                if used_idx.numel() > 0:
                    # Strictly causal: global slot attention has no triangular mask, so only
                    # expose prior-episode LTM slots (never future intra-sequence STM states).
                    ctx_parts.append(self.memory.long_term.slot_vals[used_idx].unsqueeze(0).expand(B, -1, -1))
            if ep_vals_saved is not None and ep_vals_saved.size(1) > 0:
                if ep_mask_saved is not None:
                    ctx_parts.append(ep_vals_saved * ep_mask_saved.unsqueeze(-1).to(ep_vals_saved.dtype))
                else:
                    ctx_parts.append(ep_vals_saved)
            if ctx_parts:
                memory_context = torch.cat(ctx_parts, dim=1)

        mem_ctx_tok = mem_out["ltm_out"] + mem_out["stm_out"] if mem_out is not None else None
        tgt_lat: Optional[torch.Tensor] = None
        pc_err_pre: Optional[torch.Tensor] = None
        pc_write_gate: Optional[torch.Tensor] = None
        if self.enable_world_model and self.world_model is not None:
            tgt_lat = self.world_model.encode_target(x)

        # Run Layer 0 contextual trunk (causal STM + causal Attention) before MoE routing
        h_ctx0 = self.layers[0].forward_context(
            h, memory_context=memory_context, write_mask=None
        )

        # Extract sequence routing signatures from contextual trunk latents
        if self.use_trunk_routing:
            seq_sig = F.normalize((0.5 * h_embed + 0.5 * h_ctx0.detach()).mean(dim=1), p=2, dim=-1)
        else:
            seq_sig = F.normalize(h_embed.mean(dim=1), p=2, dim=-1)
        proto_sig: Optional[torch.Tensor] = (
            self.causal_router.extract_signature(
                x,
                self.byte_embed,
                context_h=h_ctx0 if self.use_trunk_routing else None,
                context_blend=self.router_context_blend if self.use_trunk_routing else 0.0,
            )
            if (self.router_mode == "causal_proto" and self.causal_router is not None)
            else None
        )

        # Route sequences
        moe0 = self.layers[0].moe
        force_idx: Optional[int] = None
        is_mixed_batch: bool = False
        if force_expert is not None and 0 <= int(force_expert) < len(moe0.experts):
            fe = int(force_expert)
            is_frozen_fe = bool(moe0._expert_frozen_flags[fe]) if fe < len(moe0._expert_frozen_flags) else False
            unfrozen = [] if is_frozen_fe else [fe]
            chosen_training_expert = fe
            assigned_experts = x.new_full((B,), fe, dtype=torch.long)
            is_novel = x.new_zeros(B, dtype=torch.bool) if proto_sig is not None else False
            self.last_assigned_experts = assigned_experts
            self.last_routed_expert = fe
            force_idx = fe
            for layer in self.layers:
                layer.moe.assigned_experts = assigned_experts
                layer.moe.force_routed_expert = fe
                layer.moe.skip_uniform_check = False
        elif not self.auto_lifecycle and len(moe0.experts) == 1:
            is_frozen0 = moe0._expert_frozen_flags[0]
            unfrozen = [] if is_frozen0 else [0]
            chosen_training_expert = 0
            assigned_experts = x.new_zeros(B, dtype=torch.long)
            is_novel = x.new_ones(B, dtype=torch.bool) if proto_sig is not None else False
            self.last_assigned_experts = assigned_experts
            self.last_routed_expert = 0
            force_idx = 0
            for layer in self.layers:
                layer.moe.assigned_experts = assigned_experts
                layer.moe.force_routed_expert = 0
                layer.moe.skip_uniform_check = False
        else:
            raw_exemplars: List[Optional[torch.Tensor]] = []
            for expert in moe0.experts:
                if hasattr(expert, "exemplars") and expert.exemplars.numel() > 0:
                    raw_exemplars.append(expert.exemplars)
                else:
                    raw_exemplars.append(None)

            dormant_mask: List[bool] = [False] * len(moe0.experts)
            if self.auto_lifecycle and self.lifecycle is not None:
                self.lifecycle.sync_num_experts(len(moe0.experts), moe0.expert_frozen)
                revived_idx: Optional[int] = None
                if proto_sig is not None and self.causal_router is not None:
                    self.causal_router.sync_num_experts(len(moe0.experts))
                    # Only revive via raw exemplars or high similarity when dormant exists
                    if any(s == STATE_DORMANT for s in self.lifecycle.expert_states):
                        revived_idx = self.lifecycle.check_dormant_revival(
                            seq_sig, raw_exemplars, sim_threshold=0.95
                        )
                else:
                    revived_idx = self.lifecycle.check_dormant_revival(seq_sig, raw_exemplars)
                if revived_idx is not None:
                    for layer in self.layers:
                        if revived_idx < layer.moe.expert_age.numel():
                            layer.moe.expert_age[revived_idx] = 0.0
                dormant_mask = [s == STATE_DORMANT for s in self.lifecycle.expert_states]
                expert_exemplars: List[Optional[torch.Tensor]] = [
                    ex if not dormant_mask[i] else None
                    for i, ex in enumerate(raw_exemplars)
                ]
            else:
                expert_exemplars = [ex for ex in raw_exemplars if ex is not None]

            unfrozen = [i for i, f in enumerate(moe0._expert_frozen_flags) if not f]
            fallback = unfrozen[0] if unfrozen else 0

            if proto_sig is not None and self.causal_router is not None:
                self.causal_router.sync_num_experts(len(moe0.experts))
                assigned_experts, max_sims, is_novel = self.causal_router.route_sequences(
                    proto_sig, dormant_mask=dormant_mask, fallback_expert=fallback
                )
            else:
                assigned_experts, max_sims, is_novel = self.sequence_router.route_sequences(
                    seq_sig, expert_exemplars, fallback_expert=fallback
                )

            if self.training and unfrozen:
                chosen_training_expert = unfrozen[0]
                if len(unfrozen) > 1:
                    unfrozen_t = torch.tensor(unfrozen, dtype=torch.long, device=assigned_experts.device)
                    if (
                        self.causal_router is not None
                        and self.causal_router.last_proto_sims is not None
                        and self.causal_router.last_proto_sims.size(0) == B
                        and self.causal_router.last_proto_sims.size(1) > max(unfrozen)
                    ):
                        ps_act = self.causal_router.last_proto_sims[:, unfrozen]
                        active_target = unfrozen_t[ps_act.argmax(dim=-1)]
                        all_empty = ps_act.max(dim=-1).values < -0.5
                        active_target = torch.where(
                            all_empty, torch.full_like(assigned_experts, chosen_training_expert), active_target
                        )
                    else:
                        active_target = torch.where(
                            torch.isin(assigned_experts, unfrozen_t),
                            assigned_experts,
                            torch.full_like(assigned_experts, chosen_training_expert),
                        )
                else:
                    active_target = torch.full_like(assigned_experts, chosen_training_expert)

                if (
                    self.route_during_train
                    and any(moe0._expert_frozen_flags)
                    and proto_sig is not None
                    and not (
                        self.lifecycle is not None
                        and (self.lifecycle.novel_streak > 0 or self.lifecycle.cooldown_remaining > 0)
                    )
                ):
                    is_novel_batch = float(is_novel.float().mean().item()) >= 0.5
                    if is_novel_batch:
                        assigned_experts = active_target
                    elif self.train_revisit_conf_margin > 0.0 and self.causal_router is not None and self.causal_router.last_proto_sims is not None:
                        ps = self.causal_router.last_proto_sims
                        thrs = self.causal_router.last_thr_per_exp
                        allow_frozen = ~is_novel
                        if ps.size(0) == B and ps.size(1) >= len(moe0.experts):
                            asg_c = assigned_experts.clamp(0, ps.size(1) - 1)
                            asg_sim = ps.gather(1, asg_c.unsqueeze(1)).squeeze(1)
                            act_sim = ps.gather(1, active_target.unsqueeze(1)).squeeze(1)
                            has_act = act_sim > -0.5
                            beats_act = (~has_act) | (asg_sim > act_sim + self.train_revisit_conf_margin)
                            allow_frozen = allow_frozen & beats_act
                        assigned_experts = torch.where(allow_frozen, assigned_experts, active_target)
                    else:
                        assigned_experts = torch.where(is_novel, active_target, assigned_experts)
                else:
                    assigned_experts = active_target
                self.last_assigned_experts = assigned_experts
                n_asg = assigned_experts.numel()
                if len(moe0.experts) == 1:
                    force_idx = 0
                    is_mixed_batch = False
                    self.last_routed_expert = 0
                elif n_asg > 0:
                    e0 = int(assigned_experts[0].item())
                    is_unif = bool((assigned_experts == e0).all().item())
                    force_idx = e0 if is_unif else None
                    is_mixed_batch = not is_unif
                    self.last_routed_expert = e0 if is_unif else int(chosen_training_expert)
                else:
                    force_idx = None
                    is_mixed_batch = False
                    self.last_routed_expert = int(chosen_training_expert)
            else:
                chosen_training_expert = fallback
                self.last_assigned_experts = assigned_experts
                n_asg = assigned_experts.numel()
                if len(moe0.experts) == 1:
                    force_idx = 0
                    is_mixed_batch = False
                    self.last_routed_expert = 0
                elif n_asg > 0:
                    e0 = int(assigned_experts[0].item())
                    is_unif = bool((assigned_experts == e0).all().item())
                    force_idx = e0 if is_unif else None
                    is_mixed_batch = not is_unif
                    self.last_routed_expert = e0
                else:
                    force_idx = 0
                    is_mixed_batch = False
                    self.last_routed_expert = 0

            for layer in self.layers:
                layer.moe.assigned_experts = assigned_experts
                layer.moe.force_routed_expert = force_idx
                layer.moe.skip_uniform_check = is_mixed_batch

        # Complete Layer 0 MoE dispatch and remaining layers
        h, _ = self.layers[0].forward_moe(h_ctx0, update_stats=update_stats)
        h_layer0: Optional[torch.Tensor] = h
        for layer in self.layers[1:]:
            h, _ = layer(
                h,
                update_stats=update_stats,
                memory_context=memory_context,
                write_mask=None,
                assigned_experts=self.last_assigned_experts,
                force_expert=force_idx,
                skip_uniform_check=is_mixed_batch,
            )

        wm_imagine: Optional[Dict[str, torch.Tensor]] = None
        wm_loss_dict: Optional[Dict[str, torch.Tensor]] = None
        think_pre: Optional[Dict[str, Any]] = None
        rectified_pass = False
        pass2_gate: Optional[torch.Tensor] = None
        after_first_mask: Optional[torch.Tensor] = None
        pc_needs_key_mask = False
        if self.enable_world_model and self.world_model is not None and self.enable_pc_rectification:
            # Stage 3 active predictive-coding belief rectification:
            # 1) Ponder/decode pass computes prior expectation z_pc_src & epistemic confidence.
            decode_fn_pre = lambda z_state: self.decoder(
                z_state,
                assigned_experts=self.last_assigned_experts,
                force_expert=force_idx,
                skip_uniform_check=is_mixed_batch,
            )
            if self.enable_thinking and self.thinker is not None:
                think_pre = self.thinker(
                    h,
                    decode_fn_pre,
                    force_steps=force_thinking_steps,
                    expert_ids=self.last_assigned_experts,
                    force_expert=force_idx,
                    skip_uniform_check=is_mixed_batch,
                )
                z_pc_src = think_pre["z_final"]
            else:
                z_pc_src = h

            wm_imagine = self.world_model.imagine(
                z_pc_src, horizon=self.world_model.max_rollout_horizon, mem_ctx=mem_ctx_tok
            )
            if tgt_lat is None:
                tgt_lat = self.world_model.encode_target(x)
            pc_err_lat = self.world_model.compute_predictive_coding_error(
                wm_imagine, tgt_lat
            )
            if self.training:
                h_wm = 0.2 * z_pc_src + 0.8 * z_pc_src.detach()
                wm_loss_dict = self.world_model.compute_world_model_loss(
                    h_wm, tgt_lat, mem_ctx=mem_ctx_tok
                )
                if update_stats and not any(moe0._expert_frozen_flags):
                    self.world_model.update_target_encoder(
                        self.byte_embed, self.chunk_proj, self.pos_embed
                    )
            with torch.no_grad():
                wm_unc_h1 = wm_imagine["imagined_uncertainty"][:, :, 0]
                if wm_unc_h1.size(1) >= 2:
                    wm_unc_aligned = torch.cat([wm_unc_h1[:, :1], wm_unc_h1[:, :-1]], dim=1)
                else:
                    wm_unc_aligned = wm_unc_h1
                if think_pre is not None:
                    probs_pre = F.softmax(think_pre["logits_final"], dim=-1)
                    ans_unc_pre = 1.0 - probs_pre.max(dim=-1).values
                    if ans_unc_pre.dim() == 3:
                        ans_unc_pre = ans_unc_pre.mean(dim=-1)
                    if ans_unc_pre.size(1) >= 2:
                        prev_ans_unc = torch.cat([torch.ones_like(ans_unc_pre[:, :1]), ans_unc_pre[:, :-1]], dim=1)
                    else:
                        prev_ans_unc = torch.ones_like(ans_unc_pre)
                    prior_conf = (1.0 - prev_ans_unc).clamp(0.0, 1.0)
                    lat_rel = F.relu(pc_err_lat - 0.35 * wm_unc_aligned)
                    if self.chunk_size == 1 and probs_pre.size(1) >= 2 and probs_pre.size(1) == L:
                        probs_tok = probs_pre.squeeze(-2) if probs_pre.dim() == 4 else probs_pre
                        log_p = torch.log(probs_tok[:, :-1, :].clamp_min(1e-9))
                        ent_prev = -(probs_tok[:, :-1, :] * log_p).sum(dim=-1)
                        max_ent = math.log(float(probs_tok.size(-1)))
                        prec_prev = (1.0 - ent_prev / max(max_ent, 1e-6)).clamp(0.0, 1.0)
                        prec_ent = torch.cat([torch.zeros_like(prec_prev[:, :1]), prec_prev], dim=1)
                        obs_prob = torch.gather(probs_tok[:, :-1, :], -1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
                        nll_prev = (-torch.log(obs_prob.clamp_min(1e-9)) / max(max_ent, 1e-6)).clamp(0.0, 2.0)
                        nll_norm = torch.cat([torch.zeros_like(nll_prev[:, :1]), nll_prev], dim=1)
                        if self.pc_mode == "adaptive_rel":
                            ent_norm_prev = (ent_prev / max(max_ent, 1e-6)).clamp(0.0, 1.0)
                            ent_norm = torch.cat([torch.ones_like(ent_norm_prev[:, :1]), ent_norm_prev], dim=1)
                            prior_gate = F.relu(prec_ent - 0.30) / 0.70
                            nll_base = (ent_norm + 0.10).clamp(min=0.32)
                            nll_surp = F.relu(nll_norm - nll_base)
                            has_surp = (nll_surp > 0.0).float()
                            pc_err_top = (0.40 * lat_rel * has_surp + 0.90 * nll_surp) * prior_gate
                        else:
                            prior_gate = F.relu(prec_ent - 0.62) / 0.38
                            nll_surp = F.relu(nll_norm - 0.60)
                            has_surp = (nll_norm > 0.60).float()
                            pc_err_top = (0.35 * lat_rel * has_surp + 0.65 * nll_surp) * prior_gate
                    else:
                        pc_err_top = lat_rel * prior_conf
                else:
                    pc_err_top = F.relu(pc_err_lat - 0.35 * wm_unc_aligned)
                pc_err_top[:, 0] = 0.0
                wm_imagine["predictive_coding_error"] = pc_err_top
                pc_err_pre = pc_err_top
                pc_excess = F.relu(pc_err_top - self.pc_thresh)
                gate_floor = 1e-3
                gate_lin = (1.0 - self.pc_suppress_gain * pc_excess).clamp(math.sqrt(gate_floor), 1.0)
                pc_write_gate = gate_lin * gate_lin
                pc_needs_key_mask = bool((pc_write_gate < 0.98).any().item())

            rectified_pass = False
            pass2_gate: Optional[torch.Tensor] = None
            after_first_mask: Optional[torch.Tensor] = None
            if (not self.training) and pc_needs_key_mask:
                # Full-path causal rectification: suppress contradicting token(s) and their
                # preceding span (up to pc_span_window tokens) before memory/layer STM and
                # attention mix them forward, splicing Pass 1 for q <= first_trigger and Pass 2 for q > first_trigger.
                h_pass1 = h
                triggered = pc_write_gate < 0.98
                pass2_gate = pc_write_gate.clone()
                trig_gate = torch.where(triggered, pc_write_gate, torch.ones_like(pc_write_gate))
                L_g = pass2_gate.size(1)
                for w in range(1, min(self.pc_span_window, L_g)):
                    pass2_gate[:, : L_g - w] = torch.minimum(pass2_gate[:, : L_g - w], trig_gate[:, w:])
                after_first_mask = (triggered.long().cumsum(dim=1) - triggered.long()) > 0

                rect_scale = (0.05 + 0.95 * pass2_gate).unsqueeze(-1).to(dtype=h_embed.dtype)
                h_rect = h_embed * rect_scale
                if self.enable_memory and self.memory is not None:
                    mem_out_rect = self.memory(
                        h_rect,
                        update_ltm=False,
                        ep_keys=ep_keys_saved,
                        ep_vals=ep_vals_saved,
                        ep_mask=ep_mask_saved,
                        write_mask=pass2_gate,
                        return_stm_state=False,
                    )
                    h_rect = mem_out_rect["fused"]
                for l_idx, layer in enumerate(self.layers):
                    use_asg = self.last_assigned_experts if (l_idx > 0 or len(self.layers) == 1) else None
                    use_fe = force_idx if (l_idx > 0 or len(self.layers) == 1) else None
                    h_rect, _ = layer(
                        h_rect,
                        update_stats=False,
                        memory_context=memory_context,
                        write_mask=pass2_gate,
                        assigned_experts=use_asg,
                        force_expert=use_fe,
                        skip_uniform_check=is_mixed_batch if (l_idx > 0 or len(self.layers) == 1) else False,
                    )
                h = torch.where(after_first_mask.unsqueeze(-1), h_rect, h_pass1)
                rectified_pass = True
            if (not rectified_pass) and self.wm_imagine_proj is not None and self.wm_imagine_gate is not None:
                z_imag_1 = wm_imagine["imagined_latents"][:, :, 0, :]
                conf_1 = (1.0 - wm_imagine["imagined_uncertainty"][:, :, :1]).detach()
                h = h + torch.tanh(self.wm_imagine_gate) * conf_1 * self.wm_imagine_proj(z_imag_1)
        elif self.enable_world_model and self.world_model is not None:
            wm_imagine = self.world_model.imagine(
                h, horizon=self.world_model.max_rollout_horizon, mem_ctx=mem_ctx_tok
            )
            if tgt_lat is None:
                tgt_lat = self.world_model.encode_target(x)
            pc_err_top = self.world_model.compute_predictive_coding_error(
                wm_imagine, tgt_lat
            )
            wm_imagine["predictive_coding_error"] = pc_err_top
            pc_err_pre = pc_err_top
            if self.training:
                h_wm = 0.1 * h + 0.9 * h.detach()
                wm_loss_dict = self.world_model.compute_world_model_loss(
                    h_wm, tgt_lat, mem_ctx=mem_ctx_tok
                )
                if update_stats and not any(moe0._expert_frozen_flags):
                    self.world_model.update_target_encoder(
                        self.byte_embed, self.chunk_proj, self.pos_embed
                    )
            if self.wm_imagine_proj is not None and self.wm_imagine_gate is not None:
                z_imag_1 = wm_imagine["imagined_latents"][:, :, 0, :]
                conf_1 = (1.0 - wm_imagine["imagined_uncertainty"][:, :, :1]).detach()
                h = h + torch.tanh(self.wm_imagine_gate) * conf_1 * self.wm_imagine_proj(z_imag_1)

        if self.enable_thinking and self.thinker is not None:
            decode_fn = lambda z_state: self.decoder(
                z_state,
                assigned_experts=self.last_assigned_experts,
                force_expert=force_idx,
                skip_uniform_check=is_mixed_batch,
            )
            if rectified_pass and pass2_gate is not None and after_first_mask is not None:
                think_rect = self.thinker(
                    h,
                    decode_fn,
                    force_steps=force_thinking_steps,
                    key_mask=pass2_gate,
                    expert_ids=self.last_assigned_experts,
                    force_expert=force_idx,
                    skip_uniform_check=is_mixed_batch,
                )
                if think_pre is not None:
                    m2 = after_first_mask
                    m3 = m2.unsqueeze(-1)
                    m4 = m3.unsqueeze(-1)
                    n_s = max(len(think_pre["step_logits"]), len(think_rect["step_logits"]))
                    def _splice_seq(a_list, b_list, mask):
                        out_l = []
                        for k_i in range(n_s):
                            ta = a_list[min(k_i, len(a_list) - 1)]
                            tb = b_list[min(k_i, len(b_list) - 1)]
                            out_l.append(torch.where(mask, tb, ta))
                        return out_l
                    think_out = {
                        "z_final": torch.where(m3, think_rect["z_final"], think_pre["z_final"]),
                        "logits_final": torch.where(m4, think_rect["logits_final"], think_pre["logits_final"]),
                        "step_logits": _splice_seq(think_pre["step_logits"], think_rect["step_logits"], m4),
                        "step_energies": _splice_seq(think_pre["step_energies"], think_rect["step_energies"], m2),
                        "surprise_trace": _splice_seq(think_pre["surprise_trace"], think_rect["surprise_trace"], m2),
                        "token_steps": torch.where(m2, think_rect["token_steps"], think_pre["token_steps"]),
                        "steps_used": think_rect["steps_used"],
                        "active_fraction": think_rect["active_fraction"],
                        "is_unknown": torch.where(m2, think_rect["is_unknown"], think_pre["is_unknown"]),
                    }
                else:
                    think_out = think_rect
            elif think_pre is not None:
                think_out = think_pre
            elif self.enable_pc_rectification and pc_write_gate is not None and pc_needs_key_mask:
                think_out = self.thinker(
                    h,
                    decode_fn,
                    force_steps=force_thinking_steps,
                    key_mask=pc_write_gate,
                    expert_ids=self.last_assigned_experts,
                    force_expert=force_idx,
                    skip_uniform_check=is_mixed_batch,
                )
            else:
                think_out = self.thinker(
                    h,
                    decode_fn,
                    force_steps=force_thinking_steps,
                    expert_ids=self.last_assigned_experts,
                    force_expert=force_idx,
                    skip_uniform_check=is_mixed_batch,
                )
            tok_surprise = think_out["surprise_trace"][-1]
            lifecycle_events: List[str] = []
            if self.auto_lifecycle and self.lifecycle is not None and update_stats:
                mean_surprise = float(think_out["surprise_trace"][0].mean().item())
                tok_nll: Optional[torch.Tensor] = None
                if targets is not None and targets.shape == (B, h.size(1) * self.chunk_size):
                    logits_f = think_out["logits_final"].detach()
                    tok_nll = F.cross_entropy(
                        logits_f.reshape(-1, logits_f.size(-1)),
                        targets.reshape(-1),
                        reduction="none",
                    ).view(B, h.size(1), self.chunk_size).mean(dim=-1)
                _, lifecycle_events = self.lifecycle.step_lifecycle(
                    self,
                    seq_sig=seq_sig,
                    assigned_experts=assigned_experts,
                    is_novel_input=is_novel,
                    mean_surprise=mean_surprise,
                    token_latents=h,
                    token_surprise=tok_surprise,
                    token_nll=tok_nll,
                    is_training=self.training,
                    proto_sig=proto_sig,
                )
            self._maybe_update_expert_prototypes(
                moe0, chosen_training_expert, assigned_experts, seq_sig, proto_sig, is_novel, lifecycle_events, unfrozen, update_stats
            )
            if (
                self.enable_memory
                and self.memory is not None
                and self.auto_write_ltm
                and self.training
                and update_stats
            ):
                self.memory.long_term.write(h, surprise=tok_surprise)
            self.last_lifecycle_events = lifecycle_events
            return {
                "output": think_out["logits_final"],
                "latent": think_out["z_final"],
                "anchor_latent": h,
                "assigned_experts": assigned_experts,
                "thinking_steps_used": think_out["steps_used"],
                "active_token_fraction": think_out["active_fraction"],
                "token_steps": think_out["token_steps"],
                "is_unknown": think_out["is_unknown"],
                "surprise_trace": think_out["surprise_trace"],
                "step_logits": think_out["step_logits"],
                "thinking_out": think_out,
                "lifecycle_events": lifecycle_events,
                "novelty_verdict": self.lifecycle.last_novelty_verdict if self.lifecycle is not None else None,
                "num_experts": len(self.layers[0].moe.experts),
                "expert_states": list(self.lifecycle.expert_states) if self.lifecycle is not None else [],
                "memory_out": mem_out,
                "level_weights": [l.last_level_weights for l in self.layers if l.last_level_weights is not None],
                "world_model_imagine": wm_imagine,
                "world_model_loss": wm_loss_dict,
                "pre_pc_error": pc_err_pre,
                "pc_write_gate": pc_write_gate,
            }

        logits = self.decoder(
            h,
            assigned_experts=self.last_assigned_experts,
            force_expert=force_idx,
            skip_uniform_check=is_mixed_batch,
        )
        lifecycle_events = []
        if self.auto_lifecycle and self.lifecycle is not None and update_stats:
            probs = F.softmax(logits.detach(), dim=-1)
            log_probs = F.log_softmax(logits.detach(), dim=-1)
            ent = -(probs * log_probs).sum(dim=-1)
            max_ent = math.log(float(logits.size(-1)))
            tok_surprise = (ent / max(max_ent, 1e-6)).clamp(0.0, 1.0)
            if tok_surprise.dim() == 3:
                tok_surprise = tok_surprise.mean(dim=-1)
            mean_surprise = float(tok_surprise.mean().item())
            tok_nll = None
            if targets is not None and targets.shape == (B, h.size(1) * self.chunk_size):
                tok_nll = F.cross_entropy(
                    logits.detach().reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="none",
                ).view(B, h.size(1), self.chunk_size).mean(dim=-1)
            _, lifecycle_events = self.lifecycle.step_lifecycle(
                self,
                seq_sig=seq_sig,
                assigned_experts=assigned_experts,
                is_novel_input=is_novel,
                mean_surprise=mean_surprise,
                token_latents=h,
                token_surprise=tok_surprise,
                token_nll=tok_nll,
                is_training=self.training,
                proto_sig=proto_sig,
            )
        self._maybe_update_expert_prototypes(
            moe0, chosen_training_expert, assigned_experts, seq_sig, proto_sig, is_novel, lifecycle_events, unfrozen, update_stats
        )
        if (
            self.enable_memory
            and self.memory is not None
            and self.auto_write_ltm
            and self.training
            and update_stats
        ):
            probs = F.softmax(logits.detach(), dim=-1)
            log_probs = F.log_softmax(logits.detach(), dim=-1)
            ent = -(probs * log_probs).sum(dim=-1)
            max_ent = math.log(float(logits.size(-1)))
            ent_surp = (ent / max(max_ent, 1e-6)).clamp(0.0, 1.0)
            if ent_surp.dim() == 3:
                ent_surp = ent_surp.mean(dim=-1)
            self.memory.long_term.write(h, surprise=ent_surp)
        self.last_lifecycle_events = lifecycle_events
        return {
            "output": logits,
            "latent": h,
            "assigned_experts": assigned_experts,
            "thinking_steps_used": 0,
            "active_token_fraction": 0.0,
            "lifecycle_events": lifecycle_events,
            "novelty_verdict": self.lifecycle.last_novelty_verdict if self.lifecycle is not None else None,
            "num_experts": len(self.layers[0].moe.experts),
            "expert_states": list(self.lifecycle.expert_states) if self.lifecycle is not None else [],
            "memory_out": mem_out,
            "level_weights": [l.last_level_weights for l in self.layers if l.last_level_weights is not None],
            "world_model_imagine": wm_imagine,
            "world_model_loss": wm_loss_dict,
            "pre_pc_error": pc_err_pre,
            "pc_write_gate": pc_write_gate,
        }

    def compute_total_loss(
        self,
        out: Dict[str, Any],
        targets: torch.Tensor,
        wm_weight: float = 0.25,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified training loss combining:
        - Task / Thinking deep-supervision loss (when thinking is enabled) or cross-entropy
        - Internal Latent World-Model causal prediction + VICReg anti-collapse loss
        """
        if self.enable_thinking and self.thinker is not None and "thinking_out" in out:
            task_dict = self.thinker.compute_thinking_loss(out["thinking_out"], targets)
            task_loss = task_dict["loss"]
        else:
            logits = out["output"]
            task_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        wm_loss = (
            out["world_model_loss"]["loss"]
            if out.get("world_model_loss") is not None
            else (task_loss * 0.0)
        )
        total = task_loss + wm_weight * wm_loss
        return {"loss": total, "task_loss": task_loss.detach(), "wm_loss": wm_loss.detach()}

    def update_sequence_exemplars(self, exemplars_list: List[torch.Tensor]) -> None:
        for exp_idx, sig in enumerate(exemplars_list):
            for layer in self.layers:
                if exp_idx < len(layer.moe.experts):
                    layer.moe.experts[exp_idx].update_exemplars(sig)

    @torch.no_grad()
    def assess_epistemic_uncertainty(
        self,
        out: Dict[str, Any],
        abstain_threshold: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Combine strictly causal epistemic uncertainty signals from a forward output:
        - Predictive softmax uncertainty (1 - max probability)
        - Thinker epistemic surprise (if enabled)
        - World-model calibrated transition uncertainty & causal predictive-coding error (if enabled)
        """
        logits = out["output"]
        probs = F.softmax(logits, dim=-1)
        softmax_unc = (1.0 - probs.max(dim=-1).values).clamp(0.0, 1.0)
        if softmax_unc.dim() == 3:
            softmax_unc = softmax_unc.mean(dim=-1)
        answer_unc = softmax_unc

        result: Dict[str, torch.Tensor] = {"softmax_uncertainty": softmax_unc}

        if out.get("surprise_trace"):
            think_surp = out["surprise_trace"][-1].clamp(0.0, 1.0)
            result["think_surprise"] = think_surp
            answer_unc = torch.maximum(answer_unc, think_surp)

        uncertainty = answer_unc
        wm_im = out.get("world_model_imagine")
        if wm_im is not None:
            wm_unc = wm_im["imagined_uncertainty"][:, :, 0].clamp(0.0, 1.0)
            result["wm_uncertainty"] = wm_unc
            uncertainty = torch.maximum(uncertainty, wm_unc)
            if "predictive_coding_error" in wm_im:
                pc_err = wm_im["predictive_coding_error"].clamp(0.0, 1.0)
                result["pc_error"] = pc_err
                uncertainty = torch.maximum(uncertainty, pc_err)

        thresh = self.abstain_threshold if abstain_threshold is None else float(abstain_threshold)
        result["answer_uncertainty"] = answer_unc
        result["uncertainty"] = uncertainty
        result["is_unknown"] = uncertainty > thresh
        return result
