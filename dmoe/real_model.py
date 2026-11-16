"""
Real-Valued Dynamic Morphic MoE (FP32 Core).
Replaces complex-valued arithmetic with standard real-valued FP32 linear and attention layers:
- Uses native real matrix multiplications (2x to 4x faster).
- Fully parameter-matched.
- Compatible with Evidence-Accumulating Upkeeper and modular per-domain output heads.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.hierarchical_attention import ThreeLevelAttention
from dmoe.lifecycle import AutonomousExpertLifecycleManager, STATE_DORMANT
from dmoe.memory import DualTimescaleMemory, GatedDeltaWorkingMemory
from dmoe.rope import RotaryEmbedding
from dmoe.routing import CausalPrototypeRouter, SequenceContentRouter
from dmoe.thinking import SurpriseDrivenLatentThinker
from dmoe.upkeeper import EvidenceAccumulatingUpkeeper
from dmoe.world_model import LatentWorldModel


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
    """Dynamic Morphic MoE layer with real-valued feed-forward experts."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        initial_experts: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.experts = nn.ModuleList([
            RealExpert(d_model, d_ff) for _ in range(initial_experts)
        ])
        self.register_buffer("expert_usage", torch.ones(initial_experts) / max(1, initial_experts))
        self.register_buffer("expert_age", torch.zeros(initial_experts))
        self.register_buffer("expert_frozen", torch.zeros(initial_experts, dtype=torch.bool))
        self._expert_frozen_flags: List[bool] = [False] * initial_experts

        self.assigned_experts: Optional[torch.Tensor] = None
        self.force_routed_expert: Optional[int] = None
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
        device = self.expert_frozen.device
        if source_idx is None:
            source_idx = int(torch.argmax(self.expert_usage).item()) if len(self.experts) > 0 else 0

        if fresh_init:
            new_expert = RealExpert(self.d_model, self.d_ff).to(device)
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
        if len(self.experts) <= 1 or not (0 <= expert_idx < len(self.experts)):
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
        if 0 <= expert_idx < len(self.experts):
            for p in self.experts[expert_idx].parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                self.expert_frozen[expert_idx] = True
            self._expert_frozen_flags[expert_idx] = True

    def forward(self, x: torch.Tensor, update_stats: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, L, d_model]
        Dispatches sequences to assigned experts.
        """
        B, L, D = x.shape
        if self.assigned_experts is not None and self.assigned_experts.numel() == B and len(self.experts) > 1:
            out = torch.zeros_like(x)
            for exp_idx, expert in enumerate(self.experts):
                mask = (self.assigned_experts == exp_idx).nonzero(as_tuple=True)[0]
                if len(mask) > 0:
                    out[mask] = expert(x[mask])
            return out, self._zero_entropy

        chosen = self.force_routed_expert if self.force_routed_expert is not None else 0
        chosen = min(chosen, len(self.experts) - 1)
        return self.experts[chosen](x), self._zero_entropy


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
        self.moe = RealDynamicMorphicMoE(d_model, d_ff, initial_experts=initial_experts)

    def forward(
        self,
        x: torch.Tensor,
        update_stats: bool = True,
        memory_context: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.layer_stm is not None and self.stm_norm is not None and self.stm_gate is not None:
            h_stm = self.stm_norm(x)
            stm_out, _ = self.layer_stm(h_stm, return_state=False, write_mask=write_mask)
            x = x + torch.sigmoid(self.stm_gate(h_stm)) * stm_out

        # Multi-head attention (3-level hierarchical or flat)
        h = self.norm1(x)
        if self.use_three_level_attn and self.hier_attn is not None:
            attn_out, level_weights = self.hier_attn(h, memory_context=memory_context, key_mask=write_mask)
            self.last_level_weights = level_weights
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
                causal = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
                past_only = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=-1)
                pair_past = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=-2)
                key_gate = write_mask[:, :L].to(device=q.device, dtype=q.dtype).clamp(min=1e-4, max=1.0)
                log_gate = torch.log(key_gate).view(B, 1, 1, L)
                log_next = F.pad(log_gate[..., 1:], (0, 1), value=0.0)
                bias = torch.zeros(B, 1, L, L, device=q.device, dtype=q.dtype)
                bias = bias.masked_fill(~causal.view(1, 1, L, L), -1e9)
                zeros = torch.zeros_like(log_gate)
                bias = bias + torch.where(past_only.view(1, 1, L, L), log_gate, zeros)
                bias = bias + torch.where(pair_past.view(1, 1, L, L), log_next, zeros)
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            else:
                attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal_attn)
            context = attn_out.transpose(1, 2).contiguous().view(B, L, D)
            x = x + self.out_proj(context)

        # MoE block
        moe_out, entropy = self.moe(self.norm2(x), update_stats=update_stats)
        x = x + moe_out
        return x, entropy


class RealDecoder(nn.Module):
    """Modular output head decoder: spawns and freezes dedicated heads per domain."""

    def __init__(self, d_model: int, chunk_size: int = 4, vocab_size: int = 256):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.vocab_size = vocab_size
        self.final_norm = nn.LayerNorm(d_model)
        self.expert_heads = nn.ModuleList([
            nn.Linear(d_model, chunk_size * vocab_size)
        ])

    def spawn_expert_head(self) -> int:
        device = next(self.parameters()).device
        new_head = nn.Linear(self.d_model, self.chunk_size * self.vocab_size).to(device)
        self.expert_heads.append(new_head)
        return len(self.expert_heads) - 1

    def freeze_expert_head(self, idx: int) -> None:
        if 0 <= idx < len(self.expert_heads):
            for p in self.expert_heads[idx].parameters():
                p.requires_grad_(False)

    def prune_expert_head(self, idx: int) -> bool:
        """Cleanly remove a decoder head matching a pruned expert."""
        if len(self.expert_heads) <= 1 or not (0 <= idx < len(self.expert_heads)):
            return False
        del self.expert_heads[idx]
        return True

    def forward(
        self,
        h: torch.Tensor,
        assigned_experts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, seq_chunks, D = h.shape
        h_norm = self.final_norm(h)

        if assigned_experts is not None and len(self.expert_heads) > 1:
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
        route_during_train: bool = False,
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
        self.route_during_train = bool(route_during_train)

        self.byte_embed = nn.Embedding(258, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        self.chunk_proj = nn.Linear(d_model * chunk_size, d_model)
        self.sequence_router = SequenceContentRouter(d_model=d_model, sim_threshold=0.5)
        self.causal_router: Optional[CausalPrototypeRouter] = (
            CausalPrototypeRouter(
                d_model=d_model,
                max_protos=num_router_protos,
                lda_reg=router_lda_reg,
                sim_threshold=router_sim_threshold,
            )
            if self.router_mode == "causal_proto"
            else None
        )

        self.layers = nn.ModuleList([
            RealDynamicMorphicLayer(
                d_model,
                num_heads,
                d_ff,
                initial_experts=initial_experts,
                use_three_level_attn=enable_three_level_attn,
                local_window=local_window,
                chunk_stride=chunk_stride,
                causal_attn=self.causal_attn,
                use_rope=self.use_rope,
                enable_layer_stm=enable_layer_stm,
                signed_beta=signed_beta,
                layer_stm_delta_steps=layer_stm_delta_steps,
            )
            for _ in range(initial_layers)
        ])
        self.decoder = RealDecoder(d_model, chunk_size=chunk_size, vocab_size=256)
        self.upkeeper = EvidenceAccumulatingUpkeeper()

        if enable_memory:
            self.memory: Optional[DualTimescaleMemory] = DualTimescaleMemory(
                d_model=d_model,
                num_heads=num_heads,
                num_ltm_slots=num_ltm_slots,
                signed_beta=signed_beta,
            )
        else:
            self.memory = None

        self.abstain_threshold = float(abstain_threshold)
        if enable_world_model:
            self.world_model: Optional[LatentWorldModel] = LatentWorldModel(
                d_model=d_model,
                d_ff=d_ff,
                max_rollout_horizon=wm_rollout_horizon,
            )
            self.world_model.bind_encoder(self.byte_embed, self.chunk_proj, self.pos_embed, chunk_size)
            self.wm_imagine_proj = nn.Linear(d_model, d_model)
            self.wm_imagine_gate = nn.Parameter(torch.tensor(0.0))
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
            )
        else:
            self.lifecycle = None

        self.last_assigned_experts: Optional[torch.Tensor] = None
        self.last_routed_expert: int = 0
        self.last_lifecycle_events: List[str] = []

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode byte sequences into chunk latents."""
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

    def extract_episode_kv(
        self,
        x_ep: torch.Tensor,
        top_k: int = 8,
        min_nll: float = 0.7,
        cached_out: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Differentiably extract key->value episodic memory candidates from an episode x_ep [B, L]
        by finding high-surprise transitions x[t] -> x[t+1] (chunk_size == 1).
        Uses NO target labels outside x_ep itself.
        """
        if not self.enable_memory or self.memory is None or self.chunk_size != 1 or x_ep.size(1) < 2:
            return None

        B, L = x_ep.shape
        was_training = self.training
        # Reuse episode forward output when provided, else run cleanly without touching stats
        out = cached_out if cached_out is not None else self.forward(x_ep, update_stats=False)
        mem_out = out.get("memory_out")
        if mem_out is None:
            return None

        logits = out["output"].reshape(B, L, -1)
        V = logits.size(-1)
        with torch.no_grad():
            nll = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, V),
                x_ep[:, 1:].reshape(-1),
                reduction="none",
            ).view(B, L - 1)
            k = min(max(1, int(top_k)), L - 1)
            top_vals, top_idx = torch.topk(nll, k=k, dim=-1)
            ep_mask = top_vals >= float(min_nll)
            # Normalized surprise score for persistent consolidation salience
            max_ent = math.log(float(V))
            salience = (top_vals / max(max_ent, 1e-6)).clamp(0.0, 1.0)
            if self.enable_pc_rectification and out.get("world_model_imagine") is not None:
                pc_err_full = out["world_model_imagine"].get("predictive_coding_error")
                if pc_err_full is not None and pc_err_full.size(1) >= L:
                    pc_next = torch.gather(pc_err_full[:, 1:].to(salience.device), 1, top_idx)
                    salience = salience * (1.0 - 0.85 * pc_next).clamp(0.05, 1.0)
                    ep_mask = ep_mask & (pc_next < (self.pc_thresh + 0.20))

        ltm_q = mem_out["ltm_q"][:, :-1, :]
        x_norm_next = mem_out["x_norm"][:, 1:, :]
        D = ltm_q.size(-1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, D)
        raw_k = torch.gather(ltm_q, 1, gather_idx)
        raw_v = torch.gather(x_norm_next, 1, gather_idx)
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
            if defer_proto or any(ev.startswith("spawned_expert_") for ev in lifecycle_events):
                return
        active_mask = assigned_experts == cur_idx
        if not moe0._expert_frozen_flags[cur_idx] and bool(active_mask.any().item()):
            active_seq = seq_sig[active_mask]
            for layer in self.layers:
                layer.moe.experts[cur_idx].update_exemplars(active_seq)
            if proto_sig is not None and self.causal_router is not None:
                self.causal_router.update_prototypes(cur_idx, proto_sig[active_mask])

    def forward(
        self,
        x: torch.Tensor,
        update_stats: bool = True,
        output_head: str = "byte_chunk",
        force_thinking_steps: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        episode_memory: Optional[Dict[str, torch.Tensor]] = None,
        force_expert: Optional[int] = None,
    ) -> Dict[str, Any]:
        B, L = x.shape
        h = self.encode(x)  # [B, n_chunks, d_model]

        # Extract sequence routing signatures
        seq_sig = F.normalize(h.mean(dim=1), p=2, dim=-1)
        proto_sig: Optional[torch.Tensor] = (
            self.causal_router.extract_signature(x, self.byte_embed)
            if (self.router_mode == "causal_proto" and self.causal_router is not None)
            else None
        )

        # Route sequences
        moe0 = self.layers[0].moe
        if force_expert is not None and 0 <= int(force_expert) < len(moe0.experts):
            fe = int(force_expert)
            is_frozen_fe = bool(moe0._expert_frozen_flags[fe]) if fe < len(moe0._expert_frozen_flags) else False
            unfrozen = [] if is_frozen_fe else [fe]
            chosen_training_expert = fe
            assigned_experts = x.new_full((B,), fe, dtype=torch.long)
            is_novel = x.new_zeros(B, dtype=torch.bool) if proto_sig is not None else False
            self.last_assigned_experts = assigned_experts
            self.last_routed_expert = fe
            for layer in self.layers:
                layer.moe.assigned_experts = assigned_experts
                layer.moe.force_routed_expert = fe
        elif not self.auto_lifecycle and len(moe0.experts) == 1:
            is_frozen0 = moe0._expert_frozen_flags[0]
            unfrozen = [] if is_frozen0 else [0]
            chosen_training_expert = 0
            assigned_experts = x.new_zeros(B, dtype=torch.long)
            is_novel = x.new_ones(B, dtype=torch.bool) if proto_sig is not None else False
            self.last_assigned_experts = assigned_experts
            self.last_routed_expert = 0
            for layer in self.layers:
                layer.moe.assigned_experts = assigned_experts
                layer.moe.force_routed_expert = 0
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
                        assigned_experts = torch.full_like(assigned_experts, chosen_training_expert)
                    else:
                        assigned_experts = torch.where(is_novel, chosen_training_expert, assigned_experts)
                else:
                    assigned_experts = torch.full_like(assigned_experts, chosen_training_expert)
                self.last_assigned_experts = assigned_experts
                self.last_routed_expert = (
                    int(assigned_experts[0].item())
                    if bool((assigned_experts == assigned_experts[0]).all().item())
                    else int(chosen_training_expert)
                )
                force_idx: Optional[int] = (
                    int(assigned_experts[0].item())
                    if bool((assigned_experts == assigned_experts[0]).all().item())
                    else None
                )
            else:
                chosen_training_expert = fallback
                self.last_assigned_experts = assigned_experts
                self.last_routed_expert = int(assigned_experts[0].item()) if assigned_experts.numel() > 0 else 0
                force_idx = (
                    self.last_routed_expert
                    if (len(moe0.experts) == 1 or bool((assigned_experts == assigned_experts[0]).all().item()))
                    else None
                )

            for layer in self.layers:
                layer.moe.assigned_experts = assigned_experts
                layer.moe.force_routed_expert = force_idx

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
            )
            h = mem_out["fused"]
            ctx_parts = []
            if self.memory.long_term._num_used_slots > 0:
                used_idx = self.memory.long_term.slot_used.nonzero(as_tuple=True)[0]
                if used_idx.numel() > 0:
                    # Strictly causal: global slot attention has no triangular mask, so only
                    # expose prior-episode LTM slots (never future intra-sequence STM states).
                    ctx_parts.append(self.memory.long_term.slot_vals[used_idx].unsqueeze(0).expand(B, -1, -1))
        if self.enable_memory and self.memory is not None:
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

        h_layer0: Optional[torch.Tensor] = None
        for layer_idx, layer in enumerate(self.layers):
            h, _ = layer(
                h,
                update_stats=update_stats,
                memory_context=memory_context,
                write_mask=None,
            )
            if layer_idx == 0:
                h_layer0 = h

        wm_imagine: Optional[Dict[str, torch.Tensor]] = None
        wm_loss_dict: Optional[Dict[str, torch.Tensor]] = None
        think_pre: Optional[Dict[str, Any]] = None
        rectified_pass = False
        pass2_gate: Optional[torch.Tensor] = None
        after_first_mask: Optional[torch.Tensor] = None
        if self.enable_world_model and self.world_model is not None and self.enable_pc_rectification:
            # Stage 3 active predictive-coding belief rectification:
            # 1) Ponder/decode pass computes prior expectation z_pc_src & epistemic confidence.
            decode_fn_pre = lambda z_state: self.decoder(z_state, assigned_experts=self.last_assigned_experts)
            if self.enable_thinking and self.thinker is not None:
                think_pre = self.thinker(h, decode_fn_pre, force_steps=force_thinking_steps)
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
                if update_stats:
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

            rectified_pass = False
            pass2_gate: Optional[torch.Tensor] = None
            after_first_mask: Optional[torch.Tensor] = None
            if (not self.training) and bool((pc_write_gate < 0.98).any().item()):
                # Full-path causal rectification: suppress contradicting token and its immediately
                # preceding antecedent slot before memory/layer STM and attention mix them forward,
                # splicing Pass 1 for q <= first_trigger and Pass 2 for q > first_trigger.
                h_pass1 = h
                triggered = pc_write_gate < 0.98
                has_trig = triggered.any(dim=1)
                first_s = triggered.long().argmax(dim=1)
                pass2_gate = pc_write_gate.clone()
                b_idx = torch.arange(B, device=h.device)[has_trig]
                s1_idx = first_s[has_trig]
                valid_prev = s1_idx > 0
                if bool(valid_prev.any().item()):
                    bp = b_idx[valid_prev]
                    sp = s1_idx[valid_prev]
                    pass2_gate[bp, sp - 1] = torch.minimum(pass2_gate[bp, sp - 1], pc_write_gate[bp, sp])
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
                    )
                    h_rect = mem_out_rect["fused"]
                for layer in self.layers:
                    h_rect, _ = layer(
                        h_rect,
                        update_stats=False,
                        memory_context=memory_context,
                        write_mask=pass2_gate,
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
                if update_stats:
                    self.world_model.update_target_encoder(
                        self.byte_embed, self.chunk_proj, self.pos_embed
                    )
            if self.wm_imagine_proj is not None and self.wm_imagine_gate is not None:
                z_imag_1 = wm_imagine["imagined_latents"][:, :, 0, :]
                conf_1 = (1.0 - wm_imagine["imagined_uncertainty"][:, :, :1]).detach()
                h = h + torch.tanh(self.wm_imagine_gate) * conf_1 * self.wm_imagine_proj(z_imag_1)

        if self.enable_thinking and self.thinker is not None:
            decode_fn = lambda z_state: self.decoder(z_state, assigned_experts=self.last_assigned_experts)
            if rectified_pass and pass2_gate is not None and after_first_mask is not None:
                think_rect = self.thinker(h, decode_fn, force_steps=force_thinking_steps, key_mask=pass2_gate)
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
            elif self.enable_pc_rectification and pc_write_gate is not None and bool((pc_write_gate < 0.98).any().item()):
                think_out = self.thinker(h, decode_fn, force_steps=force_thinking_steps, key_mask=pc_write_gate)
            else:
                think_out = self.thinker(h, decode_fn, force_steps=force_thinking_steps)
            tok_surprise = think_out["surprise_trace"][-1]
            mean_surprise = float(think_out["surprise_trace"][0].mean().item())
            lifecycle_events: List[str] = []
            if self.auto_lifecycle and self.lifecycle is not None and update_stats:
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

        logits = self.decoder(h, assigned_experts=self.last_assigned_experts)
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
