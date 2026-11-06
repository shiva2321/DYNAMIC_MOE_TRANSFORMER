"""
Real-Valued Dynamic Morphic MoE (FP32 Core).
Replaces complex-valued arithmetic with standard real-valued FP32 linear and attention layers:
- Uses native real matrix multiplications (2x to 4x faster).
- Fully parameter-matched.
- Compatible with Evidence-Accumulating Upkeeper and modular per-domain output heads.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.routing import SequenceContentRouter
from dmoe.upkeeper import EvidenceAccumulatingUpkeeper


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

        self.assigned_experts: Optional[torch.Tensor] = None
        self.force_routed_expert: Optional[int] = None

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def spawn_expert(self, source_idx: Optional[int] = None) -> int:
        device = self.expert_frozen.device
        if source_idx is None:
            source_idx = int(torch.argmax(self.expert_usage).item()) if len(self.experts) > 0 else 0

        new_expert = self.experts[source_idx].clone_with_mutation(mutation_std=0.015).to(device)
        new_expert.exemplars = torch.zeros(0, self.d_model, dtype=torch.float32, device=device)
        self.experts.append(new_expert)
        new_idx = len(self.experts) - 1

        self.expert_usage = torch.cat([self.expert_usage, torch.tensor([0.1], device=device)])
        self.expert_age = torch.cat([self.expert_age, torch.zeros(1, device=device)])
        self.expert_frozen = torch.cat([self.expert_frozen, torch.zeros(1, dtype=torch.bool, device=device)])
        return new_idx

    def freeze_expert(self, expert_idx: int) -> None:
        if 0 <= expert_idx < len(self.experts):
            for p in self.experts[expert_idx].parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                self.expert_frozen[expert_idx] = True

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
            return out, torch.tensor(0.0, device=x.device)

        chosen = self.force_routed_expert if self.force_routed_expert is not None else 0
        chosen = min(chosen, len(self.experts) - 1)
        return self.experts[chosen](x), torch.tensor(0.0, device=x.device)


class RealDynamicMorphicLayer(nn.Module):
    """Layer combining Real Multi-Head Attention and Real Dynamic MoE."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, initial_experts: int = 1):
        super().__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.moe = RealDynamicMorphicMoE(d_model, d_ff, initial_experts=initial_experts)

    def forward(self, x: torch.Tensor, update_stats: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # Multi-head attention
        h = self.norm1(x)
        B, L, D = h.shape
        head_dim = D // self.num_heads

        q = self.q_proj(h).view(B, L, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.num_heads, head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
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
    Complete Real-Valued Dynamic Morphic MoE (FP32).
    """

    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 2,
        d_ff: int = 128,
        initial_layers: int = 2,
        initial_experts: int = 1,
        chunk_size: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.chunk_size = chunk_size

        self.byte_embed = nn.Embedding(258, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        self.chunk_proj = nn.Linear(d_model * chunk_size, d_model)
        self.sequence_router = SequenceContentRouter(d_model=d_model, sim_threshold=0.5)

        self.layers = nn.ModuleList([
            RealDynamicMorphicLayer(d_model, num_heads, d_ff, initial_experts=initial_experts)
            for _ in range(initial_layers)
        ])
        self.decoder = RealDecoder(d_model, chunk_size=chunk_size, vocab_size=256)
        self.upkeeper = EvidenceAccumulatingUpkeeper()

        self.last_assigned_experts: Optional[torch.Tensor] = None
        self.last_routed_expert: int = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode byte sequences into chunk latents."""
        B, L = x.shape
        embedded = self.byte_embed(x)  # [B, L, d_model]
        n_chunks = L // self.chunk_size
        chunked = embedded.view(B, n_chunks, self.chunk_size * self.d_model)
        h = self.chunk_proj(chunked)  # [B, n_chunks, d_model]
        if n_chunks <= self.pos_embed.size(1):
            h = h + self.pos_embed[:, :n_chunks, :]
        return h

    def forward(
        self,
        x: torch.Tensor,
        update_stats: bool = True,
        output_head: str = "byte_chunk",
    ) -> Dict[str, Any]:
        B, L = x.shape
        h = self.encode(x)  # [B, n_chunks, d_model]

        # Extract sequence routing signatures
        seq_sig = F.normalize(h.mean(dim=1), p=2, dim=-1)

        # Route sequences
        expert_exemplars = []
        moe0 = self.layers[0].moe
        for expert in moe0.experts:
            if hasattr(expert, "exemplars") and expert.exemplars.numel() > 0:
                expert_exemplars.append(expert.exemplars)

        unfrozen = [i for i, f in enumerate(moe0.expert_frozen) if not bool(f)]
        fallback = unfrozen[0] if unfrozen else 0

        assigned_experts, max_sims, is_novel = self.sequence_router.route_sequences(
            seq_sig, expert_exemplars, fallback_expert=fallback
        )

        if self.training and unfrozen:
            chosen_training_expert = unfrozen[0]
            assigned_experts = torch.full_like(assigned_experts, chosen_training_expert)
            if update_stats and not bool(moe0.expert_frozen[chosen_training_expert]):
                for layer in self.layers:
                    layer.moe.experts[chosen_training_expert].update_exemplars(seq_sig)

        self.last_assigned_experts = assigned_experts
        self.last_routed_expert = int(assigned_experts[0].item()) if assigned_experts.numel() > 0 else 0

        for layer in self.layers:
            layer.moe.assigned_experts = assigned_experts
            layer.moe.force_routed_expert = (
                self.last_routed_expert if (assigned_experts == assigned_experts[0]).all() else None
            )

        for layer in self.layers:
            h, _ = layer(h, update_stats=update_stats)

        logits = self.decoder(h, assigned_experts=self.last_assigned_experts)
        return {
            "output": logits,
            "latent": h,
            "assigned_experts": assigned_experts,
        }

    def update_sequence_exemplars(self, exemplars_list: List[torch.Tensor]) -> None:
        for exp_idx, sig in enumerate(exemplars_list):
            for layer in self.layers:
                if exp_idx < len(layer.moe.experts):
                    layer.moe.experts[exp_idx].update_exemplars(sig)
