import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class UniversalProjectiveDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int = 32000,
        continuous_dim: int = 8,
        tokenizer_free: bool = False,
        chunk_size: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.tokenizer_free = tokenizer_free
        self.chunk_size = chunk_size

        self.proj_real = nn.Linear(d_model, d_model)
        self.proj_imag = nn.Linear(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)

        if self.tokenizer_free:
            self.token_head = None
            self.expert_heads = nn.ModuleList([nn.Linear(d_model, chunk_size * 256)])
            self.byte_chunk_head = self.expert_heads[0]
            # Deterministic intra-chunk de-chunker queries (zero RNG consumed on init)
            slot_idx = torch.arange(chunk_size, dtype=torch.float32).unsqueeze(-1)
            dim_idx = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
            angles = slot_idx * (2.0 * math.pi / max(chunk_size, 1)) + dim_idx * (math.pi / max(d_model, 1))
            self.register_buffer("dechunk_slot_basis", torch.cos(angles).unsqueeze(0).unsqueeze(0))
            self.dechunk_slot_gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.expert_heads = nn.ModuleList([nn.Linear(d_model, vocab_size)])
            self.token_head = self.expert_heads[0]
            self.byte_chunk_head = None
        self.continuous_head = nn.Linear(d_model, continuous_dim)

    def spawn_expert_head(self) -> int:
        """Spawn a new modular output classification head for a new expert."""
        device = self.proj_real.weight.device
        if self.tokenizer_free:
            new_head = nn.Linear(self.d_model, self.chunk_size * 256).to(device)
        else:
            new_head = nn.Linear(self.d_model, self.token_head.out_features).to(device)
        self.expert_heads.append(new_head)
        return len(self.expert_heads) - 1

    def freeze_expert_head(self, idx: int) -> None:
        """Freeze output classification head for a mature/completed expert."""
        if 0 <= idx < len(self.expert_heads):
            for p in self.expert_heads[idx].parameters():
                p.requires_grad_(False)

    def extract_slot_features(
        self, unified_hidden: torch.Tensor, byte_residuals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Expands chunk-level latent [B, S_chunks, D] into per-byte slot latents
        [B, S_chunks, chunk_size, D] for the Byte De-Chunker and Fast Memory.
        """
        if not self.tokenizer_free:
            return unified_hidden.unsqueeze(2)
        base_slots = unified_hidden.unsqueeze(2) + self.dechunk_slot_basis * (
            0.25 + torch.tanh(self.dechunk_slot_gate)
        )
        if byte_residuals is not None and byte_residuals.shape == base_slots.shape:
            base_slots = torch.cat([base_slots, byte_residuals], dim=-1)
        return base_slots

    def _project_from_latent(
        self,
        unified_hidden: torch.Tensor,
        head: str = "token",
        byte_residuals: Optional[torch.Tensor] = None,
        assigned_experts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if head == "byte_chunk" or (head == "token" and self.tokenizer_free):
            if self.byte_chunk_head is None and len(self.expert_heads) == 0:
                raise ValueError("byte_chunk head requires tokenizer_free=True on UniversalProjectiveDecoder")
            bsz, seq_len, _ = unified_hidden.shape
            if assigned_experts is not None and len(self.expert_heads) > 1:
                logits = torch.zeros(bsz, seq_len, self.chunk_size, 256, device=unified_hidden.device)
                handled = torch.zeros(bsz, dtype=torch.bool, device=unified_hidden.device)
                for exp_idx, exp_head in enumerate(self.expert_heads):
                    mask = (assigned_experts == exp_idx).nonzero(as_tuple=True)[0]
                    if len(mask) > 0:
                        handled[mask] = True
                        exp_out = exp_head(unified_hidden[mask]).view(len(mask), seq_len, self.chunk_size, 256)
                        if byte_residuals is not None and torch.abs(self.dechunk_slot_gate).item() > 0.0:
                            slot_mod = (byte_residuals[mask] * self.dechunk_slot_basis).mean(dim=-1, keepdim=True)
                            exp_out = exp_out + torch.tanh(self.dechunk_slot_gate) * slot_mod
                        logits[mask] = exp_out
                unhandled = (~handled).nonzero(as_tuple=True)[0]
                if len(unhandled) > 0:
                    fb_out = self.expert_heads[0](unified_hidden[unhandled]).view(len(unhandled), seq_len, self.chunk_size, 256)
                    if byte_residuals is not None and torch.abs(self.dechunk_slot_gate).item() > 0.0:
                        slot_mod = (byte_residuals[unhandled] * self.dechunk_slot_basis).mean(dim=-1, keepdim=True)
                        fb_out = fb_out + torch.tanh(self.dechunk_slot_gate) * slot_mod
                    logits[unhandled] = fb_out
                return logits
            else:
                active_head = self.expert_heads[-1] if len(self.expert_heads) > 0 else self.byte_chunk_head
                logits = active_head(unified_hidden).view(bsz, seq_len, self.chunk_size, 256)
                if byte_residuals is not None and torch.abs(self.dechunk_slot_gate).item() > 0.0:
                    slot_mod = (byte_residuals * self.dechunk_slot_basis).mean(dim=-1, keepdim=True)
                    logits = logits + torch.tanh(self.dechunk_slot_gate) * slot_mod
                return logits
        elif head == "token":
            if assigned_experts is not None and len(self.expert_heads) > 1:
                bsz = unified_hidden.size(0)
                logits = torch.zeros(bsz, unified_hidden.size(1), self.token_head.out_features, device=unified_hidden.device)
                handled = torch.zeros(bsz, dtype=torch.bool, device=unified_hidden.device)
                for exp_idx, exp_head in enumerate(self.expert_heads):
                    mask = (assigned_experts == exp_idx).nonzero(as_tuple=True)[0]
                    if len(mask) > 0:
                        handled[mask] = True
                        logits[mask] = exp_head(unified_hidden[mask])
                unhandled = (~handled).nonzero(as_tuple=True)[0]
                if len(unhandled) > 0:
                    logits[unhandled] = self.expert_heads[0](unified_hidden[unhandled])
                return logits
            active_head = self.expert_heads[-1] if len(self.expert_heads) > 0 else self.token_head
            return active_head(unified_hidden)
        elif head == "continuous":
            return self.continuous_head(unified_hidden)
        elif head == "latent":
            return unified_hidden
        else:
            raise ValueError(f"Unknown output head specification: {head}")

    def forward_with_latent(
        self,
        z_r: torch.Tensor,
        z_i: torch.Tensor,
        head: str = "token",
        byte_residuals: Optional[torch.Tensor] = None,
        assigned_experts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        unified_hidden = self.final_norm(self.proj_real(z_r) + self.proj_imag(z_i))
        return (
            self._project_from_latent(
                unified_hidden, head=head, byte_residuals=byte_residuals, assigned_experts=assigned_experts
            ),
            unified_hidden,
        )

    def forward(
        self,
        z_r: torch.Tensor,
        z_i: torch.Tensor,
        head: str = "token",
        byte_residuals: Optional[torch.Tensor] = None,
        assigned_experts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output, _ = self.forward_with_latent(
            z_r, z_i, head=head, byte_residuals=byte_residuals, assigned_experts=assigned_experts
        )
        return output
