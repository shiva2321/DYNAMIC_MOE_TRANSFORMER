"""
Native Spherical Dynamic MoE Transformer.
A sovereign General Intelligence language architecture built natively on
S^7 x R+ Riemannian Geometry, DeepSeek-Style Shared Expert Isolation,
Multi-Token Prediction (MTP), and Rectified Born Rule Output Decoding.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from ishmn.geometry import project_to_sphere, geodesic_distance_sphere
from ishmn.spherical_attention import DualStreamRiemannianAttention
from ishmn.spherical_dynamic_moe import SphericalDynamicMoE
from ishmn.vsa_memory import VSAEpisodicMemory
from ishmn.born_rule import rectified_born


class NativeSphericalBlock(nn.Module):
    """
    Core computational block combining:
    1. Dual-Stream Riemannian Attention (Phase stream on S^{D-1} + Magnitude stream with SwiGLU).
    2. Spherical Dynamic MoE with Shared Expert Isolation & Aux-Loss-Free Dynamic Balancing.
    """
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        hyperspace_dim: int = 8,
        d_ff: int = 256,
        num_experts: int = 4,
        top_k: int = 2,
        max_experts: Optional[int] = None,
        enable_shared_expert: bool = True,
        enable_aux_loss_free_balancing: bool = True,
        enable_gwt_coalition: bool = True,
        use_drift_guard: bool = False,
        drift_sigma: float = 3.0,
        orthogonalize_on_spawn: bool = False,
        spawn_threshold: float = 0.0,
        device: torch.device = None
    ):
        super().__init__()
        self.d_model = d_model
        self.hyperspace_dim = hyperspace_dim

        # 1. Dual-Stream Riemannian Attention
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.attn = DualStreamRiemannianAttention(
            d_model=d_model,
            num_heads=num_heads,
            hyperspace_dim=hyperspace_dim,
            d_phase=num_heads * hyperspace_dim,
            d_mag=d_model,
            device=device
        )

        # 2. Coordinate Projectors for Spherical MoE routing
        self.to_u = nn.Linear(d_model, hyperspace_dim, bias=False, device=device)
        self.to_r = nn.Linear(d_model, 1, bias=False, device=device)

        # 3. Spherical Dynamic MoE with Shared Expert Isolation & GWT Coalition Bidding
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.moe = SphericalDynamicMoE(
            d_model=d_model,
            d_ff=d_ff,
            d_context=d_model,
            hyperspace_dim=hyperspace_dim,
            top_k=top_k,
            initial_experts=num_experts,
            max_experts=max_experts if max_experts is not None else num_experts,
            spawn_threshold=spawn_threshold,
            enable_shared_expert=enable_shared_expert,
            enable_aux_loss_free_balancing=enable_aux_loss_free_balancing,
            enable_gwt_coalition=enable_gwt_coalition,
            use_drift_guard=use_drift_guard,
            drift_sigma=drift_sigma,
            orthogonalize_on_spawn=orthogonalize_on_spawn,
            device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mag: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_out: [B, T, d_model]
            x_mag_out: [B, T, d_model]
            u_tokens: [B * T, hyperspace_dim] on S^{D-1}
        """
        # Step 1: Dual-Stream Riemannian Attention
        h_norm1 = self.norm1(x)
        out_fused, x_mag_out, _ = self.attn(h_norm1, x_mag=x_mag, mask=mask, start_pos=start_pos)
        x = x + out_fused

        # Step 2: Project tokens onto S^{D-1} x R+ for MoE routing
        h_norm2 = self.norm2(x)
        B, T, D = h_norm2.shape
        flat_h = h_norm2.reshape(B * T, D)
        u_tokens = project_to_sphere(self.to_u(flat_h))
        r_tokens = F.softplus(self.to_r(flat_h))

        if context is not None:
            if context.dim() == 2:
                flat_ctx = context.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
            elif context.dim() == 3 and context.size(1) != T:
                flat_ctx = context.expand(B, T, -1).reshape(B * T, -1)
            else:
                flat_ctx = context.reshape(B * T, -1)
        else:
            flat_ctx = None
        flat_moe_out, _ = self.moe(flat_h, u_tokens, r_tokens, context=flat_ctx)
        moe_out = flat_moe_out.reshape(B, T, D)
        x = x + moe_out

        return x, x_mag_out, u_tokens


class NativeSphericalTransformer(nn.Module):
    """
    Sovereign Cognitive Language Model.
    Integrates Dual-Stream attention, Shared-Expert MoE, Multi-Token Prediction (MTP),
    and non-parametric VSA Episodic Memory.
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        hyperspace_dim: int = 8,
        d_ff: int = 256,
        max_seq_len: int = 512,
        mtp_future_tokens: int = 2,
        vsa_capacity: int = 512,
        num_experts: int = 4,
        top_k: int = 2,
        max_experts: Optional[int] = None,
        enable_shared_expert: bool = True,
        enable_aux_loss_free_balancing: bool = True,
        enable_gwt_coalition: bool = True,
        enable_vsa_memory: bool = True,
        enable_world_model: bool = False,
        tie_weights: bool = False,
        use_drift_guard: bool = False,
        drift_sigma: float = 3.0,
        orthogonalize_on_spawn: bool = False,
        spawn_threshold: float = 0.0,
        device: torch.device = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hyperspace_dim = hyperspace_dim
        self.max_seq_len = max_seq_len
        self.mtp_future_tokens = mtp_future_tokens
        self.enable_vsa_memory = enable_vsa_memory
        self.enable_world_model = enable_world_model
        self.tie_weights = tie_weights
        self.device = device

        # Token & Positional Embeddings
        self.tok_embed = nn.Embedding(vocab_size, d_model, device=device)
        self.pos_embed = nn.Embedding(max_seq_len, d_model, device=device)

        # Transformer Blocks with GWT Coalition Bidding
        self.blocks = nn.ModuleList([
            NativeSphericalBlock(
                d_model=d_model,
                num_heads=num_heads,
                hyperspace_dim=hyperspace_dim,
                d_ff=d_ff,
                num_experts=num_experts,
                top_k=top_k,
                max_experts=max_experts,
                enable_shared_expert=enable_shared_expert,
                enable_aux_loss_free_balancing=enable_aux_loss_free_balancing,
                enable_gwt_coalition=enable_gwt_coalition,
                use_drift_guard=use_drift_guard,
                drift_sigma=drift_sigma,
                orthogonalize_on_spawn=orthogonalize_on_spawn,
                spawn_threshold=spawn_threshold,
                device=device
            )
            for _ in range(num_layers)
        ])

        # Final Normalization
        self.norm_f = nn.LayerNorm(d_model, device=device)

        # Output Heads: Primary Head (w_{t+1}) + MTP Auxiliary Heads (w_{t+2}, w_{t+3})
        # Output Heads: Primary Head (w_{t+1}) + MTP Auxiliary Heads (w_{t+2}, w_{t+3})
        self.head_primary = nn.Linear(d_model, vocab_size, bias=False, device=device)
        self.mtp_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False, device=device)
            for _ in range(max(0, mtp_future_tokens - 1))
        ])

        # Non-Parametric VSA Episodic Memory Buffer & Read Gate
        self.vsa_memory = VSAEpisodicMemory(
            capacity=vsa_capacity,
            hyperspace_dim=hyperspace_dim,
            payload_dim=d_model,
            device=device
        )
        self.vsa_read_gate = nn.Parameter(torch.tensor(0.15, device=device))
        self.vsa_write_threshold = 0.50

        # Causal World Model Trajectory Predictor (JEPA)
        if enable_world_model:
            self.w_causal_fwd = nn.Sequential(
                nn.Linear(d_model, d_model, device=device),
                nn.SiLU(),
                nn.Linear(d_model, d_model, device=device)
            )
            self.to_u_wm = nn.Linear(d_model, hyperspace_dim, bias=False, device=device)

        # Initialize weights with standard transformer variance
        self.apply(self._init_weights)

        # Tie weights after initialization if requested
        if tie_weights:
            self.head_primary.weight = self.tok_embed.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # 1. Adapt pos_embed if sequence length differs
        pos_key = prefix + "pos_embed.weight"
        if pos_key in state_dict:
            ckpt_pe = state_dict[pos_key]
            target_weight = self.pos_embed.weight
            if ckpt_pe.shape != target_weight.shape:
                adapted = target_weight.clone()
                min_len = min(ckpt_pe.shape[0], adapted.shape[0])
                adapted[:min_len] = ckpt_pe[:min_len]
                state_dict[pos_key] = adapted

        # 2. Handle world model mismatch
        if not self.enable_world_model:
            # Strip internal world model keys so they don't trigger unexpected_keys
            for k in list(state_dict.keys()):
                if k.startswith(prefix + "w_causal_fwd.") or k.startswith(prefix + "to_u_wm."):
                    state_dict.pop(k)
        else:
            # Seed missing internal world model weights if checkpoint lacked them
            for name, param in self.named_parameters():
                if name.startswith("w_causal_fwd.") or name.startswith("to_u_wm."):
                    full_key = prefix + name
                    if full_key not in state_dict:
                        state_dict[full_key] = param.clone()

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        return_mtp: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass with Multi-Token Prediction (MTP).
        Args:
            input_ids: [B, T]
            targets: Optional [B, T] target token indices
            context: Optional external context [B, T, d_model]
        Returns:
            Dict containing primary logits, auxiliary MTP logits, loss, and internal cone norm.
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        target_dev = self.tok_embed.weight.device
        input_ids = input_ids.to(target_dev)
        if targets is not None:
            targets = targets.to(target_dev)

        positions = torch.arange(0, T, device=target_dev).unsqueeze(0).expand(B, T)
        x = self.tok_embed(input_ids) + self.pos_embed(positions)
        x_mag = x.clone()

        # Causal Attention Mask
        mask = torch.triu(torch.full((T, T), float("-inf"), device=input_ids.device), diagonal=1)

        all_u_tokens = []
        for block in self.blocks:
            x, x_mag, u_tokens = block(x, x_mag=x_mag, mask=mask, context=context)
            all_u_tokens.append(u_tokens)

        h = self.norm_f(x) # [B, T, d_model]

        # 1. Non-Parametric VSA Episodic Memory Soft-Attention Read
        if self.enable_vsa_memory and bool(self.vsa_memory.occupied.any()) and len(all_u_tokens) > 0:
            last_u = all_u_tokens[-1] # [B * T, hyperspace_dim]
            flat_h = h.reshape(B * T, self.d_model)
            last_r = F.softplus(self.blocks[-1].to_r(flat_h)) # [B * T, 1]
            echo, conf = self.vsa_memory(last_u, last_r) # [B * T, d_model], [B * T, 1]
            echo_3d = echo.reshape(B, T, self.d_model)
            conf_3d = conf.reshape(B, T, 1)
            h = h + self.vsa_read_gate * conf_3d * echo_3d

        # Primary Next-Token Logits: predicts t+1
        logits_primary = self.head_primary(h) # [B, T, vocab_size]

        # Auxiliary MTP Heads
        mtp_logits = []
        if return_mtp and len(self.mtp_heads) > 0:
            for offset_idx, mtp_head in enumerate(self.mtp_heads, start=2):
                # Head predicts t + offset_idx
                # Sequence length is T - offset_idx
                if T > offset_idx:
                    h_valid = h[:, :T - offset_idx, :]
                    logits_aux = mtp_head(h_valid)
                    mtp_logits.append((offset_idx, logits_aux))

        # Spherical Circular Variance Loss (Anti-Collapse on S^7)
        if len(all_u_tokens) > 0:
            stacked_u = torch.cat(all_u_tokens, dim=0) # [N_total, hyperspace_dim]
            mean_vector = torch.mean(stacked_u, dim=0)
            cone_norm = torch.norm(mean_vector, p=2)
            # Penalize cone norm if R > 0.50
            loss_cone = 0.2 * torch.clamp(cone_norm - 0.50, min=0.0) ** 2
        else:
            cone_norm = torch.tensor(0.0, device=input_ids.device)
            loss_cone = torch.tensor(0.0, device=input_ids.device)

        # Loss Computation
        loss = None
        loss_components = {}
        if targets is not None:
            # 1. Primary Loss: target is shifted by 1
            # logits_primary[:, :-1] predicts targets[:, 1:]
            loss_primary = F.cross_entropy(
                logits_primary[:, :-1, :].reshape(-1, self.vocab_size),
                targets[:, 1:].reshape(-1)
            )
            total_loss = loss_primary + loss_cone
            loss_components["primary_ce"] = loss_primary.item()
            loss_components["cone_norm"] = cone_norm.item()

            # 2. MTP Auxiliary Losses
            for offset, aux_logits in mtp_logits:
                # aux_logits predicts targets[:, offset:]
                tgt_aux = targets[:, offset:].reshape(-1)
                flat_aux_logits = aux_logits.reshape(-1, self.vocab_size)
                if flat_aux_logits.size(0) == tgt_aux.size(0) and tgt_aux.numel() > 0:
                    loss_aux = F.cross_entropy(flat_aux_logits, tgt_aux)
                    weight = 0.5 ** (offset - 1) # Exponential decay for further future tokens
                    total_loss = total_loss + weight * loss_aux
                    loss_components[f"mtp_offset_{offset}"] = loss_aux.item()

            # 3. Universal World Model Causal Trajectory Coherence
            if self.enable_world_model and T > 1:
                z_curr = h[:, :-1, :] # [B, T-1, d_model]
                z_next = h[:, 1:, :]  # [B, T-1, d_model]
                z_pred = self.w_causal_fwd(z_curr)
                loss_wm_mse = F.mse_loss(z_pred, z_next)

                flat_z_pred = z_pred.reshape(-1, self.d_model)
                flat_z_next = z_next.reshape(-1, self.d_model)
                u_pred = project_to_sphere(self.to_u_wm(flat_z_pred))
                u_next = project_to_sphere(self.to_u_wm(flat_z_next))
                loss_wm_geo = geodesic_distance_sphere(u_pred, u_next).mean()

                loss_wm = 0.1 * (loss_wm_mse + loss_wm_geo)
                total_loss = total_loss + loss_wm
                loss_components["world_model_causal"] = loss_wm.item()

            # 4. Optional Surprise-Adaptive Hebbian Writes into VSA Memory
            if self.enable_vsa_memory and len(all_u_tokens) > 0 and (not self.training or random.random() < 0.05):
                with torch.no_grad():
                    per_token_loss = F.cross_entropy(
                        logits_primary[:, :-1, :].reshape(-1, self.vocab_size),
                        targets[:, 1:].reshape(-1),
                        reduction="none"
                    )
                    high_surprise_mask = per_token_loss > self.vsa_write_threshold
                    if high_surprise_mask.any():
                        high_indices = torch.nonzero(high_surprise_mask).squeeze(-1)
                        # Write up to 4 high surprise tokens per batch
                        for idx in high_indices[:4]:
                            if idx < all_u_tokens[-1].size(0):
                                u_w = all_u_tokens[-1][idx:idx+1]
                                flat_h = h.reshape(-1, self.d_model)
                                r_w = F.softplus(self.blocks[-1].to_r(flat_h[idx:idx+1]))
                                v_w = flat_h[idx:idx+1]
                                self.vsa_memory.write(u_w, r_w, v_w)

            loss = total_loss
            loss_components["total_loss"] = total_loss.item()

        last_r = None
        if len(self.blocks) > 0:
            flat_h = h.reshape(B * T, self.d_model)
            last_r = F.softplus(self.blocks[-1].to_r(flat_h))

        return {
            "logits": logits_primary,
            "mtp_logits": mtp_logits,
            "loss": loss,
            "loss_components": loss_components,
            "cone_norm": cone_norm.item(),
            "hidden_states": h,
            "u_tokens": all_u_tokens[-1] if len(all_u_tokens) > 0 else None,
            "r_tokens": last_r
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 40,
        enable_born_rule: bool = True,
        born_alpha: float = 3.5,
        repetition_window: int = 32,
        enable_satiation: bool = False,
        satiation_alpha: float = 2.5,
        satiation_decay: float = 0.70,
        satiation_threshold: float = 0.05,
        logits_processor: Optional[Any] = None,
        context: Optional[torch.Tensor] = None,
        write_to_vsa: bool = False
    ) -> torch.Tensor:
        """
        Autoregressive generation with:
        1. Cognitive Context Latent Conditioning (Hopfield Attractor & Episodic Grounding).
        2. Prefrontal Neuro-Symbolic Scratchpad Interception (exact arithmetic/tool evaluation).
        3. Rectified Born Rule quantum destructive interference (eliminates loops).
        4. Gem 1: Continuous Refractory Satiation Field sigma(v) (organic loop prevention).
        5. Surprise-Adaptive VSA Episodic Memory consolidation.
        6. Strict deterministic argmax decoding when temperature <= 0.0.
        """
        self.eval()
        curr_ids = prompt_ids.clone()
        B = curr_ids.size(0)

        # Gem 1: Refractory Satiation Field over vocabulary
        satiation_field = torch.zeros(B, self.vocab_size, device=curr_ids.device) if enable_satiation else None

        for _ in range(max_new_tokens):
            if curr_ids.size(1) >= self.max_seq_len:
                break

            out = self.forward(curr_ids, context=context, return_mtp=False)
            next_token_logits = out["logits"][:, -1, :].clone() # [B, vocab_size]

            # 1. Prefrontal Neuro-Symbolic Scratchpad Interceptor
            if logits_processor is not None:
                try:
                    next_token_logits = logits_processor(curr_ids, next_token_logits)
                except Exception:
                    pass

            # 2. Gem 1: Continuous Refractory Satiation Field
            if enable_satiation and satiation_field is not None:
                next_token_logits = next_token_logits - satiation_alpha * satiation_field

            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            if enable_born_rule:
                # Apply Rectified Born Rule Destructive Interference
                modified_logits = next_token_logits.clone()
                for b in range(B):
                    ctx = curr_ids[b, -repetition_window:].tolist()
                    if len(ctx) > 0:
                        # 1. Immediate previous token suppression (prevents tied-embedding identity loop)
                        last_tok = curr_ids[b, -1].item()
                        if last_tok < modified_logits.size(1):
                            modified_logits[b, last_tok] -= 12.0

                        # 2. Window frequency destructive interference
                        counts = {}
                        for tid in ctx:
                            counts[tid] = counts.get(tid, 0) + 1
                        for tid, cnt in counts.items():
                            if tid < modified_logits.size(1):
                                penalty = (cnt ** 1.5) * born_alpha * 2.0
                                modified_logits[b, tid] -= penalty
                next_token_logits = modified_logits

            # Top-K filtering (only when sampling)
            if temperature > 0 and top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

            # Selection: Strict deterministic argmax for greedy, multinomial for sampling
            if temperature <= 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) # [B, 1]
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1) # [B, 1]
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            # Update Refractory Satiation Field
            if enable_satiation and satiation_field is not None:
                satiation_field = satiation_field * satiation_decay
                satiation_field[satiation_field < satiation_threshold] = 0.0
                satiation_field.scatter_(1, next_token, 1.0)

        # 3. Optional VSA Memory Consolidation of Generated Episode
        if write_to_vsa and self.enable_vsa_memory and out.get("u_tokens") is not None:
            last_u = out["u_tokens"][-1:]
            flat_h = out["hidden_states"].reshape(-1, self.d_model)[-1:]
            last_r = F.softplus(self.blocks[-1].to_r(flat_h))
            self.vsa_memory.write(last_u, last_r, flat_h)

        return curr_ids

    def spawn_expert(
        self,
        u_coord: torch.Tensor,
        r_coord: torch.Tensor,
        parent_idx: Optional[int] = None
    ) -> List[int]:
        """
        Synchronously spawns a new expert across all transformer blocks.
        Returns a list of newly assigned expert indices for each block.
        """
        new_indices = []
        for block in self.blocks:
            if hasattr(block, "moe") and isinstance(block.moe, SphericalDynamicMoE):
                idx = block.moe._spawn_expert_raw(u_coord, r_coord, parent_idx=parent_idx)
                new_indices.append(idx)
        return new_indices

    @torch.no_grad()
    def mitose_all_eligible(self, max_splits: int = 1) -> List[List[Dict[str, Any]]]:
        """
        Executes anisotropic mitosis on eligible over-stretched experts across all transformer blocks.
        """
        all_events = []
        for block in self.blocks:
            if hasattr(block, "moe") and isinstance(block.moe, SphericalDynamicMoE):
                events = block.moe.mitose_all_eligible(max_splits=max_splits)
                all_events.append(events)
        return all_events

