from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.chunker import BYTE_PAD_ID
from dmoe.fast_memory import ExactLdaFastMemory, MultiSlotExactLdaMemory
from dmoe.model import rebuild_optimizer_preserving_state
from dmoe.replay_buffer import ReplayBuffer


class ContinualLearner:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer_capacity: int = 500,
        replay_batch_size: int = 8,
        distillation_weight: float = 0.5,
        maturity_age: Optional[int] = None,
        tau_novelty: Optional[float] = None,
        entropy_floor: Optional[float] = None,
        theta_novelty_vertical: Optional[float] = None,
        mutation_check_interval: int = 20,
        enable_fast_memory: bool = False,
        fast_memory_blend: float = 0.5,
        fast_memory_reg: float = 0.5,
        enable_local_lda: Optional[bool] = None,
        enable_setgate: Optional[bool] = None,
        freeze_backbone_on_domain: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity) if replay_buffer_capacity > 0 else None
        self.replay_batch_size = replay_batch_size
        self.distillation_weight = distillation_weight
        self.maturity_age = maturity_age
        self.tau_novelty = tau_novelty
        self.entropy_floor = entropy_floor
        self.theta_novelty_vertical = theta_novelty_vertical
        self.mutation_check_interval = mutation_check_interval
        self.enable_fast_memory = enable_fast_memory
        self.fast_memory_blend = fast_memory_blend
        self.fast_memory_reg = fast_memory_reg
        self.enable_local_lda = (
            getattr(model, "enable_local_lda", False)
            if enable_local_lda is None
            else bool(enable_local_lda)
        )
        self.enable_setgate = (
            getattr(model, "enable_setgate", False)
            if enable_setgate is None
            else bool(enable_setgate)
        )
        self.freeze_backbone_on_domain = freeze_backbone_on_domain
        self.fast_memory: Optional[nn.Module] = None
        self.global_step = 0
        self._apply_thresholds()

    def _ensure_fast_memory(self, out: Dict[str, Any], output_head: str, device: torch.device) -> None:
        if not self.enable_fast_memory or self.fast_memory is not None:
            return
        d_model = getattr(self.model, "d_model", out["decoder_latent"].size(-1))
        tokenizer_free = getattr(self.model, "tokenizer_free", False)
        chunk_size = getattr(self.model, "chunk_size", 4)
        if output_head == "byte_chunk" or (output_head == "token" and tokenizer_free):
            has_byte_res = out.get("byte_residuals") is not None
            slot_dim = 2 * d_model if has_byte_res else d_model
            self.fast_memory = MultiSlotExactLdaMemory(
                d_in=slot_dim, num_classes=256, num_slots=chunk_size, reg=self.fast_memory_reg
            ).to(device)
        elif output_head == "token":
            vocab_size = out["output"].size(-1)
            self.fast_memory = ExactLdaFastMemory(
                d_in=d_model, num_classes=vocab_size, reg=self.fast_memory_reg
            ).to(device)

    def _extract_fast_memory_features(self, out: Dict[str, Any]) -> torch.Tensor:
        latent = self._extract_latent(out)
        if isinstance(self.fast_memory, MultiSlotExactLdaMemory):
            return self.model.decoder.extract_slot_features(latent, out.get("byte_residuals"))
        return latent

    def _apply_thresholds(self) -> None:
        if self.theta_novelty_vertical is not None and hasattr(self.model, "theta_novelty_vertical"):
            self.model.theta_novelty_vertical = self.theta_novelty_vertical
        if hasattr(self.model, "layers"):
            for layer in self.model.layers:
                if self.maturity_age is not None:
                    layer.moe.maturity_age = self.maturity_age
                if self.tau_novelty is not None:
                    layer.moe.tau_novelty = self.tau_novelty
                if self.entropy_floor is not None:
                    layer.moe.entropy_floor = self.entropy_floor

    def _extract_latent(self, out: Dict[str, Any]) -> torch.Tensor:
        if "decoder_latent" in out:
            return out["decoder_latent"]
        return self.model.decoder(out["z_real"], out["z_imag"], head="latent")

    def _forward_replay(self, replay_x: torch.Tensor, replay_head: str) -> Dict[str, Any]:
        try:
            return self.model(replay_x, output_head=replay_head, update_stats=False)
        except TypeError:
            return self.model(replay_x, output_head=replay_head)

    def step(self, x: torch.Tensor, targets: torch.Tensor, output_head: str = "token") -> Dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(x, output_head=output_head)
        input_novelty_score = (
            out["novelty_score"].item()
            if "novelty_score" in out
            else self.model.manifold_corr.last_novelty_score.item()
        )
        task_loss = self._task_loss(out["output"], targets, output_head)
        with torch.no_grad():
            latent_snapshot = self._extract_latent(out).detach().clone()

        rehearsal_loss = torch.tensor(0.0, device=x.device)
        distillation_loss = torch.tensor(0.0, device=x.device)
        replay_samples = self.replay_buffer.sample(self.replay_batch_size) if self.replay_buffer is not None else []
        if replay_samples:
            groups: Dict[Tuple[Any, ...], List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = defaultdict(list)
            for replay_x, replay_targets, replay_head, replay_latent in replay_samples:
                key = (
                    replay_head,
                    tuple(replay_x.shape[1:]),
                    tuple(replay_targets.shape[1:]),
                    tuple(replay_latent.shape[1:]),
                )
                groups[key].append((replay_x, replay_targets, replay_latent))

            rehearsal_losses = []
            distillation_losses = []
            weights = []
            for (replay_head, _, _, _), items in groups.items():
                batch_x = torch.cat([item[0] for item in items], dim=0)
                batch_targets = torch.cat([item[1] for item in items], dim=0)
                batch_latent = torch.cat([item[2] for item in items], dim=0)

                replay_out = self._forward_replay(batch_x, replay_head)
                rehearsal_losses.append(self._task_loss(replay_out["output"], batch_targets, replay_head))
                current_latent = self._extract_latent(replay_out)
                distillation_losses.append(F.mse_loss(current_latent, batch_latent))
                weights.append(float(len(items)))

            weight_tensor = torch.tensor(weights, device=x.device)
            weight_tensor = weight_tensor / weight_tensor.sum()
            rehearsal_loss = (torch.stack(rehearsal_losses) * weight_tensor).sum()
            distillation_loss = (torch.stack(distillation_losses) * weight_tensor).sum()

        total_loss = task_loss + rehearsal_loss + self.distillation_weight * distillation_loss
        total_loss.backward()
        self.optimizer.step()

        if self.enable_local_lda and hasattr(self.model, "layers"):
            with torch.no_grad():
                slot_feats = self.model.decoder.extract_slot_features(
                    out["decoder_latent"], out.get("byte_residuals")
                )
                for layer in self.model.layers:
                    if hasattr(layer, "moe") and getattr(layer.moe, "enable_local_lda", False):
                        chosen_idx = getattr(layer.moe, "last_routed_expert", None)
                        if chosen_idx is not None and chosen_idx < len(layer.moe.experts):
                            if not bool(layer.moe.expert_frozen[chosen_idx]):
                                layer.moe.experts[chosen_idx].absorb(slot_feats, targets, ignore_index=BYTE_PAD_ID)

        if self.enable_fast_memory:
            with torch.no_grad():
                post_out = self._forward_replay(x, output_head)
                self._ensure_fast_memory(post_out, output_head, x.device)
                if isinstance(self.fast_memory, MultiSlotExactLdaMemory):
                    slot_feats = self._extract_fast_memory_features(post_out)
                    self.fast_memory.absorb_slots(slot_feats, targets, ignore_index=BYTE_PAD_ID)
                elif isinstance(self.fast_memory, ExactLdaFastMemory):
                    feats = self._extract_fast_memory_features(post_out)
                    self.fast_memory.absorb(feats, targets, ignore_index=BYTE_PAD_ID)

        if self.replay_buffer is not None:
            self.replay_buffer.add((x.detach().clone(), targets.detach().clone(), output_head, latent_snapshot))

        self.global_step += 1
        mutation_report = None
        if not getattr(self.model, "enable_setgate", False) and self.global_step % self.mutation_check_interval == 0:
            mutation_report = self.model.execute_graph_mutation(novelty_score=input_novelty_score)
            self._apply_thresholds()
            if mutation_report["mutation_logs"]:
                self.optimizer = rebuild_optimizer_preserving_state(
                    self.optimizer,
                    self.model,
                    lr=self.optimizer.defaults["lr"],
                    weight_decay=self.optimizer.defaults.get("weight_decay", 0.01),
                )
            for layer in self.model.layers:
                layer.moe.freeze_mature_experts()

        return {
            "task_loss": task_loss.item(),
            "rehearsal_loss": rehearsal_loss.item(),
            "distillation_loss": distillation_loss.item(),
            "total_loss": total_loss.item(),
            "novelty_score": input_novelty_score,
            "mutation_report": mutation_report,
        }

    def domain_transition(self, new_domain_id: Optional[int] = None) -> None:
        """
        Transition to a new domain:
        - Freeze all active experts from the completed domain.
        - Spawn a dedicated expert for the new domain in each layer.
        - If freeze_backbone_on_domain, freeze encoder, attention, and decoder.
        - Rebuild the optimizer for trainable parameters only.
        """
        if not hasattr(self.model, "layers"):
            return

        for layer in self.model.layers:
            if hasattr(layer, "moe"):
                for idx, is_frozen in enumerate(layer.moe.expert_frozen):
                    if not bool(is_frozen):
                        layer.moe.freeze_expert(idx)
                layer.moe.spawn_expert()

        if self.freeze_backbone_on_domain and (new_domain_id is None or new_domain_id >= 1):
            if hasattr(self.model, "encoder"):
                for p in self.model.encoder.parameters():
                    p.requires_grad_(False)
            for layer in self.model.layers:
                for sub in ("norm1", "attn", "norm2"):
                    if hasattr(layer, sub):
                        for p in getattr(layer, sub).parameters():
                            p.requires_grad_(False)
            if hasattr(self.model, "decoder"):
                for p in self.model.decoder.parameters():
                    p.requires_grad_(False)

        active_params = [p for p in self.model.parameters() if p.requires_grad]
        if active_params:
            self.optimizer = torch.optim.AdamW(
                active_params,
                lr=self.optimizer.defaults["lr"],
                weight_decay=self.optimizer.defaults.get("weight_decay", 0.01),
            )

    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor, output_head: str = "token") -> torch.Tensor:
        """
        Fast-slow inference readout:
        combines neural output with either per-expert LocalLdaExpert or global ExactLdaFastMemory.
        """
        was_training = self.model.training
        self.model.eval()
        out = self._forward_replay(x, output_head)
        slow_logits = out["output"]
        if was_training:
            self.model.train()

        if self.enable_local_lda and hasattr(self.model, "layers"):
            last_moe = self.model.layers[-1].moe
            chosen_idx = getattr(last_moe, "last_routed_expert", 0)
            if chosen_idx < len(last_moe.experts):
                expert = last_moe.experts[chosen_idx]
                if hasattr(expert, "predict_logits"):
                    slot_feats = self.model.decoder.extract_slot_features(
                        out["decoder_latent"], out.get("byte_residuals")
                    )
                    fast_logits = expert.predict_logits(slot_feats)
                    if fast_logits is not None:
                        return fast_logits
            return slow_logits

        if not self.enable_fast_memory or self.fast_memory is None:
            return slow_logits

        feats = self._extract_fast_memory_features(out)
        if isinstance(self.fast_memory, MultiSlotExactLdaMemory):
            fast_logits = self.fast_memory.predict_slot_logits(feats)
        elif isinstance(self.fast_memory, ExactLdaFastMemory):
            fast_logits = self.fast_memory.predict_logits(feats)
        else:
            return slow_logits

        slow_norm = F.log_softmax(slow_logits, dim=-1)
        fast_norm = F.log_softmax(fast_logits, dim=-1)
        w = self.fast_memory_blend
        return (1.0 - w) * slow_norm + w * fast_norm

    @torch.no_grad()
    def unlearn(self, x: torch.Tensor, targets: torch.Tensor, output_head: str = "token") -> None:
        """
        Exact closed-form downdating of sufficient statistics from fast_memory.
        """
        if not self.enable_fast_memory or self.fast_memory is None:
            return
        was_training = self.model.training
        self.model.eval()
        out = self._forward_replay(x, output_head)
        if was_training:
            self.model.train()

        feats = self._extract_fast_memory_features(out)
        if isinstance(self.fast_memory, MultiSlotExactLdaMemory):
            self.fast_memory.downdate_slots(feats, targets, ignore_index=BYTE_PAD_ID)
        elif isinstance(self.fast_memory, ExactLdaFastMemory):
            self.fast_memory.downdate(feats, targets, ignore_index=BYTE_PAD_ID)

    @staticmethod
    def _task_loss(output: torch.Tensor, targets: torch.Tensor, output_head: str) -> torch.Tensor:
        if output.dim() == 4:
            bsz, s_chunks, c_size, vocab = output.shape
            flat_out = output.reshape(bsz, s_chunks * c_size, vocab)
            flat_tgt = targets.reshape(bsz, -1)
            if flat_tgt.size(1) < flat_out.size(1):
                flat_out = flat_out[:, : flat_tgt.size(1), :]
            return F.cross_entropy(
                flat_out.reshape(-1, vocab),
                flat_tgt.reshape(-1),
                ignore_index=BYTE_PAD_ID,
            )
        if output_head in ("token", "byte_chunk"):
            return F.cross_entropy(output.reshape(-1, output.size(-1)), targets.reshape(-1))
        return F.mse_loss(output, targets)
