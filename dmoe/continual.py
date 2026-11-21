from collections import defaultdict
import inspect
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.chunker import BYTE_PAD_ID
from dmoe.fast_memory import ExactLdaFastMemory, MultiSlotExactLdaMemory
from dmoe.model import rebuild_optimizer_preserving_state
from dmoe.null_space import NullSpaceProjector
from dmoe.replay_buffer import ReplayBuffer
from dmoe.spgc import LatentQuarantineGate, MuonBONGOptimizer, classify_spgc_param_groups


class ContinualLearner:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
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
        enable_null_space: bool = False,
        null_space_energy_threshold: float = 0.98,
        null_space_max_rank_ratio: float = 0.6,
        enable_spgc: bool = False,
        spgc_tau_skip: float = 0.12,
        spgc_min_sources: int = 2,
        spgc_max_bytes: int = 16384,
        spgc_lr: float = 4e-3,
        spgc_wd_trunk: float = 0.05,
        spgc_seed: int = 0,
        spgc_store_samples: bool = False,
        spgc_min_flush: int = 1,
        spgc_auto_consolidate_steps: Optional[int] = None,
        spgc_auto_consolidate_loss: float = 0.15,
        spgc_bong_prior_pull: float = 0.0,
        spgc_fast_module_names: Tuple[str, ...] = ("ctx_proj", "fast_adapter", "moe", "experts", "decoder", "head"),
        spgc_wm_loss_weight: float = 0.25,
        spgc_enable_token_gating: bool = True,
        spgc_min_steps_before_skip: int = 0,
        spgc_wd_fast: float = 0.01,
    ):
        self.model = model
        fwd_params = inspect.signature(self.model.forward).parameters
        self._forward_accepts_targets = "targets" in fwd_params
        self._forward_accepts_update_stats = "update_stats" in fwd_params
        self._forward_accepts_output_head = "output_head" in fwd_params

        self.enable_spgc = bool(enable_spgc)
        self.spgc_tau_skip = float(spgc_tau_skip)
        self.spgc_lr = float(spgc_lr)
        self.spgc_wd_trunk = float(spgc_wd_trunk)
        self.spgc_wd_fast = float(spgc_wd_fast)
        self.spgc_store_samples = bool(spgc_store_samples)
        self.spgc_min_flush = max(1, int(spgc_min_flush))
        self.spgc_auto_consolidate_steps = (
            int(spgc_auto_consolidate_steps) if spgc_auto_consolidate_steps is not None else None
        )
        self.spgc_auto_consolidate_loss = float(spgc_auto_consolidate_loss)
        self.spgc_bong_prior_pull = float(spgc_bong_prior_pull)
        self.spgc_fast_module_names = tuple(spgc_fast_module_names)
        self.spgc_wm_loss_weight = float(spgc_wm_loss_weight)
        self.spgc_enable_token_gating = bool(spgc_enable_token_gating)
        self.spgc_min_steps_before_skip = max(0, int(spgc_min_steps_before_skip))
        self.spgc_ema_loss: Optional[float] = None
        self.spgc_gate: Optional[LatentQuarantineGate] = None
        self._spgc_last_trainable_ids = {id(p) for p in self.model.parameters() if p.requires_grad}
        if self.enable_spgc:
            groups = classify_spgc_param_groups(
                self.model,
                fast_module_names=self.spgc_fast_module_names,
                wd_trunk=self.spgc_wd_trunk,
                wd_fast=self.spgc_wd_fast,
                lr_muon=self.spgc_lr * 2.0,
                lr_adam=self.spgc_lr,
                lr_fast=self.spgc_lr * 1.5,
            )
            self.optimizer: Any = (
                optimizer
                if optimizer is not None
                else MuonBONGOptimizer(
                    groups,
                    bong_prior_pull=self.spgc_bong_prior_pull,
                    mode="acquire",
                    track_fisher=True,
                )
            )
            d_model = int(getattr(self.model, "d_model", 64))
            need_queue = self.spgc_min_flush > 1
            self.spgc_gate = LatentQuarantineGate(
                latent_dim=d_model,
                seed=int(spgc_seed),
                max_bytes=int(spgc_max_bytes),
                tau_skip=self.spgc_tau_skip,
                min_sources=int(spgc_min_sources),
                bytes_per_sample=32 if (self.spgc_store_samples or need_queue) else 0,
                store_samples=self.spgc_store_samples,
                max_queue_samples=(max(32, self.spgc_min_flush * 2) if need_queue else None),
            )
        else:
            if optimizer is None:
                optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.spgc_lr)
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
        self.enable_null_space = enable_null_space
        self.null_space_projector: Optional[NullSpaceProjector] = (
            NullSpaceProjector(
                model=self.model,
                energy_threshold=null_space_energy_threshold,
                max_rank_ratio=null_space_max_rank_ratio,
            )
            if enable_null_space
            else None
        )
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
        if "latent" in out:
            return out["latent"]
        return self.model.decoder(out["z_real"], out["z_imag"], head="latent")

    def _forward_replay(self, replay_x: torch.Tensor, replay_head: str) -> Dict[str, Any]:
        try:
            return self.model(replay_x, output_head=replay_head, update_stats=False)
        except TypeError:
            return self.model(replay_x, output_head=replay_head)

    def _rebuild_spgc_optimizer(
        self,
        consolidate: bool = False,
        new_weight_decay: Optional[float] = None,
    ) -> None:
        if not isinstance(self.optimizer, MuonBONGOptimizer):
            return
        groups = classify_spgc_param_groups(
            self.model,
            fast_module_names=self.spgc_fast_module_names,
            wd_trunk=self.spgc_wd_trunk,
            wd_fast=0.01,
            lr_muon=self.spgc_lr * 2.0,
            lr_adam=self.spgc_lr,
            lr_fast=self.spgc_lr * 1.5,
        )
        self.optimizer.rebuild_preserving_state(groups)
        self._spgc_last_trainable_ids = {id(p) for p in self.model.parameters() if p.requires_grad}
        if consolidate:
            self.optimizer.set_mode(
                "consolidate",
                snapshot_anchor=(self.spgc_bong_prior_pull > 0.0),
                new_weight_decay=0.01 if new_weight_decay is None else new_weight_decay,
            )

    def set_spgc_consolidate(self, new_weight_decay: float = 0.01) -> None:
        """Transition SPGC optimizer from fast Muon acquisition into BONG consolidation."""
        if isinstance(self.optimizer, MuonBONGOptimizer):
            self.optimizer.set_mode(
                "consolidate",
                snapshot_anchor=(self.spgc_bong_prior_pull > 0.0),
                new_weight_decay=new_weight_decay,
            )

    def _fit_latent_dim(self, z: torch.Tensor) -> torch.Tensor:
        target_d = self.spgc_gate.latent_dim
        if z.size(-1) == target_d:
            return z
        if z.size(-1) > target_d:
            return z[..., :target_d]
        pad = torch.zeros(*z.shape[:-1], target_d - z.size(-1), device=z.device, dtype=z.dtype)
        return torch.cat([z, pad], dim=-1)

    def _extract_spgc_latents(
        self,
        screen_out: Dict[str, Any],
        B: int,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        if "latent" in screen_out:
            z_raw = screen_out["latent"]
        elif "decoder_latent" in screen_out:
            z_raw = screen_out["decoder_latent"]
        else:
            z_raw = torch.zeros(B, int(getattr(self.model, "d_model", 64)), device=device)

        if T <= 1:
            z_out = z_raw.mean(dim=1) if z_raw.dim() == 3 else z_raw
            return self._fit_latent_dim(z_out)

        if z_raw.dim() == 3:
            if z_raw.size(1) == T:
                z_tok = z_raw
            elif z_raw.size(1) > 0 and T % z_raw.size(1) == 0:
                z_tok = z_raw.repeat_interleave(T // z_raw.size(1), dim=1)
            elif z_raw.size(1) >= T:
                z_tok = z_raw[:, :T, :]
            else:
                z_tok = z_raw.mean(dim=1, keepdim=True).expand(B, T, z_raw.size(-1))
        else:
            z_tok = z_raw.unsqueeze(1).expand(B, T, z_raw.size(-1))
        return self._fit_latent_dim(z_tok)

    def _forward_spgc_train(self, xb: torch.Tensor, yb: torch.Tensor, output_head: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self._forward_accepts_output_head:
            kwargs["output_head"] = output_head
        if self._forward_accepts_targets and (not yb.is_floating_point()):
            kwargs["targets"] = yb
        return self.model(xb, **kwargs)

    def _compute_spgc_loss(
        self,
        out: Dict[str, Any],
        yb: torch.Tensor,
        mb: Optional[torch.Tensor],
        output_head: str,
    ) -> Tuple[torch.Tensor, float]:
        is_continuous = (output_head == "continuous") or yb.is_floating_point()
        if is_continuous:
            task_loss = F.mse_loss(out["output"], yb)
        else:
            raw_out = out["output"]
            if mb is not None:
                bsz = xb_bsz = raw_out.size(0)
                flat_logits = raw_out.reshape(bsz, -1, raw_out.size(-1))
                flat_tgt = yb.reshape(xb_bsz, -1)
                t_min = min(flat_logits.size(1), flat_tgt.size(1), mb.size(1))
                ce = F.cross_entropy(
                    flat_logits[:, :t_min, :].reshape(-1, flat_logits.size(-1)),
                    flat_tgt[:, :t_min].reshape(-1),
                    reduction="none",
                    ignore_index=BYTE_PAD_ID,
                ).view(bsz, t_min)
                eff_mask = mb[:, :t_min] & (flat_tgt[:, :t_min] != BYTE_PAD_ID)
                if (
                    getattr(self.model, "enable_thinking", False)
                    and "thinking_out" in out
                    and getattr(self.model, "thinker", None) is not None
                    and yb.dim() == 2
                    and yb.size(1) == eff_mask.size(1)
                ):
                    task_loss = self.model.thinker.compute_thinking_loss(
                        out["thinking_out"], yb, valid_mask=eff_mask
                    )["loss"]
                else:
                    denom = eff_mask.float().sum().clamp_min(1.0)
                    task_loss = (ce * eff_mask.float()).sum() / denom
            elif getattr(self.model, "enable_thinking", False) and "thinking_out" in out and getattr(self.model, "thinker", None) is not None:
                task_loss = self.model.thinker.compute_thinking_loss(out["thinking_out"], yb)["loss"]
            else:
                task_loss = self._task_loss(raw_out, yb, output_head)

        total_loss = task_loss
        if out.get("world_model_loss") is not None and isinstance(out["world_model_loss"], dict):
            wm_l = out["world_model_loss"].get("loss")
            if isinstance(wm_l, torch.Tensor):
                total_loss = total_loss + self.spgc_wm_loss_weight * wm_l
        for aux_key in ("aux_loss", "load_balance_loss"):
            aux_v = out.get(aux_key)
            if isinstance(aux_v, torch.Tensor) and aux_v.requires_grad:
                total_loss = total_loss + aux_v
        return total_loss, float(task_loss.detach().item())

    def _run_spgc_backward_and_step(
        self,
        xb: torch.Tensor,
        yb: torch.Tensor,
        mb: Optional[torch.Tensor],
        output_head: str,
        flips: int,
    ) -> Dict[str, Any]:
        if flips > 0 and isinstance(self.optimizer, MuonBONGOptimizer):
            self.optimizer.temper_precision(roles=("fast",), factor=0.25)

        if self.spgc_store_samples and self.spgc_gate is not None and mb is None:
            rx, ry = self.spgc_gate.sample_core_replay(min(self.replay_batch_size, 16), xb.device)
            if rx is not None and rx.shape[1:] == xb.shape[1:] and ry is not None and ry.shape[1:] == yb.shape[1:]:
                xb = torch.cat([xb, rx], dim=0)
                yb = torch.cat([yb, ry], dim=0)

        self.model.train()
        self.optimizer.zero_grad()
        out = self._forward_spgc_train(xb, yb, output_head)
        input_novelty_score = (
            float(out["novelty_score"].item())
            if "novelty_score" in out
            else (
                float(self.model.manifold_corr.last_novelty_score.item())
                if hasattr(self.model, "manifold_corr")
                else 0.0
            )
        )
        total_loss, task_loss_val = self._compute_spgc_loss(out, yb, mb, output_head)
        total_loss.backward()

        # Sync optimizer if `auto_lifecycle` spawned/froze/pruned experts during forward
        cur_trainable_ids = {id(p) for p in self.model.parameters() if p.requires_grad}
        if cur_trainable_ids != self._spgc_last_trainable_ids:
            self._rebuild_spgc_optimizer(consolidate=False)

        if getattr(self, "null_space_projector", None) is not None:
            self.null_space_projector.project_gradients()
        self.optimizer.step()
        self.global_step += 1

        # Track EMA loss & trigger optional automatic Acquire -> Consolidate transition
        self.spgc_ema_loss = (
            task_loss_val if self.spgc_ema_loss is None else (0.95 * self.spgc_ema_loss + 0.05 * task_loss_val)
        )
        auto_consolidated = False
        if (
            self.spgc_auto_consolidate_steps is not None
            and isinstance(self.optimizer, MuonBONGOptimizer)
            and self.optimizer.mode == "acquire"
            and self.global_step >= self.spgc_auto_consolidate_steps
            and self.spgc_ema_loss <= self.spgc_auto_consolidate_loss
        ):
            self.set_spgc_consolidate()
            auto_consolidated = True

        # Support periodic structural mutation hooks (SelfStabilizingDynamicMorphicMoE)
        mutation_report = None
        if (
            hasattr(self.model, "execute_graph_mutation")
            and not getattr(self.model, "enable_setgate", False)
            and self.mutation_check_interval > 0
            and self.global_step % self.mutation_check_interval == 0
        ):
            mutation_report = self.model.execute_graph_mutation(novelty_score=input_novelty_score)
            self._apply_thresholds()
            if mutation_report.get("mutation_logs"):
                self._rebuild_spgc_optimizer(consolidate=False)
            for layer in self.model.layers:
                if hasattr(layer, "moe") and hasattr(layer.moe, "freeze_mature_experts"):
                    layer.moe.freeze_mature_experts()
            post_ids = {id(p) for p in self.model.parameters() if p.requires_grad}
            if post_ids != self._spgc_last_trainable_ids:
                self._rebuild_spgc_optimizer(consolidate=False)

        return {
            "task_loss": task_loss_val,
            "rehearsal_loss": 0.0,
            "distillation_loss": 0.0,
            "total_loss": float(total_loss.detach().item()),
            "novelty_score": input_novelty_score,
            "mutation_report": mutation_report,
            "lifecycle_events": out.get("lifecycle_events", []),
            "skipped_backward": False,
            "flips": flips,
            "auto_consolidated": auto_consolidated,
        }

    def flush_spgc_queue(self, output_head: str = "token") -> Dict[str, Any]:
        """Flush any buffered micro-queue surprises at epoch/phase boundaries."""
        if not self.enable_spgc or self.spgc_gate is None or not self.spgc_gate.queue_x:
            return {
                "task_loss": 0.0,
                "rehearsal_loss": 0.0,
                "distillation_loss": 0.0,
                "total_loss": 0.0,
                "novelty_score": 0.0,
                "mutation_report": None,
                "skipped_backward": True,
                "flips": 0,
            }
        dev = next(self.model.parameters()).device
        if self.spgc_gate.queue_m:
            xb, yb, mb = self.spgc_gate.enqueue_or_flush_masked(None, None, None, force_flush=True, device=dev)
        else:
            xb, yb = self.spgc_gate.enqueue_or_flush(None, None, force_flush=True, device=dev)
            mb = None
        if xb is None or yb is None:
            return {
                "task_loss": 0.0,
                "rehearsal_loss": 0.0,
                "distillation_loss": 0.0,
                "total_loss": 0.0,
                "novelty_score": 0.0,
                "mutation_report": None,
                "skipped_backward": True,
                "flips": 0,
            }
        return self._run_spgc_backward_and_step(xb, yb, mb, output_head=output_head, flips=0)

    def _spgc_step(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        output_head: str = "token",
        source_id: str = "default",
        allow_core_write: bool = True,
        skip_familiar: bool = True,
        enable_quarantine: bool = True,
        force_flush: bool = False,
    ) -> Dict[str, Any]:
        # During early acquisition warmup (before `spgc_min_steps_before_skip`), run direct dense updates
        # with zero screening forward overhead so grokking / representation formation is never prematurely halted.
        in_acquire = (not isinstance(self.optimizer, MuonBONGOptimizer)) or (self.optimizer.mode == "acquire")
        if in_acquire and self.global_step < self.spgc_min_steps_before_skip:
            return self._run_spgc_backward_and_step(x, targets, None, output_head=output_head, flips=0)

        was_training = self.model.training
        self.model.eval()
        ns_proj = getattr(self, "null_space_projector", None)
        prev_ns_track = ns_proj.is_tracking if ns_proj is not None else False
        if ns_proj is not None:
            ns_proj.is_tracking = False
        try:
            with torch.no_grad():
                screen_out = self._forward_replay(x, output_head)
                raw_out = screen_out["output"]
                B = x.size(0)
                is_continuous = (output_head == "continuous") or targets.is_floating_point()

                xb: Optional[torch.Tensor] = None
                yb: Optional[torch.Tensor] = None
                mb: Optional[torch.Tensor] = None
                flips = 0

                if is_continuous:
                    z_lat = self._extract_spgc_latents(screen_out, B=B, T=1, device=x.device)
                    x_act, y_act, flips = self.spgc_gate.triage_continuous(
                        x=x,
                        y=targets,
                        pred=raw_out,
                        z_latent=z_lat,
                        source_id=source_id,
                        skip_familiar=skip_familiar,
                        enable_quarantine=enable_quarantine,
                    )
                    xb, yb = self.spgc_gate.enqueue_or_flush(
                        x_act,
                        y_act,
                        min_flush=self.spgc_min_flush,
                        force_flush=force_flush,
                        device=x.device,
                    )
                else:
                    if raw_out.dim() == 4:
                        flat_logits = raw_out.reshape(B, -1, raw_out.size(-1))
                    elif raw_out.dim() == 3:
                        flat_logits = raw_out
                    else:
                        flat_logits = raw_out.unsqueeze(1)

                    use_tok_gating = (
                        self.spgc_enable_token_gating
                        and targets.dim() >= 2
                        and flat_logits.size(1) > 1
                    )
                    if use_tok_gating:
                        flat_t = targets.reshape(B, -1)
                        t_min = min(flat_logits.size(1), flat_t.size(1))
                        valid_mask = flat_t[:, :t_min] != BYTE_PAD_ID
                        z_tok = self._extract_spgc_latents(screen_out, B=B, T=t_min, device=x.device)
                        if bool(valid_mask.any().item()):
                            v_logits = flat_logits[:, :t_min, :][valid_mask]
                            v_targets = flat_t[:, :t_min][valid_mask]
                            v_lat = z_tok[valid_mask]
                            _, _, flips, v_active = self.spgc_gate.triage_batch(
                                x=v_targets,
                                y=v_targets,
                                logits=v_logits,
                                z_latent=v_lat,
                                source_id=source_id,
                                allow_core_write=False,
                                skip_familiar=skip_familiar,
                                enable_quarantine=enable_quarantine,
                                compact=False,
                                return_active_mask=True,
                            )
                            tok_active = torch.zeros(B, t_min, dtype=torch.bool, device=x.device)
                            tok_active[valid_mask] = v_active
                            seq_keep = tok_active.any(dim=1)
                            if bool(seq_keep.any().item()):
                                x_sel = x[seq_keep]
                                tgt_sel = targets[seq_keep]
                                m_sel = tok_active[seq_keep]
                            else:
                                x_sel, tgt_sel, m_sel = None, None, None
                        else:
                            x_sel, tgt_sel, m_sel = None, None, None

                        xb, yb, mb = self.spgc_gate.enqueue_or_flush_masked(
                            x_sel,
                            tgt_sel,
                            m_sel,
                            min_flush=self.spgc_min_flush,
                            force_flush=force_flush,
                            device=x.device,
                        )
                    else:
                        if targets.dim() >= 2:
                            flat_t = targets.reshape(B, -1)
                            t_min = min(flat_logits.size(1), flat_t.size(1))
                            q_logits = flat_logits[:, t_min - 1, :]
                            q_targets = flat_t[:, t_min - 1]
                        else:
                            q_logits = flat_logits[:, -1, :]
                            q_targets = targets
                        z_lat = self._extract_spgc_latents(screen_out, B=B, T=1, device=x.device)
                        _, _, flips, active_mask = self.spgc_gate.triage_batch(
                            x=x,
                            y=q_targets,
                            logits=q_logits,
                            z_latent=z_lat,
                            source_id=source_id,
                            allow_core_write=allow_core_write and self.spgc_store_samples,
                            skip_familiar=skip_familiar,
                            enable_quarantine=enable_quarantine,
                            compact=False,
                            return_active_mask=True,
                        )
                        if bool(active_mask.any().item()):
                            x_sel = x[active_mask]
                            tgt_sel = targets[active_mask]
                        else:
                            x_sel, tgt_sel = None, None
                        xb, yb = self.spgc_gate.enqueue_or_flush(
                            x_sel,
                            tgt_sel,
                            min_flush=self.spgc_min_flush,
                            force_flush=force_flush,
                            device=x.device,
                        )
        finally:
            if ns_proj is not None:
                ns_proj.is_tracking = prev_ns_track
            if was_training:
                self.model.train()

        if xb is None or yb is None:
            self.global_step += 1
            return {
                "task_loss": 0.0,
                "rehearsal_loss": 0.0,
                "distillation_loss": 0.0,
                "total_loss": 0.0,
                "novelty_score": 0.0,
                "mutation_report": None,
                "lifecycle_events": [],
                "skipped_backward": True,
                "flips": flips,
                "auto_consolidated": False,
            }

        return self._run_spgc_backward_and_step(xb, yb, mb, output_head=output_head, flips=flips)

    def step(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        output_head: str = "token",
        source_id: str = "default",
    ) -> Dict[str, Any]:
        if self.enable_spgc and self.spgc_gate is not None:
            return self._spgc_step(x, targets, output_head=output_head, source_id=source_id)
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(x, output_head=output_head)
        input_novelty_score = (
            float(out["novelty_score"].item())
            if "novelty_score" in out
            else (
                float(self.model.manifold_corr.last_novelty_score.item())
                if hasattr(self.model, "manifold_corr")
                else 0.0
            )
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
        if getattr(self, "null_space_projector", None) is not None:
            self.null_space_projector.project_gradients()
        self.optimizer.step()

        if self.enable_local_lda and hasattr(self.model, "layers"):
            with torch.no_grad():
                slot_feats = self.model.decoder.extract_slot_features(
                    out["decoder_latent"], out.get("byte_residuals")
                )
                assigned = getattr(self.model, "last_assigned_experts", None)
                for layer in self.model.layers:
                    if hasattr(layer, "moe") and getattr(layer.moe, "enable_local_lda", False):
                        if assigned is not None and assigned.numel() == x.size(0):
                            for exp_idx, expert in enumerate(layer.moe.experts):
                                if not bool(layer.moe.expert_frozen[exp_idx]):
                                    seq_mask = (assigned == exp_idx).nonzero(as_tuple=True)[0]
                                    if len(seq_mask) > 0:
                                        layer.moe.experts[exp_idx].absorb(
                                            slot_feats[seq_mask], targets[seq_mask], ignore_index=BYTE_PAD_ID
                                        )
                        else:
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
        if (
            hasattr(self.model, "execute_graph_mutation")
            and not getattr(self.model, "enable_setgate", False)
            and self.global_step % self.mutation_check_interval == 0
        ):
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

    def domain_transition(
        self,
        new_domain_id: Optional[int] = None,
        spawn_new_expert: bool = True,
    ) -> None:
        """
        Transition to a new domain:
        - Update null-space projection basis for completed domain.
        - Freeze all active experts from the completed domain.
        - Spawn a dedicated expert for the new domain in each layer if requested.
        - If freeze_backbone_on_domain (and not using always-plastic null-space projection),
          freeze encoder, attention, and decoder.
        - Rebuild the optimizer for trainable parameters only.
        """
        if not hasattr(self.model, "layers"):
            return

        if getattr(self, "null_space_projector", None) is not None and new_domain_id is not None and new_domain_id >= 1:
            self.null_space_projector.update_basis_for_domain(new_domain_id - 1)

        for layer in self.model.layers:
            if hasattr(layer, "moe"):
                for idx, is_frozen in enumerate(layer.moe.expert_frozen):
                    if not bool(is_frozen):
                        layer.moe.freeze_expert(idx)
                        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "freeze_expert_head"):
                            self.model.decoder.freeze_expert_head(idx)
                if spawn_new_expert:
                    layer.moe.spawn_expert()

        if spawn_new_expert and hasattr(self.model, "decoder") and hasattr(self.model.decoder, "spawn_expert_head"):
            self.model.decoder.spawn_expert_head()

        if self.freeze_backbone_on_domain and not self.enable_null_space and (new_domain_id is None or new_domain_id >= 1):
            if hasattr(self.model, "encoder"):
                for p in self.model.encoder.parameters():
                    p.requires_grad_(False)
            if hasattr(self.model, "manifold_corr"):
                for p in self.model.manifold_corr.parameters():
                    p.requires_grad_(False)
            for layer in self.model.layers:
                for sub in ("norm1", "attn", "norm2", "alpha_residual"):
                    if hasattr(layer, sub):
                        val = getattr(layer, sub)
                        if isinstance(val, nn.Parameter):
                            val.requires_grad_(False)
                        elif isinstance(val, nn.Module):
                            for p in val.parameters():
                                p.requires_grad_(False)
            if hasattr(self.model, "decoder"):
                if hasattr(self.model.decoder, "proj_real"):
                    for p in self.model.decoder.proj_real.parameters():
                        p.requires_grad_(False)
                if hasattr(self.model.decoder, "proj_imag"):
                    for p in self.model.decoder.proj_imag.parameters():
                        p.requires_grad_(False)
                if hasattr(self.model.decoder, "final_norm"):
                    for p in self.model.decoder.final_norm.parameters():
                        p.requires_grad_(False)
                if hasattr(self.model.decoder, "continuous_head"):
                    for p in self.model.decoder.continuous_head.parameters():
                        p.requires_grad_(False)
                if hasattr(self.model.decoder, "expert_heads") and len(self.model.decoder.expert_heads) > 0 and spawn_new_expert:
                    for p in self.model.decoder.expert_heads[-1].parameters():
                        p.requires_grad_(True)

        active_params = [p for p in self.model.parameters() if p.requires_grad]
        if active_params:
            if isinstance(self.optimizer, MuonBONGOptimizer):
                self._rebuild_spgc_optimizer(consolidate=True, new_weight_decay=0.01)
            else:
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
            assigned = getattr(self.model, "last_assigned_experts", None)
            slot_feats = self.model.decoder.extract_slot_features(
                out["decoder_latent"], out.get("byte_residuals")
            )
            if assigned is not None and assigned.numel() == x.size(0):
                n_slots = slot_feats.size(1) * slot_feats.size(2)
                all_logits = torch.zeros(x.size(0), n_slots, 256, device=x.device)
                for exp_idx, expert in enumerate(last_moe.experts):
                    seq_mask = (assigned == exp_idx).nonzero(as_tuple=True)[0]
                    if len(seq_mask) > 0 and hasattr(expert, "predict_logits"):
                        exp_logits = expert.predict_logits(slot_feats[seq_mask])
                        if exp_logits is not None:
                            all_logits[seq_mask] = exp_logits.reshape(len(seq_mask), n_slots, 256)
                if self.fast_memory_blend < 1.0:
                    slow_reshaped = slow_logits.view(x.size(0), n_slots, 256)
                    slow_norm = F.log_softmax(slow_reshaped, dim=-1)
                    fast_norm = F.log_softmax(all_logits, dim=-1)
                    w = self.fast_memory_blend
                    return (1.0 - w) * slow_norm + w * fast_norm
                return all_logits

            chosen_idx = getattr(last_moe, "last_routed_expert", 0)
            if chosen_idx < len(last_moe.experts):
                expert = last_moe.experts[chosen_idx]
                if hasattr(expert, "predict_logits"):
                    fast_logits = expert.predict_logits(slot_feats)
                    if fast_logits is not None:
                        if self.fast_memory_blend < 1.0:
                            n_slots = slot_feats.size(1) * slot_feats.size(2)
                            slow_reshaped = slow_logits.view(x.size(0), n_slots, 256)
                            slow_norm = F.log_softmax(slow_reshaped, dim=-1)
                            fast_norm = F.log_softmax(fast_logits, dim=-1)
                            w = self.fast_memory_blend
                            return (1.0 - w) * slow_norm + w * fast_norm
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
