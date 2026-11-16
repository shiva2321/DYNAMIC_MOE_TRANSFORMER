"""
Autonomous Expert Lifecycle Manager (M02 + M05 + M21 + M08-safe pruning).

Manages self-triggered expert growth, consolidation, dormancy/revival, and pruning
for RealSelfStabilizingDMoE without requiring an external script to manually call
spawn_expert() or freeze_expert().

Lifecycle states per expert:
    0 = ACTIVE  : plastic, currently accepting gradient updates
    1 = FROZEN  : consolidated, read-only, participates in routing
    2 = DORMANT : offloaded/hibernated after prolonged idleness; centroid kept for revival
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.novelty import NoveltyVerdict, UnifiedNoveltyDetector


STATE_ACTIVE = 0
STATE_FROZEN = 1
STATE_DORMANT = 2


class AutonomousExpertLifecycleManager(nn.Module):
    """
    Monitors routing similarity, latent surprise, EMA usage, and idle age to:
    1. Consolidate (freeze) mature active experts when a new regime arrives.
    2. Self-trigger spawn (or revive a dormant expert) using two-signal novelty
       (input similarity + surprise) gated by an anti-fragmentation confirmation window.
    3. Downgrade long-idle frozen experts to DORMANT (zero compute/routing overhead).
    4. Prune dead/oldest dormant experts cleanly under capacity pressure (max_experts).
    """

    def __init__(
        self,
        d_model: int = 32,
        max_experts: int = 8,
        min_mature_steps: int = 25,
        novel_confirm_steps: int = 3,
        novel_surprise_threshold: float = 0.45,
        weak_parent_surprise: float = 0.65,
        dormant_age: int = 60,
        prune_age: int = 150,
        min_usage_threshold: float = 0.02,
        revival_sim_threshold: float = 0.80,
        ema_decay: float = 0.95,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.max_experts = int(max_experts)
        self.min_mature_steps = int(min_mature_steps)
        self.novel_confirm_steps = int(novel_confirm_steps)
        self.novel_surprise_threshold = float(novel_surprise_threshold)
        self.weak_parent_surprise = float(weak_parent_surprise)
        self.dormant_age = int(dormant_age)
        self.prune_age = int(prune_age)
        self.min_usage_threshold = float(min_usage_threshold)
        self.revival_sim_threshold = float(revival_sim_threshold)
        self.ema_decay = float(ema_decay)

        self.novelty_detector = UnifiedNoveltyDetector(
            d_model=self.d_model,
            surp_threshold=self.novel_surprise_threshold,
        )
        self.last_novelty_verdict: Optional[NoveltyVerdict] = None

        self.expert_states: List[int] = [STATE_ACTIVE]
        self.expert_steps_trained: List[int] = [0]
        self.expert_converged: List[bool] = [False]
        self.novel_streak: int = 0
        self.cooldown_remaining: int = 0
        self.should_defer_update: bool = False
        self.total_spawns: int = 0
        self.total_revivals: int = 0
        self.total_prunes: int = 0
        self.last_events: List[str] = []

    def sync_num_experts(self, num_experts: int, expert_frozen: torch.Tensor) -> None:
        """Ensure internal state lists match the model's current expert count."""
        while len(self.expert_states) < num_experts:
            idx = len(self.expert_states)
            is_f = bool(expert_frozen[idx].item()) if idx < expert_frozen.numel() else False
            self.expert_states.append(STATE_FROZEN if is_f else STATE_ACTIVE)
            self.expert_steps_trained.append(0)
            self.expert_converged.append(is_f)
        if len(self.expert_states) > num_experts:
            self.expert_states = self.expert_states[:num_experts]
            self.expert_steps_trained = self.expert_steps_trained[:num_experts]
            self.expert_converged = self.expert_converged[:num_experts]

    @torch.no_grad()
    def update_usage_and_age(
        self,
        moe_layer: nn.Module,
        assigned_experts: torch.Tensor,
    ) -> None:
        """Update EMA expert_usage and idle expert_age buffers on a MoE layer."""
        num_experts = len(moe_layer.experts)
        if num_experts == 0:
            return
        B = max(1, assigned_experts.numel())
        counts = torch.zeros(num_experts, device=moe_layer.expert_usage.device, dtype=torch.float32)
        for e_idx in range(num_experts):
            counts[e_idx] = float((assigned_experts == e_idx).sum().item()) / float(B)

        moe_layer.expert_usage.mul_(self.ema_decay).add_((1.0 - self.ema_decay) * counts)
        idle_mask = counts == 0.0
        moe_layer.expert_age = torch.where(
            idle_mask,
            moe_layer.expert_age + 1.0,
            torch.zeros_like(moe_layer.expert_age),
        )

    @torch.no_grad()
    def check_dormant_revival(
        self,
        seq_sig: torch.Tensor,
        expert_exemplars: List[Optional[torch.Tensor]],
        sim_threshold: Optional[float] = None,
    ) -> Optional[int]:
        """
        Check if an incoming batch closely matches a DORMANT expert's centroid.
        If so, revive that expert to FROZEN so it can serve its domain again.
        """
        if not expert_exemplars:
            return None
        batch_cent = F.normalize(seq_sig.mean(dim=0, keepdim=True).float(), p=2, dim=-1)
        best_sim = -1.0
        best_idx: Optional[int] = None
        for idx, state in enumerate(self.expert_states):
            if state != STATE_DORMANT or idx >= len(expert_exemplars):
                continue
            ex = expert_exemplars[idx]
            if ex is None or ex.numel() == 0:
                continue
            cent = F.normalize(ex.mean(dim=0, keepdim=True).float(), p=2, dim=-1).to(batch_cent.device)
            sim = float((batch_cent @ cent.T).item())
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        thresh = float(self.revival_sim_threshold if sim_threshold is None else sim_threshold)
        if best_idx is not None and best_sim >= thresh:
            self.expert_states[best_idx] = STATE_FROZEN
            self.total_revivals += 1
            self.last_events.append(f"revived_expert_{best_idx}_sim_{best_sim:.3f}")
            return best_idx
        return None

    def step_lifecycle(
        self,
        model: Any,
        seq_sig: torch.Tensor,
        assigned_experts: torch.Tensor,
        is_novel_input: torch.Tensor,
        mean_surprise: Optional[float] = None,
        token_latents: Optional[torch.Tensor] = None,
        token_surprise: Optional[torch.Tensor] = None,
        token_nll: Optional[torch.Tensor] = None,
        is_training: bool = True,
    ) -> Tuple[int, List[str]]:
        """
        Execute one autonomous lifecycle tick:
        - Update usage/age and training step counters.
        - Detect two-signal novelty (input novelty OR high surprise on frozen experts).
        - Trigger freeze + spawn when confirmed past cooldown.
        - Downgrade idle frozen experts to DORMANT.
        - Prune oldest dormant/lowest-utility experts if over capacity or past prune_age.

        Returns:
            target_expert_idx: expert index to route training updates to (if training).
            events: list of lifecycle events triggered on this step.
        """
        self.last_events = []
        moe0 = model.layers[0].moe
        self.sync_num_experts(len(moe0.experts), moe0.expert_frozen)

        # 1. Update usage and age across all MoE layers
        for layer in model.layers:
            self.update_usage_and_age(layer.moe, assigned_experts)

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        active_indices = [i for i, s in enumerate(self.expert_states) if s == STATE_ACTIVE]
        if not active_indices:
            # Ensure at least one active expert or pick highest-usage non-dormant
            active_idx = 0
        else:
            active_idx = active_indices[0]

        if is_training:
            self.expert_steps_trained[active_idx] += 1
            # Schmitt-trigger hysteresis: only latch convergence once surprise drops
            # well below novel_surprise_threshold (0.75x) after initial warmup steps,
            # preventing stochastic mini-batch jitter around the threshold from latching
            # convergence and immediately firing a false novelty alarm.
            if (
                mean_surprise is not None
                and self.expert_steps_trained[active_idx] >= max(5, self.min_mature_steps // 2)
                and mean_surprise < self.novel_surprise_threshold * 0.75
            ):
                self.expert_converged[active_idx] = True

        # 2. Multi-scale novelty evaluation (coarse + near-twin + local token nuance)
        self.novelty_detector.sync_num_experts(len(moe0.experts))
        if assigned_experts.numel() > 0:
            exp_counts = torch.bincount(assigned_experts.detach().long().view(-1), minlength=len(moe0.experts))
            routed_idx = int(exp_counts.argmax().item())
        else:
            routed_idx = active_idx
        routed_centroid = (
            moe0.experts[routed_idx].exemplars
            if routed_idx < len(moe0.experts) and moe0.experts[routed_idx].exemplars.numel() > 0
            else None
        )
        e_list = [int(e) for e in assigned_experts.detach().cpu().tolist() if int(e) < len(self.expert_states)]
        frozen_frac = (
            sum(1 for e in e_list if self.expert_states[e] == STATE_FROZEN) / float(len(e_list))
            if e_list
            else 0.0
        )
        routed_to_frozen = bool(e_list) and (frozen_frac >= 0.5)
        has_router_protos = (
            getattr(model, "router_mode", "content") == "causal_proto"
            and hasattr(model, "causal_router")
            and routed_idx < len(model.causal_router.expert_protos)
            and model.causal_router.expert_protos[routed_idx] is not None
            and model.causal_router.expert_protos[routed_idx].numel() > 0
        )
        has_reference = (routed_centroid is not None) or has_router_protos
        input_novel_frac = (
            float(is_novel_input.float().mean().item())
            if (has_reference and is_novel_input.numel() > 0)
            else 0.0
        )
        input_novel_batch = input_novel_frac >= 0.5
        routed_converged = (
            self.expert_converged[routed_idx]
            if routed_idx < len(self.expert_converged)
            else False
        )
        eval_sig = seq_sig
        eval_latents = token_latents
        eval_surprise = token_surprise
        eval_nll = token_nll
        eval_mean_surprise = mean_surprise
        if (
            routed_to_frozen
            and (not input_novel_batch)
            and assigned_experts.numel() == seq_sig.size(0)
        ):
            routed_mask = assigned_experts == routed_idx
            if bool(routed_mask.any().item()) and not bool(routed_mask.all().item()):
                rm_cpu = routed_mask.to(seq_sig.device)
                eval_sig = seq_sig[rm_cpu]
                if token_latents is not None:
                    eval_latents = token_latents[routed_mask.to(token_latents.device)]
                if token_surprise is not None:
                    eval_surprise = token_surprise[routed_mask.to(token_surprise.device)]
                    eval_mean_surprise = float(eval_surprise.float().mean().item())
                if token_nll is not None:
                    eval_nll = token_nll[routed_mask.to(token_nll.device)]

        verdict = self.novelty_detector.evaluate(
            expert_idx=routed_idx,
            seq_sig=eval_sig,
            expert_centroid=routed_centroid,
            token_latents=eval_latents,
            token_surprise=eval_surprise,
            token_nll=eval_nll,
            expert_is_converged=routed_converged,
            update_baseline=is_training,
            external_novel=bool(input_novel_batch and routed_converged),
        )
        self.last_novelty_verdict = verdict

        surprise_novel = (
            eval_mean_surprise is not None
            and eval_mean_surprise >= self.novel_surprise_threshold
            and (routed_to_frozen or self.expert_converged[active_idx])
        )
        macro_verdict_novel = verdict.novelty_type in ("coarse_domain", "near_twin_rule")
        if routed_to_frozen and (not input_novel_batch) and eval_nll is not None:
            if float(eval_nll.float().mean().item()) < 3.5:
                macro_verdict_novel = False
        if verdict.novelty_type == "local_nuance":
            self.last_events.append("local_nuance_detected")
        batch_is_novel = input_novel_batch or surprise_novel or macro_verdict_novel
        in_warmup = (
            is_training
            and (not routed_to_frozen)
            and (not self.expert_converged[active_idx])
            and self.expert_steps_trained[active_idx] < max(6, self.min_mature_steps // 2)
        )

        if is_training and batch_is_novel and (not in_warmup) and self.cooldown_remaining == 0:
            self.novel_streak += 1
        else:
            self.novel_streak = 0

        # 3. M05 Transfer Check & Self-triggered Freeze + Spawn when novelty is confirmed
        active_is_mature = self.expert_steps_trained[active_idx] >= self.min_mature_steps
        should_spawn = (
            is_training
            and self.novel_streak >= self.novel_confirm_steps
            and active_is_mature
            and len(moe0.experts) < self.max_experts
        )
        self.should_defer_update = bool(
            is_training
            and batch_is_novel
            and self.expert_converged[active_idx]
            and len(moe0.experts) < self.max_experts
        )
        if should_spawn and eval_nll is not None and not surprise_novel:
            # M05: if existing converged expert already predicts the new cluster within margin
            # (and NLL trigger did not fire), extend current expert instead of duplicate spawning.
            r_stats = self.novelty_detector.expert_stats[routed_idx]
            if r_stats.count >= 5 and ("nll_mean" in r_stats.means) and (not macro_verdict_novel):
                cur_nll = float(eval_nll.float().mean().item())
                base_nll = r_stats.means.get("nll_mean", 0.0)
                margin = float(getattr(self, "extend_loss_margin", 0.25))
                if cur_nll <= base_nll + margin:
                    should_spawn = False
                    self.novel_streak = 0
                    self.last_events.append(f"extended_expert_{routed_idx}")

        if should_spawn:
            # Consolidate current mature active expert
            for layer in model.layers:
                layer.moe.freeze_expert(active_idx)
            model.decoder.freeze_expert_head(active_idx)
            self.expert_states[active_idx] = STATE_FROZEN
            self.last_events.append(f"froze_expert_{active_idx}")

            # Weak-parent rule (M05): if parent surprise is worse than threshold, fresh init
            fresh_init = mean_surprise is not None and mean_surprise >= self.weak_parent_surprise
            new_idx = -1
            for layer in model.layers:
                new_idx = layer.moe.spawn_expert(source_idx=active_idx, fresh_init=fresh_init)
            model.decoder.spawn_expert_head()
            if hasattr(model, "causal_router") and model.causal_router is not None:
                model.causal_router.sync_num_experts(len(moe0.experts))

            self.expert_states.append(STATE_ACTIVE)
            self.expert_steps_trained.append(0)
            self.expert_converged.append(False)
            self.novel_streak = 0
            self.cooldown_remaining = self.min_mature_steps // 2
            self.total_spawns += 1
            prev_active_idx = active_idx
            active_idx = new_idx
            self.last_events.append(
                f"spawned_expert_{new_idx}_{'fresh' if fresh_init else f'from_{prev_active_idx}'}"
            )

        # 4. Downgrade long-idle FROZEN experts to DORMANT
        num_non_dormant = sum(1 for s in self.expert_states if s != STATE_DORMANT)
        for idx in range(len(self.expert_states)):
            if (
                self.expert_states[idx] == STATE_FROZEN
                and num_non_dormant > 1
                and float(moe0.expert_age[idx].item()) >= self.dormant_age
                and float(moe0.expert_usage[idx].item()) < self.min_usage_threshold
            ):
                self.expert_states[idx] = STATE_DORMANT
                num_non_dormant -= 1
                self.last_events.append(f"dormant_expert_{idx}")

        # 5. Prune experts that exceeded prune_age while DORMANT (or over max_experts)
        # Prune at most 1 expert per step in reverse index order to keep indices clean
        prune_candidate: Optional[int] = None
        if len(self.expert_states) > 1:
            dormant_candidates = [
                (float(moe0.expert_age[i].item()), i)
                for i, s in enumerate(self.expert_states)
                if s == STATE_DORMANT and float(moe0.expert_age[i].item()) >= self.prune_age
            ]
            if dormant_candidates:
                dormant_candidates.sort(reverse=True)
                prune_candidate = dormant_candidates[0][1]
            elif len(self.expert_states) > self.max_experts:
                # Capacity pressure: prune lowest-usage non-active expert
                non_active = [
                    (float(moe0.expert_usage[i].item()), -float(moe0.expert_age[i].item()), i)
                    for i, s in enumerate(self.expert_states)
                    if s != STATE_ACTIVE
                ]
                if non_active:
                    non_active.sort()
                    prune_candidate = non_active[0][2]

        if prune_candidate is not None:
            for layer in model.layers:
                layer.moe.prune_expert(prune_candidate)
            model.decoder.prune_expert_head(prune_candidate)
            if hasattr(model, "causal_router") and model.causal_router is not None:
                model.causal_router.prune_expert(prune_candidate)
            self.expert_states.pop(prune_candidate)
            self.expert_steps_trained.pop(prune_candidate)
            self.expert_converged.pop(prune_candidate)
            self.novelty_detector.prune_expert(prune_candidate)
            self.total_prunes += 1
            self.last_events.append(f"pruned_expert_{prune_candidate}")
            # Recompute active_idx after pop
            active_indices = [i for i, s in enumerate(self.expert_states) if s == STATE_ACTIVE]
            active_idx = active_indices[0] if active_indices else 0

        return active_idx, list(self.last_events)

