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
        extend_loss_margin: float = 0.25,
        preserve_input_novel_on_extend: bool = False,
        defer_on_novel_streak: bool = False,
        enable_mixed_split: bool = False,
        mixed_split_gap: float = 0.12,
        max_coactive_experts: int = 4,
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
        self.extend_loss_margin = float(extend_loss_margin)
        self.preserve_input_novel_on_extend = bool(preserve_input_novel_on_extend)
        self.defer_on_novel_streak = bool(defer_on_novel_streak)
        self.enable_mixed_split = bool(enable_mixed_split)
        self.mixed_split_gap = float(mixed_split_gap)
        self.max_coactive_experts = max(2, int(max_coactive_experts))

        self.novelty_detector = UnifiedNoveltyDetector(
            d_model=self.d_model,
            surp_threshold=self.novel_surprise_threshold,
        )
        self.last_novelty_verdict: Optional[NoveltyVerdict] = None

        self.expert_states: List[int] = [STATE_ACTIVE]
        self.expert_steps_trained: List[int] = [0]
        self.expert_converged: List[bool] = [False]
        self.novel_streak: int = 0
        self.mixed_streak: int = 0
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
        proto_sig: Optional[torch.Tensor] = None,
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
        moe_ref_idx = int(getattr(model, "_moe_ref_layer_idx", 0))
        moe0 = model.layers[moe_ref_idx].moe
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
            for ai in active_indices:
                if len(active_indices) == 1 or bool((assigned_experts == ai).any().item()):
                    self.expert_steps_trained[ai] += 1
                    a_stats = (
                        self.novelty_detector.expert_stats[ai]
                        if ai < len(self.novelty_detector.expert_stats)
                        else None
                    )
                    has_cal_nll = (
                        token_nll is not None
                        and a_stats is not None
                        and a_stats.key_counts.get("nll_mean", 0) >= 5
                    )
                    if (
                        mean_surprise is not None
                        and self.expert_steps_trained[ai] >= max(5, self.min_mature_steps // 2)
                        and mean_surprise < self.novel_surprise_threshold * 0.75
                    ) or (
                        self.expert_steps_trained[ai] >= self.min_mature_steps
                        and has_cal_nll
                    ):
                        self.expert_converged[ai] = True

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

        routed_past_warmup = (
            routed_converged
            or (
                routed_idx < len(self.expert_steps_trained)
                and self.expert_steps_trained[routed_idx] >= max(5, self.min_mature_steps // 2)
            )
        )
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
            record_outcome_baseline=routed_past_warmup,
        )
        self.last_novelty_verdict = verdict

        r_stats = self.novelty_detector.expert_stats[routed_idx]
        eff_surp_thresh = self.novel_surprise_threshold
        if r_stats.key_counts.get("surp_mean", 0) >= 5 and "surp_mean" in r_stats.means:
            surp_std = max(r_stats.vars.get("surp_mean", 0.0025) ** 0.5, 0.04)
            eff_surp_thresh = max(
                self.novel_surprise_threshold,
                r_stats.means["surp_mean"] + max(0.08, 2.75 * surp_std),
            )
        surprise_novel = (
            eval_mean_surprise is not None
            and eval_mean_surprise >= eff_surp_thresh
            and (routed_to_frozen or self.expert_converged[active_idx])
        )
        macro_verdict_novel = verdict.novelty_type in ("coarse_domain", "near_twin_rule")
        if routed_to_frozen and (not input_novel_batch) and eval_nll is not None:
            if float(eval_nll.float().mean().item()) < 3.5:
                macro_verdict_novel = False
        if verdict.novelty_type == "local_nuance":
            self.last_events.append("local_nuance_detected")

        # M05 Outcome-Based Transfer Rule: if existing calibrated expert (or current active expert)
        # already predicts the candidate batch within scale-invariant margin of its reference NLL,
        # extend rather than spawning a duplicate expert.
        margin = float(getattr(self, "extend_loss_margin", 0.25))
        allow_extend_check = (not self.preserve_input_novel_on_extend) or (not input_novel_batch)
        if (
            allow_extend_check
            and eval_nll is not None
            and r_stats.key_counts.get("nll_mean", 0) >= 5
            and "nll_mean" in r_stats.means
        ):
            cur_nll = float(eval_nll.float().mean().item())
            base_nll = r_stats.means["nll_mean"]
            eff_margin = max(margin, 0.25 * base_nll)
            if cur_nll <= base_nll + eff_margin:
                if (input_novel_batch or surprise_novel or macro_verdict_novel) and is_training:
                    self.last_events.append(f"extended_expert_{routed_idx}")
                    if eval_latents is not None:
                        self.novelty_detector.update_token_prototypes(routed_idx, eval_latents)
                input_novel_batch = False
                surprise_novel = False
                macro_verdict_novel = False

        if (
            allow_extend_check
            and routed_idx != active_idx
            and eval_nll is not None
            and active_idx < len(self.novelty_detector.expert_stats)
        ):
            a_stats_chk = self.novelty_detector.expert_stats[active_idx]
            if a_stats_chk.key_counts.get("nll_mean", 0) >= 5 and "nll_mean" in a_stats_chk.means:
                cur_nll_a = float(eval_nll.float().mean().item())
                a_base = a_stats_chk.means["nll_mean"]
                if cur_nll_a <= a_base + max(margin, 0.25 * a_base):
                    input_novel_batch = False
                    surprise_novel = False
                    macro_verdict_novel = False

        batch_is_novel = input_novel_batch or surprise_novel or macro_verdict_novel
        in_warmup = (
            is_training
            and (not self.expert_converged[active_idx])
            and self.expert_steps_trained[active_idx] < max(6, self.min_mature_steps // 2)
        )

        if is_training and batch_is_novel and (not in_warmup) and self.cooldown_remaining == 0:
            self.novel_streak += 1
        else:
            self.novel_streak = 0

        # 3. M05 Transfer Check & Self-triggered Freeze + Spawn when novelty is confirmed
        active_is_mature = self.expert_steps_trained[active_idx] >= self.min_mature_steps
        can_freeze_spawn = (
            not self.enable_mixed_split
            or len(active_indices) >= self.max_coactive_experts
        )
        should_spawn = (
            is_training
            and can_freeze_spawn
            and self.novel_streak >= self.novel_confirm_steps
            and active_is_mature
            and len(moe0.experts) < self.max_experts
        )
        defer_trigger = batch_is_novel or (self.defer_on_novel_streak and self.novel_streak >= 1)
        self.should_defer_update = bool(
            is_training
            and defer_trigger
            and can_freeze_spawn
            and self.expert_converged[active_idx]
            and len(moe0.experts) < self.max_experts
        )

        # 3b. Optional Mixed-Stream Mitosis: split bimodal active clusters without freezing
        should_mixed_split = False
        split_parent_idx = active_idx
        split_idx_global: Optional[torch.Tensor] = None
        split_mask_sub: Optional[torch.Tensor] = None
        split_src = proto_sig if proto_sig is not None else seq_sig
        if (
            self.enable_mixed_split
            and is_training
            and (not should_spawn)
            and (not batch_is_novel)
            and self.cooldown_remaining == 0
            and len(moe0.experts) < min(self.max_experts, self.max_coactive_experts)
            and len(active_indices) < self.max_coactive_experts
            and split_src.size(0) >= 8
        ):
            # Pick most populated mature active expert in this batch to check for internal bimodality
            best_cnt, cand_p = 0, None
            for ai in active_indices:
                if self.expert_steps_trained[ai] >= self.min_mature_steps:
                    cnt = int((assigned_experts == ai).sum().item())
                    if cnt > best_cnt:
                        best_cnt, cand_p = cnt, ai
            if cand_p is not None and best_cnt >= 6:
                sub_idx = (assigned_experts == cand_p).nonzero(as_tuple=True)[0]
                p_raw = split_src[sub_idx.to(split_src.device)].detach().float()
                p_cent = F.normalize(p_raw - p_raw.mean(dim=0, keepdim=True), p=2, dim=-1, eps=1e-6)
                sim_m = p_cent @ p_cent.T
                pair_min = torch.argmin(sim_m)
                i0 = int((pair_min // sim_m.size(1)).item())
                i1 = int((pair_min % sim_m.size(1)).item())
                c0, c1 = p_cent[i0], p_cent[i1]
                a_mask = (p_cent @ c0) >= (p_cent @ c1)
                for _ in range(3):
                    if bool(a_mask.any()) and bool((~a_mask).any()):
                        c0 = F.normalize(p_cent[a_mask].mean(dim=0), p=2, dim=-1, eps=1e-6)
                        c1 = F.normalize(p_cent[~a_mask].mean(dim=0), p=2, dim=-1, eps=1e-6)
                        a_mask = (p_cent @ c0) >= (p_cent @ c1)
                frac_b = float((~a_mask).float().mean().item())
                if 0.25 <= frac_b <= 0.75:
                    intra0 = float((p_cent[a_mask] @ c0).mean().item())
                    intra1 = float((p_cent[~a_mask] @ c1).mean().item())
                    cross = float((c0 @ c1).item())
                    gap = 0.5 * (intra0 + intra1) - max(cross, -1.0)
                    if gap >= self.mixed_split_gap:
                        self.mixed_streak += 1
                        split_parent_idx = cand_p
                        split_idx_global = sub_idx
                        split_mask_sub = ~a_mask
                    else:
                        self.mixed_streak = 0
                else:
                    self.mixed_streak = 0
            else:
                self.mixed_streak = 0
            should_mixed_split = self.mixed_streak >= self.novel_confirm_steps
        elif not batch_is_novel:
            self.mixed_streak = 0

        if should_spawn:
            # Automatically consolidate GPM null-space basis before freezing active expert(s)
            nsp = getattr(model, "null_space_projector", None)
            if nsp is not None and hasattr(nsp, "update_basis_for_domain"):
                nsp.update_basis_for_domain(active_idx)
            if getattr(self, "freeze_embed_after_first", False) and hasattr(model, "byte_embed"):
                model.byte_embed.weight.requires_grad_(False)

            for freeze_i in active_indices:
                for layer in model.layers:
                    layer.moe.freeze_expert(freeze_i)
                    if hasattr(layer, "freeze_expert_adapters"):
                        layer.freeze_expert_adapters(freeze_i)
                model.decoder.freeze_expert_head(freeze_i)
                if getattr(model, "thinker", None) is not None and hasattr(model.thinker, "freeze_expert_adapter"):
                    model.thinker.freeze_expert_adapter(freeze_i)
                self.expert_states[freeze_i] = STATE_FROZEN
                self.last_events.append(f"froze_expert_{freeze_i}")

            # Weak-parent rule (M05): if parent surprise is worse than threshold, fresh init
            fresh_init = mean_surprise is not None and mean_surprise >= self.weak_parent_surprise
            new_idx = -1
            for layer in model.layers:
                new_idx = layer.moe.spawn_expert(source_idx=active_idx, fresh_init=fresh_init)
                if hasattr(layer, "spawn_expert_adapters"):
                    layer.spawn_expert_adapters(source_idx=active_idx, fresh_init=fresh_init)
            model.decoder.spawn_expert_head()
            if getattr(model, "thinker", None) is not None and hasattr(model.thinker, "spawn_expert_adapter"):
                model.thinker.spawn_expert_adapter(source_idx=active_idx, fresh_init=fresh_init)
            if hasattr(model, "causal_router") and model.causal_router is not None:
                model.causal_router.sync_num_experts(len(moe0.experts))

            self.expert_states.append(STATE_ACTIVE)
            self.expert_steps_trained.append(0)
            self.expert_converged.append(False)
            self.novel_streak = 0
            self.mixed_streak = 0
            self.cooldown_remaining = self.min_mature_steps // 2
            self.total_spawns += 1
            prev_active_idx = active_idx
            active_idx = new_idx
            self.last_events.append(
                f"spawned_expert_{new_idx}_{'fresh' if fresh_init else f'from_{prev_active_idx}'}"
            )
        elif should_mixed_split and split_idx_global is not None and split_mask_sub is not None:
            new_idx = -1
            for layer in model.layers:
                new_idx = layer.moe.spawn_expert(source_idx=split_parent_idx, fresh_init=False)
                if hasattr(layer, "spawn_expert_adapters"):
                    layer.spawn_expert_adapters(source_idx=split_parent_idx, fresh_init=False)
            model.decoder.spawn_expert_head()
            if getattr(model, "thinker", None) is not None and hasattr(model.thinker, "spawn_expert_adapter"):
                model.thinker.spawn_expert_adapter(source_idx=split_parent_idx, fresh_init=False)
            if hasattr(model, "causal_router") and model.causal_router is not None:
                model.causal_router.sync_num_experts(len(moe0.experts))
                if proto_sig is not None:
                    sub_p = proto_sig[split_idx_global.to(proto_sig.device)].detach()
                    c_b = sub_p[split_mask_sub.to(sub_p.device)]
                    c_a = sub_p[(~split_mask_sub).to(sub_p.device)]
                    if c_a.numel() > 0:
                        model.causal_router.expert_protos[split_parent_idx] = None
                        if split_parent_idx < len(model.causal_router._sim_counts):
                            model.causal_router._sim_counts[split_parent_idx] = 0
                            model.causal_router._sim_means[split_parent_idx] = 0.0
                            model.causal_router._sim_vars[split_parent_idx] = 0.01
                        model.causal_router.update_prototypes(split_parent_idx, c_a)
                    if c_b.numel() > 0:
                        model.causal_router.update_prototypes(new_idx, c_b)
            self.expert_states.append(STATE_ACTIVE)
            self.expert_steps_trained.append(0)
            self.expert_converged.append(False)
            self.mixed_streak = 0
            self.cooldown_remaining = self.min_mature_steps // 2
            self.total_spawns += 1
            self.last_events.append(f"mixed_split_coactive_expert_{new_idx}_from_{split_parent_idx}")

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
                if hasattr(layer, "prune_expert_adapters"):
                    layer.prune_expert_adapters(prune_candidate)
            model.decoder.prune_expert_head(prune_candidate)
            if getattr(model, "thinker", None) is not None and hasattr(model.thinker, "prune_expert_adapter"):
                model.thinker.prune_expert_adapter(prune_candidate)
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

    def step_contextual_nuance_lifecycle(
        self,
        model: Any,
        nuance_mask: torch.Tensor,
        q_tok: torch.Tensor,
        nuance_score: torch.Tensor,
        top_ids: torch.Tensor,
        top_w: torch.Tensor,
        nuance_buffer: Any,
        *,
        is_training: bool = True,
        cluster_min_count: int = 20,
        cluster_min_cos: float = 0.72,
        coarse_macro_sim_thresh: float = 0.45,
        mean_residual_err: Optional[float] = None,
    ) -> Tuple[int, List[str]]:
        """
        v7 Tier-B Nuance Lifecycle Step:
          - Tracks usage/age over active non-trunk MoE layer.
          - Buffers `nuance_mask` chunk queries in `NuanceClusterBuffer`.
          - Spawns a targeted specialist ONLY when a dense coherent nuance cluster
            accumulates and unresolved error persists (either nuance extension or
            full GPM + freeze macro domain consolidation).
        """
        self.last_events = []
        moe_ref_idx = int(getattr(model, "_moe_ref_layer_idx", 0))
        moe0 = model.layers[moe_ref_idx].moe
        n_exp = len(moe0.experts)
        if n_exp == 0:
            return 0, []
        self.sync_num_experts(n_exp, moe0.expert_frozen)

        # Update usage/age from primary assigned chunks (fallback to active expert 0)
        primary_seq = top_ids[:, 0, 0] if top_ids.dim() >= 3 else top_ids.reshape(-1)
        for layer in model.layers:
            if not getattr(layer.moe, "v7_trunk_only", False) and len(layer.moe.experts) > 0:
                self.update_usage_and_age(layer.moe, primary_seq)

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        active_indices = [i for i, s in enumerate(self.expert_states) if s == STATE_ACTIVE]
        active_idx = active_indices[0] if active_indices else 0
        if is_training:
            for ai in active_indices:
                if ai < len(self.expert_steps_trained):
                    self.expert_steps_trained[ai] += 1

        if (not is_training) or nuance_buffer is None:
            return active_idx, []

        m_flat = nuance_mask.reshape(-1)
        if bool(m_flat.any().item()):
            q_flat = q_tok.detach().reshape(-1, q_tok.size(-1))[m_flat]
            s_flat = nuance_score.detach().reshape(-1)[m_flat]
            cr = getattr(model, "context_router", None)
            if cr is not None:
                c_sim = cr.compute_queries_similarity(q_flat) if hasattr(cr, "compute_queries_similarity") else None
            else:
                c_sim = None
            if c_sim is None and cr is not None and hasattr(cr, "expert_keys"):
                keys_e = torch.nn.functional.normalize(cr.expert_keys[: cr.num_experts].float(), dim=-1)
                q_n = torch.nn.functional.normalize(q_flat.float(), dim=-1)
                c_sim = torch.einsum("nd,emd->nem", q_n, keys_e).max(dim=-1).values.max(dim=-1).values
            nuance_buffer.push(q_flat, err_vals=s_flat, coarse_sims=c_sim)
            self.last_events.append("local_nuance_detected")

        if self.cooldown_remaining > 0 or len(self.expert_states) >= self.max_experts:
            return active_idx, list(self.last_events)

        ready, _, c_vecs, mean_cos, mean_coarse = nuance_buffer.check_coherent_cluster(
            min_count=cluster_min_count,
            min_cos=cluster_min_cos,
        )
        newest_idx = active_indices[-1] if active_indices else active_idx
        mature_ok = (
            newest_idx < len(self.expert_steps_trained)
            and self.expert_steps_trained[newest_idx] >= self.min_mature_steps
        )
        if not (ready and mature_ok and c_vecs is not None):
            return active_idx, list(self.last_events)

        is_macro_shift = mean_coarse < coarse_macro_sim_thresh
        nsp = getattr(model, "null_space_projector", None)
        cr = getattr(model, "context_router", None)
        if is_macro_shift:
            if nsp is not None and hasattr(nsp, "update_basis_for_domain"):
                nsp.update_basis_for_domain(active_idx)
            for freeze_i in list(active_indices):
                for layer in model.layers:
                    layer.moe.freeze_expert(freeze_i)
                    if hasattr(layer, "freeze_expert_adapters"):
                        layer.freeze_expert_adapters(freeze_i)
                if getattr(model, "decoder", None) is not None and hasattr(model.decoder, "freeze_expert_head"):
                    model.decoder.freeze_expert_head(freeze_i)
                if getattr(model, "thinker", None) is not None and hasattr(model.thinker, "freeze_expert_adapter"):
                    model.thinker.freeze_expert_adapter(freeze_i)
                if cr is not None and hasattr(cr, "freeze_expert"):
                    cr.freeze_expert(freeze_i)
                self.expert_states[freeze_i] = STATE_FROZEN
                self.last_events.append(f"froze_expert_{freeze_i}")
        else:
            # Localized nuance extension: keep existing mature specialists co-active unless over capacity
            unfrozen = [i for i, s in enumerate(self.expert_states) if s == STATE_ACTIVE]
            if len(unfrozen) >= self.max_coactive_experts:
                oldest_u = unfrozen[0]
                for layer in model.layers:
                    layer.moe.freeze_expert(oldest_u)
                    if hasattr(layer, "freeze_expert_adapters"):
                        layer.freeze_expert_adapters(oldest_u)
                if cr is not None and hasattr(cr, "freeze_expert"):
                    cr.freeze_expert(oldest_u)
                self.expert_states[oldest_u] = STATE_FROZEN
                self.last_events.append(f"froze_expert_{oldest_u}")

        fresh_init = bool(is_macro_shift)
        new_idx = -1
        for layer in model.layers:
            idx_l = layer.moe.spawn_expert(source_idx=active_idx, fresh_init=fresh_init)
            if not getattr(layer.moe, "v7_trunk_only", False):
                new_idx = idx_l
            if hasattr(layer, "spawn_expert_adapters"):
                layer.spawn_expert_adapters(source_idx=active_idx, fresh_init=fresh_init)
        if getattr(model, "decoder", None) is not None and hasattr(model.decoder, "spawn_expert_head"):
            model.decoder.spawn_expert_head()
        if getattr(model, "thinker", None) is not None and hasattr(model.thinker, "spawn_expert_adapter"):
            model.thinker.spawn_expert_adapter(source_idx=active_idx, fresh_init=fresh_init)
        if cr is not None:
            cr.sync_num_experts(len(moe0.experts))
            if hasattr(cr, "seed_expert_from_vectors"):
                cr.seed_expert_from_vectors(new_idx, c_vecs)

        nuance_buffer.pop_cluster(min_cos=cluster_min_cos)
        self.expert_states.append(STATE_ACTIVE)
        self.expert_steps_trained.append(0)
        self.expert_converged.append(False)
        self.cooldown_remaining = max(5, self.min_mature_steps // 2)
        self.total_spawns += 1
        tag = "spawned_expert" if is_macro_shift else "spawned_nuance"
        self.last_events.append(f"{tag}_{new_idx}")
        return new_idx, list(self.last_events)

