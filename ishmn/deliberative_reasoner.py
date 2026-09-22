"""
Deliberative Reasoner Engine: Imagination-Based Multi-Step Causal Reasoning.

Enables the Universal Cognitive Engine to reason endogenously through imagination:
1. Endogenous Forward Rollout: Projects candidate hypotheses/actions K steps into the future
   using the Causal World Model (W_fwd) and Mixture-of-Experts.
2. Abductive Inversion Check: Validates imagined consequences via W_bwd to detect causal breakdowns.
3. Tri-Tier Memory Grounding: Evaluates resonance with episodic experiences and semantic invariants.
4. Dead-End & Contradiction Pruning: Identifies and eliminates invalid branches in imagination
   before physical action or commitment.
5. Structured Cognitive Trace: Emits full thought trajectories with causal confidence and rationale.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ishmn.geometry import (
    project_to_sphere,
    geodesic_distance_sphere,
    exp_map_sphere
)

@dataclass
class ReasonedBranch:
    branch_id: Any
    score: float
    causal_consistency: float
    memory_alignment: float
    steps_survived: int
    is_dead_end: bool
    is_contradiction: bool
    final_latent: torch.Tensor
    imagined_trajectory: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ReasoningResult:
    selected_branch: Any
    confidence: float
    system_mode: str
    total_candidates: int
    viable_candidates: int
    pruned_dead_ends: int
    pruned_contradictions: int
    ranked_branches: List[ReasonedBranch]
    reasoning_trace: List[Dict[str, Any]]
    explanation: str

class DeliberativeReasoner:
    """
    General Reasoner Engine leveraging the Universal World Model and Tri-Tier Memory.
    Completely domain-agnostic: operates on continuous latents in S^{D-1} x R+ x R^{d_model}.
    """
    def __init__(self, brain: Any):
        self.brain = brain
        self.device = brain.device

    def imagine_branch(
        self,
        z_start: torch.Tensor,
        z_context: torch.Tensor,
        candidate_input: Optional[torch.Tensor] = None,
        lookahead_depth: int = 8,
        goal_u: Optional[torch.Tensor] = None,
        external_evaluator: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Rolls out an imagined forward trajectory for K steps:
        - Simulates transitions: z_{t+1} = W_fwd(z_t, c) + 0.3 * MoE(z_fwd)
        - Computes causal reversibility error at each tick
        - Evaluates geodesic resonance against semantic invariants
        """
        z_curr = z_start.clone()
        c_curr = z_context.clone()
        if candidate_input is not None:
            # Safely align candidate input dimensions
            if candidate_input.size(-1) != z_curr.size(-1):
                if candidate_input.size(-1) < z_curr.size(-1):
                    pad_len = z_curr.size(-1) - candidate_input.size(-1)
                    padded = F.pad(candidate_input, (0, pad_len))
                    z_curr = 0.8 * z_curr + 0.2 * padded
                else:
                    z_curr = 0.8 * z_curr + 0.2 * candidate_input[:, :z_curr.size(-1)]
            else:
                z_curr = 0.8 * z_curr + 0.2 * candidate_input

        # Bound initial state norm
        init_norm = z_curr.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        if (init_norm > 8.0).any():
            z_curr = torch.where(init_norm > 8.0, (z_curr / init_norm) * 8.0, z_curr)

        trajectory = []
        total_causal_err = 0.0
        total_sem_res = 0.0
        is_dead_end = False
        is_contradiction = False
        steps_survived = 0

        # Retrieve semantic memory invariants for grounding
        semantic_anchors = []
        for concept in self.brain.memory.semantic_concepts:
            anchor = concept.get("centroid_u", concept.get("u"))
            if anchor is not None:
                semantic_anchors.append(anchor.to(self.device))

        for step in range(1, lookahead_depth + 1):
            steps_survived = step

            # 1. Forward World Model transition
            z_fwd = self.brain.world_model.forward_transition(z_curr, c_curr)
            u_fwd, r_fwd = self.brain.world_model.project_manifold(z_fwd)

            # 2. Enrich via MoE specialists
            moe_out, _ = self.brain.moe(z_fwd, u_fwd, r_fwd)
            z_next = z_fwd + 0.3 * moe_out

            # Bound z_next norm to preserve numerical stability on Riemannian manifold
            z_norm = z_next.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
            if (z_norm > 8.0).any():
                z_next = torch.where(z_norm > 8.0, (z_next / z_norm) * 8.0, z_next)

            u_next, r_next = self.brain.world_model.project_manifold(z_next)

            # 3. Abductive Inversion Check (Pearl Level 3 Causal Check)
            z_reconstructed = self.brain.world_model.backward_transition(z_next, c_curr)
            u_rec, r_rec = self.brain.world_model.project_manifold(z_reconstructed)
            step_causal_err = (
                geodesic_distance_sphere(u_fwd, u_rec).mean() +
                0.5 * torch.abs(r_fwd - r_rec).mean()
            ).item()
            total_causal_err += step_causal_err

            # 4. Semantic Alignment
            sem_resonance = 0.5
            if semantic_anchors:
                res_vals = [
                    torch.exp(torch.sum(u_next * anchor, dim=-1)).item()
                    for anchor in semantic_anchors
                ]
                sem_resonance = max(res_vals)
            total_sem_res += sem_resonance

            # 5. External domain-specific check (e.g. maze walls/obstacles if provided)
            external_info = {}
            if external_evaluator is not None:
                eval_res = external_evaluator(step, z_next, u_next, r_next)
                external_info = eval_res
                if eval_res.get("is_dead_end", False):
                    is_dead_end = True
                    trajectory.append({
                        "step": step,
                        "causal_error": step_causal_err,
                        "sem_resonance": sem_resonance,
                        "terminated": True,
                        "reason": "dead_end",
                        "external": external_info
                    })
                    break

            # Check for causal contradiction explosion
            if step_causal_err > 2.5:
                is_contradiction = True
                trajectory.append({
                    "step": step,
                    "causal_error": step_causal_err,
                    "sem_resonance": sem_resonance,
                    "terminated": True,
                    "reason": "causal_contradiction",
                    "external": external_info
                })
                break

            trajectory.append({
                "step": step,
                "causal_error": step_causal_err,
                "sem_resonance": sem_resonance,
                "terminated": False,
                "external": external_info
            })

            # Check goal convergence
            if goal_u is not None:
                dist_to_goal = geodesic_distance_sphere(u_next, goal_u).mean().item()
                if dist_to_goal < 0.15:
                    break

            z_curr = z_next

        avg_causal_err = total_causal_err / max(steps_survived, 1)
        causal_consistency = 1.0 / (1.0 + avg_causal_err)

        avg_sem_res = total_sem_res / max(steps_survived, 1)

        return {
            "final_latent": z_curr,
            "steps_survived": steps_survived,
            "avg_causal_error": avg_causal_err,
            "causal_consistency": causal_consistency,
            "semantic_resonance": avg_sem_res,
            "is_dead_end": is_dead_end,
            "is_contradiction": is_contradiction,
            "trajectory": trajectory
        }

    def reason(
        self,
        premise: Any,
        context: Optional[Any] = None,
        candidate_branches: Optional[List[Tuple[Any, Any]]] = None,
        modality: str = "continuous",
        lookahead_depth: int = 12,
        goal: Optional[Any] = None,
        external_branch_evaluator: Optional[Any] = None
    ) -> ReasoningResult:
        """
        Coordinates multi-branch deliberative reasoning:
        1. Encodes premise and context into Riemannian hyperspace.
        2. Recalls resonant experiences and semantic constraints from memory.
        3. Simulates each candidate branch forward in imagination.
        4. Prunes dead ends and causal contradictions.
        5. Computes a multi-criteria decision score and emits a structured ReasoningResult.
        """
        with torch.no_grad():
            # 1. Perception
            u_premise, r_premise, z_premise = self.brain.perceive(premise, modality=modality)
            if context is not None:
                u_ctx, r_ctx, z_ctx = self.brain.perceive(context, modality=modality)
            else:
                z_ctx = torch.zeros_like(z_premise)

            goal_u = None
            if goal is not None:
                goal_u, _, _ = self.brain.perceive(goal, modality=modality)

            # 2. Tri-tier memory recall
            _, mem_meta, _ = self.brain.memory.tri_tier_recall(
                self.brain.mem_state,
                u_query=u_premise,
                r_query=r_premise,
                kappa=torch.clamp(self.brain.kappa_mem, 0.1, 10.0),
                beta_r=torch.clamp(self.brain.beta_mem, 0.01, 5.0)
            )

            # 3. Default candidate branching if none provided
            if not candidate_branches:
                candidate_branches = [("default_continuation", z_premise)]

            # 4. Mental Simulation across candidate branches
            evaluated_branches = []
            pruned_dead_ends = 0
            pruned_contradictions = 0

            for branch_id, branch_input in candidate_branches:
                if isinstance(branch_input, torch.Tensor):
                    b_tensor = branch_input.to(self.device)
                else:
                    _, _, b_tensor = self.brain.perceive(branch_input, modality=modality)

                # Custom branch evaluator if provided
                branch_eval_fn = None
                if external_branch_evaluator is not None:
                    branch_eval_fn = external_branch_evaluator(branch_id)

                sim = self.imagine_branch(
                    z_start=z_premise,
                    z_context=z_ctx,
                    candidate_input=b_tensor,
                    lookahead_depth=lookahead_depth,
                    goal_u=goal_u,
                    external_evaluator=branch_eval_fn
                )

                if sim["is_dead_end"]:
                    pruned_dead_ends += 1
                if sim["is_contradiction"]:
                    pruned_contradictions += 1

                # Abductive counterfactual consistency check (Pearl Level 3: W_bwd(b_latent, c) ~ z_premise)
                if b_tensor.size(-1) != self.brain.d_model:
                    _, _, b_latent = self.brain.perceive(b_tensor, modality="continuous")
                else:
                    b_latent = b_tensor
                z_bwd_cand = self.brain.world_model.backward_transition(b_latent, z_ctx)
                u_bwd_cand, r_bwd_cand = self.brain.world_model.project_manifold(z_bwd_cand)
                direct_abductive_err = (
                    geodesic_distance_sphere(u_bwd_cand, u_premise).mean() +
                    0.5 * torch.abs(r_bwd_cand - r_premise).mean()
                ).item()
                abductive_consistency = 1.0 / (1.0 + direct_abductive_err)

                # Multi-criteria scoring
                # Score combines:
                # - Forward & Backward Causal consistency (W_bwd agreement)
                # - Step survival (longevity in imagination)
                # - Penalties for dead ends and contradictions
                causal_harmonic = 0.5 * sim["causal_consistency"] + 0.5 * abductive_consistency
                score = causal_harmonic * 45.0
                score += (sim["steps_survived"] / max(lookahead_depth, 1)) * 25.0
                score += sim.get("semantic_resonance", 1.0) * 15.0
                if sim["is_dead_end"]:
                    score -= 40.0
                if sim["is_contradiction"] or direct_abductive_err > 2.2:
                    score -= 50.0

                rb = ReasonedBranch(
                    branch_id=branch_id,
                    score=float(score),
                    causal_consistency=float(causal_harmonic),
                    memory_alignment=float(sim.get("semantic_resonance", 0.5)),
                    steps_survived=int(sim["steps_survived"]),
                    is_dead_end=bool(sim["is_dead_end"]),
                    is_contradiction=bool(sim["is_contradiction"]),
                    final_latent=sim["final_latent"],
                    imagined_trajectory=sim["trajectory"]
                )
                evaluated_branches.append(rb)

            # 5. Rank branches
            evaluated_branches.sort(key=lambda b: b.score, reverse=True)
            viable = [b for b in evaluated_branches if not b.is_dead_end and not b.is_contradiction]
            best_branch = evaluated_branches[0]

            # Calculate confidence safely
            if len(evaluated_branches) > 1:
                spread = evaluated_branches[0].score - evaluated_branches[1].score
                if math.isnan(spread):
                    confidence = 0.5
                else:
                    confidence = float(torch.sigmoid(torch.tensor(spread * 0.1)).clamp(0.01, 0.99).item())
            else:
                conf_val = best_branch.causal_consistency
                confidence = 0.5 if math.isnan(conf_val) else float(max(min(conf_val, 0.99), 0.01))

            # System mode determination
            system_mode = "System 1 (Intuitive Consensus)" if len(candidate_branches) <= 1 else f"System 2 Deliberative Search ({lookahead_depth} Ticks)"

            trace = [
                {
                    "branch_id": b.branch_id,
                    "score": b.score,
                    "causal_consistency": b.causal_consistency,
                    "pruned_dead_end": b.is_dead_end,
                    "pruned_contradiction": b.is_contradiction,
                    "steps": b.steps_survived
                }
                for b in evaluated_branches
            ]

            explanation = (
                f"Evaluated {len(candidate_branches)} candidate branches over {lookahead_depth} mental ticks. "
                f"Selected '{best_branch.branch_id}' (score: {best_branch.score:.2f}, causal consistency: {best_branch.causal_consistency:.2f}). "
                f"Pruned {pruned_dead_ends} dead ends and {pruned_contradictions} causal contradictions in imagination."
            )

            return ReasoningResult(
                selected_branch=best_branch.branch_id,
                confidence=confidence,
                system_mode=system_mode,
                total_candidates=len(candidate_branches),
                viable_candidates=len(viable),
                pruned_dead_ends=pruned_dead_ends,
                pruned_contradictions=pruned_contradictions,
                ranked_branches=evaluated_branches,
                reasoning_trace=trace,
                explanation=explanation
            )
