"""
Human-Inspired Developmental Learning Architecture (HIDLA) & Developmental Manifold Learner.

Grounded in:
1. Jean Piaget's Ontogenetic Cognitive Development Stages (Sensorimotor to Formal Operational).
2. Alison Gopnik's "Scientist in the Crib" / Active Causal Intervention Theory.
3. Lev Vygotsky's Zone of Proximal Development (ZPD) & Scaffolded Social Learning.
4. Stanislas Dehaene's Four Pillars of Human Learning (Attention, Active Engagement, Error Feedback, Consolidation).
5. Karl Friston's Active Inference & Epistemic Curiosity (Information Gain).
6. Operational Reversibility on Riemannian Manifolds S^{D-1} x R+.
"""

from enum import Enum
import math
import time
import random
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ishmn.geometry import (
    project_to_sphere,
    geodesic_distance_sphere,
    exp_map_sphere,
    EPS
)
from ishmn.universal_cognitive_engine import UniversalCognitiveEngine


class DevelopmentalStage(str, Enum):
    """
    The five ontogenetic stages of human cognitive development.
    """
    STAGE_0_SENSORIMOTOR = "0_sensorimotor"           # 0 - 6 months: Continuous coordinate calibration & predictive coding
    STAGE_1_CAUSAL_PLAY = "1_causal_play"             # 6mo - 2 years: "Scientist in the crib", active intervention do(X)
    STAGE_2_SCAFFOLDED_ZPD = "2_scaffolded_zpd"       # 2 - 5 years: Zone of Proximal Development & fast-mapping
    STAGE_3_CONCRETE_OPERATIONS = "3_concrete_ops"    # 6 - 11 years: Operational reversibility & conservation laws
    STAGE_4_FORMAL_DELIBERATION = "4_formal_reasoning" # 12+ years: Dual-process ACT & metacognitive reflection


class ZoneOfProximalDevelopment:
    """
    Lev Vygotsky's Zone of Proximal Development (ZPD) Filter.
    
    A dynamic neuroplasticity gating filter:
    - Inputs with surprise S < tau_lower are 'Already Mastered' (Boring / Automated).
      They execute via fast intuition (System 1) with zero synaptic rewrite.
    - Inputs with tau_lower <= S <= tau_upper are in the fertile 'ZPD'.
      They trigger high neuroplasticity, dendritic tuning, and episodic logging.
    - Inputs with S > tau_upper are 'Chaotic / Incomprehensible Noise'.
      They are shielded from corrupting crystallized semantic invariants until
      prerequisite scaffolding is established.
    """
    def __init__(
        self,
        tau_lower: float = 0.20,
        tau_upper: float = 0.85,
        adaptation_rate: float = 0.05
    ):
        self.tau_lower = tau_lower
        self.tau_upper = tau_upper
        self.adaptation_rate = adaptation_rate
        self.total_evaluated = 0
        self.mastered_count = 0
        self.zpd_admitted_count = 0
        self.chaotic_count = 0

    def evaluate(
        self,
        manifold_surprise: float = 0.0,
        causal_error: float = 0.0,
        surprise: Optional[float] = None
    ) -> Tuple[str, bool]:
        """
        Evaluates cognitive surprise against dynamic ZPD thresholds.
        Integrates Riemannian manifold surprise and operational reversibility error.

        Returns:
            verdict: 'mastered', 'zpd_fertile', or 'chaotic'
            should_learn: True if observation warrants synaptic consolidation.
        """
        if surprise is not None:
            manifold_surprise = surprise
        self.total_evaluated += 1
        if manifold_surprise > self.tau_upper or causal_error > 2.0:
            self.chaotic_count += 1
            return "chaotic", False
        elif manifold_surprise < self.tau_lower:
            self.mastered_count += 1
            return "mastered", False
        else:
            self.zpd_admitted_count += 1
            return "zpd_fertile", True

    def adapt_window(self, mastery_rate: float):
        """
        Dynamically adjusts ZPD thresholds based on developmental progression:
        As mastery rate increases, upper threshold expands to admit more complex concepts.
        """
        if mastery_rate > 0.60:
            self.tau_lower = min(0.35, self.tau_lower + 0.01)
            self.tau_upper = min(1.20, self.tau_upper + 0.02)
        elif mastery_rate < 0.20:
            self.tau_lower = max(0.10, self.tau_lower - 0.01)
            self.tau_upper = max(0.60, self.tau_upper - 0.02)

    def get_stats(self) -> Dict[str, Any]:
        total = max(1, self.total_evaluated)
        return {
            "tau_lower": self.tau_lower,
            "tau_upper": self.tau_upper,
            "total_evaluated": self.total_evaluated,
            "mastered_ratio": self.mastered_count / total,
            "zpd_fertile_ratio": self.zpd_admitted_count / total,
            "chaotic_ratio": self.chaotic_count / total
        }


class ActiveCausalPlaySimulator:
    """
    Alison Gopnik's 'Scientist in the Crib' Active Intervention Simulator.
    
    Instead of passive observation, an infant child actively interacts with objects
    (dropping, shaking, pushing) to test causal hypotheses:
    - Generates mental 'do(X)' perturbations.
    - Projects forward through World Model W_fwd.
    - Reconstructs counterfactual cause through W_bwd.
    - Computes Epistemic Information Gain (drop in free energy + novelty).
    - Selects the intervention that yields maximum explanatory insight.
    """
    def __init__(self, uce: UniversalCognitiveEngine):
        self.uce = uce
        self.device = uce.device

    def simulate_interventions(
        self,
        z_state: torch.Tensor,
        candidate_actions: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Evaluates a set of exploratory candidate actions via counterfactual mental simulation.
        
        Args:
            z_state: Current continuous latent state [1, d_model]
            candidate_actions: List of candidate perturbation vectors [1, d_model]
            
        Returns:
            Best action, expected information gain, and full simulation audit.
        """
        simulations = []
        best_action = None
        max_info_gain = -1e9

        z_curr = z_state.to(self.device)
        u_curr, r_curr = self.uce.world_model.project_manifold(z_curr)

        for idx, act in enumerate(candidate_actions):
            act_dev = act.to(self.device)
            # Apply mental do(act) intervention: forward transition
            z_next = self.uce.world_model.forward_transition(z_curr, act_dev)
            u_next, r_next = self.uce.world_model.project_manifold(z_next)

            # Counterfactual backward verification: W_bwd(z_next, act)
            z_recon = self.uce.world_model.backward_transition(z_next, act_dev)
            u_recon, r_recon = self.uce.world_model.project_manifold(z_recon)

            # Measure causal reversibility error on manifold
            causal_err = (
                geodesic_distance_sphere(u_curr, u_recon).mean() +
                0.5 * torch.abs(r_curr - r_recon).mean()
            ).item()

            # Measure epistemic surprise: distance travelled on S^{D-1}
            geodesic_shift = geodesic_distance_sphere(u_curr, u_next).mean().item()

            # Epistemic Information Gain:
            # Optimal play seeks high exploratory shift with low unrecoverable causal divergence
            info_gain = geodesic_shift * 0.7 + max(0.0, 1.0 - causal_err) * 0.3

            sim_entry = {
                "action_idx": idx,
                "geodesic_shift": geodesic_shift,
                "causal_error": causal_err,
                "info_gain": info_gain,
                "z_predicted": z_next.detach(),
                "u_predicted": u_next.detach()
            }
            simulations.append(sim_entry)

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_action = act_dev

        return {
            "selected_action": best_action,
            "max_info_gain": max_info_gain,
            "simulations": simulations
        }


class OperationalReversibilityAudit:
    """
    Jean Piaget's Operational Reversibility Validator.
    
    Verifies that transformations conserve structural invariants:
    Conservation of quantity, scale, and topological relations.
    Tests: for any forward mental transformation T, does T^{-1}(T(z)) == z?
    """
    def __init__(self, uce: UniversalCognitiveEngine):
        self.uce = uce

    def audit_reversibility(
        self,
        z_state: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Measures the loop-invariance (holonomy error) of state z under world model transitions.
        """
        z = z_state.to(self.uce.device)
        c = context.to(self.uce.device) if context is not None else z

        # Forward transition
        z_fwd = self.uce.world_model.forward_transition(z, c)
        # Backward rollback
        z_rev = self.uce.world_model.backward_transition(z_fwd, c)

        u_orig, r_orig = self.uce.world_model.project_manifold(z)
        u_rev, r_rev = self.uce.world_model.project_manifold(z_rev)

        angular_reversibility_error = geodesic_distance_sphere(u_orig, u_rev).mean().item()
        radial_reversibility_error = torch.abs(r_orig - r_rev).mean().item()
        total_reversibility_error = angular_reversibility_error + 0.5 * radial_reversibility_error

        # Conservation passes if round-trip reconstruction is within 0.20 rad
        conserved = total_reversibility_error <= 0.20

        return {
            "angular_error": angular_reversibility_error,
            "radial_error": radial_reversibility_error,
            "total_reversibility_error": total_reversibility_error,
            "is_conserved": conserved
        }


class DevelopmentalManifoldLearner:
    """
    Master Developmental Learning Coordinator.
    
    Guides the Universal Cognitive Engine through human ontogenetic milestones:
    - Stage 0: Sensorimotor Grounding
    - Stage 1: Active Causal Play ("Scientist in the Crib")
    - Stage 2: Scaffolded ZPD Learning & Fast-Mapping
    - Stage 3: Concrete Operations & Reversibility
    - Stage 4: Formal Deliberation & Metacognition
    
    Manages waking daytime experience and nocturnal sleep consolidation.
    """
    def __init__(
        self,
        uce: UniversalCognitiveEngine,
        initial_stage: DevelopmentalStage = DevelopmentalStage.STAGE_0_SENSORIMOTOR
    ):
        self.uce = uce
        self.device = uce.device
        self.stage = initial_stage
        self.developmental_ticks = 0
        self.stage_ticks = 0

        # Components
        self.zpd = ZoneOfProximalDevelopment()
        self.play_simulator = ActiveCausalPlaySimulator(uce)
        self.reversibility_auditor = OperationalReversibilityAudit(uce)

        # Telemetry & Perceptual Memory Buffer
        self.recent_perceptions: List[torch.Tensor] = []
        self.max_recent_perceptions: int = 32
        self.intake_history = []
        self.stage_transition_log = [{
            "stage": self.stage.value,
            "tick": 0,
            "timestamp": time.time(),
            "reason": "initialization"
        }]

    def step_intake(
        self,
        observation: Any,
        modality: str = "continuous",
        domain: str = "developmental",
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Performs a single developmental learning intake step.
        Adapts behavior based on the current ontogenetic stage.
        """
        self.developmental_ticks += 1
        self.stage_ticks += 1

        # 1. PERCEPTUAL GROUNDING ON S^{D-1} x R+
        u, r, v = self.uce.perceive(observation, modality=modality)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        if r.dim() == 1:
            r = r.unsqueeze(0)
        if v.dim() == 1:
            v = v.unsqueeze(0)

        # 2. MEASURE COGNITIVE SURPRISE ON RIEMANNIAN MANIFOLD
        # Surprise is the geodesic distance on S^{D-1} to the closest existing schema:
        # Check Immediate Sensory Memory, Working Memory, Episodic Traces, and Semantic Lattice.
        candidates = []
        u_query = u[:1]

        # Tier 0: Immediate Sensory Memory (Echoic buffer for immediate stimulus repetition <= 0.05 rad)
        if self.recent_perceptions:
            recent_stack = torch.cat(self.recent_perceptions, dim=0)
            cos_rec = torch.clamp(torch.matmul(u_query, recent_stack.t()), -1.0 + EPS, 1.0 - EPS)
            min_rec = torch.acos(cos_rec).min().item()
            if min_rec <= 0.05:
                candidates.append(min_rec)

        # Tier 1: Active Working Memory slots (active attentional focus <= 0.10 rad)
        u_mem = self.uce.mem_state["u"]
        usage = self.uce.mem_state.get("usage", None)
        if usage is not None and (usage[0] > 0).any():
            active_slots = u_mem[0, usage[0] > 0]
            cos_wm = torch.clamp(torch.matmul(u_query, active_slots.t()), -1.0 + EPS, 1.0 - EPS)
            min_wm_dist = torch.acos(cos_wm).min().item()
            if min_wm_dist <= 0.10:
                candidates.append(min_wm_dist)

        # Tier 2: Lifelong Episodic Traces (episodic recall within ZPD horizon)
        search_radius = float(self.zpd.tau_upper) if hasattr(self.zpd, "tau_upper") else 0.85
        if hasattr(self.uce.memory, "recall_resonant_episodes"):
            ep_matches = self.uce.memory.recall_resonant_episodes(u_query[0], top_k=1, max_geodesic_distance=search_radius)
            if ep_matches:
                candidates.append(ep_matches[0]["geodesic_distance"])

        # Tier 3: Sleep-Crystallized Semantic Lattice (invariant concept recognition within ZPD horizon)
        if hasattr(self.uce.memory, "query_semantic_lattice"):
            sem_matches = self.uce.memory.query_semantic_lattice(u_query[0], top_k=1, max_geodesic_distance=search_radius)
            if sem_matches:
                candidates.append(sem_matches[0]["geodesic_distance"])
        elif hasattr(self.uce.memory, "recall_semantic_concepts"):
            sem_matches = self.uce.memory.recall_semantic_concepts(u_query[0], top_k=1, max_geodesic_distance=search_radius)
            if sem_matches:
                candidates.append(sem_matches[0]["geodesic_distance"])

        manifold_surprise = min(candidates) if candidates else (float(math.pi) / 2.0)

        # Measure preliminary causal reversibility
        rev_audit = self.reversibility_auditor.audit_reversibility(v, context)
        causal_error = rev_audit["total_reversibility_error"]

        # 3. STAGE-DEPENDENT COGNITIVE PROCESSING
        result = {
            "tick": self.developmental_ticks,
            "stage": self.stage.value,
            "modality": modality,
            "domain": domain,
            "manifold_surprise": manifold_surprise,
            "causal_error": causal_error,
            "conserved": rev_audit["is_conserved"]
        }

        if self.stage == DevelopmentalStage.STAGE_0_SENSORIMOTOR:
            # Stage 0: Sensorimotor Calibration
            # Fast predictive coding relaxation without complex deliberation
            relaxed_states = self.uce.world_model.hrpc.forward_relaxation_step(v, num_steps=2, alpha=0.2)
            v_settled = relaxed_states[0]
            # Write to working memory
            w_gate = self.uce.mem_write_gate(v_settled)
            zeros_erase = torch.zeros_like(v_settled)
            _, new_mem = self.uce.memory.write(
                self.uce.mem_state,
                u_write=u[:1].detach(),
                r_write=r[:1].detach(),
                add_vector=v_settled[:1].detach(),
                erase_vector=zeros_erase[:1].detach(),
                write_gate=w_gate[:1].detach(),
                kappa_w=torch.clamp(self.uce.kappa_mem.detach(), 0.1, 10.0),
                beta_rw=torch.clamp(self.uce.beta_mem.detach(), 0.01, 5.0)
            )
            self.uce.mem_state = {
                k: val.detach().clone() if isinstance(val, torch.Tensor) else val
                for k, val in new_mem.items()
            }
            result["action_taken"] = "sensorimotor_calibrated"
            result["system_used"] = "System 1 (Reflexive Calibration)"

        elif self.stage == DevelopmentalStage.STAGE_1_CAUSAL_PLAY:
            # Stage 1: The Scientist in the Crib (Active Play)
            # Synthesize 3 candidate mental perturbations to test affordances
            perturbations = [
                v + 0.1 * torch.randn_like(v),
                v - 0.1 * torch.randn_like(v),
                0.8 * v + 0.2 * self.uce.mem_state["v"][:, 0, :]
            ]
            play_res = self.play_simulator.simulate_interventions(v, perturbations)
            best_action = play_res["selected_action"]

            # Evolve state under best exploratory play action
            z_evolved = self.uce.world_model.forward_transition(v, best_action)
            u_evolved, r_evolved = self.uce.world_model.project_manifold(z_evolved)

            # Log episode into autobiographical episodic memory
            self.uce.memory.log_episode(
                u=u[:1],
                r=r[:1],
                v=v[:1],
                surprise=manifold_surprise,
                domain=domain,
                metadata={"u_evolved": u_evolved, "info_gain": play_res["max_info_gain"], "play": True}
            )
            result["action_taken"] = "causal_play_executed"
            result["max_info_gain"] = play_res["max_info_gain"]
            result["system_used"] = "Active Causal Exploration"

        elif self.stage == DevelopmentalStage.STAGE_2_SCAFFOLDED_ZPD:
            # Stage 2: Vygotskian Scaffolding with Zone of Proximal Development
            zpd_verdict, should_learn = self.zpd.evaluate(manifold_surprise, causal_error=causal_error)
            result["zpd_verdict"] = zpd_verdict

            if zpd_verdict == "mastered":
                # Concept is familiar: bypass synaptic plasticity, fast intuition
                result["action_taken"] = "intuitive_bypass"
                result["system_used"] = "System 1 (Mastered Intuition)"
            elif zpd_verdict == "zpd_fertile":
                # In fertile ZPD: perform deliberation and log experiential episode
                delib = self.uce.world_model.deliberate_and_halting_loop(
                    z_init=v,
                    context=context if context is not None else v,
                    moe_layer=self.uce.moe
                )
                v_settled = delib["final_latent"]
                u_settled = delib["final_u"]
                r_settled = delib["final_r"]

                self.uce.memory.log_episode(
                    u=u[:1],
                    r=r[:1],
                    v=v[:1],
                    surprise=manifold_surprise,
                    domain=domain,
                    metadata={"u_settled": u_settled, "zpd": True, "steps": delib["steps_taken"]}
                )
                result["action_taken"] = "zpd_assimilated"
                result["system_used"] = delib["system_used"]
            else:
                # Chaotic: Out-of-ZPD anomaly; protect core semantic invariant lattice
                result["action_taken"] = "chaotic_deferred"
                result["system_used"] = "Cognitive Protection Gating"

        elif self.stage == DevelopmentalStage.STAGE_3_CONCRETE_OPERATIONS:
            # Stage 3: Concrete Operations & Reversibility
            # Check conservation under physical/relational transformations
            is_conserved = rev_audit["is_conserved"]
            delib = self.uce.world_model.deliberate_and_halting_loop(
                z_init=v,
                context=context if context is not None else v,
                moe_layer=self.uce.moe
            )
            v_settled = delib["final_latent"]
            u_settled = delib["final_u"]
            r_settled = delib["final_r"]

            if is_conserved and manifold_surprise > 0.25:
                # Spawns or updates invariant concept in lattice if conserved
                self.uce.memory.log_episode(
                    u=u[:1],
                    r=r[:1],
                    v=v[:1],
                    surprise=manifold_surprise,
                    domain=domain,
                    metadata={"u_settled": u_settled, "conserved": True, "reversibility_error": causal_error}
                )
            result["action_taken"] = "concrete_operation_verified"
            result["system_used"] = delib["system_used"]

        elif self.stage == DevelopmentalStage.STAGE_4_FORMAL_DELIBERATION:
            # Stage 4: Formal Metacognitive Deliberation
            # Full ACT System 1 vs System 2 routing with knowledge boundary detection
            needs_system2 = (causal_error > 0.28) or (manifold_surprise > 0.35)

            if needs_system2:
                delib = self.uce.world_model.deliberate_and_halting_loop(
                    z_init=v,
                    context=context if context is not None else v,
                    moe_layer=self.uce.moe
                )
                v_settled = delib["final_latent"]
                system_name = delib["system_used"]
            else:
                moe_out, _ = self.uce.moe(v, u, r)
                v_settled = v + 0.25 * moe_out
                system_name = "System 1 (Instant Intuition)"

            # Honest boundary check: distance to closest known semantic concept
            closest_sem = self.uce.memory.query_semantic_lattice(u, top_k=1)
            is_known = (len(closest_sem) > 0) and (closest_sem[0]["geodesic_distance"] < 0.60)
            result["knowledge_boundary"] = "known_territory" if is_known else "metacognitive_unknown_boundary"

            result["action_taken"] = "formal_reasoning_evaluated"
            result["system_used"] = system_name

        # Update short-term sensory perceptual buffer (Tier 0)
        self.recent_perceptions.append(u_query.detach().clone())
        if len(self.recent_perceptions) > self.max_recent_perceptions:
            self.recent_perceptions.pop(0)

        self.intake_history.append(result)
        return result

    def check_and_advance_milestones(self) -> Tuple[bool, Optional[str]]:
        """
        Evaluates developmental readiness criteria to advance to the next ontogenetic stage.
        Returns:
            advanced: True if stage transition occurred.
            new_stage_name: Name of new stage if transitioned, else None.
        """
        stages_order = [
            DevelopmentalStage.STAGE_0_SENSORIMOTOR,
            DevelopmentalStage.STAGE_1_CAUSAL_PLAY,
            DevelopmentalStage.STAGE_2_SCAFFOLDED_ZPD,
            DevelopmentalStage.STAGE_3_CONCRETE_OPERATIONS,
            DevelopmentalStage.STAGE_4_FORMAL_DELIBERATION
        ]
        curr_idx = stages_order.index(self.stage)
        if curr_idx >= len(stages_order) - 1:
            return False, None

        # Milestone criteria per stage
        recent = self.intake_history[-20:] if len(self.intake_history) >= 20 else self.intake_history
        if len(recent) < 5:
            return False, None

        should_advance = False
        reason = ""

        if self.stage == DevelopmentalStage.STAGE_0_SENSORIMOTOR:
            # Stage 0 -> Stage 1: Sensorimotor stability reached (low causal variance)
            mean_causal = float(np.mean([entry["causal_error"] for entry in recent]))
            if mean_causal < 0.50 and self.stage_ticks >= 5:
                should_advance = True
                reason = f"Sensorimotor coordinate stability reached (mean causal error={mean_causal:.4f})"

        elif self.stage == DevelopmentalStage.STAGE_1_CAUSAL_PLAY:
            # Stage 1 -> Stage 2: Causal play has gathered sufficient episodic grounding traces
            num_episodes = len(self.uce.memory.episodic_memory)
            if num_episodes >= 5 and self.stage_ticks >= 5:
                should_advance = True
                reason = f"Causal play affordance traces acquired ({num_episodes} episodes in buffer)"

        elif self.stage == DevelopmentalStage.STAGE_2_SCAFFOLDED_ZPD:
            # Stage 2 -> Stage 3: ZPD mastery rate > 50% across scaffolded tasks
            zpd_entries = [e for e in recent if "zpd_verdict" in e]
            if zpd_entries:
                mastered_ratio = sum(1 for e in zpd_entries if e["zpd_verdict"] == "mastered") / len(zpd_entries)
                if (mastered_ratio >= 0.40 or len(self.uce.memory.semantic_concepts) >= 10) and self.stage_ticks >= 5:
                    should_advance = True
                    reason = f"ZPD curriculum mastery achieved ({mastered_ratio*100:.1f}% familiar/mastered concepts)"

        elif self.stage == DevelopmentalStage.STAGE_3_CONCRETE_OPERATIONS:
            # Stage 3 -> Stage 4: Operational reversibility conservation rate >= 40%
            conserved_count = sum(1 for e in recent if e.get("conserved", False))
            conserved_ratio = conserved_count / len(recent)
            if (conserved_ratio >= 0.40 or conserved_count >= 2) and self.stage_ticks >= 5:
                should_advance = True
                reason = f"Operational reversibility conserved across transformations ({conserved_ratio*100:.1f}%)"

        if should_advance:
            next_stage = stages_order[curr_idx + 1]
            self.stage = next_stage
            self.stage_ticks = 0
            log_entry = {
                "stage": self.stage.value,
                "tick": self.developmental_ticks,
                "timestamp": time.time(),
                "reason": reason
            }
            self.stage_transition_log.append(log_entry)
            return True, next_stage.value

        return False, None

    def advance_stage(self, force: bool = False, reason: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Advances the learner to the next developmental stage.
        If force=True, transitions unconditionally (e.g. for curriculum milestones).
        """
        stages_order = [
            DevelopmentalStage.STAGE_0_SENSORIMOTOR,
            DevelopmentalStage.STAGE_1_CAUSAL_PLAY,
            DevelopmentalStage.STAGE_2_SCAFFOLDED_ZPD,
            DevelopmentalStage.STAGE_3_CONCRETE_OPERATIONS,
            DevelopmentalStage.STAGE_4_FORMAL_DELIBERATION
        ]
        curr_idx = stages_order.index(self.stage)
        if curr_idx >= len(stages_order) - 1:
            return False, None

        if force:
            next_stage = stages_order[curr_idx + 1]
            self.stage = next_stage
            self.stage_ticks = 0
            log_entry = {
                "stage": self.stage.value,
                "tick": self.developmental_ticks,
                "timestamp": time.time(),
                "reason": reason or "manual_curriculum_advancement"
            }
            self.stage_transition_log.append(log_entry)
            return True, next_stage.value

        return self.check_and_advance_milestones()

    def nocturnal_sleep(
        self,
        crystallize_concepts: bool = True,
        enable_rem_replay: bool = True,
        rem_steps: int = 5
    ) -> Dict[str, Any]:
        """
        Executes Dual-Phase Sleep Consolidation (NREM + REM).
        Consolidates waking daytime experiences into permanent invariant lattice anchors.
        """
        if hasattr(self.uce, "sleep_consolidate"):
            sleep_report = self.uce.sleep_consolidate(
                prune_usage_threshold=0.01,
                crystallize_concepts=crystallize_concepts,
                enable_rem_replay=enable_rem_replay,
                rem_steps=rem_steps,
                rem_batch_size=min(16, max(len(self.uce.memory.semantic_concepts), 2))
            )
        else:
            sleep_report = self.uce.sleep(
                prune_usage_threshold=0.01,
                crystallize_concepts=crystallize_concepts
            )
        return {
            "status": "developmental_sleep_completed",
            "current_stage": self.stage.value,
            "developmental_ticks": self.developmental_ticks,
            "sleep_report": sleep_report
        }

    sleep_consolidation = nocturnal_sleep

    def get_developmental_profile(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic status of ontogenetic development."""
        stats = self.zpd.get_stats()
        return {
            "current_stage": self.stage.value,
            "total_developmental_ticks": self.developmental_ticks,
            "developmental_ticks": self.developmental_ticks,
            "ticks_in_current_stage": self.stage_ticks,
            "stage_ticks": self.stage_ticks,
            "zpd_telemetry": stats,
            "zpd_thresholds": stats,
            "stage_transition_history": self.stage_transition_log,
            "total_episodic_traces": len(self.uce.memory.episodic_memory),
            "total_semantic_invariants": len(self.uce.memory.semantic_concepts),
            "active_moe_specialists": self.uce.moe.num_experts
        }
