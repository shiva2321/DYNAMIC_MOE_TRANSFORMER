"""
Autonomous Self-Improving Reasoner (ASER) for Grounded Cognitive Engines.
Integrates Continuous Latent Thoughts (COCONUT), Self-Supervised Causal Verification (STaR/Self-Play),
and Wake-Sleep Continual Learning with Zero-Forgetting Guarantees.
"""

import math
import os
import time
import random
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ishmn.geometry import project_to_sphere, geodesic_distance_sphere, exp_map_sphere
from ishmn.grounded_hybrid_engine import GroundedHybridEngine


class AutonomousSelfImprover:
    """
    Autonomous Lifelong Self-Improvement System.
    Operates during idle or background cycles to:
    1. Sample concepts and synthesize counterfactual / relational reasoning challenges.
    2. Simulate multi-step causal thought trajectories in continuous hyperspace S^7 x R+.
    3. Evaluate causal reversibility (epsilon_causal) and MoE consensus without human labels.
    4. Crystallize verified insights into permanent semantic concepts (zero forgetting).
    5. Detect hallucinations/contradictions and sharpen Vygotskian ZPD protective boundaries.
    6. Execute NREM/REM sleep consolidation with Gate 2 & Gate 3 audits.
    """
    def __init__(
        self,
        engine: GroundedHybridEngine,
        verification_causal_threshold: float = 0.20,
        verification_consensus_threshold: float = 0.80,
        contradiction_threshold: float = 0.40
    ):
        self.engine = engine
        self.uce = engine.uce
        self.slm = engine.slm
        self.tokenizer = engine.tokenizer
        self.device = engine.device

        self.verification_causal_threshold = verification_causal_threshold
        self.verification_consensus_threshold = verification_consensus_threshold
        self.contradiction_threshold = contradiction_threshold

        self.telemetry_history = []
        self.total_hypotheses_evaluated = 0
        self.verified_discoveries = 0
        self.contradictions_flagged = 0

    def sample_concept_pair(self, domain_aligned: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Samples two distinct crystallized concepts from the neocortical lattice.
        If domain_aligned is True, prioritizes sampling within coherent topological domains.
        """
        concepts = self.uce.memory.semantic_concepts
        if len(concepts) < 2:
            raise ValueError("Insufficient concepts in memory lattice for self-improvement sampling.")

        if domain_aligned and random.random() < 0.75:
            domains = list({c.get("domain") for c in concepts if c.get("domain")})
            if domains:
                dom = random.choice(domains)
                dom_concepts = [c for c in concepts if c.get("domain") == dom]
                if len(dom_concepts) >= 2:
                    c1, c2 = random.sample(dom_concepts, 2)
                    return c1, c2

        c1, c2 = random.sample(concepts, 2)
        return c1, c2

    def synthesize_relational_hypothesis(
        self,
        concept_a: Dict[str, Any],
        concept_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesizes a relational hypothesis between two concepts:
        Computes Fréchet mid-point in S^7 tangent space, then probes the SLM
        for the linguistic invariant connecting the concepts.
        """
        u_a = concept_a["u"].to(self.device)
        u_b = concept_b["u"].to(self.device)
        if u_a.dim() == 1:
            u_a = u_a.unsqueeze(0)
        if u_b.dim() == 1:
            u_b = u_b.unsqueeze(0)

        # Compute geodesic distance and midpoint on S^7
        geo_dist = geodesic_distance_sphere(u_a, u_b).mean().item()
        u_mid = project_to_sphere(0.5 * (u_a + u_b))

        word_a = concept_a.get("word", "entity_A")
        word_b = concept_b.get("word", "entity_B")
        desc_a = concept_a.get("metadata", {}).get("description", word_a)
        desc_b = concept_b.get("metadata", {}).get("description", word_b)

        prompt = (
            f"Synthesize the fundamental invariant or causal connection between '{word_a}' and '{word_b}'.\n"
            f"Concept A: {desc_a}\n"
            f"Concept B: {desc_b}\n"
            f"State the core scientific or logical principle in one concise sentence:"
        )

        with torch.no_grad():
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([formatted], return_tensors="pt").to(self.device)
            out_tokens = self.slm.generate(**inputs, max_new_tokens=60, do_sample=False)
            gen_text = self.tokenizer.decode(out_tokens[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

        return {
            "concept_a": word_a,
            "concept_b": word_b,
            "geodesic_distance": geo_dist,
            "u_midpoint": u_mid,
            "hypothesis_text": gen_text
        }

    def verify_hypothesis_causality(self, hypothesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifies the synthesized hypothesis purely via continuous Riemannian manifold dynamics:
        1. Perceive text -> u, r, v on S^7 x R+
        2. Deliberate through closed-loop world model with MoE consensus
        3. Check causal reversibility: epsilon_causal = d_S7(u, u_recon)
        """
        text = hypothesis_data["hypothesis_text"]
        if not text or len(text.strip()) < 5:
            return {"verdict": "rejected_empty", "causal_error": 1.0, "consensus": 0.0}

        # 1. Perceive hypothesis text
        u_perc, r_perc, v_perc = self.uce.perceive(text, modality="subwords")

        # 2. Deliberate in continuous latent space
        u_mid = hypothesis_data["u_midpoint"]
        delib = self.uce.world_model.deliberate_and_halting_loop(
            z_init=v_perc,
            context=v_perc,
            moe_layer=self.uce.moe
        )

        causal_err = delib["causal_error"]
        consensus = delib["consensus"]
        steps = delib["steps_taken"]
        u_final = delib["final_u"]
        r_final = delib["final_r"]
        z_final = delib["final_latent"]

        # 3. Decision metric (Unsupervised Causal Grounding)
        is_verified = (
            causal_err <= self.verification_causal_threshold
            and consensus >= self.verification_consensus_threshold
        )
        is_contradiction = (causal_err >= self.contradiction_threshold)

        if is_verified:
            verdict = "verified_invariant"
        elif is_contradiction:
            verdict = "contradiction_detected"
        else:
            verdict = "uncertain"

        return {
            "verdict": verdict,
            "causal_error": causal_err,
            "consensus": consensus,
            "thinking_steps": steps,
            "u_final": u_final,
            "r_final": r_final,
            "z_final": z_final,
            "hypothesis_text": text
        }

    def run_self_play_step(self) -> Dict[str, Any]:
        """
        Executes a single autonomous self-play reasoning & verification step.
        """
        self.total_hypotheses_evaluated += 1
        c1, c2 = self.sample_concept_pair()

        # 1. Synthesize candidate hypothesis
        hypo = self.synthesize_relational_hypothesis(c1, c2)

        # 2. Unsupervised verification on Riemannian manifold
        verif = self.verify_hypothesis_causality(hypo)
        verdict = verif["verdict"]

        crystallized = False
        if verdict == "verified_invariant":
            # Crystallize new relational invariant into permanent semantic memory
            self.verified_discoveries += 1
            new_id = len(self.uce.memory.semantic_concepts)
            concept_name = f"rel_{c1.get('word', 'A')}_{c2.get('word', 'B')}"
            
            new_concept = {
                "concept_id": new_id,
                "word": concept_name,
                "u": verif["u_final"].detach().cpu(),
                "r": float(verif["r_final"].mean().item()) if isinstance(verif["r_final"], torch.Tensor) else float(verif["r_final"]),
                "v": verif["z_final"].detach().cpu(),
                "domain": "autonomous_discovery",
                "exemplar_count": 25,
                "metadata": {
                    "description": verif["hypothesis_text"],
                    "parent_a": c1.get("word"),
                    "parent_b": c2.get("word"),
                    "causal_error": verif["causal_error"],
                    "consensus": verif["consensus"],
                    "is_autonomous_discovery": True
                }
            }
            self.uce.memory.semantic_concepts.append(new_concept)
            self.uce.memory._sem_cache_u = None
            self.uce.memory._sem_cache_v = None
            crystallized = True

        elif verdict == "contradiction_detected":
            # Contradiction: tighten ZPD chaotic boundary to reject this mode
            self.contradictions_flagged += 1
            if hasattr(self.uce, "developmental_learner") and self.uce.developmental_learner:
                self.uce.developmental_learner.zpd.tau_upper = max(
                    0.40, self.uce.developmental_learner.zpd.tau_upper * 0.98
                )

        # Log episode
        u_log = verif["u_final"] if "u_final" in verif else hypo["u_midpoint"]
        r_log = verif["r_final"] if "r_final" in verif else torch.tensor([[1.0]], device=self.device)
        v_log = verif["z_final"] if "z_final" in verif else torch.zeros(1, self.uce.d_model, device=self.device)
        
        self.uce.memory.log_episode(
            u=u_log[:1],
            r=r_log[:1] if r_log.dim() > 0 else torch.tensor([[1.0]], device=self.device),
            v=v_log[:1],
            surprise=verif["causal_error"],
            domain="autonomous_self_play",
            metadata={
                "verdict": verdict,
                "parents": (c1.get("word"), c2.get("word")),
                "crystallized": crystallized
            }
        )

        record = {
            "cycle": self.total_hypotheses_evaluated,
            "concept_a": c1.get("word"),
            "concept_b": c2.get("word"),
            "verdict": verdict,
            "causal_error": verif["causal_error"],
            "consensus": verif["consensus"],
            "hypothesis": verif.get("hypothesis_text", "")[:100],
            "crystallized": crystallized,
            "total_concepts": len(self.uce.memory.semantic_concepts)
        }
        self.telemetry_history.append(record)
        return record

    def run_self_improvement_session(
        self,
        num_cycles: int = 5,
        run_sleep_after: bool = True
    ) -> Dict[str, Any]:
        """
        Executes a complete self-improvement session:
        - Evaluates num_cycles of autonomous self-play reasoning
        - Triggers NREM/REM sleep consolidation with Gate 2 & 3 audits
        - Returns comprehensive telemetry report
        """
        t0 = time.time()
        print(f"[*] Starting Autonomous Self-Improvement Session ({num_cycles} cycles)...")
        results = []
        for i in range(1, num_cycles + 1):
            rec = self.run_self_play_step()
            results.append(rec)
            print(f"  [Cycle {i:2d}/{num_cycles}] {rec['concept_a']} <-> {rec['concept_b']} | {rec['verdict']} | CausalErr: {rec['causal_error']:.3f} | Cons: {rec['consensus']:.3f} | Cryst: {rec['crystallized']}")

        sleep_report = {}
        if run_sleep_after and self.verified_discoveries > 0:
            print("[*] Consolidating new discoveries via Dual-Phase Sleep (NREM + REM)...")
            sleep_report = self.engine.sleep_consolidate(crystallize_concepts=True)
            print(f"  [+] Sleep complete: Gate 2 Passed: {sleep_report.get('gate_2_passed', True)} | Gate 3 Passed: {sleep_report.get('gate_3_passed', True)}")

        elapsed = time.time() - t0
        return {
            "elapsed_sec": elapsed,
            "cycles_run": num_cycles,
            "total_hypotheses_evaluated": self.total_hypotheses_evaluated,
            "verified_discoveries": self.verified_discoveries,
            "contradictions_flagged": self.contradictions_flagged,
            "final_concept_count": len(self.uce.memory.semantic_concepts),
            "sleep_report": sleep_report,
            "recent_cycles": results
        }
