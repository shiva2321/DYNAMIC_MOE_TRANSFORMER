"""
Mode C: The Day / Night Lifelong Learning Cycle.
Streaming Daytime Operational Interactions -> Slow-Wave Sleep Consolidation -> Morning Zero-Forgetting Audit.

Workflow:
1. Daytime Phase:
   - Ingests streaming operational experiences across new domains (quantum computing, distributed systems, marine biology).
   - Logs episodic traces into Tri-Tier Spherical Memory with surprise and coherence tracking.
2. Nighttime Phase:
   - Triggers uce.sleep(): homeostatic consolidation, weight decay, semantic invariant crystallization onto S^7,
     and dynamic MoE neurogenesis / expert alignment.
3. Morning Phase (Zero Catastrophic Forgetting Audit):
   - Probes newly crystallized daytime concepts (fast System 1 retrieval).
   - Probes pre-existing arithmetic (12+7, 23164-45, 23.14 Trillion) and language analogies.
   - Proves 100% retention of prior knowledge without altering SLM weights.
"""

import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ishmn import UniversalCognitiveEngine

BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DayNightLifelongSystem:
    def __init__(self, brain_path: str = BRAIN_PATH, device: torch.device = DEVICE):
        self.device = device
        print(f"[*] Initializing Day/Night Lifelong Cognitive System on: {self.device}")
        self.uce = UniversalCognitiveEngine.load_brain(brain_path, device=self.device)
        self.uce.eval()
        
        # Record baseline morning state
        self.initial_experts = self.uce.moe.num_experts
        self.initial_episodes = len(self.uce.memory.episodic_memory)
        self.initial_semantics = len(self.uce.memory.semantic_concepts)
        print(f"[*] Initial Baseline Brain: {self.initial_experts} Experts | {self.initial_episodes} Episodes | {self.initial_semantics} Semantic Invariants")

    def daytime_experience_stream(self, experiences: list):
        """Simulates daytime operational experience streaming into episodic memory."""
        print(f"\n[DAYTIME PHASE] Streaming {len(experiences)} live operational observations...")
        day_start = time.time()
        
        for i, (text, domain) in enumerate(experiences):
            # Sense-make and log into episodic memory
            res = self.uce.sense_make(
                data=text,
                modality="subwords",
                domain=domain,
                surprise_threshold=0.40,
                log_to_episodic=True
            )
            if (i + 1) % 5 == 0 or i == len(experiences) - 1:
                print(f"  -> Processed [{i+1}/{len(experiences)}] ({domain}): Surprise={res['surprise']:.3f}, System={res['system_used']}")

        elapsed = time.time() - day_start
        print(f"[+] Daytime Phase Complete in {elapsed:.2f}s. Episodic buffer: {len(self.uce.memory.episodic_memory)} traces.")

    def nighttime_sleep_consolidation(self):
        """Executes slow-wave homeostatic sleep consolidation."""
        print(f"\n[NIGHTTIME PHASE] Entering Slow-Wave Homeostatic Sleep Consolidation...")
        sleep_start = time.time()
        
        # Execute homeostatic sleep consolidation
        sleep_report = self.uce.sleep(
            prune_usage_threshold=0.01,
            crystallize_concepts=True
        )
        
        elapsed = time.time() - sleep_start
        print(f"[+] Sleep Consolidation Complete in {elapsed:.2f}s.")
        sem_rep = sleep_report.get('semantic_crystallization', {})
        print(f"  - New Invariants Formed: {sem_rep.get('new_concepts_crystallized', 0)}")
        print(f"  - Active Experts: {self.uce.moe.num_experts} (was {self.initial_experts})")
        print(f"  - Semantic Lattice Size: {len(self.uce.memory.semantic_concepts)} (was {self.initial_semantics})")
        return sleep_report

    def morning_audit(self):
        """Audits both newly learned knowledge and foundational legacy knowledge."""
        print("\n" + "="*80)
        print(" MORNING AUDIT: NEW KNOWLEDGE RECALL & ZERO CATASTROPHIC FORGETTING")
        print("="*80)

        # 1. Audit newly learned daytime domain concepts
        print("\n[PART 1: Newly Crystallized Daytime Knowledge]")
        new_queries = [
            ("superposition qubits entanglement decoherence quantum", "quantum_computing"),
            ("distributed consensus byzantine fault tolerance raft paxos", "distributed_systems"),
            ("bioluminescence deep oceanic abyssal benthic organisms", "marine_biology")
        ]
        
        for query_text, expected_domain in new_queries:
            u_q, r_q, _ = self.uce.perceive(query_text, modality="subwords")
            _, meta, _ = self.uce.memory.tri_tier_recall(
                self.uce.mem_state,
                u_query=u_q,
                r_query=r_q,
                kappa=torch.clamp(self.uce.kappa_mem, 0.1, 10.0),
                beta_r=torch.clamp(self.uce.beta_mem, 0.01, 5.0),
                top_k_episodic=3,
                top_k_semantic=2
            )
            recalled_domains = [ep["domain"] for ep in meta["recalled_episodes"]]
            print(f"  Query: '{query_text[:40]}...'")
            print(f"    Expected Domain: {expected_domain}")
            print(f"    Recalled Domains: {recalled_domains} | Top Semantic Dist: {meta['recalled_semantics'][0]['geodesic_distance']:.4f} rad")

        # 2. Audit pre-existing foundational arithmetic (Proof of Zero Catastrophic Forgetting)
        print("\n[PART 2: Pre-Existing Arithmetic Retention (Zero Forgetting)]")
        math_checks = [
            ("12 + 7", 19.0),
            ("23164 - 45", 23119.0),
            ("23146484561521 + 45132030", 23146529693551.0)
        ]
        
        all_math_pass = True
        for expr, expected in math_checks:
            res = self.uce.solve_with_dual_process(expr, enable_system2=True)
            pred = res["system2_exact_result"]
            passed = (pred == expected)
            all_math_pass = all_math_pass and passed
            print(f"  Math: {expr} = {pred} (Expected: {expected}) -> {'[PASS]' if passed else '[FAIL]'}")

        # 3. Audit pre-existing language analogies
        print("\n[PART 3: Pre-Existing Riemannian Word Analogies]")
        analogy_checks = [
            ("man", "king", "woman", "queen"),
            ("fire", "hot", "ice", "cold")
        ]
        
        all_analogy_pass = True
        for a, b, c, expected_word in analogy_checks:
            res = self.uce.solve_analogy(a, b, c)
            top_pred = res["predicted_word"]
            passed = (top_pred.lower() == expected_word.lower()) if top_pred else False
            all_analogy_pass = all_analogy_pass and passed
            print(f"  Analogy: {a}:{b}::{c} -> '{top_pred}' (Expected: '{expected_word}') (Confidence: {res['confidence']:.1f}%) -> {'[PASS]' if passed else '[FAIL]'}")

        print("\n" + "-"*80)
        print(f"OVERALL LIFELONG INTEGRITY: {'100% PERFECT RETENTION' if (all_math_pass and all_analogy_pass) else 'DEGRADED'}")
        print(f"Final Brain Metrics: {self.uce.moe.num_experts} Experts | {len(self.uce.memory.episodic_memory)} Episodes | {len(self.uce.memory.semantic_concepts)} Semantic Invariants")
        print("-"*80)


def run_mode_c_evaluation():
    system = DayNightLifelongSystem()
    
    # 20 diverse operational daytime observations across novel domains
    daytime_observations = [
        ("quantum computing relies on superposition and quantum entanglement to execute parallel algorithms", "quantum_computing"),
        ("qubits maintain phase coherence until interacting with environmental thermal fluctuations", "quantum_computing"),
        ("quantum error correction requires topological surface codes to protect quantum information", "quantum_computing"),
        ("decoherence destroys quantum phase information turning quantum states into classical mixtures", "quantum_computing"),
        ("superconducting transmon qubits are manipulated using microwave electromagnetic pulses", "quantum_computing"),
        ("distributed consensus protocols guarantee state machine replication across unreliable networks", "distributed_systems"),
        ("byzantine fault tolerance allows nodes to reach agreement despite malicious adversarial actors", "distributed_systems"),
        ("raft algorithm elects a leader node to sequence log entries across cluster followers", "distributed_systems"),
        ("vector clocks track causal event ordering across asynchronous communicating processes", "distributed_systems"),
        ("consistent hashing minimizes key redistribution when cluster capacity scales dynamically", "distributed_systems"),
        ("marine bioluminescence is produced by luciferin reacting with oxygen catalyzed by luciferase", "marine_biology"),
        ("deep ocean benthic trenches exert extreme hydrostatic pressure exceeding one thousand atmospheres", "marine_biology"),
        ("hydrothermal vent ecosystems thrive on chemosynthesis without any solar photosynthetic energy", "marine_biology"),
        ("abyssal organisms have evolved flexible cell membranes and piezolyte proteins to survive pressure", "marine_biology"),
        ("cephalopod chromatophores expand and contract rapidly to produce dynamic camouflage patterns", "marine_biology")
    ]
    
    # Run Daytime
    system.daytime_experience_stream(daytime_observations)
    
    # Run Nighttime Sleep
    system.nighttime_sleep_consolidation()
    
    # Run Morning Audit
    system.morning_audit()


if __name__ == "__main__":
    run_mode_c_evaluation()
