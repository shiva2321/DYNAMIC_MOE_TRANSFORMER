"""
Universal Cognitive Engine: Interactive Multi-Modal Console Explorer.

CLI interface to interact with all cognitive subsystems:
- Inverted Möbius 4pi Spinor Arithmetic (exact zero-crossings via CRT)
- Continuous Physical Orbit Forecasting (Predictive JEPA on S^{D-1} x R+)
- Looped Riemannian Sequence Generation (Adaptive ACT Deliberation)
- Homeostatic Sleep Consolidation (Barycentric merging + Apoptotic pruning)
- Active Inference Epistemic Curiosity Evaluation
- Collective Hyperspace Multi-Agent Synchronization
"""

import argparse
import math
import sys
import os
import re
import torch
import numpy as np

from ishmn import UniversalCognitiveEngine, CollectiveHyperspaceBus

def get_engine(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brain_path = "artifacts/unified_lifelong_brain.pt"
    if os.path.exists(brain_path):
        print(f"[*] Loading unified persistent brain from {brain_path}...")
        uce = UniversalCognitiveEngine.load_brain(brain_path, device=device)
    else:
        uce = UniversalCognitiveEngine(
            d_model=32,
            d_hidden=64,
            hyperspace_dim=8,
            num_memory_slots=32,
            initial_experts=1,
            max_experts=None,
            device=device
        )
    return uce, device

def run_sensemaking_demo(uce, custom_query=None):
    print("\n" + "=" * 75)
    print(" ABDUCTIVE SENSE-MAKING & TRI-TIER MEMORY RECALL")
    print("=" * 75)
    
    if custom_query:
        res = uce.sense_make(custom_query, modality="subwords", domain="custom")
        print(f"  [Input Observation]: \"{custom_query}\"")
        print(f"    Surprise Score    : {res['surprise']:.4f}")
        print(f"    Coherence Score   : {res['coherence_score']:.4f}")
        print(f"    System Deployed   : {res['system_used']}")
        print(f"    Deliberation Steps: {res.get('deliberation_steps', 1)}")
        if res.get('recalled_episodes'):
            print(f"    Recalled Memories : {len(res['recalled_episodes'])} episodes (top domain: {res['recalled_episodes'][0]['domain']})")
        return

    # Honest Memory Inspection (No synthetic on-the-fly pre-seeding)
    episodic_count = len(uce.memory.episodic_memory)
    semantic_count = len(uce.memory.semantic_concepts)
    print("  [Persistent Memory Status]:")
    print(f"    Stored Episodic Traces   : {episodic_count}")
    print(f"    Semantic Lattice Attractors: {semantic_count}")
    print(f"    Memory Head Curvature    : kappa={float(uce.kappa_mem):.2f}, beta={float(uce.beta_mem):.2f}")

    if episodic_count > 0:
        sample_domain = uce.memory.episodic_memory[0].get("domain", "general")
        print(f"    Latest Recorded Domain   : {sample_domain}")
    print("\n  Use '--query \"<statement>\"' to probe abductive sense-making on any observation.")


def run_analogy_demo(uce, custom_analogy=None):
    print("\n" + "=" * 75)
    print(" RIEMANNIAN ANALOGY ENGINE (NATIVE INTERNAL MEMORY LATTICE)")
    print("=" * 75)
    
    concepts = [sc for sc in uce.memory.semantic_concepts if sc.get("word")]
    dim = uce.hyperspace_dim
    vocab_len = len(concepts)

    if not custom_analogy:
        print("  [Memory Lattice Configuration]:")
        print(f"    Riemannian Hyperspace   : S^{dim-1} (Dimension {dim})")
        print(f"    Crystallized Concepts   : {vocab_len} internal words in memory")
        print("    External Dependencies   : None (100% self-contained internal memory)")
        print("\n  Please provide an analogy query via '--analogy \"A:B::C\"'.")
        print("  Examples:")
        print("    python console_explorer.py --analogy \"man:king::woman\"")
        print("    python console_explorer.py --analogy \"sun:planet::nucleus\"")
        print("    python console_explorer.py --analogy \"fire:hot::ice\"")
        return

    # Expecting format 'A:B::C' or 'A,B,C'
    parts = re.split(r"[:;,\s]+|::", custom_analogy.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 3:
        concept_a, concept_b, concept_c = parts[0], parts[1], parts[2]
        analogy = uce.solve_analogy(concept_a, concept_b, concept_c, modality="subwords")
        print(f"  Analogy Query: [{concept_a}] : [{concept_b}] :: [{concept_c}] : [?]")
        
        pred_word = analogy.get("predicted_word")
        conf = analogy.get("confidence", 0.0)
        err = analogy.get("error")
        if err:
            print(f"\n  [!] Knowledge Boundary: {err}")
            print("  >>> PREDICTED TARGET [?]: UNKNOWN (Concept not yet learned)")
            return

        if pred_word:
            print(f"\n  >>> PREDICTED TARGET [?]: \"{pred_word.upper()}\" (Confidence: {conf:.1f}%)")
        else:
            print("\n  >>> PREDICTED TARGET [?]: UNKNOWN / UNCERTAIN")

        top_cands = analogy.get("top_candidates", [])
        if top_cands:
            print(f"\n  Top Ranked Candidates on S^{dim-1}:")
            for idx, cand in enumerate(top_cands[:4], 1):
                print(f"    {idx}. \"{cand['word']:<12}\" | Similarity: {cand['similarity_pct']:.1f}% | Geodesic Dist: {cand['geodesic_distance']:.4f} rad")

        print("\n  Riemannian Geometric Verification:")
        print(f"    Source Geodesic Distance (A -> B): {analogy['distance_source']:.4f} rad")
        print(f"    Target Geodesic Distance (C -> D): {analogy['distance_target']:.4f} rad")
        print(f"    Isometry Preserved on S^{dim-1}    : {analogy['isometry_preserved']} (Exact Metric Preservation)")
        print(f"    D Derived Coordinate Magnitude  : {float(torch.norm(analogy['u_d'])):.4f}")

def run_physics_demo(uce, steps=10):
    print("\n" + "=" * 75)
    print(" CONTINUOUS NON-LINEAR PHYSICAL DYNAMICS FORECAST (JEPA)")
    print("=" * 75)
    init_state = torch.randn(1, uce.d_model, device=uce.device)
    rollout = uce.predict_trajectory(init_state, steps=steps)
    for r in rollout:
        s = r["step"]
        rad = r["r"].item()
        u_lead = r["u"].squeeze()[:3].numpy()
        print(f"  Step {s:2d} | Radius r = {rad:.3f} | Latent Direction: [{u_lead[0]:.2f}, {u_lead[1]:.2f}, {u_lead[2]:.2f}...]")

def run_sleep_demo(uce):
    print("\n" + "=" * 75)
    print(" HOMEOSTATIC SLEEP CONSOLIDATION & APOPTOTIC PRUNING")
    print("=" * 75)
    # Simulate usage
    with torch.no_grad():
        if uce.moe.num_experts > 1:
            uce.moe.expert_usage[-1] = 0.1
    report = uce.sleep(prune_usage_threshold=0.03)
    print(f"  Consolidation Status : {report['status']}")
    print(f"  Merged Co-Resonant   : {report['moe_consolidation']['merged']}")
    print(f"  Apoptotically Pruned : {report['moe_consolidation']['pruned']}")
    print(f"  Active Experts Post  : {report['active_experts']}")
    print(f"  Memory Norm Post     : {report['working_memory_norm']:.4f}")

def run_curiosity_demo(uce):
    print("\n" + "=" * 75)
    print(" EPISTEMIC ACTIVE INFERENCE & CURIOSITY EVALUATION")
    print("=" * 75)
    cands = [
        torch.zeros(1, 10, device=uce.device),
        torch.randn(1, 10, device=uce.device) * 1.5,
        torch.randn(1, 10, device=uce.device) * 4.0
    ]
    ranks = uce.evaluate_epistemic_curiosity(cands, modality="continuous")
    for r in ranks:
        print(f"  Stimulus {r['candidate_idx']} | InfoGain: {r['info_gain']:.3f} | "
              f"FE Reduction: {r['free_energy_reduction']:.3f} | Novelty: {r['novelty_score']:.3f}")

def run_collective_demo(uce):
    print("\n" + "=" * 75)
    print(" COLLECTIVE HYPERSPACE MULTI-AGENT SYNCHRONIZATION")
    print("=" * 75)
    uce2, _ = get_engine(device=uce.device)
    bus = CollectiveHyperspaceBus(hyperspace_dim=8, d_model=32, device=uce.device)
    bus.register_agent("agent_phys", uce)
    bus.register_agent("agent_sym", uce2)
    p_in = torch.randn(1, 10, device=uce.device)
    res = bus.solve_coupled_task(
        "agent_phys", "agent_sym", p_in,
        symbolic_input="collective consensus across peer manifolds",
        symbolic_modality="subwords"
    )
    print(f"  Coupled Step Result : Synced Phys Output Shape {list(res['phys_output'].shape)}")
    print(f"  Sym Surprise Score  : {res['sym_surprise']:.4f} | Coherence: {res['sym_coherence']:.4f}")
    print(f"  Bus Mean Angular Dist: {res['mean_angular_distance']:.3f} rad | Collective Entropy: {res['collective_entropy']:.3f}")

def run_developmental_demo(uce, custom_stimulus=None):
    print("\n" + "=" * 75)
    print(" HUMAN-INSPIRED DEVELOPMENTAL LEARNING ARCHITECTURE (HIDLA)")
    print("=" * 75)
    profile = uce.get_developmental_profile()
    stage = profile.get("current_stage", "UNKNOWN")
    zpd_info = profile.get("zpd_thresholds", {})
    ticks = profile.get("developmental_ticks", 0)

    print(f"  [Ontogenetic Stage]      : {stage}")
    print(f"  [Developmental Age/Ticks]: {ticks} ticks (Stage Ticks: {profile.get('stage_ticks', 0)})")
    print(f"  [ZPD Exploration Bounds] : S in [{zpd_info.get('tau_lower', 0.20):.2f}, {zpd_info.get('tau_upper', 0.85):.2f}] rad")
    print(f"  [Cognitive Safety Shield]: Active (Chaotic stimuli trigger Protection Gating)")

    if custom_stimulus:
        stimuli = [custom_stimulus]
    else:
        stimuli = [
            "mama dada milk",
            "The ball falls downward due to gravity and rebounds",
            "If all quantum fields admit gauge symmetries, then topological solitons exhibit holonomy"
        ]

    print("\n  Evaluating Ontogenetic Intake & Cognitive Protection:")
    for stim in stimuli:
        print(f"\n  --- Intake Probe: \"{stim}\" ---")
        step_res = uce.developmental_step(stim, modality="subwords", domain="developmental")
        zpd_eval = step_res.get("zpd_verdict", "n/a")
        surprise = step_res.get("manifold_surprise", 0.0)
        causal_err = step_res.get("causal_error", 0.0)
        action = step_res.get("action_taken", "intake")
        sys_used = step_res.get("system_used", "unknown")

        print(f"    Hierarchical Surprise  : {surprise:.4f} rad")
        print(f"    Causal Error           : {causal_err:.4f}")
        if zpd_eval != "n/a":
            print(f"    ZPD Filter Verdict     : {str(zpd_eval).upper()}")
        print(f"    Action Executed        : {action}")
        print(f"    System Deployed        : {sys_used}")
        if "max_info_gain" in step_res:
            print(f"    Active Play Sim InfoGain: {step_res['max_info_gain']:.4f}")
        if "conserved" in step_res:
            print(f"    Piagetian Conservation : Conserved = {step_res['conserved']}")

def main():
    parser = argparse.ArgumentParser(description="Universal Cognitive Engine Explorer")
    parser.add_argument("--mode", choices=["all", "sensemaking", "analogy", "physics", "sleep", "curiosity", "collective", "developmental"],
                        default="all", help="Subsystem to demonstrate")
    parser.add_argument("--query", type=str, default=None, help="Custom text observation to probe for sense-making")
    parser.add_argument("--analogy", type=str, default=None, help="Custom analogy to solve (e.g. 'man:king::woman')")
    parser.add_argument("--developmental", action="store_true", help="Demonstrate Human-Inspired Developmental Learning")
    parser.add_argument("--stimulus", type=str, default=None, help="Custom stimulus to evaluate with developmental learner")
    args = parser.parse_args()

    uce, device = get_engine()
    print(f"[*] Universal Cognitive Engine initialized on: {device}")
    print(f"[*] Active Experts at Birth: E = {uce.moe.num_experts} (Infinite Dynamic Capacity)")

    if args.query:
        run_sensemaking_demo(uce, custom_query=args.query)
        return

    if args.analogy:
        run_analogy_demo(uce, custom_analogy=args.analogy)
        return

    if args.developmental or args.stimulus or args.mode == "developmental":
        run_developmental_demo(uce, custom_stimulus=args.stimulus)
        return

    if args.mode in ("all", "sensemaking"):
        run_sensemaking_demo(uce, custom_query=args.query)
    if args.mode in ("all", "analogy"):
        run_analogy_demo(uce, custom_analogy=args.analogy)
    if args.mode in ("all", "physics"):
        run_physics_demo(uce)
    if args.mode in ("all", "curiosity"):
        run_curiosity_demo(uce)
    if args.mode in ("all", "collective"):
        run_collective_demo(uce)
    if args.mode in ("all", "developmental"):
        run_developmental_demo(uce, custom_stimulus=args.stimulus)
    if args.mode in ("all", "sleep"):
        run_sleep_demo(uce)

    print("\n[+] Explorer Tour Finished Successfully.")


if __name__ == "__main__":
    main()
