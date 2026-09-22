"""
Experiment: Unified General Intelligence Engine (Fusion of Mode B + Mode C + LLM).
Evaluates the GroundedHybridEngine across a complete 4-Phase Lifelong Cognitive Cycle:
1. Phase 1: Baseline Morning Manifold Generation (Layer 14 ACT Deliberation).
2. Phase 2: Daytime Operational Streaming (18 observations across Astrophysics, SynBio, and Crypto).
3. Phase 3: Nighttime Slow-Wave Sleep Consolidation (Crystallizing invariants onto S^7).
4. Phase 4: Next-Day Emergence & 100% Zero Catastrophic Forgetting Verification.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ishmn import GroundedHybridEngine

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)
TELEMETRY_LOG = OUTPUT_DIR / "unified_intelligence_telemetry.json"


def run_unified_general_intelligence_experiment():
    print("=" * 80)
    print(" UNIFIED GENERAL INTELLIGENCE ENGINE: MODE B + MODE C + LINGUISTIC CORTEX")
    print("=" * 80)

    # Initialize Grounded Hybrid Engine
    engine = GroundedHybridEngine(alpha=0.25)
    
    experiment_report = {
        "timestamp": time.time(),
        "baseline_metrics": {},
        "daytime_metrics": {},
        "sleep_metrics": {},
        "nextday_metrics": {},
        "lifelong_audit": {}
    }

    # =========================================================================
    # PHASE 1: Baseline Morning Probes (Layer 14 Deliberation Active)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" PHASE 1: BASELINE MORNING REASONING & LAYER 14 ACT DELIBERATION")
    print("=" * 80)

    baseline_prompts = [
        ("Causal Physics", "Explain in 2 sentences what happens when liquid nitrogen at -196 C contacts a white-hot glowing iron anvil."),
        ("Topological Geometry", "Explain in 2 sentences why a coffee mug and a donut (torus) are topologically equivalent."),
        ("Physical Contradiction", "Can a closed mechanical pendulum swing higher than its release point without external energy? Explain.")
    ]

    phase1_results = []
    for label, prompt in baseline_prompts:
        print(f"\n[Probe: {label}]")
        print(f"Prompt: \"{prompt}\"")
        res = engine.generate_grounded(prompt, max_new_tokens=75)
        print(f"  Generated Text: \"{res['text']}\"")
        print(f"  Telemetry: {res['tokens_generated']} tokens | System 1: {res['system1_tokens']} | System 2: {res['system2_tokens']} ({res['system2_tokens']/max(res['tokens_generated'],1)*100:.1f}%)")
        print(f"  Avg Thinking Ticks: {res['avg_thinking_ticks']:.2f} ticks/token | Mean Causal Error: {res['mean_causal_error']:.4f} rad")
        phase1_results.append({
            "label": label,
            "tokens": res["tokens_generated"],
            "s1_tokens": res["system1_tokens"],
            "s2_tokens": res["system2_tokens"],
            "avg_ticks": res["avg_thinking_ticks"],
            "causal_error": res["mean_causal_error"]
        })
    experiment_report["baseline_metrics"] = phase1_results

    # =========================================================================
    # PHASE 2: Daytime Operational Environment Streaming
    # =========================================================================
    print("\n" + "=" * 80)
    print(" PHASE 2: CONTINUOUS DAYTIME STREAMING (EPISODIC BUFFER EXPANSION)")
    print("=" * 80)

    daytime_observations = [
        # Domain 1: Astrophysics & Extreme Gravity
        ("neutron stars maintain hydrostatic equilibrium through neutron degeneracy pressure opposing gravitational collapse", "astrophysics"),
        ("gravitational microlensing warps background starlight into einstein rings around massive intervening compact bodies", "astrophysics"),
        ("event horizon photon sphere orbits black holes at three halves the schwarzschild radius where light rays orbit continuously", "astrophysics"),
        ("magnetar magnetic fields exceed one hundred gigatesla generating intense soft gamma repeater bursts", "astrophysics"),
        ("chandrasekhar mass limit defines the maximum stable mass of an electron degenerate white dwarf star", "astrophysics"),
        ("supermassive black holes drive relativistic plasma jets through active galactic nuclei powered by magnetohydrodynamic accretion", "astrophysics"),

        # Domain 2: Synthetic Biology & Bio-Engineering
        ("crispr cas nine endonuclease complexes utilize single guide rna to introduce targeted double strand breaks", "synthetic_biology"),
        ("synthetic genetic toggle switches implement bistable memory in living bacteria using reciprocal transcriptional repression", "synthetic_biology"),
        ("metabolic flux engineering reroutes cellular biochemical pathways to optimize biofuel synthesis yields", "synthetic_biology"),
        ("orthogonal artificial base pairs expand the genetic alphabet allowing ribosomal incorporation of non canonical amino acids", "synthetic_biology"),
        ("bacterial quorum sensing regulates collective biofilm formation via acyl homoserine lactone autoinducer signaling", "synthetic_biology"),
        ("chimeric antigen receptor t cells are engineered with synthetic scfv receptors to target malignant tumor antigens", "synthetic_biology"),

        # Domain 3: Cryptographic Consensus & Zero Knowledge
        ("zero knowledge succinct non interactive arguments of knowledge enable verifiable computation with zero secret leakage", "cryptographic_consensus"),
        ("elliptic curve bilinear pairings map cryptographic points into target multiplicative groups to verify bls signatures", "cryptographic_consensus"),
        ("verifiable delay functions enforce deterministic sequential execution time resistant to parallel compute acceleration", "cryptographic_consensus"),
        ("merkle patricia tries provide cryptographic inclusion proofs for decentralized state machine state verification", "cryptographic_consensus"),
        ("byzantine fault tolerance requires greater than two thirds honest validator voting power to reach deterministic finality", "cryptographic_consensus"),
        ("sharded consensus partitions transaction throughput across validator committees with cross shard message verification", "cryptographic_consensus")
    ]

    initial_episodes = len(engine.uce.memory.episodic_memory)
    day_res = engine.learn_daytime_stream(daytime_observations)
    print(f"[+] Daytime Streaming Completed in {day_res['elapsed_sec']:.2f}s.")
    print(f"    - Observations Ingested: {day_res['observations_ingested']}")
    print(f"    - Episodic Buffer Growth: {initial_episodes} -> {day_res['total_episodic_traces']} traces")
    print(f"    - Mean Perceptual Surprise: {day_res['mean_surprise']:.4f} rad")
    experiment_report["daytime_metrics"] = day_res

    # =========================================================================
    # PHASE 3: Nighttime Slow-Wave Sleep Consolidation
    # =========================================================================
    print("\n" + "=" * 80)
    print(" PHASE 3: NIGHTTIME SLOW-WAVE SLEEP CONSOLIDATION")
    print("=" * 80)

    initial_semantics = len(engine.uce.memory.semantic_concepts)
    initial_experts = engine.uce.moe.num_experts

    sleep_res = engine.sleep_consolidate(prune_usage_threshold=0.01, crystallize_concepts=True)
    new_semantics = len(engine.uce.memory.semantic_concepts)
    new_experts = engine.uce.moe.num_experts

    print(f"[+] Slow-Wave Sleep Completed in {sleep_res['elapsed_sec']:.2f}s.")
    print(f"    - Active MoE Specialists: {initial_experts} -> {new_experts}")
    print(f"    - Semantic Lattice Growth: {initial_semantics} -> {new_semantics} invariant points")
    print(f"    - New Invariants Crystallized: {sleep_res.get('semantic_crystallization', {}).get('new_concepts_crystallized', new_semantics - initial_semantics)}")
    print(f"    - Working Memory Energy Norm: {sleep_res.get('working_memory_norm', 0.0):.4f}")
    experiment_report["sleep_metrics"] = {
        "elapsed_sec": sleep_res["elapsed_sec"],
        "initial_semantics": initial_semantics,
        "new_semantics": new_semantics,
        "initial_experts": initial_experts,
        "new_experts": new_experts
    }

    # =========================================================================
    # PHASE 4: Next-Day Evaluation & Emergence Testing
    # =========================================================================
    print("\n" + "=" * 80)
    print(" PHASE 4: NEXT-DAY EVALUATION & ZERO CATASTROPHIC FORGETTING PROOF")
    print("=" * 80)

    # 4A. Test newly crystallized daytime concepts through Layer 14
    print("\n[PART 4A: Generating with Newly Crystallized Daytime Concepts]")
    nextday_prompts = [
        ("Astrophysics Query", "Describe the equilibrium of a neutron star and how degeneracy pressure resists gravitational collapse."),
        ("SynBio Query", "Explain how synthetic genetic toggle switches maintain memory in engineered bacteria."),
        ("Crypto Query", "How do zero-knowledge proofs enable verifiable computation without leaking private secret data?")
    ]

    phase4_gen = []
    for label, prompt in nextday_prompts:
        print(f"\n--- {label} ---")
        print(f"Prompt: \"{prompt}\"")
        res = engine.generate_grounded(prompt, max_new_tokens=80)
        print(f"  Generated Text: \"{res['text'][:200]}...\"")
        print(f"  Telemetry: {res['tokens_generated']} tokens | System 1: {res['system1_tokens']} | System 2: {res['system2_tokens']}")
        print(f"  Avg Thinking Ticks: {res['avg_thinking_ticks']:.2f} ticks/token | Mean Causal Error: {res['mean_causal_error']:.4f} rad")
        phase4_gen.append({
            "label": label,
            "tokens": res["tokens_generated"],
            "s1_tokens": res["system1_tokens"],
            "s2_tokens": res["system2_tokens"],
            "avg_ticks": res["avg_thinking_ticks"],
            "causal_error": res["mean_causal_error"]
        })
    experiment_report["nextday_metrics"] = phase4_gen

    # 4B. Rigorous Lifelong Retention Audit (Proof of Zero Catastrophic Forgetting)
    print("\n[PART 4B: Rigorous Lifelong Retention Audit]")
    audit_res = engine.audit_lifelong_integrity()
    print("  Mathematical Precision Audit:")
    for m in audit_res["math_checks"]:
        print(f"    {m['expr']} = {m['pred']} (Expected: {m['expected']}) -> {'[100% PASS]' if m['passed'] else '[FAIL]'}")

    print("\n  Riemannian Word Analogy Audit:")
    for a in audit_res["analogy_checks"]:
        print(f"    {a['query']} -> '{a['pred']}' (Expected: '{a['expected']}') -> {'[PASS]' if a['passed'] else '[FAIL]'}")

    print("\n" + "-" * 80)
    print(f"LIFELONG ZERO-FORGETTING STATUS: {'100% PERFECT RETENTION' if audit_res['all_passed'] else 'DEGRADED'}")
    print(f"Final Cognitive Manifold State: {audit_res['active_experts']} Experts | {audit_res['semantic_invariants']} Semantic Invariants | {audit_res['episodic_traces']} Episodic Traces")
    print("-" * 80)
    experiment_report["lifelong_audit"] = audit_res

    # Save Telemetry to disk
    with open(TELEMETRY_LOG, "w") as f:
        json.dump(experiment_report, f, indent=2)
    print(f"\n[+] Complete Telemetry logged to: {TELEMETRY_LOG}")


if __name__ == "__main__":
    run_unified_general_intelligence_experiment()
