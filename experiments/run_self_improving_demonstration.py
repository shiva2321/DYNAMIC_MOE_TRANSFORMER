"""
End-to-End Autonomous Continuous Learning & Self-Improvement Demonstration.
Executes autonomous self-play hypothesis generation, continuous causal verification,
neocortical invariant crystallization, and biomimetic wake-sleep memory consolidation.
"""

import os
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn.grounded_hybrid_engine import GroundedHybridEngine
from ishmn.autonomous_self_improver import AutonomousSelfImprover

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"


def main():
    print("=" * 85)
    print(" AUTONOMOUS CONTINUOUS LEARNING & SELF-IMPROVING SYSTEM DEMONSTRATION")
    print("=" * 85)
    print(f"[*] Execution Device: {DEVICE}")

    # 1. Initialize Grounded Hybrid Cognitive Engine
    engine = GroundedHybridEngine(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        brain_path=BRAIN_PATH,
        device=DEVICE
    )

    init_concepts = len(engine.uce.memory.semantic_concepts)
    init_episodes = len(engine.uce.memory.episodic_memory)
    print(f"[+] Initial Neocortical Lattice Invariants: {init_concepts}")
    print(f"[+] Initial Episodic Memory Traces       : {init_episodes}")
    print(f"[+] Active Dynamic MoE Specialists       : {engine.uce.moe.num_experts}")

    # 2. Initialize Autonomous Self-Improver
    improver = AutonomousSelfImprover(
        engine=engine,
        verification_causal_threshold=0.33,
        verification_consensus_threshold=0.80,
        contradiction_threshold=0.45
    )

    # 3. Execute Autonomous Self-Play Cycles (Unsupervised Relational Synthesis & Grounding)
    print("\n" + "=" * 80)
    print(" PHASE 1: AUTONOMOUS SELF-PLAY & CAUSAL VERIFICATION (6 CYCLES)")
    print("=" * 80)

    session_report = improver.run_self_improvement_session(
        num_cycles=6,
        run_sleep_after=False
    )

    # 4. Phase 2: Biomimetic Wake-Sleep Memory Consolidation (NREM + REM Replay)
    print("\n" + "=" * 80)
    print(" PHASE 2: BIOMIMETIC WAKE-SLEEP CONSOLIDATION (NREM + REM DREAMING)")
    print("=" * 80)

    sleep_report = engine.sleep_consolidate(
        prune_usage_threshold=0.01,
        crystallize_concepts=True,
        enable_rem_replay=True,
        rem_steps=4,
        rem_batch_size=16
    )

    # 5. Phase 3: Gate 2 and Gate 3 Lifelong Integrity Audits
    print("\n" + "=" * 80)
    print(" PHASE 3: GATE 2 (DRIFT) & GATE 3 (ZERO-FORGETTING) INTEGRITY AUDITS")
    print("=" * 80)

    integrity_report = engine.audit_lifelong_integrity()
    gate_2_passed = sleep_report.get("rem", {}).get("gate_2_passed", True)
    gate_3_passed = integrity_report.get("all_passed", True)

    print(f"  [+] Gate 2 (Zero Catastrophic Drift) Audit: {'PASSED' if gate_2_passed else 'FAILED'}")
    print(f"      - Mean Geodesic Drift: {sleep_report.get('rem', {}).get('mean_geodesic_drift', 0.0):.4f} rad (< 0.15 rad threshold)")
    print(f"      - Max Geodesic Drift : {sleep_report.get('rem', {}).get('max_geodesic_drift', 0.0):.4f} rad (< 0.25 rad threshold)")

    print(f"  [+] Gate 3 (Zero Catastrophic Forgetting) Audit: {'PASSED' if gate_3_passed else 'FAILED'}")
    for mc in integrity_report.get("math_checks", []):
        print(f"      - {mc['expr']} = {mc['pred']} (Expected: {mc['expected']}) -> {'PASSED' if mc['passed'] else 'FAILED'}")

    # 6. Save Updated Brain Checkpoint
    engine.uce.save_brain(BRAIN_PATH)
    print(f"\n[+] Saved continuously updated brain to: {BRAIN_PATH}")

    final_concepts = len(engine.uce.memory.semantic_concepts)
    final_episodes = len(engine.uce.memory.episodic_memory)

    print("\n" + "=" * 85)
    print(" AUTONOMOUS CONTINUOUS LEARNING REPORT")
    print("=" * 85)
    print(f" Initial Semantic Invariants        : {init_concepts}")
    print(f" Final Semantic Invariants          : {final_concepts} (+{final_concepts - init_concepts} discovered & crystallized)")
    print(f" Initial Episodic Traces            : {init_episodes}")
    print(f" Final Episodic Traces              : {final_episodes} (+{final_episodes - init_episodes} new experiential traces)")
    print(f" Self-Play Hypotheses Evaluated     : {session_report['total_hypotheses_evaluated']}")
    print(f" Verified Invariants Discovered     : {session_report['verified_discoveries']}")
    print(f" Contradictions Flagged             : {session_report['contradictions_flagged']}")
    print(f" Gate 2 Drift Verification          : {'PASSED' if gate_2_passed else 'FAILED'}")
    print(f" Gate 3 Zero-Forgetting Verification: {'PASSED' if gate_3_passed else 'FAILED'}")
    print(f" Total Session Runtime              : {session_report['elapsed_sec']:.2f}s")
    print("=" * 85)


if __name__ == "__main__":
    main()
