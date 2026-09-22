"""
Stage 2 Empirical Verification: Dual-Phase Sleep (NREM + REM) & Geodesic Drift Audit on Live Brain.
Tests the GroundedHybridEngine with:
1. Daytime operational streaming (new sensory experiences)
2. Phase 1 (NREM): Barycentric invariant crystallization on S^7
3. Phase 2 (REM): Generative tangent Brownian replay + LeJEPA SIGReg
4. Gate 2 Falsification Gate: Geodesic drift audit (mean <= 0.15 rad, max <= 0.25 rad)
5. Gate 3 Zero-Forgetting Verification: 100% exact retention across arithmetic & analogies.
"""

import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ishmn import GroundedHybridEngine

BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_stage2_live_verification():
    print("=" * 78)
    print("STAGE 2: LIVE DUAL-PHASE SLEEP (NREM + REM) & GEODESIC DRIFT AUDIT")
    print(f"Device: {DEVICE} | Target Brain: {BRAIN_PATH}")
    print("=" * 78)

    t0 = time.time()
    engine = GroundedHybridEngine(brain_path=BRAIN_PATH, device=DEVICE)
    print(f"[+] Engine loaded in {time.time() - t0:.2f}s.")

    # 1. Daytime Operational Streaming
    print("\n--- PHASE 1: DAYTIME SENSORY INGESTION ---")
    daytime_events = [
        ("Topological insulators exhibit conducting edge states protected by time-reversal symmetry.", "physics"),
        ("CRISPR Cas12a induces staggered double-strand breaks with 5-nucleotide overhangs.", "biology"),
        ("Zero-knowledge rollups compress execution state transitions using succinct non-interactive arguments.", "cryptography")
    ]
    daytime_report = engine.learn_daytime_stream(daytime_events)
    print(f"[*] Ingested {daytime_report['observations_ingested']} daytime observations in {daytime_report['elapsed_sec']:.2f}s.")
    print(f"    - Mean Abductive Surprise: {daytime_report['mean_surprise']:.4f}")
    print(f"    - Active Episodic Buffer: {daytime_report['total_episodic_traces']} traces")

    # 2. Dual-Phase Sleep Consolidation (NREM + REM Generative Replay)
    print("\n--- PHASE 2: DUAL-PHASE SLEEP CONSOLIDATION (NREM + REM) ---")
    sleep_report = engine.sleep_consolidate(
        prune_usage_threshold=0.01,
        crystallize_concepts=True,
        enable_rem_replay=True,
        rem_steps=8,
        rem_batch_size=16,
        perturbation_sigma=0.05
    )

    nrem = sleep_report["nrem"]
    rem = sleep_report["rem"]

    print(f"[*] Sleep Cycle Completed in {sleep_report['elapsed_sec']:.2f}s.")
    print(f"    - NREM Crystallized Invariants: {nrem.get('semantic_crystallization', {}).get('num_crystallized', 0)}")
    print(f"    - Active Semantic Invariants on S^7: {sleep_report['total_semantic_concepts']}")
    print(f"    - REM Replay Executed: {rem['rem_executed']} ({rem.get('rem_steps', 0)} steps)")
    print(f"    - REM Mean Loss: {rem.get('mean_rem_loss', 0.0):.4f}")
    print(f"    - Mean Geodesic Drift: {rem.get('mean_geodesic_drift', 0.0):.6f} rad (Threshold: <= 0.15 rad)")
    print(f"    - Max Geodesic Drift: {rem.get('max_geodesic_drift', 0.0):.6f} rad (Threshold: <= 0.25 rad)")
    print(f"    - Gate 2 Falsification Status: {'PASSED' if rem.get('gate_2_passed') else 'FAILED'}")

    # 3. Lifelong Zero-Forgetting Audit (Gate 3)
    print("\n--- PHASE 3: LIFELONG INTEGRITY & ZERO-FORGETTING AUDIT (GATE 3) ---")
    audit = engine.audit_lifelong_integrity()

    print("\n[Scale-Invariant Arithmetic Integrity]")
    for m in audit["math_checks"]:
        status = "PASS" if m["passed"] else "FAIL"
        print(f"  [{status}] {m['expr']} = {m['pred']} (expected: {m['expected']})")

    print("\n[Riemannian Geometric Analogy Invariants]")
    for a in audit["analogy_checks"]:
        status = "PASS" if a["passed"] else "FAIL"
        print(f"  [{status}] {a['query']} -> {a['pred']} (expected: {a['expected']})")

    print("\n" + "=" * 78)
    print(f"Gate 2 (Geodesic Drift Audit):     {'PASSED' if rem.get('gate_2_passed') else 'FAILED'}")
    print(f"Gate 3 (Zero-Forgetting Audit):    {'PASSED' if audit['all_passed'] else 'FAILED'}")
    print("=" * 78)

    if rem.get("gate_2_passed") and audit["all_passed"]:
        print(">>> ALL STAGE 2 GATES PASSED! Dual-phase sleep operates with zero representation drift.")
    else:
        print(">>> CRITICAL: STAGE 2 FAILED. Drift exceeded or historical capabilities degraded.")


if __name__ == "__main__":
    run_stage2_live_verification()
