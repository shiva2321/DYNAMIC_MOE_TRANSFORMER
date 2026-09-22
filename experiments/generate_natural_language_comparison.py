"""
Generate Natural Language Comparison: Baseline SLM vs Grounded Hybrid Engine (ISHMN).

Demonstrates actual side-by-side natural language text generation:
- Baseline SLM (Qwen2.5-1.5B) without manifold grounding.
- Grounded Hybrid Engine (Qwen2.5-1.5B + ISHMN Latent Manifold Sandwich + HIDLA).
- Step-by-step cognitive deliberation telemetry (System 1 vs System 2, ZPD verdicts, thinking ticks).
"""

import os
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn import UniversalCognitiveEngine
from ishmn.grounded_hybrid_engine import GroundedHybridEngine

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def run_comparative_generation():
    print("=" * 85)
    print(" LIVE NATURAL LANGUAGE GENERATION: BASELINE SLM vs GROUNDED HYBRID ENGINE")
    print(" Architectural Bridge: Small Language Model (Qwen2.5-1.5B) + ISHMN S^7 Manifold")
    print("=" * 85)
    print(f"[*] Execution Device: {DEVICE}")

    # Initialize Grounded Hybrid Engine
    engine = GroundedHybridEngine(
        model_name=MODEL_NAME,
        brain_path=BRAIN_PATH,
        device=DEVICE
    )

    test_prompts = [
        {
            "category": "Causal Thermodynamics & Physical Invariance",
            "prompt": "Explain in 2 precise sentences what happens when liquid nitrogen at -196 C is poured directly onto a white-hot glowing iron anvil."
        },
        {
            "category": "Neuroscience & Memory Systems (Trained Knowledge)",
            "prompt": "How does the hippocampus organize spatial cognitive maps and theta oscillations during memory consolidation?"
        },
        {
            "category": "Mathematical Signal Processing (Trained Knowledge)",
            "prompt": "Explain the fundamental mathematical function of the Fourier Transform in 2 concise sentences."
        },
        {
            "category": "Operational Conservation & Counterfactual Reversibility",
            "prompt": "Can a frictionless pendulum in a closed vacuum swing higher than its release point without external energy? Explain why."
        },
        {
            "category": "Chaotic Nonsense / Adversarial Hallucination Attack",
            "prompt": "Purple gasoline bananas violently freeze into solid copper gears, creating quantum cheese. How does this power a steam locomotive?"
        }
    ]

    for idx, item in enumerate(test_prompts, 1):
        cat = item["category"]
        prompt = item["prompt"]

        print("\n" + "#" * 85)
        print(f" TEST PROMPT {idx}: [{cat}]")
        print(f" Prompt: \"{prompt}\"")
        print("#" * 85)

        # --------------------------------------------------------------------
        # 1. Baseline SLM Generation (Standard Autoregressive Next-Token Guessing)
        # --------------------------------------------------------------------
        t0 = time.time()
        res_baseline = engine.generate_grounded(prompt, max_new_tokens=85, enable_sandwich=False)
        t_base = time.time() - t0
        base_text = res_baseline["text"]

        # --------------------------------------------------------------------
        # 2. Grounded Hybrid Generation (Latent Manifold Sandwich with HIDLA)
        # --------------------------------------------------------------------
        t0 = time.time()
        res_grounded = engine.generate_grounded(prompt, max_new_tokens=85, enable_sandwich=True)
        t_ground = time.time() - t0
        grounded_text = res_grounded["text"]

        # --------------------------------------------------------------------
        # 3. Print Side-by-Side Outputs
        # --------------------------------------------------------------------
        print("\n[A. BASELINE SLM RESPONSE (No Manifold Deliberation)]:")
        print(f"  \"{base_text}\"")
        print(f"  (Tokens: {res_baseline['tokens_generated']} | Time: {t_base:.2f}s | Mode: Pure Unanchored Probability)")

        print("\n[B. GROUNDED HYBRID ENGINE RESPONSE (With ISHMN Manifold Sandwich)]:")
        print(f"  \"{grounded_text}\"")
        print(f"  (Tokens: {res_grounded['tokens_generated']} | Time: {t_ground:.2f}s)")

        # --------------------------------------------------------------------
        # 4. Cognitive Telemetry & Thinking Mechanics
        # --------------------------------------------------------------------
        telem = res_grounded["telemetry"]
        s1_toks = res_grounded.get("system1_tokens", 0)
        s2_toks = res_grounded.get("system2_tokens", 0)
        avg_ticks = res_grounded.get("avg_thinking_ticks", 1.0)
        mean_err = res_grounded.get("mean_causal_error", 0.0)
        mean_surp = res_grounded.get("mean_surprise", 0.0)

        shield_count = sum(1 for t in telem if "Cognitive Protection" in t.get("system", ""))
        fertile_count = sum(1 for t in telem if "zpd_fertile" == t.get("zpd_verdict", ""))
        mastered_count = sum(1 for t in telem if "mastered" == t.get("zpd_verdict", ""))

        print("\n  [Cognitive Deliberation Telemetry (Layer 14 Intercept)]:")
        print(f"    Total Forward Passes      : {len(telem)}")
        print(f"    System 1 Instant Intuition: {s1_toks} tokens (1-pass reflex)")
        print(f"    System 2 Deliberation     : {s2_toks} tokens (looped world model deliberation)")
        print(f"    Average Thinking Ticks    : {avg_ticks:.2f} ticks / token")
        print(f"    Mean Manifold Surprise    : {mean_surp:.4f} rad on S^7")
        print(f"    Mean Causal Rollback Error: {mean_err:.4f}")
        print(f"    ZPD Classification        : {mastered_count} Mastered | {fertile_count} Fertile ZPD | {shield_count} Protection Shields Active")

    print("\n" + "=" * 85)
    print(" COMPARATIVE NATURAL LANGUAGE GENERATION COMPLETED!")
    print("=" * 85)


if __name__ == "__main__":
    run_comparative_generation()
