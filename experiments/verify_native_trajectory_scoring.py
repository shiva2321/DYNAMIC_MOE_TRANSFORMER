"""
Native Trajectory-Level Geometric Verifier & Falsification Experiment.

Evaluates whether our Native Dual-Cortex Substrate (NativeSphericalTransformer + UniversalCognitiveEngine)
generates geometric trajectories whose path-integrated causal residual and hypersphere geodesic jitter
correlate with reasoning consistency:
    S^{(k)} = - [ (1/T) sum_t epsilon_causal(z_t, c_t) + beta * Var_t[d_S(u_t, u_{t+1})] ]

100% Sovereign: Zero external LLMs (No Qwen, no LLaMA, no external dependencies).
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn.geometry import project_to_sphere, geodesic_distance_sphere
from ishmn.universal_cognitive_engine import UniversalCognitiveEngine
from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.data_loader import SovereignTokenizer
from ishmn.training_engine import MODEL_PROFILES
from experiments.benchmark_official_reasoning import extract_gsm8k_answer, answers_match

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"
DEFAULT_500M_CHECKPOINT = "artifacts/checkpoints/native_moe_500m_peak.pt"


def score_native_trajectory(
    wm,
    z_seq: torch.Tensor,
    beta: float = 0.5
) -> Tuple[float, float, float]:
    """
    Computes path-integrated causal residual and geodesic jitter for a native latent trajectory.
    z_seq: [T, d_model]
    Returns: (total_score, mean_causal_err, geodesic_jitter)
    """
    T = z_seq.size(0)
    if T < 2:
        return 0.0, 0.0, 0.0

    # Running cumulative context: c_t = mean(z_{1:t})
    cumsum_z = torch.cumsum(z_seq, dim=0)
    steps = torch.arange(1, T + 1, device=z_seq.device, dtype=z_seq.dtype).unsqueeze(-1)
    c_seq = cumsum_z / steps

    # 1. Path-integrated Pearl-L3 Causal Residual
    with torch.no_grad():
        causal_res = wm.causal_residual(z_seq, c_seq)  # [T]
        mean_causal = causal_res.mean().item()

    # 2. Geodesic Trajectory Jitter on Hypersphere
    u_seq, _ = wm.project_manifold(z_seq)  # [T, hyperspace_dim]
    u_curr = u_seq[:-1]
    u_next = u_seq[1:]
    step_geodesics = geodesic_distance_sphere(u_curr, u_next).squeeze(-1)  # [T-1]
    jitter_var = torch.var(step_geodesics).item() if len(step_geodesics) > 1 else 0.0

    # Total score: Geometric Coherence (Higher is better)
    total_score = -(mean_causal + beta * jitter_var)
    return total_score, mean_causal, jitter_var


def run_native_experiment(
    num_samples: int = 15,
    k_completions: int = 4,
    temperature: float = 0.8,
    max_new_tokens: int = 60,
    beta: float = 0.5,
    output_path: str = "artifacts/native_trajectory_scoring_results.json"
):
    print("=" * 88)
    print(" 100% NATIVE TRAJECTORY-LEVEL GEOMETRIC VERIFIER (ZERO THIRD-PARTY LLMS)")
    print(f" Execution Hardware : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Samples: {num_samples} | K: {k_completions} | Temp: {temperature}")
    print("=" * 88, flush=True)

    # 1. Load Native 500M Architecture & UCE
    cfg = MODEL_PROFILES["500m"]
    tokenizer = SovereignTokenizer("gpt2")
    max_seq_len = 96

    model = NativeSphericalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        hyperspace_dim=cfg["hyperspace_dim"],
        d_ff=cfg["d_ff"],
        max_seq_len=max_seq_len,
        mtp_future_tokens=cfg["mtp_future_tokens"],
        vsa_capacity=cfg["vsa_capacity"],
        num_experts=cfg["num_experts"],
        top_k=cfg["top_k"],
        enable_shared_expert=cfg["enable_shared_expert"],
        enable_aux_loss_free_balancing=cfg["enable_aux_loss_free_balancing"],
        enable_vsa_memory=True,
        enable_world_model=True,
        tie_weights=True,
        device=DEVICE
    ).to(DEVICE)

    uce = UniversalCognitiveEngine(
        d_model=cfg["d_model"],
        d_hidden=cfg["d_ff"],
        hyperspace_dim=cfg["hyperspace_dim"],
        num_memory_slots=32,
        vocab_size=tokenizer.vocab_size,
        initial_experts=cfg["num_experts"],
        device=DEVICE
    ).to(DEVICE)
    uce.attach_native_transformer(model, tokenizer=tokenizer)

    if os.path.exists(DEFAULT_500M_CHECKPOINT):
        print(f"[*] Loading Native 500M Checkpoint: {DEFAULT_500M_CHECKPOINT}...")
        ckpt = torch.load(DEFAULT_500M_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        uce.load_state_dict(ckpt["uce_state_dict"], strict=True)
        print(f"[+] Successfully loaded Native Dual-Cortex 500M Brain from step {ckpt.get('step', 0)}.")
    
    model.eval()
    uce.eval()
    wm = uce.world_model

    # 3. Load GSM8K Test Questions
    from datasets import load_dataset
    print("[*] Loading GSM8K test split for native reasoning evaluation...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    greedy_correct = 0
    maj_vote_correct = 0
    verifier_correct = 0
    total_evaluated = 0

    all_scores = []
    all_correctness = []
    results_log = []

    for i, item in enumerate(ds):
        if total_evaluated >= num_samples:
            break

        question = item["question"]
        gold_answer_str = item["answer"]
        gold_num = extract_gsm8k_answer(gold_answer_str)
        if gold_num is None:
            continue

        prompt = f"Question: {question}\nAnswer:"
        full_enc = tokenizer.encode(prompt)
        max_prompt_len = max_seq_len - max_new_tokens - 2
        if len(full_enc) > max_prompt_len:
            full_enc = full_enc[-max_prompt_len:]
        prompt_ids = torch.tensor([full_enc], dtype=torch.long, device=DEVICE)
        prompt_len = prompt_ids.size(1)

        # A. Greedy Completion (T=0)
        with torch.no_grad():
            greedy_gen = model.generate(prompt_ids, max_new_tokens=max_new_tokens, temperature=0.0)
            greedy_txt = tokenizer.decode(greedy_gen[0, prompt_len:])
            greedy_num = extract_gsm8k_answer(greedy_txt)
            is_greedy_correct = answers_match(greedy_num, gold_num)
            if is_greedy_correct:
                greedy_correct += 1

        # B. K Sampled Completions with Native Trajectory Extraction
        completions = []
        for k in range(k_completions):
            with torch.no_grad():
                gen_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=40
                )
                gen_tokens = gen_ids[0, prompt_len:]
                gen_txt = tokenizer.decode(gen_tokens)
                pred_num = extract_gsm8k_answer(gen_txt)
                is_correct = answers_match(pred_num, gold_num)

                # Extract native latent trajectory h through model forward pass
                out = model(gen_ids, return_mtp=False)
                # Hidden states corresponding to the generated reasoning trajectory
                h_seq = out["hidden_states"][0, prompt_len:, :]  # [T_gen, d_model]

                score, mean_causal, jitter = score_native_trajectory(wm, h_seq, beta=beta)

                completions.append({
                    "k": k,
                    "text": gen_txt,
                    "pred_num": pred_num,
                    "is_correct": is_correct,
                    "score": score,
                    "causal_residual": mean_causal,
                    "jitter": jitter
                })

                if pred_num is not None:
                    all_scores.append(score)
                    all_correctness.append(1 if is_correct else 0)

        # C. Self-Consistency Majority Vote
        valid_preds = [c["pred_num"] for c in completions if c["pred_num"] is not None]
        if valid_preds:
            maj_vote_pred = Counter(valid_preds).most_common(1)[0][0]
        else:
            maj_vote_pred = greedy_num
        is_maj_correct = answers_match(maj_vote_pred, gold_num)
        if is_maj_correct:
            maj_vote_correct += 1

        # D. Manifold Geometric Verifier (Select argmax Geometric Coherence)
        scored_candidates = [c for c in completions if c["pred_num"] is not None]
        if scored_candidates:
            # Maximizes geometric coherence (highest score)
            best_cand = max(scored_candidates, key=lambda c: c["score"])
            verifier_pred = best_cand["pred_num"]
        else:
            verifier_pred = greedy_num
        is_verifier_correct = answers_match(verifier_pred, gold_num)
        if is_verifier_correct:
            verifier_correct += 1

        total_evaluated += 1
        results_log.append({
            "idx": i,
            "gold": gold_num,
            "greedy": {"pred": greedy_num, "correct": is_greedy_correct},
            "majority_vote": {"pred": maj_vote_pred, "correct": is_maj_correct},
            "manifold_verifier": {"pred": verifier_pred, "correct": is_verifier_correct},
            "completions": completions
        })

        print(f"  [{total_evaluated:02d}/{num_samples}] Gold: {str(gold_num):>6s} | Greedy: {str(is_greedy_correct):5s} | MajVote: {str(is_maj_correct):5s} | GeoVerifier: {str(is_verifier_correct):5s}", flush=True)

    # 4. Statistical Analysis
    greedy_acc = greedy_correct / max(total_evaluated, 1)
    maj_acc = maj_vote_correct / max(total_evaluated, 1)
    verifier_acc = verifier_correct / max(total_evaluated, 1)

    spearman_rho, p_val = 0.0, 1.0
    if len(all_scores) > 3 and len(set(all_correctness)) > 1:
        spearman_rho, p_val = stats.spearmanr(all_scores, all_correctness)

    print("\n" + "=" * 88)
    print(" NATIVE TRAJECTORY VERIFICATION RESULTS (100% Sovereign Native Model)")
    print("=" * 88)
    print(f"Total Evaluated Items           : {total_evaluated}")
    print(f"Greedy Baseline Accuracy (T=0) : {greedy_acc * 100:.1f}% ({greedy_correct}/{total_evaluated})")
    print(f"Self-Consistency Majority Vote  : {maj_acc * 100:.1f}% ({maj_vote_correct}/{total_evaluated})")
    print(f"Manifold Geometric Verifier     : {verifier_acc * 100:.1f}% ({verifier_correct}/{total_evaluated})")
    print(f"Spearman Correlation (rho)      : {spearman_rho:.4f} (p-value: {p_val:.4e})")
    print("=" * 88, flush=True)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "samples_evaluated": total_evaluated,
            "greedy_accuracy": greedy_acc,
            "majority_vote_accuracy": maj_acc,
            "manifold_verifier_accuracy": verifier_acc,
            "spearman_rho": float(spearman_rho),
            "p_value": float(p_val),
            "details": results_log
        }, f, indent=2)
    print(f"[+] Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--beta", type=float, default=0.5)
    args = parser.parse_args()

    run_native_experiment(
        num_samples=args.samples,
        k_completions=args.k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        beta=args.beta
    )
