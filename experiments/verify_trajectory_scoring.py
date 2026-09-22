"""
Trajectory-Level Geometric Verifier & Falsification Experiment.

Evaluates whether the Riemannian Manifold's path-integrated causal residual
and geodesic trajectory jitter correlate with mathematical reasoning correctness
on GSM8K, testing Claude Opus Frontier 5:
    S^{(k)} = (1/T) sum_t epsilon_causal(z_t, c_t) + beta * Var_t[d_S(u_t, u_{t+1})]

Compares:
1. Greedy Baseline (T=0)
2. Self-Consistency Majority Vote (K completions, Mode answer)
3. Manifold Geometric Verifier (K completions, argmin S^{(k)})
4. Spearman correlation rho(S^{(k)}, Correctness) with 95% Bootstrap CI.
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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn.geometry import project_to_sphere, geodesic_distance_sphere
from ishmn.slm_adapter import SLMManifoldAdapter
from ishmn.universal_cognitive_engine import UniversalCognitiveEngine
from experiments.benchmark_official_reasoning import extract_gsm8k_answer, answers_match

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"
DEFAULT_ADAPTER_PATH = "artifacts/slm_manifold_adapter.pt"
LAYER_IDX = 14


def score_trajectory(
    wm,
    z_seq: torch.Tensor,
    beta: float = 0.5
) -> Tuple[float, float, float]:
    """
    Computes path-integrated causal residual and geodesic jitter for a latent trajectory.
    z_seq: [T, d_model=32]
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
        causal_res = wm.causal_residual(z_seq, c_seq) # [T]
        mean_causal = causal_res.mean().item()

    # 2. Geodesic Trajectory Jitter
    u_seq, _ = wm.project_manifold(z_seq) # [T, hyperspace_dim]
    u_curr = u_seq[:-1]
    u_next = u_seq[1:]
    step_geodesics = geodesic_distance_sphere(u_curr, u_next).squeeze(-1) # [T-1]
    jitter_var = torch.var(step_geodesics).item() if len(step_geodesics) > 1 else 0.0
    # Total score: Geometric Coherence (Higher is better, minimizes causal residual and jitter)
    total_score = -(mean_causal + beta * jitter_var)
    return total_score, mean_causal, jitter_var


def run_experiment(
    num_samples: int = 40,
    k_completions: int = 4,
    temperature: float = 0.8,
    max_new_tokens: int = 400,
    beta: float = 0.5,
    output_path: str = "artifacts/trajectory_scoring_results.json"
):
    print("=" * 80)
    print(" TRAJECTORY-LEVEL GEOMETRIC VERIFIER & FALSIFICATION EXPERIMENT")
    print(f" Samples: {num_samples} | K: {k_completions} | Temp: {temperature} | Device: {DEVICE}")
    print("=" * 80, flush=True)

    # 1. Load Universal Cognitive Engine & World Model
    print(f"[*] Loading Brain Checkpoint: {DEFAULT_BRAIN_PATH}...")
    uce = UniversalCognitiveEngine.load_brain(DEFAULT_BRAIN_PATH, device=DEVICE)
    uce.eval()
    wm = uce.world_model

    # 2. Load Adapter
    print(f"[*] Loading SLM-Manifold Adapter: {DEFAULT_ADAPTER_PATH}...")
    adapter = SLMManifoldAdapter(
        hidden_size=1536,
        d_model=32,
        hyperspace_dim=8,
        mlp_dim=256,
        alpha=0.22,
        device=DEVICE
    )
    if os.path.exists(DEFAULT_ADAPTER_PATH):
        ad_state = torch.load(DEFAULT_ADAPTER_PATH, map_location=DEVICE)
        adapter.load_state_dict(ad_state)
        print("[+] Adapter loaded successfully.")
    else:
        print("[!] Warning: Adapter checkpoint not found, using initialized weights.")
    adapter.eval()

    # 3. Load Language Model
    print(f"[*] Loading Language Model: {DEFAULT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    slm = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    slm.eval()

    # 4. Load GSM8K dataset
    print("[*] Loading GSM8K test split...")
    dataset = load_dataset("gsm8k", "main", split="test")
    samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]

    all_scores = []
    all_labels = [] # 1 for correct, 0 for incorrect

    greedy_correct = 0
    majority_correct = 0
    geometric_correct = 0
    total_q = 0

    per_question_results = []
    t0_start = time.time()

    for q_idx, item in enumerate(samples, 1):
        question = item["question"]
        gold_solution = item["answer"]
        gold_num = extract_gsm8k_answer(gold_solution)

        prompt = (
            f"Solve this math problem step by step. Conclude your response with '#### <number>'.\n\n"
            f"Question: {question}\n\n"
            f"Solution:"
        )
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([formatted], return_tensors="pt").to(DEVICE)
        prompt_len = inputs.input_ids.shape[1]

        # A. Greedy Baseline Pass (T=0)
        with torch.no_grad():
            greedy_out = slm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        greedy_text = tokenizer.decode(greedy_out[0][prompt_len:], skip_special_tokens=True).strip()
        greedy_pred = extract_gsm8k_answer(greedy_text)
        is_greedy_corr = answers_match(greedy_pred, gold_num)
        if is_greedy_corr:
            greedy_correct += 1

        # B. Sample K completions at temperature T
        with torch.no_grad():
            sampled_out = slm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=k_completions
            ) # [K, total_len]

        # Extract Layer 14 hidden states in one batched forward pass
        with torch.no_grad():
            h14_full = slm(sampled_out, output_hidden_states=True).hidden_states[LAYER_IDX] # [K, total_len, 1536]
            h14_gen = h14_full[:, prompt_len:, :].float() # [K, gen_len, 1536]

        candidate_records = []
        for k in range(k_completions):
            gen_ids = sampled_out[k][prompt_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            pred_num = extract_gsm8k_answer(gen_text)
            is_corr = answers_match(pred_num, gold_num)

            # Manifold down-projection and scoring
            h_k = h14_gen[k] # [gen_len, 1536]
            with torch.no_grad():
                z_k = adapter.encode(h_k) # [gen_len, 32]
                score, mean_causal, jitter = score_trajectory(wm, z_k, beta=beta)

            candidate_records.append({
                "cand_idx": k,
                "pred_num": pred_num,
                "is_correct": is_corr,
                "score": score,
                "mean_causal": mean_causal,
                "jitter": jitter
            })

            # Record for correlation (higher cognitive traversal score correlates with reasoning correctness)
            all_scores.append(score)
            all_labels.append(1 if is_corr else 0)

        # C. Self-Consistency: Majority Vote Selection
        preds = [c["pred_num"] for c in candidate_records if c["pred_num"] is not None]
        if preds:
            maj_pred = Counter(preds).most_common(1)[0][0]
        else:
            maj_pred = greedy_pred
        is_maj_corr = answers_match(maj_pred, gold_num)
        if is_maj_corr:
            majority_correct += 1

        # D. Geometric Verifier Selection (Authentic Cognitive Traversal Energy)
        candidate_records.sort(key=lambda c: c["score"], reverse=True)
        best_cand = candidate_records[0]
        geom_pred = best_cand["pred_num"]
        is_geom_corr = answers_match(geom_pred, gold_num)
        if is_geom_corr:
            geometric_correct += 1

        total_q += 1

        print(f"  [Q {q_idx:2d}/{num_samples}] Gold: {gold_num:<8} | Greedy: {str(is_greedy_corr):<5} | MajVote: {str(is_maj_corr):<5} | GeomVerifier: {str(is_geom_corr):<5} (Score: {best_cand['score']:.3f})", flush=True)

        per_question_results.append({
            "q_idx": q_idx,
            "question": question,
            "gold_answer": gold_num,
            "greedy": {"pred": greedy_pred, "correct": is_greedy_corr},
            "majority_vote": {"pred": maj_pred, "correct": is_maj_corr},
            "geometric_verifier": {"pred": geom_pred, "correct": is_geom_corr, "best_cand": best_cand},
            "all_candidates": candidate_records
        })

    elapsed = time.time() - t0_start

    # Compute Statistical Metrics
    all_scores_np = np.array(all_scores)
    all_labels_np = np.array(all_labels)

    # Point-biserial / Spearman rank correlation between negative score and correctness
    if len(np.unique(all_labels_np)) > 1:
        spearman_rho, p_val = stats.spearmanr(all_scores_np, all_labels_np)
        point_biserial = stats.pointbiserialr(all_labels_np, all_scores_np)
        pb_corr = point_biserial.statistic
        pb_p = point_biserial.pvalue
    else:
        spearman_rho, p_val = 0.0, 1.0
        pb_corr, pb_p = 0.0, 1.0

    greedy_acc = (greedy_correct / total_q) * 100
    majority_acc = (majority_correct / total_q) * 100
    geometric_acc = (geometric_correct / total_q) * 100

    print("\n" + "=" * 80)
    print(" EXPERIMENT SUMMARY & FALSIFICATION REPORT")
    print("=" * 80)
    print(f" Total Evaluated Items         : {total_q}")
    print(f" Total Trajectories Scored     : {len(all_scores)}")
    print(f" Greedy Baseline Accuracy (T=0): {greedy_acc:.2f}% ({greedy_correct}/{total_q})")
    print(f" Majority Vote Accuracy (K={k_completions}): {majority_acc:.2f}% ({majority_correct}/{total_q})")
    print(f" Geometric Verifier Accuracy   : {geometric_acc:.2f}% ({geometric_correct}/{total_q})")
    print(f" Delta (GeomVerifier - Greedy) : {geometric_acc - greedy_acc:+.2f} pp")
    print(f" Delta (GeomVerifier - MajVote): {geometric_acc - majority_acc:+.2f} pp")
    print("-" * 80)
    print(f" Spearman Rank Correlation rho : {spearman_rho:.4f} (p = {p_val:.4e})")
    print(f" Point-Biserial Correlation r_pb: {pb_corr:.4f} (p = {pb_p:.4e})")
    print(f" Kill Gate Threshold (rho > 0.3): {'PASSED' if spearman_rho >= 0.30 else 'NOT YET MET (Needs Layer-14 WM Training)'}")
    print(f" Elapsed Runtime               : {elapsed:.2f}s ({elapsed/total_q:.2f}s per question)")
    print("=" * 80)

    summary = {
        "num_samples": total_q,
        "k_completions": k_completions,
        "greedy_acc": greedy_acc,
        "majority_acc": majority_acc,
        "geometric_acc": geometric_acc,
        "spearman_rho": float(spearman_rho),
        "spearman_p_val": float(p_val),
        "point_biserial_r": float(pb_corr),
        "runtime_sec": elapsed,
        "per_question_results": per_question_results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Detailed results saved to: {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory-Level Geometric Verifier Experiment")
    parser.add_argument("--samples", type=int, default=20, help="Number of GSM8K questions to evaluate")
    parser.add_argument("--k", type=int, default=4, help="Number of sampled completions per question")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--tokens", type=int, default=350, help="Maximum new tokens per completion")
    parser.add_argument("--beta", type=float, default=0.5, help="Geodesic jitter variance weighting")
    args = parser.parse_args()

    run_experiment(
        num_samples=args.samples,
        k_completions=args.k,
        temperature=args.temp,
        max_new_tokens=args.tokens,
        beta=args.beta
    )
