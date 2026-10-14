"""
bench_y5_sparse_retrieval.py - Benchmark runner for Round Y5 (E5 Sparse Retrieval) on TRAINED models.
Part 1: Top-B captured attention mass (B in {4, 8, 16, 32}) for Y6-trained hyp arms and Euclid controls on test batches.
Part 2: Branch-and-bound tree retrieval on trained keys across N in {512, 1024, 2048, 4096} vs dense attention.

ASCII output only. Peak memory strictly < 4 GB.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import time
import math
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F

from geometry.lorentz import LorentzModel
from yggformer.config import YggCfg
from yggformer.model import YggFormer
from yggformer.tree_search import LorentzKeyTree, branch_and_bound_top_b
from research.yggformer_results.bench_y6_lm_sinks import load_stdlib_corpus, get_batches


def run_part1_trained_mass_capture(device: torch.device) -> Dict:
    print("\n--- PART 1: TOP-B ATTENTION MASS CAPTURE ON TRAINED MODELS (B in 4, 8, 16, 32) ---")
    B_list = [4, 8, 16, 32]
    arms = ["euclid", "euclid_temp", "hyp", "hyp_phase", "hyp_phase_bias", "hyp_phase_bias_null"]

    _, test_text, char2idx, _ = load_stdlib_corpus()
    test_batches = get_batches(test_text, char2idx, seq_len=64, batch_size=8, n_batches=10, seed=10)

    results = {}

    for arm in arms:
        ckpt_path = f"research/yggformer_results/y6_model_{arm}.pt"
        if not os.path.exists(ckpt_path):
            print(f"Warning: checkpoint {ckpt_path} not found, skipping {arm}")
            continue

        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = YggFormer(saved["cfg"]).to(device)
        model.load_state_dict(saved["model_state"])
        model.eval()

        arm_masses = {b: [] for b in B_list}

        with torch.no_grad():
            for x, _ in test_batches:
                x = x.to(device)
                B_sz, T = x.shape
                positions = torch.arange(T, device=device)
                h = model.tok_emb(x) + model.pos_emb(positions)

                for block in model.blocks:
                    norm_in = block.norm1(h)
                    attn = block.attn
                    H = attn.n_heads
                    DH = attn.head_dim
                    qkv = attn.qkv(norm_in).view(B_sz, T, 3, H, DH).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]

                    if attn.null_k is not None:
                        k = torch.cat([attn.null_k.expand(B_sz, -1, 1, -1), k], dim=2)
                        causal_mask = torch.ones(T, T + 1, dtype=torch.bool, device=device)
                        causal_mask[:, 1:] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
                        causal_mask = causal_mask.view(1, 1, T, T + 1)
                    else:
                        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device)).view(1, 1, T, T)

                    if attn.geometry in {"euclid", "euclid_gate"}:
                        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(DH)
                    elif attn.geometry == "euclid_temp":
                        inv_tau = (1.0 / (F.softplus(attn.g_tau) + 1e-3)).view(1, H, 1, 1)
                        logits = torch.matmul(q * inv_tau, k.transpose(-1, -2)) / math.sqrt(DH)
                    else:
                        c = (F.softplus(attn.g_c) + 1e-3).view(1, H, 1, 1)
                        tau = (F.softplus(attn.g_tau) + 1e-3).view(1, H, 1, 1)
                        q_scaled = q * attn.q_scale
                        k_scaled = k * attn.k_scale
                        if attn.r_max is not None:
                            from geometry.attention import clip_radius
                            q_scaled = clip_radius(q_scaled, attn.r_max)
                            k_scaled = clip_radius(k_scaled, attn.r_max)

                        th_q, th_k = None, None
                        if attn.phase is not None:
                            ph = attn.phase(norm_in).view(B_sz, T, 2, H, attn.n_phase).permute(2, 0, 3, 1, 4)
                            th_q, th_k = ph[0], ph[1]
                            if attn.null_k is not None:
                                th_k = torch.cat([torch.zeros(B_sz, H, 1, attn.n_phase, device=device), th_k], dim=2)

                        kb = None
                        if attn.bias_proj is not None:
                            kb = attn.bias_proj(norm_in).permute(0, 2, 1)
                            if attn.null_k is not None:
                                kb = torch.cat([attn.null_beta.expand(B_sz, -1, 1), kb], dim=2)

                        from geometry.attention import augment
                        kappa = F.softplus(attn.g_kappa).view(1, H, 1, 1) if attn.phase is not None else 1.0
                        q_aug, k_aug, _, _ = augment(q_scaled, k_scaled, k_scaled, c=c, tau=tau, theta_q=th_q, theta_k=th_k, kappa=kappa, key_bias=kb)
                        logits = torch.matmul(q_aug, k_aug.transpose(-1, -2))

                    logits = logits.masked_fill(~causal_mask, float("-inf"))
                    probs = F.softmax(logits, dim=-1) # (B_sz, H, T, S)

                    # Sample queries with sufficient history (t >= 32)
                    for t in range(32, T):
                        # valid keys up to position t (plus null key if present)
                        p_t = probs[:, :, t, : t + 2 if attn.null_k is not None else t + 1] # (B_sz, H, S_valid)
                        for b in B_list:
                            if p_t.shape[-1] >= b:
                                top_p, _ = torch.topk(p_t, k=b, dim=-1)
                                arm_masses[b].append(top_p.sum(dim=-1).mean().item())

                    h = block(h, is_causal=True)

        results[arm] = {}
        for b in B_list:
            m_mean = float(np.mean(arm_masses[b]))
            m_std = float(np.std(arm_masses[b]))
            results[arm][str(b)] = {"mean": m_mean, "std": m_std}
            print(f"  Arm {arm:20s} | Top-{b:2d} Mass: {m_mean*100:5.2f}% +- {m_std*100:4.2f}%")

        del model
        torch.cuda.empty_cache()

    return results


def run_part2_trained_tree_retrieval(device: torch.device) -> Dict:
    print("\n--- PART 2: BRANCH-AND-BOUND RETRIEVAL ON TRAINED KEYS (N in 512, 1024, 2048, 4096) ---")
    N_list = [512, 1024, 2048, 4096]
    B_list = [4, 8, 16, 32]

    ckpt_path = "research/yggformer_results/y6_model_hyp.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return {}

    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = YggFormer(saved["cfg"]).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()

    _, test_text, char2idx, _ = load_stdlib_corpus()
    indices = [char2idx[c] for c in test_text if c in char2idx]

    attn0 = model.blocks[0].attn
    head_idx = 0
    c_val = float((F.softplus(attn0.g_c[head_idx]) + 1e-3).item())
    tau_val = float((F.softplus(attn0.g_tau[head_idx]) + 1e-3).item())
    dim = attn0.head_dim
    lorentz = LorentzModel(dim=dim, c=c_val)

    # Extract trained keys and queries by projecting token embeddings through layer 0
    max_N = max(N_list)
    toks = torch.tensor(indices[: max_N + 64], dtype=torch.long, device=device)
    with torch.no_grad():
        emb = model.tok_emb(toks)
        norm_in = model.blocks[0].norm1(emb)
        qkv = attn0.qkv(norm_in).view(-1, 3, attn0.n_heads, dim)
        k_s = attn0.k_scale.view(-1)[head_idx]
        q_s = attn0.q_scale.view(-1)[head_idx]
        all_k = qkv[:max_N, 1, head_idx, :] * k_s
        all_q = qkv[max_N : max_N + 20, 0, head_idx, :] * q_s

        # Project to Lorentz manifold
        all_k_lorentz = lorentz.exp_map_zero(all_k.cpu())
        all_q_lorentz = lorentz.exp_map_zero(all_q.cpu())

    results = {}

    for N in N_list:
        print(f"\nTesting Tree Retrieval at N = {N} keys...")
        keys_subset = all_k_lorentz[:N]
        t0_build = time.time()
        tree = LorentzKeyTree.build(keys_subset, max_leaf_size=16, c=c_val)
        build_time = time.time() - t0_build

        results[str(N)] = {}

        for b in B_list:
            recalls = []
            mass_ratios = []
            eval_fractions = []
            t_dense_list = []
            t_bb_list = []

            recalls = []
            mass_99_recalls = []
            mass_ratios = []
            eval_fractions = []
            t_vectorized_dense_list = []
            t_bb_list = []

            for q in all_q_lorentz:
                # Fair vectorized dense search (GPU/CPU vectorized inner product + topk)
                t0 = time.perf_counter()
                inner = -q[0] * keys_subset[:, 0] + (q[1:] * keys_subset[:, 1:]).sum(dim=-1)
                dists = torch.acosh((-c_val * inner).clamp(min=1.0 + 1e-7)) / math.sqrt(c_val)
                logits = -(torch.cosh(math.sqrt(c_val) * dists) - 1.0) / (c_val * tau_val)
                dense_top_indices = torch.topk(logits, k=b).indices.tolist()
                t_dense = (time.perf_counter() - t0) * 1000.0

                # Branch-and-bound tree search
                t1 = time.perf_counter()
                bb_top_indices, leaf_evals, node_visits = branch_and_bound_top_b(
                    tree, q, B=b, c=c_val, tau=tau_val
                )
                t_bb = (time.perf_counter() - t1) * 1000.0

                # Raw top-B index recall
                recall = len(set(dense_top_indices).intersection(set(bb_top_indices))) / b
                recalls.append(recall)

                # Mass-filtered recall over keys carrying 99% of attention mass
                probs = F.softmax(logits, dim=-1)
                p_sorted, p_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(p_sorted, dim=-1)
                k_99 = max(1, int((cum <= 0.99).sum().item()))
                mass_99_keys = set(p_idx[:k_99].tolist())
                mass_99_rec = len(mass_99_keys.intersection(set(bb_top_indices))) / len(mass_99_keys)
                mass_99_recalls.append(mass_99_rec)

                true_mass = probs[dense_top_indices].sum().item()
                bb_mass = probs[bb_top_indices].sum().item()
                mass_ratios.append(bb_mass / max(true_mass, 1e-8))

                eval_fractions.append(leaf_evals / N)
                t_vectorized_dense_list.append(t_dense)
                t_bb_list.append(t_bb)

            mean_rec = float(np.mean(recalls))
            mean_m99_rec = float(np.mean(mass_99_recalls))
            mean_mass_r = float(np.mean(mass_ratios))
            mean_eval_f = float(np.mean(eval_fractions))
            mean_td = float(np.mean(t_vectorized_dense_list))
            mean_tbb = float(np.mean(t_bb_list))

            results[str(N)][str(b)] = {
                "top_b_recall": mean_rec,
                "mass_99_recall": mean_m99_rec,
                "mass_ratio": mean_mass_r,
                "eval_fraction": mean_eval_f,
                "t_vectorized_dense_ms": mean_td,
                "t_bb_ms": mean_tbb,
                "tree_slowdown_factor": mean_tbb / max(mean_td, 1e-6),
            }
            print(f"  N={N:4d}, B={b:2d} | Top-{b:2d} Rec: {mean_rec*100:5.1f}% | 99% Mass Rec: {mean_m99_rec*100:5.1f}% | Evaluated: {mean_eval_f*100:4.1f}% keys | BB: {mean_tbb:5.2f}ms vs Dense: {mean_td:6.3f}ms ({mean_tbb/mean_td:.0f}x slower)")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    print("=" * 70)
    print("STARTING ROUND Y5: E5 SPARSE RETRIEVAL BENCHMARKS ON TRAINED MODELS")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    part1_res = run_part1_trained_mass_capture(device)
    part2_res = run_part2_trained_tree_retrieval(device)

    out_data = {
        "metadata": {
            "device": str(device),
            "prediction_top16_mass_target": "> 0.99 (missed on random, re-evaluating on trained)",
        },
        "part1_mass_capture": part1_res,
        "part2_tree_retrieval": part2_res,
    }

    os.makedirs("research/yggformer_results", exist_ok=True)
    with open("research/yggformer_results/e5_results.json", "w") as f:
        json.dump(out_data, f, indent=2)
    print("\nResults successfully saved to research/yggformer_results/e5_results.json")


if __name__ == "__main__":
    main()
