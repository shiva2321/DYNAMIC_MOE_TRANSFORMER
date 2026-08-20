import os
import sys
import json
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.nanogpt import HyperTransformerLM
from exp_10domain_continual_learning import Streamer10Domain

def analyze_10domain_checkpoint(
    ckpt_path: str,
    output_prefix: str = "10domain_analysis",
    num_val_samples: int = 128
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 105)
    print(f"  [10-DOMAIN FROZEN POST-TRAINING DIAGNOSTIC PROBE & SPECIALIZATION HEATMAP]")
    print(f"  Checkpoint: {ckpt_path}")
    print("=" * 105)

    streamer = Streamer10Domain(seq_len=256, batch_size=2)
    domains = streamer.domains
    domain_titles = [streamer.metadata["domains"][d]["title"] for d in domains]
    num_domains = len(domains)

    # Inspect checkpoint topology
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]

    layer_expert_counts = []
    for l in range(4):
        keys_r = [k for k in state_dict.keys() if f"blocks.{l}.hyper_moe.memory.keys_r" in k]
        layer_expert_counts.append(len(keys_r) if len(keys_r) > 0 else 2)

    print(f"  Discovered Layer Expert Counts: {layer_expert_counts}")

    model = HyperTransformerLM(
        vocab_size=50304,
        d_model=384,
        n_layers=4,
        n_heads=6,
        d_ff=768,
        d_hyper=2048,
        top_k=2,
        max_k=4,
        top_p=0.85,
        dynamic_k=True,
        spawn_threshold=1.0,
        max_experts=32,
        initial_experts=2,
        use_sparse_attn=True,
        foveal_window=128,
        num_landmarks=4,
        num_sinks=4,
        max_seq_len=576,
        dropout=0.0,
        use_bus=True,
        use_context_binding=True
    ).to(device)

    # Spawn slots to match discovered state dict
    for l_idx, count in enumerate(layer_expert_counts):
        while model.blocks[l_idx].hyper_moe.num_experts < count:
            dummy_seed = torch.randn(1, 2048, dtype=torch.cfloat, device=device)
            model.blocks[l_idx].hyper_moe._spawn_expert(dummy_seed)

    model.load_state_dict(state_dict)
    model.eval()

    # Collect Routing Tensors across all 10 domains
    max_exp = max(layer_expert_counts)
    routing_mass = np.zeros((4, num_domains, max_exp)) # [L, D, E]

    eval_micro_batch = 4
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for d_idx, domain in enumerate(domains):
            vx, vy = streamer.get_val_batch(domain, num_samples=num_val_samples)
            accum_counts = [torch.zeros(layer_expert_counts[l], device=device) for l in range(4)]
            
            for i in range(0, vx.shape[0], eval_micro_batch):
                bx = vx[i : i + eval_micro_batch].to(device)
                _, _, telems = model(bx, allow_spawning=False)
                for l_idx, telem in enumerate(telems):
                    top_idx = telem['top_indices'].flatten()
                    counts = torch.bincount(top_idx, minlength=layer_expert_counts[l_idx]).float()
                    accum_counts[l_idx] += counts

            for l_idx in range(4):
                total = accum_counts[l_idx].sum().item()
                if total > 0:
                    probs = (accum_counts[l_idx] / total).cpu().numpy()
                    routing_mass[l_idx, d_idx, :len(probs)] = probs

    # Gini Specialization Index Calculation
    def calc_gini(arr):
        sorted_arr = np.sort(arr)
        n = len(arr)
        if np.sum(sorted_arr) == 0: return 0.0
        index = np.arange(1, n + 1)
        return float((np.sum((2 * index - n - 1) * sorted_arr)) / (n * np.sum(sorted_arr)))

    print("\n--- Per-Domain Gini Specialization Index (Layer 0 to Layer 3) ---")
    for d_idx, domain in enumerate(domains):
        ginis = [calc_gini(routing_mass[l, d_idx, :layer_expert_counts[l]]) for l in range(4)]
        gini_str = " | ".join([f"L{l}:{ginis[l]:.3f}" for l in range(4)])
        print(f"  {domain:<22} -> {gini_str} (Mean Gini: {np.mean(ginis):.3f})")

    # Residual Cosine Similarity (Backbone-stripped)
    print("\n" + "=" * 90)
    print("  [RESIDUAL AUXILIARY CROSS-DOMAIN COSINE SIMILARITIES (LAYER 0)]")
    print("=" * 90)
    l0_mass = routing_mass[0, :, :layer_expert_counts[0]] # [10, E]
    top_shared = int(np.argmax(np.mean(l0_mass, axis=0)))
    res_l0 = l0_mass.copy()
    res_l0[:, top_shared] = 0.0
    res_norms = np.linalg.norm(res_l0, axis=-1, keepdims=True) + 1e-8
    res_l0_norm = res_l0 / res_norms
    res_cossim = res_l0_norm @ res_l0_norm.T

    print(f"  Top Shared Generalist Expert in Layer 0: Expert #{top_shared} (Stripped for residual analysis)")
    print("\n  1. Fresh Data Domains (Zero Repetition, >1.1M tokens):")
    fresh_domains = ["github_code", "python_code", "freelaw_legal", "financial_market", "gutenberg_literature", "wikitext_facts"]
    for i in range(len(fresh_domains)):
        for j in range(i + 1, len(fresh_domains)):
            d1, d2 = fresh_domains[i], fresh_domains[j]
            idx1, idx2 = domains.index(d1), domains.index(d2)
            print(f"    • Sim({d1:<20}, {d2:<20}) = {res_cossim[idx1, idx2]:.4f}")

    print("\n  2. Systems Code vs Prose Manifolds:")
    gh_idx = domains.index("github_code")
    for d_name in domains:
        if d_name != "github_code":
            d_i = domains.index(d_name)
            print(f"    • Sim(github_code, {d_name:<20}) = {res_cossim[gh_idx, d_i]:.4f}")

    print("\n  3. Repetition-Affected Group (Confounded by <1.1M tokens / multi-pass):")
    rep_domains = ["fineweb_edu", "openweb_math", "pubmed_biomedical", "arxiv_physics"]
    for i in range(len(rep_domains)):
        for j in range(i + 1, len(rep_domains)):
            d1, d2 = rep_domains[i], rep_domains[j]
            idx1, idx2 = domains.index(d1), domains.index(d2)
            print(f"    • Sim({d1:<20}, {d2:<20}) = {res_cossim[idx1, idx2]:.4f} [REPETITION-CONFOUNDED]")

    # Save full numerical matrix
    cossim_data = {
        "domains": domains,
        "top_shared_expert_l0": top_shared,
        "residual_cossim_matrix": res_cossim.tolist(),
        "routing_mass_per_layer": routing_mass.tolist()
    }
    cossim_out = os.path.join(PROJECT_ROOT, "experiments", f"{output_prefix}_residual_cossim.json")
    with open(cossim_out, "w", encoding="utf-8") as f:
        json.dump(cossim_data, f, indent=2)
    print(f"\n[SAVED] Full 10x10 Residual Cosine Similarity Matrix saved to: {cossim_out}")

    # Plot 10-Domain Routing Heatmap
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for l in range(4):
        ax = axes[l]
        data = routing_mass[l, :, :layer_expert_counts[l]]
        sns.heatmap(
            data,
            ax=ax,
            cmap="mako",
            annot=True,
            fmt=".2f",
            cbar=True,
            yticklabels=[d[:14] for d in domains],
            xticklabels=[f"E{e}" for e in range(layer_expert_counts[l])]
        )
        ax.set_title(f"Layer {l} Routing Mass Distribution (Gini = {np.mean([calc_gini(data[d]) for d in range(num_domains)]):.3f})")
        ax.set_ylabel("Domain")
        ax.set_xlabel("Expert Index")

    plt.tight_layout()
    heatmap_out = os.path.join(PROJECT_ROOT, "experiments", f"{output_prefix}_specialization_heatmap.png")
    plt.savefig(heatmap_out, dpi=300)
    plt.close()
    print(f"\n[SAVED] 10-Domain Specialization Heatmap saved to: {heatmap_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="10domain_analysis")
    args = parser.parse_args()

    analyze_10domain_checkpoint(args.ckpt, output_prefix=args.out)
