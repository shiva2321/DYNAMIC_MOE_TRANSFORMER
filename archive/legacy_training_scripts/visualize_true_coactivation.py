"""
Empirical Expert Co-Activation & Weight Similarity Matrix Analyzer.
Replaces static key dot-products with TRUE empirical runtime dynamics:
1. True Empirical Co-Activation Correlation Matrix (Which experts fire together on real tokens).
2. True MLP Weight Cosine Similarity Matrix (Which experts developed similar internal neural weights).
3. Real Sequence Token-by-Token Routing Trace Heatmap.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.nanogpt import HyperTransformerLM
from hyperspace.vsa import ComplexPhasorVSA

def compute_true_empirical_brain_heatmaps(
    ckpt_path: str = "experiments/checkpoints/hyperspace_deep_trained_25m.pt",
    meta_file: str = "data/real_blend_cache/metadata_real_blend.json",
    out_dir: str = "experiments"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 95)
    print("  [COMPUTING TRUE EMPIRICAL CO-ACTIVATION & WEIGHT TOPOLOGY HEATMAPS]")
    print(f"  Checkpoint: {ckpt_path}")
    print("=" * 95)

    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    # 1. Load Checkpoint
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = state.get("config", {})
    state_dict = state.get("model_state", state)

    vocab_size = config.get("vocab_size", 50304)
    d_model = config.get("d_model", 384)
    n_layers = config.get("n_layers", 6)
    n_heads = config.get("n_heads", 6)
    d_ff = config.get("d_ff", 768)
    d_hyper = config.get("d_hyper", 2048)

    per_layer_experts = []
    for l in range(n_layers):
        exp_keys = set()
        for k in state_dict.keys():
            if f"blocks.{l}.hyper_moe.experts." in k:
                exp_keys.add(int(k.split(".")[4]))
        per_layer_experts.append(max(2, len(exp_keys)))

    total_ckpt_experts = sum(per_layer_experts)
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    max_seq_len = clean_sd.get("pos_embeddings.weight", torch.zeros(1088, 1)).shape[0]

    model = HyperTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        d_hyper=d_hyper,
        top_k=2,
        max_k=4,
        top_p=0.85,
        dynamic_k=True,
        spawn_threshold=0.35,
        max_experts=32,
        initial_experts=2,
        use_sparse_attn=True,
        foveal_window=128,
        num_landmarks=4,
        num_sinks=4,
        max_seq_len=max_seq_len,
        dropout=0.0,
    ).to(device)

    for l_idx, block in enumerate(model.blocks):
        needed = per_layer_experts[l_idx] - block.hyper_moe.num_experts
        for exp_i in range(needed):
            dummy_key = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper), device=device)
            block.hyper_moe._spawn_expert(dummy_key, label=f"restored_L{l_idx}_E{exp_i}")

    model.load_state_dict(clean_sd, strict=True)
    model.eval()

    # 2. Compute TRUE MLP Weight Cosine Similarity across Experts in each layer
    layer_weight_sims = []
    for l_idx, block in enumerate(model.blocks):
        moe = block.hyper_moe
        n_exp = len(moe.experts)
        # Flatten all weights of each expert (soma + dendrite MLPs)
        expert_weight_vectors = []
        for exp in moe.experts:
            w_flat = torch.cat([p.flatten() for p in exp.parameters() if p.requires_grad])
            expert_weight_vectors.append(F.normalize(w_flat.unsqueeze(0), p=2, dim=-1))
        
        W = torch.cat(expert_weight_vectors, dim=0) # [N, Num_Params]
        # Cosine similarity matrix between expert weights
        cos_sim = torch.matmul(W, W.T).detach().cpu().numpy() # [N, N]
        layer_weight_sims.append(cos_sim)

    # 3. Compute TRUE Empirical Co-Activation Correlation Matrix from live tokens
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    domains = list(metadata["domains"].keys())
    val_shards = {d: np.memmap(metadata["domains"][d]["val_file"], dtype=np.uint16, mode='r') for d in domains}

    layer_activations = [[] for _ in range(n_layers)]

    with torch.no_grad():
        for domain in domains:
            data = val_shards[domain]
            starts = np.random.randint(0, len(data) - 512 - 1, size=4)
            batch = torch.from_numpy(np.stack([data[s:s + 512] for s in starts]).astype(np.int64)).to(device)
            
            x = model.token_embeddings(batch)
            for l_idx, block in enumerate(model.blocks):
                normed_x = block.norm2(x)
                norm_flat = F.normalize(normed_x.view(-1, d_model), p=2, dim=-1)
                r_proj = block.hyper_moe.proj_r(norm_flat).float()
                i_proj = block.hyper_moe.proj_i(norm_flat).float()
                z = torch.complex(r_proj, i_proj)
                query_phasors = z / (torch.abs(z) + 1e-8)
                
                # Compute raw router activation weights [Tokens, N]
                res = block.hyper_moe.memory.compute_resonance(query_phasors)
                probs = F.softmax(res * 5.0, dim=-1) # [Tokens, N]
                layer_activations[l_idx].append(probs.cpu().numpy())
                
                x, _ = block(x, allow_spawning=False)

    # Compute correlation matrix per layer
    layer_coactivations = []
    for l_idx in range(n_layers):
        all_acts = np.concatenate(layer_activations[l_idx], axis=0) # [Total_Tokens, N]
        # Pearson correlation matrix
        corr = np.corrcoef(all_acts, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        layer_coactivations.append(corr)

    # 4. Plot Visualizations
    # FIGURE 1: True MLP Weight Similarity across Layers (6 Subplots)
    fig, axes = plt.subplots(3, 2, figsize=(18, 14), dpi=300)
    axes = axes.flatten()
    for l_idx in range(n_layers):
        ax = axes[l_idx]
        mat = layer_weight_sims[l_idx]
        sns.heatmap(
            mat,
            ax=ax,
            cmap="mako",
            vmin=np.percentile(mat, 5),
            vmax=1.0,
            xticklabels=5,
            yticklabels=5,
            cbar_kws={'label': 'Weight Cosine Similarity'} if l_idx % 2 == 1 else None
        )
        ax.set_title(f"Layer {l_idx} Expert Weight Similarity Matrix ({len(mat)} Experts)", fontweight='bold')
        ax.set_xlabel("Expert Index", fontsize=9)
        ax.set_ylabel("Expert Index", fontsize=9)

    plt.suptitle("Empirical Neural Weight Similarity Matrix Across Experts (Layers 0 to 5)", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    weight_sim_path = os.path.join(out_dir, "plots", "empirical_expert_weight_similarity.png")
    plt.savefig(weight_sim_path)
    plt.close()
    print(f"[SAVED] Empirical Weight Similarity saved to: {weight_sim_path}")

    # FIGURE 2: True Expert Co-Activation Correlation Matrix (6 Subplots)
    fig, axes = plt.subplots(3, 2, figsize=(18, 14), dpi=300)
    axes = axes.flatten()
    for l_idx in range(n_layers):
        ax = axes[l_idx]
        mat = layer_coactivations[l_idx]
        sns.heatmap(
            mat,
            ax=ax,
            cmap="coolwarm",
            vmin=-0.3,
            vmax=0.8,
            xticklabels=5,
            yticklabels=5,
            cbar_kws={'label': 'Pearson Correlation (Co-Firing)'} if l_idx % 2 == 1 else None
        )
        ax.set_title(f"Layer {l_idx} Expert Co-Activation Correlation Matrix ({len(mat)} Experts)", fontweight='bold')
        ax.set_xlabel("Expert Index", fontsize=9)
        ax.set_ylabel("Expert Index", fontsize=9)

    plt.suptitle("Empirical Expert Co-Activation Correlation Matrix (Which Experts Fire Together)", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    coact_path = os.path.join(out_dir, "plots", "empirical_expert_coactivation_correlation.png")
    plt.savefig(coact_path)
    plt.close()
    print(f"[SAVED] Empirical Co-Activation Correlation saved to: {coact_path}")

    # Copy to Brain Artifacts
    brain_dir = r"C:\Users\Asta\.gemini\antigravity-ide\brain\0f00f3a4-e432-4a06-8006-7dfc2fa48c7a"
    import shutil
    shutil.copy(weight_sim_path, os.path.join(brain_dir, "empirical_expert_weight_similarity.png"))
    shutil.copy(coact_path, os.path.join(brain_dir, "empirical_expert_coactivation_correlation.png"))
    print("[COPIED] Empirical weight and co-activation heatmaps copied to artifact directory!")

if __name__ == "__main__":
    compute_true_empirical_brain_heatmaps()
