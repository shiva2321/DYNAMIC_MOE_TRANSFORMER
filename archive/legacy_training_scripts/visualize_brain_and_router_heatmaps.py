"""
Brain Topology & Dynamic Router Heatmap Visualizer.
Computes and plots:
1. The Router Heatmap: Real-World Domains (8) x Micro-Experts (157 across 6 Layers) Routing Matrix.
2. The Brain Heatmap: Inter-Expert Semantic Attractor Similarity Matrix in Complex Phasor Space C^2048.
3. Layer-Wise Hierarchical Router Heatmaps for each of the 6 Transformer Layers.
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

def generate_brain_and_router_heatmaps(
    ckpt_path: str = "experiments/checkpoints/hyperspace_deep_trained_25m.pt",
    meta_file: str = "data/real_blend_cache/metadata_real_blend.json",
    out_dir: str = "experiments"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 95)
    print("  [GENERATING BRAIN TOPOLOGY & DYNAMIC ROUTER HEATMAPS]")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 95)

    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    # 1. Load Model
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

    # 2. Compute the Brain's Internal Attractor Topology Heatmap (Inter-Expert Resonance Matrix)
    # Aggregate all 157 expert keys across all layers
    all_keys = []
    expert_labels = []
    layer_slices = []
    current_idx = 0

    for l_idx, block in enumerate(model.blocks):
        mem = block.hyper_moe.memory
        keys = mem.expert_keys # [Num_Exp, D] in C^D
        all_keys.append(keys)
        n_exp = keys.shape[0]
        layer_slices.append((current_idx, current_idx + n_exp, f"Layer {l_idx}"))
        for e in range(n_exp):
            expert_labels.append(f"L{l_idx}-E{e}")
        current_idx += n_exp

    all_keys_tensor = torch.cat(all_keys, dim=0) # [157, 2048]
    # Compute full pairwise Hermitian dot product: Real(K . K^H)
    r = all_keys_tensor.real
    i = all_keys_tensor.imag
    # [157, 157]
    sim_matrix = (torch.matmul(r, r.T) + torch.matmul(i, i.T)).cpu().numpy() / d_hyper

    # 3. Compute the Router Heatmap across 8 Real-World Domains
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    domains = list(metadata["domains"].keys())
    domain_display_names = [d.replace("_", " ").title() for d in domains]
    val_shards = {d: np.memmap(metadata["domains"][d]["val_file"], dtype=np.uint16, mode='r') for d in domains}

    # [8, 157] Global Router Routing Matrix
    global_router_matrix = np.zeros((len(domains), total_ckpt_experts), dtype=np.float32)
    # [6, 8, max_exp] Per-Layer Router Matrices
    layer_router_matrices = [np.zeros((len(domains), per_layer_experts[l]), dtype=np.float32) for l in range(n_layers)]

    with torch.no_grad():
        for d_idx, domain in enumerate(domains):
            data = val_shards[domain]
            starts = np.random.randint(0, len(data) - 512 - 1, size=8)
            batch = torch.from_numpy(np.stack([data[s:s + 512] for s in starts]).astype(np.int64)).to(device)
            
            x = model.token_embeddings(batch)
            global_exp_offset = 0

            for l_idx, block in enumerate(model.blocks):
                moe = block.hyper_moe
                normed_x = block.norm2(x)
                
                # Complex projection to C^D
                norm_flat = F.normalize(normed_x.view(-1, d_model), p=2, dim=-1)
                r_proj = moe.proj_r(norm_flat).float()
                i_proj = moe.proj_i(norm_flat).float()
                z = torch.complex(r_proj, i_proj)
                query_phasors = z / (torch.abs(z) + 1e-8)
                
                # Hermitian resonance against expert keys
                res = moe.memory.compute_resonance(query_phasors) # [Tokens, N]
                probs = F.softmax(res * 10.0, dim=-1).mean(dim=0).cpu().numpy() # [N]
                
                # Store in layer matrix
                layer_router_matrices[l_idx][d_idx, :len(probs)] = probs
                # Store in global matrix
                global_router_matrix[d_idx, global_exp_offset:global_exp_offset + len(probs)] = probs
                
                global_exp_offset += len(probs)
                x, _ = block(x, allow_spawning=False)

    # Normalize Router Matrix per domain
    global_router_matrix = global_router_matrix / (global_router_matrix.sum(axis=1, keepdims=True) + 1e-8)

    # 4. Plot Visualizations
    # FIGURE 1: The Brain Topology Heatmap (157x157 Attractor Similarity Matrix)
    plt.figure(figsize=(14, 12), dpi=300)
    sns.heatmap(
        sim_matrix,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Hermitian Resonance Similarity in $\\mathbb{C}^{2048}$'}
    )
    # Draw layer boundary lines
    for start, end, label in layer_slices:
        plt.axhline(start, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.axvline(start, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
    
    plt.title("System Brain Topology Heatmap\nPairwise Semantic Attractor Similarity Matrix across All 157 Micro-Experts", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Micro-Expert Index (Grouped by Layer 0 -> Layer 5)", fontsize=11, fontweight='bold')
    plt.ylabel("Micro-Expert Index (Grouped by Layer 0 -> Layer 5)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    brain_fig_path = os.path.join(out_dir, "plots", "system_brain_topology_heatmap.png")
    plt.savefig(brain_fig_path)
    plt.close()
    print(f"[SAVED] Brain Topology Heatmap saved to: {brain_fig_path}")

    # FIGURE 2: The Router Heatmap (8 Domains x 157 Micro-Experts)
    plt.figure(figsize=(18, 8), dpi=300)
    sns.heatmap(
        global_router_matrix,
        cmap="viridis",
        yticklabels=domain_display_names,
        xticklabels=[f"E{i}" if i % 5 == 0 else "" for i in range(total_ckpt_experts)],
        cbar_kws={'label': 'Empirical Routing Probability Share $R_{d, e}$'}
    )
    for start, end, label in layer_slices:
        plt.axvline(start, color='white', linestyle='-', linewidth=1.5, alpha=0.9)
        plt.text(start + (end - start) / 2.0, -0.5, label, ha='center', va='bottom', color='black', fontweight='bold', fontsize=9)

    plt.title("System Router Heatmap\nEmpirical Routing Activation Matrix: 8 Real-World Domains -> 157 Micro-Experts", fontsize=14, fontweight='bold', pad=25)
    plt.xlabel("Micro-Expert Index across Transformer Layers (0 -> 5)", fontsize=11, fontweight='bold')
    plt.ylabel("Real-World Knowledge Domains", fontsize=11, fontweight='bold')
    plt.tight_layout()
    router_fig_path = os.path.join(out_dir, "plots", "system_router_activation_heatmap.png")
    plt.savefig(router_fig_path)
    plt.close()
    print(f"[SAVED] Router Heatmap saved to: {router_fig_path}")

    # FIGURE 3: Layer-by-Layer Router Sub-Heatmaps (6 Subplots)
    fig, axes = plt.subplots(3, 2, figsize=(18, 14), dpi=300)
    axes = axes.flatten()

    for l_idx in range(n_layers):
        ax = axes[l_idx]
        mat = layer_router_matrices[l_idx]
        # Normalize per domain for visual clarity
        mat_norm = mat / (mat.sum(axis=1, keepdims=True) + 1e-8)
        sns.heatmap(
            mat_norm,
            ax=ax,
            cmap="mako",
            yticklabels=domain_display_names if l_idx % 2 == 0 else False,
            xticklabels=[f"E{e}" for e in range(per_layer_experts[l_idx])],
            cbar_kws={'label': 'Routing Share'} if l_idx % 2 == 1 else None
        )
        ax.set_title(f"Layer {l_idx} Router Heatmap ({per_layer_experts[l_idx]} Experts)", fontweight='bold')
        ax.set_xlabel("Expert Slot Index", fontsize=9)
        if l_idx % 2 == 0:
            ax.set_ylabel("Domain", fontsize=9)

    plt.suptitle("Hierarchical Router Heatmaps Across Transformer Depth (Layers 0 to 5)", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    layer_router_fig_path = os.path.join(out_dir, "plots", "layerwise_router_heatmaps.png")
    plt.savefig(layer_router_fig_path)
    plt.close()
    print(f"[SAVED] Layer-wise Router Heatmaps saved to: {layer_router_fig_path}")

    # Copy to Brain Artifacts directory
    brain_dir = r"C:\Users\Asta\.gemini\antigravity-ide\brain\0f00f3a4-e432-4a06-8006-7dfc2fa48c7a"
    import shutil
    shutil.copy(brain_fig_path, os.path.join(brain_dir, "system_brain_topology_heatmap.png"))
    shutil.copy(router_fig_path, os.path.join(brain_dir, "system_router_activation_heatmap.png"))
    shutil.copy(layer_router_fig_path, os.path.join(brain_dir, "layerwise_router_heatmaps.png"))
    print("[COPIED] All 3 heatmaps successfully copied to brain artifact directory!")

if __name__ == "__main__":
    generate_brain_and_router_heatmaps()
