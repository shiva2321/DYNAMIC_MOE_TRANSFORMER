"""
Audit Current Brain State, Router Matrices, and Domain Coherence on the Genuine Diverse Engine.
Features:
1. Exact Expert Census per Layer and Domain Attribution.
2. Domain x Expert Routing Heatmaps across all 4 Layers.
3. Learnable Hyperspace Centroid Similarity Matrix in C^2048.
4. Domain-by-Domain Live Generation with Expert Firing Telemetry.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import tiktoken

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.nanogpt import HyperTransformerLM
from hyperspace.vsa import ComplexPhasorVSA

def audit_brain_state():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "experiments/checkpoints/hyperspace_genuine_diverse_engine.pt"
    enc = tiktoken.get_encoding("gpt2")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    state_dict = checkpoint.get("model_state", checkpoint)

    vocab_size = config.get("vocab_size", 50304)
    d_model = config.get("d_model", 384)
    n_layers = config.get("n_layers", 4)
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

    total_experts = sum(per_layer_experts)
    print("\n" + "=" * 95)
    print("  [UNIVERSAL SUBSTRAIT: BRAIN STATE, ROUTER MATRICES & DOMAIN COHERENCE AUDIT]")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Total Experts: {total_experts} Active Experts (Layer 0: {per_layer_experts[0]}, Layer 1: {per_layer_experts[1]}, Layer 2: {per_layer_experts[2]}, Layer 3: {per_layer_experts[3]})")
    print("=" * 95)

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
        spawn_threshold=0.30,
        max_experts=16,
        initial_experts=2,
        use_sparse_attn=True,
        foveal_window=128,
        num_landmarks=4,
        num_sinks=4,
        max_seq_len=576,
        dropout=0.0,
    ).to(device)

    for l_idx, block in enumerate(model.blocks):
        needed = per_layer_experts[l_idx] - block.hyper_moe.num_experts
        for exp_i in range(needed):
            dummy_key = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper), device=device)
            block.hyper_moe._spawn_expert(dummy_key, label=f"restored_L{l_idx}_E{exp_i}")

    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=True)
    model.eval()

    # 1. Measure Domain-Specific Expert Routing Profiles
    cache_dir = "data/genuine_diverse_cache"
    meta_file = os.path.join(cache_dir, "metadata_genuine.json")
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    domains = ["natural_stories", "python_code", "wikitext_facts", "fineweb_reasoning"]
    domain_display = {
        "natural_stories": "Stories (TinyStories)",
        "python_code": "Code (Python)",
        "wikitext_facts": "Knowledge (WikiText)",
        "fineweb_reasoning": "Reasoning (FineWeb)"
    }

    # Record average routing weight per expert for each domain across layers
    routing_profiles = {l: {d: np.zeros(per_layer_experts[l]) for d in domains} for l in range(n_layers)}

    seq_len = 256
    num_eval_samples = 12
    with torch.no_grad():
        for d in domains:
            val_file = metadata["domains"][d]["val_file"]
            data = np.memmap(val_file, dtype=np.uint16, mode='r')
            starts = np.random.randint(0, len(data) - seq_len - 1, size=num_eval_samples)
            batch = torch.from_numpy(np.stack([data[s:s + seq_len] for s in starts]).astype(np.int64)).to(device)
            
            with torch.amp.autocast('cuda'):
                _, _, telemetries = model(batch, allow_spawning=False)
            
            for l in range(n_layers):
                weights = telemetries[l]["top_weights"].cpu().numpy() # [B, S, top_k]
                indices = telemetries[l]["top_indices"].cpu().numpy() # [B, S, top_k]
                
                # Accumulate expert activation mass
                for b_i in range(batch.shape[0]):
                    for s_i in range(batch.shape[1]):
                        for k_i in range(weights.shape[2]):
                            e_idx = indices[b_i, s_i, k_i]
                            w = weights[b_i, s_i, k_i]
                            if e_idx < per_layer_experts[l]:
                                routing_profiles[l][d][e_idx] += w
                
                # Normalize per domain
                total_w = routing_profiles[l][d].sum()
                if total_w > 0:
                    routing_profiles[l][d] /= total_w

    # 2. Extract and Plot Learnable Hyperspace Keys Similarity Matrix
    fig = plt.figure(figsize=(20, 14), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # Plot Layer 1 and Layer 2 Routing Heatmaps
    for idx, l in enumerate([1, 2]):
        ax = fig.add_subplot(gs[0, idx])
        matrix = np.stack([routing_profiles[l][d] for d in domains]) # [4, num_experts]
        
        im = ax.imshow(matrix, aspect='auto', cmap='magma', interpolation='nearest')
        ax.set_xticks(range(per_layer_experts[l]))
        ax.set_xticklabels([f"E{i}" for i in range(per_layer_experts[l])], fontsize=9)
        ax.set_yticks(range(len(domains)))
        ax.set_yticklabels([domain_display[d] for d in domains], fontsize=10, fontweight='bold')
        ax.set_title(f"Layer {l} Domain-to-Expert Firing Matrix ({per_layer_experts[l]} Experts)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Specialized Expert IDs", fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Routing Probability")

    # Plot Centroid Hyperspace Key Cosine Similarity in Layer 2 (Learned C^2048)
    ax_keys = fig.add_subplot(gs[1, 0])
    block = model.blocks[2]
    kr = torch.stack(list(block.hyper_moe.memory.keys_r)).squeeze(1) # [num_exp, 2048]
    ki = torch.stack(list(block.hyper_moe.memory.keys_i)).squeeze(1) # [num_exp, 2048]
    
    # Hermitian Inner Product: Real(K * K^†) / D
    sim_matrix = (torch.matmul(kr, kr.T) + torch.matmul(ki, ki.T)).cpu().detach().numpy() / d_hyper
    np.fill_diagonal(sim_matrix, 1.0)
    
    im_k = ax_keys.imshow(sim_matrix, cmap='coolwarm', vmin=-0.2, vmax=1.0)
    ax_keys.set_title("Layer 2 Learnable Expert Centroid Similarity (C^2048)", fontsize=12, fontweight='bold')
    ax_keys.set_xticks(range(per_layer_experts[2]))
    ax_keys.set_xticklabels([f"E{i}" for i in range(per_layer_experts[2])], fontsize=8)
    ax_keys.set_yticks(range(per_layer_experts[2]))
    ax_keys.set_yticklabels([f"E{i}" for i in range(per_layer_experts[2])], fontsize=8)
    plt.colorbar(im_k, ax=ax_keys, fraction=0.046, pad=0.04, label="Resonance (Hermitian Dot Product)")

    # Plot Multi-Domain Validation Accuracy Scorecard
    ax_score = fig.add_subplot(gs[1, 1])
    history = checkpoint.get("history", {})
    if "domain_top1" in history:
        steps = history["step"]
        for d, col in zip(domains, ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']):
            ax_score.plot(steps, history["domain_top1"][d], label=domain_display[d], color=col, linewidth=2.0)
        ax_score.set_xlabel("Training Steps", fontweight='bold')
        ax_score.set_ylabel("Top-1 Token Prediction Accuracy (%)", fontweight='bold')
        ax_score.set_title("Multi-Domain Token Accuracy Convergence", fontweight='bold')
        ax_score.legend()
        ax_score.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_out = "experiments/plots/current_router_and_domain_specialization_matrix.png"
    plt.savefig(plot_out)
    plt.close()
    print(f"[SAVED] Routing & Brain Matrix plot saved to: {plot_out}")

    import shutil
    brain_dir = r"C:\Users\Asta\.gemini\antigravity-ide\brain\0f00f3a4-e432-4a06-8006-7dfc2fa48c7a"
    shutil.copy(plot_out, os.path.join(brain_dir, "current_router_and_domain_specialization_matrix.png"))

    # 3. Live Multi-Domain Generation & Expert Attribution Audit
    print("\n" + "=" * 95)
    print("  [LIVE MULTI-DOMAIN GENERATION & ROUTING ATTRIBUTION AUDIT]")
    print("=" * 95)

    test_prompts = [
        {
            "domain": "Natural Language Story & Narrative (TinyStories)",
            "prompt": "Lily and her puppy Max were playing in the garden. Suddenly, Max found a little",
            "max_tokens": 50,
            "temp": 0.65
        },
        {
            "domain": "Real Python Source Code",
            "prompt": "def merge_intervals(intervals):\n    \"\"\"Merges overlapping intervals in a list.\"\"\"\n",
            "max_tokens": 50,
            "temp": 0.65
        },
        {
            "domain": "Encyclopedic Knowledge & Facts (WikiText)",
            "prompt": "The solar system consists of the Sun and the objects that orbit it, including",
            "max_tokens": 50,
            "temp": 0.65
        },
        {
            "domain": "Educational Reasoning & Science (FineWeb-Edu)",
            "prompt": "In cell biology, mitochondria are organelles that generate most of the chemical",
            "max_tokens": 50,
            "temp": 0.65
        }
    ]

    for tp in test_prompts:
        p_str = tp["prompt"]
        p_toks = enc.encode(p_str)
        curr = torch.tensor([p_toks], dtype=torch.long, device=device)
        gen = []
        l2_experts_fired = []

        for _ in range(tp["max_tokens"]):
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    l_out, _, telem = model(curr, allow_spawning=False)
            
            top_l2 = telem[2]["top_indices"][0, -1].cpu().numpy().tolist()
            l2_experts_fired.extend(top_l2)

            scaled_logits = l_out[0, -1, :vocab_size] / tp["temp"]
            v, top_idx = torch.topk(scaled_logits, 40)
            probs = F.softmax(v, dim=-1)
            nxt = top_idx[torch.multinomial(probs, 1)].item()
            gen.append(nxt)
            curr = torch.cat([curr, torch.tensor([[nxt]], device=device)], dim=1)

        unique_l2, counts_l2 = np.unique(l2_experts_fired, return_counts=True)
        top_routed_exp = [f"E{e} ({c}x)" for e, c in sorted(zip(unique_l2, counts_l2), key=lambda x: x[1], reverse=True)[:4]]

        print("\n" + "#" * 80)
        print(f"  DOMAIN: {tp['domain']}")
        print(f"  PROMPT: {p_str.strip()}")
        print(f"  ROUTER FIRING PROFILE (Layer 2 Top Experts): {', '.join(top_routed_exp)}")
        print("#" * 80)
        print(f"[MODEL OUTPUT]:\n{p_str}{enc.decode(gen)}\n")

if __name__ == "__main__":
    audit_brain_state()
