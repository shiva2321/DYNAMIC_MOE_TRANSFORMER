import os
import sys
import math
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.nanogpt import HyperTransformerLM
from train_scaled_production_engine import ScaledProductionDataStreamer

def load_dynamic_hyper_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state"]
    
    # 1. Determine how many experts each layer actually has in the saved state dict
    layer_experts = {}
    for k in sd.keys():
        if "hyper_moe.experts." in k:
            parts = k.split(".")
            # format: blocks.<layer_idx>.hyper_moe.experts.<exp_idx>...
            layer_idx = int(parts[1])
            exp_idx = int(parts[4])
            layer_experts[layer_idx] = max(layer_experts.get(layer_idx, 0), exp_idx + 1)
            
    print(f"[CHECKPOINT LOAD] Discovered layer expert topology: {layer_experts}")

    # 2. Instantiate base model (initial_experts=2)
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
        spawn_threshold=1.0, # Disable spawn during eval
        max_experts=16,
        initial_experts=2,
        use_sparse_attn=True,
        foveal_window=128,
        num_landmarks=4,
        num_sinks=4,
        max_seq_len=576,
        dropout=0.0,
        use_bus=True
    ).to(device)

    # 3. Expand each layer's experts to match state dict
    for layer_idx, num_exp in layer_experts.items():
        moe = model.blocks[layer_idx].hyper_moe
        current_exp = len(moe.experts)
        for i in range(current_exp, num_exp):
            dummy_key = torch.randn(1, moe.d_hyper, dtype=torch.cfloat, device=device)
            dummy_key = dummy_key / (torch.abs(dummy_key) + 1e-8)
            moe._spawn_expert(dummy_key, label=f"restored_{i}")

    # 4. Load weights
    model.load_state_dict(sd)
    model.eval()
    print("[CHECKPOINT LOAD] Successfully loaded state dict into frozen evaluation model!")
    return model, layer_experts

def compute_gini(array):
    """Compute Gini coefficient of an activation distribution."""
    array = np.array(array, dtype=np.float64)
    if np.sum(array) == 0:
        return 0.0
    array = np.sort(array)
    n = len(array)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

def probe_frozen_inference(ckpt_path="experiments/checkpoint_hyperspace_budgeted_spawn.pt", seed_name="Seed 1337"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 115)
    print(f"  [FROZEN INFERENCE ROUTING PROBE: {seed_name.upper()}]")
    print(f"  Target Checkpoint: {ckpt_path}")
    print("=" * 115)

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint {ckpt_path} not found!")
        return None

    model, layer_experts = load_dynamic_hyper_model(ckpt_path, device)
    
    streamer = ScaledProductionDataStreamer(cache_dir="data/scaled_real_corpus", seq_len=256, batch_size=4)
    domains = ["fineweb_edu", "python_code", "wikitext_facts", "natural_stories"]
    domain_titles = {
        "fineweb_edu": "Web / Reasoning",
        "python_code": "Python Code",
        "wikitext_facts": "Wiki / Facts",
        "natural_stories": "Tiny Stories"
    }

    # Tracking metrics
    routing_mass = {l: {d: np.zeros(layer_experts[l], dtype=np.float64) for d in domains} for l in range(4)}
    domain_token_counts = {d: 0 for d in domains}
    val_losses = {d: [] for d in domains}
    val_accs = {d: [] for d in domains}

    num_eval_batches = 50 # 50 batches * 4 seqs = 200 sequences per domain (51,200 tokens/domain)

    with torch.no_grad():
        for domain in domains:
            for _ in range(num_eval_batches):
                x, y = streamer.get_domain_val_batch(domain, num_samples=4)
                x = x.to(device)
                y = y.to(device)
                b, s = x.shape
                domain_token_counts[domain] += (b * s)
                
                # Forward with frozen model (no spawning, eval mode)
                logits, loss, layer_telemetries = model(x, targets=y, allow_spawning=False)
                
                # Compute loss & top-1 accuracy
                val_losses[domain].append(loss.item())
                preds = logits.argmax(dim=-1)
                acc = (preds == y).float().mean().item()
                val_accs[domain].append(acc)

                # Collect per-layer routing mass
                for l in range(4):
                    telem = layer_telemetries[l]
                    top_indices = telem["top_indices"].view(-1, telem["top_indices"].shape[-1]) # [N, k]
                    top_weights = telem["top_weights"].view(-1, telem["top_weights"].shape[-1]) # [N, k]
                    num_exp = layer_experts[l]
                    
                    for k_idx in range(top_indices.shape[-1]):
                        idx_k = top_indices[:, k_idx]
                        w_k = top_weights[:, k_idx]
                        for exp_id in range(num_exp):
                            mask = (idx_k == exp_id)
                            if mask.any():
                                routing_mass[l][domain][exp_id] += w_k[mask].sum().item()

    # Aggregate results
    print("\n" + "-" * 115)
    print(f"  [HELD-OUT INFERENCE VALIDATION PERFORMANCE ({seed_name.upper()})]")
    print("-" * 115)
    summary_eval = {}
    for domain in domains:
        mean_loss = np.mean(val_losses[domain])
        mean_acc = np.mean(val_accs[domain]) * 100.0
        summary_eval[domain] = {"loss": float(mean_loss), "acc": float(mean_acc)}
        print(f"  • {domain_titles[domain]:<20} | Held-out Loss: {mean_loss:.4f} nats | Top-1 Acc: {mean_acc:.2f}%")

    print("\n" + "=" * 115)
    print(f"  [FROZEN INFERENCE DOMAIN-TO-EXPERT ROUTING MATRIX ({seed_name.upper()})]")
    print("=" * 115)

    routing_matrix = {l: {} for l in range(4)}
    gini_scores = {l: {} for l in range(4)}

    for l in range(4):
        num_exp = layer_experts[l]
        print(f"\n--- LAYER {l} (Total Spawned Experts: {num_exp}) ---")
        header = f"{'Domain':<20} | " + " | ".join([f"E{e:<2}" for e in range(num_exp)]) + " | Top Specialist | Gini"
        print(header)
        print("-" * len(header))

        for domain in domains:
            raw_mass = routing_mass[l][domain]
            pct_dist = (raw_mass / (raw_mass.sum() + 1e-8)) * 100.0
            
            top_exp = int(np.argmax(pct_dist))
            gini = compute_gini(pct_dist)
            
            routing_matrix[l][domain] = pct_dist.tolist()
            gini_scores[l][domain] = float(gini)

            dist_str = " | ".join([f"{pct_dist[e]:4.1f}%" for e in range(num_exp)])
            print(f"{domain_titles[domain]:<20} | {dist_str} | E{top_exp:<2} ({pct_dist[top_exp]:.1f}%) | {gini:.3f}")

    # Compute Cross-Domain Routing Cosine Similarity per layer
    print("\n" + "-" * 115)
    print("  [CROSS-DOMAIN ROUTING OVERLAP (COSINE SIMILARITY: 1.0 = IDENTICAL ROUTING, 0.0 = DISJOINT SPECIALIZATION)]")
    print("-" * 115)
    overlap_metrics = {}
    for l in range(4):
        p_vec = np.array(routing_matrix[l]["python_code"])
        w_vec = np.array(routing_matrix[l]["wikitext_facts"])
        s_vec = np.array(routing_matrix[l]["natural_stories"])
        f_vec = np.array(routing_matrix[l]["fineweb_edu"])

        def cos_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        sim_code_wiki = cos_sim(p_vec, w_vec)
        sim_code_stories = cos_sim(p_vec, s_vec)
        sim_wiki_stories = cos_sim(w_vec, s_vec)
        sim_web_code = cos_sim(f_vec, p_vec)

        overlap_metrics[f"layer_{l}"] = {
            "code_vs_wiki": sim_code_wiki,
            "code_vs_stories": sim_code_stories,
            "wiki_vs_stories": sim_wiki_stories,
            "web_vs_code": sim_web_code
        }
        print(f"  Layer {l}: Python vs Wiki = {sim_code_wiki:.4f} | Python vs Stories = {sim_code_stories:.4f} | Python vs Web = {sim_web_code:.4f}")

    return {
        "checkpoint": ckpt_path,
        "seed_name": seed_name,
        "eval_metrics": summary_eval,
        "routing_matrix": routing_matrix,
        "gini_scores": gini_scores,
        "overlap_metrics": overlap_metrics,
        "layer_experts": layer_experts
    }

if __name__ == "__main__":
    res_1337 = probe_frozen_inference("experiments/checkpoint_hyperspace_budgeted_spawn.pt", seed_name="Seed 1337")
    res_42 = probe_frozen_inference("experiments/checkpoint_hyperspace_budgeted_spawn_seed_42.pt", seed_name="Seed 42")
    
    with open("experiments/frozen_inference_routing_probe_results.json", "w") as f:
        json.dump({"seed_1337": res_1337, "seed_42": res_42}, f, indent=2)
    print("\n[SAVED] Frozen inference probe results saved to: experiments/frozen_inference_routing_probe_results.json")
