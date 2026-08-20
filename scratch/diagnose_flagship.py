import os
import sys
import json
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.nanogpt import HyperTransformerLM
from train_scaled_production_engine import ScaledProductionDataStreamer

def diagnose_checkpoint(ckpt_path="experiments/checkpoint_hyperspace.pt", out_json="experiments/flagship_diagnostics_results.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at: {ckpt_path}")
        return

    print("=" * 100)
    print(f"  [DIAGNOSTIC: ROUTING BREAKDOWN FOR {ckpt_path}]")
    print("=" * 100)

    ckpt = torch.load(ckpt_path, map_location=device)
    model_type = ckpt.get("model_type", "")
    use_context_binding = "context_bound" in model_type
    
    # Check max_experts capacity from buffer shape in checkpoint
    if "blocks.0.hyper_moe.expert_usage_counts" in ckpt["model_state"]:
        max_experts = ckpt["model_state"]["blocks.0.hyper_moe.expert_usage_counts"].shape[0]
    else:
        max_experts = 16

    # Reconstruct exact expert capacity per layer from checkpoint state dict
    layer_expert_counts = [0] * 4
    for k in ckpt["model_state"].keys():
        if "hyper_moe.memory.keys_r" in k:
            parts = k.split(".")
            l_idx = int(parts[1])
            exp_idx = int(parts[-1])
            layer_expert_counts[l_idx] = max(layer_expert_counts[l_idx], exp_idx + 1)
            
    print(f"  [RECONSTRUCTING TOPOLOGY] Model Type: {model_type} | Max Capacity: {max_experts} | Active Experts per layer: {layer_expert_counts}")

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
        spawn_threshold=0.30,
        max_experts=max_experts,
        initial_experts=min(2, min(layer_expert_counts)),
        use_sparse_attn=True,
        foveal_window=128,
        num_landmarks=4,
        num_sinks=4,
        max_seq_len=576,
        dropout=0.0,
        use_bus=True,
        use_context_binding=use_context_binding,
        ortho_loss_weight=0.005,
        load_bal_weight=0.01
    ).to(device)

    for l_idx, count in enumerate(layer_expert_counts):
        while model.blocks[l_idx].hyper_moe.num_experts < count:
            dummy_seed = torch.randn(1, 2048, dtype=torch.cfloat).to(device)
            model.blocks[l_idx].hyper_moe._spawn_expert(dummy_seed)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    streamer = ScaledProductionDataStreamer(cache_dir="data/scaled_real_corpus", seq_len=256, batch_size=8)
    domains = ["fineweb_edu", "python_code", "wikitext_facts", "natural_stories"]

    results = {}
    with torch.no_grad():
        for l_idx in range(4):
            print(f"\n--- LAYER {l_idx} ---")
            weighted_usage = {d: np.zeros(16) for d in domains}
            for d in domains:
                for _ in range(15):
                    vx, vy = streamer.get_domain_val_batch(d, num_samples=8)
                    vx = vx.to(device)
                    _, _, telem = model(vx, allow_spawning=False)
                    idx = telem[l_idx]['top_indices'].reshape(-1).cpu().numpy()
                    w = telem[l_idx]['top_weights'].reshape(-1).cpu().numpy()
                    for e_id, weight in zip(idx, w):
                        if e_id < 16:
                            weighted_usage[d][e_id] += weight

            layer_res = {}
            for d in domains:
                tot = weighted_usage[d].sum()
                if tot == 0:
                    tot = 1e-8
                order = np.argsort(-weighted_usage[d])
                shares = weighted_usage[d][order] / tot * 100
                top_str = ', '.join([f'E{order[i]}:{shares[i]:.1f}%' for i in range(5)])
                top3_sum = shares[:3].sum()
                layer_res[d] = {
                    "top_5": [f"E{order[i]}:{shares[i]:.1f}%" for i in range(5)],
                    "top3_mass_pct": float(top3_sum)
                }
                print(f"  {d:<18} -> {top_str} (Top-3 mass: {top3_sum:.1f}%)")
            results[f"L{l_idx}"] = layer_res

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Diagnostic results saved to: {out_json}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "experiments/checkpoint_hyperspace.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else "experiments/flagship_diagnostics_results.json"
    diagnose_checkpoint(ckpt, out)
