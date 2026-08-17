"""Weighted version: measure each expert's share of ACTUAL routing weight mass per domain,
not raw slot occupancy (which double-counts near-zero-weight dynamic-k slots)."""
import os, sys, time
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import torch
from model.nanogpt import HyperTransformerLM
from hyperspace.vsa import ComplexPhasorVSA
from train_scaled_production_engine import ScaledProductionDataStreamer

device = torch.device("cpu")
checkpoint = torch.load("experiments/checkpoints/hyperspace_scaled_production_master.pt", map_location=device, weights_only=False)
sd = checkpoint.get("model_state", checkpoint)
model = HyperTransformerLM(vocab_size=50304, d_model=384, n_layers=4, n_heads=6, d_ff=768, d_hyper=2048,
    top_k=2, max_k=4, top_p=0.85, dynamic_k=True, spawn_threshold=0.30, max_experts=16, initial_experts=2,
    use_sparse_attn=True, foveal_window=128, num_landmarks=4, num_sinks=4, max_seq_len=576, dropout=0.0).to(device)
for block in model.blocks:
    for _ in range(16 - block.hyper_moe.num_experts):
        dk = ComplexPhasorVSA.random_hyperspace_vector((1, 2048), device=device)
        block.hyper_moe._spawn_expert(dk, label="restore")
model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()}, strict=True)
model.eval()

streamer = ScaledProductionDataStreamer(cache_dir="data/scaled_real_corpus", seq_len=256, batch_size=8)
domains = ["fineweb_edu", "python_code", "wikitext_facts", "natural_stories"]
last_l = 3
weighted_usage = {d: np.zeros(16) for d in domains}
mean_active_k = {d: [] for d in domains}

t0=time.time()
with torch.no_grad():
    for d in domains:
        for _ in range(10):
            vx, vy = streamer.get_domain_val_batch(d, num_samples=8)
            logits, _, telem = model(vx, allow_spawning=False)
            idx = telem[last_l]["top_indices"].reshape(-1)      # [B*S*k]
            w = telem[last_l]["top_weights"].reshape(-1)         # [B*S*k]
            for e_id, weight in zip(idx.tolist(), w.tolist()):
                weighted_usage[d][e_id] += weight
            # mean active k (weight > 1e-4 counts as "active")
            active_per_token = (telem[last_l]["top_weights"] > 1e-4).float().sum(dim=-1)
            mean_active_k[d].append(active_per_token.mean().item())
print(f"done in {time.time()-t0:.1f}s\n")

print("="*100)
print(" WEIGHT-BASED PER-DOMAIN EXPERT USAGE (last layer) -- this is what actually contributes to output")
print("="*100)
domain_top = {}
for d in domains:
    total = weighted_usage[d].sum()
    order = np.argsort(-weighted_usage[d])
    shares = weighted_usage[d][order] / total * 100
    domain_top[d] = set(order[:3].tolist())
    print(f"\n{d}  (mean active k = {np.mean(mean_active_k[d]):.2f})")
    print("  Top-5 by weight:", ", ".join(f"E{order[i]}:{shares[i]:.1f}%" for i in range(5)))

print("\n" + "="*100)
print(" CROSS-DOMAIN TOP-3 OVERLAP (weight-based)")
print("="*100)
for i, d1 in enumerate(domains):
    for d2 in domains[i+1:]:
        print(f"  {d1} ∩ {d2}: {domain_top[d1] & domain_top[d2] or 'NONE'}")
