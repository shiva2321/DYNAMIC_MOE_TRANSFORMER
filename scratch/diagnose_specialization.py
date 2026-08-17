"""Direct empirical check: does per-domain routing actually concentrate on distinct experts,
and are those experts' usage histories dominated by that domain, or is routing diffuse/shared?"""
import os, sys, json
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter

from model.nanogpt import HyperTransformerLM
from hyperspace.vsa import ComplexPhasorVSA
from train_scaled_production_engine import ScaledProductionDataStreamer

device = torch.device("cpu")
ckpt_path = "experiments/checkpoints/hyperspace_scaled_production_master.pt"
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
print("Experts per layer in this checkpoint:", per_layer_experts, "n_layers:", n_layers)

model = HyperTransformerLM(
    vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
    d_hyper=d_hyper, top_k=2, max_k=4, top_p=0.85, dynamic_k=True, spawn_threshold=0.30,
    max_experts=max(per_layer_experts), initial_experts=2, use_sparse_attn=True,
    foveal_window=128, num_landmarks=4, num_sinks=4, max_seq_len=576, dropout=0.0,
).to(device)

for l_idx, block in enumerate(model.blocks):
    needed = per_layer_experts[l_idx] - block.hyper_moe.num_experts
    for _ in range(needed):
        dummy_key = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper), device=device)
        block.hyper_moe._spawn_expert(dummy_key, label="restore")

clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(clean_sd, strict=True)
model.eval()
print("Loaded. Total experts per layer:", [b.hyper_moe.num_experts for b in model.blocks])

streamer = ScaledProductionDataStreamer(cache_dir="data/scaled_real_corpus", seq_len=256, batch_size=8)
domains = ["fineweb_edu", "python_code", "wikitext_facts", "natural_stories"]

per_domain_layer_usage = {d: [Counter() for _ in range(n_layers)] for d in domains}

with torch.no_grad():
    for d in domains:
        for _ in range(15):  # 15 batches x 8 seqs x 256 tokens = ~30,720 tokens sampled per domain
            vx, vy = streamer.get_domain_val_batch(d, num_samples=8)
            logits, _, telemetries = model(vx, allow_spawning=False)
            for l_idx, telem in enumerate(telemetries):
                top_idx = telem["top_indices"]  # [B, S, k]
                flat = top_idx.reshape(-1).tolist()
                per_domain_layer_usage[d][l_idx].update(flat)

print("\n" + "="*100)
print(" PER-DOMAIN EXPERT USAGE DISTRIBUTION (last layer, most semantically differentiated)")
print("="*100)
last_l = n_layers - 1
domain_top_experts = {}
for d in domains:
    counts = per_domain_layer_usage[d][last_l]
    total = sum(counts.values())
    ranked = counts.most_common()
    top1_share = ranked[0][1] / total if ranked else 0
    top2_share = sum(c for _, c in ranked[:2]) / total if ranked else 0
    domain_top_experts[d] = set(e for e, c in ranked[:3])
    print(f"\n{d}: total_routed_slots={total}, distinct_experts_touched={len(ranked)}")
    print(f"  Top-1 expert concentration: {top1_share*100:.1f}%  |  Top-2 concentration: {top2_share*100:.1f}%")
    print(f"  Top 5 experts (id: count, %): " + ", ".join(f"E{e}:{c}({c/total*100:.1f}%)" for e, c in ranked[:5]))

print("\n" + "="*100)
print(" CROSS-DOMAIN OVERLAP (top-3 experts per domain)")
print("="*100)
for i, d1 in enumerate(domains):
    for d2 in domains[i+1:]:
        overlap = domain_top_experts[d1] & domain_top_experts[d2]
        print(f"  {d1} ∩ {d2}: shared top-3 experts = {overlap if overlap else 'NONE'}")

print("\n" + "="*100)
print(" EXPERT USAGE ACROSS ALL LAYERS SUMMARY")
print("="*100)
for l_idx in range(n_layers):
    print(f"\nLayer {l_idx}:")
    for d in domains:
        counts = per_domain_layer_usage[d][l_idx]
        total = sum(counts.values())
        ranked = counts.most_common(3)
        top1_share = ranked[0][1]/total*100 if ranked else 0
        print(f"  {d:<16} top1={ranked[0][0] if ranked else '-'} ({top1_share:.1f}%)  top3={[e for e,c in ranked]}")
