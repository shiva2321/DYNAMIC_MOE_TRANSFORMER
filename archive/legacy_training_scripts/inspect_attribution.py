"""
CLI Tool for Token-Level Expert Attribution and Layer Routing Inspection.
Demonstrates:
1. Exact Expert IDs contributing to each generated token
2. Expert contribution routing weights (Softmax probabilities)
3. Layer-by-Layer Expert Activation Breakdown
4. ANSI Color-Coded Terminal Output
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.nanogpt import HyperTransformerLM
from data.dataset_hub import MultiDomainDatasetHub
from hyperspace.vsa import ComplexPhasorVSA

# ANSI Colors for terminal expert attribution
EXPERT_COLORS = [
    "\033[91m", # Red (Expert 0)
    "\033[92m", # Green (Expert 1)
    "\033[94m", # Blue (Expert 2)
    "\033[93m", # Yellow (Expert 3)
    "\033[95m", # Magenta (Expert 4)
    "\033[96m", # Cyan (Expert 5)
    "\033[97m", # White (Expert 6)
]
RESET = "\033[0m"

def load_dynamic_checkpoint(ckpt_path: str, device: torch.device) -> HyperTransformerLM:
    """Loads model checkpoint and dynamically instantiates spawned experts to match checkpoint state."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = state.get("config", {})
    state_dict = state.get("model_state", state)

    d_model = config.get("d_model", 384)
    n_layers = config.get("n_layers", 6)
    n_heads = config.get("n_heads", 6)
    d_ff = config.get("d_ff", 1024)
    d_hyper = config.get("d_hyper", 2048)
    top_k = config.get("top_k", 2)
    max_experts = config.get("max_experts", 16)
    spawn_threshold = config.get("spawn_threshold", 0.25)
    seq_len = config.get("seq_len", 256)
    max_seq_len = seq_len + 32

    model = HyperTransformerLM(
        vocab_size=50304,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        d_hyper=d_hyper,
        top_k=top_k,
        max_experts=max_experts,
        spawn_threshold=spawn_threshold,
        max_seq_len=max_seq_len,
    ).to(device)

    # Spawn experts to match saved checkpoint
    for b_idx, block in enumerate(model.blocks):
        exp_keys = [k for k in state_dict.keys() if k.startswith(f"blocks.{b_idx}.hyper_moe.experts.")]
        expert_ids = set()
        for k in exp_keys:
            parts = k.split(".")
            exp_id = int(parts[4])
            expert_ids.add(exp_id)

        target_num_experts = max(len(expert_ids), 2)
        while block.hyper_moe.num_experts < target_num_experts:
            seed = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper), device=device)
            block.hyper_moe._spawn_expert(seed, label="loaded_expert")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"[Loaded Checkpoint: {ckpt_path} | Layers: {n_layers} | Experts/Layer: {model.blocks[0].hyper_moe.num_experts}]")
    return model

def trace_prompt_attribution(
    model: HyperTransformerLM,
    hub: MultiDomainDatasetHub,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.75,
    top_k: int = 40,
    device: torch.device = None,
):
    model.eval()
    prompt_tokens = hub.encode(prompt)
    curr_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    print("\n" + "=" * 90)
    print("  [EXPERT ATTRIBUTION TRACER] REAL-TIME TOKEN-LEVEL ROUTING INSPECTION")
    print("=" * 90)
    print(f"PROMPT: {prompt}\n")

    token_trace = []
    layer_expert_usage = {layer_idx: {} for layer_idx in range(len(model.blocks))}

    for _ in range(max_new_tokens):
        idx_cond = curr_ids if curr_ids.size(1) <= model.max_seq_len else curr_ids[:, -model.max_seq_len:]
        with torch.no_grad():
            logits, _, telemetries = model(idx_cond, allow_spawning=False)

        last_logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
            last_logits[last_logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        curr_ids = torch.cat([curr_ids, next_token], dim=1)

        token_str = hub.decode([next_token.item()])

        # Extract per-layer routing for the newly generated token
        step_routing_per_layer = []
        for l_idx, telem in enumerate(telemetries):
            # top_indices: [Batch, SeqLen, K]
            top_exps = telem["top_indices"][0, -1].tolist()
            top_wgts = telem["top_weights"][0, -1].tolist()
            step_routing_per_layer.append(list(zip(top_exps, top_wgts)))

            for exp_id in top_exps:
                layer_expert_usage[l_idx][exp_id] = layer_expert_usage[l_idx].get(exp_id, 0) + 1

        primary_exp, primary_weight = step_routing_per_layer[0][0]

        token_trace.append({
            "token_id": next_token.item(),
            "token_str": token_str,
            "primary_expert": primary_exp,
            "primary_weight": primary_weight,
            "layer_routing": step_routing_per_layer,
        })

    # 1. Print Color-Coded Inline Text
    print("--- 1. Color-Coded Generated Response (by Layer 0 Primary Expert) ---")
    sys.stdout.write(prompt)
    for tok in token_trace:
        color = EXPERT_COLORS[tok["primary_expert"] % len(EXPERT_COLORS)]
        sys.stdout.write(f"{color}{tok['token_str']}{RESET}")
    print("\n")

    # 2. Print Token-by-Token Routing Table
    print("--- 2. Token-by-Token Routing Breakdown (First 15 Tokens) ---")
    print(f"{'Token':<18} | {'Layer 0 Active Experts (Weights)':<36} | {'Layer 1 Active Experts (Weights)':<36}")
    print("-" * 95)
    for tok in token_trace[:15]:
        tok_repr = repr(tok['token_str'])
        l0_str = ", ".join([f"E{e} ({w:.2f})" for e, w in tok['layer_routing'][0]])
        l1_str = ", ".join([f"E{e} ({w:.2f})" for e, w in tok['layer_routing'][1]]) if len(tok['layer_routing']) > 1 else "N/A"
        print(f"{tok_repr:<18} | {l0_str:<36} | {l1_str:<36}")
    if len(token_trace) > 15:
        print(f"... [{len(token_trace) - 15} additional tokens generated]")

    # 3. Print Layer-by-Layer Global Expert Utilization
    print("\n--- 3. Total Expert Contribution Breakdown across Layers ---")
    for l_idx, counts in layer_expert_usage.items():
        total = sum(counts.values())
        breakdown = " | ".join([f"Expert {e}: {c / total * 100:.1f}% ({c} hits)" for e, c in sorted(counts.items())])
        print(f"Layer {l_idx}: {breakdown}")
    print("=" * 90 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="def dijkstra_shortest_path(graph, start):", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=35)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--checkpoint", type=str, default="experiments/checkpoints/scaled_production_hyperspace.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    hub = MultiDomainDatasetHub(seq_len=256, batch_size=4, use_bpe=True)

    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        ckpt = "experiments/checkpoints/benchmark_hyperspace_moe.pt"

    model = load_dynamic_checkpoint(ckpt, device=device)

    trace_prompt_attribution(
        model=model,
        hub=hub,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )

if __name__ == "__main__":
    main()
