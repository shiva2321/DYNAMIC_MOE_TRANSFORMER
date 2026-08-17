"""
Multi-Domain Training Pipeline for NanoGPT with Dynamic Hyper-MoE.
Simulates a lifelong learning stream across Code, Math, Natural Language, and JSON.
Demonstrates:
- Dynamic expert spawning when new domains appear
- Continuous optimization without losing prior optimizer states
- Domain-specific expert specialization tracking
- Sample autoregressive generation
"""

import sys
import time
import math
import argparse
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data.stream_generator import MultiDomainStreamGenerator
from model.nanogpt import HyperTransformerLM

def dynamic_param_optimizer(model: HyperTransformerLM, lr: float = 3e-4, weight_decay: float = 0.01):
    """Creates AdamW optimizer with support for dynamically appending newly spawned expert parameters."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def sync_optimizer_params(optimizer: torch.optim.Optimizer, model: HyperTransformerLM):
    """
    Checks if new parameters were spawned (e.g. newly added experts) and attaches them to the optimizer.
    """
    existing_params = set()
    for group in optimizer.param_groups:
        for p in group['params']:
            existing_params.add(p)
            
    new_params = [p for p in model.parameters() if p not in existing_params and p.requires_grad]
    if new_params:
        optimizer.add_param_group({'params': new_params})

def main():
    parser = argparse.ArgumentParser(description="Train NanoGPT with Dynamic Hyper-MoE")
    parser.add_argument("--steps", type=int, default=250, help="Total training steps")
    parser.add_argument("--d_model", type=int, default=256, help="Model embedding dimension")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of Transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=512, help="Expert FFN hidden dimension")
    parser.add_argument("--d_hyper", type=int, default=2048, help="Hyperspace dimension")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k active experts")
    parser.add_argument("--spawn_threshold", type=float, default=0.28, help="Novelty threshold for spawning")
    parser.add_argument("--max_experts", type=int, default=16, help="Maximum experts per layer")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=96, help="Sequence length")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("================================================================================")
    print("  [INIT] DYNAMIC HYPER-MoE LANGUAGE MODEL (NANOGPT + HYPERSPACE)")
    print("================================================================================")
    print(f"Device: {device} | Hyperspace Dim (D): {args.d_hyper} | Layers: {args.n_layers}")
    print(f"Embedding Dim: {args.d_model} | Expert SwiGLU Dim: {args.d_ff} | Top-K: {args.top_k}")
    print(f"Spawn Novelty Threshold: {args.spawn_threshold} | Max Experts/Layer: {args.max_experts}\n")

    # Initialize data stream generator
    stream_gen = MultiDomainStreamGenerator(seq_len=args.seq_len, batch_size=args.batch_size)
    
    # Initialize Model
    model = HyperTransformerLM(
        vocab_size=stream_gen.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        d_hyper=args.d_hyper,
        top_k=args.top_k,
        spawn_threshold=args.spawn_threshold,
        max_experts=args.max_experts,
        max_seq_len=args.seq_len + 32,
    ).to(device)

    optimizer = dynamic_param_optimizer(model, lr=args.lr)

    # Domain cycle schedule
    domains = [0, 1, 2, 3] # Code, Math, Language, JSON
    steps_per_domain = 40

    print("--- Beginning Multi-Domain Streaming Training ---")
    start_time = time.time()
    
    # Track domain expert affinities: [domain_id -> expert_activation_counts]
    domain_expert_counts = {d: {} for d in domains}
    
    for step in range(1, args.steps + 1):
        # Current domain based on schedule
        active_domain_id = domains[(step // steps_per_domain) % len(domains)]
        
        inputs, targets, domain_name = stream_gen.get_domain_batch(active_domain_id)
        inputs, targets = inputs.to(device), targets.to(device)

        model.train()
        optimizer.zero_grad()
        
        # Track expert count before forward pass
        prev_experts = [b.hyper_moe.num_experts for b in model.blocks]
        
        logits, loss, layer_telemetries = model(inputs, targets=targets, allow_spawning=True)
        
        # Check if new experts were spawned in any layer
        curr_experts = [b.hyper_moe.num_experts for b in model.blocks]
        if curr_experts != prev_experts:
            sync_optimizer_params(optimizer, model)
            print(f"\n[SPAWN EVENT @ Step {step:3d}] Domain: {domain_name:<8} | Experts/Layer: {curr_experts}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track expert routing for this domain
        top_indices = layer_telemetries[0]["top_indices"] # Layer 0 expert routing: [Tokens, K]
        for exp_idx in top_indices.view(-1).tolist():
            domain_expert_counts[active_domain_id][exp_idx] = domain_expert_counts[active_domain_id].get(exp_idx, 0) + 1

        if step % 20 == 0 or step == 1 or step == args.steps:
            expert_summary = "/".join(str(b.hyper_moe.num_experts) for b in model.blocks)
            print(f"Step {step:4d}/{args.steps} | Domain: {domain_name:<8} | Loss: {loss.item():.4f} | Perplexity: {math.exp(min(15.0, loss.item())):.2f} | Experts: [{expert_summary}]")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s ({args.steps / total_time:.1f} steps/sec)\n")

    print("================================================================================")
    print("  [STATS] DOMAIN SPECIALIZATION MATRIX (Layer 0 Expert Routing Distribution)")
    print("================================================================================")
    total_experts = model.blocks[0].hyper_moe.num_experts
    header = f"{'Domain':<12} | " + " | ".join(f"Expert {i:2d}" for i in range(total_experts))
    print(header)
    print("-" * len(header))
    for d_id, (d_name, _) in stream_gen.domain_map.items():
        counts = domain_expert_counts.get(d_id, {})
        total_d_tokens = sum(counts.values()) or 1
        row = f"{d_name:<12} | "
        for i in range(total_experts):
            pct = (counts.get(i, 0) / total_d_tokens) * 100.0
            row += f"{pct:8.1f}% | "
        print(row)

    print("\n================================================================================")
    print("  [EVAL] AUTOREGRESSIVE GENERATION TEST ACROSS ALL 4 DOMAINS")
    print("================================================================================")
    sample_prompts = [
        ("Code", "def quicksort("),
        ("Math", "Theorem: For any right triangle"),
        ("Language", "The architecture of"),
        ("JSON", '{"system": "hyper_space",')
    ]

    for domain_name, prompt_text in sample_prompts:
        prompt_bytes = stream_gen.encode_text(prompt_text)
        prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
        
        gen_tokens = model.generate(prompt_tensor, max_new_tokens=48, temperature=0.7)
        gen_text = stream_gen.decode_tokens(gen_tokens[0].tolist())
        
        print(f"\n--- Domain: {domain_name} ---")
        print(f"Prompt: {prompt_text}")
        print(f"Output:\n{gen_text}")
        print("-" * 60)

if __name__ == "__main__":
    main()
