"""
High-Performance Multi-Dataset Training Engine for Hyperspace 2.0.
Supports 7 distinct domains with BPE Tokenization, CUDA Mixed Precision,
Dynamic Expert Spawning, and Continual Lifelong Learning streams.
"""

import os
import sys
import time
import math
import json
import argparse
from typing import Dict, List, Any, Optional
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data.dataset_hub import MultiDomainDatasetHub, DOMAIN_LIST, DOMAIN_METADATA
from model.nanogpt import HyperTransformerLM

def sync_dynamic_optimizer(optimizer: torch.optim.Optimizer, model: HyperTransformerLM) -> int:
    """Detects any newly spawned expert parameters and dynamically registers them to optimizer."""
    existing = set()
    for group in optimizer.param_groups:
        for p in group['params']:
            existing.add(p)
    new_params = [p for p in model.parameters() if p not in existing and p.requires_grad]
    if new_params:
        optimizer.add_param_group({'params': new_params})
    return len(new_params)

def evaluate_all_domains(
    model: HyperTransformerLM,
    hub: MultiDomainDatasetHub,
    device: torch.device,
    num_batches: int = 4,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Computes evaluation loss and perplexity across all 7 domains."""
    model.eval()
    domain_losses = {}
    with torch.no_grad():
        for domain_key in DOMAIN_LIST:
            total_loss = 0.0
            for _ in range(num_batches):
                inputs, targets, _ = hub.get_batch(domain_key, is_eval=True)
                inputs, targets = inputs.to(device), targets.to(device)
                if use_amp and device.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        _, loss, _ = model(inputs, targets=targets, allow_spawning=False)
                else:
                    _, loss, _ = model(inputs, targets=targets, allow_spawning=False)
                total_loss += loss.item()
            domain_losses[domain_key] = total_loss / num_batches
    return domain_losses

def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset Training Engine for Hyperspace 2.0")
    parser.add_argument("--mode", type=str, default="sequential", choices=["sequential", "multitask", "transfer"],
                        help="Training regime: sequential (continual), multitask (interleaved), or transfer (zero-shot holdout)")
    parser.add_argument("--steps_per_domain", type=int, default=80, help="Steps per domain in sequential mode")
    parser.add_argument("--total_steps", type=int, default=560, help="Total steps for multitask mode")
    parser.add_argument("--d_model", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--n_layers", type=int, default=3, help="Transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--d_ff", type=int, default=512, help="Expert FFN hidden dimension")
    parser.add_argument("--d_hyper", type=int, default=2048, help="Hyperspace dimension")
    parser.add_argument("--top_k", type=int, default=2, help="Active top-k experts per token")
    parser.add_argument("--spawn_threshold", type=float, default=0.26, help="Novelty threshold for spawning")
    parser.add_argument("--max_experts", type=int, default=16, help="Max experts per layer")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Directory for logs and checkpoints")
    parser.add_argument("--save_checkpoint", action="store_true", default=True, help="Save final checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device)
    use_amp = (device.type == "cuda")

    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    print("================================================================================")
    print("  [TRAIN] MULTI-DATASET HYPERSPACE 2.0 ENGINE")
    print("================================================================================")
    print(f"Device: {device} | CUDA Mixed Precision (AMP): {use_amp} | Mode: {args.mode.upper()}")
    print(f"Layers: {args.n_layers} | d_model: {args.d_model} | d_hyper: {args.d_hyper} | Top-K: {args.top_k}")
    print(f"Spawn Novelty Threshold: {args.spawn_threshold} | Max Experts: {args.max_experts}\n")

    # Initialize Multi-Domain Dataset Hub
    hub = MultiDomainDatasetHub(seq_len=args.seq_len, batch_size=args.batch_size, use_bpe=True)

    # Initialize Hyperspace 2.0 Model
    model = HyperTransformerLM(
        vocab_size=hub.vocab_size,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Training Schedule Setup
    if args.mode == "sequential":
        active_domains = DOMAIN_LIST
        total_training_steps = len(active_domains) * args.steps_per_domain
    elif args.mode == "transfer":
        active_domains = DOMAIN_LIST[:5] # Train on first 5, hold out last 2
        total_training_steps = len(active_domains) * args.steps_per_domain
    else: # multitask
        active_domains = DOMAIN_LIST
        total_training_steps = args.total_steps

    print(f"-> Active Domains ({len(active_domains)}): {', '.join(active_domains)}")
    print(f"-> Total Training Steps: {total_training_steps}\n")

    telemetry_log = []
    domain_expert_activations = {d: {} for d in DOMAIN_LIST}
    spawn_events = []

    start_time = time.time()

    for step in range(1, total_training_steps + 1):
        # Determine current domain
        if args.mode == "sequential" or args.mode == "transfer":
            domain_idx = (step - 1) // args.steps_per_domain
            current_domain_key = active_domains[domain_idx % len(active_domains)]
        else: # multitask
            current_domain_key = random.choice(active_domains)

        inputs, targets, domain_name = hub.get_batch(current_domain_key)
        inputs, targets = inputs.to(device), targets.to(device)

        model.train()
        optimizer.zero_grad()

        prev_expert_count = model.blocks[0].hyper_moe.num_experts

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits, loss, layer_telemetries = model(inputs, targets=targets, allow_spawning=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss, layer_telemetries = model(inputs, targets=targets, allow_spawning=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        curr_expert_count = model.blocks[0].hyper_moe.num_experts
        if curr_expert_count != prev_expert_count:
            new_params = sync_dynamic_optimizer(optimizer, model)
            spawn_event = {
                "step": step,
                "domain": current_domain_key,
                "domain_name": domain_name,
                "prev_experts": prev_expert_count,
                "curr_experts": curr_expert_count,
                "new_params_count": new_params,
            }
            spawn_events.append(spawn_event)
            print(f"\n⚡ [SPAWN EVENT @ Step {step:3d}] Domain: {domain_name:<30} | Experts: {prev_expert_count} -> {curr_expert_count}")

        # Track routing activations for Layer 0
        if layer_telemetries:
            top_indices = layer_telemetries[0]["top_indices"].view(-1)
            for idx in top_indices.tolist():
                domain_expert_activations[current_domain_key][idx] = domain_expert_activations[current_domain_key].get(idx, 0) + 1

        # Periodic logging and evaluation
        if step % 20 == 0 or step == total_training_steps:
            eval_losses = evaluate_all_domains(model, hub, device, num_batches=2, use_amp=use_amp)
            avg_eval_loss = sum(eval_losses.values()) / len(eval_losses)
            ppl = math.exp(min(avg_eval_loss, 20.0))
            
            sigma = layer_telemetries[0]["branching_ratio"] if layer_telemetries else 1.0
            r_temp = layer_telemetries[0]["routing_temperature"] if layer_telemetries else 10.0
            
            step_record = {
                "step": step,
                "active_domain": current_domain_key,
                "train_loss": round(loss.item(), 4),
                "avg_eval_loss": round(avg_eval_loss, 4),
                "eval_perplexity": round(ppl, 2),
                "domain_eval_losses": {k: round(v, 4) for k, v in eval_losses.items()},
                "num_experts": curr_expert_count,
                "branching_ratio_sigma": round(sigma, 4),
                "routing_temperature": round(r_temp, 2),
            }
            telemetry_log.append(step_record)
            
            print(f"Step {step:4d}/{total_training_steps:4d} | Domain: {current_domain_key:<12} | "
                  f"Train Loss: {loss.item():.4f} | Avg Eval Loss: {avg_eval_loss:.4f} (PPL: {ppl:6.2f}) | "
                  f"Experts: {curr_expert_count} | σ: {sigma:.4f}")

    total_time = time.time() - start_time
    print(f"\n[COMPLETE] Training completed in {total_time:.2f}s ({total_training_steps / total_time:.1f} steps/sec)")
    print(f"Total Spawn Events: {len(spawn_events)} | Final Layer Experts: {model.blocks[0].hyper_moe.num_experts}")

    # Comprehensive Final Evaluation Across All 7 Domains
    print("\n================================================================================")
    print("  [EVALUATION] FINAL RETENTION & PERPLEXITY ACROSS ALL 7 DOMAINS")
    print("================================================================================")
    final_eval_losses = evaluate_all_domains(model, hub, device, num_batches=6, use_amp=use_amp)
    for domain_key in DOMAIN_LIST:
        d_loss = final_eval_losses[domain_key]
        d_ppl = math.exp(min(d_loss, 20.0))
        d_meta = DOMAIN_METADATA[domain_key]
        print(f"Domain {d_meta['id']}: {d_meta['name']:<35} | Loss: {d_loss:.4f} | PPL: {d_ppl:7.2f}")
    print("================================================================================\n")

    # Sample Coherent Text Generation for Each Domain
    print("================================================================================")
    print("  [GENERATION] MULTI-DOMAIN AUTOREGRESSIVE TEXT GENERATION")
    print("================================================================================")
    generation_results = {}
    for domain_key in DOMAIN_LIST:
        sample_prompt = DOMAIN_METADATA[domain_key]["sample_prompt"]
        prompt_tokens = hub.encode(sample_prompt)
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
        with torch.no_grad():
            gen_tokens = model.generate(prompt_tensor, max_new_tokens=40, temperature=0.75, top_k=30)
            gen_text = hub.decode(gen_tokens[0].tolist())
            
        generation_results[domain_key] = {
            "prompt": sample_prompt,
            "generated_text": gen_text,
        }
        print(f"\n--- Domain: {DOMAIN_METADATA[domain_key]['name']} ---")
        print(f"PROMPT: {sample_prompt}")
        print(f"GENERATION:\n{gen_text}\n")

    # Save Experiment Telemetry & Checkpoint
    log_file = os.path.join(args.output_dir, "logs", f"multi_dataset_{args.mode}_log.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "total_time_seconds": total_time,
            "spawn_events": spawn_events,
            "telemetry_log": telemetry_log,
            "domain_expert_activations": domain_expert_activations,
            "final_eval_losses": final_eval_losses,
            "generations": generation_results,
        }, f, indent=2)
    print(f"[SAVED] Experiment logs saved to {log_file}")

    if args.save_checkpoint:
        ckpt_path = os.path.join(args.output_dir, "checkpoints", f"hyperspace_multidomain_{args.mode}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "config": vars(args),
            "final_eval_losses": final_eval_losses,
            "domain_activations": domain_expert_activations,
        }, ckpt_path)
        print(f"[SAVED] Model checkpoint saved to {ckpt_path}\n")

if __name__ == "__main__":
    main()
