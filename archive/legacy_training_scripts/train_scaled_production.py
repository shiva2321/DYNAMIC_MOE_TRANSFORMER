"""
Production-Scale High-Throughput Training Engine for Hyperspace 2.0.
Trains 25M-45M Parameter Neuro-Cognitive Language Models on Millions of Tokens
with Cosine Warmup, Gradient Accumulation, AMP Mixed Precision, and Memory-Mapped Shards.
"""

import os
import sys
import time
import math
import json
import argparse
from typing import Dict, List, Any, Tuple
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data.scale_data_engine import ScaledMultiDomainHub, DOMAIN_KEYS
from model.nanogpt import HyperTransformerLM

def get_lr_scheduler(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate scheduler with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / (warmup_steps + 1)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def sync_dynamic_optimizer(optimizer: torch.optim.Optimizer, model: HyperTransformerLM) -> int:
    """Attaches newly spawned expert parameters to the optimizer while preserving state."""
    existing = set()
    for group in optimizer.param_groups:
        for p in group['params']:
            existing.add(p)
    new_params = [p for p in model.parameters() if p not in existing and p.requires_grad]
    if new_params:
        optimizer.add_param_group({'params': new_params, 'weight_decay': 0.01})
    return len(new_params)

def evaluate_validation_loss(
    model: HyperTransformerLM,
    hub: ScaledMultiDomainHub,
    device: torch.device,
    eval_batches_per_domain: int = 4,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Computes validation loss across all 7 domains on held-out binary shards."""
    model.eval()
    domain_val_losses = {}
    with torch.no_grad():
        for domain in DOMAIN_KEYS:
            total_loss = 0.0
            for _ in range(eval_batches_per_domain):
                x, y = hub.get_batch(domain, split="val", device=device)
                if use_amp and device.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        _, loss, _ = model(x, targets=y, allow_spawning=False)
                else:
                    _, loss, _ = model(x, targets=y, allow_spawning=False)
                total_loss += loss.item()
            domain_val_losses[domain] = total_loss / eval_batches_per_domain
    return domain_val_losses

def main():
    parser = argparse.ArgumentParser(description="Production-Scale Training for Hyperspace 2.0")
    parser.add_argument("--steps", type=int, default=1200, help="Total optimization steps")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Linear warmup steps")
    parser.add_argument("--d_model", type=int, default=384, help="Embedding dimension (25M model scale)")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of Transformer blocks")
    parser.add_argument("--n_heads", type=int, default=6, help="Attention heads")
    parser.add_argument("--d_ff", type=int, default=1024, help="Expert FFN dimension")
    parser.add_argument("--d_hyper", type=int, default=2048, help="Hyperspace dimension")
    parser.add_argument("--top_k", type=int, default=2, help="Active top-k experts per token")
    parser.add_argument("--spawn_threshold", type=float, default=0.25, help="Novelty threshold for spawning")
    parser.add_argument("--max_experts", type=int, default=16, help="Max experts per layer")
    parser.add_argument("--batch_size", type=int, default=16, help="Micro-batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length in tokens")
    parser.add_argument("--max_lr", type=float, default=8e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=8e-5, help="Minimum learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Output directory")
    args = parser.parse_args()

    device = torch.device(args.device)
    use_amp = (device.type == "cuda")

    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    tokens_per_step = args.batch_size * args.grad_accum_steps * args.seq_len
    total_tokens_trained = args.steps * tokens_per_step

    print("================================================================================")
    print("  [PRODUCTION TRAINER] SCALED HYPERSPACE 2.0 (25M-45M PARAMETERS)")
    print("================================================================================")
    print(f"Device: {device} | CUDA Mixed Precision (AMP): {use_amp}")
    print(f"Architecture: d_model={args.d_model} | Layers={args.n_layers} | Heads={args.n_heads} | d_ff={args.d_ff}")
    print(f"Hyperspace Dim: {args.d_hyper} | Top-K: {args.top_k} | Spawn Threshold: {args.spawn_threshold}")
    print(f"Tokens/Step: {tokens_per_step:,d} (Micro-batch: {args.batch_size} × Accum: {args.grad_accum_steps} × Seq: {args.seq_len})")
    print(f"Total Steps: {args.steps:,d} | Total Tokens to Process: {total_tokens_trained:,d}\n")

    # Initialize Scaled Binary Multi-Domain Hub
    hub = ScaledMultiDomainHub(seq_len=args.seq_len, batch_size=args.batch_size)

    # Initialize Scaled Hyperspace 2.0 Model
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"-> Total Model Parameters: {total_params:,d} ({total_params / 1e6:.2f}M params)\n")

    optimizer = model.configure_optimizers(lr=args.max_lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_time = time.time()
    last_log_time = time.time()
    last_log_tokens = 0

    telemetry_records = []
    spawn_events = []

    print("--- Beginning Scaled Multi-Domain Training Run ---")

    for step in range(1, args.steps + 1):
        # Update learning rate via Cosine Schedule
        lr = get_lr_scheduler(step, args.warmup_steps, args.steps, args.max_lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        prev_experts = model.blocks[0].hyper_moe.num_experts

        # Gradient Accumulation Loop
        for micro_step in range(args.grad_accum_steps):
            # Interleave domains across accumulation micro-steps
            active_domain = DOMAIN_KEYS[(step * args.grad_accum_steps + micro_step) % len(DOMAIN_KEYS)]
            x, y = hub.get_batch(active_domain, split="train", device=device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits, loss, telemetries = model(x, targets=y, allow_spawning=True)
                    loss = loss / args.grad_accum_steps
                scaler.scale(loss).backward()
            else:
                logits, loss, telemetries = model(x, targets=y, allow_spawning=True)
                loss = loss / args.grad_accum_steps
                loss.backward()

            accum_loss += loss.item() * args.grad_accum_steps

        # Check for newly spawned experts and dynamically sync optimizer
        curr_experts = model.blocks[0].hyper_moe.num_experts
        if curr_experts != prev_experts:
            new_p = sync_dynamic_optimizer(optimizer, model)
            spawn_event = {
                "step": step,
                "prev_experts": prev_experts,
                "curr_experts": curr_experts,
                "new_params": new_p,
            }
            spawn_events.append(spawn_event)
            print(f"\n⚡ [SPAWN EVENT @ Step {step:4d}] Experts: {prev_experts} -> {curr_experts} (Dynamic Param Sync)")

        # Optimizer step with gradient clipping
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Telemetry & Throughput Logging
        if step % 25 == 0 or step == args.steps:
            now = time.time()
            elapsed_chunk = now - last_log_time
            tokens_chunk = 25 * tokens_per_step
            tok_per_sec = tokens_chunk / max(1e-5, elapsed_chunk)
            last_log_time = now

            sigma = telemetries[0]["branching_ratio"] if telemetries else 1.0
            r_temp = telemetries[0]["routing_temperature"] if telemetries else 10.0

            val_losses = evaluate_validation_loss(model, hub, device, eval_batches_per_domain=2, use_amp=use_amp)
            avg_val_loss = sum(val_losses.values()) / len(val_losses)
            val_ppl = math.exp(min(avg_val_loss, 20.0))

            telemetry_records.append({
                "step": step,
                "train_loss": round(accum_loss, 4),
                "avg_val_loss": round(avg_val_loss, 4),
                "val_perplexity": round(val_ppl, 2),
                "val_losses_per_domain": {k: round(v, 4) for k, v in val_losses.items()},
                "lr": round(lr, 7),
                "tok_per_sec": round(tok_per_sec, 1),
                "num_experts": curr_experts,
                "sigma": round(sigma, 4),
            })

            print(f"Step {step:4d}/{args.steps:4d} | Train Loss: {accum_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:6.2f}) | "
                  f"LR: {lr:.2e} | Speed: {tok_per_sec:6,.0f} tok/s | Exp: {curr_experts} | σ: {sigma:.4f}")

        # Periodic Generation Checkpoint (Every 200 steps)
        if step % 200 == 0 or step == args.steps:
            print("\n--------------------------------------------------------------------------------")
            print(f"  [CHECKPOINT GENERATION @ Step {step}]")
            print("--------------------------------------------------------------------------------")
            sample_prompts = [
                ("Python Code", "def dijkstra_shortest_path(graph, start):"),
                ("Mathematics", "Theorem: In any Hilbert space H, the spectral projection operator"),
                ("Sci-Fi Fiction", "Captain Elena Vance looked out at the dormant orbital array and"),
            ]
            for label, prompt in sample_prompts:
                p_tokens = hub.encode(prompt)
                p_tensor = torch.tensor([p_tokens], dtype=torch.long, device=device)
                with torch.no_grad():
                    gen_ids = model.generate(p_tensor, max_new_tokens=35, temperature=0.75, top_k=40)
                    gen_str = hub.decode(gen_ids[0].tolist())
                print(f"[{label}]\n{gen_str}\n")
            print("--------------------------------------------------------------------------------\n")

    total_time = time.time() - start_time
    avg_speed = total_tokens_trained / total_time
    print(f"\n[PRODUCTION RUN COMPLETE] Processed {total_tokens_trained:,d} tokens in {total_time:.2f}s ({avg_speed:,.0f} tokens/sec)")

    # Save Checkpoint & Telemetry Logs
    ckpt_path = os.path.join(args.output_dir, "checkpoints", "scaled_production_hyperspace.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": vars(args),
        "total_params": total_params,
        "telemetry_records": telemetry_records,
    }, ckpt_path)
    print(f"[SAVED] Production model checkpoint saved to {ckpt_path}")

    log_path = os.path.join(args.output_dir, "logs", "scaled_production_training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "total_params": total_params,
            "total_tokens_trained": total_tokens_trained,
            "total_time_seconds": round(total_time, 2),
            "average_throughput_tok_sec": round(avg_speed, 1),
            "telemetry_records": telemetry_records,
            "spawn_events": spawn_events,
        }, f, indent=2)
    print(f"[SAVED] Production training telemetry saved to {log_path}\n")

if __name__ == "__main__":
    main()
