"""
16-Domain Autonomous Production Training Engine for Hyperspace 2.0.
Trains on 21.6 Million tokens across 16 scientific and technical disciplines with:
1. DynamicWarmupAdamW with localized group warmup (no gradient shocks).
2. Autonomous on-the-fly expert spawning (Zero hardcoding - model starts with 2 bootstrap experts).
3. Real-time autonomous spawn event logger and domain attribution tracking.
4. PyTorch CUDA AMP mixed precision with gradient accumulation (16,384 tokens/step).
"""

import os
import sys

# Configure PyTorch CUDA memory allocator to use virtual expandable segments (prevents fragmentation during on-the-fly expert allocation)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import json
import math
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.nanogpt import HyperTransformerLM
from hyperspace.dynamic_optimizer import DynamicWarmupAdamW
from data.dataset_hub import MultiDomainDatasetHub

class Scaled16DomainMemmapLoader:
    def __init__(self, cache_dir: str = "data/scaled_cache_16d", seq_len: int = 256, batch_size: int = 16):
        self.cache_dir = cache_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        meta_file = os.path.join(cache_dir, "metadata_16d.json")
        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        self.domains = list(self.metadata["domains"].keys())
        self.train_shards = {}
        self.val_shards = {}
        
        for d in self.domains:
            tr_file = self.metadata["domains"][d]["train_file"]
            va_file = self.metadata["domains"][d]["val_file"]
            
            self.train_shards[d] = np.memmap(tr_file, dtype=np.uint16, mode='r')
            self.val_shards[d] = np.memmap(va_file, dtype=np.uint16, mode='r')

    def get_batch(self, split: str = "train", domain: str = None) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if domain is None:
            domain = np.random.choice(self.domains)
            
        data = self.train_shards[domain] if split == "train" else self.val_shards[domain]
        max_start = len(data) - self.seq_len - 1
        starts = np.random.randint(0, max_start, size=self.batch_size)
        
        x = np.stack([data[s:s + self.seq_len] for s in starts]).astype(np.int64)
        y = np.stack([data[s + 1:s + self.seq_len + 1] for s in starts]).astype(np.int64)
        
        return torch.from_numpy(x), torch.from_numpy(y), domain

def train_16domain_production(args):
    device = torch.device(args.device)
    loader = Scaled16DomainMemmapLoader(cache_dir="data/scaled_cache_16d", seq_len=args.seq_len, batch_size=args.batch_size)
    hub = MultiDomainDatasetHub(seq_len=args.seq_len, batch_size=args.batch_size, use_bpe=True)

    print("=" * 90)
    print("  [16-DOMAIN PRODUCTION ENGINE] AUTONOMOUS HYPERSPACE 2.0 PRETRAINING")
    print("=" * 90)
    print(f"Device: {device} | Mixed Precision (AMP): {torch.cuda.is_available()}")
    print(f"16 Knowledge Domains: {', '.join(loader.domains[:8])} ... (+8 more)")
    print(f"Tokens / Step: {args.batch_size * args.grad_accum_steps * args.seq_len:,d} | Total Steps: {args.steps}")

    # Model initialized with ONLY 2 bootstrap generic experts per layer
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

    # Dynamic optimizer with localized per-group warmup
    decay_params, nodecay_params = [], []
    for pn, p in model.named_parameters():
        if p.requires_grad:
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

    optimizer = DynamicWarmupAdamW(
        [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        default_group_warmup_steps=30,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    total_tokens_trained = 0
    spawn_events_log = []
    telemetry_records = []
    start_time = time.time()

    print(f"-> Model Initialized with {model.blocks[0].hyper_moe.num_experts} Bootstrap Experts / Layer")
    print("--- Beginning 16-Domain Training Run ---")

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_train_loss = 0.0
        current_domain = None

        # Gradient accumulation micro-batches
        for accum_idx in range(args.grad_accum_steps):
            # Round-robin cycle across all 16 knowledge domains
            domain_to_fetch = loader.domains[((step - 1) * args.grad_accum_steps + accum_idx) % len(loader.domains)]
            x, y, current_domain = loader.get_batch(split="train", domain=domain_to_fetch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Record experts before forward pass
            initial_experts_per_layer = [b.hyper_moe.num_experts for b in model.blocks]

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits, loss, telemetries = model(x, targets=y, allow_spawning=True)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            step_train_loss += loss.item() * args.grad_accum_steps

            # Check if autonomous spawning occurred in any layer
            for l_idx, b in enumerate(model.blocks):
                if b.hyper_moe.num_experts > initial_experts_per_layer[l_idx]:
                    new_exp_id = b.hyper_moe.num_experts - 1
                    spawn_event = {
                        "event": "autonomous_spawn",
                        "step": step,
                        "layer": l_idx,
                        "new_expert_id": new_exp_id,
                        "triggering_domain": current_domain,
                        "total_experts_in_layer": b.hyper_moe.num_experts,
                        "timestamp_epoch": time.time(),
                    }
                    spawn_events_log.append(spawn_event)
                    print(f"\n  ⚡ [AUTONOMOUS ON-THE-FLY SPAWN @ Step {step:4d} | Layer {l_idx}]", flush=True)
                    print(f"     -> Mathematical Trigger: Mean Max Phasor Resonance < tau_spawn ({args.spawn_threshold})", flush=True)
                    print(f"     -> Triggering Knowledge Domain Stream: '{current_domain.upper()}'", flush=True)
                    print(f"     -> Layer Total Experts: {b.hyper_moe.num_experts} (Spawned Expert #{new_exp_id})", flush=True)
                    print(f"     -> Initializing DynamicWarmupAdamW Group (30-step local warmup, v0 conditioned from warm variance)\n", flush=True)

                    # Attach newly created expert parameters to DynamicWarmupAdamW
                    new_exp_module = b.hyper_moe.experts[new_exp_id]
                    optimizer.add_dynamic_param_group(
                        list(new_exp_module.parameters()),
                        lr=args.lr,
                        weight_decay=0.01,
                        warmup_steps=30,
                        group_name=f"layer_{l_idx}_expert_{new_exp_id}",
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        # Gradient clipping and optimizer step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Cosine LR scheduler progression
        if step < 50:
            lr_mult = float(step) / 50.0
        else:
            progress = float(step - 50) / float(max(1, args.steps - 50))
            lr_mult = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        optimizer.update_dynamic_schedules(global_lr_multiplier=lr_mult)

        tokens_in_step = args.batch_size * args.grad_accum_steps * args.seq_len
        total_tokens_trained += tokens_in_step

        # Periodic logging and multi-domain validation
        is_eval_step = (step % args.eval_interval == 0 or step == 1 or step == args.steps)
        if is_eval_step:
            model.eval()
            val_losses = {}
            with torch.no_grad():
                for d in loader.domains[:8]: # Sample 8 domains for speed
                    vx, vy, _ = loader.get_batch(split="val", domain=d)
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        _, vloss, _ = model(vx, targets=vy, allow_spawning=False)
                    val_losses[d] = float(vloss.item())

            avg_val_loss = float(np.mean(list(val_losses.values())))
            val_ppl = math.exp(min(avg_val_loss, 20.0))
            last_val_loss = avg_val_loss
            last_val_ppl = val_ppl
        else:
            avg_val_loss = last_val_loss if 'last_val_loss' in locals() else step_train_loss
            val_ppl = last_val_ppl if 'last_val_ppl' in locals() else math.exp(min(step_train_loss, 20.0))

        elapsed = time.time() - start_time
        throughput = total_tokens_trained / max(1.0, elapsed)
        layer_experts = [b.hyper_moe.num_experts for b in model.blocks]

        telemetry_records.append({
            "step": step,
            "train_loss": step_train_loss,
            "avg_val_loss": avg_val_loss,
            "val_ppl": val_ppl,
            "tokens_sec": throughput,
            "experts_per_layer": layer_experts,
        })

        eval_marker = " [VAL EVAL]" if is_eval_step else ""
        print(f"Step {step:4d}/{args.steps:4d} | Train Loss: {step_train_loss:8.4f} | Val Loss: {avg_val_loss:7.4f} (PPL: {val_ppl:8.2f}) | Tok/s: {throughput:6.0f} | Experts/Layer: {layer_experts} | Spawns: {len(spawn_events_log)}{eval_marker}", flush=True)

    total_time = time.time() - start_time
    print("\n" + "=" * 90)
    print(f"  [TRAINING COMPLETE] Processed {total_tokens_trained:,d} tokens in {total_time:.2f}s ({total_tokens_trained/total_time:,.0f} tok/s)")
    print(f"  Total Autonomous Expert Spawns Logged: {len(spawn_events_log)}")
    print("=" * 90)

    # Save checkpoint & logs
    os.makedirs("experiments/checkpoints", exist_ok=True)
    os.makedirs("experiments/logs", exist_ok=True)

    ckpt_path = "experiments/checkpoints/hyperspace_16domain_autonomous.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "d_hyper": args.d_hyper,
            "top_k": args.top_k,
            "max_experts": args.max_experts,
            "spawn_threshold": args.spawn_threshold,
            "seq_len": args.seq_len,
        },
        "total_tokens_trained": total_tokens_trained,
        "spawn_events_log": spawn_events_log,
    }, ckpt_path)
    print(f"[SAVED] Model checkpoint saved to {ckpt_path}")

    log_path = "experiments/logs/hyperspace_16domain_training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_tokens_trained": total_tokens_trained,
            "total_time_seconds": total_time,
            "spawn_events_log": spawn_events_log,
            "telemetry_records": telemetry_records,
        }, f, indent=2)
    print(f"[SAVED] Training telemetry log saved to {log_path}")

    return ckpt_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--d_hyper", type=int, default=2048)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--spawn_threshold", type=float, default=0.20)
    parser.add_argument("--max_experts", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_16domain_production(args)

if __name__ == "__main__":
    main()
