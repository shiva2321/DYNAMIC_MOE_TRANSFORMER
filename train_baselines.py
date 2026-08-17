"""
Canonical Baseline Training Pipeline: Matched-Budget Comparison on Identical Corpus.
Trains:
1. DenseTransformerLM (Dense baseline with weight-tied head & RoPE)
2. StaticSoftmaxMoELM (Static Top-2 Softmax MoE baseline with 16 fixed experts)
on the exact same 11.4M-token HuggingFace corpus (data/scaled_real_corpus).
"""

import os
import sys
import time
import json
import math
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.baselines import DenseTransformerLM, StaticSoftmaxMoELM
from train_scaled_production_engine import ScaledProductionDataStreamer

def train_baseline_model(
    model_type: str = "dense", # "dense" or "static_moe"
    total_steps: int = 1500,
    seq_len: int = 256,
    micro_batch_size: int = 4,
    accum_steps: int = 3,
    max_lr: float = 6e-4,
    min_lr: float = 6e-5,
    warmup_steps: int = 200,
    out_dir: str = "experiments"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 95)
    print(f"  [UNIVERSAL SUBSTRAIT: CANONICAL BASELINE TRAINING -> {model_type.upper()}]")
    print(f"  Dataset: 11,402,565 Real Tokens (Matched Budget: {total_steps * micro_batch_size * accum_steps * seq_len:,} Tokens)")
    print(f"  Architecture: 4 Layers, d_model=384, Weight-Tied Head, Pure RoPE")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 95)

    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    vocab_size = 50304
    d_model = 384
    n_layers = 4
    n_heads = 6
    d_ff = 768

    if model_type == "dense":
        model = DenseTransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff * 2, # Matched compute capacity
            max_seq_len=seq_len + 64
        ).to(device)
    elif model_type == "static_moe":
        model = StaticSoftmaxMoELM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            num_experts=16,
            top_k=2,
            max_seq_len=seq_len + 64
        ).to(device)
    else:
        raise ValueError(f"Unknown baseline model_type: {model_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda')
    streamer = ScaledProductionDataStreamer(cache_dir="data/scaled_real_corpus", seq_len=seq_len, batch_size=micro_batch_size)

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    history = {
        "step": [],
        "train_loss": [],
        "domain_losses": {d: [] for d in streamer.domains},
        "domain_top1": {d: [] for d in streamer.domains},
    }

    t_start = time.perf_counter()
    tokens_processed = 0

    for step in range(1, total_steps + 1):
        curr_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr

        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(accum_steps):
            x, y, _ = streamer.get_interleaved_batch(split="train")
            x, y = x.to(device), y.to(device)
            tokens_processed += (micro_batch_size * seq_len)

            with torch.amp.autocast('cuda'):
                logits, loss, _ = model(x, targets=y)
                loss = loss / accum_steps

            accum_loss += loss.item() * accum_steps
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Validation Every 100 Steps
        if step % 100 == 0 or step == 1:
            model.eval()
            dom_loss_report = {}
            with torch.no_grad():
                for d in streamer.domains:
                    vx, vy = streamer.get_domain_val_batch(d, num_samples=8)
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast('cuda'):
                        val_logits, val_loss, _ = model(vx, targets=vy)
                    preds = torch.argmax(val_logits, dim=-1)
                    acc1 = (preds == vy).float().mean().item() * 100.0
                    history["domain_losses"][d].append(val_loss.item())
                    history["domain_top1"][d].append(acc1)
                    dom_loss_report[d] = (val_loss.item(), acc1)

            history["step"].append(step)
            history["train_loss"].append(accum_loss)
            
            fine_l, fine_a1 = dom_loss_report["fineweb_edu"]
            code_l, code_a1 = dom_loss_report["python_code"]
            wiki_l, wiki_a1 = dom_loss_report["wikitext_facts"]
            story_l, story_a1 = dom_loss_report["natural_stories"]
            
            elapsed = time.perf_counter() - t_start
            tok_s = tokens_processed / max(0.1, elapsed)
            print(f"Step {step:4d}/{total_steps} | LR: {curr_lr:.1e} | FineWeb: {fine_l:.3f} ({fine_a1:4.1f}%) | Code: {code_l:.3f} ({code_a1:4.1f}%) | Wiki: {wiki_l:.3f} ({wiki_a1:4.1f}%) | Story: {story_l:.3f} ({story_a1:4.1f}%) | Speed: {tok_s:.0f} tok/s")

    # Save Baseline Checkpoint
    ckpt_path = os.path.join(out_dir, "checkpoints", f"canonical_baseline_{model_type}.pt")
    torch.save({
        "model_type": model_type,
        "model_state": model.state_dict(),
        "config": {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "vocab_size": vocab_size
        },
        "history": history
    }, ckpt_path)
    print(f"[SAVED] Baseline {model_type.upper()} checkpoint saved to: {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonical Baseline Trainer")
    parser.add_argument("--model", type=str, choices=["dense", "static_moe"], default="dense")
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    train_baseline_model(model_type=args.model, total_steps=args.steps)
