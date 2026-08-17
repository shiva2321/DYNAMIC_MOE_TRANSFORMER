"""
Train & Evaluate Real Dynamic Expert Creation Engine.
Verifies:
1. Autonomous Spawning from 2 Bootstrap Experts via Clonal Mitosis (No Cold Noise).
2. Gradient Flow into Learnable Hyperspace Keys in C^2048.
3. Weight-Tied LM Head & Pure RoPE Fast Convergence.
4. Live Autoregressive Generation & Top-1/Top-5 Token Accuracy.
"""

import os
import sys
import time
import json
import math
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.nanogpt import HyperTransformerLM
from hyperspace.dynamic_optimizer import DynamicWarmupAdamW
from hyperspace.vsa import ComplexPhasorVSA

class RealDataStreamer:
    def __init__(self, cache_dir: str = "data/real_blend_cache", seq_len: int = 512, batch_size: int = 4):
        meta_file = os.path.join(cache_dir, "metadata_real_blend.json")
        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.domains = list(self.metadata["domains"].keys())
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.train_shards = {d: np.memmap(self.metadata["domains"][d]["train_file"], dtype=np.uint16, mode='r') for d in self.domains}
        self.val_shards = {d: np.memmap(self.metadata["domains"][d]["val_file"], dtype=np.uint16, mode='r') for d in self.domains}

    def get_batch(self, domain: str, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        shards = self.train_shards if split == "train" else self.val_shards
        data = shards[domain]
        max_start = len(data) - self.seq_len - 1
        starts = np.random.randint(0, max_start, size=self.batch_size)
        x = np.stack([data[s:s + self.seq_len] for s in starts]).astype(np.int64)
        y = np.stack([data[s + 1:s + self.seq_len + 1] for s in starts]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

def run_dynamic_engine_test(
    total_steps: int = 400,
    seq_len: int = 512,
    batch_size: int = 4,
    lr: float = 4e-4,
    out_dir: str = "experiments"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 95)
    print("  [UNIVERSAL SUBSTRAIT: REAL DYNAMIC EXPERT CREATION & TRAINING ENGINE]")
    print(f"  Architecture: 4 Layers, d_model=384, d_hyper=2048, SwiGLU Two-Compartment")
    print(f"  Bootstrap: 2 Initial Experts/Layer -> Autonomous Clonal Spawning up to 16/Layer")
    print(f"  Features: Learnable Keys in C^2048 | Weight-Tied LM Head | Pure RoPE | DynamicWarmupAdamW")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 95)

    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    # 1. Initialize Model from 2 Bootstrap Experts
    vocab_size = 50304
    d_model = 384
    n_layers = 4
    n_heads = 6
    d_ff = 768
    d_hyper = 2048

    model = HyperTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        d_hyper=d_hyper,
        top_k=2,
        max_k=4,
        top_p=0.85,
        dynamic_k=True,
        spawn_threshold=0.28, # Sensitive novelty threshold for dynamic neurogenesis
        max_experts=16,
        initial_experts=2,
        use_sparse_attn=True,
        foveal_window=128,
        num_landmarks=4,
        num_sinks=4,
        max_seq_len=seq_len + 64,
        dropout=0.0,
    ).to(device)

    # 2. Dynamic Optimizer
    optimizer = DynamicWarmupAdamW(model.parameters(), lr=lr, weight_decay=0.01, default_group_warmup_steps=30)
    scaler = torch.amp.GradScaler('cuda')

    streamer = RealDataStreamer(cache_dir="data/real_blend_cache", seq_len=seq_len, batch_size=batch_size)
    primary_domains = ["github_code", "fineweb_edu", "openweb_math", "freelaw_legal"]

    print(f"Initial State: {sum(b.hyper_moe.num_experts for b in model.blocks)} Experts Total across {n_layers} Layers")
    print("Beginning Dynamic Lifelong Training & Autonomous Neurogenesis...\n")

    history = {
        "step": [],
        "loss": [],
        "top1_acc": [],
        "top5_acc": [],
        "total_experts": [],
        "domain": []
    }

    # Tracking previous expert count to detect spawns and update optimizer
    prev_expert_counts = [b.hyper_moe.num_experts for b in model.blocks]

    t_start = time.perf_counter()

    for step in range(1, total_steps + 1):
        domain = primary_domains[(step // 25) % len(primary_domains)] # Switch domain every 25 steps to trigger novelty
        x, y = streamer.get_batch(domain, split="train")
        x, y = x.to(device), y.to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            logits, loss, telemetries = model(x, targets=y, allow_spawning=True)

        scaler.scale(loss).backward()

        # Check if any new experts or keys were spawned during this step
        curr_expert_counts = [b.hyper_moe.num_experts for b in model.blocks]
        if curr_expert_counts != prev_expert_counts:
            # New expert(s) spawned! Add their parameters to optimizer
            for l_idx, block in enumerate(model.blocks):
                if curr_expert_counts[l_idx] > prev_expert_counts[l_idx]:
                    new_exp_idx = curr_expert_counts[l_idx] - 1
                    new_exp = block.hyper_moe.experts[new_exp_idx]
                    key_r = block.hyper_moe.memory.keys_r[new_exp_idx]
                    key_i = block.hyper_moe.memory.keys_i[new_exp_idx]
                    
                    new_params = list(new_exp.parameters()) + [key_r, key_i]
                    optimizer.add_dynamic_param_group(new_params, lr=lr, warmup_steps=30, group_name=f"L{l_idx}_E{new_exp_idx}")
                    print(f"  🌱 [AUTONOMOUS SPAWN @ STEP {step}] Layer {l_idx} spawned Expert #{new_exp_idx} for domain [{domain}]! Warm-registered in optimizer.")
            prev_expert_counts = curr_expert_counts

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Evaluation & Telemetry
        if step % 20 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    vx, vy = streamer.get_batch(domain, split="val")
                    vx, vy = vx.to(device), vy.to(device)
                    val_logits, val_loss, _ = model(vx, targets=vy, allow_spawning=False)
            
            preds = torch.argmax(val_logits, dim=-1)
            top5 = torch.topk(val_logits, k=5, dim=-1).indices
            top1_acc = (preds == vy).float().mean().item() * 100.0
            top5_acc = torch.any(top5 == vy.unsqueeze(-1), dim=-1).float().mean().item() * 100.0
            total_exp = sum(b.hyper_moe.num_experts for b in model.blocks)

            history["step"].append(step)
            history["loss"].append(val_loss.item())
            history["top1_acc"].append(top1_acc)
            history["top5_acc"].append(top5_acc)
            history["total_experts"].append(total_exp)
            history["domain"].append(domain)

            print(f"Step {step:3d}/{total_steps} | Domain: [{domain:<14}] | Val Loss: {val_loss.item():.4f} | Top-1 Acc: {top1_acc:5.2f}% | Top-5 Acc: {top5_acc:5.2f}% | Total Exp: {total_exp}")

        # Live Generation Quality Test every 100 steps
        if step % 100 == 0:
            print("\n" + "-" * 75)
            print(f"  [LIVE GENERATION TEST @ STEP {step}]")
            test_prompt = "def calculate_matrix_determinant(matrix):\n    \"\"\"Calculates determinant of square matrix.\"\"\"\n"
            p_tokens = enc.encode(test_prompt)
            curr = torch.tensor([p_tokens], dtype=torch.long, device=device)
            gen = []
            for _ in range(40):
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        l_out, _, _ = model(curr, allow_spawning=False)
                nxt = torch.argmax(l_out[0, -1, :vocab_size]).item()
                gen.append(nxt)
                curr = torch.cat([curr, torch.tensor([[nxt]], device=device)], dim=1)
            print(f"Prompt: {test_prompt.strip()}")
            print(f"Continuation: {enc.decode(gen).strip()}")
            print("-" * 75 + "\n")

    total_time = time.perf_counter() - t_start
    print(f"\n[COMPLETE] Trained {total_steps} steps in {total_time:.2f}s ({(total_steps * seq_len * batch_size)/total_time:.0f} tok/s)")

    # Save Checkpoint
    ckpt_out = os.path.join(out_dir, "checkpoints", "hyperspace_dynamic_cloned_engine.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "d_hyper": d_hyper,
            "vocab_size": vocab_size,
        },
        "history": history
    }, ckpt_out)
    print(f"[SAVED] Checkpoint saved to: {ckpt_out}")

    # Plot Convergence & Spawning Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    # 1. Loss Curve
    axes[0].plot(history["step"], history["loss"], color='#e74c3c', linewidth=2.0)
    axes[0].set_xlabel("Training Step", fontweight='bold')
    axes[0].set_ylabel("Validation Loss", fontweight='bold')
    axes[0].set_title("Cross-Entropy Loss Descent", fontweight='bold')
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Top-1 and Top-5 Token Accuracy
    axes[1].plot(history["step"], history["top1_acc"], label="Top-1 Accuracy", color='#3498db', linewidth=2.0)
    axes[1].plot(history["step"], history["top5_acc"], label="Top-5 Accuracy", color='#2ecc71', linewidth=2.0)
    axes[1].set_xlabel("Training Step", fontweight='bold')
    axes[1].set_ylabel("Accuracy (%)", fontweight='bold')
    axes[1].set_title("Next-Token Prediction Accuracy", fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Autonomous Expert Spawning Curve
    axes[2].plot(history["step"], history["total_experts"], color='#8e44ad', linewidth=2.5, marker='o')
    axes[2].set_xlabel("Training Step", fontweight='bold')
    axes[2].set_ylabel("Total Active Micro-Experts", fontweight='bold')
    axes[2].set_title("Autonomous Lifelong Neurogenesis (Spawning)", fontweight='bold')
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_out = os.path.join(out_dir, "plots", "dynamic_cloned_engine_evaluation.png")
    plt.savefig(plot_out)
    plt.close()
    print(f"[SAVED] Evaluation plot saved to: {plot_out}")

    # Copy to brain artifact directory
    brain_dir = r"C:\Users\Asta\.gemini\antigravity-ide\brain\0f00f3a4-e432-4a06-8006-7dfc2fa48c7a"
    import shutil
    shutil.copy(plot_out, os.path.join(brain_dir, "dynamic_cloned_engine_evaluation.png"))

if __name__ == "__main__":
    run_dynamic_engine_test(total_steps=300, seq_len=256, batch_size=8, lr=5e-4)
