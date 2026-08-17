"""
Long-Context Real-World Multi-Domain Training Engine (Option A).
Integrates:
1. Dynamic & Sparse Multi-Head Attention (HDSA) with RoPE (seq_len = 1024 to 2048).
2. Dynamic Hyperspace 2.0 MoE with Continuous Complex Phasor Projections (C^2048).
3. Autonomous Centroid-Seeded Novelty Spawning across 8 Real-World Knowledge Domains.
4. Two-Compartment Pyramidal Soma-Dendrite Micro-Experts with Global Workspace Bus.
5. Context-Conditioned Dynamic-k Routing with Shannon Entropy Modulation.
6. Constant-Memory O(N_active) Training with DynamicWarmupAdamW.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from model.nanogpt import HyperTransformerLM
from hyperspace.dynamic_optimizer import DynamicWarmupAdamW

class RealBlendMemmapLoader:
    def __init__(self, cache_dir: str = "data/real_blend_cache", seq_len: int = 1024, batch_size: int = 4):
        meta_file = os.path.join(cache_dir, "metadata_real_blend.json")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")
            
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

def train_real_multidomain_longcontext(
    total_steps: int = 100,
    seq_len: int = 1024,
    micro_batch_size: int = 1,
    grad_accum_steps: int = 4,
    cache_dir: str = "data/real_blend_cache",
    out_dir: str = "experiments"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_batch_size = micro_batch_size * grad_accum_steps
    tokens_per_step = seq_len * effective_batch_size
    print("\n" + "=" * 95)
    print("  [UNIVERSAL SUBSTRAIT (HYPERSPACE 2.0): LONG-CONTEXT REAL-WORLD MULTI-DOMAIN TRAINING]")
    print(f"  Sequence Length: {seq_len} Tokens (HDSA Dynamic Sparse Attention with RoPE)")
    print(f"  Micro-Batch: {micro_batch_size} x {grad_accum_steps} Accumulation = Effective Batch {effective_batch_size} ({tokens_per_step:,} Tokens/Step)")
    print(f"  Target Training Steps: {total_steps} (~{total_steps * tokens_per_step / 1e6:.2f} Million Tokens)")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 95)

    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    loader = RealBlendMemmapLoader(cache_dir=cache_dir, seq_len=seq_len, batch_size=micro_batch_size)
    domains = loader.domains
    print(f"Loaded {len(domains)} Real-World Domains: {', '.join(domains)}")

    # Model Configuration
    vocab_size = loader.metadata.get("vocab_size", 50304)
    d_model = 384
    n_layers = 6
    n_heads = 6
    d_ff = 768
    d_hyper = 2048
    initial_experts = 2
    max_experts = 32

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
        spawn_threshold=0.35,
        max_experts=max_experts,
        initial_experts=initial_experts,
        use_sparse_attn=True,
        foveal_window=128,
        num_landmarks=4,
        num_sinks=4,
        max_seq_len=seq_len + 64,
        dropout=0.0,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Initialized: {total_params:,} Parameters ({total_params/1e6:.2f}M)")

    optimizer = DynamicWarmupAdamW(model.parameters(), lr=1e-3, default_group_warmup_steps=40)
    scaler = torch.amp.GradScaler('cuda')

    # Training Tracking
    steps_per_domain = max(1, total_steps // len(domains))
    history = {
        "step": [],
        "domain": [],
        "train_loss": [],
        "val_loss": [],
        "perplexity": [],
        "total_experts": [],
        "mean_k": [],
        "tokens_per_sec": []
    }

    step_idx = 0
    t_start = time.perf_counter()

    for d_idx, domain in enumerate(domains):
        domain_title = loader.metadata["domains"][domain]["title"]
        print(f"\n[{d_idx+1}/{len(domains)}] >>> Streaming Real-World Domain: [{domain_title}] ({steps_per_domain} Steps)...")
        
        for local_step in range(steps_per_domain):
            step_idx += 1
            t0 = time.perf_counter()
            model.train()

            # Dynamic learning rate decay per step
            progress = min(1.0, step_idx / total_steps)
            curr_lr = 1e-4 + 0.5 * (1e-3 - 1e-4) * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = curr_lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            last_telemetries = []

            for accum_i in range(grad_accum_steps):
                x_batch, y_batch = loader.get_batch(domain, split="train")
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                with torch.amp.autocast('cuda'):
                    logits, loss, telemetries = model(x_batch, targets=y_batch, allow_spawning=True)
                    loss = loss / grad_accum_steps

                scaler.scale(loss).backward()
                accum_loss += loss.item() * grad_accum_steps
                last_telemetries = telemetries

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            dur = time.perf_counter() - t0
            tok_s = tokens_per_step / max(1e-5, dur)

            # Compute telemetry
            total_active_exp = sum(b.hyper_moe.num_experts for b in model.blocks)
            mean_k_step = np.mean([t.get("mean_active_k", 2.0) for t in last_telemetries]) if last_telemetries else 2.0

            if step_idx % 5 == 0 or step_idx == 1:
                # Validation Loss on current domain
                model.eval()
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        vx, vy = loader.get_batch(domain, split="val")
                        vx, vy = vx.to(device), vy.to(device)
                        _, val_loss, _ = model(vx, targets=vy, allow_spawning=False)

                val_loss_val = val_loss.item()
                ppl = math.exp(min(20.0, val_loss_val))

                history["step"].append(step_idx)
                history["domain"].append(domain)
                history["train_loss"].append(accum_loss)
                history["val_loss"].append(val_loss_val)
                history["perplexity"].append(ppl)
                history["total_experts"].append(total_active_exp)
                history["mean_k"].append(mean_k_step)
                history["tokens_per_sec"].append(tok_s)

                print(f"  Step {step_idx:3d}/{total_steps:3d} | Train: {accum_loss:.4f} | Val: {val_loss_val:.4f} | PPL: {ppl:8.2f} | Exp/L: {total_active_exp//n_layers:2d} ({total_active_exp:3d} Total) | k*: {mean_k_step:.2f} | {tok_s:5.0f} tok/s")

    total_time = time.perf_counter() - t_start
    total_tokens_trained = total_steps * seq_len * effective_batch_size

    print("\n" + "=" * 95)
    print(f"  [TRAINING COMPLETE] Processed {total_tokens_trained:,} Tokens in {total_time/60:.2f} Minutes ({total_tokens_trained/total_time:.0f} Tokens/Sec)")
    print(f"  Final Multi-Domain Perplexity: {history['perplexity'][-1]:.2f}")
    print(f"  Final Active Micro-Experts: {history['total_experts'][-1]} Total across {n_layers} Layers")
    print("=" * 95)

    # Save Checkpoint
    ckpt_path = os.path.join(out_dir, "checkpoints", "hyperspace_real_multidomain_longcontext.pt")
    torch.save({
        "config": {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "d_hyper": d_hyper,
            "max_seq_len": seq_len + 64,
            "use_sparse_attn": True,
            "dynamic_k": True,
            "top_k": 2,
            "max_k": 4,
            "num_experts": history['total_experts'][-1] // n_layers
        },
        "model_state": model.state_dict(),
        "history": history
    }, ckpt_path)
    print(f"[SAVED] Checkpoint saved to: {ckpt_path}")

    # Generate Learning Curves Plot
    generate_longcontext_plots(history, out_dir)
    generate_training_report(history, total_tokens_trained, total_time, out_dir)

def generate_longcontext_plots(history: Dict[str, Any], out_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)

    # 1. Validation Loss & Train Loss
    ax1 = axes[0, 0]
    ax1.plot(history["step"], history["train_loss"], label="Train Loss", color="#1f77b4", alpha=0.6, linestyle="--")
    ax1.plot(history["step"], history["val_loss"], label="Validation Loss", color="#d62728", linewidth=2.5)
    ax1.set_xlabel("Training Steps", fontweight='bold')
    ax1.set_ylabel("Cross-Entropy Loss", fontweight='bold')
    ax1.set_title("Multi-Domain Loss Convergence (S = 1024 HDSA)", fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Perplexity Drop
    ax2 = axes[0, 1]
    ax2.plot(history["step"], history["perplexity"], color="#2ca02c", linewidth=2.5)
    ax2.set_yscale("log")
    ax2.set_xlabel("Training Steps", fontweight='bold')
    ax2.set_ylabel("Perplexity (Log Scale)", fontweight='bold')
    ax2.set_title("Validation Perplexity Reduction", fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. Autonomous Expert Spawning Dynamics
    ax3 = axes[1, 0]
    ax3.plot(history["step"], history["total_experts"], color="#9467bd", linewidth=2.5)
    ax3.set_xlabel("Training Steps", fontweight='bold')
    ax3.set_ylabel("Total Active Micro-Experts", fontweight='bold')
    ax3.set_title("Autonomous On-The-Fly Expert Growth across 8 Domains", fontweight='bold')
    ax3.grid(True, linestyle="--", alpha=0.6)

    # 4. Context-Conditioned Dynamic k* Distribution
    ax4 = axes[1, 1]
    ax4.plot(history["step"], history["mean_k"], color="#ff7f0e", linewidth=2.5)
    ax4.set_xlabel("Training Steps", fontweight='bold')
    ax4.set_ylabel("Mean Recruited Experts (k*)", fontweight='bold')
    ax4.set_title("Dynamic-k Recruitment Aperture vs Context", fontweight='bold')
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_fig = os.path.join(out_dir, "plots", "real_multidomain_longcontext_training.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"[SAVED] Training plot saved to: {out_fig}")

def generate_training_report(history: Dict[str, Any], total_tokens: int, duration: float, out_dir: str):
    out_md = os.path.join(out_dir, "real_multidomain_longcontext_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Real-World Multi-Domain Long-Context Training Report (Option A)\n\n")
        f.write("**Architecture**: Universal Substrait (Hyperspace 2.0) with HDSA Dynamic Sparse Attention\n")
        f.write(f"**Sequence Length**: $S = 1024$ tokens  \n")
        f.write(f"**Total Tokens Trained**: `{total_tokens:,}` Tokens ({total_tokens/1e6:.2f}M) in `{duration/60:.2f}` Minutes  \n")
        f.write(f"**Average Throughput**: `{total_tokens/duration:.0f}` Tokens / Sec  \n\n")
        f.write("---\n\n")
        f.write("## 1. Key Training Metrics\n\n")
        f.write(f"* **Initial Validation Loss**: `{history['val_loss'][0]:.4f}` (Perplexity: `{history['perplexity'][0]:.2f}`)\n")
        f.write(f"* **Final Validation Loss**: `{history['val_loss'][-1]:.4f}` (Perplexity: `{history['perplexity'][-1]:.2f}`)\n")
        f.write(f"* **Initial Experts**: `{history['total_experts'][0]}` $\\to$ **Final Active Experts**: `{history['total_experts'][-1]}` Total Experts\n")
        f.write(f"* **Mean Dynamic-$k^*$**: `{history['mean_k'][-1]:.2f}` Experts / Token\n\n")
        f.write("---\n\n")
        f.write("### Training Progression Visualization\n")
        f.write("![Long Context Training](plots/real_multidomain_longcontext_training.png)\n")

    print(f"[SAVED] Training report saved to: {out_md}")

if __name__ == "__main__":
    train_real_multidomain_longcontext(total_steps=100, seq_len=1024, micro_batch_size=1, grad_accum_steps=4)
