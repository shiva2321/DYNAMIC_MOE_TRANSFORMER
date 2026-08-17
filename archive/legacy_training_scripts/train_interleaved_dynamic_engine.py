"""
Interleaved Multi-Domain Dynamic Engine Training & Domain-Discriminative Routing.
Features:
1. Interleaved Multi-Domain Streaming (Code, Web Reasoning, Math, Law in every step).
2. Contrastive Hyperspace Centroid Optimization in C^2048.
3. Clonal Mitosis Dynamic Spawning & Warm Parameter Registration.
4. Domain-by-Domain Generation & Expert Attribution Audit.
"""

import os
import sys
import time
import json
import random
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

class InterleavedDataStreamer:
    def __init__(self, cache_dir: str = "data/real_blend_cache", seq_len: int = 256, batch_size: int = 8):
        meta_file = os.path.join(cache_dir, "metadata_real_blend.json")
        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.domains = ["github_code", "fineweb_edu", "openweb_math", "freelaw_legal"]
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.train_shards = {d: np.memmap(self.metadata["domains"][d]["train_file"], dtype=np.uint16, mode='r') for d in self.domains}
        self.val_shards = {d: np.memmap(self.metadata["domains"][d]["val_file"], dtype=np.uint16, mode='r') for d in self.domains}

    def get_interleaved_batch(self, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        shards = self.train_shards if split == "train" else self.val_shards
        # Interleave samples across domains in each batch
        batch_x = []
        batch_y = []
        batch_domains = []
        
        samples_per_domain = max(1, self.batch_size // len(self.domains))
        for d in self.domains:
            data = shards[d]
            max_start = len(data) - self.seq_len - 1
            starts = np.random.randint(0, max_start, size=samples_per_domain)
            for s in starts:
                batch_x.append(data[s:s + self.seq_len].astype(np.int64))
                batch_y.append(data[s + 1:s + self.seq_len + 1].astype(np.int64))
                batch_domains.append(d)

        # Shuffle within batch
        combined = list(zip(batch_x, batch_y, batch_domains))
        random.shuffle(combined)
        bx, by, b_domains = zip(*combined)

        return torch.from_numpy(np.stack(bx)), torch.from_numpy(np.stack(by)), list(b_domains)

    def get_domain_val_batch(self, domain: str, num_samples: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.val_shards[domain]
        max_start = len(data) - self.seq_len - 1
        starts = np.random.randint(0, max_start, size=num_samples)
        x = np.stack([data[s:s + self.seq_len] for s in starts]).astype(np.int64)
        y = np.stack([data[s + 1:s + self.seq_len + 1] for s in starts]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

def run_interleaved_training(
    total_steps: int = 600,
    seq_len: int = 256,
    batch_size: int = 8,
    lr: float = 5e-4,
    out_dir: str = "experiments"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 95)
    print("  [UNIVERSAL SUBSTRAIT: INTERLEAVED MULTI-DOMAIN DYNAMIC ENGINE TRAINING]")
    print(f"  Architecture: 4 Layers, d_model=384, d_hyper=2048, SwiGLU Two-Compartment")
    print(f"  Batching: {batch_size} Interleaved Samples/Step across 4 Core Disciplines")
    print(f"  Features: Learnable Keys in C^2048 | Clonal Mitosis | Weight-Tied Head | Pure RoPE")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 95)

    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

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
        spawn_threshold=0.30,
        max_experts=16,
        initial_experts=2,
        use_sparse_attn=True,
        foveal_window=128,
        num_landmarks=4,
        num_sinks=4,
        max_seq_len=seq_len + 64,
        dropout=0.0,
    ).to(device)

    optimizer = DynamicWarmupAdamW(model.parameters(), lr=lr, weight_decay=0.01, default_group_warmup_steps=30)
    scaler = torch.amp.GradScaler('cuda')
    streamer = InterleavedDataStreamer(cache_dir="data/real_blend_cache", seq_len=seq_len, batch_size=batch_size)

    history = {
        "step": [],
        "train_loss": [],
        "domain_losses": {d: [] for d in streamer.domains},
        "domain_top1": {d: [] for d in streamer.domains},
        "total_experts": []
    }

    prev_expert_counts = [b.hyper_moe.num_experts for b in model.blocks]
    t_start = time.perf_counter()

    for step in range(1, total_steps + 1):
        # 1. Interleaved Batch Sampled Simultaneously Across All 4 Domains
        x, y, b_domains = streamer.get_interleaved_batch(split="train")
        x, y = x.to(device), y.to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            logits, loss, telemetries = model(x, targets=y, allow_spawning=True)

        scaler.scale(loss).backward()

        # Dynamic Spawning Registration
        curr_expert_counts = [b.hyper_moe.num_experts for b in model.blocks]
        if curr_expert_counts != prev_expert_counts:
            for l_idx, block in enumerate(model.blocks):
                if curr_expert_counts[l_idx] > prev_expert_counts[l_idx]:
                    new_exp_idx = curr_expert_counts[l_idx] - 1
                    new_exp = block.hyper_moe.experts[new_exp_idx]
                    key_r = block.hyper_moe.memory.keys_r[new_exp_idx]
                    key_i = block.hyper_moe.memory.keys_i[new_exp_idx]
                    
                    new_params = list(new_exp.parameters()) + [key_r, key_i]
                    optimizer.add_dynamic_param_group(new_params, lr=lr, warmup_steps=30, group_name=f"L{l_idx}_E{new_exp_idx}")
                    print(f"  🌱 [AUTONOMOUS CLONAL SPAWN @ STEP {step:3d}] Layer {l_idx} spawned Expert #{new_exp_idx}! Warm-registered in optimizer.")
            prev_expert_counts = curr_expert_counts

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Evaluation Every 30 Steps Across All 4 Domains Separately
        if step % 30 == 0 or step == 1:
            model.eval()
            dom_loss_report = {}
            total_exp = sum(b.hyper_moe.num_experts for b in model.blocks)

            with torch.no_grad():
                for d in streamer.domains:
                    vx, vy = streamer.get_domain_val_batch(d, num_samples=8)
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast('cuda'):
                        val_logits, val_loss, _ = model(vx, targets=vy, allow_spawning=False)
                    preds = torch.argmax(val_logits, dim=-1)
                    acc1 = (preds == vy).float().mean().item() * 100.0
                    
                    history["domain_losses"][d].append(val_loss.item())
                    history["domain_top1"][d].append(acc1)
                    dom_loss_report[d] = (val_loss.item(), acc1)

            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["total_experts"].append(total_exp)

            code_l, code_a = dom_loss_report["github_code"]
            law_l, law_a = dom_loss_report["freelaw_legal"]
            math_l, math_a = dom_loss_report["openweb_math"]
            web_l, web_a = dom_loss_report["fineweb_edu"]

            print(f"Step {step:3d}/{total_steps} | Exp: {total_exp:2d} | Code: {code_l:.3f} ({code_a:4.1f}%) | Law: {law_l:.3f} ({law_a:4.1f}%) | Math: {math_l:.3f} ({math_a:4.1f}%) | Web: {web_l:.3f} ({web_a:4.1f}%)")

        # Domain Discriminative Generation Check at Steps 200, 400, 600
        if step % 200 == 0:
            print("\n" + "=" * 80)
            print(f"  [DOMAIN-DISCRIMINATIVE GENERATION AUDIT @ STEP {step}]")
            print("=" * 80)
            
            audit_prompts = [
                ("Code", "def quick_sort(arr):\n    \"\"\"Sorts array using quicksort algorithm.\"\"\"\n"),
                ("Law", "Section 4.02 Indemnification by Seller. Seller agrees to indemnify and hold harmless Buyer from"),
                ("Math", "Let V be a finite dimensional vector space. Then the dimension of V satisfies")
            ]

            for dom_tag, p_str in audit_prompts:
                p_toks = enc.encode(p_str)
                curr = torch.tensor([p_toks], dtype=torch.long, device=device)
                gen_toks = []
                for _ in range(35):
                    with torch.no_grad():
                        with torch.amp.autocast('cuda'):
                            l_out, _, telem = model(curr, allow_spawning=False)
                    nxt = torch.argmax(l_out[0, -1, :vocab_size]).item()
                    gen_toks.append(nxt)
                    curr = torch.cat([curr, torch.tensor([[nxt]], device=device)], dim=1)
                
                # Check top expert in Layer 2
                top_exp_l2 = telem[2].get("top_indices", None)
                exp_str = f"L2 Top Experts: {top_exp_l2[0, -1].cpu().numpy().tolist()}" if top_exp_l2 is not None else ""
                
                print(f"\n--- [{dom_tag} Prompt] --- ({exp_str})")
                print(f"{p_str.strip()} {enc.decode(gen_toks).strip()}")

            print("\n" + "=" * 80 + "\n")

    total_time = time.perf_counter() - t_start
    print(f"\n[TRAINING COMPLETE] Processed {total_steps * batch_size * seq_len:,} Tokens in {total_time:.2f}s ({((total_steps * batch_size * seq_len)/total_time):.0f} tok/s)")

    # Save Final Checkpoint
    ckpt_out = os.path.join(out_dir, "checkpoints", "hyperspace_interleaved_dynamic_engine.pt")
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

    # Plot Multi-Domain Convergence Curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # 1. Multi-Domain Loss Descent
    for d, c in zip(streamer.domains, ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']):
        axes[0].plot(history["step"], history["domain_losses"][d], label=f"{d}", color=c, linewidth=2.0)
    axes[0].set_xlabel("Interleaved Training Step", fontweight='bold')
    axes[0].set_ylabel("Validation Loss", fontweight='bold')
    axes[0].set_title("Balanced Multi-Domain Cross-Entropy Descent", fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Multi-Domain Top-1 Accuracy
    for d, c in zip(streamer.domains, ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']):
        axes[1].plot(history["step"], history["domain_top1"][d], label=f"{d}", color=c, linewidth=2.0)
    axes[1].set_xlabel("Interleaved Training Step", fontweight='bold')
    axes[1].set_ylabel("Top-1 Token Accuracy (%)", fontweight='bold')
    axes[1].set_title("Balanced Multi-Domain Token Accuracy", fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_out = os.path.join(out_dir, "plots", "interleaved_dynamic_engine_evaluation.png")
    plt.savefig(plot_out)
    plt.close()
    print(f"[SAVED] Multi-Domain Evaluation plot saved to: {plot_out}")

    # Copy to brain artifact directory
    brain_dir = r"C:\Users\Asta\.gemini\antigravity-ide\brain\0f00f3a4-e432-4a06-8006-7dfc2fa48c7a"
    import shutil
    shutil.copy(plot_out, os.path.join(brain_dir, "interleaved_dynamic_engine_evaluation.png"))

if __name__ == "__main__":
    run_interleaved_training(total_steps=600, seq_len=256, batch_size=8, lr=5e-4)
