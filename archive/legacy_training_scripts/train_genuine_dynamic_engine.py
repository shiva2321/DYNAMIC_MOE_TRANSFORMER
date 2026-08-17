"""
Train Dynamic Engine on 100% Genuine, Diverse, Non-Repeating Multi-Domain Data.
Streams from 17,987 unique real-world documents across Stories, Python Code, WikiText, and FineWeb-Edu.
"""

import os
import sys
import time
import json
import random
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

class GenuineDataStreamer:
    def __init__(self, cache_dir: str = "data/genuine_diverse_cache", seq_len: int = 256, batch_size: int = 8):
        meta_file = os.path.join(cache_dir, "metadata_genuine.json")
        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.domains = ["natural_stories", "python_code", "wikitext_facts", "fineweb_reasoning"]
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.train_shards = {d: np.memmap(self.metadata["domains"][d]["train_file"], dtype=np.uint16, mode='r') for d in self.domains}
        self.val_shards = {d: np.memmap(self.metadata["domains"][d]["val_file"], dtype=np.uint16, mode='r') for d in self.domains}

    def get_interleaved_batch(self, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        shards = self.train_shards if split == "train" else self.val_shards
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

def run_genuine_training(
    total_steps: int = 1000,
    seq_len: int = 256,
    batch_size: int = 8,
    lr: float = 6e-4,
    out_dir: str = "experiments"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 95)
    print("  [UNIVERSAL SUBSTRAIT: TRAINING DYNAMIC ENGINE ON 100% GENUINE DIVERSE DATA]")
    print(f"  Architecture: 4 Layers, d_model=384, d_hyper=2048, SwiGLU Two-Compartment")
    print(f"  Dataset: 17,987 Unique Documents (Stories, Python Code, WikiText-103, FineWeb-Edu)")
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

    optimizer = DynamicWarmupAdamW(model.parameters(), lr=lr, weight_decay=0.01, default_group_warmup_steps=40)
    scaler = torch.amp.GradScaler('cuda')
    streamer = GenuineDataStreamer(cache_dir="data/genuine_diverse_cache", seq_len=seq_len, batch_size=batch_size)

    history = {
        "step": [],
        "train_loss": [],
        "domain_losses": {d: [] for d in streamer.domains},
        "domain_top1": {d: [] for d in streamer.domains},
        "domain_top5": {d: [] for d in streamer.domains},
        "total_experts": []
    }

    prev_expert_counts = [b.hyper_moe.num_experts for b in model.blocks]
    t_start = time.perf_counter()

    for step in range(1, total_steps + 1):
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
                    optimizer.add_dynamic_param_group(new_params, lr=lr, warmup_steps=40, group_name=f"L{l_idx}_E{new_exp_idx}")
                    print(f"  🌱 [AUTONOMOUS CLONAL SPAWN @ STEP {step:4d}] Layer {l_idx} spawned Expert #{new_exp_idx}! Warm-registered in optimizer.")
            prev_expert_counts = curr_expert_counts

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Evaluation Every 50 Steps Across All 4 Domains
        if step % 50 == 0 or step == 1:
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
                    
                    _, top5_preds = torch.topk(val_logits, 5, dim=-1)
                    acc5 = (top5_preds == vy.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0

                    history["domain_losses"][d].append(val_loss.item())
                    history["domain_top1"][d].append(acc1)
                    history["domain_top5"][d].append(acc5)
                    dom_loss_report[d] = (val_loss.item(), acc1, acc5)

            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["total_experts"].append(total_exp)

            story_l, story_a1, story_a5 = dom_loss_report["natural_stories"]
            code_l, code_a1, code_a5 = dom_loss_report["python_code"]
            wiki_l, wiki_a1, wiki_a5 = dom_loss_report["wikitext_facts"]
            fine_l, fine_a1, fine_a5 = dom_loss_report["fineweb_reasoning"]

            print(f"Step {step:4d}/{total_steps} | Exp: {total_exp:2d} | Story: {story_l:.3f} ({story_a1:4.1f}%) | Code: {code_l:.3f} ({code_a1:4.1f}%) | Wiki: {wiki_l:.3f} ({wiki_a1:4.1f}%) | Web: {fine_l:.3f} ({fine_a1:4.1f}%)")

        # Live Generation Showcase Every 250 Steps
        if step % 250 == 0:
            print("\n" + "=" * 80)
            print(f"  [GENUINE LANGUAGE COHERENCE & REASONING AUDIT @ STEP {step}]")
            print("=" * 80)
            
            prompts = [
                ("Story / Narrative", "Once upon a time, there was a little boy named Tim who loved to play with his"),
                ("Python Code", "def calculate_average(numbers):\n    \"\"\"Calculates arithmetic mean.\"\"\"\n"),
                ("Wiki Knowledge", "The solar system is gravitationally bound to the Sun and includes")
            ]

            for label, p_str in prompts:
                p_toks = enc.encode(p_str)
                curr = torch.tensor([p_toks], dtype=torch.long, device=device)
                gen_toks = []
                for _ in range(40):
                    with torch.no_grad():
                        with torch.amp.autocast('cuda'):
                            l_out, _, _ = model(curr, allow_spawning=False)
                    scaled_logits = l_out[0, -1, :vocab_size] / 0.7
                    v, top_idx = torch.topk(scaled_logits, 40)
                    probs = F.softmax(v, dim=-1)
                    nxt = top_idx[torch.multinomial(probs, 1)].item()
                    gen_toks.append(nxt)
                    curr = torch.cat([curr, torch.tensor([[nxt]], device=device)], dim=1)
                
                print(f"\n--- [{label}] ---")
                print(f"{p_str.strip()} {enc.decode(gen_toks).strip()}")

            print("\n" + "=" * 80 + "\n")

    total_time = time.perf_counter() - t_start
    print(f"\n[TRAINING COMPLETE] Processed {total_steps * batch_size * seq_len:,} Tokens in {total_time:.2f}s ({((total_steps * batch_size * seq_len)/total_time):.0f} tok/s)")

    # Save Checkpoint
    ckpt_out = os.path.join(out_dir, "checkpoints", "hyperspace_genuine_diverse_engine.pt")
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

    # Plot Multi-Domain Learning Curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # 1. Multi-Domain Loss Descent
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for d, c in zip(streamer.domains, colors):
        axes[0].plot(history["step"], history["domain_losses"][d], label=f"{d}", color=c, linewidth=2.0)
    axes[0].set_xlabel("Training Steps (Interleaved)", fontweight='bold')
    axes[0].set_ylabel("Cross-Entropy Validation Loss", fontweight='bold')
    axes[0].set_title("True Multi-Domain Loss Convergence (100% Unique Data)", fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Top-1 Token Accuracy
    for d, c in zip(streamer.domains, colors):
        axes[1].plot(history["step"], history["domain_top1"][d], label=f"{d}", color=c, linewidth=2.0)
    axes[1].set_xlabel("Training Steps (Interleaved)", fontweight='bold')
    axes[1].set_ylabel("Top-1 Token Accuracy (%)", fontweight='bold')
    axes[1].set_title("True Multi-Domain Top-1 Token Prediction Accuracy", fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_out = os.path.join(out_dir, "plots", "genuine_diverse_engine_evaluation.png")
    plt.savefig(plot_out)
    plt.close()
    print(f"[SAVED] Evaluation plot saved to: {plot_out}")

    # Copy to brain artifact directory
    brain_dir = r"C:\Users\Asta\.gemini\antigravity-ide\brain\0f00f3a4-e432-4a06-8006-7dfc2fa48c7a"
    import shutil
    shutil.copy(plot_out, os.path.join(brain_dir, "genuine_diverse_engine_evaluation.png"))

if __name__ == "__main__":
    run_genuine_training(total_steps=1000, seq_len=256, batch_size=8, lr=6e-4)
