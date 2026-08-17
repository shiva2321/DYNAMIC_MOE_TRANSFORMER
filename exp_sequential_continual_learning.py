"""
Genuine Sequential Continual Learning & Catastrophic Forgetting Experiment.
Protocol:
- Phase 1: Train strictly on FineWeb-Edu (Domain 1). Record baseline L_{1, :}.
- Phase 2: Train strictly on Python Code (Domain 2) without revisiting Domain 1. Record L_{2, :}.
- Phase 3: Train strictly on WikiText-103 (Domain 3) without revisiting Domains 1 & 2. Record L_{3, :}.
- Phase 4: Train strictly on TinyStories (Domain 4) without revisiting Domains 1, 2 & 3. Record L_{4, :}.
Computes exact 4x4 Continual Learning Matrix and true Backward Transfer (R_BWT).
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
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.nanogpt import HyperTransformerLM
from hyperspace.vsa import ComplexPhasorVSA
from train_scaled_production_engine import ScaledProductionDataStreamer

def run_sequential_continual_learning(
    steps_per_phase: int = 300,
    seq_len: int = 256,
    micro_batch_size: int = 4,
    accum_steps: int = 3,
    lr: float = 5e-4,
    out_dir: str = "experiments"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 95)
    print("  [UNIVERSAL SUBSTRAIT: GENUINE SEQUENTIAL CONTINUAL LEARNING BENCHMARK]")
    print("  Protocol: 4 Isolated Sequential Phases (Strict Non-Interleaved Domain Streams)")
    print(f"  Steps Per Phase: {steps_per_phase} | Total Steps: {steps_per_phase * 4}")
    print("=" * 95)

    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    streamer = ScaledProductionDataStreamer(cache_dir="data/scaled_real_corpus", seq_len=seq_len, batch_size=micro_batch_size)
    domains = ["fineweb_edu", "python_code", "wikitext_facts", "natural_stories"]
    domain_titles = [streamer.metadata["domains"][d]["title"] for d in domains]

    # Initialize Hyperspace MoE Model
    model = HyperTransformerLM(
        vocab_size=50304,
        d_model=384,
        n_layers=4,
        n_heads=6,
        d_ff=768,
        d_hyper=2048,
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
        max_seq_len=576,
        dropout=0.0
    ).to(device)

    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    # Evaluation Matrix: R[phase_idx, domain_idx]
    loss_matrix = np.zeros((4, 4))
    acc_matrix = np.zeros((4, 4))

    def evaluate_all_domains() -> Tuple[List[float], List[float]]:
        model.eval()
        dom_losses = []
        dom_accs = []
        with torch.no_grad():
            for d in domains:
                total_l = 0.0
                total_correct = 0
                total_tokens = 0
                for _ in range(10):
                    vx, vy = streamer.get_domain_val_batch(d, num_samples=8)
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast('cuda'):
                        v_logits, _, _ = model(vx, allow_spawning=False)
                        loss = F.cross_entropy(v_logits.view(-1, model.vocab_size), vy.view(-1), reduction='sum')
                    total_l += loss.item()
                    total_tokens += vy.numel()
                    preds = torch.argmax(v_logits, dim=-1)
                    total_correct += (preds == vy).sum().item()
                dom_losses.append(total_l / total_tokens)
                dom_accs.append((total_correct / total_tokens) * 100.0)
        return dom_losses, dom_accs

    # Initial Zero-Shot Evaluation
    init_l, init_a = evaluate_all_domains()
    print(f"\nInitial Untrained Loss across domains: {[round(l, 3) for l in init_l]}")

    for phase_idx, phase_domain in enumerate(domains):
        dom_name = domain_titles[phase_idx]
        print("\n" + "-" * 95)
        print(f"  PHASE {phase_idx + 1}/4: TRAINING EXCLUSIVELY ON [{dom_name}]")
        print(f"  (All other domains strictly absent from training stream)")
        print("-" * 95)

        for step in range(1, steps_per_phase + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(accum_steps):
                x, y = streamer.get_domain_batch(phase_domain, split="train")
                x, y = x.to(device), y.to(device)

                with torch.amp.autocast('cuda'):
                    logits, loss, _ = model(x, targets=y, allow_spawning=True)
                    loss = loss / accum_steps

                accum_loss += loss.item() * accum_steps
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if step % 100 == 0 or step == steps_per_phase:
                total_exp = sum(b.hyper_moe.num_experts for b in model.blocks)
                print(f"  Phase {phase_idx+1} Step {step:3d}/{steps_per_phase} | Train Loss: {accum_loss:.4f} | Total Experts: {total_exp}")

        # Post-Phase Evaluation across ALL domains
        phase_losses, phase_accs = evaluate_all_domains()
        loss_matrix[phase_idx, :] = phase_losses
        acc_matrix[phase_idx, :] = phase_accs

        print(f"\n  >>> Post-Phase {phase_idx + 1} Evaluation Matrix Row:")
        for d_i, (d_loss, d_acc) in enumerate(zip(phase_losses, phase_accs)):
            status = ""
            if d_i == phase_idx:
                status = "(Active Task Target)"
            elif d_i < phase_idx:
                delta = d_loss - loss_matrix[d_i, d_i]
                status = f"(Past Task: Delta={delta:+.3f} nats)"
            else:
                status = "(Future Task: Zero-Shot)"
            print(f"      • {domains[d_i]:<16}: Loss = {d_loss:.4f} | Acc = {d_acc:4.1f}%  {status}")

    # Compute True Backward Transfer (BWT)
    # R_BWT = 1/(T-1) * sum_{i=1}^{T-1} (L_{T, i} - L_{i, i})
    bwt_deltas = []
    print("\n" + "=" * 95)
    print("  [TRUE SEQUENTIAL BACKWARD TRANSFER (BWT) MATRIX RESULT]")
    print("=" * 95)
    print(f"{'Task / Domain':<25} | {'Immediate Loss (L_ii)':<22} | {'Final Loss (L_Ti)':<18} | {'Delta (L_Ti - L_ii)':<20} | {'Status'}")
    print("-" * 95)
    for i in range(3):
        l_ii = loss_matrix[i, i]
        l_ti = loss_matrix[3, i]
        delta = l_ti - l_ii
        bwt_deltas.append(delta)
        status = "Degradation (Forgetting)" if delta > 0.05 else ("Retention (Transfer)" if delta < -0.05 else "Stable Memory")
        print(f"{domains[i]:<25} | {l_ii:<22.4f} | {l_ti:<18.4f} | {delta:<+20.4f} | {status}")
    print("-" * 95)
    mean_bwt = np.mean(bwt_deltas)
    print(f"Mean Sequential Backward Transfer (R_BWT): {mean_bwt:+.4f} nats")
    print(f"Overall Lifelong Learning Verdict: {'Degradation (Forgetting Observed)' if mean_bwt > 0.05 else 'Robust Forgetting Immunity / Retention'}")
    print("=" * 95)

    # Save Checkpoint & Results
    results_path = os.path.join(out_dir, "sequential_continual_learning_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "domains": domains,
            "loss_matrix": loss_matrix.tolist(),
            "acc_matrix": acc_matrix.tolist(),
            "mean_bwt_nats": float(mean_bwt),
            "bwt_deltas": [float(d) for d in bwt_deltas]
        }, f, indent=2)

    # Plot Heatmap
    plt.figure(figsize=(9, 7))
    plt.imshow(loss_matrix, cmap="viridis_r", aspect="auto")
    plt.colorbar(label="Validation Loss (nats)")
    plt.xticks(range(4), [d.replace("_", " ").title() for d in domains], rotation=25)
    plt.yticks(range(4), [f"After Phase {i+1} ({domains[i]})" for i in range(4)])
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{loss_matrix[i, j]:.2f}", ha="center", va="center", color="white" if loss_matrix[i, j] > 5.0 else "black", fontweight="bold")
    plt.title("Sequential Continual Learning: Empirical Loss Matrix $R_{t, i}$")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plots", "sequential_continual_learning_matrix.png"), dpi=300)
    plt.close()
    print(f"[SAVED] Results written to: {results_path}")
    print(f"[SAVED] Plot written to: experiments/plots/sequential_continual_learning_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential Continual Learning Experiment")
    parser.add_argument("--steps", type=int, default=300, help="Steps per phase")
    args = parser.parse_args()
    run_sequential_continual_learning(steps_per_phase=args.steps)
