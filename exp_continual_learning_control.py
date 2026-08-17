"""
Unified Continual Learning Control Benchmark: Attribution Isolation.
Runs the exact same 4-phase sequential protocol with 20% TinyExemplarBuffer across:
1. DenseTransformerLM (Monolithic Baseline)
2. StaticSoftmaxMoELM (Static MoE Baseline)
3. HyperTransformerLM (Universal Substrait Dynamic MoE)
to cleanly isolate architectural contributions from generic buffer effects.
"""

import os
import sys
import time
import json
import math
import argparse
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.nanogpt import HyperTransformerLM
from model.baselines import DenseTransformerLM, StaticSoftmaxMoELM
from train_scaled_production_engine import ScaledProductionDataStreamer
from exp_sequential_exemplar_replay import TinyExemplarBuffer

def run_control_experiment(
    model_type: str = "dense", # "dense", "static_moe", or "hyperspace"
    steps_per_phase: int = 300,
    replay_ratio: float = 0.20,
    seq_len: int = 256,
    micro_batch_size: int = 4,
    accum_steps: int = 3,
    lr: float = 5e-4,
    out_dir: str = "experiments"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 95)
    print(f"  [CONTINUAL LEARNING CONTROL EXPERIMENT: MODEL = {model_type.upper()}]")
    print(f"  Protocol: 4 Sequential Phases | Replay Ratio: {replay_ratio*100:.0f}% Exemplar Buffer")
    print(f"  Steps Per Phase: {steps_per_phase} | Total Steps: {steps_per_phase * 4}")
    print("=" * 95)

    streamer = ScaledProductionDataStreamer(cache_dir="data/scaled_real_corpus", seq_len=seq_len, batch_size=micro_batch_size)
    domains = ["fineweb_edu", "python_code", "wikitext_facts", "natural_stories"]
    domain_titles = [streamer.metadata["domains"][d]["title"] for d in domains]

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
            d_ff=d_ff * 2, # Matched compute
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
    elif model_type == "hyperspace":
        model = HyperTransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
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
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    memory_buffer = TinyExemplarBuffer(max_samples_per_domain=256)

    loss_matrix = np.zeros((4, 4))
    acc_matrix = np.zeros((4, 4))

    def evaluate_all_domains() -> Tuple[List[float], List[float]]:
        model.eval()
        dom_losses, dom_accs = [], []
        with torch.no_grad():
            for d in domains:
                total_l = 0.0
                total_correct = 0
                total_tokens = 0
                for _ in range(10):
                    vx, vy = streamer.get_domain_val_batch(d, num_samples=8)
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast('cuda'):
                        if model_type == "hyperspace":
                            v_logits, _, _ = model(vx, allow_spawning=False)
                        else:
                            v_logits, _, _ = model(vx)
                        loss = F.cross_entropy(v_logits.view(-1, vocab_size), vy.view(-1), reduction='sum')
                    total_l += loss.item()
                    total_tokens += vy.numel()
                    preds = torch.argmax(v_logits, dim=-1)
                    total_correct += (preds == vy).sum().item()
                dom_losses.append(total_l / total_tokens)
                dom_accs.append((total_correct / total_tokens) * 100.0)
        return dom_losses, dom_accs

    for phase_idx, phase_domain in enumerate(domains):
        dom_name = domain_titles[phase_idx]
        print(f"\n--- Phase {phase_idx+1}/4: [{dom_name}] (Replay Domains: {list(memory_buffer.buffer.keys())}) ---")

        for step in range(1, steps_per_phase + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(accum_steps):
                if phase_idx > 0 and np.random.rand() < replay_ratio:
                    replay_data = memory_buffer.sample_batch(micro_batch_size)
                    if replay_data is not None:
                        x, y = replay_data
                    else:
                        x, y = streamer.get_domain_batch(phase_domain, split="train")
                else:
                    x, y = streamer.get_domain_batch(phase_domain, split="train")

                x, y = x.to(device), y.to(device)

                with torch.amp.autocast('cuda'):
                    if model_type == "hyperspace":
                        logits, loss, _ = model(x, targets=y, allow_spawning=True)
                    else:
                        logits, loss, _ = model(x, targets=y)
                    loss = loss / accum_steps

                accum_loss += loss.item() * accum_steps
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        memory_buffer.add_exemplars(phase_domain, streamer, num_samples=256)
        phase_losses, phase_accs = evaluate_all_domains()
        loss_matrix[phase_idx, :] = phase_losses
        acc_matrix[phase_idx, :] = phase_accs

        print(f"  >>> Post-Phase {phase_idx+1} Val Losses: FineWeb={phase_losses[0]:.3f} | Python={phase_losses[1]:.3f} | Wiki={phase_losses[2]:.3f} | Stories={phase_losses[3]:.3f}")

    # BWT Calculation
    bwt_deltas = [loss_matrix[3, i] - loss_matrix[i, i] for i in range(3)]
    mean_bwt = np.mean(bwt_deltas)

    print("\n" + "=" * 95)
    print(f"  [CONTROL EXPERIMENT RESULT: {model_type.upper()} + 20% EXEMPLAR REPLAY]")
    print(f"  Mean R_BWT: {mean_bwt:+.4f} nats")
    print(f"  Phase 2 Python: Immediate Acc = {acc_matrix[1, 1]:.1f}% -> Final Displaced Acc = {acc_matrix[3, 1]:.1f}%")
    print("=" * 95)

    results_path = os.path.join(out_dir, f"continual_control_{model_type}_replay_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": model_type,
            "loss_matrix": loss_matrix.tolist(),
            "acc_matrix": acc_matrix.tolist(),
            "mean_bwt_nats": float(mean_bwt),
            "bwt_deltas": [float(d) for d in bwt_deltas]
        }, f, indent=2)
    print(f"[SAVED] Results saved to: {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual Learning Control Runner")
    parser.add_argument("--model", type=str, choices=["dense", "static_moe", "hyperspace"], default="dense")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--ratio", type=float, default=0.20)
    args = parser.parse_args()
    run_control_experiment(model_type=args.model, steps_per_phase=args.steps, replay_ratio=args.ratio)
