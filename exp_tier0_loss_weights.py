"""
Tier 0 Experiment: Testing High-Strength Auxiliary Orthogonality and Load-Balancing Weights.
Evaluates whether 10x-20x stronger auxiliary loss penalties (ortho=0.05, load_bal=0.10)
prevent Layer 3 output-layer collapse in Universal Substrait under identical 4-phase sequential protocol.
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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.nanogpt import HyperTransformerLM
from hyperspace.dynamic_optimizer import DynamicWarmupAdamW
from train_scaled_production_engine import ScaledProductionDataStreamer

def run_tier0_experiment(steps_per_phase: int = 300, replay_ratio: float = 0.20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("  [TIER 0 EXPERIMENT: HIGH-STRENGTH AUXILIARY ORTHO & LOAD-BALANCING WEIGHTS]")
    print(f"  Protocol: 4 Sequential Phases | Replay Ratio: {int(replay_ratio*100)}% Exemplar Buffer")
    print(f"  Auxiliary Weights: ortho_loss_weight = 0.05 (10x) | load_bal_weight = 0.10 (10x)")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 100)

    seq_len = 256
    batch_size = 12
    domains = ["fineweb_edu", "python_code", "wikitext_facts", "natural_stories"]
    streamer = ScaledProductionDataStreamer(cache_dir="data/scaled_real_corpus", seq_len=seq_len, batch_size=batch_size)

    # 1. Instantiate Model with 10x Auxiliary Loss Weights
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
        dropout=0.0,
        use_bus=True,
        ortho_loss_weight=0.05,
        load_bal_weight=0.10
    ).to(device)

    max_lr = 6e-4
    min_lr = 6e-5
    optimizer = DynamicWarmupAdamW(model.parameters(), lr=max_lr, weight_decay=0.01, default_group_warmup_steps=30)
    scaler = torch.amp.GradScaler('cuda')

    # Exemplar Buffer
    exemplar_buffer: Dict[str, torch.Tensor] = {}
    exemplars_per_domain = 256

    def add_exemplars(dom: str):
        shards = streamer.train_shards[dom]
        max_start = len(shards) - seq_len - 1
        starts = np.random.randint(0, max_start, size=exemplars_per_domain)
        x_ex = np.stack([shards[s:s + seq_len] for s in starts]).astype(np.int64)
        exemplar_buffer[dom] = torch.from_numpy(x_ex)

    def sample_batch(dom: str) -> Tuple[torch.Tensor, torch.Tensor]:
        prior_doms = [d for d in domains if d in exemplar_buffer and d != dom]
        if not prior_doms or replay_ratio <= 0.0:
            return streamer.get_domain_batch(dom, split="train")

        n_replay = max(1, int(batch_size * replay_ratio))
        n_curr = batch_size - n_replay

        curr_x, curr_y = streamer.get_domain_batch(dom, split="train", num_samples=n_curr)
        replay_xs = []
        for _ in range(n_replay):
            r_dom = random.choice(prior_doms)
            buf = exemplar_buffer[r_dom]
            idx = random.randint(0, buf.shape[0] - 1)
            replay_xs.append(buf[idx])

        r_x = torch.stack(replay_xs)
        r_y = torch.roll(r_x, shifts=-1, dims=-1)
        r_y[:, -1] = r_x[:, -1]

        combined_x = torch.cat([curr_x, r_x], dim=0)
        combined_y = torch.cat([curr_y, r_y], dim=0)
        perm = torch.randperm(batch_size)
        return combined_x[perm], combined_y[perm]

    def evaluate_domains() -> Dict[str, float]:
        model.eval()
        res = {}
        with torch.no_grad():
            for d in domains:
                vx, vy = streamer.get_domain_val_batch(d, num_samples=16)
                vx, vy = vx.to(device), vy.to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _, l, _ = model(vx, targets=vy)
                res[d] = float(l.item())
        return res

    # 2. Sequential 4-Phase Training Loop
    loss_matrix = np.zeros((4, 4))
    t_start = time.perf_counter()

    for phase_idx, curr_domain in enumerate(domains):
        print(f"\n--- Phase {phase_idx + 1}/4: [{streamer.domain_names[curr_domain]}] ---")
        warmup_steps = 30
        for step in range(1, steps_per_phase + 1):
            if step < warmup_steps:
                curr_lr = max_lr * (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / (steps_per_phase - warmup_steps)
                curr_lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

            for pg in optimizer.param_groups:
                pg["base_lr"] = curr_lr
                pg["lr"] = curr_lr

            model.train()
            optimizer.zero_grad(set_to_none=True)
            bx, by = sample_batch(curr_domain)
            bx, by = bx.to(device), by.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss, aux = model(bx, targets=by, allow_spawning=True)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        add_exemplars(curr_domain)
        val_losses = evaluate_domains()
        for j, d in enumerate(domains):
            loss_matrix[phase_idx, j] = val_losses[d]
        print(f"  >>> Post-Phase {phase_idx + 1} Val Losses: FineWeb={loss_matrix[phase_idx,0]:.3f} | Python={loss_matrix[phase_idx,1]:.3f} | Wiki={loss_matrix[phase_idx,2]:.3f} | Stories={loss_matrix[phase_idx,3]:.3f}")

    # Save Checkpoint
    ckpt_path = "experiments/checkpoints/hyperspace_tier0_high_aux.pt"
    torch.save({"model_state": model.state_dict(), "loss_matrix": loss_matrix.tolist()}, ckpt_path)
    print(f"\n[SAVED] Tier 0 checkpoint saved to: {ckpt_path}")

    # 3. Full Layer-Wise Routing Mass Diagnostic
    print("\n" + "=" * 100)
    print("  [GROUND-TRUTH LAYER-WISE ROUTING BREAKDOWN (TIER 0: HIGH AUX LOSS)]")
    print("=" * 100)

    model.eval()
    routing_breakdown = {}

    with torch.no_grad():
        for l_idx in range(4):
            print(f"\n--- LAYER {l_idx} ---")
            weighted_usage = {d: np.zeros(16) for d in domains}
            for d in domains:
                for _ in range(12):
                    vx, vy = streamer.get_domain_val_batch(d, num_samples=8)
                    vx = vx.to(device)
                    _, _, telem = model(vx, allow_spawning=False)
                    idx = telem[l_idx]['top_indices'].reshape(-1).cpu().numpy()
                    w = telem[l_idx]['top_weights'].reshape(-1).cpu().numpy()
                    for e_id, weight in zip(idx, w):
                        if e_id < 16:
                            weighted_usage[d][e_id] += weight

            layer_data = {}
            for d in domains:
                tot = weighted_usage[d].sum()
                order = np.argsort(-weighted_usage[d])
                shares = weighted_usage[d][order] / tot * 100
                top_str = ', '.join([f'E{order[i]}:{shares[i]:.1f}%' for i in range(5)])
                top3_sum = shares[:3].sum()
                layer_data[d] = {
                    "top_experts": [f"E{order[i]}:{shares[i]:.1f}%" for i in range(5)],
                    "top3_mass_pct": float(top3_sum)
                }
                print(f"  {d:<18} -> {top_str} (Top-3 mass: {top3_sum:.1f}%)")
            routing_breakdown[f"L{l_idx}"] = layer_data

    # Calculate Backward Transfer Drift
    deltas = [loss_matrix[3, i] - loss_matrix[i, i] for i in range(3)]
    mean_bwt = float(np.mean(deltas))

    print("\n" + "=" * 100)
    print(f"  [TIER 0 OUTCOME SUMMARY]")
    print(f"  Mean R_BWT: {mean_bwt:+.4f} nats (FineWeb: {deltas[0]:+.4f}, Python: {deltas[1]:+.4f}, Wiki: {deltas[2]:+.4f})")
    print("=" * 100)

    # Save Diagnostic JSON
    out_payload = {
        "loss_matrix": loss_matrix.tolist(),
        "mean_bwt": mean_bwt,
        "deltas": deltas,
        "routing_breakdown": routing_breakdown
    }
    with open("experiments/tier0_high_aux_diagnostic_results.json", "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

if __name__ == "__main__":
    run_tier0_experiment()
