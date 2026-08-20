"""
Sequential Continual Learning with Exemplar Memory Replay (Experience Replay / TinyReplayBuffer).
Implements the canonical continual learning benchmark:
- Fixed-capacity exemplar ring buffer per completed domain.
- Replay sampling ratio alpha = 0.20 (80% current domain, 20% past domain exemplars).
- Evaluates true 4x4 loss matrix R_{t, i} and R_BWT to quantify forgetting mitigation.
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
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.nanogpt import HyperTransformerLM
from train_scaled_production_engine import ScaledProductionDataStreamer

class TinyExemplarBuffer:
    """Fixed-capacity episodic memory buffer storing exemplar sequences from completed tasks."""
    def __init__(self, max_samples_per_domain: int = 256):
        self.max_samples = max_samples_per_domain
        self.buffer: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    def add_exemplars(self, domain: str, streamer: ScaledProductionDataStreamer, num_samples: int = 256):
        if domain not in self.buffer:
            self.buffer[domain] = []
        
        # Sample held-out exemplars directly from training shard
        shards = streamer.train_shards[domain]
        max_start = len(shards) - streamer.seq_len - 1
        starts = np.random.randint(0, max_start, size=num_samples)
        for s in starts:
            x = shards[s:s + streamer.seq_len].astype(np.int64)
            y = shards[s + 1:s + streamer.seq_len + 1].astype(np.int64)
            self.buffer[domain].append((x, y))
            if len(self.buffer[domain]) > self.max_samples:
                self.buffer[domain].pop(0)
        print(f"  [Memory Buffer] Buffered {len(self.buffer[domain])} real exemplar sequences for domain: '{domain}'")

    def sample_batch(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        all_domains = list(self.buffer.keys())
        if not all_domains:
            return None
        
        bx, by = [], []
        for _ in range(batch_size):
            chosen_dom = np.random.choice(all_domains)
            samples = self.buffer[chosen_dom]
            idx = np.random.randint(0, len(samples))
            x, y = samples[idx]
            bx.append(x)
            by.append(y)
            
        return torch.from_numpy(np.stack(bx)), torch.from_numpy(np.stack(by))

def run_sequential_exemplar_replay(
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
    print("  [UNIVERSAL SUBSTRAIT: SEQUENTIAL CONTINUAL LEARNING WITH EXEMPLAR REPLAY]")
    print(f"  Protocol: 4 Sequential Phases | Replay Ratio: {replay_ratio*100:.0f}% Past Task Rehearsal")
    print(f"  Steps Per Phase: {steps_per_phase} | Total Steps: {steps_per_phase * 4}")
    print("=" * 95)

    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    streamer = ScaledProductionDataStreamer(cache_dir="data/scaled_real_corpus", seq_len=seq_len, batch_size=micro_batch_size)
    domains = ["fineweb_edu", "python_code", "wikitext_facts", "natural_stories"]
    domain_titles = [streamer.metadata["domains"][d]["title"] for d in domains]

    # Initialize Fresh Hyperspace MoE Model
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
    from hyperspace.dynamic_optimizer import DynamicWarmupAdamW
    optimizer = DynamicWarmupAdamW(model.parameters(), lr=lr, weight_decay=0.01, default_group_warmup_steps=50)
    prev_expert_counts = [b.hyper_moe.num_experts for b in model.blocks]
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
                        v_logits, _, _ = model(vx, allow_spawning=False)
                        loss = F.cross_entropy(v_logits.view(-1, model.vocab_size), vy.view(-1), reduction='sum')
                    total_l += loss.item()
                    total_tokens += vy.numel()
                    preds = torch.argmax(v_logits, dim=-1)
                    total_correct += (preds == vy).sum().item()
                dom_losses.append(total_l / total_tokens)
                dom_accs.append((total_correct / total_tokens) * 100.0)
        return dom_losses, dom_accs

    for phase_idx, phase_domain in enumerate(domains):
        dom_name = domain_titles[phase_idx]
        print("\n" + "-" * 95)
        print(f"  PHASE {phase_idx + 1}/4: TRAINING ON [{dom_name}]")
        if phase_idx > 0:
            print(f"  (Replay Enabled: Mixing in exemplars from {list(memory_buffer.buffer.keys())})")
        print("-" * 95)

        for step in range(1, steps_per_phase + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(accum_steps):
                # Sample batch with exemplar replay if past domains exist
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
                    logits, loss, _ = model(x, targets=y, allow_spawning=True)
                    loss = loss / accum_steps

                accum_loss += loss.item() * accum_steps
                scaler.scale(loss).backward()

                # Dynamic Spawning Parameter Registration
                curr_expert_counts = [b.hyper_moe.num_experts for b in model.blocks]
                if curr_expert_counts != prev_expert_counts:
                    for l_idx, block in enumerate(model.blocks):
                        if curr_expert_counts[l_idx] > prev_expert_counts[l_idx]:
                            for new_exp_idx in range(prev_expert_counts[l_idx], curr_expert_counts[l_idx]):
                                new_exp = block.hyper_moe.experts[new_exp_idx]
                                key_r = block.hyper_moe.memory.keys_r[new_exp_idx]
                                key_i = block.hyper_moe.memory.keys_i[new_exp_idx]
                                new_params = list(new_exp.parameters()) + [key_r, key_i]
                                optimizer.add_dynamic_param_group(new_params, lr=lr, warmup_steps=50, group_name=f"L{l_idx}_E{new_exp_idx}")
                                print(f"  🌱 [AUTONOMOUS SPAWN REGISTERED @ PHASE {phase_idx+1} STEP {step}] Layer {l_idx} spawned Expert #{new_exp_idx}! Registered in DynamicWarmupAdamW.", flush=True)
                    prev_expert_counts = curr_expert_counts

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if step % 100 == 0 or step == steps_per_phase:
                total_exp = sum(b.hyper_moe.num_experts for b in model.blocks)
                print(f"  Phase {phase_idx+1} Step {step:3d}/{steps_per_phase} | Train Loss: {accum_loss:.4f} | Total Experts: {total_exp}")

        # Add exemplars of this finished phase to memory buffer
        memory_buffer.add_exemplars(phase_domain, streamer, num_samples=256)

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

    # Compute Backward Transfer (BWT) with Replay
    bwt_deltas = []
    print("\n" + "=" * 95)
    print("  [SEQUENTIAL CONTINUAL LEARNING WITH EXEMPLAR REPLAY: BWT RESULT]")
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
    print(f"Mean Sequential Backward Transfer with Replay (R_BWT): {mean_bwt:+.4f} nats")
    print("=" * 95)

    # Save JSON results
    results_path = os.path.join(out_dir, "sequential_exemplar_replay_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "domains": domains,
            "loss_matrix": loss_matrix.tolist(),
            "acc_matrix": acc_matrix.tolist(),
            "mean_bwt_nats": float(mean_bwt),
            "bwt_deltas": [float(d) for d in bwt_deltas],
            "replay_ratio": replay_ratio
        }, f, indent=2)

    # Plot Comparison Matrix
    plt.figure(figsize=(9, 7))
    plt.imshow(loss_matrix, cmap="viridis_r", aspect="auto")
    plt.colorbar(label="Validation Loss (nats)")
    plt.xticks(range(4), [d.replace("_", " ").title() for d in domains], rotation=25)
    plt.yticks(range(4), [f"After Phase {i+1} ({domains[i]})" for i in range(4)])
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{loss_matrix[i, j]:.2f}", ha="center", va="center", color="white" if loss_matrix[i, j] > 5.0 else "black", fontweight="bold")
    plt.title(f"Sequential Continual Learning with Exemplar Replay (R_BWT = {mean_bwt:+.3f} nats)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plots", "sequential_exemplar_replay_matrix.png"), dpi=300)
    plt.close()

    print(f"[SAVED] Results written to: {results_path}")
    print(f"[SAVED] Plot written to: experiments/plots/sequential_exemplar_replay_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential Continual Learning with Replay")
    parser.add_argument("--steps", type=int, default=300, help="Steps per phase")
    parser.add_argument("--ratio", type=float, default=0.20, help="Replay ratio")
    args = parser.parse_args()
    run_sequential_exemplar_replay(steps_per_phase=args.steps, replay_ratio=args.ratio)
