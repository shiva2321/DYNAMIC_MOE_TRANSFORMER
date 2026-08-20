"""
Unified Continual Learning Control Benchmark: Attribution Isolation.
Runs the exact same 4-phase sequential protocol with 20% TinyExemplarBuffer across:
1. DenseTransformerLM (Monolithic Baseline)
2. StaticSoftmaxMoELM (Static MoE Baseline)
3. HyperTransformerLM (Universal Substrait Dynamic MoE)
to cleanly isolate architectural contributions from generic buffer effects.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
from model.baselines import DenseTransformerLM, StaticSoftmaxMoELM, DendriticStaticSoftmaxMoELM
from train_scaled_production_engine import ScaledProductionDataStreamer
from exp_sequential_exemplar_replay import TinyExemplarBuffer

def run_control_experiment(
    model_type: str = "dense", # "dense", "static_moe", or "hyperspace"
    steps_per_phase: int = 300,
    replay_ratio: float = 0.20,
    seq_len: int = 256,
    micro_batch_size: int = 2,
    accum_steps: int = 6,
    lr: float = 5e-4,
    seed: int = 1337,
    out_dir: str = "experiments"
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 95)
    print(f"  [CONTINUAL LEARNING CONTROL EXPERIMENT: MODEL = {model_type.upper()} | SEED = {seed}]")
    print(f"  Protocol: 4 Sequential Phases | Replay Ratio: {replay_ratio*100:.0f}% Exemplar Buffer")
    print(f"  Steps Per Phase: {steps_per_phase} | Total Steps: {steps_per_phase * 4} | Effective Batch: {micro_batch_size * accum_steps}")
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
    elif model_type == "static_moe_30exp": # Clean Capacity-Matched (128.09M params vs Hyperspace 128.92M)
        model = StaticSoftmaxMoELM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            num_experts=30,
            top_k=2,
            max_seq_len=seq_len + 64
        ).to(device)
    elif model_type == "hyperspace_core_2exp":
        # Pure Concentrated 2-Expert Core Champion (Drift-Guard OFF, Binding OFF)
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
            spawn_threshold=1.0, # Pure 2-expert concentrated nucleus (no spawning)
            max_experts=2,
            initial_experts=2,
            use_sparse_attn=True,
            foveal_window=128,
            num_landmarks=4,
            num_sinks=4,
            max_seq_len=576,
            dropout=0.0,
            use_bus=True,
            use_context_binding=False,
            use_drift_guard=False,
            ortho_loss_weight=0.005,
            load_bal_weight=0.01
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
            dropout=0.0,
            use_bus=True
        ).to(device)
    elif model_type == "hyperspace_budgeted_spawn":
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
            dropout=0.0,
            use_bus=True,
            ortho_loss_weight=0.005,
            load_bal_weight=0.01
        ).to(device)
    elif model_type == "hyperspace_drift_guard":
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
            dropout=0.0,
            use_bus=True,
            use_drift_guard=True,
            drift_sigma=2.0,
            ortho_loss_weight=0.005,
            load_bal_weight=0.01
        ).to(device)
    elif model_type == "hyperspace_high_aux":
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
            dropout=0.0,
            use_bus=True,
            ortho_loss_weight=0.05,
            load_bal_weight=0.10
        ).to(device)
    elif model_type == "hyperspace_no_bus":
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
            dropout=0.0,
            use_bus=False
        ).to(device)
    elif model_type == "hyperspace_fixed_16exp":
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
            dynamic_k=False,
            spawn_threshold=1.0, # Disable dynamic neurogenesis
            max_experts=16,
            initial_experts=16,
            use_sparse_attn=True,
            foveal_window=128,
            num_landmarks=4,
            num_sinks=4,
            max_seq_len=576,
            dropout=0.0,
            use_bus=True
        ).to(device)
    elif model_type == "hyperspace_all_layer_drift_guard_2exp":
        # ISOLATION TEST: All-4-Layers Drift-Guard on Pure 2-Expert Core
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
            spawn_threshold=1.0, # Pure 2-expert concentrated nucleus
            max_experts=2,
            initial_experts=2,
            use_sparse_attn=True,
            foveal_window=128,
            num_landmarks=4,
            num_sinks=4,
            max_seq_len=576,
            dropout=0.0,
            use_bus=True,
            use_context_binding=False,
            use_drift_guard=True, # All 4 layers
            drift_sigma=2.0,
            ortho_loss_weight=0.005,
            load_bal_weight=0.01
        ).to(device)
    elif model_type == "hyperspace_l3_drift_guard":
        # ISOLATION TEST 1: Layer 3 Drift-Guard ALONE (Context Binding OFF)
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
            spawn_threshold=1.0, # Pure 2-expert concentrated nucleus
            max_experts=2,
            initial_experts=2,
            use_sparse_attn=True,
            foveal_window=128,
            num_landmarks=4,
            num_sinks=4,
            max_seq_len=576,
            dropout=0.0,
            use_bus=True,
            use_context_binding=False,
            use_drift_guard=[False, False, False, True],
            drift_sigma=2.0,
            ortho_loss_weight=0.005,
            load_bal_weight=0.01
        ).to(device)
    elif model_type == "hyperspace_context_bundle":
        # ISOLATION TEST 2: Context Bundling ALONE (Drift-Guard OFF)
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
            spawn_threshold=1.0, # Pure 2-expert concentrated nucleus
            max_experts=2,
            initial_experts=2,
            use_sparse_attn=True,
            foveal_window=128,
            num_landmarks=4,
            num_sinks=4,
            max_seq_len=576,
            dropout=0.0,
            use_bus=True,
            use_context_binding=True,
            use_drift_guard=False,
            drift_sigma=2.0,
            ortho_loss_weight=0.005,
            load_bal_weight=0.01
        ).to(device)
    elif model_type == "hyperspace_context_bound":
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
            spawn_threshold=1.0, # Pure 2-expert concentrated nucleus
            max_experts=2,
            initial_experts=2,
            use_sparse_attn=True,
            foveal_window=128,
            num_landmarks=4,
            num_sinks=4,
            max_seq_len=576,
            dropout=0.0,
            use_bus=True,
            use_context_binding=True,
            use_drift_guard=[False, False, False, True],
            drift_sigma=2.0,
            ortho_loss_weight=0.005,
            load_bal_weight=0.01
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    scaler = torch.amp.GradScaler('cuda')
    if "hyperspace" in model_type:
        from hyperspace.dynamic_optimizer import DynamicWarmupAdamW
        optimizer = DynamicWarmupAdamW(model.parameters(), lr=lr, weight_decay=0.01, default_group_warmup_steps=50)
        prev_expert_counts = [b.hyper_moe.num_experts for b in model.blocks]
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
        prev_expert_counts = None

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
                        if "hyperspace" in model_type:
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
        
        # LIVE GUARDRAIL 1: Mathematical Validity Check (Loss <= ln(vocab_size))
        max_valid_loss = math.log(vocab_size)
        for d_name, d_l in zip(domains, dom_losses):
            if d_l > max_valid_loss or math.isnan(d_l):
                raise AssertionError(f"[MATH VALIDITY ERROR] Domain '{d_name}' loss {d_l:.4f} exceeds theoretical uniform bound {max_valid_loss:.4f} or is NaN!")

        return dom_losses, dom_accs

    for phase_idx, phase_domain in enumerate(domains):
        dom_name = domain_titles[phase_idx]
        print(f"\n--- Phase {phase_idx+1}/4: [{dom_name}] (Replay Domains: {list(memory_buffer.buffer.keys())}) ---")

        if "hyperspace" in model_type:
            phase_start_expert_counts = [b.hyper_moe.num_experts for b in model.blocks]

        model.train()
        for step in range(1, steps_per_phase + 1):
            accum_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            # Determine allow_spawning flags per block
            if model_type == "hyperspace_budgeted_spawn":
                curr_counts = [b.hyper_moe.num_experts for b in model.blocks]
                if phase_idx == 0 and step <= 100:
                    # 100-step grace period for Phase 1 to let E0/E1 anchor
                    allow_spawning = [False] * len(model.blocks)
                else:
                    # Max 2 new experts per phase per layer
                    allow_spawning = [(curr_counts[l] - phase_start_expert_counts[l]) < 2 for l in range(len(model.blocks))]
            elif "hyperspace" in model_type:
                allow_spawning = True
            else:
                allow_spawning = False

            for accum_i in range(accum_steps):
                is_replay = False
                if phase_idx > 0 and np.random.rand() < replay_ratio:
                    replay_data = memory_buffer.sample_batch(micro_batch_size)
                    if replay_data is not None:
                        x, y = replay_data
                        is_replay = True
                    else:
                        x, y = streamer.get_domain_batch(phase_domain, split="train")
                else:
                    x, y = streamer.get_domain_batch(phase_domain, split="train")

                x, y = x.to(device), y.to(device)

                with torch.amp.autocast('cuda'):
                    if "hyperspace" in model_type:
                        logits, loss, telems = model(x, targets=y, allow_spawning=allow_spawning, is_replay=is_replay)
                        if is_replay and (step % 50 == 0 or step == 1) and accum_i == 0:
                            l0_top = torch.mode(telems[0]['top_indices'].flatten())[0].item()
                            l1_top = torch.mode(telems[1]['top_indices'].flatten())[0].item()
                            l2_top = torch.mode(telems[2]['top_indices'].flatten())[0].item()
                            l3_top = torch.mode(telems[3]['top_indices'].flatten())[0].item()
                            print(f"  [REPLAY ROUTING @ PHASE {phase_idx+1} STEP {step}] Replay Batch -> L0:E{l0_top} | L1:E{l1_top} | L2:E{l2_top} | L3:E{l3_top}")
                    else:
                        logits, loss, _ = model(x, targets=y)
                    loss = loss / accum_steps

                accum_loss += loss.item() * accum_steps
                scaler.scale(loss).backward()

                # Dynamic Spawning Parameter Registration
                if "hyperspace" in model_type:
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

            if step % 50 == 0 or step == steps_per_phase:
                print(f"  [Step {step}/{steps_per_phase}] Mean Accum Loss: {accum_loss:.4f}", flush=True)
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        memory_buffer.add_exemplars(phase_domain, streamer, num_samples=256)
        phase_losses, phase_accs = evaluate_all_domains()
        loss_matrix[phase_idx, :] = phase_losses
        acc_matrix[phase_idx, :] = phase_accs
        torch.cuda.empty_cache()

        print(f"  >>> Post-Phase {phase_idx+1} Val Losses: FineWeb={phase_losses[0]:.3f} | Python={phase_losses[1]:.3f} | Wiki={phase_losses[2]:.3f} | Stories={phase_losses[3]:.3f}", flush=True)
        print(f"  >>> Post-Phase {phase_idx+1} Val Accs  : FineWeb={phase_accs[0]:.1f}% | Python={phase_accs[1]:.1f}% | Wiki={phase_accs[2]:.1f}% | Stories={phase_accs[3]:.1f}%", flush=True)

        # LIVE GUARDRAIL 2: Live Per-Phase 4-Layer Routing Snapshots
        if "hyperspace" in model_type:
            print(f"\n  [LIVE ROUTING SNAPSHOT @ POST-PHASE {phase_idx+1}]")
            model.eval()
            with torch.no_grad(), torch.amp.autocast('cuda'):
                for d in domains:
                    vx, vy = streamer.get_domain_val_batch(d, num_samples=16)
                    vx = vx.to(device)
                    _, _, telems = model(vx, allow_spawning=False)
                    routing_summary = []
                    for l_idx, telem in enumerate(telems):
                        top_idx = telem['top_indices'].flatten()
                        counts = torch.bincount(top_idx, minlength=model.blocks[l_idx].hyper_moe.num_experts)
                        tot = counts.sum().item()
                        e_str = ", ".join([f"E{e}:{(counts[e].item()/tot)*100:.1f}%" for e in range(min(3, len(counts)))])
                        routing_summary.append(f"L{l_idx}:[{e_str}]")
                    print(f"    {d:<16} -> " + " | ".join(routing_summary), flush=True)

        # LIVE GUARDRAIL 3: Phase-Level Checkpointing
        phase_ckpt_path = os.path.join(out_dir, f"checkpoint_{model_type}_phase_{phase_idx+1}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "model_type": model_type,
            "phase_idx": phase_idx,
            "loss_matrix": loss_matrix[:phase_idx+1].tolist(),
            "acc_matrix": acc_matrix[:phase_idx+1].tolist()
        }, phase_ckpt_path)
        print(f"  [SAVED] Phase {phase_idx+1} checkpoint saved to: {phase_ckpt_path}\n", flush=True)

    # BWT Calculation
    bwt_deltas = [loss_matrix[3, i] - loss_matrix[i, i] for i in range(3)]
    mean_bwt = np.mean(bwt_deltas)

    print("\n" + "=" * 95)
    print(f"  [CONTROL EXPERIMENT RESULT: {model_type.upper()} + 20% EXEMPLAR REPLAY]")
    print(f"  Mean R_BWT: {mean_bwt:+.4f} nats")
    print(f"  Phase 2 Python: Immediate Acc = {acc_matrix[1, 1]:.1f}% -> Final Displaced Acc = {acc_matrix[3, 1]:.1f}%")
    print("=" * 95)

    suffix = f"_seed_{seed}" if seed != 1337 else ""
    results_path = os.path.join(out_dir, f"continual_control_{model_type}{suffix}_replay_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": model_type,
            "seed": seed,
            "loss_matrix": loss_matrix.tolist(),
            "acc_matrix": acc_matrix.tolist(),
            "mean_bwt_nats": float(mean_bwt),
            "bwt_deltas": [float(d) for d in bwt_deltas]
        }, f, indent=2)
    print(f"[SAVED] Results saved to: {results_path}")

    # Save model weights for routing diagnostics
    ckpt_path = os.path.join(out_dir, f"checkpoint_{model_type}{suffix}.pt")
    torch.save({"model_state": model.state_dict(), "model_type": model_type, "seed": seed}, ckpt_path)
    print(f"[SAVED] Model checkpoint saved to: {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual Learning Control Runner")
    parser.add_argument("--model", type=str, choices=["dense", "static_moe", "static_moe_30exp", "hyperspace", "hyperspace_core_2exp", "hyperspace_budgeted_spawn", "hyperspace_drift_guard", "hyperspace_high_aux", "hyperspace_no_bus", "hyperspace_fixed_16exp", "hyperspace_context_bound", "hyperspace_l3_drift_guard", "hyperspace_all_layer_drift_guard_2exp", "hyperspace_context_bundle"], default="dense")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    run_control_experiment(model_type=args.model, steps_per_phase=args.steps, replay_ratio=args.ratio, seed=args.seed)
