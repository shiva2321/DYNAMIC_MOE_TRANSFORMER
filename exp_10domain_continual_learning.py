import os
import sys
import time
import math
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.nanogpt import HyperTransformerLM
from model.baselines import DenseTransformerLM, StaticSoftmaxMoELM
from hyperspace.dynamic_optimizer import DynamicWarmupAdamW
from exp_sequential_exemplar_replay import TinyExemplarBuffer

class Streamer10Domain:
    """
    Zero-latency memory-mapped streamer for the 10-Domain corpus.
    """
    def __init__(self, data_dir: str = "data/corpus_10domain", seq_len: int = 256, batch_size: int = 2):
        self.data_dir = os.path.join(PROJECT_ROOT, data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size

        meta_path = os.path.join(self.data_dir, "metadata_10domain.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.domains = list(self.metadata["domains"].keys())
        self.shards = {}

        for d in self.domains:
            info = self.metadata["domains"][d]
            train_path = os.path.join(self.data_dir, info["train_file"])
            val_path = os.path.join(self.data_dir, info["val_file"])
            self.shards[d] = {
                "train": np.memmap(train_path, dtype=np.uint16, mode='r'),
                "val": np.memmap(val_path, dtype=np.uint16, mode='r')
            }

    def get_batch(self, domain: str, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.shards[domain][split]
        max_idx = len(data) - self.seq_len - 1
        ix = np.random.randint(0, max_idx, size=self.batch_size)
        x_list = [torch.from_numpy((data[i : i + self.seq_len]).astype(np.int64)) for i in ix]
        y_list = [torch.from_numpy((data[i + 1 : i + 1 + self.seq_len]).astype(np.int64)) for i in ix]
        x = torch.stack(x_list)
        y = torch.stack(y_list)
        return x, y

    def get_val_batch(self, domain: str, num_samples: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.shards[domain]["val"]
        max_idx = len(data) - self.seq_len - 1
        step = max(1, max_idx // num_samples)
        ix = [i * step for i in range(num_samples)]
        x_list = [torch.from_numpy((data[i : i + self.seq_len]).astype(np.int64)) for i in ix]
        y_list = [torch.from_numpy((data[i + 1 : i + 1 + self.seq_len]).astype(np.int64)) for i in ix]
        return torch.stack(x_list), torch.stack(y_list)

class TinyExemplarBuffer10D:
    """Fixed-capacity episodic memory buffer storing exemplar sequences for 10-domain replay."""
    def __init__(self, max_samples_per_domain: int = 256):
        self.max_samples = max_samples_per_domain
        self.buffer: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

    def add_exemplars(self, domain: str, streamer: Streamer10Domain, num_samples: int = 256):
        if domain not in self.buffer:
            self.buffer[domain] = []
        data = streamer.shards[domain]["train"]
        max_start = len(data) - streamer.seq_len - 1
        starts = np.random.randint(0, max_start, size=num_samples)
        for s in starts:
            x = torch.from_numpy(data[s : s + streamer.seq_len].astype(np.int64))
            y = torch.from_numpy(data[s + 1 : s + 1 + streamer.seq_len].astype(np.int64))
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
            d = np.random.choice(all_domains)
            idx = np.random.randint(0, len(self.buffer[d]))
            x, y = self.buffer[d][idx]
            bx.append(x)
            by.append(y)
        return torch.stack(bx), torch.stack(by)

def run_10domain_experiment(
    model_type: str = "hyperspace_budgeted_spawn",
    steps_per_phase: int = 360,
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
    streamer = Streamer10Domain(seq_len=seq_len, batch_size=micro_batch_size)
    domains = streamer.domains
    domain_titles = [streamer.metadata["domains"][d]["title"] for d in domains]
    num_domains = len(domains)

    total_steps = steps_per_phase * num_domains
    effective_batch = micro_batch_size * accum_steps
    tokens_per_step = effective_batch * seq_len
    total_tokens = total_steps * tokens_per_step

    print("\n" + "=" * 105)
    print(f"  [10-DOMAIN CONTINUAL LEARNING DUEL: MODEL = {model_type.upper()} | SEED = {seed}]")
    print(f"  Curriculum: 10 Diverse Knowledge Domains | Replay Ratio: {replay_ratio*100:.0f}% Exemplar Buffer")
    print(f"  Steps / Domain: {steps_per_phase} | Total Steps: {total_steps:,} | Effective Batch: {effective_batch} ({tokens_per_step:,} tok/step)")
    print(f"  Total Budget: {total_tokens:,} tokens (~{total_tokens/1e6:.2f}M tokens) on {torch.cuda.get_device_name(0)}")
    print("=" * 105)

    vocab_size = 50304
    d_model = 384
    n_layers = 4
    n_heads = 6
    d_ff = 768
    d_hyper = 2048

    if model_type == "hyperspace_budgeted_spawn":
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
            max_experts=32,
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
            ortho_loss_weight=0.005,
            load_bal_weight=0.01
        ).to(device)
    elif model_type == "hyperspace_core_2exp":
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
            ortho_loss_weight=0.005,
            load_bal_weight=0.01
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
            max_seq_len=576
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if "hyperspace" in model_type:
        optimizer = DynamicWarmupAdamW(model.parameters(), lr=lr, weight_decay=0.01, default_group_warmup_steps=40)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    scaler = torch.amp.GradScaler('cuda')
    memory_buffer = TinyExemplarBuffer10D(max_samples_per_domain=256)

    # 10x10 Evaluation Matrices
    loss_matrix = np.zeros((num_domains, num_domains))
    acc_matrix = np.zeros((num_domains, num_domains))

    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_seed_{seed}" if seed != 1337 else ""

    def evaluate_all_10_domains():
        model.eval()
        dom_losses = []
        dom_accs = []
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for d in domains:
                vx, vy = streamer.get_val_batch(d, num_samples=32)
                vx, vy = vx.to(device), vy.to(device)
                total_l = 0.0
                total_correct = 0
                total_tokens = 0
                for i in range(0, vx.shape[0], micro_batch_size):
                    bx = vx[i : i + micro_batch_size]
                    by = vy[i : i + micro_batch_size]
                    if "hyperspace" in model_type:
                        v_logits, _, _ = model(bx, allow_spawning=False)
                    else:
                        v_logits, _, _ = model(bx)
                    loss = F.cross_entropy(v_logits.view(-1, vocab_size), by.view(-1), reduction='sum')
                    total_l += loss.item()
                    total_tokens += by.numel()
                    preds = torch.argmax(v_logits, dim=-1)
                    total_correct += (preds == by).sum().item()
                dom_losses.append(total_l / total_tokens)
                dom_accs.append((total_correct / total_tokens) * 100.0)

        # LIVE GUARDRAIL 1: Mathematical Validity Check (Loss <= ln(vocab_size))
        max_valid_loss = math.log(vocab_size)
        for d_name, d_l in zip(domains, dom_losses):
            if d_l > max_valid_loss or math.isnan(d_l):
                raise AssertionError(f"[MATH VALIDITY ERROR] Domain '{d_name}' loss {d_l:.4f} exceeds uniform bound {max_valid_loss:.4f} or is NaN!")

        return dom_losses, dom_accs

    # Main 10-Phase Training Loop
    t_global_start = time.perf_counter()

    for phase_idx, phase_domain in enumerate(domains):
        dom_title = domain_titles[phase_idx]
        print(f"\n--- Phase {phase_idx+1}/10: [{dom_title}] (Replay Domains: {list(memory_buffer.buffer.keys())}) ---")

        if "hyperspace" in model_type:
            phase_start_expert_counts = [b.hyper_moe.num_experts for b in model.blocks]

        model.train()
        t_phase_start = time.perf_counter()

        for step in range(1, steps_per_phase + 1):
            accum_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            if model_type == "hyperspace_budgeted_spawn":
                curr_counts = [b.hyper_moe.num_experts for b in model.blocks]
                if phase_idx == 0 and step <= 100:
                    # 100-step grace period for Phase 1 to let bootstrap experts anchor
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
                        x, y = streamer.get_batch(phase_domain, split="train")
                else:
                    x, y = streamer.get_batch(phase_domain, split="train")

                x, y = x.to(device), y.to(device)

                if "hyperspace" in model_type:
                    prev_counts = [b.hyper_moe.num_experts for b in model.blocks]
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits, loss, telems = model(x, targets=y, allow_spawning=allow_spawning, is_replay=is_replay)
                        loss = loss / accum_steps
                    scaler.scale(loss).backward()
                    accum_loss += loss.item() * accum_steps

                    # Check for newly spawned experts and attach to DynamicWarmupAdamW
                    new_counts = [b.hyper_moe.num_experts for b in model.blocks]
                    for l_idx, (p_cnt, n_cnt) in enumerate(zip(prev_counts, new_counts)):
                        if n_cnt > p_cnt:
                            for new_exp_idx in range(p_cnt, n_cnt):
                                new_exp_module = model.blocks[l_idx].hyper_moe.experts[new_exp_idx]
                                optimizer.add_dynamic_param_group(
                                    new_exp_module.parameters(),
                                    lr=lr,
                                    weight_decay=0.01,
                                    warmup_steps=40,
                                    group_name=f"layer_{l_idx}_exp_{new_exp_idx}_phase_{phase_idx+1}"
                                )
                                print(f"  🌱 [AUTONOMOUS SPAWN REGISTERED @ PHASE {phase_idx+1} STEP {step}] Layer {l_idx} spawned Expert #{new_exp_idx}! Registered in DynamicWarmupAdamW.", flush=True)
                else:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits, loss, _ = model(x, targets=y)
                        loss = loss / accum_steps
                    scaler.scale(loss).backward()
                    accum_loss += loss.item() * accum_steps

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if step % 60 == 0 or step == steps_per_phase:
                print(f"  [Step {step:>3}/{steps_per_phase}] Mean Loss: {accum_loss:.4f}", flush=True)

        t_phase_end = time.perf_counter()
        phase_time_s = t_phase_end - t_phase_start
        print(f"  [Phase {phase_idx+1} Complete] Elapsed: {phase_time_s:.1f}s ({phase_time_s/60:.2f}m) | Speed: {(steps_per_phase * tokens_per_step)/phase_time_s:.1f} tok/s")

        # Buffer exemplars
        memory_buffer.add_exemplars(phase_domain, streamer, num_samples=256)

        # Evaluate across all 10 domains
        dom_l, dom_a = evaluate_all_10_domains()
        loss_matrix[phase_idx] = dom_l
        acc_matrix[phase_idx] = dom_a

        loss_str = " | ".join([f"{domains[i][:4]}={dom_l[i]:.3f}" for i in range(num_domains)])
        acc_str = " | ".join([f"{domains[i][:4]}={dom_a[i]:.1f}%" for i in range(num_domains)])
        print(f"  >>> Post-Phase {phase_idx+1} Val Losses: {loss_str}")
        print(f"  >>> Post-Phase {phase_idx+1} Val Accs  : {acc_str}")

        # LIVE GUARDRAIL 2: Live Per-Phase 4-Layer Routing Snapshots
        if "hyperspace" in model_type:
            print(f"\n  [LIVE ROUTING SNAPSHOT @ POST-PHASE {phase_idx+1}]")
            model.eval()
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                for d in domains:
                    vx, vy = streamer.get_val_batch(d, num_samples=16)
                    vx = vx.to(device)
                    _, _, telems = model(vx, allow_spawning=False)
                    routing_summary = []
                    for l_idx, telem in enumerate(telems):
                        top_idx = telem['top_indices'].flatten()
                        counts = torch.bincount(top_idx, minlength=model.blocks[l_idx].hyper_moe.num_experts)
                        tot = counts.sum().item()
                        e_str = ", ".join([f"E{e}:{(counts[e].item()/tot)*100:.1f}%" for e in range(min(4, len(counts)))])
                        routing_summary.append(f"L{l_idx}:[{e_str}]")
                    print(f"    {d:<20} -> " + " | ".join(routing_summary), flush=True)

        # LIVE GUARDRAIL 3: Phase-Level Checkpointing
        phase_ckpt_path = os.path.join(out_dir, f"checkpoint_10d_{model_type}{suffix}_phase_{phase_idx+1}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "model_type": model_type,
            "phase_idx": phase_idx,
            "loss_matrix": loss_matrix[:phase_idx+1].tolist(),
            "acc_matrix": acc_matrix[:phase_idx+1].tolist()
        }, phase_ckpt_path)
        print(f"  [SAVED] Phase {phase_idx+1} checkpoint saved to: {phase_ckpt_path}\n", flush=True)

    t_global_end = time.perf_counter()
    total_elapsed_s = t_global_end - t_global_start

    # Compute Backward Transfer across all 9 prior domains
    bwt_deltas = [loss_matrix[9, i] - loss_matrix[i, i] for i in range(9)]
    mean_bwt = float(np.mean(bwt_deltas))

    print("\n" + "=" * 105)
    print(f"  [10-DOMAIN CONTINUAL LEARNING RESULTS: {model_type.upper()} | SEED = {seed}]")
    print(f"  Mean R_BWT (Retention on 9 historical domains): {mean_bwt:+.4f} nats")
    print(f"  Total Run Duration: {total_elapsed_s:.1f}s ({total_elapsed_s/60:.2f} mins / {total_elapsed_s/3600:.2f} hours)")
    print(f"  Average Global Throughput: {total_tokens/total_elapsed_s:.1f} tokens/sec")
    print("=" * 105)

    res_file = os.path.join(out_dir, f"continual_10domain_{model_type}{suffix}_results.json")
    results_data = {
        "model_type": model_type,
        "seed": seed,
        "domains": domains,
        "domain_titles": domain_titles,
        "steps_per_phase": steps_per_phase,
        "total_steps": total_steps,
        "total_tokens": total_tokens,
        "total_elapsed_seconds": total_elapsed_s,
        "mean_throughput_tok_s": total_tokens / total_elapsed_s,
        "mean_bwt_nats": mean_bwt,
        "bwt_deltas_nats": bwt_deltas,
        "loss_matrix": loss_matrix.tolist(),
        "acc_matrix": acc_matrix.tolist(),
        "final_expert_counts": [b.hyper_moe.num_experts for b in model.blocks] if "hyperspace" in model_type else [16]*4
    }

    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"[SAVED] Results saved to: {res_file}")

    final_model_path = os.path.join(out_dir, f"checkpoint_10domain_{model_type}{suffix}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"[SAVED] Final model checkpoint saved to: {final_model_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="hyperspace_budgeted_spawn", choices=["hyperspace_budgeted_spawn", "hyperspace_core_2exp", "static_moe"])
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    run_10domain_experiment(
        model_type=args.model,
        steps_per_phase=args.steps,
        replay_ratio=args.ratio,
        seed=args.seed
    )
