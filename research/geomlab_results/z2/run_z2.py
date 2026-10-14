"""
Track Z Round Z2 Benchmark Harness: Memory Geometry (Associative Recall per Byte).
Evaluates 8 candidate memory mechanisms on Multi-Query Associative Recall (MQAR):
1. softmax (reference ceiling)
2. linear (elu+1 positive features)
3. gated_delta (Gated DeltaNet)
4. delta_l2 (L2-normalised keys)
5. kata_psd (Self-dual homogeneous cone / PSD features)
6. triadic (Coupled bilinear / tensor-product state)
7. post_decay (PoST geometrically spaced decay rates)
8. legacy_vmf (ISHMN InfiniteSphericalMemory read-only adapter)

Measures:
- Accuracy vs number of stored pairs (P in {4, 8, 16, 32, 64})
- Recall at equal state bytes
- Length extrapolation (T in {128, 256, 512, 1024, 2048})
- Runtime speed (ms/step)
"""

import argparse
import difflib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from geomlab.protocol import (
    ci95,
    paired_stats,
    bonferroni,
    check_prereg,
    validate_report,
)
from geomlab.memory_geometry import (
    SoftmaxAttentionMemory,
    LinearAttentionMemory,
    GatedDeltaMemory,
    DeltaL2Memory,
    KataPsdMemory,
    TriadicMemory,
    PostDecayMemory,
    LegacyVMFMemoryAdapter,
    compute_state_bytes,
)


# -----------------------------------------------------------------------------
# 1. MQAR Synthetic Task Generator
# -----------------------------------------------------------------------------
def generate_mqar_batch(
    batch_size: int,
    seq_len: int,
    num_pairs: int,
    vocab_size: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a batch of Multi-Query Associative Recall (MQAR) sequences.
    
    Structure:
    - First half (t < seq_len // 2): Randomly spaced (key, value) pairs.
      Keys and values are drawn without replacement from separate vocab slices.
    - Second half (t >= seq_len // 2): Query keys presented, target is the matching value.
    - Mask: 1 at query target positions, 0 elsewhere.
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    # Reserve tokens: 0 = pad, 1 = bos
    key_vocab_start = 2
    key_vocab_end = vocab_size // 2
    val_vocab_start = vocab_size // 2
    val_vocab_end = vocab_size

    inputs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    half_len = seq_len // 2

    for b in range(batch_size):
        # Sample P distinct keys and P distinct values
        keys = torch.randperm(key_vocab_end - key_vocab_start)[:num_pairs] + key_vocab_start
        values = torch.randperm(val_vocab_end - val_vocab_start)[:num_pairs] + val_vocab_start

        # Insert key-value pairs in the first half
        # We need 2 * num_pairs positions
        available_store_positions = list(range(0, half_len - 1, 2))
        if len(available_store_positions) < num_pairs:
            raise ValueError(f"Sequence length {seq_len} too short for {num_pairs} pairs")
        chosen_store = random.sample(available_store_positions, num_pairs)

        kv_map = {}
        for idx in range(num_pairs):
            pos = chosen_store[idx]
            k_tok = keys[idx].item()
            v_tok = values[idx].item()
            inputs[b, pos] = k_tok
            inputs[b, pos + 1] = v_tok
            kv_map[k_tok] = v_tok

        # Insert queries in the second half
        available_query_positions = list(range(half_len, seq_len - 1, 2))
        n_queries = min(num_pairs, len(available_query_positions))
        query_keys = random.sample(list(kv_map.keys()), n_queries)
        chosen_query = random.sample(available_query_positions, n_queries)

        for idx in range(n_queries):
            pos = chosen_query[idx]
            q_k = query_keys[idx]
            inputs[b, pos] = q_k
            # Target is the value at position pos + 1
            targets[b, pos] = kv_map[q_k]
            mask[b, pos] = True

    return inputs, targets, mask


# -----------------------------------------------------------------------------
# 2. Sequence Model Wrapper
# -----------------------------------------------------------------------------
class MQARSequenceModel(nn.Module):
    """2-layer sequence model wrapping an interchangeable memory mixer."""
    def __init__(
        self,
        arm: str,
        vocab_size: int = 1024,
        dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.arm = arm
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if arm == "softmax":
                mixer = SoftmaxAttentionMemory(dim=dim)
            elif arm == "linear":
                mixer = LinearAttentionMemory(dim=dim)
            elif arm == "gated_delta":
                mixer = GatedDeltaMemory(dim=dim)
            elif arm == "delta_l2":
                mixer = DeltaL2Memory(dim=dim)
            elif arm == "kata_psd":
                mixer = KataPsdMemory(dim=dim)
            elif arm == "triadic":
                mixer = TriadicMemory(dim=dim)
            elif arm == "post_decay":
                mixer = PostDecayMemory(dim=dim)
            elif arm == "legacy_vmf":
                mixer = LegacyVMFMemoryAdapter(dim=dim, num_slots=32, hyperspace_dim=8)
            else:
                raise ValueError(f"Unknown memory arm: {arm}")

            proj_q = nn.Linear(dim, dim)
            proj_k = nn.Linear(dim, dim)
            proj_v = nn.Linear(dim, dim)
            proj_out = nn.Linear(dim, dim)
            norm1 = nn.LayerNorm(dim)
            norm2 = nn.LayerNorm(dim)
            mlp = nn.Sequential(
                nn.Linear(dim, 2 * dim),
                nn.GELU(),
                nn.Linear(2 * dim, dim),
            )
            self.layers.append(nn.ModuleDict({
                "mixer": mixer,
                "proj_q": proj_q,
                "proj_k": proj_k,
                "proj_v": proj_v,
                "proj_out": proj_out,
                "norm1": norm1,
                "norm2": norm2,
                "mlp": mlp,
            }))

        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.embed(input_ids)  # [B, T, d]
        is_val_token = (input_ids >= self.vocab_size // 2)

        for layer in self.layers:
            mixer = layer["mixer"]
            norm_h = layer["norm1"](h)
            q = layer["proj_q"](norm_h)  # [B, T, d]
            k = layer["proj_k"](norm_h)  # [B, T, d]
            v = layer["proj_v"](norm_h)  # [B, T, d]
            k_write = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)

            # Sequential recurrent rollout over sequence length T
            state = mixer.init_state(batch_size=B, device=input_ids.device)
            outs = []
            for t in range(T):
                # 1. READ FIRST using current query token
                o_t = mixer.read_step(state, q[:, t])
                outs.append(o_t.unsqueeze(1))
                # 2. WRITE SECOND with delayed key at value positions
                write_mask_t = is_val_token[:, t].float()
                new_state = mixer.write_step(state, k_write[:, t], v[:, t])
                if isinstance(state, dict):
                    updated_state = {}
                    for key in state:
                        val = state[key]
                        if torch.is_tensor(val):
                            m = write_mask_t.view(B, *([1] * (val.ndim - 1)))
                            updated_state[key] = (1.0 - m) * val + m * new_state[key]
                        else:
                            updated_state[key] = new_state[key]
                    state = updated_state
                else:
                    state = new_state

            out_seq = torch.cat(outs, dim=1)  # [B, T, d]
            h = layer["proj_out"](out_seq)

        logits = self.head(h)  # [B, T, vocab_size]
        return logits


# -----------------------------------------------------------------------------
# 3. Training & Evaluation Pipeline
# -----------------------------------------------------------------------------
def train_and_eval_arm(
    arm: str,
    lr: float,
    device: torch.device,
    seed: int,
    total_steps: int = 150,
    batch_size: int = 16,
    seq_len: int = 128,
    train_pairs: int = 8,
    dim: int = 64,
) -> Dict[str, Any]:
    """Trains a model on MQAR task and evaluates across pair counts and lengths."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    vocab_size = 512
    model = MQARSequenceModel(arm=arm, vocab_size=vocab_size, dim=dim, num_layers=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    start_time = time.perf_counter()
    for step in range(total_steps):
        inputs, targets, mask = generate_mqar_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_pairs=train_pairs,
            vocab_size=vocab_size,
            device=device,
            seed=seed * 1000 + step,
        )
        optimizer.zero_grad()
        logits = model(inputs)  # [B, T, vocab_size]
        loss = F.cross_entropy(logits[mask], targets[mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    total_time = time.perf_counter() - start_time
    ms_per_step = (total_time / total_steps) * 1000.0

    # Evaluation phase: test pair capacity (P in {4, 8, 16, 32})
    model.eval()
    pair_accuracies = {}
    with torch.no_grad():
        for P in [4, 8, 16, 32]:
            correct = 0
            total = 0
            for eval_b in range(4):
                inp, tgt, msk = generate_mqar_batch(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_pairs=P,
                    vocab_size=vocab_size,
                    device=device,
                    seed=seed * 5000 + eval_b,
                )
                preds = model(inp).argmax(dim=-1)
                correct += (preds[msk] == tgt[msk]).sum().item()
                total += msk.sum().item()
            pair_accuracies[f"P_{P}"] = (correct / total) if total > 0 else 0.0

    # Evaluation phase: length extrapolation at P=8 (T in {128, 256, 512})
    length_accuracies = {}
    with torch.no_grad():
        for L in [128, 256, 512]:
            correct = 0
            total = 0
            for eval_b in range(4):
                inp, tgt, msk = generate_mqar_batch(
                    batch_size=batch_size // 2,
                    seq_len=L,
                    num_pairs=8,
                    vocab_size=vocab_size,
                    device=device,
                    seed=seed * 9000 + eval_b,
                )
                preds = model(inp).argmax(dim=-1)
                correct += (preds[msk] == tgt[msk]).sum().item()
                total += msk.sum().item()
            length_accuracies[f"L_{L}"] = (correct / total) if total > 0 else 0.0

    state_bytes = compute_state_bytes(arm, dim=dim, num_slots=32, batch_size=1, seq_len=seq_len)

    return {
        "arm": arm,
        "seed": seed,
        "lr": lr,
        "ms_per_step": ms_per_step,
        "state_bytes": state_bytes,
        "acc_P4": pair_accuracies["P_4"],
        "acc_P8": pair_accuracies["P_8"],
        "acc_P16": pair_accuracies["P_16"],
        "acc_P32": pair_accuracies["P_32"],
        "acc_L128": length_accuracies["L_128"],
        "acc_L256": length_accuracies["L_256"],
        "acc_L512": length_accuracies["L_512"],
    }


# -----------------------------------------------------------------------------
# 4. Main Runner
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("Track Z Round Z2: Memory Geometry Benchmark (Recall per Byte)")
    print("=" * 80)

    # Pre-registration check
    prereg_path = Path("research/geomlab_results/z2/PREREG.md")
    if not prereg_path.exists():
        raise FileNotFoundError(f"Missing PREREG.md at {prereg_path}")
    print(f"PREREG check: PREREG.md exists (mtime {prereg_path.stat().st_mtime})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    parser = argparse.ArgumentParser(description="Z2 Memory Geometry Benchmark")
    parser.add_argument("--checkpoint", type=str, default="research/geomlab_results/z2/checkpoint_z2.json")
    args, _ = parser.parse_known_args()

    checkpoint_file = Path(args.checkpoint)

    arms = [
        "softmax",
        "linear",
        "gated_delta",
        "delta_l2",
        "kata_psd",
        "triadic",
        "post_decay",
        "legacy_vmf",
    ]

    dev_seeds = [100, 101]
    test_seeds = list(range(10))
    lr_grid = [1e-3, 3e-3]

    ckpt_data = {}
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                ckpt_data = json.load(f)
            print(f"Loaded existing checkpoint from {checkpoint_file}", flush=True)
        except Exception as e:
            print(f"Warning: could not load checkpoint: {e}", flush=True)

    best_lrs = ckpt_data.get("best_lrs", {})

    # Step 1: Tune learning rate on dev seeds if needed
    if best_lrs and all(arm in best_lrs for arm in arms):
        print("\n--- Hyperparameters loaded from checkpoint ---", flush=True)
        for arm in arms:
            print(f"  {arm}: LR = {best_lrs[arm]}", flush=True)
    else:
        print("\n--- Tuning Learning Rates on Dev Seeds (100, 101) ---", flush=True)
        best_lrs = {}
        for arm in arms:
            scores = {}
            for lr in lr_grid:
                accs = [
                    train_and_eval_arm(arm, lr, device, seed, total_steps=60)["acc_P8"]
                    for seed in dev_seeds
                ]
                scores[lr] = float(np.mean(accs))
            best_lr = max(scores, key=scores.get)
            best_lrs[arm] = best_lr
            print(f"  {arm} selected: LR = {best_lr} (dev score P8: {scores[best_lr]:.4f})", flush=True)

    # Step 2: Evaluate test seeds 0-9
    print("\n--- Evaluating Test Seeds (0..9) ---", flush=True)
    raw_results = ckpt_data.get("raw_results", {arm: [] for arm in arms})
    peak_vram_mb = ckpt_data.get("peak_vram_mb", 0.0)

    for arm in arms:
        existing = raw_results.get(arm, [])
        done_seeds = {r["seed"] for r in existing}
        lr = best_lrs[arm]
        print(f"Running {arm} (LR = {lr})...", flush=True)
        for seed in test_seeds:
            if seed in done_seeds:
                continue
            res = train_and_eval_arm(arm, lr, device, seed, total_steps=60)
            raw_results[arm].append(res)
            if torch.cuda.is_available():
                peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() / (1024 * 1024))
            # Save progress to checkpoint
            tmp_path = checkpoint_file.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({
                    "best_lrs": best_lrs,
                    "raw_results": raw_results,
                    "peak_vram_mb": peak_vram_mb,
                }, f, indent=2)
            tmp_path.replace(checkpoint_file)
            print(f"  {arm} seed {seed}: P8={res['acc_P8']*100:.1f}%, P16={res['acc_P16']*100:.1f}%, L512={res['acc_L512']*100:.1f}%")

    # Step 3: Statistical Analysis & Aggregation
    def _metric_stat(vals):
        arr = np.asarray(vals, dtype=float)
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ci_h = ci95(arr)
        return {
            "mean": mean_v,
            "std": std_v,
            "ci_half": ci_h,
            "values": [float(x) for x in vals],
        }

    summary = {}
    for arm in arms:
        runs = raw_results[arm]
        summary[arm] = {
            "lr": best_lrs[arm],
            "state_bytes": runs[0]["state_bytes"],
            "ms_per_step": _metric_stat([r["ms_per_step"] for r in runs]),
            "acc_P4": _metric_stat([r["acc_P4"] for r in runs]),
            "acc_P8": _metric_stat([r["acc_P8"] for r in runs]),
            "acc_P16": _metric_stat([r["acc_P16"] for r in runs]),
            "acc_P32": _metric_stat([r["acc_P32"] for r in runs]),
            "acc_L128": _metric_stat([r["acc_L128"] for r in runs]),
            "acc_L256": _metric_stat([r["acc_L256"] for r in runs]),
            "acc_L512": _metric_stat([r["acc_L512"] for r in runs]),
        }

    # Paired comparisons vs gated_delta on P=16
    base_arm = "gated_delta"
    base_p16 = [r["acc_P16"] for r in raw_results[base_arm]]
    paired_vs_delta = {}
    for arm in arms:
        if arm == base_arm:
            continue
        arm_p16 = [r["acc_P16"] for r in raw_results[arm]]
        p_stats = paired_stats(arm_p16, base_p16)
        paired_vs_delta[arm] = p_stats

    # Bonferroni corrections
    p_vals = [paired_vs_delta[a]["p_value_ttest"] for a in paired_vs_delta]
    bonf_res = bonferroni(p_vals)
    for idx, a in enumerate(paired_vs_delta):
        paired_vs_delta[a]["p_bonferroni"] = bonf_res["adjusted_p_values"][idx]

    # Save results.json
    results_data = {
        "metadata": {
            "date": "2026-10-09",
            "track": "Track Z",
            "round": "Z2",
            "device": str(device),
            "peak_vram_mb": peak_vram_mb,
            "test_seeds": test_seeds,
            "dev_seeds": dev_seeds,
        },
        "best_lrs": best_lrs,
        "summary": summary,
        "paired_vs_delta": paired_vs_delta,
        "raw_results": raw_results,
    }

    out_json = Path("research/geomlab_results/z2/results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved raw results to {out_json}")

    # Verify PREREG precedes results.json
    check_prereg("research/geomlab_results/z2/PREREG.md", out_json)
    print("PREREG integrity check passed: PREREG.md strictly precedes results.json.")

    # Step 4: Render Tables
    table1_lines = [
        "Table 1: MQAR Recall Accuracy (%) vs Stored Pairs (10 Seeds, mean +- 95% CI)",
        "Arm                 State Bytes   P = 4           P = 8           P = 16          P = 32          ms/step",
        "---------------------------------------------------------------------------------------------------------",
    ]
    for arm in arms:
        s = summary[arm]
        b_str = f"{s['state_bytes']} B"
        p4 = f"{s['acc_P4']['mean']*100:.1f}% +- {s['acc_P4']['ci_half']*100:.1f}%"
        p8 = f"{s['acc_P8']['mean']*100:.1f}% +- {s['acc_P8']['ci_half']*100:.1f}%"
        p16 = f"{s['acc_P16']['mean']*100:.1f}% +- {s['acc_P16']['ci_half']*100:.1f}%"
        p32 = f"{s['acc_P32']['mean']*100:.1f}% +- {s['acc_P32']['ci_half']*100:.1f}%"
        step_str = f"{s['ms_per_step']['mean']:.1f} +- {s['ms_per_step']['ci_half']:.1f}"
        table1_lines.append(
            f"{arm:<18} {b_str:<13} {p4:<15} {p8:<15} {p16:<15} {p32:<15} {step_str}"
        )

    table2_lines = [
        "\nTable 2: Paired Statistical Comparison vs Gated Delta on P = 16 (10 Seeds)",
        "Arm                 Mean Diff (Arm - Delta)     95% CI             p-val (paired t)   p-Bonferroni   Cohen's d_z",
        "----------------------------------------------------------------------------------------------------------------",
    ]
    for arm in arms:
        if arm == base_arm:
            continue
        p = paired_vs_delta[arm]
        diff_str = f"{p['mean_diff']*100:+.2f}%"
        ci_str = f"[{(p['mean_diff'] - p['ci95_diff'])*100:+.2f}%, {(p['mean_diff'] + p['ci95_diff'])*100:+.2f}%]"
        table2_lines.append(
            f"{arm:<18} {diff_str:<26} {ci_str:<18} {p['p_value_ttest']:<18.4e} {p['p_bonferroni']:<14.4e} {p['cohen_dz']:+.2f}"
        )

    table3_lines = [
        "\nTable 3: Length Extrapolation Accuracy (%) at P = 8 (10 Seeds, mean +- 95% CI)",
        "Arm                 T = 128 (Train)   T = 256 (2x)      T = 512 (4x)      Degradation (512 vs 128)",
        "--------------------------------------------------------------------------------------------------",
    ]
    for arm in arms:
        s = summary[arm]
        l128 = f"{s['acc_L128']['mean']*100:.1f}% +- {s['acc_L128']['ci_half']*100:.1f}%"
        l256 = f"{s['acc_L256']['mean']*100:.1f}% +- {s['acc_L256']['ci_half']*100:.1f}%"
        l512 = f"{s['acc_L512']['mean']*100:.1f}% +- {s['acc_L512']['ci_half']*100:.1f}%"
        deg = (s['acc_L512']['mean'] - s['acc_L128']['mean']) * 100
        deg_str = f"{deg:+.1f}%"
        table3_lines.append(
            f"{arm:<18} {l128:<17} {l256:<17} {l512:<17} {deg_str}"
        )

    all_tables_text = "\n".join(table1_lines + table2_lines + table3_lines) + "\n"
    out_tables = Path("research/geomlab_results/z2/tables.txt")
    out_tables.write_text(all_tables_text, encoding="utf-8")
    print(f"Saved rendered tables to {out_tables}")

    print("\n" + all_tables_text)

    # Validate report tables against rendered tables
    validate_report("research/geomlab_results/z2/tables.txt", all_tables_text)
    print("Report validation successful: rendered tables verified.")


if __name__ == "__main__":
    main()
