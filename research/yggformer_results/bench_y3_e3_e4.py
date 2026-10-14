"""
bench_y3_e3_e4.py - Execution script for Round Y3 (C2 Re-evaluation):
1. E3 Sweep: Arms [euclid, hyp, hyp_phase, hyp_phase_bias] across head_dim in {8, 16},
   tree depth in {3, 5}, 10 test seeds (0-9). Parameter-matched baselines via match_params.
   Tuned learning rate grid [1e-3, 3e-3, 1e-2, 3e-2] on dev seed 100 with n_steps=500.
2. E4 Numerics: Precision sweep (fp32, fp16, bf16) with safe_radius vs unclipped.
3. Operating point report for all arms.

Outputs saved to research/yggformer_results/e3_results.json.
ASCII output only (cp1252 safe).
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import time
import math
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from yggformer.config import YggCfg
from yggformer.model import YggFormer
from yggformer.params import match_params, count_parameters
from yggformer.report import operating_point
from yggformer.tasks.hierarchy import TreeHierarchyDataset


class ModelWithAdapter(nn.Module):
    """Wraps YggFormer with a linear projection for continuous leaf vectors."""
    def __init__(self, cfg: YggCfg, d_data: int = 64):
        super().__init__()
        self.cfg = cfg
        self.d_data = d_data
        self.leaf_proj = nn.Linear(d_data, cfg.d_model, bias=False)
        self.model = YggFormer(cfg)

    def forward(self, leaf_vecs: torch.Tensor, label_tokens: torch.Tensor, query_vec: torch.Tensor, record_stats: bool = False):
        B, N_ctx, _ = leaf_vecs.shape
        proj_leaves = self.leaf_proj(leaf_vecs)
        label_embeds = self.model.tok_emb(label_tokens)

        # Interleave leaf and label tokens: [leaf_0, label_0, leaf_1, label_1, ..., query_leaf]
        seq = []
        for i in range(N_ctx):
            seq.append(proj_leaves[:, i:i+1, :])
            seq.append(label_embeds[:, i:i+1, :])
        seq.append(self.leaf_proj(query_vec.unsqueeze(1)))

        inputs_embeds = torch.cat(seq, dim=1)
        logits = self.model(inputs_embeds=inputs_embeds, is_causal=True, record_stats=record_stats)
        query_logits = logits[:, -1, :]
        return query_logits


def collate_episodes(episodes: List[Dict], device: torch.device):
    B = len(episodes)
    N_ctx = len(episodes[0]["context_leaf_ids"])
    d_data = episodes[0]["query_vector"].shape[0]

    leaf_vecs = torch.stack([ep["context_vectors"] for ep in episodes]).to(device)
    label_tokens = torch.stack([ep["context_labels"] for ep in episodes]).to(device)
    query_vec = torch.stack([ep["query_vector"] for ep in episodes]).to(device)
    targets = torch.tensor([ep["query_label"] for ep in episodes], dtype=torch.long, device=device)

    return leaf_vecs, label_tokens, query_vec, targets


def train_and_eval(
    arm: str,
    head_dim: int,
    depth: int,
    lr: float,
    seed: int,
    device: torch.device,
    n_steps: int = 500,
    batch_size: int = 16,
    n_eval_episodes: int = 100,
    qk_dtype: str = "float32",
    r_max: float = None,
) -> Tuple[float, Dict, float]:
    """Train on episodes and return (test_accuracy, op_report, peak_mem_mb)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    branching = 2
    cluster_depth = 2
    n_classes = branching ** cluster_depth
    d_data = 64
    d_model = max(head_dim * 2, 32)
    n_heads = max(d_model // head_dim, 1)

    ref_cfg = YggCfg(
        vocab=n_classes,
        d_model=d_model,
        n_layers=2,
        n_heads=n_heads,
        head_dim=head_dim,
        head_geometry="euclid",
        ffn_mult=2.0,
        max_len=64,
        seed=seed,
    )

    target_cfg = YggCfg(
        vocab=n_classes,
        d_model=d_model,
        n_layers=2,
        n_heads=n_heads,
        head_dim=head_dim,
        head_geometry=arm,
        ffn_mult=2.0,
        max_len=64,
        qk_dtype=qk_dtype,
        r_max=r_max,
        seed=seed,
    )

    matched_cfg = match_params(ref_cfg, target_cfg)
    model = ModelWithAdapter(matched_cfg, d_data=d_data).to(device)

    train_ds = TreeHierarchyDataset(depth=depth, branching=branching, cluster_depth=cluster_depth, d_data=d_data, n_samples=n_steps * batch_size, n_context=12, seed=seed)
    test_ds = TreeHierarchyDataset(depth=depth, branching=branching, cluster_depth=cluster_depth, d_data=d_data, n_samples=n_eval_episodes, n_context=12, seed=seed + 10000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    torch.cuda.reset_peak_memory_stats(device)
    model.train()

    # Training loop
    for step in range(n_steps):
        batch_eps = [train_ds[step * batch_size + b] for b in range(batch_size)]
        leaf_vecs, label_tokens, query_vec, targets = collate_episodes(batch_eps, device)

        optimizer.zero_grad()
        logits = model(leaf_vecs, label_tokens, query_vec)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = len(test_ds)

    eval_eps = [test_ds[i] for i in range(min(32, total))]
    e_leaves, e_labels, e_query, e_targets = collate_episodes(eval_eps, device)

    with torch.no_grad():
        test_logits = model(e_leaves, e_labels, e_query, record_stats=True)
        for start_idx in range(0, total, 32):
            chunk = [test_ds[i] for i in range(start_idx, min(start_idx + 32, total))]
            c_leaves, c_labels, c_query, c_targets = collate_episodes(chunk, device)
            c_logits = model(c_leaves, c_labels, c_query)
            preds = c_logits.argmax(dim=-1)
            correct += (preds == c_targets).sum().item()

    acc = correct / total

    # Operating point extraction
    dummy_batch = torch.randint(0, n_classes, (2, 8), device=device)
    op_rep = operating_point(model.model, dummy_batch)

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    del model, optimizer, train_ds, test_ds
    torch.cuda.empty_cache()

    return acc, op_rep, peak_mem_mb


def main():
    print("=" * 70)
    print("STARTING ROUND Y3: E3 SWEEP & E4 NUMERICS (C2 RE-EVALUATION)")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    os.makedirs("research/yggformer_results", exist_ok=True)

    # 1. TUNING LEARNING RATE GRID ON DEV SEED 100
    lr_grid = [1e-3, 3e-3, 1e-2, 3e-2]
    arms = ["euclid", "hyp", "hyp_phase", "hyp_phase_bias"]
    dev_seed = 100

    print("\n--- TUNING LR ON DEV SEED 100 (depth=5, head_dim=16, n_steps=500) ---")
    best_lrs = {}
    for arm in arms:
        best_lr = lr_grid[1]
        best_score = -1.0
        for lr in lr_grid:
            acc, _, _ = train_and_eval(arm, head_dim=16, depth=5, lr=lr, seed=dev_seed, device=device, n_steps=500)
            print(f"  Arm {arm:<15} LR {lr:1.0e} -> Acc {acc*100:.1f}%")
            if acc > best_score:
                best_score = acc
                best_lr = lr
        best_lrs[arm] = best_lr
        print(f"  >> Chosen LR for {arm}: {best_lr:1.0e} (Dev Acc: {best_score*100:.1f}%)")

    # 2. E3 SWEEP OVER TEST SEEDS 0-9
    head_dims = [8, 16]
    depths = [3, 5]
    test_seeds = list(range(10))

    results = {
        "metadata": {
            "device": str(device),
            "test_seeds": test_seeds,
            "dev_seed": dev_seed,
            "lr_grid": lr_grid,
            "best_lrs": {k: float(v) for k, v in best_lrs.items()},
            "head_dims": head_dims,
            "depths": depths,
            "arms": arms,
            "n_steps": 500,
        },
        "sweep": {},
        "operating_points": {},
    }

    t0 = time.time()
    print("\n--- RUNNING E3 SWEEP (10 SEEDS: 0-9, n_steps=500) ---")
    total_configs = len(depths) * len(head_dims) * len(arms)
    cfg_idx = 0

    for depth in depths:
        for hd in head_dims:
            key = f"d{depth}_hd{hd}"
            results["sweep"][key] = {}
            for arm in arms:
                cfg_idx += 1
                lr = best_lrs[arm]
                accs = []
                peaks = []
                op_reps = []

                for seed in test_seeds:
                    acc, op_rep, peak_mb = train_and_eval(arm, head_dim=hd, depth=depth, lr=lr, seed=seed, device=device, n_steps=500)
                    accs.append(acc)
                    peaks.append(peak_mb)
                    op_reps.append(op_rep)

                acc_mean = float(np.mean(accs))
                acc_std = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
                ci95 = 2.262 * (acc_std / math.sqrt(len(accs)))
                peak_mean = float(np.mean(peaks))

                results["sweep"][key][arm] = {
                    "accs": [float(x) for x in accs],
                    "mean": acc_mean,
                    "ci95": ci95,
                    "peak_mb": peak_mean,
                }
                results["operating_points"][f"{key}_{arm}"] = op_reps[0]

                print(f"[{cfg_idx:2d}/{total_configs:2d}] {key:<10} Arm {arm:<15} Acc: {acc_mean*100:5.1f}% +- {ci95*100:4.1f}% (Peak {peak_mean:4.1f}MB)")

    elapsed = time.time() - t0
    results["metadata"]["wall_clock_s"] = elapsed
    print(f"\nE3 Sweep Completed in {elapsed:.1f}s")

    # 3. E4 NUMERICS: PRECISION AND SAFE RADIUS SWEEP
    print("\n--- RUNNING E4 NUMERICS SWEEP (depth=5, head_dim=16, arm=hyp, n_steps=500) ---")
    dtypes = ["float32", "float16", "bfloat16"]
    clip_options = [True, False]
    e4_results = {}

    for dt in dtypes:
        for clip in clip_options:
            clip_label = "clipped" if clip else "unclipped"
            # clip=True uses safe_radius(dtype) via None; clip=False sets huge r_max=1000.0 (unclipped)
            r_max = None if clip else 1000.0
            exp_key = f"{dt}_{clip_label}"

            accs = []
            nan_count = 0
            for seed in range(5):
                try:
                    acc, op_rep, _ = train_and_eval(
                        "hyp", head_dim=16, depth=5, lr=best_lrs["hyp"], seed=seed,
                        device=device, n_steps=500, qk_dtype=dt, r_max=r_max
                    )
                    if math.isnan(acc) or math.isinf(acc):
                        nan_count += 1
                        accs.append(0.0)
                    else:
                        accs.append(acc)
                except Exception as e:
                    nan_count += 1
                    accs.append(0.0)

            mean_acc = float(np.mean(accs))
            e4_results[exp_key] = {
                "dtype": dt,
                "clipped": clip,
                "nan_rate": nan_count / 5.0,
                "mean_acc": mean_acc,
            }
            print(f"  E4 {exp_key:<20} NaN Rate: {nan_count/5.0:3.1f} | Acc: {mean_acc*100:5.1f}%")

    results["e4_numerics"] = e4_results

    with open("research/yggformer_results/e3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults successfully saved to research/yggformer_results/e3_results.json")


if __name__ == "__main__":
    main()
