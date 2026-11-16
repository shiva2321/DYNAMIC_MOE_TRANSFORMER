"""
10-Seed Evaluation of Dual-Timescale Memory, Three-Level Attention, and Latent World Model.

Compares:
- Baseline_Flat: RealSelfStabilizingDMoE (flat attention, enable_memory=False, enable_three_level_attn=False, enable_world_model=False)
- Full_Cognitive: RealSelfStabilizingDMoE (enable_memory=True, enable_three_level_attn=True, enable_world_model=True, enable_thinking=True)

Benchmarks:
1. Long-Range Associative Recall with 12-step Distractor Gap (seq_len=24):
   Requires recalling 4 key-value pairs stored at positions 0..7 after 12 distractor tokens.
2. Multi-Step Latent World-Model Prediction + Downstream Rule Accuracy:
   Measures 1-step latent cosine similarity (cos_sim_1step) and downstream next-token accuracy
   when trained jointly with `compute_total_loss(out, yb, wm_weight=0.25)`.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from dmoe.real_model import RealSelfStabilizingDMoE


def make_associative_recall_batch(
    batch_size: int,
    seed: int,
    device: torch.device,
    num_pairs: int = 4,
    distractor_len: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build [k0, v0, k1, v1, ..., distractor x 12, q0, q1, q2, q3] (length = 24).
    Keys in [32..39], Values in [48..55], Distractors in [64..71].
    Returns (xb, yb, eval_mask) where eval_mask is True on the 4 query positions.
    """
    g = torch.Generator().manual_seed(seed)
    seq_len = num_pairs * 2 + distractor_len + num_pairs
    xb = torch.zeros(batch_size, seq_len, dtype=torch.long)
    yb = torch.zeros(batch_size, seq_len, dtype=torch.long)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    for b in range(batch_size):
        keys = torch.randperm(8, generator=g)[:num_pairs] + 32
        vals = torch.randint(48, 56, (num_pairs,), generator=g)
        for i in range(num_pairs):
            xb[b, 2 * i] = keys[i]
            xb[b, 2 * i + 1] = vals[i]
            yb[b, 2 * i] = vals[i]
            yb[b, 2 * i + 1] = keys[i]

        d_start = num_pairs * 2
        dist = torch.randint(64, 72, (distractor_len,), generator=g)
        xb[b, d_start : d_start + distractor_len] = dist
        yb[b, d_start : d_start + distractor_len] = dist

        q_perm = torch.randperm(num_pairs, generator=g)
        q_start = d_start + distractor_len
        for j in range(num_pairs):
            qi = int(q_perm[j].item())
            xb[b, q_start + j] = keys[qi]
            yb[b, q_start + j] = vals[qi]
            mask[b, q_start + j] = True

    return xb.to(device), yb.to(device), mask.to(device)


def make_wm_trajectory_batch(batch_size: int, seq_len: int, seed: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Structured stateful token trajectory: x_{t+1} depends on (x_t + x_{t-1}) mod 12."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(batch_size, seq_len + 1, dtype=torch.long)
    x[:, 0] = torch.randint(0, 12, (batch_size,), generator=g)
    x[:, 1] = torch.randint(0, 12, (batch_size,), generator=g)
    for t in range(2, seq_len + 1):
        x[:, t] = (x[:, t - 1] + x[:, t - 2] + 1) % 12
    x = x + 80
    return x[:, :-1].to(device), x[:, 1:].to(device)


def run_seed(seed: int, steps: int, device: torch.device) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}

    for name, flags in [
        (
            "Baseline_Flat",
            dict(
                enable_thinking=True,
                enable_memory=False,
                enable_three_level_attn=False,
                enable_world_model=False,
            ),
        ),
        (
            "Full_Cognitive",
            dict(
                enable_thinking=True,
                enable_memory=True,
                num_ltm_slots=32,
                enable_three_level_attn=True,
                local_window=4,
                chunk_stride=4,
                enable_world_model=True,
                wm_rollout_horizon=2,
            ),
        ),
    ]:
        # 1. Associative Recall Task
        torch.manual_seed(seed)
        m_rec = RealSelfStabilizingDMoE(
            d_model=32,
            num_heads=2,
            d_ff=64,
            initial_layers=1,
            chunk_size=1,
            max_thinking_steps=2,
            **flags,
        ).to(device)
        opt_r = torch.optim.AdamW(m_rec.parameters(), lr=5e-3)
        m_rec.train()
        for st in range(steps):
            xb, yb, mask = make_associative_recall_batch(24, seed=seed * 1000 + st, device=device)
            opt_r.zero_grad()
            out = m_rec(xb, force_thinking_steps=2, targets=yb)
            logits = out["output"].squeeze(2)
            task_loss = F.cross_entropy(logits[mask], yb[mask])
            wm_loss = out["world_model_loss"]["loss"] if out.get("world_model_loss") is not None else 0.0
            loss = task_loss + 0.1 * wm_loss
            loss.backward()
            opt_r.step()

        m_rec.eval()
        with torch.no_grad():
            xe, ye, me = make_associative_recall_batch(64, seed=seed * 9000 + 1, device=device)
            oe = m_rec(xe, force_thinking_steps=2)
            preds = oe["output"].squeeze(2).argmax(dim=-1)
            recall_acc = float((preds[me] == ye[me]).float().mean().item())

        # 2. World-Model Latent Trajectory + Downstream Next-Token Task
        torch.manual_seed(seed + 500)
        m_wm = RealSelfStabilizingDMoE(
            d_model=32,
            num_heads=2,
            d_ff=64,
            initial_layers=1,
            chunk_size=1,
            max_thinking_steps=2,
            **flags,
        ).to(device)
        opt_w = torch.optim.AdamW(m_wm.parameters(), lr=5e-3)
        m_wm.train()
        for st in range(steps):
            xb, yb = make_wm_trajectory_batch(24, 16, seed=seed * 2000 + st, device=device)
            opt_w.zero_grad()
            out = m_wm(xb, force_thinking_steps=2, targets=yb)
            loss_d = m_wm.compute_total_loss(out, yb, wm_weight=0.25)
            loss_d["loss"].backward()
            opt_w.step()

        # Measure world-model 1-step cosine similarity & downstream accuracy
        m_wm.eval()
        with torch.no_grad():
            xt, yt = make_wm_trajectory_batch(64, 16, seed=seed * 8000 + 2, device=device)
            ot = m_wm(xt, force_thinking_steps=2)
            traj_acc = float((ot["output"].squeeze(2).argmax(dim=-1) == yt).float().mean().item())
            if m_wm.world_model is not None:
                m_wm.train()
                ow_tr = m_wm(xt, update_stats=False, force_thinking_steps=2)
                wm_cos = float(ow_tr["world_model_loss"]["cos_sim_1step"].item())
            else:
                wm_cos = 0.0

        results[name] = {
            "recall_acc": recall_acc,
            "traj_acc": traj_acc,
            "wm_cos_1step": wm_cos,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--out", type=str, default="results/dmoe_v4_cognitive/results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    all_seeds: List[Dict[str, Dict[str, float]]] = []
    for s in range(args.seeds):
        res = run_seed(seed=100 + s, steps=args.steps, device=device)
        all_seeds.append(res)
        b = res["Baseline_Flat"]
        f = res["Full_Cognitive"]
        print(
            f"[Seed {s+1:02d}/{args.seeds:02d}] "
            f"Base(Recall={b['recall_acc']:.3f}, Traj={b['traj_acc']:.3f}) | "
            f"Cognitive(Recall={f['recall_acc']:.3f}, Traj={f['traj_acc']:.3f}, WM_cos={f['wm_cos_1step']:.3f})",
            flush=True,
        )

    summary = {}
    for cond in ("Baseline_Flat", "Full_Cognitive"):
        summary[cond] = {}
        for m in ("recall_acc", "traj_acc", "wm_cos_1step"):
            vals = [r[cond][m] for r in all_seeds]
            mean = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1))
            summary[cond][m] = {"mean": mean, "std": sd}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"elapsed_sec": time.time() - t0, "summary": summary, "seeds": all_seeds}, f, indent=2)

    print("\n=== COGNITIVE ARCHITECTURE SUMMARY (10 SEEDS) ===")
    for cond, stats in summary.items():
        print(
            f"{cond:16s} | Recall={stats['recall_acc']['mean']:.3f}+/-{stats['recall_acc']['std']:.3f} "
            f"| TrajAcc={stats['traj_acc']['mean']:.3f}+/-{stats['traj_acc']['std']:.3f} "
            f"| WM_Cos1Step={stats['wm_cos_1step']['mean']:.3f}+/-{stats['wm_cos_1step']['std']:.3f}"
        )


if __name__ == "__main__":
    main()

