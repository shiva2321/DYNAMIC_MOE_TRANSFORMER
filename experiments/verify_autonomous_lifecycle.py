"""
Task-ID-Free Autonomous Expert Lifecycle Verification (10 Seeds).

Feeds a continual stream with ZERO task-ID hints and ZERO manual spawn_expert()/freeze_expert() calls:
1. Domain 0 (tokens 32..39)
2. Short 2-step noise burst (tokens 200..207) -> must NOT fragment into a false expert
3. Domain 1 (tokens 96..103) -> must self-freeze Expert 0 and self-spawn Expert 1
4. Domain 2 (tokens 160..167) -> must self-freeze Expert 1 and self-spawn Expert 2

Compares:
- Baseline_NoLifecycle (single plastic expert, no auto_lifecycle)
- dMoE_AutoLifecycle (auto_lifecycle=True, enable_thinking=True)
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from dmoe.real_model import RealSelfStabilizingDMoE


def make_domain_batch(domain_id: int, batch_size: int, seq_len: int, seed: int, device: torch.device):
    g = torch.Generator().manual_seed(seed)
    offsets = [32, 96, 160, 200]
    shifts = [1, 3, 5, 7]
    off = offsets[domain_id]
    sh = shifts[domain_id]
    xb = torch.randint(off, off + 8, (batch_size, seq_len), generator=g)
    yb = ((xb - off + sh) % 8) + off
    return xb.to(device), yb.to(device)


def run_seed(seed: int, domain_steps: int, device: torch.device) -> Dict[str, Dict[str, float]]:
    # Shared evaluation sets for Domains 0, 1, 2
    eval_sets = [
        make_domain_batch(d, 64, 8, seed=9000 + d * 100 + seed, device=device)
        for d in range(3)
    ]

    results: Dict[str, Dict[str, float]] = {}

    # 1. Baseline_NoLifecycle (single plastic expert)
    torch.manual_seed(seed)
    m_base = RealSelfStabilizingDMoE(
        d_model=24, num_heads=2, d_ff=64, initial_layers=1, chunk_size=1,
        enable_thinking=True, max_thinking_steps=2, auto_lifecycle=False,
    ).to(device)
    opt_b = torch.optim.AdamW(m_base.parameters(), lr=5e-3)
    m_base.train()
    for d, steps in [(0, domain_steps), (3, 2), (1, domain_steps), (2, domain_steps)]:
        for st in range(steps):
            xb, yb = make_domain_batch(d, 16, 8, seed=seed * 1000 + d * 200 + st, device=device)
            opt_b.zero_grad()
            out = m_base(xb, force_thinking_steps=2)
            m_base.thinker.compute_thinking_loss(out["thinking_out"], yb)["loss"].backward()
            opt_b.step()
    m_base.eval()
    with torch.no_grad():
        accs_b = [
            (m_base(xe)["output"].squeeze(2).argmax(dim=-1) == ye).float().mean().item()
            for xe, ye in eval_sets
        ]
    results["Baseline_NoLifecycle"] = {
        "acc_d0": accs_b[0],
        "acc_d1": accs_b[1],
        "acc_d2": accs_b[2],
        "mean_acc": sum(accs_b) / 3.0,
        "num_experts": float(len(m_base.layers[0].moe.experts)),
        "false_spawns": 0.0,
    }

    # 2. dMoE_AutoLifecycle (zero manual spawn/freeze calls)
    torch.manual_seed(seed)
    m_auto = RealSelfStabilizingDMoE(
        d_model=24, num_heads=2, d_ff=64, initial_layers=1, chunk_size=1,
        enable_thinking=True, max_thinking_steps=2,
        auto_lifecycle=True, max_experts=6, min_mature_steps=20,
        novel_confirm_steps=3, novel_surprise_threshold=0.42,
        dormant_age=500, prune_age=1000,
    ).to(device)
    m_auto.sequence_router.sim_threshold = 0.55
    opt_a = torch.optim.AdamW(m_auto.parameters(), lr=5e-3)
    m_auto.train()
    false_spawns = 0
    for d, steps in [(0, domain_steps), (3, 2), (1, domain_steps), (2, domain_steps)]:
        for st in range(steps):
            xb, yb = make_domain_batch(d, 16, 8, seed=seed * 1000 + d * 200 + st, device=device)
            opt_a.zero_grad()
            out = m_auto(xb, force_thinking_steps=2)
            if any(ev.startswith("spawned_expert_") for ev in out["lifecycle_events"]):
                if d == 3:
                    false_spawns += 1
                # Differential learning rate: protect consolidated representations (0.05x lr on shared backbone)
                # while giving full plasticity (1.0x lr) to the newly spawned expert & head
                expert_params = [
                    p for layer in m_auto.layers for p in layer.moe.experts[-1].parameters() if p.requires_grad
                ] + [p for p in m_auto.decoder.expert_heads[-1].parameters() if p.requires_grad]
                exp_ids = {id(p) for p in expert_params}
                backbone_params = [p for p in m_auto.parameters() if p.requires_grad and id(p) not in exp_ids]
                opt_a = torch.optim.AdamW([
                    {"params": expert_params, "lr": 5e-3},
                    {"params": backbone_params, "lr": 2.5e-4},
                ])
            m_auto.thinker.compute_thinking_loss(out["thinking_out"], yb)["loss"].backward()
            opt_a.step()

    m_auto.eval()
    with torch.no_grad():
        accs_a = [
            (m_auto(xe)["output"].squeeze(2).argmax(dim=-1) == ye).float().mean().item()
            for xe, ye in eval_sets
        ]
    results["dMoE_AutoLifecycle"] = {
        "acc_d0": accs_a[0],
        "acc_d1": accs_a[1],
        "acc_d2": accs_a[2],
        "mean_acc": sum(accs_a) / 3.0,
        "num_experts": float(len(m_auto.layers[0].moe.experts)),
        "false_spawns": float(false_spawns),
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--domain-steps", type=int, default=45)
    parser.add_argument("--out", type=str, default="results/dmoe_v4_autonomous_lifecycle/results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    all_seeds = []
    for s in range(args.seeds):
        res = run_seed(seed=100 + s, domain_steps=args.domain_steps, device=device)
        all_seeds.append(res)
        b = res["Baseline_NoLifecycle"]
        a = res["dMoE_AutoLifecycle"]
        print(
            f"[Seed {s+1:02d}/{args.seeds:02d}] "
            f"Base(Mean={b['mean_acc']:.4f}, D0={b['acc_d0']:.4f}, Exp={int(b['num_experts'])}) | "
            f"Auto(Mean={a['mean_acc']:.4f}, D0={a['acc_d0']:.4f}, D1={a['acc_d1']:.4f}, D2={a['acc_d2']:.4f}, Exp={int(a['num_experts'])}, FalseSpawns={int(a['false_spawns'])})",
            flush=True,
        )

    summary = {}
    for cond in all_seeds[0]:
        summary[cond] = {}
        for m in all_seeds[0][cond]:
            vals = [r[cond][m] for r in all_seeds]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
            sd = math.sqrt(var)
            summary[cond][m] = {"mean": mean, "sd": sd}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"elapsed_sec": time.time() - t0, "summary": summary, "per_seed": all_seeds}, f, indent=2)

    print("\n=== AUTONOMOUS LIFECYCLE SUMMARY (10 SEEDS) ===")
    for cond, md in summary.items():
        print(
            f"{cond:24s} | Mean={md['mean_acc']['mean']:.4f}+/-{md['mean_acc']['sd']:.4f} "
            f"| D0={md['acc_d0']['mean']:.4f} | D1={md['acc_d1']['mean']:.4f} | D2={md['acc_d2']['mean']:.4f} "
            f"| Experts={md['num_experts']['mean']:.2f} | FalseSpawns={md['false_spawns']['mean']:.2f}"
        )


if __name__ == "__main__":
    main()

