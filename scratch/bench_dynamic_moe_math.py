"""
bench_dynamic_moe_math.py - Complete Empirical Benchmark on Continual Arithmetic.

Compares:
1. Monolithic Sandwich (YggdrasilSandwichModel): Full parameter drift.
2. Dynamic MoE with Shared Drifting Encoder: Prototypes frozen, encoder drifts.
3. Dynamic MoE with Frozen Base + Dynamic Adapters: True modular isolation (Zero Forgetting).

Stream: Task 1 (Add) -> Task 2 (Sub) -> Task 3 (Mul) -> Task 4 (Mod) -> Task 5 (Div)
Seeds: 10 dev seeds (100-109). Device: NVIDIA GeForce RTX 3060.
ASCII output only (safe for Windows cp1252).
"""

import math
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry.layers import YggdrasilSandwichModel
from geometry.dynamic_moe import DynamicYggdrasilMoE

SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mean_ci(arr):
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return float(arr[0]), 0.0
    m = float(np.mean(arr))
    se = float(scipy.stats.sem(arr))
    ci = se * scipy.stats.t.ppf(0.975, df=n - 1)
    return m, ci


def generate_task_data(op_type, num_samples=1000, max_val=30, seed=42):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, max_val + 1, size=num_samples)
    if op_type in ['mod', 'div']:
        b = rng.integers(1, max_val + 1, size=num_samples)
    else:
        b = rng.integers(0, max_val + 1, size=num_samples)

    if op_type == 'add':
        c = a + b
    elif op_type == 'sub':
        c = a - b
    elif op_type == 'mul':
        c = a * b
    elif op_type == 'mod':
        c = a % b
    elif op_type == 'div':
        c = a // b
    else:
        raise ValueError(f"Unknown op_type: {op_type}")

    x = np.stack([
        a.astype(np.float32) / max_val,
        b.astype(np.float32) / max_val,
        np.log1p(a.astype(np.float32)) / np.log1p(max_val),
        np.log1p(b.astype(np.float32)) / np.log1p(max_val),
    ], axis=1)

    y = c.astype(np.float32)
    return torch.tensor(x, dtype=torch.float32, device=DEVICE), torch.tensor(y, dtype=torch.float32, device=DEVICE).unsqueeze(-1)


def run_benchmark():
    print("=" * 78)
    print("EMPIRICAL CONTINUAL LEARNING BENCHMARK ACROSS 10 SEEDS (100-109)")
    print("Stream: Task 1 (Add) -> Task 2 (Sub) -> Task 3 (Mul) -> Task 4 (Mod) -> Task 5 (Div)")
    print("Comparing Architectures on NVIDIA GeForce RTX 3060:")
    print("  1. Monolithic Sandwich (Full parameter drift)")
    print("  2. Dynamic MoE with Shared Drifting Encoder (Encoder drift ablation)")
    print("  3. Dynamic MoE with Frozen Base + Adapters (True Modular MoE)")
    print("=" * 78)

    ops = ['add', 'sub', 'mul', 'mod', 'div']

    results = {
        "Monolithic_Sandwich": {"T1_init": [], "T1_ret": [], "T2_init": [], "T2_ret": []},
        "Dynamic_MoE_Drifting_Encoder": {"T1_init": [], "T1_ret": [], "T2_init": [], "T2_ret": []},
        "Dynamic_MoE_True_Modular": {"T1_init": [], "T1_ret": [], "T2_init": [], "T2_ret": []},
    }

    for seed in SEEDS:
        torch.manual_seed(seed)

        tasks = [generate_task_data(op, num_samples=800, seed=seed + i) for i, op in enumerate(ops)]
        t1_te_x, t1_te_y = generate_task_data('add', num_samples=300, seed=seed + 100)
        t2_te_x, t2_te_y = generate_task_data('sub', num_samples=300, seed=seed + 200)

        # ------------------------------------------------------------------
        # 1. Monolithic Sandwich
        # ------------------------------------------------------------------
        m_mono = YggdrasilSandwichModel(in_features=4, hidden_dim=64, num_prototypes=16, num_classes=1).to(DEVICE)
        opt_mono = torch.optim.AdamW(m_mono.parameters(), lr=0.005)

        # Train T1
        for _ in range(60):
            opt_mono.zero_grad()
            F.mse_loss(m_mono(tasks[0][0]), tasks[0][1]).backward()
            opt_mono.step()
        with torch.no_grad():
            results["Monolithic_Sandwich"]["T1_init"].append(float((m_mono(t1_te_x).round() == t1_te_y).float().mean().item()))

        # Train T2
        for _ in range(60):
            opt_mono.zero_grad()
            F.mse_loss(m_mono(tasks[1][0]), tasks[1][1]).backward()
            opt_mono.step()
        with torch.no_grad():
            results["Monolithic_Sandwich"]["T2_init"].append(float((m_mono(t2_te_x).round() == t2_te_y).float().mean().item()))

        # Train T3, T4, T5
        for t_idx in [2, 3, 4]:
            for _ in range(60):
                opt_mono.zero_grad()
                F.mse_loss(m_mono(tasks[t_idx][0]), tasks[t_idx][1]).backward()
                opt_mono.step()

        with torch.no_grad():
            results["Monolithic_Sandwich"]["T1_ret"].append(float((m_mono(t1_te_x).round() == t1_te_y).float().mean().item()))
            results["Monolithic_Sandwich"]["T2_ret"].append(float((m_mono(t2_te_x).round() == t2_te_y).float().mean().item()))

        # ------------------------------------------------------------------
        # 2. Dynamic MoE with Shared Drifting Encoder
        # ------------------------------------------------------------------
        m_drift = DynamicYggdrasilMoE(in_features=4, hidden_dim=64, out_features=1).to(DEVICE)
        m_drift.spawn_expert("task_add")
        opt_d = torch.optim.AdamW(m_drift.parameters(), lr=0.005)

        for _ in range(60):
            opt_d.zero_grad()
            F.mse_loss(m_drift(tasks[0][0], expert_id=0), tasks[0][1]).backward()
            opt_d.step()
        with torch.no_grad():
            results["Dynamic_MoE_Drifting_Encoder"]["T1_init"].append(float((m_drift(t1_te_x, expert_id=0).round() == t1_te_y).float().mean().item()))

        # Spawn T2, freeze expert 0 (encoder remains trainable)
        m_drift.freeze_expert(0)
        m_drift.spawn_expert("task_sub")
        opt_d2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, m_drift.parameters()), lr=0.005)
        for _ in range(60):
            opt_d2.zero_grad()
            F.mse_loss(m_drift(tasks[1][0], expert_id=1), tasks[1][1]).backward()
            opt_d2.step()
        with torch.no_grad():
            results["Dynamic_MoE_Drifting_Encoder"]["T2_init"].append(float((m_drift(t2_te_x, expert_id=1).round() == t2_te_y).float().mean().item()))

        # Train T3, T4, T5 (continually updating shared encoder)
        for t_idx in [2, 3, 4]:
            m_drift.freeze_all_past_experts()
            m_drift.spawn_expert(f"task_{ops[t_idx]}")
            opt_dt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m_drift.parameters()), lr=0.005)
            for _ in range(60):
                opt_dt.zero_grad()
                F.mse_loss(m_drift(tasks[t_idx][0], expert_id=t_idx), tasks[t_idx][1]).backward()
                opt_dt.step()

        with torch.no_grad():
            results["Dynamic_MoE_Drifting_Encoder"]["T1_ret"].append(float((m_drift(t1_te_x, expert_id=0).round() == t1_te_y).float().mean().item()))
            results["Dynamic_MoE_Drifting_Encoder"]["T2_ret"].append(float((m_drift(t2_te_x, expert_id=1).round() == t2_te_y).float().mean().item()))

        # ------------------------------------------------------------------
        # 3. Dynamic MoE with Frozen Base + Adapters (True Modular MoE)
        # ------------------------------------------------------------------
        m_true = DynamicYggdrasilMoE(in_features=4, hidden_dim=64, out_features=1).to(DEVICE)
        e0 = m_true.spawn_expert("task_add")
        opt_t0 = torch.optim.AdamW(m_true.parameters(), lr=0.005)

        for _ in range(60):
            opt_t0.zero_grad()
            F.mse_loss(m_true(tasks[0][0], expert_id=0), tasks[0][1]).backward()
            opt_t0.step()
        with torch.no_grad():
            results["Dynamic_MoE_True_Modular"]["T1_init"].append(float((m_true(t1_te_x, expert_id=0).round() == t1_te_y).float().mean().item()))

        # Freeze Base Encoder and Projector to lock the coordinate manifold!
        for p in m_true.encoder.parameters():
            p.requires_grad = False
        for p in m_true.projector.parameters():
            p.requires_grad = False
        m_true.freeze_expert(0)

        # Train T2 on Expert 1's local adapter and cortex
        e1 = m_true.spawn_expert("task_sub")
        opt_t1 = torch.optim.AdamW(e1.parameters(), lr=0.005)
        for _ in range(60):
            opt_t1.zero_grad()
            F.mse_loss(m_true(tasks[1][0], expert_id=1), tasks[1][1]).backward()
            opt_t1.step()
        with torch.no_grad():
            results["Dynamic_MoE_True_Modular"]["T2_init"].append(float((m_true(t2_te_x, expert_id=1).round() == t2_te_y).float().mean().item()))

        # Train T3, T4, T5 with independent expert branches
        for t_idx in [2, 3, 4]:
            m_true.freeze_all_past_experts()
            new_e = m_true.spawn_expert(f"task_{ops[t_idx]}")
            opt_tk = torch.optim.AdamW(new_e.parameters(), lr=0.005)
            for _ in range(60):
                opt_tk.zero_grad()
                F.mse_loss(m_true(tasks[t_idx][0], expert_id=t_idx), tasks[t_idx][1]).backward()
                opt_tk.step()

        # Evaluate Retained Accuracy
        with torch.no_grad():
            results["Dynamic_MoE_True_Modular"]["T1_ret"].append(float((m_true(t1_te_x, expert_id=0).round() == t1_te_y).float().mean().item()))
            results["Dynamic_MoE_True_Modular"]["T2_ret"].append(float((m_true(t2_te_x, expert_id=1).round() == t2_te_y).float().mean().item()))

    print("\nFINAL MEASURED RESULTS (Mean +/- 95% Confidence Interval across 10 Seeds):")
    print("-" * 88)
    print(f"{'Architecture':<32} | {'Task':<8} | {'Initial Acc':<16} | {'Retained Acc':<16} | {'Retention %':<12}")
    print("-" * 88)

    for arch in ["Monolithic_Sandwich", "Dynamic_MoE_Drifting_Encoder", "Dynamic_MoE_True_Modular"]:
        for t_name, i_key, r_key in [("T1 (Add)", "T1_init", "T1_ret"), ("T2 (Sub)", "T2_init", "T2_ret")]:
            i_m, i_ci = mean_ci(results[arch][i_key])
            r_m, r_ci = mean_ci(results[arch][r_key])
            ret_ratio = (r_m / max(i_m, 1e-6)) * 100.0
            print(f"{arch:<32} | {t_name:<8} | {i_m*100:5.2f}% +/- {i_ci*100:4.2f}% | {r_m*100:5.2f}% +/- {r_ci*100:4.2f}% | {ret_ratio:5.1f}%")
        print("-" * 88)


if __name__ == "__main__":
    run_benchmark()
