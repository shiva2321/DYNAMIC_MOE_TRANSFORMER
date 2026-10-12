"""
test_growmoe_math.py - Benchmarking PhyloMoE (Dynamic MoE) on Continual Arithmetic.

Compares:
1. Monolithic Fine-Tuning Baseline (B0 FineTune)
2. Joint Replay Baseline (B2 ReplayNet)
3. Architecture A: PhyloMoE (Dynamic MoE with conformal novelty gating and spawning)

Stream: Task 1 (Add) -> Task 2 (Sub) -> Task 3 (Mul) -> Task 4 (Mod) -> Task 5 (Div)
ASCII output only.
"""

import math
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from growmoe.config import PhyloCfg, CommonCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.baselines import FineTune, ReplayNet

SEEDS = [100, 101, 102, 103, 104]
D = 64
N_CLASSES = 64


def mean_ci(arr):
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return float(arr[0]), 0.0
    m = float(np.mean(arr))
    se = float(scipy.stats.sem(arr))
    ci = se * scipy.stats.t.ppf(0.975, df=n - 1)
    return m, ci


def make_math_stream_data(op_type, num_samples, seed=42):
    """
    Generates unit vector inputs and discrete integer class labels for arithmetic operations.
    Inputs are projected onto the D-dimensional unit hypersphere.
    """
    rng = np.random.default_rng(seed)
    max_val = 7
    a = rng.integers(0, max_val + 1, size=num_samples)
    if op_type in ['mod', 'div']:
        b = rng.integers(1, max_val + 1, size=num_samples)
    else:
        b = rng.integers(0, max_val + 1, size=num_samples)

    if op_type == 'add':
        c = a + b  # 0..14
        op_idx = 0
    elif op_type == 'sub':
        c = a - b + max_val  # offset so >= 0: 0..14
        op_idx = 1
    elif op_type == 'mul':
        c = a * b  # 0..49
        op_idx = 2
    elif op_type == 'mod':
        c = a % b  # 0..7
        op_idx = 3
    elif op_type == 'div':
        c = a // b  # 0..7
        op_idx = 4
    else:
        raise ValueError()

    # Create distinct directional clustering per operation + continuous features
    # Base direction per operation on the sphere
    op_dirs = rng.standard_normal((5, D))
    op_dirs = op_dirs / np.linalg.norm(op_dirs, axis=-1, keepdims=True)

    # Embed (a, b) into coordinates
    raw_feats = np.zeros((num_samples, D), dtype=np.float32)
    raw_feats[:, 0] = a / max_val
    raw_feats[:, 1] = b / max_val
    raw_feats[:, 2] = np.sin(2 * np.pi * a / 8.0)
    raw_feats[:, 3] = np.cos(2 * np.pi * a / 8.0)
    raw_feats[:, 4] = np.sin(2 * np.pi * b / 8.0)
    raw_feats[:, 5] = np.cos(2 * np.pi * b / 8.0)

    # Blend with operation-specific direction to simulate distinct arithmetic regimes
    noise = rng.standard_normal((num_samples, D)).astype(np.float32) * 0.1
    vecs = 0.7 * op_dirs[op_idx] + 0.3 * raw_feats + noise
    vecs = vecs / np.linalg.norm(vecs, axis=-1, keepdims=True)

    x = torch.tensor(vecs, dtype=torch.float32)
    y = torch.tensor(c, dtype=torch.long)
    return x, y


def run_growmoe_benchmark():
    print("=" * 78)
    print("BENCHMARKING DYNAMIC MOE (PHYLOMOE) ON CONTINUAL ARITHMETIC")
    print("Stream: Task 1 (Add) -> Task 2 (Sub) -> Task 3 (Mul) -> Task 4 (Mod) -> Task 5 (Div)")
    print("Comparison: B0 FineTune vs. B2 ReplayNet vs. PhyloMoE (Dynamic MoE)")
    print("=" * 78)

    ops = ['add', 'sub', 'mul', 'mod', 'div']
    results_t1 = {'B0_FineTune': [], 'B2_Replay': [], 'PhyloMoE': []}
    results_t2 = {'B0_FineTune': [], 'B2_Replay': [], 'PhyloMoE': []}
    spawn_counts = []

    for seed in SEEDS:
        torch.manual_seed(seed)

        # Build stream batches: 5 tasks, 640 samples per task (10 batches of 64)
        tasks_data = [make_math_stream_data(op, 640, seed=seed + i * 10) for i, op in enumerate(ops)]
        t1_test_x, t1_test_y = make_math_stream_data('add', 200, seed=seed + 999)
        t2_test_x, t2_test_y = make_math_stream_data('sub', 200, seed=seed + 998)

        # 1. B0 FineTune
        cfg_b0 = CommonCfg(d=D, hidden=64, depth=1, n_classes=N_CLASSES, lr=3e-3, seed=seed)
        m_b0 = FineTune(cfg_b0)
        for x_t, y_t in tasks_data:
            for b_idx in range(0, len(x_t), 64):
                bx, by = x_t[b_idx:b_idx+64], y_t[b_idx:b_idx+64]
                m_b0.observe(bx, by)
        pred1 = m_b0.predict(t1_test_x).probs.argmax(dim=-1)
        pred2 = m_b0.predict(t2_test_x).probs.argmax(dim=-1)
        results_t1['B0_FineTune'].append(float((pred1 == t1_test_y).float().mean().item()))
        results_t2['B0_FineTune'].append(float((pred2 == t2_test_y).float().mean().item()))

        # 2. B2 ReplayNet (256-sample reservoir)
        cfg_b2 = CommonCfg(d=D, hidden=64, depth=1, n_classes=N_CLASSES, lr=3e-3, replay_size=256, seed=seed)
        m_b2 = ReplayNet(cfg_b2)
        for x_t, y_t in tasks_data:
            for b_idx in range(0, len(x_t), 64):
                bx, by = x_t[b_idx:b_idx+64], y_t[b_idx:b_idx+64]
                m_b2.observe(bx, by)
        pred1 = m_b2.predict(t1_test_x).probs.argmax(dim=-1)
        pred2 = m_b2.predict(t2_test_x).probs.argmax(dim=-1)
        results_t1['B2_Replay'].append(float((pred1 == t1_test_y).float().mean().item()))
        results_t2['B2_Replay'].append(float((pred2 == t2_test_y).float().mean().item()))

        # 3. PhyloMoE (Dynamic MoE)
        cfg_phylo = PhyloCfg(
            d=D,
            hidden=64,
            depth=1,
            n_classes=N_CLASSES,
            lr=3e-3,
            q=0.05,
            m_min=32,
            min_share=0.05,
            pool_cap=512,
            probe_steps=30,
            seed=seed,
            grow=False,
        )
        pool = PhyloMoE(cfg_phylo)
        for x_t, y_t in tasks_data:
            for b_idx in range(0, len(x_t), 64):
                bx, by = x_t[b_idx:b_idx+64], y_t[b_idx:b_idx+64]
                pool.observe(bx, by)

        pred1 = pool.predict(t1_test_x).probs.argmax(dim=-1)
        pred2 = pool.predict(t2_test_x).probs.argmax(dim=-1)
        results_t1['PhyloMoE'].append(float((pred1 == t1_test_y).float().mean().item()))
        results_t2['PhyloMoE'].append(float((pred2 == t2_test_y).float().mean().item()))
        spawn_counts.append(len(pool.experts))

    print(f"\n{'Architecture':<22} | {'Retained T1 (Add)':<18} | {'Retained T2 (Sub)':<18} | {'Active Experts':<14}")
    print("-" * 78)
    for name in ['B0_FineTune', 'B2_Replay', 'PhyloMoE']:
        m1, ci1 = mean_ci(results_t1[name])
        m2, ci2 = mean_ci(results_t2[name])
        exp_str = f"{np.mean(spawn_counts):.1f}" if name == 'PhyloMoE' else "1.0"
        print(f"{name:<22} | {m1*100:5.2f}% +/- {ci1*100:4.2f}% | {m2*100:5.2f}% +/- {ci2*100:4.2f}% | {exp_str:<14}")
    print("=" * 78)


if __name__ == "__main__":
    run_growmoe_benchmark()
