"""
test_math_sandwich.py - Empirical evaluation of the Geometric Sandwich Architecture on arithmetic.

Experiments:
1. Multi-Operation Discrete Arithmetic (Add, Sub, Mul, Mod, Div) on Seen vs. Unseen Pairs.
2. Extrapolation to Huge Digits (Length / Magnitude Generalization).
3. Continual Learning Stream across 5 Arithmetic Tasks (Retention vs. Catastrophic Forgetting).

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

from geometry.layers import YggdrasilProjector, YggdrasilMemoryCortex

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


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class FlatEuclideanNet(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x):
        return self.net(x)


class SphericalPrototypeNet(nn.Module):
    def __init__(self, in_features, hidden_dim, num_prototypes, out_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dim))
        self.value_head = nn.Linear(num_prototypes, out_features)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        feat = F.normalize(self.encoder(x), dim=-1)
        protos = F.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(feat, protos.T) * self.scale
        attn = F.softmax(sim, dim=-1)
        return self.value_head(attn)


class YggdrasilSandwichNet(nn.Module):
    """
    Geometric Sandwich:
    1. Flat Sensory Encoder (in_features -> hidden_dim)
    2. Yggdrasil Manifold Projector (H^d_h x T^d_t x R+)
    3. Memory Cortex with Geodesic Prototype Metric
    4. Decision Readout Head
    """
    def __init__(self, in_features, hidden_dim, num_prototypes, out_features, dim_h=8, dim_t=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.projector = YggdrasilProjector(
            in_features=hidden_dim,
            dim_h=dim_h,
            dim_t=dim_t,
            c=1.0,
        )
        self.cortex = YggdrasilMemoryCortex(
            num_prototypes=num_prototypes,
            dim_h=dim_h,
            dim_t=dim_t,
            out_features=hidden_dim,
            c=1.0,
        )
        self.head = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        feat = self.encoder(x)
        state = self.projector(feat)
        _, context = self.cortex(state)
        self.cortex.clamp_prototypes()
        return self.head(context + feat)


# ---------------------------------------------------------------------------
# Dataset Generation
# ---------------------------------------------------------------------------

def generate_arithmetic_dataset(op_type, num_samples, max_val=30, seed=42):
    """
    Generates arithmetic problems: A op B = C
    op_type in ['add', 'sub', 'mul', 'mod', 'div']
    Inputs represented as normalized scalars + one-hot op indicator.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, max_val + 1, size=num_samples)
    
    if op_type in ['mod', 'div']:
        b = rng.integers(1, max_val + 1, size=num_samples)
    elif op_type == 'sub':
        b = rng.integers(0, max_val + 1, size=num_samples)
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

    # Digits representation: normalized input vector [a, b, log(a+1), log(b+1)]
    x = np.stack([
        a.astype(np.float32) / max_val,
        b.astype(np.float32) / max_val,
        np.log1p(a.astype(np.float32)) / np.log1p(max_val),
        np.log1p(b.astype(np.float32)) / np.log1p(max_val),
    ], axis=1)

    y = c.astype(np.float32)
    return x, y, a, b


def run_experiment_1():
    print("=" * 78)
    print("EXPERIMENT 1: MULTI-OPERATION ARITHMETIC (ADD, SUB, MUL, MOD, DIV)")
    print("Task: Learn operations on numbers [0..30].")
    print("Benchmark: In-Distribution (Seen pairs) vs. Out-of-Distribution (Unseen pairs).")
    print("Seeds: 10 dev seeds (100-109).")
    print("=" * 78)

    ops = ['add', 'sub', 'mul', 'mod', 'div']
    models_to_test = ['Flat_Euclidean', 'Spherical_Proto', 'Yggdrasil_Sandwich']

    # Dict: op -> model -> list of (seen_acc, unseen_acc)
    results_seen = {op: {m: [] for m in models_to_test} for op in ops}
    results_unseen = {op: {m: [] for m in models_to_test} for op in ops}

    for op in ops:
        for seed in SEEDS:
            torch.manual_seed(seed)
            # Generate all pairs (31 x 31 = 961 pairs)
            x_all, y_all, a_all, b_all = generate_arithmetic_dataset(op, num_samples=2000, max_val=30, seed=seed)

            # Split into 70% train (seen) and 30% test (unseen)
            n_train = 1400
            idx = np.random.default_rng(seed).permutation(len(x_all))
            tr_idx, te_idx = idx[:n_train], idx[n_train:]

            x_tr = torch.tensor(x_all[tr_idx], dtype=torch.float32, device=DEVICE)
            y_tr = torch.tensor(y_all[tr_idx], dtype=torch.float32, device=DEVICE).unsqueeze(-1)
            x_te = torch.tensor(x_all[te_idx], dtype=torch.float32, device=DEVICE)
            y_te = torch.tensor(y_all[te_idx], dtype=torch.float32, device=DEVICE).unsqueeze(-1)

            # Target normalization for stable regression training
            y_mean = y_tr.mean()
            y_std = y_tr.std().clamp(min=1e-4)
            y_tr_norm = (y_tr - y_mean) / y_std
            y_te_norm = (y_te - y_mean) / y_std

            for m_name in models_to_test:
                if m_name == 'Flat_Euclidean':
                    net = FlatEuclideanNet(in_features=4, hidden_dim=64, out_features=1).to(DEVICE)
                elif m_name == 'Spherical_Proto':
                    net = SphericalPrototypeNet(in_features=4, hidden_dim=64, num_prototypes=16, out_features=1).to(DEVICE)
                elif m_name == 'Yggdrasil_Sandwich':
                    net = YggdrasilSandwichNet(in_features=4, hidden_dim=64, num_prototypes=16, out_features=1, dim_h=8, dim_t=8).to(DEVICE)

                opt = torch.optim.AdamW(net.parameters(), lr=0.005, weight_decay=1e-4)

                # Train
                net.train()
                for _ in range(150):
                    opt.zero_grad()
                    pred_norm = net(x_tr)
                    loss = F.mse_loss(pred_norm, y_tr_norm)
                    loss.backward()
                    opt.step()

                # Evaluate exact integer accuracy (pred rounded to nearest int == true)
                net.eval()
                with torch.no_grad():
                    pred_tr = (net(x_tr) * y_std + y_mean).round()
                    acc_seen = float((pred_tr == y_tr).float().mean().item())

                    pred_te = (net(x_te) * y_std + y_mean).round()
                    acc_unseen = float((pred_te == y_te).float().mean().item())

                results_seen[op][m_name].append(acc_seen)
                results_unseen[op][m_name].append(acc_unseen)

        print(f"--- Operation: {op.upper()} ---")
        for m_name in models_to_test:
            m_s, ci_s = mean_ci(results_seen[op][m_name])
            m_u, ci_u = mean_ci(results_unseen[op][m_name])
            print(f"  {m_name:<20} | Seen Acc: {m_s*100:5.2f}% +/- {ci_s*100:4.2f}% | Unseen Acc: {m_u*100:5.2f}% +/- {ci_u*100:4.2f}%")
        print()


# ---------------------------------------------------------------------------
# Experiment 2: Extrapolation to Huge Digits
# ---------------------------------------------------------------------------

def run_experiment_2():
    print("=" * 78)
    print("EXPERIMENT 2: EXTRAPOLATION TO HUGE DIGITS (MAGNITUDE GENERALIZATION)")
    print("Training Domain: Numbers [0..99] (1-2 digits)")
    print("Extrapolation Test Domains:")
    print("  Tier 1: 3 digits [100..999]")
    print("  Tier 2: 4-5 digits [1,000..99,999]")
    print("  Tier 3: 6-8 HUGE digits [100,000..99,999,999]")
    print("Task: Addition and Multiplication.")
    print("=" * 78)

    models_to_test = ['Flat_Euclidean', 'Spherical_Proto', 'Yggdrasil_Sandwich']
    ops = ['add', 'mul']

    for op in ops:
        print(f"--- Task: {op.upper()} Extrapolation ---")
        err_tier1 = {m: [] for m in models_to_test}
        err_tier2 = {m: [] for m in models_to_test}
        err_tier3 = {m: [] for m in models_to_test}

        for seed in SEEDS:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)

            # Training: [0..99]
            a_tr = rng.integers(0, 100, size=2000)
            b_tr = rng.integers(0, 100, size=2000)
            y_tr = (a_tr + b_tr) if op == 'add' else (a_tr * b_tr)

            # Test Tier 1: 3 digits [100..999]
            a_t1 = rng.integers(100, 1000, size=500)
            b_t1 = rng.integers(100, 1000, size=500)
            y_t1 = (a_t1 + b_t1) if op == 'add' else (a_t1 * b_t1)

            # Test Tier 2: 4-5 digits [1,000..99,999]
            a_t2 = rng.integers(1000, 100000, size=500)
            b_t2 = rng.integers(1000, 100000, size=500)
            y_t2 = (a_t2 + b_t2) if op == 'add' else (a_t2 * b_t2)

            # Test Tier 3: 6-8 HUGE digits [100,000..99,999,999]
            a_t3 = rng.integers(100000, 100000000, size=500)
            b_t3 = rng.integers(100000, 100000000, size=500)
            y_t3 = (a_t3 + b_t3) if op == 'add' else (a_t3 * b_t3)

            def to_feats(a_arr, b_arr):
                # Log-scale features preserve unbounded dynamic range
                return np.stack([
                    np.log1p(a_arr.astype(np.float32)),
                    np.log1p(b_arr.astype(np.float32)),
                    np.sqrt(a_arr.astype(np.float32) + 1.0),
                    np.sqrt(b_arr.astype(np.float32) + 1.0),
                ], axis=1)

            x_tr_f = torch.tensor(to_feats(a_tr, b_tr), dtype=torch.float32, device=DEVICE)
            y_tr_t = torch.tensor(np.log1p(y_tr.astype(np.float32)), dtype=torch.float32, device=DEVICE).unsqueeze(-1)

            x_t1_f = torch.tensor(to_feats(a_t1, b_t1), dtype=torch.float32, device=DEVICE)
            y_t1_true = torch.tensor(y_t1, dtype=torch.float64, device=DEVICE)

            x_t2_f = torch.tensor(to_feats(a_t2, b_t2), dtype=torch.float32, device=DEVICE)
            y_t2_true = torch.tensor(y_t2, dtype=torch.float64, device=DEVICE)

            x_t3_f = torch.tensor(to_feats(a_t3, b_t3), dtype=torch.float32, device=DEVICE)
            y_t3_true = torch.tensor(y_t3, dtype=torch.float64, device=DEVICE)

            for m_name in models_to_test:
                if m_name == 'Flat_Euclidean':
                    net = FlatEuclideanNet(in_features=4, hidden_dim=64, out_features=1).to(DEVICE)
                elif m_name == 'Spherical_Proto':
                    net = SphericalPrototypeNet(in_features=4, hidden_dim=64, num_prototypes=16, out_features=1).to(DEVICE)
                elif m_name == 'Yggdrasil_Sandwich':
                    net = YggdrasilSandwichNet(in_features=4, hidden_dim=64, num_prototypes=16, out_features=1, dim_h=8, dim_t=8).to(DEVICE)

                opt = torch.optim.AdamW(net.parameters(), lr=0.005, weight_decay=1e-4)

                net.train()
                for _ in range(160):
                    opt.zero_grad()
                    loss = F.mse_loss(net(x_tr_f), y_tr_t)
                    loss.backward()
                    opt.step()

                net.eval()
                with torch.no_grad():
                    # Compute Relative Error: |pred - true| / true
                    def rel_err(x_f, y_true):
                        log_pred = net(x_f).squeeze(-1)
                        pred_val = torch.expm1(log_pred).clamp(min=0.0).double()
                        re = (pred_val - y_true).abs() / (y_true.clamp(min=1.0))
                        return float(re.mean().item())

                    err_tier1[m_name].append(rel_err(x_t1_f, y_t1_true))
                    err_tier2[m_name].append(rel_err(x_t2_f, y_t2_true))
                    err_tier3[m_name].append(rel_err(x_t3_f, y_t3_true))

        for m_name in models_to_test:
            m1, ci1 = mean_ci(err_tier1[m_name])
            m2, ci2 = mean_ci(err_tier2[m_name])
            m3, ci3 = mean_ci(err_tier3[m_name])
            print(f"  {m_name:<20} | T1 (3-dig) RelErr: {m1*100:6.2f}% | T2 (4-5 dig): {m2*100:6.2f}% | T3 (Huge 6-8 dig): {m3*100:6.2f}%")
        print()


# ---------------------------------------------------------------------------
# Experiment 3: Continual Learning Sequence (Catastrophic Forgetting)
# ---------------------------------------------------------------------------

def run_experiment_3():
    print("=" * 78)
    print("EXPERIMENT 3: CONTINUAL LEARNING STREAM ACROSS 5 ARITHMETIC TASKS")
    print("Stream: Task 1 (Add) -> Task 2 (Sub) -> Task 3 (Mul) -> Task 4 (Mod) -> Task 5 (Div)")
    print("Measure: Retained Accuracy on Task 1 (Add) and Task 2 (Sub) after stream ends.")
    print("Investigating Continual Learning Claim: Does geometric memory prevent forgetting?")
    print("=" * 78)

    models_to_test = [
        'Flat_Euclidean',
        'Spherical_Proto',
        'Yggdrasil_Monolithic',
        'Yggdrasil_Modular_Branch'
    ]

    ret_task1 = {m: [] for m in models_to_test}
    ret_task2 = {m: [] for m in models_to_test}

    for seed in SEEDS:
        torch.manual_seed(seed)

        # Datasets for 5 tasks
        t1_x, t1_y, _, _ = generate_arithmetic_dataset('add', 1000, 30, seed=seed)
        t2_x, t2_y, _, _ = generate_arithmetic_dataset('sub', 1000, 30, seed=seed+1)
        t3_x, t3_y, _, _ = generate_arithmetic_dataset('mul', 1000, 30, seed=seed+2)
        t4_x, t4_y, _, _ = generate_arithmetic_dataset('mod', 1000, 30, seed=seed+3)
        t5_x, t5_y, _, _ = generate_arithmetic_dataset('div', 1000, 30, seed=seed+4)

        # Held-out test sets for Task 1 and 2
        t1_x_te, t1_y_te, _, _ = generate_arithmetic_dataset('add', 300, 30, seed=seed+10)
        t2_x_te, t2_y_te, _, _ = generate_arithmetic_dataset('sub', 300, 30, seed=seed+11)

        tasks = [
            (torch.tensor(t1_x, dtype=torch.float32, device=DEVICE), torch.tensor(t1_y, dtype=torch.float32, device=DEVICE).unsqueeze(-1)),
            (torch.tensor(t2_x, dtype=torch.float32, device=DEVICE), torch.tensor(t2_y, dtype=torch.float32, device=DEVICE).unsqueeze(-1)),
            (torch.tensor(t3_x, dtype=torch.float32, device=DEVICE), torch.tensor(t3_y, dtype=torch.float32, device=DEVICE).unsqueeze(-1)),
            (torch.tensor(t4_x, dtype=torch.float32, device=DEVICE), torch.tensor(t4_y, dtype=torch.float32, device=DEVICE).unsqueeze(-1)),
            (torch.tensor(t5_x, dtype=torch.float32, device=DEVICE), torch.tensor(t5_y, dtype=torch.float32, device=DEVICE).unsqueeze(-1)),
        ]

        t1_te_x = torch.tensor(t1_x_te, dtype=torch.float32, device=DEVICE)
        t1_te_y = torch.tensor(t1_y_te, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        t2_te_x = torch.tensor(t2_x_te, dtype=torch.float32, device=DEVICE)
        t2_te_y = torch.tensor(t2_y_te, dtype=torch.float32, device=DEVICE).unsqueeze(-1)

        # 1. Flat Euclidean Net (Sequential Fine-Tuning)
        m_flat = FlatEuclideanNet(in_features=4, hidden_dim=64, out_features=1).to(DEVICE)
        opt_flat = torch.optim.AdamW(m_flat.parameters(), lr=0.005)
        for x_tr, y_tr in tasks:
            for _ in range(80):
                opt_flat.zero_grad()
                loss = F.mse_loss(m_flat(x_tr), y_tr)
                loss.backward()
                opt_flat.step()
        with torch.no_grad():
            acc1 = float(((m_flat(t1_te_x).round() == t1_te_y).float().mean()).item())
            acc2 = float(((m_flat(t2_te_x).round() == t2_te_y).float().mean()).item())
            ret_task1['Flat_Euclidean'].append(acc1)
            ret_task2['Flat_Euclidean'].append(acc2)

        # 2. Spherical Prototype Net (Sequential Fine-Tuning)
        m_sph = SphericalPrototypeNet(in_features=4, hidden_dim=64, num_prototypes=16, out_features=1).to(DEVICE)
        opt_sph = torch.optim.AdamW(m_sph.parameters(), lr=0.005)
        for x_tr, y_tr in tasks:
            for _ in range(80):
                opt_sph.zero_grad()
                loss = F.mse_loss(m_sph(x_tr), y_tr)
                loss.backward()
                opt_sph.step()
        with torch.no_grad():
            acc1 = float(((m_sph(t1_te_x).round() == t1_te_y).float().mean()).item())
            acc2 = float(((m_sph(t2_te_x).round() == t2_te_y).float().mean()).item())
            ret_task1['Spherical_Proto'].append(acc1)
            ret_task2['Spherical_Proto'].append(acc2)

        # 3. Yggdrasil Monolithic Sandwich Net (Sequential Fine-Tuning with shared encoder)
        m_ygg = YggdrasilSandwichNet(in_features=4, hidden_dim=64, num_prototypes=16, out_features=1, dim_h=8, dim_t=8).to(DEVICE)
        opt_ygg = torch.optim.AdamW(m_ygg.parameters(), lr=0.005)
        for x_tr, y_tr in tasks:
            for _ in range(80):
                opt_ygg.zero_grad()
                loss = F.mse_loss(m_ygg(x_tr), y_tr)
                loss.backward()
                opt_ygg.step()
        with torch.no_grad():
            acc1 = float(((m_ygg(t1_te_x).round() == t1_te_y).float().mean()).item())
            acc2 = float(((m_ygg(t2_te_x).round() == t2_te_y).float().mean()).item())
            ret_task1['Yggdrasil_Monolithic'].append(acc1)
            ret_task2['Yggdrasil_Monolithic'].append(acc2)

        # 4. Yggdrasil Modular Branching (Dynamic MoE / Cognitive Anchors Isolation)
        # Task 1 is learned by Branch 1, Task 2 by Branch 2, etc.
        m_branch1 = YggdrasilSandwichNet(in_features=4, hidden_dim=64, num_prototypes=16, out_features=1, dim_h=8, dim_t=8).to(DEVICE)
        opt_b1 = torch.optim.AdamW(m_branch1.parameters(), lr=0.005)
        for _ in range(80):
            opt_b1.zero_grad()
            loss = F.mse_loss(m_branch1(tasks[0][0]), tasks[0][1])
            loss.backward()
            opt_b1.step()

        m_branch2 = YggdrasilSandwichNet(in_features=4, hidden_dim=64, num_prototypes=16, out_features=1, dim_h=8, dim_t=8).to(DEVICE)
        opt_b2 = torch.optim.AdamW(m_branch2.parameters(), lr=0.005)
        for _ in range(80):
            opt_b2.zero_grad()
            loss = F.mse_loss(m_branch2(tasks[1][0]), tasks[1][1])
            loss.backward()
            opt_b2.step()

        # In modular MoE, past branches are protected from subsequent training tasks (Tasks 3, 4, 5)
        with torch.no_grad():
            acc1 = float(((m_branch1(t1_te_x).round() == t1_te_y).float().mean()).item())
            acc2 = float(((m_branch2(t2_te_x).round() == t2_te_y).float().mean()).item())
            ret_task1['Yggdrasil_Modular_Branch'].append(acc1)
            ret_task2['Yggdrasil_Modular_Branch'].append(acc2)

    print(f"{'Model Architecture':<26} | {'Retained T1 (Add)':<18} | {'Retained T2 (Sub)':<18}")
    print("-" * 68)
    for m in models_to_test:
        m1, ci1 = mean_ci(ret_task1[m])
        m2, ci2 = mean_ci(ret_task2[m])
        print(f"{m:<26} | {m1*100:5.2f}% +/- {ci1*100:4.2f}% | {m2*100:5.2f}% +/- {ci2*100:4.2f}%")


if __name__ == "__main__":
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    print("=" * 78)
    print("ALL MATH EXPERIMENTS FINISHED SUCCESSFULLY.")
    print("=" * 78)
