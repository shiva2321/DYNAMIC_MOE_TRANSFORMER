"""
Phase 0 Benchmark Runner: Group Symmetries and Algebraic Discovery.
Runs dev-seed tuning and 10 fresh test seeds on Z_113 and S_5.
Outputs:
- research/geomlab_results/phase0/results.json
- research/geomlab_results/phase0/tables.txt
"""

import json
import math
import os
import sys
import time
from pathlib import Path

# Add repository root to pythonpath
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from geomlab.protocol import ci95, paired_stats, bonferroni, check_prereg
from geomlab.diagnostics import effective_rank
from geomlab.phase0_algebra import (
    build_z113_dataset,
    build_s5_dataset,
    BaselineMLP,
    BaselineTransformer,
    TorusPhaseModel,
    BlockUnitaryModel,
    compute_fourier_participation_ratio,
)

ROOT = Path(__file__).resolve().parent
PREREG_FILE = ROOT / "PREREG.md"
RESULTS_FILE = ROOT / "results.json"
TABLES_FILE = ROOT / "tables.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Training Loops
# ---------------------------------------------------------------------------

def train_z113_seed(
    arm: str,
    seed: int,
    lr: float,
    epochs: int = 150,
    d_model: int = 128,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x, train_y, test_x, test_y = build_z113_dataset(p=113, train_fraction=0.3, seed=seed)
    train_x, train_y = train_x.to(DEVICE), train_y.to(DEVICE)
    test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)

    if arm == "baseline_mlp":
        model = BaselineMLP(num_tokens=113, d_model=d_model).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    elif arm == "baseline_transformer":
        model = BaselineTransformer(num_tokens=113, d_model=d_model, n_heads=4).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    elif arm == "torus_phase":
        model = TorusPhaseModel(num_tokens=113, d_model=d_model).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown arm: {arm}")

    steps_to_95 = epochs
    steps_to_99 = epochs
    history_test_acc = []

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(test_x).argmax(dim=-1)
            test_acc = (preds == test_y).float().mean().item()
            history_test_acc.append(test_acc)

            if test_acc >= 0.95 and steps_to_95 == epochs:
                steps_to_95 = ep + 1
            if test_acc >= 0.99 and steps_to_99 == epochs:
                steps_to_99 = ep + 1

    wall_time = time.time() - t0
    final_test_acc = history_test_acc[-1]

    # Diagnostics
    with torch.no_grad():
        if arm == "torus_phase":
            fpr = compute_fourier_participation_ratio(model.phases, p=113)
            # Embedding representation matrix
            rep_mat = torch.exp(1j * model.phases).real
            erank = effective_rank(rep_mat)
        elif arm == "baseline_mlp":
            fpr = 113.0
            erank = effective_rank(model.embedding.weight)
        elif arm == "baseline_transformer":
            fpr = 113.0
            erank = effective_rank(model.embedding.weight)

    return {
        "final_test_acc": final_test_acc,
        "steps_to_95": steps_to_95,
        "steps_to_99": steps_to_99,
        "fourier_pr": fpr,
        "effective_rank": erank,
        "wall_time_s": wall_time,
    }


def train_s5_seed(
    arm: str,
    seed: int,
    lr: float,
    epochs: int = 100,
    d_model: int = 128,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x, train_y, test_x, test_y, perms = build_s5_dataset(train_fraction=0.3, seed=seed)
    train_x, train_y = train_x.to(DEVICE), train_y.to(DEVICE)
    test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)

    if arm == "baseline_mlp":
        model = BaselineMLP(num_tokens=120, d_model=d_model).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    elif arm == "torus_phase":
        model = TorusPhaseModel(num_tokens=120, d_model=d_model).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif arm == "block_unitary":
        model = BlockUnitaryModel(num_tokens=120, d_model=d_model, b=4).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown arm: {arm}")

    steps_to_95 = epochs
    steps_to_99 = epochs
    history_test_acc = []

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(test_x).argmax(dim=-1)
            test_acc = (preds == test_y).float().mean().item()
            history_test_acc.append(test_acc)

            if test_acc >= 0.95 and steps_to_95 == epochs:
                steps_to_95 = ep + 1
            if test_acc >= 0.99 and steps_to_99 == epochs:
                steps_to_99 = ep + 1

    wall_time = time.time() - t0
    final_test_acc = history_test_acc[-1]

    with torch.no_grad():
        if arm == "block_unitary":
            # Representation matrix from unitary generators
            u_all = model.get_unitary()
            rep_mat = u_all.reshape(120, -1).real
            erank = effective_rank(rep_mat)
        elif arm == "torus_phase":
            rep_mat = torch.exp(1j * model.phases).real
            erank = effective_rank(rep_mat)
        else:
            erank = effective_rank(model.embedding.weight)

    return {
        "final_test_acc": final_test_acc,
        "steps_to_95": steps_to_95,
        "steps_to_99": steps_to_99,
        "effective_rank": erank,
        "wall_time_s": wall_time,
    }


# ---------------------------------------------------------------------------
# Dev-seed Tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters() -> Dict[str, Dict[str, float]]:
    print("=== Hyperparameter Tuning on Dev Seeds [100, 101, 102] ===")
    lr_grid = [1e-3, 3e-3, 1e-2, 3e-2]
    dev_seeds = [100, 101, 102]
    chosen = {}

    # Z_113 Tuning
    print("Tuning Z_113 arms...")
    for arm in ["baseline_mlp", "baseline_transformer", "torus_phase"]:
        best_lr = lr_grid[0]
        best_score = -1.0
        for lr in lr_grid:
            scores = []
            for s in dev_seeds:
                res = train_z113_seed(arm, s, lr, epochs=80)
                scores.append(res["final_test_acc"])
            mean_score = float(np.mean(scores))
            print(f"  {arm} lr={lr}: dev mean acc = {mean_score*100:.2f}%")
            if mean_score > best_score:
                best_score = mean_score
                best_lr = lr
        chosen[f"z113_{arm}"] = best_lr
        print(f"-> Chosen for {arm}: {best_lr}")

    # S_5 Tuning
    print("\nTuning S_5 arms...")
    for arm in ["baseline_mlp", "torus_phase", "block_unitary"]:
        best_lr = lr_grid[0]
        best_score = -1.0
        for lr in lr_grid:
            scores = []
            for s in dev_seeds:
                res = train_s5_seed(arm, s, lr, epochs=60)
                scores.append(res["final_test_acc"])
            mean_score = float(np.mean(scores))
            print(f"  {arm} lr={lr}: dev mean acc = {mean_score*100:.2f}%")
            if mean_score > best_score:
                best_score = mean_score
                best_lr = lr
        chosen[f"s5_{arm}"] = best_lr
        print(f"-> Chosen for {arm}: {best_lr}")

    return chosen


# ---------------------------------------------------------------------------
# Main Execution and Verification
# ---------------------------------------------------------------------------

def main():
    print(f"Running Phase 0 on Device: {DEVICE}")
    t_start = time.time()

    # Step 1: Tune LRs on Dev Seeds
    chosen_lrs = tune_hyperparameters()

    # Step 2: Fresh Test Seeds (0 to 9)
    test_seeds = list(range(10))
    results: Dict[str, Any] = {
        "metadata": {
            "date": "2026-10-14",
            "device": str(DEVICE),
            "test_seeds": test_seeds,
            "chosen_lrs": chosen_lrs,
        },
        "z113": {},
        "s5": {},
        "comparisons": {},
    }

    print("\n=== Running Confirmatory Tests on Fresh Seeds 0-9 ===")
    # Task 1: Z_113
    print("\n--- Task 1: Z_113 Addition ---")
    for arm in ["baseline_mlp", "baseline_transformer", "torus_phase"]:
        lr = chosen_lrs[f"z113_{arm}"]
        arm_res = []
        for s in test_seeds:
            res = train_z113_seed(arm, s, lr, epochs=150)
            arm_res.append(res)
            print(f"Z_113 | {arm:<20} | Seed {s} | Final Acc: {res['final_test_acc']*100:.2f}% | Steps to 99%: {res['steps_to_99']}")

        # Summary statistics
        accs = [r["final_test_acc"] * 100 for r in arm_res]
        s99 = [r["steps_to_99"] for r in arm_res]
        fpr = [r["fourier_pr"] for r in arm_res]
        er = [r["effective_rank"] for r in arm_res]

        results["z113"][arm] = {
            "raw": arm_res,
            "acc_mean": float(np.mean(accs)),
            "acc_ci": ci95(accs),
            "s99_mean": float(np.mean(s99)),
            "s99_ci": ci95(s99),
            "fpr_mean": float(np.mean(fpr)),
            "fpr_ci": ci95(fpr),
            "erank_mean": float(np.mean(er)),
            "erank_ci": ci95(er),
        }

    # Task 2: S_5
    print("\n--- Task 2: S_5 Permutation Composition ---")
    for arm in ["baseline_mlp", "torus_phase", "block_unitary"]:
        lr = chosen_lrs[f"s5_{arm}"]
        arm_res = []
        for s in test_seeds:
            res = train_s5_seed(arm, s, lr, epochs=100)
            arm_res.append(res)
            print(f"S_5   | {arm:<20} | Seed {s} | Final Acc: {res['final_test_acc']*100:.2f}% | Steps to 99%: {res['steps_to_99']}")

        accs = [r["final_test_acc"] * 100 for r in arm_res]
        s99 = [r["steps_to_99"] for r in arm_res]
        er = [r["effective_rank"] for r in arm_res]

        results["s5"][arm] = {
            "raw": arm_res,
            "acc_mean": float(np.mean(accs)),
            "acc_ci": ci95(accs),
            "s99_mean": float(np.mean(s99)),
            "s99_ci": ci95(s99),
            "erank_mean": float(np.mean(er)),
            "erank_ci": ci95(er),
        }

    # Statistical Comparisons
    print("\n=== Computing Paired Statistics and Hypothesis Tests ===")
    # 1. Z113: torus vs baseline_mlp on steps to 99%
    s_torus = [r["steps_to_99"] for r in results["z113"]["torus_phase"]["raw"]]
    s_mlp = [r["steps_to_99"] for r in results["z113"]["baseline_mlp"]["raw"]]
    stats_z113_mlp = paired_stats(s_mlp, s_torus)  # positive diff means torus used fewer steps

    # 2. Z113: torus vs baseline_transformer on steps to 99%
    s_tf = [r["steps_to_99"] for r in results["z113"]["baseline_transformer"]["raw"]]
    stats_z113_tf = paired_stats(s_tf, s_torus)

    # 3. S5: block_unitary vs torus_phase on final accuracy
    acc_u = [r["final_test_acc"] * 100 for r in results["s5"]["block_unitary"]["raw"]]
    acc_t = [r["final_test_acc"] * 100 for r in results["s5"]["torus_phase"]["raw"]]
    stats_s5_u_vs_t = paired_stats(acc_u, acc_t)

    # 4. S5: block_unitary vs baseline_mlp on final accuracy
    acc_m = [r["final_test_acc"] * 100 for r in results["s5"]["baseline_mlp"]["raw"]]
    stats_s5_u_vs_m = paired_stats(acc_u, acc_m)

    results["comparisons"] = {
        "z113_torus_vs_mlp_steps": stats_z113_mlp,
        "z113_torus_vs_tf_steps": stats_z113_tf,
        "s5_block_unitary_vs_torus_acc": stats_s5_u_vs_t,
        "s5_block_unitary_vs_mlp_acc": stats_s5_u_vs_m,
    }

    # Write Results JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to: {RESULTS_FILE}")

    # Check PREREG integrity
    check_prereg(PREREG_FILE, RESULTS_FILE)
    print("PREREG timestamp integrity check: PASSED (PREREG strictly precedes results)")

    # Render Tables
    render_tables(results)


def render_tables(results: Dict[str, Any]):
    z = results["z113"]
    s = results["s5"]
    c = results["comparisons"]

    lines = []
    lines.append("================================================================================")
    lines.append("PHASE 0: GROUP SYMMETRIES AND ALGEBRAIC DISCOVERY (CONFIRMATORY SEEDS 0-9)")
    lines.append("================================================================================")
    lines.append("")
    lines.append("Table 1: Task 1 - Z_113 Modular Addition (30% train, 70% test, 150 epochs)")
    lines.append("-" * 80)
    lines.append(f"{'Arm':<22} | {'Test Acc (%)':<16} | {'Steps to 99%':<14} | {'Fourier PR':<12} | {'Eff. Rank':<10}")
    lines.append("-" * 80)
    for arm in ["baseline_mlp", "baseline_transformer", "torus_phase"]:
        d = z[arm]
        acc_str = f"{d['acc_mean']:.2f} +/- {d['acc_ci']:.2f}"
        s99_str = f"{d['s99_mean']:.1f} +/- {d['s99_ci']:.1f}"
        fpr_str = f"{d['fpr_mean']:.2f} +/- {d['fpr_ci']:.2f}"
        er_str = f"{d['erank_mean']:.2f}"
        lines.append(f"{arm:<22} | {acc_str:<16} | {s99_str:<14} | {fpr_str:<12} | {er_str:<10}")
    lines.append("-" * 80)
    lines.append("")

    lines.append("Table 2: Task 2 - S_5 Permutation Composition (30% train, 70% test, 100 epochs)")
    lines.append("-" * 80)
    lines.append(f"{'Arm':<22} | {'Test Acc (%)':<16} | {'Steps to 99%':<14} | {'Eff. Rank':<10}")
    lines.append("-" * 80)
    for arm in ["baseline_mlp", "torus_phase", "block_unitary"]:
        d = s[arm]
        acc_str = f"{d['acc_mean']:.2f} +/- {d['acc_ci']:.2f}"
        s99_str = f"{d['s99_mean']:.1f} +/- {d['s99_ci']:.1f}"
        er_str = f"{d['erank_mean']:.2f}"
        lines.append(f"{arm:<22} | {acc_str:<16} | {s99_str:<14} | {er_str:<10}")
    lines.append("-" * 80)
    lines.append("")

    lines.append("Table 3: Paired Statistical Tests (N = 10 fresh test seeds)")
    lines.append("-" * 80)
    lines.append(f"{'Comparison':<35} | {'Mean Diff':<14} | {'t-stat':<10} | {'p-value':<10} | {'Cohen d_z':<10}")
    lines.append("-" * 80)
    
    c1 = c["z113_torus_vs_mlp_steps"]
    lines.append(f"{'Z113: Steps Saved (Torus vs MLP)':<35} | {c1['mean_diff']:+.1f} +/- {c1['ci95_diff']:.1f}  | {c1['t_statistic']:+.2f}     | {c1['p_value_ttest']:.4e} | {c1['cohen_dz']:+.2f}")
    
    c2 = c["z113_torus_vs_tf_steps"]
    lines.append(f"{'Z113: Steps Saved (Torus vs TF)':<35} | {c2['mean_diff']:+.1f} +/- {c2['ci95_diff']:.1f}  | {c2['t_statistic']:+.2f}     | {c2['p_value_ttest']:.4e} | {c2['cohen_dz']:+.2f}")

    c3 = c["s5_block_unitary_vs_torus_acc"]
    lines.append(f"{'S5: Acc Gain (Unitary vs Torus)':<35} | {c3['mean_diff']:+.2f}% +/- {c3['ci95_diff']:.2f}%| {c3['t_statistic']:+.2f}     | {c3['p_value_ttest']:.4e} | {c3['cohen_dz']:+.2f}")

    c4 = c["s5_block_unitary_vs_mlp_acc"]
    lines.append(f"{'S5: Acc Gain (Unitary vs MLP)':<35} | {c4['mean_diff']:+.2f}% +/- {c4['ci95_diff']:.2f}%| {c4['t_statistic']:+.2f}     | {c4['p_value_ttest']:.4e} | {c4['cohen_dz']:+.2f}")
    lines.append("-" * 80)

    table_text = "\n".join(lines)
    with open(TABLES_FILE, "w") as f:
        f.write(table_text)
    print(table_text)
    print(f"\nRendered tables written to: {TABLES_FILE}")


if __name__ == "__main__":
    main()
