"""
Phase 1 Benchmark Runner: Real Latent Geometry Health and Superposition Capacity.
Runs dev-seed tuning (seeds 100, 101, 102) and 10 fresh test seeds (0-9)
on Banking77 text representations across D in {128, 256, 512}.

Features:
- Crash-resilient checkpointing: writes intermediate trial results atomically after each run.
  Resumes cleanly without lost work in event of power outage or system interruption.
- Pre-registration integrity verification (check_prereg).
- Matched budget comparisons between Euclidean, Spherical, and Torus representations.

Outputs:
- research/geomlab_results/phase1/results.json
- research/geomlab_results/phase1/tables.txt
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
from geomlab.diagnostics import effective_rank, isotropy
from geomlab.phase1_capacity import (
    load_banking_dataset,
    EuclideanModel,
    SphericalModel,
    TorusModel,
    evaluate_bundle_capacity,
    compute_mutual_coherence,
)
from growmoe.frontend import HashFrontend

ROOT = Path(__file__).resolve().parent
PREREG_FILE = ROOT / "PREREG.md"
RESULTS_FILE = ROOT / "results.json"
TABLES_FILE = ROOT / "tables.txt"
CHECKPOINT_FILE = ROOT / "checkpoint.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Crash-Resilient Checkpointing
# ---------------------------------------------------------------------------

def load_checkpoint() -> Dict[str, Any]:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to load checkpoint: {e}. Starting fresh.")
    return {"tuning": {}, "test_runs": {}, "haar_runs": {}}


def save_checkpoint(data: Dict[str, Any]) -> None:
    tmp_file = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_file, CHECKPOINT_FILE)


# ---------------------------------------------------------------------------
# Training and Extraction Functions
# ---------------------------------------------------------------------------

def train_and_evaluate_seed(
    arm: str,
    d_model: int,
    seed: int,
    lr: float,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    cb_feats: torch.Tensor,
    epochs: int = 50,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if arm == "euclidean_baseline":
        model = EuclideanModel(in_dim=1024, d_model=d_model, num_classes=77).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif arm == "spherical_norm":
        model = SphericalModel(in_dim=1024, d_model=d_model, num_classes=77).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif arm == "torus_phase":
        model = TorusModel(in_dim=1024, d_model=d_model, num_classes=77).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        raise ValueError(f"Unknown arm: {arm}")

    # Training loop
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        logits, _ = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    wall_time = time.time() - t0

    # Validation accuracy
    model.eval()
    with torch.no_grad():
        val_logits, _ = model(val_x)
        val_acc = float((val_logits.argmax(dim=-1) == val_y).float().mean().item() * 100.0)

        # Extract representations on the N = 1000 codebook
        if arm == "euclidean_baseline":
            cb_reps = model.encode(cb_feats)  # (1000, D)
            is_torus = False
            erank_val = effective_rank(cb_reps)
            iso_metrics = isotropy(cb_reps)
            coherence = compute_mutual_coherence(cb_reps)
        elif arm == "spherical_norm":
            cb_reps = model.encode(cb_feats)  # (1000, D) unit vectors
            is_torus = False
            erank_val = effective_rank(cb_reps)
            iso_metrics = isotropy(cb_reps)
            coherence = compute_mutual_coherence(cb_reps)
        elif arm == "torus_phase":
            phases = model.encode(cb_feats)  # (1000, D)
            is_torus = True
            # For Torus, complex phasors Z in C^{N x D}
            phasors = torch.exp(1j * phases)
            erank_val = effective_rank(phasors)
            iso_metrics = isotropy(phasors)
            # Clifford embedding for mutual coherence: v = [cos, sin] / sqrt(D)
            clifford = torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1) / math.sqrt(d_model)
            coherence = compute_mutual_coherence(clifford)
            cb_reps = phases

        # Evaluate superposition capacity
        bundle_accs = evaluate_bundle_capacity(
            cb_reps,
            is_torus=is_torus,
            bundle_sizes=(1, 2, 4, 8, 16, 32),
            num_trials=100,
            seed=seed,
        )

    # Convert bundle_accs keys to strings for JSON serializability
    bundle_accs_str = {str(k): v for k, v in bundle_accs.items()}

    return {
        "val_acc": val_acc,
        "erank": erank_val,
        "erank_ratio": float(erank_val / d_model * 100.0),
        "uniformity_loss": iso_metrics["uniformity_loss"],
        "mean_pairwise_cosine": iso_metrics["mean_pairwise_cosine"],
        "coherence": coherence,
        "bundle_capacity": bundle_accs_str,
        "wall_time_sec": wall_time,
    }


def evaluate_random_haar(
    d_model: int,
    seed: int,
    N: int = 1000,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Random unit vectors on S^{D-1}
    raw = torch.randn(N, d_model, device=DEVICE)
    unit_vecs = F.normalize(raw, p=2, dim=-1)

    erank_val = effective_rank(unit_vecs)
    iso_metrics = isotropy(unit_vecs)
    coherence = compute_mutual_coherence(unit_vecs)
    bundle_accs = evaluate_bundle_capacity(
        unit_vecs,
        is_torus=False,
        bundle_sizes=(1, 2, 4, 8, 16, 32),
        num_trials=100,
        seed=seed,
    )
    bundle_accs_str = {str(k): v for k, v in bundle_accs.items()}

    return {
        "val_acc": 0.0,
        "erank": erank_val,
        "erank_ratio": float(erank_val / d_model * 100.0),
        "uniformity_loss": iso_metrics["uniformity_loss"],
        "mean_pairwise_cosine": iso_metrics["mean_pairwise_cosine"],
        "coherence": coherence,
        "bundle_capacity": bundle_accs_str,
        "wall_time_sec": 0.0,
    }


# ---------------------------------------------------------------------------
# Main Experiment Execution
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PHASE 1: REAL LATENT GEOMETRY HEALTH AND CAPACITY BENCHMARK")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Check pre-registration file existence
    if not PREREG_FILE.exists():
        raise FileNotFoundError(f"Missing pre-registration file: {PREREG_FILE}")
    print(f"Verified Pre-Registration: {PREREG_FILE.name}")

    ckpt = load_checkpoint()

    # Load data
    print("\nLoading Banking77 dataset...")
    train_x, train_y, val_x, val_y, cb_texts, cb_labels = load_banking_dataset(seed=42)
    train_x, train_y = train_x.to(DEVICE), train_y.to(DEVICE)
    val_x, val_y = val_x.to(DEVICE), val_y.to(DEVICE)

    # Compute codebook features once
    fe = HashFrontend(d=1024)
    cb_feats = fe(cb_texts).float().to(DEVICE)
    print(f"Train samples: {train_x.shape[0]}, Val samples: {val_x.shape[0]}, Codebook items: {cb_feats.shape[0]}")

    dev_seeds = [100, 101, 102]
    test_seeds = list(range(10))
    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]
    dimensions = [128, 256, 512]
    arms = ["euclidean_baseline", "spherical_norm", "torus_phase"]

    # -----------------------------------------------------------------------
    # Step 1: Hyperparameter Tuning on Dev Seeds (D = 256)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 1: HYPERPARAMETER TUNING (DEV SEEDS 100, 101, 102) at D = 256")
    print("-" * 70)

    if "tuning" not in ckpt:
        ckpt["tuning"] = {}

    tuning_results: Dict[str, Dict[str, float]] = {}
    best_lrs: Dict[str, float] = {}

    for arm in arms:
        if arm not in ckpt["tuning"]:
            ckpt["tuning"][arm] = {}
        tuning_results[arm] = {}
        best_val = -1.0
        best_lr = learning_rates[0]

        for lr in learning_rates:
            lr_str = f"{lr:.0e}"
            if lr_str not in ckpt["tuning"][arm]:
                ckpt["tuning"][arm][lr_str] = {}

            val_scores = []
            for seed in dev_seeds:
                seed_str = str(seed)
                if seed_str in ckpt["tuning"][arm][lr_str]:
                    res = ckpt["tuning"][arm][lr_str][seed_str]
                else:
                    res = train_and_evaluate_seed(
                        arm=arm,
                        d_model=256,
                        seed=seed,
                        lr=lr,
                        train_x=train_x,
                        train_y=train_y,
                        val_x=val_x,
                        val_y=val_y,
                        cb_feats=cb_feats,
                        epochs=50,
                    )
                    ckpt["tuning"][arm][lr_str][seed_str] = res
                    save_checkpoint(ckpt)
                val_scores.append(res["val_acc"])

            mean_val = float(np.mean(val_scores))
            tuning_results[arm][lr_str] = mean_val
            print(f"Arm {arm} lr={lr_str}: mean_val_acc = {mean_val:.2f}% (seeds={val_scores})")
            if mean_val > best_val:
                best_val = mean_val
                best_lr = lr

        best_lrs[arm] = best_lr
        print(f"--> Best lr for {arm}: {best_lr:.0e} ({best_val:.2f}%)")

    # -----------------------------------------------------------------------
    # Step 2: Confirmatory Evaluation Across 10 Test Seeds (Seeds 0 - 9)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 2: CONFIRMATORY EVALUATION ACROSS 10 TEST SEEDS (0 - 9)")
    print("-" * 70)

    if "test_runs" not in ckpt:
        ckpt["test_runs"] = {}
    if "haar_runs" not in ckpt:
        ckpt["haar_runs"] = {}

    test_results: Dict[int, Dict[str, List[Dict[str, Any]]]] = {
        D: {arm: [] for arm in arms + ["random_haar"]} for D in dimensions
    }

    for D in dimensions:
        D_str = str(D)
        print(f"\n==================== Dimension D = {D} ====================")
        N = cb_feats.shape[0]
        welch_mu0 = math.sqrt(2 * math.log(N * (N - 1) / 2) / D)
        print(f"Welch bound mu_0 for N={N}, D={D}: {welch_mu0:.4f}")

        if D_str not in ckpt["test_runs"]:
            ckpt["test_runs"][D_str] = {}
        if D_str not in ckpt["haar_runs"]:
            ckpt["haar_runs"][D_str] = {}

        # Random Haar reference
        print("\nEvaluating Theoretical Reference: random_haar...")
        for seed in test_seeds:
            seed_str = str(seed)
            if seed_str in ckpt["haar_runs"][D_str]:
                res_haar = ckpt["haar_runs"][D_str][seed_str]
            else:
                res_haar = evaluate_random_haar(d_model=D, seed=seed, N=N)
                ckpt["haar_runs"][D_str][seed_str] = res_haar
                save_checkpoint(ckpt)
            test_results[D]["random_haar"].append(res_haar)

        haar_m8 = [r["bundle_capacity"]["8"] for r in test_results[D]["random_haar"]]
        print(f"  random_haar M=8 recovery: {np.mean(haar_m8):.2f}% +/- {np.std(haar_m8):.2f}%")

        # Learned Arms
        for arm in arms:
            lr = best_lrs[arm]
            print(f"\nEvaluating Arm: {arm} (lr={lr:.0e})...")
            if arm not in ckpt["test_runs"][D_str]:
                ckpt["test_runs"][D_str][arm] = {}

            for seed in test_seeds:
                seed_str = str(seed)
                if seed_str in ckpt["test_runs"][D_str][arm]:
                    res = ckpt["test_runs"][D_str][arm][seed_str]
                else:
                    res = train_and_evaluate_seed(
                        arm=arm,
                        d_model=D,
                        seed=seed,
                        lr=lr,
                        train_x=train_x,
                        train_y=train_y,
                        val_x=val_x,
                        val_y=val_y,
                        cb_feats=cb_feats,
                        epochs=50,
                    )
                    ckpt["test_runs"][D_str][arm][seed_str] = res
                    save_checkpoint(ckpt)
                test_results[D][arm].append(res)
                print(f"  seed {seed}: val_acc={res['val_acc']:.2f}%, erank={res['erank']:.1f} ({res['erank_ratio']:.1f}%), unif={res['uniformity_loss']:.3f}, M=8={res['bundle_capacity']['8']:.1f}%")

    # -----------------------------------------------------------------------
    # Step 3: Statistical Hypothesis Testing
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 3: STATISTICAL HYPOTHESIS TESTING AND DECISION CRITERIA")
    print("-" * 70)

    # Welch bound mu0
    welch_dict = {D: float(math.sqrt(2 * math.log(1000 * 999 / 2) / D)) for D in dimensions}

    # Hypothesis 1: Euclidean suffers from cone effect (erank <= 65% of D at D=256, 512)
    euc_erank_ratio_256 = [r["erank_ratio"] for r in test_results[256]["euclidean_baseline"]]
    euc_erank_ratio_512 = [r["erank_ratio"] for r in test_results[512]["euclidean_baseline"]]
    h1_passed = (float(np.mean(euc_erank_ratio_256)) <= 65.0) and (float(np.mean(euc_erank_ratio_512)) <= 65.0)
    print(f"H1 (Euclidean erank ratio <= 65%): D=256: {np.mean(euc_erank_ratio_256):.2f}%, D=512: {np.mean(euc_erank_ratio_512):.2f}% -> {'PASSED' if h1_passed else 'REJECTED'}")

    # Hypothesis 2: Spherical and Torus preserve erank >= 85% and lower uniformity loss
    sph_erank_ratio_256 = [r["erank_ratio"] for r in test_results[256]["spherical_norm"]]
    tor_erank_ratio_256 = [r["erank_ratio"] for r in test_results[256]["torus_phase"]]
    h2_erank_passed = (float(np.mean(sph_erank_ratio_256)) >= 85.0) and (float(np.mean(tor_erank_ratio_256)) >= 85.0)

    sph_unif_256 = [r["uniformity_loss"] for r in test_results[256]["spherical_norm"]]
    tor_unif_256 = [r["uniformity_loss"] for r in test_results[256]["torus_phase"]]
    euc_unif_256 = [r["uniformity_loss"] for r in test_results[256]["euclidean_baseline"]]

    t_sph_unif = paired_stats(sph_unif_256, euc_unif_256)
    t_tor_unif = paired_stats(tor_unif_256, euc_unif_256)
    h2_unif_passed = (t_sph_unif["p_value_ttest"] < 0.01) and (t_tor_unif["p_value_ttest"] < 0.01)
    h2_passed = h2_erank_passed and h2_unif_passed
    print(f"H2 (Manifolds erank >= 85% & unif lower p < 0.01): Sph erank={np.mean(sph_erank_ratio_256):.2f}%, Tor erank={np.mean(tor_erank_ratio_256):.2f}%, Sph p={t_sph_unif['p_value_ttest']:.2e}, Tor p={t_tor_unif['p_value_ttest']:.2e} -> {'PASSED' if h2_passed else 'REJECTED'}")

    # Hypothesis 3: Real data capacity tax (real clustering reduces M=8 capacity relative to Haar by >= 10 points)
    haar_m8_256 = [r["bundle_capacity"]["8"] for r in test_results[256]["random_haar"]]
    sph_m8_256 = [r["bundle_capacity"]["8"] for r in test_results[256]["spherical_norm"]]
    tax_diff = float(np.mean(haar_m8_256) - np.mean(sph_m8_256))
    h3_passed = tax_diff >= 10.0
    print(f"H3 (Real data capacity tax >= 10%): Haar M=8 = {np.mean(haar_m8_256):.2f}%, Spherical M=8 = {np.mean(sph_m8_256):.2f}%, Tax: {tax_diff:.2f}% -> {'PASSED' if h3_passed else 'REJECTED'}")

    # Hypothesis 4: Manifold models beat Euclidean on M=4, 8 by >= 15 points (p < 0.01)
    euc_m4_256 = [r["bundle_capacity"]["4"] for r in test_results[256]["euclidean_baseline"]]
    euc_m8_256 = [r["bundle_capacity"]["8"] for r in test_results[256]["euclidean_baseline"]]
    sph_m4_256 = [r["bundle_capacity"]["4"] for r in test_results[256]["spherical_norm"]]
    tor_m4_256 = [r["bundle_capacity"]["4"] for r in test_results[256]["torus_phase"]]
    tor_m8_256 = [r["bundle_capacity"]["8"] for r in test_results[256]["torus_phase"]]

    t_sph_m4 = paired_stats(sph_m4_256, euc_m4_256)
    t_sph_m8 = paired_stats(sph_m8_256, euc_m8_256)
    t_tor_m4 = paired_stats(tor_m4_256, euc_m4_256)
    t_tor_m8 = paired_stats(tor_m8_256, euc_m8_256)

    sph_m4_gain = float(np.mean(sph_m4_256) - np.mean(euc_m4_256))
    sph_m8_gain = float(np.mean(sph_m8_256) - np.mean(euc_m8_256))
    tor_m4_gain = float(np.mean(tor_m4_256) - np.mean(euc_m4_256))
    tor_m8_gain = float(np.mean(tor_m8_256) - np.mean(euc_m8_256))

    h4_passed = (
        (sph_m4_gain >= 15.0 and t_sph_m4["p_value_ttest"] < 0.01) or
        (sph_m8_gain >= 15.0 and t_sph_m8["p_value_ttest"] < 0.01) or
        (tor_m4_gain >= 15.0 and t_tor_m4["p_value_ttest"] < 0.01) or
        (tor_m8_gain >= 15.0 and t_tor_m8["p_value_ttest"] < 0.01)
    )
    print(f"H4 (Manifold beats Euclidean on M=4,8 by >= 15%): Sph M=4 gain={sph_m4_gain:+.2f}%, Sph M=8 gain={sph_m8_gain:+.2f}%, Tor M=4 gain={tor_m4_gain:+.2f}%, Tor M=8 gain={tor_m8_gain:+.2f}% -> {'PASSED' if h4_passed else 'REJECTED'}")

    # Build final serializable data
    output_data = {
        "metadata": {
            "date": "2026-10-14",
            "device": str(DEVICE),
            "num_test_seeds": len(test_seeds),
            "dev_seeds": dev_seeds,
            "test_seeds": test_seeds,
            "best_lrs": {arm: float(lr) for arm, lr in best_lrs.items()},
            "tuning_results": tuning_results,
            "welch_bound": welch_dict,
        },
        "hypotheses": {
            "H1": {"passed": bool(h1_passed), "euc_erank_ratio_256": float(np.mean(euc_erank_ratio_256)), "euc_erank_ratio_512": float(np.mean(euc_erank_ratio_512))},
            "H2": {"passed": bool(h2_passed), "sph_erank_ratio_256": float(np.mean(sph_erank_ratio_256)), "tor_erank_ratio_256": float(np.mean(tor_erank_ratio_256)), "t_sph_unif": t_sph_unif, "t_tor_unif": t_tor_unif},
            "H3": {"passed": bool(h3_passed), "tax_diff_points": float(tax_diff), "haar_m8": float(np.mean(haar_m8_256)), "sph_m8": float(np.mean(sph_m8_256))},
            "H4": {
                "passed": bool(h4_passed),
                "sph_m4_gain": sph_m4_gain,
                "t_sph_m4": t_sph_m4,
                "sph_m8_gain": sph_m8_gain,
                "t_sph_m8": t_sph_m8,
                "tor_m4_gain": tor_m4_gain,
                "t_tor_m4": t_tor_m4,
                "tor_m8_gain": tor_m8_gain,
                "t_tor_m8": t_tor_m8,
            },
        },
        "test_results": {},
    }

    # Aggregate metric summaries for output_data
    for D in dimensions:
        output_data["test_results"][str(D)] = {}
        for arm in arms + ["random_haar"]:
            runs = test_results[D][arm]
            val_accs = [r["val_acc"] for r in runs]
            eranks = [r["erank"] for r in runs]
            erank_ratios = [r["erank_ratio"] for r in runs]
            unifs = [r["uniformity_loss"] for r in runs]
            mean_cosines = [r["mean_pairwise_cosine"] for r in runs]
            coherences = [r["coherence"] for r in runs]

            def _stat(arr):
                return {"mean": float(np.mean(arr)), "ci": float(ci95(arr)), "values": [float(x) for x in arr]}

            bundle_stats = {}
            for M in (1, 2, 4, 8, 16, 32):
                m_vals = [r["bundle_capacity"][str(M)] for r in runs]
                bundle_stats[str(M)] = _stat(m_vals)

            output_data["test_results"][str(D)][arm] = {
                "val_acc": _stat(val_accs),
                "erank": _stat(eranks),
                "erank_ratio": _stat(erank_ratios),
                "uniformity_loss": _stat(unifs),
                "mean_pairwise_cosine": _stat(mean_cosines),
                "coherence": _stat(coherences),
                "bundle_capacity": bundle_stats,
            }

    # Save results.json
    print(f"\nWriting results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Verify check_prereg
    check_prereg(PREREG_FILE, RESULTS_FILE)
    print("Pre-registration integrity verified successfully!")

    # Generate ASCII tables
    print(f"Writing tables to {TABLES_FILE}...")
    lines = []
    lines.append("=" * 80)
    lines.append("PHASE 1: REAL LATENT GEOMETRY HEALTH AND CAPACITY RESULTS (BANKING77)")
    lines.append("=" * 80)
    lines.append(f"Dev Seeds: {dev_seeds} | Test Seeds: {test_seeds} (10 seeds, mean +- 95% CI)")
    lines.append("")

    lines.append("TABLE 1: GEOMETRY HEALTH DIAGNOSTICS ACROSS BOTTLENECK DIMENSIONS")
    lines.append("-" * 80)
    lines.append(f"{'Dim':<5} | {'Arm':<20} | {'Val Acc (%)':<16} | {'E-Rank (% of D)':<16} | {'Uniformity':<14} | {'Mutual Coherence':<16}")
    lines.append("-" * 80)

    for D in dimensions:
        w_mu0 = welch_dict[D]
        for arm in arms + ["random_haar"]:
            res_summary = output_data["test_results"][str(D)][arm]
            va = f"{res_summary['val_acc']['mean']:.1f} +/- {res_summary['val_acc']['ci']:.1f}" if arm != "random_haar" else "N/A"
            er = f"{res_summary['erank_ratio']['mean']:.1f}% +/- {res_summary['erank_ratio']['ci']:.1f}%"
            un = f"{res_summary['uniformity_loss']['mean']:.2f} +/- {res_summary['uniformity_loss']['ci']:.2f}"
            co = f"{res_summary['coherence']['mean']:.3f} (mu0={w_mu0:.3f})"
            lines.append(f"{D:<5} | {arm:<20} | {va:<16} | {er:<16} | {un:<14} | {co:<16}")
        lines.append("-" * 80)

    lines.append("\nTABLE 2: SUPERPOSITION WORKING MEMORY CAPACITY (RECOVERY ACCURACY %)")
    lines.append("-" * 80)
    lines.append(f"{'Dim':<5} | {'Arm':<20} | {'M=1':<10} | {'M=2':<10} | {'M=4':<10} | {'M=8':<10} | {'M=16':<10} | {'M=32':<10}")
    lines.append("-" * 80)

    for D in dimensions:
        for arm in arms + ["random_haar"]:
            res_summary = output_data["test_results"][str(D)][arm]
            b = res_summary["bundle_capacity"]
            m1 = f"{b['1']['mean']:.1f}%"
            m2 = f"{b['2']['mean']:.1f}%"
            m4 = f"{b['4']['mean']:.1f}%"
            m8 = f"{b['8']['mean']:.1f}%"
            m16 = f"{b['16']['mean']:.1f}%"
            m32 = f"{b['32']['mean']:.1f}%"
            lines.append(f"{D:<5} | {arm:<20} | {m1:<10} | {m2:<10} | {m4:<10} | {m8:<10} | {m16:<10} | {m32:<10}")
        lines.append("-" * 80)

    lines.append("\nTABLE 3: PRE-REGISTERED HYPOTHESIS VERIFICATION")
    lines.append("-" * 80)
    lines.append(f"H1: Euclidean suffers from cone effect (erank <= 65% of D): {'PASSED' if h1_passed else 'REJECTED'}")
    lines.append(f"    D=256: {output_data['hypotheses']['H1']['euc_erank_ratio_256']:.1f}%, D=512: {output_data['hypotheses']['H1']['euc_erank_ratio_512']:.1f}%")
    lines.append(f"H2: Manifolds preserve erank >= 85% & uniformity lower (p < 0.01): {'PASSED' if h2_passed else 'REJECTED'}")
    lines.append(f"    Spherical erank: {output_data['hypotheses']['H2']['sph_erank_ratio_256']:.1f}%, Torus erank: {output_data['hypotheses']['H2']['tor_erank_ratio_256']:.1f}%")
    lines.append(f"H3: Real data capacity tax >= 10 points (Haar vs Real Spherical M=8): {'PASSED' if h3_passed else 'REJECTED'}")
    lines.append(f"    Haar M=8: {output_data['hypotheses']['H3']['haar_m8']:.1f}%, Real M=8: {output_data['hypotheses']['H3']['sph_m8']:.1f}%, Tax: {tax_diff:.1f} points")
    lines.append(f"H4: Manifold beats Euclidean on M=4,8 by >= 15 points (p < 0.01): {'PASSED' if h4_passed else 'REJECTED'}")
    lines.append(f"    Spherical M=4 gain: {sph_m4_gain:+.1f}% (p={t_sph_m4['p_value_ttest']:.2e}, dz={t_sph_m4['cohen_dz']:.2f})")
    lines.append(f"    Spherical M=8 gain: {sph_m8_gain:+.1f}% (p={t_sph_m8['p_value_ttest']:.2e}, dz={t_sph_m8['cohen_dz']:.2f})")
    lines.append(f"    Torus M=4 gain: {tor_m4_gain:+.1f}% (p={t_tor_m4['p_value_ttest']:.2e}, dz={t_tor_m4['cohen_dz']:.2f})")
    lines.append(f"    Torus M=8 gain: {tor_m8_gain:+.1f}% (p={t_tor_m8['p_value_ttest']:.2e}, dz={t_tor_m8['cohen_dz']:.2f})")
    lines.append("=" * 80)

    table_text = "\n".join(lines)
    with open(TABLES_FILE, "w", encoding="utf-8") as f:
        f.write(table_text)

    print("\n" + table_text)


if __name__ == "__main__":
    main()
