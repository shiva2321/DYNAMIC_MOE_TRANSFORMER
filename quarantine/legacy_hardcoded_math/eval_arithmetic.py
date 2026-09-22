"""
Evaluation of Arithmetic Generalization and Out-Of-Distribution Reasoning.

Evaluates:
1. In-Distribution (ID) Unseen Arithmetic Generalization
2. Out-Of-Distribution (OOD) Number Extrapolation ([31, 70])
3. Unseen Composite Operation Reasoning
4. Spherical Expert Specialization Heatmap (which expert specialized in which operator?)
5. Generates high-res report plot: artifacts/arithmetic_generalization_report.png
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arithmetic_engine import generate_arithmetic_splits, NORMALIZATION_SCALE
from train_arithmetic import ArithmeticSphericalMoENet

def evaluate_arithmetic_generalization(
    checkpoint_path: str = "artifacts/arithmetic_model.pt",
    output_image: str = "artifacts/arithmetic_generalization_report.png"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Arithmetic Eval] Evaluating model on {device}...")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    history = ckpt["history"]

    # Re-generate test splits
    splits = generate_arithmetic_splits(
        num_train=100, # not used during eval
        num_id_val=500,
        num_ood=500,
        num_composite=500,
        d_model=32,
        hyperspace_dim=8,
        device=device
    )

    val_id = splits["val_id"]
    val_ood = splits["val_ood"]
    val_composite = splits["val_composite"]

    # Rebuild model structure
    final_num_experts = history["num_experts"][-1]
    model = ArithmeticSphericalMoENet(
        d_model=32,
        d_ff=128,
        hyperspace_dim=8,
        initial_experts=final_num_experts,
        max_experts=max(8, final_num_experts),
        device=device
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def evaluate_dataset(dataset, title: str):
        with torch.no_grad():
            pred, meta = model(dataset["X"], dataset["U"], dataset["R"])
            y_true = dataset["Y"] * NORMALIZATION_SCALE  # De-normalize back to original math scale
            y_pred = pred * NORMALIZATION_SCALE

            mse = F.mse_loss(y_pred, y_true).item()
            mae = torch.abs(y_pred - y_true).mean().item()
            # Accuracy within tolerance of +- 1.5 units
            acc_tol = (torch.abs(y_pred - y_true) <= 1.5).float().mean().item() * 100.0

            # Per-operator breakdown
            ops = dataset["ops"]
            per_op = {}
            for op_type in set(ops):
                indices = [i for i, o in enumerate(ops) if o == op_type]
                sub_true = y_true[indices]
                sub_pred = y_pred[indices]
                sub_mae = torch.abs(sub_pred - sub_true).mean().item()
                sub_acc = (torch.abs(sub_pred - sub_true) <= 1.5).float().mean().item() * 100.0
                per_op[op_type] = {"mae": sub_mae, "acc": sub_acc, "count": len(indices)}

            return {
                "title": title,
                "mse": mse,
                "mae": mae,
                "acc_tol": acc_tol,
                "y_true": y_true.squeeze(-1).cpu().numpy(),
                "y_pred": y_pred.squeeze(-1).cpu().numpy(),
                "per_op": per_op,
                "topk_indices": meta["topk_indices"],
                "ops": ops
            }

    print("\n" + "="*70)
    print(">>> 1. IN-DISTRIBUTION UNSEEN GENERALIZATION (Operands in [1, 30])")
    print("="*70)
    id_results = evaluate_dataset(val_id, "In-Distribution (ID)")
    print(f"Overall MAE: {id_results['mae']:.3f} | Accuracy (Tol +-1.5): {id_results['acc_tol']:.1f}%")
    for op, stats in id_results["per_op"].items():
        print(f"  Operator '{op}': MAE = {stats['mae']:.3f} | Accuracy = {stats['acc']:.1f}% ({stats['count']} samples)")

    print("\n" + "="*70)
    print(">>> 2. OUT-OF-DISTRIBUTION NUMBER EXTRAPOLATION (Operands in [31, 70])")
    print("="*70)
    ood_results = evaluate_dataset(val_ood, "Out-Of-Distribution Extrapolation")
    print(f"Overall MAE: {ood_results['mae']:.3f} | Accuracy (Tol +-1.5): {ood_results['acc_tol']:.1f}%")
    for op, stats in ood_results["per_op"].items():
        print(f"  Operator '{op}': MAE = {stats['mae']:.3f} | Accuracy = {stats['acc']:.1f}% ({stats['count']} samples)")

    print("\n" + "="*70)
    print(">>> 3. UNSEEN COMPOSITE OPERATIONS (Chained a*b+c, a+b-c)")
    print("="*70)
    comp_results = evaluate_dataset(val_composite, "Unseen Composite Operations")
    print(f"Overall MAE: {comp_results['mae']:.3f} | Accuracy (Tol +-1.5): {comp_results['acc_tol']:.1f}%")

    # Routing Specialization Matrix: Operator -> Expert Usage
    all_ops = ["+", "-", "*", "%", "composite"]
    num_exp = model.moe.num_experts
    specialization_matrix = np.zeros((len(all_ops), num_exp))

    # Aggregate across ID and Composite
    combined_ops = id_results["ops"] + comp_results["ops"]
    combined_routes = id_results["topk_indices"] + comp_results["topk_indices"]

    for op_name, route in zip(combined_ops, combined_routes):
        op_idx = all_ops.index(op_name)
        for exp_id in route:
            specialization_matrix[op_idx, exp_id] += 1

    # Normalize by row (operator)
    row_sums = specialization_matrix.sum(axis=1, keepdims=True)
    specialization_matrix = specialization_matrix / np.maximum(row_sums, 1e-6)

    print("\n--- Operator-to-Expert Hyperspace Specialization Matrix ---")
    for op_idx, op_name in enumerate(all_ops):
        weights_str = " | ".join([f"E{e}: {specialization_matrix[op_idx, e]:.2f}" for e in range(num_exp)])
        print(f"  Operator {op_name:<10}: {weights_str}")

    # Generate Scientific Report Plot
    render_report_figure(history, id_results, ood_results, specialization_matrix, all_ops, num_exp, output_image)
    return id_results, ood_results, comp_results

def render_report_figure(history, id_res, ood_res, spec_matrix, ops, num_exp, output_image):
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=150)
    fig.suptitle("Arithmetic Generalization & Hyperspherical MoE Specialization Report", fontsize=15, fontweight='bold')

    # 1. Training & Generalization Loss Curves
    epochs = history["epoch"]
    axs[0, 0].plot(epochs, history["train_loss"], color='#3b82f6', label='Training MSE Loss', linewidth=2)
    axs[0, 0].plot(epochs, history["val_id_loss"], color='#10b981', label='Unseen ID Validation Loss', linestyle='--', linewidth=2)
    axs[0, 0].set_title("Training Loss Convergence & Generalization", fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("MSE Loss")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # 2. In-Distribution vs Out-of-Distribution MAE by Operator
    common_ops = ["+", "-", "*", "%"]
    id_maes = [id_res["per_op"].get(op, {}).get("mae", 0.0) for op in common_ops]
    ood_maes = [ood_res["per_op"].get(op, {}).get("mae", 0.0) for op in common_ops]

    x_pos = np.arange(len(common_ops))
    w = 0.35
    axs[0, 1].bar(x_pos - w/2, id_maes, width=w, label='In-Distribution ([1, 30])', color='#10b981', alpha=0.85)
    axs[0, 1].bar(x_pos + w/2, ood_maes, width=w, label='OOD Extrapolation ([31, 70])', color='#f59e0b', alpha=0.85)
    axs[0, 1].set_title("Error by Mathematical Operator (ID vs OOD Extrapolation)", fontsize=11, fontweight='bold')
    axs[0, 1].set_xticks(x_pos)
    axs[0, 1].set_xticklabels(common_ops, fontsize=11, fontweight='bold')
    axs[0, 1].set_ylabel("Mean Absolute Error (Original Scale)")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    # 3. Dynamic Expert Specialization Heatmap
    cax = axs[1, 0].imshow(spec_matrix, aspect='auto', cmap='magma', interpolation='nearest')
    fig.colorbar(cax, ax=axs[1, 0], label='Routing Probability')
    axs[1, 0].set_title("Dendritic Expert Specialization on Hypersphere S^{D-1} x R+", fontsize=11, fontweight='bold')
    axs[1, 0].set_yticks(range(len(ops)))
    axs[1, 0].set_yticklabels(ops, fontsize=10, fontweight='bold')
    axs[1, 0].set_xticks(range(num_exp))
    axs[1, 0].set_xticklabels([f"Expert {e}" for e in range(num_exp)], fontsize=9)
    axs[1, 0].set_xlabel("Dynamically Spawned Dendritic Expert")
    axs[1, 0].set_ylabel("Mathematical Operator")

    # 4. True vs Predicted Parity Plot on Unseen ID Data
    y_true = id_res["y_true"]
    y_pred = id_res["y_pred"]
    # Downsample points for scatter if large
    indices = np.random.choice(len(y_true), size=min(250, len(y_true)), replace=False)
    axs[1, 1].scatter(y_true[indices], y_pred[indices], color='#8b5cf6', alpha=0.7, edgecolors='black', label='Unseen Test Predictions')
    min_v = min(y_true[indices].min(), y_pred[indices].min())
    max_v = max(y_true[indices].max(), y_pred[indices].max())
    axs[1, 1].plot([min_v, max_v], [min_v, max_v], color='#ef4444', linestyle='--', label='Perfect Accuracy Line (y=x)')
    axs[1, 1].set_title("Prediction Fidelity on Unseen Arithmetic Problems", fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel("True Mathematical Result")
    axs[1, 1].set_ylabel("Neural Network Predicted Result")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    plt.close()
    print(f"\n[Arithmetic Eval] Evaluation report figure saved to: {output_image}")

if __name__ == "__main__":
    evaluate_arithmetic_generalization()
