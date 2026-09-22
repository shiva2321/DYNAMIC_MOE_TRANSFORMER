"""
Training Script for Grounded Understanding & Causal Reversibility in Infinite Hyperspace.

Trains the ArithmeticSphericalMoENet with:
1. Grounded Single-Digit Primitives (0..9) for 100% discrete ALU accuracy in System 2.
2. Broad Domain (1..50) for continuous geometric System 1 intuition.
3. Bidirectional Causal Inversion Loss (Pearl Level 3):
   L_causal = || f( f(a, op, b), inv(op), b ) - a ||^2
   Forces the neural network to be inherently self-reversing and consistent.
"""

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arithmetic_engine import NORMALIZATION_SCALE, build_dataset_vectorized
from train_arithmetic import ArithmeticSphericalMoENet
from dynamic_optimizer import DynamicWarmupAdamW
from grounded_perceptor import OPERATOR_SEMANTICS

def generate_grounded_dataset(device: torch.device):
    """
    Generates dataset enriched with single-digit primitives and causal inversion pairs.
    """
    samples = []
    
    # 1. Single-Digit Primitives (0..9 x 0..9) for +, -, *, %
    # Replicated 5x so the network acquires crystal-clear discrete precision
    for _ in range(5):
        for a in range(10):
            for b in range(10):
                samples.append((float(a), "+", float(b), float(a + b)))
                samples.append((float(a), "-", float(b), float(a - b)))
                samples.append((float(a), "*", float(b), float(a * b)))
                if b != 0:
                    samples.append((float(a), "%", float(b), float(a % 5)))

    # 2. General Continuous Range (1..50)
    for _ in range(2500):
        op = random.choice(["+", "-", "*", "%"])
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        if op == "+":
            res = a + b
        elif op == "-":
            res = a - b
        elif op == "*":
            res = a * b
        elif op == "%":
            res = a % 5
        samples.append((float(a), op, float(b), float(res)))

    random.shuffle(samples)
    print(f"[Grounded Data] Generated {len(samples)} grounded samples (primitives + general domain).")
    return build_dataset_vectorized(samples, device=device)

def train_grounded_causal_system(
    epochs: int = 30,
    batch_size: int = 64,
    save_path: str = "artifacts/grounded_model.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Grounded Trainer] Training Grounded Causal Network on {device}...")
    
    dataset = generate_grounded_dataset(device=device)
    N = dataset["X"].size(0)

    model = ArithmeticSphericalMoENet(
        d_model=32,
        d_ff=128,
        hyperspace_dim=8,
        initial_experts=3,
        max_experts=8,
        spawn_threshold=0.38,
        device=device
    ).to(device)

    optimizer = DynamicWarmupAdamW(
        list(model.parameters()),
        lr=0.003,
        weight_decay=1e-5,
        default_group_warmup_steps=25
    )

    criterion = nn.MSELoss()
    history = {"epoch": [], "task_loss": [], "causal_loss": [], "num_experts": []}

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N)
        epoch_task_loss = []
        epoch_causal_loss = []

        for b_start in range(0, N, batch_size):
            b_idx = perm[b_start:b_start + batch_size]
            x_b = dataset["X"][b_idx]
            u_b = dataset["U"][b_idx]
            r_b = dataset["R"][b_idx]
            y_b = dataset["Y"][b_idx]
            batch_samples = [dataset["samples"][i] for i in b_idx.cpu().tolist()]

            optimizer.update_dynamic_schedules()
            optimizer.zero_grad()

            # 1. Forward Task Prediction
            pred, _ = model(x_b, u_b, r_b, optimizer=optimizer)
            loss_task = criterion(pred, y_b)

            # 2. Bidirectional Causal Inversion Loss (on a subset of batch for speed)
            # Take linear operations (+, -) from batch
            causal_pairs = []
            for s_idx, (a_s, op_s, b_s, _) in enumerate(batch_samples[:16]):
                if op_s in ["+", "-"]:
                    inv_op = OPERATOR_SEMANTICS[op_s]["inverse"]
                    cand_y = (pred[s_idx, 0] * NORMALIZATION_SCALE).item()
                    causal_pairs.append((cand_y, inv_op, b_s, a_s))

            if len(causal_pairs) > 0:
                ds_causal = build_dataset_vectorized(causal_pairs, device=device)
                pred_recon, _ = model(ds_causal["X"], ds_causal["U"], ds_causal["R"])
                target_recon = ds_causal["Y"]
                loss_causal = criterion(pred_recon, target_recon)
            else:
                loss_causal = torch.tensor(0.0, device=device)

            total_loss = loss_task + 0.3 * loss_causal
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_task_loss.append(loss_task.item())
            epoch_causal_loss.append(loss_causal.item())

        mean_task = sum(epoch_task_loss) / len(epoch_task_loss)
        mean_causal = sum(epoch_causal_loss) / len(epoch_causal_loss)
        history["epoch"].append(ep)
        history["task_loss"].append(mean_task)
        history["causal_loss"].append(mean_causal)
        history["num_experts"].append(model.moe.num_experts)

        if ep % 2 == 0 or ep == 1 or ep == epochs:
            print(
                f"Epoch {ep:2d}/{epochs} | Task MSE: {mean_task:.4f} | "
                f"Causal Rollback MSE: {mean_causal:.4f} | Experts: {model.moe.num_experts}",
                flush=True
            )

    # Save checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "history": history,
        "config": {"d_model": 32, "d_ff": 128, "hyperspace_dim": 8}
    }, save_path)
    print(f"\n[Grounded Trainer] Training complete! Grounded causal model saved to: {save_path}", flush=True)

    return model, history

if __name__ == "__main__":
    train_grounded_causal_system(epochs=26)
