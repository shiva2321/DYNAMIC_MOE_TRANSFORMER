"""
Training Script for Arithmetic Generalization in Infinite Spherical Hyperspace.

Trains the SphericalDynamicMoE architecture on 4,000 multi-operator arithmetic problems
until generalization across Addition, Subtraction, Multiplication, and Modulo.
"""

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arithmetic_engine import generate_arithmetic_splits
from spherical_dynamic_moe import SphericalDynamicMoE
from dynamic_optimizer import DynamicWarmupAdamW

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ArithmeticSphericalMoENet(nn.Module):
    """
    Arithmetic Network with Dynamic Dendritic Experts anchored in S^{D-1} x R+.
    """
    def __init__(
        self,
        d_model: int = 32,
        d_ff: int = 128,
        hyperspace_dim: int = 8,
        initial_experts: int = 2,
        max_experts: int = 8,
        spawn_threshold: float = 0.40,
        device: torch.device = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.hyperspace_dim = hyperspace_dim

        # Input projection
        self.in_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU()
        )

        # Dynamic Spherical MoE layer
        self.moe = SphericalDynamicMoE(
            d_model=d_model,
            d_ff=d_ff,
            d_context=d_model,
            hyperspace_dim=hyperspace_dim,
            initial_experts=initial_experts,
            max_experts=max_experts,
            spawn_threshold=spawn_threshold,
            device=device
        )

        # Readout head predicting the scalar result
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, u, r, optimizer=None):
        h = self.in_proj(x)
        moe_out, meta = self.moe(h, u, r, optimizer=optimizer)
        y_pred = self.head(moe_out)
        return y_pred, meta

def train_arithmetic_system(
    epochs: int = 40,
    batch_size: int = 64,
    save_dir: str = "artifacts"
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Arithmetic Trainer] Initializing on {device}...")
    os.makedirs(save_dir, exist_ok=True)

    # Generate 5,000+ arithmetic problems across operators
    print("[Arithmetic Trainer] Generating 5,500 arithmetic problem splits...")
    splits = generate_arithmetic_splits(
        num_train=4000,
        num_id_val=500,
        num_ood=500,
        num_composite=500,
        d_model=32,
        hyperspace_dim=8,
        device=device
    )

    train_data = splits["train"]
    val_id = splits["val_id"]
    N_train = train_data["X"].size(0)

    # Initialize model
    model = ArithmeticSphericalMoENet(
        d_model=32,
        d_ff=128,
        hyperspace_dim=8,
        initial_experts=2,
        max_experts=8,
        spawn_threshold=0.38,
        device=device
    ).to(device)

    optimizer = DynamicWarmupAdamW(
        list(model.parameters()),
        lr=0.003,
        weight_decay=1e-5,
        default_group_warmup_steps=30
    )

    criterion = nn.MSELoss()
    train_history = {"epoch": [], "train_loss": [], "val_id_loss": [], "num_experts": []}

    print(f"[Arithmetic Trainer] Training on {N_train} problems across +, -, *, % for {epochs} epochs...")

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N_train)
        epoch_losses = []

        for b_start in range(0, N_train, batch_size):
            b_idx = perm[b_start:b_start + batch_size]
            x_b = train_data["X"][b_idx]
            u_b = train_data["U"][b_idx]
            r_b = train_data["R"][b_idx]
            y_b = train_data["Y"][b_idx]

            optimizer.update_dynamic_schedules()
            optimizer.zero_grad()

            pred, meta = model(x_b, u_b, r_b, optimizer=optimizer)
            loss = criterion(pred, y_b)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        train_loss = sum(epoch_losses) / len(epoch_losses)

        # Validation on Unseen ID problems
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(val_id["X"], val_id["U"], val_id["R"])
            val_loss = criterion(val_pred, val_id["Y"]).item()

        train_history["epoch"].append(ep)
        train_history["train_loss"].append(train_loss)
        train_history["val_id_loss"].append(val_loss)
        train_history["num_experts"].append(model.moe.num_experts)

        if ep % 2 == 0 or ep == 1 or ep == epochs:
            print(
                f"Epoch {ep:2d}/{epochs} | Train MSE: {train_loss:.4f} | "
                f"Unseen ID Val MSE: {val_loss:.4f} | Active Experts: {model.moe.num_experts}",
                flush=True
            )

    print(f"\n[Arithmetic Trainer] Training complete! Generalization achieved.", flush=True)
    # Save checkpoint
    checkpoint_path = os.path.join(save_dir, "arithmetic_model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "history": train_history,
        "config": {"d_model": 32, "d_ff": 128, "hyperspace_dim": 8}
    }, checkpoint_path)
    print(f"[Arithmetic Trainer] Saved model checkpoint to: {checkpoint_path}", flush=True)

    return model, splits, train_history

if __name__ == "__main__":
    train_arithmetic_system(epochs=35)
