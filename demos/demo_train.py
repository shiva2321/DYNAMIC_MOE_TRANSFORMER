"""
Training and Demonstration of Spherical Hyperspace Controller Network (ISHMN).

Task: Hierarchical Associative Memory in Infinite Spherical Hyperspace
1. Presentation Phase: Network receives distinct key-value pairs associated with multi-shell spherical positions.
2. Distractor/Deliberation Phase: Network must maintain and steer memory in hyperspace while processing noise.
3. Recall Phase: Network is queried with keys and must reconstruct the target values.

Logs convergence metrics, attention entropy, and saves 3D trajectory traces for visualization.
"""

import json
import os
import random
import numpy as np
import torch
import torch.optim as optim
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn import SphericalHyperspaceControllerNet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_spherical_memory_batch(
    batch_size: int,
    num_items: int = 4,
    item_dim: int = 8,
    distractor_steps: int = 4,
    device: torch.device = None
):
    """
    Generates synthetic sequences:
    - Input layout: [Batch, Total_Len, Input_Dim]
      where Input_Dim = item_dim + 2 (flag_store, flag_recall)
    - Total_Len = num_items (store) + distractor_steps (hold) + num_items (recall)
    """
    total_len = num_items + distractor_steps + num_items
    input_dim = item_dim + 2

    inputs = torch.zeros(batch_size, total_len, input_dim, device=device)
    targets = torch.zeros(batch_size, total_len, item_dim, device=device)
    loss_mask = torch.zeros(batch_size, total_len, 1, device=device)

    for b in range(batch_size):
        # Generate random unique binary/continuous vectors for items
        keys = torch.randn(num_items, item_dim, device=device)
        keys = torch.sign(keys)  # Distinct binary patterns for robust keys
        values = torch.randn(num_items, item_dim, device=device)

        # 1. Store phase
        for i in range(num_items):
            inputs[b, i, :item_dim] = keys[i] + values[i] * 0.5
            inputs[b, i, item_dim] = 1.0  # flag_store = 1

        # 2. Distractor phase (zero keys, no flags)
        # inputs remain zero

        # 3. Recall phase (shuffled query order)
        query_order = list(range(num_items))
        random.shuffle(query_order)

        for step_idx, q_idx in enumerate(query_order):
            t = num_items + distractor_steps + step_idx
            inputs[b, t, :item_dim] = keys[q_idx]
            inputs[b, t, item_dim + 1] = 1.0  # flag_recall = 1
            targets[b, t, :] = values[q_idx]
            loss_mask[b, t, 0] = 1.0

    return inputs, targets, loss_mask

def train_spherical_memory_network(
    epochs: int = 200,
    batch_size: int = 16,
    num_items: int = 4,
    item_dim: int = 8,
    distractor_steps: int = 4,
    hyperspace_dim: int = 6,
    device: torch.device = None,
    save_dir: str = "artifacts"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ISHMN] Initializing model on device: {device}")

    os.makedirs(save_dir, exist_ok=True)

    input_dim = item_dim + 2
    hidden_dim = 64
    payload_dim = item_dim
    num_slots = 16
    initial_radius_shells = 4

    model = SphericalHyperspaceControllerNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=item_dim,
        hyperspace_dim=hyperspace_dim,
        payload_dim=payload_dim,
        num_slots=num_slots,
        initial_radius_shells=initial_radius_shells,
        enable_steering=True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
    criterion = nn.MSELoss(reduction='none')

    history = {
        "epoch": [],
        "loss": [],
        "recall_mse": [],
        "cosine_sim": []
    }

    print(f"[ISHMN] Beginning training for {epochs} epochs...")

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        inputs, targets, mask = generate_spherical_memory_batch(
            batch_size=batch_size,
            num_items=num_items,
            item_dim=item_dim,
            distractor_steps=distractor_steps,
            device=device
        )

        outputs, _, _ = model(inputs, record_traces=False)

        # Masked loss only on recall steps
        raw_loss = criterion(outputs, targets)
        loss = (raw_loss * mask).sum() / mask.sum().clamp(min=1.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if ep % 20 == 0 or ep == 1 or ep == epochs:
            model.eval()
            with torch.no_grad():
                val_in, val_tgt, val_mask = generate_spherical_memory_batch(
                    batch_size=32,
                    num_items=num_items,
                    item_dim=item_dim,
                    distractor_steps=distractor_steps,
                    device=device
                )
                val_out, _, _ = model(val_in)
                val_loss = ((criterion(val_out, val_tgt) * val_mask).sum() / val_mask.sum()).item()
                
                # Cosine similarity between target and retrieved output during recall
                pred_recall = val_out[val_mask.squeeze(-1) == 1.0]
                tgt_recall = val_tgt[val_mask.squeeze(-1) == 1.0]
                cos_sim = nn.functional.cosine_similarity(pred_recall, tgt_recall, dim=-1).mean().item()

                history["epoch"].append(ep)
                history["loss"].append(val_loss)
                history["recall_mse"].append(val_loss)
                history["cosine_sim"].append(cos_sim)

                print(f"Epoch {ep:3d}/{epochs} | Val MSE: {val_loss:.4f} | Recall CosSim: {cos_sim:.4f}")

    print("[ISHMN] Training complete! Saving model checkpoint and evaluation traces...")
    torch.save(model.state_dict(), os.path.join(save_dir, "ishmn_model.pt"))

    # Generate rich trace from test evaluation for 3D visualization
    model.eval()
    with torch.no_grad():
        test_in, test_tgt, test_mask = generate_spherical_memory_batch(
            batch_size=1,
            num_items=num_items,
            item_dim=item_dim,
            distractor_steps=distractor_steps,
            device=device
        )
        test_out, final_state, traces = model(test_in, record_traces=True)

    # Serialize traces into JSON-friendly format
    serializable_traces = []
    for t_idx, trace in enumerate(traces):
        serializable_traces.append({
            "step": t_idx,
            "u_q": trace["u_q"][0].cpu().numpy().tolist(),
            "r_q": float(trace["r_q"][0, 0].cpu().item()),
            "u_w": trace["u_w"][0].cpu().numpy().tolist(),
            "r_w": float(trace["r_w"][0, 0].cpu().item()),
            "read_weights": trace["read_weights"][0].cpu().numpy().tolist(),
            "write_weights": trace["write_weights"][0].cpu().numpy().tolist(),
            "write_gate": float(trace["write_gate"][0, 0].cpu().item()),
            "mem_u": trace["mem_u"][0].cpu().numpy().tolist(),
            "mem_r": trace["mem_r"][0, :, 0].cpu().numpy().tolist()
        })

    eval_data = {
        "history": history,
        "traces": serializable_traces,
        "config": {
            "hyperspace_dim": hyperspace_dim,
            "num_slots": num_slots,
            "num_items": num_items,
            "distractor_steps": distractor_steps,
            "total_len": inputs.shape[1]
        }
    }

    trace_path = os.path.join(save_dir, "traces.json")
    with open(trace_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    print(f"[ISHMN] Visualization traces saved to: {trace_path}")
    return model, eval_data

if __name__ == "__main__":
    set_seed(42)
    train_spherical_memory_network(epochs=160, save_dir="artifacts")
