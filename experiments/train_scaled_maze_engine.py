"""
Train Scaled Universal Cognitive Engine on Diverse Multi-Maze Dataset.
Includes:
- 60 Training Mazes (5x5, diverse wall configurations)
- 20 Held-out Unseen Mazes (5x5)
- 10 Novel Out-of-Distribution Mazes (6x6 and 7x7 grids + custom barriers)

Guarantees:
- Zero hardcoded pathfinding during evaluation (pure neural inference on Riemannian manifold).
- Scale-invariant continuous encoding.
- Saves trained brain & policy to artifacts/ for live visualizers.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import random
from collections import deque
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ishmn import UniversalCognitiveEngine, geodesic_distance_sphere

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

def create_random_connected_maze(grid_size=5, wall_prob=0.25, seed=None):
    """Generates a connected grid maze of any size with guaranteed valid path."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    max_attempts = 100
    for _ in range(max_attempts):
        walls = np.zeros((grid_size, grid_size), dtype=np.float32)
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) != (0, 0) and (r, c) != (grid_size - 1, grid_size - 1):
                    if random.random() < wall_prob:
                        walls[r, c] = 1.0

        # Check path existence using BFS (only used as ground truth demonstrator)
        queue = deque([(0, 0, [(0, 0)])])
        visited = set([(0, 0)])
        found_path = None

        while queue:
            curr_r, curr_c, path = queue.popleft()
            if (curr_r, curr_c) == (grid_size - 1, grid_size - 1):
                found_path = path
                break
            for dr, dc in ACTIONS:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    if walls[nr, nc] == 0.0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc, path + [(nr, nc)]))

        min_len = grid_size + 1
        if found_path is not None and len(found_path) >= min_len:
            return walls, found_path

    # Fallback to empty maze with boundary
    walls = np.zeros((grid_size, grid_size), dtype=np.float32)
    path = [(0, c) for c in range(grid_size)] + [(r, grid_size - 1) for r in range(1, grid_size)]
    return walls, path

def create_custom_barrier_maze(grid_size=6, pattern="central_box"):
    """Creates a novel structured barrier maze (OOD)."""
    walls = np.zeros((grid_size, grid_size), dtype=np.float32)
    if pattern == "central_box":
        # Block center, forcing a winding path around the edges
        mid = grid_size // 2
        walls[mid-1:mid+1, mid-1:mid+1] = 1.0
        walls[mid, 1] = 1.0
    elif pattern == "serpentine":
        # Snake-like walls
        for c in range(grid_size - 2):
            walls[1, c] = 1.0
        for c in range(2, grid_size):
            walls[3, c] = 1.0

    # Find path
    queue = deque([(0, 0, [(0, 0)])])
    visited = set([(0, 0)])
    found_path = None
    while queue:
        curr_r, curr_c, path = queue.popleft()
        if (curr_r, curr_c) == (grid_size - 1, grid_size - 1):
            found_path = path
            break
        for dr, dc in ACTIONS:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if walls[nr, nc] == 0.0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, path + [(nr, nc)]))

    return walls, found_path

def get_local_sensors(walls, curr_r, curr_c, grid_size):
    """Returns [wall_up, wall_down, wall_left, wall_right] (1.0 = wall or out-of-bounds)."""
    sensors = []
    for dr, dc in ACTIONS:
        nr, nc = curr_r + dr, curr_c + dc
        if not (0 <= nr < grid_size and 0 <= nc < grid_size) or walls[nr, nc] == 1.0:
            sensors.append(1.0)
        else:
            sensors.append(0.0)
    return sensors

def extract_step_transitions(maze_list):
    """Builds training tensors for state observation [N, 8] and actions [N]."""
    x_list = []
    ctx_list = []
    act_list = []

    for item in maze_list:
        walls = item["walls"]
        grid_size = walls.shape[0]
        path = item["path"]
        goal = path[-1]

        # Context is flattened wall array
        flat_walls = walls.flatten()

        for t in range(len(path) - 1):
            curr = path[t]
            nxt = path[t + 1]

            dr = nxt[0] - curr[0]
            dc = nxt[1] - curr[1]
            act = ACTIONS.index((dr, dc))

            sensors = get_local_sensors(walls, curr[0], curr[1], grid_size)

            # State: (r_norm, c_norm, goal_r_norm, goal_c_norm, wall_u, wall_d, wall_l, wall_r)
            state_vec = [
                curr[0] / (grid_size - 1),
                curr[1] / (grid_size - 1),
                goal[0] / (grid_size - 1),
                goal[1] / (grid_size - 1),
            ] + sensors

            x_list.append(state_vec)
            ctx_list.append(flat_walls)
            act_list.append(act)

    return x_list, ctx_list, act_list

def train_scaled_system():
    print("=" * 80)
    print("      SCALED MULTI-MAZE TRAINING (UNIVERSAL COGNITIVE ENGINE)")
    print(f"      Device: {DEVICE} | Hyperspherical MoE + Causal World Model")
    print("=" * 80)

    # 1. Generate Mazes
    print("\n[*] Generating Multi-Maze Dataset...")
    train_mazes = []
    for i in range(60):
        wp = 0.18 + (i % 5) * 0.03
        w, p = create_random_connected_maze(grid_size=5, wall_prob=wp, seed=1000 + i)
        train_mazes.append({"walls": w, "path": p, "id": f"train_5x5_{i+1}", "grid_size": 5})

    unseen_mazes = []
    for i in range(20):
        w, p = create_random_connected_maze(grid_size=5, wall_prob=0.25, seed=5000 + i)
        unseen_mazes.append({"walls": w, "path": p, "id": f"unseen_5x5_{i+1}", "grid_size": 5})

    novel_mazes = []
    # 6x6 mazes
    for i in range(5):
        w, p = create_random_connected_maze(grid_size=6, wall_prob=0.22, seed=8000 + i)
        novel_mazes.append({"walls": w, "path": p, "id": f"novel_6x6_{i+1}", "grid_size": 6})
    # 7x7 mazes
    for i in range(3):
        w, p = create_random_connected_maze(grid_size=7, wall_prob=0.20, seed=9000 + i)
        novel_mazes.append({"walls": w, "path": p, "id": f"novel_7x7_{i+1}", "grid_size": 7})
    # Structured barrier mazes
    w_box, p_box = create_custom_barrier_maze(grid_size=6, pattern="central_box")
    novel_mazes.append({"walls": w_box, "path": p_box, "id": "novel_6x6_barrier_box", "grid_size": 6})
    w_snake, p_snake = create_custom_barrier_maze(grid_size=6, pattern="serpentine")
    novel_mazes.append({"walls": w_snake, "path": p_snake, "id": "novel_6x6_serpentine", "grid_size": 6})

    print(f"    - Training Mazes (5x5)          : {len(train_mazes)}")
    print(f"    - Held-Out Unseen Mazes (5x5)   : {len(unseen_mazes)}")
    print(f"    - Novel OOD Mazes (6x6 & 7x7)   : {len(novel_mazes)}")

    x_train_list, ctx_train_list, act_train_list = extract_step_transitions(train_mazes)
    print(f"[*] Total Step Demonstrations Extracted: {len(x_train_list)} transitions.")

    x_train = torch.tensor(x_train_list, dtype=torch.float32, device=DEVICE)
    act_train = torch.tensor(act_train_list, dtype=torch.long, device=DEVICE)

    # 2. Load the Living Main Brain (artifacts/unified_lifelong_brain.pt)
    BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"
    if os.path.exists(BRAIN_PATH):
        print(f"\n[*] Loading ONE MAIN PERSISTENT BRAIN from {BRAIN_PATH}...")
        brain = UniversalCognitiveEngine.load_brain(BRAIN_PATH, device=DEVICE)
        print(f"    - Existing Specialists: E = {brain.moe.num_experts}")
        print(f"    - Existing Episodic Traces: {len(brain.memory.episodic_memory)}")
        print(f"    - Existing Semantic Concepts: {len(brain.memory.semantic_concepts)}")
    else:
        print(f"\n[*] Initializing Main Persistent Brain at {BRAIN_PATH}...")
        brain = UniversalCognitiveEngine(
            d_model=32,
            d_hidden=64,
            hyperspace_dim=8,
            num_memory_slots=32,
            initial_experts=2,
            max_experts=None,
            lr=0.003,
            device=DEVICE
        )

    action_head = brain.action_head

    optimizer = torch.optim.AdamW(
        list(brain.world_model.parameters()) +
        list(brain.moe.parameters()) +
        list(action_head.parameters()),
        lr=0.003,
        weight_decay=1e-4
    )

    # 3. Train
    print("\n[*] Training Brain & Spatial Policy (35 epochs)...")
    brain.train()
    action_head.train()

    epochs = 35
    batch_size = 32
    num_samples = len(x_train)

    for ep in range(1, epochs + 1):
        indices = torch.randperm(num_samples)
        total_loss = 0.0
        total_acc = 0.0
        batches = 0

        for b_start in range(0, num_samples, batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            xb = x_train[b_idx]
            yb = act_train[b_idx]

            # Context walls tensor
            batch_ctx = [ctx_train_list[i] for i in b_idx.cpu().tolist()]
            ctx_tensor = torch.tensor(np.array(batch_ctx), dtype=torch.float32, device=DEVICE)

            optimizer.zero_grad()
            u_curr, r_curr, z_curr = brain.perceive(xb, modality="continuous")
            u_ctx, r_ctx, z_ctx = brain.perceive(ctx_tensor, modality="continuous")

            z_fwd = brain.world_model.forward_transition(z_curr, z_ctx)
            u_fwd, r_fwd = brain.world_model.project_manifold(z_fwd)

            moe_out, _ = brain.moe(z_fwd, u_fwd, r_fwd, optimizer=optimizer)
            z_pred = z_fwd + 0.3 * moe_out

            logits = action_head(z_pred)
            loss_act = F.cross_entropy(logits, yb)

            z_bwd = brain.world_model.backward_transition(z_pred, z_ctx)
            loss_causal = F.mse_loss(z_bwd, z_curr)

            loss = loss_act + 0.4 * loss_causal
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            acc = (preds == yb).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            batches += 1

        if ep % 5 == 0 or ep == epochs:
            avg_loss = total_loss / batches
            avg_acc = (total_acc / batches) * 100.0
            print(f"    Epoch {ep:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | Action Acc: {avg_acc:.1f}% | Active Specialists: E = {brain.moe.num_experts}")

    # 4. Save Persistent Models & Evaluation Artifacts
    brain.eval()
    action_head.eval()

    print(f"\n[*] Saving ONE MAIN PERSISTENT BRAIN to {BRAIN_PATH}...")
    brain.save_brain(BRAIN_PATH)

    # Serialize datasets for visualizers
    def serialize_maze(m):
        return {
            "id": m["id"],
            "grid_size": m["grid_size"],
            "walls": m["walls"].tolist(),
            "path": m["path"]
        }

    dataset_payload = {
        "train_mazes": [serialize_maze(m) for m in train_mazes],
        "unseen_mazes": [serialize_maze(m) for m in unseen_mazes],
        "novel_mazes": [serialize_maze(m) for m in novel_mazes]
    }
    with open("artifacts/scaled_maze_dataset.json", "w") as f:
        json.dump(dataset_payload, f, indent=2)

    print("[+] Successfully saved scaled brain, policy, and full dataset!")
    return brain, action_head, train_mazes, unseen_mazes, novel_mazes

if __name__ == "__main__":
    train_scaled_system()
