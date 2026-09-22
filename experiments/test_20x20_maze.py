"""
Zero-Shot Extrapolation Experiment: 20x20 Mazes with ZERO Retraining.
Tests whether the Universal Cognitive Engine trained on 5x5 mazes can navigate
a 20x20 maze (16x larger area, 400 cells, deep corridors).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn

from ishmn import UniversalCognitiveEngine
from demos.live_maze_solver import solve_and_record_maze, DEVICE
from experiments.train_scaled_maze_engine import create_random_connected_maze

def test_20x20_zero_shot():
    brain_path = "artifacts/unified_lifelong_brain.pt"
    policy_path = "artifacts/maze_policy_head.pt"

    print("=" * 80)
    print("      ZERO-SHOT EXTRAPOLATION TEST: 20x20 MAZES (ZERO RETRAINING)")
    print("      Model Trained Strictly on 5x5 Mazes (Area: 25 cells -> 400 cells)")
    print("=" * 80)

    brain = UniversalCognitiveEngine.load_brain(brain_path, device=DEVICE)
    action_head = nn.Sequential(
        nn.Linear(brain.d_model, 64),
        nn.LayerNorm(64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 4)
    ).to(DEVICE)
    action_head.load_state_dict(torch.load(policy_path, map_location=DEVICE))
    brain.eval()
    action_head.eval()

    # Test on 10 different 20x20 mazes
    num_tests = 10
    successes = 0
    results = []

    print(f"\n[*] Generating and testing {num_tests} distinct 20x20 mazes...")
    for seed in range(num_tests):
        w, p = create_random_connected_maze(grid_size=20, wall_prob=0.18, seed=7000 + seed)
        maze_item = {
            "id": f"maze_20x20_{seed+1}",
            "grid_size": 20,
            "walls": w,
            "path": p
        }
        opt_len = len(p) - 1

        # Give generous step budget for 400 cells (e.g. 350 steps)
        res = solve_and_record_maze(brain, action_head, maze_item, max_steps=350)
        reached = res["reached_goal"]
        steps = res["steps_taken"]
        crashes = res["crashes"]

        if reached:
            successes += 1

        eff = steps / opt_len if opt_len > 0 else 1.0
        results.append((seed + 1, opt_len, steps, eff, reached, crashes))
        print(f"  Maze #{seed+1:2d} | Opt: {opt_len:2d} steps | Agent: {steps:3d} steps | Eff: {eff:.2f}x | Reached: {'YES' if reached else 'NO'} | Crashes: {crashes}")

    print("\n" + "=" * 80)
    print(f"  Summary on 20x20 Zero-Shot: {successes}/{num_tests} ({successes/num_tests*100:.1f}% Success)")
    print("=" * 80)

if __name__ == "__main__":
    test_20x20_zero_shot()
