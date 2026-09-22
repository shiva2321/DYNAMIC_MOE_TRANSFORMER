"""
Export step-by-step rollouts for live web visualizer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch
import torch.nn as nn

from ishmn import UniversalCognitiveEngine
from demos.live_maze_solver import solve_and_record_maze, DEVICE

def export_web_data():
    dataset_path = "artifacts/scaled_maze_dataset.json"
    brain_path = "artifacts/unified_lifelong_brain.pt"
    policy_path = "artifacts/maze_policy_head.pt"

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

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    from experiments.train_scaled_maze_engine import create_random_connected_maze
    w_20, p_20 = create_random_connected_maze(grid_size=20, wall_prob=0.18, seed=7003)
    maze_20 = {
        "id": "novel_20x20_extrapolation",
        "grid_size": 20,
        "walls": w_20.tolist(),
        "path": p_20
    }

    curated = [
        {"item": dataset["train_mazes"][0], "category": "Seen (Training)", "label": "Seen Maze #1 (5x5)"},
        {"item": dataset["train_mazes"][3], "category": "Seen (Training)", "label": "Seen Maze #2 (5x5)"},
        {"item": dataset["unseen_mazes"][0], "category": "Unseen (In-Distribution)", "label": "Unseen Maze #1 (5x5)"},
        {"item": dataset["unseen_mazes"][2], "category": "Unseen (In-Distribution)", "label": "Unseen Maze #2 (5x5)"},
        {"item": dataset["novel_mazes"][0], "category": "Novel (Out-of-Distribution)", "label": "Novel 6x6 Maze"},
        {"item": dataset["novel_mazes"][5], "category": "Novel (Out-of-Distribution)", "label": "Novel 7x7 Maze"},
        {"item": dataset["novel_mazes"][-2], "category": "Novel (Out-of-Distribution)", "label": "Novel Central Barrier Box (6x6)"},
        {"item": dataset["novel_mazes"][-1], "category": "Novel (Out-of-Distribution)", "label": "Novel Serpentine Snake (6x6)"},
        {"item": maze_20, "category": "Novel (Out-of-Distribution)", "label": "Zero-Shot 20x20 Maze (16x Scale)"}
    ]

    export_mazes = []
    for entry in curated:
        m = entry["item"]
        res = solve_and_record_maze(brain, action_head, m)
        export_mazes.append({
            "id": m["id"],
            "label": entry["label"],
            "category": entry["category"],
            "grid_size": m["grid_size"],
            "walls": m["walls"],
            "optimal_path": m["path"],
            "reached_goal": res["reached_goal"],
            "steps_taken": res["steps_taken"],
            "optimal_steps": res["optimal_steps"],
            "crashes": res["crashes"],
            "path_history": res["path_history"],
            "actions_taken": res["actions_taken"],
            "frames": res["telemetry_frames"]
        })

    with open("artifacts/maze_visualizer_data.json", "w") as f:
        json.dump(export_mazes, f, indent=2)

    print(f"[+] Successfully exported {len(export_mazes)} curated mazes to artifacts/maze_visualizer_data.json!")

if __name__ == "__main__":
    export_web_data()
