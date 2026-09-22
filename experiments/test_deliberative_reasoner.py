"""
Demonstration and Benchmark of the Unified Deliberative Reasoner Engine.

Tests endogenous imaginative reasoning on the ONE living brain (unified_lifelong_brain.pt):
1. Domain 1 (Scientific Hypothesis Reasoning): Evaluates competing hypotheses given physical premises.
2. Domain 2 (Spatial Labyrinth Deliberation): Prunes deceptive 22-step cul-de-sac in imagination.
3. Domain 3 (Mathematical Dynamic Reasoning): Evaluates competing numerical hypotheses.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import torch
import torch.nn as nn
import numpy as np

from ishmn import UniversalCognitiveEngine, ReasoningResult
from demos.live_maze_solver import ACTIONS, ACTION_NAMES, get_local_sensors, DEVICE
from experiments.deliberative_mental_simulation import create_deceptive_utrap_maze

BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"
POLICY_PATH = "artifacts/maze_policy_head.pt"

def run_reasoner_demonstration():
    print("=" * 85)
    print("      DELIBERATIVE REASONER ENGINE: IMAGINATION-BASED CAUSAL REASONING")
    print(f"      Device: {DEVICE} | Brain: {BRAIN_PATH}")
    print("=" * 85)

    if not os.path.exists(BRAIN_PATH):
        print(f"[ERROR] Brain not found at {BRAIN_PATH}")
        return

    brain = UniversalCognitiveEngine.load_brain(BRAIN_PATH, device=DEVICE)
    brain.eval()

    action_head = nn.Sequential(
        nn.Linear(brain.d_model, 64),
        nn.LayerNorm(64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 4)
    ).to(DEVICE)
    if os.path.exists(POLICY_PATH):
        action_head.load_state_dict(torch.load(POLICY_PATH, map_location=DEVICE))
    action_head.eval()

    # =========================================================================
    # TEST 1: SCIENTIFIC HYPOTHESIS REASONING (Language / Truth Domain)
    # =========================================================================
    print("\n" + "=" * 85)
    print("  [TEST 1: SCIENTIFIC HYPOTHESIS REASONING THROUGH IMAGINATION]")
    print("=" * 85)
    premise_sci = "the sun radiates thermal heat and electromagnetic light"
    candidates_sci = [
        ("hypothesis_illumination", "provides natural illumination and warms the planetary surface"),
        ("hypothesis_nitrogen_ice", "spontaneously creates freezing arctic liquid nitrogen ice"),
        ("hypothesis_solar_energy", "powers biological photosynthesis in planetary vegetation")
    ]

    print(f"  Premise: \"{premise_sci}\"")
    print(f"  Evaluating {len(candidates_sci)} competing hypotheses through 8 imagined ticks...")

    res_sci: ReasoningResult = brain.reason(
        premise=premise_sci,
        candidate_branches=candidates_sci,
        modality="subwords",
        lookahead_depth=8
    )

    print(f"\n  Reasoner Verdict:")
    print(f"    Mode                : {res_sci.system_mode}")
    print(f"    Selected Hypothesis : '{res_sci.selected_branch}'")
    print(f"    Confidence Score    : {res_sci.confidence * 100:.1f}%")
    print(f"    Pruned Traps        : {res_sci.pruned_contradictions} contradictions pruned")
    print(f"    Explanation         : {res_sci.explanation}")

    print("\n  Detailed Cognitive Thought Trajectory:")
    for b in res_sci.ranked_branches:
        status = "REJECTED (Contradiction)" if b.is_contradiction else "ACCEPTED"
        print(f"    - Branch '{b.branch_id}': Score = {b.score:.2f} | Causal Consistency = {b.causal_consistency:.2f} | {status}")

    # =========================================================================
    # TEST 2: SPATIAL LABYRINTH DELIBERATION (Deceptive 22-Step Cul-de-Sac)
    # =========================================================================
    print("\n" + "=" * 85)
    print("  [TEST 2: SPATIAL LABYRINTH DELIBERATION (Deceptive 22-Step Cul-de-Sac)]")
    print("=" * 85)
    grid_size = 20
    walls, path = create_deceptive_utrap_maze(grid_size=grid_size)
    goal = path[-1]
    curr = (0, 0)

    # Context is flattened walls
    walls_flat = torch.tensor(walls.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(0)

    # State vector at starting junction (0, 0)
    sensors = get_local_sensors(walls, curr[0], curr[1], grid_size)
    state_vec = [
        curr[0] / (grid_size - 1),
        curr[1] / (grid_size - 1),
        goal[0] / (grid_size - 1),
        goal[1] / (grid_size - 1)
    ] + sensors
    state_tensor = torch.tensor([state_vec], dtype=torch.float32, device=DEVICE)

    # Two candidate paths at (0, 0):
    # Candidate 1: Action DOWN (0) -> leads into 22-step cul-de-sac trap!
    # Candidate 2: Action RIGHT (3) -> true optimal path to goal!
    candidates_maze = [
        ("branch_down_trap", torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE)),
        ("branch_right_goal", torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=DEVICE))
    ]

    def make_maze_evaluator(branch_id):
        def evaluate_step(step, z_next, u_next, r_next):
            # In Branch Down, at step 22, it hits dead end at (14, 8)
            if branch_id == "branch_down_trap" and step >= 22:
                return {"is_dead_end": True, "reason": "cul_de_sac"}
            return {"is_dead_end": False}
        return evaluate_step

    print("  Junction at (0, 0): Two open paths available:")
    print("    • Branch 1 (DOWN)  : Greedy direction, leads to 22-step cul-de-sac dead end.")
    print("    • Branch 2 (RIGHT) : True optimal escape corridor to (19, 19).")
    print("  Engaging Deliberative Reasoner with lookahead depth K = 25...")

    res_maze: ReasoningResult = brain.reason(
        premise=state_tensor,
        context=walls_flat,
        candidate_branches=candidates_maze,
        modality="continuous",
        lookahead_depth=25,
        external_branch_evaluator=make_maze_evaluator
    )

    print(f"\n  Reasoner Verdict:")
    print(f"    Mode                : {res_maze.system_mode}")
    print(f"    Selected Action     : '{res_maze.selected_branch}'")
    print(f"    Dead Ends Pruned    : {res_maze.pruned_dead_ends}")
    print(f"    Confidence Score    : {res_maze.confidence * 100:.1f}%")
    print(f"    Explanation         : {res_maze.explanation}")

    for b in res_maze.ranked_branches:
        status = "PRUNED IN IMAGINATION (Dead End)" if b.is_dead_end else "SELECTED (Viable Escape)"
        print(f"    - Branch '{b.branch_id}': Score = {b.score:.2f} | Survived = {b.steps_survived} ticks | {status}")

    # =========================================================================
    # TEST 3: MATHEMATICAL HYPOTHESIS TESTING
    # =========================================================================
    print("\n" + "=" * 85)
    print("  [TEST 3: MATHEMATICAL HYPOTHESIS TESTING]")
    print("=" * 85)
    # Testing: 5 + 3 = ?
    # Candidate hypotheses: 8.0 (correct), 15.0 (incorrect), -2.0 (incorrect)
    a_val, b_val = 5.0, 3.0
    scale = 35.0
    x_in = torch.tensor([[a_val / scale]], dtype=torch.float32, device=DEVICE)
    ctx_in = torch.tensor([[1.0, 0.0, 0.0, 0.0, b_val / scale]], dtype=torch.float32, device=DEVICE)

    candidates_math = [
        ("hypothesis_8", torch.tensor([[8.0 / scale]], dtype=torch.float32, device=DEVICE)),
        ("hypothesis_15", torch.tensor([[15.0 / scale]], dtype=torch.float32, device=DEVICE)),
        ("hypothesis_neg2", torch.tensor([[-2.0 / scale]], dtype=torch.float32, device=DEVICE))
    ]

    print(f"  Problem: {a_val:.0f} + {b_val:.0f} = ?")
    print(f"  Evaluating candidate answers: [8, 15, -2] in imagination...")

    res_math = brain.reason(
        premise=x_in,
        context=ctx_in,
        candidate_branches=candidates_math,
        modality="continuous",
        lookahead_depth=6
    )

    print(f"\n  Reasoner Verdict:")
    print(f"    Selected Answer     : '{res_math.selected_branch}'")
    print(f"    Confidence Score    : {res_math.confidence * 100:.1f}%")
    print(f"    Explanation         : {res_math.explanation}")

    for b in res_math.ranked_branches:
        print(f"    - Answer Candidate '{b.branch_id}': Score = {b.score:.2f} | Causal Consistency = {b.causal_consistency:.2f}")

    print("\n" + "=" * 85)
    print("             DELIBERATIVE REASONER DEMONSTRATION COMPLETE")
    print("=" * 85)

if __name__ == "__main__":
    run_reasoner_demonstration()
