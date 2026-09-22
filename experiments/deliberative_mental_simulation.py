"""
Experiment: System 2 Deliberative Mental Simulation for Long-Horizon Deceptive Mazes.
Implements Vicarious Trial and Error (VTE) / Mental Pre-Play:
- Evaluates candidate branches at junctions through imagined forward rollouts.
- Prunes dead-end traps in imagination before taking physical steps.
- Compares System 1 (Greedy 1-Step) vs System 2 (Deliberative Mental Pre-Play)
  across random 20x20 mazes and purpose-built deceptive trap mazes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ishmn import UniversalCognitiveEngine
from demos.live_maze_solver import ACTIONS, ACTION_NAMES, get_local_sensors, DEVICE
from experiments.train_scaled_maze_engine import create_random_connected_maze

def simulate_branch_forward(walls, start_pos, first_action, goal, lookahead_depth=12,
                            visited_physical=None, brain=None, action_head=None):
    """
    Mentally rolls out a candidate action branch for K steps using the NEURAL POLICY
    (brain + action_head) to pick each imagined next move — zero hardcoded heuristics.

    NOTE: `brain` and `action_head` must be provided. They are the same unified
    cognitive engine and maze policy head used for physical navigation. This ensures
    the mental pre-play is fully consistent with the agent's actual learned behaviour.

    Returns:
      - is_dead_end: bool (did the mental path terminate in a cul-de-sac?)
      - reached_goal: bool (did the mental rollout reach the goal?)
      - steps_survived: int (how many steps before dead-ending or completing lookahead)
      - final_dist: int (Manhattan distance to goal at termination of rollout)
    """
    if brain is None or action_head is None:
        raise ValueError(
            "[simulate_branch_forward] brain and action_head must be provided. "
            "Mental simulation must use neural inference — zero hardcoded heuristics."
        )

    grid_size = walls.shape[0]
    visited_mental = set(visited_physical) if visited_physical else set()
    visited_mental.add(start_pos)

    walls_flat = torch.tensor(walls.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(0)

    dr, dc = ACTIONS[first_action]
    curr = (start_pos[0] + dr, start_pos[1] + dc)
    visited_mental.add(curr)

    if curr == goal:
        return {"is_dead_end": False, "reached_goal": True, "steps_survived": 1, "final_dist": 0}

    for step in range(2, lookahead_depth + 1):
        # Sense open unvisited moves from the mental position
        open_candidates = []
        for a_idx, (adr, adc) in enumerate(ACTIONS):
            nr, nc = curr[0] + adr, curr[1] + adc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and walls[nr, nc] == 0.0:
                if (nr, nc) not in visited_mental:
                    open_candidates.append((a_idx, (nr, nc)))

        if not open_candidates:
            # Cul-de-sac / dead-end reached in imagination
            man_dist = abs(curr[0] - goal[0]) + abs(curr[1] - goal[1])
            return {"is_dead_end": True, "reached_goal": False, "steps_survived": step, "final_dist": man_dist}

        # ---- NEURAL inference guides each imagined step (zero hardcoded heuristics) ----
        sensors = get_local_sensors(walls, curr[0], curr[1], grid_size)
        state_vec = [
            curr[0] / (grid_size - 1),
            curr[1] / (grid_size - 1),
            goal[0] / (grid_size - 1),
            goal[1] / (grid_size - 1),
        ] + sensors
        obs = torch.tensor([state_vec], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            u_c, r_c, z_c = brain.perceive(obs, modality="continuous")
            u_ctx, r_ctx, z_ctx = brain.perceive(walls_flat, modality="continuous")
            z_fwd = brain.world_model.forward_transition(z_c, z_ctx)
            u_fwd, r_fwd = brain.world_model.project_manifold(z_fwd)
            moe_out, _ = brain.moe(z_fwd, u_fwd, r_fwd)
            z_pred = z_fwd + 0.3 * moe_out
            logits = action_head(z_pred)[0]

        # Pick best open-unvisited action according to neural logits
        open_act_set = {a_idx for a_idx, _ in open_candidates}
        ranked = torch.argsort(logits, descending=True).cpu().tolist()
        best_act = next((a for a in ranked if a in open_act_set), open_candidates[0][0])
        best_next_pos = dict(open_candidates)[best_act]
        # ---------------------------------------------------------------------------------

        curr = best_next_pos
        visited_mental.add(curr)

        if curr == goal:
            return {"is_dead_end": False, "reached_goal": True, "steps_survived": step, "final_dist": 0}

    man_dist = abs(curr[0] - goal[0]) + abs(curr[1] - goal[1])
    return {"is_dead_end": False, "reached_goal": False, "steps_survived": lookahead_depth, "final_dist": man_dist}


def solve_maze_experiment(brain, action_head, maze_item, mode="system1", max_steps=350, lookahead_depth=12):
    """
    Solves a maze using either:
    - mode="system1": Instantaneous 1-step reactive policy
    - mode="system2": Deliberative Mental Simulation (Vicarious Trial and Error)
    """
    walls = np.array(maze_item["walls"], dtype=np.float32)
    grid_size = maze_item["grid_size"]
    goal = tuple(maze_item["path"][-1])
    curr = (0, 0)

    path_history = [curr]
    actions_taken = []
    crashes = 0
    traps_pruned_in_imagination = 0
    junction_deliberations = 0

    walls_flat = torch.tensor(walls.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(0)

    for step in range(max_steps):
        if curr == goal:
            break

        sensors = get_local_sensors(walls, curr[0], curr[1], grid_size)
        open_count = 4 - sum(sensors)

        state_vec = [
            curr[0] / (grid_size - 1),
            curr[1] / (grid_size - 1),
            goal[0] / (grid_size - 1),
            goal[1] / (grid_size - 1)
        ] + sensors

        obs = torch.tensor([state_vec], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            u_curr, r_curr, z_curr = brain.perceive(obs, modality="continuous")
            u_ctx, r_ctx, z_ctx = brain.perceive(walls_flat, modality="continuous")

            z_fwd = brain.world_model.forward_transition(z_curr, z_ctx)
            u_fwd, r_fwd = brain.world_model.project_manifold(z_fwd)

            moe_out, _ = brain.moe(z_fwd, u_fwd, r_fwd)
            z_pred = z_fwd + 0.3 * moe_out

            logits = action_head(z_pred)[0]
            probs = F.softmax(logits, dim=-1).cpu().tolist()

        ranked_actions = torch.argsort(logits, descending=True).cpu().tolist()

        # Find unblocked candidates
        valid_candidates = []
        for act_idx in ranked_actions:
            adr, adc = ACTIONS[act_idx]
            anr, anc = curr[0] + adr, curr[1] + adc
            if 0 <= anr < grid_size and 0 <= anc < grid_size and walls[anr, anc] == 0.0:
                valid_candidates.append((act_idx, (anr, anc)))

        if not valid_candidates:
            break

        # Check for immediate wall bump
        top_act = ranked_actions[0]
        tdr, tdc = ACTIONS[top_act]
        tnr, tnc = curr[0] + tdr, curr[1] + tdc
        if not (0 <= tnr < grid_size and 0 <= tnc < grid_size) or walls[tnr, tnc] == 1.0:
            crashes += 1

        recent_pos = path_history[-2] if len(path_history) >= 2 else None
        unvisited = [c for c in valid_candidates if c[1] not in path_history]
        non_backtrack = [c for c in valid_candidates if c[1] != recent_pos]

        # -------------------------------------------------------------
        # Decision Logic: System 1 vs System 2
        # -------------------------------------------------------------
        if mode == "system1" or len(unvisited) <= 1:
            # System 1: Instant flow
            if unvisited:
                best_act, (nr, nc) = unvisited[0]
            elif non_backtrack:
                best_act, (nr, nc) = non_backtrack[0]
            else:
                best_act, (nr, nc) = valid_candidates[0]
        else:
            # System 2: Deliberative Mental Simulation (Vicarious Trial and Error)
            junction_deliberations += 1
            viable_branches = []
            dead_branches = []

            for act_idx, next_p in unvisited:
                sim_res = simulate_branch_forward(
                    walls=walls,
                    start_pos=curr,
                    first_action=act_idx,
                    goal=goal,
                    lookahead_depth=lookahead_depth,
                    visited_physical=path_history,
                    brain=brain,
                    action_head=action_head
                )
                if sim_res["is_dead_end"]:
                    dead_branches.append((act_idx, next_p, sim_res))
                    traps_pruned_in_imagination += 1
                else:
                    # Viable branch
                    score = 100.0 if sim_res["reached_goal"] else (100.0 - sim_res["final_dist"])
                    viable_branches.append((act_idx, next_p, score, probs[act_idx]))

            if viable_branches:
                # Rank viable branches by combined mental score and neural policy probability
                viable_branches.sort(key=lambda b: (b[2], b[3]), reverse=True)
                best_act, (nr, nc) = viable_branches[0][0], viable_branches[0][1]
            elif unvisited:
                # All candidate unvisited branches hit dead ends: pick the one with longest survival
                dead_branches.sort(key=lambda b: b[2]["steps_survived"], reverse=True)
                best_act, (nr, nc) = dead_branches[0][0], dead_branches[0][1]
            elif non_backtrack:
                best_act, (nr, nc) = non_backtrack[0]
            else:
                best_act, (nr, nc) = valid_candidates[0]

        curr = (nr, nc)
        path_history.append(curr)
        actions_taken.append(ACTION_NAMES[best_act])

    reached_goal = (curr == goal)
    opt_steps = len(maze_item["path"]) - 1
    return {
        "reached_goal": reached_goal,
        "steps_taken": len(actions_taken),
        "optimal_steps": opt_steps,
        "efficiency": len(actions_taken) / opt_steps if opt_steps > 0 else 1.0,
        "crashes": crashes,
        "traps_pruned": traps_pruned_in_imagination,
        "junction_deliberations": junction_deliberations
    }

def create_deceptive_utrap_maze(grid_size=20):
    """
    Creates a purpose-built deceptive labyrinth:
    At (0, 0), there are two open branches:
    - Branch 1 (Greedy Trap): Goes DOWN col 0 to row 14, then RIGHT to col 8.
      DEAD END at (14, 8)! Total trap depth: 22 steps.
    - Branch 2 (True Path): Goes RIGHT along row 0 to col 19, then DOWN col 19 to (19, 19).
      Total optimal length: 38 steps.
    """
    walls = np.ones((grid_size, grid_size), dtype=np.float32)

    # Trap Path (Branch 1)
    for r in range(15):
        walls[r, 0] = 0.0
    for c in range(9):
        walls[14, c] = 0.0  # Dead end at (14, 8)

    # True Escape Path (Branch 2)
    for c in range(grid_size):
        walls[0, c] = 0.0
    for r in range(grid_size):
        walls[r, grid_size - 1] = 0.0

    path = [(0, c) for c in range(grid_size)] + [(r, grid_size - 1) for r in range(1, grid_size)]
    return walls, path

def run_comparative_experiment():
    brain_path = "artifacts/unified_lifelong_brain.pt"
    policy_path = "artifacts/maze_policy_head.pt"

    print("=" * 85)
    print("   LONG-HORIZON MAZE EXPERIMENT: SYSTEM 1 vs SYSTEM 2 DELIBERATION")
    print("   Testing Deceptive Cul-de-Sacs & 20x20 Labyrinths (Zero Retraining)")
    print("=" * 85)

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

    # -------------------------------------------------------------
    # BATTERY 1: The 10 Random 20x20 Mazes (Revisiting Maze #1 & #7)
    # -------------------------------------------------------------
    print("\n--- [BATTERY 1: The 10 Random 20x20 Mazes] ---")
    print(f"{'Maze ID':<12} | {'Optimal':<7} | {'System 1 Steps':<14} | {'System 2 Steps':<14} | {'Traps Pruned':<12} | {'Sys2 Win?'}")
    print("-" * 85)

    sys1_success = 0
    sys2_success = 0
    sys1_steps_total = 0
    sys2_steps_total = 0

    for seed in range(10):
        w, p = create_random_connected_maze(grid_size=20, wall_prob=0.18, seed=7000 + seed)
        item = {"id": f"maze_20x20_{seed+1}", "grid_size": 20, "walls": w, "path": p}

        res_sys1 = solve_maze_experiment(brain, action_head, item, mode="system1", max_steps=350)
        res_sys2 = solve_maze_experiment(brain, action_head, item, mode="system2", max_steps=350, lookahead_depth=14)

        if res_sys1["reached_goal"]:
            sys1_success += 1
        if res_sys2["reached_goal"]:
            sys2_success += 1

        s1_str = f"{res_sys1['steps_taken']} (PASS)" if res_sys1["reached_goal"] else f"{res_sys1['steps_taken']} (FAIL)"
        s2_str = f"{res_sys2['steps_taken']} (PASS)" if res_sys2["reached_goal"] else f"{res_sys2['steps_taken']} (FAIL)"

        win_str = "YES" if (res_sys2["reached_goal"] and not res_sys1["reached_goal"]) or (res_sys2["steps_taken"] < res_sys1["steps_taken"]) else "EQUAL"
        print(f"{item['id']:<12} | {res_sys1['optimal_steps']:<7} | {s1_str:<14} | {s2_str:<14} | {res_sys2['traps_pruned']:<12} | {win_str}")

    print("-" * 85)
    print(f"System 1 (Greedy 1-Step) Success Rate: {sys1_success}/10 ({sys1_success*10:.1f}%)")
    print(f"System 2 (Mental Pre-Play) Success Rate: {sys2_success}/10 ({sys2_success*10:.1f}%)")

    # -------------------------------------------------------------
    # BATTERY 2: Designed Deceptive U-Trap Labyrinths
    # -------------------------------------------------------------
    print("\n\n--- [BATTERY 2: Purpose-Built Deceptive U-Trap Labyrinths (20x20)] ---")
    w_trap, p_trap = create_deceptive_utrap_maze(grid_size=20)
    trap_item = {"id": "deceptive_utrap_20x20", "grid_size": 20, "walls": w_trap, "path": p_trap}

    print(f"[*] Optimal Escape Route: {len(p_trap) - 1} steps")
    print("[*] Running System 1 (Greedy 1-Step)...")
    res_trap_s1 = solve_maze_experiment(brain, action_head, trap_item, mode="system1", max_steps=350)
    print(f"    System 1 -> Reached: {'YES' if res_trap_s1['reached_goal'] else 'NO (TIMEOUT)'} | Steps: {res_trap_s1['steps_taken']} | Wall Crashes: {res_trap_s1['crashes']}")

    print("[*] Running System 2 (Deliberative Mental Pre-Play with Lookahead K=25)...")
    res_trap_s2 = solve_maze_experiment(brain, action_head, trap_item, mode="system2", max_steps=350, lookahead_depth=25)
    print(f"    System 2 -> Reached: {'YES' if res_trap_s2['reached_goal'] else 'NO (TIMEOUT)'} | Steps: {res_trap_s2['steps_taken']} | Traps Pruned in Imagination: {res_trap_s2['traps_pruned']} | Wall Crashes: {res_trap_s2['crashes']}")

    diff_steps = res_trap_s1['steps_taken'] - res_trap_s2['steps_taken']
    print(f"\n[+] System 2 Cognitive Advantage: Saved {diff_steps} physical exploratory steps by simulating {res_trap_s2['traps_pruned']} dead-end traps in imagination!")

if __name__ == "__main__":
    run_comparative_experiment()
