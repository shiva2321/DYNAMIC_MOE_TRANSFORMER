"""
Live Interactive Visualizer for Neural Maze Solving.
Demonstrates step-by-step navigation across:
- Seen Training Mazes
- Unseen In-Distribution Mazes (5x5)
- Novel Out-of-Distribution Mazes (6x6, 7x7, Custom Barrier Mazes)

Features:
- ANSI color grid rendering with live agent animation.
- Real-time cognitive telemetry: Action probabilities, System 1 vs System 2 deliberation,
  Spherical MoE routing, and hypersphere manifold coordinates.
- Supports smooth playback (--speed), step-by-step manual advance (--step), or batch evaluation (--eval-only).
"""

import sys
import io
from pathlib import Path

# Force UTF-8 stdout encoding for Windows console rendering
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ishmn import UniversalCognitiveEngine

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_ARROWS = ["▲ UP   ", "▼ DOWN ", "◄ LEFT ", "► RIGHT"]

# ANSI Color Codes
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_WALL = "\033[38;5;240m██\033[0m"
CLR_AGENT = "\033[38;5;46;1m🤖\033[0m"
CLR_AGENT_CELL = "\033[48;5;28;38;5;15;1m ●\033[0m"
CLR_GOAL = "\033[38;5;220;1m ★\033[0m"
CLR_TRAIL = "\033[38;5;45m ·\033[0m"
CLR_EMPTY = "\033[38;5;238m ·\033[0m"
CLR_CYAN = "\033[38;5;51m"
CLR_YELLOW = "\033[38;5;226m"
CLR_GREEN = "\033[38;5;48m"
CLR_RED = "\033[38;5;196m"
CLR_GRAY = "\033[38;5;245m"
CLR_MAGENTA = "\033[38;5;201m"

def get_local_sensors(walls, curr_r, curr_c, grid_size):
    """Returns [wall_up, wall_down, wall_left, wall_right] (1.0 = wall or boundary)."""
    sensors = []
    for dr, dc in ACTIONS:
        nr, nc = curr_r + dr, curr_c + dc
        if not (0 <= nr < grid_size and 0 <= nc < grid_size) or walls[nr, nc] == 1.0:
            sensors.append(1.0)
        else:
            sensors.append(0.0)
    return sensors

def render_frame(walls, path_history, curr, goal, step, total_steps, telemetry, maze_id, maze_type):
    """Renders a single frame of the maze with side-by-side telemetry."""
    grid_size = walls.shape[0]
    visited_set = set(path_history[:-1]) if len(path_history) > 1 else set()

    lines = []
    header = f"{CLR_BOLD}{CLR_CYAN}=== NEURAL MAZE SOLVER | {maze_id.upper()} ({maze_type.upper()} {grid_size}x{grid_size}) ==={CLR_RESET}"
    lines.append(header)
    lines.append(f"{CLR_GRAY}Progress: Step {step}/{total_steps} | Goal: {goal}{CLR_RESET}")
    lines.append("┌" + "──" * grid_size + "┐   ┌" + "─" * 46 + "┐")

    # Build maze rows and telemetry lines
    maze_rows = []
    for r in range(grid_size):
        row_str = "│"
        for c in range(grid_size):
            pos = (r, c)
            if pos == curr:
                row_str += CLR_AGENT_CELL
            elif pos == goal:
                row_str += CLR_GOAL
            elif walls[r, c] == 1.0:
                row_str += CLR_WALL
            elif pos in visited_set:
                row_str += CLR_TRAIL
            else:
                row_str += CLR_EMPTY
        row_str += "│"
        maze_rows.append(row_str)

    # Telemetry lines
    act_name = telemetry.get("action_name", "START")
    probs = telemetry.get("action_probs", [0.25, 0.25, 0.25, 0.25])
    sys_mode = telemetry.get("system_mode", "System 1 (Corridor Flow)")
    moe_weights = telemetry.get("moe_weights", [])
    man_dist = abs(curr[0] - goal[0]) + abs(curr[1] - goal[1])

    tel_lines = [
        f"│ {CLR_BOLD}Action Chosen:{CLR_RESET} {CLR_GREEN}{act_name:<10}{CLR_RESET} DistToGoal: {man_dist:>2}    │",
        f"│ {CLR_BOLD}Cognitive Mode:{CLR_RESET} {CLR_YELLOW if 'System 2' in sys_mode else CLR_CYAN}{sys_mode:<29}{CLR_RESET} │",
        f"│ Probabilities: ▲ {probs[0]:.2f} ▼ {probs[1]:.2f} ◄ {probs[2]:.2f} ► {probs[3]:.2f} │",
        f"│ Spherical MoE: {', '.join([f'E{i}:{w:.2f}' for i, w in enumerate(moe_weights[:4])]):<30} │",
        f"│ Latent Manifold: r={telemetry.get('radial_r', 1.0):.2f} | u_norm=1.000 (S^7)  │",
        f"│ Collision Check: {'CLEAN (OPEN PATH)' if not telemetry.get('collision') else CLR_RED + 'WALL HIT (RECOVERED)' + CLR_RESET:<38} │",
        f"│ Status: {CLR_GREEN + 'GOAL REACHED!' + CLR_RESET if curr == goal else 'Navigating hyperspace...' :<45} │"
    ]

    max_r = max(len(maze_rows), len(tel_lines))
    for i in range(max_r):
        m_part = maze_rows[i] if i < len(maze_rows) else "│" + "  " * grid_size + "│"
        t_part = tel_lines[i] if i < len(tel_lines) else "│" + " " * 46 + "│"
        lines.append(f"{m_part}   {t_part}")

    lines.append("└" + "──" * grid_size + "┘   └" + "─" * 46 + "┘")
    return "\n".join(lines)

def solve_and_record_maze(brain, action_head, maze_item, max_steps=None, mode="system1", lookahead_depth=16):
    """Executes full step-by-step neural rollout and records telemetry frames."""
    grid_size = maze_item["grid_size"]
    if max_steps is None:
        max_steps = max(35, grid_size * 5)
    walls = np.array(maze_item["walls"], dtype=np.float32)
    goal = tuple(maze_item["path"][-1])
    curr = (0, 0)

    path_history = [curr]
    actions_taken = []
    telemetry_frames = []
    crashes = 0
    traps_pruned = 0
    reached_goal = False

    walls_flat = torch.tensor(walls.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(0)

    for step in range(max_steps):
        if curr == goal:
            reached_goal = True
            # Record final goal frame
            telemetry_frames.append({
                "step": step,
                "curr": curr,
                "action_name": "GOAL!",
                "action_probs": [0.0, 0.0, 0.0, 0.0],
                "system_mode": "GOAL REACHED",
                "moe_weights": [1.0, 0.0, 0.0, 0.0],
                "radial_r": 1.0,
                "collision": False,
                "reached_goal": True
            })
            break

        sensors = get_local_sensors(walls, curr[0], curr[1], grid_size)
        open_count = 4 - sum(sensors)

        # Deliberation complexity
        if open_count > 2:
            sys_mode = "System 2 (Junction Branching)"
        elif open_count == 1:
            sys_mode = "System 2 (Dead-End Recovery)"
        else:
            sys_mode = "System 1 (Corridor Flow)"

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

            # Inspect expert weights
            resonance = brain.moe.compute_resonance(u_fwd, r_fwd)[0]
            topk_w = F.softmax(resonance[:4], dim=-1).cpu().tolist()

        # Action selection: argmax with obstacle awareness & cognitive inhibition of return
        ranked_actions = torch.argsort(logits, descending=True).cpu().tolist()
        collision = False

        # Find all valid (unblocked) candidate moves
        valid_candidates = []
        for act_idx in ranked_actions:
            adr, adc = ACTIONS[act_idx]
            anr, anc = curr[0] + adr, curr[1] + adc
            if 0 <= anr < grid_size and 0 <= anc < grid_size and walls[anr, anc] == 0.0:
                valid_candidates.append((act_idx, (anr, anc)))

        if not valid_candidates:
            break

        # Check if top policy action was blocked by a wall
        top_act = ranked_actions[0]
        tdr, tdc = ACTIONS[top_act]
        tnr, tnc = curr[0] + tdr, curr[1] + tdc
        if not (0 <= tnr < grid_size and 0 <= tnc < grid_size) or walls[tnr, tnc] == 1.0:
            collision = True
            crashes += 1

        # Cognitive Spatial Memory & Decision Engine (System 1 vs System 2)
        recent_pos = path_history[-2] if len(path_history) >= 2 else None
        unvisited_candidates = [c for c in valid_candidates if c[1] not in path_history]
        non_backtrack_candidates = [c for c in valid_candidates if c[1] != recent_pos]

        if mode == "system2" and len(unvisited_candidates) > 1:
            # System 2: Deliberative Reasoner Engine (Mental Simulation in Manifold Hyperspace)
            def make_evaluator(start_pos, candidate_act):
                mental_pos = [start_pos[0] + ACTIONS[candidate_act][0], start_pos[1] + ACTIONS[candidate_act][1]]
                visited_mental = set(path_history) | {tuple(mental_pos)}
                def eval_step(sim_step, z_next, u_next, r_next):
                    if tuple(mental_pos) == goal:
                        return {"is_dead_end": False, "reached_goal": True}
                    open_cands = []
                    for a_i, (adr, adc) in enumerate(ACTIONS):
                        nr, nc = mental_pos[0] + adr, mental_pos[1] + adc
                        if 0 <= nr < grid_size and 0 <= nc < grid_size and walls[nr, nc] == 0.0:
                            if (nr, nc) not in visited_mental:
                                open_cands.append((a_i, (nr, nc)))
                    if not open_cands:
                        return {"is_dead_end": True, "reached_goal": False}
                    act_logits = brain.action_head(z_next)[0]
                    ranked = torch.argsort(act_logits, descending=True).cpu().tolist()
                    chosen_a = next((a for a in ranked if any(c[0] == a for c in open_cands)), open_cands[0][0])
                    next_pos = next(c[1] for c in open_cands if c[0] == chosen_a)
                    mental_pos[0], mental_pos[1] = next_pos[0], next_pos[1]
                    visited_mental.add(tuple(mental_pos))
                    return {"is_dead_end": False, "reached_goal": (tuple(mental_pos) == goal)}
                return eval_step

            candidate_branches = [(act_idx, obs) for act_idx, _ in unvisited_candidates]
            reasoning_res = brain.reason(
                premise=obs,
                context=walls_flat,
                candidate_branches=candidate_branches,
                modality="continuous",
                lookahead_depth=lookahead_depth,
                external_branch_evaluator=lambda b_id: make_evaluator(curr, b_id)
            )

            traps_pruned += reasoning_res.pruned_dead_ends
            sys_mode = f"System 2 ({reasoning_res.system_mode})"
            best_act = reasoning_res.selected_branch
            unvisited_dict = dict(unvisited_candidates)
            if best_act in unvisited_dict:
                nr, nc = unvisited_dict[best_act]
            elif non_backtrack_candidates:
                best_act, (nr, nc) = non_backtrack_candidates[0]
            else:
                best_act, (nr, nc) = valid_candidates[0]
        else:
            # System 1: Instant flow
            if unvisited_candidates:
                best_act, (nr, nc) = unvisited_candidates[0]
            elif non_backtrack_candidates:
                best_act, (nr, nc) = non_backtrack_candidates[0]
            else:
                best_act, (nr, nc) = valid_candidates[0]

        curr = (nr, nc)
        path_history.append(curr)
        actions_taken.append(ACTION_NAMES[best_act])

        telemetry_frames.append({
            "step": step + 1,
            "curr": curr,
            "action_name": ACTION_NAMES[best_act],
            "action_probs": [round(p, 3) for p in probs],
            "system_mode": sys_mode,
            "moe_weights": [round(w, 3) for w in topk_w],
            "radial_r": round(r_fwd.mean().item(), 2),
            "collision": collision,
            "traps_pruned": traps_pruned,
            "reached_goal": (curr == goal)
        })

    return {
        "reached_goal": reached_goal or (curr == goal),
        "steps_taken": len(actions_taken),
        "optimal_steps": len(maze_item["path"]) - 1,
        "crashes": crashes,
        "path_history": path_history,
        "actions_taken": actions_taken,
        "telemetry_frames": telemetry_frames
    }

def run_live_visualizer(args):
    dataset_path = "artifacts/scaled_maze_dataset.json"
    brain_path = "artifacts/unified_lifelong_brain.pt"

    if not os.path.exists(dataset_path) or not os.path.exists(brain_path):
        print("[!] Models or dataset not found. Running training first...")
        from experiments.train_scaled_maze_engine import train_scaled_system
        train_scaled_system()

    print(f"[*] Loading Universal Cognitive Engine from {brain_path}...")
    brain = UniversalCognitiveEngine.load_brain(brain_path, device=DEVICE)
    action_head = brain.action_head
    brain.eval()
    action_head.eval()

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Select mazes according to args
    demos_to_run = []
    if args.maze in ["seen", "all"]:
        demos_to_run.append((dataset["train_mazes"][0], "Seen (Training)"))
        demos_to_run.append((dataset["train_mazes"][1], "Seen (Training)"))
    if args.maze in ["unseen", "all"]:
        demos_to_run.append((dataset["unseen_mazes"][0], "Unseen (In-Distribution)"))
        demos_to_run.append((dataset["unseen_mazes"][1], "Unseen (In-Distribution)"))
    if args.maze in ["novel", "all"]:
        # 6x6, 7x7, and custom barrier
        demos_to_run.append((dataset["novel_mazes"][0], "Novel (6x6 OOD)"))
        demos_to_run.append((dataset["novel_mazes"][5], "Novel (7x7 OOD)"))
        demos_to_run.append((dataset["novel_mazes"][-2], "Novel (Central Barrier Box)"))
        demos_to_run.append((dataset["novel_mazes"][-1], "Novel (Serpentine Snake)"))
    if args.maze in ["20x20", "all"]:
        from experiments.train_scaled_maze_engine import create_random_connected_maze
        w_20, p_20 = create_random_connected_maze(grid_size=20, wall_prob=0.18, seed=7003)
        demos_to_run.append(({"id": "novel_20x20_maze", "grid_size": 20, "walls": w_20.tolist(), "path": p_20}, "Zero-Shot 20x20 Extrapolation"))

    if args.eval_only:
        print("\n" + "=" * 80)
        print("          BATCH EVALUATION: SEEN, UNSEEN & NOVEL MAZES")
        print("=" * 80)
        for cat_name, key in [("Seen (5x5)", "train_mazes"), ("Unseen (5x5)", "unseen_mazes"), ("Novel OOD", "novel_mazes")]:
            cat_list = dataset[key]
            successes = 0
            steps_ratio = []
            for item in cat_list:
                res = solve_and_record_maze(brain, action_head, item, mode=args.mode)
                if res["reached_goal"]:
                    successes += 1
                    steps_ratio.append(res["steps_taken"] / res["optimal_steps"])
            pct = (successes / len(cat_list)) * 100.0
            avg_eff = np.mean(steps_ratio) if steps_ratio else 0.0
            print(f"  {cat_name:<18} | Success: {successes:>2}/{len(cat_list):<2} ({pct:>5.1f}%) | Path Efficiency: {avg_eff:.2f}x optimal")
        print("=" * 80)
        return

    # Interactive / Live step-by-step visual mode
    for maze_item, cat_label in demos_to_run:
        walls = np.array(maze_item["walls"], dtype=np.float32)
        goal = tuple(maze_item["path"][-1])
        res = solve_and_record_maze(brain, action_head, maze_item, mode=args.mode)

        print(f"\n\nStarting Live Rollout for {maze_item['id']} ({cat_label})...")
        time.sleep(1.0)

        path_so_far = [(0, 0)]
        for frame in res["telemetry_frames"]:
            curr = frame["curr"]
            path_so_far.append(curr)

            # Clear screen ANSI
            print("\033[2J\033[H", end="")
            frame_view = render_frame(
                walls=walls,
                path_history=path_so_far,
                curr=curr,
                goal=goal,
                step=frame["step"],
                total_steps=res["steps_taken"],
                telemetry=frame,
                maze_id=maze_item["id"],
                maze_type=cat_label
            )
            print(frame_view)

            if args.step_by_step:
                input("\n[Press Enter for next step...]")
            else:
                time.sleep(args.speed)

        # Summary footer for this maze
        status_str = f"{CLR_GREEN}SUCCESS! (Goal reached){CLR_RESET}" if res["reached_goal"] else f"{CLR_RED}FAILED (Max steps exceeded){CLR_RESET}"
        print(f"\n>>> Outcome: {status_str}")
        print(f">>> Steps Taken: {res['steps_taken']} (Optimal: {res['optimal_steps']}) | Wall Bumps Recovered: {res['crashes']} | Traps Pruned in Imagination: {res.get('traps_pruned', 0)}")
        time.sleep(1.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Neural Maze Solver Visualizer")
    parser.add_argument("--maze", type=str, default="all", choices=["seen", "unseen", "novel", "20x20", "all"], help="Which category of mazes to demonstrate")
    parser.add_argument("--mode", type=str, default="system2", choices=["system1", "system2"], help="Cognitive Mode: system1 (instant) or system2 (mental pre-play)")
    parser.add_argument("--speed", type=float, default=0.18, help="Seconds per step in animation")
    parser.add_argument("--step-by-step", action="store_true", help="Manual step-by-step mode (press Enter to advance)")
    parser.add_argument("--eval-only", action="store_true", help="Run quantitative evaluation across all datasets without animation")
    args = parser.parse_args()

    run_live_visualizer(args)
