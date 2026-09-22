"""
Autonomous Self-Evolving Multi-Network System in Infinite Spherical Hyperspace.

Orchestrates:
1. Universal Perception across heterogeneous data streams
2. Controller Navigation in Infinite Spherical Hyperspace
3. Predictive World Model Foresight & Riemannian Surprise
4. Sleep/Dreaming Consolidation & Shell Migration
5. Continual Self-Optimization & Live Telemetry Streaming
"""

import argparse
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ishmn.geometry import project_to_sphere, geodesic_distance_sphere
from ishmn.memory import InfiniteSphericalMemory
from ishmn.network import SphericalHyperspaceControllerNet
from ishmn.perception import UniversalPerceptor
from ishmn.evolution_engine import (
    PredictiveWorldModel,
    IntrinsicCuriosityEngine,
    DreamingConsolidationEngine
)

class SelfEvolvingHyperspaceSystem:
    def __init__(
        self,
        hyperspace_dim: int = 8,
        payload_dim: int = 32,
        hidden_dim: int = 64,
        num_slots: int = 16,
        max_slots: int = 64,
        device: torch.device = None
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.hyperspace_dim = hyperspace_dim
        self.payload_dim = payload_dim

        print(f"[Self-Evolving System] Initializing on {device}...")

        # 1. Perception Network
        self.perceptor = UniversalPerceptor(
            hyperspace_dim=hyperspace_dim,
            payload_dim=payload_dim,
            embedding_dim=hidden_dim
        ).to(device)

        # 2. Hyperspace Controller Network
        self.controller = SphericalHyperspaceControllerNet(
            input_dim=payload_dim,
            hidden_dim=hidden_dim,
            output_dim=payload_dim,
            hyperspace_dim=hyperspace_dim,
            payload_dim=payload_dim,
            num_slots=num_slots,
            max_slots=max_slots,
            enable_steering=True
        ).to(device)

        # 3. Predictive World Model
        self.world_model = PredictiveWorldModel(
            hyperspace_dim=hyperspace_dim,
            payload_dim=payload_dim,
            hidden_dim=hidden_dim
        ).to(device)

        # 4. Intrinsic Curiosity Engine
        self.curiosity_engine = IntrinsicCuriosityEngine()

        # 5. Dreaming & Consolidation Engine
        self.dreamer = DreamingConsolidationEngine(
            memory_module=self.controller.memory,
            world_model=self.world_model,
            consolidation_rate=0.12,
            clustering_angle=0.15
        )

        # Optimizers
        self.optimizer_controller = optim.AdamW(self.controller.parameters(), lr=0.002, weight_decay=1e-5)
        self.optimizer_perceptor = optim.AdamW(self.perceptor.parameters(), lr=0.002, weight_decay=1e-5)
        self.optimizer_world = optim.AdamW(self.world_model.parameters(), lr=0.002, weight_decay=1e-5)

        # Initialize internal recurrent and memory state
        self.system_state = self.controller.init_state(batch_size=1, device=device)
        self.prev_coord = None

        # Telemetry history
        self.history = {
            "cycle": [],
            "phase": [],
            "surprise": [],
            "prediction_error": [],
            "active_slots": [],
            "max_shell_radius": [],
            "min_shell_radius": [],
            "avg_retention_energy": []
        }

    def generate_streaming_stimulus(self, cycle_idx: int) -> tuple[torch.Tensor, str]:
        """
        Generates diverse multimodal streaming inputs:
        - Cycles between continuous sensory vectors, semantic tokens, and harmonic waveforms.
        """
        modality_selector = cycle_idx % 3
        if modality_selector == 0:
            # Continuous sensory stream: high-dimensional continuous features
            freq = 0.2 * (cycle_idx % 5 + 1)
            t = torch.linspace(0, 1, 16, device=self.device)
            wave = torch.sin(2 * np.pi * freq * t).unsqueeze(0)
            noise = torch.randn(1, 16, device=self.device) * 0.1
            return (wave + noise), "continuous"
        elif modality_selector == 1:
            # Discrete token stream: concept sequence
            token_ids = torch.randint(0, 500, (1, 6), device=self.device)
            return token_ids, "tokens"
        else:
            # Relational associative pattern
            vec = torch.randn(1, 24, device=self.device)
            return vec, "continuous"

    def evolve_cycle(self, cycle_idx: int) -> dict:
        """
        Executes one full self-evolution cycle:
        1. Wake Phase: Perceives input, predicts next coordinate, reads memory, steers, computes surprise.
        2. Plasticity & Adaptation: High surprise triggers shell spawning and weight updates.
        3. Sleep / Dream Phase: Periodic counterfactual replay, shell migration towards core, cluster compaction.
        """
        self.controller.train()
        self.perceptor.train()
        self.world_model.train()

        # -----------------------------
        # Phase 1: Wake & Perception
        # -----------------------------
        stimulus, modality = self.generate_streaming_stimulus(cycle_idx)
        u_stim, r_stim, v_stim = self.perceptor(stimulus, modality=modality)

        # If we had a previous coordinate, evaluate World Model foresight
        surprise_val = 0.0
        pred_loss_val = 0.0
        if self.prev_coord is not None:
            u_prev, r_prev, v_prev = self.prev_coord
            u_pred, r_pred, v_pred = self.world_model(u_prev, r_prev, v_prev)

            surprise, surprise_metrics = self.curiosity_engine.compute_surprise(
                u_pred, r_pred, v_pred, u_stim, r_stim, v_stim
            )
            surprise_val = surprise.mean().item()

            # Train world model to predict the dynamics
            loss_wm = (
                geodesic_distance_sphere(u_pred, u_stim).mean() +
                torch.abs(r_pred - r_stim).mean() +
                F.mse_loss(v_pred, v_stim)
            )
            self.optimizer_world.zero_grad()
            loss_wm.backward(retain_graph=True)
            self.optimizer_world.step()
            pred_loss_val = loss_wm.item()

        self.prev_coord = (u_stim.detach(), r_stim.detach(), v_stim.detach())

        # Step Controller with perceived payload
        y_t, self.system_state, trace = self.controller.step(
            v_stim, self.system_state, record_trace=True
        )

        # -----------------------------
        # Phase 2: Plasticity & Shell Expansion
        # -----------------------------
        # If surprise is high (> 1.2), this is a novel paradigm -> spawn into outer hyperspace shell (r -> inf)
        spawned = False
        if surprise_val > 1.2:
            self.system_state["memory"] = self.controller.memory.spawn_infinite_shell(
                self.system_state["memory"],
                u_new=u_stim.detach(),
                v_new=v_stim.detach(),
                shell_delta=1.5
            )
            spawned = True

        # Online self-supervised associative reconstruction loss
        recon_loss = F.mse_loss(y_t, v_stim)
        self.optimizer_controller.zero_grad()
        self.optimizer_perceptor.zero_grad()
        recon_loss.backward()
        self.optimizer_controller.step()
        self.optimizer_perceptor.step()

        # Detach states to prevent unbounded BPTT across infinite lifelong time
        self.system_state["h"] = self.system_state["h"].detach()
        self.system_state["read_payload"] = self.system_state["read_payload"].detach()
        self.system_state["memory"]["u"] = self.system_state["memory"]["u"].detach()
        self.system_state["memory"]["r"] = self.system_state["memory"]["r"].detach()
        self.system_state["memory"]["v"] = self.system_state["memory"]["v"].detach()

        # -----------------------------
        # Phase 3: Sleep & Dreaming Consolidation (Every 5 cycles)
        # -----------------------------
        dream_metrics = {}
        if cycle_idx % 5 == 0 and cycle_idx > 0:
            self.system_state["memory"], dream_metrics = self.dreamer.dream_and_consolidate(
                self.system_state["memory"],
                dream_steps=4,
                optimizer=self.optimizer_world
            )
            current_phase = "Sleep / Dreaming Consolidation"
        else:
            current_phase = "Wake / Online Perception"

        mem = self.system_state["memory"]
        num_active = mem["u"].size(1)
        max_r = float(mem["r"].max().item())
        min_r = float(mem["r"].min().item())
        avg_energy = float(mem["energy"].mean().item())

        # Log cycle metrics
        self.history["cycle"].append(cycle_idx)
        self.history["phase"].append(current_phase)
        self.history["surprise"].append(surprise_val)
        self.history["prediction_error"].append(pred_loss_val)
        self.history["active_slots"].append(num_active)
        self.history["max_shell_radius"].append(max_r)
        self.history["min_shell_radius"].append(min_r)
        self.history["avg_retention_energy"].append(avg_energy)

        cycle_summary = {
            "cycle": cycle_idx,
            "phase": current_phase,
            "modality": modality,
            "surprise": surprise_val,
            "prediction_error": pred_loss_val,
            "recon_loss": recon_loss.item(),
            "active_slots": num_active,
            "max_shell_radius": max_r,
            "min_shell_radius": min_r,
            "avg_energy": avg_energy,
            "shell_spawned": spawned,
            "dream_metrics": dream_metrics,
            "trace": {
                "u_q": trace["u_q"][0].cpu().numpy().tolist(),
                "r_q": float(trace["r_q"][0, 0].cpu().item()),
                "u_w": trace["u_w"][0].cpu().numpy().tolist(),
                "r_w": float(trace["r_w"][0, 0].cpu().item()),
                "mem_u": mem["u"][0].cpu().numpy().tolist(),
                "mem_r": mem["r"][0, :, 0].cpu().numpy().tolist(),
                "mem_energy": mem["energy"][0].cpu().numpy().tolist()
            }
        }
        return cycle_summary

    def run_evolution_loop(
        self,
        cycles: int = 50,
        delay_sec: float = 0.0,
        telemetry_file: str = "artifacts/evolution_telemetry.json"
    ):
        print(f"[Self-Evolving System] Starting evolution loop for {cycles} cycles...")
        os.makedirs(os.path.dirname(telemetry_file), exist_ok=True)

        recent_traces = []

        for c in range(1, cycles + 1):
            summary = self.evolve_cycle(c)
            recent_traces.append(summary["trace"])
            if len(recent_traces) > 30:
                recent_traces.pop(0)

            if c % 5 == 0 or c == 1 or c == cycles:
                print(
                    f"Cycle {c:3d}/{cycles} | Phase: {summary['phase']:<30} | "
                    f"Surprise: {summary['surprise']:.3f} | Slots: {summary['active_slots']} | "
                    f"Max Shell r: {summary['max_shell_radius']:.2f} | Energy: {summary['avg_energy']:.2f}"
                )

            # Export live telemetry checkpoint
            telemetry_data = {
                "history": self.history,
                "current_summary": {k: v for k, v in summary.items() if k != "trace"},
                "recent_traces": recent_traces,
                "config": {
                    "hyperspace_dim": self.hyperspace_dim,
                    "payload_dim": self.payload_dim,
                    "total_cycles": cycles
                }
            }
            with open(telemetry_file, "w") as f:
                json.dump(telemetry_data, f, indent=2)

            if delay_sec > 0:
                time.sleep(delay_sec)

        print(f"[Self-Evolving System] Lifelong evolution loop completed! Telemetry saved to: {telemetry_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=40, help="Number of self-evolution cycles")
    parser.add_argument("--delay", type=float, default=0.0, help="Optional delay between cycles")
    args = parser.parse_args()

    system = SelfEvolvingHyperspaceSystem(
        hyperspace_dim=8,
        payload_dim=32,
        num_slots=16,
        max_slots=48
    )
    system.run_evolution_loop(cycles=args.cycles, delay_sec=args.delay)
