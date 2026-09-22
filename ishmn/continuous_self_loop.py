"""
Continuous Autonomous Self-Loop with Dynamic MoE and Triggered I/O.

Architecture Paradigm:
- The Self-Loop is the primary living entity running continuous internal dynamics.
- Inputs are external perturbations that nudge the continuous trajectory in S^{D-1} x R+.
- Outputs are triggered emissions released only when attractor consensus or confidence
  thresholds are reached (no forced step-by-step regurgitation).
- Memory and Neural Networks co-evolve:
  - Memory spawns new concentric shells (r -> inf).
  - Dynamic MoE spawns new Dendritic Experts via Clonal Mitosis and DynamicWarmupAdamW.
"""

import queue
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from ishmn.geometry import project_to_sphere, exp_map_sphere
from ishmn.memory import InfiniteSphericalMemory
from ishmn.perception import UniversalPerceptor
from ishmn.neuro.dendritic_expert import TwoCompartmentDendriticExpert
from ishmn.spherical_dynamic_moe import SphericalDynamicMoE
from ishmn.dynamic_optimizer import DynamicWarmupAdamW
from ishmn.spherical_attention import RiemannianSphericalAttention

@dataclass
class TriggeredEmission:
    step: int
    payload: torch.Tensor
    coordinate_u: List[float]
    coordinate_r: float
    confidence: float
    active_experts: int
    trigger_reason: str
    metadata: Dict[str, Any]
    action_type: int = 2  # 0: DELIBERATE, 1: MOTOR_STEP, 2: EMIT_COMMUNICATION, 3: INVOKE_SYMBOLIC_TOOL
    action_result: Optional[Any] = None

class ContinuousSelfLoop(nn.Module):
    def __init__(
        self,
        uce: Optional[Any] = None,
        d_model: int = 32,
        d_hidden: int = 64,
        hyperspace_dim: int = 8,
        num_memory_slots: int = 16,
        initial_experts: int = 1,
        max_experts: Optional[int] = None,
        emission_threshold: float = 0.52,
        device: torch.device = None
    ):
        super().__init__()
        self.uce = uce
        if uce is not None:
            self.device = uce.device
            self.d_model = uce.d_model
            self.d_hidden = uce.d_hidden
            self.hyperspace_dim = uce.hyperspace_dim
            self.perceptor = uce.perceptor
            self.memory = uce.memory
            self.moe = uce.moe
        else:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            self.d_model = d_model
            self.d_hidden = d_hidden
            self.hyperspace_dim = hyperspace_dim

            # 1. Universal Perception Engine (turns input perturbations into coordinates)
            self.perceptor = UniversalPerceptor(
                hyperspace_dim=hyperspace_dim,
                payload_dim=d_model,
                embedding_dim=d_hidden
            ).to(device)

            # 2. Infinite Spherical Hyperspace Memory
            self.memory = InfiniteSphericalMemory(
                num_slots=num_memory_slots,
                hyperspace_dim=hyperspace_dim,
                payload_dim=d_model,
                initial_radius_shells=4
            ).to(device)

            # 3. Dynamic Spherical MoE (spawns and evolves dendritic experts)
            self.moe = SphericalDynamicMoE(
                d_model=d_model,
                d_ff=d_hidden,
                d_context=d_model,
                hyperspace_dim=hyperspace_dim,
                initial_experts=initial_experts,
                max_experts=max_experts,
                spawn_threshold=0.35,
                device=device
            ).to(device)

        self.emission_threshold = emission_threshold

        # 4. Core Controller with Attractor Memory & Riemannian Spherical Attention
        self.recurrent_cell = nn.GRUCell(self.d_model * 2, self.d_hidden).to(self.device)
        self.attractor_attn = RiemannianSphericalAttention(
            d_model=self.d_hidden,
            num_heads=4,
            hyperspace_dim=self.hyperspace_dim,
            device=self.device
        ).to(self.device)
        self.attractor_buffer = []
        self.max_attractor_window = 16

        # Heads
        self.fc_steer_u = nn.Linear(self.d_hidden, self.hyperspace_dim).to(self.device)
        self.fc_steer_r = nn.Linear(self.d_hidden, 1).to(self.device)
        self.fc_read_query_u = nn.Linear(self.d_hidden, self.hyperspace_dim).to(self.device)
        self.fc_read_query_r = nn.Linear(self.d_hidden, 1).to(self.device)
        self.fc_write_add = nn.Linear(self.d_hidden, self.d_model).to(self.device)
        self.fc_write_gate = nn.Linear(self.d_hidden, 1).to(self.device)

        # Trigger Gate: Decides whether internal state reaches consensus to emit an output
        self.fc_trigger_gate = nn.Sequential(
            nn.Linear(self.d_hidden + self.d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)

        # Emission Readout Head
        self.fc_emission_payload = nn.Sequential(
            nn.Linear(self.d_hidden + self.d_model, self.d_hidden),
            nn.GELU(),
            nn.Linear(self.d_hidden, self.d_model)
        ).to(self.device)

        # 5. Dynamic Optimizer with Warmup
        if self.uce is not None:
            self.optimizer = self.uce.optimizer
        else:
            all_params = list(self.perceptor.parameters()) + \
                         list(self.recurrent_cell.parameters()) + \
                         list(self.attractor_attn.parameters()) + \
                         list(self.fc_steer_u.parameters()) + \
                         list(self.fc_steer_r.parameters()) + \
                         list(self.fc_read_query_u.parameters()) + \
                         list(self.fc_read_query_r.parameters()) + \
                         list(self.fc_write_add.parameters()) + \
                         list(self.fc_write_gate.parameters()) + \
                         list(self.fc_trigger_gate.parameters()) + \
                         list(self.fc_emission_payload.parameters()) + \
                         list(self.moe.parameters())
            self.optimizer = DynamicWarmupAdamW(all_params, lr=0.002, default_group_warmup_steps=30)


        # State initialization
        self.reset_state()

        # Input perturbation queue (external inputs pushed here)
        self.input_queue: queue.Queue = queue.Queue()

        # Telemetry logs
        self.telemetry = {
            "step": [],
            "mode": [],
            "u_trajectory": [],
            "r_trajectory": [],
            "active_experts": [],
            "memory_slots": [],
            "max_shell": [],
            "trigger_confidence": [],
            "emissions_count": 0
        }

    def reset_state(self):
        self.curr_u = project_to_sphere(torch.randn(1, self.hyperspace_dim, device=self.device))
        self.curr_r = torch.tensor([[1.5]], device=self.device)
        self.curr_h = torch.zeros(1, self.d_hidden, device=self.device)
        self.curr_read_payload = torch.zeros(1, self.d_model, device=self.device)
        self.attractor_buffer = []
        self.mem_state = self.memory.init_memory(batch_size=1, device=self.device)
        self.step_count = 0
        self.idle_count = 0

    def inject_input(self, data: torch.Tensor, modality: str = "continuous", tag: str = ""):
        """Injects an external input as a perturbation to the self-loop."""
        self.input_queue.put({"data": data, "modality": modality, "tag": tag})

    def loop_tick(self) -> Optional[TriggeredEmission]:
        """
        Executes a single continuous tick of the living self-loop:
        1. Checks for input perturbations (or engages in endogenous oscillation).
        2. Routes through Dynamic MoE (with neurogenesis if novelty is detected).
        3. Reads & writes to Infinite Spherical Memory.
        4. Steers internal trajectory along hypersphere geodesics.
        5. Checks emission trigger: fires output ONLY when consensus threshold is met.
        6. Periodically triggers sleep consolidation when idle.
        """
        self.step_count += 1
        self.train()
        self.optimizer.update_dynamic_schedules()

        # ----------------------------------------------------
        # 1. Sensory Ingestion or Endogenous Oscillation
        # ----------------------------------------------------
        has_input = not self.input_queue.empty()
        input_tag = ""

        if has_input:
            item = self.input_queue.get_nowait()
            stim_data = item["data"].to(self.device)
            modality = item["modality"]
            input_tag = item.get("tag", "")

            # Universal perceptor turns input into coordinates
            u_stim, r_stim, v_stim = self.perceptor(stim_data, modality=modality)

            # Perturb internal coordinate towards stimulus location
            self.curr_u = project_to_sphere(0.6 * self.curr_u + 0.4 * u_stim)
            self.curr_r = 0.6 * self.curr_r + 0.4 * r_stim
            x_feed = v_stim
            mode = "Perturbed (Input Injected)"
            self.idle_count = 0
        else:
            # Endogenous oscillation: internal harmonic drift
            drift_noise = torch.randn_like(self.curr_u) * 0.05
            self.curr_u = project_to_sphere(self.curr_u + drift_noise)
            x_feed = self.curr_read_payload  # Feed back previous memory read
            mode = "Endogenous Oscillation"
            self.idle_count += 1

        # ----------------------------------------------------
        # 2. Dynamic MoE Routing & Autonomous Neurogenesis
        # ----------------------------------------------------
        moe_out, moe_meta = self.moe(
            x=x_feed,
            u_x=self.curr_u,
            r_x=self.curr_r,
            context=self.curr_read_payload,
            optimizer=self.optimizer
        )

        # ----------------------------------------------------
        # 3. Recurrent Controller Update & RSA Attractor Attention
        # ----------------------------------------------------
        cell_in = torch.cat([moe_out, self.curr_read_payload], dim=-1)
        h_cand = self.recurrent_cell(cell_in, self.curr_h)

        # Attractor trajectory maintenance and Riemannian Spherical Attention
        self.attractor_buffer.append(h_cand.unsqueeze(1))
        if len(self.attractor_buffer) > self.max_attractor_window:
            self.attractor_buffer.pop(0)

        if len(self.attractor_buffer) > 1:
            attractor_seq = torch.cat(self.attractor_buffer, dim=1)
            attended_h, _ = self.attractor_attn(attractor_seq)
            self.curr_h = h_cand + 0.3 * attended_h[:, -1, :]
        else:
            self.curr_h = h_cand

        # ----------------------------------------------------
        # 4. Memory Read, Write, and Geodesic Steering
        # ----------------------------------------------------
        u_q = project_to_sphere(self.fc_read_query_u(self.curr_h))
        r_q = F.softplus(self.fc_read_query_r(self.curr_h)) + 0.2
        kappa_q = torch.tensor([[4.0]], device=self.device)
        beta_q = torch.tensor([[1.0]], device=self.device)

        read_payload, read_weights, self.mem_state = self.memory.read(
            self.mem_state, u_q, r_q, kappa_q, beta_q
        )
        self.curr_read_payload = read_payload

        # Write to memory
        write_add = torch.tanh(self.fc_write_add(self.curr_h))
        write_gate = torch.sigmoid(self.fc_write_gate(self.curr_h))
        _, self.mem_state = self.memory.write(
            self.mem_state, self.curr_u, self.curr_r,
            add_vector=write_add,
            erase_vector=torch.ones_like(write_add) * 0.3,
            write_gate=write_gate,
            kappa_w=kappa_q, beta_rw=beta_q
        )

        # Steer internal trajectory in hyperspace
        steer_v = torch.tanh(self.fc_steer_u(self.curr_h)) * 0.2
        self.curr_u = exp_map_sphere(self.curr_u, steer_v)
        dr = torch.tanh(self.fc_steer_r(self.curr_h)) * 0.1
        self.curr_r = F.softplus(self.curr_r + dr)

        # ----------------------------------------------------
        # 5. Triggered Emission Check
        # ----------------------------------------------------
        trigger_in = torch.cat([self.curr_h, read_payload], dim=-1)
        gate_score = self.fc_trigger_gate(trigger_in)

        # Phase-lock resonance between MoE processing and retrieved memory
        res_align = F.cosine_similarity(moe_out, read_payload, dim=-1, eps=1e-6).unsqueeze(-1)
        res_norm = 0.5 * (res_align + 1.0) # [0, 1]

        # Combined confidence: internal gate + phase-lock resonance + memory-aligned trigger
        comb_conf = 0.30 * gate_score + 0.40 * res_norm
        if has_input or (self.idle_count <= 2 and self.idle_count > 0):
            # Input perturbation settling into attractor: boost proportional to resonance
            # High resonance (>0.7) passes threshold 0.60; unresonant noise (<0.55) stays silent
            comb_conf = comb_conf + 0.30 * res_norm

        trigger_confidence = float(comb_conf.item())

        emission = None
        if trigger_confidence >= self.emission_threshold:
            out_payload = self.fc_emission_payload(trigger_in)
            act_type = 2  # default communication
            act_res = None
            if self.uce is not None and hasattr(self.uce, "action_head"):
                with torch.no_grad():
                    act_logits = self.uce.action_head(out_payload)[0]
                    act_type = int(torch.argmax(act_logits).item())
                    # If input tag contains a math query, ensure symbolic tool dispatch
                    if any(c in input_tag for c in ["+", "-", "*", "/", "%", "^"]):
                        act_type = 3  # ACTION_INVOKE_SYMBOLIC_TOOL
                    if act_type == 3 and hasattr(self.uce, "dispatch_action"):
                        dispatch_out = self.uce.dispatch_action(act_type, query=input_tag)
                        act_res = dispatch_out.get("result", None)

            emission = TriggeredEmission(
                step=self.step_count,
                payload=out_payload.detach(),
                coordinate_u=self.curr_u[0].detach().cpu().numpy().tolist(),
                coordinate_r=float(self.curr_r[0, 0].item()),
                confidence=trigger_confidence,
                active_experts=self.moe.num_experts,
                trigger_reason=f"Attractor Reached (Conf: {trigger_confidence:.3f}) on tag '{input_tag}'",
                metadata={
                    "mode": mode,
                    "topk_experts": moe_meta["topk_indices"],
                    "routing_entropy": moe_meta["routing_entropy"],
                    "spawned_expert": moe_meta["spawned_expert"]
                },
                action_type=act_type,
                action_result=act_res
            )
            self.telemetry["emissions_count"] += 1

        # ----------------------------------------------------
        # 6. Periodic Homeostatic Sleep Consolidation (when idle)
        # ----------------------------------------------------
        if self.idle_count >= 10:
            mode = "Homeostatic Sleep & Consolidation"
            self.trigger_sleep_cycle()
            self.idle_count = 0

        # Detach states to maintain infinite lifecycle without graph explosion
        self.curr_h = self.curr_h.detach()
        self.curr_u = self.curr_u.detach()
        self.curr_r = self.curr_r.detach()
        self.curr_read_payload = self.curr_read_payload.detach()

        # Log telemetry
        max_r = float(self.mem_state["r"].max().item())
        self.telemetry["step"].append(self.step_count)
        self.telemetry["mode"].append(mode)
        self.telemetry["u_trajectory"].append(self.curr_u[0].cpu().numpy().tolist())
        self.telemetry["r_trajectory"].append(float(self.curr_r[0, 0].cpu().item()))
        self.telemetry["active_experts"].append(self.moe.num_experts)
        self.telemetry["memory_slots"].append(self.mem_state["u"].size(1))
        self.telemetry["max_shell"].append(max_r)
        self.telemetry["trigger_confidence"].append(trigger_confidence)

        return emission

    def trigger_sleep_cycle(self, prune_usage_threshold: float = 0.02) -> Dict[str, Any]:
        """
        Executes an autonomous homeostatic sleep consolidation phase:
        1. Migrates and consolidates associative memory shells in S^{D-1} x R+.
        2. Merges co-resonant experts and apoptotically prunes dead experts.
        3. Cleans up DynamicWarmupAdamW parameter groups and momentum buffers.
        """
        self.mem_state = self.memory.consolidate_and_migrate_shells(self.mem_state, consolidation_rate=0.15)
        self.mem_state = self.memory.compact_geodesic_clusters(self.mem_state)
        moe_report = self.moe.sleep_consolidate_and_prune(
            prune_usage_threshold=prune_usage_threshold,
            optimizer=self.optimizer
        )
        return {
            "moe_consolidation": moe_report,
            "memory_slots": self.mem_state["u"].size(1),
            "max_shell_radius": float(self.mem_state["r"].max().item()),
            "status": "consolidated"
        }

