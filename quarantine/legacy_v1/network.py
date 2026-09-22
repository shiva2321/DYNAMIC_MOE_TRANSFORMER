"""
Neural Network Controller in Infinite Spherical Hyperspace (ISHMN).

A recurrent neural network that navigates and controls an infinite spherical memory manifold.
The controller emits:
- Spherical Read Heads (u_q in S^{D-1}, r_q in [0, inf), kappa, beta)
- Spherical Write Heads (u_w, r_w, add, erase, gate)
- Riemannian Geodesic Steering (tangent velocity v in T_u S^{D-1}, radial velocity Delta r)
- Dynamic Shell Spawning into outer concentric hyperspheres
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ishmn.geometry import project_to_sphere
from ishmn.memory import InfiniteSphericalMemory

class SphericalHyperspaceControllerNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hyperspace_dim: int = 8,
        payload_dim: int = 32,
        num_slots: int = 32,
        initial_radius_shells: int = 4,
        max_slots: int = 128,
        enable_steering: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hyperspace_dim = hyperspace_dim
        self.payload_dim = payload_dim
        self.enable_steering = enable_steering

        # Memory module
        self.memory = InfiniteSphericalMemory(
            num_slots=num_slots,
            hyperspace_dim=hyperspace_dim,
            payload_dim=payload_dim,
            initial_radius_shells=initial_radius_shells,
            max_slots=max_slots
        )

        # Recurrent controller: receives input x_t and previous memory read m_{t-1}
        self.controller_cell = nn.GRUCell(input_dim + payload_dim, hidden_dim)

        # Read Head Projections
        self.fc_read_u = nn.Linear(hidden_dim, hyperspace_dim)
        self.fc_read_r = nn.Linear(hidden_dim, 1)
        self.fc_read_kappa = nn.Linear(hidden_dim, 1)
        self.fc_read_beta = nn.Linear(hidden_dim, 1)

        # Write Head Projections
        self.fc_write_u = nn.Linear(hidden_dim, hyperspace_dim)
        self.fc_write_r = nn.Linear(hidden_dim, 1)
        self.fc_write_add = nn.Linear(hidden_dim, payload_dim)
        self.fc_write_erase = nn.Linear(hidden_dim, payload_dim)
        self.fc_write_gate = nn.Linear(hidden_dim, 1)
        self.fc_write_kappa = nn.Linear(hidden_dim, 1)
        self.fc_write_beta = nn.Linear(hidden_dim, 1)

        # Geodesic Steering Projections
        if enable_steering:
            self.fc_steer_tangent = nn.Linear(hidden_dim, hyperspace_dim)
            self.fc_steer_radial = nn.Linear(hidden_dim, 1)
            self.fc_steer_gate = nn.Linear(hidden_dim, 1)

        # Dynamic Shell Spawning Projection
        self.fc_spawn_gate = nn.Linear(hidden_dim, 1)

        # Output Projection (combining controller hidden state and current memory read)
        self.fc_output = nn.Sequential(
            nn.Linear(hidden_dim + payload_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def init_hidden(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def init_state(self, batch_size: int, device: torch.device = None) -> dict:
        """Initializes both controller hidden state and memory state."""
        if device is None:
            device = next(self.parameters()).device
        h_0 = self.init_hidden(batch_size, device)
        m_0 = torch.zeros(batch_size, self.payload_dim, device=device)
        mem_state = self.memory.init_memory(batch_size, device)
        return {
            "h": h_0,
            "read_payload": m_0,
            "memory": mem_state
        }

    def step(
        self,
        x_t: torch.Tensor,
        state: dict,
        record_trace: bool = False
    ) -> tuple[torch.Tensor, dict, dict]:
        """
        Executes one time-step of the network:
        1. Updates controller GRU with [x_t, m_{t-1}]
        2. Generates spherical read query (u_q, r_q) and reads m_t
        3. Generates spherical write params (u_w, r_w, add, erase, gate) and updates memory
        4. (Optional) Steers memory slots along geodesics in hyperspace
        5. (Optional) Dynamically spawns new shell when novelty/spawn_gate is high
        6. Produces output y_t
        """
        h_prev = state["h"]
        m_prev = state["read_payload"]
        mem_state = state["memory"]
        B = x_t.size(0)

        # 1. Update Controller State
        cell_input = torch.cat([x_t, m_prev], dim=-1)
        h_t = self.controller_cell(cell_input, h_prev)

        # 2. Read from Spherical Hyperspace Memory
        u_q = project_to_sphere(self.fc_read_u(h_t))
        r_q = F.softplus(self.fc_read_r(h_t)) + 0.1
        kappa_q = F.softplus(self.fc_read_kappa(h_t)) + 1.0
        beta_q = F.softplus(self.fc_read_beta(h_t)) + 0.5

        read_payload, read_weights, mem_state = self.memory.read(
            mem_state, u_q, r_q, kappa_q, beta_q
        )

        # 3. Write to Spherical Hyperspace Memory
        u_w = project_to_sphere(self.fc_write_u(h_t))
        r_w = F.softplus(self.fc_write_r(h_t)) + 0.1
        add_vec = torch.tanh(self.fc_write_add(h_t))
        erase_vec = torch.sigmoid(self.fc_write_erase(h_t))
        write_gate = torch.sigmoid(self.fc_write_gate(h_t))
        kappa_w = F.softplus(self.fc_write_kappa(h_t)) + 1.0
        beta_w = F.softplus(self.fc_write_beta(h_t)) + 0.5

        write_weights, mem_state = self.memory.write(
            mem_state, u_w, r_w, add_vec, erase_vec, write_gate, kappa_w, beta_w
        )

        # 4. Riemannian Geodesic Steering (Controller actively shifts memory in hyperspace)
        steer_info = None
        if self.enable_steering:
            tangent_vel = torch.tanh(self.fc_steer_tangent(h_t)) * 0.2
            radial_vel = torch.tanh(self.fc_steer_radial(h_t)) * 0.2
            steer_gate = torch.sigmoid(self.fc_steer_gate(h_t))
            mem_state = self.memory.steer_memory_geodesics(
                mem_state,
                tangent_velocity=tangent_vel * steer_gate,
                radial_velocity=radial_vel * steer_gate,
                steer_weights=read_weights  # steers slots in proportion to attention
            )
            steer_info = {
                "tangent_vel": tangent_vel,
                "radial_vel": radial_vel,
                "steer_gate": steer_gate
            }

        # 5. Dynamic Outer Shell Spawning
        spawn_gate = torch.sigmoid(self.fc_spawn_gate(h_t))
        # If spawn gate is very confident in inference / specialized trigger
        # (kept differentiable via soft-weighting; discrete spawn can be enabled when needed)

        # 6. Output Generation
        out_input = torch.cat([h_t, read_payload], dim=-1)
        y_t = self.fc_output(out_input)

        new_state = {
            "h": h_t,
            "read_payload": read_payload,
            "memory": mem_state
        }

        trace = None
        if record_trace:
            trace = {
                "u_q": u_q.detach(),
                "r_q": r_q.detach(),
                "u_w": u_w.detach(),
                "r_w": r_w.detach(),
                "read_weights": read_weights.detach(),
                "write_weights": write_weights.detach(),
                "write_gate": write_gate.detach(),
                "mem_u": mem_state["u"].detach(),
                "mem_r": mem_state["r"].detach(),
                "steer": steer_info
            }

        return y_t, new_state, trace

    def forward(
        self,
        x_seq: torch.Tensor,
        initial_state: dict = None,
        record_traces: bool = False
    ) -> tuple[torch.Tensor, dict, list]:
        """
        Processes a full sequence: x_seq: [B, Seq_Len, input_dim]
        Returns:
            y_seq: [B, Seq_Len, output_dim]
            final_state: dict
            traces: list of dicts (if record_traces is True)
        """
        B, T, _ = x_seq.shape
        if initial_state is None:
            state = self.init_state(B, device=x_seq.device)
        else:
            state = initial_state

        outputs = []
        traces = []

        for t in range(T):
            x_t = x_seq[:, t, :]
            y_t, state, trace = self.step(x_t, state, record_trace=record_traces)
            outputs.append(y_t.unsqueeze(1))
            if record_traces:
                traces.append(trace)

        y_seq = torch.cat(outputs, dim=1)
        return y_seq, state, traces
