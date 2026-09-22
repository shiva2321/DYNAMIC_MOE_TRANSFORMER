"""
Collective Hyperspace Multi-Agent Synchronization Bus.

Grounding:
1. Multiple Universal Cognitive Engines (UCEs) co-existing in the same Infinite Spherical Hyperspace.
2. Riemannian Manifold Consensus: Agents broadcast (u_i in S^{D-1}, r_i in R+, z_i in R^{d_model}).
3. Geodesic Resonant Coupling: Mutual alignment via pairwise spherical cross-attention.
4. Complementary Specialization: Allows a physical mechanics specialist and a symbolic/arithmetic
   specialist to synchronize in real time to solve coupled non-linear problems.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ishmn.geometry import project_to_sphere, geodesic_distance_sphere, exp_map_sphere
from ishmn.universal_cognitive_engine import UniversalCognitiveEngine


class CollectiveHyperspaceBus(nn.Module):
    """
    Manages collective manifold synchronization across multiple Universal Cognitive Engines.
    """
    def __init__(
        self,
        hyperspace_dim: int = 8,
        d_model: int = 32,
        kappa_bus: float = 4.0,
        device: torch.device = None
    ):
        super().__init__()
        self.hyperspace_dim = hyperspace_dim
        self.d_model = d_model
        self.kappa_bus = kappa_bus
        self.device = device if device is not None else torch.device("cpu")

        # Agents registry
        self.agents: Dict[str, UniversalCognitiveEngine] = {}

    def register_agent(self, agent_id: str, agent: UniversalCognitiveEngine):
        """Registers a cognitive agent into the collective manifold."""
        self.agents[agent_id] = agent

    def broadcast_and_synchronize(
        self,
        agent_states: Dict[str, torch.Tensor],
        coupling_strength: float = 0.30
    ) -> Dict[str, Any]:
        """
        Gathers belief states from all agents, computes spherical cross-resonance,
        and exchanges tangent consensus perturbations.
        Args:
            agent_states: Dict mapping agent_id -> z_state [1, d_model]
        Returns:
            synchronized_states: Dict mapping agent_id -> z_synced [1, d_model]
            consensus_metrics: Pairwise geodesic distances, mutual resonance matrix, entropy
        """
        agent_ids = list(agent_states.keys())
        N = len(agent_ids)
        if N <= 1:
            return {
                "synchronized_states": agent_states,
                "mean_angular_distance": 0.0,
                "collective_entropy": 0.0,
                "resonance_matrix": [[1.0]]
            }

        # 1. Project each agent's state onto the manifold
        u_list = []
        r_list = []
        z_list = []
        for aid in agent_ids:
            z_i = agent_states[aid].to(self.device)
            agent = self.agents[aid]
            u_i, r_i = agent.world_model.project_manifold(z_i)
            u_list.append(u_i)
            r_list.append(r_i)
            z_list.append(z_i)

        u_stack = torch.cat(u_list, dim=0) # [N, hyperspace_dim]
        r_stack = torch.cat(r_list, dim=0) # [N, 1]
        z_stack = torch.cat(z_list, dim=0) # [N, d_model]

        # 2. Pairwise Spherical Resonance Matrix
        # cos_sim: [N, N]
        cos_sim = torch.mm(u_stack, u_stack.t()).clamp(-1.0, 1.0)
        dist_matrix = torch.acos(cos_sim) # [N, N] in radians

        # Attention weights via vMF kernel
        weights = F.softmax(self.kappa_bus * cos_sim, dim=-1) # [N, N]

        # 3. Tangent Consensus Blending
        # z_consensus_i = sum_j weights[i, j] * z_j
        z_consensus = torch.mm(weights, z_stack) # [N, d_model]

        # Update each agent's state with peer coupling
        z_synced_stack = (1.0 - coupling_strength) * z_stack + coupling_strength * z_consensus

        synchronized_states = {}
        for idx, aid in enumerate(agent_ids):
            synchronized_states[aid] = z_synced_stack[idx:idx+1]

        # 4. Collective Metrics
        mean_ang_dist = float(dist_matrix.sum().item() / max(1, N * (N - 1)))
        collective_entropy = float(-(weights * torch.log(weights.clamp(min=1e-8))).sum(dim=-1).mean().item())

        return {
            "synchronized_states": synchronized_states,
            "mean_angular_distance": mean_ang_dist,
            "collective_entropy": collective_entropy,
            "resonance_matrix": weights.detach().cpu().numpy().tolist()
        }

    def solve_coupled_task(
        self,
        agent_phys_id: str,
        agent_sym_id: str,
        physical_input: torch.Tensor,
        symbolic_input: Any = "energy conservation in dynamical manifold",
        symbolic_modality: str = "subwords"
    ) -> Dict[str, Any]:
        """
        Executes a coupled cognitive step where two peer agents collaborate:
        - Agent Phys evolves continuous physical dynamics trajectory.
        - Agent Sym performs abductive sense-making / symbolic interpretation.
        - The bus exchanges cross-attention to maintain mutual consensus on S^{D-1}.
        """
        agent_phys = self.agents[agent_phys_id]
        agent_sym = self.agents[agent_sym_id]

        # 1. Individual evaluations
        # Phys perceives continuous
        _, _, z_phys = agent_phys.perceive(physical_input, modality="continuous")
        phys_pred = agent_phys.world_model.forward_transition(z_phys, z_phys)

        # Sym perceives and sense-makes
        sym_sense = agent_sym.sense_make(symbolic_input, modality=symbolic_modality)
        z_sym = sym_sense["latent_state"]

        # 2. Synchronize across the Hyperspace Bus
        sync_res = self.broadcast_and_synchronize(
            agent_states={agent_phys_id: phys_pred, agent_sym_id: z_sym},
            coupling_strength=0.30
        )

        z_phys_synced = sync_res["synchronized_states"][agent_phys_id]
        z_sym_synced = sync_res["synchronized_states"][agent_sym_id]

        # Phys continuous readout post-sync
        out_phys = agent_phys.continuous_head(z_phys_synced)

        return {
            "phys_output": out_phys,
            "sym_surprise": sym_sense["surprise"],
            "sym_coherence": sym_sense["coherence_score"],
            "sym_system_used": sym_sense["system_used"],
            "mean_angular_distance": sync_res["mean_angular_distance"],
            "collective_entropy": sync_res["collective_entropy"]
        }
