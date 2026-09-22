"""
Autonomous Evolution Engine for Infinite Spherical Hyperspace.

Components:
1. PredictiveWorldModel: Anticipates future coordinates in S^{D-1} x R+
2. IntrinsicCuriosityEngine: Computes Riemannian surprise on the hypersphere
3. DreamingConsolidationEngine: Simulates sleep/dream replays, shell migrations, and cluster compaction
4. MetaEvolutionaryAdaptor: Adjusts plasticity and meta-parameters to continually self-improve
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ishmn.geometry import project_to_sphere, geodesic_distance_sphere, exp_map_sphere
from ishmn.memory import InfiniteSphericalMemory

class PredictiveWorldModel(nn.Module):
    """
    Anticipates future trajectories in Infinite Spherical Hyperspace.
    Predicts next directional orientation u_{t+1} and radial scale r_{t+1}.
    """
    def __init__(self, hyperspace_dim: int = 8, payload_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.hyperspace_dim = hyperspace_dim
        self.payload_dim = payload_dim

        self.net = nn.Sequential(
            nn.Linear(hyperspace_dim + 1 + payload_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.pred_u = nn.Linear(hidden_dim, hyperspace_dim)
        self.pred_r = nn.Linear(hidden_dim, 1)
        self.pred_v = nn.Linear(hidden_dim, payload_dim)

    def forward(
        self,
        u_curr: torch.Tensor,
        r_curr: torch.Tensor,
        v_curr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([u_curr, r_curr, v_curr], dim=-1)
        feat = self.net(x)

        u_next = project_to_sphere(self.pred_u(feat))
        r_next = F.softplus(self.pred_r(feat)) + 0.1
        v_next = torch.tanh(self.pred_v(feat))

        return u_next, r_next, v_next

class IntrinsicCuriosityEngine(nn.Module):
    """
    Calculates intrinsic surprise on the Riemannian manifold S^{D-1} x R+.
    Surprise = Geodesic angle distance + radial disparity + payload reconstruction error.
    """
    def __init__(self, lambda_angle: float = 1.0, lambda_radial: float = 0.5, lambda_payload: float = 0.2):
        super().__init__()
        self.lambda_angle = lambda_angle
        self.lambda_radial = lambda_radial
        self.lambda_payload = lambda_payload

    def compute_surprise(
        self,
        u_pred: torch.Tensor,
        r_pred: torch.Tensor,
        v_pred: torch.Tensor,
        u_actual: torch.Tensor,
        r_actual: torch.Tensor,
        v_actual: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # 1. Geodesic distance on S^{D-1} in radians [0, pi]
        angle_dist = geodesic_distance_sphere(u_pred, u_actual).squeeze(-1) # [B]

        # 2. Radial difference
        radial_dist = torch.abs(r_pred - r_actual).squeeze(-1) # [B]

        # 3. Payload feature disparity
        payload_dist = torch.norm(v_pred - v_actual, p=2, dim=-1) # [B]

        total_surprise = (
            self.lambda_angle * angle_dist +
            self.lambda_radial * radial_dist +
            self.lambda_payload * payload_dist
        )

        metrics = {
            "surprise_mean": total_surprise.mean().item(),
            "angle_dist_mean": angle_dist.mean().item(),
            "radial_dist_mean": radial_dist.mean().item(),
            "payload_dist_mean": payload_dist.mean().item()
        }

        return total_surprise, metrics

class DreamingConsolidationEngine:
    """
    Offline sleep/dreaming processor.
    Performs generative memory replay, merges close concepts, and migrates
    stable memories inward into permanent core shells.
    """
    def __init__(
        self,
        memory_module: InfiniteSphericalMemory,
        world_model: PredictiveWorldModel,
        consolidation_rate: float = 0.15,
        clustering_angle: float = 0.12
    ):
        self.memory = memory_module
        self.world_model = world_model
        self.consolidation_rate = consolidation_rate
        self.clustering_angle = clustering_angle

    def dream_and_consolidate(
        self,
        state: dict[str, torch.Tensor],
        dream_steps: int = 5,
        optimizer: torch.optim.Optimizer = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """
        Executes a dreaming phase:
        1. Counterfactual geodesic imagination using PredictiveWorldModel
        2. Radial shell migration towards core
        3. Geodesic cluster compaction on S^{D-1}
        """
        u_mem = state["u"]
        r_mem = state["r"]
        v_mem = state["v"]
        B, N, D = u_mem.shape

        # Pick highest energy slots as dream seeds
        seed_indices = torch.topk(state["energy"], k=min(4, N), dim=-1).indices

        dream_losses = []
        for step in range(dream_steps):
            # Select seed points across batch
            for b in range(B):
                idx = seed_indices[b, step % seed_indices.size(1)]
                u_seed = u_mem[b:b+1, idx]
                r_seed = r_mem[b:b+1, idx]
                v_seed = v_mem[b:b+1, idx]

                # Run predictive world model imagination
                u_dream, r_dream, v_dream = self.world_model(u_seed, r_seed, v_seed)

                # World model self-consistency loss
                cycle_u, cycle_r, _ = self.world_model(u_dream, r_dream, v_dream)
                loss = (geodesic_distance_sphere(u_seed, cycle_u) + torch.abs(r_seed - cycle_r)).mean()

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                dream_losses.append(loss.item())

        # Consolidate shells and compact clusters
        consolidated_state = self.memory.consolidate_and_migrate_shells(
            state, consolidation_rate=self.consolidation_rate
        )
        compacted_state = self.memory.compact_geodesic_clusters(
            consolidated_state, angle_threshold=self.clustering_angle
        )

        metrics = {
            "dream_loss_avg": sum(dream_losses) / max(1, len(dream_losses)),
            "active_slots": compacted_state["u"].size(1),
            "max_shell_radius": float(compacted_state["r"].max().item()),
            "min_shell_radius": float(compacted_state["r"].min().item()),
            "avg_energy": float(compacted_state["energy"].mean().item())
        }

        return compacted_state, metrics
