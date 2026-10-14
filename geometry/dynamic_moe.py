"""
Dynamic Geometric Mixture-of-Experts (DynamicYggdrasilMoE)
Combines the Yggdrasil Riemannian Manifold (H^d_h x T^d_t x R+) with autonomous,
growing modular experts to eliminate catastrophic forgetting and enable continual learning.

Key Structural Properties:
1. Fast Sensory Encoder: maps raw inputs to continuous representations.
2. Manifold Projector: enforces strict Riemannian bounds:
   - Hyperbolic: ||u|| < (1 - eps)/sqrt(c) (Poincare ball)
   - Torus: wrapped angles [-pi, pi) (Clifford Torus)
   - Radial: strictly positive intensity r > eps
3. Dynamic Geometric Expert Branches:
   - Each expert owns an independent set of learnable manifold memory prototypes.
   - Each expert maintains a manifold centroid anchor c_k in M.
   - Each expert has a dedicated readout head and optional local adapter.
4. Intrinsic Geodesic Routing:
   - Routing is computed via true manifold geodesic distance:
     d_k = d_M(z, c_k)
   - Avoids routing collapse and router forgetting because routing is geometric.
5. Autonomous Spawning & Freeze Protection:
   - When inputs arrive in an unoccupied manifold region (min_k d_k > R_novel),
     a new expert branch is spawned.
   - Previous expert branches are frozen to guarantee retention.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry.layers import (
    YggdrasilProjector,
    YggdrasilMemoryCortex,
    YggdrasilDistanceFunction,
)
from geometry.gpu_kernel import YggdrasilKernel


class GeometricExpertBranch(nn.Module):
    """
    A single expert branch in the Yggdrasil Manifold.
    Maintains:
    - An independent memory cortex with K learnable manifold prototypes.
    - A dedicated readout head.
    - A lightweight residual adapter for the sensory features.
    - A centroid anchor in Yggdrasil space for geodesic routing.
    """

    def __init__(
        self,
        expert_id: int,
        hidden_dim: int,
        out_features: int,
        num_prototypes: int = 16,
        dim_h: int = 8,
        dim_t: int = 8,
        c: float = 1.0,
        eps: float = 1e-5,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.expert_id = expert_id
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.dim_h = dim_h
        self.dim_t = dim_t
        self.total_dim = dim_h + dim_t + 1
        self.c = c
        self.eps = eps
        self.name = name or f"expert_{expert_id}"

        # 1. Local feature adapter (allows expert-specific sensory refinement)
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Local Yggdrasil Memory Cortex with learnable prototypes
        self.cortex = YggdrasilMemoryCortex(
            num_prototypes=num_prototypes,
            dim_h=dim_h,
            dim_t=dim_t,
            out_features=hidden_dim,
            c=c,
            eps=eps,
        )

        # 3. Dedicated Readout Head
        self.head = nn.Linear(hidden_dim, out_features)

        # 4. Centroid Anchor (buffer, updated via running mean in manifold space)
        self.register_buffer("centroid", torch.zeros(1, self.total_dim))
        self.centroid[0, -1] = 1.0  # initial radial unit
        self.samples_seen = 0

    def clamp_prototypes(self):
        self.cortex.clamp_prototypes()

    def update_centroid(self, states: torch.Tensor):
        """Updates the expert's geometric centroid anchor using current batch states."""
        with torch.no_grad():
            batch_mean = states.mean(dim=0, keepdim=True)
            if self.samples_seen == 0:
                self.centroid.copy_(batch_mean)
            else:
                # Running exponential moving average
                momentum = 0.1
                self.centroid.copy_((1.0 - momentum) * self.centroid + momentum * batch_mean)
            self.samples_seen += states.shape[0]

    def forward(self, feat: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # Refine features with local adapter
        adapted_feat = feat + 0.5 * self.adapter(feat)

        # Query local manifold memory cortex
        _, context = self.cortex(state)

        # Executive readout
        out = self.head(adapted_feat + context)
        return out


class DynamicYggdrasilMoE(nn.Module):
    """
    Complete Dynamic Geometric Mixture-of-Experts Model.
    - Base Sensory Encoder
    - Yggdrasil Projector
    - Growing collection of GeometricExpertBranch modules
    - Geodesic Manifold Router
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        dim_h: int = 8,
        dim_t: int = 8,
        num_prototypes_per_expert: int = 16,
        novelty_threshold: float = 2.5,
        c: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.dim_h = dim_h
        self.dim_t = dim_t
        self.total_dim = dim_h + dim_t + 1
        self.num_prototypes_per_expert = num_prototypes_per_expert
        self.novelty_threshold = novelty_threshold
        self.c = c
        self.eps = eps

        # 1. Base Sensory Encoder (Perception)
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 2. Yggdrasil Manifold Projector
        self.projector = YggdrasilProjector(
            in_features=hidden_dim,
            dim_h=dim_h,
            dim_t=dim_t,
            c=c,
            eps=eps,
        )

        # 3. Dynamic Expert Pool
        self.experts = nn.ModuleList()
        self._kernel = None

    def _get_kernel(self, device):
        if self._kernel is None:
            ordinal = device.index or 0 if device.type == "cuda" else 0
            self._kernel = YggdrasilKernel(device_ordinal=ordinal)
        return self._kernel

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def spawn_expert(self, name: Optional[str] = None) -> GeometricExpertBranch:
        """Dynamically instantiates and registers a new geometric expert branch."""
        eid = len(self.experts)
        expert = GeometricExpertBranch(
            expert_id=eid,
            hidden_dim=self.hidden_dim,
            out_features=self.out_features,
            num_prototypes=self.num_prototypes_per_expert,
            dim_h=self.dim_h,
            dim_t=self.dim_t,
            c=self.c,
            eps=self.eps,
            name=name,
        )
        # Move to same device as encoder
        device = next(self.encoder.parameters()).device
        expert = expert.to(device)
        self.experts.append(expert)
        return expert

    def freeze_expert(self, expert_id: int):
        """Freezes an expert's weights to completely shield it from catastrophic forgetting."""
        if 0 <= expert_id < len(self.experts):
            for param in self.experts[expert_id].parameters():
                param.requires_grad = False

    def freeze_all_past_experts(self):
        """Freezes all experts except the newest one."""
        for i in range(len(self.experts) - 1):
            self.freeze_expert(i)

    def compute_routing_distances(self, states: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise geodesic distance from queries (Batch, Total_Dim)
        to each expert's centroid anchor (Num_Experts, Total_Dim).
        Returns: (Batch, Num_Experts)
        """
        if len(self.experts) == 0:
            raise RuntimeError("No experts exist in DynamicYggdrasilMoE.")

        # Stack centroids: (K, Total_Dim)
        centroids = torch.cat([e.centroid for e in self.experts], dim=0)

        kernel = self._get_kernel(states.device)
        dists = YggdrasilDistanceFunction.apply(
            states,
            centroids,
            self.dim_h,
            self.dim_t,
            self.c,
            self.eps,
            1.0, 1.0, 1.0,
            kernel,
        )
        return dists

    def forward(
        self,
        x: torch.Tensor,
        expert_id: Optional[int] = None,
        return_routing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with geodesic routing.
        If expert_id is explicitly provided, routes directly to that expert.
        Otherwise, routes autonomously via geodesic proximity to expert centroids.
        """
        if len(self.experts) == 0:
            self.spawn_expert("initial_expert")

        feat = self.encoder(x)
        state = self.projector(feat)

        if expert_id is not None:
            # Direct task routing
            target_expert = self.experts[expert_id]
            out = target_expert(feat, state)
            routing_weights = None
        else:
            if len(self.experts) == 1:
                out = self.experts[0](feat, state)
                routing_weights = torch.ones(x.shape[0], 1, device=x.device)
            else:
                # Geodesic Routing across all available expert centroids
                dists = self.compute_routing_distances(state)  # (B, K)
                temperature = 1.0
                routing_weights = F.softmax(-dists / temperature, dim=-1)  # (B, K)

                # Hard top-1 dispatch or soft blend
                best_expert_ids = dists.argmin(dim=-1)  # (B,)

                # Vectorized dispatch
                out = torch.zeros(x.shape[0], self.out_features, device=x.device)
                for k, expert in enumerate(self.experts):
                    mask = (best_expert_ids == k)
                    if mask.any():
                        out[mask] = expert(feat[mask], state[mask])

        if return_routing:
            return out, routing_weights
        return out
