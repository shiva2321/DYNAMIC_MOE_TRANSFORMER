"""
Semantic Hyperspace Item Memory and Dynamic Expert Spawning Engine.
Maintains the high-dimensional codebook of LEARNABLE expert addresses in C^D,
detects novel knowledge regions via cluster quantile resonance,
and updates expert centroids via direct backpropagation gradients.
"""

import math
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperspace.vsa import ComplexPhasorVSA

class SemanticHyperspaceMemory(nn.Module):
    """
    Continuous Item Memory for Neural Experts in Complex Phasor Hyperspace.
    Expert address keys are learnable parameters in C^D optimized by backpropagation.
    """
    def __init__(
        self,
        d_hyper: int = 2048,
        spawn_threshold: float = 0.35,
        max_experts: int = 64,
        use_drift_guard: bool = False,
        drift_sigma: float = 2.0
    ):
        super().__init__()
        self.d_hyper = d_hyper
        self.spawn_threshold = spawn_threshold
        self.max_experts = max_experts
        self.use_drift_guard = use_drift_guard
        self.drift_sigma = drift_sigma
        
        # Learnable parameter lists for real and imaginary components in C^D
        self.keys_r = nn.ParameterList()
        self.keys_i = nn.ParameterList()
        self.expert_metadata: List[Dict[str, Any]] = []

        # Statistical Outlier Drift-Guard Buffers (Faithful to VSA_GROUND_REBUILD)
        self.register_buffer("running_sim_mean", torch.zeros(max_experts))
        self.register_buffer("running_sim_var", torch.ones(max_experts) * 0.04)
        self.register_buffer("usage_counts", torch.zeros(max_experts, dtype=torch.long))

    @property
    def num_experts(self) -> int:
        return len(self.keys_r)

    @property
    def expert_keys(self) -> torch.Tensor:
        """
        Returns normalized complex unit phasor keys: [N_experts, D] in C^D.
        Preserves autograd gradient tracking into keys_r and keys_i.
        """
        if self.num_experts == 0:
            return torch.empty((0, self.d_hyper), dtype=torch.cfloat)
        
        kr = torch.cat([p for p in self.keys_r], dim=0) # [N, D]
        ki = torch.cat([p for p in self.keys_i], dim=0) # [N, D]
        z = torch.complex(kr, ki)
        unit_keys = z / (torch.abs(z) + 1e-8)
        return unit_keys

    def register_expert_address(self, key_vector: torch.Tensor, label: str = "general") -> int:
        """
        Registers a new learnable expert key into the hyperspace memory.
        key_vector: [1, D] (complex phasor)
        """
        # Ensure unit magnitude initialization
        angles = torch.angle(key_vector)
        init_r = torch.cos(angles).detach().float()
        init_i = torch.sin(angles).detach().float()
        
        param_r = nn.Parameter(init_r)
        param_i = nn.Parameter(init_i)
        
        self.keys_r.append(param_r)
        self.keys_i.append(param_i)
        
        expert_id = self.num_experts - 1
        self.expert_metadata.append({"id": expert_id, "label": label})

        # Initialize running similarity distribution for new expert
        if expert_id < self.max_experts:
            self.running_sim_mean[expert_id] = 0.35
            self.running_sim_var[expert_id] = 0.04
            self.usage_counts[expert_id] = 0

        return expert_id

    def compute_resonance(self, query_phasors: torch.Tensor) -> torch.Tensor:
        """
        Computes the semantic resonance (Hermitian cosine similarity) between input queries and learnable expert keys.
        query_phasors: [Batch, D] (unit complex phasors)
        Returns: [Batch, num_experts] in [-1.0, 1.0].
        If use_drift_guard=True during training, straight-through gates gradients for outlier tokens.
        """
        if self.num_experts == 0:
            raise ValueError("No experts registered in Hyperspace Memory.")
            
        keys = self.expert_keys # [N, D] in C^D
        qr = query_phasors.real
        qi = query_phasors.imag
        kr = keys.real
        ki = keys.imag
        
        sim = (torch.matmul(qr, kr.T) + torch.matmul(qi, ki.T)) / self.d_hyper # [B, N]

        if self.training and self.use_drift_guard and self.num_experts > 0:
            with torch.no_grad():
                for e in range(self.num_experts):
                    e_sim = sim[:, e]
                    mean_val = self.running_sim_mean[e]
                    std_val = torch.sqrt(self.running_sim_var[e] + 1e-6)
                    active_mask = e_sim > (mean_val - 2.5 * std_val)
                    if active_mask.any():
                        batch_mean = e_sim[active_mask].mean()
                        batch_var = torch.var(e_sim[active_mask], unbiased=False) if active_mask.sum() > 1 else torch.tensor(0.01, device=sim.device)
                        
                        alpha = 0.95
                        self.running_sim_mean[e] = alpha * self.running_sim_mean[e] + (1.0 - alpha) * batch_mean
                        self.running_sim_var[e] = (alpha * self.running_sim_var[e] + (1.0 - alpha) * batch_var).clamp(min=1e-4, max=1.0)
                        self.usage_counts[e] += active_mask.sum()

            std_expanded = torch.sqrt(self.running_sim_var[:self.num_experts] + 1e-6).unsqueeze(0) # [1, N]
            mean_expanded = self.running_sim_mean[:self.num_experts].unsqueeze(0) # [1, N]
            is_outlier = torch.abs(sim - mean_expanded) > (self.drift_sigma * std_expanded)
            
            # Straight-through gradient detachment: forward value exact, backward gradient zero for outliers
            sim = torch.where(is_outlier, sim.detach(), sim)

        return sim

    def evaluate_novelty(
        self, query_phasors: torch.Tensor, min_novel_token_fraction: float = 0.08
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, Optional[torch.Tensor]]:
        """
        Evaluates whether queries represent a novel domain requiring a new expert.
        Uses cluster-based quantile novelty detection rather than batch-mean.
        """
        resonance = self.compute_resonance(query_phasors) # [B, N]
        max_sim, _ = resonance.max(dim=-1) # [B]
        
        novel_mask = max_sim < self.spawn_threshold
        novel_fraction = novel_mask.float().mean().item()
        
        should_spawn = (novel_fraction >= min_novel_token_fraction) and (self.num_experts < self.max_experts)
        
        seed_vector = None
        if should_spawn:
            novel_tokens = query_phasors[novel_mask]
            centroid = novel_tokens.mean(dim=0, keepdim=True)
            seed_vector = ComplexPhasorVSA.normalize(centroid).detach()
            
        return resonance, max_sim, should_spawn, seed_vector

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """
        Computes soft orthogonality penalty among all active learnable expert keys:
        L_ortho = mean_{i != j} (Sim(K_i, K_j))^2
        """
        if self.num_experts <= 1:
            device = self.keys_r[0].device if self.num_experts > 0 else torch.device("cpu")
            return torch.tensor(0.0, device=device)
            
        keys = self.expert_keys # [N, D]
        kr = keys.real
        ki = keys.imag
        gram = (torch.matmul(kr, kr.T) + torch.matmul(ki, ki.T)) / self.d_hyper # [N, N]
        
        mask = ~torch.eye(self.num_experts, dtype=torch.bool, device=gram.device)
        off_diag = gram[mask]
        return torch.mean(off_diag ** 2)
