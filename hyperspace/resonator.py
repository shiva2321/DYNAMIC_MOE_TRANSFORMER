"""
Resonator Network in Complex Phasor Hyperspace with Grover-Inspired Amplitude Amplification.
Solves the vector-symbolic inverse problem:
Given a composite query S = x_1 (x) x_2 (x) ... (x) x_M + noise,
identifies each constituent x_m from discrete or continuous codebooks in O(sqrt(N)) iterations.
"""

import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperspace.vsa import ComplexPhasorVSA

class ResonatorFactorizer(nn.Module):
    """
    Multi-Codebook Resonator Network with Grover-Inspired Amplitude Amplification.
    """
    def __init__(
        self,
        codebooks: List[torch.Tensor],
        max_iterations: int = 15,
        convergence_tol: float = 1e-4,
        grover_amplification: float = 1.5,
    ):
        """
        codebooks: List of complex tensors, each of shape [num_items_m, D]
        """
        super().__init__()
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.grover_amplification = grover_amplification
        
        self.num_factors = len(codebooks)
        for i, cb in enumerate(codebooks):
            self.register_buffer(f"codebook_{i}", cb)

    def get_codebook(self, index: int) -> torch.Tensor:
        return getattr(self, f"codebook_{index}")

    def forward(self, query: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """
        query: [Batch, D] (complex phasor)
        Returns:
            factor_estimates: List of [Batch, D] converged complex phasor estimates
            factor_indices: List of [Batch] codebook indices for best match
            iterations: Number of steps executed
        """
        batch_size, d = query.shape
        device = query.device
        
        # 1. Initialize factor estimates as uniform superposition of each codebook
        estimates = []
        for i in range(self.num_factors):
            cb = self.get_codebook(i) # [M_i, D]
            init_superposition = ComplexPhasorVSA.bundle(cb.unsqueeze(0).expand(batch_size, -1, -1), dim=1)
            estimates.append(init_superposition)

        # 2. Resonator iterative relaxation loop with Grover Amplitude Amplification
        converged = False
        steps = 0
        for step in range(self.max_iterations):
            steps += 1
            max_delta = 0.0
            new_estimates = []
            
            for i in range(self.num_factors):
                # Compute unbinding key from all other factor estimates
                unbind_key = None
                for j in range(self.num_factors):
                    if j == i:
                        continue
                    if unbind_key is None:
                        unbind_key = estimates[j]
                    else:
                        unbind_key = ComplexPhasorVSA.bind(unbind_key, estimates[j])
                
                # Unbind query with unbind_key: unmasked = query (x) (unbind_key)^(-1)
                unmasked = ComplexPhasorVSA.unbind(query, unbind_key)
                
                # Project onto codebook i (Associative Memory inner products)
                cb = self.get_codebook(i) # [M_i, D]
                raw_sims = torch.matmul(unmasked, torch.conj(cb).T).real / d # [Batch, M_i]
                
                # Grover Quantum-Inspired Inversion-About-The-Mean Operator:
                # s_amplified = s + gamma * (s - mean(s))
                mean_sim = raw_sims.mean(dim=-1, keepdim=True)
                amplified_sims = raw_sims + self.grover_amplification * (raw_sims - mean_sim)
                
                # Non-linear sharp activation (temperature scaled softmax)
                weights = F.softmax(amplified_sims * 15.0, dim=-1) # [Batch, M_i]
                
                # Reconstruct cleaned-up phasor estimate: [Batch, D]
                weighted_cb = torch.matmul(weights.to(cb.dtype), cb)
                angles = torch.angle(weighted_cb)
                cleaned = torch.complex(torch.cos(angles), torch.sin(angles))
                
                delta = torch.norm(cleaned - estimates[i], dim=-1).mean().item()
                if delta > max_delta:
                    max_delta = delta
                    
                new_estimates.append(cleaned)
                
            estimates = new_estimates
            if max_delta < self.convergence_tol:
                converged = True
                break

        # 3. Retrieve final nearest codebook indices
        factor_indices = []
        for i in range(self.num_factors):
            cb = self.get_codebook(i)
            sims = torch.matmul(estimates[i], torch.conj(cb).T).real / d
            factor_indices.append(sims.argmax(dim=-1))
            
        return estimates, factor_indices, steps
