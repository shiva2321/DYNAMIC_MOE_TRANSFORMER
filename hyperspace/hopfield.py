"""
Modern Hopfield Network (Dense Associative Memory).
Implements continuous energy landscape descent with exponential storage capacity (2^{D/2}):
xi^{(t+1)} = Y * Softmax(beta * Y^T * xi^{(t)})
Used to clean up noisy superposition vectors and restore pristine stored attractor states.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernHopfieldMemory(nn.Module):
    """
    Continuous Modern Hopfield Layer (Dense Associative Memory).
    """
    def __init__(self, d_dim: int, num_patterns: int = 32, beta: float = 8.0, max_iter: int = 3):
        super().__init__()
        self.d_dim = d_dim
        self.num_patterns = num_patterns
        self.beta = beta
        self.max_iter = max_iter
        
        # Stored memory patterns Y in R^(num_patterns x d_dim)
        self.stored_patterns = nn.Parameter(torch.randn(num_patterns, d_dim))
        nn.init.orthogonal_(self.stored_patterns)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes Hopfield Energy: E(xi) = - (1/beta) * log( sum_i exp(beta * y_i^T * xi) ) + 0.5 * ||xi||^2
        """
        norm_patterns = F.normalize(self.stored_patterns, p=2, dim=-1)
        norm_state = F.normalize(state, p=2, dim=-1)
        
        # Similarities: [..., num_patterns]
        sims = torch.matmul(norm_state, norm_patterns.T)
        lse = torch.logsumexp(self.beta * sims, dim=-1)
        energy = - (1.0 / self.beta) * lse + 0.5 * torch.sum(state ** 2, dim=-1)
        return energy

    def forward(self, query_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs iterative Hopfield energy minimization to retrieve nearest clean attractor state.
        query_state: [..., d_dim]
        Returns:
            cleaned_state: [..., d_dim]
            energy_val: [...]
        """
        norm_patterns = F.normalize(self.stored_patterns, p=2, dim=-1) # [M, D]
        curr_state = query_state
        
        for _ in range(self.max_iter):
            norm_state = F.normalize(curr_state, p=2, dim=-1)
            # Dot products with stored patterns: [..., M]
            sims = torch.matmul(norm_state, norm_patterns.T) * math.sqrt(self.d_dim)
            weights = F.softmax(self.beta * sims, dim=-1) # [..., M]
            
            # Update state towards attractor
            curr_state = torch.matmul(weights, norm_patterns)
            
        energy_val = self.energy(curr_state)
        return curr_state, energy_val
