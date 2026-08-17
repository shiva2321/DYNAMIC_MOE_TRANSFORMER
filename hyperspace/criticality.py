"""
Self-Organized Criticality Controller.
Maintains neural computational dynamics at the Edge of Chaos (Branching Ratio sigma = 1.0).
Dynamically adjusts routing temperature and expert firing thresholds to maximize
Shannon information transmission and dynamic range without explosive chaos or dead quiescence.
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfOrganizedCriticalityController(nn.Module):
    """
    Monitors neural avalanche statistics and tunes routing dynamics to the critical point (sigma = 1.0).
    """
    def __init__(self, target_branching_ratio: float = 1.0, base_temperature: float = 10.0, adaptation_rate: float = 0.02):
        super().__init__()
        self.target_sigma = target_branching_ratio
        self.adaptation_rate = adaptation_rate
        
        # Dynamic running temperature buffer
        self.register_buffer("current_temperature", torch.tensor(base_temperature, dtype=torch.float32))
        self.register_buffer("prev_activity_level", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("running_branching_ratio", torch.tensor(1.0, dtype=torch.float32))

    def update_criticality(self, routing_logits: torch.Tensor, top_weights: torch.Tensor) -> Tuple[float, float]:
        """
        routing_logits: [Batch*Seq, num_experts]
        top_weights: [Batch*Seq, k]
        Returns:
            effective_temperature: float
            branching_ratio: float
        """
        # Measure current activity level via routing entropy: H = - sum(p * log(p))
        probs = F.softmax(routing_logits / self.current_temperature, dim=-1)
        entropy = - torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
        curr_activity = entropy.detach()
        
        # Compute branching ratio sigma = curr_activity / prev_activity
        if self.prev_activity_level > 1e-4:
            instant_sigma = (curr_activity / self.prev_activity_level).item()
        else:
            instant_sigma = 1.0

        # Update running exponential average of sigma
        self.running_branching_ratio = 0.9 * self.running_branching_ratio + 0.1 * instant_sigma
        self.prev_activity_level = curr_activity

        if self.training:
            # Self-Organizing Feedback Rule:
            # If sigma > target (Supercritical / excess chaotic firing) -> Increase temperature sharpness (higher contrast)
            # If sigma < target (Subcritical / dying signal) -> Lower temperature sharpness (broader excitation)
            if self.running_branching_ratio > (self.target_sigma + 0.05):
                self.current_temperature = self.current_temperature * (1.0 + self.adaptation_rate)
            elif self.running_branching_ratio < (self.target_sigma - 0.05):
                self.current_temperature = self.current_temperature * (1.0 - self.adaptation_rate)
                
            # Clamp to safe bounds [2.0, 30.0]
            self.current_temperature = torch.clamp(self.current_temperature, min=2.0, max=30.0)

        return self.current_temperature.item(), self.running_branching_ratio.item()
