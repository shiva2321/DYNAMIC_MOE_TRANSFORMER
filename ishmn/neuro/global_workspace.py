"""
Predictive Global Workspace & Hyperspace Superposition Bus.
Harvested from E:\\universal_substrait\\hyperspace\\bus.py and E:\\unified\\predictive_gwt.py.

Key Features:
1. Complex Phasor Global Workspace Bus:
   Enables active neural experts to broadcast and listen to bound messages in continuous superposition:
       Psi_bus = Sum_{k in Active} (y_k (x) Key_k)
   Allows E experts to communicate without O(E^2) all-to-all quadratic connectivity.
2. Predictive GWT Ticket Arena:
   Processors/experts bid with interpretations based on resonance to prediction error.
   The winner broadcasts globally to update all processor states (conscious workspace).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperspaceGlobalBus(nn.Module):
    """
    Global Workspace Bus residing in continuous phasor superposition space.
    Communicates across active MoE experts in O(E) linear time instead of O(E^2) quadratic cross-attention.
    """
    def __init__(self, d_model: int = 32, d_hyper: int = 64, device: Optional[torch.device] = None):
        super().__init__()
        self.d_model = d_model
        self.d_hyper = d_hyper
        self.device = device or torch.device("cpu")

        self.to_bus = nn.Linear(d_model, d_hyper, bias=False, device=self.device)
        self.from_bus = nn.Linear(d_hyper, d_model, bias=False, device=self.device)
        self.bus_gate = nn.Parameter(torch.zeros(1, device=self.device))

        nn.init.xavier_uniform_(self.to_bus.weight)
        nn.init.xavier_uniform_(self.from_bus.weight)

    def broadcast_and_listen(
        self,
        expert_outputs: torch.Tensor,
        expert_keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            expert_outputs: [B, K, d_model]
            expert_keys: [B, K, d_hyper] (unit normalized)
        Returns:
            bus_context: [B, K, d_model]
        """
        B, K, D = expert_outputs.shape

        # 1. Project outputs to bus space
        bus_msgs = F.normalize(self.to_bus(expert_outputs), p=2, dim=-1) # [B, K, d_hyper]

        # 2. Key binding via circular correlation / Hadamard product
        bound_msgs = bus_msgs * expert_keys # [B, K, d_hyper]

        # 3. Superposition broadcast: bundle across all active experts
        bus_state = bound_msgs.sum(dim=1, keepdim=True) # [B, 1, d_hyper]

        # 4. Each expert unbinds with its own key: bus_state * expert_key
        received_msgs = bus_state * expert_keys # [B, K, d_hyper]

        # 5. Project back to model space
        bus_context = self.from_bus(received_msgs) # [B, K, d_model]
        gate = torch.tanh(self.bus_gate)

        return gate * bus_context


@dataclass
class GWTTicket:
    processor_id: str
    interpretation: torch.Tensor
    resonance: float
    confidence: float


class PredictiveGlobalWorkspace:
    """
    Predictive Global Workspace Theory (GWT) Manager.
    Arbitrates conscious access among competing expert modules via error-driven resonance.
    """
    def __init__(self, surprise_threshold: float = 0.35):
        self.surprise_threshold = surprise_threshold
        self.last_broadcast: Optional[GWTTicket] = None
        self.broadcast_history: List[Dict[str, Any]] = []

    def arbitrate(
        self,
        tickets: List[GWTTicket]
    ) -> Optional[GWTTicket]:
        """
        Conducts GWT ticket competition. The ticket with highest resonance wins conscious broadcast.
        """
        if not tickets:
            return None

        # Sort by resonance * confidence
        ranked = sorted(tickets, key=lambda t: t.resonance * t.confidence, reverse=True)
        winner = ranked[0]

        self.last_broadcast = winner
        self.broadcast_history.append({
            "winner": winner.processor_id,
            "resonance": winner.resonance,
            "confidence": winner.confidence
        })
        return winner
