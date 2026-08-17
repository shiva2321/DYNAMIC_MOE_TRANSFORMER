"""
Shared Hyperspace Global Workspace / Communication Bus.
Enables active neural experts to broadcast and listen to bound messages in continuous superposition:
Psi_bus = Sum_{k in Active} (y_k (x) Role_k)
Allows experts to communicate across layers without all-to-all dense quadratic connectivity.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperspace.vsa import ComplexPhasorVSA

class HyperspaceGlobalBus(nn.Module):
    """
    Global Workspace Bus residing in Complex Phasor Hyperspace C^D.
    """
    def __init__(self, d_model: int, d_hyper: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.d_hyper = d_hyper
        
        # Projections between Model Space and Bus Space
        self.to_bus = nn.Linear(d_model, d_hyper, bias=False)
        self.from_bus = nn.Linear(d_hyper, d_model, bias=False)
        self.bus_gate = nn.Parameter(torch.zeros(1)) # Learnable gating multiplier initialized near 0

    def broadcast_and_listen(
        self,
        expert_outputs: torch.Tensor,
        expert_keys: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        expert_outputs: [Batch*Seq, K, d_model]
        expert_keys: [Batch*Seq, K, d_hyper] (complex phasors)
        expert_weights: [Batch*Seq, K, 1]
        
        Returns:
            bus_context: [Batch*Seq, K, d_model] (context received by each expert from all other active peers)
        """
        # 1. Project outputs to complex phasors
        real_msg = self.to_bus(expert_outputs) # [N, K, d_hyper]
        phasor_msg = ComplexPhasorVSA.project_real_to_phasor(real_msg) # [N, K, d_hyper] (complex)
        
        # 2. Bind each message with its sender expert's key: msg_k (x) Key_k
        bound_msgs = ComplexPhasorVSA.bind(phasor_msg, expert_keys) # [N, K, d_hyper]
        
        # 3. Superposition Broadcast: Combine all active messages onto the global bus
        # Global Bus State: [N, d_hyper]
        bus_state = ComplexPhasorVSA.bundle(bound_msgs, dim=1)
        
        # 4. Each expert listens by unbinding with its own key: Bus (x) Key_k^(-1)
        bus_state_expanded = bus_state.unsqueeze(1).expand_as(bound_msgs) # [N, K, d_hyper]
        received_phasors = ComplexPhasorVSA.unbind(bus_state_expanded, expert_keys) # [N, K, d_hyper]
        
        # 5. Project back to model dimension
        received_real = ComplexPhasorVSA.project_phasor_to_real(received_phasors) # [N, K, d_hyper]
        bus_context = self.from_bus(received_real) # [N, K, d_model]
        
        return torch.tanh(self.bus_gate) * bus_context
