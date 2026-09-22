"""
Two-Compartment Dendritic Expert with Somatic NMDA Gating.
Adapted from Universal SubStrait for Infinite Spherical Hyperspace MoE.

Compartmental Structure:
1. Basal Compartment: Feedforward sensory/token processing.
2. Apical Compartment: Top-down context from Spherical Hyperspace Memory.
3. Somatic Integration: Non-linear NMDA spike modulation.
   y_e = W_down( h_basal * (1 + beta_nmda * tanh(h_apical)) )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoCompartmentDendriticExpert(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        d_ff: int = 64,
        d_context: int = 32,
        beta_nmda: float = 0.5,
        device: torch.device = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_context = d_context
        self.beta_nmda = beta_nmda

        # Basal Compartment (Feedforward)
        self.w_gate_b = nn.Linear(d_model, d_ff, bias=False, device=device)
        self.w_up_b = nn.Linear(d_model, d_ff, bias=False, device=device)

        # Apical Compartment (Top-Down Memory Context)
        self.w_gate_a = nn.Linear(d_context, d_ff, bias=False, device=device)
        self.w_up_a = nn.Linear(d_context, d_ff, bias=False, device=device)

        # Somatic Down-Projection
        self.w_down = nn.Linear(d_ff, d_model, bias=False, device=device)

        self._init_weights()

    def _init_weights(self):
        # Small initialization to ensure stable gradients
        for w in [self.w_gate_b, self.w_up_b, self.w_gate_a, self.w_up_a]:
            nn.init.xavier_uniform_(w.weight, gain=1.0)
        nn.init.xavier_uniform_(self.w_down.weight, gain=0.5)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, d_model] Basal feedforward input
            context: [B, d_context] Apical context from hyperspace memory (optional)
        Returns:
            y: [B, d_model] Somatically integrated output
        """
        # 1. Basal Compartment: SwiGLU feedforward activation
        h_basal = F.silu(self.w_gate_b(x)) * self.w_up_b(x)

        # 2. Apical Compartment: Contextual modulation
        if context is not None:
            h_apical = F.silu(self.w_gate_a(context)) * self.w_up_a(context)
            # Somatic dendritic multiplication (NMDA plateau potential)
            modulation = 1.0 + self.beta_nmda * torch.tanh(h_apical)
            soma = h_basal * modulation
        else:
            soma = h_basal

        # 3. Somatic output down-projection
        y = self.w_down(soma)
        return y
