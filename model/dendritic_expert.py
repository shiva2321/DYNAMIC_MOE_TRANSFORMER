"""
Two-Compartment Dendritic Pyramidal Micro-Expert.
Models the biological biophysics of pyramidal cortical neurons:
- Basal Dendrites: Process bottom-up token representations (feedforward drive).
- Apical Dendrites: Receive top-down context from the Hyperspace Global Bus (contextual modulation).
- Somatic Integration: Non-linear NMDA coincidence gate combining basal and apical drives:
  h_soma = h_basal + alpha * h_apical + beta * (h_basal (*) h_apical)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoCompartmentDendriticExpert(nn.Module):
    """
    Biological Two-Compartment Neural Expert.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, device: Optional[torch.device] = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        kwargs = {"device": device} if device is not None else {}
        
        # 1. Basal Dendritic Compartment (Bottom-Up Sensory Feedforward)
        self.basal_gate = nn.Linear(d_model, d_ff, bias=False, **kwargs)
        self.basal_up = nn.Linear(d_model, d_ff, bias=False, **kwargs)
        
        # 2. Apical Dendritic Compartment (Top-Down Hyperspace Workspace Feedback)
        self.apical_gate = nn.Linear(d_model, d_ff, bias=False, **kwargs)
        self.apical_up = nn.Linear(d_model, d_ff, bias=False, **kwargs)
        
        # 3. Learnable Dendritic Coincidence Gains (Apical linear gain & NMDA multiplicative gain)
        self.alpha_apical = nn.Parameter(torch.tensor(0.1, **kwargs))
        self.beta_nmda = nn.Parameter(torch.tensor(0.5, **kwargs))
        
        # 4. Somatic Output Projection
        self.w_down = nn.Linear(d_ff, d_model, bias=False, **kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize with scaled normals
        nn.init.normal_(self.basal_gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.basal_up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.apical_gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.apical_up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_down.weight, mean=0.0, std=0.02)

    def forward(self, x_basal: torch.Tensor, c_apical: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_basal: [num_tokens, d_model] (bottom-up feedforward input)
        c_apical: Optional [num_tokens, d_model] (top-down contextual feedback from Hyperspace Bus)
        Returns:
            y: [num_tokens, d_model]
        """
        # 1. Basal computation: SiLU(x * W_gate) * (x * W_up)
        h_basal = F.silu(self.basal_gate(x_basal)) * self.basal_up(x_basal)
        
        if c_apical is not None:
            # 2. Apical computation: SiLU(c * W_gate) * (c * W_up)
            h_apical = F.silu(self.apical_gate(c_apical)) * self.apical_up(c_apical)
            
            # 3. Active NMDA Somatic Coincidence Detection
            # h_soma = h_basal + alpha * h_apical + beta * (h_basal * h_apical)
            h_soma = h_basal + (self.alpha_apical * h_apical) + (self.beta_nmda * (h_basal * h_apical))
        else:
            h_soma = h_basal

        h_soma = self.dropout(h_soma)
        return self.w_down(h_soma)
