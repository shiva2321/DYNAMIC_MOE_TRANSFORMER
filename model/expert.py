"""
Lightweight Neural Micro-Expert for the Hyperspace MoE.
Uses SwiGLU feed-forward architecture (Shazeer, 2020 / LLaMA standard) for high capacity-to-parameter ratio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroExpert(nn.Module):
    """
    SwiGLU Neural Micro-Expert:
    FFN_SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize weights with standard scaled normal
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_down.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [num_tokens, d_model]
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        return self.w_down(hidden)
