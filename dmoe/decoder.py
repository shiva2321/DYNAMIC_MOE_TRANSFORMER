import torch
import torch.nn as nn


class UniversalProjectiveDecoder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int = 32000, continuous_dim: int = 8):
        super().__init__()
        self.d_model = d_model

        self.proj_real = nn.Linear(d_model, d_model)
        self.proj_imag = nn.Linear(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)

        self.token_head = nn.Linear(d_model, vocab_size)
        self.continuous_head = nn.Linear(d_model, continuous_dim)

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor, head: str = "token") -> torch.Tensor:
        unified_hidden = self.final_norm(self.proj_real(z_r) + self.proj_imag(z_i))
        if head == "token":
            return self.token_head(unified_hidden)
        elif head == "continuous":
            return self.continuous_head(unified_hidden)
        elif head == "latent":
            return unified_hidden
        else:
            raise ValueError(f"Unknown output head specification: {head}")

