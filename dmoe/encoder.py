import math
from typing import Tuple

import torch
import torch.nn as nn

from dmoe.chunker import MultiScaleByteChunker


class UniversalMultimodalEncoder(nn.Module):
    def __init__(self, d_model: int, tokenizer_free: bool = False, chunk_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.tokenizer_free = tokenizer_free
        self.chunk_size = chunk_size

        if self.tokenizer_free:
            self.byte_chunker = MultiScaleByteChunker(d_model, chunk_size=chunk_size)
            self.text_embed = None
        else:
            self.text_embed = nn.Embedding(32000, d_model)
            self.byte_chunker = None
        self.text_phase_proj = nn.Linear(d_model, d_model)

        self.image_patch_proj = nn.Conv2d(in_channels=3, out_channels=d_model, kernel_size=4, stride=4)
        self.image_phase_proj = nn.Linear(d_model, d_model)

        self.numeric_projectors = nn.ModuleDict()
        self.num_phase_proj = nn.Linear(d_model, d_model)

        self.unify_real = nn.Linear(d_model, d_model)
        self.unify_imag = nn.Linear(d_model, d_model)

    def _get_numeric_projector(self, feature_dim: int, device: torch.device) -> nn.Linear:
        key = str(feature_dim)
        if key not in self.numeric_projectors:
            self.numeric_projectors[key] = nn.Linear(feature_dim, self.d_model).to(device)
        return self.numeric_projectors[key]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2 and not x.is_floating_point():
            if self.tokenizer_free:
                emb = self.byte_chunker(x)
            else:
                emb = self.text_embed(x)
            seq_len = emb.size(1)
            real_comp = self.unify_real(emb)
            positions = torch.arange(seq_len, device=x.device, dtype=torch.float32).unsqueeze(-1)
            omega = torch.exp(
                torch.arange(0, self.d_model, 2, device=x.device, dtype=torch.float32)
                * -(math.log(10000.0) / self.d_model)
            )
            phase_sin = torch.sin(positions * omega)
            phase_cos = torch.cos(positions * omega)
            phase_pos = torch.cat([phase_sin, phase_cos], dim=-1)[:, : self.d_model].unsqueeze(0)
            imag_comp = self.unify_imag(self.text_phase_proj(emb) + phase_pos)

        elif x.dim() == 4:
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
            patches = self.image_patch_proj(x)
            patches = patches.flatten(2).transpose(1, 2)
            real_comp = self.unify_real(patches)
            imag_comp = self.unify_imag(self.image_phase_proj(patches))

        elif x.dim() == 3:
            feature_dim = x.size(2)
            projector = self._get_numeric_projector(feature_dim, x.device)
            proj = projector(x)
            real_comp = self.unify_real(proj)
            imag_comp = self.unify_imag(self.num_phase_proj(proj))

        else:
            raise ValueError(f"Unsupported multimodal tensor shape: {tuple(x.shape)}")

        return real_comp, imag_comp
