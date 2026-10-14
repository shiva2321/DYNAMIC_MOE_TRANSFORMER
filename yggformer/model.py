"""
YggFormer: Complete language model and sequence transformer with Yggdrasil attention.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import YggCfg
from .block import YggBlock


class YggFormer(nn.Module):
    """
    Decoder-only Transformer with Yggdrasil Geometric Attention.
    Tied token embeddings and output projection.
    """

    def __init__(self, cfg: YggCfg):
        super().__init__()
        self.cfg = cfg
        self.vocab = cfg.vocab
        self.d_model = cfg.d_model
        self.max_len = cfg.max_len

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)

        self.blocks = nn.ModuleList([YggBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.RMSNorm(cfg.d_model)

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        record_stats: bool = False,
    ) -> torch.Tensor:
        if tokens is not None:
            B, T = tokens.shape
            if T > self.max_len:
                raise ValueError(f"Sequence length {T} exceeds maximum context {self.max_len}")
            positions = torch.arange(T, device=tokens.device)
            x = self.tok_emb(tokens) + self.pos_emb(positions)
        elif inputs_embeds is not None:
            B, T, _ = inputs_embeds.shape
            if T > self.max_len:
                raise ValueError(f"Sequence length {T} exceeds maximum context {self.max_len}")
            positions = torch.arange(T, device=inputs_embeds.device)
            x = inputs_embeds + self.pos_emb(positions)
        else:
            raise ValueError("Either tokens or inputs_embeds must be provided")

        for block in self.blocks:
            x = block(x, is_causal=is_causal, record_stats=record_stats)

        x = self.final_norm(x)
        # Weight tying with token embedding
        logits = F.linear(x, self.tok_emb.weight)
        return logits
