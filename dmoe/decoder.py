import torch
import torch.nn as nn


class UniversalProjectiveDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int = 32000,
        continuous_dim: int = 8,
        tokenizer_free: bool = False,
        chunk_size: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.tokenizer_free = tokenizer_free
        self.chunk_size = chunk_size

        self.proj_real = nn.Linear(d_model, d_model)
        self.proj_imag = nn.Linear(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)

        if self.tokenizer_free:
            self.token_head = None
            self.byte_chunk_head = nn.Linear(d_model, chunk_size * 256)
        else:
            self.token_head = nn.Linear(d_model, vocab_size)
            self.byte_chunk_head = None
        self.continuous_head = nn.Linear(d_model, continuous_dim)

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor, head: str = "token") -> torch.Tensor:
        unified_hidden = self.final_norm(self.proj_real(z_r) + self.proj_imag(z_i))
        if head == "byte_chunk" or (head == "token" and self.tokenizer_free):
            if self.byte_chunk_head is None:
                raise ValueError("byte_chunk head requires tokenizer_free=True on UniversalProjectiveDecoder")
            bsz, seq_len, _ = unified_hidden.shape
            return self.byte_chunk_head(unified_hidden).view(bsz, seq_len, self.chunk_size, 256)
        elif head == "token":
            return self.token_head(unified_hidden)
        elif head == "continuous":
            return self.continuous_head(unified_hidden)
        elif head == "latent":
            return unified_hidden
        else:
            raise ValueError(f"Unknown output head specification: {head}")
