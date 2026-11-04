from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

BYTE_MASK_ID = 256
BYTE_PAD_ID = 257


class MultiScaleByteChunker(nn.Module):
    """
    Tokenizer-free byte stream chunker combining Charformer GBST-style
    multi-scale 1D convolutions (for shift-resilient local n-gram context)
    with MEGABYTE/BLT-style strided patch projection into d_model.
    """

    def __init__(
        self,
        d_model: int,
        chunk_size: int = 4,
        kernel_sizes: Tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.kernel_sizes = tuple(kernel_sizes)

        # 0..255 for raw UTF-8/binary bytes, 256 for BYTE_MASK_ID, 257 for BYTE_PAD_ID
        self.byte_embed = nn.Embedding(258, d_model, padding_idx=BYTE_PAD_ID)

        self.conv_bank = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, kernel_size=k, padding=0) for k in self.kernel_sizes]
        )
        self.scale_gate = nn.Linear(d_model, len(self.kernel_sizes))

        self.chunk_proj = nn.Linear(chunk_size * d_model, d_model)
        self.chunk_norm = nn.LayerNorm(d_model)

    @staticmethod
    def text_to_bytes(
        texts: Union[str, bytes, Sequence[Union[str, bytes]]],
        device: Union[torch.device, str, None] = None,
    ) -> torch.Tensor:
        if isinstance(texts, (str, bytes)):
            texts = [texts]
        encoded = [t.encode("utf-8") if isinstance(t, str) else bytes(t) for t in texts]
        max_len = max((len(b) for b in encoded), default=1)
        max_len = max(max_len, 1)
        out = torch.full((len(encoded), max_len), BYTE_PAD_ID, dtype=torch.long, device=device)
        for idx, b in enumerate(encoded):
            if len(b) > 0:
                out[idx, : len(b)] = torch.tensor(list(b), dtype=torch.long, device=device)
        return out

    @staticmethod
    def bytes_to_text(
        byte_tensor: torch.Tensor, mask_placeholder: bytes = b"\xef\xbf\xbd"
    ) -> List[str]:
        if byte_tensor.dim() == 3:
            bsz, s_chunks, c_size = byte_tensor.shape
            byte_tensor = byte_tensor.reshape(bsz, s_chunks * c_size)
        results = []
        for row in byte_tensor.detach().cpu().tolist():
            buf = bytearray()
            for val in row:
                v = int(val)
                if v == BYTE_PAD_ID:
                    continue
                elif v == BYTE_MASK_ID:
                    buf.extend(mask_placeholder)
                elif 0 <= v <= 255:
                    buf.append(v)
                else:
                    raise ValueError(f"Out-of-range byte/token ID in bytes_to_text: {v}")
            results.append(bytes(buf).decode("utf-8", errors="replace"))
        return results

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        if byte_ids.dim() != 2:
            raise ValueError(f"Expected 2D byte tensor [B, L], got shape {tuple(byte_ids.shape)}")

        ids = byte_ids.long().clamp(0, BYTE_PAD_ID)
        bsz, seq_len = ids.shape
        valid_mask = (ids != BYTE_PAD_ID).unsqueeze(-1).float()  # [B, L, 1]

        emb = self.byte_embed(ids) * valid_mask  # [B, L, d_model]
        emb_t = emb.transpose(1, 2)  # [B, d_model, L]

        scale_outputs = []
        for k, conv in zip(self.kernel_sizes, self.conv_bank):
            padded = F.pad(emb_t, (k - 1, 0)) if k > 1 else emb_t
            conv_out = F.gelu(conv(padded)).transpose(1, 2)  # [B, L, d_model]
            scale_outputs.append(conv_out)

        # Softmax-gated mixture across multi-scale n-gram convolutions [B, L, M, 1]
        gate_weights = F.softmax(self.scale_gate(emb), dim=-1).unsqueeze(-1)
        stacked = torch.stack(scale_outputs, dim=2)  # [B, L, M, d_model]
        ms_features = ((stacked * gate_weights).sum(dim=2) + emb) * valid_mask  # [B, L, d_model]

        remainder = seq_len % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            ms_features = F.pad(ms_features, (0, 0, 0, pad_len))

        num_chunks = ms_features.size(1) // self.chunk_size
        chunked = ms_features.reshape(bsz, num_chunks, self.chunk_size * self.d_model)
        return self.chunk_norm(self.chunk_proj(chunked))
