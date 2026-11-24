import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

BYTE_MASK_ID = 256
BYTE_PAD_ID = 257


class MultiScaleByteChunker(nn.Module):
    """
    Tokenizer-free byte stream chunker combining Charformer GBST-style
    multi-scale 1D convolutions (for shift-resilient local n-gram context),
    intra-chunk complex phase-rotation / cyclic roll binding, and
    MEGABYTE/BLT-style strided patch projection into d_model.
    """

    def __init__(
        self,
        d_model: int,
        chunk_size: int = 4,
        kernel_sizes: Tuple[int, ...] = (1, 2, 4, 8),
        shift_invariant_pool: bool = False,
    ):
        super().__init__()
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.kernel_sizes = tuple(kernel_sizes)
        self.shift_invariant_pool = bool(shift_invariant_pool)

        # 0..255 for raw UTF-8/binary bytes, 256 for BYTE_MASK_ID, 257 for BYTE_PAD_ID
        self.byte_embed = nn.Embedding(258, d_model, padding_idx=BYTE_PAD_ID)

        self.conv_bank = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, kernel_size=k, padding=0) for k in self.kernel_sizes]
        )
        self.scale_gate = nn.Linear(d_model, len(self.kernel_sizes))

        self.chunk_proj = nn.Linear(chunk_size * d_model, d_model)
        self.chunk_norm = nn.LayerNorm(d_model)

        # Deterministic intra-chunk phase-rotation basis (consumes zero RNG on init)
        slot_idx = torch.arange(chunk_size, dtype=torch.float32).unsqueeze(-1)
        dim_idx = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
        slot_angles = slot_idx * (2.0 * math.pi / max(chunk_size, 1)) + dim_idx * (
            math.pi / max(d_model, 1)
        )
        self.register_buffer("slot_phase_basis", torch.sin(slot_angles).unsqueeze(0).unsqueeze(0))
        self.slot_phase_gate = nn.Parameter(torch.tensor(0.0))
        if self.shift_invariant_pool:
            # Causal triangular BlurPool kernel for shift-resilient strided chunking (Zhang, ICML 2019)
            if chunk_size == 1:
                bw = torch.tensor([1.0], dtype=torch.float32)
            else:
                half = (chunk_size + 1) / 2.0
                idx_b = torch.arange(chunk_size, dtype=torch.float32) + 1.0
                bw = torch.minimum(idx_b, (chunk_size + 1.0) - idx_b)
                bw = (bw + 0.25) / (bw + 0.25).sum()
            self.register_buffer("blur_kernel", bw.view(1, 1, chunk_size))
        self.last_byte_features: Optional[torch.Tensor] = None
        self.last_pos_chunks: Optional[torch.Tensor] = None

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
        ms_raw = (stacked * gate_weights).sum(dim=2)
        ms_features = (ms_raw + emb) * valid_mask  # [B, L, d_model]
        pos_features = F.relu(ms_raw) * valid_mask  # Sparse nonnegative multi-scale k-gram detector

        if self.shift_invariant_pool and self.chunk_size > 1:
            # Strictly causal low-pass BlurPool before strided chunking reduces 1..3-byte shift aliasing
            bk = self.blur_kernel.to(dtype=ms_features.dtype, device=ms_features.device)
            bk_d = bk.expand(self.d_model, 1, self.chunk_size)
            ms_pad_t = F.pad(ms_features.transpose(1, 2), (self.chunk_size - 1, 0))
            ms_blur = F.conv1d(ms_pad_t, bk_d, groups=self.d_model).transpose(1, 2) * valid_mask
            pos_pad_t = F.pad(pos_features.transpose(1, 2), (self.chunk_size - 1, 0))
            pos_blur = F.conv1d(pos_pad_t, bk_d, groups=self.d_model).transpose(1, 2) * valid_mask
            ms_pool_src = 0.65 * ms_blur + 0.35 * ms_features
            pos_pool_src = 0.65 * pos_blur + 0.35 * pos_features
        else:
            ms_pool_src = ms_features
            pos_pool_src = pos_features

        remainder = seq_len % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            ms_features = F.pad(ms_features, (0, 0, 0, pad_len))
            ms_pool_src = F.pad(ms_pool_src, (0, 0, 0, pad_len))
            pos_pool_src = F.pad(pos_pool_src, (0, 0, 0, pad_len))
            valid_mask = F.pad(valid_mask, (0, 0, 0, pad_len))

        num_chunks = ms_features.size(1) // self.chunk_size
        clean_slots = ms_features.reshape(bsz, num_chunks, self.chunk_size, self.d_model)
        pool_slots = ms_pool_src.reshape(bsz, num_chunks, self.chunk_size, self.d_model)
        if self.shift_invariant_pool:
            vm_slots = valid_mask.reshape(bsz, num_chunks, self.chunk_size, 1)
            pre_phase_mean = (pool_slots * vm_slots).sum(dim=2) / vm_slots.sum(dim=2).clamp_min(1.0)
        else:
            pre_phase_mean = clean_slots.mean(dim=2)
        pos_slots = pos_pool_src.reshape(bsz, num_chunks, self.chunk_size, self.d_model)
        self.last_pos_chunks = 0.5 * pos_slots.mean(dim=2) + 0.5 * pos_slots.max(dim=2).values
        # Apply gated intra-chunk phase-rotation + cyclic roll binding per byte slot
        rolled = torch.roll(clean_slots, shifts=1, dims=-1)
        slot_features = clean_slots + torch.tanh(self.slot_phase_gate) * (
            clean_slots * self.slot_phase_basis + 0.25 * rolled
        )
        self.last_byte_features = slot_features
        chunked = slot_features.reshape(bsz, num_chunks, self.chunk_size * self.d_model)
        if self.shift_invariant_pool:
            return self.chunk_norm(0.45 * self.chunk_proj(chunked) + 0.85 * pre_phase_mean)
        return self.chunk_norm(self.chunk_proj(chunked) + 0.65 * pre_phase_mean)
