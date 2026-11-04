import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.encoder import DEFAULT_VOCAB_MASK_ID, UniversalMultimodalEncoder
from dmoe.experts import ComplexExpert

MASK_TOKEN_ID = DEFAULT_VOCAB_MASK_ID


class JEPAModule(nn.Module):
    """
    Wraps an EXISTING encoder by reference (not a copy). The encoder passed
    in is assumed to only ever process text/byte sequences while wrapped here,
    since the EMA update in update_target_encoder() zips context/target
    parameters positionally and verifies they remain structurally identical.
    """

    def __init__(
        self,
        encoder: UniversalMultimodalEncoder,
        predictor_hidden_dim: int,
        mask_token_id: Optional[int] = None,
        mask_ratio: float = 0.3,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = ComplexExpert(encoder.d_model, predictor_hidden_dim)
        self.mask_token_id = (
            encoder.default_mask_token_id if mask_token_id is None else mask_token_id
        )
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay

    def _masked_span(self, seq_len: int) -> Tuple[int, int]:
        mask_len = max(1, int(seq_len * self.mask_ratio))
        mask_len = min(mask_len, seq_len)
        start = int(torch.randint(0, seq_len - mask_len + 1, (1,)).item())
        return start, start + mask_len

    def compute_loss(self, token_ids: torch.Tensor) -> torch.Tensor:
        step = self.context_encoder.effective_chunk_size
        seq_len = (token_ids.size(1) + step - 1) // step
        start, end = self._masked_span(seq_len)

        masked_ids = token_ids.clone()
        if self.context_encoder.tokenizer_free:
            masked_ids = masked_ids.long()
        masked_ids[:, start * step : end * step] = self.mask_token_id

        context_r, context_i = self.context_encoder(masked_ids)
        with torch.no_grad():
            target_r, target_i = self.target_encoder(token_ids)

        ctx_slice_r = context_r[:, start:end, :]
        ctx_slice_i = context_i[:, start:end, :]
        tgt_slice_r = target_r[:, start:end, :].detach()
        tgt_slice_i = target_i[:, start:end, :].detach()

        if self.context_encoder.tokenizer_free:
            unmasked = [idx for idx in range(seq_len) if idx < start or idx >= end]
            if unmasked:
                vis_r = context_r[:, unmasked, :]
                vis_i = context_i[:, unmasked, :]
                ctx_pool_r = vis_r.mean(dim=1, keepdim=True)
                ctx_pool_i = vis_i.mean(dim=1, keepdim=True)
            else:
                vis_r = context_r
                vis_i = context_i
                ctx_pool_r = torch.zeros_like(context_r[:, :1, :])
                ctx_pool_i = torch.zeros_like(context_i[:, :1, :])

            pred_r, pred_i = self.predictor(ctx_slice_r + ctx_pool_r, ctx_slice_i + ctx_pool_i)
            pred_loss = F.mse_loss(pred_r, tgt_slice_r) + F.mse_loss(pred_i, tgt_slice_i)

            std_r = torch.sqrt(vis_r.var(dim=(0, 1), unbiased=False) + 1e-4)
            std_i = torch.sqrt(vis_i.var(dim=(0, 1), unbiased=False) + 1e-4)
            var_loss = F.relu(1.0 - std_r).mean() + F.relu(1.0 - std_i).mean()
            return pred_loss + 0.1 * var_loss

        pred_r, pred_i = self.predictor(ctx_slice_r, ctx_slice_i)
        return F.mse_loss(pred_r, tgt_slice_r) + F.mse_loss(pred_i, tgt_slice_i)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        target_params = list(self.target_encoder.parameters())
        context_params = list(self.context_encoder.parameters())
        if len(target_params) != len(context_params):
            raise RuntimeError(
                f"Target and context encoders have mismatched parameter counts: "
                f"{len(target_params)} != {len(context_params)}"
            )
        for target_p, context_p in zip(target_params, context_params, strict=True):
            if target_p.shape != context_p.shape:
                raise RuntimeError(
                    f"Target and context parameter shape mismatch: "
                    f"{tuple(target_p.shape)} != {tuple(context_p.shape)}"
                )
            target_p.mul_(self.ema_decay).add_(context_p.detach() * (1.0 - self.ema_decay))

    def trainable_parameters(self):
        return self.predictor.parameters()
