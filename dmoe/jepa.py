import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.encoder import UniversalMultimodalEncoder
from dmoe.experts import ComplexExpert

MASK_TOKEN_ID = 31999


class JEPAModule(nn.Module):
    """
    Wraps an EXISTING encoder by reference (not a copy). The encoder passed
    in is assumed to only ever process text while wrapped here, since the
    EMA update in update_target_encoder() zips context/target parameters
    positionally and assumes they stay structurally identical -- that
    breaks if the wrapped encoder's lazily-created numeric_projectors
    ModuleDict grows differently on one side than the other.
    """

    def __init__(
        self,
        encoder: UniversalMultimodalEncoder,
        predictor_hidden_dim: int,
        mask_token_id: int = MASK_TOKEN_ID,
        mask_ratio: float = 0.3,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = ComplexExpert(encoder.d_model, predictor_hidden_dim)
        self.mask_token_id = mask_token_id
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay

    def _masked_span(self, seq_len: int) -> Tuple[int, int]:
        mask_len = max(1, int(seq_len * self.mask_ratio))
        mask_len = min(mask_len, seq_len)
        start = int(torch.randint(0, seq_len - mask_len + 1, (1,)).item())
        return start, start + mask_len

    def compute_loss(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        start, end = self._masked_span(seq_len)

        masked_ids = token_ids.clone()
        masked_ids[:, start:end] = self.mask_token_id

        context_r, context_i = self.context_encoder(masked_ids)
        with torch.no_grad():
            target_r, target_i = self.target_encoder(token_ids)

        ctx_slice_r = context_r[:, start:end, :]
        ctx_slice_i = context_i[:, start:end, :]
        tgt_slice_r = target_r[:, start:end, :].detach()
        tgt_slice_i = target_i[:, start:end, :].detach()

        pred_r, pred_i = self.predictor(ctx_slice_r, ctx_slice_i)

        return F.mse_loss(pred_r, tgt_slice_r) + F.mse_loss(pred_i, tgt_slice_i)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        for target_p, context_p in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target_p.mul_(self.ema_decay).add_(context_p.detach() * (1.0 - self.ema_decay))

    def trainable_parameters(self):
        return self.predictor.parameters()
