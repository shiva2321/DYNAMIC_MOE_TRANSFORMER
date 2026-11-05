from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from dmoe.chunker import BYTE_PAD_ID


class ExactLdaFastMemory(nn.Module):
    """
    Closed-form Linear Discriminant / Ridge sufficient-statistics fast memory,
    adapted from spherical_hyperspace_memory (unimoe/heads.py & v1/downdate.py).

    Stores exact float64 sufficient statistics:
      - n[c]: count of observations for class c
      - sum_x[c]: sum of feature vectors for class c
      - scatter: shared second-moment matrix sum_i x_i x_i^T

    Supports:
      - absorb(x, y): O(1)-step closed-form learning without gradient descent
      - downdate(x, y): exact bit-identical unlearning of any subset of data
      - predict_logits(x): closed-form Cholesky discriminant logits
    """

    def __init__(self, d_in: int, num_classes: int, reg: float = 1.0):
        super().__init__()
        if d_in <= 0 or num_classes <= 0 or reg <= 0.0:
            raise ValueError("d_in, num_classes, and reg must be positive")
        self.d_in = d_in
        self.num_classes = num_classes
        self.reg = float(reg)

        self.register_buffer("n", torch.zeros(num_classes, dtype=torch.float64))
        self.register_buffer("sum_x", torch.zeros(num_classes, d_in, dtype=torch.float64))
        self.register_buffer("scatter", torch.zeros(d_in, d_in, dtype=torch.float64))
        self.register_buffer("total", torch.tensor(0.0, dtype=torch.float64))

        self._dirty = True
        self._w: Optional[torch.Tensor] = None
        self._b: Optional[torch.Tensor] = None

    def _filter_valid(
        self, x: torch.Tensor, y: torch.Tensor, ignore_index: int = BYTE_PAD_ID
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_flat = x.detach().reshape(-1, self.d_in).to(dtype=torch.float64, device=self.n.device)
        y_flat = y.detach().reshape(-1).to(dtype=torch.long, device=self.n.device)
        valid = (y_flat != ignore_index) & (y_flat >= 0) & (y_flat < self.num_classes)
        return x_flat[valid], y_flat[valid]

    @torch.no_grad()
    def absorb(self, x: torch.Tensor, y: torch.Tensor, ignore_index: int = BYTE_PAD_ID) -> None:
        xv, yv = self._filter_valid(x, y, ignore_index=ignore_index)
        if xv.numel() == 0:
            return
        ones = torch.ones(yv.size(0), dtype=torch.float64, device=self.n.device)
        self.n.index_add_(0, yv, ones)
        self.sum_x.index_add_(0, yv, xv)
        self.scatter.add_(xv.T @ xv)
        self.total.add_(float(xv.size(0)))
        self._dirty = True

    @torch.no_grad()
    def downdate(self, x: torch.Tensor, y: torch.Tensor, ignore_index: int = BYTE_PAD_ID) -> None:
        xv, yv = self._filter_valid(x, y, ignore_index=ignore_index)
        if xv.numel() == 0:
            return
        ones = torch.ones(yv.size(0), dtype=torch.float64, device=self.n.device)
        self.n.index_add_(0, yv, -ones)
        self.n.clamp_min_(0.0)
        self.sum_x.index_add_(0, yv, -xv)
        zero_mask = self.n == 0.0
        if zero_mask.any():
            self.sum_x[zero_mask] = 0.0
        self.scatter.sub_(xv.T @ xv)
        self.total.sub_(float(xv.size(0))).clamp_min_(0.0)
        self._dirty = True

    @torch.no_grad()
    def _solve(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._dirty and self._w is not None and self._b is not None:
            return self._w, self._b

        seen = self.n > 0.0
        w = torch.zeros(self.num_classes, self.d_in, dtype=torch.float64, device=self.n.device)
        b = torch.full((self.num_classes,), -1e9, dtype=torch.float64, device=self.n.device)

        if not seen.any() or self.total.item() <= 0.0:
            self._w, self._b = w, b
            self._dirty = False
            return w, b

        mu = torch.zeros_like(self.sum_x)
        mu[seen] = self.sum_x[seen] / self.n[seen].unsqueeze(1)

        total_n = max(self.total.item(), 1.0)
        between = (mu[seen] * self.n[seen].unsqueeze(1)).T @ mu[seen]
        cov = (self.scatter - between) / total_n
        cov = 0.5 * (cov + cov.T)
        eye = torch.eye(self.d_in, dtype=torch.float64, device=self.n.device)
        reg_cov = cov + self.reg * eye

        try:
            chol = torch.linalg.cholesky(reg_cov)
            w_seen = torch.cholesky_solve(mu[seen].T, chol).T
        except RuntimeError:
            w_seen = torch.linalg.solve(reg_cov + 1e-3 * eye, mu[seen].T).T

        w[seen] = w_seen
        prior = torch.log(self.n[seen] / total_n)
        b[seen] = -0.5 * (mu[seen] * w_seen).sum(dim=1) + prior

        self._w, self._b = w, b
        self._dirty = False
        return w, b

    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.d_in).to(dtype=torch.float64, device=self.n.device)
        w, b = self._solve()
        logits = x_flat @ w.T + b.unsqueeze(0)
        return logits.to(dtype=x.dtype).reshape(*orig_shape, self.num_classes)


class MultiSlotExactLdaMemory(nn.Module):
    """
    Slot-indexed bank of ExactLdaFastMemory heads for chunked byte-stream
    or token-stream prediction. For chunk_size=W, slot s in 0..W-1 predicts
    the s-th byte in each chunk from the concatenated [chunk_latent, slot_feature].
    """

    def __init__(self, d_in: int, num_classes: int = 256, num_slots: int = 4, reg: float = 0.5):
        super().__init__()
        self.d_in = d_in
        self.num_classes = num_classes
        self.num_slots = num_slots
        self.slots = nn.ModuleList(
            [ExactLdaFastMemory(d_in=d_in, num_classes=num_classes, reg=reg) for _ in range(num_slots)]
        )

    @torch.no_grad()
    def absorb_slots(
        self, slot_features: torch.Tensor, targets: torch.Tensor, ignore_index: int = BYTE_PAD_ID
    ) -> None:
        """
        slot_features: [B, S_chunks, num_slots, d_in]
        targets: [B, L] where L <= S_chunks * num_slots
        """
        bsz, s_chunks, n_slots, _ = slot_features.shape
        padded_targets = torch.full(
            (bsz, s_chunks * n_slots), ignore_index, dtype=torch.long, device=targets.device
        )
        valid_len = min(targets.size(1), s_chunks * n_slots)
        padded_targets[:, :valid_len] = targets[:, :valid_len]
        reshaped_targets = padded_targets.reshape(bsz, s_chunks, n_slots)

        for s in range(n_slots):
            self.slots[s].absorb(slot_features[:, :, s, :], reshaped_targets[:, :, s], ignore_index=ignore_index)

    @torch.no_grad()
    def downdate_slots(
        self, slot_features: torch.Tensor, targets: torch.Tensor, ignore_index: int = BYTE_PAD_ID
    ) -> None:
        bsz, s_chunks, n_slots, _ = slot_features.shape
        padded_targets = torch.full(
            (bsz, s_chunks * n_slots), ignore_index, dtype=torch.long, device=targets.device
        )
        valid_len = min(targets.size(1), s_chunks * n_slots)
        padded_targets[:, :valid_len] = targets[:, :valid_len]
        reshaped_targets = padded_targets.reshape(bsz, s_chunks, n_slots)

        for s in range(n_slots):
            self.slots[s].downdate(slot_features[:, :, s, :], reshaped_targets[:, :, s], ignore_index=ignore_index)

    @torch.no_grad()
    def predict_slot_logits(self, slot_features: torch.Tensor) -> torch.Tensor:
        """
        slot_features: [B, S_chunks, num_slots, d_in]
        Returns: [B, S_chunks, num_slots, num_classes]
        """
        out_slots: List[torch.Tensor] = []
        for s in range(self.num_slots):
            out_slots.append(self.slots[s].predict_logits(slot_features[:, :, s, :]))
        return torch.stack(out_slots, dim=2)

