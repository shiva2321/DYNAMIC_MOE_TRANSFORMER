"""A novelty reader on the sphere built from the same running statistics as the LDA head: how far, in the pooled-covariance metric, a message is from the nearest class that was taught.
Kept apart from ``heads.py`` on purpose: that file is part of the code stamp every cached P4 cell is keyed on, and an addition there would have made all of them look stale."""

import torch
from torch import Tensor

from .heads import RIDGE, LdaHead


class DensityHead(LdaHead):
    def __init__(self, d: int, n_classes: int, shrink: float = 0.1) -> None:
        super().__init__(d, n_classes, shrink)
        self._prec = None

    def update(self, x: Tensor, y: Tensor) -> None:
        super().update(x, y)
        self._prec = None

    def _precision(self) -> Tensor:
        if self._prec is None:
            seen = self.seen()
            mu = self.s[seen] / self.n[seen, None]
            cov = (self.S - (mu.T * self.n[seen]) @ mu) / max(self.total - int(seen.sum()), 1)
            cov = 0.5 * (cov + cov.T)
            cov = cov + (self.shrink * float(torch.trace(cov)) / self.d + RIDGE) * torch.eye(self.d, dtype=torch.float64)
            self._prec = torch.linalg.inv(cov)
        return self._prec

    def distance(self, x: Tensor) -> Tensor:
        """Squared Mahalanobis distance from each row to the NEAREST taught class mean: small for text like what was taught, large for anything else. A novelty score."""
        seen = self.seen()
        if not bool(seen.any()):
            raise ValueError("the head has seen no class")
        prec = self._precision()
        mu = self.s[seen] / self.n[seen, None]
        a = mu @ prec
        xs = x.detach().double()
        d2 = ((xs @ prec) * xs).sum(1, keepdim=True) - 2.0 * xs @ a.T + (mu * a).sum(1)
        return d2.min(1).values.clamp_min(0.0).float()
