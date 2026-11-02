"""The fast-slow expert: a closed-form part and a small network on the same inputs, several heads on top.

* fast part: a running LDA head (class means and one pooled covariance; learns in one pass, never forgets, fixed memory) = ``unimoe.heads.LdaHead`` through ``DensityHead``;
* slow part: a residual MLP expert (``growmoe.experts.MLPExpert``) trained by gradient steps;
* heads: label (the two parts blended in log-probability, ``lam`` = weight of the fast part), density (squared Mahalanobis distance to the nearest taught class, from the fast part's statistics),
  confidence (one minus the largest blended probability) and disagreement (Jensen-Shannon divergence between the fast and slow class distributions on the same input).
Nothing here edits growmoe/ or unimoe/.
"""

import numpy as np
import torch
from torch import Tensor

from growmoe.experts import MLPExpert
from unimoe.density import DensityHead
from unimoe.pools import blend


def jsd(p: Tensor, q: Tensor) -> Tensor:
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.clamp_min(1e-12).log() - b.clamp_min(1e-12).log())).sum(1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class FastSlow:
    def __init__(self, d: int, n_classes: int, hidden: int = 128, depth: int = 1, lam: float = 0.6, epochs: int = 40, lr: float = 3e-3, batch: int = 128, shrink: float = 0.1,
                 seed: int = 0) -> None:
        self.d, self.n_classes, self.lam, self.epochs, self.lr, self.batch, self.seed = d, n_classes, lam, epochs, lr, batch, seed
        self.fast = DensityHead(d, n_classes, shrink)
        self.slow = MLPExpert(d, hidden, depth, n_classes, seed=seed)

    def fit(self, X: Tensor, y: Tensor) -> "FastSlow":
        X, y = X.float(), y.long()
        self.fast.update(X, y)
        opt = torch.optim.Adam(self.slow.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            order = torch.randperm(X.shape[0], generator=g)
            for lo in range(0, X.shape[0], self.batch):
                idx = order[lo:lo + self.batch]
                self.slow.train_step(X[idx], y[idx], opt)
        self.slow.eval()
        return self

    # ---- heads
    def p_fast(self, X: Tensor) -> Tensor:
        return self.fast.proba(X.float())

    @torch.no_grad()
    def p_slow(self, X: Tensor) -> Tensor:
        self.slow.eval()
        return self.slow.proba(X.float())

    def p(self, X: Tensor) -> Tensor:
        return blend(self.p_slow(X), self.p_fast(X), self.lam)

    def density(self, X: Tensor) -> np.ndarray:
        return self.fast.distance(X.float()).numpy()

    def confidence(self, X: Tensor) -> np.ndarray:
        return (1.0 - self.p(X).max(1).values).numpy()

    def disagreement(self, X: Tensor) -> np.ndarray:
        return jsd(self.p_fast(X), self.p_slow(X)).numpy()

    def macs(self) -> dict:
        """Multiply-adds per input of each head (the slow part's forward pass is shared by every head that reads it)."""
        c = int(self.fast.seen().sum())
        h = self.slow.hidden
        slow = len(self.slow.blocks) * 2 * self.d * h + self.d * self.n_classes
        fast = self.d * self.d + c * self.d
        return {"density": fast, "fast": fast, "slow": slow, "label": fast + slow, "confidence": fast + slow, "disagreement": fast + slow}
