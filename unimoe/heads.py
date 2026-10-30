"""A running linear discriminant (LDA) head: class means and one pooled covariance kept as sufficient statistics, so it learns in one pass, never forgets what it has seen, and costs a fixed amount
of memory. It is the closed-form fit that online gradient steps only approach; the pool uses it as a second opinion inside an expert (``pools.UniPool``) and, over domain ids, as a domain head."""

import torch
from torch import Tensor

MASK = -1e9
RIDGE = 1e-4                                                     # absolute floor on the covariance: inputs are unit vectors, a direction's variance is of the order 1 / d


class LdaHead:
    def __init__(self, d: int, n_classes: int, shrink: float = 0.1) -> None:
        self.d, self.n_classes, self.shrink = d, n_classes, shrink
        self.n = torch.zeros(n_classes, dtype=torch.float64)
        self.s = torch.zeros(n_classes, d, dtype=torch.float64)
        self.S = torch.zeros(d, d, dtype=torch.float64)
        self.total = 0
        self._fit_cache = None

    def update(self, x: Tensor, y: Tensor) -> None:
        x, y = x.detach().double(), y.detach().long()
        for c in y.unique().tolist():
            m = y == c
            self.n[c] += int(m.sum())
            self.s[c] += x[m].sum(0)
        self.S += x.T @ x
        self.total += int(y.numel())
        self._fit_cache = None

    def seen(self) -> Tensor:
        return self.n > 0

    def _fit(self):
        if self._fit_cache is None:
            seen = self.seen()
            mu = torch.zeros_like(self.s)
            mu[seen] = self.s[seen] / self.n[seen, None]
            within = self.S - (mu.T * self.n) @ mu
            dof = max(self.total - int(seen.sum()), 1)
            cov = within / dof
            cov = 0.5 * (cov + cov.T)
            ridge = self.shrink * float(torch.trace(cov)) / self.d + RIDGE
            cov = cov + ridge * torch.eye(self.d, dtype=torch.float64)
            a = torch.linalg.solve(cov, mu.T).T                               # row c = cov^-1 mu_c
            const = -0.5 * (mu * a).sum(1)
            const[seen] += torch.log(self.n[seen] / self.total)
            self._fit_cache = (a, const)
        return self._fit_cache

    def logits(self, x: Tensor) -> Tensor:
        a, const = self._fit()
        z = x.detach().double() @ a.T + const
        z[:, ~self.seen()] = MASK
        return z.float()

    def proba(self, x: Tensor) -> Tensor:
        return torch.softmax(self.logits(x), dim=-1)

    def n_scalars(self) -> int:
        """What it stores: the class sums, the pooled scatter, the class counts and the total."""
        return self.n_classes * self.d + self.d * self.d + self.n_classes + 1
