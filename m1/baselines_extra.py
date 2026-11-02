"""Extra baselines for M1 benchmark: MahalanobisProto (A5b) and RandomFeatureRidge (A6).

A5b: Prototype classifier with per-class covariance shrunk towards pooled covariance.
A6: Random feature projection with closed-form ridge regression.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from growmoe.baselines import _Base
from growmoe.config import CommonCfg
from growmoe.memory_baselines import _MemoryPool
from growmoe.types import Prediction, StepRecord


class MahalanobisProto(_MemoryPool):
    """Arm A5b: Nearest class prototype with class-specific Mahalanobis distance.

    Maintains incremental sufficient statistics (count, sum, sum of outer products) per class.
    Class covariance is shrunk towards the pooled covariance by shrinkage alpha:
        Sigma_shrunk_c = (1 - alpha) * Sigma_c + alpha * Sigma_pooled + epsilon * I
    Score is negative squared Mahalanobis distance:
        score_c = - (x - mu_c)^T (Sigma_shrunk_c)^(-1) (x - mu_c)
    Probabilities obtained via softmax(scale * score).
    """

    name = "M_mahalanobis_proto"

    def __init__(self, cfg: Optional[CommonCfg] = None, shrinkage: float = 0.1, scale: float = 1.0) -> None:
        super().__init__(cfg)
        self.shrinkage = float(shrinkage)
        self.scale = float(scale)
        self._cnt = torch.zeros(self.n_classes, dtype=torch.float64)
        self._sum = torch.zeros(self.n_classes, self.d, dtype=torch.float64)
        self._outer = torch.zeros(self.n_classes, self.d, self.d, dtype=torch.float64)
        self._inv_covs: Optional[Tensor] = None
        self._means: Optional[Tensor] = None

    def stored_vectors(self) -> int:
        """Number of d-vectors held: 1 mean vector + d covariance vectors per class seen."""
        seen = int((self._cnt > 0).sum().item())
        return seen * (1 + self.d)

    def lookup_vectors(self) -> int:
        """Lookup vectors for pool_report accounting."""
        return max(1, int((self._cnt > 0).sum().item()))

    def observe(self, x: Tensor, y: Optional[Tensor] = None, ctx: Optional[Tensor] = None) -> StepRecord:
        """Update incremental statistics with batch (x, y)."""
        self.step += 1
        if y is not None:
            xd = self._unit(x).double()
            yd = y.long()
            for i in range(xd.shape[0]):
                c = int(yd[i])
                xi = xd[i]
                self._cnt[c] += 1.0
                self._sum[c] += xi
                self._outer[c] += torch.outer(xi, xi)
            self._inv_covs = None
            self._means = None
        return self._record(x.shape[0], -1 if y is None else 0, n_experts=1)

    def _update_parameters(self) -> None:
        """Compute means and inverted shrunk covariances for all seen classes."""
        seen_mask = self._cnt > 0
        means = torch.zeros(self.n_classes, self.d, dtype=torch.float64)
        covs = torch.zeros(self.n_classes, self.d, self.d, dtype=torch.float64)

        total_samples = float(self._cnt.sum().item())
        if total_samples <= 0:
            return

        # 1. Compute per-class sample covariances and accumulate pooled covariance
        pooled_cov = torch.zeros(self.d, self.d, dtype=torch.float64)
        for c in range(self.n_classes):
            n_c = float(self._cnt[c].item())
            if n_c > 0:
                mu_c = self._sum[c] / n_c
                means[c] = mu_c
                # Sigma_c = (1/n_c) * (sum x x^T - n_c * mu_c mu_c^T)
                scatter_c = self._outer[c] - n_c * torch.outer(mu_c, mu_c)
                cov_c = scatter_c / n_c
                covs[c] = cov_c
                pooled_cov += scatter_c

        pooled_cov = pooled_cov / max(1.0, total_samples)

        # 2. Shrink each class covariance towards pooled covariance and invert
        inv_covs = torch.zeros(self.n_classes, self.d, self.d, dtype=torch.float64)
        eye_d = torch.eye(self.d, dtype=torch.float64)
        ridge = 1e-4

        for c in range(self.n_classes):
            if seen_mask[c]:
                sigma_shrunk = (1.0 - self.shrinkage) * covs[c] + self.shrinkage * pooled_cov + ridge * eye_d
                # Symmetrize for stability
                sigma_shrunk = 0.5 * (sigma_shrunk + sigma_shrunk.T)
                try:
                    inv_covs[c] = torch.linalg.inv(sigma_shrunk)
                except torch.linalg.LinAlgError:
                    inv_covs[c] = torch.linalg.pinv(sigma_shrunk)

        self._means = means
        self._inv_covs = inv_covs

    def predict(self, x: Tensor, ctx: Optional[Tensor] = None) -> Prediction:
        """Predict class probabilities via Mahalanobis distance."""
        seen = self._cnt > 0
        if not bool(seen.any()):
            return self._empty_prediction(x.shape[0])

        if self._inv_covs is None or self._means is None:
            self._update_parameters()

        xd = self._unit(x).double()
        b = xd.shape[0]
        scores = torch.full((b, self.n_classes), float("-inf"), dtype=torch.float32)

        for c in range(self.n_classes):
            if seen[c]:
                diff = xd - self._means[c]  # (B, d)
                # d_M^2 = diff @ inv_cov @ diff^T per row
                v = diff @ self._inv_covs[c]  # (B, d)
                d_sq = (diff * v).sum(dim=-1).float()  # (B,)
                scores[:, c] = -d_sq

        logits = scores * self.scale
        logits = logits.masked_fill(~seen, float("-inf"))
        return self._prediction(torch.softmax(logits, dim=1), 0)


class RandomFeatureRidge(_Base):
    """Arm A6: Random feature ridge regression.

    Fixed random Gaussian projection W_proj (d -> D) regenerated from seed, followed by ReLU.
    Incrementally accumulates G = phi^T phi and B = phi^T Y.
    Weights are solved lazily via Cholesky solve of (G + ridge * I) W = B at predict time.
    Memory footprint: D * D + D * n_classes floats exposed to pool_report via summary().
    """

    name = "M_random_feature_ridge"

    def __init__(
        self,
        cfg: Optional[CommonCfg] = None,
        D: int = 1024,
        ridge: float = 0.1,
        scale: float = 1.0,
    ) -> None:
        super().__init__(cfg)
        self.d = self.cfg.d
        self.n_classes = self.cfg.n_classes
        self.D = int(D)
        self.ridge = float(ridge)
        self.scale = float(scale)

        # Fixed Gaussian projection regenerated from seed (not counted as stored parameters)
        rng = torch.Generator().manual_seed((self.cfg.seed ^ 0x524650) & 0xFFFFFFFF)
        self.W_proj = torch.randn(self.d, self.D, generator=rng, dtype=torch.float32) * (1.0 / math.sqrt(self.d))

        self._G = torch.zeros(self.D, self.D, dtype=torch.float64)
        self._B = torch.zeros(self.D, self.n_classes, dtype=torch.float64)
        self._cnt = torch.zeros(self.n_classes, dtype=torch.long)
        self._W: Optional[Tensor] = None

    @staticmethod
    def _unit(x: Tensor) -> Tensor:
        x = x.float()
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _empty_prediction(self, b: int) -> Prediction:
        return self._prediction(
            torch.full((b, self.n_classes), 1.0 / self.n_classes),
            -1,
            p_nov=torch.zeros(b),
            unknown=torch.ones(b, dtype=torch.bool),
        )

    def summary(self) -> dict:
        """Memory accounting for pool_report: D*D + D*classes float parameters."""
        return {
            "n_experts": 1,
            "n_params": self.D * self.D + self.D * self.n_classes,
        }

    def observe(self, x: Tensor, y: Optional[Tensor] = None, ctx: Optional[Tensor] = None) -> StepRecord:
        """Incrementally accumulate G and B."""
        self.step += 1
        if y is not None:
            x_unit = self._unit(x)
            phi = F.relu(x_unit @ self.W_proj).double()  # (B, D)
            self._G += phi.T @ phi
            y_long = y.long()
            Y = F.one_hot(y_long, self.n_classes).double()  # (B, C)
            self._B += phi.T @ Y
            self._cnt.index_add_(0, y_long, torch.ones(len(y), dtype=torch.long))
            self._W = None
        return self._record(x.shape[0], -1 if y is None else 0, n_experts=1)

    def _solve_weights(self) -> Tensor:
        """Solve (G + ridge * I) W = B lazily via Cholesky decomposition."""
        if self._W is None:
            A = self._G + self.ridge * torch.eye(self.D, dtype=torch.float64)
            A = 0.5 * (A + A.T)
            try:
                L = torch.linalg.cholesky(A)
                self._W = torch.cholesky_solve(self._B, L)
            except torch.linalg.LinAlgError:
                self._W = torch.linalg.solve(A, self._B)
        return self._W

    def predict(self, x: Tensor, ctx: Optional[Tensor] = None) -> Prediction:
        """Predict class probabilities via random feature linear projection."""
        seen = self._cnt > 0
        if not bool(seen.any()):
            return self._empty_prediction(x.shape[0])

        W = self._solve_weights()  # (D, C)
        x_unit = self._unit(x)
        phi = F.relu(x_unit @ self.W_proj).double()  # (B, D)
        logits = (phi @ W).float() * self.scale
        logits = logits.masked_fill(~seen, float("-inf"))
        return self._prediction(torch.softmax(logits, dim=1), 0)
