"""v1/downdate.py: Exact downdating of closed-form heads and ExactLda implementation.

Part of Job V-3 / Probe P2 (is removal exact?).
Implements exact downdating of:
  - ClassMean
  - KnnMemory
  - MahalanobisProto (Arm A5b)
  - RandomFeatureRidge (Arm A6, with data=(x, y) for non-separable Gram matrix G)
  - ExactLda (New class with per-class sufficient statistics)
Also assesses StreamingLDA (Arm A5a) recursive covariance running statistics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from growmoe.memory_baselines import (
    ClassMean,
    KnnMemory,
    Prediction,
    StepRecord,
    StreamingLDA,
    _MemoryPool,
)
from growmoe.pools.base import CommonCfg
from m1.baselines_extra import MahalanobisProto, RandomFeatureRidge

TOL_EXACT = 1e-6


class ExactLda(_MemoryPool):
    """Batch Linear Discriminant Analysis with incremental per-class sufficient statistics.

    Unlike StreamingLDA (which updates a single running shared covariance recursively
    with step-dependent order effects), ExactLda maintains:
      - per-class count n_c
      - per-class sum s_c = sum_{i in c} x_i
      - per-class uncentered outer products outer_c = sum_{i in c} x_i x_i^T

    The pooled within-class covariance is computed from these separable statistics:
      scatter_c = outer_c - n_c * (mu_c mu_c^T)
      Sigma_W = (1 / N) * sum_{c in seen} scatter_c
    Shrinkage towards identity:
      Sigma_shrunk = (1 - shrinkage) * Sigma_W + shrinkage * I

    Because the sufficient statistics are strictly partitioned by class, removing
    any class c (setting n_c = 0, s_c = 0, outer_c = 0) is mathematically EXACT
    to training without that class.
    """

    name = "M_exact_lda"

    def __init__(
        self,
        cfg: Optional[CommonCfg] = None,
        shrinkage: float = 1e-2,
        scale: float = 1.0,
    ) -> None:
        super().__init__(cfg)
        self.shrinkage = float(shrinkage)
        self.scale = float(scale)
        self._cnt = torch.zeros(self.n_classes, dtype=torch.float64)
        self._sum = torch.zeros(self.n_classes, self.d, dtype=torch.float64)
        self._outer = torch.zeros(self.n_classes, self.d, self.d, dtype=torch.float64)
        self._W: Optional[Tuple[Tensor, Tensor]] = None

    def stored_vectors(self) -> int:
        """Parameters in d-vector units: C mean vectors + C*d outer product vectors."""
        seen = int((self._cnt > 0).sum().item())
        return seen * (1 + self.d)

    def lookup_vectors(self) -> int:
        return max(1, int((self._cnt > 0).sum().item()))

    def observe(self, x: Tensor, y: Optional[Tensor] = None, ctx: Optional[Tensor] = None) -> StepRecord:
        """Incrementally accumulate per-class sufficient statistics."""
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
            self._W = None
        return self._record(x.shape[0], -1 if y is None else 0, n_experts=1)

    def _weights(self) -> Optional[Tuple[Tensor, Tensor]]:
        """Compute LDA weights (w, b) from current sufficient statistics."""
        if self._W is None:
            seen = self._cnt > 0
            if not bool(seen.any()):
                return None
            total_n = float(self._cnt.sum().item())
            if total_n <= 0:
                return None

            mu = torch.zeros(self.n_classes, self.d, dtype=torch.float64)
            pooled_scatter = torch.zeros(self.d, self.d, dtype=torch.float64)

            for c in range(self.n_classes):
                n_c = float(self._cnt[c].item())
                if n_c > 0:
                    mu_c = self._sum[c] / n_c
                    mu[c] = mu_c
                    scatter_c = self._outer[c] - n_c * torch.outer(mu_c, mu_c)
                    pooled_scatter += scatter_c

            cov = pooled_scatter / total_n
            cov = 0.5 * (cov + cov.T)
            sig = (1.0 - self.shrinkage) * cov + self.shrinkage * torch.eye(self.d, dtype=torch.float64)

            try:
                w = torch.linalg.solve(sig, mu.T).T  # (C, d)
            except torch.linalg.LinAlgError:
                w = torch.linalg.pinv(sig) @ mu.T
                w = w.T

            b = -0.5 * (mu * w).sum(dim=1)
            self._W = (w, b)
        return self._W

    def predict(self, x: Tensor, ctx: Optional[Tensor] = None) -> Prediction:
        """Predict class probabilities via solved linear discriminant functions."""
        seen = self._cnt > 0
        if not bool(seen.any()):
            return self._empty_prediction(x.shape[0])
        wb = self._weights()
        if wb is None:
            return self._empty_prediction(x.shape[0])
        w, b = wb
        logits = (self._unit(x).double() @ w.T + b).float() * self.scale
        logits = logits.masked_fill(~seen, float("-inf"))
        return self._prediction(torch.softmax(logits, dim=1), 0)


def remove_classes(
    model: Any,
    labels: Union[int, Sequence[int], Iterable[int], Tensor],
    data: Optional[Tuple[Tensor, Tensor]] = None,
) -> Dict[str, Any]:
    """Remove one or more classes from a trained closed-form head.

    Args:
        model: Trained closed-form head (ClassMean, KnnMemory, MahalanobisProto,
               RandomFeatureRidge, ExactLda, or StreamingLDA).
        labels: Class label(s) to remove.
        data: Optional tuple of (x, y) tensors of the removed samples, required
              for exact downdating of models with non-class-separable statistics
              (e.g., RandomFeatureRidge Gram matrix G).

    Returns:
        Metadata dict detailing what was removed and classification (exact / approximate / impossible).

    Raises:
        ValueError: If any requested class is not present in the model.
        TypeError: If the model type is unsupported.
    """
    if isinstance(labels, (int, np.integer)):
        labels_set = {int(labels)}
    elif isinstance(labels, torch.Tensor):
        labels_set = {int(lbl) for lbl in labels.view(-1).tolist()}
    else:
        labels_set = {int(lbl) for lbl in labels}

    if not labels_set:
        return {"removed_labels": [], "status": "no_op", "classification": "exact"}

    # 1. ClassMean
    if isinstance(model, ClassMean):
        # Validate presence
        for c in labels_set:
            if c < 0 or c >= model.n_classes or model._cnt[c].item() == 0:
                raise ValueError(f"Class {c} is not present in ClassMean model")

        for c in labels_set:
            model._sum[c].zero_()
            model._cnt[c] = 0.0

        return {
            "model": model.name,
            "removed_labels": sorted(labels_set),
            "classification": "exact",
            "downdated_fields": ["_sum", "_cnt"],
            "notes": "Per-class sums and counts zeroed. Retained classes completely untouched.",
        }

    # 2. KnnMemory
    elif isinstance(model, KnnMemory):
        # Validate presence
        held_y = model._Y[: model._n]
        for c in labels_set:
            if not bool((held_y == c).any()):
                raise ValueError(f"Class {c} is not present in KnnMemory model")

        keep_mask = ~torch.isin(held_y, torch.tensor(list(labels_set), dtype=torch.long))
        n_kept = int(keep_mask.sum().item())
        n_removed = model._n - n_kept

        new_X = model._X[: model._n][keep_mask]
        new_Y = model._Y[: model._n][keep_mask]
        model._X[:n_kept] = new_X
        model._Y[:n_kept] = new_Y
        model._X[n_kept : model._n].zero_()
        model._Y[n_kept : model._n].zero_()
        model._n = n_kept
        # Note: self._seen is stream count. If capacity is None, reservoir was not stochastic.
        is_exact = model.capacity is None

        return {
            "model": model.name,
            "removed_labels": sorted(labels_set),
            "removed_samples": n_removed,
            "classification": "exact" if is_exact else "approximate",
            "downdated_fields": ["_X", "_Y", "_n"],
            "notes": "Exact when capacity=None (stream preservation). Approximate if bounded capacity reservoir evicted prior items.",
        }

    # 3. MahalanobisProto (Arm A5b)
    elif isinstance(model, MahalanobisProto):
        for c in labels_set:
            if c < 0 or c >= model.n_classes or model._cnt[c].item() == 0:
                raise ValueError(f"Class {c} is not present in MahalanobisProto model")

        for c in labels_set:
            model._cnt[c] = 0.0
            model._sum[c].zero_()
            model._outer[c].zero_()

        model._inv_covs = None
        model._means = None

        return {
            "model": model.name,
            "removed_labels": sorted(labels_set),
            "classification": "exact",
            "downdated_fields": ["_cnt", "_sum", "_outer", "_inv_covs", "_means"],
            "notes": "Sufficient statistics are strictly class-separable; recomputed pooled covariance and class covariances are exact.",
        }

    # 4. RandomFeatureRidge (Arm A6)
    elif isinstance(model, RandomFeatureRidge):
        for c in labels_set:
            if c < 0 or c >= model.n_classes or model._cnt[c].item() == 0:
                raise ValueError(f"Class {c} is not present in RandomFeatureRidge model")

        if data is None:
            # Without data, G cannot be downdated exactly because G = sum phi^T phi is pooled
            for c in labels_set:
                model._B[:, c] = 0.0
                model._cnt[c] = 0
            model._W = None
            return {
                "model": model.name,
                "removed_labels": sorted(labels_set),
                "classification": "approximate",
                "downdated_fields": ["_B", "_cnt"],
                "notes": "Without removed data samples, Gram matrix G could not be downdated; solved weights approximate.",
            }

        x_rem, y_rem = data
        x_rem_t = torch.as_tensor(x_rem, dtype=torch.float32)
        y_rem_t = torch.as_tensor(y_rem, dtype=torch.long)

        mask = torch.isin(y_rem_t, torch.tensor(list(labels_set), dtype=torch.long))
        if bool(mask.any()):
            x_target = x_rem_t[mask]
            x_unit = model._unit(x_target)
            phi = F.relu(x_unit @ model.W_proj).double()  # (M, D)
            model._G -= phi.T @ phi
            # Symmetrize for numerical stability
            model._G = 0.5 * (model._G + model._G.T)

        for c in labels_set:
            model._B[:, c] = 0.0
            model._cnt[c] = 0
        model._W = None

        return {
            "model": model.name,
            "removed_labels": sorted(labels_set),
            "classification": "exact",
            "downdated_fields": ["_G", "_B", "_cnt", "_W"],
            "notes": "Gram matrix G downdated using removed sample features (phi^T phi); B zeroed on removed classes. Exact up to float64 roundoff.",
        }

    # 5. ExactLda
    elif isinstance(model, ExactLda):
        for c in labels_set:
            if c < 0 or c >= model.n_classes or model._cnt[c].item() == 0:
                raise ValueError(f"Class {c} is not present in ExactLda model")

        for c in labels_set:
            model._cnt[c] = 0.0
            model._sum[c].zero_()
            model._outer[c].zero_()

        model._W = None

        return {
            "model": model.name,
            "removed_labels": sorted(labels_set),
            "classification": "exact",
            "downdated_fields": ["_cnt", "_sum", "_outer", "_W"],
            "notes": "Per-class count, sum, and outer product statistics zeroed. Recomputed pooled covariance and discriminant weights are exact.",
        }

    # 6. StreamingLDA (Arm A5a)
    elif isinstance(model, StreamingLDA):
        for c in labels_set:
            if c < 0 or c >= model.n_classes or model._cnt[c].item() == 0:
                raise ValueError(f"Class {c} is not present in StreamingLDA model")

        c_removed_cnt = sum(float(model._cnt[c].item()) for c in labels_set)
        for c in labels_set:
            model._cnt[c] = 0.0
            model._mu[c].zero_()

        model._n = max(0, model._n - int(c_removed_cnt))
        model._W = None

        return {
            "model": model.name,
            "removed_labels": sorted(labels_set),
            "classification": "approximate",
            "downdated_fields": ["_cnt", "_mu", "_n", "_W"],
            "notes": "Running covariance _cov has order-dependent non-linear updates mixing all classes; cannot be downdated without stream replay.",
        }

    else:
        raise TypeError(f"Unsupported model type: {type(model).__name__}")
