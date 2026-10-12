"""Memory baselines for continual-learning tables (M0): kNN lookup, nearest class mean, streaming LDA.

They learn nothing (or almost nothing), so they cannot forget; any method that claims to be a continual learner has to
beat them at the same memory. All three are ``ContinualPool`` s and read only ``(x, y, ctx)``.

``KnnMemory(capacity=None)`` stores every labelled example; with ``capacity`` it keeps a uniform reservoir (Vitter's Algorithm R,
seeded from ``cfg.seed``). Queries are processed in chunks so the (queries x stored) similarity matrix never gets large.
"""

import numpy as np
import torch
from torch import Tensor

from .baselines import _Base
from .config import CommonCfg
from .types import Prediction, StepRecord

_CHUNK = 512


class _MemoryPool(_Base):
    def __init__(self, cfg: CommonCfg = None) -> None:
        super().__init__(cfg)
        self.d, self.n_classes = self.cfg.d, self.cfg.n_classes

    @staticmethod
    def _unit(x: Tensor) -> Tensor:
        x = x.float()
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _empty_prediction(self, b: int) -> Prediction:
        return self._prediction(torch.full((b, self.n_classes), 1.0 / self.n_classes), -1,
                                p_nov=torch.zeros(b), unknown=torch.ones(b, dtype=torch.bool))

    def summary(self) -> dict:
        return self._summary([], 0, n_experts=1, stored_vectors=self.stored_vectors())

    def stored_vectors(self) -> int:                                                  # pragma: no cover - overridden
        raise NotImplementedError

    def lookup_vectors(self) -> int:
        """How many stored vectors one query is compared with (drives the multiply-add count of a prediction)."""
        return self.n_classes


class KnnMemory(_MemoryPool):
    def __init__(self, cfg: CommonCfg = None, k: int = 5, capacity: int = None) -> None:
        super().__init__(cfg)
        self.k, self.capacity = k, capacity
        self.name = "M_knn_all" if capacity is None else f"M_knn_cap{capacity}"
        self._X = torch.zeros(capacity if capacity is not None else 1024, self.d)
        self._Y = torch.zeros(self._X.shape[0], dtype=torch.long)
        self._n = 0
        self._seen = 0
        self._rng = np.random.default_rng([self.cfg.seed, 0x4B4E4E])

    @property
    def X(self) -> Tensor:
        return self._X[: self._n]

    @property
    def Y(self) -> Tensor:
        return self._Y[: self._n]

    def stored_vectors(self) -> int:
        return self._n

    def lookup_vectors(self) -> int:
        return self._n                                                                 # a query is compared with every stored example

    def _grow(self, need: int) -> None:
        cap = max(need, 2 * self._X.shape[0])
        self._X = torch.cat([self._X[: self._n], torch.zeros(cap - self._n, self.d)])
        self._Y = torch.cat([self._Y[: self._n], torch.zeros(cap - self._n, dtype=torch.long)])

    def observe(self, x: Tensor, y: Tensor = None, ctx: Tensor = None) -> StepRecord:
        self.step += 1
        b = x.shape[0]
        if y is not None:
            x, y = self._unit(x), y.long()
            if self.capacity is None:
                if self._n + b > self._X.shape[0]:
                    self._grow(self._n + b)
                self._X[self._n:self._n + b], self._Y[self._n:self._n + b] = x, y
                self._n += b
                self._seen += b
            else:
                for i in range(b):
                    self._seen += 1
                    if self._n < self.capacity:
                        slot = self._n
                        self._n += 1
                    else:
                        slot = int(self._rng.integers(0, self._seen))
                        if slot >= self.capacity:
                            continue
                    self._X[slot], self._Y[slot] = x[i], y[i]
        return self._record(b, -1 if y is None else 0, n_experts=1)

    def predict(self, x: Tensor, ctx: Tensor = None) -> Prediction:
        b = x.shape[0]
        if self._n == 0:
            return self._empty_prediction(b)
        x = self._unit(x)
        X, Y = self.X, self.Y
        kk = min(self.k, self._n)
        probs = torch.zeros(b, self.n_classes)
        for lo in range(0, b, _CHUNK):
            idx = (x[lo:lo + _CHUNK] @ X.T).topk(kk, dim=1).indices
            votes = torch.zeros(idx.shape[0], self.n_classes)
            votes.scatter_add_(1, Y[idx], torch.ones(idx.shape, dtype=torch.float32))
            probs[lo:lo + _CHUNK] = votes / kk
        return self._prediction(probs, 0)


class ClassMean(_MemoryPool):
    name = "M_class_mean"

    def __init__(self, cfg: CommonCfg = None, scale: float = 20.0) -> None:
        super().__init__(cfg)
        self.scale = scale
        self._sum = torch.zeros(self.n_classes, self.d)
        self._cnt = torch.zeros(self.n_classes)

    def stored_vectors(self) -> int:
        return self.n_classes

    def observe(self, x: Tensor, y: Tensor = None, ctx: Tensor = None) -> StepRecord:
        self.step += 1
        if y is not None:
            self._sum.index_add_(0, y.long(), self._unit(x))
            self._cnt.index_add_(0, y.long(), torch.ones(x.shape[0]))
        return self._record(x.shape[0], -1 if y is None else 0, n_experts=1)

    def predict(self, x: Tensor, ctx: Tensor = None) -> Prediction:
        seen = self._cnt > 0
        if not bool(seen.any()):
            return self._empty_prediction(x.shape[0])
        mu = self._sum / self._cnt.clamp_min(1.0).unsqueeze(1)
        mu = mu / mu.norm(dim=1, keepdim=True).clamp_min(1e-12)
        logits = (self.scale * (self._unit(x) @ mu.T)).masked_fill(~seen, float("-inf"))
        return self._prediction(torch.softmax(logits, dim=1), 0)


class StreamingLDA(_MemoryPool):
    """Streaming linear discriminant analysis (Hayes and Kanan, 2020): running class means plus ONE shared running covariance,
    classified with a shrunk inverse. Memory is C x d + d x d, independent of the stream length."""

    name = "M_streaming_lda"

    def __init__(self, cfg: CommonCfg = None, shrinkage: float = 1e-2) -> None:
        super().__init__(cfg)
        self.shrinkage = shrinkage
        self._mu = torch.zeros(self.n_classes, self.d, dtype=torch.float64)
        self._cnt = torch.zeros(self.n_classes, dtype=torch.float64)
        self._cov = torch.zeros(self.d, self.d, dtype=torch.float64)
        self._n = 0
        self._W = None

    def stored_vectors(self) -> int:
        return self.n_classes + self.d                                                 # means + the covariance, in vector units

    def observe(self, x: Tensor, y: Tensor = None, ctx: Tensor = None) -> StepRecord:
        self.step += 1
        if y is not None:
            xd, yd = self._unit(x).double(), y.long()
            for i in range(xd.shape[0]):
                c = int(yd[i])
                diff = xd[i] - self._mu[c]
                self._cov = (self._n * self._cov + (self._n / (self._n + 1.0)) * torch.outer(diff, diff)) / (self._n + 1.0)
                self._mu[c] += diff / (self._cnt[c] + 1.0)
                self._cnt[c] += 1.0
                self._n += 1
            self._W = None
        return self._record(x.shape[0], -1 if y is None else 0, n_experts=1)

    def _weights(self):
        if self._W is None:
            sig = (1.0 - self.shrinkage) * self._cov + self.shrinkage * torch.eye(self.d, dtype=torch.float64)
            w = torch.linalg.solve(sig, self._mu.T).T                                  # (C, d)
            self._W = (w, -0.5 * (self._mu * w).sum(dim=1))
        return self._W

    def predict(self, x: Tensor, ctx: Tensor = None) -> Prediction:
        seen = self._cnt > 0
        if not bool(seen.any()):
            return self._empty_prediction(x.shape[0])
        w, b = self._weights()
        logits = (self._unit(x).double() @ w.T + b).float().masked_fill(~seen, float("-inf"))
        return self._prediction(torch.softmax(logits, dim=1), 0)
