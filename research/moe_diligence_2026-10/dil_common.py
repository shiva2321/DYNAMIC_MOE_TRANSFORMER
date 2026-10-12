"""Throwaway diligence helpers (Claude, 2026-10-09). Lives in the scratchpad; imports growmoe read-only."""
import os
import sys

sys.path.insert(0, r"D:\spherical_hyperspace_memory")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

torch.set_num_threads(4)

from growmoe.telemetry import EventLog
from growmoe.types import Prediction, StepRecord


class _PoolBase:
    name = "base"

    def __init__(self, n_classes: int, d: int = 128):
        self.n_classes, self.d = n_classes, d
        self.events = EventLog()
        self.step = 0

    def _rec(self, b: int) -> StepRecord:
        return StepRecord(novel=torch.zeros(b, dtype=torch.bool), p_nov=torch.ones(b),
                          assigned=torch.zeros(b, dtype=torch.long), loss=float("nan"), spawned=[], n_experts=1)

    def _pred(self, probs: torch.Tensor) -> Prediction:
        b = probs.shape[0]
        return Prediction(probs=probs, expert=torch.zeros(b, dtype=torch.long), p_nov=torch.ones(b),
                          unknown=torch.zeros(b, dtype=torch.bool))

    def summary(self) -> dict:
        return {"n_experts": 1, "n_params": 0, "active_params_per_sample": 0, "lineage": {}, "states": {}}


class KnnPool(_PoolBase):
    """Memory-based k-NN on the unit-sphere features. capacity=None keeps every training sample (unbounded memory)."""

    def __init__(self, n_classes, d=128, k=5, capacity=None, seed=0):
        super().__init__(n_classes, d)
        self.k, self.capacity = k, capacity
        self.name = f"knn_k{k}_" + ("all" if capacity is None else f"cap{capacity}")
        self.X = torch.zeros(0, d)
        self.Y = torch.zeros(0, dtype=torch.long)
        self.rng = np.random.default_rng([seed, 0x4B4E4E])
        self.n_seen = 0

    def observe(self, x, y=None, ctx=None):
        self.step += 1
        if y is not None:
            for i in range(x.shape[0]):
                self.n_seen += 1
                if self.capacity is None or self.X.shape[0] < self.capacity:
                    self.X = torch.cat([self.X, x[i:i + 1]])
                    self.Y = torch.cat([self.Y, y[i:i + 1].long()])
                else:
                    j = int(self.rng.integers(0, self.n_seen))
                    if j < self.capacity:
                        self.X[j] = x[i]
                        self.Y[j] = y[i].long()
        return self._rec(x.shape[0])

    def predict(self, x, ctx=None):
        b = x.shape[0]
        if self.X.shape[0] == 0:
            return self._pred(torch.full((b, self.n_classes), 1.0 / self.n_classes))
        sims = x @ self.X.T
        kk = min(self.k, self.X.shape[0])
        top = sims.topk(kk, dim=1)
        votes = torch.zeros(b, self.n_classes)
        votes.scatter_add_(1, self.Y[top.indices], torch.ones(b, kk))
        return self._pred(votes / votes.sum(dim=1, keepdim=True).clamp_min(1e-9))

    def stored_vectors(self):
        return int(self.X.shape[0])


class NcmPool(_PoolBase):
    """Nearest class mean (one running mean per label)."""

    name = "ncm"

    def __init__(self, n_classes, d=128, seed=0):
        super().__init__(n_classes, d)
        self.sum = torch.zeros(n_classes, d)
        self.cnt = torch.zeros(n_classes)

    def observe(self, x, y=None, ctx=None):
        self.step += 1
        if y is not None:
            self.sum.index_add_(0, y.long(), x)
            self.cnt.index_add_(0, y.long(), torch.ones(x.shape[0]))
        return self._rec(x.shape[0])

    def predict(self, x, ctx=None):
        seen = self.cnt > 0
        mu = self.sum / self.cnt.clamp_min(1.0).unsqueeze(1)
        mu = mu / mu.norm(dim=1, keepdim=True).clamp_min(1e-9)
        logits = 20.0 * (x @ mu.T)
        logits = logits.masked_fill(~seen, -1e9)
        return self._pred(torch.softmax(logits, dim=1))

    def stored_vectors(self):
        return self.n_classes


class SldaPool(_PoolBase):
    """Streaming LDA (Hayes and Kanan 2020 style): per-class means plus one shared running covariance, shrinkage eps."""

    name = "slda"

    def __init__(self, n_classes, d=128, eps=1e-2, seed=0):
        super().__init__(n_classes, d)
        self.eps = eps
        self.mu = torch.zeros(n_classes, d, dtype=torch.float64)
        self.cnt = torch.zeros(n_classes, dtype=torch.float64)
        self.cov = torch.zeros(d, d, dtype=torch.float64)
        self.n = 0

    def observe(self, x, y=None, ctx=None):
        self.step += 1
        if y is not None:
            xd = x.double()
            for i in range(xd.shape[0]):
                c = int(y[i])
                diff = xd[i] - self.mu[c]
                delta = (self.n / (self.n + 1.0)) * torch.outer(diff, diff)
                self.cov = (self.n * self.cov + delta) / (self.n + 1.0)
                self.mu[c] += diff / (self.cnt[c] + 1.0)
                self.cnt[c] += 1.0
                self.n += 1
        return self._rec(x.shape[0])

    def predict(self, x, ctx=None):
        seen = self.cnt > 0
        sig = (1.0 - self.eps) * self.cov + self.eps * torch.eye(self.d, dtype=torch.float64)
        w = torch.linalg.solve(sig, self.mu.T).T                      # (C, d)
        b = -0.5 * (self.mu * w).sum(dim=1)
        logits = (x.double() @ w.T + b).float().masked_fill(~seen, -1e9)
        return self._pred(torch.softmax(logits, dim=1))

    def stored_vectors(self):
        return self.n_classes + self.d


def mean_ci(vals):
    from growmoe.metrics import mean_ci as _m
    m, lo, hi = _m(vals)
    return m, lo, hi


def fmt(vals, scale=100.0, nd=2):
    m, lo, hi = mean_ci(vals)
    return f"{m*scale:.{nd}f} [{lo*scale:.{nd}f}, {hi*scale:.{nd}f}]"
