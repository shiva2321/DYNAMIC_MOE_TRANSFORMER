"""Novelty detectors on unit vectors and the shared calibration rule.

A detector is fitted on teaching rows (``fit(X, y)``) and gives every input a novelty ``score`` (higher means less like what was taught). Each reports ``macs``, the multiply-adds one scored input costs
(an honest count of the matrix products, not a timing). ``conformal_cut`` is the one calibration rule used by every arm: the cut is the ceil((n+1)(1-q))-th smallest score of n known inputs, so a
known input is flagged with probability (n-k+1)/(n+1), which ``effective_rate`` returns (q is an aim, the effective rate is what is true).
"""

import math
from typing import Optional

import numpy as np
import torch
from scipy.special import ive
from scipy.stats import rankdata
from torch import Tensor

from unimoe.density import DensityHead


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties counting half. nan when either side is empty."""
    neg, pos = np.asarray(neg, dtype=np.float64), np.asarray(pos, dtype=np.float64)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def cut_rank(n: int, q: float) -> int:
    return int(math.ceil((n + 1) * (1.0 - q)))


def conformal_cut(cal: np.ndarray, q: float) -> float:
    """Flag a score strictly above the returned cut. +inf when n is too small for any cut at q (nothing can then be flagged)."""
    n = len(cal)
    k = cut_rank(n, q)
    return float("inf") if k > n else float(np.sort(cal)[k - 1])


def effective_rate(n: int, q: float) -> float:
    k = cut_rank(n, q)
    return 0.0 if k > n else (n - k + 1) / (n + 1)


class Detector:
    name = ""
    macs = 0

    def fit(self, X: Tensor, y: Tensor) -> "Detector":
        raise NotImplementedError

    def score(self, X: Tensor) -> np.ndarray:
        raise NotImplementedError


class Knn(Detector):
    """1 - the mean cosine of the k nearest teaching vectors."""
    name = "knn"

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.X: Optional[Tensor] = None

    def fit(self, X: Tensor, y: Tensor) -> "Knn":
        self.X = X.float()
        self.macs = int(X.shape[0] * X.shape[1])
        return self

    def score(self, X: Tensor) -> np.ndarray:
        out = []
        for lo in range(0, X.shape[0], 1024):
            sims = X[lo:lo + 1024].float() @ self.X.T
            out.append(1.0 - sims.topk(min(self.k, sims.shape[1]), dim=1).values.mean(1))
        return torch.cat(out).numpy()


class Maha(Detector):
    """Squared Mahalanobis distance to the nearest taught class mean (the LDA head's own statistics)."""
    name = "maha"

    def __init__(self, shrink: float = 0.1) -> None:
        self.shrink = shrink
        self.head: Optional[DensityHead] = None

    def fit(self, X: Tensor, y: Tensor) -> "Maha":
        self.head = DensityHead(X.shape[1], int(y.max()) + 1, self.shrink)
        self.head.update(X, y)
        d, c = X.shape[1], int(self.head.seen().sum())
        self.macs = d * d + c * d
        return self

    def score(self, X: Tensor) -> np.ndarray:
        return self.head.distance(X).numpy()


class Vmf(Detector):
    """Negative log density under the best of the taught classes, each a von Mises-Fisher distribution on the sphere (mean direction, concentration from the mean resultant length)."""
    name = "vmf"

    def fit(self, X: Tensor, y: Tensor) -> "Vmf":
        d = X.shape[1]
        mus, kap = [], []
        for c in sorted(set(y.tolist())):
            m = X[y == c].float().mean(0).double()
            R = float(m.norm())
            kap.append(float(np.clip(R * (d - R * R) / max(1.0 - R * R, 1e-9), 1.0, 5e3)))
            mus.append((m / m.norm().clamp_min(1e-12)).float())
        self.mu = torch.stack(mus)
        k = np.asarray(kap)
        v = d / 2.0 - 1.0
        self.kappa = torch.from_numpy(k).float()
        self.logC = torch.from_numpy(v * np.log(k) - (d / 2.0) * math.log(2 * math.pi) - (np.log(ive(v, k)) + k)).float()
        self.macs = int(len(kap) * d)
        return self

    def score(self, X: Tensor) -> np.ndarray:
        ll = self.logC[None] + self.kappa[None] * (X.float() @ self.mu.T)
        return (-ll.max(1).values).numpy()
