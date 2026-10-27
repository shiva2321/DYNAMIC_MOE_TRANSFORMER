"""The standard novelty detectors from the literature, every one with the same two calls so a claim about the status machine always has its control (repository rule D-06):

    detector.fit(X, y)             X: (n, d) vectors of the TAUGHT examples, y: one class name per row (the skill or "other")
    detector.novelty_score(X)      (n,) floats, HIGHER = MORE NOVEL

The seven (spec 6.2): ``vmf_conformal`` (the expert pool's per-sample gate: one von Mises-Fisher signature per class, novelty = 1 - the best conformal p-value of the signature's own leave-one-out
calibration), ``knn_distance`` (distance to the k-th nearest taught vector; Sun et al. 2022), ``mahalanobis`` (Lee et al. 2018: class means, one shared covariance, distance to the nearest class),
``relative_mahalanobis`` (Ren et al. 2021: that distance minus the distance under a background Gaussian fitted to general text, here the Other-content corpora), ``classifier_confidence``
(one minus the largest softmax probability; Hendrycks and Gimpel 2017), ``energy`` (minus the log-sum-exp of the logits; Liu et al. 2020) and ``autoencoder_descriptor`` (one small autoencoder per class,
novelty = the lowest z-score of its reconstruction error; the descriptor of SEMA, Wang et al.). ``group_test`` is the permutation test of ``growmoe.settest`` on a batch of 1, 4, 8 or 16 parts.

All of them judge the vectors they are given. For a decoder reader (R4) the raw vectors share one big direction (pairwise cosines 0.75 to 0.99), so the caller passes them through
``centred_unit(taught_vectors)`` first, for the baselines and for the status machine alike; the Mahalanobis family and the classifiers would not mind, the cosine-based ones do.

Calibration is shared: ``conformal_threshold(novelty of held-out taught data, q)`` gives the score above which a part is flagged novel, with at most q of exchangeable taught data flagged.
"""

import math
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from growmoe import settest
from growmoe.signature import VmfSignature

BASELINE_NAMES = ("vmf_conformal", "knn_distance", "mahalanobis", "relative_mahalanobis", "classifier_confidence", "energy", "autoencoder_descriptor")
_CHUNK = 2048
_TIE_BREAK = 1e-7


# ---------------------------------------------------------------------------------------------------------------------------------- helpers
def _matrix(X, what: str = "X") -> np.ndarray:
    a = np.asarray(X, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"{what} must be a non-empty 2-D array, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{what} must be finite")
    return a


def _unit(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    if (n < 1e-12).any():
        raise ValueError("a zero vector has no direction")
    return a / n


def centred_unit(fit_vectors) -> Callable[[np.ndarray], np.ndarray]:
    """A function that subtracts the mean of ``fit_vectors`` (the taught examples; fixed now, not re-estimated on what is passed later) and rescales every row to unit length."""
    mu = _matrix(fit_vectors, "fit_vectors").mean(axis=0)

    def apply(Z) -> np.ndarray:
        a = _matrix(Z, "vectors") - mu
        if (np.linalg.norm(a, axis=1) < 1e-12).any():
            raise ValueError("a vector equal to the centre has no direction after centring")
        return _unit(a)

    return apply


def conformal_threshold(cal_scores, q: float) -> float:
    """The novelty score above which a part is flagged, from the scores of HELD-OUT taught data: the ceil((n+1)(1-q))-th smallest, so that at most q of exchangeable taught data lie above
    it (finite-sample). Too few scores to ever flag at q give +infinity (nothing is flagged), the same rule as the 19-score minimum of the status machine at q = 0.05."""
    s = np.sort(np.asarray(cal_scores, dtype=np.float64).ravel())
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must lie strictly between 0 and 1, got {q}")
    if s.size == 0:
        raise ValueError("conformal_threshold needs at least one calibration score")
    k = math.ceil((s.size + 1) * (1.0 - q) - 1e-9)
    return math.inf if k > s.size else float(s[k - 1])


def group_test(ref, batch, size: int, n_perm: int = 199, seed: int = 0) -> float:
    """The permutation p-value (``growmoe.settest.perm_pvalue``, mean-distance statistic) of the first ``size`` rows of ``batch`` against the reference vectors ``ref``; small = the batch is
    unlike the reference. ``size`` is 1, 4, 8 or 16 in the study. Valid at every level, with the smallest attainable p-value 1/(1 + n_perm)."""
    ref, batch = _matrix(ref, "ref"), _matrix(batch, "batch")
    if size < 1 or size > len(batch):
        raise ValueError(f"size must be between 1 and the {len(batch)} rows of the batch, got {size}")
    if ref.shape[1] != batch.shape[1]:
        raise ValueError("ref and batch must have the same dimension")
    gen = settest.generator(int(seed), 0x6774, int(size))
    return float(settest.perm_pvalue(torch.from_numpy(ref).float(), torch.from_numpy(batch[:size]).float(), n_perm=n_perm, gen=gen))


class _Detector:
    name = ""

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, X, y) -> "_Detector":
        X = _matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(f"X and y must have the same number of rows, got {len(X)} and {y.shape}")
        self.d = X.shape[1]
        self._fit(X, y)
        self._fitted = True
        return self

    def novelty_score(self, X) -> np.ndarray:
        if not self._fitted:
            raise ValueError(f"{self.name}: call fit before novelty_score")
        X = _matrix(X)
        if X.shape[1] != self.d:
            raise ValueError(f"{self.name}: expected vectors of dimension {self.d}, got {X.shape[1]}")
        return np.asarray(self._score(X), dtype=np.float64)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def _score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _classes(y: np.ndarray, at_least: int = 1):
    cls = sorted(set(y.tolist()))
    if len(cls) < at_least:
        raise ValueError(f"need at least two classes, got {len(cls)}" if at_least == 2 else "need at least one class")
    return cls


# ---------------------------------------------------------------------------------------------------------------------------------- the seven
class VmfConformal(_Detector):
    name = "vmf_conformal"

    def __init__(self, seed: int = 0, max_protos: int = 256) -> None:
        super().__init__()
        self.seed, self.max_protos = seed, max_protos

    def _fit(self, X, y) -> None:
        self._sigs = []
        for k, c in enumerate(_classes(y)):
            sig = VmfSignature(self.d, max_protos=self.max_protos, seed=self.seed + k, dedupe=True)
            sig.update(_unit(X[y == c]).astype(np.float32))
            if sig.n < 2:
                raise ValueError(f"class {c!r} has fewer than two distinct vectors")
            self._sigs.append(sig.fit())

    def _score(self, X) -> np.ndarray:
        """1 - the best conformal p-value over the classes (the pool's gate), with the continuous kernel score as a tie-break. The p-values of a class of n examples come in steps of
        1/(n+1), so a small class (a Thin skill) can never say "novel" more strongly than 1/(n+1): without a tie-break every far point ties with the calibration points that sit at that
        maximum and a shared threshold flags nothing. The tie-break is 1e-7 times the highest kernel log-density over the classes (low for a point far from every class, high for one
        near some class), far below one p-value step, so the p-value ordering stays the primary key."""
        U = _unit(X).astype(np.float32)
        best_p = np.max([s.pvalue(U).double().numpy() for s in self._sigs], axis=0)
        best_density = np.max([s.score(U).double().numpy() for s in self._sigs], axis=0)
        return 1.0 - best_p - _TIE_BREAK * best_density


class KnnDistance(_Detector):
    name = "knn_distance"

    def __init__(self, k: int = 10) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k

    def _fit(self, X, y) -> None:
        self._P = _unit(X).astype(np.float32)

    def _score(self, X) -> np.ndarray:
        U = _unit(X).astype(np.float32)
        kth = min(self.k, len(self._P))
        out = np.empty(len(U))
        for s in range(0, len(U), _CHUNK):
            sims = U[s:s + _CHUNK] @ self._P.T
            out[s:s + _CHUNK] = 1.0 - np.partition(sims, len(self._P) - kth, axis=1)[:, len(self._P) - kth]        # cosine of the k-th nearest
        return out


def _gaussian_precision(residuals: np.ndarray) -> np.ndarray:
    """Inverse of a shrunk (Ledoit-Wolf) covariance, so that a dimension larger than the sample does not break the inverse."""
    return LedoitWolf(assume_centered=True).fit(residuals).precision_


def _mahalanobis_sq(X: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    diff = X - mean
    return np.einsum("nd,de,ne->n", diff, precision, diff, optimize=True)


class Mahalanobis(_Detector):
    name = "mahalanobis"

    def _fit(self, X, y) -> None:
        cls = _classes(y)
        self._means = np.stack([X[y == c].mean(axis=0) for c in cls])
        resid = np.vstack([X[y == c] - m for c, m in zip(cls, self._means)])
        if len(resid) < 2:
            raise ValueError("a covariance needs at least two rows")
        self._precision = _gaussian_precision(resid)

    def _distance(self, X) -> np.ndarray:
        return np.min([_mahalanobis_sq(X, m, self._precision) for m in self._means], axis=0)

    def _score(self, X) -> np.ndarray:
        return self._distance(X)


class RelativeMahalanobis(Mahalanobis):
    name = "relative_mahalanobis"

    def __init__(self, background) -> None:
        super().__init__()
        self._background = _matrix(background, "background")
        if len(self._background) < 2:
            raise ValueError("the background needs at least two rows")

    def _fit(self, X, y) -> None:
        if self._background.shape[1] != X.shape[1]:
            raise ValueError("the background and the taught vectors must have the same dimension")
        super()._fit(X, y)
        self._bg_mean = self._background.mean(axis=0)
        self._bg_precision = _gaussian_precision(self._background - self._bg_mean)

    def _score(self, X) -> np.ndarray:
        return self._distance(X) - _mahalanobis_sq(X, self._bg_mean, self._bg_precision)


class _LogisticLogits(_Detector):
    """A multinomial logistic regression on standardised vectors, the classifier whose logits the two output-space detectors read. ``C`` is the inverse regularisation strength."""

    def __init__(self, C: float = 1.0) -> None:
        super().__init__()
        if not C > 0:
            raise ValueError(f"C must be positive, got {C}")
        self.C = C

    def _fit(self, X, y) -> None:
        _classes(y, at_least=2)
        self._scaler = StandardScaler().fit(X)
        self._clf = LogisticRegression(C=self.C, max_iter=1000).fit(self._scaler.transform(X), y)

    def _logits(self, X) -> np.ndarray:
        z = self._clf.decision_function(self._scaler.transform(X))
        return np.stack([-z / 2.0, z / 2.0], axis=1) if z.ndim == 1 else z               # two classes: the single margin is the difference of two logits


class ClassifierConfidence(_LogisticLogits):
    name = "classifier_confidence"

    def _score(self, X) -> np.ndarray:
        z = self._logits(X)
        z = z - z.max(axis=1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return 1.0 - probs.max(axis=1)


class Energy(_LogisticLogits):
    name = "energy"

    def _score(self, X) -> np.ndarray:
        z = self._logits(X)
        top = z.max(axis=1)
        return -(top + np.log(np.exp(z - top[:, None]).sum(axis=1)))


class AutoencoderDescriptor(_Detector):
    name = "autoencoder_descriptor"

    def __init__(self, hidden: Optional[int] = None, epochs: int = 300, lr: float = 1e-2, seed: int = 0) -> None:
        super().__init__()
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed

    def _fit(self, X, y) -> None:
        h = self.hidden or max(2, min(32, self.d // 2))
        self._parts = []
        for k, c in enumerate(_classes(y)):
            rows = torch.from_numpy(X[y == c]).float()
            if len(rows) < 2:
                raise ValueError(f"class {c!r} has fewer than two vectors")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed + k)
                net = torch.nn.Sequential(torch.nn.Linear(self.d, h), torch.nn.ReLU(), torch.nn.Linear(h, self.d))
                opt = torch.optim.Adam(net.parameters(), lr=self.lr)
                for _ in range(self.epochs):
                    loss = ((net(rows) - rows) ** 2).sum(dim=1).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            net.eval()
            err = self._error(net, rows)
            self._parts.append((net, float(err.mean()), max(float(err.std()), 1e-8)))

    @staticmethod
    def _error(net, rows: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return ((net(rows) - rows) ** 2).sum(dim=1)

    def _score(self, X) -> np.ndarray:
        rows = torch.from_numpy(X).float()
        return np.min([((self._error(net, rows) - mu) / sd).double().numpy() for net, mu, sd in self._parts], axis=0)


def make_baseline(name: str, background=None, **kw) -> _Detector:
    """A fresh, unfitted detector by its spec name. ``relative_mahalanobis`` needs ``background``: vectors of general text (the Other-content corpora) in the same space."""
    if name == "relative_mahalanobis":
        if background is None:
            raise ValueError("relative_mahalanobis needs a background sample of general text (background=...)")
        return RelativeMahalanobis(background, **kw)
    table: Dict[str, Callable[..., _Detector]] = {"vmf_conformal": VmfConformal, "knn_distance": KnnDistance, "mahalanobis": Mahalanobis,
                                                  "classifier_confidence": ClassifierConfidence, "energy": Energy, "autoencoder_descriptor": AutoencoderDescriptor}
    if name not in table:
        raise ValueError(f"unknown baseline {name!r}; the baselines are {list(BASELINE_NAMES)}")
    return table[name](**kw)
