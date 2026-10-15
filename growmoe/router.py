"""Routing v2, stage 4a: which expert ANSWERS (spec docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md, section 4).

The pool's density gate keeps deciding "known or unknown" and keeps its conformal guarantee. The router only decides, among the experts, who answers. It is trained on nothing
but what the experts already store (each expert's signature prototypes, labelled with the expert that holds them), chooses its own rule among a few cheap candidates by
cross-validated owner accuracy, and falls back to the density rule (today's) unless another rule is clearly better.

Everything here is a pure function of its arguments (a generator carries every random draw), so equal inputs give equal outputs. Scorers are small classes, not closures, so a pool
that holds one can still be pickled (the diagnosis pipeline pickles pools).
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .signature import VmfSignature

CANDIDATES = ("density", "lda", "softmax", "knn5")
NEG_INF = -math.inf


class DensityScorer:
    """One kernel density per expert, rebuilt from the rows it was given; score = log kernel density + the vMF normaliser (the rule of the pool's own gate, without its conformal part)."""

    def __init__(self, sigs: List[VmfSignature], present: List[bool]) -> None:
        self.sigs, self.present = sigs, present

    def __call__(self, Z: Tensor) -> Tensor:
        cols = [s.score(Z) + s._log_c[s._kappa] if ok else torch.full((Z.shape[0],), NEG_INF) for s, ok in zip(self.sigs, self.present)]
        return torch.stack(cols, dim=1)


def fit_density(X: Tensor, y: Tensor, k: int, gen: Optional[torch.Generator] = None) -> DensityScorer:
    sigs, present = [], []
    for c in range(k):
        rows = X[y == c]
        ok = rows.shape[0] >= 2
        s = VmfSignature(d=X.shape[1], max_protos=max(256, rows.shape[0]), seed=c)
        if ok:
            s.update(rows)
            s.fit()
        sigs.append(s)
        present.append(ok)
    return DensityScorer(sigs, present)


class LdaScorer:
    """Linear discriminants of class means with one pooled within-class covariance."""

    def __init__(self, w: Tensor, b: Tensor, present: Tensor) -> None:
        self.w, self.b, self.present = w, b, present

    def __call__(self, Z: Tensor) -> Tensor:
        return (Z.double() @ self.w + self.b).masked_fill(~self.present, NEG_INF).float()


def fit_lda(X: Tensor, y: Tensor, k: int, gen: Optional[torch.Generator] = None, reg: float = 1e-3) -> LdaScorer:
    """Class means, the pooled within-class covariance plus ``reg`` times the identity, log priors. Closed form (``gen`` is unused)."""
    Xd = X.double()
    n, d = Xd.shape
    counts = torch.bincount(y, minlength=k)
    present = counts > 0
    mu = torch.zeros(k, d, dtype=torch.float64).index_add_(0, y, Xd) / counts.clamp_min(1).double().unsqueeze(1)
    xc = Xd - mu[y]
    cov = xc.T @ xc / max(n - int(present.sum()), 1) + reg * torch.eye(d, dtype=torch.float64)
    w = torch.linalg.solve(cov, mu.T)                                  # (d, k) = precision @ means^T
    b = -0.5 * (mu * w.T).sum(dim=1) + torch.log(counts.clamp_min(1).double() / n)
    return LdaScorer(w, b, present)


class SoftmaxScorer:
    def __init__(self, w: Tensor, b: Tensor, present: Tensor) -> None:
        self.w, self.b, self.present = w, b, present

    def __call__(self, Z: Tensor) -> Tensor:
        return (Z @ self.w + self.b).masked_fill(~self.present, NEG_INF)


def fit_softmax(X: Tensor, y: Tensor, k: int, gen: torch.Generator, steps: int = 500, lr: float = 0.05, l2: float = 1e-3, batch: int = 2048) -> SoftmaxScorer:
    """Multinomial logistic regression, Adam, L2, **class-balanced loss** (every expert counts the same however many prototypes it holds)."""
    n, d = X.shape
    counts = torch.bincount(y, minlength=k)
    w = torch.zeros(d, k, requires_grad=True)
    b = torch.zeros(k, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    cls_w = (n / (k * counts.clamp_min(1))).float()
    with torch.enable_grad():                                           # predict is usually called under torch.no_grad(), and the lazy refit inside it has to train
        for _ in range(steps):
            idx = torch.randint(0, n, (min(batch, n),), generator=gen)
            loss = F.cross_entropy(X[idx] @ w + b, y[idx], weight=cls_w) + l2 * (w ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return SoftmaxScorer(w.detach(), b.detach(), counts > 0)


class KnnScorer:
    """Cosine vote of the ``nn`` nearest stored rows (a score per expert: the summed positive similarities of its rows among the neighbours)."""

    def __init__(self, X: Tensor, y: Tensor, k: int, nn: int = 5) -> None:
        self.X, self.y, self.k, self.nn = X, y, k, nn

    def __call__(self, Z: Tensor) -> Tensor:
        top = (Z @ self.X.T).topk(min(self.nn, self.X.shape[0]), dim=1)
        out = torch.zeros(Z.shape[0], self.k)
        out.scatter_add_(1, self.y[top.indices], top.values.clamp_min(0) + 1e-6)
        return out


def fit_knn(X: Tensor, y: Tensor, k: int, gen: Optional[torch.Generator] = None, nn: int = 5) -> KnnScorer:
    return KnnScorer(X.clone(), y.clone(), k, nn)


FITS = {"density": fit_density, "lda": fit_lda, "softmax": fit_softmax, "knn5": fit_knn}


def fold_assignment(y: Tensor, k: int, folds: int, gen: torch.Generator) -> Tensor:
    """Fold of every row: the rows of each expert are shuffled and dealt round-robin, so every fold holds (nearly) the same share of every expert."""
    fold_of = torch.zeros(y.shape[0], dtype=torch.long)
    for c in range(k):
        idx = (y == c).nonzero().squeeze(1)
        if idx.numel():
            fold_of[idx[torch.randperm(idx.numel(), generator=gen)]] = torch.arange(idx.numel()) % folds
    return fold_of


def cv_owner_accuracy(X: Tensor, y: Tensor, k: int, folds: int, gen: torch.Generator, candidates: Sequence[str] = CANDIDATES) -> Dict[str, float]:
    """Mean over folds of how often each candidate, fitted on the other folds, names the expert that holds a held-out row."""
    fold_of = fold_assignment(y, k, folds, gen)
    acc: Dict[str, List[float]] = {name: [] for name in candidates}
    for f in range(folds):
        te = fold_of == f
        if not bool(te.any()):
            continue
        tr = ~te
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen))
        for name in candidates:
            scorer = FITS[name](X[tr], y[tr], k, torch.Generator().manual_seed(seed))
            acc[name].append(float((scorer(X[te]).argmax(1) == y[te]).float().mean()))
    return {name: (sum(v) / len(v) if v else 0.0) for name, v in acc.items()}


def select_rule(cv: Dict[str, float], current: Optional[str], margin: float) -> str:
    """``density`` unless another candidate beats density's cross-validated accuracy by MORE than ``margin`` (then the best such candidate). Hysteresis: when a rule is already in
    use, a different one replaces it only if its accuracy exceeds the current rule's by more than ``margin`` too."""
    others = {n: a for n, a in cv.items() if n != "density"}
    best = max(others, key=others.get) if others else None
    candidate = best if best is not None and cv[best] > cv["density"] + margin else "density"
    if current is not None and current in cv and candidate != current and not cv[candidate] > cv[current] + margin:
        return current
    return candidate


class RouteRule:
    """What the pool holds: the selected rule, its fitted scorer (None for density), the experts (ascending ids) it was fitted for, and the last cross-validation."""

    def __init__(self) -> None:
        self.name: str = "density"
        self.scorer = None
        self.eids: Optional[List[int]] = None
        self.cv: Dict[str, float] = {}
        self.rows: int = 0

    def set(self, name: str, scorer, eids: List[int], cv: Dict[str, float], rows: int) -> None:
        self.name, self.scorer, self.eids, self.cv, self.rows = name, scorer, list(eids), dict(cv), int(rows)

    def cost(self, d: int) -> Tuple[int, int, int]:
        """(parameters, extra multiply-adds per prediction, extra stored scalars) of the selected rule."""
        k = len(self.eids or ())
        if self.name in ("lda", "softmax"):
            return k * d + k, k * d, 0
        if self.name == "knn5":
            return 0, self.rows * d, self.rows * d
        return 0, 0, 0
