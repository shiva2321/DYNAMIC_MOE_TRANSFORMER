"""Non-parametric von Mises-Fisher kernel-density signature of what an expert has seen.

A signature keeps a bounded, uniformly random reservoir of unit-vector prototypes (Vitter's Algorithm R)
and scores a unit vector u by the log of its vMF kernel density estimate

    score(u) = log( (1/n) sum_j exp(kappa <p_j, u>) )

(the exact vMF normaliser C_d(kappa) is a constant added by ``logpdf``). There is no cluster count, no
covariance and no low-dimensional-subspace assumption, so it works on any frontend's output. The kernel
concentration kappa is chosen from a fixed grid by leave-one-out likelihood using the exact normaliser,
and p-values are conformal ones against the prototypes' own leave-one-out scores (small score = novel).
"""

import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .conformal import conformal_pvalues
from .vmf import log_vmf_norm

DEFAULT_KAPPA_GRID = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)


class VmfSignature:
    def __init__(self, d: int, max_protos: int = 256,
                 kappa_grid: Sequence[float] = DEFAULT_KAPPA_GRID, seed: int = 0, dedupe: bool = False) -> None:
        grid = tuple(sorted(float(k) for k in kappa_grid))
        if d < 2 or max_protos < 2 or not grid or grid[0] <= 0.0:
            raise ValueError("need d >= 2, max_protos >= 2 and a non-empty positive kappa_grid")
        self.d = d
        self.max_protos = max_protos
        self.kappa_grid = grid
        self.seed = seed
        self.dedupe = dedupe                      # True: a point already held is not added again (see ``update``)
        self.seen = 0
        self._rng = np.random.default_rng([seed, 0x5167])
        self._log_c = {k: log_vmf_norm(d, k) for k in grid}
        self._protos = torch.zeros(max_protos, d)
        self._ranks = torch.zeros(max_protos, dtype=torch.long)
        self._n = 0
        self._kappa = grid[len(grid) // 2]
        self._cal = None
        self._dirty = True

    # ------------------------------------------------------------------ state
    @property
    def n(self) -> int:
        return self._n

    @property
    def protos(self) -> Tensor:
        return self._protos[: self._n]

    @property
    def kappa(self) -> float:
        self._refit_if_stale()
        return self._kappa

    @property
    def cal_scores(self) -> Tensor:
        """Leave-one-out scores of the prototypes (the conformal calibration set)."""
        if self._n < 2:
            raise ValueError("calibration needs at least 2 prototypes")
        self._refit_if_stale()
        return self._cal

    def arrival_ranks(self) -> Tensor:
        """Stream position (0-based) at which each held prototype arrived."""
        return self._ranks[: self._n].clone()

    def _unit(self, x) -> Tensor:
        x = torch.as_tensor(x).detach().to(torch.float32)
        if x.ndim == 1:
            x = x[None]
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"expected vectors of length {self.d}, got shape {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("signature input contains non-finite values")
        norm = x.norm(dim=1, keepdim=True)
        fixed = torch.zeros_like(x)
        fixed[:, 0] = 1.0
        return torch.where(norm > 1e-12, x / norm.clamp_min(1e-12), fixed)

    # ----------------------------------------------------------------- update
    def update(self, x) -> None:
        """Stream points into the reservoir (Algorithm R: every point seen so far is kept w.p. max_protos/seen). With ``dedupe`` a point that is already a prototype (cosine within 1e-5 of
        1, which is float32 rounding of the same vector) is skipped and not counted as seen: a signature that holds a point and its copy has a twin for that point in every leave-one-out score,
        which drives all calibration scores to the same value and the p-value of fresh points of the same region to the floor 1/(n+1)."""
        x = self._unit(x)
        if getattr(self, "dedupe", False) and self._n > 0:
            x = x[(x @ self.protos.T).max(dim=1).values < 1.0 - 1e-5]
            if x.shape[0] == 0:
                return
        b = x.shape[0]
        ts = self.seen + np.arange(b, dtype=np.int64)
        fill = min(b, self.max_protos - self._n)
        changed = fill > 0
        if fill:
            sl = slice(self._n, self._n + fill)
            self._protos[sl] = x[:fill]
            self._ranks[sl] = torch.from_numpy(ts[:fill])
            self._n += fill
        if fill < b:
            slots = self._rng.integers(0, ts[fill:] + 1)
            for k in np.flatnonzero(slots < self.max_protos):
                self._protos[slots[k]] = x[fill + k]
                self._ranks[slots[k]] = int(ts[fill + k])
                changed = True
        self.seen += b
        if changed:
            self._dirty = True

    def merge_from(self, other: "VmfSignature") -> None:
        """Absorb another signature; each prototype is weighted by how many stream points it stands for."""
        if other.d != self.d:
            raise ValueError("cannot merge signatures of different dimension")
        if other._n == 0:
            return
        protos = torch.cat([self.protos, other.protos])
        ranks = torch.cat([self._ranks[: self._n], other._ranks[: other._n] + self.seen])
        if protos.shape[0] > self.max_protos:
            w = np.concatenate([np.full(self._n, self.seen / max(self._n, 1)),
                                np.full(other._n, other.seen / other._n)])
            keep = np.sort(self._rng.choice(protos.shape[0], size=self.max_protos, replace=False, p=w / w.sum()))
            protos, ranks = protos[keep], ranks[keep]
        m = protos.shape[0]
        self._protos[:m] = protos
        self._ranks[:m] = ranks
        self._n = m
        self.seen += other.seen
        self._dirty = True

    # -------------------------------------------------------------------- fit
    def fit(self) -> "VmfSignature":
        """Pick kappa by leave-one-out log-likelihood (with the exact normaliser) and store the LOO scores."""
        if self._n < 2:
            raise ValueError("fitting kappa needs at least 2 prototypes")
        p = self.protos.double()
        gram = p @ p.T
        gram.fill_diagonal_(-math.inf)
        log_m = math.log(self._n - 1)
        best = None
        for k in self.kappa_grid:
            loo = torch.logsumexp(k * gram, dim=1) - log_m
            ll = loo.mean().item() + self._log_c[k]
            if best is None or ll > best[0]:
                best = (ll, k, loo)
        self._kappa = best[1]
        self._cal = best[2].to(torch.float32)
        self._dirty = False
        return self

    def _refit_if_stale(self) -> None:
        if self._dirty and self._n >= 2:
            self.fit()

    # ---------------------------------------------------------------- scoring
    def score(self, x) -> Tensor:
        if self._n == 0:
            raise ValueError("an empty signature cannot score")
        k = self.kappa
        u = self._unit(x)
        return torch.logsumexp(k * (u @ self.protos.T), dim=1) - math.log(self._n)

    def logpdf(self, x) -> Tensor:
        s = self.score(x)
        return s + self._log_c[self._kappa]

    def pvalue(self, x) -> Tensor:
        """Conformal p-value; small means unlike everything this signature has seen."""
        cal = self.cal_scores
        return conformal_pvalues(cal, self.score(x))

    # --------------------------------------------------------------- geometry
    def nn_similarity(self) -> Tensor:
        """Cosine similarity of every prototype to its nearest other prototype."""
        if self._n < 2:
            raise ValueError("nearest-neighbour similarity needs at least 2 prototypes")
        gram = self.protos @ self.protos.T
        gram.fill_diagonal_(-2.0)
        return gram.max(dim=1).values

    def affinity(self, x, n_random: int = 500, seed: int = 0) -> float:
        """How close new data are to what this signature holds, on a 0..1 scale anchored at both ends: 1 means as close
        to the prototypes as the prototypes are to each other, 0 means no closer than uniformly random directions.
        Used to set how much of a parent a child inherits (Phase-0 spike 2: it tracks the class-mean cosine of a
        rotated copy almost linearly, 0.01 / 0.54 / 0.84 / 0.94 for unrelated / 0.7 / 0.9 / 0.97)."""
        if self._n < 2:
            raise ValueError("affinity needs at least 2 prototypes")
        u = self._unit(x)
        s_new = (u @ self.protos.T).max(dim=1).values.mean().item()
        s_self = self.nn_similarity().mean().item()
        z = np.random.default_rng([self.seed, seed, 0x41464E]).standard_normal((n_random, self.d))
        rand = torch.from_numpy((z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32))
        s_floor = (rand @ self.protos.T).max(dim=1).values.mean().item()
        return float(np.clip((s_new - s_floor) / max(s_self - s_floor, 1e-9), 0.0, 1.0))

    def mean_direction(self) -> Tensor:
        if self._n == 0:
            raise ValueError("an empty signature has no mean direction")
        m = self.protos.mean(dim=0)
        return m / m.norm().clamp_min(1e-12)
