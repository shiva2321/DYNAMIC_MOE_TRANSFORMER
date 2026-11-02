"""The frontend: raw reader states become unit vectors, the same map for every domain.

The raw input is the concatenation of the readers' blocks (``dims``, default Qwen 1536 + MiniLM 384). Each block is standardised with the mean and spread of the rows the frontend is fitted on (raw
language-model states are nearly degenerate: a few huge dimensions), projected to d numbers (a fixed Gaussian projection, or the top principal directions of those rows) and normalised; the blocks
are concatenated and scaled so that the whole vector has unit length (d x number of blocks numbers). It is fitted ONCE, before any stream, on UNLABELLED TEACHING rows only: rows of taught skills in
teaching wordings. It never sees a held-out wording or a withheld skill (``fit`` checks the flags it is given).
"""

from typing import List, Optional, Sequence

import numpy as np
import torch


class Frontend:
    def __init__(self, d: int = 128, mode: str = "rp", seed: int = 0, dims: Sequence[int] = (1536, 384)) -> None:
        if mode not in ("rp", "pca"):
            raise ValueError(f"mode must be 'rp' or 'pca', got {mode!r}")
        self.d, self.mode, self.seed, self.dims = d, mode, seed, tuple(int(k) for k in dims)
        self.mu: List[np.ndarray] = []
        self.sd: List[np.ndarray] = []
        self.P: List[np.ndarray] = []

    @property
    def out_dim(self) -> int:
        return self.d * len(self.dims)

    def _blocks(self, raw: np.ndarray) -> List[np.ndarray]:
        raw = np.asarray(raw)
        if raw.ndim != 2 or raw.shape[1] != sum(self.dims):
            raise ValueError(f"expected rows of {sum(self.dims)} numbers (blocks {self.dims}), got shape {raw.shape}")
        edges = np.cumsum((0,) + self.dims)
        return [raw[:, edges[k]:edges[k + 1]].astype(np.float32) for k in range(len(self.dims))]

    def fit(self, raw: np.ndarray, teaching: np.ndarray) -> "Frontend":
        """``teaching`` is a boolean flag per row of ``raw``; every row must be flagged."""
        teaching = np.asarray(teaching, dtype=bool)
        if np.asarray(raw).shape[0] != teaching.shape[0] or not teaching.all():
            raise ValueError("the frontend may only be fitted on teaching rows (every row must be flagged as one)")
        self.mu, self.sd, self.P = [], [], []
        for k, x in enumerate(self._blocks(raw)):
            mu, sd = x.mean(0), x.std(0) + 1e-3
            z = (x - mu) / sd
            if self.mode == "rp":
                rng = np.random.default_rng([self.seed, 0xF50E, k])
                P = (rng.standard_normal((x.shape[1], self.d)) / np.sqrt(self.d)).astype(np.float32)
            else:
                _, _, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
                P = vt[:self.d].T.astype(np.float32)
            self.mu.append(mu)
            self.sd.append(sd)
            self.P.append(P)
        return self

    def __call__(self, raw: np.ndarray) -> torch.Tensor:
        if not self.P:
            raise RuntimeError("fit the frontend first")
        out = []
        for k, x in enumerate(self._blocks(raw)):
            t = torch.from_numpy(((x - self.mu[k]) / self.sd[k]) @ self.P[k])
            out.append(t / t.norm(dim=1, keepdim=True).clamp_min(1e-12))
        return torch.cat(out, 1) / float(np.sqrt(len(out)))
