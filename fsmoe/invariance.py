"""A learned invariance map: a small linear map trained contrastively so that the same skill in different wordings lands together and a near-twin skill is pushed hardest.

Supervised contrastive loss on TEACHING rows only (taught skills, teaching wordings): positives are the other rows of the same skill, negatives are the rows of every other skill, and rows of the
anchor's twin skill count ``twin_weight`` times in the denominator (the hard negatives: one operator or one word away). The map takes unit vectors of the frontend and returns unit vectors of
``out`` numbers. It cannot be fitted on a held-out wording or a withheld skill: the caller passes only teaching rows, and ``fit`` checks that every row is flagged as one.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class InvarianceMap:
    def __init__(self, d_in: int, out: int = 64, tau: float = 0.1, twin_weight: float = 4.0, epochs: int = 200, lr: float = 3e-3, weight_decay: float = 1e-2, seed: int = 0) -> None:
        self.d_in, self.out, self.tau, self.twin_weight, self.epochs, self.lr, self.wd, self.seed = d_in, out, tau, twin_weight, epochs, lr, weight_decay, seed
        self.W: Optional[Tensor] = None

    def fit(self, X: Tensor, y: Tensor, twin_of: Dict[int, int], teaching: Sequence[bool]) -> "InvarianceMap":
        if not bool(np.all(np.asarray(teaching, dtype=bool))):
            raise ValueError("the map may only be fitted on teaching rows (every row must be flagged as one)")
        X, y = X.float(), y.long()
        n = X.shape[0]
        g = torch.Generator().manual_seed(self.seed)
        W = torch.nn.init.orthogonal_(torch.empty(self.d_in, self.out), generator=g).requires_grad_(True)
        classes = sorted(set(y.tolist()))
        twin_row = torch.tensor([twin_of.get(int(c), -1) for c in y.tolist()])
        same = y[:, None] == y[None, :]
        eye = torch.eye(n, dtype=torch.bool)
        pos = same & ~eye
        twin = (y[None, :] == twin_row[:, None]) & ~same
        weight = torch.where(twin, torch.tensor(self.twin_weight), torch.tensor(1.0)) * (~same).float()
        has_pos = pos.any(1)
        opt = torch.optim.AdamW([W], lr=self.lr, weight_decay=self.wd)
        for _ in range(self.epochs):
            z = F.normalize(X @ W, dim=1)
            s = (z @ z.T) / self.tau
            s = s.masked_fill(eye, -1e9)
            e = torch.exp(s - s.max(1, keepdim=True).values)
            denom = (e * pos.float()).sum(1) + (e * weight).sum(1)
            logp = (s - s.max(1, keepdim=True).values) - torch.log(denom.clamp_min(1e-30))[:, None]
            loss = -(logp * pos.float()).sum(1)[has_pos] / pos.float().sum(1)[has_pos]
            opt.zero_grad()
            loss.mean().backward()
            opt.step()
        self.W = W.detach()
        return self

    def __call__(self, X: Tensor) -> Tensor:
        if self.W is None:
            raise RuntimeError("fit the map first")
        return F.normalize(X.float() @ self.W, dim=1)
