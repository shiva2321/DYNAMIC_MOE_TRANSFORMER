"""Residual-MLP expert with a seen-class mask and function-preserving growth.

    block:   h <- h + W2 gelu(W1 LN(h) + b1) + b2
    expert:  logits = head(LN_f(blocks(u))), classes this expert has never seen masked to -1e9

* ``widen(k)``   k new hidden units in every block: random incoming weights, ZERO outgoing weights.
* ``deepen()``   a new block with zero output weights, i.e. the identity on the residual stream.
* ``inherit(rho, sigma, rng)``  child copy in which a fraction ``rho`` of every block's hidden units is
  re-drawn from the parent's weight statistics and N(0, sigma * std) noise is added to the block weights
  (partial re-initialisation: rho and sigma are meant to grow as the parent explains the new data less well).

Growth replaces parameters, so an optimiser built before ``widen``/``deepen`` must be rebuilt afterwards.
No global RNG is touched: weights come from a per-expert generator, growth draws from a numpy generator.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

MASK_VALUE = -1e9


def agem_project(g: Tensor, g_ref: Tensor) -> Tensor:
    """A-GEM (Chaudhry et al., 2019): if g conflicts with the reference gradient, remove the conflicting component."""
    dot = g @ g_ref
    if dot >= 0:
        return g
    return g - dot / (g_ref @ g_ref).clamp_min(1e-20) * g_ref


class ResBlock(nn.Module):
    def __init__(self, d: int, hidden: int, gen: torch.Generator) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.w1 = nn.Parameter(torch.randn(hidden, d, generator=gen) / math.sqrt(d))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.randn(d, hidden, generator=gen) / math.sqrt(hidden))
        self.b2 = nn.Parameter(torch.zeros(d))

    def forward(self, h: Tensor) -> Tensor:
        return h + F.linear(F.gelu(F.linear(self.ln(h), self.w1, self.b1)), self.w2, self.b2)


class MLPExpert(nn.Module):
    def __init__(self, d_in: int = 128, hidden: int = 128, depth: int = 1, n_classes: int = 64,
                 seed: int = 0) -> None:
        super().__init__()
        self.d_in, self.n_classes, self.seed = d_in, n_classes, seed
        gen = torch.Generator().manual_seed(seed)
        self.blocks = nn.ModuleList(ResBlock(d_in, hidden, gen) for _ in range(depth))
        self.norm = nn.LayerNorm(d_in)
        with torch.random.fork_rng(devices=[]):          # nn.Linear's default init would advance the global RNG
            self.head = nn.Linear(d_in, n_classes)
        with torch.no_grad():
            self.head.weight.copy_(torch.randn(n_classes, d_in, generator=gen) / math.sqrt(d_in))
            self.head.bias.zero_()
        self.register_buffer("seen", torch.zeros(n_classes, dtype=torch.bool))
        self._ops = 0

    # ------------------------------------------------------------------ forward
    @property
    def hidden(self) -> int:
        return self.blocks[0].w1.shape[0]

    def features(self, u: Tensor) -> Tensor:
        h = u
        for block in self.blocks:
            h = block(h)
        return self.norm(h)

    def forward(self, u: Tensor) -> Tensor:
        return self.head(self.features(u)).masked_fill(~self.seen, MASK_VALUE)

    def proba(self, u: Tensor) -> Tensor:
        return F.softmax(self(u), dim=-1)

    def mark_seen(self, y: Tensor) -> None:
        y = torch.as_tensor(y).long().flatten()
        if y.numel() and (int(y.min()) < 0 or int(y.max()) >= self.n_classes):
            raise ValueError(f"labels must lie in [0, {self.n_classes})")
        self.seen[y] = True

    def loss(self, u: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy(self(u), y.long())

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def clone(self) -> "MLPExpert":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------- growth
    def _next_rng(self) -> np.random.Generator:
        self._ops += 1
        return np.random.default_rng([self.seed, 0x47524F57, self._ops])

    @torch.no_grad()
    def widen(self, k: int, rng: np.random.Generator = None) -> "MLPExpert":
        if k <= 0:
            raise ValueError("widen needs k >= 1")
        rng = rng if rng is not None else self._next_rng()
        for b in self.blocks:
            d = b.w1.shape[1]
            fresh = torch.from_numpy(rng.standard_normal((k, d)) / math.sqrt(d)).to(b.w1)
            b.w1 = nn.Parameter(torch.cat([b.w1.data, fresh]))
            b.b1 = nn.Parameter(torch.cat([b.b1.data, torch.zeros(k).to(b.b1)]))
            b.w2 = nn.Parameter(torch.cat([b.w2.data, torch.zeros(d, k).to(b.w2)], dim=1))
        return self

    @torch.no_grad()
    def deepen(self, rng: np.random.Generator = None) -> "MLPExpert":
        rng = rng if rng is not None else self._next_rng()
        ref = self.blocks[-1]
        gen = torch.Generator().manual_seed(int(rng.integers(0, 2 ** 31 - 1)))
        block = ResBlock(ref.w1.shape[1], ref.w1.shape[0], gen).to(ref.w1)
        block.w2.zero_()
        self.blocks.append(block)
        return self

    @torch.no_grad()
    def inherit(self, rho: float, sigma: float, rng: np.random.Generator = None) -> "MLPExpert":
        rng = rng if rng is not None else self._next_rng()
        child = self.clone()
        for b in child.blocks:
            h, d = b.w1.shape
            s1, s2 = float(b.w1.std()), float(b.w2.std())
            m = int(round(rho * h))
            if m:
                idx = torch.from_numpy(np.sort(rng.choice(h, size=m, replace=False)))
                b.w1[idx] = torch.from_numpy(rng.standard_normal((m, d)) * s1).to(b.w1)
                b.b1[idx] = 0.0
                b.w2[:, idx] = torch.from_numpy(rng.standard_normal((d, m)) * s2).to(b.w2)
            if sigma > 0.0:
                b.w1.add_(torch.from_numpy(rng.standard_normal(tuple(b.w1.shape)) * sigma * s1).to(b.w1))
                b.w2.add_(torch.from_numpy(rng.standard_normal(tuple(b.w2.shape)) * sigma * s2).to(b.w2))
        return child

    @torch.no_grad()
    def shrink_perturb(self, lam: float, seed: int = 0) -> "MLPExpert":
        """Child initialisation: ``lam * parent + (1 - lam) * fresh`` over EVERY parameter (shrink-and-perturb, Ash and
        Adams 2020, with the shrink factor chosen by the caller, e.g. the parent's affinity to the new data).
        ``lam = 0`` is exactly a fresh expert with this seed, ``lam = 1`` a clone. The child keeps the parent's
        seen-class mask. Phase-0 spike 2 measured this to match a clone on related regimes (zero-shot accuracy 0.99
        at class-mean cosine 0.9) and to equal scratch on unrelated ones, where a clone is 4-14 % worse."""
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam must lie in [0, 1]")
        child = self.clone()
        fresh = MLPExpert(self.d_in, self.hidden, len(self.blocks), self.n_classes, seed=seed)
        for pc, pf in zip(child.parameters(), fresh.parameters()):
            pc.mul_(lam).add_((1.0 - lam) * pf.to(pc))
        return child

    # ----------------------------------------------------------------- training
    @staticmethod
    def _flat_grad(loss: Tensor, params) -> Tensor:
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(grads, params)])

    def train_step(self, u: Tensor, y: Tensor, opt, replay=None, project: bool = False) -> float:
        """One optimiser step on (u, y); returns the loss on the new batch measured before the update.

        ``replay=(x, y)`` is mixed into the batch; with ``project=True`` it is instead used as an A-GEM
        reference: if the new gradient conflicts with the replay gradient (negative dot product) the
        conflicting component is projected out (Chaudhry et al., 2019).
        """
        self.train()
        y = y.long()
        self.mark_seen(y)
        if replay is not None:
            xr, yr = replay[0], replay[1].long()
            self.mark_seen(yr)
        if replay is not None and project:
            params = [p for p in self.parameters() if p.requires_grad]
            loss_new = F.cross_entropy(self(u), y)
            g = agem_project(self._flat_grad(loss_new, params),
                             self._flat_grad(F.cross_entropy(self(xr), yr), params))
            opt.zero_grad(set_to_none=True)
            offset = 0
            for p in params:
                p.grad = g[offset:offset + p.numel()].view_as(p).clone()
                offset += p.numel()
            opt.step()
            return float(loss_new.detach())
        if replay is not None:
            x_all, y_all = torch.cat([u, xr]), torch.cat([y, yr])
        else:
            x_all, y_all = u, y
        logits = self(x_all)
        loss = F.cross_entropy(logits, y_all)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(F.cross_entropy(logits[: u.shape[0]], y).detach())
