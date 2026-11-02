"""Routing v2, stage 4b: the spawn-or-extend question asked of the labels (spec docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md, section 5).

Today a 60-step look-ahead on samples the networks have memorised decides whether a new cluster is a new function or a harder part of an old one; it is close to a coin flip.
The plain question is: does the best existing expert already predict the new cluster's labels? Its zero-shot loss on the cluster is compared with its own honest prequential losses
(its reference). The comparison is an exact one-sided permutation test (the cluster's mean loss is larger than the reference's) with a practical-significance margin, because inside an
expert that holds several operators the harder one is slightly worse than the pooled reference and a bare test over-fires at large n (spike B, E-032).

p = (1 + c) / (1 + n_perm) with c = permutations at least as extreme as the observed split (Phipson and Smyth 2010), valid at every level. All ``n_perm`` permutations are drawn
(a call costs about ten milliseconds and happens once per cluster), so the reported p is the full exact value; the spec's early stop was dropped on purpose (refinement R-5):
it would turn p into a lower bound in the event log.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from .settest import max_exceedances


def cluster_conflict(cluster_losses: Tensor, ref_losses: Tensor, alpha: float, n_perm: int, margin: float, gen: torch.Generator, chunk: int = 64) -> Tuple[bool, float, float]:
    """``(conflict, p, diff)``: ``diff`` = mean cluster loss - mean reference loss; ``conflict`` = (p <= alpha and diff > margin). Cluster losses that are not finite (labels the
    expert has never seen) are a conflict outright (p = 0)."""
    c = torch.as_tensor(cluster_losses, dtype=torch.float64).flatten()
    r = torch.as_tensor(ref_losses, dtype=torch.float64).flatten()
    if c.numel() < 1 or r.numel() < 1:
        raise ValueError("the cluster and the reference must each hold at least one loss")
    if not bool(torch.isfinite(r).all()):
        raise ValueError("reference losses must be finite")
    allowed = max_exceedances(alpha, n_perm)
    if allowed < 0:
        raise ValueError(f"n_perm={n_perm} is too small for alpha={alpha}: the smallest attainable p-value 1/(1+n_perm) exceeds alpha, so the test could never reject")
    if not bool(torch.isfinite(c).all()):
        return True, 0.0, math.inf
    pooled = torch.cat([c, r])
    n_c, n_all = c.numel(), pooled.numel()
    total = pooled.sum()

    def stat(idx: Tensor) -> Tensor:                                    # idx (m, n_c): rows of the pooled sample forming the cluster
        cm = pooled[idx].mean(1)
        return cm - (total - cm * n_c) / (n_all - n_c)

    obs = float(stat(torch.arange(n_c)[None, :])[0])
    exceed = 0
    for lo in range(0, n_perm, chunk):
        m = min(chunk, n_perm - lo)
        exceed += int((stat(torch.rand(m, n_all, generator=gen).argsort(1)[:, :n_c]) >= obs).sum())
    p = (1 + exceed) / (1 + n_perm)
    diff = float(c.mean() - r.mean())
    return bool(p <= alpha and diff > margin), p, diff


@dataclass
class TransferVerdict:
    parent: int          # the expert with the lowest zero-shot loss on the cluster
    conflict: bool
    p: float
    diff: float
    loss_cluster: float
    loss_ref: float
    margin: float
    n_cluster: int
    n_ref: int
    chance_veto: bool = False    # the worse-than-chance rule (cfg.transfer_chance) turned a no-conflict verdict into a conflict

    def fields(self) -> dict:
        """What the ``spawn`` and ``extend`` events carry."""
        out = {"p_conflict": self.p, "loss_cluster": self.loss_cluster, "loss_ref": self.loss_ref, "margin": self.margin, "n_ref": self.n_ref}
        if self.chance_veto:
            out["chance_veto"] = True
        return out
