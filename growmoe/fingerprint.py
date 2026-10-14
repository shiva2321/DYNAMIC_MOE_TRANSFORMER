"""Stage-1 fingerprints for the gate shortlist (docs/superpowers/specs/2026-10-10-routing-v2-design.md section 5).

An expert's fingerprint is k unit sub-centroids of its stored prototypes. A first, cheap pass scores an input against every expert's
sub-centroids (best cosine) and keeps the top few experts; only those get the exact conformal score. No pool state lives here.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def spherical_kmeans(points: Tensor, k: int, iters: int = 10, seed: int = 0) -> Tensor:
    """k unit-length centroids of unit-length ``points`` (rows); k is capped at the number of points."""
    n = points.shape[0]
    k = max(1, min(int(k), n))
    g = torch.Generator().manual_seed(int(seed))
    centroids = points[torch.randperm(n, generator=g)[:k]].clone()
    for _ in range(iters):
        assign = (points @ centroids.T).argmax(1)
        for j in range(k):
            members = assign == j
            if bool(members.any()):
                centroids[j] = F.normalize(points[members].mean(0), dim=0)
    return centroids


class Fingerprint:
    """k sub-centroids of one expert, rebuilt when its signature has seen ``refresh`` more stream points than at the last build."""

    def __init__(self, centroids: Tensor, built_seen: int) -> None:
        self.centroids, self.built_seen = centroids, int(built_seen)

    @classmethod
    def build(cls, sig, k: int, seed: int) -> "Fingerprint":
        return cls(spherical_kmeans(sig.protos, k, seed=seed), sig.seen)

    def stale(self, sig, refresh: int) -> bool:
        return sig.seen - self.built_seen >= refresh

    def padded(self, k: int) -> Tensor:
        """Exactly k rows (repeating rows when the expert has fewer), so all experts stack into one (E, k, d) tensor."""
        c = self.centroids
        if c.shape[0] >= k:
            return c[:k]
        return c.repeat(-(-k // c.shape[0]), 1)[:k]
