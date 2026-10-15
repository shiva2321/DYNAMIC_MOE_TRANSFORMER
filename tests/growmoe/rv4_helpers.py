"""Synthetic data shared by the RV4 tests."""

import numpy as np
import torch
import torch.nn.functional as F

from growmoe.config import PhyloCfg
from growmoe.pools.phylo import PhyloMoE


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def aniso(d=12, k=6, n=16, spacing=0.6, s_info=0.08, s_nuis=1.0, n_nuis=4, seed=0):
    """k classes whose means differ only along coordinate 0, where the noise is small; four other coordinates carry large noise that says nothing about the class.
    A shared-covariance discriminant (LDA) discounts the noisy coordinates; a kernel density over a handful of prototypes is dominated by them.
    (Chosen by a scratch search on 2026-10-15: CV owner accuracy lda 0.97-0.99 against density 0.46-0.60 over seeds 0-3.)"""
    g = np.random.default_rng(seed)
    scale = np.full(d, 0.08)
    scale[0] = s_info
    scale[1:1 + n_nuis] = s_nuis
    X, y = [], []
    for c in range(k):
        m = np.zeros(d)
        m[d - 1] = 3.0                                                 # a shared offset keeps the points away from the origin, so normalising keeps the geometry
        m[0] = (c - (k - 1) / 2) * spacing
        X.append(m + scale * g.standard_normal((n, d)))
        y += [c] * n
    return torch.from_numpy(unit(np.concatenate(X))), torch.tensor(y)


def antipodal(d=16, k=4, n=40, sigma=0.05, seed=0):
    """Expert c holds two OPPOSITE clusters (+v_c and -v_c): its class mean is near zero, so no linear rule can see it, while a density rule can."""
    g = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        v = np.zeros(d)
        v[c] = 1.0
        for s in (1.0, -1.0):
            X.append(s * v + sigma * g.standard_normal((n // 2, d)))
            y += [c] * (n // 2)
    return torch.from_numpy(unit(np.concatenate(X))), torch.tensor(y)


D, C = 16, 4


def teacher(seed):
    """A random linear teacher: label = argmax(x @ W)."""
    return torch.randn(D, C, generator=torch.Generator().manual_seed(seed))


def regime_batch(W, n, seed, perm=0):
    """n inputs uniform on the sphere, labelled by the teacher; ``perm`` shifts every label (the same inputs under a different function).
    A scratch run on 2026-10-15 (hidden 32, 5 inner steps, 40 batches of 64): accuracy 0.81, loss 0.45 on the same function and 4.9 under perm=1 (chance is 1.39)."""
    x = F.normalize(torch.randn(n, D, generator=torch.Generator().manual_seed(seed)), dim=1)
    return x, ((x @ W).argmax(1) + perm) % C


def trained_pool(n_batches=40, n_classes=C, **kw):
    """A pool with ONE expert that learned the teacher; returns (pool, W)."""
    cfg = dict(d=D, hidden=32, n_classes=n_classes, seed=3, q=0.02, alpha_spawn=0.01, m_min=32, inner_steps=5, route_rule="accept_then_logf", set_gate=False, group_vote=False,
               spawn_rule="transfer")
    cfg.update(kw)
    pool, W = PhyloMoE(PhyloCfg(**cfg)), teacher(1)
    for k in range(n_batches):
        pool.observe(*regime_batch(W, 64, 100 + k))
    return pool, W
