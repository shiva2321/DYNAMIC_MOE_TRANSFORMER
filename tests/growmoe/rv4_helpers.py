"""Synthetic data shared by the RV4 tests."""

import numpy as np
import torch


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
