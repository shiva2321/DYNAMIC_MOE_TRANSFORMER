"""von Mises-Fisher helpers on S^{d-1}: exact log-normaliser, mean resultant length, Wood's sampler.

The vMF density is f(x; mu, kappa) = C_d(kappa) exp(kappa <mu, x>) with
C_d(kappa) = kappa^{d/2-1} / ((2 pi)^{d/2} I_{d/2-1}(kappa)).
"""

import math

import numpy as np
import torch
from scipy.special import gammaln, ive


def _log_bessel_i(nu: float, kappa: float) -> float:
    v = ive(nu, kappa)
    if v > 0.0 and math.isfinite(v):
        return math.log(v) + kappa
    # underflow of the scaled Bessel function (huge order, small kappa): use the leading series
    # I_nu(k) = (k/2)^nu / Gamma(nu+1) * sum_m (k^2/4)^m / (m! (nu+1)_m)
    x = 0.25 * kappa * kappa
    term, total = 1.0, 1.0
    for m in range(1, 60):
        term *= x / (m * (nu + m))
        total += term
        if term < 1e-18 * total:
            break
    return nu * math.log(0.5 * kappa) - gammaln(nu + 1.0) + math.log(total)


def log_vmf_norm(d: int, kappa: float) -> float:
    """log C_d(kappa). For kappa <= 0 this is the log density of the uniform distribution on the sphere."""
    if kappa <= 0.0:
        return float(gammaln(d / 2.0) - math.log(2.0) - (d / 2.0) * math.log(math.pi))
    nu = d / 2.0 - 1.0
    return nu * math.log(kappa) - (d / 2.0) * math.log(2.0 * math.pi) - _log_bessel_i(nu, kappa)


def mean_resultant_length(d: int, kappa: float) -> float:
    """E[<mu, x>] = I_{d/2}(kappa) / I_{d/2-1}(kappa)."""
    if kappa <= 0.0:
        return 0.0
    return float(ive(d / 2.0, kappa) / ive(d / 2.0 - 1.0, kappa))


def sample_vmf(mu, kappa: float, n: int, rng: np.random.Generator) -> torch.Tensor:
    """Draw ``n`` unit vectors from vMF(mu, kappa) (Wood 1994). Deterministic given ``rng``."""
    mu = np.asarray(mu.detach().cpu().numpy() if isinstance(mu, torch.Tensor) else mu, dtype=np.float64)
    d = mu.size
    mu = mu / np.linalg.norm(mu)
    if kappa <= 0.0:
        x = rng.standard_normal((n, d))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return torch.from_numpy(x.astype(np.float32))

    b = (d - 1.0) / (2.0 * kappa + math.sqrt(4.0 * kappa * kappa + (d - 1.0) ** 2))
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (d - 1.0) * math.log(1.0 - x0 * x0)
    w = np.empty(n)
    filled = 0
    while filled < n:
        m = 2 * (n - filled) + 8
        z = rng.beta((d - 1.0) / 2.0, (d - 1.0) / 2.0, size=m)
        u = rng.random(m)
        wc = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        ok = kappa * wc + (d - 1.0) * np.log(1.0 - x0 * wc) - c >= np.log(u)
        take = wc[ok][: n - filled]
        w[filled:filled + take.size] = take
        filled += take.size

    v = rng.standard_normal((n, d))
    v -= (v @ mu)[:, None] * mu[None, :]
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    x = w[:, None] * mu[None, :] + np.sqrt(np.clip(1.0 - w * w, 0.0, None))[:, None] * v
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return torch.from_numpy(x.astype(np.float32))
