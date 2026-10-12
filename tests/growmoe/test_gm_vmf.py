import math

import numpy as np
import pytest
import torch
from scipy.special import gammaln, logsumexp

from growmoe.vmf import log_vmf_norm, mean_resultant_length, sample_vmf


def test_log_norm_matches_closed_form_in_three_dimensions():
    for kappa in (1.0, 5.0, 20.0):
        closed = math.log(kappa / (4 * math.pi * math.sinh(kappa)))
        assert abs(log_vmf_norm(3, kappa) - closed) < 1e-8


@pytest.mark.parametrize("d", [3, 8, 32, 128])
@pytest.mark.parametrize("kappa", [1.0, 10.0, 100.0])
def test_density_integrates_to_one_on_the_sphere(d, kappa):
    # integral over S^{d-1} of C exp(kappa cos t) = int_0^pi C exp(kappa cos t) A_{d-2} sin^{d-2} t dt
    t = np.linspace(1e-9, math.pi - 1e-9, 400001)
    log_area = math.log(2.0) + 0.5 * (d - 1) * math.log(math.pi) - gammaln(0.5 * (d - 1))
    log_integrand = log_vmf_norm(d, kappa) + kappa * np.cos(t) + log_area + (d - 2) * np.log(np.sin(t))
    log_total = logsumexp(log_integrand) + math.log(t[1] - t[0])
    assert abs(math.exp(log_total) - 1.0) < 2e-3


@pytest.mark.parametrize("d,kappa", [(8, 5.0), (128, 50.0), (128, 300.0)])
def test_sampler_mean_cosine_matches_theory(d, kappa):
    rng = np.random.default_rng(0)
    mu = torch.zeros(d)
    mu[3] = 1.0
    x = sample_vmf(mu, kappa, 4000, rng)
    assert x.shape == (4000, d)
    assert torch.allclose(x.norm(dim=1), torch.ones(4000), atol=1e-5)
    cos = (x @ mu).numpy()
    se = cos.std() / math.sqrt(len(cos))
    assert abs(cos.mean() - mean_resultant_length(d, kappa)) < 4 * se + 1e-3


def test_kappa_zero_is_uniform_and_sampling_is_reproducible():
    mu = torch.nn.functional.normalize(torch.randn(64), dim=0)
    x = sample_vmf(mu, 0.0, 3000, np.random.default_rng(1))
    assert abs((x @ mu).mean().item()) < 0.01
    a = sample_vmf(mu, 20.0, 50, np.random.default_rng(7))
    b = sample_vmf(mu, 20.0, 50, np.random.default_rng(7))
    assert torch.equal(a, b)
