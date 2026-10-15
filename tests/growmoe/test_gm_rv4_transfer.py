"""RV4 stage 4b, the pure test: does the best existing expert fail on the new cluster's labels? (spec section 5.1)"""

import math

import pytest
import torch

from growmoe.settest import generator
from growmoe.transfer import cluster_conflict


def test_false_conflicts_stay_at_or_below_alpha_under_exchangeability():
    false = 0
    for t in range(400):
        g = torch.Generator().manual_seed(1000 + t)
        c, r = torch.randn(32, generator=g).abs(), torch.randn(256, generator=g).abs()                  # one distribution
        conflict, _, _ = cluster_conflict(c, r, 0.05, 199, 0.0, generator(7, t))
        false += conflict
    assert false <= 34                                                                                  # Binomial(400, 0.05): mean 20, its 99.9th percentile is about 34


def test_a_large_shift_is_called_a_conflict():
    g = torch.Generator().manual_seed(1)
    c, r = 2.0 + torch.rand(32, generator=g), torch.rand(256, generator=g)
    conflict, p, diff = cluster_conflict(c, r, 0.01, 999, 0.25, generator(1))
    assert conflict and p <= 0.01 and diff > 1.4


def test_the_margin_blocks_a_small_but_significant_shift():
    g = torch.Generator().manual_seed(2)
    r = torch.randn(512, generator=g).abs()
    c = torch.randn(600, generator=g).abs() + 0.15                                                       # +0.15 nats at n = 600 is significant ...
    open_, p_open, _ = cluster_conflict(c, r, 0.01, 999, 0.0, generator(2))
    shut, p_shut, _ = cluster_conflict(c, r, 0.01, 999, 0.25, generator(2))
    assert open_ and not shut and p_open == p_shut                                                       # ... but not practically significant; only the margin differs


def test_a_cluster_that_is_easier_than_the_reference_is_never_a_conflict():
    g = torch.Generator().manual_seed(3)
    r = 1.0 + torch.rand(256, generator=g)
    conflict, p, diff = cluster_conflict(r[:40] - 0.8, r, 0.01, 999, 0.0, generator(3))
    assert not conflict and p > 0.5 and diff < 0


def test_non_finite_cluster_losses_are_a_conflict():
    for bad in (float("inf"), float("nan")):
        conflict, p, diff = cluster_conflict(torch.tensor([bad, 1.0]), torch.rand(64), 0.01, 999, 0.25, generator(1))
        assert conflict and p == 0.0 and diff == math.inf


def test_the_p_value_is_an_integer_count_over_n_perm_plus_one_and_the_call_is_deterministic():
    g = torch.Generator().manual_seed(4)
    c, r = torch.randn(32, generator=g).abs(), torch.randn(128, generator=g).abs()
    a = cluster_conflict(c, r, 0.5, 99, 0.0, generator(3))
    assert a == cluster_conflict(c, r, 0.5, 99, 0.0, generator(3))
    assert abs(a[1] * 100 - round(a[1] * 100)) < 1e-9


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError):
        cluster_conflict(torch.empty(0), torch.rand(8), 0.01, 999, 0.0, generator(1))
    with pytest.raises(ValueError):
        cluster_conflict(torch.rand(8), torch.empty(0), 0.01, 999, 0.0, generator(1))
    with pytest.raises(ValueError):
        cluster_conflict(torch.rand(8), torch.rand(8), 0.001, 99, 0.0, generator(1))                      # 1/(1+99) > 0.001: could never reject
    with pytest.raises(ValueError):
        cluster_conflict(torch.rand(8), torch.tensor([1.0, float("inf")]), 0.01, 999, 0.0, generator(1))   # a reference must be finite
