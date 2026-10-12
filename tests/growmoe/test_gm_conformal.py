import math

import pytest
import torch

from growmoe.conformal import benjamini_hochberg, conformal_pvalues


def test_pvalue_formula_with_ties_counts_less_or_equal():
    cal = torch.tensor([0.0, 0.0, 0.0, 1.0])
    p = conformal_pvalues(cal, torch.tensor([0.0, 0.5, 2.0, -1.0]))
    # (1 + #{cal <= s}) / (n + 1)
    expected = torch.tensor([(1 + 3) / 5, (1 + 3) / 5, (1 + 4) / 5, (1 + 0) / 5])
    assert torch.allclose(p, expected, atol=1e-6)


def test_empty_calibration_set_is_an_error():
    with pytest.raises(ValueError):
        conformal_pvalues(torch.tensor([]), torch.tensor([0.0]))


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
def test_marginal_validity_under_exchangeability(alpha):
    g = torch.Generator().manual_seed(0)
    reps, n = 4000, 99
    hits = 0
    for _ in range(reps):
        cal = torch.randn(n, generator=g)
        s = torch.randn(1, generator=g)
        hits += int(conformal_pvalues(cal, s).item() <= alpha)
    freq = hits / reps
    sd = math.sqrt(alpha * (1 - alpha) / reps)
    # P(p <= alpha) = floor(alpha (n+1)) / (n+1) <= alpha; allow 3 sd of binomial noise
    assert freq <= alpha + 3 * sd
    assert freq >= alpha - 3 * sd - 1.0 / (n + 1)


def test_power_against_a_shifted_distribution():
    g = torch.Generator().manual_seed(1)
    cal = torch.randn(1000, generator=g)
    test = torch.randn(5000, generator=g) - 3.5
    power = (conformal_pvalues(cal, test) <= 0.05).float().mean().item()
    assert power > 0.9


def test_benjamini_hochberg_controls_fdr_on_mixed_batches():
    g = torch.Generator().manual_seed(2)
    q, reps, m = 0.1, 500, 100
    fdps, powers = [], []
    for _ in range(reps):
        cal = torch.randn(300, generator=g)
        is_alt = torch.rand(m, generator=g) < 0.2
        s = torch.randn(m, generator=g) - 3.5 * is_alt.float()
        rej = benjamini_hochberg(conformal_pvalues(cal, s), q)
        v = (rej & ~is_alt).sum().item()
        fdps.append(v / max(rej.sum().item(), 1))
        if is_alt.any():
            powers.append((rej & is_alt).sum().item() / is_alt.sum().item())
    assert sum(fdps) / reps <= q + 0.02
    assert sum(powers) / len(powers) > 0.7


def test_benjamini_hochberg_step_up_rule_by_hand():
    p = torch.tensor([0.001, 0.008, 0.039, 0.041, 0.9])
    # q=0.05, m=5: thresholds q*k/m = .01,.02,.03,.04,.05 -> largest k with p_(k)<=thr is k=2
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True, False, False, False]
    assert benjamini_hochberg(torch.tensor([]), 0.05).numel() == 0
    assert benjamini_hochberg(torch.tensor([0.9, 0.8]), 0.05).tolist() == [False, False]
