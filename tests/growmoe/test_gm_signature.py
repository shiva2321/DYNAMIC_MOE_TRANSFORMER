import math

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from growmoe.signature import VmfSignature
from growmoe.vmf import log_vmf_norm, sample_vmf

D = 128


def _mu(seed):
    g = np.random.default_rng(seed)
    return torch.nn.functional.normalize(torch.from_numpy(g.standard_normal(D)).float(), dim=0)


def _cluster(seed, kappa, n, mu=None):
    mu = _mu(seed) if mu is None else mu
    return sample_vmf(mu, kappa, n, np.random.default_rng(seed + 1000))


def test_tight_cluster_selects_larger_kernel_concentration_than_diffuse():
    tight, diffuse = VmfSignature(D), VmfSignature(D)
    tight.update(_cluster(0, 250.0, 200))
    diffuse.update(_cluster(0, 10.0, 200))
    tight.fit()
    diffuse.fit()
    assert tight.kappa in tight.kappa_grid and diffuse.kappa in diffuse.kappa_grid
    assert tight.kappa > diffuse.kappa


def test_scores_separate_two_clusters_at_high_dimension():
    # kappa = 100 (mean resultant length 0.55). The one-class score is weaker on broad clusters: measured
    # AUROC at d = 128 with 200 prototypes is 0.64 / 0.88 / 0.9997 for kappa = 10 / 30 / 100.
    sig = VmfSignature(D)
    sig.update(_cluster(1, 100.0, 200))
    a = _cluster(1, 100.0, 300, mu=_mu(1))         # same cluster, fresh draws
    b = _cluster(2, 100.0, 300)                    # different cluster
    scores = torch.cat([sig.score(a), sig.score(b)]).numpy()
    labels = np.r_[np.ones(300), np.zeros(300)]
    assert roc_auc_score(labels, scores) > 0.99


def test_conformal_pvalues_on_fresh_same_distribution_are_calibrated():
    # marginal validity, averaged over independent reservoirs: one calibration set's coverage is
    # Beta-distributed and moves by about +-0.013 at n = 256, so a single seed would test luck
    mu = _mu(3)
    hits, means = [], []
    for s in range(10):
        sig = VmfSignature(D, max_protos=256, seed=s)
        sig.update(sample_vmf(mu, 60.0, 256, np.random.default_rng(100 + s)))
        p = sig.pvalue(sample_vmf(mu, 60.0, 1000, np.random.default_rng(200 + s)))
        hits.append((p <= 0.05).float().mean().item())
        means.append(p.mean().item())
    assert 0.03 <= float(np.mean(hits)) <= 0.065      # valid (<= alpha + 0.02) and not wastefully conservative
    assert abs(float(np.mean(means)) - 0.5) < 0.03


def test_pvalue_is_monotone_in_score():
    sig = VmfSignature(D)
    sig.update(_cluster(4, 50.0, 120))
    x = torch.cat([_cluster(4, 50.0, 40, mu=_mu(4)), _cluster(5, 50.0, 40)])
    s, p = sig.score(x), sig.pvalue(x)
    order = torch.argsort(s)
    assert (p[order][1:] >= p[order][:-1] - 1e-7).all()


def test_logpdf_adds_the_normaliser():
    sig = VmfSignature(D)
    sig.update(_cluster(6, 80.0, 100))
    x = _cluster(6, 80.0, 5, mu=_mu(6))
    assert torch.allclose(sig.logpdf(x), sig.score(x) + log_vmf_norm(D, sig.kappa), atol=1e-4)


def test_reservoir_is_bounded_and_uniform_over_the_stream():
    stream = _cluster(7, 30.0, 10000)
    ranks = []
    for seed in range(5):
        sig = VmfSignature(D, max_protos=256, seed=seed)
        for i in range(0, 10000, 64):
            sig.update(stream[i:i + 64])
        assert sig.n == 256 and sig.seen == 10000
        ranks.append(sig.arrival_ranks())
    ranks = torch.cat(ranks)
    assert abs(ranks.float().mean().item() - 5000.0) < 300          # a single seed has sd 178, five pooled have 80
    quartile = torch.bucketize(ranks, torch.tensor([2500, 5000, 7500]), right=True)
    counts = torch.bincount(quartile, minlength=4)
    assert (counts - 320).abs().max().item() < 60                   # 320 +- 3.9 sd; catches symmetric distortions


def test_kappa_choice_is_stable_across_stream_orders():
    pts = _cluster(8, 70.0, 1000)
    chosen = []
    for seed in (0, 1):
        perm = torch.from_numpy(np.random.default_rng(seed).permutation(1000))
        sig = VmfSignature(D, max_protos=256, seed=seed)
        sig.update(pts[perm])
        sig.fit()
        chosen.append(sig.kappa_grid.index(sig.kappa))
    assert abs(chosen[0] - chosen[1]) <= 1


def test_updates_normalise_inputs_and_validate_shape():
    sig = VmfSignature(D)
    x = _cluster(9, 40.0, 50)
    sig.update(3.0 * x)
    a = sig.score(x)
    sig2 = VmfSignature(D)
    sig2.update(x)
    assert torch.allclose(a, sig2.score(x), atol=1e-4)
    with pytest.raises(ValueError):
        sig.update(torch.randn(4, D + 1))


def test_too_few_prototypes_is_an_error():
    sig = VmfSignature(D)
    sig.update(_cluster(10, 20.0, 1))
    with pytest.raises(ValueError):
        sig.fit()
    with pytest.raises(ValueError):
        sig.pvalue(_cluster(10, 20.0, 3))


def test_merge_from_adds_prototypes_within_capacity():
    a, b = VmfSignature(D, max_protos=64), VmfSignature(D, max_protos=64)
    a.update(_cluster(11, 40.0, 40))
    b.update(_cluster(12, 40.0, 40))
    a.merge_from(b)
    assert 40 < a.n <= 64


def test_nearest_neighbour_similarity_and_mean_direction():
    sig = VmfSignature(D)
    x = _cluster(13, 60.0, 30)
    sig.update(x)
    nn = sig.nn_similarity()
    sims = x @ x.T
    sims.fill_diagonal_(-2.0)
    assert torch.allclose(nn, sims.max(dim=1).values, atol=1e-5)
    m = sig.mean_direction()
    assert abs(m.norm().item() - 1.0) < 1e-5 and (m @ _mu(13)).item() > 0.3


# --------------------------------------------------------------------------------------------- affinity
def test_affinity_is_near_one_for_the_same_distribution_and_near_zero_for_unrelated_directions():
    sig = VmfSignature(D)
    sig.update(_cluster(20, 60.0, 256))
    same = _cluster(20, 60.0, 300, mu=_mu(20))
    rand = torch.nn.functional.normalize(torch.randn(300, D, generator=torch.Generator().manual_seed(0)), dim=1)
    unrelated = _cluster(21, 60.0, 300)
    assert 0.85 <= sig.affinity(same) <= 1.0
    assert 0.0 <= sig.affinity(rand) <= 0.05
    assert sig.affinity(unrelated) < 0.1


def test_affinity_falls_monotonically_as_the_new_cluster_rotates_away_from_the_signature():
    sig = VmfSignature(D)
    mu = _mu(22)
    sig.update(_cluster(22, 60.0, 256, mu=mu))
    other = torch.nn.functional.normalize(torch.randn(D, generator=torch.Generator().manual_seed(1)), dim=0)
    other = torch.nn.functional.normalize(other - (other @ mu) * mu, dim=0)          # unit vector orthogonal to mu
    values = []
    for cos in (1.0, 0.95, 0.85, 0.7, 0.5, 0.2):
        centre = cos * mu + math.sqrt(1 - cos ** 2) * other
        values.append(sig.affinity(_cluster(23, 60.0, 300, mu=centre)))
    assert all(a > b for a, b in zip(values, values[1:]))
    assert values[0] > 0.8 and values[-1] < 0.35


def test_affinity_is_deterministic_in_its_seed_and_needs_two_prototypes():
    sig = VmfSignature(D)
    sig.update(_cluster(24, 60.0, 100))
    x = _cluster(25, 60.0, 50)
    assert sig.affinity(x, seed=3) == sig.affinity(x, seed=3)
    one = VmfSignature(D)
    one.update(_cluster(24, 60.0, 1))
    with pytest.raises(ValueError):
        one.affinity(x)
