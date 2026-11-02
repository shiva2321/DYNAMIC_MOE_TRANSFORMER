import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fsmoe import detectors as DT
from fsmoe.fastslow import FastSlow, jsd
from fsmoe.invariance import InvarianceMap


def _clusters(n_per, k, d, seed, spread=0.15):
    g = torch.Generator().manual_seed(seed)
    centres = F.normalize(torch.randn(k, d, generator=g), dim=1)
    x = centres.repeat_interleave(n_per, 0) + spread * torch.randn(k * n_per, d, generator=g) / d ** 0.5 * 6
    return F.normalize(x, dim=1), torch.arange(k).repeat_interleave(n_per), centres


def test_auroc_edge_cases():
    assert DT.auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert DT.auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 0.0
    assert DT.auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    assert np.isnan(DT.auroc(np.array([]), np.array([1.0])))


def test_conformal_cut_rank_and_effective_rate():
    assert DT.cut_rank(40, 0.05) == 39 and abs(DT.effective_rate(40, 0.05) - 2 / 41) < 1e-12
    assert DT.conformal_cut(np.arange(10.0), 0.05) == float("inf") and DT.effective_rate(10, 0.05) == 0.0
    assert DT.conformal_cut(np.arange(100.0), 0.1) == 90.0           # rank ceil(101 * 0.9) = 91, the 91st smallest is 90


def test_conformal_flag_rate_matches_the_formula():
    rng = np.random.default_rng(0)
    flags = []
    for _ in range(4000):
        cal, new = rng.standard_normal(40), rng.standard_normal()
        flags.append(new > DT.conformal_cut(cal, 0.05))
    assert abs(np.mean(flags) - DT.effective_rate(40, 0.05)) < 0.012


@pytest.mark.parametrize("cls", [DT.Knn, DT.Maha, DT.Vmf])
def test_detectors_separate_a_far_cluster_and_count_their_cost(cls):
    X, y, centres = _clusters(60, 5, 32, 1)
    det = cls().fit(X[:200], y[:200])
    known = det.score(X[200:])
    g = torch.Generator().manual_seed(9)
    far = F.normalize(-centres.mean(0, keepdim=True) + 0.1 * torch.randn(40, 32, generator=g), dim=1)
    assert DT.auroc(known, det.score(far)) > 0.95 and det.macs > 0


def test_fastslow_learns_and_its_heads_have_the_right_shape():
    X, y, centres = _clusters(80, 4, 32, 2)
    fs = FastSlow(32, 6, hidden=32, epochs=20).fit(X[::2], y[::2])
    p = fs.p(X[1::2])
    assert (p.argmax(1) == y[1::2]).float().mean() > 0.9 and torch.allclose(p.sum(1), torch.ones(len(p)), atol=1e-4)
    assert fs.disagreement(X).min() >= 0 and fs.confidence(X).shape == (len(X),) and fs.density(X).min() >= 0
    assert fs.macs()["label"] == fs.macs()["fast"] + fs.macs()["slow"]
    same = torch.softmax(torch.randn(5, 7), 1)
    assert float(jsd(same, same).max()) < 1e-6 and float(jsd(same, same.flip(1)).min()) >= 0


def test_invariance_map_refuses_non_teaching_rows():
    X, y, _ = _clusters(10, 3, 16, 3)
    with pytest.raises(ValueError):
        InvarianceMap(16, 8).fit(X, y, {}, [True] * (len(X) - 1) + [False])


def test_invariance_map_pulls_the_same_skill_together_across_a_nuisance_direction():
    d, k, n = 24, 4, 40
    g = torch.Generator().manual_seed(4)
    signal = F.normalize(torch.randn(k, d, generator=g), dim=1)
    nuis = F.normalize(torch.randn(3, d, generator=g), dim=1)                  # three "wordings": a shift along their own direction
    fam = torch.randint(0, 3, (k * n,), generator=g)
    y = torch.arange(k).repeat_interleave(n)
    X = F.normalize(signal[y] + 1.2 * nuis[fam] + 0.1 * torch.randn(k * n, d, generator=g), dim=1)
    m = InvarianceMap(d, 8, epochs=150).fit(X, y, {0: 1, 1: 0, 2: 3, 3: 2}, [True] * len(X))
    Z = m(X)
    within = lambda A: float(((A @ A.T)[y[:, None] == y[None, :]]).mean())
    between = lambda A: float(((A @ A.T)[y[:, None] != y[None, :]]).mean())
    assert within(Z) > within(X) and within(Z) - between(Z) > within(X) - between(X)
    assert np.allclose(Z.norm(dim=1).numpy(), 1.0, atol=1e-5)
