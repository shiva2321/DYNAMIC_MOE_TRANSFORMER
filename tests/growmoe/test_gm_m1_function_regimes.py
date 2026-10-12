"""M1: the function-regime stream. Each regime lives in its own region of the sphere and is labelled by its own hidden random
teacher applied to the within-region deviation, so memorising neighbours fails but every regime is learnable.

Thresholds below come from the diligence pilot (research/moe_diligence_2026-10/spike_lookup_fails_out.txt: 5-NN about 22 %, a small
trained network 45 % at 300 and 59 % at 1000 samples, chance 12.5 %) and leave a wide margin.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from growmoe.experts import MLPExpert
from growmoe.streams_m1 import function_regimes

N, D, C = 6, 128, 8


@pytest.fixture(scope="module")
def stream():
    return function_regimes(0, n_regimes=N, d=D, n_train=800, n_test=400)


def _knn_accuracy(x_tr, y_tr, x_te, y_te, k=5):
    idx = (x_te @ x_tr.T).topk(k, dim=1).indices
    votes = torch.zeros(len(x_te), C)
    votes.scatter_add_(1, y_tr[idx], torch.ones(idx.shape))
    return float((votes.argmax(1) == y_te).float().mean())


def _train_net(x, y, seed=0, steps=900):
    net = MLPExpert(D, 64, 1, C, seed=seed)
    net.mark_seen(torch.arange(C))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    gen = np.random.default_rng([seed, 7])
    for _ in range(steps):
        b = torch.from_numpy(gen.integers(0, len(x), 64))
        net.train_step(x[b], y[b], opt)
    net.eval()
    return net


def _accuracy(net, x, y):
    with torch.no_grad():
        return float((net(x).argmax(1) == y).float().mean())


@pytest.fixture(scope="module")
def net0(stream):
    seg = stream.segments[0]
    return _train_net(seg.x, seg.y)


# ------------------------------------------------------------------------------------------------ layout
def test_layout_follows_the_preregistration(stream):
    assert stream.d == D and stream.n_classes == C
    assert sorted(stream.regimes) == list(range(N))
    schedule = [seg.regime for seg in stream.segments]
    assert schedule == list(range(N)) + [0, N // 2]                                      # every regime once, then revisits of 0 and n/2
    assert stream.meta["schedule"] == schedule
    assert [seg.first_visit for seg in stream.segments] == [True] * N + [False] * 2
    assert all(len(seg.y) == 800 and seg.x.shape == (800, D) for seg in stream.segments)
    assert all(reg.x_te.shape == (400, D) and len(reg.y_te) == 400 for reg in stream.regimes.values())


def test_inputs_are_unit_vectors_and_labels_are_valid_integers(stream):
    for seg in stream.segments:
        assert seg.x.dtype == torch.float32 and seg.y.dtype == torch.int64
        assert torch.allclose(seg.x.norm(dim=1), torch.ones(len(seg.x)), atol=1e-5)
        assert int(seg.y.min()) >= 0 and int(seg.y.max()) < C
    for reg in stream.regimes.values():
        assert torch.allclose(reg.x_te.norm(dim=1), torch.ones(len(reg.x_te)), atol=1e-5)
        assert int(reg.y_te.min()) >= 0 and int(reg.y_te.max()) < C


def test_the_stream_is_deterministic_in_the_seed_and_changes_with_it():
    a = function_regimes(3, n_regimes=3, n_train=64, n_test=32)
    b = function_regimes(3, n_regimes=3, n_train=64, n_test=32)
    c = function_regimes(4, n_regimes=3, n_train=64, n_test=32)
    for sa, sb in zip(a.segments, b.segments):
        assert torch.equal(sa.x, sb.x) and torch.equal(sa.y, sb.y)
    assert not torch.equal(a.segments[0].x, c.segments[0].x)
    assert not torch.equal(a.regimes[0].y_te, c.regimes[0].y_te)


def test_too_few_regimes_are_refused():
    with pytest.raises(ValueError):
        function_regimes(0, n_regimes=1)


def test_labels_are_roughly_balanced_in_every_regime(stream):
    for seg in stream.segments:
        share = torch.bincount(seg.y, minlength=C).float() / len(seg.y)
        assert float(share.min()) >= 0.07 and float(share.max()) <= 0.20


# ------------------------------------------------------------------------------------------------ regions
def test_regimes_live_in_separate_regions_of_the_sphere(stream):
    centres = torch.as_tensor(stream.meta["centres"], dtype=torch.float32)
    assert centres.shape == (N, D)
    for r, reg in stream.regimes.items():
        assert float(((reg.x_te @ centres.T).argmax(1) == r).float().mean()) >= 0.99


def test_a_revisit_brings_fresh_samples_from_the_same_region(stream):
    first, again = stream.segments[0], stream.segments[N]
    assert first.regime == again.regime == 0 and not again.first_visit
    assert not torch.equal(first.x, again.x)
    assert float(F.normalize(first.x.mean(0), dim=0) @ F.normalize(again.x.mean(0), dim=0)) > 0.95
    assert not any(bool((first.x == row).all(dim=1).any()) for row in again.x[:20])      # no sample is repeated


# ------------------------------------------------------------------------------------------------ the point of the stream
def test_a_lookup_cannot_solve_a_regime_but_a_small_network_can(stream, net0):
    seg, reg = stream.segments[0], stream.regimes[0]
    knn = _knn_accuracy(seg.x, seg.y, reg.x_te, reg.y_te)
    mlp = _accuracy(net0, reg.x_te, reg.y_te)
    assert knn < 0.32, f"5-NN reached {knn:.3f}: neighbours share labels, the stream tests memorisation"
    assert mlp > 0.40, f"the network reached {mlp:.3f}: the regime is not learnable"
    assert mlp - knn > 0.10


def test_knowledge_of_one_regime_does_not_transfer_to_another(stream, net0):
    own = _accuracy(net0, stream.regimes[0].x_te, stream.regimes[0].y_te)
    other = [_accuracy(net0, stream.regimes[r].x_te, stream.regimes[r].y_te) for r in range(1, N)]
    assert own - max(other) > 0.15
    assert float(np.mean(other)) < 0.25
