"""M0: no-learning / light-learning memory baselines (kNN lookup, class means, streaming LDA).

They exist so every continual-learning table includes the simplest thing that cannot forget: remember the examples.
"""

import numpy as np
import pytest
import torch

from growmoe.config import CommonCfg
from growmoe.memory_baselines import ClassMean, KnnMemory, StreamingLDA
from growmoe.telemetry import EventLog
from growmoe.types import ContinualPool, Prediction, StepRecord

D, C = 16, 6
CFG = CommonCfg(d=D, n_classes=C, seed=3)


def _unit(n, d=D, seed=0):
    z = np.random.default_rng(seed).standard_normal((n, d))
    return torch.from_numpy((z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32))


def _blobs(n_per_class, seed=0, classes=(0, 1, 2), spread=0.35, centre_seed=123):
    """Blobs, one per class; the class centres depend only on ``centre_seed`` so train and test sets share them."""
    rng = np.random.default_rng(seed)
    centres = np.linalg.qr(np.random.default_rng(centre_seed).standard_normal((D, D)))[0][:, : len(classes)].T
    xs, ys = [], []
    for k, c in enumerate(classes):
        z = centres[k][None, :] + spread * rng.standard_normal((n_per_class, D)) / np.sqrt(D)
        xs.append(z / np.linalg.norm(z, axis=1, keepdims=True))
        ys.append(np.full(n_per_class, c))
    x = torch.from_numpy(np.concatenate(xs).astype(np.float32))
    y = torch.from_numpy(np.concatenate(ys).astype(np.int64))
    perm = torch.from_numpy(rng.permutation(len(y)))
    return x[perm], y[perm]


MAKERS = {
    "knn": lambda: KnnMemory(CFG, k=3),
    "knn_capped": lambda: KnnMemory(CFG, k=3, capacity=40),
    "ncm": lambda: ClassMean(CFG),
    "slda": lambda: StreamingLDA(CFG),
}


@pytest.mark.parametrize("name", sorted(MAKERS))
def test_every_memory_baseline_satisfies_the_pool_protocol(name):
    pool = MAKERS[name]()
    assert isinstance(pool, ContinualPool) and isinstance(pool.events, EventLog) and pool.name
    x, y = _blobs(20)
    rec = pool.observe(x, y)
    assert isinstance(rec, StepRecord) and rec.assigned.shape == (x.shape[0],)
    pred = pool.predict(x)
    assert isinstance(pred, Prediction)
    assert pred.probs.shape == (x.shape[0], C)
    assert torch.allclose(pred.probs.sum(dim=1), torch.ones(x.shape[0]), atol=1e-5)
    s = pool.summary()
    assert s["n_params"] == 0 and s["n_experts"] == 1


@pytest.mark.parametrize("name", sorted(MAKERS))
def test_unlabelled_data_teaches_nothing_and_an_empty_pool_answers_uniformly(name):
    pool = MAKERS[name]()
    empty = pool.predict(_unit(4))
    assert torch.allclose(empty.probs, torch.full((4, C), 1.0 / C), atol=1e-6)
    pool.observe(_unit(10), None)
    assert torch.allclose(pool.predict(_unit(4)).probs, torch.full((4, C), 1.0 / C), atol=1e-6)


def test_knn_stores_only_labelled_examples():
    pool = KnnMemory(CFG)
    pool.observe(_unit(10), None)
    assert pool.stored_vectors() == 0
    pool.observe(_unit(10), torch.zeros(10, dtype=torch.long))
    assert pool.stored_vectors() == 10


@pytest.mark.parametrize("name", ["knn", "ncm", "slda"])
def test_separable_blobs_are_classified_and_unseen_classes_get_no_probability(name):
    pool = MAKERS[name]()
    x, y = _blobs(60, seed=1)
    pool.observe(x, y)
    xt, yt = _blobs(40, seed=2)
    pred = pool.predict(xt)
    assert float((pred.probs.argmax(1) == yt).float().mean()) >= 0.95
    assert float(pred.probs[:, 3:].sum()) == pytest.approx(0.0, abs=1e-6)         # classes 3..5 were never shown


def test_knn_returns_the_stored_label_for_a_stored_vector_and_votes_by_majority():
    pool = KnnMemory(CFG, k=1)
    x, y = _unit(30, seed=5), torch.arange(30) % C
    pool.observe(x, y)
    assert torch.equal(pool.predict(x).probs.argmax(1), y)
    pool3 = KnnMemory(CFG, k=3)
    base = _unit(1, seed=9)
    near = torch.nn.functional.normalize(base + 1e-3 * _unit(3, seed=10), dim=1)
    far = _unit(5, seed=11)
    pool3.observe(torch.cat([near, far]), torch.tensor([2, 2, 4, 0, 0, 0, 0, 0]))
    assert int(pool3.predict(base).probs.argmax(1)) == 2


def test_knn_capacity_is_respected_deterministic_and_uniform_over_time():
    x = _unit(2000, seed=7)
    y = torch.zeros(2000, dtype=torch.long)
    a, b, c = KnnMemory(CFG, capacity=200), KnnMemory(CFG, capacity=200), KnnMemory(CommonCfg(d=D, n_classes=C, seed=4), capacity=200)
    for pool in (a, b, c):
        for i in range(0, 2000, 50):
            pool.observe(x[i:i + 50], y[i:i + 50])
    assert a.stored_vectors() == b.stored_vectors() == 200
    assert torch.equal(a.X, b.X)                                                       # same seed, same memory
    assert not torch.equal(a.X, c.X)                                                   # different seed, different memory
    first_half = int((a.X @ x[:1000].T).max(dim=1).values.gt(0.9999).sum())           # kept samples that came from the first half
    assert 0.35 <= first_half / 200 <= 0.65                                            # reservoir sampling is uniform in time


def test_streaming_lda_with_a_single_seen_class_gives_it_probability_one():
    pool = StreamingLDA(CFG)
    x, _ = _blobs(50, seed=3, classes=(0, 1))
    pool.observe(x, torch.zeros(x.shape[0], dtype=torch.long))
    assert torch.allclose(pool.predict(x).probs[:, 0], torch.ones(x.shape[0]), atol=1e-5)


def test_streaming_lda_tracks_scikit_learn_lda_on_overlapping_classes():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    x, y = _blobs(300, seed=4, spread=2.5)                                            # heavy overlap: not trivially separable
    xt, yt = _blobs(300, seed=5, spread=2.5)
    pool = StreamingLDA(CFG)
    for i in range(0, x.shape[0], 50):
        pool.observe(x[i:i + 50], y[i:i + 50])
    ours = float((pool.predict(xt).probs.argmax(1) == yt).float().mean())
    ref = float(LinearDiscriminantAnalysis().fit(x.numpy(), y.numpy()).score(xt.numpy(), yt.numpy()))
    assert ref > 0.5                                                                   # the task is learnable but not trivial
    assert ours >= ref - 0.05
