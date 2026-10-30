"""The real-data domains (unimoe/domains.py): local datasets turned into labelled unit vectors with regimes, never downloaded."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from growmoe import arith
from unimoe import domains as DM

HUB = Path.home() / ".cache" / "huggingface" / "hub"
needs_data = pytest.mark.skipif(not (HUB / "datasets--banking77").exists(), reason="local Hugging Face cache not present")


class FakeFE:
    """A cheap stand-in for the sphere frontend: hashed bag of words to 128 numbers, no model."""
    d = 128

    def text(self, texts):
        out = np.zeros((len(texts), self.d), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in str(t).lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest()[:6], 16) % self.d] += 1.0
        out += 1e-3
        return torch.from_numpy(out / np.linalg.norm(out, axis=1, keepdims=True))

    def digits(self, codes):
        x = np.zeros((len(codes), self.d), dtype=np.float32)
        x[:, :codes.shape[1]] = codes
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return torch.from_numpy(np.where(n > 0, x / np.maximum(n, 1e-12), x))


def test_group_classes_partitions_the_classes_and_is_deterministic():
    rng = np.random.default_rng(0)
    centers = np.eye(6)[[0, 0, 1, 1, 2, 2]] + 0.01 * rng.standard_normal((6, 6))
    v = torch.from_numpy(centers / np.linalg.norm(centers, axis=1, keepdims=True)).float()
    g = DM.group_classes(v, 3, seed=1)
    assert sorted(c for grp in g for c in grp) == list(range(6)) and all(len(grp) >= 1 for grp in g)
    assert g == DM.group_classes(v, 3, seed=1)
    assert sorted(map(tuple, g)) == [(0, 1), (2, 3), (4, 5)]                                       # the clusters that are there are found
    assert DM.group_classes(v, 1, seed=0) == [[0, 1, 2, 3, 4, 5]]
    assert sorted(c for grp in DM.group_classes(v, 9, seed=0) for c in grp) == list(range(6))      # more groups than classes: capped, still a partition


def test_stratified_cap_keeps_every_class_and_never_exceeds_the_cap():
    y = np.array([0] * 100 + [1] * 10 + [2] * 3)
    idx = DM.stratified(y, 30, np.random.default_rng(0))
    assert len(idx) <= 30 and set(y[idx]) == {0, 1, 2} and len(set(idx.tolist())) == len(idx)
    assert len(DM.stratified(y, 1000, np.random.default_rng(0))) == 113                           # a cap above the data returns all of it


@needs_data
@pytest.mark.parametrize("name,n_classes", [("banking", 77), ("agnews", 4), ("newsgroups", 20), ("emotion", 6), ("sst2", 2), ("svamp", 4)])
def test_every_text_domain_loads_with_regimes_that_partition_its_classes(name, n_classes):
    d = DM.load_domain(name, FakeFE(), seed=0, cap_train=600, cap_test=200)
    assert d.kind == "text" and d.n_classes == n_classes and len(d.class_names) == n_classes
    assert d.x_train.shape[1] == 128 and torch.allclose(d.x_train.norm(dim=1), torch.ones(len(d.x_train)), atol=1e-4)
    assert d.y_train.min() >= 0 and d.y_train.max() < n_classes and d.y_test.max() < n_classes
    assert 0 < len(d.x_train) <= 600 and 0 < len(d.x_test) <= 200
    assert set(d.r_train.tolist()) == set(d.r_test.tolist()) == set(range(len(d.regime_names)))
    for r in range(len(d.regime_names)):                                                           # a class belongs to exactly one regime
        assert set(d.y_train[d.r_train == r].tolist()) == set(d.y_test[d.r_test == r].tolist()) or len(d.y_test[d.r_test == r]) > 0
    cls_regime = {}
    for y, r in zip(d.y_train.tolist(), d.r_train.tolist()):
        assert cls_regime.setdefault(y, r) == r
    assert not (set(d.meta["texts_train"]) & set(d.meta["texts_test"]))                           # no test text was trained on


@needs_data
def test_a_small_domain_with_a_cap_above_its_size_gives_what_it_has_without_duplicates():
    d = DM.load_domain("svamp", FakeFE(), seed=0, cap_train=4000, cap_test=1500)
    assert len(d.x_train) <= 700 and len(d.x_test) <= 300
    assert len(set(d.meta["texts_train"])) == len(d.meta["texts_train"])
    assert sorted(d.class_names) == ["Addition", "Common-Division", "Multiplication", "Subtraction"]   # the dataset's one typo'd label is merged


@needs_data
def test_the_arithmetic_domain_has_one_regime_per_operator_with_shared_digit_labels_and_held_out_test_problems():
    d = DM.load_domain("arith", FakeFE(), seed=0, cap_train=1200, cap_test=600)
    assert d.kind == "digits" and d.regime_names == list(arith.OPS) and d.n_classes == arith.N_CLASS
    assert d.x_train.shape[1] == 128 and d.y_train.max() < arith.N_CLASS
    assert set(d.r_train.tolist()) == set(d.r_test.tolist()) == set(range(6))
    tr, te = d.meta["problems_train"], d.meta["problems_test"]
    assert not arith.heldout(tr["op"], tr["a"], tr["b"]).any() and arith.heldout(te["op"], te["a"], te["b"]).all()    # "unseen" is exact
    assert len(d.meta["problem_id_test"]) == len(d.x_test)
    assert DM.load_domain("arith", FakeFE(), seed=0, cap_train=1200, cap_test=600).y_train.tolist() == d.y_train.tolist()


@needs_data
def test_out_of_distribution_sets_are_unit_vectors_of_text_the_pool_never_trains_on():
    ood = DM.ood_sets(FakeFE(), seed=0, n=50)
    assert set(ood) == {"stories", "wiki", "science_questions"}
    for name, x in ood.items():
        assert x.shape == (50, 128) and torch.allclose(x.norm(dim=1), torch.ones(50), atol=1e-4), name


def test_an_unknown_domain_is_refused_loudly():
    with pytest.raises(ValueError):
        DM.load_domain("nonexistent", FakeFE(), seed=0)
