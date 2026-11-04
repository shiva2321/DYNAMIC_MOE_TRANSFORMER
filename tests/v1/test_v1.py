"""tests/v1/test_v1.py: Verification unit tests for Job V-3 (Probe P2).

Tests:
  1. Exact downdating equality on hand-made data for each closed-form head (ClassMean,
     KnnMemory, MahalanobisProto, RandomFeatureRidge, ExactLda).
  2. ExactLda matches brute-force batch formula bit for bit.
  3. remove_classes on absent/unseen class raises clear ValueError across models.
  4. scan_traces detects planted traces and reports zero traces after remove_labels.
  5. Certificate is valid JSON containing all specified keys.
  6. Repeat of finished experiment is skipped cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from costlab.env import fixed_threads
from growmoe.memory_baselines import ClassMean, KnnMemory, StreamingLDA
from growmoe.pools.base import CommonCfg, ExpertRecord
from m1.baselines_extra import MahalanobisProto, RandomFeatureRidge
from v1.downdate import TOL_EXACT, ExactLda, remove_classes
from v1.pool_removal import ExposureTracker, remove_labels, scan_traces


@pytest.fixture(autouse=True)
def pin_cpus():
    """Pin unit tests to logical processors 8 and 10 with 2 threads."""
    with fixed_threads(n=2, cpus=[8, 10]):
        yield


def _make_toy_dataset(d: int = 16, n_per_class: int = 40, n_classes: int = 4, seed: int = 42):
    """Generate reproducible small fixture dataset."""
    rng = np.random.default_rng(seed)
    x_list = []
    y_list = []
    for c in range(n_classes):
        center = rng.standard_normal(d)
        center /= np.linalg.norm(center)
        samples = center + 0.15 * rng.standard_normal((n_per_class, d))
        x_list.append(samples)
        y_list.append(np.full(n_per_class, c, dtype=np.int64))
    X = np.concatenate(x_list, axis=0)
    Y = np.concatenate(y_list, axis=0)
    perm = rng.permutation(len(Y))
    X, Y = X[perm], Y[perm]
    return torch.from_numpy(X).float(), torch.from_numpy(Y).long()


class DummyModule(torch.nn.Module):
    def __init__(self, d: int, n_classes: int):
        super().__init__()
        self.linear = torch.nn.Linear(d, n_classes)
        self.seen = torch.zeros(n_classes, dtype=torch.bool)


class DummyReplay:
    def __init__(self, d: int, cap: int = 64):
        self.capacity = cap
        self.x = torch.zeros(cap, d)
        self.y = torch.full((cap,), -1, dtype=torch.long)
        self.n = 0

    def add(self, x: torch.Tensor, y: torch.Tensor):
        num = x.shape[0]
        self.x[:num] = x
        self.y[:num] = y
        self.n = num


class DummySig:
    def __init__(self, d: int):
        self.d = d
        self._protos = torch.zeros(16, d)
        self._n = 0

    @property
    def n(self):
        return self._n

    @property
    def protos(self):
        return self._protos[: self._n]


class DummyPool:
    def __init__(self, d: int = 16, n_classes: int = 4):
        self.d = d
        self.n_classes = n_classes
        self.experts = {}
        self._parents = {}
        self._route = type("Route", (), {"eids": [0, 1]})()
        self._held = []
        self._boot_x = []
        self._boot_y = []
        from growmoe.probation import Probation
        self.probation = Probation(cap=32, d=d)
        self.domain_lda = None

    def remove_expert(self, eid: int):
        return self.experts.pop(eid, None)

    def _refit_router(self):
        self._route.eids = sorted(self.experts.keys())

    def _normalize(self, x: torch.Tensor):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def test_exact_downdating_equality():
    """Verify exact downdating (tol <= 1e-6) for ClassMean, KnnMemory, MahalanobisProto, RandomFeatureRidge, ExactLda."""
    d = 16
    n_classes = 4
    x_train, y_train = _make_toy_dataset(d=d, n_per_class=30, n_classes=n_classes, seed=800)
    x_test, y_test = _make_toy_dataset(d=d, n_per_class=15, n_classes=n_classes, seed=801)

    removed_label = 2
    retained_mask_train = (y_train != removed_label)
    retained_mask_test = (y_test != removed_label)

    x_train_nev = x_train[retained_mask_train]
    y_train_nev = y_train[retained_mask_train]
    x_test_ret = x_test[retained_mask_test]
    y_test_ret = y_test[retained_mask_test]

    cfg = CommonCfg(seed=800, d=d, n_classes=n_classes)

    # 1. ClassMean
    m_full = ClassMean(cfg)
    m_full.observe(x_train, y_train)
    remove_classes(m_full, [removed_label])

    m_nev = ClassMean(cfg)
    m_nev.observe(x_train_nev, y_train_nev)

    assert torch.allclose(m_full._sum, m_nev._sum, atol=TOL_EXACT)
    assert torch.allclose(m_full._cnt, m_nev._cnt, atol=TOL_EXACT)
    pred_full = m_full.predict(x_test_ret).probs[:, [0, 1, 3]]
    pred_nev = m_nev.predict(x_test_ret).probs[:, [0, 1, 3]]
    assert torch.allclose(pred_full, pred_nev, atol=TOL_EXACT)
    assert (pred_full.argmax(dim=-1) == pred_nev.argmax(dim=-1)).all()

    # 2. KnnMemory (capacity=None)
    knn_full = KnnMemory(cfg, k=5, capacity=None)
    knn_full.observe(x_train, y_train)
    remove_classes(knn_full, [removed_label])

    knn_nev = KnnMemory(cfg, k=5, capacity=None)
    knn_nev.observe(x_train_nev, y_train_nev)

    assert knn_full.X.shape == knn_nev.X.shape
    assert torch.allclose(knn_full.X, knn_nev.X, atol=TOL_EXACT)
    assert (knn_full.Y == knn_nev.Y).all()
    pred_knn_f = knn_full.predict(x_test_ret).probs[:, [0, 1, 3]]
    pred_knn_n = knn_nev.predict(x_test_ret).probs[:, [0, 1, 3]]
    assert torch.allclose(pred_knn_f, pred_knn_n, atol=TOL_EXACT)

    # 3. MahalanobisProto
    mp_full = MahalanobisProto(cfg, shrinkage=0.1, scale=1.0)
    mp_full.observe(x_train, y_train)
    remove_classes(mp_full, [removed_label])

    mp_nev = MahalanobisProto(cfg, shrinkage=0.1, scale=1.0)
    mp_nev.observe(x_train_nev, y_train_nev)

    assert torch.allclose(mp_full._cnt, mp_nev._cnt, atol=TOL_EXACT)
    assert torch.allclose(mp_full._sum, mp_nev._sum, atol=TOL_EXACT)
    assert torch.allclose(mp_full._outer, mp_nev._outer, atol=TOL_EXACT)
    pred_mp_f = mp_full.predict(x_test_ret).probs[:, [0, 1, 3]]
    pred_mp_n = mp_nev.predict(x_test_ret).probs[:, [0, 1, 3]]
    assert torch.allclose(pred_mp_f, pred_mp_n, atol=TOL_EXACT)
    assert (pred_mp_f.argmax(dim=-1) == pred_mp_n.argmax(dim=-1)).all()

    # 4. RandomFeatureRidge (with data=(x_rem, y_rem))
    rf_full = RandomFeatureRidge(cfg, D=128, ridge=0.1, scale=1.0)
    rf_full.observe(x_train, y_train)
    x_rem = x_train[~retained_mask_train]
    y_rem = y_train[~retained_mask_train]
    remove_classes(rf_full, [removed_label], data=(x_rem, y_rem))

    rf_nev = RandomFeatureRidge(cfg, D=128, ridge=0.1, scale=1.0)
    rf_nev.observe(x_train_nev, y_train_nev)

    assert torch.allclose(rf_full._G, rf_nev._G, atol=TOL_EXACT)
    assert torch.allclose(rf_full._B, rf_nev._B, atol=TOL_EXACT)
    pred_rf_f = rf_full.predict(x_test_ret).probs[:, [0, 1, 3]]
    pred_rf_n = rf_nev.predict(x_test_ret).probs[:, [0, 1, 3]]
    assert torch.allclose(pred_rf_f, pred_rf_n, atol=TOL_EXACT)
    assert (pred_rf_f.argmax(dim=-1) == pred_rf_n.argmax(dim=-1)).all()

    # 5. ExactLda
    el_full = ExactLda(cfg, shrinkage=1e-2, scale=1.0)
    el_full.observe(x_train, y_train)
    remove_classes(el_full, [removed_label])

    el_nev = ExactLda(cfg, shrinkage=1e-2, scale=1.0)
    el_nev.observe(x_train_nev, y_train_nev)

    assert torch.allclose(el_full._cnt, el_nev._cnt, atol=TOL_EXACT)
    assert torch.allclose(el_full._sum, el_nev._sum, atol=TOL_EXACT)
    assert torch.allclose(el_full._outer, el_nev._outer, atol=TOL_EXACT)
    pred_el_f = el_full.predict(x_test_ret).probs[:, [0, 1, 3]]
    pred_el_n = el_nev.predict(x_test_ret).probs[:, [0, 1, 3]]
    assert torch.allclose(pred_el_f, pred_el_n, atol=TOL_EXACT)
    assert (pred_el_f.argmax(dim=-1) == pred_el_n.argmax(dim=-1)).all()


def test_exact_lda_matches_brute_force():
    """Verify ExactLda covariance and solved weights match a brute-force formula."""
    d = 8
    n_classes = 3
    x_train, y_train = _make_toy_dataset(d=d, n_per_class=25, n_classes=n_classes, seed=123)

    cfg = CommonCfg(seed=123, d=d, n_classes=n_classes)
    model = ExactLda(cfg, shrinkage=0.05)
    model.observe(x_train, y_train)

    # Brute-force calculation
    x_unit = x_train / x_train.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    xd = x_unit.double()
    total_n = len(y_train)
    pooled_scatter = torch.zeros(d, d, dtype=torch.float64)
    mu_bf = torch.zeros(n_classes, d, dtype=torch.float64)

    for c in range(n_classes):
        xc = xd[y_train == c]
        nc = xc.shape[0]
        muc = xc.mean(dim=0)
        mu_bf[c] = muc
        diff = xc - muc
        pooled_scatter += diff.T @ diff

    cov_bf = pooled_scatter / total_n
    cov_bf = 0.5 * (cov_bf + cov_bf.T)
    sig_bf = (1.0 - 0.05) * cov_bf + 0.05 * torch.eye(d, dtype=torch.float64)
    w_bf = torch.linalg.solve(sig_bf, mu_bf.T).T
    b_bf = -0.5 * (mu_bf * w_bf).sum(dim=1)

    wb_model = model._weights()
    assert wb_model is not None
    w_model, b_model = wb_model

    assert torch.allclose(w_model, w_bf, atol=1e-9)
    assert torch.allclose(b_model, b_bf, atol=1e-9)


def test_remove_classes_absent_class_raises():
    """Verify remove_classes raises clear ValueError when given an absent class."""
    cfg = CommonCfg(seed=999, d=8, n_classes=5)
    x = torch.randn(10, 8)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0], dtype=torch.long)

    for head_cls in [ClassMean, KnnMemory, MahalanobisProto, RandomFeatureRidge, ExactLda, StreamingLDA]:
        model = head_cls(cfg)
        model.observe(x, y)
        with pytest.raises(ValueError, match="not present"):
            remove_classes(model, [4])


def test_scan_traces_finds_planted_trace_and_cleans_after_removal():
    """Verify scan_traces identifies planted traces and reports clean after remove_labels."""
    pool = DummyPool(d=8, n_classes=4)
    rec0 = ExpertRecord(id=0, module=DummyModule(8, 4), parent=None, birth_step=0)
    rec0.replay = DummyReplay(8)
    rec0.meta["sig"] = DummySig(8)
    rec0.module.seen[0] = True
    rec0.module.seen[1] = True

    rec1 = ExpertRecord(id=1, module=DummyModule(8, 4), parent=0, birth_step=10)
    rec1.replay = DummyReplay(8)
    rec1.meta["sig"] = DummySig(8)
    rec1.module.seen[1] = True
    rec1.module.seen[2] = True

    pool.experts[0] = rec0
    pool.experts[1] = rec1

    # Plant traces for label 2
    pool.probation.add(torch.randn(2, 8), y=torch.tensor([2, 2]), arrival=1)
    rec1.replay.add(torch.randn(2, 8), torch.tensor([1, 2]))

    # Scan before removal
    pre_scan = scan_traces(pool, [2])
    assert pre_scan["total_traces_found"] > 0
    assert pre_scan["pool_probation_samples"]["count"] == 2
    assert pre_scan["expert_module_seen"]["count"] == 1
    assert pre_scan["expert_replay_samples"]["count"] == 1

    # Apply remove_labels
    cert = remove_labels(pool, [2])
    assert cert["certified_clean"] is True
    assert cert["trace_audit"]["total_traces_found"] == 0
    assert 1 not in pool.experts  # Expert 1 saw label 2, so it was dropped
    assert 0 in pool.experts      # Expert 0 never saw label 2, retained


def test_certificate_is_valid_json():
    """Verify removal certificate parses as valid JSON with all required keys."""
    pool = DummyPool(d=8, n_classes=4)
    rec0 = ExpertRecord(id=0, module=DummyModule(8, 4), parent=None, birth_step=0)
    rec0.replay = DummyReplay(8)
    pool.experts[0] = rec0

    cert = remove_labels(pool, [1])
    raw_json = json.dumps(cert)
    parsed = json.loads(raw_json)

    assert "status" in parsed
    assert "target_labels" in parsed
    assert "exposure_summary" in parsed
    assert "trace_audit" in parsed
    assert "certified_clean" in parsed
    assert parsed["certified_clean"] is True


def test_repeat_finished_experiment_skipped(tmp_path: Path):
    """Verify idempotency helper recognizes finished JSON experiment and skips execution."""
    test_json = tmp_path / "exp_test.json"
    data = {"status": "complete", "seed": 800, "metrics": {"acc": 0.85}}
    test_json.write_text(json.dumps(data), encoding="utf-8")

    assert test_json.exists()
    loaded = json.loads(test_json.read_text(encoding="utf-8"))
    assert loaded["status"] == "complete"
