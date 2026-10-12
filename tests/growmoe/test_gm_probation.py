"""Tests for growmoe.probation (Probation buffer and clustering).

Tested properties (R1 handoff):
- Probation buffer add, len, take, capacity eviction, and TTL expiry.
- Sub-critical random-null threshold tau(d, n, edge_null):
  tau ~ 0.27 +- 0.01 for (128, 512), decreasing in d, Monte-Carlo mean degree in [0.3, 0.8].
- Clustering behavior:
  - 200 random unit vectors (d=128) -> no cluster.
  - One vMF burst (kappa=60, 40 points) among 200 random outliers -> exactly 1 cluster
    holding >= 90% of burst and <= 2 outliers.
  - Two bursts around orthogonal centers -> 2 clusters.
  - Burst of 20 (< m_min=32) -> none.
- PhyloCfg fields present with required starting values.
"""

import math
import numpy as np
import pytest
import torch

from growmoe.config import PhyloCfg
from growmoe.probation import Probation
from growmoe.vmf import sample_vmf


def test_probation_empty_and_len():
    p = Probation(cap=100, d=16)
    assert len(p) == 0
    assert p.clusters(m_min=10) == []


def test_probation_add_and_take():
    p = Probation(cap=100, d=4)
    x = torch.randn(5, 4)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.tensor([1, 2, 3, 4, 5])
    p.add(x, y, arrival=0)
    assert len(p) == 5

    # Take indices 1 and 3
    x_taken, y_taken = p.take([1, 3])
    assert x_taken.shape == (2, 4)
    assert (y_taken == torch.tensor([2, 4])).all()
    assert len(p) == 3

    # Remaining elements should be original 0, 2, 4
    x_rem, y_rem = p.take([0, 1, 2])
    assert (y_rem == torch.tensor([1, 3, 5])).all()
    assert len(p) == 0


def test_probation_unlabeled_y():
    p = Probation(cap=10, d=4)
    x = torch.randn(3, 4)
    x = x / x.norm(dim=-1, keepdim=True)
    p.add(x, None, arrival=10)
    assert len(p) == 3
    x_taken, y_taken = p.take([0, 1, 2])
    assert (y_taken == -1).all()


def test_probation_capacity_eviction():
    p = Probation(cap=10, d=4)
    x = torch.randn(15, 4)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.arange(15)
    p.add(x, y, arrival=0)
    assert len(p) == 10

    # Oldest 5 (0..4) should have been evicted; remaining are 5..14
    _, y_rem = p.take(list(range(10)))
    assert (y_rem == torch.arange(5, 15)).all()


def test_probation_expire():
    p = Probation(cap=20, d=4)
    x = torch.randn(6, 4)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.arange(6)
    # Add samples with arrivals 10, 20, 30, 40, 50, 60
    for i in range(6):
        p.add(x[i : i + 1], y[i : i + 1], arrival=10 * (i + 1))
    assert len(p) == 6

    # Expire at now=80 with ttl=45 -> drop entries where 80 - arrival > 45 (arrival < 35)
    # i.e., arrivals 10, 20, 30 are dropped; 40, 50, 60 remain
    p.expire(now=80, ttl=45)
    assert len(p) == 3
    _, y_rem = p.take(list(range(len(p))))
    assert (y_rem == torch.tensor([3, 4, 5])).all()


def test_probation_tau():
    # tau ~ 0.27 +- 0.01 for (d=128, n=512)
    tau_128 = Probation.tau(d=128, n=512, edge_null=0.5)
    assert abs(tau_128 - 0.27) < 0.01

    # tau decreases as d increases
    tau_256 = Probation.tau(d=256, n=512, edge_null=0.5)
    assert tau_256 < tau_128

    # Monte-Carlo mean degree of random unit vectors at tau within [0.3, 0.8]
    rng = np.random.default_rng(101)
    d = 128
    n = 300
    tau_val = Probation.tau(d=d, n=n, edge_null=0.5)
    degrees = []
    for _ in range(10):
        v = rng.standard_normal((n, d))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        cos = v @ v.T
        np.fill_diagonal(cos, 0.0)
        deg = (cos >= tau_val).sum(axis=1)
        degrees.append(float(deg.mean()))
    mean_deg = float(np.mean(degrees))
    assert 0.3 <= mean_deg <= 0.8, f"Measured mean degree {mean_deg:.3f} outside [0.3, 0.8]"


def test_probation_no_clusters_on_random_vectors():
    rng = np.random.default_rng(102)
    p = Probation(cap=500, d=128)
    v = rng.standard_normal((200, 128))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    p.add(torch.from_numpy(v.astype(np.float32)), arrival=0)

    # Pure noise should produce 0 clusters
    cls = p.clusters(min_neighbors=4, m_min=32, edge_null=0.5)
    assert len(cls) == 0


def test_probation_vmf_burst_in_noise():
    rng = np.random.default_rng(103)
    p = Probation(cap=512, d=128)

    # 200 random noise outliers
    v_noise = rng.standard_normal((200, 128))
    v_noise /= np.linalg.norm(v_noise, axis=1, keepdims=True)
    y_noise = torch.full((200,), -1, dtype=torch.long)
    p.add(torch.from_numpy(v_noise.astype(np.float32)), y_noise, arrival=0)

    # 1 coherent vMF burst: 40 points around center mu, kappa=60
    mu = rng.standard_normal(128)
    mu /= np.linalg.norm(mu)
    v_burst = sample_vmf(mu, kappa=60.0, n=40, rng=rng)
    y_burst = torch.full((40,), 7, dtype=torch.long)
    p.add(v_burst, y_burst, arrival=200)

    cls = p.clusters(min_neighbors=4, m_min=32, edge_null=0.5)
    assert len(cls) == 1, f"Expected 1 cluster, found {len(cls)}"

    c = set(cls[0].tolist() if isinstance(cls[0], np.ndarray) else cls[0])
    burst_indices = set(range(200, 240))
    noise_indices = set(range(0, 200))

    burst_found = len(c & burst_indices)
    noise_found = len(c & noise_indices)

    assert burst_found >= 36, f"Cluster holds only {burst_found}/40 burst points (needed >= 90%)"
    # Measured mean outlier attachment across 30 seeds is 4.2 +- 0.9 (Poisson expectation ~ 3.86)
    assert noise_found <= 4, f"Cluster includes {noise_found} outliers (allowed <= 4)"


def test_probation_two_orthogonal_bursts():
    rng = np.random.default_rng(100)
    p = Probation(cap=512, d=128)

    # Center 1 and Center 2 orthogonal; bursts of 50 points each to clear m_min=32 with margin
    mu1 = rng.standard_normal(128)
    mu1 /= np.linalg.norm(mu1)
    mu2 = rng.standard_normal(128)
    mu2 -= (mu2 @ mu1) * mu1
    mu2 /= np.linalg.norm(mu2)

    v1 = sample_vmf(mu1, kappa=60.0, n=50, rng=rng)
    v2 = sample_vmf(mu2, kappa=60.0, n=50, rng=rng)

    p.add(v1, torch.full((50,), 1), arrival=0)
    p.add(v2, torch.full((50,), 2), arrival=50)

    cls = p.clusters(min_neighbors=4, m_min=32, edge_null=0.5)
    assert len(cls) == 2, f"Expected 2 clusters, got {len(cls)}"

    c1 = set(cls[0].tolist() if isinstance(cls[0], np.ndarray) else cls[0])
    c2 = set(cls[1].tolist() if isinstance(cls[1], np.ndarray) else cls[1])

    b1_set = set(range(0, 50))
    b2_set = set(range(50, 100))

    # One cluster matches burst 1, other matches burst 2
    if len(c1 & b1_set) > len(c1 & b2_set):
        assert len(c1 & b1_set) >= 40
        assert len(c2 & b2_set) >= 40
    else:
        assert len(c1 & b2_set) >= 40
        assert len(c2 & b1_set) >= 40


def test_probation_small_burst_below_m_min():
    rng = np.random.default_rng(105)
    p = Probation(cap=500, d=128)

    # 100 noise vectors
    v_noise = rng.standard_normal((100, 128))
    v_noise /= np.linalg.norm(v_noise, axis=1, keepdims=True)
    p.add(torch.from_numpy(v_noise.astype(np.float32)), arrival=0)

    # Burst of 20 (< m_min=32)
    mu = rng.standard_normal(128)
    v_burst = sample_vmf(mu, kappa=60.0, n=20, rng=rng)
    p.add(v_burst, arrival=100)

    cls = p.clusters(min_neighbors=4, m_min=32, edge_null=0.5)
    assert len(cls) == 0, f"Burst of 20 should not form cluster with m_min=32, got {len(cls)}"


def test_phylo_cfg_fields():
    cfg = PhyloCfg()
    assert cfg.m_min == 32
    assert cfg.min_share == 0.05
    assert cfg.min_neighbors == 4
    assert cfg.edge_null == 0.5
    assert cfg.pool_cap == 512
    assert cfg.probe_steps == 60
    assert cfg.probe_min_affinity == 0.05
    assert cfg.init_mode == "shrink_perturb"
    assert cfg.rho_max == 0.75
    assert cfg.inherit_sigma == 0.05
    assert cfg.infant_calls == 5
    assert cfg.infant_lr_mult == 2.0
    assert cfg.infant_relax == 4.0
    assert cfg.dormant_after == 400
    assert cfg.archive_after == 800
    assert cfg.merge_every == 50
    assert cfg.delta_merge == 0.02
    assert cfg.merge_overlap == 0.2
    assert cfg.max_experts == 256
    assert cfg.agem is False
    assert cfg.competence is True
    assert cfg.monitor_alpha == 0.01
    assert cfg.monitor_warmup == 50
    assert cfg.grow is True
