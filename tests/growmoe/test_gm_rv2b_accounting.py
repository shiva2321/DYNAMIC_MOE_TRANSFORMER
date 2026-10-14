"""RV2-B: the gate cost the pool actually spent, and its agreement with the static scan figure."""

import numpy as np
import pytest
import torch

from growmoe.accounting import measured_gate_macs_per_input, pool_report
from growmoe.config import PhyloCfg
from growmoe.experts import MLPExpert
from growmoe.pools.phylo import PhyloMoE
from growmoe.signature import VmfSignature

D = 32


def _unit(n, seed):
    z = np.random.default_rng(seed).standard_normal((n, D))
    return torch.from_numpy((z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32))


def _pool(n_experts, **cfg):
    pool = PhyloMoE(PhyloCfg(d=D, hidden=16, n_classes=4, seed=0, q=0.02, max_experts=4096, route_rule="accept_then_logf", **cfg))
    centres = _unit(n_experts, 99)
    for e in range(n_experts):
        sig = VmfSignature(D, 256, seed=e)
        sig.update(torch.nn.functional.normalize(_unit(128, e) * 0.6 + centres[e], dim=1))
        sig.fit()
        pool.add_expert(MLPExpert(D, 16, 1, 4, seed=e), parent=None, state="active", meta={"sig": sig})
    return pool, centres


def _x(centres, n=64):
    idx = torch.from_numpy(np.random.default_rng(3).integers(0, len(centres), n))
    return torch.nn.functional.normalize(centres[idx] + _unit(n, 4) * 0.6, dim=1)


def test_with_the_cascade_off_the_measured_cost_equals_the_static_scan():
    pool, centres = _pool(20)
    pool.predict(_x(centres))
    assert measured_gate_macs_per_input(pool) == pytest.approx(pool_report(pool, D).gate_macs)


def test_the_cascade_measures_far_fewer_multiply_adds():
    exact, centres = _pool(20)
    casc, _ = _pool(20, gate_shortlist=2, gate_fingerprint_k=4, gate_shortlist_min_experts=4)
    x = _x(centres)
    exact.predict(x)
    casc.predict(x)
    assert measured_gate_macs_per_input(casc) < 0.5 * measured_gate_macs_per_input(exact)


def test_the_phase_argument_selects_the_counters():
    pool, centres = _pool(6)
    pool.predict(_x(centres, 10))
    assert measured_gate_macs_per_input(pool, "observe") == 0.0


def test_no_samples_means_zero_not_a_division_error():
    pool, _ = _pool(3)
    assert measured_gate_macs_per_input(pool) == 0.0
