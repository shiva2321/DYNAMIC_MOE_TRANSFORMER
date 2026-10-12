"""M0: honest resource accounting.

A shared module (the frozen base of Pool C) must be counted once, not once per expert, and every pool must report how
much it stores and how much arithmetic one prediction needs, in the same units, so methods can be compared fairly.
"""

import numpy as np
import torch

from growmoe.accounting import pool_report, unique_param_count
from growmoe.baselines import ReplayNet
from growmoe.config import CommonCfg, KeyedCfg, PhyloCfg
from growmoe.experts import MLPExpert
from growmoe.memory_baselines import ClassMean, KnnMemory
from growmoe.pools.keyed import KeyedDeltaMoE
from growmoe.pools.phylo import PhyloMoE
from growmoe.signature import VmfSignature
from growmoe.vmf import sample_vmf


def test_unique_param_count_counts_a_shared_module_once():
    shared = torch.nn.Linear(4, 3)                                            # 4*3 + 3 = 15 parameters
    a = torch.nn.ModuleDict({"base": shared, "own": torch.nn.Linear(3, 2)})   # own: 3*2 + 2 = 8
    b = torch.nn.ModuleDict({"base": shared, "own": torch.nn.Linear(3, 2)})
    assert unique_param_count([a, b]) == 15 + 8 + 8
    assert unique_param_count([a]) == 23
    assert unique_param_count([]) == 0


def _two_regime_keyed_pool(seed=0):
    cfg = KeyedCfg(d=64, n_stm=32, rank=16, seed=seed)
    pool = KeyedDeltaMoE(cfg=cfg)
    rng = np.random.default_rng(seed)
    mu1 = rng.standard_normal(64)
    mu1 /= np.linalg.norm(mu1)
    mu2 = rng.standard_normal(64)
    mu2 /= np.linalg.norm(mu2)
    while float(mu1 @ mu2) > 0.2:
        mu2 = rng.standard_normal(64)
        mu2 /= np.linalg.norm(mu2)
    pool.observe(sample_vmf(mu1, 40.0, 40, rng), torch.zeros(40, dtype=torch.long))
    pool.observe(sample_vmf(mu2, 40.0, 40, rng), torch.ones(40, dtype=torch.long))
    return pool


def test_pool_c_summary_counts_the_shared_base_once():
    pool = _two_regime_keyed_pool()
    assert len(pool.experts) >= 2
    s = pool.summary()
    naive_sum = sum(p.numel() for r in pool.experts.values() for p in r.module.parameters())
    base = sum(p.numel() for p in pool.base_mlp.parameters())
    own = [r.module.num_params() for r in pool.experts.values()]
    assert naive_sum > s["n_params"], "the naive per-expert sum double counts the shared base"
    assert s["n_params"] == base + sum(own)
    assert s["active_params_per_sample"] == base + float(np.mean(own))


# ------------------------------------------------------------------------------------------------ pool_report
D = 32


def _unit(n, seed=0, d=D):
    z = np.random.default_rng(seed).standard_normal((n, d))
    return torch.from_numpy((z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32))


def _weight_macs(module):
    return sum(p.numel() for p in module.parameters() if p.dim() == 2)


def test_report_for_a_replay_network_counts_parameters_buffer_and_matrix_multiplies():
    cfg = CommonCfg(d=D, hidden=16, n_classes=4, replay_size=64)
    pool = ReplayNet(cfg, replay_size=50)
    pool.observe(_unit(80), torch.randint(0, 4, (80,)))
    r = pool_report(pool, d=D)
    assert r.params == sum(p.numel() for p in pool.net.parameters())
    assert r.stored_vectors == 50                                                       # the buffer is full (capacity 50)
    assert r.memory_bytes == 4 * r.params + 4 * D * 50
    assert r.predict_macs == _weight_macs(pool.net) and r.gate_macs == 0 and r.n_experts == 1


def test_report_for_knn_and_class_mean_counts_stored_vectors_and_lookup_cost():
    cfg = CommonCfg(d=D, n_classes=4)
    knn = KnnMemory(cfg)
    knn.observe(_unit(70), torch.randint(0, 4, (70,)))
    r = pool_report(knn, d=D)
    assert (r.params, r.stored_vectors, r.memory_bytes, r.predict_macs) == (0, 70, 4 * D * 70, 70 * D)
    ncm = ClassMean(cfg)
    ncm.observe(_unit(70), torch.randint(0, 4, (70,)))
    r = pool_report(ncm, d=D)
    assert (r.params, r.stored_vectors, r.predict_macs) == (0, 4, 4 * D)


def _fake_phylo_pool():
    """Three experts with known numbers of signature prototypes (64, 100, 200) and replay samples (30, 0, 12)."""
    pool = PhyloMoE(PhyloCfg(d=D, hidden=16, n_classes=4, seed=0))
    for e, (n_sig, n_rep) in enumerate([(64, 30), (100, 0), (200, 12)]):
        sig = VmfSignature(D, 256, seed=e)
        sig.update(_unit(n_sig, seed=10 + e))
        sig.fit()
        rec = pool.add_expert(MLPExpert(D, 16, 1, 4, seed=e), parent=None, state="active", meta={"sig": sig})
        if n_rep:
            rec.replay.add(_unit(n_rep, seed=20 + e), torch.zeros(n_rep, dtype=torch.long))
    return pool


def test_report_for_a_pool_adds_the_gate_scan_over_every_experts_stored_vectors():
    pool = _fake_phylo_pool()
    r = pool_report(pool, d=D)
    params = sum(p.numel() for rec in pool.experts.values() for p in rec.module.parameters())
    assert r.params == params and r.n_experts == 3
    assert r.stored_vectors == (64 + 100 + 200) + (30 + 0 + 12)
    assert r.gate_macs == (64 + 100 + 200) * D                                          # one scan of every stored signature vector
    expert_macs = _weight_macs(pool.experts[0].module)
    assert r.expert_macs == expert_macs and r.predict_macs == r.gate_macs + expert_macs
    assert r.memory_bytes == 4 * params + 4 * D * r.stored_vectors


def test_report_for_pool_c_uses_the_unique_parameter_count():
    pool = _two_regime_keyed_pool()
    r = pool_report(pool, d=64)
    assert r.params == pool.summary()["n_params"]


def test_report_keeps_the_naive_per_expert_sum_next_to_the_unique_count():
    pool = _two_regime_keyed_pool()
    r = pool_report(pool, d=64)
    naive = sum(p.numel() for rec in pool.experts.values() for p in rec.module.parameters())
    assert r.naive_params == naive and r.naive_params > r.params                         # the shared base is repeated in the naive sum
    plain = pool_report(_fake_phylo_pool(), d=D)
    assert plain.naive_params == plain.params                                           # nothing is shared: the two counts agree
    knn = pool_report(KnnMemory(CommonCfg(d=D, n_classes=4)), d=D)
    assert knn.naive_params == knn.params == 0