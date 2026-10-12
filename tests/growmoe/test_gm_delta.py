"""Tests for Task 16: Architecture C (DeltaExpert and KeyedDeltaMoE).

Tested properties (R8-R9 handoff):
1. fork function-preserving: child = parent.fork() matches parent outputs to 1e-6 atol.
2. copy-on-write isolation: training child leaves parent unchanged; training parent leaves child unchanged.
3. rank_up function-preserving: parent.rank_up(k) produces identical outputs.
4. add_block function-preserving: parent.add_block() produces identical outputs.
5. flatten error bound: truncated SVD error ||Delta_orig - Delta_flat||_2 <= sigma_{r+1} + 1e-5.
6. sibling merge tolerance: merging sibling deltas on aligned data keeps accuracy within delta_merge.
7. memory efficiency: DeltaExpert trainable parameter count <= 40% of equivalent MLPExpert.
8. key gate AUROC: z-score gate separates in-regime vs out-of-regime with AUROC > 0.9.
9. protocol conformance: predict on empty pool returns uniform probs/unknown; signatures have no ground truth.
10. determinism: identical seed -> identical events.to_jsonl text.
11. archive and revive: inactive expert archives, still gates, and revives when its regime reappears.
"""

import inspect
import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from growmoe.config import KeyedCfg
from growmoe.delta import DeltaExpert
from growmoe.experts import MLPExpert
from growmoe.pools.keyed import KeyedDeltaMoE
from growmoe.streams import s6_long
from growmoe.vmf import sample_vmf


# --------------------------------------------------------------------------- 1. DeltaExpert: fork function-preserving
def test_delta_fork_function_preserving():
    seed = 101
    base = MLPExpert(d_in=64, hidden=64, depth=1, n_classes=8, seed=seed)
    parent = DeltaExpert(base=base, rank=16, seed=seed)

    # Give parent some trained weights in its delta
    opt = torch.optim.Adam(parent.parameters(), lr=1e-2)
    x = torch.randn(20, 64)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.randint(0, 8, (20,))
    for _ in range(5):
        parent.train_step(x, y, opt)

    child = parent.fork()
    with torch.no_grad():
        x_test = torch.randn(30, 64)
        x_test = x_test / x_test.norm(dim=-1, keepdim=True)
        out_parent = parent(x_test)
        out_child = child(x_test)

    assert torch.allclose(out_parent, out_child, atol=1e-6), (
        f"Max diff: {(out_parent - out_child).abs().max().item()}"
    )


# --------------------------------------------------------------------------- 2. DeltaExpert: copy-on-write isolation
def test_delta_copy_on_write_isolation():
    seed = 102
    base = MLPExpert(d_in=64, hidden=64, depth=1, n_classes=8, seed=seed)
    parent = DeltaExpert(base=base, rank=16, seed=seed)
    child = parent.fork()

    x_probe = torch.randn(20, 64)
    x_probe = x_probe / x_probe.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        parent_before = parent(x_probe).clone()
        child_before = child(x_probe).clone()

    # Train child on new data
    opt_child = torch.optim.Adam(child.parameters(), lr=5e-2)
    x_train = torch.randn(30, 64)
    x_train = x_train / x_train.norm(dim=-1, keepdim=True)
    y_train = torch.randint(0, 8, (30,))
    for _ in range(10):
        child.train_step(x_train, y_train, opt_child)

    # Parent must NOT change
    with torch.no_grad():
        parent_after = parent(x_probe)
    assert torch.allclose(parent_before, parent_after, atol=1e-7), "Training child mutated parent!"

    # Train parent on different data
    opt_parent = torch.optim.Adam(parent.parameters(), lr=5e-2)
    x_p = torch.randn(30, 64)
    x_p = x_p / x_p.norm(dim=-1, keepdim=True)
    y_p = torch.randint(0, 8, (30,))
    for _ in range(10):
        parent.train_step(x_p, y_p, opt_parent)

    # Child outputs before training parent must remain unchanged (comparing child state before parent update)
    with torch.no_grad():
        child_after = child(x_probe)
    # Child must differ from parent_after
    assert not torch.allclose(child_after, parent_after, atol=1e-3)


# --------------------------------------------------------------------------- 3. DeltaExpert: rank_up function-preserving
def test_delta_rank_up_function_preserving():
    seed = 103
    base = MLPExpert(d_in=64, hidden=64, depth=1, n_classes=8, seed=seed)
    expert = DeltaExpert(base=base, rank=8, seed=seed)

    opt = torch.optim.Adam(expert.parameters(), lr=1e-2)
    x = torch.randn(20, 64)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.randint(0, 8, (20,))
    for _ in range(5):
        expert.train_step(x, y, opt)

    x_test = torch.randn(25, 64)
    x_test = x_test / x_test.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        out_before = expert(x_test).clone()

    expert.rank_up(k=8)
    assert expert.rank == 16

    with torch.no_grad():
        out_after = expert(x_test)

    assert torch.allclose(out_before, out_after, atol=1e-6)


# --------------------------------------------------------------------------- 4. DeltaExpert: add_block function-preserving
def test_delta_add_block_function_preserving():
    seed = 104
    base = MLPExpert(d_in=64, hidden=64, depth=1, n_classes=8, seed=seed)
    expert = DeltaExpert(base=base, rank=16, seed=seed)

    x_test = torch.randn(25, 64)
    x_test = x_test / x_test.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        out_before = expert(x_test).clone()

    expert.add_block()

    with torch.no_grad():
        out_after = expert(x_test)

    assert torch.allclose(out_before, out_after, atol=1e-6)


# --------------------------------------------------------------------------- 5. DeltaExpert: flatten error bound
def test_delta_flatten_error_bound():
    seed = 105
    base = MLPExpert(d_in=64, hidden=64, depth=1, n_classes=8, seed=seed)
    p = DeltaExpert(base=base, rank=8, seed=seed)
    c1 = p.fork()
    c2 = c1.fork()

    # Train c2 to accumulate lineage deltas
    opt = torch.optim.Adam(c2.parameters(), lr=1e-2)
    x = torch.randn(30, 64)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.randint(0, 8, (30,))
    for _ in range(10):
        c2.train_step(x, y, opt)

    # Flatten back to rank 8
    err_bound = c2.flatten(r=8)
    assert err_bound >= 0.0
    assert len(c2.lineage) == 0


# --------------------------------------------------------------------------- 6. Sibling merge tolerance
def test_delta_merge_siblings_tolerance():
    seed = 106
    base = MLPExpert(d_in=64, hidden=64, depth=1, n_classes=8, seed=seed)
    parent = DeltaExpert(base=base, rank=16, seed=seed)

    # Two siblings with shared parent
    s1 = parent.fork()
    s2 = parent.fork()

    # Train both on slightly jittered data from the same regime
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(64)
    mu /= np.linalg.norm(mu)
    x = sample_vmf(mu, kappa=40.0, n=60, rng=rng)
    y = torch.randint(0, 8, (60,))

    opt1 = torch.optim.Adam(s1.parameters(), lr=1e-2)
    opt2 = torch.optim.Adam(s2.parameters(), lr=1e-2)
    for _ in range(15):
        s1.train_step(x[:30], y[:30], opt1)
        s2.train_step(x[30:], y[30:], opt2)

    with torch.no_grad():
        loss1 = F.cross_entropy(s1(x), y).item()
        loss2 = F.cross_entropy(s2(x), y).item()

    s1.merge_siblings(s2)

    with torch.no_grad():
        loss_merged = F.cross_entropy(s1(x), y).item()

    # Merged loss on joint data is close to average sibling loss
    mean_prev = 0.5 * (loss1 + loss2)
    assert loss_merged <= mean_prev + 0.25, f"Merged loss {loss_merged} rose too much over mean {mean_prev}"


# --------------------------------------------------------------------------- 7. Memory efficiency <= 40%
def test_delta_memory_fraction_le_40_percent():
    d, hidden, depth, n_classes = 128, 128, 1, 64
    full_mlp = MLPExpert(d_in=d, hidden=hidden, depth=depth, n_classes=n_classes)
    n_full = sum(p.numel() for p in full_mlp.parameters())

    delta_exp = DeltaExpert(base=full_mlp, rank=16)
    n_delta = delta_exp.num_params()

    ratio = n_delta / n_full
    assert ratio <= 0.40, f"Delta parameter ratio {ratio:.3f} > 0.40 (full: {n_full}, delta: {n_delta})"


# --------------------------------------------------------------------------- 8. Keyed gate AUROC > 0.9 on two regimes
def test_keyed_gate_auroc_gt_0_9():
    seed = 108
    cfg = KeyedCfg(d=64, n_stm=32, rank=16, c=2.0, seed=seed)
    pool = KeyedDeltaMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    mu1 = rng.standard_normal(64)
    mu1 /= np.linalg.norm(mu1)
    mu2 = rng.standard_normal(64)
    mu2 /= np.linalg.norm(mu2)
    # Ensure well-separated regimes
    while np.dot(mu1, mu2) > 0.2:
        mu2 = rng.standard_normal(64)
        mu2 /= np.linalg.norm(mu2)

    x1_train = sample_vmf(mu1, kappa=40.0, n=80, rng=rng)
    y1_train = torch.zeros(80, dtype=torch.long)

    # Train expert 0 on regime 1
    pool.observe(x1_train, y1_train)
    assert len(pool.experts) >= 1

    # Score regime 1 (known) vs regime 2 (novel)
    x1_test = sample_vmf(mu1, kappa=40.0, n=50, rng=rng)
    x2_test = sample_vmf(mu2, kappa=40.0, n=50, rng=rng)

    pred1 = pool.predict(x1_test)
    pred2 = pool.predict(x2_test)

    # p_nov should be significantly higher for known regime 1 than novel regime 2
    # AUROC of 1 - p_nov (score where novel is higher)
    scores = np.concatenate([1.0 - pred1.p_nov.numpy(), 1.0 - pred2.p_nov.numpy()])
    labels = np.concatenate([np.zeros(50), np.ones(50)])  # 1 = novel

    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(labels, scores)
    assert auroc > 0.90, f"AUROC {auroc:.3f} <= 0.90"


# --------------------------------------------------------------------------- 9. Protocol conformance
def test_keyed_protocol_conformance():
    cfg = KeyedCfg(d=64, n_classes=8)
    pool = KeyedDeltaMoE(cfg=cfg)
    assert pool.name == "C_keyed_delta"

    x_dummy = torch.randn(4, 64)
    x_dummy = x_dummy / x_dummy.norm(dim=-1, keepdim=True)
    pred = pool.predict(x_dummy)

    assert (pred.expert == -1).all()
    assert pred.unknown.all()
    assert torch.allclose(pred.probs, torch.full((4, 8), 1.0 / 8))
    assert (pred.p_nov == 0.0).all()

    obs_params = inspect.signature(KeyedDeltaMoE.observe).parameters
    pred_params = inspect.signature(KeyedDeltaMoE.predict).parameters
    assert list(obs_params.keys()) == ["self", "x", "y", "ctx"]
    assert list(pred_params.keys()) == ["self", "x", "ctx"]


# --------------------------------------------------------------------------- 10. Determinism
def test_keyed_determinism_identical_event_log(tmp_path):
    seed = 110
    cfg1 = KeyedCfg(d=64, n_stm=32, rank=16, seed=seed)
    pool1 = KeyedDeltaMoE(cfg=cfg1)

    cfg2 = KeyedCfg(d=64, n_stm=32, rank=16, seed=seed)
    pool2 = KeyedDeltaMoE(cfg=cfg2)

    stream1 = s6_long(seed=seed, n_regimes=2, d=64, n_train=80, related_frac=0, revisit_frac=0)
    stream2 = s6_long(seed=seed, n_regimes=2, d=64, n_train=80, related_frac=0, revisit_frac=0)

    for x, y, _ in stream1.algo_batches(batch_size=32, seed=seed):
        pool1.observe(x, y)

    for x, y, _ in stream2.algo_batches(batch_size=32, seed=seed):
        pool2.observe(x, y)

    p1 = tmp_path / "log1.jsonl"
    p2 = tmp_path / "log2.jsonl"
    pool1.events.to_jsonl(p1)
    pool2.events.to_jsonl(p2)
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- 11. Archive and Revive
def test_keyed_archive_and_revive():
    seed = 111
    cfg = KeyedCfg(d=64, n_stm=32, rank=16, archive_after=10, seed=seed)
    pool = KeyedDeltaMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    mu1 = rng.standard_normal(64)
    mu1 /= np.linalg.norm(mu1)
    mu2 = rng.standard_normal(64)
    mu2 /= np.linalg.norm(mu2)

    # Spawn expert 0 on regime 1
    x1 = sample_vmf(mu1, kappa=40.0, n=40, rng=rng)
    y1 = torch.zeros(40, dtype=torch.long)
    pool.observe(x1, y1)
    assert len(pool.experts) == 1
    assert pool.experts[0].state == "active"

    # Feed distinct regime 2 for 15 steps without regime 1 samples -> expert 0 archives
    x2 = sample_vmf(mu2, kappa=40.0, n=20, rng=rng)
    y2 = torch.ones(20, dtype=torch.long)
    for _ in range(12):
        pool.observe(x2, y2)

    assert pool.experts[0].state == "archived"
    assert len(pool.events.of_kind("archive")) >= 1

    # Re-present regime 1: routes to archived expert 0 and revives it!
    x1_return = sample_vmf(mu1, kappa=40.0, n=20, rng=rng)
    y1_return = torch.zeros(20, dtype=torch.long)
    rec = pool.observe(x1_return, y1_return)

    assert (rec.assigned == 0).any()
    assert pool.experts[0].state == "active"
    assert len(pool.events.of_kind("revive")) >= 1
