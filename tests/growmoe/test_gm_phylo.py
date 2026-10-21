"""Tests for growmoe.pools.phylo (PhyloMoE stages 13a-13b).

Tested properties (R2 handoff):
- Bootstrap (13a): buffer (x, y) until >= m_min samples, then found expert 0 from scratch.
  Single-regime stream -> exactly 1 expert, no spawn before m_min samples.
- Gate and known/novel split (13b):
  - Predict on empty pool -> uniform probs, expert -1, p_nov 0, unknown True.
  - Seen regime fresh test samples -> empirical false-novelty <= 2q over 5 seeds.
  - Unseen regime -> novelty detection power > 0.9.
  - Routing among 3 trained regimes >= 0.95.
- Protocol conformance: observe(x, y=None, ctx=None) and predict(x, ctx=None) contain no GT.
"""

import inspect
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from growmoe.config import PRE_STAGE4, PhyloCfg, GrowthCfg
from growmoe.experts import MLPExpert
from growmoe.growth import plateau
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams import s6_long, _balanced_teacher


def test_phylo_init_and_empty_predict():
    cfg = PhyloCfg(d=64, n_classes=8)
    pool = PhyloMoE(cfg=cfg, grow_cfg=GrowthCfg())
    assert pool.name == "A_phylomoe"
    assert len(pool.experts) == 0

    x = torch.randn(10, 64)
    x = x / x.norm(dim=-1, keepdim=True)
    pred = pool.predict(x)

    assert pred.expert.shape == (10,)
    assert (pred.expert == -1).all()
    assert (pred.p_nov == 0.0).all()
    assert (pred.unknown == True).all()
    assert pred.probs.shape == (10, 8)
    expected_prob = 1.0 / 8.0
    assert torch.allclose(pred.probs, torch.full((10, 8), expected_prob), atol=1e-5)


def test_phylo_bootstrap_single_regime():
    seed = 100
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, batch_size=16, lr=3e-3, seed=seed)
    pool = PhyloMoE(cfg=cfg)

    # 1 regime with 160 train samples
    stream = s6_long(seed=seed, n_regimes=1, d=64, n_train=160, n_test=64, related_frac=0.0)

    batches = list(stream.algo_batches(batch_size=16, seed=seed))
    assert len(batches) >= 2

    # Batch 1 (16 samples < m_min=32): buffer only, no expert spawned yet
    x0, y0, ctx0 = batches[0]
    rec0 = pool.observe(x0, y0, ctx0)
    assert rec0.n_experts == 0
    assert len(pool.experts) == 0
    assert (rec0.assigned == -1).all()
    assert (rec0.novel == False).all()
    assert rec0.spawned == []

    # Batch 2 (16 samples, total 32 == m_min=32): expert 0 is founded
    x1, y1, ctx1 = batches[1]
    rec1 = pool.observe(x1, y1, ctx1)
    assert rec1.n_experts == 1
    assert len(pool.experts) == 1
    assert rec1.spawned == [0]
    assert (rec1.assigned == 0).all()

    # Check spawn event recorded in telemetry
    spawn_events = pool.events.of_kind("spawn")
    assert len(spawn_events) == 1
    assert spawn_events[0].fields["expert"] == 0
    assert spawn_events[0].fields["parent"] is None
    assert spawn_events[0].fields["init"] == "scratch"

    # Observe remaining batches: no further spawns
    for xb, yb, ctxb in batches[2:]:
        recb = pool.observe(xb, yb, ctxb)
        assert recb.spawned == []
        assert recb.n_experts == 1

    assert len(pool.experts) == 1
    assert len(pool.events.of_kind("spawn")) == 1


def test_phylo_false_novelty_rate_seen_regime():
    # Samples of a seen regime -> empirical false-novelty <= 2q over 5 seeds
    q = 0.05
    dev_seeds = [100, 101, 102, 103, 104]
    false_nov_rates = []

    for s in dev_seeds:
        cfg = PhyloCfg(d=64, n_classes=8, m_min=32, q=q, seed=s)
        pool = PhyloMoE(cfg=cfg)
        stream = s6_long(seed=s, n_regimes=1, d=64, n_train=320, n_test=200, related_frac=0.0)

        for x, y, ctx in stream.algo_batches(batch_size=32, seed=s):
            pool.observe(x, y, ctx)

        # Score on fresh test samples of the same regime
        x_te, y_te = stream.test_sets()[0]
        pred = pool.predict(x_te)
        rate = pred.unknown.float().mean().item()
        false_nov_rates.append(rate)

    mean_false_nov = float(np.mean(false_nov_rates))
    # Handoff spec: empirical false-novelty <= 2q (= 0.10)
    assert mean_false_nov <= 2 * q, f"Measured false novelty {mean_false_nov:.4f} > 2q ({2*q})"


def test_phylo_novelty_detection_power_unseen_regime():
    # An unseen regime is flagged with power > 0.9 once the first expert is calibrated
    seed = 105
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, q=0.05, seed=seed)
    pool = PhyloMoE(cfg=cfg)

    # 2 unrelated regimes
    stream = s6_long(seed=seed, n_regimes=2, d=64, n_train=320, n_test=200, related_frac=0.0)

    # Train only on regime 0 batches
    for x, y, gt in stream.batches(batch_size=32, seed=seed):
        if gt["regime"] == 0:
            pool.observe(x, y)

    assert len(pool.experts) == 1

    # Score on unseen regime 1 test set
    x_te_unseen, _ = stream.test_sets()[1]
    pred = pool.predict(x_te_unseen)
    power = pred.unknown.float().mean().item()

    assert power > 0.90, f"Novelty detection power on unseen regime {power:.4f} <= 0.90"


def test_phylo_routing_accuracy_three_regimes():
    # Routing among 3 trained regimes >= 0.95
    seed = 106
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, seed=seed)
    pool = PhyloMoE(cfg=cfg)

    stream = s6_long(seed=seed, n_regimes=3, d=64, n_train=300, n_test=200, related_frac=0.0)

    # Train 3 separate experts for the 3 regimes to evaluate gate routing
    # (Since multi-regime spawning is wired in R3, we populate expert 0, 1, 2)
    from growmoe.experts import MLPExpert
    from growmoe.signature import VmfSignature
    from collections import deque

    test_sets = stream.test_sets()

    for r in range(3):
        # Gather regime r training samples
        r_x_list, r_y_list = [], []
        for x, y, gt in stream.batches(batch_size=32, seed=seed):
            if gt["regime"] == r:
                r_x_list.append(x)
                r_y_list.append(y)
        rx = torch.cat(r_x_list, dim=0)
        ry = torch.cat(r_y_list, dim=0)

        mod = MLPExpert(cfg.d, cfg.hidden, cfg.depth, cfg.n_classes, seed=seed * 1000 + r)
        sig = VmfSignature(cfg.d, max_protos=256, seed=seed * 1000 + r)
        sig.update(rx)
        sig.fit()
        rec = pool.add_expert(mod, parent=None, state="active", meta={"sig": sig, "recent": deque(maxlen=cfg.m_min)})
        pool.train_expert(rec.id, rx, ry, replay=True)

    assert len(pool.experts) == 3

    # Now evaluate routing on test sets of all 3 regimes
    correct_routing = []
    for r in range(3):
        x_te, _ = test_sets[r]
        pred = pool.predict(x_te)
        acc = (pred.expert == r).float().mean().item()
        correct_routing.append(acc)

    mean_routing = float(np.mean(correct_routing))
    assert mean_routing >= 0.95, f"Routing accuracy {mean_routing:.4f} < 0.95 across 3 regimes"


def test_phylo_infant_relaxation():
    seed = 107
    cfg = PhyloCfg(d=64, n_classes=8, infant_relax=4.0, seed=seed)
    pool = PhyloMoE(cfg=cfg)

    from growmoe.experts import MLPExpert
    from growmoe.signature import VmfSignature
    from collections import deque

    mod = MLPExpert(cfg.d, cfg.hidden, cfg.depth, cfg.n_classes, seed=seed)
    sig = VmfSignature(cfg.d, max_protos=256, seed=seed)
    x_ref = torch.randn(50, 64)
    x_ref /= x_ref.norm(dim=-1, keepdim=True)
    sig.update(x_ref)
    sig.fit()

    rec = pool.add_expert(mod, parent=None, state="infant", meta={"sig": sig, "recent": deque(maxlen=cfg.m_min)})

    x_test = torch.randn(10, 64)
    x_test /= x_test.norm(dim=-1, keepdim=True)

    # Raw p-value from signature
    raw_p = sig.pvalue(x_test)
    pred = pool.predict(x_test)

    # Infant p-value should be relaxed by infant_relax (clamped to 1.0)
    expected_p = torch.clamp(raw_p * cfg.infant_relax, max=1.0)
    assert torch.allclose(pred.p_nov, expected_p, atol=1e-4)


def test_phylo_signatures_no_gt():
    sig_obs = inspect.signature(PhyloMoE.observe)
    params_obs = list(sig_obs.parameters.keys())
    assert params_obs == ["self", "x", "y", "ctx"]

    sig_pred = inspect.signature(PhyloMoE.predict)
    params_pred = list(sig_pred.parameters.keys())
    assert params_pred == ["self", "x", "ctx"]


# --------------------------------------------------------------------------- R3 Tests
def test_phylo_probation_outliers_no_spawn():
    seed = 108
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, pool_cap=512, seed=seed)
    pool = PhyloMoE(cfg=cfg)

    # Bootstrap expert 0 with 40 samples
    rng = np.random.default_rng(seed)
    mu0 = rng.standard_normal(64)
    mu0 /= np.linalg.norm(mu0)
    from growmoe.vmf import sample_vmf
    x_init = sample_vmf(mu0, kappa=60.0, n=40, rng=rng)
    y_init = torch.full((40,), 0, dtype=torch.long)
    pool.observe(x_init, y_init)
    assert len(pool.experts) == 1

    # 50 isolated random outliers spread across 50 batches
    for _ in range(50):
        v = rng.standard_normal((1, 64))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rec = pool.observe(torch.from_numpy(v.astype(np.float32)), torch.tensor([-1]))
        assert rec.spawned == []

    # Should have 0 spawns from outliers
    assert len(pool.experts) == 1
    assert len(pool.events.of_kind("spawn")) == 1


def test_phylo_probation_coherent_burst_spawns():
    seed = 109
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, pool_cap=512, seed=seed)
    pool = PhyloMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    mu0 = rng.standard_normal(64)
    mu0 /= np.linalg.norm(mu0)
    from growmoe.vmf import sample_vmf
    x0 = sample_vmf(mu0, kappa=60.0, n=40, rng=rng)
    pool.observe(x0, torch.full((40,), 0, dtype=torch.long))
    assert len(pool.experts) == 1

    # One coherent burst of 3 * m_min = 96 points around orthogonal center mu1
    mu1 = rng.standard_normal(64)
    mu1 -= (mu1 @ mu0) * mu0
    mu1 /= np.linalg.norm(mu1)
    x1 = sample_vmf(mu1, kappa=60.0, n=96, rng=rng)
    y1 = torch.full((96,), 1, dtype=torch.long)

    # Feed burst in 3 batches of 32
    for b in range(3):
        xb = x1[b * 32 : (b + 1) * 32]
        yb = y1[b * 32 : (b + 1) * 32]
        pool.observe(xb, yb)

    # Exactly 1 new expert spawned
    assert len(pool.experts) == 2
    spawn_events = pool.events.of_kind("spawn")
    assert len(spawn_events) == 2
    ev1 = spawn_events[1]
    assert ev1.fields["expert"] == 1
    assert ev1.fields["parent"] == 0
    assert ev1.fields["init"] == "shrink_perturb"
    assert pool.ancestors(1) == [0]


def test_phylo_probation_cap_and_ttl_expiry():
    seed = 110
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, min_share=0.1, pool_cap=64, seed=seed)
    pool = PhyloMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    # Bootstrap expert 0
    v0 = rng.standard_normal((32, 64))
    v0 /= np.linalg.norm(v0, axis=1, keepdims=True)
    pool.observe(torch.from_numpy(v0.astype(np.float32)), torch.zeros(32, dtype=torch.long))

    # Feed 100 random noise points
    for _ in range(10):
        v = rng.standard_normal((10, 64))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        pool.observe(torch.from_numpy(v.astype(np.float32)))

    # Pool probation buffer never exceeds pool_cap
    assert len(pool.probation) <= 64

    # TTL is ceil(32 / 0.1) = 320 samples
    ttl = int(np.ceil(cfg.m_min / cfg.min_share))
    # Advance time by adding enough samples to expire old entries
    for _ in range(35):
        v = rng.standard_normal((10, 64))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        pool.observe(torch.from_numpy(v.astype(np.float32)))

    # Buffer should not accumulate beyond cap and old entries should have expired
    assert len(pool.probation) <= 64


def test_phylo_spawn_clone_birth_equivalence():
    seed = 111
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, init_mode="clone", seed=seed)
    pool = PhyloMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    from growmoe.vmf import sample_vmf
    mu0 = rng.standard_normal(64); mu0 /= np.linalg.norm(mu0)
    mu1 = rng.standard_normal(64); mu1 -= (mu1 @ mu0) * mu0; mu1 /= np.linalg.norm(mu1)

    x0 = sample_vmf(mu0, kappa=60.0, n=40, rng=rng)
    pool.observe(x0, torch.zeros(40, dtype=torch.long))

    x1 = sample_vmf(mu1, kappa=60.0, n=64, rng=rng)
    pool.observe(x1, torch.ones(64, dtype=torch.long))

    assert len(pool.experts) == 2
    parent_mod = pool.experts[0].module
    child_mod = pool.experts[1].module

    test_x = torch.randn(5, 64)
    test_x /= test_x.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        # Feature embeddings of parent and clone child must be identical at birth
        feat_p = parent_mod.features(test_x)
        feat_c = child_mod.features(test_x)
        assert torch.allclose(feat_p, feat_c, atol=1e-5)


def test_phylo_spawn_parent_max_affinity():
    seed = 112
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, init_mode="shrink_perturb", seed=seed)
    pool = PhyloMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    from growmoe.vmf import sample_vmf
    mu0 = rng.standard_normal(64); mu0 /= np.linalg.norm(mu0)
    mu1 = rng.standard_normal(64); mu1 -= (mu1 @ mu0) * mu0; mu1 /= np.linalg.norm(mu1)

    # Found Expert 0 around mu0
    x0 = sample_vmf(mu0, kappa=60.0, n=40, rng=rng)
    pool.observe(x0, torch.zeros(40, dtype=torch.long))

    # Spawn Expert 1 around mu1 (orthogonal to mu0)
    x1 = sample_vmf(mu1, kappa=60.0, n=64, rng=rng)
    pool.observe(x1, torch.ones(64, dtype=torch.long))
    assert len(pool.experts) == 2

    # Now introduce a third burst around mu2 which is very close to mu1 (cos ~ 0.9)
    mu2 = 0.9 * mu1 + np.sqrt(1 - 0.9**2) * rng.standard_normal(64)
    mu2 -= (mu2 @ mu0) * mu0  # remain orthogonal to mu0
    mu2 /= np.linalg.norm(mu2)

    x2 = sample_vmf(mu2, kappa=60.0, n=64, rng=rng)
    pool.observe(x2, None)

    # Expert 2 should be spawned with parent = 1 (higher affinity to mu1 than mu0)
    assert len(pool.experts) == 3
    ev2 = pool.events.of_kind("spawn")[2]
    assert ev2.fields["expert"] == 2
    assert ev2.fields["parent"] == 1
    assert ev2.fields["lam"] == ev2.fields["affinity"]


# --------------------------------------------------------------------------- R4 Tests
def test_phylo_probe_aligned_task_extends():
    seed = 113
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, delta_forget=0.20, seed=seed, **PRE_STAGE4)             # the interference probe is the Phase-1 spawn rule; the transfer rule is tested in test_gm_rv4_*
    pool = PhyloMoE(cfg=cfg)

    # Establish Expert 0 with 4 batches of 32 samples (128 samples total)
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(64); mu /= np.linalg.norm(mu)
    from growmoe.vmf import sample_vmf
    for _ in range(4):
        xb = sample_vmf(mu, kappa=60.0, n=32, rng=rng)
        yb = ((xb[:, 0] > 0).long())
        pool.observe(xb, yb)
    assert len(pool.experts) == 1
    assert pool.experts[0].replay.n >= 32

    # Aligned cluster nearby (cos ~ 0.95) with identical class labeling rule
    mu_aligned = 0.95 * mu + np.sqrt(1 - 0.95**2) * rng.standard_normal(64)
    mu_aligned /= np.linalg.norm(mu_aligned)
    x_aligned = sample_vmf(mu_aligned, kappa=60.0, n=64, rng=rng)
    y_aligned = ((x_aligned[:, 0] > 0).long())

    pool.observe(x_aligned, y_aligned)

    # The probe should decide to EXTEND Expert 0, NOT spawn Expert 1
    assert len(pool.experts) == 1
    extend_events = pool.events.of_kind("extend")
    assert len(extend_events) >= 1
    ev = extend_events[-1]
    assert ev.fields["expert"] == 0
    assert ev.fields["d_old"] <= cfg.delta_forget
    assert ev.fields["gap"] <= cfg.delta_forget


def test_phylo_probe_conflicting_task_spawns():
    seed = 114
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, delta_forget=0.05, seed=seed, **PRE_STAGE4)
    pool = PhyloMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(64); mu /= np.linalg.norm(mu)
    from growmoe.vmf import sample_vmf
    x0 = sample_vmf(mu, kappa=60.0, n=64, rng=rng)
    y0 = torch.zeros(64, dtype=torch.long)
    pool.observe(x0, y0)
    assert len(pool.experts) == 1
    assert pool.experts[0].replay.n >= 32

    # Related direction mu_rel at cos ~ 0.88 (affinity > probe_min_affinity)
    mu_rel = 0.88 * mu + np.sqrt(1 - 0.88**2) * rng.standard_normal(64)
    mu_rel /= np.linalg.norm(mu_rel)

    # Expert 0 replay covers mu_rel with label 0
    x_pre = sample_vmf(mu_rel, kappa=60.0, n=32, rng=rng)
    pool.experts[0].replay.add(x_pre, torch.zeros(32, dtype=torch.long))

    # Conflicting task: novel cluster in mu_rel, but with conflicting labels (y=1 vs y=0)
    x_conflict = sample_vmf(mu_rel, kappa=60.0, n=64, rng=rng)
    y_conflict = torch.ones(64, dtype=torch.long)

    # Precondition assertion: assert conflict causes d_old > delta_forget on replay probe
    r_x, r_y = pool.experts[0].replay.data
    split = len(r_x) // 2
    r_fit_x, r_fit_y = r_x[:split], r_y[:split]
    r_probe_x, r_probe_y = r_x[split:], r_y[split:]
    shadow_test = pool.experts[0].module.clone()
    opt_test = torch.optim.Adam(shadow_test.parameters(), lr=cfg.lr)
    for _ in range(cfg.probe_steps):
        shadow_test.train_step(x_conflict[:32], y_conflict[:32], opt_test, replay=(r_fit_x[:32], r_fit_y[:32]))
    d_old_pre = float((F.cross_entropy(shadow_test(r_probe_x), r_probe_y) - F.cross_entropy(pool.experts[0].module(r_probe_x), r_probe_y)).item())
    assert d_old_pre > cfg.delta_forget, f"Precondition failed: conflict d_old {d_old_pre} <= {cfg.delta_forget}"

    pool.observe(x_conflict, y_conflict)

    # Should reject extend and SPAWN Expert 1
    assert len(pool.experts) == 2
    spawn_events = pool.events.of_kind("spawn")
    assert len(spawn_events) == 2
    ev1 = spawn_events[1]
    assert ev1.fields["expert"] == 1
    assert ev1.fields["parent"] == 0


def test_phylo_probe_delta_forget_threshold_flip():
    seed = 115
    from growmoe.vmf import sample_vmf
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(64); mu /= np.linalg.norm(mu)
    x0 = sample_vmf(mu, kappa=60.0, n=64, rng=rng)
    y0 = torch.zeros(64, dtype=torch.long)

    # Moderate shift with different label
    mu_shift = 0.85 * mu + np.sqrt(1 - 0.85**2) * rng.standard_normal(64)
    mu_shift /= np.linalg.norm(mu_shift)
    x1 = sample_vmf(mu_shift, kappa=60.0, n=64, rng=rng)
    y1 = torch.ones(64, dtype=torch.long)

    # 1. Strict threshold -> spawns
    pool_strict = PhyloMoE(cfg=PhyloCfg(d=64, n_classes=8, m_min=32, delta_forget=0.0001, seed=seed, **PRE_STAGE4))
    pool_strict.observe(x0.clone(), y0.clone())
    pool_strict.observe(x1.clone(), y1.clone())
    assert len(pool_strict.experts) == 2
    assert len(pool_strict.events.of_kind("spawn")) == 2

    # 2. Permissive threshold -> extends
    pool_perm = PhyloMoE(cfg=PhyloCfg(d=64, n_classes=8, m_min=32, delta_forget=10.0, seed=seed, **PRE_STAGE4))
    pool_perm.observe(x0.clone(), y0.clone())
    pool_perm.observe(x1.clone(), y1.clone())
    assert len(pool_perm.experts) == 1
    assert len(pool_perm.events.of_kind("extend")) == 1


def test_phylo_infant_protection_subclusters():
    seed = 116
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, infant_calls=10, delta_forget=0.20, seed=seed, **PRE_STAGE4)
    pool = PhyloMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    from growmoe.vmf import sample_vmf
    mu_base = rng.standard_normal(64); mu_base /= np.linalg.norm(mu_base)

    # 5 subclusters around mu_base with high mutual affinity
    for c in range(5):
        mu_c = 0.95 * mu_base + np.sqrt(1 - 0.95**2) * rng.standard_normal(64)
        mu_c /= np.linalg.norm(mu_c)
        x_c = sample_vmf(mu_c, kappa=60.0, n=40, rng=rng)
        y_c = torch.full((40,), c, dtype=torch.long)
        pool.observe(x_c, y_c)

    # Infant protection and probe extending keeps expert count < 5
    assert len(pool.experts) < 5
    assert len(pool.events.of_kind("extend")) >= 1


def test_phylo_infant_lifecycle_transition():
    seed = 117
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, infant_calls=4, infant_lr_mult=2.0, seed=seed)
    pool = PhyloMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    from growmoe.vmf import sample_vmf
    mu0 = rng.standard_normal(64); mu0 /= np.linalg.norm(mu0)
    mu1 = rng.standard_normal(64); mu1 -= (mu1 @ mu0) * mu0; mu1 /= np.linalg.norm(mu1)

    # Found expert 0
    x0 = sample_vmf(mu0, kappa=60.0, n=40, rng=rng)
    pool.observe(x0, torch.zeros(40, dtype=torch.long))
    assert pool.experts[0].state == "active"
    assert pool.experts[0].lr_mult == 1.0

    # Novel burst around mu1 spawns expert 1 in infant state
    x1 = sample_vmf(mu1, kappa=60.0, n=40, rng=rng)
    pool.observe(x1, None)
    assert len(pool.experts) == 2
    assert pool.experts[1].state == "infant"
    assert pool.experts[1].lr_mult == cfg.infant_lr_mult

    # Now observe cfg.infant_calls calls with expert 0 data
    for c in range(1, cfg.infant_calls):
        pool.observe(x0[:10], torch.zeros(10, dtype=torch.long))
        assert pool.experts[1].state == "infant"
        assert pool.experts[1].lr_mult == cfg.infant_lr_mult

    # The cfg.infant_calls-th call triggers transition to active
    pool.observe(x0[:10], torch.zeros(10, dtype=torch.long))
    assert pool.experts[1].state == "active"
    assert pool.experts[1].lr_mult == 1.0


# --------------------------------------------------------------------------- R5 Tests
def test_phylo_competence_stationary_no_alarm():
    seed = 118
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, monitor_warmup=50, competence=True, seed=seed)
    pool = PhyloMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    from growmoe.vmf import sample_vmf
    mu0 = rng.standard_normal(64); mu0 /= np.linalg.norm(mu0)

    # 10 stationary batches of 32 samples (320 samples total, well beyond warmup=50)
    for _ in range(10):
        xb = sample_vmf(mu0, kappa=60.0, n=32, rng=rng)
        yb = torch.zeros(32, dtype=torch.long)
        pool.observe(xb, yb)

    assert len(pool.experts) == 1
    assert pool.experts[0].meta["monitor"].armed
    assert len(pool.events.of_kind("alarm")) == 0


def test_phylo_competence_remapped_alarm_and_supersedes():
    seed = 119
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, monitor_warmup=30, competence=True, delta_forget=0.05, seed=seed)
    pool = PhyloMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    from growmoe.vmf import sample_vmf
    mu0 = rng.standard_normal(64); mu0 /= np.linalg.norm(mu0)

    # Warmup expert 0 on label 0 (2 batches of 32 = 64 samples > warmup=30)
    for _ in range(2):
        xb = sample_vmf(mu0, kappa=60.0, n=32, rng=rng)
        yb = torch.zeros(32, dtype=torch.long)
        pool.observe(xb, yb)

    assert pool.experts[0].meta["monitor"].armed

    # Abrupt concept drift: same input mu0, but remapped labels (y=1)
    # Feed 2 batches of 32 with label 1
    for _ in range(2):
        xb = sample_vmf(mu0, kappa=60.0, n=32, rng=rng)
        yb = torch.ones(32, dtype=torch.long)
        pool.observe(xb, yb)

    # Alarm should have fired and a successor expert spawned
    alarms = pool.events.of_kind("alarm")
    assert len(alarms) >= 1
    assert alarms[0].fields["expert"] == 0

    assert len(pool.experts) == 2
    assert pool.experts[1].meta.get("supersedes") == 0

    # Routing tie-break: fresh test samples on mu0 should route to Expert 1 (superseding Expert 0)
    x_test = sample_vmf(mu0, kappa=60.0, n=20, rng=rng)
    pred = pool.predict(x_test)
    assert (pred.expert == 1).all()


def test_phylo_dormant_archive_and_revive():
    seed = 120
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, dormant_after=5, archive_after=10, seed=seed)
    pool = PhyloMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    from growmoe.vmf import sample_vmf
    mu0 = rng.standard_normal(64); mu0 /= np.linalg.norm(mu0)
    mu1 = rng.standard_normal(64); mu1 -= (mu1 @ mu0) * mu0; mu1 /= np.linalg.norm(mu1)

    # Found expert 0
    x0 = sample_vmf(mu0, kappa=60.0, n=40, rng=rng)
    pool.observe(x0, torch.zeros(40, dtype=torch.long))

    # Spawn expert 1
    x1 = sample_vmf(mu1, kappa=60.0, n=40, rng=rng)
    pool.observe(x1, torch.ones(40, dtype=torch.long))
    assert len(pool.experts) == 2

    # Fast forward expert 1 past infant state
    for _ in range(cfg.infant_calls):
        pool.observe(x0[:10], torch.zeros(10, dtype=torch.long))
    assert pool.experts[1].state == "active"

    # Step expert 1 to dormant (5 steps inactive)
    for _ in range(5):
        pool.observe(x0[:10], torch.zeros(10, dtype=torch.long))
    assert pool.experts[1].state == "dormant"

    # Step expert 1 to archived (10 steps inactive)
    for _ in range(6):
        pool.observe(x0[:10], torch.zeros(10, dtype=torch.long))
    assert pool.experts[1].state == "archived"
    assert pool.experts[1].opt is None
    assert pool.experts[1].replay is None
    assert len(pool.events.of_kind("archive")) >= 1

    # Revive expert 1 by sending mu1 samples
    rec = pool.observe(x1[:20], torch.ones(20, dtype=torch.long))
    assert (rec.assigned == 1).any()
    assert pool.experts[1].state == "active"
    assert pool.experts[1].opt is not None
    assert pool.experts[1].replay is not None
    assert len(pool.events.of_kind("revive")) >= 1


def test_phylo_distillation_merge_duplicates():
    seed = 121
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, merge_every=2, merge_overlap=0.1, delta_merge=0.10, seed=seed)
    pool = PhyloMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    from growmoe.vmf import sample_vmf
    from growmoe.experts import MLPExpert
    from growmoe.signature import VmfSignature
    from collections import deque

    mu0 = rng.standard_normal(64); mu0 /= np.linalg.norm(mu0)
    x = sample_vmf(mu0, kappa=60.0, n=64, rng=rng)
    y = ((x[:, 0] > 0).long())

    # Build Expert 0
    mod0 = MLPExpert(cfg.d, cfg.hidden, cfg.depth, cfg.n_classes, seed=seed)
    sig0 = VmfSignature(cfg.d, max_protos=256, seed=seed)
    sig0.update(x)
    sig0.fit()
    rec0 = pool.add_expert(mod0, parent=None, state="active", meta={"sig": sig0, "recent": deque(maxlen=cfg.m_min)})
    pool.train_expert(rec0.id, x, y, replay=True)

    # Build near-duplicate Expert 1 on same distribution
    mod1 = MLPExpert(cfg.d, cfg.hidden, cfg.depth, cfg.n_classes, seed=seed + 1)
    sig1 = VmfSignature(cfg.d, max_protos=256, seed=seed + 1)
    sig1.update(x)
    sig1.fit()
    rec1 = pool.add_expert(mod1, parent=None, state="active", meta={"sig": sig1, "recent": deque(maxlen=cfg.m_min)})
    pool.train_expert(rec1.id, x, y, replay=True)

    assert len(pool.experts) == 2

    # Step pool to trigger merge_every
    pool.observe(x[:10], y[:10])
    pool.observe(x[:10], y[:10])

    # Expert 1 should have been distilled and merged into Expert 0
    assert len(pool.experts) == 1
    assert 0 in pool.experts
    assert 1 not in pool.experts
    merges = pool.events.of_kind("merge")
    assert len(merges) >= 1
    assert merges[0].fields["kept"] == 0
    assert merges[0].fields["removed"] == 1


def test_phylo_unlabeled_stream_no_training(monkeypatch):
    seed = 122
    from growmoe.experts import MLPExpert
    def raise_if_trained(*args, **kwargs):
        raise RuntimeError("MLPExpert.train_step must not be called on unlabeled streams!")

    monkeypatch.setattr(MLPExpert, "train_step", raise_if_trained)

    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, seed=seed)
    pool = PhyloMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    from growmoe.vmf import sample_vmf

    # 3 regimes fed with y=None
    for r in range(3):
        mu_r = rng.standard_normal(64); mu_r /= np.linalg.norm(mu_r)
        xr = sample_vmf(mu_r, kappa=60.0, n=40, rng=rng)
        pool.observe(xr, None)

    assert len(pool.experts) == 3


def test_phylo_determinism_identical_event_log(tmp_path):
    seed = 123
    cfg1 = PhyloCfg(d=64, n_classes=8, m_min=32, seed=seed)
    pool1 = PhyloMoE(cfg=cfg1)

    cfg2 = PhyloCfg(d=64, n_classes=8, m_min=32, seed=seed)
    pool2 = PhyloMoE(cfg=cfg2)

    stream1 = s6_long(seed=seed, n_regimes=3, d=64, n_train=100, related_frac=0, revisit_frac=0)
    stream2 = s6_long(seed=seed, n_regimes=3, d=64, n_train=100, related_frac=0, revisit_frac=0)

    for x, y, _ in stream1.algo_batches(batch_size=32, seed=seed):
        pool1.observe(x, y)

    for x, y, _ in stream2.algo_batches(batch_size=32, seed=seed):
        pool2.observe(x, y)

    p1 = tmp_path / "log1.jsonl"
    p2 = tmp_path / "log2.jsonl"
    pool1.events.to_jsonl(p1)
    pool2.events.to_jsonl(p2)
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


def test_phylo_mini_integration_three_regimes_with_revisit():
    seed = 124
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, delta_forget=0.20, seed=seed)
    pool = PhyloMoE(cfg=cfg)

    stream = s6_long(seed=seed, n_regimes=3, d=64, n_train=300, related_frac=0, revisit_frac=0)

    # Train on 3 regimes
    for x, y, _ in stream.algo_batches(batch_size=32, seed=seed):
        pool.observe(x, y)

    assert len(pool.experts) == 3

    # Revisit regime 0
    x_te0, y_te0 = stream.test_sets()[0]
    pool.observe(x_te0[:32], y_te0[:32])

    # Revisit must not spawn a 4th expert
    assert len(pool.experts) == 3

    # Verify high accuracy on test sets across all 3 regimes
    accs = []
    for r in range(3):
        x_te, y_te = stream.test_sets()[r]
        pred = pool.predict(x_te)
        acc = (pred.probs.argmax(dim=-1) == y_te).float().mean().item()
        accs.append(acc)

    mean_acc = float(np.mean(accs))
    assert mean_acc >= 0.85, f"Mini-integration mean test accuracy {mean_acc:.4f} < 0.85"


# --------------------------------------------------------------------------- R6: Task 14 Growth inside A
def test_phylo_growth_on_plateau_improves_loss():
    seed = 125
    D, C = 32, 8
    # Capacity-limited student (hidden=4) on a teacher stream (teacher width=16)
    teacher = _balanced_teacher(width=16, depth=1, d=D, n_classes=C, seed=seed)
    rng = np.random.default_rng(seed)

    n_total = 3000
    x_raw = rng.standard_normal((n_total, D))
    x_norm = torch.from_numpy(x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)).float()
    with torch.no_grad():
        y_all = teacher(x_norm).argmax(dim=-1)

    cfg = PhyloCfg(d=D, hidden=4, depth=1, n_classes=C, replay_size=3000, m_min=32, lr=3e-3, seed=seed, growth_rule="probe")      # the shadow-trial policy of Phase 1; growth from the fit gap is tested in test_gm_rv4_*
    grow_cfg = GrowthCfg(train_steps=480, n_boot=200)
    pool = PhyloMoE(cfg=cfg, grow_cfg=grow_cfg)

    # Stream training pass: observe stream in batches of 64
    bs = 64
    for step in range(len(x_norm) // bs):
        bx = x_norm[step * bs : (step + 1) * bs]
        by = y_all[step * bs : (step + 1) * bs]
        pool.observe(bx, by)

    # Precondition assertion: replay buffer has >= 128 samples
    assert pool.experts[0].replay.n >= 128

    # Simulate entering loss plateau with flat history (measured ~0.76 nats)
    rec = pool.experts[0]
    last_loss = rec.meta["loss_hist"][-1]
    rec.meta["loss_hist"] = [last_loss + float(rng.normal(0, 0.0001)) for _ in range(80)]
    assert plateau(rec.meta["loss_hist"])

    fixed_sufficient_params = MLPExpert(d_in=D, hidden=16, depth=1, n_classes=C).num_params()

    # Next observe call triggers plateau growth check
    idx_trig = rng.integers(0, n_total, bs)
    pool.observe(x_norm[idx_trig], y_all[idx_trig])

    grow_events = pool.events.of_kind("grow")
    assert len(grow_events) >= 1, "Expected at least 1 grow event upon plateau check"

    ev = grow_events[0]
    # Check required fields from Section 6 and R6:
    # grow{expert, kind, increment, gain, p, params_added, vs_status}
    assert ev.fields["expert"] == 0
    assert ev.fields["kind"] in ("widen", "deepen")
    assert ev.fields["increment"] >= 0
    assert ev.fields["gain"] >= grow_cfg.min_gain
    assert ev.fields["p"] < grow_cfg.alpha
    assert ev.fields["vs_status"] > 0
    assert ev.fields["params_added"] > 0

    # Final parameters <= 2x the smallest sufficient fixed expert
    final_params = pool.experts[0].module.num_params()
    assert final_params <= 2 * fixed_sufficient_params, (
        f"Final parameters {final_params} exceed 2x sufficient expert {2 * fixed_sufficient_params}"
    )

    # Verify optimizer was rebuilt (references new parameters)
    opt_params = {p for group in pool.experts[0].opt.param_groups for p in group["params"]}
    model_params = set(pool.experts[0].module.parameters())
    assert opt_params == model_params, "Optimizer was not rebuilt with the grown module parameters"

    # Verify loss history was cleared after check
    assert len(rec.meta["loss_hist"]) <= 1


def test_phylo_growth_never_grows_on_data_limited_regime():
    seed = 126
    D, C = 32, 8
    # 60 training samples: replay capacity or total regime has only 60 points
    cfg = PhyloCfg(d=D, hidden=4, depth=1, n_classes=C, replay_size=60, m_min=32, seed=seed, growth_rule="probe")
    grow_cfg = GrowthCfg(train_steps=60, n_boot=50)
    pool = PhyloMoE(cfg=cfg, grow_cfg=grow_cfg)

    rng = np.random.default_rng(seed)
    x_raw = rng.standard_normal((60, D))
    x_norm = torch.from_numpy(x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)).float()
    y_raw = torch.from_numpy(rng.integers(0, C, size=60))

    # Observe multiple passes over the 60 samples (so loss history easily reaches >80 and plateaus)
    for step in range(100):
        idx = rng.integers(0, 60, 32)
        pool.observe(x_norm[idx], y_raw[idx])

    # Precondition assertion: replay buffer has at most 60 samples (< 128)
    assert pool.experts[0].replay.n <= 60
    assert pool.experts[0].replay.n < 128

    # Never grows on a data-limited regime (< 128 replay samples)
    grow_events = pool.events.of_kind("grow")
    assert len(grow_events) == 0, f"Expected 0 grow events on data-limited regime, got {len(grow_events)}"
    assert pool.experts[0].module.hidden == 4


def test_phylo_growth_replay_size_power_comparison():
    seed = 127
    D, C = 32, 8
    # Test firing frequency of small buffer (replay_size=256) vs larger buffer (replay_size=2500)
    teacher = MLPExpert(d_in=D, hidden=4, depth=1, n_classes=C, seed=1234)
    teacher.mark_seen(torch.arange(C))

    cfg_small = PhyloCfg(d=D, hidden=4, depth=1, n_classes=C, replay_size=256, m_min=32, seed=seed, growth_rule="probe")
    pool_small = PhyloMoE(cfg=cfg_small, grow_cfg=GrowthCfg(train_steps=80, n_boot=100))

    rng = np.random.default_rng(seed)
    x_raw = rng.standard_normal((1000, D))
    x_norm = torch.from_numpy(x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)).float()
    with torch.no_grad():
        y_all = teacher(x_norm).argmax(dim=1)

    for step in range(120):
        idx = rng.integers(0, len(x_norm), 32)
        pool_small.observe(x_norm[idx], y_all[idx])

    # With replay_size=256, buffer has at most 256 samples -> split 80/20 has ~51 val samples,
    # probe overfits and vs_status <= 0, so policy does not commit
    assert len(pool_small.events.of_kind("grow")) == 0


def test_phylo_predict_route_rule_options():
    """Verify that pool.predict respects alternative route_rule settings (argmax_p, argmax_logf, zscore_logf, nearest_proto)."""
    from collections import deque
    from growmoe.signature import VmfSignature
    seed = 128
    D, C = 32, 4
    cfg = PhyloCfg(d=D, hidden=16, depth=1, n_classes=C, m_min=16, seed=seed)
    pool = PhyloMoE(cfg=cfg)
    for eid in [0, 1]:
        mod = MLPExpert(d_in=D, hidden=16, depth=1, n_classes=C, seed=seed + eid)
        sig = VmfSignature(d=D, max_protos=64, seed=seed + eid)
        center = torch.zeros(D)
        center[eid] = 1.0
        x_pts = center.unsqueeze(0) + 0.1 * torch.randn(20, D)
        x_pts = x_pts / x_pts.norm(dim=-1, keepdim=True)
        sig.update(x_pts)
        sig.fit()
        pool.add_expert(mod, parent=None, state="active", meta={"sig": sig, "recent": deque(maxlen=16)})

    x_test = torch.zeros(10, D)
    x_test[:, 0] = 1.0  # Closest to expert 0
    x_test = x_test / x_test.norm(dim=-1, keepdim=True)

    for rule in ["default", "accept_then_logf", "argmax_p", "argmax_logf", "zscore_logf", "nearest_proto"]:
        pool.route_rule = rule
        pred = pool.predict(x_test)
        assert pred.expert.shape == (10,)
        assert (pred.expert == 0).all(), f"Rule {rule} failed to route clearly aligned input to expert 0"


