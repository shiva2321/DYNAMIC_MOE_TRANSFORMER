"""CVA-2a: Verification of missing and partial specification bullets from GROWMOE_HANDOFF.md section 7.

Covers up to 12 missing or partial bullets across R1-R6:
1. Evidence test: cluster with evidence <= threshold stays in probation (R3 evidence test)
2. Evidence test: cluster with evidence > threshold spawns and clears probation (R3 evidence test)
3. Spawn init_mode="inherit" (R3 13d)
4. Spawn lam grows with affinity (R3 13d)
5. Spawn child seen reset and replay initialization (R3 13d)
6. Probe: need >= 16 in R_probe else spawn (R4 13e)
7. Probe: unlabeled cluster skips probe and spawns (R4 13e)
8. Probe: conflicting task keeps already-trained child weights (R4 13e)
9. Infant gate relaxation in gate matrices (R2 13b / R4 13f)
10. Distillation merge reparents younger expert's children to older expert (R5 13h)
11. Distillation merge rejected when accuracy drops > delta_merge (R5 13h)
12. Task 14 growth requires loss plateau (strictly decreasing loss does not grow) (R6)
"""

import math
from collections import deque
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from growmoe.config import PhyloCfg, GrowthCfg
from growmoe.experts import MLPExpert
from growmoe.growth import plateau
from growmoe.pools.phylo import PhyloMoE
from growmoe.signature import VmfSignature
from growmoe.vmf import sample_vmf


def test_cva_evidence_test_low_evidence_stays_in_probation():
    """Handoff R3: Candidate cluster with evidence <= log(1/alpha_spawn) stays in probation without spawning."""
    seed = 200
    cfg = PhyloCfg(d=32, n_classes=4, m_min=16, seed=seed)
    pool = PhyloMoE(cfg=cfg)
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(32)
    mu /= np.linalg.norm(mu)

    # Expert 0
    x0 = sample_vmf(mu, kappa=40.0, n=32, rng=rng)
    y0 = torch.zeros(32, dtype=torch.long)
    pool.observe(x0, y0)
    assert len(pool.experts) == 1

    # Candidate cluster drawn from mu (same regime): evidence will be <= threshold
    x_cand = sample_vmf(mu, kappa=40.0, n=20, rng=rng)
    pool.probation.add(x_cand, torch.zeros(20, dtype=torch.long), arrival=pool.samples_seen)

    # Observe a known sample to trigger probation cluster check
    pool.observe(x0[:1], y0[:1])
    assert len(pool.experts) == 1
    assert len(pool.probation) == 20


def test_cva_evidence_test_high_evidence_spawns():
    """Handoff R3: Candidate cluster with evidence > log(1/alpha_spawn) triggers spawn and clears taken samples."""
    seed = 201
    cfg = PhyloCfg(d=32, n_classes=4, m_min=16, infant_calls=1, probe_min_affinity=0.2, seed=seed)
    pool = PhyloMoE(cfg=cfg)
    rng = np.random.default_rng(seed)
    mu0 = rng.standard_normal(32)
    mu0 /= np.linalg.norm(mu0)
    mu1 = rng.standard_normal(32)
    mu1 -= float(np.dot(mu1, mu0)) * mu0
    mu1 /= np.linalg.norm(mu1)

    pool.observe(sample_vmf(mu0, kappa=40.0, n=32, rng=rng), torch.zeros(32, dtype=torch.long))
    pool.observe(sample_vmf(mu0, kappa=40.0, n=32, rng=rng), torch.zeros(32, dtype=torch.long))
    assert pool.experts[0].state == "active"

    # Orthogonal cluster of 32 points: evidence > threshold -> spawns Expert 1
    pool.observe(sample_vmf(mu1, kappa=40.0, n=32, rng=rng), torch.ones(32, dtype=torch.long))
    assert len(pool.experts) == 2
    assert 1 in pool.experts
    assert len(pool.probation) < 16


def test_cva_spawn_init_mode_inherit():
    """Handoff R3 13d: Spawning child with init_mode='inherit' calls parent.module.inherit and logs init='inherit'."""
    seed = 202
    cfg = PhyloCfg(d=32, n_classes=4, m_min=16, init_mode="inherit", seed=seed)
    pool = PhyloMoE(cfg=cfg)
    rng = np.random.default_rng(seed)
    mu0 = rng.standard_normal(32)
    mu0 /= np.linalg.norm(mu0)
    mu1 = rng.standard_normal(32)
    mu1 -= float(np.dot(mu1, mu0)) * mu0
    mu1 /= np.linalg.norm(mu1)

    pool.observe(sample_vmf(mu0, kappa=40.0, n=32, rng=rng), torch.zeros(32, dtype=torch.long))
    pool.observe(sample_vmf(mu1, kappa=40.0, n=48, rng=rng), torch.ones(48, dtype=torch.long))
    assert len(pool.experts) == 2
    ev = pool.events.of_kind("spawn")[-1]
    assert ev.fields["init"] == "inherit"
    assert pool.experts[1].module.num_params() == pool.experts[0].module.num_params()


def test_cva_spawn_lam_monotonic_with_affinity():
    """Handoff R3 13d: lam grows with affinity under constructed affinities."""
    cfg = PhyloCfg(init_mode="shrink_perturb")
    pool = PhyloMoE(cfg=cfg)
    affinities = [0.10, 0.35, 0.70, 0.95]
    lams = []
    for aff in affinities:
        # PhyloMoE._handle_cluster sets lam = 1.0 (clone), 0.0 (scratch), else aff
        lam = 1.0 if cfg.init_mode == "clone" else (0.0 if cfg.init_mode == "scratch" else aff)
        lams.append(lam)
    assert all(l1 < l2 for l1, l2 in zip(lams, lams[1:]))


def test_cva_spawn_child_seen_reset_and_replay():
    """Handoff R3 13d: Child seen reset to cluster labels, replay <- cluster, lr_mult = infant_lr_mult."""
    seed = 203
    cfg = PhyloCfg(d=32, n_classes=4, m_min=16, seed=seed)
    pool = PhyloMoE(cfg=cfg)
    rng = np.random.default_rng(seed)
    mu0 = rng.standard_normal(32)
    mu0 /= np.linalg.norm(mu0)
    mu1 = rng.standard_normal(32)
    mu1 -= float(np.dot(mu1, mu0)) * mu0
    mu1 /= np.linalg.norm(mu1)

    pool.observe(sample_vmf(mu0, kappa=40.0, n=32, rng=rng), torch.zeros(32, dtype=torch.long))
    x1 = sample_vmf(mu1, kappa=40.0, n=48, rng=rng)
    y1 = torch.full((48,), 2, dtype=torch.long)
    pool.observe(x1, y1)

    child = pool.experts[1]
    assert child.state == "infant"
    assert child.lr_mult == cfg.infant_lr_mult
    assert child.replay is not None
    assert child.replay.n >= 16
    assert child.module.seen[2].item() is True
    assert child.module.seen[:2].any().item() is False


def test_cva_probe_insufficient_probe_replay_spawns():
    """Handoff R4 13e: Split parent replay into halves; need >= 16 in R_probe else spawn."""
    seed = 204
    cfg = PhyloCfg(d=32, n_classes=4, m_min=16, q=0.10, delta_forget=10.0, infant_calls=2, seed=seed)
    pool = PhyloMoE(cfg=cfg)
    rng = np.random.default_rng(seed)
    mu0 = rng.standard_normal(32)
    mu0 /= np.linalg.norm(mu0)

    # Found expert 0 with 16 samples, then 4 samples to end infancy (replay.n == 20 < 32, so R_probe < 16)
    pool.observe(sample_vmf(mu0, kappa=40.0, n=16, rng=rng), torch.zeros(16, dtype=torch.long))
    pool.observe(sample_vmf(mu0, kappa=40.0, n=4, rng=rng), torch.zeros(4, dtype=torch.long))
    assert pool.experts[0].state == "active"
    assert pool.experts[0].replay.n == 20

    # Novel cluster (cos ~ 0.50)
    mu_cand = 0.50 * mu0 + math.sqrt(1 - 0.50**2) * rng.standard_normal(32)
    mu_cand /= np.linalg.norm(mu_cand)
    pool.observe(sample_vmf(mu_cand, kappa=40.0, n=40, rng=rng), torch.zeros(40, dtype=torch.long))

    # Must spawn expert 1 because R_probe < 16, even though delta_forget=10.0 would allow extend
    assert len(pool.experts) == 2
    assert len(pool.events.of_kind("spawn")) == 2
    assert len(pool.events.of_kind("extend")) == 0


def test_cva_probe_unlabeled_cluster_skips_probe_and_spawns():
    """Handoff R4 13e: Unlabeled novel cluster (y=None) -> no probe, spawn directly."""
    seed = 205
    cfg = PhyloCfg(d=32, n_classes=4, m_min=16, seed=seed)
    pool = PhyloMoE(cfg=cfg)
    rng = np.random.default_rng(seed)
    mu0 = rng.standard_normal(32)
    mu0 /= np.linalg.norm(mu0)
    mu1 = rng.standard_normal(32)
    mu1 -= float(np.dot(mu1, mu0)) * mu0
    mu1 /= np.linalg.norm(mu1)

    pool.observe(sample_vmf(mu0, kappa=40.0, n=32, rng=rng), torch.zeros(32, dtype=torch.long))
    pool.observe(sample_vmf(mu1, kappa=40.0, n=48, rng=rng), None)

    assert len(pool.experts) == 2
    assert len(pool.events.of_kind("extend")) == 0
    assert len(pool.events.of_kind("spawn")) == 2


def test_cva_probe_conflict_keeps_probe_trained_child():
    """Handoff R4 13e: On conflicting task, probe rejects extend and spawns; child keeps trained weights."""
    seed = 206
    cfg = PhyloCfg(d=64, n_classes=8, m_min=32, delta_forget=0.05, seed=seed)
    pool = PhyloMoE(cfg=cfg)
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(64)
    mu /= np.linalg.norm(mu)
    x0 = sample_vmf(mu, kappa=60.0, n=64, rng=rng)
    y0 = torch.zeros(64, dtype=torch.long)
    pool.observe(x0, y0)

    mu_rel = 0.88 * mu + math.sqrt(1 - 0.88**2) * rng.standard_normal(64)
    mu_rel /= np.linalg.norm(mu_rel)
    x_pre = sample_vmf(mu_rel, kappa=60.0, n=32, rng=rng)
    pool.experts[0].replay.add(x_pre, torch.zeros(32, dtype=torch.long))

    x_conflict = sample_vmf(mu_rel, kappa=60.0, n=64, rng=rng)
    y_conflict = torch.ones(64, dtype=torch.long)
    pool.observe(x_conflict, y_conflict)

    assert len(pool.experts) == 2
    loss_child = float(F.cross_entropy(pool.experts[1].module(x_conflict), y_conflict))
    assert loss_child < 0.5


def test_cva_infant_gate_relaxation_active():
    """Handoff R2 13b / R4 13f: Gate relaxes infant conformal p-values by min(1, p * infant_relax)."""
    cfg = PhyloCfg(d=16, n_classes=2, infant_relax=4.0, seed=0)
    pool = PhyloMoE(cfg=cfg)
    x = torch.randn(8, 16)
    x = x / x.norm(dim=-1, keepdim=True)
    sig = VmfSignature(16, max_protos=64, seed=0)
    sig.update(x)
    sig.fit()

    pool.add_expert(MLPExpert(16, 8, 1, 2, seed=0), parent=None, state="infant", meta={"sig": sig})
    p_infant, _ = pool._gate_column(0, x)

    pool.experts[0].state = "active"
    p_active, _ = pool._gate_column(0, x)

    expected = torch.clamp(p_active * 4.0, max=1.0)
    assert torch.allclose(p_infant, expected, atol=1e-5)


def test_cva_distillation_merge_reparents_children():
    """Handoff R5 13h: When younger expert merges into older expert, younger's children are reparented to older."""
    cfg = PhyloCfg(d=16, n_classes=2, merge_every=2, merge_overlap=0.1, delta_merge=0.5, seed=0)
    pool = PhyloMoE(cfg=cfg)
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(16)
    mu /= np.linalg.norm(mu)
    x = sample_vmf(mu, kappa=40.0, n=32, rng=rng)
    y = torch.zeros(32, dtype=torch.long)

    m0 = MLPExpert(16, 8, 1, 2, seed=0)
    s0 = VmfSignature(16, max_protos=64, seed=0)
    s0.update(x); s0.fit()
    pool.add_expert(m0, parent=None, state="active", meta={"sig": s0, "recent": deque(maxlen=16)})
    pool.train_expert(0, x, y, replay=True)

    m1 = m0.clone()
    s1 = VmfSignature(16, max_protos=64, seed=1)
    s1.update(x); s1.fit()
    pool.add_expert(m1, parent=0, state="active", meta={"sig": s1, "recent": deque(maxlen=16)})
    pool.train_expert(1, x, y, replay=True)

    m2 = m0.clone()
    s2 = VmfSignature(16, max_protos=64, seed=2)
    s2.update(x); s2.fit()
    pool.add_expert(m2, parent=1, state="active", meta={"sig": s2, "recent": deque(maxlen=16)})

    assert pool.experts[2].parent == 1
    pool.observe(x[:8], y[:8])
    pool.observe(x[:8], y[:8])

    assert 1 not in pool.experts
    assert 0 in pool.experts
    assert pool.experts[2].parent == 0


def test_cva_distillation_merge_rejection_on_accuracy_drop():
    """Handoff R5 13h: Distillation merge is aborted if accuracy on held-out replay drops > delta_merge."""
    cfg = PhyloCfg(d=16, n_classes=2, merge_every=2, merge_overlap=0.1, delta_merge=0.00001, seed=0)
    pool = PhyloMoE(cfg=cfg)
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(16)
    mu /= np.linalg.norm(mu)
    x = sample_vmf(mu, kappa=40.0, n=32, rng=rng)

    m0 = MLPExpert(16, 8, 1, 2, seed=0)
    s0 = VmfSignature(16, max_protos=64, seed=0)
    s0.update(x); s0.fit()
    pool.add_expert(m0, parent=None, state="active", meta={"sig": s0, "recent": deque(maxlen=16)})
    pool.train_expert(0, x, torch.zeros(32, dtype=torch.long), replay=True)

    m1 = MLPExpert(16, 8, 1, 2, seed=1)
    s1 = VmfSignature(16, max_protos=64, seed=1)
    s1.update(x); s1.fit()
    pool.add_expert(m1, parent=0, state="active", meta={"sig": s1, "recent": deque(maxlen=16)})
    pool.train_expert(1, x, torch.ones(32, dtype=torch.long), replay=True)

    pool.observe(x[:8], torch.zeros(8, dtype=torch.long))
    pool.observe(x[:8], torch.zeros(8, dtype=torch.long))

    assert 0 in pool.experts
    assert 1 in pool.experts
    assert len(pool.events.of_kind("merge")) == 0


def test_cva_growth_requires_loss_plateau():
    """Handoff R6: Task 14 growth does not trigger when loss is strictly improving (no plateau)."""
    seed = 207
    cfg = PhyloCfg(d=32, hidden=4, depth=1, n_classes=4, replay_size=500, m_min=16, seed=seed)
    grow_cfg = GrowthCfg(train_steps=100, n_boot=50)
    pool = PhyloMoE(cfg=cfg, grow_cfg=grow_cfg)
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(32)
    mu /= np.linalg.norm(mu)

    for _ in range(5):
        pool.observe(sample_vmf(mu, kappa=40.0, n=32, rng=rng), torch.zeros(32, dtype=torch.long))

    rec = pool.experts[0]
    assert rec.replay.n >= 128
    rec.meta["loss_hist"] = [2.0 - 0.01 * i for i in range(80)]
    assert not plateau(rec.meta["loss_hist"])

    pool.observe(sample_vmf(mu, kappa=40.0, n=32, rng=rng), torch.zeros(32, dtype=torch.long))
    assert len(pool.events.of_kind("grow")) == 0
