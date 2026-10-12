"""Tests for growmoe.pools.bnp (Task 15 Architecture B: BNP-MoE).

Tested properties (R7 handoff):
- VAEDensity: tanh MLP encoder/decoder, Gaussian decoder with clamped log-var [-8, 4].
  ELBO improves with training and is higher in-cluster than out.
- STM gating: samples below n_stm accumulate in STM without spawning.
- Scratch spawn: >= n_stm coherent samples spawn exactly one scratch expert (fresh classifier + fresh VAE).
- tau_b monotonic: larger tau_b => fewer spawns on the same stream.
- Protocol conformance: predict on empty pool returns uniform probs/unknown; inspect.signature has no GT.
- Determinism: same seed -> identical events.to_jsonl text.
- Unlabeled stream: y=None runs density-based discovery without classifier training calls.
"""

import inspect
import numpy as np
import pytest
import torch

from growmoe.config import BNPCfg
from growmoe.experts import MLPExpert
from growmoe.pools.bnp import BNPMoE, VAEDensity
from growmoe.streams import s6_long
from growmoe.vmf import sample_vmf


# --------------------------------------------------------------------------- 1. VAEDensity
def test_bnp_vae_elbo_improves_and_separates_clusters():
    seed = 101
    d, h, z = 64, 32, 8
    vae = VAEDensity(d=d, h=h, z=z, seed=seed)

    rng = np.random.default_rng(seed)
    mu0 = rng.standard_normal(d)
    mu0 /= np.linalg.norm(mu0)
    mu1 = rng.standard_normal(d)
    mu1 -= (mu1 @ mu0) * mu0
    mu1 /= np.linalg.norm(mu1)

    # In-cluster samples and out-of-cluster samples
    x_in_train = sample_vmf(mu0, kappa=40.0, n=200, rng=rng)
    x_in_test = sample_vmf(mu0, kappa=40.0, n=50, rng=rng)
    x_out_test = sample_vmf(mu1, kappa=40.0, n=50, rng=rng)

    # Prior to training: scoring ELBO on in-cluster test points
    elbo_pre = vae.elbo(x_in_test).detach().mean().item()

    # Train VAE on in-cluster training data
    opt = torch.optim.Adam(vae.parameters(), lr=2e-3)
    for _ in range(60):
        idx = torch.from_numpy(rng.integers(0, len(x_in_train), 32))
        vae.fit_step(x_in_train[idx], opt)

    # Post-training: scoring ELBO (posterior-mean z)
    elbo_post = vae.elbo(x_in_test).detach().mean().item()
    elbo_out = vae.elbo(x_out_test).detach().mean().item()

    # ELBO improves with training
    assert elbo_post > elbo_pre, f"ELBO did not improve: post {elbo_post} <= pre {elbo_pre}"

    # ELBO is significantly higher in-cluster than out
    assert elbo_post > elbo_out, f"In-cluster ELBO {elbo_post} <= out-cluster {elbo_out}"
    assert elbo_post - elbo_out > 5.0, f"Separation gap {elbo_post - elbo_out} <= 5.0 nats"


# --------------------------------------------------------------------------- 2. STM below threshold
def test_bnp_stm_below_threshold_never_spawns():
    seed = 102
    cfg = BNPCfg(d=64, n_stm=32, seed=seed)
    pool = BNPMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(64)
    mu /= np.linalg.norm(mu)
    x = sample_vmf(mu, kappa=40.0, n=20, rng=rng)
    y = torch.zeros(20, dtype=torch.long)

    rec = pool.observe(x, y)
    assert len(pool.experts) == 0
    assert len(pool.stm_x) == 20
    assert len(pool.events.of_kind("spawn")) == 0
    assert (rec.assigned == -1).all()


# --------------------------------------------------------------------------- 3. STM at threshold spawns scratch expert
def test_bnp_stm_at_threshold_spawns_exactly_one_scratch_expert():
    seed = 103
    cfg = BNPCfg(d=64, n_stm=32, seed=seed)
    pool = BNPMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(64)
    mu /= np.linalg.norm(mu)
    x = sample_vmf(mu, kappa=40.0, n=40, rng=rng)
    y = torch.zeros(40, dtype=torch.long)

    rec = pool.observe(x, y)
    assert len(pool.experts) == 1
    assert rec.spawned == [0]
    assert len(pool.stm_x) == 0

    spawns = pool.events.of_kind("spawn")
    assert len(spawns) == 1
    assert spawns[0].fields["expert"] == 0
    assert spawns[0].fields["init"] == "scratch"
    assert spawns[0].fields["parent"] is None
    assert spawns[0].fields["n_cluster"] == 40

    # Live expert contains both MLPExpert classifier and VAEDensity density
    assert isinstance(pool.experts[0].module, MLPExpert)
    assert isinstance(pool.experts[0].meta["vae"], VAEDensity)


# --------------------------------------------------------------------------- 4. Monotonic tau_b spawns
def test_bnp_tau_b_monotonic_spawns():
    seed = 104
    stream = s6_long(seed=seed, n_regimes=3, d=64, n_train=100, related_frac=0, revisit_frac=0)

    cfg_loose = BNPCfg(d=64, n_stm=32, tau_b=1.0, seed=seed)
    pool_loose = BNPMoE(cfg=cfg_loose)

    cfg_tight = BNPCfg(d=64, n_stm=32, tau_b=4.0, seed=seed)
    pool_tight = BNPMoE(cfg=cfg_tight)

    for x, y, _ in stream.algo_batches(batch_size=32, seed=seed):
        pool_loose.observe(x, y)
        pool_tight.observe(x, y)

    # Larger tau_b => lower novelty threshold => fewer spawns
    assert len(pool_loose.experts) >= len(pool_tight.experts), (
        f"Loose pool ({len(pool_loose.experts)}) has fewer experts than tight ({len(pool_tight.experts)})"
    )


# --------------------------------------------------------------------------- 5. Protocol conformance
def test_bnp_protocol_conformance():
    cfg = BNPCfg(d=64, n_classes=8)
    pool = BNPMoE(cfg=cfg)
    assert pool.name == "B_bnpmoe"

    x_dummy = torch.randn(4, 64)
    x_dummy = x_dummy / x_dummy.norm(dim=-1, keepdim=True)
    pred = pool.predict(x_dummy)

    assert (pred.expert == -1).all()
    assert pred.unknown.all()
    assert torch.allclose(pred.probs, torch.full((4, 8), 1.0 / 8))
    assert (pred.p_nov == 0.0).all()

    # Signatures must contain no ground-truth arguments
    obs_params = inspect.signature(BNPMoE.observe).parameters
    pred_params = inspect.signature(BNPMoE.predict).parameters
    assert list(obs_params.keys()) == ["self", "x", "y", "ctx"]
    assert list(pred_params.keys()) == ["self", "x", "ctx"]


# --------------------------------------------------------------------------- 6. Determinism
def test_bnp_determinism_identical_event_log(tmp_path):
    seed = 105
    cfg1 = BNPCfg(d=64, n_stm=32, seed=seed)
    pool1 = BNPMoE(cfg=cfg1)

    cfg2 = BNPCfg(d=64, n_stm=32, seed=seed)
    pool2 = BNPMoE(cfg=cfg2)

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


# --------------------------------------------------------------------------- 7. Unlabeled stream
def test_bnp_unlabeled_stream_no_classifier_training(monkeypatch):
    seed = 106
    def raise_if_trained(*args, **kwargs):
        raise RuntimeError("MLPExpert.train_step must not be called on unlabeled streams!")

    monkeypatch.setattr(MLPExpert, "train_step", raise_if_trained)

    cfg = BNPCfg(d=64, n_stm=32, seed=seed)
    pool = BNPMoE(cfg=cfg)

    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(64)
    mu /= np.linalg.norm(mu)
    x = sample_vmf(mu, kappa=40.0, n=50, rng=rng)

    rec = pool.observe(x, None)
    assert len(pool.experts) == 1
    assert torch.isnan(torch.tensor(rec.loss))
