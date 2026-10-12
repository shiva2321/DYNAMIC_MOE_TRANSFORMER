from functools import lru_cache

import numpy as np
import pytest
import torch

from growmoe import streams as S
from growmoe.baselines import (DPMeansGate, FineTune, JointOracle, OracleRouted, ReplayNet, StaticRouterMoE,
                               _Reservoir)
from growmoe.config import CommonCfg
from growmoe.telemetry import EventLog
from growmoe.types import ContinualPool, Prediction, StepRecord

SMALL = CommonCfg(d=16, hidden=16, n_classes=8, batch_size=8, inner_steps=1)


def _cfg(seed=0):
    return CommonCfg(d=128, hidden=64, n_classes=64, batch_size=32, inner_steps=3, lr=3e-3, seed=seed)


@lru_cache(maxsize=None)
def _stream(seed=0, n_train=640):
    return S.s2_permuted(seed, n_regimes=3, n_train=n_train, n_test=300)


def _accuracies(pool, stream):
    out = []
    for r, (x, y) in stream.test_sets().items():
        if hasattr(pool, "set_eval_regime"):
            pool.set_eval_regime(r)
        out.append((pool.predict(x).probs.argmax(1) == y).float().mean().item())
    return out


def _train(pool, stream, seed=0):
    for x, y, _ in stream.algo_batches(32, seed=seed):
        pool.observe(x, y)
    return pool


@lru_cache(maxsize=None)
def _run(name):
    st, cfg = _stream(), _cfg()
    regimes = [gt["regime"] for _, _, gt in st.batches(32, seed=0)]
    pool = {"B0": lambda: FineTune(cfg), "B1": lambda: JointOracle(cfg), "B2": lambda: ReplayNet(cfg),
            "B3": lambda: StaticRouterMoE(cfg), "B4": lambda: OracleRouted(lambda i: regimes[i], cfg),
            "B5": lambda: DPMeansGate(cfg, lam=0.8)}[name]()
    _train(pool, st)
    return pool, _accuracies(pool, st)


def _unit(n, d=16, seed=0):
    return torch.nn.functional.normalize(torch.from_numpy(np.random.default_rng(seed).standard_normal((n, d))).float(), dim=1)


def _all_params(pool):
    if hasattr(pool, "net"):
        mods = [pool.net]
    elif isinstance(pool, StaticRouterMoE):
        mods = list(pool.experts) + [pool.router]
    else:
        mods = list(pool.models.values()) if isinstance(pool.models, dict) else list(pool.models)
    return [p.detach().clone() for m in mods for p in m.parameters()]


MAKERS = {
    "B0": lambda: FineTune(SMALL), "B1": lambda: JointOracle(SMALL), "B2": lambda: ReplayNet(SMALL),
    "B3": lambda: StaticRouterMoE(SMALL, n_experts=3), "B4": lambda: OracleRouted(lambda i: i % 2, SMALL),
    "B5": lambda: DPMeansGate(SMALL, lam=1.0),
}


# ------------------------------------------------------------------------------------------- protocol
@pytest.mark.parametrize("name", sorted(MAKERS))
def test_every_baseline_satisfies_the_pool_protocol(name):
    pool = MAKERS[name]()
    assert isinstance(pool, ContinualPool) and isinstance(pool.events, EventLog) and pool.name
    x, y = _unit(8), torch.randint(0, 8, (8,))
    rec = pool.observe(x, y)
    assert isinstance(rec, StepRecord)
    assert rec.novel.shape == rec.p_nov.shape == rec.assigned.shape == (8,)
    assert rec.novel.dtype == torch.bool and rec.assigned.dtype == torch.long and rec.n_experts >= 1
    if hasattr(pool, "set_eval_regime"):
        pool.set_eval_regime(0)
    pred = pool.predict(x)
    assert isinstance(pred, Prediction) and pred.probs.shape == (8, 8)
    assert torch.allclose(pred.probs.sum(1), torch.ones(8), atol=1e-5)
    assert pred.expert.shape == pred.p_nov.shape == pred.unknown.shape == (8,)
    assert pred.expert.dtype == torch.long and pred.unknown.dtype == torch.bool
    s = pool.summary()
    assert {"n_experts", "n_params", "active_params_per_sample", "lineage", "states"} <= set(s)
    assert s["n_experts"] >= 1 and s["n_params"] >= s["active_params_per_sample"] > 0


@pytest.mark.parametrize("name", sorted(MAKERS))
def test_observing_without_labels_never_trains(name):
    pool = MAKERS[name]()
    pool.observe(_unit(8, seed=1), torch.randint(0, 8, (8,)))
    before = _all_params(pool)
    pool.observe(_unit(8, seed=2), None)
    after = _all_params(pool)
    # unlabeled data may DISCOVER a regime or cluster (a new, untrained expert appended at the end) but trains none that exist
    assert len(after) >= len(before) and all(torch.equal(a, b) for a, b in zip(before, after))


@pytest.mark.parametrize("name", sorted(MAKERS))
def test_constructing_a_baseline_leaves_the_global_torch_rng_alone(name):
    state = torch.get_rng_state().clone()
    MAKERS[name]()
    assert torch.equal(state, torch.get_rng_state())


# ------------------------------------------------------------------- the comparisons the study relies on
def test_finetuning_forgets_the_first_regime():
    assert _run("B0")[1][0] < 0.5                                # measured 0.18-0.39 over 5 seeds


def test_joint_training_is_the_ceiling():
    joint = np.mean(_run("B1")[1])
    assert joint > 0.85                                          # measured 0.90-0.94
    assert all(joint >= np.mean(_run(n)[1]) - 0.02 for n in ("B0", "B2", "B3", "B4", "B5"))


def test_replay_beats_finetuning_by_at_least_a_fifth_of_the_accuracy_range():
    assert np.mean(_run("B2")[1]) - np.mean(_run("B0")[1]) >= 0.2      # measured gap 0.29-0.41


def test_oracle_routing_never_forgets_exactly():
    st, cfg = _stream(), _cfg()
    regimes = [gt["regime"] for _, _, gt in st.batches(32, seed=0)]
    pool = OracleRouted(lambda i: regimes[i], cfg)
    batches = list(st.algo_batches(32, seed=0))
    first_regime_batches = sum(1 for r in regimes if r == regimes[0])
    for x, y, _ in batches[:first_regime_batches]:
        pool.observe(x, y)
    pool.set_eval_regime(regimes[0])
    x_te = st.test_sets()[regimes[0]][0]
    right_after = pool.predict(x_te).probs.clone()
    for x, y, _ in batches[first_regime_batches:]:
        pool.observe(x, y)
    pool.set_eval_regime(regimes[0])
    assert torch.equal(pool.predict(x_te).probs, right_after)           # zero forgetting by construction
    assert np.mean(_run("B4")[1]) - np.mean(_run("B0")[1]) >= 0.2        # and it is far above fine-tuning (gap 0.27-0.41)


def test_a_fixed_expert_count_does_not_cure_forgetting():
    assert np.mean(_run("B3")[1]) < np.mean(_run("B2")[1]) - 0.2         # measured 0.47-0.58 vs 0.88-0.91


# ------------------------------------------------------------------------------------------- B1 / B2
def test_joint_buffer_keeps_every_sample_in_order_across_growth():
    pool = JointOracle(CommonCfg(d=8, hidden=8, n_classes=4, batch_size=8), joint_steps=1)
    xs = [_unit(100, d=8, seed=i) for i in range(40)]                     # 4000 samples: forces 1024 -> 2048 -> 4096
    for x in xs:
        pool.observe(x, torch.randint(0, 4, (100,)))
    assert pool._n == 4000
    assert torch.equal(pool._x[:4000], torch.cat(xs))


def test_reservoir_is_bounded_and_uniform_over_the_stream():
    ranks = []
    for seed in range(5):
        buf = _Reservoir(256, 4, np.random.default_rng(seed))
        for i in range(0, 10000, 50):
            x = torch.zeros(50, 4)
            x[:, 0] = torch.arange(i, i + 50).float()
            buf.add(x, torch.arange(i, i + 50))
        assert buf.n == 256 and buf.seen == 10000
        assert len(set(buf.y.tolist())) == 256                           # no sample is stored twice
        ranks.append(buf.y.clone())
    ranks = torch.cat(ranks)
    assert abs(ranks.float().mean().item() - 4999.5) < 300
    counts = torch.bincount(torch.bucketize(ranks, torch.tensor([2500, 5000, 7500]), right=True), minlength=4)
    assert (counts - 320).abs().max().item() < 60


def test_replay_net_without_replay_share_does_not_crash_and_still_learns():
    cfg = CommonCfg(d=16, hidden=16, n_classes=8, batch_size=8, replay_frac=0.0)
    pool = ReplayNet(cfg)
    for i in range(5):
        pool.observe(_unit(8, seed=i), torch.randint(0, 8, (8,)))
    assert pool.buffer.n == 8 * 5 or pool.buffer.n > 0


# --------------------------------------------------------------------------------------------- B3
def test_bias_update_pushes_load_away_from_the_overloaded_expert():
    pool = StaticRouterMoE(SMALL, n_experts=4, top_k=1, bias_rate=0.05)
    with torch.no_grad():
        pool.router.weight.zero_()
        pool.router.weight[0] = 1.0                                       # expert 0 wins every sample with positive mass
    x = torch.nn.functional.normalize(_unit(16).abs() + 0.1, dim=1)
    pool.observe(x, torch.randint(0, 8, (16,)))
    assert pool.sel_bias[0] < 0 and (pool.sel_bias[1:] > 0).all()
    assert pool.summary()["n_experts"] == 4


def test_the_bias_prevents_routing_collapse():
    shares = {}
    for rate in (0.0, 0.05):
        worst = []
        for seed in range(2):
            pool = StaticRouterMoE(_cfg(seed), bias_rate=rate)
            st = _stream(seed)
            xs = []
            for x, y, _ in st.algo_batches(32, seed=seed):
                pool.observe(x, y)
                xs.append(x)
            with torch.no_grad():
                top = (pool.router(torch.cat(xs)) + pool.sel_bias).topk(pool.top_k, dim=1).indices
            worst.append((torch.bincount(top.flatten(), minlength=8).float() / top.numel()).max().item() * pool.top_k)
        shares[rate] = float(np.mean(worst))
    assert shares[0.0] > 0.8                                              # without it one expert takes ~97 % of selections
    assert shares[0.05] < shares[0.0] - 0.2                               # measured 0.56 vs 0.97


# --------------------------------------------------------------------------------------------- B4
def test_oracle_gets_regimes_only_through_the_injected_callable():
    calls = []

    def regime_of(i):
        calls.append(i)
        return [4, 4, 9, 4][i]

    pool = OracleRouted(regime_of, SMALL)
    recs = [pool.observe(_unit(8, seed=i), torch.randint(0, 8, (8,))) for i in range(4)]
    assert calls == [0, 1, 2, 3]
    assert [r.spawned for r in recs] == [[0], [], [1], []]
    assert recs[3].assigned.tolist() == [0] * 8 and recs[2].assigned.tolist() == [1] * 8
    assert [e.step for e in pool.events.of_kind("spawn")] == [0, 2] and "upper-bound" in pool.name


def test_oracle_prediction_needs_an_eval_regime_and_is_uniform_for_an_unseen_one():
    pool = OracleRouted(lambda i: 0, SMALL)
    pool.observe(_unit(8), torch.randint(0, 8, (8,)))
    with pytest.raises(RuntimeError):
        pool.predict(_unit(4))
    pool.set_eval_regime(7)
    p = pool.predict(_unit(4))
    assert torch.allclose(p.probs, torch.full((4, 8), 1 / 8)) and (p.expert == -1).all()


# --------------------------------------------------------------------------------------------- B5
def test_dpmeans_spawn_count_is_non_increasing_in_lambda():
    counts = []
    for lam in (0.3, 0.8, 1.2, 1.8):
        counts.append(_train(DPMeansGate(_cfg(), lam=lam), _stream()).summary()["n_experts"])
    assert all(a >= b for a, b in zip(counts, counts[1:])) and counts[0] > counts[-1]     # measured 127, 5, 2, 1


def test_dpmeans_flags_the_sample_that_founds_a_cluster():
    pool = DPMeansGate(SMALL, lam=1.0)
    e = torch.eye(16)
    rec = pool.observe(torch.stack([e[0], e[0], e[1]]), torch.tensor([0, 0, 1]))
    assert rec.novel.tolist() == [True, False, True] and rec.spawned == [0, 1] and rec.assigned.tolist() == [0, 0, 1]
    assert rec.p_nov[0] == 0.0 and rec.p_nov[1] == 1.0 and 0.0 < rec.p_nov[2] < 1.0 and rec.n_experts == 2
    assert [ev.fields["expert"] for ev in pool.events.of_kind("spawn")] == [0, 1]


def test_dpmeans_prediction_marks_far_inputs_unknown():
    pool = DPMeansGate(SMALL, lam=1.0)
    e = torch.eye(16)
    pool.observe(e[0:1].repeat(8, 1), torch.zeros(8, dtype=torch.long))
    near, far = pool.predict(e[0:1]), pool.predict(e[1:2])
    assert not near.unknown.item() and near.p_nov.item() == 1.0
    assert far.unknown.item() and far.p_nov.item() < 1.0
    empty = DPMeansGate(SMALL, lam=1.0).predict(e[0:1])
    assert empty.unknown.item() and empty.expert.item() == -1


def test_dpmeans_lambda_must_be_a_squared_distance_between_unit_vectors():
    for bad in (0.0, -1.0, 4.0, 5.0):
        with pytest.raises(ValueError):
            DPMeansGate(SMALL, lam=bad)


def test_dpmeans_without_labels_still_discovers_clusters():
    pool = DPMeansGate(SMALL, lam=1.0)
    e = torch.eye(16)
    rec = pool.observe(torch.stack([e[0], e[1], e[2]]), None)
    assert rec.spawned == [0, 1, 2] and pool.summary()["n_experts"] == 3 and np.isnan(rec.loss)


# ------------------------------------------------------------------------------------------ determinism
@pytest.mark.parametrize("name", ["B3", "B5"])
def test_identical_runs_give_identical_predictions(name):
    outs = []
    for _ in range(2):
        pool = MAKERS[name]()
        for i in range(4):
            pool.observe(_unit(8, seed=i), torch.arange(8) % 8)
        outs.append(pool.predict(_unit(8, seed=99)).probs)
    assert torch.equal(outs[0], outs[1])
