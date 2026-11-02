import numpy as np
import pytest
import torch
import torch.nn.functional as F

from growmoe.experts import MLPExpert
from growmoe.pools.phylo import PhyloMoE
from fsmoe.grow.world import GrowthWorld
from fsmoe.nbpool import NbCfg, NbPool, NotebookExpert, nb_cfg

D, K = 32, 6
REC = {"inner_steps": 2, "q": 0.02, "alpha_spawn": 0.01, "m_min": 32}


def _pool(**nb):
    return NbPool(nb_cfg(K, 3, REC, d=D, **nb))


def _blob(c, n, seed, noise=0.8):
    g = torch.Generator().manual_seed(seed)
    centre = torch.zeros(D)
    centre[c] = 1.0
    return F.normalize(centre + noise * torch.randn(n, D, generator=g) / np.sqrt(D), dim=1)


def _feed(pool, regimes, n_batches=12, batch=64):
    for b in range(n_batches):
        for c in regimes:
            x = _blob(c, batch, 100 * c + b)
            y = (x[:, (c + 1) % D] > x[:, (c + 1) % D].median()).long() + 2 * (x[:, (c + 2) % D] > 0).long()
            pool.observe(x, y)


# ------------------------------------------------------------------ the notebook expert
def test_notebook_expert_answers_with_its_head_learns_and_round_trips():
    nb = NotebookExpert(D, K)
    x = _blob(0, 40, 1)
    assert torch.allclose(nb.proba(x), torch.full((40, K), 1.0 / K)) and int(nb.seen.sum()) == 0          # empty: uniform
    y = (x[:, 1] > x[:, 1].median()).long()
    loss = nb.train_step(x, y, None)
    assert np.isnan(loss) and nb.head.total == 40 and int(nb.seen.sum()) == 2
    assert float((nb.proba(x).argmax(1) == y).float().mean()) > 0.8
    sd = nb.state_dict()
    twin = NotebookExpert(D, K)
    twin.load_state_dict(sd)
    assert torch.allclose(twin.proba(x), nb.proba(x)) and twin.head.total == 40
    c = nb.clone()
    c.train_step(x, y, None)
    assert nb.head.total == 40 and c.head.total == 80                                                      # a clone is independent
    assert nb.num_params() == 0 and nb.inherit(0.5, 0.1).head.total == 0 and nb.shrink_perturb(0.5, 1).head.total == 0


def test_notebook_expert_can_be_given_an_optimiser_and_a_replay_batch():
    nb = NotebookExpert(D, K)
    opt = torch.optim.Adam(nb.parameters(), lr=1e-3)                                                        # the pool builds one for every expert
    x = _blob(0, 20, 2)
    nb.train_step(x, torch.zeros(20, dtype=torch.long), opt, replay=(x, torch.ones(20, dtype=torch.long)))
    assert nb.head.total == 40


# ------------------------------------------------------------------ the pool
def test_nb_first_needs_the_shipped_spawn_and_growth_rules():
    with pytest.raises(ValueError):
        nb_cfg(K, 0, {**REC, "spawn_rule": "probe"}, d=D)
    with pytest.raises(ValueError):
        nb_cfg(K, 0, {**REC, "growth_rule": "probe"}, d=D)
    assert isinstance(NbCfg, type) and nb_cfg(K, 0, REC, d=D).nb_first


def test_every_newborn_is_a_notebook_and_the_pool_still_answers():
    pool = _pool(nb_min=10 ** 9)                                           # judging off: all notebooks
    _feed(pool, (0, 5))
    mods = [r.module for r in pool.experts.values()]
    assert len(mods) >= 2 and all(isinstance(m, NotebookExpert) for m in mods)
    pred = pool.predict(_blob(0, 30, 999))
    assert pred.probs.shape == (30, K) and torch.allclose(pred.probs.sum(1), torch.ones(30), atol=1e-4)
    s = pool.summary()
    assert s["nb"]["notebook_experts"] == len(mods) and s["nb"]["net_experts"] == 0 and s["n_params"] == len(mods)     # one inert scalar each


def test_a_notebook_expert_holds_out_a_share_of_its_samples_and_never_trains_on_them():
    pool = _pool(nb_min=10 ** 9)
    _feed(pool, (0,), n_batches=10)
    rec = next(iter(pool.experts.values()))
    hold = rec.meta["nb"]["hold"]
    assert hold.n > 0 and rec.module.head.total + hold.n <= rec.n_assigned + 64 and rec.replay.n == min(rec.replay.capacity, rec.module.head.total)
    both = torch.cat([rec.replay.data[0], hold.data[0]])
    assert torch.unique(both, dim=0).shape[0] == both.shape[0]              # no row is both trained on and held out


def test_a_child_notebook_is_primed_once_with_its_founding_samples():
    pool = _pool(nb_min=10 ** 9)
    _feed(pool, (0,), n_batches=4)
    n_before = len(pool.experts)
    x = _blob(9, 80, 5)
    y = torch.randint(0, K, (80,))
    pool._handle_cluster(x, y, evidence=1.0)
    assert len(pool.experts) == n_before + 1
    child = pool.experts[max(pool.experts)]
    assert isinstance(child.module, NotebookExpert) and child.module.head.total == 80 and child.parent is not None      # once, not once per pretraining step


def test_a_kept_notebook_whose_replay_is_full_releases_its_held_out_samples_and_then_trains_on_everything():
    pool = _pool(nb_min=50, nb_steps=5, nb_sizes=((8, 1),), nb_margin=1.1, replay_size=200, nb_every=10 ** 9)
    _feed(pool, (0,), n_batches=12)
    rec = next(iter(pool.experts.values()))
    assert rec.meta["nb"]["hold"] is not None and rec.meta["nb"]["judged"] == 0 and rec.replay.n == 200
    pool._nb_judge(rec.id)
    assert rec.meta["nb"]["judged"] == 1 and rec.meta["nb"]["next_at"] == 2 * rec.replay.n and rec.meta["nb"]["hold"] is None
    before = rec.module.head.total
    x = _blob(0, 64, 77)
    pool.train_expert(rec.id, x, torch.randint(0, K, (64,)))
    assert rec.module.head.total == before + 64                                                              # nothing held out any more


def test_an_infant_notebook_is_not_judged():
    pool = _pool(nb_min=50, nb_steps=5, nb_sizes=((8, 1),))
    _feed(pool, (0,), n_batches=8)
    rec = next(iter(pool.experts.values()))
    rec.state = "infant"
    pool._nb_judge(rec.id)
    assert rec.meta["nb"]["judged"] == 0 and pool.nb_macs == 0.0


def test_judging_gives_a_network_to_a_regime_that_needs_one_and_keeps_the_notebook_for_one_that_does_not():
    w = GrowthWorld(1, ["wide", "lin"], per_phase=2, phase_batches=1)
    pool = NbPool(nb_cfg(K, 5, {"inner_steps": 2, "q": 0.02, "alpha_spawn": 0.01, "m_min": 32, "replay_size": 2000}, d=D, nb_min=300, nb_steps=250, nb_every=5, nb_sizes=((8, 1), (64, 1))))
    rng = np.random.default_rng(0)
    for step in range(60):
        for rid in (0, 1):
            x, y = w.regimes[rid].sample(128, rng)
            pool.observe(torch.from_numpy(x), torch.from_numpy(y))
    s = pool.summary()["nb"]
    kinds = [e.kind for e in pool.events if e.kind.startswith("nb_")]
    assert "nb_keep" in kinds and s["notebook_experts"] >= 1                                               # the linear regime keeps its notebook
    assert "nb_promote" in kinds and s["net_experts"] >= 1 and all(h in (8, 64) for h, _ in s["net_sizes"].values())      # the harder regime gets a network, of one of the candidate sizes
    for r in pool.experts.values():
        if not isinstance(r.module, NotebookExpert):
            assert isinstance(r.module, MLPExpert) and r.meta["lda"] is not None and r.meta["nb"]["hold"] is None and r.opt is not None


def test_notebook_experts_sit_out_merges_and_come_back_in_place():
    pool = _pool(nb_min=10 ** 9)
    _feed(pool, (0, 5), n_batches=6)
    d = pool.experts
    ids = list(d)
    pool._attempt_merges()
    assert pool.experts is d and list(pool.experts) == sorted(ids)


def test_a_judged_notebook_is_not_judged_again_before_its_data_has_doubled():
    pool = _pool(nb_min=50, nb_steps=5, nb_sizes=((8, 1),), nb_margin=1.1)                                  # no candidate can win by more than 1
    _feed(pool, (0,), n_batches=8)
    rec = next(iter(pool.experts.values()))
    pool._nb_judge(rec.id)
    judged = rec.meta["nb"]["judged"]
    macs = pool.nb_macs
    pool._nb_judge(rec.id)
    assert judged >= 1 and rec.meta["nb"]["judged"] == judged and pool.nb_macs == macs


def test_accounting_counts_the_notebook_heads_and_the_held_out_buffers():
    pool = _pool(nb_min=10 ** 9)
    _feed(pool, (0, 5), n_batches=6)
    params, macs, scalars = pool.extra_cost(D)
    heads = sum(r.module.head.n_scalars() for r in pool.experts.values())
    holds = sum(r.meta["nb"]["hold"].n * (D + 1) for r in pool.experts.values())
    assert scalars >= heads + holds and heads > 0 and holds > 0


def test_the_notebook_first_pool_turns_on_the_worse_than_chance_rule_of_the_spawn_test():
    assert nb_cfg(K, 0, REC, d=D).transfer_chance == 1.0
    assert nb_cfg(K, 0, REC, d=D, transfer_chance=2.0).transfer_chance == 2.0
    assert nb_cfg(K, 0, REC, d=D, nb_first=False).transfer_chance == 1.0                     # also for the shipped networks (the ablation arms of E-061)


def test_the_width_growth_step_is_skipped_for_a_notebook_and_kept_for_a_network(monkeypatch):
    pool = _pool(nb_min=10 ** 9)
    _feed(pool, (0,), n_batches=6)
    rec = next(iter(pool.experts.values()))
    calls = []
    monkeypatch.setattr(PhyloMoE, "_gap_growth_step", lambda self, r, x, y, pre: calls.append(r.id))
    x, y = _blob(0, 64, 3), torch.randint(0, K, (64,))
    pool._gap_growth_step(rec, x, y, torch.zeros(64))
    assert calls == []
    rec.module = MLPExpert(D, 8, 1, K)
    pool._gap_growth_step(rec, x, y, torch.zeros(64))
    assert calls == [rec.id]


def test_a_promotion_carries_the_notebook_over_as_the_experts_head():
    pool = _pool(nb_min=50, nb_steps=5, nb_sizes=((8, 1),), nb_margin=-1.0, nb_every=10 ** 9)             # any candidate "wins"
    _feed(pool, (0,), n_batches=10)
    rec = next(iter(pool.experts.values()))
    head, total = rec.module.head, rec.module.head.total
    pool._nb_judge(rec.id)
    assert isinstance(rec.module, MLPExpert) and rec.meta["lda"] is head and head.total == total and rec.meta["nb"]["hold"] is None


def test_the_candidate_is_judged_as_it_would_answer_blended_with_the_notebook():
    w = GrowthWorld(4, ["wide"], per_phase=1, phase_batches=1)
    rng = np.random.default_rng(0)
    x, y = (torch.from_numpy(a) for a in w.regimes[0].sample(7000, rng))
    out = {}
    for blend_w in (0.0, 1.0):
        pool = NbPool(nb_cfg(K, 5, {"inner_steps": 2, "q": 0.02, "alpha_spawn": 0.01, "m_min": 32, "replay_size": 6000, "lda_blend": max(blend_w, 0.01)}, d=D, nb_min=100, nb_steps=300, nb_sizes=((64, 1),), nb_margin=0.001, nb_every=10 ** 9))
        pool.observe(x[:64], y[:64])
        rec = next(iter(pool.experts.values()))
        rec.module.head = type(rec.module.head)(D, K, 0.1)
        rec.module.head.update(x[:6000], y[:6000])
        rec.replay.add(x[:6000], y[:6000])
        rec.meta["nb"]["hold"].add(x[6000:], y[6000:])
        rec.state = "active"
        pool.cfg.lda_blend = blend_w
        pool._nb_judge(rec.id)
        out[blend_w] = [e for e in pool.events if e.kind in ("nb_promote", "nb_keep")][-1].kind
    assert out[0.0] == "nb_promote" and out[1.0] == "nb_keep"                                                # a pure-notebook answer cannot be beaten by blending with it fully


def test_notebook_experts_return_to_the_pool_in_id_order_when_a_network_expert_sits_between_them():
    pool = _pool(nb_min=10 ** 9)
    _feed(pool, (0, 5, 9), n_batches=6)
    ids = sorted(pool.experts)
    assert len(ids) >= 3
    pool.experts[ids[1]].module = MLPExpert(D, 8, 1, K)
    d = pool.experts
    pool._attempt_merges()
    assert pool.experts is d and list(pool.experts) == ids


def test_a_cluster_that_joins_a_notebook_parent_unchanged_enters_its_head_once(monkeypatch):
    pool = _pool(nb_min=10 ** 9)
    _feed(pool, (0,), n_batches=4)
    eid = next(iter(pool.experts))
    before = pool.experts[eid].module.head.total

    def fake(self, x, y, *a, **k):
        self.events.add(kind="extend", step=self.step, expert=eid)
    monkeypatch.setattr(PhyloMoE, "_handle_cluster", fake)
    pool._handle_cluster(_blob(0, 50, 8), torch.randint(0, K, (50,)))
    assert pool.experts[eid].module.head.total == before + 50


def test_a_notebook_never_skips_a_batch_but_a_network_expert_of_the_same_pool_may():
    pool = _pool(skip_known=1.0, skip_min_assigned=0, nb_every=10 ** 9)                                  # never judged, so every expert stays a notebook
    _feed(pool, [0, 1])
    assert all(isinstance(r.module, NotebookExpert) for r in pool.experts.values())
    nb = next(r for r in pool.experts.values() if isinstance(r.module, NotebookExpert))
    assert pool._known_batch(nb, torch.zeros(4, dtype=torch.float64)) is False                         # a notebook's training IS its closed-form absorb
    assert pool.skip_stats["skipped"] == 0 and pool.skip_stats["trained"] >= 1
    net = next(iter(pool.experts.values()))
    net.module = MLPExpert(D, 8, 1, K, seed=0)                                                         # the same record holding a network
    net.state, net.n_assigned = "active", 10 ** 6
    assert pool._known_batch(net, torch.zeros(4, dtype=torch.float64)) is True
