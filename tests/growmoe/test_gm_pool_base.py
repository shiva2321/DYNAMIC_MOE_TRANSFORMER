import numpy as np
import pytest
import torch

from growmoe.config import CommonCfg
from growmoe.experts import MLPExpert
from growmoe.pools.base import BasePool, ExpertRecord

CFG = CommonCfg(d=16, hidden=8, n_classes=8, batch_size=8, lr=1e-2, replay_size=32, seed=0)


def _module(seed=0):
    return MLPExpert(CFG.d, CFG.hidden, CFG.depth, CFG.n_classes, seed=seed)


def _units(n, seed=0, d=16):
    z = np.random.default_rng(seed).standard_normal((n, d))
    return torch.from_numpy(z / np.linalg.norm(z, axis=1, keepdims=True)).float()


def _tree():
    """0 -> 1 -> 2, 0 -> 3."""
    pool = BasePool(CFG)
    a = pool.add_expert(_module(0))
    b = pool.add_expert(_module(1), parent=a.id)
    c = pool.add_expert(_module(2), parent=b.id)
    d = pool.add_expert(_module(3), parent=a.id)
    return pool, a, b, c, d


# ---------------------------------------------------------------------------------------------- registry
def test_ids_are_unique_monotone_and_never_reused_after_removal():
    pool = BasePool(CFG)
    ids = [pool.add_expert(_module(i)).id for i in range(3)]
    assert ids == [0, 1, 2]
    pool.remove_expert(1)
    assert pool.add_expert(_module(9)).id == 3
    assert sorted(pool.experts) == [0, 2, 3]


def test_a_record_carries_the_bookkeeping_the_architectures_share():
    pool = BasePool(CFG)
    pool.step = 7
    rec = pool.add_expert(_module(), state="infant")
    assert isinstance(rec, ExpertRecord) and rec.birth_step == 7 and rec.state == "infant" and rec.parent is None
    assert rec.n_assigned == 0 and rec.opt is not None and rec.meta == {}
    assert rec.replay_x.shape == (0, 16) and rec.replay_y.shape == (0,)
    with pytest.raises(ValueError):
        pool.add_expert(_module(), state="zombie")


def test_step_counter_and_events_start_empty():
    pool = BasePool(CFG)
    assert pool.step == 0 and len(pool.events) == 0
    pool.tick()
    pool.tick()
    assert pool.step == 2


# ----------------------------------------------------------------------------------------------- lineage
def test_ancestors_on_a_hand_built_tree():
    pool, a, b, c, d = _tree()
    assert pool.ancestors(a.id) == [] and pool.ancestors(b.id) == [a.id]
    assert pool.ancestors(c.id) == [b.id, a.id] and pool.ancestors(d.id) == [a.id]


def test_a_parent_must_already_exist():
    pool = BasePool(CFG)
    with pytest.raises(KeyError):
        pool.add_expert(_module(), parent=99)
    with pytest.raises(KeyError):
        pool.ancestors(99)


def test_inserting_a_cycle_raises_and_leaves_the_tree_unchanged():
    pool, a, b, c, d = _tree()
    for child, parent in ((a.id, c.id), (a.id, b.id), (b.id, b.id)):
        with pytest.raises(ValueError):
            pool.set_parent(child, parent)
    assert pool.ancestors(c.id) == [b.id, a.id] and pool.experts[a.id].parent is None
    pool.set_parent(d.id, c.id)                                        # a legal re-parenting still works
    assert pool.ancestors(d.id) == [c.id, b.id, a.id]


def test_lineage_survives_removing_an_ancestor():
    pool, a, b, c, d = _tree()
    pool.remove_expert(b.id)
    assert pool.ancestors(c.id) == [b.id, a.id]
    assert b.id not in pool.experts and c.id in pool.experts


# ------------------------------------------------------------------------------------------ optimisers
def test_every_expert_has_its_own_optimiser_over_exactly_its_own_parameters():
    pool, a, b, c, d = _tree()
    for rec in (a, b, c, d):
        owned = {id(p) for p in rec.module.parameters()}
        assert {id(p) for g in rec.opt.param_groups for p in g["params"]} == owned
        assert rec.opt.param_groups[0]["lr"] == pytest.approx(CFG.lr)
    assert len({id(r.opt) for r in (a, b, c, d)}) == 4


def test_removing_an_expert_detaches_only_its_own_optimiser():
    pool, a, b, c, d = _tree()
    x, y = _units(8), torch.randint(0, 8, (8,))
    pool.train_expert(a.id, x, y)
    pool.train_expert(d.id, x, y)
    state_a = [t.clone() for st in a.opt.state.values() for t in st.values() if torch.is_tensor(t)]
    opt_a, opt_d = a.opt, d.opt
    pool.remove_expert(b.id)
    assert pool.experts[a.id].opt is opt_a and pool.experts[d.id].opt is opt_d
    after = [t for st in a.opt.state.values() for t in st.values() if torch.is_tensor(t)]
    assert all(torch.equal(u, v) for u, v in zip(state_a, after))
    assert b.id not in pool.experts


def test_lr_multiplier_scales_only_that_experts_learning_rate():
    pool, a, b, *_ = _tree()
    pool.set_lr_mult(b.id, 0.25)
    assert b.opt.param_groups[0]["lr"] == pytest.approx(0.25 * CFG.lr) and a.opt.param_groups[0]["lr"] == pytest.approx(CFG.lr)


# ----------------------------------------------------------------------------------------------- training
def test_train_expert_trains_replays_counts_and_updates_the_seen_mask():
    pool = BasePool(CFG)
    rec = pool.add_expert(_module())
    x, y = _units(8, 1), torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    before = [p.detach().clone() for p in rec.module.parameters()]
    loss = pool.train_expert(rec.id, x, y)
    assert np.isfinite(loss) and rec.n_assigned == 8 and rec.last_active == pool.step
    assert any(not torch.equal(a, b) for a, b in zip(before, rec.module.parameters()))
    assert rec.module.seen.tolist() == [True] * 4 + [False] * 4
    assert rec.replay_x.shape == (8, 16)                               # the batch entered the reservoir after training on it
    pool.train_expert(rec.id, _units(40, 2), torch.randint(0, 4, (40,)))
    assert rec.replay_x.shape[0] == CFG.replay_size and rec.n_assigned == 48


def test_train_expert_without_replay_skips_the_reservoir():
    pool = BasePool(CFG)
    rec = pool.add_expert(_module())
    pool.train_expert(rec.id, _units(8), torch.randint(0, 8, (8,)), replay=False)
    assert rec.replay_x.shape[0] == 0 and rec.n_assigned == 8


# ------------------------------------------------------------------------------------------------ summary
def test_summary_counts_parameters_states_and_lineage():
    pool, a, b, c, d = _tree()
    pool.set_state(c.id, "dormant")
    s = pool.summary()
    assert s["n_experts"] == 4
    assert s["n_params"] == sum(p.numel() for r in (a, b, c, d) for p in r.module.parameters())
    assert s["lineage"] == {a.id: None, b.id: a.id, c.id: b.id, d.id: a.id}
    assert s["states"][c.id] == "dormant" and s["state_counts"] == {"active": 3, "dormant": 1}
    assert s["active_params_per_sample"] == pytest.approx(s["n_params"] / 4)
    pool.remove_expert(d.id)
    assert pool.summary()["n_params"] == sum(p.numel() for r in (a, b, c) for p in r.module.parameters())


def test_summary_of_an_empty_pool_is_well_formed():
    s = BasePool(CFG).summary()
    assert s["n_experts"] == 0 and s["n_params"] == 0 and s["active_params_per_sample"] == 0 and s["lineage"] == {}


def test_state_changes_are_validated():
    pool, a, *_ = _tree()
    with pytest.raises(ValueError):
        pool.set_state(a.id, "asleep")
    with pytest.raises(KeyError):
        pool.set_state(99, "active")
