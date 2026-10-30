"""The systems under test (unimoe/systems.py): UniMoE variants with their declared switches, and the baselines at equal memory, all behind one pool protocol."""

import numpy as np
import pytest
import torch

from unimoe import systems as SY

D, C = 128, 20


def toy(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.nn.functional.normalize(torch.randn(C, D, generator=g), dim=1)
    y = torch.randint(0, C, (n,), generator=g)
    x = torch.nn.functional.normalize(centers[y] + 0.15 * torch.randn(n, D, generator=g), dim=1)
    return x, y


def test_every_system_runs_a_few_batches_and_answers_with_probabilities():
    x, y = toy()
    ctx = (y % 3).long()
    for name in SY.UNIMOE + SY.BASELINES:
        kw = {"mem_bytes": 6_000_000, "total_samples": 1000} if name in ("replay_eqmem", "knn_eqmem") else {}
        pool = SY.make_system(name, C, seed=1, recipe={"inner_steps": 2}, **kw)
        for lo in range(0, 192, 64):
            pool.observe(x[lo:lo + 64], y[lo:lo + 64], ctx=ctx[lo:lo + 64] if getattr(pool, "uses_regime", False) else None)
        out = pool.predict(x[192:], ctx=ctx[192:] if getattr(pool, "uses_regime", False) else None)
        assert out.probs.shape == (64, C), name
        assert torch.allclose(out.probs.sum(dim=1), torch.ones(64), atol=1e-3), name
        s = pool.summary()
        assert s["n_experts"] >= 1 and s["n_params"] >= 0, name
        assert hasattr(pool, "events"), name


def test_the_unimoe_variants_have_the_switches_the_design_declares_and_the_recipe_and_seed_rule_apply():
    shipped = SY.make_system("shipped", C, seed=3, recipe={"inner_steps": 7}).cfg
    real = SY.make_system("U-real", C, seed=3, recipe={"inner_steps": 7}).cfg
    deep = SY.make_system("U-deep", C, seed=3, recipe={"inner_steps": 7}).cfg
    assert not shipped.set_gate and shipped.growth_rule == "probe" and not shipped.sig_dedupe
    assert real.set_gate and real.set_gate_confirm == 2 and real.growth_rule == "gap" and real.sig_dedupe and real.infant_relax == 1.0
    assert deep.set_gate and deep.growth_rule == "probe" and deep.sig_dedupe
    for c in (shipped, real, deep):
        assert c.inner_steps == 7 and c.seed == 3 + 7 and c.n_classes == C and c.d == D          # the pool seed differs from the stream seed (C-75)
        assert c.route_rule == "accept_then_logf" and c.router and c.spawn_rule == "transfer" and c.batch_vote
    assert SY.make_system("U-recipe", C, seed=3, recipe={"inner_steps": 20, "replay_size": 512}).cfg.replay_size == 512


def test_a_replay_network_at_equal_memory_never_exceeds_the_budget_and_refuses_when_there_is_no_room():
    budget = 6_000_000
    pool = SY.make_system("replay_eqmem", C, seed=0, recipe={"hidden": 256, "inner_steps": 1}, mem_bytes=budget, total_samples=100_000)
    params = pool.summary()["n_params"]
    assert 4 * params + 4 * D * pool.buffer.capacity <= budget
    assert pool.buffer.capacity >= 512
    with pytest.raises(ValueError):
        SY.make_system("replay_eqmem", C, seed=0, recipe={"hidden": 4096, "inner_steps": 1}, mem_bytes=1_000_000, total_samples=100_000)    # the weights alone are larger than the budget
    knn = SY.make_system("knn_eqmem", C, seed=0, mem_bytes=budget)
    assert knn.capacity == budget // (4 * D)


def test_the_regime_told_experts_are_an_upper_bound_that_needs_the_regime_and_says_so():
    x, y = toy(512, seed=1)
    regime = (torch.arange(512) % 2).long()
    y2 = torch.where(regime == 1, (y + 5) % C, y)                                                # regime 1 relabels the same inputs: one network cannot know which map applies
    oracle = SY.make_system("oracle_experts", C, seed=0, recipe={"inner_steps": 30})
    single = SY.make_system("finetune", C, seed=0, recipe={"inner_steps": 30})
    assert oracle.uses_regime and not getattr(single, "uses_regime", False)
    for lo in range(0, 384, 64):
        oracle.observe(x[lo:lo + 64], y2[lo:lo + 64], ctx=regime[lo:lo + 64])
        single.observe(x[lo:lo + 64], y2[lo:lo + 64])
    acc = lambda p, **kw: float((p.predict(x[384:], **kw).probs.argmax(1) == y2[384:]).float().mean())
    assert acc(oracle, ctx=regime[384:]) > acc(single) + 0.1


def test_an_unknown_system_name_is_refused():
    with pytest.raises(ValueError):
        SY.make_system("nonsense", C, seed=0)
