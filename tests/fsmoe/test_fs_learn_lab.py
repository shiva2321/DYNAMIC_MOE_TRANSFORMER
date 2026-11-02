import numpy as np
import pytest
import torch

from fsmoe.learn_lab import VerifiedLearner


class Rec:
    def __init__(self):
        self.calls = []

    def __call__(self, x, y):
        self.calls.append((x.clone(), y.clone()))


def _learner(mode, **kw):
    rec = Rec()
    mem_x = torch.randn(50, 4)
    mem_y = torch.randint(0, 3, (50,))
    return rec, VerifiedLearner(rec, mode, {0, 1, 2}, mem_x, mem_y, np.random.default_rng(0), **kw)


def test_interleaved_sends_mixed_chunks_in_order_of_arrival():
    rec, L = _learner("interleaved", chunk=4)
    for i in range(8):
        L.add(torch.full((4,), float(i)), 10 + i % 3)
    L.finish()
    assert len(rec.calls) == 2 and all(len(y) == 4 for _, y in rec.calls) and len(set(rec.calls[0][1].tolist())) > 1


def test_focused_teaches_a_new_skill_as_its_own_batch_every_m_items():
    rec, L = _learner("focused", m=5, passes=3, batch=8)
    for i in range(12):
        L.add(torch.randn(4), 7)
    assert L.n_consolidations == 2 and len(rec.calls) == 2 * 3                       # items 5 and 10 trigger, 3 passes each
    assert all(set(y.tolist()) == {7} and len(y) == 8 for _, y in rec.calls)
    L.finish()
    assert L.n_consolidations == 2                                                    # 2 items left over: fewer than 3, not taught


def test_focused_replay_mixes_half_known_items_into_every_batch_and_remembers_new_ones():
    rec, L = _learner("focused_replay", m=5, passes=2, batch=8)
    for _ in range(5):
        L.add(torch.randn(4), 9)
    assert len(rec.calls) == 2
    for x, y in rec.calls:
        assert len(y) == 16 and int((y == 9).sum()) == 8 and set(y[8:].tolist()) <= {0, 1, 2}
    assert int((L.mem_y == 9).sum()) == 5 or len(L.mem_y) == L.mem_cap


def test_known_skills_always_take_the_interleaved_path_and_enter_the_memory():
    rec, L = _learner("focused_replay", chunk=3, mem_cap=60)
    for i in range(3):
        L.add(torch.randn(4), 1)
    assert L.n_chunks == 1 and len(L.mem_y) == 53
    assert L.n_consolidations == 0


def test_memory_stays_bounded_and_is_counted():
    rec, L = _learner("focused_replay", chunk=100, mem_cap=50)
    for _ in range(30):
        L.add(torch.randn(4), 2)
    assert len(L.mem_y) == 50 and L.memory_bytes() == 4 * 50 * 4


class _FakePool:
    """Predicts a fixed distribution that never contains class 3; records what it was shown."""
    name = "fake"
    events = []

    def __init__(self):
        self.seen = []

    def observe(self, x, y=None, ctx=None):
        self.seen.append(y)

    def predict(self, x, ctx=None):
        from growmoe.types import Prediction
        probs = torch.tensor([0.4, 0.3, 0.3, 0.0]).repeat(len(x), 1)
        return Prediction(probs=probs, expert=torch.zeros(len(x), dtype=torch.long), p_nov=torch.ones(len(x)), unknown=torch.zeros(len(x), dtype=torch.bool))

    def summary(self):
        return {"n_experts": 1}


def test_global_fast_lets_a_class_the_pool_never_saw_win_after_a_few_examples():
    from fsmoe.learn_lab import GlobalFast
    pool = GlobalFast(_FakePool(), 4, 4, lam=0.5)
    x = torch.nn.functional.normalize(torch.randn(8, 4), dim=1)
    assert pool.predict(x).probs.argmax(1).tolist() == [0] * 8                       # no closed-form statistics yet: the pool's answer unchanged
    g = torch.Generator().manual_seed(0)
    centres = torch.nn.functional.normalize(torch.randn(4, 4, generator=g), dim=1)
    xs = torch.nn.functional.normalize(centres.repeat_interleave(12, 0) + 0.05 * torch.randn(48, 4, generator=g), dim=1)
    ys = torch.arange(4).repeat_interleave(12)
    pool.observe(xs, ys)
    out = pool.predict(centres[3:4]).probs
    assert int(out.argmax()) == 3 and pool.pool.seen and torch.allclose(out.sum(), torch.tensor(1.0), atol=1e-4)
    assert pool.summary() == {"n_experts": 1}


def test_uniform_priors_stop_a_very_frequent_class_from_outvoting_a_rare_one():
    from fsmoe.learn_lab import GlobalFast
    g = torch.Generator().manual_seed(1)
    centres = torch.nn.functional.normalize(torch.randn(4, 4, generator=g), dim=1)
    xs, ys = [], []
    for c, n in ((0, 400), (1, 20), (2, 20), (3, 6)):
        xs.append(torch.nn.functional.normalize(centres[c] + 0.9 * torch.randn(n, 4, generator=g), dim=1))      # overlapping classes, so the prior is not drowned by the evidence
        ys.append(torch.full((n,), c))
    xs, ys = torch.cat(xs), torch.cat(ys)
    with_priors, uniform = GlobalFast(_FakePool(), 4, 4, lam=1.0), GlobalFast(_FakePool(), 4, 4, lam=1.0, uniform_priors=True)
    for m in (with_priors, uniform):
        m.observe(xs, ys)
    probe = centres[3:4]
    p_prior, p_uni = float(with_priors.predict(probe).probs[0, 3]), float(uniform.predict(probe).probs[0, 3])
    assert 0.0 < p_prior < 0.99 and p_uni > p_prior + 0.05
    assert torch.allclose(uniform.predict(probe).probs.sum(), torch.tensor(1.0), atol=1e-4)


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        VerifiedLearner(Rec(), "nonsense", set(), torch.zeros(1, 2), torch.zeros(1, dtype=torch.long), np.random.default_rng(0))
