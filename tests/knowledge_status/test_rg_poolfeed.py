"""Lab G5 (routing_general/lab_poolfeed.py): the stream, the keys, the arm options, the verdict cache and the dynamic cards, with stand-ins for the language model and the pool."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
for p in (_KS, _KS / "routing_general", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import cards as CD  # noqa: E402
import lab_poolfeed as LP  # noqa: E402
import skills  # noqa: E402


class FakeReader:
    """Stands in for the language model: a fixed, deterministic distribution over the cards and "none of these" for every text; counts the texts it was asked."""
    asked = []

    def __init__(self, cards, with_exemplars, cache_dir, **kw):
        self.n = len(cards) + 1

    def probabilities(self, texts):
        FakeReader.asked += list(texts)
        P = np.full((len(texts), self.n), 0.02)
        for i, t in enumerate(texts):
            P[i, sum(map(ord, t)) % self.n] = 1.0
        return P / P.sum(1, keepdims=True)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _cards(k=3):
    return [CD.Card(f"skill {i}", (f"ex {i}a", f"ex {i}b")) for i in range(k)]


@pytest.fixture(autouse=True)
def fake_reader(monkeypatch):
    FakeReader.asked = []
    monkeypatch.setattr(LP.RD, "CardReader", FakeReader)


# ------------------------------------------------------------------ arm options
def test_arm_options_default_and_parse():
    base, o = LP.parse_arm("both")
    assert base == "both" and o == {"chance": False, "hard": False, "wk": 1.0, "wd": LP.W_DIGITS, "dyn": False, "we": 1.0, "cache": None, "shuffle": False, "head": None, "msim": 0.6, "warm": 0, "sharp": False, "alpha": None, "mmin": None}
    base, o = LP.parse_arm("key:c,h,w2,d0.35,y,e0.5,k0.95")
    assert base == "key" and o == {"chance": True, "hard": True, "wk": 2.0, "wd": 0.35, "dyn": True, "we": 0.5, "cache": 0.95, "shuffle": False, "head": None, "msim": 0.6, "warm": 0, "sharp": False, "alpha": None, "mmin": None}
    o = LP.parse_arm("emb:a0.3,p16")[1]
    assert o["alpha"] == 0.3 and o["mmin"] == 16
    assert LP.parse_arm("both:t0.9,n40,u")[1]["warm"] == 40 and LP.parse_arm("both:t0.9,n40,u")[1]["sharp"] is True
    assert LP.parse_arm("both:t0.9,m0.8")[1]["head"] == 0.9 and LP.parse_arm("both:t0.9,m0.8")[1]["msim"] == 0.8
    assert LP.parse_arm("both:s")[1]["shuffle"] is True
    with pytest.raises(ValueError):
        LP.parse_arm("key:z")
    with pytest.raises(ValueError):
        LP.parse_arm("nope")


# ------------------------------------------------------------------ the stream
def test_the_stream_is_deterministic_and_has_the_declared_phases():
    a, b = LP.build_stream(150, n1=8, n2=10, n3_novel=10, n3_taught=4, n_test=5), LP.build_stream(150, n1=8, n2=10, n3_novel=10, n3_taught=4, n_test=5)
    assert [m.span for ph in a["phases"] for m in ph] == [m.span for ph in b["phases"] for m in ph]
    assert a["taught"] == b["taught"] and a["novel"] == b["novel"] and a["novel"] not in a["taught"] and len(set(a["taught"])) == 3
    p1, p2, p3 = a["phases"]
    assert len(p1) == 24 and len(p2) == 30 and len(p3) == 10 + 12
    assert {m.family for m in p1} <= {"S1", "S2", "S3"}                                       # phase 1 is in the teaching wordings only
    assert {m.family for m in p2} <= {"S4", "S5", "S6", "S7", "S8"}                           # phase 2 only in new wordings
    assert {m.skill for m in p1} == set(a["taught"]) and {m.skill for m in p2} == set(a["taught"])
    assert a["novel"] in {m.skill for m in p3} and sum(m.skill == a["novel"] for m in p3) == 10
    blocks = [m.skill for m in p1]
    assert blocks == sorted(blocks, key=lambda s: a["taught"].index(s))                        # phase 1 teaches the skills one after another
    assert all(m.answer == skills.truth(m.skill, m.a, m.b) and 0 <= m.a <= 9 and 0 <= m.b <= 9 for ph in a["phases"] for m in ph)


def test_the_test_groups_are_in_the_right_wordings_and_skills():
    s = LP.build_stream(151, n1=6, n2=6, n3_novel=6, n3_taught=3, n_test=4)
    t = s["test"]
    assert set(t) == set(LP.GROUPS)
    assert {m.skill for m in t["taught_teach"]} == set(s["taught"]) and {m.skill for m in t["taught_new"]} == set(s["taught"])
    assert {m.skill for m in t["novel_teach"]} == {s["novel"]} == {m.skill for m in t["novel_new"]}
    assert {m.family for m in t["taught_teach"]} | {m.family for m in t["novel_teach"]} <= {"S1", "S2", "S3"}
    assert {m.family for m in t["taught_new"]} | {m.family for m in t["novel_new"]} <= {"S4", "S5", "S6", "S7", "S8"}
    assert len(s["cards"]) == 3 and all(len(c.exemplars) == 2 for c in s["cards"])             # the closed list: the taught skills, two taught messages each


# ------------------------------------------------------------------ the keys
def _msgs(n=3):
    return [LP.Msg(type("E", (), {"span": f"msg {i}", "skill": skills.SKILLS[i % 3], "a": i + 1, "b": 2, "answer": 0, "family": "S1"})()) for i in range(n)]


def test_digit_features_are_two_scaled_one_hots():
    X = LP.digit_features(_msgs(2), wd=0.5)
    assert X.shape == (2, 20) and np.isclose(X[0, 1], 0.5) and np.isclose(X[0, 12], 0.5) and np.isclose(X.sum(), 2.0)


def test_key_parts_have_the_widths_and_scales_of_each_arm():
    m = _msgs(3)
    emb = {x.span: _unit(np.arange(128) + i) for i, x in enumerate(m)}
    probs = {x.span: np.array([0.1, 0.7, 0.1, 0.1], dtype=np.float32) for x in m}
    opt = {"hard": False, "wk": 1.0, "wd": LP.W_DIGITS, "we": 1.0}
    assert LP.key_parts(m, {}, "emb", emb, probs, opt).shape == (3, 148)
    assert LP.key_parts(m, {}, "key", emb, probs, opt).shape == (3, 24)
    assert LP.key_parts(m, {}, "both", emb, probs, opt).shape == (3, 152)
    X = LP.key_parts(m, {}, "oracle", emb, probs, opt)
    assert X.shape == (3, 26) and X[:, :6].sum() == 3 and X[0, skills.SKILLS.index(m[0].skill)] == 1.0
    h = LP.key_parts(m, {}, "key", emb, probs, {**opt, "hard": True})
    assert np.allclose(h[:, :4], [[0, 1, 0, 0]] * 3)                                           # a hard key is the one-hot of the most probable option
    w = LP.key_parts(m, {}, "both", emb, probs, {**opt, "we": 0.5, "wk": 2.0})
    assert np.allclose(np.linalg.norm(w[0, :128]), 0.5) and np.allclose(w[0, 128:132], 2.0 * probs[m[0].span])


# ------------------------------------------------------------------ the verdict cache
def _cache(sim=0.95, k=3):
    rng = np.random.default_rng(0)
    base = {f"t{i}": _unit(rng.normal(size=128)) for i in range(6)}
    emb = dict(base)
    emb["t0 again"] = _unit(base["t0"] + 0.01 * rng.normal(size=128))                           # nearly the same sentence vector as t0
    truth = {s: FakeReader(_cards(k), True, None).probabilities([s])[0] for s in emb}
    FakeReader.asked = []                                                                      # building the truth is not a question the cache asked
    return LP.VerdictCache(_cards(k), emb, sim, truth), emb


def test_the_verdict_cache_asks_the_model_once_and_serves_near_duplicates():
    c, _ = _cache()
    out = c.probs(["t0", "t1"])
    assert c.asked == 2 and c.hits == 0 and FakeReader.asked == ["t0", "t1"]
    again = c.probs(["t0 again", "t2"])
    assert c.hits == 1 and c.asked == 3 and FakeReader.asked == ["t0", "t1", "t2"]               # "t0 again" came from the cache, t2 was new
    assert np.allclose(again["t0 again"], out["t0"])                                           # it got the verdict of the nearest earlier message
    assert c.served == 4


def test_the_verdict_cache_does_not_learn_from_test_messages_and_counts_wrong_hits():
    c, _ = _cache()
    c.probs(["t0"], learn=False)
    assert len(c.E) == 0                                                                       # a test message is answered but not remembered
    c.probs(["t0"])
    c.truth["t0 again"] = np.eye(4)[(int(c.P[0].argmax()) + 1) % 4]                             # the full reader would have said something else
    c.probs(["t0 again"])
    assert c.hits == 1 and c.wrong == 1


def test_a_stricter_similarity_serves_fewer_requests():
    strict, _ = _cache(sim=0.99999)
    strict.probs(["t0"])
    strict.probs(["t0 again"])
    assert strict.hits == 0 and strict.asked == 2


# ------------------------------------------------------------------ dynamic cards
class FakePool:
    def __init__(self):
        self.experts = {0: object(), 1: object()}
        self.route = {}

    def predict(self, x):
        return type("P", (), {"expert": torch.tensor([self.route.get(i, -1) for i in range(len(x))])})()


def test_dynamic_probabilities_have_a_fixed_width_with_none_last():
    r = LP.Cards(_cards(3), dynamic=True)
    P = r.probs(["a", "b"])
    assert P["a"].shape == (LP.KMAX + 1,) and np.isclose(P["a"].sum(), 1.0) and np.allclose(P["a"][3:LP.KMAX], 0.0)
    s = LP.Cards(_cards(3), dynamic=False)
    assert s.probs(["a"])["a"].shape == (4,)
    P = r.probs(["c", "d"])                                                                    # the stand-in puts its weight on option 3 ("none") for "c" and on card 0 for "d"
    assert P["c"].argmax() == LP.KMAX and np.isclose(P["c"][:3].sum(), 0.06 / 1.06, atol=1e-6)
    assert P["d"].argmax() == 0 and P["d"][LP.KMAX] < 0.05


def test_an_expert_born_after_teaching_gets_a_card_once_two_messages_are_routed_to_it():
    r, pool = LP.Cards(_cards(3), dynamic=True), FakePool()
    pool.route = {0: 1}
    r.after_batch(pool, torch.zeros(3, 2), ["m0", "m1", "m2"], [1], phase=1)
    assert len(r.cards) == 3 and r.pending == {1: ["m0"]}                                      # one message is not enough
    pool.route = {1: 1, 2: 1}
    r.after_batch(pool, torch.zeros(3, 2), ["m3", "m4", "m5"], [], phase=2)
    assert len(r.cards) == 4 and r.cards[3].exemplars == ("m0", "m4") and r.pending == {}


def test_no_card_is_made_during_teaching_for_a_dead_expert_or_beyond_the_slots():
    r, pool = LP.Cards(_cards(3), dynamic=True), FakePool()
    pool.route = {0: 1, 1: 1}
    r.after_batch(pool, torch.zeros(2, 2), ["x", "y"], [1], phase=0)
    assert len(r.cards) == 3 and r.pending == {}                                               # phase 0 = teaching: the teacher's cards only
    r.after_batch(pool, torch.zeros(2, 2), ["x", "y"], [7], phase=1)
    assert r.pending == {}                                                                      # an expert that is not in the pool is dropped
    full = LP.Cards(_cards(LP.KMAX), dynamic=True)
    pool.route = {0: 1, 1: 1}
    full.after_batch(pool, torch.zeros(2, 2), ["x", "y"], [1], phase=1)
    assert len(full.cards) == LP.KMAX                                                           # the slots are full
    static = LP.Cards(_cards(3), dynamic=False)
    static.after_batch(pool, torch.zeros(2, 2), ["x", "y"], [1], phase=1)
    assert len(static.cards) == 3


def test_the_shuffle_control_keeps_the_verdicts_but_gives_them_to_other_messages():
    probs = {f"m{i}": np.eye(4, dtype=np.float32)[i % 4] * 0.9 + 0.025 for i in range(40)}
    a, b = LP.shuffle_verdicts(probs, 3), LP.shuffle_verdicts(probs, 3)
    assert all(np.array_equal(a[k], b[k]) for k in probs)                                      # deterministic in the seed
    assert sorted(map(tuple, a.values())) == sorted(map(tuple, probs.values()))                # the same verdicts ...
    assert sum(not np.array_equal(a[k], probs[k]) for k in probs) > 20                         # ... on other messages
    assert any(not np.array_equal(LP.shuffle_verdicts(probs, 4)[k], a[k]) for k in probs)      # another seed shuffles differently


# ------------------------------------------------------------------ the trained head
class DictReader:
    """A language model that knows the class of every text (a dict) and is sure about it, except for texts listed as unsure."""
    labels, unsure, asked = {}, set(), []

    def __init__(self, cards, with_exemplars, cache_dir, **kw):
        self.n = len(cards) + 1

    def probabilities(self, texts):
        DictReader.asked += list(texts)
        P = np.full((len(texts), self.n), 0.01)
        for i, t in enumerate(texts):
            P[i, DictReader.labels[t]] = 0.4 if t in DictReader.unsure else 0.97
            if t in DictReader.unsure:
                P[i, (DictReader.labels[t] + 1) % self.n] = 0.35
        return P / P.sum(1, keepdims=True)


def _head_world(monkeypatch, conf=0.9, msim=0.5, k=3):
    rng = np.random.default_rng(1)
    centres = [_unit(rng.normal(size=128)) for _ in range(k + 1)]
    emb, labels = {}, {}
    for c in range(k + 1):
        for j in range(8):
            s = f"c{c}m{j}"
            emb[s] = _unit(centres[c] + 0.05 * rng.normal(size=128))
            labels[s] = c
    emb["far"] = _unit(rng.normal(size=128))
    labels["far"] = 0
    DictReader.labels, DictReader.unsure, DictReader.asked = labels, set(), []
    monkeypatch.setattr(LP.RD, "CardReader", DictReader)
    truth = {s: DictReader(_cards(k), True, None).probabilities([s])[0] for s in emb}
    DictReader.asked = []
    return LP.VerdictHead(_cards(k), emb, conf, msim, truth)


def test_the_head_asks_the_model_until_it_has_seen_two_verdicts_and_then_answers_alone(monkeypatch):
    h = _head_world(monkeypatch)
    h.probs([f"c0m{j}" for j in range(4)])
    assert h.clf is None and h.asked == 4                                                      # one class cannot be fitted
    h.probs([f"c1m{j}" for j in range(4)])
    assert h.clf is not None and h.asked == 8 and h.hits == 0
    out = h.probs(["c0m5", "c1m6"])
    assert h.asked == 8 and h.hits == 2 and h.wrong == 0                                       # answered from the head, no model call
    assert out["c0m5"].argmax() == 0 and out["c1m6"].argmax() == 1 and np.isclose(out["c0m5"].sum(), 1.0)


def test_the_head_abstains_on_a_message_unlike_anything_it_was_trained_on(monkeypatch):
    h = _head_world(monkeypatch, msim=0.9)
    h.probs([f"c0m{j}" for j in range(4)] + [f"c1m{j}" for j in range(4)])
    h.probs(["far"])
    assert h.hits == 0 and DictReader.asked[-1] == "far"                                       # far from every training vector: asked, however confident the head is


def test_the_head_does_not_learn_from_test_messages_or_from_unsure_verdicts(monkeypatch):
    h = _head_world(monkeypatch)
    h.probs(["c0m0", "c1m0"], learn=False)
    assert len(h.E) == 0 and h.clf is None
    DictReader.unsure = {"c1m1"}
    h.probs(["c0m1", "c1m1"])
    assert len(h.E) == 1 and h.y == [0]                                                        # the unsure verdict (probability 0.4) is not a training label


def test_the_head_counts_answers_the_full_reader_would_not_have_given(monkeypatch):
    h = _head_world(monkeypatch)
    h.probs([f"c0m{j}" for j in range(4)] + [f"c1m{j}" for j in range(4)])
    h.truth["c0m6"] = np.eye(4)[2]                                                             # the full reader would have said something else about this one
    h.probs(["c0m6", "c0m7"])
    assert h.hits == 2 and h.wrong == 1


def test_a_stricter_confidence_makes_the_head_ask_more(monkeypatch):
    easy, strict = _head_world(monkeypatch, conf=0.5), _head_world(monkeypatch, conf=0.9999)
    for h in (easy, strict):
        h.probs([f"c0m{j}" for j in range(4)] + [f"c1m{j}" for j in range(4)])
        h.probs([f"c0m{j}" for j in range(4, 8)])
    assert easy.hits == 4 and strict.hits < easy.hits


def test_the_head_waits_for_enough_verdicts_before_it_answers_and_can_sharpen_them(monkeypatch):
    h = _head_world(monkeypatch)
    h.warm = 12
    h.probs([f"c0m{j}" for j in range(4)] + [f"c1m{j}" for j in range(4)])
    h.probs(["c0m5"])
    assert h.hits == 0 and h.asked == 9                                                        # 8 verdicts learned: under the warm-up of 12, still asked
    h.probs([f"c0m{j}" for j in range(5, 8)] + ["c1m5"])
    assert len(h.y) >= 12
    out = h.probs(["c0m6"])
    assert h.hits == 1
    sharp = _head_world(monkeypatch)
    sharp.sharp = True
    sharp.probs([f"c0m{j}" for j in range(4)] + [f"c1m{j}" for j in range(4)])
    s = sharp.probs(["c0m5"])["c0m5"]
    assert np.isclose(s.max(), 0.97) and np.isclose(s.sum(), 1.0) and s.argmax() == 0           # the head's choice with the probability the language model gives when it is sure
    assert not np.isclose(out["c0m6"].max(), 0.97)                                              # without the switch the head hands over its own probabilities


def test_the_cheap_readers_count_each_distinct_text_once_and_keep_stream_counts_apart_from_test_evaluations(monkeypatch):
    c, _ = _cache()
    c.probs(["t0", "t1"])
    c.probs(["t2"], learn=False)
    c.probs(["t2"], learn=False)                                                               # the same test message evaluated again after the next phase
    assert c.asked == 4 and len(c.asked_set) == 3                                              # asked four times, three distinct texts
    assert c.s_served == 2 and c.s_hits == 0
    c.probs(["t0 again"])
    assert c.s_served == 3 and c.s_hits == 1 and c.s_wrong == 0 and c.served == 5

    h = _head_world(monkeypatch)
    h.probs([f"c0m{j}" for j in range(4)] + [f"c1m{j}" for j in range(4)])
    h.probs(["far"], learn=False)
    h.probs(["far"], learn=False)
    h.probs(["c0m5", "c1m6"])
    assert h.asked == 10 and len(h.asked_set) == 9                                             # "far" is asked twice but is one distinct text
    assert h.s_served == 10 and h.s_hits == 2 and h.served == 12 and h.hits == 2


def test_the_head_does_not_answer_a_message_that_is_confidently_classified_but_far_from_everything_it_saw(monkeypatch):
    h = _head_world(monkeypatch, conf=0.9, msim=0.5)
    h.probs([f"c0m{j}" for j in range(4)] + [f"c1m{j}" for j in range(4)])
    monkeypatch.setattr(h, "_head_probs", lambda E: np.tile(np.array([0.99, 0.005, 0.005, 0.0], dtype=np.float32), (len(E), 1)))    # a head that is sure of everything
    assert float((h.emb["far"] @ h.E.T).max()) < 0.5                                           # "far" is unlike every message it was trained on
    hits = h.hits
    h.probs(["far"])
    assert h.hits == hits and DictReader.asked[-1] == "far"                                    # the nearest-neighbour check sends it to the language model
    h.probs(["c0m5"])
    assert h.hits == hits + 1                                                                  # a familiar message is answered by the same (sure) head


def test_the_warm_up_is_inclusive(monkeypatch):
    h = _head_world(monkeypatch)
    h.warm = 8
    h.probs([f"c0m{j}" for j in range(4)] + [f"c1m{j}" for j in range(4)])
    assert len(h.y) == 8
    h.probs(["c0m5"])
    assert h.hits == 1                                                                         # exactly the warm-up number of verdicts is enough
