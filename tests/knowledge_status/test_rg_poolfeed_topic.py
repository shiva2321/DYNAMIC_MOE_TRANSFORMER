"""Lab G6 (routing_general/lab_poolfeed_topic.py): the topical stream, the wording shift and the keys, on a small made-up corpus (no dataset and no language model needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
for p in (_KS, _KS / "routing_general", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import lab_poolfeed_topic as LT  # noqa: E402

SIZES = {"n1": 8, "n2": 6, "n3_novel": 10, "n3_taught": 4, "n_test": 3}


def _corpus():
    names = [f"topic_{c}" for c in range(6)]
    tr = [(f"message number {i} concerning {names[c].replace('_', ' ')} matters entirely", c) for c in range(6) for i in range(40)]
    te = [(f"question number {i} regarding {names[c].replace('_', ' ')} affairs entirely", c) for c in range(6) for i in range(10)]
    return (([t for t, _ in tr], [c for _, c in tr]), ([t for t, _ in te], [c for _, c in te]), names)


@pytest.fixture(autouse=True)
def fake_domain(monkeypatch):
    monkeypatch.setitem(LT.DOMAINS, "fake", _corpus)


def test_the_shift_changes_the_message_and_wraps_it_in_polite_chatter():
    rng = np.random.default_rng(0)
    chatter = ["Thanks for helping me out.", "Hope you are well."]
    text = "please explain the transfer limits of international payments"
    s = LT.shift_text(text, rng, chatter)
    assert s != text and s.startswith(tuple(chatter)) and s.endswith(tuple(chatter))
    assert LT.shift_text(text, np.random.default_rng(0), chatter) == s                        # deterministic in the generator
    assert any(c in s for c in chatter) and len(s) > len(text)


def test_the_topical_stream_is_deterministic_and_has_the_declared_phases():
    a, b = LT.build_topic_stream("fake", 5, **SIZES), LT.build_topic_stream("fake", 5, **SIZES)
    assert [m.span for ph in a["phases"] for m in ph] == [m.span for ph in b["phases"] for m in ph]
    assert a["taught"] == b["taught"] and a["novel"] == b["novel"] and a["novel"] not in a["taught"] and len(set(a["taught"])) == 3
    p1, p2, p3 = a["phases"]
    assert len(p1) == 24 and len(p2) == 18 and len(p3) == 10 + 12
    blocks = [m.cls for m in p1]
    assert blocks == sorted(blocks, key=lambda c: a["taught"].index(c)) and not any(m.shifted for m in p1)       # taught one after another, natural wording
    assert {m.cls for m in p2} == set(a["taught"]) and sum(m.shifted for m in p2) == 9                            # half of phase 2 is shifted
    assert sum(m.cls == a["novel"] for m in p3) == 10 and sum(m.shifted for m in p3) == 11


def test_the_tests_are_new_messages_in_four_groups_and_the_cards_come_from_phase_one():
    s = LT.build_topic_stream("fake", 6, **SIZES)
    t = s["test"]
    assert set(t) == set(LT.GROUPS)
    assert {m.cls for m in t["taught_plain"]} == set(s["taught"]) == {m.cls for m in t["taught_shift"]}
    assert {m.cls for m in t["novel_plain"]} == {s["novel"]} == {m.cls for m in t["novel_shift"]}
    assert len(t["taught_plain"]) == 9 and len(t["novel_shift"]) == 3
    assert not any(m.shifted for m in t["taught_plain"] + t["novel_plain"]) and all(m.shifted for m in t["taught_shift"] + t["novel_shift"])
    stream_spans = {m.span for ph in s["phases"] for m in ph}
    assert not ({m.span for g in t.values() for m in g} & stream_spans)                       # no test message is in the stream
    p1 = s["phases"][0]
    for card, c in zip(s["cards"], s["taught"]):
        assert card.name == s["names"][c].replace("_", " ") and len(card.exemplars) == 2 and all(e in {m.span for m in p1 if m.cls == c} for e in card.exemplars)


def test_key_parts_have_the_width_and_weights_of_each_arm():
    s = LT.build_topic_stream("fake", 7, **SIZES)
    msgs = s["phases"][0][:4]
    emb = {m.span: np.ones(128, dtype=np.float32) / np.sqrt(128) for m in msgs}
    probs = {m.span: np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32) for m in msgs}
    opt = {"hard": False, "wk": 1.0, "we": 1.0}
    assert LT.key_parts(msgs, s, "emb", emb, probs, opt).shape == (4, 128)
    assert LT.key_parts(msgs, s, "key", emb, probs, opt).shape == (4, 4)
    assert LT.key_parts(msgs, s, "both", emb, probs, opt).shape == (4, 132)
    X = LT.key_parts(msgs, s, "oracle", emb, probs, opt)
    assert X.shape == (4, 4) and X.sum() == 4 and X[0, s["taught"].index(msgs[0].cls)] == 1.0
    w = LT.key_parts(msgs, s, "both", emb, probs, {**opt, "we": 0.5, "wk": 2.0})
    assert np.isclose(np.linalg.norm(w[0, :128]), 0.5) and np.allclose(w[0, 128:], 2.0 * probs[msgs[0].span])
    h = LT.key_parts(msgs, s, "key", emb, probs, {**opt, "hard": True})
    assert np.allclose(h, [[1, 0, 0, 0]] * 4)


def test_all_spans_lists_every_text_once():
    s = LT.build_topic_stream("fake", 8, **SIZES)
    spans = LT.all_spans(s)
    assert len(spans) == len(set(spans)) and {m.span for ph in s["phases"] for m in ph} <= set(spans)


def test_the_shift_uses_two_different_chatter_sentences_and_changes_the_message_itself():
    chatter = [f"Chatter sentence number {i}." for i in range(6)]
    text = "please explain the transfer limits of international payments"
    for seed in range(5):
        s = LT.shift_text(text, np.random.default_rng(seed), chatter)
        pre = next(c for c in chatter if s.startswith(c))
        post = next(c for c in chatter if s.endswith(c))
        assert pre != post                                                                     # one sentence before, a different one after
        assert text not in s                                                                  # typos, case flips or doubled spaces changed the message


def test_shifted_test_messages_are_other_messages_than_the_plain_ones_and_are_wrapped_in_chatter():
    import re
    chatter = list(LT.pools.load_pool("dev").chatter)
    s = LT.build_topic_stream("fake", 9, **SIZES)
    for kind in ("taught", "novel"):
        plain = {(m.cls, tuple(re.findall(r"\d+", m.span))) for m in s["test"][f"{kind}_plain"]}
        shifted = {(m.cls, tuple(re.findall(r"\d+", m.span))) for m in s["test"][f"{kind}_shift"]}
        assert plain and shifted and not (plain & shifted)                                    # the shifted tests are different messages (digits are never touched by the noise)
        assert all(any(m.span.startswith(c) for c in chatter) for m in s["test"][f"{kind}_shift"])
