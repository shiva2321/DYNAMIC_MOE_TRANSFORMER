import re

import numpy as np
import pytest

import corpora
import generator as G
import pools
import skills

CUE_TOKEN = re.compile(r"divided by|plus|minus|times|mod|max|[+\-*/%]")


def request_text(msg):
    p = next(q for q in msg.parts if q.kind == "skill")
    return msg.text[p.start:p.end]


def test_is_clean_matches_whole_words_only():
    assert corpora.is_clean("The modern world is changing quickly for everyone")      # "modern" is not the cue "mod"
    assert not corpora.is_clean("I go there seven times a day without fail ok")        # the cue "times"
    assert not corpora.is_clean("I bought 5 apples at the market this morning")         # a digit
    assert not corpora.is_clean("Take the sum of everything we spent last week")        # the cue "sum"
    assert not corpora.is_clean("Is two + two really four or something else here")      # a symbol
    assert not corpora.is_clean("too short")                                           # fewer than 5 words
    assert not corpora.is_clean(" ".join(["word"] * 41))                               # more than 40 words


@pytest.mark.parametrize("source", ["banking77", "agnews"])
def test_chatter_has_no_digits_or_operator_cues(source):
    sentences = corpora.load_other_sentences(1000, seed=0, source=source)
    assert len(sentences) == 1000 and len(set(sentences)) == 1000
    for s in sentences:
        assert not re.search(r"\d", s), s
        assert not (set(re.findall(r"[a-z]+", s.lower())) & pools.ARITH_CUES), s
        assert not (set(s) & pools.ARITH_SYMBOLS), s
        assert 5 <= len(s.split()) <= 40, s
    assert corpora.ARITH_CUES is pools.ARITH_CUES


def test_loaders_are_deterministic_for_a_seed():
    a = corpora.load_other_sentences(50, seed=3, source="banking77")
    b = corpora.load_other_sentences(50, seed=3, source="banking77")
    c = corpora.load_other_sentences(50, seed=4, source="banking77")
    assert a == b and a != c


def test_near_twin_differs_only_in_the_operator_cue():
    rng = np.random.default_rng(0)
    made = 0
    for _ in range(600):
        skill = [s for s in skills.SKILLS if s != "max"][int(rng.integers(0, 5))]
        fam = ("S1", "S2", "S6", "S8")[int(rng.integers(0, 4))]
        msg = G.make_single(rng, "dev", skill, fam)
        if not msg.template_ids[1].startswith(("S1.", "S2.")):
            continue                                                                   # S6 and S8 sometimes wrap a verbal (S3) core, which has no parallel form
        to = [s for s in skills.SKILLS if s != skill][int(rng.integers(0, 5))]
        if to in ("div", "mod") and next(x for x in msg.parts if x.kind == "skill").b == 0:
            continue                                                                   # the twin would divide by zero (refused, tested below)
        twin = G.make_near_twin(rng, msg, to)
        p, q = (next(x for x in m.parts if x.kind == "skill") for m in (msg, twin))
        assert (p.a, p.b) == (q.a, q.b) and q.skill == to != p.skill
        assert q.answer == skills.truth(to, q.a, q.b)
        assert CUE_TOKEN.sub("<OP>", msg.text) == CUE_TOKEN.sub("<OP>", twin.text), (msg.text, twin.text)
        assert CUE_TOKEN.sub("<OP>", request_text(msg)) == CUE_TOKEN.sub("<OP>", request_text(twin))
        assert twin.text != msg.text
        assert twin.text[q.start:q.end] == request_text(twin) and q.end <= len(twin.text)
        made += 1
    assert made > 150


def test_near_twin_refuses_a_zero_divisor_and_an_unparallel_core():
    rng = np.random.default_rng(1)
    zero = G.Message("12 + 0", (G.Part(0, 6, "skill", "add", 12, 0, 12),), "S1", ("S1.f0", "S1.add.c0"), "dev")
    with pytest.raises(ValueError):
        G.make_near_twin(rng, zero, "div")                                             # the twin would divide by zero
    assert G.make_near_twin(rng, zero, "sub").text == "12 - 0"
    verbal = G.make_single(rng, "dev", "add", "S3")
    with pytest.raises(ValueError):
        G.make_near_twin(rng, verbal, "sub")                                           # no parallel form for verbal cores


def test_trap_has_no_skill_label_but_looks_arithmetic():
    rng = np.random.default_rng(2)
    for _ in range(100):
        m = G.make_trap(rng, "dev")
        assert len(m.parts) == 1 and m.parts[0].kind == "other" and m.parts[0].skill is None
        assert m.parts[0].start == 0 and m.parts[0].end == len(m.text)
        words = set(re.findall(r"[a-z]+", m.text.lower()))
        assert (words & pools.ARITH_CUES) or re.search(r"\d", m.text)                  # a trap must look like arithmetic to a word-matcher


def test_other_and_unseen_content_are_single_other_parts_and_disjoint():
    rng = np.random.default_rng(3)
    others = [G.make_other(rng, "dev") for _ in range(200)]
    unseen = [G.make_unseen_content(rng, "dev") for _ in range(200)]
    for m in others:
        assert len(m.parts) == 1 and m.parts[0].kind == "other" and corpora.is_clean(m.text)
    for m in unseen:
        assert len(m.parts) == 1 and m.parts[0].kind == "unseen_content" and corpora.is_clean(m.text)
    assert not ({m.text for m in others} & {m.text for m in unseen})
    a = set(corpora.load_other_sentences(300, seed=0, source="banking77")) | set(corpora.load_other_sentences(300, seed=0, source="agnews"))
    b = set(corpora.load_unseen_content(300, seed=0))
    assert len(b) == 300 and not (a & b)


def test_loaders_fail_loudly_when_the_cache_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(corpora, "HUB", tmp_path)
    monkeypatch.setattr(corpora, "CACHE_DIR", tmp_path / "cache")
    corpora._MEMO.clear()
    with pytest.raises(FileNotFoundError, match="banking77"):
        corpora.load_other_sentences(10, seed=0, source="banking77")
    with pytest.raises(FileNotFoundError, match="ai2_arc"):
        corpora.load_unseen_content(10, seed=0)
    corpora._MEMO.clear()
