import re

import numpy as np
import pytest

import corpora
import generator as G
import pools
import skills
from test_ks_generator import read_skill


def digits_of(text):
    return [int(x) for x in re.findall(r"\d+", text)]


def test_spans_cut_the_text_exactly():
    rng = np.random.default_rng(0)
    for kinds in (("skill", "other"), ("skill", "skill")):
        for _ in range(120):
            msg = G.make_tangled(rng, "dev", kinds)
            parts = sorted(msg.parts, key=lambda q: q.start)
            assert len(parts) == 2 and msg.family == "tangled"
            assert 0 == parts[0].start and parts[1].end == len(msg.text)
            assert parts[0].end <= parts[1].start                                       # a separator sits between the parts and belongs to neither
            for p in parts:
                span = msg.text[p.start:p.end]
                assert span == span.strip() and len(span) > 0
                if p.kind == "skill":
                    assert digits_of(span) == [p.a, p.b] or sorted(digits_of(span)) == sorted([p.a, p.b]), (span, p)
                    assert p.answer == skills.truth(p.skill, p.a, p.b)
                else:
                    assert p.kind == "other" and p.skill is None and corpora.is_clean(span)


def test_skill_part_labels_survive_the_join():
    rng = np.random.default_rng(1)
    for _ in range(200):
        msg = G.make_tangled(rng, "dev", ("skill", "skill"), families=("S1",))
        for p in msg.parts:
            assert read_skill(msg.text[p.start:p.end]) == p.skill


def test_both_orders_occur():
    rng = np.random.default_rng(2)
    firsts = [sorted(G.make_tangled(rng, "dev", ("skill", "other")).parts, key=lambda q: q.start)[0].kind for _ in range(300)]
    assert 80 < firsts.count("skill") < 220 and 80 < firsts.count("other") < 220


def test_known_plus_unseen_pairs_use_two_different_skills():
    rng = np.random.default_rng(3)
    for _ in range(300):
        msg = G.make_tangled(rng, "dev", ("skill", "skill"))
        a, b = (p.skill for p in msg.parts)
        assert a != b
    msg = G.make_tangled(rng, "dev", ("skill", "skill"), skill_labels=("add", "mul"))
    assert {p.skill for p in msg.parts} == {"add", "mul"}
    with pytest.raises(ValueError):
        G.make_tangled(rng, "dev", ("skill", "skill"), skill_labels=("add", "add"))
    with pytest.raises(ValueError):
        G.make_tangled(rng, "dev", ("other", "other"))


def test_two_requests_in_one_message_are_both_recoverable():
    rng = np.random.default_rng(4)
    for _ in range(200):
        msg = G.make_tangled(rng, "dev", ("skill", "skill"), families=("S1",))
        parts = sorted(msg.parts, key=lambda q: q.start)
        assert digits_of(msg.text) == [parts[0].a, parts[0].b, parts[1].a, parts[1].b]


def test_separators_carry_no_digits_or_operator_cues():
    for sep in G.SEPARATORS:
        assert not re.search(r"\d", sep)
        assert not (set(re.findall(r"[a-z]+", sep.lower())) & pools.ARITH_CUES), sep
        assert not (set(sep) & pools.ARITH_SYMBOLS)
        assert sep.startswith(" ") and len(sep.strip()) >= 0
