"""The infix cores of the confirmatory pool (S1 symbols and words, S2 number words) carry the operator of their own skill and of no other skill, and the commutative S5 cores keep both operands.
This is a check written from the meaning of the words, not from the pool file, and it reads the pool only through ``pools.load_pool``."""

import re

import pytest

import pools
import skills

CUE = {   # the operator cues of each skill: words (whole-word match) and symbols
    "add": (("plus",), ("+",)), "sub": (("minus",), ("-",)), "mul": (("times",), ("*",)), "div": (("divided by",), ("/",)), "mod": (("mod",), ("%",)), "max": (("max", "maximum", "bigger"), ()),
}


def cues_in(text: str):
    found = set()
    low = text.lower()
    for skill, (words, symbols) in CUE.items():
        if any(re.search(rf"\b{re.escape(w)}\b", low) for w in words) or any(s in text for s in symbols):
            found.add(skill)
    return found


@pytest.fixture
def confirm(monkeypatch):
    monkeypatch.setenv("KS_ALLOW_CONFIRM", "1")
    pools.load_pool.cache_clear()
    yield pools.load_pool("confirm")
    pools.load_pool.cache_clear()


def test_every_infix_core_has_exactly_the_cue_of_its_own_skill(confirm):
    for fam in ("S1", "S2"):
        for c in confirm.cores[fam]:
            assert cues_in(c.text) == {c.skill}, (c.core_id, c.text, cues_in(c.text))


def test_every_infix_core_shows_both_operands_once_in_the_order_a_then_b(confirm):
    for fam, a, b in (("S1", "{a}", "{b}"), ("S2", "{aw}", "{bw}")):
        for c in confirm.cores[fam]:
            if c.skill == "max":
                continue                                                  # max may name its operands in either order; it is commutative
            assert c.text.count(a) == 1 and c.text.count(b) == 1 and c.text.index(a) < c.text.index(b), (c.core_id, c.text)


def test_the_swapped_commutative_cores_show_b_before_a_and_are_marked_swapped(confirm):
    for c in confirm.cores["S5"]:
        assert c.swap and c.skill in skills.COMMUTATIVE and c.text.count("{a}") == 1 and c.text.count("{b}") == 1 and c.text.index("{b}") < c.text.index("{a}"), (c.core_id, c.text)


def test_story_cores_never_use_an_operator_word_or_symbol(confirm):
    for c in confirm.cores["S4"]:
        assert not re.search(r"[+*/%=]|\bplus\b|\bminus\b|\btimes\b|\bdivided\b|\bmod\b", c.text.lower()), (c.core_id, c.text)
        assert c.text.count("{a}") == 1 and c.text.count("{b}") == 1


def test_verbal_cores_name_both_operands_and_no_symbol(confirm):
    for c in confirm.cores["S3"]:
        assert c.text.count("{a}") == 1 and c.text.count("{b}") == 1 and not re.search(r"[+*/%=]", c.text), (c.core_id, c.text)
