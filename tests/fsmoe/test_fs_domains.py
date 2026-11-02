import numpy as np
import pytest

from fsmoe.domains import ArithDomain, CodeDomain, LangDomain, table
from fsmoe.domains import arith as AR
from fsmoe.domains import code as CO


# ------------------------------------------------------------------------------------------------------------------------------ arithmetic
def test_arith_every_cell_is_checked_and_deterministic():
    d = ArithDomain()
    for s in d.skills:
        for f in d.families:
            items = d.make(s, f, 6, 3)
            assert len(items) == 6 and all(it.skill == s and it.family == f and it.text.strip() for it in items)
            assert all(d.check(it) for it in items)
            assert [it.text for it in items] == [it.text for it in d.make(s, f, 6, 3)]
            assert [it.text for it in items] != [it.text for it in d.make(s, f, 6, 4)]


def test_arith_answers_and_twins():
    assert [AR.answer(s, 17, 5) for s in AR.SKILLS] == [22, 12, 85, 3, 17, 2, 5, 12]
    assert all(AR.TWINS[s] in AR.SKILLS and AR.TWINS[s] != s for s in AR.SKILLS)
    d = ArithDomain()
    assert set(d.teach_families).isdisjoint(d.heldout_families)
    assert set(d.withheld) <= set(d.skills) and len(d.taught) == 5


def test_arith_the_checker_catches_a_wrong_answer():
    d = ArithDomain()
    it = d.make("add", "symbols", 1, 0)[0]
    bad = type(it)(it.text, it.domain, it.skill, it.family, (it.meta[0], it.meta[1], it.meta[2] + 1))
    assert d.check(it) and not d.check(bad)


def test_arith_numbers_are_in_the_text_for_digit_families():
    d = ArithDomain()
    for s in d.skills:
        for it in d.make(s, "symbols", 20, 1):
            a, b, _ = it.meta
            assert str(a) in it.text and str(b) in it.text


# ------------------------------------------------------------------------------------------------------------------------------ code
def test_code_reference_signatures_are_distinct():
    assert len(set(CO.REF_SIGNATURE.values())) == len(CO.SKILLS)


def test_code_every_variant_runs_and_matches_its_family():
    d = CodeDomain()
    for s in d.skills:
        for f in d.families:
            for it in d.make(s, f, 8, 5):
                assert d.check(it), (s, f, it.text)


def test_code_a_snippet_of_another_family_fails_the_check():
    d = CodeDomain()
    it = d.make("sum_list", "loop", 1, 0)[0]
    wrong = type(it)(it.text, it.domain, "product_list", it.family, it.meta)
    assert d.check(it) and not d.check(wrong)


def test_code_a_broken_snippet_fails_the_check():
    assert CO.signature("def f(x):\n    return x[100]\n", "f", "list") is None
    assert CO.signature("def f(x):\n    return 1\n", "g", "list") is None


def test_code_naming_families_use_their_pools():
    d = CodeDomain()
    for it in d.make("sum_list", "renamed", 20, 2):
        assert it.meta[0] in CO.NEUTRAL["f"]
    for it in d.make("sum_list", "short", 20, 2):
        assert it.meta[0] in CO.SHORT["f"]
    for it in d.make("sum_list", "loop", 20, 2):
        assert it.meta[0] in CO.FUNC_DESCRIPTIVE["sum_list"]


def test_code_boilerplate_and_formatted_look_different_but_behave_the_same():
    d = CodeDomain()
    a, b, c = d.make("max_list", "boilerplate", 1, 9)[0], d.make("max_list", "formatted", 1, 9)[0], d.make("max_list", "loop", 1, 9)[0]
    assert "__main__" in a.text and "->" in b.text and "__main__" not in c.text
    assert d.check(a) and d.check(b) and d.check(c)


def test_code_withheld_split_has_near_twins_and_far_skills():
    d = CodeDomain()
    assert set(d.far_withheld) <= set(d.withheld)
    assert all(d.twins[s] in d.taught for s in d.near_withheld())
    assert len(d.taught) == 11 and len(d.withheld) == 5


# ------------------------------------------------------------------------------------------------------------------------------ language
@pytest.fixture(scope="module")
def lang():
    return LangDomain()


def test_lang_withheld_split_is_ten_near_and_ten_far(lang):
    assert len(lang.skills) == 77 and len(lang.withheld) == 20 and len(lang.far_withheld) == 10 and len(lang.near_withheld()) == 10
    assert all(lang.twins[s] in lang.taught for s in lang.near_withheld())


def test_lang_teaching_and_evaluation_pools_are_disjoint(lang):
    for s in lang.skills[:12]:
        assert set(lang._d["teach"][s]).isdisjoint(lang._d["eval"][s])


def test_lang_families_checked_deterministic_and_never_taught_when_held_out(lang):
    s = lang.taught[0]
    for f in lang.families:
        items = lang.make(s, f, 5, 2, "eval")
        assert all(lang.check(it) and it.family == f for it in items)
        assert [it.text for it in items] == [it.text for it in lang.make(s, f, 5, 2, "eval")]
    with pytest.raises(ValueError):
        lang.make(s, "typos", 3, 0, "teach")


def test_lang_transforms_change_the_text_but_keep_the_utterance_recognisable(lang):
    s = lang.taught[1]
    base = lang.make(s, "original", 4, 1, "eval")
    for f in ("chatter", "register", "framing"):
        wrapped = lang.make(s, f, 4, 1, "eval")
        assert all(b.text.lower().split()[0] != "" for b in wrapped)
        assert any(w.text != b.text for w, b in zip(wrapped, base))


# ------------------------------------------------------------------------------------------------------------------------------ the table
def test_table_row_counts_and_withheld_skills_have_no_teaching_rows():
    d = ArithDomain()
    rows = table(d, 0, 2, 1)
    teach = [it for it, sp in rows if sp == "teach"]
    ev = [it for it, sp in rows if sp == "eval"]
    assert len(teach) == len(d.taught) * len(d.teach_families) * 2
    assert len(ev) == len(d.skills) * len(d.families) * 1
    assert not {it.skill for it in teach} & set(d.withheld)
    assert not {it.family for it in teach} & set(d.heldout_families)
