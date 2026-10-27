"""Task 11, cutting a message into parts: the gold parts of the generator, and the two automatic cutters (sentences and clauses; sliding word windows)."""

import numpy as np
import pytest

import generator as G
import segment as SG


def messages(n=40, seed=0):
    rng = np.random.default_rng(seed)
    singles = [G.make_single(rng, "dev", s, f) for s in ("add", "sub", "mul", "max") for f in ("S1", "S3", "S4", "S6", "S8") for _ in range(2)]
    tangled = [G.make_tangled(rng, "dev", kinds) for kinds in (("skill", "skill"), ("skill", "other"), ("other", "skill")) for _ in range(n // 3)]
    return singles, tangled


def test_gold_parts_equal_the_message_spans():
    singles, tangled = messages()
    for m in singles + tangled:
        spans = SG.gold_parts(m)
        assert spans == [(p.start, p.end) for p in m.parts]
        assert all(m.text[s:e].strip() == m.text[s:e] and e > s for s, e in spans)
    for m in tangled:
        spans = SG.gold_parts(m)
        assert spans[0][0] == 0 and spans[-1][1] == len(m.text) and spans[0][1] <= spans[1][0]       # the separator belongs to neither part


def check_partition(text, spans):
    assert spans == sorted(spans)
    covered = np.zeros(len(text), dtype=bool)
    for s, e in spans:
        assert 0 <= s < e <= len(text)
        assert text[s:e] == text[s:e].strip()
        assert not covered[s:e].any(), f"overlap in {spans} of {text!r}"
        covered[s:e] = True
    uncovered = [c for c, flag in zip(text, covered) if not flag]
    assert all(c.isspace() for c in uncovered), f"non-space text left out of every part: {text!r} {spans}"


def test_sentence_clause_parts_cover_the_text_without_overlap():
    singles, tangled = messages()
    for m in singles + tangled:
        check_partition(m.text, SG.sentence_clause_parts(m.text))
    for text in ("", "   ", "one", "What is 3 plus 4? Also: what is 8 times 2?", "Hi, and thanks. Bye.", "a; b; c d e f", "  padded   text with   gaps  "):
        check_partition(text, SG.sentence_clause_parts(text))


def test_sentence_clause_parts_cut_where_a_person_would():
    text = "What is 3 plus 4? Also: what is 8 times 2?"
    assert [text[s:e] for s, e in SG.sentence_clause_parts(text)] == ["What is 3 plus 4?", "Also: what is 8 times 2?"]
    text = "I want to know my balance, and then please add 17 and 5 together."
    assert [text[s:e] for s, e in SG.sentence_clause_parts(text)] == ["I want to know my balance,", "and then please add 17 and 5 together."]
    text = "What is 17 and 5 together"                                  # a bare "and" inside a request is not a clause break
    assert [text[s:e] for s, e in SG.sentence_clause_parts(text)] == [text]


def test_parts_are_meaning_sized_never_a_word_or_two():
    text = "Hi. 17 plus 5? Thanks."                       # "Hi." joins the request after it, and the trailing "Thanks." joins the request before it
    assert [text[s:e] for s, e in SG.sentence_clause_parts(text)] == [text]
    text = "Hi there friend. 17 plus 5? Thanks a lot, bye."
    assert [text[s:e] for s, e in SG.sentence_clause_parts(text)] == ["Hi there friend.", "17 plus 5?", "Thanks a lot, bye."]
    for text in ("a b. c d. e f g h i. j", "ok. 5 plus 6. yes"):
        parts = [text[s:e] for s, e in SG.sentence_clause_parts(text)]
        assert all(len(p.split()) >= 3 for p in parts) or len(parts) == 1


def test_windows_slide_over_words_and_reach_the_end():
    words = [f"w{i}" for i in range(23)]
    text = " ".join(words)
    spans = SG.window_parts(text, width=8, stride=4)
    pieces = [text[s:e].split() for s, e in spans]
    assert all(len(p) == 8 for p in pieces)
    assert pieces[0][0] == "w0" and pieces[-1][-1] == "w22"                 # the tail is covered by a last window anchored at the end
    assert [p[0] for p in pieces[:-1]] == [f"w{i}" for i in range(0, 4 * (len(pieces) - 1), 4)]
    assert set(w for p in pieces for w in p) == set(words)


def test_short_text_is_one_window_and_empty_text_none():
    assert SG.window_parts("only four words here", width=8, stride=4) == [(0, len("only four words here"))]
    assert SG.window_parts("", 8, 4) == [] and SG.window_parts("   ", 8, 4) == []


def test_window_arguments_are_checked():
    for width, stride in ((0, 1), (4, 0), (4, -1), (4, 5)):
        with pytest.raises(ValueError):
            SG.window_parts("a b c d e f", width, stride)
