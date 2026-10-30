"""Cards and the domain-free question (routing_general/cards.py)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "knowledge_status" / "routing_general"))

import cards as CD  # noqa: E402

TWO = [CD.Card("addition", ("17 plus 5", "what is 3 and 4 together")), CD.Card("card arrival", ("when will my card get here",))]


def test_options_are_the_cards_in_order_and_a_last_none_option():
    lines = CD.option_lines(TWO, with_exemplars=False)
    assert lines[0] == "A. addition"
    assert lines[1] == "B. card arrival"
    assert lines[2].startswith("C. none of these")
    assert len(lines) == 3


def test_exemplars_are_shown_in_quotes_when_asked_and_cut_to_one_line():
    long = CD.Card("weather", ("it will\nrain\ttomorrow " + "very " * 60 + "much",))
    line = CD.option_lines([long], with_exemplars=True)[0]
    assert line.startswith('A. weather (examples: "it will rain tomorrow very')
    assert "\n" not in line and "\t" not in line
    assert line.count('"') == 2
    assert "..." in line                                                                     # the cut is visible, never silent
    assert len(line) < len('A. weather (examples: "") ') + CD.EXEMPLAR_CHARS + 10
    assert CD.option_lines(TWO, with_exemplars=True)[0] == 'A. addition (examples: "17 plus 5"; "what is 3 and 4 together")'
    assert CD.option_lines([CD.Card("empty")], with_exemplars=True)[0] == "A. empty"          # a card without exemplars shows only its name


def test_prompt_holds_the_message_and_every_name_and_no_domain_word():
    p = CD.prompt_text("when does my card arrive?", TWO, with_exemplars=False)
    assert "when does my card arrive?" in p
    assert "addition" in p and "card arrival" in p and "none of these" in p
    assert p.rstrip().endswith("Answer with one letter.")
    template = CD.prompt_text("X", [CD.Card("Y")], with_exemplars=False).lower()
    for word in ("arithmetic", "calculat", "number", "operation", "bank", "intent", "topic", "sentiment"):
        assert word not in template, f"the template mentions {word!r}: it must stay domain-free"


def test_card_set_is_validated():
    with pytest.raises(ValueError):
        CD.option_lines([], with_exemplars=False)
    with pytest.raises(ValueError):
        CD.option_lines([CD.Card("a"), CD.Card("a")], with_exemplars=False)
    with pytest.raises(ValueError):
        CD.option_lines([CD.Card(" ")], with_exemplars=False)
    with pytest.raises(ValueError):
        CD.option_lines([CD.Card(f"c{i}") for i in range(CD.MAX_CARDS + 1)], with_exemplars=False)
    assert len(CD.option_lines([CD.Card(f"c{i}") for i in range(CD.MAX_CARDS)], with_exemplars=False)) == CD.MAX_CARDS + 1


def test_signature_names_the_question_and_nothing_else():
    a = CD.signature(TWO, False)
    assert a == CD.signature(list(TWO), False)
    assert a != CD.signature(TWO, True)
    assert a != CD.signature(TWO[::-1], False)                                                # the order of the options is part of the question
    assert a != CD.signature([CD.Card("addition"), CD.Card("card arrival 2")], False)
    assert CD.signature([CD.Card("addition", ("x",)), TWO[1]], False) == CD.signature([CD.Card("addition", ("y",)), TWO[1]], False)   # exemplars do not matter when they are not shown


def test_letters_cover_the_options_exactly():
    assert CD.letters_for(TWO) == ["A", "B", "C"]
    assert CD.letters_for([CD.Card("only")]) == ["A", "B"]
