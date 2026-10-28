"""The function check (and the number reader and verifier that Task 12 reuses): does a taught skill's solver reproduce the answer that comes with a message, from the numbers in it?
The numbers do not change with the wording, so this separates a known skill in a new wording from a skill that was never taught without looking at how the message reads."""

import numpy as np
import pytest

import function_check as FC
import generator as G
import skills


class ExactSolver:
    """A solver that is always right (so the tests are about the checking, not about learning)."""

    def __init__(self, skill):
        self.skill = skill

    def predict(self, a, b):
        return np.array([skills.truth(self.skill, int(x), int(y)) if not (self.skill in ("div", "mod") and int(y) == 0) else -10 ** 9 for x, y in zip(a, b)], dtype=np.int64)


SOLVERS = {s: ExactSolver(s) for s in skills.SKILLS}


# ------------------------------------------------------------------------------------------------------------------------------ the number reader
def test_digits_are_read_in_order_and_the_operator_is_ignored():
    assert FC.parse_numbers("17 + 5") == [17, 5] and FC.parse_numbers("17+5") == [17, 5] and FC.parse_numbers("17-5") == [17, 5]
    assert FC.parse_numbers("max(17, 5)") == [17, 5] and FC.parse_numbers("what you get when you add 5 to 17") == [5, 17]
    assert FC.parse_numbers("17 plus 5") == FC.parse_numbers("17 ? 5") == FC.parse_numbers("17 times 5") == [17, 5]          # no operator word is read


def test_number_words_are_read_when_there_are_no_digits():
    assert FC.parse_numbers("seventeen plus five") == [17, 5] and FC.parse_numbers("twenty-one minus eight") == [21, 8]
    assert FC.parse_numbers("what is ninety nine times seven") == [99, 7] and FC.parse_numbers("which is larger, twelve or seven") == [12, 7]
    assert FC.parse_numbers("one hundred twenty three plus four") == [123, 4] and FC.parse_numbers("one thousand two hundred thirty-four minus one") == [1234, 1]


def test_the_word_one_is_not_a_number_when_digits_are_present():
    assert FC.parse_numbers("There are 17 apples in one bag and 5 in another. How many apples altogether?") == [17, 5]
    assert FC.parse_numbers("One jar holds 17 coins and another holds 5.") == [17, 5]


def test_nothing_is_read_from_text_without_numbers_and_typos_in_number_words_are_not_guessed():
    assert FC.parse_numbers("") == [] and FC.parse_numbers("Hope your day is going well.") == []
    assert FC.parse_numbers("sevneteen plus five") == [5]                                                          # a misspelt number word is simply not read


@pytest.mark.parametrize("family", ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"])
def test_the_numbers_of_generated_requests_are_recovered_in_one_order_or_the_other(family):
    rng = np.random.default_rng(0)
    bad = 0
    n = 0
    for skill in skills.SKILLS:
        if family == "S5" and skill not in skills.COMMUTATIVE:
            continue
        for _ in range(60):
            m = G.make_single(rng, "dev", skill, family)
            p = next(x for x in m.parts if x.kind == "skill")
            got = FC.parse_numbers(m.text[p.start:p.end])
            n += 1
            bad += sorted(got) != sorted([p.a, p.b])
    assert bad / n <= 0.01, (family, bad, n)


# ------------------------------------------------------------------------------------------------------------------------------ the verifier
def test_a_story_is_explained_by_the_skill_that_reproduces_its_answer():
    assert FC.explain_one(SOLVERS, ["add", "sub", "max"], "I had 17 apples and got 5 more. How many do I have now?", 22) == ["add"]
    assert FC.explain_one(SOLVERS, ["add", "sub", "max"], "I had 17 apples and gave away 5. How many are left?", 12) == ["sub"]
    assert FC.explain_one(SOLVERS, ["add", "sub", "max"], "Ann has 17 apples and Ben has 5. What is the largest number either of them has?", 17) == ["max"]


def test_a_skill_that_was_never_taught_is_not_explained():
    assert FC.explain_one(SOLVERS, ["add", "sub", "max"], "17 times 5", 85) == []
    assert FC.explain_one(SOLVERS, ["add", "sub", "max"], "the remainder when 17 is divided by 5", 2) == []


def test_the_other_order_of_the_numbers_is_tried():
    assert FC.explain_one(SOLVERS, ["sub"], "5 subtracted from 17", 12) == ["sub"]                                  # the text shows 5 before 17
    assert FC.explain_one(SOLVERS, ["sub"], "17 subtracted from 5", -12) == ["sub"]


class LenientSolver:
    """A solver that does not refuse a zero divisor: it divides by one instead, as a learned network may do (it just returns something)."""

    def __init__(self, skill):
        self.skill = skill

    def predict(self, a, b):
        b = np.where(np.asarray(b) == 0, 1, b)
        return np.asarray(a) // b if self.skill == "div" else np.asarray(a) % b


def test_only_competent_skills_can_explain_and_a_zero_divisor_never_does():
    assert FC.explain_one(SOLVERS, [], "17 plus 5", 22) == []
    assert FC.explain_one(SOLVERS, ["div", "mod"], "17 and 0", 17) == []
    lenient = {"div": LenientSolver("div"), "mod": LenientSolver("mod")}
    assert lenient["div"].predict(np.array([17]), np.array([0]))[0] == 17                           # the solver itself would say 17
    assert FC.explain_one(lenient, ["div", "mod"], "17 and 0", 17) == [] and FC.explain_one(lenient, ["div"], "0 and 17", 0) == ["div"]
    assert FC.explain_one(SOLVERS, ["div"], "how many times 5 goes into 17", 3) == ["div"]


def test_several_skills_may_explain_one_example_and_the_list_says_which():
    assert sorted(FC.explain_one(SOLVERS, ["sub", "mod", "max", "add"], "5 and 5", 0)) == ["mod", "sub"]


def test_text_that_does_not_give_exactly_two_numbers_is_not_explained():
    for text in ("17 apples", "17 plus 5 plus 3", "no numbers here", ""):
        assert FC.explain_one(SOLVERS, list(skills.SKILLS), text, 22) == []


def test_the_batch_version_gives_the_same_answers_as_one_at_a_time():
    texts = ["I had 17 apples and got 5 more.", "17 times 5", "5 subtracted from 17", "seventeen plus five", "no numbers"]
    answers = [22, 85, 12, 22, 3]
    one = [FC.explain_one(SOLVERS, ["add", "sub", "max"], t, a) for t, a in zip(texts, answers)]
    assert FC.explain_many(SOLVERS, ["add", "sub", "max"], texts, answers) == one


def test_no_operator_word_is_used_to_decide_the_skill():
    """Replace the operator wording of S2 style requests by a question mark: the explaining skills do not change, because the numbers and the answer are all that is read."""
    for skill, a, b in (("add", 17, 5), ("mul", 12, 3), ("sub", 30, 4)):
        text, masked = f"{G.number_to_words(a)} plus {G.number_to_words(b)}", f"{G.number_to_words(a)} ? {G.number_to_words(b)}"
        ans = skills.truth(skill, a, b)
        assert FC.explain_one(SOLVERS, list(skills.SKILLS), text, ans) == FC.explain_one(SOLVERS, list(skills.SKILLS), masked, ans)
