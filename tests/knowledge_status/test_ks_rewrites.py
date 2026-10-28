"""Generic rewrites of teaching sentences (option (a), wording diversity in teaching): how they are asked for, which ones are kept, and how they are cached. No language model here:
the generator is passed in, so every rule is tested on text we control."""

import json

import numpy as np
import pytest

import rewrites as RW


class Fake:
    """A generator that returns canned text per prompt and counts its calls."""

    def __init__(self, table=None, default=None):
        self.table, self.default, self.calls = table or {}, default, 0

    def __call__(self, prompts, seed):
        self.calls += len(prompts)
        return [self.table.get(p, self.default if self.default is not None else "17 and 5 put together") for p in prompts]


def test_the_instructions_name_no_test_wording():
    """The protocol of option (a): the rewrites are GENERIC. No instruction may name a wording family of the test bed (stories, noise, register, chatter), nor ask for a style."""
    banned = ("story", "stories", "typo", "misspell", "spelling", "case", "capital", "space", "noise", "formal", "casual", "terse", "polite", "slang", "chatter", "banter", "joke", "friendly")
    assert len(RW.INSTRUCTIONS) >= 3
    for text in RW.INSTRUCTIONS:
        low = text.lower()
        assert not any(b in low for b in banned), text
        assert "number" in low and "meaning" in low                                   # every instruction says to keep the numbers and the meaning


def test_the_prompt_carries_the_instruction_and_the_source_and_nothing_about_the_skill():
    p = RW.build_prompt(RW.INSTRUCTIONS[0], "17 plus 5")
    assert RW.INSTRUCTIONS[0] in p and "17 plus 5" in p
    for skill in ("addition", "subtraction", "multiplication", "division", "remainder", "maximum", "arithmetic", "calculate"):
        assert skill not in p.lower()


@pytest.mark.parametrize("source,rewrite,ok", [
    ("17 plus 5", "What do you get if you add 5 to 17?", True),
    ("17 plus 5", "What do you get if you add 5 to 18?", False),                       # a number changed
    ("17 plus 5", "What do you get if you add 5 to 17 and 3?", False),                 # a number added
    ("seventeen plus five", "what is 17 and 5 added together", True),                  # words to digits: same numbers
    ("17 plus 5", "17 plus 5", False),                                                 # nothing changed
    ("17 plus 5", "  17   PLUS 5 ", False),                                            # nothing but case and spacing
    ("17 plus 5", "Sure! Here is the rewritten request: add 5 to 17.", False),         # the model talking about its answer
    ("17 plus 5", "add 5 to 17\nor maybe 17 plus 5", False),                           # more than one line
    ("17 plus 5", "", False),
    ("17 plus 5", "add 5 to 17 " + "and so on " * 40, False),                           # far too long
    ("I had 17 apples and got 5 more", "Five more apples were added to my 17 apples", True),
    ("17 / 5", "how many times does 5 fit into 17", True),
])
def test_which_rewrites_are_kept(source, rewrite, ok):
    assert RW.keep(source, rewrite) is ok


def test_rewrites_are_generated_once_per_source_and_instruction_and_cached(tmp_path):
    gen = Fake(default="add 5 to 17 please")
    out = RW.rewrites_for(["17 plus 5"], RW.INSTRUCTIONS[:2], gen, tmp_path, seed=1)
    assert out == {"17 plus 5": ["add 5 to 17 please"]} and gen.calls == 2                # two instructions, identical output kept once
    again = RW.rewrites_for(["17 plus 5"], RW.INSTRUCTIONS[:2], gen, tmp_path, seed=1)
    assert again == out and gen.calls == 2                                                  # nothing generated the second time


def test_the_filter_can_change_without_generating_again(tmp_path, monkeypatch):
    gen = Fake(default="add 5 to 17 please")
    RW.rewrites_for(["17 plus 5"], RW.INSTRUCTIONS[:1], gen, tmp_path, seed=1)
    monkeypatch.setattr(RW, "keep", lambda s, r: False)
    assert RW.rewrites_for(["17 plus 5"], RW.INSTRUCTIONS[:1], gen, tmp_path, seed=1) == {"17 plus 5": []} and gen.calls == 1


def test_a_damaged_cache_block_is_generated_again(tmp_path):
    gen = Fake(default="add 5 to 17 please")
    RW.rewrites_for(["17 plus 5"], RW.INSTRUCTIONS[:1], gen, tmp_path, seed=1)
    for f in tmp_path.glob("*.json"):
        f.write_text('{"raw": ["add 5', encoding="utf-8")
    out = RW.rewrites_for(["17 plus 5"], RW.INSTRUCTIONS[:1], gen, tmp_path, seed=1)
    assert out == {"17 plus 5": ["add 5 to 17 please"]} and gen.calls == 2


def test_the_cache_does_not_mix_instructions_or_seeds(tmp_path):
    a, b = Fake(default="add 5 to 17 please"), Fake(default="put 5 with 17")
    RW.rewrites_for(["17 plus 5"], RW.INSTRUCTIONS[:1], a, tmp_path, seed=1)
    assert RW.rewrites_for(["17 plus 5"], RW.INSTRUCTIONS[1:2], b, tmp_path, seed=1) == {"17 plus 5": ["put 5 with 17"]}
    assert RW.rewrites_for(["17 plus 5"], RW.INSTRUCTIONS[:1], b, tmp_path, seed=2) == {"17 plus 5": ["put 5 with 17"]}


def test_many_sources_are_batched_and_every_source_gets_an_entry(tmp_path):
    srcs = [f"{i} plus {i + 1}" for i in range(1, 150)]
    gen = Fake(default="")
    out = RW.rewrites_for(srcs, RW.INSTRUCTIONS[:1], gen, tmp_path, seed=0, batch=64)
    assert set(out) == set(srcs) and all(v == [] for v in out.values()) and gen.calls == 149
    assert len(list(tmp_path.glob("*.json"))) == 3                                          # 64 + 64 + 21


def test_duplicate_sources_are_generated_once_and_empty_input_does_nothing(tmp_path):
    gen = Fake(default="add 5 to 17 please")
    assert RW.rewrites_for([], RW.INSTRUCTIONS, gen, tmp_path, seed=0) == {} and gen.calls == 0
    RW.rewrites_for(["17 plus 5", "17 plus 5"], RW.INSTRUCTIONS[:1], gen, tmp_path, seed=0)
    assert gen.calls == 1


def test_order_and_content_do_not_depend_on_the_batch_a_source_lands_in(tmp_path):
    srcs = [f"{i} plus {i + 1}" for i in range(1, 20)]
    table = {RW.build_prompt(RW.INSTRUCTIONS[0], s): f"add {s.split()[2]} to {s.split()[0]} please" for s in srcs}
    a = RW.rewrites_for(srcs, RW.INSTRUCTIONS[:1], Fake(table), tmp_path / "a", seed=0, batch=4)
    b = RW.rewrites_for(srcs[::-1], RW.INSTRUCTIONS[:1], Fake(table), tmp_path / "b", seed=0, batch=7)
    assert a == b


# ------------------------------------------------------------------------------------------------------------------------------ the answer check
class Solver:
    """Canned answers by rewrite text; counts the prompts it is asked."""

    def __init__(self, table):
        self.table, self.calls = table, 0

    def __call__(self, prompts, seed):
        self.calls += len(prompts)
        return [self.table.get(p.split("\n\n", 1)[1], "no idea") for p in prompts]


def test_a_rewrite_is_kept_only_if_its_answer_matches_the_true_answer(tmp_path):
    rws = {"90 modulo 3": ["What is the remainder when 90 is divided by 3?", "Result of 90 divided by 3, disregarding any remainder", "2 is not a factor of 47"], "5 plus 3": []}
    solver = Solver({"What is the remainder when 90 is divided by 3?": "0", "Result of 90 divided by 3, disregarding any remainder": "30", "2 is not a factor of 47": "True"})
    out = RW.verify(rws, {"90 modulo 3": 0, "5 plus 3": 8}, solver, tmp_path)
    assert out == {"90 modulo 3": ["What is the remainder when 90 is divided by 3?"], "5 plus 3": []}


def test_the_answer_is_the_first_whole_number_in_the_reply():
    assert RW.first_int("22") == 22 and RW.first_int("The answer is 22.") == 22 and RW.first_int("-5") == -5 and RW.first_int("It is 7 or 8") == 7
    assert RW.first_int("no idea") is None and RW.first_int("") is None


def test_the_solver_prompt_asks_for_one_number_and_names_no_operation():
    p = RW.solve_prompt("add 5 to 17")
    assert "single whole number" in p and p.endswith("add 5 to 17") and "remainder" not in p.lower()


def test_answers_are_solved_once_cached_and_a_damaged_block_is_solved_again(tmp_path):
    rws = {"17 plus 5": ["add 5 to 17 please"]}
    solver = Solver({"add 5 to 17 please": "22"})
    RW.verify(rws, {"17 plus 5": 22}, solver, tmp_path)
    RW.verify(rws, {"17 plus 5": 22}, solver, tmp_path)
    assert solver.calls == 1
    for f in tmp_path.glob("*.json"):
        f.write_text('{"raw": ["2', encoding="utf-8")
    assert RW.verify(rws, {"17 plus 5": 22}, solver, tmp_path) == rws and solver.calls == 2


def test_a_source_without_a_known_answer_keeps_nothing(tmp_path):
    out = RW.verify({"17 plus 5": ["add 5 to 17 please"]}, {}, Solver({"add 5 to 17 please": "22"}), tmp_path)
    assert out == {"17 plus 5": []}
