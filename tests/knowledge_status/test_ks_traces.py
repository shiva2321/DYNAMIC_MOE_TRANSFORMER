"""Task 12, worked traces and relation: "3 times 4: 4 + 4 = 8, 8 + 4 = 12, so 12" is a worked example whose steps are instances of a skill the system already knows (addition), so addition is
named as the parent of multiplication; a division trace is repeated subtraction. Decoys check that the relation is not handed out for free. The machine and the reader are stubs here:
the logic under test is the trace, its chain and the verification."""

import numpy as np
import pytest

import function_check as FC
import generator as G
import skills
import status as ST
import traces as TR
from test_ks_function_check import ExactSolver

NAN = float("nan")
SOLVERS = {s: ExactSolver(s) for s in skills.SKILLS}
VARIANTS = range(TR.N_VARIANTS)


def step_skill_table(trace):
    """What an ideal reader plus machine would say about each step text: the skill of the step."""
    return {st.text: st.skill for st in trace.steps}


def assessor(table, status="solid"):
    def assess(text):
        s = table.get(text)
        if s is None or status != "solid":
            return ST.Assessment("unseen" if status != "solid" else "unseen", None, 0.01, 0, NAN, 1, "")
        return ST.Assessment("solid", s, 0.9, 10, 0.97, 1, s)

    return assess


# ------------------------------------------------------------------------------------------------------------------------------ the traces themselves
@pytest.mark.parametrize("variant", VARIANTS)
def test_mul_trace_steps_chain_to_the_answer(variant):
    rng = np.random.default_rng(variant)
    for _ in range(40):
        t = TR.make_trace(rng, "mul", variant)
        assert t.header_skill == "mul" and t.decoy is None and t.answer == t.a * t.b and len(t.steps) == t.a - 1
        prev = t.b
        for st in t.steps:
            assert st.skill == "add" and (st.a, st.b) == (prev, t.b) and st.result == prev + t.b
            assert FC.parse_numbers(st.text)[:3] == [st.a, st.b, st.result]                       # the step text really shows its numbers, in this order
            prev = st.result
        assert prev == t.answer
        assert FC.parse_numbers(t.text)[0] == t.a and FC.parse_numbers(t.text)[1] == t.b and FC.parse_numbers(t.text)[-1] == t.answer


@pytest.mark.parametrize("variant", VARIANTS)
def test_div_trace_steps_chain_to_the_answer(variant):
    rng = np.random.default_rng(100 + variant)
    for _ in range(40):
        t = TR.make_trace(rng, "div", variant)
        assert t.header_skill == "div" and t.decoy is None and t.answer == t.a // t.b and len(t.steps) == t.answer
        prev = t.a
        for st in t.steps:
            assert st.skill == "sub" and (st.a, st.b) == (prev, t.b) and st.result == prev - t.b
            assert FC.parse_numbers(st.text)[:3] == [st.a, st.b, st.result]
            prev = st.result
        assert prev == t.a % t.b and prev < t.b
        assert FC.parse_numbers(t.text)[0] == t.a and FC.parse_numbers(t.text)[1] == t.b and FC.parse_numbers(t.text)[-1] == t.answer


def test_traces_come_in_several_wordings_and_are_deterministic():
    texts = {TR.make_trace(np.random.default_rng(7), "mul", v).text for v in VARIANTS}
    assert len(texts) == TR.N_VARIANTS
    a, b = TR.make_trace(np.random.default_rng(3), "div", 1), TR.make_trace(np.random.default_rng(3), "div", 1)
    assert a == b
    with pytest.raises(ValueError, match="variant"):
        TR.make_trace(np.random.default_rng(0), "mul", TR.N_VARIANTS)
    with pytest.raises(ValueError, match="skill"):
        TR.make_trace(np.random.default_rng(0), "add", 0)
    with pytest.raises(ValueError, match="decoy"):
        TR.make_trace(np.random.default_rng(0), "mul", 0, decoy="nope")


# ------------------------------------------------------------------------------------------------------------------------------ the verification
@pytest.mark.parametrize("variant", VARIANTS)
def test_parent_set_of_a_mul_trace_is_exactly_add(variant):
    rng = np.random.default_rng(variant)
    for _ in range(25):
        t = TR.make_trace(rng, "mul", variant)
        v = TR.verify_trace(t, assessor(step_skill_table(t)), SOLVERS)
        assert v.parents == frozenset({"add"}) and v.chain_valid and v.verified_steps == len(t.steps)


@pytest.mark.parametrize("variant", VARIANTS)
def test_parent_set_of_a_div_trace_is_exactly_sub(variant):
    rng = np.random.default_rng(50 + variant)
    for _ in range(25):
        t = TR.make_trace(rng, "div", variant)
        v = TR.verify_trace(t, assessor(step_skill_table(t)), SOLVERS)
        assert v.parents == frozenset({"sub"}) and v.chain_valid and v.verified_steps == len(t.steps)


def test_wrong_steps_decoy_is_not_chain_valid():
    rng = np.random.default_rng(1)
    for skill in ("mul", "div"):
        for variant in VARIANTS:
            for _ in range(15):
                t = TR.make_trace(rng, skill, variant, decoy="wrong_steps")
                assert t.decoy == "wrong_steps"
                v = TR.verify_trace(t, assessor(step_skill_table(t)), SOLVERS)
                assert not v.chain_valid and v.parents == frozenset(), (skill, variant, t.text)


def test_irrelevant_valid_steps_decoy_gets_no_relation():
    """A max request with additions that are correct but have nothing to do with the answer: every step is a verified addition, and still no relation."""
    rng = np.random.default_rng(2)
    for variant in VARIANTS:
        for _ in range(15):
            t = TR.make_trace(rng, "max", variant, decoy="irrelevant_valid_steps")
            assert t.header_skill == "max" and t.steps and all(st.result == st.a + st.b for st in t.steps)
            v = TR.verify_trace(t, assessor(step_skill_table(t)), SOLVERS)
            assert v.verified_steps == len(t.steps) and not v.chain_valid and v.parents == frozenset()


def test_unrelated_skill_decoy_gets_no_relation():
    rng = np.random.default_rng(3)
    for variant in VARIANTS:
        for _ in range(15):
            t = TR.make_trace(rng, "mod", variant, decoy="unrelated_skill")
            assert t.header_skill == "mod"
            v = TR.verify_trace(t, assessor(step_skill_table(t)), SOLVERS)
            assert not v.chain_valid and v.parents == frozenset()


def test_relation_needs_the_parent_to_be_solid():
    """If the machine does not hold addition as a Solid skill (it is Unseen or only Thin), the steps are not instances of a known skill and the parent set is empty."""
    t = TR.make_trace(np.random.default_rng(0), "mul", 0)
    unseen = TR.verify_trace(t, assessor(step_skill_table(t), status="unseen"), SOLVERS)
    assert unseen.parents == frozenset() and unseen.verified_steps == 0

    def thin(text):
        return ST.Assessment("thin", "add", 0.5, 3, 0.5, 1, "add")

    thin_v = TR.verify_trace(t, thin, SOLVERS)
    assert thin_v.parents == frozenset() and thin_v.verified_steps == 0


def test_a_step_counts_only_if_the_solver_of_the_skill_reproduces_its_result():
    t = TR.make_trace(np.random.default_rng(0), "mul", 0)

    class Off:
        def predict(self, a, b):
            return np.asarray(a) + np.asarray(b) + 1

    v = TR.verify_trace(t, assessor(step_skill_table(t)), {**SOLVERS, "add": Off()})
    assert v.verified_steps == 0 and v.parents == frozenset()                           # the machine says addition, the solver of addition disagrees


def test_the_machine_must_name_the_right_skill_for_the_step():
    t = TR.make_trace(np.random.default_rng(0), "mul", 0)
    wrong = {st.text: "sub" for st in t.steps}                                           # the machine takes every addition step for a subtraction
    v = TR.verify_trace(t, assessor(wrong), SOLVERS)
    assert v.verified_steps == 0 and v.parents == frozenset()


def test_plain_unseen_message_is_not_given_a_relation():
    plain = TR.Trace("17 times 5", "mul", 17, 5, 85, (), None)
    v = TR.verify_trace(plain, assessor({}), SOLVERS)
    assert v.parents == frozenset() and not v.chain_valid and v.verified_steps == 0


def test_verify_does_not_read_the_operator_symbol():
    """Replace every operator symbol and word in the step texts by a question mark: the numbers are all that is read, so the verdict is unchanged (the stub machine knows the steps by their numbers)."""
    rng = np.random.default_rng(9)
    for skill in ("mul", "div"):
        for variant in VARIANTS:
            t = TR.make_trace(rng, skill, variant)
            masked_steps = tuple(TR.replace_step_text(st, TR.mask_operator(st.text)) for st in t.steps)
            masked = TR.Trace(t.text, t.header_skill, t.a, t.b, t.answer, masked_steps, None)
            by_numbers = {tuple(FC.parse_numbers(st.text)[:3]): st.skill for st in t.steps}

            def assess(text, table=by_numbers):
                s = table.get(tuple(FC.parse_numbers(text)[:3]))
                return ST.Assessment("solid", s, 0.9, 10, 0.97, 1, s) if s else ST.Assessment("unseen", None, 0.01, 0, NAN, 1, "")

            assert all("?" in st.text for st in masked_steps)
            assert TR.verify_trace(masked, assess, SOLVERS) == TR.verify_trace(t, assess, SOLVERS)


def test_a_broken_chain_in_the_middle_is_not_valid_even_when_each_step_is():
    t = TR.make_trace(np.random.default_rng(4), "mul", 0)
    steps = list(t.steps)
    if len(steps) < 2:
        t = TR.make_trace(np.random.default_rng(5), "mul", 0)
        steps = list(t.steps)
    first = steps[0]
    other = TR.Step(f"{first.a + 1} + {first.b} = {first.a + 1 + first.b}", "add", first.a + 1, first.b, first.a + 1 + first.b)
    broken = TR.Trace(t.text, t.header_skill, t.a, t.b, t.answer, (first, other, *steps[2:]) if len(steps) > 1 else (first,), None)
    v = TR.verify_trace(broken, assessor({s.text: s.skill for s in broken.steps}), SOLVERS)
    assert v.verified_steps == len(broken.steps) and not v.chain_valid and v.parents == frozenset()


# ------------------------------------------------------------------------------------------------------------------------------ each part of the chain check is needed on its own
def hand_trace(text, a, b, answer, rows, op):
    steps = tuple(TR.Step(f"{x} {'+' if op == 'add' else '-'} {y} = {r}", op, x, y, r) for x, y, r in rows)
    return TR.Trace(text, "mul" if op == "add" else "div", a, b, answer, steps, None)


def valid(trace):
    return TR.verify_trace(trace, assessor({s.text: s.skill for s in trace.steps}), SOLVERS)


def test_the_right_claims_are_valid_and_each_wrong_claim_is_invalid_for_its_own_reason():
    ok_mul = hand_trace("3 times 4: 4 + 4 = 8, 8 + 4 = 12, so 12", 3, 4, 12, [(4, 4, 8), (8, 4, 12)], "add")
    ok_div = hand_trace("12 divided by 4: 12 - 4 = 8, 8 - 4 = 4, 4 - 4 = 0, 3 times, so 3", 12, 4, 3, [(12, 4, 8), (8, 4, 4), (4, 4, 0)], "sub")
    assert valid(ok_mul).chain_valid and valid(ok_div).chain_valid
    wrong = {
        "mul: starts from the wrong number": hand_trace("3 times 4: 5 + 4 = 9, 9 + 4 = 13, so 13", 3, 4, 13, [(5, 4, 9), (9, 4, 13)], "add"),
        "mul: one step too many": hand_trace("3 times 4: 4 + 4 = 8, 8 + 4 = 12, 12 + 4 = 16, so 16", 3, 4, 16, [(4, 4, 8), (8, 4, 12), (12, 4, 16)], "add"),
        "mul: ends at a different number than the one stated": hand_trace("3 times 4: 4 + 4 = 8, 8 + 4 = 12, so 13", 3, 4, 13, [(4, 4, 8), (8, 4, 12)], "add"),
        "mul: the step size changes": hand_trace("3 times 4: 4 + 4 = 8, 8 + 5 = 13, so 13", 3, 4, 13, [(4, 4, 8), (8, 5, 13)], "add"),
        "div: starts from the wrong number": hand_trace("12 divided by 4: 11 - 4 = 7, 7 - 4 = 3, 2 times, so 2", 12, 4, 2, [(11, 4, 7), (7, 4, 3)], "sub"),
        "div: the number of steps is not the stated answer": hand_trace("12 divided by 4: 12 - 4 = 8, 8 - 4 = 4, 4 - 4 = 0, 2 times, so 2", 12, 4, 2, [(12, 4, 8), (8, 4, 4), (4, 4, 0)], "sub"),
        "div: stops while there is still a whole divisor left": hand_trace("12 divided by 4: 12 - 4 = 8, 8 - 4 = 4, 2 times, so 2", 12, 4, 2, [(12, 4, 8), (8, 4, 4)], "sub"),
        "div: the step size changes": hand_trace("12 divided by 4: 12 - 4 = 8, 8 - 5 = 3, 2 times, so 2", 12, 4, 2, [(12, 4, 8), (8, 5, 3)], "sub"),
    }
    for name, t in wrong.items():
        v = valid(t)
        assert v.verified_steps == len(t.steps) and not v.chain_valid and v.parents == frozenset(), name
