"""Experts that share a number encoder (routing_lab/experts.py, lab_experts.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "knowledge_status" / "routing_lab"))

import experts as EX  # noqa: E402
import lab_experts as LE  # noqa: E402
from growmoe import arith  # noqa: E402


def test_decode_reads_digits_least_significant_first_and_the_sign_slot():
    sym = np.full((3, arith.N_POS), arith.PAD)
    sym[0, :3] = [3, 2, 1]
    sym[1, :2] = [5, 4]
    sym[1, arith.SIGN_POS] = arith.NEG
    sym[2, 0] = arith.NEG                                                                       # a non-digit symbol in a digit slot counts as 0
    assert EX.decode(sym).tolist() == [123, -45, 0]


def test_the_experts_need_distinct_names_and_matching_data_and_a_fit_before_predicting():
    with pytest.raises(ValueError):
        EX.SharedExperts(["add", "add"])
    with pytest.raises(ValueError):
        EX.SharedExperts([])
    e = EX.SharedExperts(["add", "sub"], steps=2, device="cpu")
    with pytest.raises(ValueError, match="data for"):
        e.fit({"add": LE.draw(0, "add", 20)})
    with pytest.raises(RuntimeError, match="fit the experts"):
        e.predict("add", np.array([1]), np.array([2]))


def test_a_tiny_fit_runs_predicts_integers_and_refuses_an_unknown_expert():
    e = EX.SharedExperts(["add", "sub"], steps=5, hidden=16, device="cpu").fit({"add": LE.draw(0, "add", 30), "sub": LE.draw(0, "sub", 30)})
    out = e.predict("add", np.array([12, 7]), np.array([30, 5]))
    assert out.shape == (2,) and out.dtype.kind == "i"
    assert e.solver("sub").predict(np.array([9]), np.array([4])).shape == (1,)
    with pytest.raises(KeyError):
        e.predict("mul", np.array([1]), np.array([2]))


def test_shared_experts_learn_a_skill_from_many_examples():
    data = {"add": LE.draw(1, "add", 600), "max": LE.draw(1, "max", 600)}
    e = EX.SharedExperts(["add", "max"], seed=1, steps=1500, hidden=64, device="cpu").fit(data)
    acc = EX.held_out_accuracy(e.solver("max").predict, 5, np.random.default_rng(0), n=200)
    assert acc > 0.3                                                                           # far above chance (0.06) after a short run


def test_draw_is_deterministic_and_uses_the_training_split():
    a1 = LE.draw(3, "mul", 50)
    a2 = LE.draw(3, "mul", 50)
    assert all((x == y).all() for x, y in zip(a1, a2)) and len(a1[0]) == 50
    assert not arith.heldout(np.full(50, 2), a1[0], a1[1]).any()


def test_held_out_accuracy_scores_exact_answers():
    perfect = lambda a, b: a + b
    assert EX.held_out_accuracy(perfect, 0, np.random.default_rng(0), n=100) == 1.0
    assert EX.held_out_accuracy(lambda a, b: a + b + 1, 0, np.random.default_rng(0), n=100) == 0.0
