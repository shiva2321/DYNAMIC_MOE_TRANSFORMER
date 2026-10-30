"""Shift diagnostics of the general routing lab (routing_general/shift.py): rewrites of the roster's own held-out messages, a trust rule, a rank correlation."""

import sys
from pathlib import Path

import numpy as np
import pytest

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
for p in (_KS, _KS / "routing_general"):
    sys.path.insert(0, str(p))

import shift as SH  # noqa: E402


def test_the_generic_filter_keeps_a_real_rewrite_and_drops_the_rest():
    src = "I want to know when my card will arrive"
    assert SH.keep_general(src, "When is my card going to get here?")
    assert not SH.keep_general(src, src.upper())                                      # the same text in other case is not a rewrite
    assert not SH.keep_general(src, "")
    assert not SH.keep_general(src, "Sure, here is the rewrite: when will my card come")
    assert not SH.keep_general(src, "line one\nline two")
    assert not SH.keep_general(src, " ".join(["word"] * (SH.MAX_WORDS + 1)))
    assert SH.keep_general("pay 17 by 5", "settle 5 and 17 together")                  # digits kept, order free
    assert not SH.keep_general("pay 17 by 5", "pay 18 by 5")                          # a digit that changed
    assert not SH.keep_general("pay 17 by 5", "pay it by then")                       # a digit that vanished


def test_calibration_is_split_per_class_into_a_cut_half_and_a_source_half():
    cal = [(f"m{c}{j}", c) for c in range(3) for j in range(6)]
    a, b = SH.split_calibration(cal)
    assert len(a) == len(b) == 9 and not {t for t, _ in a} & {t for t, _ in b}
    for c in range(3):
        assert sum(1 for _, l in a if l == c) == sum(1 for _, l in b if l == c) == 3
    assert SH.split_calibration(cal) == (a, b)                                          # deterministic
    assert SH.cap_per_class(b, 2) == [x for c in range(3) for x in [y for y in b if y[1] == c][:2]]
    odd = SH.split_calibration([("x", 0), ("y", 0), ("z", 0)])
    assert len(odd[0]) == 2 and len(odd[1]) == 1


def test_shift_set_writes_rewrites_with_the_label_of_their_source_and_caches_them(tmp_path):
    calls = []

    def gen(prompts, seed):
        calls.append(list(prompts))
        return [f"reworded: {p.split('Request: ')[-1]}" for p in prompts]

    src = [("alpha beta", 0), ("gamma delta", 1), ("alpha beta", 0)]
    out = SH.shift_set(src, gen, tmp_path, seed=3, keep=lambda s, r: True, instructions=("I1", "I2"), batch=2)
    assert sorted(out) == [("reworded: alpha beta", 0), ("reworded: gamma delta", 1)]       # each source once, the label of its source, the second instruction's copies dropped
    n_calls = len(calls)
    assert SH.shift_set(src, gen, tmp_path, seed=3, keep=lambda s, r: True, instructions=("I1", "I2"), batch=2) == out and len(calls) == n_calls     # second time: from the cache
    only_alpha = SH.shift_set(src, gen, tmp_path, seed=3, keep=lambda s, r: "gamma" not in r, instructions=("I1", "I2"), batch=2)   # the filter runs when the cache is read
    assert only_alpha == [("reworded: alpha beta", 0)] and len(calls) == n_calls
    with pytest.raises(RuntimeError):
        SH.shift_set([("a b", 0)], lambda prompts, seed: [], tmp_path / "bad", seed=0, instructions=("I",))


def test_trust_rule_keeps_readers_under_the_limit_or_the_least_shifted_one():
    flags = {"ask": 0.06, "vmf": 0.30, "maha": 0.09}
    assert SH.trusted(flags, 0.10) == ["ask", "maha"]
    assert SH.trusted({"a": 0.5, "b": 0.3, "c": 0.4}, 0.10) == ["b"]                        # nobody under the limit: the least shifted one
    assert SH.trusted({"a": 0.3, "b": 0.3}, 0.10) == ["a"]                                    # a tie goes to the first
    with pytest.raises(ValueError):
        SH.trusted({}, 0.1)


def test_spearman_handles_ties_and_constants():
    assert SH.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert SH.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert SH.spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)
    assert np.isnan(SH.spearman([1, 1, 1], [1, 2, 3]))
    with pytest.raises(ValueError):
        SH.spearman([1, 2], [1])
