"""The group function test (routing_lab/group_check.py): perfect solvers, the hit matrix, the permutation null, windows, overturning."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "knowledge_status" / "routing_lab"))

import group_check as GC  # noqa: E402
import skills  # noqa: E402

SOLVERS = {s: GC.PerfectSolver(s) for s in skills.SKILLS}


def parts(skill, pairs):
    return [list(p) for p in pairs], [skills.truth(skill, a, b) for a, b in pairs]


def test_perfect_solvers_match_the_truth_function_for_every_skill():
    rng = np.random.default_rng(1)
    a, b = rng.integers(1, 100, 200), rng.integers(1, 100, 200)
    for s in skills.SKILLS:
        got = GC.PerfectSolver(s).predict(a, b)
        assert got.tolist() == [skills.truth(s, int(x), int(y)) for x, y in zip(a, b)]
    assert GC.PerfectSolver("div").predict(np.array([7]), np.array([0])).tolist() == [0]
    with pytest.raises(KeyError):
        GC.PerfectSolver("pow")


def test_hit_matrix_reads_either_order_and_skips_parts_without_two_numbers():
    nums = [[17, 5], [5, 17], [3], [40, 8]]
    ans = [12, 12, 99, 32]                                                                       # 17 - 5 = 12, 5 - 17 is not, 40 - 8 = 32
    H = GC.hit_matrix(SOLVERS["sub"], nums, ans, "sub")
    assert H[0, 0] == 1 and H[1, 1] == 1                                                         # (5, 17) read the other way round is 17 - 5
    assert H[3, 3] == 1 and H[0, 1] == 1 and H[2].sum() == 0 and H[:, 2].sum() == 0


def test_hit_matrix_never_lets_a_zero_divisor_explain_div_or_mod():
    H = GC.hit_matrix(SOLVERS["div"], [[7, 0]], [5], "div")                                          # (7, 0) is skipped, (0, 7) gives 0 // 7 == 0, not 5
    assert H.sum() == 0
    assert GC.hit_matrix(SOLVERS["div"], [[7, 0]], [0], "div")[0, 0] == 1                          # either order is tried: 0 // 7 == 0 is a legitimate hit, 7 // 0 is never computed
    assert GC.hit_matrix(SOLVERS["mod"], [[0, 7]], [0], "mod")[0, 0] == 1


def test_permutation_p_is_tiny_for_a_matched_window_and_large_for_chance():
    rng = np.random.default_rng(0)
    nums, ans = parts("add", [(11, 22), (33, 44), (55, 12), (61, 29), (73, 15), (88, 7), (46, 31), (90, 6)])
    H = GC.hit_matrix(SOLVERS["add"], nums, ans, "add")
    assert GC.permutation_p(H, rng, 400) == pytest.approx(1 / 401)                                 # only the identity shuffle reaches 8 of 8 (1 in 40320)
    nums2, ans2 = parts("mul", [(11, 22), (33, 44), (55, 12), (61, 29), (73, 15), (88, 7), (46, 31), (90, 6)])
    H2 = GC.hit_matrix(SOLVERS["add"], nums2, ans2, "add")
    assert GC.permutation_p(H2, np.random.default_rng(0), 400) > 0.5
    assert GC.permutation_p(H[:1, :1], rng) == 1.0


def test_a_window_of_known_parts_shows_their_skill_present_and_not_the_others():
    rng = np.random.default_rng(3)
    nums, ans = parts("add", [(11, 22), (33, 44), (55, 12), (61, 29), (73, 15), (88, 7)])
    res = GC.explained_window(SOLVERS, ["add", "sub", "mul"], nums, ans, rng, alpha=0.01)
    assert res["add"]["present"] and res["add"]["count"] == 6 and res["add"]["p"] < 0.005 and res["add"]["parts"] == [0, 1, 2, 3, 4, 5]
    assert not res["sub"]["present"] and not res["mul"]["present"]


def test_a_window_of_parts_of_an_untaught_skill_has_no_present_skill_even_if_one_part_coincides():
    rng = np.random.default_rng(4)
    nums, ans = parts("mul", [(11, 22), (33, 14), (25, 12), (61, 9), (73, 15), (88, 7)])
    nums[0], ans[0] = [11, 1], 12                                                                  # one coincidence: add(11, 1) == 12 == stated answer
    res = GC.explained_window(SOLVERS, ["add", "sub"], nums, ans, rng, alpha=0.01)
    assert res["add"]["count"] >= 1 and not any(r["present"] for r in res.values())


def test_overturn_takes_back_only_explained_parts_in_a_mixed_window():
    rng = np.random.default_rng(5)
    nums_add, ans_add = parts("add", [(11, 22), (33, 44), (55, 12), (61, 29), (73, 15), (88, 7)])
    nums_mul, ans_mul = parts("mul", [(23, 37), (45, 18), (51, 17), (62, 13)])
    numbers = dict(enumerate(nums_add + nums_mul))
    answers = dict(enumerate(ans_add + ans_mul))
    got = GC.overturn(list(range(10)), numbers, answers, SOLVERS, ["add", "sub"], window=10, rng=rng, nearest={0: "add"})
    assert sorted(got) == [0, 1, 2, 3, 4, 5] and set(got.values()) == {"add"}


def test_overturn_prefers_the_nearest_skill_when_two_present_skills_explain_a_part():
    rng = np.random.default_rng(6)
    nums, ans = parts("max", [(40, 3), (50, 7), (60, 8), (70, 5), (80, 9), (90, 2)])
    nums = [list(n) for n in nums]
    # sub(a, b) with a > b is not max; make the answers fit both max and add by using b = 0 style pairs: max(a, 0) == a == add(a, 0)
    nums = [[a, 0] for a, _ in nums]
    ans = [a for a, _ in nums]
    got = GC.overturn(list(range(6)), dict(enumerate(nums)), dict(enumerate(ans)), SOLVERS, ["add", "max"], window=6, rng=rng, nearest={i: "max" for i in range(6)})
    assert set(got.values()) == {"max"} and len(got) == 6
    got2 = GC.overturn(list(range(6)), dict(enumerate(nums)), dict(enumerate(ans)), SOLVERS, ["add", "max"], window=6, rng=np.random.default_rng(6), nearest={})
    assert len(got2) == 6 and set(got2.values()) <= {"add", "max"}


def test_overturn_skips_windows_of_one_and_runs_with_no_competent_skill():
    numbers, answers = {0: [1, 2], 1: [3, 4], 2: [5, 6]}, {0: 3, 1: 7, 2: 11}
    assert GC.overturn([0, 1, 2], numbers, answers, SOLVERS, [], window=3, rng=np.random.default_rng(0)) == {}
    assert GC.overturn([0, 1, 2], numbers, answers, SOLVERS, ["add"], window=1, rng=np.random.default_rng(0)) == {}


def test_a_given_grouping_replaces_the_consecutive_windows():
    rng = np.random.default_rng(7)
    nums, ans = parts("add", [(11, 22), (33, 44), (55, 12), (61, 29), (73, 15), (88, 7), (46, 31), (90, 6)])
    numbers, answers = dict(enumerate(nums)), dict(enumerate(ans))
    rows = list(range(8))
    assert GC.overturn(rows, numbers, answers, SOLVERS, ["add"], window=2, rng=rng) == {}        # windows of two can never reach the threshold
    got = GC.overturn(rows, numbers, answers, SOLVERS, ["add"], window=2, rng=np.random.default_rng(7), grouping=[rows])
    assert sorted(got) == rows


def test_holm_stops_at_the_first_skill_that_fails_its_threshold():
    rng = np.random.default_rng(8)
    nums, ans = parts("add", [(11, 22), (33, 44), (55, 12), (61, 29), (73, 15), (88, 7)])
    res = GC.explained_window(SOLVERS, ["sub", "add"], nums, ans, rng, alpha=0.01, n_perm=200)
    assert res["add"]["present"] and res["add"]["p"] < res["sub"]["p"]


def test_a_zero_divisor_never_explains_in_either_order_and_other_skills_may_use_a_zero_operand():
    assert GC.hit_matrix(SOLVERS["div"], [[0, 0]], [0], "div").sum() == 0                        # 0 // 0 is never computed, so the solver's placeholder 0 cannot "explain" an answer of 0
    assert GC.hit_matrix(SOLVERS["mod"], [[0, 0]], [0], "mod").sum() == 0
    assert GC.hit_matrix(SOLVERS["sub"], [[5, 0]], [5], "sub")[0, 0] == 1                        # a zero second operand is fine for subtraction: 5 - 0 == 5 (the other order gives -5)


def test_overturn_follows_the_nearest_skill_whichever_one_it_is():
    nums = [[a, 0] for a in (40, 50, 60, 70, 80, 90)]
    ans = [a for a, _ in nums]
    for near in ("add", "max"):
        got = GC.overturn(list(range(6)), dict(enumerate(nums)), dict(enumerate(ans)), SOLVERS, ["add", "max"], window=6, rng=np.random.default_rng(6), nearest={i: near for i in range(6)})
        assert set(got.values()) == {near} and len(got) == 6
