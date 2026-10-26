import numpy as np
import pytest

import assignments as A
import skills
from growmoe import arith


def test_assignment_is_not_tied_to_the_skill_identity():
    rng = np.random.default_rng(0)
    seen = {s: set() for s in skills.SKILLS}
    for _ in range(200):
        asg = A.draw_assignment(rng)
        assert set(asg) == set(skills.SKILLS)
        assert sorted(asg.values()) == ["solid", "solid", "thin", "unseen", "unseen", "unseen"]
        for s, state in asg.items():
            seen[s].add(state)
    assert all(v == {"solid", "thin", "unseen"} for v in seen.values()), seen


def test_need_solid_is_respected():
    rng = np.random.default_rng(1)
    for need in ({"add"}, {"sub"}, {"add", "sub"}):                                     # mul runs force add Solid; div runs force sub Solid
        for _ in range(100):
            asg = A.draw_assignment(rng, need_solid=frozenset(need))
            assert all(asg[s] == "solid" for s in need) and list(asg.values()).count("solid") == 2 and list(asg.values()).count("thin") == 1
    thin_sets = {tuple(sorted(s for s, st in A.draw_assignment(rng, frozenset({"add"})).items() if st == "thin")) for _ in range(100)}
    assert len(thin_sets) > 2                                                           # the forced skill is fixed, the rest still rotate
    with pytest.raises(ValueError):
        A.draw_assignment(rng, need_solid=frozenset({"add", "sub", "mul"}))             # three forced Solid skills do not fit in n_solid=2
    with pytest.raises(ValueError):
        A.draw_assignment(rng, need_solid=frozenset({"no_such_skill"}))
    with pytest.raises(ValueError):
        A.draw_assignment(rng, n_solid=4, n_thin=3)


@pytest.mark.parametrize("count,acc,chance,expected", [
    (500, 0.90, 0.05, "solid"), (499, 0.90, 0.05, "grey"), (2000, 0.95, 0.05, "solid"), (2000, 0.89, 0.05, "grey"),
    (20, 0.30, 0.05, "thin"), (20, 0.95, 0.05, "thin"), (50, 0.95, 0.05, "thin"), (100, 0.99, 0.05, "thin"), (101, 0.99, 0.05, "grey"),
    (2000, 0.40, 0.05, "thin"), (2000, 0.60, 0.05, "thin"), (2000, 0.61, 0.05, "grey"),
    (2000, 0.14, 0.05, "grey"), (2000, 0.15, 0.05, "thin"),                              # chance + 0.10 is 0.15000000000000002 in floating point; the boundary is still inclusive
    (500, 0.55, 0.50, "grey"), (500, 0.60, 0.50, "thin"),                                # a high chance level (always answering "0") narrows the Thin band
])
def test_status_truth_boundaries(count, acc, chance, expected):
    assert A.status_truth(count, acc, chance) == expected


def test_status_truth_reads_the_thresholds_at_call_time_and_rejects_nonsense(monkeypatch):
    monkeypatch.setitem(A.THRESHOLDS, "solid_min_acc", 0.80)
    assert A.status_truth(2000, 0.85, 0.05) == "solid"
    for bad in ((0, 0.5, 0.1), (-3, 0.5, 0.1), (10, 1.2, 0.1), (10, -0.1, 0.1), (10, 0.5, 1.5)):
        with pytest.raises(ValueError):
            A.status_truth(*bad)


def test_teaching_and_test_examples_follow_their_rules():
    rng = np.random.default_rng(2)
    for skill in ("add", "div", "max"):
        teach = A.draw_teaching(rng, "dev", skill, 300)
        test = A.draw_test(rng, "dev", skill, 300)
        assert len(teach) == len(test) == 300
        for ex in teach:
            assert ex.skill == skill and ex.family in A.TEACH_FAMILIES and ex.answer == skills.truth(skill, ex.a, ex.b)
            assert not arith.heldout(np.array([skills.ARITH_OP_ID[skill]]), np.array([ex.a]), np.array([ex.b]))[0]
        for ex in test:
            assert ex.skill == skill and ex.family in A.TEST_FAMILIES and ex.answer == skills.truth(skill, ex.a, ex.b)
            assert arith.heldout(np.array([skills.ARITH_OP_ID[skill]]), np.array([ex.a]), np.array([ex.b]))[0]
    assert "S5" not in {ex.family for ex in A.draw_test(rng, "dev", "div", 300)}        # operands in the other order exist only for commutative skills
    assert "S5" in {ex.family for ex in A.draw_test(rng, "dev", "add", 300)}
    with pytest.raises(ValueError):
        A.draw_teaching(rng, "dev", "no_such_skill", 5)


def test_no_identical_text_between_teaching_and_test_sets():
    rng = np.random.default_rng(3)
    for skill in ("add", "mul"):
        teach = A.draw_teaching(rng, "dev", skill, 1000)
        test = A.draw_test(rng, "dev", skill, 600, taught_keys=A.text_keys(teach))
        assert not (A.text_keys(teach) & A.text_keys(test))
        assert not ({e.template_ids[0] for e in teach} & {e.template_ids[0] for e in test})   # a test frame is never a teaching frame
        assert not ({e.template_ids for e in teach} & {e.template_ids for e in test})         # nor is a (frame, request wording) pair
        test_cores = {e.template_ids[1] for e in test if e.family in ("S4", "S5")}
        assert test_cores and not (test_cores & {e.template_ids[1] for e in teach})           # S4 and S5 bring request wordings that were never taught


def test_text_keys_ignore_case_and_spacing():
    rng = np.random.default_rng(4)
    ex = A.draw_teaching(rng, "dev", "add", 1)[0]
    shouted = ex.__class__(**{**ex.__dict__, "text": "  " + ex.text.upper().replace(" ", "   ") + " "})
    assert A.text_keys([ex]) == A.text_keys([shouted])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_solver_is_deterministic_for_a_seed(device):
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no GPU on this machine")
    rng = np.random.default_rng(5)
    teach = A.draw_teaching(rng, "dev", "add", 200)
    a, b, y = (np.array([getattr(e, k) for e in teach]) for k in ("a", "b", "answer"))
    p1 = A.Solver(seed=7, device=device).fit(a, b, y).predict(a, b)
    p2 = A.Solver(seed=7, device=device).fit(a, b, y).predict(a, b)
    assert p1.shape == (200,) and np.issubdtype(p1.dtype, np.integer) and np.array_equal(p1, p2)


def test_competence_table_reports_chance_and_accuracy():
    rng = np.random.default_rng(6)
    asg = {"add": "solid", "sub": "thin", "mul": "unseen", "div": "unseen", "mod": "unseen", "max": "unseen"}
    table = A.competence_table(asg, A.COUNTS, rng, "dev")
    assert set(table) == {"add", "sub"}                                                 # only taught skills are measured
    assert table["add"]["examples"] == 2000 and table["sub"]["examples"] == 20
    for row in table.values():
        assert 0.0 <= row["chance"] < 0.5 and 0.0 <= row["accuracy"] <= 1.0
    assert table["add"]["accuracy"] >= 0.90, table["add"]                               # if this fails the operand range must shrink, not the bar
    assert table["add"]["status"] == "solid" and table["sub"]["status"] == "thin"
    assert A.COUNTS == {"solid": 2000, "thin": 20} and A.COUNT_SWEEP == (5, 20, 100, 500, 2000)
