"""The realism set: only one-step problems, labels that the reference answer confirms, no repeats, and a guard that keeps it out of teaching and tuning."""

import pytest

import realism as RL
import skills


def rec(eq, ans, body="Tom has 5 apples and 3 pears.", q="How many?", i="x1"):
    return {"ID": i, "Body": body, "Question": q, "Equation": eq, "Answer": ans, "Type": "t"}


def test_only_single_operation_solutions_are_kept():
    rows = RL._rows_from([rec("( 5.0 + 3.0 )", 8.0), rec("( ( 5.0 + 3.0 ) - 2.0 )", 6.0), rec("( 5.0 + ( 3.0 * 2.0 ) )", 11.0), rec("5.0", 5.0), rec("( 2.5 + 3.0 )", 5.5)], "s")
    assert [r["operands"] for r in rows] == [[5, 3]] and rows[0]["skill"] == "add"


def test_each_operation_is_read_from_its_symbol_and_bad_cases_are_dropped():
    good = RL._rows_from([rec("( 9.0 - 4.0 )", 5.0, i="a"), rec("( 6.0 * 7.0 )", 42.0, i="b"), rec("( 12.0 / 4.0 )", 3.0, i="c"), rec("( 2.0 + 2.0 )", 4.0, i="d")], "s")
    assert [r["skill"] for r in good] == ["sub", "mul", "div", "add"]
    bad = RL._rows_from([rec("( 4.0 - 9.0 )", -5.0, i="a"), rec("( 7.0 / 2.0 )", 3.5, i="b"), rec("( 7.0 / 0.0 )", 0.0, i="c")], "s")
    assert bad == []                                                                    # a negative difference, an inexact quotient and a zero divisor are not kept


def test_label_matches_the_reference_answer():
    rows = RL._rows_from([rec("( 9.0 - 4.0 )", 5.0), rec("( 9.0 - 4.0 )", 6.0, body="Another one.", i="y")], "s")
    assert len(rows) == 1                                                               # the second states an answer the operation does not give
    for r in RL.build_realism_set(120, seed=3):
        a, b = r["operands"]
        assert skills.truth(r["skill"], a, b) is not None and r["skill"] in ("add", "sub", "mul", "div")


def test_no_row_appears_twice_and_the_draw_is_balanced_and_deterministic():
    rows = RL.build_realism_set(300, seed=100)
    assert len(rows) == 300 and len({r["text"] for r in rows}) == 300
    counts = {s: sum(r["skill"] == s for r in rows) for s in ("add", "sub", "mul", "div")}
    assert all(c >= 60 for c in counts.values()), counts
    assert RL.build_realism_set(300, seed=100) == rows and RL.build_realism_set(300, seed=101) != rows
    with pytest.raises(ValueError):
        RL.build_realism_set(5000)


def test_realism_rows_never_enter_teaching_or_tuning():
    rows = RL.build_realism_set(50, seed=1)
    RL.assert_not_realism(["something else entirely", "and another"], rows)
    with pytest.raises(AssertionError, match="realism"):
        RL.assert_not_realism(["fine", rows[7]["text"]], rows)


def test_spotcheck_is_readable_markdown_with_the_asked_number_of_rows():
    rows = RL.build_realism_set(100, seed=2)
    text = RL.spotcheck(rows, k=50, seed=1)
    assert text.count("\n| ") >= 50 and "| label |" in text and RL.spotcheck(rows, k=50, seed=1) == text
    assert RL.spotcheck(rows[:10], k=50, seed=1).count("\n| ") == 10 + 1             # fewer rows than asked: all of them (plus the header row)
