import json

import pytest

import tune_scratch as TS


def test_the_grid_has_every_combination_once():
    s = TS.settings()
    assert len(s) == 8 and len({json.dumps(x, sort_keys=True) for x in s}) == 8
    assert {x["hidden"] for x in s} == {256, 1024} and {x["epochs"] for x in s} == {300, 800}


def test_the_winner_is_the_highest_score_and_ties_go_to_the_smaller_network():
    assert TS.choose({"a": 0.8, "b": 0.9, "c": 0.7}, {"a": 256, "b": 1024, "c": 256}) == "b"
    assert TS.choose({"a": 0.9, "b": 0.9}, {"a": 1024, "b": 256}) == "b"
    assert TS.choose({"a": 0.9000000001, "b": 0.9}, {"a": 1024, "b": 256}) == "b"                 # equal to nine decimals
    assert TS.choose({"a": 0.9001, "b": 0.9}, {"a": 1024, "b": 256}) == "a"


def test_score_is_the_mean_over_seeds_of_the_mean_of_the_three_numbers():
    rows = [{"b1a": 0.9, "b1b": 0.6, "b1c": 0.3}, {"b1a": 0.5, "b1b": 0.5, "b1c": 0.5}]
    assert TS.score(rows) == pytest.approx((0.6 + 0.5) / 2)


def test_it_refuses_dev_and_confirmatory_seeds(tmp_path):
    for bad in ([100], [150, 400], [149]):
        with pytest.raises(ValueError, match="150"):
            TS.run(bad, tmp_path)


def test_the_report_marks_the_winner():
    s = {"R5_FS_strict": {"winner": {"hidden": 256}, "score": 0.9, "all": {json.dumps({"hidden": 256}): 0.9, json.dumps({"hidden": 1024}): 0.8}}}
    text = TS.report(s)
    assert "(winner)" in text and text.index("0.9000") < text.index("0.8000")


def test_only_the_winning_row_is_marked():
    s = {"A": {"winner": {"hidden": 256}, "score": 0.9, "all": {json.dumps({"hidden": 256}): 0.9, json.dumps({"hidden": 1024}): 0.8}}}
    rows = [r for r in TS.report(s).splitlines() if r.startswith("| {")]
    assert len(rows) == 2 and "(winner)" in rows[0] and "0.9000" in rows[0] and "(winner)" not in rows[1]
