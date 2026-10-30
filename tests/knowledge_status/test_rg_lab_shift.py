"""The shift lab (routing_general/lab_shift.py) with stand-ins for the language model and the encoder: the cell, the verdict rules and the report."""

import sys
from pathlib import Path

import numpy as np
import pytest

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
for p in (_KS, _KS / "routing_general", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import lab_shift as LS  # noqa: E402
import test_rg_lab as TL  # noqa: E402


def reword(prompts, seed):
    return ["reworded: " + p.split("Request: ")[-1] for p in prompts]


def test_run_cell_scores_every_reader_on_the_rewrites_and_builds_every_system(tmp_path):
    w = TL.make_world()
    rec = LS.run_cell(w, TL.FOREIGN, reword, ask=TL.fake_ask_for(w), encode=TL.fake_encode, cache_dir=tmp_path / "ask", rw_dir=tmp_path / "rw", seed=1)
    assert set(rec["readers"]) == set(LS.READERS)
    for r, op in rec["readers"].items():
        assert 0.0 <= op["shift_flag"] <= 1.0 and 0.0 <= op["known_flagged"] <= 1.0, r
    assert rec["readers"]["ask_exemplars"]["shift_flag"] == 0.0                              # the stand-in reader still finds the class name in a rewrite
    assert set(rec["systems"]) == {"either_fixed", "both_fixed", "trusted_either", "trusted_either_calV", "trusted_all_either"}
    assert set(rec["trusted"]["pair"]) <= set(LS.PAIR) and "ask_exemplars" in rec["trusted"]["pair"]
    assert set(rec["ident"]) == {"ask_known", "ask_shift", "lr_known", "lr_shift"} and rec["ident"]["ask_shift"] == 1.0
    assert rec["n"]["shift"] == 10 and rec["n"]["cal"] == 10                                   # 5 cut messages and 5 rewritten ones per class, two classes


def test_a_diagnostic_with_nothing_to_measure_is_loud(tmp_path):
    w = TL.make_world()
    with pytest.raises(RuntimeError, match="no rewrite was kept"):
        LS.run_cell(w, TL.FOREIGN, lambda prompts, seed: [p.split("Request: ")[-1] for p in prompts], ask=TL.fake_ask_for(w), encode=TL.fake_encode, cache_dir=tmp_path / "a",
                    rw_dir=tmp_path / "r", seed=1)


def rec(readers, trusted_either):
    return {"readers": {r: {"shift_flag": s, "known_flagged": k, "unknown_found": u} for r, (s, k, u) in readers.items()},
            "systems": {"trusted_either": {"known_flagged": trusted_either[0], "unknown_found": trusted_either[1]}}}


GOOD = {"d1": [rec({"a": (0.05, 0.04, 0.80), "b": (0.30, 0.40, 0.90)}, (0.04, 0.80))], "d2": [rec({"a": (0.06, 0.05, 0.50), "b": (0.08, 0.06, 0.70)}, (0.05, 0.68))]}


def test_the_verdict_rules_pass_on_a_clean_case_and_fail_for_the_declared_reasons():
    v = LS.verdicts(GOOD)
    assert v["C6a"]["pass"] and v["C6a"]["rho"] == pytest.approx(1.0) and v["C6b_pass"]
    # a reader that really flags 40 % of the known test but looks trusted on the rewrites: the diagnostic fails
    liar = {"d1": [rec({"a": (0.05, 0.04, 0.80), "b": (0.05, 0.40, 0.90)}, (0.04, 0.80))], "d2": GOOD["d2"]}
    assert not LS.verdicts(liar)["C6a"]["pass"] and not LS.verdicts(liar)["C6a"]["bad_untrusted"]
    # a good reader that looks untrusted: too few good readers are trusted
    shy = {"d1": [rec({"a": (0.30, 0.04, 0.80), "b": (0.35, 0.05, 0.90)}, (0.04, 0.80))], "d2": [rec({"a": (0.30, 0.05, 0.50), "b": (0.35, 0.06, 0.70)}, (0.05, 0.68))]}
    assert LS.verdicts(shy)["C6a"]["good_trusted_share"] == 0.0 and not LS.verdicts(shy)["C6a"]["pass"]
    # the system finds too little in one domain
    weak = {"d1": GOOD["d1"], "d2": [rec({"a": (0.06, 0.05, 0.50), "b": (0.08, 0.06, 0.70)}, (0.05, 0.55))]}
    assert not LS.verdicts(weak)["C6b"]["d2"]["pass"] and LS.verdicts(weak)["C6b"]["d1"]["pass"] and not LS.verdicts(weak)["C6b_pass"]
    # the system flags too many known messages
    noisy = {"d1": [rec({"a": (0.05, 0.04, 0.80), "b": (0.30, 0.40, 0.90)}, (0.30, 0.95))], "d2": GOOD["d2"]}
    assert not LS.verdicts(noisy)["C6b"]["d1"]["pass"]


def test_the_report_prints_every_table_and_the_verdict_lines(tmp_path):
    w = TL.make_world()
    r = LS.run_cell(w, TL.FOREIGN, reword, ask=TL.fake_ask_for(w), encode=TL.fake_encode, cache_dir=tmp_path / "ask", rw_dir=tmp_path / "rw", seed=1)
    text = LS.report({"toy": [r, r]}, [0, 1])
    assert "## toy (2 seeds" in text and "| ask_exemplars |" in text and "| trusted_either |" in text and "| trusted_either_calV |" in text
    assert "C6a (the diagnostic)" in text and "C6b, toy" in text and "C6b overall" in text and "Trusted pair readers per seed" in text
