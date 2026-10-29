"""Fixes after the second independent review: the pinned realism set, the information-ceiling ladder on fresh seeds, the B7 scope, the Conflict text, the one-sided from-scratch rule."""

import json
import re

import pytest

import freeze_prereg as FZ
import report as RP
import run_confirm as RC
from test_ks_freeze import STAMPS, freeze_into, write_arm_cells, write_cells
from test_ks_run_confirm import ENV, GO, frozen, pre


def test_the_realism_rows_come_from_the_frozen_file_and_a_changed_file_is_refused(tmp_path):
    f = tmp_path / "realism_set.jsonl"
    f.write_bytes(b'{"skill": "add", "operands": [1, 2], "text": "a"}\r\n{"skill": "sub", "operands": [5, 2], "text": "b"}\r\n')
    rec = {"realism_set_sha256": FZ.file_sha256(f)}
    assert [r["text"] for r in RC.realism_rows(rec, f)] == ["a", "b"]
    f.write_bytes(b'{"skill": "add", "operands": [1, 2], "text": "changed"}\n')
    with pytest.raises(RC.ConfirmRefused, match="not the realism set that was frozen"):
        RC.realism_rows(rec, f)
    with pytest.raises(RC.ConfirmRefused, match="not the realism set that was frozen"):
        RC.realism_rows(rec, tmp_path / "missing.jsonl")
    with pytest.raises(RC.ConfirmRefused, match="no realism set was frozen"):
        RC.realism_rows({"realism_set_sha256": None}, f)


def test_preflight_refuses_a_run_whose_realism_file_is_not_the_frozen_one(tmp_path):
    rec, out, pool, files = frozen(tmp_path)
    other = tmp_path / "other_realism.jsonl"
    other.write_text('{"skill": "add", "operands": [1, 2], "text": "x"}\n')
    with pytest.raises(RC.ConfirmRefused, match="realism"):
        RC.preflight(frozen_dir=out, pool_file=pool, code_files=files, environ=GO, closed=tmp_path / "C.json", started=tmp_path / "S.json", out=tmp_path / "confirm",
                     competence_dir=tmp_path / "competence", environment=ENV, realism_file=other)
    assert pre(tmp_path, out, pool, files) == rec                                              # the real file is the frozen one


def test_the_prereg_states_the_conflict_scope_the_expected_banner_and_the_realism_details(tmp_path):
    rec, out, pool, code = freeze_into(tmp_path)
    text = (out / "PREREG.md").read_text(encoding="utf-8")
    assert "Conflict condition of spec 5.6" in text and "was not built" in text and "Nothing about Conflict is reported" in text
    assert "if it fires" not in text
    assert "status claim (B3a to B7)" in text and "B3a to B6" not in text and "expected to carry the CONDITIONAL banner" in text
    counts = rec["realism_counts"]
    assert sum(counts.values()) == 300 and set(counts) == {"add", "sub", "mul", "div"}
    assert ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) in text and "75 of each" not in text
    assert "operands are both at most 99" in text and "about 1 in 10" in text and "does not move the verdict" not in text
    assert "the spec's B1a bar OR the spec's B1b bar" in text
    assert "R1_minilm384" in text.split("Not run, stated in advance")[1] or "E1-only readers" in text.split("Not run, stated in advance")[1]
    assert rec["e1_only_readers"] == ["R1_minilm384", "R2_mpnet", "R3_bert", "R4_gpt2_L8", "R4_qwen_L22"]


def test_the_expected_b2_verdict_uses_the_spec_bar_like_the_judgement_does(tmp_path):
    results = tmp_path / "results"
    write_cells(results)
    for p in (results / "e1").glob("R0_hash__s*.json"):                                       # R0 reaches 0.85 on B1b: 0.10 below the spec bar, 0.14 below nothing tighter
        d = json.loads(p.read_text())
        d["b1b"], d["b1a"] = 0.85, 0.95
        p.write_text(json.dumps(d))
    for p in (results / "e1").glob("R5_R4_qwen_L26__s*.json"):                                  # a steady 0.999 tightens the frozen B1b bar to about 0.998
        d = json.loads(p.read_text())
        d["b1b"] = 0.999
        p.write_text(json.dumps(d))
    pool = tmp_path / "pool_confirm.py"
    pool.write_text("POOL = 1\n")
    code = tmp_path / "x.py"
    code.write_text("x = 1\n")
    out = tmp_path / "out"
    out.mkdir()
    rec = FZ.freeze(out_dir=out, results_dir=results, pool_file=pool, code_files=[code], expected_stamps=STAMPS)
    assert rec["bars"]["b1b"]["bar"] > 0.97 and rec["bars"]["b1b"]["draft"] == 0.95
    assert rec["statements"]["B2"]["expected"] == "fail" and rec["bars"]["b2_gap"]["expected"] == "fail"       # 0.95 - 0.85 = 0.10 < 0.15 against the spec bar (a frozen-bar reading would say 0.15)


def test_a_from_scratch_arm_that_is_better_is_never_worse_and_the_found_rate_is_shown(tmp_path):
    results = tmp_path / "results"
    write_cells(results)
    write_arm_cells(results, "R5_FS_strict", -0.20)                                           # better on B1b, B1c, macro F1 and false-new by 0.2
    write_arm_cells(results, "R5_FS_pre", 0.0)
    out = FZ.fs_summary(results, {**STAMPS, "fs": "FS"})
    assert out["R5_FS_strict"]["status"] == "kept" and out["R5_FS_strict"]["worst_gap"] == 0.0 and all(g == 0.0 for g in out["R5_FS_strict"]["gaps"].values())
    assert out["R5_FS_strict"]["found"] == pytest.approx(0.89) and out["R5_FS_pre"]["headline_found"] == pytest.approx(0.89)
    low = tmp_path / "low_false_new"
    write_cells(low)
    write_arm_cells(low, "R5_FS_strict", 0.10)                                                # worse on B1b/B1c/macro F1 by 0.10, and false-new HIGHER by 0.10
    g = FZ.fs_summary(low, {**STAMPS, "fs": "FS"}, arms=("R5_FS_strict",))["R5_FS_strict"]["gaps"]
    assert g["b1b"] == pytest.approx(0.10, abs=1e-9) and g["false_unseen_heldout"] == pytest.approx(0.10, abs=1e-9)


def test_the_from_scratch_table_has_the_found_column_and_the_explanation(tmp_path):
    results = tmp_path / "results"
    write_cells(results)
    write_arm_cells(results, "R5_FS_strict", 0.03)
    out = tmp_path / "out"
    out.mkdir()
    pool = tmp_path / "pool_confirm.py"
    pool.write_text("POOL = 1\n")
    code = tmp_path / "x.py"
    code.write_text("x = 1\n")
    FZ.freeze(out_dir=out, results_dir=results, pool_file=pool, code_files=[code], expected_stamps={**STAMPS, "fs": "FS"})
    text = (out / "PREREG.md").read_text(encoding="utf-8")
    assert "never-taught skills found" in text and "only being worse than the headline reader" in text and "call almost nothing new" in text and "headline reader finds 0.890" in text


def test_the_banner_scope_includes_b7():
    lines = RP.banners({"invalid": False, "status_void": True, "conditional": True})
    assert "B3a to B7" in lines[0] and "B3a to B7" in lines[1] and "B3a to B6" not in "".join(lines)


def test_the_report_prints_the_reported_small_operand_row(tmp_path):
    rec, *_ = freeze_into(tmp_path)
    confirm = tmp_path / "confirm"
    write_cells(confirm, seeds=range(400, 412), pool="confirm")                                 # eleven of the twelve seeds can be scored, enough for the minimum of ten
    text, res = RP.build(rec, confirm, STAMPS, full=False)
    assert "B7 on the problems whose operands are both at most 99" in text and ": n/a (realism minus templates" in text
    for p in (confirm / "realism").glob("s*.json"):
        d = json.loads(p.read_text())
        d["diff_small"] = -0.2 if d["diff"] is not None else None
        p.write_text(json.dumps(d))
    text2, res2 = RP.build(rec, confirm, STAMPS, full=False)
    assert res2["means"]["b7_diff_small"] == pytest.approx(-0.2) and "-0.200 (realism minus templates" in text2
    assert "b7_diff_small" not in res2["verdict"]["measures"]                                  # reported, never judged


def test_the_information_ceiling_ladder_is_part_of_the_e1_table_of_the_report(tmp_path, monkeypatch):
    import run_e1
    rec, *_ = freeze_into(tmp_path)
    confirm = tmp_path / "confirm"
    write_cells(confirm, seeds=range(400, 410), pool="confirm")
    seen = {}
    monkeypatch.setattr(run_e1, "report_e1", lambda out, reps, seeds, stamp, pool: seen.update(reps=list(reps)) or "| x |")
    import run_e2
    import run_e3
    import run_realism
    monkeypatch.setattr(run_e2, "report_e2", lambda *a, **k: "")
    monkeypatch.setattr(run_e3, "report_e4", lambda *a, **k: "")
    monkeypatch.setattr(run_realism, "report_b7", lambda *a, **k: "")
    RP.build(rec, confirm, STAMPS, full=True)
    assert seen["reps"] == rec["representations"] + rec["e1_only_readers"] and "R3_bert" in seen["reps"]


def test_strip_marks_also_removes_the_stale_header_and_the_draft_bar_columns():
    text = "\n".join(["PASS / fail is the mean against the DRAFT bar of spec section 7 (see the last table for the dev-fixed bars). A seed is one teaching assignment.",
                      "B5: the parent set of a multiplication trace is exactly {add}, of a division trace exactly {sub}, in at least 0.90 of the traces; a decoy gets a false relation in at most 0.10.",
                      "| decoy wrong_steps | 10 | 40 | n/a | 0.000 | at most 0.10: PASS |", "| mul | 0.4 | at least 0.90: fail |", "Some prose says at most 0.10 on purpose."])
    out = RP.strip_marks(text)
    assert "PASS / fail is the mean" not in out and "B5: the parent set" not in out
    assert "| decoy wrong_steps | 10 | 40 | n/a | 0.000 | |" in out and "| mul | 0.4 | |" in out and "Some prose says at most 0.10 on purpose." in out


def test_the_headline_found_rate_is_taken_from_the_first_arm_that_has_dev_cells(tmp_path):
    results = tmp_path / "results"
    write_cells(results)
    write_arm_cells(results, "R5_FS_strict", 0.03)                                          # R5_FS_pre has no dev cells and sorts first
    out = tmp_path / "out"
    out.mkdir()
    pool = tmp_path / "pool_confirm.py"
    pool.write_text("POOL = 1\n")
    code = tmp_path / "x.py"
    code.write_text("x = 1\n")
    FZ.freeze(out_dir=out, results_dir=results, pool_file=pool, code_files=[code], expected_stamps={**STAMPS, "fs": "FS"})
    assert "the headline reader finds 0.890" in (out / "PREREG.md").read_text(encoding="utf-8")
    assert FZ._fmt(None) == "n/a" and FZ._fmt(0.5) == "0.500"
