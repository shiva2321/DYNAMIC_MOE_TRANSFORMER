"""Gaps found by the planted-bug sweeps of run_confirm.py and report.py."""

import copy
import types

import pytest

import report as RP
import run_confirm as RC
from test_ks_freeze import STAMPS, freeze_into, write_arm_cells, write_cells


def frozen(tmp_path):
    return freeze_into(tmp_path)[0]


def test_the_environment_record_follows_the_device_switch(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "FakeGPU")
    monkeypatch.delenv("KS_DEVICE", raising=False)
    e = RC.environment_record()
    assert e["cuda"] is True and e["gpu"] == "FakeGPU"
    monkeypatch.setenv("KS_DEVICE", "cpu")
    e = RC.environment_record()
    assert e["cuda"] is False and e["gpu"] is None                                        # the same machine run on the CPU is another set-up
    monkeypatch.delenv("KS_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    e = RC.environment_record()
    assert e["cuda"] is False and e["gpu"] is None


def test_the_banners_are_followed_by_a_blank_line_whenever_there_is_one_and_not_otherwise(tmp_path):
    rec = frozen(tmp_path)
    confirm = tmp_path / "confirm"
    write_cells(confirm, seeds=range(400, 410), pool="confirm")
    text, res = RP.build(rec, confirm, STAMPS, full=False)
    assert res["consequences"] == {"invalid": False, "status_void": False, "conditional": True}      # only one of the three
    lines = text.splitlines()
    assert lines[2].startswith("**STATUS CLAIMS CONDITIONAL") and lines[3] == "" and lines[4].startswith("Cells read")
    clean = copy.deepcopy(rec)
    clean["bars"]["b1a"]["bar"] = clean["bars"]["b1a"]["draft"] = 0.5                        # a headline reader that passes B1a, B1b and B1c: no banner at all
    clean["bars"]["b1c"]["bar"] = clean["bars"]["b1c"]["draft"] = 0.5
    text2, res2 = RP.build(clean, confirm, STAMPS, full=False)
    assert not any(res2["consequences"].values())
    assert text2.splitlines()[1] == "" and text2.splitlines()[2].startswith("Cells read")


def test_a_measure_without_a_dev_value_prints_n_a_and_never_a_surprise(tmp_path):
    rec = frozen(tmp_path)
    confirm = tmp_path / "confirm"
    write_cells(confirm, seeds=range(400, 410), pool="confirm")
    plain = copy.deepcopy(rec)
    plain["bars"]["macro_f1"].update(dev_mean=None, dev_min=None, dev_max=None, dev_sd=None)
    text, _ = RP.build(plain, confirm, STAMPS, full=False)
    cells = [c.strip() for c in [r for r in text.splitlines() if r.startswith("| macro_f1")][0].split("|")]
    assert cells[6] == "n/a" and cells[7] == ""
    text2, _ = RP.build(rec, confirm, STAMPS, full=False)
    cells2 = [c.strip() for c in [r for r in text2.splitlines() if r.startswith("| macro_f1")][0].split("|")]
    assert cells2[6] not in ("n/a", "") and float(cells2[6]) == pytest.approx(rec["bars"]["macro_f1"]["dev_mean"], abs=1e-3)


def test_the_from_scratch_table_prints_numbers_for_every_reader(tmp_path):
    rec = frozen(tmp_path)
    rec = {**rec, "representations": rec["representations"] + ["R5_FS_strict"]}
    confirm = tmp_path / "confirm"
    write_cells(confirm, seeds=range(400, 410), pool="confirm")
    write_arm_cells(confirm, "R5_FS_strict", 0.03, stamp="FSX", seeds=range(400, 410), pool="confirm")
    st = {**STAMPS, "e1": lambda rep: "FSX" if rep == "R5_FS_strict" else "E1", "e2": lambda rep: "FSX" if rep == "R5_FS_strict" else "E2"}
    text, _ = RP.build(rec, confirm, st, full=False)
    rows = {r.split("|")[1].strip(): [c.strip() for c in r.split("|")[2:-1]] for r in text.splitlines() if r.startswith("| R5_")}
    assert set(rows) >= {"R5_R4_qwen_L26", "R5_FS_strict"}
    for name in ("R5_R4_qwen_L26", "R5_FS_strict"):
        assert len(rows[name]) == 5 and all(c != "n/a" for c in rows[name]), (name, rows[name])
    assert float(rows["R5_FS_strict"][1]) < float(rows["R5_R4_qwen_L26"][1])                   # the arm was written 0.03 below the headline reader on B1b
