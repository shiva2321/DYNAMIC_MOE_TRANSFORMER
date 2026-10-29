"""Task 15 plumbing, tested before the freeze: the single-use guards of the confirmatory runner, resumable cells, the closing marker, the cell stamps, the pool swap for E4, and the report that
applies the frozen bars. No model is run; the cells are faked."""

import json
from pathlib import Path

import pytest

import freeze_prereg as FZ
import report as RP
import run_confirm as RC
import runlib
from test_ks_freeze import STAMPS, freeze_into, write_arm_cells, write_cells

GO = {"KS_ALLOW_CONFIRM": "1"}
ENV = {"python": "3.12", "torch": "2.x", "cuda": True, "gpu": "fake"}


def frozen(tmp_path):
    rec, out, pool, code = freeze_into(tmp_path)
    files = sorted(code.glob("*.py"))
    return rec, out, pool, files


def pre(tmp_path, out, pool, files, environ=GO, closed=None, started=None, environment=ENV):
    return RC.preflight(frozen_dir=out, pool_file=pool, code_files=files, environ=environ, closed=closed or tmp_path / "CLOSED.json", started=started or tmp_path / "STARTED.json",
                        out=tmp_path / "confirm", competence_dir=tmp_path / "competence", environment=environment)


def test_preflight_passes_on_an_intact_freeze_and_returns_the_record(tmp_path):
    rec, out, pool, files = frozen(tmp_path)
    assert pre(tmp_path, out, pool, files) == rec


def test_preflight_refuses_without_the_go_flag(tmp_path):
    rec, out, pool, files = frozen(tmp_path)
    for env in ({}, {"KS_ALLOW_CONFIRM": "0"}, {"KS_ALLOW_CONFIRM": "true"}):
        with pytest.raises(RC.ConfirmRefused, match="KS_ALLOW_CONFIRM"):
            pre(tmp_path, out, pool, files, environ=env)


def test_preflight_refuses_a_closed_run_even_with_the_go_flag(tmp_path):
    rec, out, pool, files = frozen(tmp_path)
    closed = tmp_path / "CLOSED.json"
    closed.write_text("{}")
    with pytest.raises(RC.ConfirmRefused, match="single use"):
        pre(tmp_path, out, pool, files, closed=closed)


def test_preflight_refuses_changed_bars_prereg_pool_or_code(tmp_path):
    rec, out, pool, files = frozen(tmp_path)
    bars = json.loads((out / "bars.json").read_text())
    easier = json.loads(json.dumps(bars))
    easier["bars"]["false_unseen_heldout"]["bar"] = 0.5
    (out / "bars.json").write_text(json.dumps(easier))
    with pytest.raises(RC.ConfirmRefused, match="bars.json"):
        pre(tmp_path, out, pool, files)
    (out / "bars.json").write_text(json.dumps(bars))
    prereg = (out / "PREREG.md").read_text()
    (out / "PREREG.md").write_text(prereg.replace("EXPECTED TO FAIL", "expected to pass"))
    with pytest.raises(RC.ConfirmRefused, match="PREREG"):
        pre(tmp_path, out, pool, files)
    (out / "PREREG.md").write_text(prereg)
    pool.write_text("POOL = 99\n")
    with pytest.raises(RC.ConfirmRefused, match="pool_confirm"):
        pre(tmp_path, out, pool, files)
    pool.write_text("POOL = 1\n")
    files[0].write_text("x = 2\n")
    with pytest.raises(RC.ConfirmRefused, match="code"):
        pre(tmp_path, out, pool, files)
    files[0].write_text("x = 1\n")
    assert pre(tmp_path, out, pool, files) == rec                                   # restored files are accepted again


def test_a_started_run_resumes_under_the_same_freeze_and_set_up_and_is_refused_otherwise(tmp_path):
    rec, out, pool, files = frozen(tmp_path)
    started = tmp_path / "STARTED.json"
    st = RC.mark_started(rec, started, ENV)
    assert st["e4_seeds"] is None and st["bars_sha256"] == FZ.bars_hash(rec) and st["environment"] == ENV
    assert RC.set_e4_seeds([400, 402], started)["e4_seeds"] == [400, 402]
    assert RC.set_e4_seeds([999], started)["e4_seeds"] == [400, 402]                  # a resume reads the list back; it cannot change
    assert RC.mark_started(rec, started, {"other": 1})["environment"] == ENV            # an existing start record is never rewritten
    assert pre(tmp_path, out, pool, files, started=started) == rec
    with pytest.raises(RC.ConfirmRefused, match="another set-up"):
        pre(tmp_path, out, pool, files, started=started, environment={**ENV, "cuda": False})        # CPU after a power cut: not a resume
    started.write_text(json.dumps({"bars_sha256": "0" * 64}))
    with pytest.raises(RC.ConfirmRefused, match="different freeze"):
        pre(tmp_path, out, pool, files, started=started)


def test_a_first_start_refuses_when_confirmatory_results_already_exist(tmp_path):
    rec, out, pool, files = frozen(tmp_path)
    assert RC.earlier_results(tmp_path / "confirm", tmp_path / "competence") == []
    (tmp_path / "confirm" / "e2").mkdir(parents=True)
    (tmp_path / "confirm" / "e2" / "R0_hash__s400.json").write_text("{}")
    with pytest.raises(RC.ConfirmRefused, match="earlier confirmatory results"):
        pre(tmp_path, out, pool, files)
    (tmp_path / "confirm" / "e2" / "R0_hash__s400.json").unlink()
    assert pre(tmp_path, out, pool, files) == rec
    (tmp_path / "competence").mkdir()
    (tmp_path / "competence" / "confirm__add__n2000__s400.json").write_text("{}")
    with pytest.raises(RC.ConfirmRefused, match="earlier confirmatory results"):
        pre(tmp_path, out, pool, files)
    started = tmp_path / "STARTED.json"                                                  # but once started, those files are the run's own
    RC.mark_started(rec, started, ENV)
    assert pre(tmp_path, out, pool, files, started=started) == rec


def test_jobs_cover_every_cell_exactly_once(tmp_path):
    rec, *_ = frozen(tmp_path)
    rec = {**rec, "seeds": {"confirm": list(range(400, 430)), "dev": rec["seeds"]["dev"]}}
    js = RC.jobs(rec, [400, 401, 405])
    n_rep = len(rec["representations"])
    assert len(js) == n_rep * 30 * 2 + 30 + 3 and len(set(js)) == len(js)
    assert {j[0] for j in js} == {"e1", "e2", "e4", "b7"} and {j[1] for j in js if j[0] in ("e1", "e2")} == set(rec["representations"])
    assert "R5_FS_pre" in rec["representations"] and ("e1", "R5_FS_strict", 400) in js
    assert RC.cell_path(Path("o"), ("e1", "R0_hash", 400)) == Path("o/e1/R0_hash__s400.json") and RC.cell_path(Path("o"), ("e4", None, 7)) == Path("o/e4/s7.json")
    assert RC.cell_path(Path("o"), ("b7", None, 7)) == Path("o/realism/s7.json")
    assert not [j for j in RC.jobs(rec, None) if j[0] == "e4"]                           # before the E4 seeds are known


def test_cell_stamps_change_with_the_experiment_the_pool_and_the_freeze(tmp_path):
    rec, *_ = frozen(tmp_path)
    a = RC.cell_stamp("code", rec)
    assert a == RC.cell_stamp("code", rec) and a != RC.cell_stamp("other code", rec)
    assert a != RC.cell_stamp("code", {**rec, "pool_confirm_sha256": "0" * 64})
    assert a != RC.cell_stamp("code", {**rec, "q": 0.1})                                 # any change of the frozen record


def test_the_stamps_of_a_run_separate_the_from_scratch_arms_and_the_other_experiments(tmp_path, monkeypatch):
    rec, *_ = frozen(tmp_path)
    monkeypatch.setattr(FZ, "current_stamps", lambda: {"e1": lambda rep: f"e1-{rep}", "e2": lambda rep: f"e2-{rep}", "e4": "e4", "b7": "b7", "fs": "fs"})
    st = RC.stamps(rec)
    assert st["e1"]("R0_hash") == RC.cell_stamp("e1-R0_hash", rec) and st["e2"]("R0_hash") == RC.cell_stamp("e2-R0_hash", rec)
    assert st["e1"]("R5_FS_pre") == st["e2"]("R5_FS_pre") == RC.cell_stamp("fs", rec) != st["e1"]("R0_hash")
    assert st["e4"] == RC.cell_stamp("e4", rec) and st["b7"] == RC.cell_stamp("b7", rec) and st["e4"] != st["b7"]


def fake_runners(calls):
    def make(exp):
        def run(rep, seed, out_dir, stamp):
            calls.append((exp, rep, seed))
            name = f"{rep}__s{seed}.json" if rep else f"s{seed}.json"
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            runlib.atomic_write_text(Path(out_dir) / name, json.dumps({"seed": seed, "code_stamp": stamp}))
        return run
    return {e: make(e) for e in ("e1", "e2", "e4", "b7")}


STAMP_OF = {"e1": lambda rep: "A", "e2": lambda rep: "B", "e4": "C", "b7": "D"}


def test_run_all_resumes_computes_each_cell_once_and_redoes_stale_cells(tmp_path):
    rec, *_ = frozen(tmp_path)
    rec = {**rec, "seeds": {"confirm": [400, 401], "dev": rec["seeds"]["dev"]}}
    out = tmp_path / "confirm"
    calls = []
    p1 = RC.run_all(rec, fake_runners(calls), STAMP_OF, [401, 402], out, max_cells=4, log=lambda s: None)
    assert p1["ran"] == 4 and p1["done"] == 4 and len(calls) == 4                     # stopped after four new cells (a power cut looks the same)
    p2 = RC.run_all(rec, fake_runners(calls), STAMP_OF, [401, 402], out, log=lambda s: None)
    assert p2["done"] == p2["total"] == len(RC.jobs(rec, [401, 402])) and len(calls) == p2["total"] and len(set(calls)) == len(calls)
    assert (out / "realism" / "s400.json").exists() and (out / "e4" / "s402.json").exists()
    p3 = RC.run_all(rec, fake_runners(calls), STAMP_OF, [401, 402], out, log=lambda s: None)
    assert p3["ran"] == 0 and len(calls) == p2["total"]
    p4 = RC.run_all(rec, fake_runners(calls), {**STAMP_OF, "e4": "NEW"}, [401, 402], out, log=lambda s: None)
    assert p4["ran"] == 2 and p4["done"] == p4["total"]                               # only the two E4 cells were written by other code


def test_close_needs_every_cell_and_then_blocks_a_second_run(tmp_path):
    rec, out, pool, files = frozen(tmp_path)
    closed = tmp_path / "CLOSED.json"
    with pytest.raises(RC.ConfirmRefused, match="cannot close"):
        RC.close(rec, {"done": 3, "total": 4}, closed)
    assert not closed.exists()
    RC.close(rec, {"done": 4, "total": 4}, closed)
    assert json.loads(closed.read_text())["bars_sha256"] == FZ.bars_hash(rec)
    with pytest.raises(RC.ConfirmRefused, match="single use"):
        pre(tmp_path, out, pool, files, closed=closed)


def test_the_e4_wrapper_builds_the_world_from_the_confirmatory_pool_and_restores_the_module(monkeypatch):
    import run_e3 as E3
    seen = []

    def recorder(seed, pool, sizes, competence_fn, n_per_kind):
        seen.append(pool)
        return "world"

    def fake_run_seed(seed, out_dir, stamp, n_per_kind=40, **kw):
        assert E3.build_trace_world(seed, "dev", None, None, n_per_kind) == "world"      # exactly how the dev runner calls it
        return {"seed": seed}

    monkeypatch.setattr(E3, "build_trace_world", recorder)
    monkeypatch.setattr(E3, "run_seed", fake_run_seed)
    assert RC.run_e4_cell(7, "o", "s", 3) == {"seed": 7} and seen == ["confirm"]
    assert E3.build_trace_world is recorder                                              # the swap is undone

    def changed_run_seed(seed, out_dir, stamp, n_per_kind=40, **kw):
        E3.build_trace_world(seed, "somewhere else", None, None, n_per_kind)

    monkeypatch.setattr(E3, "run_seed", changed_run_seed)
    with pytest.raises(RuntimeError, match="revisited"):
        RC.run_e4_cell(7, "o", "s", 3)                                                   # a real exception, not an assert that python -O removes


def test_main_refuses_without_a_go_and_writes_nothing(monkeypatch, capsys):
    monkeypatch.delenv("KS_ALLOW_CONFIRM", raising=False)
    before = {p.name for p in RC.HERE.glob("CONFIRM_*.json")}
    out_before = sorted(RC.OUT.glob("*/*.json")) if RC.OUT.exists() else []
    assert RC.main([]) == 2 and "REFUSED" in capsys.readouterr().out
    assert {p.name for p in RC.HERE.glob("CONFIRM_*.json")} == before
    assert (sorted(RC.OUT.glob("*/*.json")) if RC.OUT.exists() else []) == out_before


def test_the_environment_record_names_the_libraries_and_the_device():
    e = RC.environment_record()
    assert {"python", "torch", "scikit-learn", "transformers", "numpy", "cuda", "gpu"} <= set(e) and e["python"] and e == RC.environment_record()


# ------------------------------------------------------------------------------------------------------------------------------------ the report
def confirm_cells(tmp_path, n=10, **kw):
    confirm = tmp_path / "confirm"
    write_cells(confirm, seeds=range(400, 400 + n), pool="confirm", **kw)
    return confirm


def test_report_applies_the_frozen_bars_marks_surprises_and_states_the_claim(tmp_path):
    rec, out, pool, files = frozen(tmp_path)
    confirm = confirm_cells(tmp_path, bump=0.2)                                            # macro F1 0.2 higher than on dev: above its bar and far outside the dev range
    text, res = RP.build(rec, confirm, STAMPS, full=False)
    v = res["verdict"]
    assert v["measures"]["macro_f1"]["verdict"] == "pass" and v["measures"]["false_unseen_heldout"]["verdict"] == "fail" and v["measures"]["trap_as_unseen"]["verdict"] == "fail"
    assert v["statements"]["B3a"] == "fail" and v["statements"]["B3b"] == "reported" and v["statements"]["B7"] == "not_run"        # 9 realism seeds are fewer than the minimum of 10
    assert "| B3a | FAIL | FAIL | yes |" in text and "| B7 | NOT RUN |" in text
    assert res["claim"]["claim"] == "fails" and "never gated" in text and res["bars_sha256"] == FZ.bars_hash(rec)
    assert [r for r in text.splitlines() if r.startswith("| macro_f1")][0].split("|")[7].strip() == "SURPRISE"
    assert "R0" in text and "gap +" in text                                               # the word-matcher control is printed with its gaps


def test_report_prints_the_consequences_of_the_kill_rules_at_the_top(tmp_path):
    rec, *_ = frozen(tmp_path)
    confirm = confirm_cells(tmp_path)                                                       # the fake reader has B1a 0.92 and B1c 0.89 (as on dev): conditional
    text, res = RP.build(rec, confirm, STAMPS, full=False)
    assert res["consequences"] == {"invalid": False, "status_void": False, "conditional": True}
    assert text.splitlines()[2].startswith("**STATUS CLAIMS CONDITIONAL") and "INVALID" not in text.split("| measure |")[0]
    lines = RP.banners({"invalid": True, "status_void": True, "conditional": True})
    assert len(lines) == 3 and lines[0].startswith("**INVALID") and lines[1].startswith("**STATUS CLAIMS VOID") and RP.banners({"invalid": False, "status_void": False, "conditional": False}) == []


def test_a_measure_with_fewer_seeds_than_the_registered_minimum_is_not_judged(tmp_path):
    rec, *_ = frozen(tmp_path)
    text, res = RP.build(rec, confirm_cells(tmp_path, n=9), STAMPS, full=False)
    assert res["verdict"]["measures"]["macro_f1"]["verdict"] == "not_run" and res["verdict"]["statements"]["B3a"] == "not_run"


def test_report_does_not_flag_a_surprise_for_a_value_inside_the_dev_range():
    row = {"dev_min": 0.80, "dev_max": 0.90, "dev_sd": 0.02}
    assert RP.surprise("m", 0.85, row) is False and RP.surprise("m", 0.78, row) is False and RP.surprise("m", 0.779, row) is True
    assert RP.surprise("m", 0.921, row) is True and RP.surprise("m", 0.92, row) is False
    assert RP.surprise("m", None, row) is False and RP.surprise("m", 0.5, {"dev_min": None}) is False


def test_report_refuses_cells_written_by_other_code_other_seeds_or_another_pool(tmp_path):
    rec, *_ = frozen(tmp_path)
    write_cells(tmp_path / "stale", seeds=range(400, 410), e2_stamp="STALE", pool="confirm")
    with pytest.raises(ValueError, match="stamp"):
        RP.build(rec, tmp_path / "stale", STAMPS, full=False)
    write_cells(tmp_path / "dev_seeds", seeds=range(100, 110), pool="confirm")
    with pytest.raises(ValueError, match="outside the expected seeds"):
        RP.build(rec, tmp_path / "dev_seeds", STAMPS, full=False)
    write_cells(tmp_path / "dev_pool", seeds=range(400, 410), pool="dev")
    with pytest.raises(ValueError, match="pool"):
        RP.build(rec, tmp_path / "dev_pool", STAMPS, full=False)


def test_report_without_any_trace_seed_leaves_b5_not_run_instead_of_failing(tmp_path):
    rec, *_ = frozen(tmp_path)
    confirm = confirm_cells(tmp_path)
    for p in (confirm / "e4").glob("*.json"):
        p.unlink()
    text, res = RP.build(rec, confirm, STAMPS, full=False)
    assert res["verdict"]["statements"]["B5"] == "not_run"


def test_report_reads_the_adapted_number_from_current_cells_only_and_skips_nan(tmp_path):
    def cell(seed, value, stamp="E2"):
        d = tmp_path / "e2"
        d.mkdir(exist_ok=True)
        (d / f"R5_R4_qwen_L26__s{seed}.json").write_text(json.dumps({"code_stamp": stamp, "machine": {"q0.05": {"adapted": {"false_unseen_heldout": value}}}}))

    cell(400, 0.10)
    cell(401, 0.20)
    cell(402, 0.90, stamp="OLD")                                                           # written by other code: not counted
    cell(403, float("nan"))                                                                # not a number: not counted
    assert RP.adapted_false_unseen(tmp_path, "R5_R4_qwen_L26", "E2", 0.05) == pytest.approx(0.15)
    assert RP.adapted_false_unseen(tmp_path, "R5_R4_qwen_L26", "NONE", 0.05) is None
    assert RP.adapted_false_unseen(tmp_path / "nowhere", "R5_R4_qwen_L26", "E2", 0.05) is None


def test_report_says_which_way_each_bar_points(tmp_path):
    rec, *_ = frozen(tmp_path)
    text, _ = RP.build(rec, confirm_cells(tmp_path), STAMPS, full=False)
    rows = {r.split("|")[1].strip(): r for r in text.splitlines() if r.startswith("| ") and r.count("|") == 8}
    assert "at least" in rows["macro_f1"] and "at most" not in rows["macro_f1"]
    assert "at most" in rows["false_unseen_heldout"] and "at least" not in rows["false_unseen_heldout"]


def test_report_shows_the_from_scratch_arms_next_to_the_headline_reader(tmp_path):
    rec, *_ = frozen(tmp_path)
    confirm = confirm_cells(tmp_path)
    write_arm_cells(confirm, "R5_FS_strict", 0.03, stamp="FSX", seeds=range(400, 410), pool="confirm")
    st = {**STAMPS, "e1": lambda rep: "FSX" if rep == "R5_FS_strict" else "E1", "e2": lambda rep: "FSX" if rep == "R5_FS_strict" else "E2"}
    text, _ = RP.build(rec, confirm, st, full=False)
    assert "From-scratch reader arms" in text and "| R5_FS_strict |" in text and "| R5_R4_qwen_L26 |" in text


def test_strip_marks_removes_the_draft_marks_the_b2_line_and_the_dev_fixed_section_and_nothing_else():
    text = "\n".join(["# E1", "| rep | B1b |", "| R5 | 0.98 [0.97 - 0.99] PASS |", "| R0 | 0.00 [0.0 - 0.0] fail |", "**B2 validity of the test bed:** R0 is below: test bed VALID",
                      "The claim may fail in prose and PASS here.", "| trace | bar |", "| mul | at least 0.90: fail |", "## Bars: draft and dev-fixed", "| x | y |"])
    out = RP.strip_marks(text)
    assert "PASS |" not in out and "fail |" not in out and "B2 validity" not in out and "dev-fixed" not in out
    assert "| R5 | 0.98 [0.97 - 0.99] |" in out and "| mul | at least 0.90 |" in out and "may fail in prose and PASS here." in out


def test_the_full_report_appends_the_experiment_tables_without_their_marks(tmp_path, monkeypatch):
    import run_e1
    import run_e2
    import run_e3
    import run_realism
    rec, *_ = frozen(tmp_path)
    confirm = confirm_cells(tmp_path)
    monkeypatch.setattr(run_e1, "report_e1", lambda *a, **k: "| R5 | 0.98 PASS |\n**B2 validity: VALID**")
    monkeypatch.setattr(run_e2, "report_e2", lambda *a, **k: "| R5 | 0.40 fail |\n## Bars: draft and dev-fixed\n| bar | 0.05 |")
    monkeypatch.setattr(run_e3, "report_e4", lambda *a, **k: "| mul | at least 0.90: fail |")
    monkeypatch.setattr(run_realism, "report_b7", lambda cells, label="": f"B7 table {label} {len(cells)}")
    text, _ = RP.build(rec, confirm, STAMPS, full=True)
    tail = text.split("## Experiment tables (diagnostic)")[1]
    assert "| R5 | 0.98 |" in tail and "| R5 | 0.40 |" in tail and "| mul | at least 0.90 |" in tail and "Bars: draft and dev-fixed" not in tail and "B2 validity" not in tail and "B7 table confirmatory 10" in tail


def test_report_main_refuses_when_nothing_is_frozen(capsys):
    assert RP.main() == 2 and "REFUSED" in capsys.readouterr().out
