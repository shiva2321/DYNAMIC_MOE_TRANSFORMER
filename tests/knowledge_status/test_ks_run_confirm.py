"""Task 15 plumbing, tested before the freeze: the single-use guards of the confirmatory runner, resumable cells, the closing marker, the pool swap for E4, and the report that applies the
frozen bars. No model is run; the cells are faked."""

import json
from pathlib import Path

import pytest

import freeze_prereg as FZ
import report as RP
import run_confirm as RC
import runlib
from test_ks_freeze import STAMPS, freeze_into, write_cells

GO = {"KS_ALLOW_CONFIRM": "1"}


def frozen(tmp_path):
    rec, out, pool, code = freeze_into(tmp_path)
    files = sorted(code.glob("*.py"))
    return rec, out, pool, files


def pre(tmp_path, out, pool, files, environ=GO, closed=None, started=None):
    return RC.preflight(frozen_dir=out, pool_file=pool, code_files=files, environ=environ, closed=closed or tmp_path / "CLOSED.json", started=started or tmp_path / "STARTED.json")


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


def test_preflight_refuses_changed_bars_pool_or_code(tmp_path):
    rec, out, pool, files = frozen(tmp_path)
    bars = json.loads((out / "bars.json").read_text())
    easier = json.loads(json.dumps(bars))
    easier["bars"]["false_unseen_heldout"]["bar"] = 0.5
    (out / "bars.json").write_text(json.dumps(easier))
    with pytest.raises(RC.ConfirmRefused, match="bars.json"):
        pre(tmp_path, out, pool, files)
    (out / "bars.json").write_text(json.dumps(bars))
    pool.write_text("POOL = 99\n")
    with pytest.raises(RC.ConfirmRefused, match="pool_confirm"):
        pre(tmp_path, out, pool, files)
    pool.write_text("POOL = 1\n")
    files[0].write_text("x = 2\n")
    with pytest.raises(RC.ConfirmRefused, match="code"):
        pre(tmp_path, out, pool, files)
    files[0].write_text("x = 1\n")
    assert pre(tmp_path, out, pool, files) == rec                                   # restored files are accepted again


def test_a_started_run_resumes_under_the_same_freeze_and_is_refused_under_another(tmp_path):
    rec, out, pool, files = frozen(tmp_path)
    started = tmp_path / "STARTED.json"
    st = RC.mark_started(rec, [400, 402], started)
    assert st["e4_seeds"] == [400, 402] and st["bars_sha256"] == FZ.bars_hash(rec)
    assert RC.mark_started(rec, [999], started)["e4_seeds"] == [400, 402]            # a resume reads the list back; it cannot change
    assert pre(tmp_path, out, pool, files, started=started) == rec
    started.write_text(json.dumps({"bars_sha256": "0" * 64}))
    with pytest.raises(RC.ConfirmRefused, match="different freeze"):
        pre(tmp_path, out, pool, files, started=started)


def test_jobs_cover_every_cell_exactly_once(tmp_path):
    rec, *_ = frozen(tmp_path)
    rec = {**rec, "seeds": {"confirm": list(range(400, 430)), "dev": rec["seeds"]["dev"]}}
    js = RC.jobs(rec, [400, 401, 405])
    assert len(js) == 3 * 30 * 2 + 3 and len(set(js)) == len(js)
    assert {j[0] for j in js} == {"e1", "e2", "e4"} and {j[1] for j in js if j[0] != "e4"} == set(rec["representations"])
    assert RC.cell_path(Path("o"), ("e1", "R0_hash", 400)) == Path("o/e1/R0_hash__s400.json") and RC.cell_path(Path("o"), ("e4", None, 7)) == Path("o/e4/s7.json")


def fake_runners(calls):
    def make(exp):
        def run(rep, seed, out_dir, stamp):
            calls.append((exp, rep, seed))
            name = f"{rep}__s{seed}.json" if rep else f"s{seed}.json"
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            runlib.atomic_write_text(Path(out_dir) / name, json.dumps({"seed": seed, "code_stamp": stamp}))
        return run
    return {e: make(e) for e in ("e1", "e2", "e4")}


def test_run_all_resumes_computes_each_cell_once_and_redoes_stale_cells(tmp_path):
    rec, *_ = frozen(tmp_path)
    rec = {**rec, "seeds": {"confirm": [400, 401], "dev": rec["seeds"]["dev"]}}
    out = tmp_path / "confirm"
    stamp_of = {"e1": lambda rep: "A", "e2": lambda rep: "B", "e4": "C"}
    calls = []
    p1 = RC.run_all(rec, fake_runners(calls), stamp_of, [401, 402], out, max_cells=4, log=lambda s: None)
    assert p1["ran"] == 4 and p1["done"] == 4 and len(calls) == 4                     # stopped after four new cells (a power cut looks the same)
    p2 = RC.run_all(rec, fake_runners(calls), stamp_of, [401, 402], out, log=lambda s: None)
    assert p2["done"] == p2["total"] == len(RC.jobs(rec, [401, 402])) and len(calls) == p2["total"] and len(set(calls)) == len(calls)
    p3 = RC.run_all(rec, fake_runners(calls), stamp_of, [401, 402], out, log=lambda s: None)
    assert p3["ran"] == 0 and len(calls) == p2["total"]
    p4 = RC.run_all(rec, fake_runners(calls), {**stamp_of, "e4": "NEW"}, [401, 402], out, log=lambda s: None)
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
    with pytest.raises(AssertionError, match="revisited"):
        RC.run_e4_cell(7, "o", "s", 3)


def test_main_refuses_without_a_go_and_writes_nothing(monkeypatch, capsys):
    monkeypatch.delenv("KS_ALLOW_CONFIRM", raising=False)
    before = {p.name for p in RC.HERE.glob("CONFIRM_*.json")}
    out_before = sorted(RC.OUT.glob("*/*.json")) if RC.OUT.exists() else []
    assert RC.main([]) == 2 and "REFUSED" in capsys.readouterr().out
    assert {p.name for p in RC.HERE.glob("CONFIRM_*.json")} == before
    assert (sorted(RC.OUT.glob("*/*.json")) if RC.OUT.exists() else []) == out_before


def test_report_applies_the_frozen_bars_marks_surprises_and_states_the_claim(tmp_path):
    rec, out, pool, files = frozen(tmp_path)
    confirm = tmp_path / "confirm"
    write_cells(confirm, seeds=range(400, 405), bump=0.2)                                  # macro F1 0.2 higher than on dev: above its bar and far outside the dev range
    text, res = RP.build(rec, confirm, STAMPS, full=False)
    v = res["verdict"]
    assert v["measures"]["macro_f1"]["verdict"] == "pass" and v["measures"]["false_unseen_heldout"]["verdict"] == "fail"
    assert v["statements"]["B3a"] == "fail" and v["statements"]["B7"] == "not_run" and v["statements"]["B3b"] == "reported"
    assert "| B3a | FAIL | FAIL | yes |" in text and "| B7 | NOT RUN | NOT RUN | yes |" in text
    assert res["claim"]["claim"] == "fails" and "never gated" in text
    assert res["bars_sha256"] == FZ.bars_hash(rec)
    assert [r for r in text.splitlines() if r.startswith("| macro_f1")][0].split("|")[7].strip() == "SURPRISE"


def test_report_does_not_flag_a_surprise_for_a_value_inside_the_dev_range():
    row = {"dev_min": 0.80, "dev_max": 0.90, "dev_sd": 0.02}
    assert RP.surprise("m", 0.85, row) is False and RP.surprise("m", 0.78, row) is False and RP.surprise("m", 0.779, row) is True
    assert RP.surprise("m", 0.921, row) is True and RP.surprise("m", 0.92, row) is False
    assert RP.surprise("m", None, row) is False and RP.surprise("m", 0.5, {"dev_min": None}) is False


def test_report_refuses_cells_written_by_other_code(tmp_path):
    rec, *_ = frozen(tmp_path)
    confirm = tmp_path / "confirm"
    write_cells(confirm, seeds=range(400, 403), e2_stamp="STALE")
    with pytest.raises(ValueError, match="stamp"):
        RP.build(rec, confirm, STAMPS, full=False)


def test_report_main_refuses_when_nothing_is_frozen(capsys):
    assert RP.main() == 2 and "REFUSED" in capsys.readouterr().out


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
    confirm = tmp_path / "confirm"
    write_cells(confirm, seeds=range(400, 405))
    text, _ = RP.build(rec, confirm, STAMPS, full=False)
    rows = {r.split("|")[1].strip(): r for r in text.splitlines() if r.startswith("| ") and r.count("|") == 8}
    assert "at least" in rows["macro_f1"] and "at most" not in rows["macro_f1"]
    assert "at most" in rows["false_unseen_heldout"] and "at least" not in rows["false_unseen_heldout"]
    assert "n/a" in rows["b7_gap"]
