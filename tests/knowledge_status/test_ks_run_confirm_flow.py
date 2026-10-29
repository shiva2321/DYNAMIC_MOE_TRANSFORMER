"""The flow of the single-use runner end to end with fakes: which pool every real runner is given, when models are freed, how main() starts, stops, resumes and closes."""

import json

import pytest

import freeze_prereg as FZ
import run_confirm as RC
import runlib
from test_ks_freeze import freeze_into
from test_ks_run_confirm import STAMP_OF, fake_runners


def test_the_real_runners_hand_the_confirmatory_pool_to_every_experiment_and_free_the_models_when_the_reader_changes(monkeypatch):
    import reps
    import run_e1
    import run_e2
    import run_realism
    import scratch_reader
    calls, freed = [], []
    monkeypatch.setattr(run_e1, "run_cell", lambda rep, seed, pool, out, stamp: calls.append(("e1", rep, seed, pool, str(out), stamp)))
    monkeypatch.setattr(run_e2, "run_cell", lambda rep, seed, pool, out, stamp: calls.append(("e2", rep, seed, pool, str(out), stamp)))
    monkeypatch.setattr(run_realism, "run_seed", lambda seed, out, stamp, pool="dev": calls.append(("b7", None, seed, pool, str(out), stamp)))
    monkeypatch.setattr(scratch_reader, "run_e1_cell", lambda arm, seed, pool, out, stamp: calls.append(("fs-e1", arm, seed, pool, str(out), stamp)))
    monkeypatch.setattr(scratch_reader, "run_e2_cell", lambda arm, seed, pool, out, stamp: calls.append(("fs-e2", arm, seed, pool, str(out), stamp)))
    monkeypatch.setattr(RC, "run_e4_cell", lambda seed, out, stamp, n: calls.append(("e4", None, seed, "wrapper", str(out), stamp, n)))
    monkeypatch.setattr(reps, "clear_models", lambda: freed.append(len(calls)))
    monkeypatch.setattr(RC, "realism_rows", lambda rec, path=None: ["frozen rows"])
    monkeypatch.setattr(run_realism, "run_seed", lambda seed, out, stamp, pool="dev", rows=None: calls.append(("b7", None, seed, pool, str(out), stamp, rows)))
    run = RC.real_runners({"e4_n_per_kind": 7})
    run["e1"]("R5_R4_qwen_L26", 400, "o1", "s1")
    run["e1"]("R5_R4_qwen_L26", 401, "o1", "s1")
    assert freed == []                                                                   # same reader: nothing is freed
    run["e1"]("R0_hash", 400, "o1", "s1")
    assert freed == [2]                                                                  # the reader changed: freed once, before the new cell
    run["e2"]("R0_hash", 400, "o2", "s2")
    run["e2"]("R4_qwen_L26", 400, "o2", "s2")
    run["e1"]("R5_FS_strict", 400, "o1", "s1")
    run["e2"]("R5_FS_pre", 400, "o2", "s2")
    run["b7"](None, 400, "ob", "sb")
    run["e4"](None, 402, "o4", "s4")
    assert [c[0] for c in calls] == ["e1", "e1", "e1", "e2", "e2", "fs-e1", "fs-e2", "b7", "e4"]
    assert [c[3] for c in calls if c[0] != "e4"] == ["confirm"] * 8                       # never the default pool
    assert calls[-1] == ("e4", None, 402, "wrapper", "o4", "s4", 7)
    assert calls[-2][6] == ["frozen rows"]                                                  # B7 gets the frozen problems, not a rebuild from the cache
    assert RC.real_runners({})["e4"] is not None
    assert len(freed) == 6                                                               # every change of reader frees the GPU


def test_the_e4_seeds_are_those_where_addition_and_subtraction_are_really_solid_on_the_confirmatory_pool(tmp_path):
    rec, *_ = freeze_into(tmp_path)
    asked = []

    def competence(pool, skill, count, seed):
        asked.append(pool)
        return {"accuracy": 0.97 if (skill == "add" or seed % 2 == 0) else 0.7, "chance": 0.02}

    got = RC.e4_seeds_for(rec, competence)
    assert got == [s for s in rec["seeds"]["confirm"] if s % 2 == 0] and set(asked) == {"confirm"}


def patch_main(monkeypatch, tmp_path, rec, e4_seeds):
    monkeypatch.setattr(RC, "preflight", lambda *a, **k: rec)
    monkeypatch.setattr(RC, "e4_seeds_for", lambda r, f=None: e4_seeds)
    calls = []
    monkeypatch.setattr(RC, "real_runners", lambda r: fake_runners(calls))
    monkeypatch.setattr(RC, "stamps", lambda r: STAMP_OF)
    monkeypatch.setattr(RC, "STARTED", tmp_path / "STARTED.json")
    monkeypatch.setattr(RC, "CLOSED", tmp_path / "CLOSED.json")
    monkeypatch.setattr(RC, "OUT", tmp_path / "confirm")
    monkeypatch.setattr(RC.runlib, "set_threads", lambda n=3: None)
    return calls


def test_main_starts_stops_after_a_few_cells_resumes_and_closes_only_when_every_cell_exists(monkeypatch, tmp_path, capsys):
    rec, *_ = freeze_into(tmp_path)
    rec = {**rec, "seeds": {"confirm": [400, 401], "dev": rec["seeds"]["dev"]}}
    calls = patch_main(monkeypatch, tmp_path, rec, [401])
    total = len(RC.jobs(rec, [401]))
    assert RC.main(["--max-cells", "3"]) == 0
    assert len(calls) == 3 and (tmp_path / "STARTED.json").exists() and not (tmp_path / "CLOSED.json").exists()
    started = json.loads((tmp_path / "STARTED.json").read_text())
    assert started["e4_seeds"] == [401] and started["bars_sha256"] == FZ.bars_hash(rec) and started["environment"]["python"]
    monkeypatch.setattr(RC, "e4_seeds_for", lambda r, f=None: pytest.fail("the E4 seeds must be read back from STARTED.json on a resume"))
    assert RC.main([]) == 0
    assert len(calls) == total and (tmp_path / "CLOSED.json").exists()
    assert "CONFIRM_CLOSED.json written" in capsys.readouterr().out
    assert RC.main([]) == 0 and len(calls) == total                                       # nothing is computed twice


def test_a_power_cut_between_the_start_record_and_the_e4_seeds_is_repaired_on_the_next_launch(monkeypatch, tmp_path):
    rec, *_ = freeze_into(tmp_path)
    rec = {**rec, "seeds": {"confirm": [400], "dev": rec["seeds"]["dev"]}}
    calls = patch_main(monkeypatch, tmp_path, rec, [400])
    RC.mark_started(rec, tmp_path / "STARTED.json")                                       # the cut: started, E4 seeds never recorded
    assert json.loads((tmp_path / "STARTED.json").read_text())["e4_seeds"] is None
    assert RC.main([]) == 0
    assert json.loads((tmp_path / "STARTED.json").read_text())["e4_seeds"] == [400] and (tmp_path / "CLOSED.json").exists() and calls


def test_main_prints_the_reason_and_computes_nothing_when_the_preflight_refuses(monkeypatch, tmp_path, capsys):
    calls = patch_main(monkeypatch, tmp_path, {}, [])
    monkeypatch.setattr(RC, "preflight", lambda *a, **k: (_ for _ in ()).throw(RC.ConfirmRefused("because")))
    assert RC.main([]) == 2 and "REFUSED: because" in capsys.readouterr().out
    assert calls == [] and not (tmp_path / "STARTED.json").exists()


def test_the_log_names_the_experiment_the_reader_and_the_seed(tmp_path):
    rec, *_ = freeze_into(tmp_path)
    rec = {**rec, "seeds": {"confirm": [400], "dev": rec["seeds"]["dev"]}}
    lines = []
    RC.run_all(rec, fake_runners([]), STAMP_OF, [400], tmp_path / "confirm", log=lines.append)
    assert any(l.startswith("e1 R0_hash seed 400: done in") for l in lines) and any(l.startswith("e4  seed 400: done in") for l in lines) and any(l.startswith("b7  seed 400") for l in lines)
    assert len(lines) == len(RC.jobs(rec, [400]))
