"""The audit of the confirmatory run on a tiny faked run: it finds nothing in a clean run and names each kind of tampering."""

import json
import sys
from pathlib import Path

import pytest

import freeze_prereg as FZ
import report as RP
import run_confirm as RC
from test_ks_freeze import STAMPS, freeze_into, write_cells

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "knowledge_status" / "post_run"))
import audit as AU  # noqa: E402


def finished_run(tmp_path):
    """A frozen record, a closed fake run with 10 confirmatory seeds laid out as run_confirm writes it, and the verdict report.py makes."""
    rec, out, pool, code = freeze_into(tmp_path)
    rec = json.loads((out / "bars.json").read_text())
    root = out
    confirm = root / "results" / "confirm"
    write_cells(confirm, seeds=range(400, 410), pool="confirm")
    # the fake cells above use the experiment stamps of STAMPS; the audit asks for exactly those
    text, result = RP.build({**rec, "seeds": {"confirm": list(range(400, 410)), "dev": rec["seeds"]["dev"]}}, confirm, STAMPS, full=False)
    (confirm / "verdict.json").write_text(json.dumps(result, default=float))
    return rec, root, pool, sorted(code.glob("*.py")), confirm, result


def spec_with_ids(tmp_path, ids=("B1a", "B1b", "B1c", "B2", "B3a", "B3b", "B4a", "B4b", "B5", "B6", "B7")):
    p = tmp_path / "spec.md"
    p.write_text("## 7. Measurements and draft bars\n" + "\n".join(f"- {i} something" for i in ids) + "\n## 8. next\n", encoding="utf-8")
    return p


def run_audit(tmp_path, root, pool, files, rec, spec=None, seeds=None, jobs_e4=tuple(range(400, 410))):
    started = root / "CONFIRM_STARTED.json"
    started.write_text(json.dumps({"bars_sha256": FZ.bars_hash(rec), "e4_seeds": list(jobs_e4)}))
    (root / "CONFIRM_CLOSED.json").write_text("{}")
    return AU.audit(root, spec or spec_with_ids(tmp_path), pool, files, stamps=STAMPS)


def test_the_spec_statements_are_read_from_the_spec_text(tmp_path):
    assert AU.spec_statements(spec_with_ids(tmp_path)) == sorted(["B1a", "B1b", "B1c", "B2", "B3a", "B3b", "B4a", "B4b", "B5", "B6", "B7"])
    real = AU.spec_statements()
    assert real == sorted(["B1a", "B1b", "B1c", "B2", "B3a", "B3b", "B4a", "B4b", "B5", "B6", "B7"])


def test_a_clean_run_has_no_problems_and_each_tampering_is_named(tmp_path):
    rec, root, pool, files, confirm, result = finished_run(tmp_path)
    tiny = {**rec, "seeds": {"confirm": list(range(400, 410)), "dev": rec["seeds"]["dev"]}}
    (root / "bars.json").write_text(json.dumps(tiny))                                  # keep the hash file consistent with the tiny record
    (root / "bars.sha256").write_text(f"{FZ.bars_hash(tiny)}  bars.json\n")
    (root / "PREREG.md").write_text(FZ.render_prereg(tiny))
    problems = [p.replace(chr(92), '/') for p in run_audit(tmp_path, root, pool, files, tiny)]
    # the faked run lacks the experiment cells run_confirm would also have written (FS arms, extra readers): they are named, nothing else is
    assert all(p.startswith("missing cell") for p in problems), problems
    assert not [p for p in problems if "differ" in p or "frozen files changed" in p or "verdict.json says" in p]


def test_the_audit_names_a_changed_verdict_a_missing_statement_and_a_changed_freeze(tmp_path):
    rec, root, pool, files, confirm, result = finished_run(tmp_path)
    tiny = {**rec, "seeds": {"confirm": list(range(400, 410)), "dev": rec["seeds"]["dev"]}}
    (root / "bars.json").write_text(json.dumps(tiny))
    (root / "bars.sha256").write_text(f"{FZ.bars_hash(tiny)}  bars.json\n")
    (root / "PREREG.md").write_text(FZ.render_prereg(tiny))
    flipped = json.loads(json.dumps(result, default=float))
    flipped["verdict"]["measures"]["macro_f1"]["verdict"] = "fail" if result["verdict"]["measures"]["macro_f1"]["verdict"] == "pass" else "pass"
    (confirm / "verdict.json").write_text(json.dumps(flipped))
    assert any(p.startswith("macro_f1: verdict.json says") for p in run_audit(tmp_path, root, pool, files, tiny))
    (confirm / "verdict.json").write_text(json.dumps(result, default=float))
    assert any("differ from the statements of spec section 7" in p for p in run_audit(tmp_path, root, pool, files, tiny, spec=spec_with_ids(tmp_path, ids=("B1a", "B1b", "B8"))))
    pool.write_text("POOL = 5\n")
    assert any(p.startswith("the frozen files changed") for p in run_audit(tmp_path, root, pool, files, tiny))
    pool.write_text("POOL = 1\n")
    (confirm / "e1" / "stray__s999.json").write_text("{}")
    probs = [p.replace(chr(92), "/") for p in run_audit(tmp_path, root, pool, files, tiny)]
    assert any("unexpected file e1/stray__s999.json" in p for p in probs) and any("outside seeds 400 to 409" in p for p in probs)


def test_the_audit_names_a_missing_start_record_and_a_start_under_another_freeze(tmp_path):
    rec, root, pool, files, confirm, result = finished_run(tmp_path)
    tiny = {**rec, "seeds": {"confirm": list(range(400, 410)), "dev": rec["seeds"]["dev"]}}
    (root / "bars.json").write_text(json.dumps(tiny))
    (root / "bars.sha256").write_text(f"{FZ.bars_hash(tiny)}  bars.json\n")
    (root / "PREREG.md").write_text(FZ.render_prereg(tiny))
    probs = AU.audit(root, spec_with_ids(tmp_path), pool, files, stamps=STAMPS)
    assert "CONFIRM_STARTED.json is missing" in probs and "CONFIRM_CLOSED.json is missing" in probs
    (root / "CONFIRM_STARTED.json").write_text(json.dumps({"bars_sha256": "0" * 64, "e4_seeds": []}))
    assert "the run was started under another freeze" in AU.audit(root, spec_with_ids(tmp_path), pool, files, stamps=STAMPS)
    (confirm / "verdict.json").unlink()
    assert AU.audit(root, spec_with_ids(tmp_path), pool, files, stamps=STAMPS)[0].startswith("cannot read bars.json or results/confirm/verdict.json")


# ------------------------------------------------------------------------------------------------------------------------------------ the audit recomputes from the raw cells
def tiny_run(tmp_path):
    rec, root, pool, files, confirm, result = finished_run(tmp_path)
    tiny = {**rec, "seeds": {"confirm": list(range(400, 410)), "dev": rec["seeds"]["dev"]}}
    (root / "bars.json").write_text(json.dumps(tiny))
    (root / "bars.sha256").write_text(f"{FZ.bars_hash(tiny)}  bars.json\n")
    (root / "PREREG.md").write_text(FZ.render_prereg(tiny))
    return tiny, root, pool, files, confirm, result


def only_real(problems):
    return [p.replace(chr(92), "/") for p in problems if not p.startswith("missing cell")]


def test_the_audit_recomputes_every_verdict_from_the_raw_cells_and_finds_nothing_in_a_clean_run(tmp_path):
    tiny, root, pool, files, confirm, result = tiny_run(tmp_path)
    assert only_real(run_audit(tmp_path, root, pool, files, tiny)) == []
    raw = AU.raw_measures(confirm, tiny, [])
    assert raw["b1a"] == pytest.approx(result["means"]["b1a"]) and raw["b6_gap"] == pytest.approx(result["means"]["b6_gap"])
    assert raw["b5_decoy_worst"] == pytest.approx(result["means"]["b5_decoy_worst"]) and "b7_gap" not in raw          # nine realism seeds are fewer than the minimum


def test_the_audit_names_a_wrong_statement_verdict_a_wrong_value_and_wrong_consequences(tmp_path):
    tiny, root, pool, files, confirm, result = tiny_run(tmp_path)
    bad = json.loads(json.dumps(result, default=float))
    bad["verdict"]["statements"]["B3a"] = "pass"
    bad["verdict"]["measures"]["macro_f1"]["value"] = 0.123
    bad["consequences"] = {"invalid": True, "status_void": False, "conditional": False}
    (confirm / "verdict.json").write_text(json.dumps(bad))
    probs = only_real(run_audit(tmp_path, root, pool, files, tiny))
    assert any(p.startswith("B3a: verdict.json says pass, recomputing gives fail") for p in probs)
    assert any(p.startswith("macro_f1: verdict.json holds the value 0.123") for p in probs)
    assert any(p.startswith("consequences in verdict.json") for p in probs)


def test_the_audit_checks_the_measures_against_the_frozen_record_not_against_the_verdict_code(tmp_path):
    tiny, root, pool, files, confirm, result = tiny_run(tmp_path)
    dropped = json.loads(json.dumps(result, default=float))
    del dropped["verdict"]["measures"]["trap_as_unseen"]                                 # a measure silently dropped from the code that judges
    (confirm / "verdict.json").write_text(json.dumps(dropped))
    probs = only_real(run_audit(tmp_path, root, pool, files, tiny))
    assert any(p.startswith("B3a was judged on") and "trap_as_unseen" not in p.split("the frozen record lists")[0] and "trap_as_unseen" in p for p in probs)


def test_an_unreadable_cell_is_named_and_the_audit_goes_on(tmp_path):
    tiny, root, pool, files, confirm, result = tiny_run(tmp_path)
    (confirm / "e2" / "R5_R4_qwen_L26__s403.json").write_text("{not json")
    probs = only_real(run_audit(tmp_path, root, pool, files, tiny))
    assert any("cannot read cell R5_R4_qwen_L26__s403.json" in p for p in probs) and any("is not stamped with this freeze" in p for p in probs)
    (confirm / "e2" / "R5_R4_qwen_L26__s403.json").write_text(json.dumps({"code_stamp": "E2", "pool": "confirm"}))
    probs2 = only_real(run_audit(tmp_path, root, pool, files, tiny))
    assert any("cell R5_R4_qwen_L26__s403.json lacks" in p for p in probs2)


def test_the_judge_applies_the_bar_in_the_right_direction_with_float_slack():
    assert AU.judge(0.95, 0.95, "min") == "pass" and AU.judge(0.9499, 0.95, "min") == "fail" and AU.judge(0.05, 0.05, "max") == "pass" and AU.judge(0.0501, 0.05, "max") == "fail"
    assert AU.judge(None, 0.5, "min") == "not_run" and AU.judge(0.95 - 1e-13, 0.95, "min") == "pass"
