"""Tests for V0 walking skeleton builder, demo, comparators, report and AUROC."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from costlab.count import MacCount
from costlab.env import fixed_threads
from m1.arms import build_arm
from m1.beds import build_bed
from m1.cell import execute_cell
from m1.queue import load_tuning_map
from v0.auroc import rank_auroc
from v0.demo import DEMO_LABEL, generate_summary_text, run_demo
from v0.report import run_report
from v0.system import build_v0

# Constants / bars / tolerances following commit-gate conventions
SPEC_BAR_AUROC_CHANCE = 0.5
SPEC_BAR_AUROC_PERFECT = 1.0
SPEC_BAR_AUROC_INVERTED = 0.0
TOL_EXACT_ACCURACY = 1e-7


@pytest.fixture(scope="module", autouse=True)
def pin_cpus_for_v0():
    """Pin test execution to logical processors 8 and 10 with 2 threads."""
    with fixed_threads(n=2, cpus=[8, 10]):
        yield


def test_build_v0_config_and_bed_rejection():
    """(a) build_v0 config equals H_ULDA config except arm id; refuses B1 and B2."""
    bed_b3 = build_bed("B3seq", 22, 800)
    tuning_map = load_tuning_map("results/m1/tuning", "B3seq")
    arm_tuning = tuning_map.get("A1q", {})

    v0_arm = build_v0(bed_b3, tuning_dir="results/m1/tuning")
    h_arm = build_arm("H_ULDA", bed_b3, tuning=arm_tuning)

    cfg_v0 = dict(v0_arm.config())
    cfg_h = dict(h_arm.config())

    assert cfg_v0.pop("arm") == "V0"
    assert cfg_h.pop("arm") == "H_ULDA"
    assert cfg_v0 == cfg_h

    # Refuses B1 and B2
    bed_b1 = build_bed("B1", 10, 800)
    with pytest.raises(ValueError, match="supported on B3 beds only"):
        build_v0(bed_b1)

    bed_b2 = build_bed("B2", 6, 800)
    with pytest.raises(ValueError, match="supported on B3 beds only"):
        build_v0(bed_b2)


def test_build_v0_missing_a1q_raises_error():
    """(item 1.5) build_v0 raises ValueError when A1q tuning entry is missing (no fallback)."""
    bed = build_bed("B3seq", 22, 800)

    # Empty tuning dict
    with pytest.raises(ValueError, match="has no 'A1q' entry"):
        build_v0(bed, tuning_dir={})

    # Tuning dict missing A1q
    with pytest.raises(ValueError, match="has no 'A1q' entry"):
        build_v0(bed, tuning_dir={"A2": {"inner_steps": 5, "q": 0.05}})


def test_demo_short_run_and_files(tmp_path: Path):
    """(b) A demo with --max-phases 2 writes all four files and carries demonstration label."""
    out_dir = tmp_path / "smoke_demo"
    record = run_demo(
        schedule="seq",
        seed=800,
        out_dir=out_dir,
        max_phases=2,
        comparators=True,
    )

    rec_file = out_dir / "record.json"
    tbl_file = out_dir / "table.md"
    fig_file = out_dir / "figure.png"
    sum_file = out_dir / "summary.txt"

    assert rec_file.exists(), "record.json must exist"
    assert tbl_file.exists(), "table.md must exist"
    assert fig_file.exists(), "figure.png must exist"
    assert sum_file.exists(), "summary.txt must exist"

    assert record["label"] == DEMO_LABEL
    assert record["status"] == "complete"
    assert "V0" in record["arms"]
    for arm_id in ["A3", "A7", "A5a", "A5b", "A6", "A7all"]:
        assert arm_id in record["arms"], f"Comparator {arm_id} must be in record.json"


def test_assembly_v0_equals_h_ulda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """(c) Assembly check: on B3seq seed 800, V0 accuracy equals execute_cell(H_ULDA) and stored dev cell."""
    # Monkeypatch count_macs in m1.cell so execute_cell does not run slow second MACs pass
    monkeypatch.setattr("m1.cell.count_macs", lambda fn, **kw: MacCount(macs=0, flops=0, by_op={}))

    bed = build_bed("B3seq", 22, 800)
    tuning_map = load_tuning_map("results/m1/tuning", "B3seq")
    arm_tuning = tuning_map.get("A1q", {})

    cell_rec = execute_cell(bed, "H_ULDA", profile="det", tuning=arm_tuning)
    cell_acc = cell_rec["accuracy"]
    cell_per_regime = cell_rec["per_regime_accuracy"]

    demo_rec = run_demo(
        schedule="seq",
        seed=800,
        out_dir=tmp_path / "assembly_v0",
        max_phases=None,
        comparators=False,
    )
    v0_data = demo_rec["arms"]["V0"]
    v0_acc = v0_data["final_accuracy"]
    v0_per_regime = v0_data["per_regime_accuracy"]

    assert math.isclose(v0_acc, cell_acc, abs_tol=TOL_EXACT_ACCURACY), (
        f"V0 final accuracy {v0_acc} does not match execute_cell {cell_acc}"
    )
    assert len(v0_per_regime) == len(cell_per_regime)
    for r_idx, (acc_v0, acc_cell) in enumerate(zip(v0_per_regime, cell_per_regime)):
        assert math.isclose(acc_v0, acc_cell, abs_tol=TOL_EXACT_ACCURACY), (
            f"Regime {r_idx} accuracy mismatch: V0={acc_v0}, cell={acc_cell}"
        )

    # Verify against stored M1 dev cell record
    stored_path = Path("results/m1/dev_b/B3seq_n22_H_ULDA__seed800.json")
    if stored_path.exists():
        stored_rec = json.loads(stored_path.read_text(encoding="utf-8"))
        stored_acc = stored_rec["accuracy"]
        assert math.isclose(v0_acc, stored_acc, abs_tol=TOL_EXACT_ACCURACY), (
            f"V0 final accuracy {v0_acc} does not match stored dev cell accuracy {stored_acc}"
        )
    else:
        pytest.skip(f"Stored record {stored_path} missing")


def test_comparators_batch_order_and_replay_memory(tmp_path: Path):
    """(d) Comparators see identical batches; A3 and A7 check final memory and capped_by_stream_size."""
    out_dir = tmp_path / "comp_demo"
    record = run_demo(
        schedule="seq",
        seed=800,
        out_dir=out_dir,
        max_phases=1,
        comparators=True,
    )

    hashes = record["batch_hashes"]
    v0_hash = hashes["V0"]
    for arm_id in ["A3", "A7", "A5a", "A5b", "A6", "A7all"]:
        assert hashes[arm_id] == v0_hash, f"{arm_id} did not see identical batches to V0"

    v0_mem = record["arms"]["V0"]["final_memory_bytes"]
    a3_data = record["arms"]["A3"]
    a7_data = record["arms"]["A7"]

    # Verify memory_budget_bytes and capped_by_stream_size
    assert a3_data["memory_budget_bytes"] == v0_mem
    assert a7_data["memory_budget_bytes"] == v0_mem
    assert "capped_by_stream_size" in a3_data
    assert "capped_by_stream_size" in a7_data
    assert isinstance(a3_data["capped_by_stream_size"], bool)
    assert isinstance(a7_data["capped_by_stream_size"], bool)


def test_auroc_handmade_scores():
    """(e) The rank AUROC function on hand-made scores with tie handling."""
    # Perfect separation -> 1.0
    neg_perf = [0.1, 0.2, 0.3]
    pos_perf = [0.7, 0.8, 0.9]
    assert math.isclose(rank_auroc(neg_perf, pos_perf), SPEC_BAR_AUROC_PERFECT)

    # Identical scores -> 0.5
    neg_ident = [0.5, 0.5, 0.5]
    pos_ident = [0.5, 0.5, 0.5]
    assert math.isclose(rank_auroc(neg_ident, pos_ident), SPEC_BAR_AUROC_CHANCE)

    # Inverted separation -> 0.0
    assert math.isclose(rank_auroc(pos_perf, neg_perf), SPEC_BAR_AUROC_INVERTED)

    # Ties across classes: neg=[0.1, 0.5], pos=[0.5, 0.9] -> 0.875
    neg_ties = [0.1, 0.5]
    pos_ties = [0.5, 0.9]
    assert math.isclose(rank_auroc(neg_ties, pos_ties), 0.875)

    # Empty inputs raise ValueError
    with pytest.raises(ValueError):
        rank_auroc([], [1.0])
    with pytest.raises(ValueError):
        rank_auroc([1.0], [])


def test_report_on_synthetic_fixtures(tmp_path: Path):
    """(f) The report generator runs on SYNTHETIC fixtures and outputs the label in first line."""
    synth_dir = tmp_path / "synth_fixture"
    synth_dir.mkdir(parents=True, exist_ok=True)

    dummy_record = {
        "synthetic": True,
        "label": DEMO_LABEL,
        "status": "complete",
        "schedule": "SYNTHETIC",
        "seed": "SYNTHETIC",
        "arms": {
            "V0": {
                "final_accuracy": 0.65,
                "final_memory_bytes": 14 * 1024 * 1024,
                "final_experts": 4,
                "answer_latency": {"1": {"latency_median_s": 0.002}},
                "auroc": {"mean": 0.82},
            },
            "A3": {
                "final_accuracy": 0.64,
                "final_memory_bytes": 7 * 1024 * 1024,
                "answer_latency": {"1": {"latency_median_s": 0.001}},
            },
            "A7": {
                "final_accuracy": 0.58,
                "final_memory_bytes": 6 * 1024 * 1024,
            },
            "A5a": {
                "final_accuracy": 0.61,
                "final_memory_bytes": 2 * 1024 * 1024,
            },
            "A5b": {
                "final_accuracy": 0.67,
                "final_memory_bytes": 1 * 1024 * 1024,
            },
            "A6": {
                "final_accuracy": 0.68,
                "final_memory_bytes": 3 * 1024 * 1024,
            },
            "A7all": {
                "final_accuracy": 0.69,
                "final_memory_bytes": 6 * 1024 * 1024,
            },
        },
    }
    (synth_dir / "record.json").write_text(json.dumps(dummy_record), encoding="utf-8")

    out_dir = tmp_path / "synth_report"
    res = run_report([synth_dir], out_dir)

    tbl_file = out_dir / "table.md"
    fig_file = out_dir / "figure.png"

    assert tbl_file.exists()
    assert fig_file.exists()

    first_line = tbl_file.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("# SYNTHETIC"), f"First line must start with # SYNTHETIC, got: {first_line!r}"
    assert res["label"] == "SYNTHETIC"


def test_rerun_finished_demo_skips(tmp_path: Path):
    """(g) Re-running a finished demo skips it without recomputing."""
    out_dir = tmp_path / "idempotent_demo"
    run_demo("seq", 800, out_dir, max_phases=1, comparators=False)

    rec_file = out_dir / "record.json"
    mtime1 = rec_file.stat().st_mtime_ns

    # Re-run
    run_demo("seq", 800, out_dir, max_phases=1, comparators=False)
    mtime2 = rec_file.stat().st_mtime_ns

    assert mtime1 == mtime2, "record.json was overwritten instead of skipped"


def test_summary_ahead_comparators_and_no_all_three():
    """(item 1.7) Summary names all comparators ahead with diff and memory; no 'all three' anywhere."""
    phases = [
        {
            "phase": 0,
            "V0": {
                "unknown_flag": {"false_new_rate": 0.05, "ood_detection_rate": 0.60},
            },
        }
    ]

    # Scenario 1: Some comparators ahead of V0
    final_data_ahead = {
        "V0": {
            "final_accuracy": 0.65,
            "final_memory_bytes": 14000000,
            "final_experts": 5,
            "forgetting": 0.02,
            "update_seconds_total": 20.0,
            "answer_latency": {},
            "auroc": {"mean": 0.75, "by_dataset": {}},
        },
        "A5b": {
            "final_accuracy": 0.68,
            "final_memory_bytes": 2000000,
            "forgetting": 0.01,
            "update_seconds_total": 0.5,
        },
        "A6": {
            "final_accuracy": 0.70,
            "final_memory_bytes": 5000000,
            "forgetting": 0.01,
            "update_seconds_total": 15.0,
        },
        "A3": {
            "final_accuracy": 0.60,
            "final_memory_bytes": 7000000,
            "forgetting": 0.05,
            "update_seconds_total": 12.0,
        },
    }

    summary1 = generate_summary_text(final_data_ahead, phases, schedule="seq", seed=800)
    assert "A5b" in summary1
    assert "A6" in summary1
    assert "+3.00%" in summary1 or "diff +3.00%" in summary1
    assert "+5.00%" in summary1 or "diff +5.00%" in summary1
    assert "vs V0" in summary1
    assert "all three" not in summary1.lower()

    # Scenario 2: No comparator ahead of V0
    final_data_none_ahead = {
        "V0": {
            "final_accuracy": 0.75,
            "final_memory_bytes": 14000000,
            "final_experts": 5,
            "forgetting": 0.02,
            "update_seconds_total": 20.0,
            "answer_latency": {},
            "auroc": {"mean": 0.75, "by_dataset": {}},
        },
        "A3": {
            "final_accuracy": 0.60,
            "final_memory_bytes": 7000000,
            "forgetting": 0.05,
        },
    }
    summary2 = generate_summary_text(final_data_none_ahead, phases, schedule="seq", seed=800)
    assert "no comparator printed here is ahead on this stream" in summary2
    assert "all three" not in summary2.lower()


def test_summary_capped_arms_and_raw_stream_comparison():
    """(item 1.7) Summary prints capped sentence and raw stream size comparison."""
    phases = [
        {
            "phase": 0,
            "V0": {
                "unknown_flag": {"false_new_rate": 0.02, "ood_detection_rate": 0.40},
            },
        }
    ]
    final_data = {
        "V0": {
            "final_accuracy": 0.65,
            "final_memory_bytes": 14000000,
            "final_experts": 5,
            "update_seconds_total": 20.0,
            "answer_latency": {},
            "auroc": {"mean": 0.75, "by_dataset": {}},
        },
        "A3": {
            "final_accuracy": 0.60,
            "final_memory_bytes": 6770000,
            "capped_by_stream_size": True,
        },
        "A7": {
            "final_accuracy": 0.58,
            "final_memory_bytes": 6200000,
            "capped_by_stream_size": True,
        },
    }

    raw_stream = 5600000
    summary = generate_summary_text(final_data, phases, schedule="seq", seed=800, raw_stream_bytes=raw_stream)

    assert "could not use the budget: it already holds the whole training stream, 6.46 MB" in summary or "holds the whole training stream" in summary
    assert "Raw training stream size" in summary
    assert "raw stream size" in summary


def test_summary_silent_flag_disclosure():
    """(item 1.7) Silent flag disclosure appears when both rates are 0 and not otherwise."""
    silent_phases = [
        {
            "phase": 0,
            "V0": {
                "unknown_flag": {"false_new_rate": 0.0, "ood_detection_rate": 0.0},
            },
        }
    ]
    active_phases = [
        {
            "phase": 0,
            "V0": {
                "unknown_flag": {"false_new_rate": 0.05, "ood_detection_rate": 0.0},
            },
        }
    ]
    data = {
        "V0": {
            "final_accuracy": 0.65,
            "final_memory_bytes": 14000000,
            "final_experts": 5,
            "update_seconds_total": 20.0,
            "answer_latency": {},
            "auroc": {"mean": 0.75, "by_dataset": {}},
        }
    }

    silent_stmt = "The unknown flag never fired at its configured level (on known or on out-of-domain examples); the AUROC measures the novelty score, not the flag."

    summary_silent = generate_summary_text(data, silent_phases, schedule="seq", seed=800)
    assert silent_stmt in summary_silent

    summary_active = generate_summary_text(data, active_phases, schedule="seq", seed=800)
    assert silent_stmt not in summary_active
