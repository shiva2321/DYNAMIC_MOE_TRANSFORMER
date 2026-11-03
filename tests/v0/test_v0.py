"""Tests for V0 walking skeleton builder, demo, comparators, report and AUROC."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from costlab.env import fixed_threads
from m1.arms import build_arm
from m1.beds import build_bed
from m1.cell import execute_cell
from m1.queue import load_tuning_map
from v0.auroc import rank_auroc
from v0.demo import DEMO_LABEL, run_demo
from v0.report import run_report
from v0.system import build_v0

# Constants / bars / tolerances following commit-gate conventions
SPEC_BAR_AUROC_CHANCE = 0.5
SPEC_BAR_AUROC_PERFECT = 1.0
SPEC_BAR_AUROC_INVERTED = 0.0
TOL_EXACT_ACCURACY = 1e-7
TOL_REPLAY_MEM_DIFF = 0.05


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


def test_demo_short_run_and_files(tmp_path: Path):
    """(b) A demo with --max-phases 3 writes all four files and carries demonstration label."""
    out_dir = tmp_path / "smoke_demo"
    record = run_demo(
        schedule="seq",
        seed=800,
        out_dir=out_dir,
        max_phases=3,
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
    assert "A3" in record["arms"]
    assert "A7" in record["arms"]
    assert "A5a" in record["arms"]


def test_assembly_v0_equals_h_ulda(tmp_path: Path):
    """(c) Assembly check: on B3seq seed 800, V0 final accuracy equals execute_cell(H_ULDA) exactly."""
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


def test_comparators_batch_order_and_replay_memory(tmp_path: Path):
    """(d) Comparators see identical batches in identical order; replay memory matches within 5%."""
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
    assert hashes["A3"] == v0_hash, "A3 did not see identical batches to V0"
    assert hashes["A7"] == v0_hash, "A7 did not see identical batches to V0"
    assert hashes["A5a"] == v0_hash, "A5a did not see identical batches to V0"

    v0_mem = record["arms"]["V0"]["final_memory_bytes"]
    a3_mem = record["arms"]["A3"]["final_memory_bytes"]

    # Replay comparator's memory within 5% of V0's final memory
    rel_diff = abs(a3_mem - v0_mem) / float(v0_mem)
    assert rel_diff < TOL_REPLAY_MEM_DIFF, f"A3 memory {a3_mem} differs by {rel_diff:.2%} from V0 memory {v0_mem}"
    assert record["arms"]["A3"]["config"]["target_pool_bytes"] == v0_mem


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
                "final_accuracy": 0.72,
                "final_memory_bytes": 1024 * 1024 * 4,
                "final_experts": 4,
                "answer_latency": {"1": {"latency_median_s": 0.002}},
                "auroc": {"mean": 0.82},
            },
            "A3": {
                "final_accuracy": 0.64,
                "final_memory_bytes": 1024 * 1024 * 4,
                "answer_latency": {"1": {"latency_median_s": 0.001}},
            },
            "A7": {
                "final_accuracy": 0.58,
                "final_memory_bytes": 1024 * 1024 * 4,
            },
            "A5a": {
                "final_accuracy": 0.69,
                "final_memory_bytes": 1024 * 1024 * 2,
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
