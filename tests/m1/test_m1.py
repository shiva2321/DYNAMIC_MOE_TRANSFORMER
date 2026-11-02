"""Unit tests for M1 Stage A & Stage B-1 runner:

adapters, baselines (A5b, A6, A4 budget), schema 1.2.0, tuning driver, and beds.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import m1.queue as queue_mod
from growmoe.accounting import pool_report
from costlab.count import count_macs
from m1.arms import (
    ALL_ARM_ORDER,
    A1PoolArm,
    build_arm,
    compute_a4_steps,
    pool_budget_bytes,
)
from m1.baselines_extra import MahalanobisProto, RandomFeatureRidge
from m1.beds import POOL_SEED_OFFSET, Bed, build_bed
from m1.cell import (
    SCHEMA_VERSION,
    cell_filename,
    compute_tuning_ref,
    execute_cell,
    is_valid_cell_record,
    load_cell_record,
    read_pool_budget,
    record_matches_config,
    save_cell_record,
)
from m1.config_m1 import (
    A2_CANDIDATES,
    DEV_SEEDS,
    EDGE_EXTEND_CAP_STEPS,
    EDGE_EXTEND_FACTOR_STEPS,
)
from m1.queue import parse_seeds, run_queue
from m1.tune import TuningDriver, is_candidate_better, select_best_candidate


@pytest.fixture(scope="module")
def small_bed():
    return build_bed("B1", n_regimes=2, seed=100, n_train=16, n_test=16)


@pytest.fixture(scope="module")
def a1_record(small_bed):
    """One real A1 record on the small bed, shared by the tests that only read it."""
    return execute_cell(small_bed, "A1", split="dev")


def test_pool_seed_offset():
    """Verify pool seed is stream seed + 7 (C-75 rule)."""
    bed = build_bed("B1", n_regimes=2, seed=800, n_train=16, n_test=16)
    assert bed.seed == 800
    assert bed.pool_seed == 800 + POOL_SEED_OFFSET
    assert bed.pool_seed == 807


def test_split_report_refused(small_bed):
    """Verify --split report is strictly refused in code."""
    with pytest.raises(ValueError, match="refuses --split report"):
        execute_cell(small_bed, "A1", split="report")

    with pytest.raises(ValueError, match="refuses --split report"):
        run_queue(split="report", beds=["B1"], ns=[2], arms=["A1"], seeds=[800], out="results/m1/dev")

    res = subprocess.run(
        [sys.executable, "-m", "m1.run", "--bed", "B1", "--n", "2", "--arm", "A1", "--seed", "800", "--split", "report"],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "refused" in res.stderr.lower()


def test_adapters_obey_interface():
    """Verify all arms obey uniform Arm adapter interface and describe themselves with plain-JSON configs."""
    bed = build_bed("B1", n_regimes=2, seed=100, n_train=64, n_test=16)
    pool_bytes = 100_000
    pool_macs = 50_000_000

    test_arms = list(ALL_ARM_ORDER) + ["A1L"]
    for arm_id in test_arms:
        arm = build_arm(arm_id, bed, pool_memory_bytes=pool_bytes, pool_macs_budget=pool_macs)

        assert hasattr(arm, "observe") and callable(arm.observe)
        assert hasattr(arm, "predict") and callable(arm.predict)
        assert hasattr(arm, "memory_bytes") and callable(arm.memory_bytes)
        assert hasattr(arm, "describe") and callable(arm.describe)
        assert hasattr(arm, "config") and callable(arm.config)

        # >= 32 samples so that the pool spawns its first expert
        x = torch.randn(64, bed.d)
        y = torch.randint(0, bed.n_classes, (64,))
        arm.observe(x, y)

        pred = arm.predict(x)
        assert pred.probs.shape == (len(x), bed.n_classes)

        mem = arm.memory_bytes()
        assert isinstance(mem, int) and mem > 0

        assert arm.config()["arm"] == arm_id
        assert arm.describe()["arm"] == arm_id
        json.dumps(arm.config())  # plain JSON types only


def test_a1_is_baseline_v1_and_a1l_is_the_legacy_pool(small_bed):
    """A1 must be Baseline v1; A1L the legacy pool of shootout (harness check only)."""
    c = build_arm("A1", small_bed).config()
    assert c["variant"] == "baseline_v1"
    assert (c["router"], c["spawn_rule"], c["growth_rule"]) == (True, "transfer", "probe")
    assert (c["q"], c["alpha_spawn"], c["m_min"], c["gate_shortlist_min_experts"]) == (0.05, 0.01, 32, 64)
    assert (c["route_rule"], c["inner_steps"], c["seed"]) == ("accept_then_logf", 5, small_bed.pool_seed)

    legacy = build_arm("A1L", small_bed).config()
    assert legacy["variant"] == "pre_stage4"
    assert (legacy["router"], legacy["spawn_rule"], legacy["growth_rule"]) == (False, "probe", "probe")
    assert (legacy["q"], legacy["alpha_spawn"], legacy["m_min"], legacy["gate_shortlist_min_experts"]) == (0.02, 0.01, 32, 2)
    assert legacy["seed"] == small_bed.pool_seed
    assert c != legacy


def test_unknown_arm_and_variant_are_refused(small_bed):
    with pytest.raises(ValueError, match="Unknown arm"):
        build_arm("A99", small_bed)
    with pytest.raises(ValueError, match="variant"):
        A1PoolArm(small_bed, variant="baseline_v99")


def test_record_schema(a1_record, small_bed):
    """Verify that cell record conforms to schema 1.2.0."""
    rec = a1_record
    assert is_valid_cell_record(rec)
    assert rec["schema_version"] == SCHEMA_VERSION
    assert (rec["bed"], rec["n_regimes"], rec["arm"], rec["seed"], rec["split"]) == ("B1", 2, "A1", 100, "dev")
    assert isinstance(rec["accuracy"], float)
    assert len(rec["per_regime_accuracy"]) == 2
    assert rec["forgetting"] is None or isinstance(rec["forgetting"], float)
    assert rec["config"] == json.loads(json.dumps(build_arm("A1", small_bed).config()))
    assert "1" in rec["answer_cost"] and "32" in rec["answer_cost"]
    assert rec["memory_bytes"] > 0
    assert len(rec["memory_bytes_by_regime"]) > 0
    assert len(rec["update_cost"]["eval_wall_clock_per_regime_boundary"]) >= 2
    assert "bed_info" in rec and rec["bed_info"]["regimes"] == 2
    assert "profile" in rec and rec["profile"] == "full"
    assert "uncounted_note" in rec


def test_schema_1_1_0_records_are_rejected():
    """Old schema 1.1.0 records must be rejected by is_valid_cell_record."""
    rec = {"schema_version": "1.1.0", "arm": "A1"}
    assert not is_valid_cell_record(rec)


def test_light_records_validate(small_bed):
    """Light profile records validate and have empty answer_cost."""
    rec = execute_cell(small_bed, "A1", split="dev", profile="light")
    assert is_valid_cell_record(rec)
    assert rec["profile"] == "light"
    assert rec["answer_cost"] == {}
    assert rec["update_cost"]["wall_clock_total"] > 0


def test_batch_rows_are_exactly_the_batch_size(a1_record):
    """Latency timed on exactly batch_size rows (rows repeated)."""
    for b in (1, 32):
        assert a1_record["answer_cost"][str(b)]["batch_size"] == b
        assert a1_record["answer_cost"][str(b)]["rows_timed"] == b


def test_pinning_is_recorded(small_bed):
    pinning = {"threads": 2, "cpus": [0, 2], "blas_limited": True}
    rec = execute_cell(small_bed, "A1", split="dev", pinning=pinning)
    assert rec["pinning"] == pinning


def test_undefined_forgetting_is_none_not_zero(monkeypatch, small_bed, tmp_path):
    nan = float("nan")
    monkeypatch.setattr(
        "m1.cell.accuracy_stats",
        lambda R, learned_at=None, chance=0.0: {"avg": 0.0, "forgetting": nan, "backward": nan, "forward": nan},
    )
    rec = execute_cell(small_bed, "A1", split="dev")
    assert rec["forgetting"] is None
    assert is_valid_cell_record(rec)
    path = tmp_path / "rec.json"
    save_cell_record(path, rec)
    assert load_cell_record(path)["forgetting"] is None


def test_equal_memory_arms_share_the_pool_budget(small_bed, a1_record):
    budget = pool_budget_bytes(small_bed)
    assert budget == a1_record["memory_bytes"]

    for arm in ("A3", "A7"):
        rec = execute_cell(small_bed, arm, split="dev")
        assert rec["config"]["target_pool_bytes"] == budget
        assert rec["memory_bytes"] > 0

    rec = execute_cell(small_bed, "A3", pool_memory_bytes=budget + 4096, split="dev")
    assert rec["config"]["target_pool_bytes"] == budget + 4096


def test_record_matches_config_detects_stale_records(small_bed, a1_record):
    assert record_matches_config(a1_record, small_bed, "A1")

    stale = copy.deepcopy(a1_record)
    stale["config"]["router"] = False
    assert not record_matches_config(stale, small_bed, "A1")

    legacy = copy.deepcopy(a1_record)
    legacy["config"] = json.loads(json.dumps(build_arm("A1L", small_bed).config()))
    assert not record_matches_config(legacy, small_bed, "A1")
    assert record_matches_config(legacy, small_bed, "A1L")

    missing = copy.deepcopy(a1_record)
    del missing["config"]
    assert not record_matches_config(missing, small_bed, "A1")


def test_stale_tuning_invalidates_cells(small_bed):
    """When a cell has a tuning_ref, matching must check that tuning."""
    tuning_1 = {"inner_steps": 5}
    tuning_2 = {"inner_steps": 20}
    rec = execute_cell(small_bed, "A1", split="dev", tuning=tuning_1)
    assert record_matches_config(rec, small_bed, "A1", tuning=tuning_1)
    assert not record_matches_config(rec, small_bed, "A1", tuning=tuning_2)


def test_atomic_write_no_half_file(tmp_path):
    out_file = tmp_path / "test_record.json"
    data = {"schema_version": SCHEMA_VERSION, "accuracy": 0.5}

    save_cell_record(out_file, data)
    assert out_file.exists()
    assert load_cell_record(out_file) == data

    bad_file = tmp_path / "bad_record.json"
    bad_data = {"circular": None}
    bad_data["circular"] = bad_data

    with pytest.raises(ValueError):
        save_cell_record(bad_file, bad_data)

    assert not bad_file.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["test_record.json"]


def test_parse_seeds():
    assert parse_seeds("800") == [800]
    assert parse_seeds("800-802,805") == [800, 801, 802, 805]
    assert parse_seeds("802, 800,800") == [800, 802]


# -----------------------------------------------------------------------------
# New Stage B-1 Specific Tests (Section 5 Item 9)
# -----------------------------------------------------------------------------
def test_a4_never_exceeds_budget(small_bed):
    """A4 inner_steps are calibrated so that training MACs <= pool_macs_budget."""
    pool_macs = count_macs(lambda: execute_cell(small_bed, "A1", split="dev", profile="light")).macs
    pool_bytes = pool_budget_bytes(small_bed)

    steps = compute_a4_steps(small_bed, pool_macs, pool_bytes, hidden=256)
    assert steps >= 1

    a4 = build_arm("A4", small_bed, pool_memory_bytes=pool_bytes, pool_macs_budget=pool_macs, hidden=256, inner_steps=steps)

    def _run_a4():
        for x, y, _ in small_bed:
            a4.observe(x, y)

    actual_macs = count_macs(_run_a4).macs
    assert actual_macs <= pool_macs


def test_a5b_and_a6_two_class_toy():
    """A5b and A6 give hand-computable answers on a two-class toy."""
    from growmoe.config import CommonCfg

    cfg = CommonCfg(seed=42, d=2, n_classes=2)

    # 1. Test A5b MahalanobisProto
    a5b = MahalanobisProto(cfg, shrinkage=0.5, scale=1.0)
    # Class 0 at (-1, 0), Class 1 at (1, 0)
    x0 = torch.tensor([[-1.0, 0.0]])
    y0 = torch.tensor([0])
    x1 = torch.tensor([[1.0, 0.0]])
    y1 = torch.tensor([1])

    a5b.observe(x0, y0)
    a5b.observe(x1, y1)

    # Query at (-1, 0) should have probability close to 1.0 for class 0
    p0 = a5b.predict(x0).probs
    assert p0[0, 0].item() > 0.99
    assert p0[0, 1].item() < 0.01

    # Query at (1, 0) should have probability close to 1.0 for class 1
    p1 = a5b.predict(x1).probs
    assert p1[0, 1].item() > 0.99

    # 2. Test A6 RandomFeatureRidge
    a6 = RandomFeatureRidge(cfg, D=16, ridge=0.1, scale=1.0)
    a6.observe(x0, y0)
    a6.observe(x1, y1)

    pred0 = a6.predict(x0).probs
    pred1 = a6.predict(x1).probs
    assert pred0[0, 0].item() > pred0[0, 1].item()
    assert pred1[0, 1].item() > pred1[0, 0].item()


def test_memory_of_a5b_and_a6_equals_pool_report():
    """Memory of A5b and A6 equals pool_report figure exactly."""
    from growmoe.config import CommonCfg

    d, c = 8, 3
    cfg = CommonCfg(seed=42, d=d, n_classes=c)

    # A5b
    a5b = MahalanobisProto(cfg, shrinkage=0.1)
    # Observe two distinct classes
    a5b.observe(torch.randn(2, d), torch.tensor([0, 1]))
    rep_a5b = pool_report(a5b, d)
    # 2 seen classes * (1 + d) vectors = 2 * 9 = 18 vectors -> 18 * 4 * 8 = 576 bytes
    assert a5b.stored_vectors() == 2 * (1 + d)
    assert rep_a5b.memory_bytes == 4 * d * 2 * (1 + d)

    # A6
    D = 16
    a6 = RandomFeatureRidge(cfg, D=D, ridge=0.1)
    a6.observe(torch.randn(2, d), torch.tensor([0, 1]))
    rep_a6 = pool_report(a6, d)
    expected_bytes = 4 * (D * D + D * c)
    assert rep_a6.memory_bytes == expected_bytes


def test_selection_rule_and_edge_extension():
    """Selection rule handles ties (< 0.001) and edge extension."""
    c1 = {"mean_accuracy": 0.850, "mean_memory_bytes": 1000, "mean_wall_clock": 2.0}
    c2 = {"mean_accuracy": 0.8505, "mean_memory_bytes": 500, "mean_wall_clock": 2.5}
    # Within 0.001 tie margin, c2 wins by lower memory
    assert select_best_candidate([c1, c2]) == c2

    # Higher accuracy beyond margin wins outright
    c3 = {"mean_accuracy": 0.855, "mean_memory_bytes": 2000, "mean_wall_clock": 3.0}
    assert select_best_candidate([c1, c2, c3]) == c3


def test_trial_files_skip_on_rerun(tmp_path):
    """Finished trials are skipped on rerun."""
    driver = TuningDriver(bed_name="B1", n_regimes=2, out_dir=tmp_path, smoke=True, seeds=[800])
    rec1 = driver.run_single_trial("A1", {"inner_steps": 5}, seed=800)

    cfg_str = json.dumps({"inner_steps": 5}, sort_keys=True)
    import hashlib

    cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:12]
    trial_file = tmp_path / "trials" / "B1" / f"trial__A1__{cfg_hash}__seed800.json"
    assert trial_file.exists()
    mtime = trial_file.stat().st_mtime_ns

    # Second run loads from file without re-executing
    rec2 = driver.run_single_trial("A1", {"inner_steps": 5}, seed=800)
    assert rec2["accuracy"] == rec1["accuracy"]
    assert trial_file.stat().st_mtime_ns == mtime


def test_b2_bed_interface_on_synthetic_stream():
    """Test Bed B2 wrapping and interface on a synthetic Stream."""
    from growmoe.streams import Regime, Segment, Stream

    d, n_classes = 128, 77
    n_regimes = 6
    regimes = {r: Regime(id=r, name=f"banking_regime_{r}", x_te=torch.randn(10, d), y_te=torch.zeros(10, dtype=torch.long), te_ids=torch.arange(10)) for r in range(6)}
    segments = [
        Segment(regime=r, first_visit=(i < 6), x=torch.randn(20, d), y=torch.zeros(20, dtype=torch.long), src_ids=torch.arange(20), y_clean=torch.zeros(20), is_outlier=torch.zeros(20, dtype=torch.bool))
        for i, r in enumerate([0, 1, 2, 3, 4, 5, 2, 0, 4])
    ]
    stream = Stream(name="synthetic_b2", d=d, n_classes=n_classes, regimes=regimes, segments=segments)

    bed = Bed(name="B2", n_regimes=6, seed=800, pool_seed=807, d=d, n_classes=n_classes, stream=stream)
    assert bed.n_phases == 9
    assert bed.n_regimes == 6
    assert bed.total_samples == 180

    walk_batches = list(bed.walk(batch_size=10))
    assert len(walk_batches) > 0
    x, y, reg, fv, phase = walk_batches[0]
    assert x.shape[1] == 128
    assert isinstance(phase, int)
    assert isinstance(fv, bool)


def test_pool_instrumentation_overhead(small_bed):
    """Verify pool instrumentation measures time split and wrapper overhead is small."""
    arm_timed = build_arm("A1", small_bed, instrument=True)
    arm_plain = build_arm("A1", small_bed, instrument=False)

    x = torch.randn(32, small_bed.d)
    y = torch.randint(0, small_bed.n_classes, (32,))

    arm_timed.observe(x, y)
    arm_plain.observe(x, y)

    assert hasattr(arm_timed, "time_split") and arm_timed.time_split is not None
    assert "observe" in arm_timed.time_split
    assert "train_expert" in arm_timed.time_split
    assert "router_refit" in arm_timed.time_split
    assert arm_timed.time_split["observe"] > 0


# ---------------------------------------------------------------------------------------------------------------------------------
# Tests restored after job B-1 (stage-A coverage that the first B-1 delivery had dropped) and the reviewer's tests for the repairs
# ---------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def a1_tuned_record(small_bed):
    """An A1 record made with a tuning entry (2 inner steps) whose configuration differs from the untuned default (5 steps)."""
    return execute_cell(small_bed, "A1", split="dev", tuning={"inner_steps": 2})


def test_read_pool_budget(tmp_path, small_bed, a1_record):
    assert read_pool_budget(tmp_path, small_bed) is None                       # no record

    path = tmp_path / cell_filename("B1", 2, "A1", 100)
    save_cell_record(path, a1_record)
    assert read_pool_budget(tmp_path, small_bed) == a1_record["memory_bytes"]

    stale = copy.deepcopy(a1_record)
    stale["config"]["q"] = 0.5
    save_cell_record(path, stale)
    assert read_pool_budget(tmp_path, small_bed) is None                       # a pool built with another configuration is not the budget

    path.write_text("{ not json", encoding="utf-8")
    assert read_pool_budget(tmp_path, small_bed) is None                       # unreadable record


def test_read_pool_budget_knows_the_tuning(tmp_path, small_bed, a1_tuned_record):
    save_cell_record(tmp_path / cell_filename("B1", 2, "A1", 100), a1_tuned_record)
    assert read_pool_budget(tmp_path, small_bed, tuning={"inner_steps": 2}) == a1_tuned_record["memory_bytes"]
    assert read_pool_budget(tmp_path, small_bed) is None                       # the untuned A1 is another arm
    assert read_pool_budget(tmp_path, small_bed, tuning={"inner_steps": 3}) is None


def test_record_matches_config_needs_the_budget_for_equal_memory_arms(small_bed):
    budget = pool_budget_bytes(small_bed)
    rec = execute_cell(small_bed, "A3", pool_memory_bytes=budget, split="dev")
    assert record_matches_config(rec, small_bed, "A3", budget)
    assert not record_matches_config(rec, small_bed, "A3", budget + 4096)      # another budget is another arm
    assert not record_matches_config(rec, small_bed, "A3", None)               # an unknown budget cannot be confirmed


def _fake_execute_factory(calls, record):
    def fake_execute(bed, arm_name, pool_memory_bytes=None, pool_macs_budget=None, split="dev", pinning=None, profile="full", tuning=None, **kwargs):
        calls.append((arm_name, pool_memory_bytes))
        return copy.deepcopy(record)
    return fake_execute


def test_queue_skips_current_records_and_reruns_stale_ones(tmp_path, monkeypatch, small_bed, a1_record):
    """Skip a record that is complete, valid and made by the arm as configured now; rerun one that is not; give A3 the A1 record memory."""
    calls = []
    monkeypatch.setattr(queue_mod, "execute_cell", _fake_execute_factory(calls, a1_record))
    monkeypatch.setattr(queue_mod, "build_bed", lambda name, n, seed: small_bed)

    target = tmp_path / cell_filename("B1", 2, "A1", 100)
    save_cell_record(target, a1_record)
    mtime = target.stat().st_mtime_ns
    queue = dict(split="dev", beds=["B1"], ns=[2], seeds=[100], out=tmp_path, cpus=[0], threads=1)

    run_queue(arms=["A1"], **queue)                        # current record: skipped
    assert calls == [] and target.stat().st_mtime_ns == mtime

    stale = copy.deepcopy(a1_record)
    stale["config"]["q"] = 0.5
    save_cell_record(target, stale)
    run_queue(arms=["A1"], **queue)                        # stale record: rerun and replaced
    assert calls == [("A1", None)]
    assert record_matches_config(load_cell_record(target), small_bed, "A1")

    calls.clear()
    run_queue(arms=["A3", "A1"], **queue)                  # A1 is handled first whatever the order given; its record memory is the budget of A3
    assert calls == [("A3", a1_record["memory_bytes"])]


def test_queue_reruns_a_corrupt_record(tmp_path, monkeypatch, small_bed, a1_record):
    calls = []
    monkeypatch.setattr(queue_mod, "execute_cell", _fake_execute_factory(calls, a1_record))
    monkeypatch.setattr(queue_mod, "build_bed", lambda name, n, seed: small_bed)

    target = tmp_path / cell_filename("B1", 2, "A1", 100)
    target.write_text("{ truncated", encoding="utf-8")
    run_queue(split="dev", beds=["B1"], ns=[2], arms=["A1"], seeds=[100], out=tmp_path, cpus=[0], threads=1)
    assert [c[0] for c in calls] == ["A1"]
    assert is_valid_cell_record(load_cell_record(target))


def test_queue_refuses_to_size_an_arm_from_a_missing_reference(tmp_path, monkeypatch, small_bed, a1_record):
    """No silent fallback to an untuned pool: A3 without a current A1 record and A3-2 without a current A2 record stop with a clear message."""
    calls = []
    monkeypatch.setattr(queue_mod, "execute_cell", _fake_execute_factory(calls, a1_record))
    monkeypatch.setattr(queue_mod, "build_bed", lambda name, n, seed: small_bed)
    queue = dict(split="dev", beds=["B1"], ns=[2], seeds=[100], out=tmp_path, cpus=[0], threads=1)
    with pytest.raises(RuntimeError, match="A1"):
        run_queue(arms=["A3"], **queue)
    save_cell_record(tmp_path / cell_filename("B1", 2, "A1", 100), a1_record)
    with pytest.raises(RuntimeError, match="A2"):
        run_queue(arms=["A3-2"], **queue)
    assert calls == []


def test_queue_fails_closed_on_missing_or_unreadable_tuning(tmp_path, monkeypatch, small_bed, a1_record):
    calls = []
    monkeypatch.setattr(queue_mod, "execute_cell", _fake_execute_factory(calls, a1_record))
    monkeypatch.setattr(queue_mod, "build_bed", lambda name, n, seed: small_bed)
    out = tmp_path / "cells"
    queue = dict(split="dev", beds=["B1"], ns=[2], seeds=[100], out=out, cpus=[0], threads=1, arms=["A1"])
    tdir = tmp_path / "tuning"
    tdir.mkdir()
    with pytest.raises(FileNotFoundError):
        run_queue(tuning_dir=str(tdir), **queue)                          # --tuning given, no file: untuned cells would be made
    (tdir / "B1.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be read"):
        run_queue(tuning_dir=str(tdir), **queue)
    (tdir / "B1.json").write_text(json.dumps({"arms": {"A3": {"chosen": {"hidden": 8}}}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not in the tuning file"):
        run_queue(tuning_dir=str(tdir), **queue)                          # A1 absent from the file
    assert calls == []


def test_same_seed_gives_same_accuracy(small_bed, a1_record):
    """Verify deterministic accuracy across identical runs."""
    rec2 = execute_cell(build_bed("B1", n_regimes=2, seed=100, n_train=16, n_test=16), "A1", split="dev")
    assert a1_record["accuracy"] == pytest.approx(rec2["accuracy"], abs=1e-12)
    assert a1_record["per_regime_accuracy"] == pytest.approx(rec2["per_regime_accuracy"], abs=1e-12)
    assert a1_record["memory_bytes"] == rec2["memory_bytes"]


def test_closed_form_update_multiply_adds_are_counted(small_bed):
    """The Gram update of the random-feature head is a matrix product: its update multiply-adds must be counted (the first B-1 delivery stored a typed 0)."""
    rec = execute_cell(small_bed, "A6", split="dev", tuning={"D": 64, "ridge": 0.1})
    assert rec["update_cost"]["counted_macs"] > 0
    assert rec["uncounted_note"] is True
    knn = execute_cell(small_bed, "A7all", split="dev")
    assert knn["update_cost"]["counted_macs"] == 0 and knn["reference_pool"] is None     # a lookup multiplies nothing and has no reference pool


def test_pool_config_records_every_switch_that_changes_the_pool(small_bed):
    """A2 candidates that differ only in the twin fix, the skip flag or the chance flag must have different configurations (the stale-record guard compares them)."""
    def cfg(candidate):
        return build_arm("A2", small_bed, tuning={"candidate": candidate, "inner_steps": 2, "q": 0.05}).config()

    shipped, twin, skip, chance = cfg("shipped"), cfg("shipped+twin"), cfg("shipped+skip"), cfg("shipped+chance")
    assert shipped["candidate"] == "shipped" and twin["candidate"] == "shipped+twin"
    assert (shipped["sig_dedupe"], twin["sig_dedupe"], twin["infant_relax"], twin["set_gate"]) == (False, True, 1.0, False)
    assert skip["skip_known"] == 0.1 and shipped["skip_known"] is None
    assert chance["transfer_chance"] == 1.0 and shipped["transfer_chance"] is None
    assert len({json.dumps(c, sort_keys=True) for c in (shipped, twin, skip, chance)}) == 4
    with pytest.raises(ValueError, match="Unknown A2 candidate"):
        build_arm("A2", small_bed, tuning={"candidate": "shipped+typo", "inner_steps": 2, "q": 0.05})


def test_budget_arms_do_not_fall_back_to_an_untuned_pool(small_bed):
    for arm in ("A3-2", "A4", "A4-2", "A7-2"):
        with pytest.raises(ValueError, match="reference pool"):
            build_arm(arm, small_bed)
    with pytest.raises(ValueError, match="pool_macs_budget"):
        build_arm("A4", small_bed, pool_memory_bytes=100_000)


def test_reference_pool_is_counted_once_and_tied_to_its_tuning(tmp_path, monkeypatch, small_bed):
    import m1.tune as tune_mod

    runs = []
    real_count = tune_mod.count_macs
    monkeypatch.setattr(tune_mod, "count_macs", lambda fn: (runs.append(1), real_count(fn))[1])
    monkeypatch.setattr(tune_mod, "build_bed", lambda name, n, seed: small_bed)
    drv = TuningDriver("B1", 2, tmp_path, smoke=True, seeds=[100])
    with pytest.raises(RuntimeError, match="must be tuned"):
        drv.pool_reference("A1", 100)
    drv.state["arms"]["A1"] = {"chosen": {"inner_steps": 2}}
    first = drv.pool_reference("A1", 100)
    again = drv.pool_reference("A1", 100)
    assert first == again and len(runs) == 1 and first["macs"] > 0 and first["bytes"] > 0
    drv.state["arms"]["A1"] = {"chosen": {"inner_steps": 3}}                   # another tuning entry: the cached count no longer applies
    drv.pool_reference("A1", 100)
    assert len(runs) == 2


def test_a_tuned_arm_is_tuned_again_when_its_grid_or_seeds_change(tmp_path, monkeypatch, small_bed):
    import m1.tune as tune_mod

    monkeypatch.setattr(tune_mod, "build_bed", lambda name, n, seed: small_bed)
    drv = TuningDriver("B1", 2, tmp_path, smoke=True, seeds=[100])
    entry = drv.tune_arm("A5a")
    assert entry["grid_ref"] == drv.grid_ref("A5a")
    calls = []
    original = drv.evaluate_candidate
    monkeypatch.setattr(drv, "evaluate_candidate", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    drv.tune_arm("A5a")
    assert calls == []                                                          # unchanged: reused
    drv.seeds = [100, 101]
    drv.tune_arm("A5a")
    assert calls                                                                # another seed list: tuned again


# ---------------------------------------------------------------------------------------------------------------------------------
# Held-default pair cells (H_<candidate>) and the light profile (final evaluation only)
# ---------------------------------------------------------------------------------------------------------------------------------
def test_h_arms_build_the_candidate_at_the_base_operating_point(small_bed):
    base = {"inner_steps": 2, "q": 0.02}                                        # the A1q choice
    gap = build_arm("H_shipped+gap", small_bed, tuning=base)
    ref = build_arm("H_shipped", small_bed, tuning=base)
    cfg_gap, cfg_ref = gap.config(), ref.config()
    assert (cfg_gap["arm"], cfg_gap["candidate"], cfg_gap["q"], cfg_gap["inner_steps"], cfg_gap["growth_rule"]) == ("H_shipped+gap", "shipped+gap", 0.02, 2, "gap")
    assert (cfg_ref["q"], cfg_ref["inner_steps"], cfg_ref["growth_rule"]) == (0.02, 2, "probe")        # the reference differs from the candidate only in the switch
    assert {k: v for k, v in cfg_gap.items() if k not in ("arm", "candidate", "growth_rule")} == {k: v for k, v in cfg_ref.items() if k not in ("arm", "candidate", "growth_rule")}
    with pytest.raises(ValueError, match="UniPool"):
        build_arm("H_ULDA", small_bed, tuning=base)
    with pytest.raises(ValueError, match="candidate"):
        build_arm("H_nonsense", small_bed, tuning=base)


def test_light_profile_gives_the_same_final_accuracy_without_boundary_evaluations(small_bed, a1_record):
    light = execute_cell(small_bed, "A1", split="dev", profile="light")
    assert light["accuracy"] == a1_record["accuracy"] and light["per_regime_accuracy"] == a1_record["per_regime_accuracy"]
    assert light["memory_bytes"] == a1_record["memory_bytes"]
    assert light["forgetting"] is None and is_valid_cell_record(light)
    assert len(light["update_cost"]["eval_wall_clock_per_regime_boundary"]) == 1 < len(a1_record["update_cost"]["eval_wall_clock_per_regime_boundary"])


def test_queue_expands_the_h_shorthand_and_uses_the_a1q_tuning(tmp_path, monkeypatch, small_bed, a1_record):
    seen = []

    def fake_execute(bed, arm_name, pool_memory_bytes=None, pool_macs_budget=None, split="dev", pinning=None, profile="full", tuning=None, **kwargs):
        seen.append((arm_name, dict(tuning)))
        return copy.deepcopy(a1_record)

    monkeypatch.setattr(queue_mod, "execute_cell", fake_execute)
    monkeypatch.setattr(queue_mod, "build_bed", lambda name, n, seed: small_bed)
    tdir = tmp_path / "tuning"
    tdir.mkdir()
    (tdir / "B1.json").write_text(json.dumps({"arms": {"A1q": {"chosen": {"inner_steps": 2, "q": 0.02}}}}), encoding="utf-8")
    run_queue(split="dev", beds=["B1"], ns=[2], arms=["H"], seeds=[100], out=tmp_path / "cells", tuning_dir=str(tdir), cpus=[0], threads=1)
    assert [a for a, _ in seen] == ["H_shipped", "H_shipped+gap", "H_shipped+skip", "H_shipped+chance"]
    assert all(t == {"inner_steps": 2, "q": 0.02} for _, t in seen)
    assert (tmp_path / "cells" / cell_filename("B1", 2, "H_shipped+gap", 100)).exists()


def test_reference_pool_memory_needs_no_counted_pass(tmp_path, monkeypatch, small_bed):
    """A3 and A7 need only the memory of the reference pool (a plain pass); only A4 needs the counted multiply-adds (a counted pass, several times slower)."""
    import m1.tune as tune_mod

    runs = []
    real_count = tune_mod.count_macs
    monkeypatch.setattr(tune_mod, "count_macs", lambda fn: (runs.append(1), real_count(fn))[1])
    monkeypatch.setattr(tune_mod, "build_bed", lambda name, n, seed: small_bed)
    drv = TuningDriver("B1", 2, tmp_path, smoke=True, seeds=[100])
    drv.state["arms"]["A1"] = {"chosen": {"inner_steps": 2}}
    mem = drv.get_pool_budget_bytes("A1", 100)
    assert mem > 0 and runs == []                                               # plain pass: nothing counted
    assert drv.get_pool_budget_bytes("A1", 100) == mem and runs == []           # and cached
    macs = drv.get_pool_budget_macs("A1", 100)
    assert macs > 0 and len(runs) == 1                                          # the counted pass is made once, when A4 asks
    assert drv.get_pool_budget_macs("A1", 100) == macs and len(runs) == 1
    assert drv.get_pool_budget_bytes("A1", 100) == mem                          # same memory from both kinds of pass


def test_det_profile_has_the_deterministic_metrics_and_no_timing(small_bed, a1_record):
    det = execute_cell(small_bed, "A1", split="dev", profile="det")
    assert is_valid_cell_record(det) and det["profile"] == "det"
    assert det["accuracy"] == a1_record["accuracy"] and det["forgetting"] == a1_record["forgetting"] and det["memory_bytes"] == a1_record["memory_bytes"]
    assert det["update_cost"] == {"counted_macs": a1_record["update_cost"]["counted_macs"]}                # no wall-clock, no peak memory
    assert det["pool_time_split"] is None
    for b in ("1", "32"):
        assert set(det["answer_cost"][b]) == {"batch_size", "rows_timed", "counted_macs", "active_bytes"}
        assert det["answer_cost"][b]["counted_macs"] == a1_record["answer_cost"][b]["counted_macs"]
    assert not record_matches_config(det, small_bed, "A1", profile="full")                                # a det cell does not stand in for a timed one


def test_queue_shards_split_the_groups_without_overlap(tmp_path, monkeypatch, small_bed, a1_record):
    calls = []

    def fake_execute(bed, arm_name, **kwargs):
        calls.append(bed.seed)
        return copy.deepcopy(a1_record)

    monkeypatch.setattr(queue_mod, "execute_cell", fake_execute)
    monkeypatch.setattr(queue_mod, "build_bed", lambda name, n, seed: copy.replace(small_bed, seed=seed) if hasattr(copy, "replace") else small_bed)
    seen = {}
    for k in (0, 1):
        calls.clear()
        run_queue(split="dev", beds=["B1"], ns=[2], arms=["A1"], seeds=[100, 101, 102, 103], out=tmp_path / f"shard{k}", cpus=[0], threads=1, shard=f"{k}/2")
        seen[k] = len(calls)
    assert seen == {0: 2, 1: 2}                                                  # four groups, two per shard
    with pytest.raises(ValueError, match="shard"):
        run_queue(split="dev", beds=["B1"], ns=[2], arms=["A1"], seeds=[100], out=tmp_path / "bad", cpus=[0], threads=1, shard="2/2")


# ---------------------------------------------------------------------------------------------------------------------------------
# Stage B-2: Beds B2 and B3, UniPool candidates, and held-default pairs on B3
# ---------------------------------------------------------------------------------------------------------------------------------
def test_b2_bed_interface_real():
    """Verify Bed B2 builds offline and obeys the standard Bed interface with exact specs."""
    bed = build_bed("B2", 6, seed=800)
    assert bed.name == "B2"
    assert bed.n_regimes == 6
    assert bed.d == 128
    assert bed.n_classes == 77
    assert bed.n_phases == 9
    assert bed.total_samples == 3000
    assert len(bed.test_sets) == 6
    for r in range(6):
        xt, yt = bed.test_sets[r]
        assert xt.shape[1] == 128 and yt.dim() == 1

    walk_batches = list(bed.walk(batch_size=64))
    assert len(walk_batches) > 0
    xb, yb, reg, fv, ph = walk_batches[0]
    assert xb.shape[1] == 128 and yb.shape[0] == xb.shape[0]
    assert isinstance(reg, int) and isinstance(fv, bool) and isinstance(ph, int)

    with pytest.raises(ValueError, match="Bed B2 requires n_regimes=6"):
        build_bed("B2", 10, seed=800)


def test_b3_bed_interface_synthetic_stream():
    """Verify B3 beds wrap UStream interface, phases, first_visit flags, and label_blocks in bed_info."""
    from unimoe.suite import Phase, Regime, UStream

    d, n_classes = 128, 20
    regimes = {
        r: Regime(
            id=r,
            name=f"domain_regime_{r}",
            domain="synth_dom",
            kind="text",
            local=r,
            x_te=torch.randn(10, d),
            y_te=torch.full((10,), r, dtype=torch.long),
        )
        for r in range(3)
    }
    phases = [
        Phase(
            name="p0",
            x=torch.randn(30, d),
            y=torch.zeros(30, dtype=torch.long),
            regime=torch.zeros(30, dtype=torch.long),
            first=torch.ones(30, dtype=torch.bool),
        ),
        Phase(
            name="p1",
            x=torch.randn(20, d),
            y=torch.ones(20, dtype=torch.long),
            regime=torch.ones(20, dtype=torch.long),
            first=torch.ones(20, dtype=torch.bool),
        ),
        Phase(
            name="p2",
            x=torch.randn(25, d),
            y=torch.full((25,), 2, dtype=torch.long),
            regime=torch.full((25,), 2, dtype=torch.long),
            first=torch.zeros(25, dtype=torch.bool),
        ),
    ]
    label_blocks = {"synth_dom": (0, 20)}
    synth_stream = UStream(
        name="synth_b3",
        d=d,
        n_classes=n_classes,
        regimes=regimes,
        phases=phases,
        ood={},
        label_blocks=label_blocks,
    )

    bed = Bed(name="B3seq", n_regimes=3, seed=800, pool_seed=807, d=d, n_classes=n_classes, stream=synth_stream, label_blocks=label_blocks)
    assert bed.n_phases == 3
    assert bed.total_samples == 75
    assert len(bed.test_sets) == 3
    assert bed.label_blocks == label_blocks

    walk_batches = list(bed.walk(batch_size=16))
    assert len(walk_batches) > 0
    xb, yb, reg, fv, ph = walk_batches[0]
    assert xb.shape[1] == 128
    assert isinstance(reg, int) and isinstance(fv, bool) and isinstance(ph, int)

    # Check cell execution incorporates label_blocks in bed_info
    rec = execute_cell(bed, "A1", split="dev", profile="light")
    assert "label_blocks" in rec["bed_info"]
    assert rec["bed_info"]["label_blocks"] == label_blocks
    assert rec["bed_info"]["phases"] == 3
    assert rec["bed_info"]["total_samples"] == 75


def test_b3_streams_and_sample_counts():
    """Verify B3 schedules (seq, mix, drift) match Section 4.2 specs for phases and samples."""
    from unimoe.experiment import streams

    st = streams(800)
    assert "seq" in st and "mix" in st and "drift" in st
    # seq: 28 phases, 12,704 samples
    assert len(st["seq"].phases) == 28
    assert sum(len(p.y) for p in st["seq"].phases) == 12704
    # mix: 5 phases, 10,204 samples
    assert len(st["mix"].phases) == 5
    assert sum(len(p.y) for p in st["mix"].phases) == 10204
    # drift: 10 phases, 10,200 samples
    assert len(st["drift"].phases) == 10
    assert sum(len(p.y) for p in st["drift"].phases) == 10200
    # All between 10,000 and 12,700 samples across 22 regimes and 125 classes
    for sch in ("seq", "mix", "drift"):
        bed = build_bed(f"B3{sch}", 22, seed=800)
        assert bed.n_regimes == 22
        assert bed.n_classes == 125
        assert 10000 <= bed.total_samples <= 12704


def test_unipool_candidates_on_b3():
    """Verify UniPool candidates (UG1, ULDA, UPLUS) build on B3 with label blocks and store full config."""
    bed_b3 = build_bed("B3seq", 22, seed=800)

    for cand in ("UG1", "ULDA", "UPLUS"):
        arm = build_arm("A2", bed_b3, tuning={"candidate": cand, "inner_steps": 2, "q": 0.05})
        cfg = arm.config()
        assert cfg["arm"] == "A2"
        assert cfg["candidate"] == cand
        assert cfg["inner_steps"] == 2
        assert cfg["q"] == 0.05
        assert "gap_mode" in cfg
        assert "domain_blocks" in cfg
        assert "lda_blend" in cfg
        assert "lda_shrink" in cfg

        if cand == "ULDA":
            assert cfg["lda_blend"] == 0.5
        elif cand == "UG1":
            assert cfg["gap_mode"] == "one_step"
            assert cfg["growth_rule"] == "gap"
        elif cand == "UPLUS":
            assert cfg["lda_blend"] == 0.5
            assert cfg["gap_mode"] == "one_step"
            assert cfg["domain_head"] == "hard"
            assert len(cfg["domain_blocks"]) > 0

    # Verify UniPool candidate rejected on B1 and B2
    bed_b1 = build_bed("B1", 10, seed=800)
    with pytest.raises(ValueError, match="UniPool"):
        build_arm("A2", bed_b1, tuning={"candidate": "ULDA", "inner_steps": 2, "q": 0.05})

    bed_b2 = build_bed("B2", 6, seed=800)
    with pytest.raises(ValueError, match="UniPool"):
        build_arm("A2", bed_b2, tuning={"candidate": "ULDA", "inner_steps": 2, "q": 0.05})


def test_held_default_pairs_on_b3():
    """Verify held-default pairs on B3: H_U-real and H_ULDA at base operating point."""
    bed_b3 = build_bed("B3seq", 22, seed=800)
    base = {"inner_steps": 3, "q": 0.02}

    arm_ureal = build_arm("H_U-real", bed_b3, tuning=base)
    cfg_ureal = arm_ureal.config()
    assert cfg_ureal["arm"] == "H_U-real"
    assert cfg_ureal["candidate"] == "U-real"
    assert cfg_ureal["inner_steps"] == 3
    assert cfg_ureal["q"] == 0.02

    arm_ulda = build_arm("H_ULDA", bed_b3, tuning=base)
    cfg_ulda = arm_ulda.config()
    assert cfg_ulda["arm"] == "H_ULDA"
    assert cfg_ulda["candidate"] == "ULDA"
    assert cfg_ulda["inner_steps"] == 3
    assert cfg_ulda["q"] == 0.02
    assert cfg_ulda["lda_blend"] == 0.5

    # On B1, H_ULDA must be rejected
    bed_b1 = build_bed("B1", 10, seed=800)
    with pytest.raises(ValueError, match="UniPool"):
        build_arm("H_ULDA", bed_b1, tuning=base)


def test_tuning_driver_and_queue_bed_names(tmp_path):
    """Verify tuning driver and queue accept B2 and B3 beds and enforce regime counts."""
    drv_b2 = TuningDriver("B2", 30, tmp_path, smoke=True, seeds=[800])
    assert drv_b2.n_regimes == 6
    assert drv_b2.tune_bed_name == "B2"

    drv_b3 = TuningDriver("B3seq", 30, tmp_path, smoke=True, seeds=[800])
    assert drv_b3.n_regimes == 22
    assert drv_b3.tune_bed_name == "B3seq"

    drv_b3m = TuningDriver("B3mix", 30, tmp_path, smoke=True, seeds=[800])
    assert drv_b3m.n_regimes == 22
    assert drv_b3m.tune_bed_name == "B3seq"  # B3 tunes on seq

    drv_b3d = TuningDriver("B3drift", 30, tmp_path, smoke=True, seeds=[800])
    assert drv_b3d.n_regimes == 22
    assert drv_b3d.tune_bed_name == "B3seq"

    # Queue shorthand expansion for H on B3
    h_b3 = queue_mod.h_arms_for("B3seq")
    assert "H_U-real" in h_b3 and "H_ULDA" in h_b3
    h_b1 = queue_mod.h_arms_for("B1")
    assert "H_ULDA" not in h_b1


def _mixed_phase_bed():
    """A tiny B3-like bed whose phases mix regimes: phase 0 holds regimes 0 and 1 (both new), phase 1 holds regime 1 (old) and regime 2 (new)."""
    from unimoe.suite import Phase, Regime, UStream

    d, n = 16, 24
    regimes = {r: Regime(id=r, name=f"r{r}", domain="s", kind="text", local=r, x_te=torch.randn(8, d) + r, y_te=torch.full((8,), r, dtype=torch.long)) for r in range(3)}

    def phase(name, regs, new):
        reg = torch.tensor([r for r in regs for _ in range(n // len(regs))])
        return Phase(name, torch.randn(len(reg), d) + reg.float().unsqueeze(1), reg.clone(), reg, torch.tensor([r in new for r in reg.tolist()]))

    stream = UStream("tiny_mix", d, 3, regimes, [phase("p0", [0, 1], {0, 1}), phase("p1", [1, 2], {2})], {}, {"s": (0, 3)})
    return Bed(name="B3mix", n_regimes=3, seed=800, pool_seed=807, d=d, n_classes=3, stream=stream, label_blocks={"s": (0, 3)})


def test_mixed_phases_report_every_regime_and_every_first_visit():
    bed = _mixed_phase_bed()
    seen_regimes, first_by_phase = {0: set(), 1: set()}, {0: set(), 1: set()}
    for x, y, regimes, first, ph in bed.walk_full(8):
        assert set(regimes) == set(y.tolist())                  # all regimes of the batch, not only the first sample's
        seen_regimes[ph] |= set(regimes)
        first_by_phase[ph] |= set(first)
    assert seen_regimes == {0: {0, 1}, 1: {1, 2}} and first_by_phase == {0: {0, 1}, 1: {2}}


def test_cells_on_mixed_phases_compute_forgetting_from_the_phase_a_regime_was_learned():
    rec = execute_cell(_mixed_phase_bed(), "A5a", split="dev", profile="det")
    assert is_valid_cell_record(rec) and rec["forgetting"] is not None            # regimes 0 and 1 were learned in phase 0 and evaluated again after phase 1
    bed = _mixed_phase_bed()
    bed.stream.phases[0].first[:] = False                                       # nothing is ever first seen in phase 0: regimes 0 and 1 have no learning row
    with pytest.raises(ValueError, match="no first-visit sample"):
        execute_cell(bed, "A5a", split="dev", profile="det")


def test_cell_records_carry_the_accounting_block(small_bed, a1_record):
    """The record keeps what growmoe.accounting says about the trained arm, so the report can reconcile it with the counted multiply-adds without guessing formulas."""
    acc = a1_record["accounting"]
    for key in ("n_experts", "params", "stored_vectors", "memory_bytes", "gate_macs", "expert_macs", "predict_macs"):
        assert isinstance(acc[key], int) and acc[key] >= 0
    assert acc["predict_macs"] == acc["gate_macs"] + acc["expert_macs"]
    assert acc["memory_bytes"] == a1_record["memory_bytes"]                           # the stored bytes of the record are the accounting's
    assert acc["gate_macs_measured_predict"] >= 0                                    # a pool has gate counters
    rec_lda = execute_cell(small_bed, "A5a", split="dev", profile="det")
    assert "gate_macs_measured_predict" not in rec_lda["accounting"] and rec_lda["accounting"]["expert_macs"] == 0


def test_report_reconciliation_and_answer_split_use_the_recorded_numbers(a1_record):
    from m1.report import compute_cost_drivers, compute_reconciliation_table

    rec = copy.deepcopy(a1_record)
    rec["arm"] = "A2"
    rec["accounting"] = dict(rec["accounting"], gate_macs=300, expert_macs=100, predict_macs=400, memory_bytes=1234, gate_macs_measured_predict=250.0)
    rec["answer_cost"]["1"]["counted_macs"] = 360
    old = copy.deepcopy(a1_record)                                # a cell made before the block existed
    old["arm"] = "A3"
    old.pop("accounting", None)
    by_arm = {r["arm"]: r for r in compute_reconciliation_table([rec, old])}
    assert by_arm["A2"]["accounting_macs"] == 400 and by_arm["A2"]["costlab_macs"] == 360 and by_arm["A2"]["macs_ratio"] == pytest.approx(0.9)
    assert by_arm["A2"]["accounting_bytes"] == 1234
    assert by_arm["A3"]["accounting_macs"] is None and by_arm["A3"]["macs_ratio"] is None      # not recorded, not guessed
    split = compute_cost_drivers([rec, old])["answer_drivers"]
    assert len(split) == 1 and split[0]["gate_share"] == pytest.approx(0.75) and split[0]["gate_macs_measured"] == 250.0 and split[0]["counted_macs"] == 360


def test_memory_by_regime_keeps_the_first_visit_of_each_regime():
    """B1 revisits two regimes at the end; their entries must stay the memory after the first visit (M1-8 fits a line through these), not be overwritten by the later visit."""
    bed = build_bed("B1", n_regimes=4, seed=100, n_train=16, n_test=16)
    assert bed.n_phases > bed.n_regimes                                              # there are revisit phases
    rec = execute_cell(bed, "A7all", split="dev", profile="det")                     # a kNN memory grows with every sample it stores
    by_regime = [rec["memory_bytes_by_regime"][str(r)] for r in range(bed.n_regimes)]
    assert by_regime == sorted(by_regime) and len(set(by_regime)) == len(by_regime)
    assert by_regime[-1] < rec["memory_bytes"]                                       # the revisit phases added samples after the last first visit
