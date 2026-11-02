"""Unit tests for M1 Stage A runner: adapters, schema, stale-record guard, equal-memory budget and queue."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest
import torch

import m1.queue as queue_mod
from m1.arms import A1PoolArm, build_arm, pool_budget_bytes
from m1.beds import POOL_SEED_OFFSET, build_bed
from m1.cell import (
    cell_filename,
    execute_cell,
    is_valid_cell_record,
    load_cell_record,
    read_pool_budget,
    record_matches_config,
    save_cell_record,
)
from m1.queue import parse_seeds, run_queue


@pytest.fixture(scope="module")
def small_bed():
    return build_bed("B1", n_regimes=2, seed=100, n_train=16, n_test=16)


@pytest.fixture(scope="module")
def a1_record(small_bed):
    """One real A1 record on the small bed, shared by the tests that only read it (tests that change it copy it first)."""
    return execute_cell(small_bed, "A1", split="dev")


def test_pool_seed_offset():
    """Verify pool seed is stream seed + 7 (C-75 rule)."""
    bed = build_bed("B1", n_regimes=2, seed=800, n_train=16, n_test=16)
    assert bed.seed == 800
    assert bed.pool_seed == 800 + POOL_SEED_OFFSET
    assert bed.pool_seed == 807


def test_split_report_refused(small_bed):
    """Verify --split report is strictly refused in stage A."""
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
    """Verify all arms obey the uniform Arm adapter interface and describe themselves with plain-JSON configurations."""
    bed = build_bed("B1", n_regimes=2, seed=100, n_train=64, n_test=16)
    pool_bytes = 100_000

    for arm_id in ("A1", "A1L", "A3", "A7", "A8"):
        arm = build_arm(arm_id, bed, pool_memory_bytes=pool_bytes)

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
        json.dumps(arm.config())                      # plain JSON types only: the record stores it and compares it after a round trip


def test_a1_is_baseline_v1_and_a1l_is_the_legacy_pool(small_bed):
    """A1 must be the pool as Baseline v1 defines it (stage-4 defaults on); A1L the pre-stage-4 pool of the shootout (harness check only)."""
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
        build_arm("A2", small_bed)
    with pytest.raises(ValueError, match="variant"):
        A1PoolArm(small_bed, variant="baseline_v2")


def test_record_schema(a1_record, small_bed):
    """Verify that the cell record contains all required fields and sane values."""
    rec = a1_record
    assert is_valid_cell_record(rec)
    assert rec["schema_version"] == "1.1.0"
    assert (rec["bed"], rec["n_regimes"], rec["arm"], rec["seed"], rec["split"]) == ("B1", 2, "A1", 100, "dev")
    assert isinstance(rec["accuracy"], float)
    assert len(rec["per_regime_accuracy"]) == 2
    assert rec["forgetting"] is None or isinstance(rec["forgetting"], float)
    assert rec["config"] == json.loads(json.dumps(build_arm("A1", small_bed).config()))
    assert "1" in rec["answer_cost"] and "32" in rec["answer_cost"]
    assert rec["memory_bytes"] > 0
    assert len(rec["memory_bytes_by_regime"]) > 0
    assert len(rec["update_cost"]["eval_wall_clock_per_regime_boundary"]) >= 2      # one evaluation per regime boundary plus the final one


def test_batch_rows_are_exactly_the_batch_size(a1_record):
    """The test set has 16 rows; the batch-32 latency must still be timed on 32 rows (the rows repeat)."""
    for b in (1, 32):
        assert a1_record["answer_cost"][str(b)]["batch_size"] == b
        assert a1_record["answer_cost"][str(b)]["rows_timed"] == b


def test_pinning_is_recorded(small_bed):
    pinning = {"threads": 2, "cpus": [0, 2], "blas_limited": True}
    rec = execute_cell(small_bed, "A1", split="dev", pinning=pinning)
    assert rec["pinning"] == pinning


def test_undefined_forgetting_is_none_not_zero(monkeypatch, small_bed, tmp_path):
    """When the metric is undefined (NaN) the record stores null, never an invented 0.0 (a function-regime stream always has two regimes, so the metric is patched)."""
    nan = float("nan")
    monkeypatch.setattr("m1.cell.accuracy_stats", lambda R, learned_at=None, chance=0.0: {"avg": 0.0, "forgetting": nan, "backward": nan, "forward": nan})
    rec = execute_cell(small_bed, "A1", split="dev")
    assert rec["forgetting"] is None
    assert is_valid_cell_record(rec)
    path = tmp_path / "rec.json"
    save_cell_record(path, rec)
    assert load_cell_record(path)["forgetting"] is None


def test_equal_memory_arms_share_the_pool_budget(small_bed, a1_record):
    """A3 and A7 are given the memory the A1 pool has at the end of its stream; the fallback computation must agree with A1's own record."""
    budget = pool_budget_bytes(small_bed)
    assert budget == a1_record["memory_bytes"]

    for arm in ("A3", "A7"):
        rec = execute_cell(small_bed, arm, split="dev")                       # no budget given: computed once, used by the arm and the counting pass
        assert rec["config"]["target_pool_bytes"] == budget
        assert rec["memory_bytes"] > 0

    rec = execute_cell(small_bed, "A3", pool_memory_bytes=budget + 4096, split="dev")
    assert rec["config"]["target_pool_bytes"] == budget + 4096


def test_record_matches_config_detects_stale_records(small_bed, a1_record):
    assert record_matches_config(a1_record, small_bed, "A1")

    stale = copy.deepcopy(a1_record)
    stale["config"]["router"] = False                      # a record written by a pool without the router is not an A1 record
    assert not record_matches_config(stale, small_bed, "A1")

    legacy = copy.deepcopy(a1_record)
    legacy["config"] = json.loads(json.dumps(build_arm("A1L", small_bed).config()))
    assert not record_matches_config(legacy, small_bed, "A1")
    assert record_matches_config(legacy, small_bed, "A1L")

    missing = copy.deepcopy(a1_record)
    del missing["config"]                                  # records of the first runner version carry no configuration: always stale
    assert not record_matches_config(missing, small_bed, "A1")


def test_record_matches_config_needs_the_budget_for_equal_memory_arms(small_bed):
    budget = pool_budget_bytes(small_bed)
    rec = execute_cell(small_bed, "A3", pool_memory_bytes=budget, split="dev")
    assert record_matches_config(rec, small_bed, "A3", budget)
    assert not record_matches_config(rec, small_bed, "A3", budget + 4096)      # another budget is another arm
    assert not record_matches_config(rec, small_bed, "A3", None)               # unknown budget cannot be confirmed


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


def test_atomic_write_no_half_file(tmp_path):
    """Verify atomic write creates complete files and leaves no corrupt or temporary files on failure."""
    out_file = tmp_path / "test_record.json"
    data = {"schema_version": "1.1.0", "accuracy": 0.5}

    save_cell_record(out_file, data)
    assert out_file.exists()
    assert load_cell_record(out_file) == data

    bad_file = tmp_path / "bad_record.json"
    bad_data = {"circular": None}
    bad_data["circular"] = bad_data                          # unserializable circular reference

    with pytest.raises(ValueError):
        save_cell_record(bad_file, bad_data)

    assert not bad_file.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["test_record.json"]       # the temporary file is gone as well


def test_parse_seeds():
    assert parse_seeds("800") == [800]
    assert parse_seeds("800-802,805") == [800, 801, 802, 805]
    assert parse_seeds("802, 800,800") == [800, 802]


def test_queue_skips_current_records_and_reruns_stale_ones(tmp_path, monkeypatch, small_bed, a1_record):
    """Skip a record that is complete, valid and made by the arm as configured now; rerun one that is not; give A3 the A1 record's memory."""
    calls = []

    def fake_execute(bed, arm_name, pool_memory_bytes=None, split="dev", pinning=None, **kwargs):
        calls.append((arm_name, pool_memory_bytes))
        return copy.deepcopy(a1_record)

    monkeypatch.setattr(queue_mod, "execute_cell", fake_execute)
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
    run_queue(arms=["A3", "A1"], **queue)                  # A1 runs first whatever the order given; its record's memory is A3's budget
    assert calls == [("A3", a1_record["memory_bytes"])]


def test_queue_reruns_a_corrupt_record(tmp_path, monkeypatch, small_bed, a1_record):
    calls = []

    def fake_execute(bed, arm_name, pool_memory_bytes=None, split="dev", pinning=None, **kwargs):
        calls.append(arm_name)
        return copy.deepcopy(a1_record)

    monkeypatch.setattr(queue_mod, "execute_cell", fake_execute)
    monkeypatch.setattr(queue_mod, "build_bed", lambda name, n, seed: small_bed)

    target = tmp_path / cell_filename("B1", 2, "A1", 100)
    target.write_text("{ truncated", encoding="utf-8")
    run_queue(split="dev", beds=["B1"], ns=[2], arms=["A1"], seeds=[100], out=tmp_path, cpus=[0], threads=1)
    assert calls == ["A1"]
    assert is_valid_cell_record(load_cell_record(target))


def test_same_seed_gives_same_accuracy(small_bed, a1_record):
    """Verify deterministic accuracy across identical runs."""
    rec2 = execute_cell(build_bed("B1", n_regimes=2, seed=100, n_train=16, n_test=16), "A1", split="dev")
    assert a1_record["accuracy"] == pytest.approx(rec2["accuracy"], abs=1e-12)
    assert a1_record["per_regime_accuracy"] == pytest.approx(rec2["per_regime_accuracy"], abs=1e-12)
    assert a1_record["memory_bytes"] == rec2["memory_bytes"]
