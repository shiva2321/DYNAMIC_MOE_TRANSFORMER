"""Unit tests for M1 verdict calculations (Rule 6.1, Rule 6.2, Predictions M1-1 to M1-9).

Tests boundary cases at thresholds, tie-breaking, R1/R2 reference pools, and synthetic tables.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np
import pytest

from m1.cell import SCHEMA_VERSION
from m1.config_m1 import (
    THRESHOLD_GAP_GROWTH_MAX_ACC_DROP,
    THRESHOLD_GAP_GROWTH_MIN_WALL_CLOCK_DROP,
    THRESHOLD_LDA_HEAD_PAIRED_GAIN,
    THRESHOLD_LDA_HEAD_WALL_CLOCK_MAX_RISE,
    THRESHOLD_RULE_6_1_ACC_MARGIN,
    THRESHOLD_SKIP_KNOWN_MAX_ACC_DROP,
    THRESHOLD_SKIP_KNOWN_MIN_OBS_DROP,
    THRESHOLD_TRANSFER_CHANCE_MAX_COST_B2_B3,
    THRESHOLD_TRANSFER_CHANCE_MIN_GAIN_ONE_BED,
)
from m1.verdict import (
    compute_arm_summary,
    evaluate_all_predictions,
    evaluate_prediction_m1_1,
    evaluate_prediction_m1_2,
    evaluate_prediction_m1_3,
    evaluate_prediction_m1_4,
    evaluate_prediction_m1_5,
    evaluate_prediction_m1_6,
    evaluate_prediction_m1_7,
    evaluate_prediction_m1_8,
    evaluate_prediction_m1_9,
    evaluate_rule_6_1,
    evaluate_rule_6_1_all,
    evaluate_rule_6_2_all,
    evaluate_rule_6_2_chance,
    evaluate_rule_6_2_growth,
    evaluate_rule_6_2_lda_head,
    evaluate_rule_6_2_skip,
)


def make_cell(
    bed: str = "B1",
    n_regimes: int = 10,
    arm: str = "A1",
    seed: int = 800,
    acc: float = 0.40,
    forg: float = 0.05,
    up_wall: float = 10.0,
    up_macs: int = 1_000_000,
    lat_b1: float = 0.005,
    lat_b32: float = 0.020,
    ans_macs: int = 50_000,
    mem_bytes: int = 1_000_000,
    split: str = "dev",
    candidate: str | None = None,
    pool_time_split: Dict[str, float] | None = None,
    memory_bytes_by_regime: Dict[str, int] | None = None,
    experts: int = 1,
) -> Dict[str, Any]:
    """Helper to create a structurally valid cell record for unit testing."""
    if memory_bytes_by_regime is None:
        memory_bytes_by_regime = {str(r): (r + 1) * (mem_bytes // n_regimes) for r in range(n_regimes)}

    return {
        "schema_version": SCHEMA_VERSION,
        "bed": bed,
        "n_regimes": n_regimes,
        "arm": arm,
        "seed": seed,
        "split": split,
        "commit": "testcommit",
        "environment": {"python": "3.12"},
        "pinning": {"threads": 2, "cpus": [0, 2]},
        "profile": "full",
        "bed_info": {
            "d": 128,
            "classes": 10,
            "regimes": n_regimes,
            "phases": n_regimes,
            "total_samples": 1000,
            "regime_names": [f"regime_{i}" for i in range(n_regimes)],
        },
        "tuning_ref": "testtuning",
        "reference_pool": None,
        "uncounted_note": False,
        "config": {
            "arm": arm,
            "candidate": candidate,
            "inner_steps": 5,
            "hidden": 512,
        },
        "accuracy": float(acc),
        "per_regime_accuracy": [float(acc)] * n_regimes,
        "forgetting": float(forg),
        "update_cost": {
            "wall_clock_per_observe": [0.01] * 10,
            "wall_clock_per_regime": {str(r): up_wall / n_regimes for r in range(n_regimes)},
            "wall_clock_total": float(up_wall),
            "counted_macs": int(up_macs),
            "peak_memory": 10_000_000,
            "eval_wall_clock_per_regime_boundary": [0.01] * n_regimes,
        },
        "answer_cost": {
            "1": {
                "batch_size": 1,
                "rows_timed": 10,
                "latency_median_s": float(lat_b1),
                "latency_mean_s": float(lat_b1),
                "latency_std_s": 0.0001,
                "latency_cv": 0.02,
                "latency_p95_s": float(lat_b1) * 1.2,
                "items_per_second": 1.0 / lat_b1 if lat_b1 > 0 else 0.0,
                "counted_macs": int(ans_macs),
                "active_bytes": 100_000,
            },
            "32": {
                "batch_size": 32,
                "rows_timed": 32,
                "latency_median_s": float(lat_b32),
                "latency_mean_s": float(lat_b32),
                "latency_std_s": 0.0005,
                "latency_cv": 0.02,
                "latency_p95_s": float(lat_b32) * 1.2,
                "items_per_second": 32.0 / lat_b32 if lat_b32 > 0 else 0.0,
                "counted_macs": int(ans_macs) * 32,
                "active_bytes": 100_000,
            },
        },
        "memory_bytes": int(mem_bytes),
        "memory_bytes_by_regime": memory_bytes_by_regime,
        "pool_time_split": pool_time_split,
        "describe": {"experts": experts},
    }


# -----------------------------------------------------------------------------
# Unit Tests for Rule 6.1
# -----------------------------------------------------------------------------


def test_rule_6_1_firing_and_boundaries() -> None:
    """Test Rule 6.1 triggers when baseline is within 0.01 margin at lower update & answer cost."""
    # Pool A1: acc=0.40, up_wall=10.0, lat_b1=0.005, up_macs=1000, ans_macs=500
    pool = make_cell(bed="B1", n_regimes=10, arm="A1", seed=800, acc=0.40, up_wall=10.0, lat_b1=0.005, up_macs=1000, ans_macs=500)

    # Baseline A3 within margin (0.395 >= 0.390) and cheaper on both costs -> FIRES
    b_pass = make_cell(bed="B1", n_regimes=10, arm="A3", seed=800, acc=0.395, up_wall=8.0, lat_b1=0.004, up_macs=800, ans_macs=400)
    res = evaluate_rule_6_1([pool, b_pass], bed="B1_n10", reference_pool="A1", reading="primary")
    assert res["fired"] is True
    assert "A3" in res["triggering_baselines"]

    # Boundary test: exactly at pool - 0.01 (0.390) -> FIRES
    b_exact = make_cell(bed="B1", n_regimes=10, arm="A3", seed=800, acc=0.390, up_wall=8.0, lat_b1=0.004)
    res = evaluate_rule_6_1([pool, b_exact], bed="B1_n10", reference_pool="A1", reading="primary")
    assert res["fired"] is True

    # Boundary test: just below margin (0.3899) -> DOES NOT FIRE
    b_below = make_cell(bed="B1", n_regimes=10, arm="A3", seed=800, acc=0.3899, up_wall=8.0, lat_b1=0.004)
    res = evaluate_rule_6_1([pool, b_below], bed="B1_n10", reference_pool="A1", reading="primary")
    assert res["fired"] is False
    assert "A3" not in res["triggering_baselines"]

    # Fails when update wall-clock is higher (11.0 > 10.0) -> DOES NOT FIRE
    b_cost1 = make_cell(bed="B1", n_regimes=10, arm="A3", seed=800, acc=0.41, up_wall=11.0, lat_b1=0.004)
    res = evaluate_rule_6_1([pool, b_cost1], bed="B1_n10", reference_pool="A1", reading="primary")
    assert res["fired"] is False

    # Fails when answer latency is higher (0.006 > 0.005) -> DOES NOT FIRE
    b_cost2 = make_cell(bed="B1", n_regimes=10, arm="A3", seed=800, acc=0.41, up_wall=8.0, lat_b1=0.006)
    res = evaluate_rule_6_1([pool, b_cost2], bed="B1_n10", reference_pool="A1", reading="primary")
    assert res["fired"] is False


def test_rule_6_1_secondary_reading() -> None:
    """Test Rule 6.1 secondary reading uses counted MACs."""
    pool = make_cell(bed="B1", n_regimes=10, arm="A1", seed=800, acc=0.40, up_wall=10.0, lat_b1=0.005, up_macs=1000, ans_macs=500)
    # Baseline with lower MACs but higher wall-clock -> Secondary FIRES, Primary DOES NOT
    b = make_cell(bed="B1", n_regimes=10, arm="A4", seed=800, acc=0.40, up_wall=12.0, lat_b1=0.007, up_macs=800, ans_macs=400)

    res_sec = evaluate_rule_6_1([pool, b], bed="B1_n10", reference_pool="A1", reading="secondary")
    assert res_sec["fired"] is True
    assert "A4" in res_sec["triggering_baselines"]

    res_prim = evaluate_rule_6_1([pool, b], bed="B1_n10", reference_pool="A1", reading="primary")
    assert res_prim["fired"] is False


def test_rule_6_1_excludes_a8() -> None:
    """Verify Arm A8 (JointOracle offline ceiling) is NEVER an eligible baseline."""
    pool = make_cell(bed="B1", n_regimes=10, arm="A1", seed=800, acc=0.40, up_wall=10.0, lat_b1=0.005)
    # A8 is much higher accuracy and cheaper -> must NEVER trigger Rule 6.1
    a8 = make_cell(bed="B1", n_regimes=10, arm="A8", seed=800, acc=0.90, up_wall=1.0, lat_b1=0.001)

    res = evaluate_rule_6_1([pool, a8], bed="B1_n10", reference_pool="A1", eligible_baselines=["A8"])
    assert res["fired"] is False
    assert "A8" not in res["triggering_baselines"]


def test_rule_6_1_r2_reference_pool() -> None:
    """Verify R2 evaluates against pool A2 with -2 baselines."""
    pool_a2 = make_cell(bed="B1", n_regimes=10, arm="A2", seed=800, acc=0.42, up_wall=10.0, lat_b1=0.005)
    b_pass = make_cell(bed="B1", n_regimes=10, arm="A3-2", seed=800, acc=0.415, up_wall=8.0, lat_b1=0.004)

    res = evaluate_rule_6_1([pool_a2, b_pass], bed="B1_n10", reference_pool="A2", reading="primary")
    assert res["fired"] is True
    assert "A3-2" in res["triggering_baselines"]


# -----------------------------------------------------------------------------
# Unit Tests for Rule 6.2 Held Defaults
# -----------------------------------------------------------------------------


def test_rule_6_2_lda_head() -> None:
    """Test adoption of LDA head: +0.01 gain in all 3 schedules, 95% CI > 0 in >= 2, lat rise <= 10%."""
    records = []
    # 3 schedules: B3seq (CI > 0), B3mix (CI > 0), B3drift (mean >= 0.01 but CI spans zero)
    # Exactly 2 schedules have CI > 0 -> must ADOPT under >= 2, but would fail under >= 3
    for s in (800, 801, 802, 803, 804):
        # B3seq: constant gain 0.015 -> CI > 0
        records.append(make_cell(bed="B3seq", n_regimes=22, arm="H_U-real", candidate="U-real", seed=s, acc=0.50, lat_b1=0.010))
        records.append(make_cell(bed="B3seq", n_regimes=22, arm="H_ULDA", candidate="ULDA", seed=s, acc=0.515, lat_b1=0.0105))

        # B3mix: constant gain 0.012 -> CI > 0
        records.append(make_cell(bed="B3mix", n_regimes=22, arm="H_U-real", candidate="U-real", seed=s, acc=0.50, lat_b1=0.010))
        records.append(make_cell(bed="B3mix", n_regimes=22, arm="H_ULDA", candidate="ULDA", seed=s, acc=0.512, lat_b1=0.0105))

    # B3drift: gains [0.030, -0.005, 0.020, 0.010, 0.005] -> mean is +0.012 (>= 0.01), but CI spans zero (ci_lo < 0)
    drift_gains = [0.030, -0.005, 0.020, 0.010, 0.005]
    for s, g in zip((800, 801, 802, 803, 804), drift_gains):
        records.append(make_cell(bed="B3drift", n_regimes=22, arm="H_U-real", candidate="U-real", seed=s, acc=0.50, lat_b1=0.010))
        records.append(make_cell(bed="B3drift", n_regimes=22, arm="H_ULDA", candidate="ULDA", seed=s, acc=0.50 + g, lat_b1=0.0105))

    res = evaluate_rule_6_2_lda_head(records)
    assert res["adopted"] is True
    assert res["status"] == "PASS"
    assert res["cond_gain_all"] is True
    assert res["cond_ci_two"] is True

    # Fails if latency rise exceeds 10% (e.g. 0.012 vs 0.010 -> +20%)
    rec_lat_fail = copy.deepcopy(records)
    for r in rec_lat_fail:
        if r["config"]["candidate"] == "ULDA":
            r["answer_cost"]["1"]["latency_median_s"] = 0.0125
    res_fail = evaluate_rule_6_2_lda_head(rec_lat_fail)
    assert res_fail["adopted"] is False

    # Fails if gain is < 0.01 on one schedule
    rec_gain_fail = copy.deepcopy(records)
    for r in rec_gain_fail:
        if r["bed"] == "B3seq" and r["config"]["candidate"] == "ULDA":
            r["accuracy"] = 0.505  # gain is only +0.005 < 0.01
    res_gain = evaluate_rule_6_2_lda_head(rec_gain_fail)
    assert res_gain["adopted"] is False


def test_rule_6_2_growth() -> None:
    """Test adoption of growth gap: acc drop <= 0.005 on all beds, wall clock drop >= 15%."""
    records = []
    for bed in ("B1_n10", "B2", "B3seq"):
        for s in (800, 801):
            records.append(make_cell(bed=bed, n_regimes=10, arm="H_shipped", candidate="shipped", seed=s, acc=0.40, up_wall=100.0))
            # gap: acc drop 0.002 (<= 0.005) and wall clock 80.0 (20% drop >= 15%)
            records.append(make_cell(bed=bed, n_regimes=10, arm="H_shipped+gap", candidate="shipped+gap", seed=s, acc=0.398, up_wall=80.0))

    res = evaluate_rule_6_2_growth(records)
    assert res["adopted"] is True

    # Fails if wall clock drop is only 10% (< 15%)
    rec_wall_fail = copy.deepcopy(records)
    for r in rec_wall_fail:
        if r["config"]["candidate"] == "shipped+gap":
            r["update_cost"]["wall_clock_total"] = 92.0
    res_w = evaluate_rule_6_2_growth(rec_wall_fail)
    assert res_w["adopted"] is False


def test_rule_6_2_skip() -> None:
    """Test adoption of skip_known: acc drop <= 0.003 on all beds, observe drop >= 10% on B2 or B3."""
    records = []
    for bed in ("B1_n10", "B2"):
        for s in (800, 801):
            records.append(make_cell(bed=bed, n_regimes=10, arm="H_shipped", candidate="shipped", seed=s, acc=0.40, up_wall=100.0))
            # on B2, wall is 88.0 (12% drop >= 10%)
            up_w = 88.0 if bed == "B2" else 99.0
            records.append(make_cell(bed=bed, n_regimes=10, arm="H_shipped+skip", candidate="shipped+skip", seed=s, acc=0.399, up_wall=up_w))

    res = evaluate_rule_6_2_skip(records)
    assert res["adopted"] is True

    # Fails if acc drop > 0.003 (e.g. 0.395 vs 0.400 -> drop of 0.005)
    rec_acc_fail = copy.deepcopy(records)
    for r in rec_acc_fail:
        if r["config"]["candidate"] == "shipped+skip":
            r["accuracy"] = 0.395
    res_a = evaluate_rule_6_2_skip(rec_acc_fail)
    assert res_a["adopted"] is False


def test_rule_6_2_chance() -> None:
    """Test adoption of transfer_chance: gains >= 0.01 on 1 bed, cost <= 0.005 on B2 and B3."""
    records = []
    # B1: gains 0.015 (>= 0.01)
    # B2: costs 0.002 (<= 0.005)
    for s in (800, 801):
        records.append(make_cell(bed="B1_n10", arm="H_shipped", candidate="shipped", seed=s, acc=0.40))
        records.append(make_cell(bed="B1_n10", arm="H_shipped+chance", candidate="shipped+chance", seed=s, acc=0.415))

        records.append(make_cell(bed="B2", arm="H_shipped", candidate="shipped", seed=s, acc=0.50))
        records.append(make_cell(bed="B2", arm="H_shipped+chance", candidate="shipped+chance", seed=s, acc=0.498))

    res = evaluate_rule_6_2_chance(records)
    assert res["adopted"] is True

    # Fails if cost on B2 is 0.010 (> 0.005)
    rec_cost_fail = copy.deepcopy(records)
    for r in rec_cost_fail:
        if r["bed"] == "B2" and r["config"]["candidate"] == "shipped+chance":
            r["accuracy"] = 0.485
    res_c = evaluate_rule_6_2_chance(rec_cost_fail)
    assert res_c["adopted"] is False


# -----------------------------------------------------------------------------
# Unit Tests for Predictions M1-1 through M1-9
# -----------------------------------------------------------------------------


def test_prediction_m1_1() -> None:
    """M1-1: B1 N=100 pool latency >= 1.0 * replay, counted MACs <= 0.25 * replay."""
    # A2: lat=0.010, macs=250
    # A3-2: lat=0.008, macs=1000 -> lat_ratio = 1.25 >= 1.0, macs_ratio = 0.25 <= 0.25
    a2 = make_cell(bed="B1", n_regimes=100, arm="A2", lat_b1=0.010, ans_macs=250)
    a3_2 = make_cell(bed="B1", n_regimes=100, arm="A3-2", lat_b1=0.008, ans_macs=1000)

    res = evaluate_prediction_m1_1([a2, a3_2])
    assert res["status"] == "PASS"

    # Fails when macs ratio > 0.25 (e.g. macs=300 -> 0.30)
    a2_fail = make_cell(bed="B1", n_regimes=100, arm="A2", lat_b1=0.010, ans_macs=300)
    res_fail = evaluate_prediction_m1_1([a2_fail, a3_2])
    assert res_fail["status"] == "FAIL"


def test_prediction_m1_2() -> None:
    """M1-2: pool-to-replay latency ratio at batch 32 is lower than at batch 1."""
    # A2: b1=0.010, b32=0.020
    # A3-2: b1=0.010, b32=0.030 -> ratio_b1 = 1.0, ratio_b32 = 0.667 < 1.0
    a2 = make_cell(bed="B1", n_regimes=100, arm="A2", lat_b1=0.010, lat_b32=0.020)
    a3_2 = make_cell(bed="B1", n_regimes=100, arm="A3-2", lat_b1=0.010, lat_b32=0.030)

    res = evaluate_prediction_m1_2([a2, a3_2])
    assert res["status"] == "PASS"

    # Fails when ratio_b32 >= ratio_b1
    a2_fail = make_cell(bed="B1", n_regimes=100, arm="A2", lat_b1=0.010, lat_b32=0.040)
    res_fail = evaluate_prediction_m1_2([a2_fail, a3_2])
    assert res_fail["status"] == "FAIL"


def test_prediction_m1_3() -> None:
    """M1-3: whole-stream wall-clock of pool <= 0.5 * replay on B1 N=30 and N=100."""
    recs = [
        make_cell(bed="B1", n_regimes=30, arm="A2", up_wall=40.0),
        make_cell(bed="B1", n_regimes=30, arm="A3-2", up_wall=100.0),  # ratio 0.4 <= 0.5
        make_cell(bed="B1", n_regimes=100, arm="A2", up_wall=120.0),
        make_cell(bed="B1", n_regimes=100, arm="A3-2", up_wall=300.0), # ratio 0.4 <= 0.5
    ]
    res = evaluate_prediction_m1_3(recs)
    assert res["status"] == "PASS"

    # Fails if ratio on N=30 > 0.5
    recs[0]["update_cost"]["wall_clock_total"] = 60.0
    res_fail = evaluate_prediction_m1_3(recs)
    assert res_fail["status"] == "FAIL"


def test_prediction_m1_4() -> None:
    """M1-4: observe time split share > 60% on B2 and B3."""
    # share = (obs - train + refit) / (obs + refit)
    # obs=10.0, train=2.0, refit=1.0 -> (10 - 2 + 1) / (10 + 1) = 9 / 11 = 81.8% > 60%
    split_pass = {"observe": 10.0, "train_expert": 2.0, "router_refit": 1.0}
    recs = [
        make_cell(bed="B2", arm="A2", pool_time_split=split_pass),
        make_cell(bed="B3seq", arm="A2", pool_time_split=split_pass),
    ]
    res = evaluate_prediction_m1_4(recs)
    assert res["status"] == "PASS"

    # Fails when share <= 0.60 (e.g. train=8.0 -> (10 - 8 + 1)/11 = 3/11 = 27%)
    split_fail = {"observe": 10.0, "train_expert": 8.0, "router_refit": 1.0}
    recs[0]["pool_time_split"] = split_fail
    res_fail = evaluate_prediction_m1_4(recs)
    assert res_fail["status"] == "FAIL"


def test_prediction_m1_5() -> None:
    """M1-5: On B2, better of kNN and LDA reaches at least pool accuracy."""
    recs = [
        make_cell(bed="B2", arm="A2", acc=0.45),
        make_cell(bed="B2", arm="A7-2", acc=0.46),  # kNN is 0.46 >= 0.45
        make_cell(bed="B2", arm="A5a", acc=0.40),
    ]
    res = evaluate_prediction_m1_5(recs)
    assert res["status"] == "PASS"

    # Fails when both are below pool
    recs[1]["accuracy"] = 0.44
    res_fail = evaluate_prediction_m1_5(recs)
    assert res_fail["status"] == "FAIL"


def test_prediction_m1_6() -> None:
    """M1-6: On B3, streaming LDA is within 0.02 of LDA-head pool in >= 2 of 3 schedules."""
    recs = []
    # 2 schedules within 0.02, 1 not
    recs.append(make_cell(bed="B3seq", arm="A5a", acc=0.48))
    recs.append(make_cell(bed="B3seq", arm="H_ULDA", candidate="ULDA", acc=0.49))   # diff 0.01 <= 0.02

    recs.append(make_cell(bed="B3mix", arm="A5a", acc=0.47))
    recs.append(make_cell(bed="B3mix", arm="H_ULDA", candidate="ULDA", acc=0.48))   # diff 0.01 <= 0.02

    recs.append(make_cell(bed="B3drift", arm="A5a", acc=0.40))
    recs.append(make_cell(bed="B3drift", arm="H_ULDA", candidate="ULDA", acc=0.45)) # diff 0.05 > 0.02

    res = evaluate_prediction_m1_6(recs)
    assert res["status"] == "PASS"

    # Fails when fewer than 2 match
    recs[2]["accuracy"] = 0.40  # mix diff becomes 0.08
    res_fail = evaluate_prediction_m1_6(recs)
    assert res_fail["status"] == "FAIL"


def test_prediction_m1_7() -> None:
    """M1-7: Rule 6.1 fires on B2 and B3 and does NOT fire on B1."""
    recs = []
    # B1: baseline is worse -> does not fire
    recs.append(make_cell(bed="B1", n_regimes=10, arm="A2", acc=0.40, up_wall=10.0, lat_b1=0.005))
    recs.append(make_cell(bed="B1", n_regimes=10, arm="A3-2", acc=0.35, up_wall=12.0, lat_b1=0.008))

    # B2: baseline within 0.01 and cheaper -> fires
    recs.append(make_cell(bed="B2", arm="A2", acc=0.40, up_wall=10.0, lat_b1=0.005))
    recs.append(make_cell(bed="B2", arm="A3-2", acc=0.395, up_wall=8.0, lat_b1=0.004))

    # B3: fires
    recs.append(make_cell(bed="B3seq", arm="A2", acc=0.40, up_wall=10.0, lat_b1=0.005))
    recs.append(make_cell(bed="B3seq", arm="A3-2", acc=0.395, up_wall=8.0, lat_b1=0.004))

    res = evaluate_prediction_m1_7(recs)
    assert res["status"] == "PASS"

    # Fails if B1 also fires
    recs[1]["accuracy"] = 0.40
    recs[1]["update_cost"]["wall_clock_total"] = 5.0
    recs[1]["answer_cost"]["1"]["latency_median_s"] = 0.002
    res_fail = evaluate_prediction_m1_7(recs)
    assert res_fail["status"] == "FAIL"


def test_prediction_m1_8() -> None:
    """M1-8: Pool memory grows linearly with regimes on B1 (R^2 >= 0.95, slope within 25% of bytes per expert)."""
    # 10 regimes, strictly linear 100,000 bytes per regime; 10 experts -> slope = 100,000, bytes_per_expert = 100,000
    mem_dict = {str(r): (r + 1) * 100_000 for r in range(10)}
    rec = make_cell(bed="B1", n_regimes=10, arm="A2", mem_bytes=1_000_000, memory_bytes_by_regime=mem_dict, experts=10)

    res = evaluate_prediction_m1_8([rec])
    assert res["status"] == "PASS"
    assert res["numbers"]["r_squared"] >= 0.99
    assert res["numbers"]["slope_rel_error"] <= 0.05

    # Fails when slope error > 25% (e.g. experts=20 -> bytes_per_expert = 50,000, slope error = 100%)
    rec_fail = make_cell(bed="B1", n_regimes=10, arm="A2", mem_bytes=1_000_000, memory_bytes_by_regime=mem_dict, experts=20)
    res_f = evaluate_prediction_m1_8([rec_fail])
    assert res_f["status"] == "FAIL"


def test_prediction_m1_9() -> None:
    """M1-9: Rules 6.2 adopt LDA and gap, but reject skip and chance."""
    records = []
    # Setup matching M1-9 expectations:
    # 1. lda_head -> ADOPT
    for s in ("B3seq", "B3mix", "B3drift"):
        for seed in (800, 801):
            records.append(make_cell(bed=s, arm="H_U-real", candidate="U-real", seed=seed, acc=0.50, lat_b1=0.010))
            records.append(make_cell(bed=s, arm="H_ULDA", candidate="ULDA", seed=seed, acc=0.52, lat_b1=0.0105))

    # 2. growth -> ADOPT (acc drop 0.002 <= 0.005, wall drop 20% >= 15%)
    for seed in (800, 801):
        records.append(make_cell(bed="B1_n10", arm="H_shipped", candidate="shipped", seed=seed, acc=0.40, up_wall=100.0))
        records.append(make_cell(bed="B1_n10", arm="H_shipped+gap", candidate="shipped+gap", seed=seed, acc=0.398, up_wall=80.0))

    # 3. skip -> REJECT (acc drop 0.005 > 0.003)
    for seed in (800, 801):
        records.append(make_cell(bed="B1_n10", arm="H_shipped+skip", candidate="shipped+skip", seed=seed, acc=0.390, up_wall=95.0))

    # 4. chance -> REJECT (no bed gains >= 0.01)
    for seed in (800, 801):
        records.append(make_cell(bed="B1_n10", arm="H_shipped+chance", candidate="shipped+chance", seed=seed, acc=0.402, up_wall=100.0))

    res = evaluate_prediction_m1_9(records)
    assert res["status"] == "PASS"


# -----------------------------------------------------------------------------
# Unit Tests for Report Loading & Pipeline
# -----------------------------------------------------------------------------


def test_schema_1_1_0_skipped_with_message(tmp_path: Any, capsys: Any) -> None:
    """Note (2): Records with schema 1.1.0 are stage-A evidence: skipped with a message, never crash, never counted."""
    from m1.cell import save_cell_record
    from m1.report import load_records

    # Save a valid 1.2.0 record
    rec_120 = make_cell(bed="B1", arm="A1", seed=800)
    save_cell_record(tmp_path / "B1_n10_A1__seed800.json", rec_120)

    # Save a stage-A 1.1.0 record
    rec_110 = copy.deepcopy(rec_120)
    rec_110["schema_version"] = "1.1.0"
    rec_110.pop("bed_info", None)
    rec_110.pop("profile", None)
    save_cell_record(tmp_path / "B1_n10_A3__seed800.json", rec_110)

    recs, split = load_records(tmp_path)
    captured = capsys.readouterr()

    # Verify message printed and 1.1.0 record not counted
    assert "Skipping invalid or other-schema record" in captured.out
    assert "schema_version=1.1.0" in captured.out
    assert len(recs) == 1
    assert recs[0]["schema_version"] == "1.2.0"


def test_report_pipeline_synthetic(tmp_path: Any) -> None:
    """Section 7 item 6: Run full report pipeline on synthetic fixtures."""
    from m1.cell import save_cell_record
    from m1.report import generate_report

    cells_dir = tmp_path / "cells"
    out_dir = tmp_path / "report_out"
    cells_dir.mkdir()

    # Generate synthetic cells for multiple beds and arms
    for b in ("B1", "B2"):
        for a in ("A1", "A2", "A3", "A5a"):
            rec = make_cell(
                bed=b,
                n_regimes=10,
                arm=a,
                seed=800,
                split="synthetic",
                pool_time_split={"observe": 10.0, "train_expert": 2.0, "router_refit": 1.0},
            )
            save_cell_record(cells_dir / f"{b}_n10_{a}__seed800.json", rec)

    verdicts_data = generate_report(cells_dir, out_dir, split="synthetic")

    assert (out_dir / "REPORT_M1.md").exists()
    assert (out_dir / "verdicts.json").exists()
    assert (out_dir / "figures" / "summary_figure.png").exists()

    md_text = (out_dir / "REPORT_M1.md").read_text(encoding="utf-8")
    assert "[SYNTHETIC]" in md_text
    assert "Executive Summary" in md_text
    assert "Scoreboards" in md_text


# -----------------------------------------------------------------------------
# Tests added by the reviewer after job B-3 (defects found in the first delivery)
# -----------------------------------------------------------------------------


def test_pair_cells_are_the_dedicated_h_cells_never_a1_or_a2() -> None:
    """A1 and A2 cells with the same candidate name must not be used as reference or candidate of a held-default pair."""
    records = []
    for s in (800, 801):
        records.append(make_cell(bed="B2", n_regimes=6, arm="H_shipped", candidate="shipped", seed=s, acc=0.40, up_wall=100.0))
        records.append(make_cell(bed="B2", n_regimes=6, arm="H_shipped+gap", candidate="shipped+gap", seed=s, acc=0.40, up_wall=100.0))
        records.append(make_cell(bed="B2", n_regimes=6, arm="A1", candidate="shipped", seed=s, acc=0.10, up_wall=500.0))          # as-built A1: other operating point
        records.append(make_cell(bed="B2", n_regimes=6, arm="A2", candidate="shipped+gap", seed=s, acc=0.60, up_wall=10.0))       # A2 happens to run the same candidate
    res = evaluate_rule_6_2_growth(records)
    assert res["status"] == "FAIL"                    # equal accuracy and equal wall-clock: no 15 % drop; the A1/A2 cells would have made it PASS
    assert res["beds"]["B2"]["mean_wall_ref"] == 100.0 and res["beds"]["B2"]["mean_wall_cand"] == 100.0


def test_lda_latency_rise_must_hold_in_every_schedule() -> None:
    records = []
    for s in (800, 801, 802):
        for bed, rise in (("B3seq", 0.0), ("B3mix", 0.0), ("B3drift", 0.25)):                # mean rise 8.3 % (under 10 %), worst schedule 25 %
            records.append(make_cell(bed=bed, n_regimes=22, arm="H_U-real", candidate="U-real", seed=s, acc=0.50 + 0.001 * s % 7 * 0.01, lat_b1=0.010))
            records.append(make_cell(bed=bed, n_regimes=22, arm="H_ULDA", candidate="ULDA", seed=s, acc=0.52 + 0.001 * s % 7 * 0.01, lat_b1=0.010 * (1 + rise)))
    res = evaluate_rule_6_2_lda_head(records)
    assert res["cond_lat_rise"] is False and res["adopted"] is False
    assert res["mean_lat_rise"] < 0.10 < res["max_lat_rise"]


def test_m1_6_is_two_sided_and_needs_the_lda_head_pool() -> None:
    """Streaming LDA far ABOVE the LDA-head pool is not 'within 0.02'; a missing H_ULDA cell is not replaced by A2."""
    recs = []
    for s_bed, lda, pool in (("B3seq", 0.60, 0.48), ("B3mix", 0.60, 0.48), ("B3drift", 0.48, 0.48)):
        recs.append(make_cell(bed=s_bed, arm="A5a", acc=lda))
        recs.append(make_cell(bed=s_bed, arm="H_ULDA", candidate="ULDA", acc=pool))
    res = evaluate_prediction_m1_6(recs)
    assert res["status"] == "FAIL" and res["matches"] == 1
    no_pool = [make_cell(bed=b, arm="A5a", acc=0.48) for b in ("B3seq", "B3mix", "B3drift")] + [make_cell(bed=b, arm="A2", acc=0.48) for b in ("B3seq", "B3mix", "B3drift")]
    assert evaluate_prediction_m1_6(no_pool)["status"] == "NOT APPLICABLE"


def test_m1_5_uses_each_knn_arm_on_its_own() -> None:
    """The better of the kNN arms counts, not their average; the pool is A2 and A1 is never mixed in."""
    recs = [
        make_cell(bed="B2", arm="A2", acc=0.50),
        make_cell(bed="B2", arm="A1", acc=0.10),
        make_cell(bed="B2", arm="A7-2", acc=0.30),
        make_cell(bed="B2", arm="A7all", acc=0.52),          # average of the two kNN arms is 0.41 < 0.50, the better one is 0.52 >= 0.50
        make_cell(bed="B2", arm="A5a", acc=0.20),
    ]
    res = evaluate_prediction_m1_5(recs)
    assert res["status"] == "PASS" and res["numbers"]["best_baseline_arm"] == "A7all"


def test_m1_4_uses_the_a2_pool_and_prints_a1_beside_it() -> None:
    split_hi = {"observe": 10.0, "train_expert": 2.0, "router_refit": 2.0}      # share (10-2+2)/(12) = 0.83
    split_lo = {"observe": 10.0, "train_expert": 8.0, "router_refit": 0.0}      # share 0.2
    recs = []
    for bed, n in (("B2", 6), ("B3seq", 22)):
        recs.append(make_cell(bed=bed, n_regimes=n, arm="A2", pool_time_split=split_hi))
        recs.append(make_cell(bed=bed, n_regimes=n, arm="A1", pool_time_split=split_lo))
    res = evaluate_prediction_m1_4(recs)
    assert res["status"] == "PASS" and res["numbers"]["a1_share_b2"] < 0.60 < res["numbers"]["share_b2"]
    assert evaluate_prediction_m1_4([r for r in recs if r["arm"] == "A1"])["status"] == "NOT APPLICABLE"
