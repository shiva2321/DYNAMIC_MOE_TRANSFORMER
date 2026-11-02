"""Pure verdict calculation functions for M1 efficiency benchmark.

Implements Pre-registration Section 6.1, Section 6.2, and Section 7 (Predictions M1-1 to M1-9).
Thresholds and constants are imported exclusively from m1.config_m1.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from growmoe.metrics import mean_ci, paired_diff
from m1.config_m1 import (
    HELD_DEFAULT_PAIRS,
    THRESHOLD_GAP_GROWTH_MAX_ACC_DROP,
    THRESHOLD_GAP_GROWTH_MIN_WALL_CLOCK_DROP,
    THRESHOLD_LDA_HEAD_PAIRED_GAIN,
    THRESHOLD_LDA_HEAD_WALL_CLOCK_MAX_RISE,
    THRESHOLD_M1_1_LATENCY_RATIO_MIN,
    THRESHOLD_M1_1_MACS_RATIO_MAX,
    THRESHOLD_M1_3_TRAIN_RATIO_MAX,
    THRESHOLD_M1_4_SHARE_MIN,
    THRESHOLD_M1_6_LDA_GAP_MAX,
    THRESHOLD_M1_8_R_SQUARED_MIN,
    THRESHOLD_M1_8_SLOPE_REL_ERR_MAX,
    THRESHOLD_RULE_6_1_ACC_MARGIN,
    THRESHOLD_SKIP_KNOWN_MAX_ACC_DROP,
    THRESHOLD_SKIP_KNOWN_MIN_OBS_DROP,
    THRESHOLD_TRANSFER_CHANCE_MAX_COST_B2_B3,
    THRESHOLD_TRANSFER_CHANCE_MIN_GAIN_ONE_BED,
)

# -----------------------------------------------------------------------------
# Record Canonicalization & Grouping
# -----------------------------------------------------------------------------


def bed_key(bed: str, n_regimes: Optional[int] = None) -> str:
    """Canonical bed key identifier string (e.g. 'B1_n10', 'B2', 'B3seq')."""
    if bed == "B1" and n_regimes is not None:
        return f"B1_n{n_regimes}"
    return str(bed)


def record_bed_key(rec: Dict[str, Any]) -> str:
    """Extract canonical bed key from a cell record."""
    b = rec.get("bed", "")
    n = rec.get("n_regimes")
    return bed_key(b, n)


def group_records(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Group cell records by bed key and arm name."""
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for rec in records:
        b_key = record_bed_key(rec)
        arm = rec.get("arm", "")
        if b_key not in grouped:
            grouped[b_key] = {}
        if arm not in grouped[b_key]:
            grouped[b_key][arm] = []
        grouped[b_key][arm].append(rec)
    return grouped


def compute_arm_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics over seeds for an arm on a bed."""
    if not records:
        return {}
    accs = [float(r["accuracy"]) for r in records if "accuracy" in r]
    mean_acc = float(np.mean(accs)) if accs else 0.0

    up_walls = [
        float(r["update_cost"]["wall_clock_total"])
        for r in records
        if "update_cost" in r and "wall_clock_total" in r["update_cost"]
    ]
    mean_up_wall = float(np.mean(up_walls)) if up_walls else 0.0

    up_macs = [
        float(r["update_cost"]["counted_macs"])
        for r in records
        if "update_cost" in r and r["update_cost"].get("counted_macs") is not None
    ]
    mean_up_macs = float(np.mean(up_macs)) if up_macs else 0.0

    ans_lats_b1 = []
    ans_macs_b1 = []
    for r in records:
        ans = r.get("answer_cost", {})
        if "1" in ans:
            if "latency_median_s" in ans["1"]:
                ans_lats_b1.append(float(ans["1"]["latency_median_s"]))
            if "counted_macs" in ans["1"] and ans["1"]["counted_macs"] is not None:
                ans_macs_b1.append(float(ans["1"]["counted_macs"]))
    mean_ans_lat_b1 = float(np.mean(ans_lats_b1)) if ans_lats_b1 else 0.0
    mean_ans_macs_b1 = float(np.mean(ans_macs_b1)) if ans_macs_b1 else 0.0

    ans_lats_b32 = []
    for r in records:
        ans = r.get("answer_cost", {})
        if "32" in ans and "latency_median_s" in ans["32"]:
            ans_lats_b32.append(float(ans["32"]["latency_median_s"]))
    mean_ans_lat_b32 = float(np.mean(ans_lats_b32)) if ans_lats_b32 else 0.0

    mem_bytes = [int(r["memory_bytes"]) for r in records if "memory_bytes" in r]
    mean_mem_bytes = float(np.mean(mem_bytes)) if mem_bytes else 0.0

    return {
        "n_records": len(records),
        "mean_accuracy": mean_acc,
        "mean_update_wall_clock": mean_up_wall,
        "mean_update_macs": mean_up_macs,
        "mean_answer_latency_b1": mean_ans_lat_b1,
        "mean_answer_latency_b32": mean_ans_lat_b32,
        "mean_answer_macs_b1": mean_ans_macs_b1,
        "mean_memory_bytes": mean_mem_bytes,
    }


# -----------------------------------------------------------------------------
# Section 6.1 Decision Rule
# -----------------------------------------------------------------------------


def evaluate_rule_6_1(
    records: Sequence[Dict[str, Any]],
    bed: str,
    reference_pool: str = "A1",
    reading: str = "primary",
    acc_margin: float = THRESHOLD_RULE_6_1_ACC_MARGIN,
    eligible_baselines: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Evaluate Pre-registration Rule 6.1 for a bed, reference pool, and reading.

    Rule 6.1: Fires if any eligible baseline (never A8) is within acc_margin (0.01)
    of the pool's accuracy at lower update cost AND lower answer cost.
    Readings:
      - 'primary': update wall-clock total and answer latency at batch 1.
      - 'secondary': update counted MACs and answer counted MACs at batch 1.
    """
    if reading not in ("primary", "secondary"):
        raise ValueError(f"Unknown reading {reading!r}; must be 'primary' or 'secondary'.")

    # Filter records for bed
    bed_recs = [r for r in records if record_bed_key(r) == bed or r.get("bed") == bed]
    if not bed_recs:
        return {
            "bed": bed,
            "reference_pool": reference_pool,
            "reading": reading,
            "fired": False,
            "status": "NOT APPLICABLE",
            "reason": f"No records found for bed {bed}",
            "triggering_baselines": [],
            "pool_metrics": {},
            "baseline_metrics": {},
        }

    pool_recs = [r for r in bed_recs if r.get("arm") == reference_pool]
    if not pool_recs:
        return {
            "bed": bed,
            "reference_pool": reference_pool,
            "reading": reading,
            "fired": False,
            "status": "NOT APPLICABLE",
            "reason": f"No records found for reference pool {reference_pool} on {bed}",
            "triggering_baselines": [],
            "pool_metrics": {},
            "baseline_metrics": {},
        }

    pool_metrics = compute_arm_summary(pool_recs)
    pool_acc = pool_metrics["mean_accuracy"]
    pool_up_wall = pool_metrics["mean_update_wall_clock"]
    pool_up_macs = pool_metrics["mean_update_macs"]
    pool_ans_lat = pool_metrics["mean_answer_latency_b1"]
    pool_ans_macs = pool_metrics["mean_answer_macs_b1"]

    if eligible_baselines is None:
        if reference_pool == "A1":
            eligible = ["A3", "A4", "A5a", "A5b", "A6", "A7", "A7all"]
        else:
            eligible = ["A3-2", "A4-2", "A5a", "A5b", "A6", "A7-2", "A7all"]
    else:
        eligible = list(eligible_baselines)

    # A8 is JointOracle (offline ceiling), never an eligible baseline method
    eligible = [arm for arm in eligible if arm != "A8"]

    triggering: List[str] = []
    baseline_metrics: Dict[str, Any] = {}

    for arm in eligible:
        arm_recs = [r for r in bed_recs if r.get("arm") == arm]
        if not arm_recs:
            continue
        m = compute_arm_summary(arm_recs)
        baseline_metrics[arm] = m

        b_acc = m["mean_accuracy"]
        # Within margin: pool_acc - b_acc <= acc_margin  <=>  b_acc >= pool_acc - acc_margin
        acc_ok = b_acc >= (pool_acc - acc_margin)

        if reading == "primary":
            up_cost_ok = m["mean_update_wall_clock"] < pool_up_wall
            ans_cost_ok = m["mean_answer_latency_b1"] < pool_ans_lat
        else:
            up_cost_ok = m["mean_update_macs"] < pool_up_macs
            ans_cost_ok = m["mean_answer_macs_b1"] < pool_ans_macs

        if acc_ok and up_cost_ok and ans_cost_ok:
            triggering.append(arm)

    fired = len(triggering) > 0
    return {
        "bed": bed,
        "reference_pool": reference_pool,
        "reading": reading,
        "fired": fired,
        "status": "PASS" if fired else "FAIL",
        "triggering_baselines": triggering,
        "pool_metrics": pool_metrics,
        "baseline_metrics": baseline_metrics,
    }


def evaluate_rule_6_1_all(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Evaluate Rule 6.1 across all beds found in records for R1 and R2, primary and secondary."""
    grouped = group_records(records)
    beds = sorted(grouped.keys())
    results: Dict[str, Dict[str, Any]] = {}
    for bed in beds:
        results[bed] = {
            "R1_primary": evaluate_rule_6_1(records, bed, reference_pool="A1", reading="primary"),
            "R1_secondary": evaluate_rule_6_1(records, bed, reference_pool="A1", reading="secondary"),
            "R2_primary": evaluate_rule_6_1(records, bed, reference_pool="A2", reading="primary"),
            "R2_secondary": evaluate_rule_6_1(records, bed, reference_pool="A2", reading="secondary"),
        }
    return results


# -----------------------------------------------------------------------------
# Section 6.2 Held Defaults Adoption Rules
# -----------------------------------------------------------------------------


def _find_candidate_records(
    records: Sequence[Dict[str, Any]], bed: str, candidate_name: str
) -> List[Dict[str, Any]]:
    """Cells of a held-default pair on a bed: the dedicated cells with arm id ``H_<candidate>`` (built at the base operating point of A1q).

    A1 and A2 cells are never used: A1 runs at the as-built q and A2 at its own tuned point, so pairing them would compare different operating points.
    """
    return [
        r for r in records
        if (record_bed_key(r) == bed or r.get("bed") == bed) and r.get("arm") == f"H_{candidate_name}"
    ]


def evaluate_rule_6_2_lda_head(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate adoption rule for LDA head (B3 only; U-real vs ULDA).

    Rule: Becomes default if paired difference to U-real on B3 is at least +0.01 in all
    three schedules (B3seq, B3mix, B3drift), 95% CI is above zero in at least two, and
    answer wall-clock rises by at most 10%.
    """
    ref_name, cand_name = HELD_DEFAULT_PAIRS["lda_head"]
    schedules = ("B3seq", "B3mix", "B3drift")
    sched_diffs: Dict[str, Any] = {}
    lat_ratios: List[float] = []

    for s in schedules:
        ref_recs = _find_candidate_records(records, s, ref_name)
        cand_recs = _find_candidate_records(records, s, cand_name)
        if not ref_recs or not cand_recs:
            return {
                "adopted": False,
                "status": "NOT APPLICABLE",
                "reason": f"Missing records for {ref_name} or {cand_name} on {s}",
                "schedules": {},
            }
        ref_by_seed = {r["seed"]: r for r in ref_recs if "seed" in r}
        cand_by_seed = {r["seed"]: r for r in cand_recs if "seed" in r}
        common = sorted(set(ref_by_seed.keys()) & set(cand_by_seed.keys()))
        if not common:
            return {
                "adopted": False,
                "status": "NOT APPLICABLE",
                "reason": f"No common seeds for {s}",
                "schedules": {},
            }

        acc_cand = [float(cand_by_seed[seed]["accuracy"]) for seed in common]
        acc_ref = [float(ref_by_seed[seed]["accuracy"]) for seed in common]
        diff = paired_diff(acc_cand, acc_ref)

        lat_cand = [
            float(cand_by_seed[seed]["answer_cost"]["1"]["latency_median_s"])
            for seed in common
            if "answer_cost" in cand_by_seed[seed] and "1" in cand_by_seed[seed]["answer_cost"]
        ]
        lat_ref = [
            float(ref_by_seed[seed]["answer_cost"]["1"]["latency_median_s"])
            for seed in common
            if "answer_cost" in ref_by_seed[seed] and "1" in ref_by_seed[seed]["answer_cost"]
        ]
        mean_lat_cand = float(np.mean(lat_cand)) if lat_cand else 0.0
        mean_lat_ref = float(np.mean(lat_ref)) if lat_ref else 0.0
        lat_ratio = (mean_lat_cand - mean_lat_ref) / mean_lat_ref if mean_lat_ref > 0 else 0.0
        lat_ratios.append(lat_ratio)

        sched_diffs[s] = {
            "paired_diff": diff,
            "lat_ratio_change": lat_ratio,
        }

    cond_gain_all = all(
        d["paired_diff"]["mean_diff"] >= THRESHOLD_LDA_HEAD_PAIRED_GAIN for d in sched_diffs.values()
    )
    cond_ci_two = (
        sum(1 for d in sched_diffs.values() if d["paired_diff"]["ci_lo"] > 0.0) >= 2
    )
    mean_lat_rise = float(np.mean(lat_ratios)) if lat_ratios else 0.0
    max_lat_rise = float(np.max(lat_ratios)) if lat_ratios else 0.0              # the rule is about answer time in each schedule: the worst one counts
    cond_lat_rise = max_lat_rise <= THRESHOLD_LDA_HEAD_WALL_CLOCK_MAX_RISE

    adopted = cond_gain_all and cond_ci_two and cond_lat_rise
    return {
        "adopted": adopted,
        "status": "PASS" if adopted else "FAIL",
        "cond_gain_all": cond_gain_all,
        "cond_ci_two": cond_ci_two,
        "cond_lat_rise": cond_lat_rise,
        "mean_lat_rise": mean_lat_rise,
        "max_lat_rise": max_lat_rise,
        "schedules": sched_diffs,
    }


def evaluate_rule_6_2_growth(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate adoption rule for fit gap growth (shipped vs shipped+gap).

    Rule: Default if accuracy is not below probe by more than 0.5 points (0.005)
    on any bed and whole-run wall-clock is at least 15% lower (-0.15).
    """
    ref_name, cand_name = HELD_DEFAULT_PAIRS["growth"]
    grouped = group_records(records)
    beds = sorted(grouped.keys())
    if not beds:
        return {"adopted": False, "status": "NOT APPLICABLE", "reason": "No beds found"}

    bed_evals: Dict[str, Any] = {}
    wall_cand_all: List[float] = []
    wall_ref_all: List[float] = []

    for b in beds:
        ref_recs = _find_candidate_records(records, b, ref_name)
        cand_recs = _find_candidate_records(records, b, cand_name)
        if not ref_recs or not cand_recs:
            continue
        ref_by_seed = {r["seed"]: r for r in ref_recs if "seed" in r}
        cand_by_seed = {r["seed"]: r for r in cand_recs if "seed" in r}
        common = sorted(set(ref_by_seed.keys()) & set(cand_by_seed.keys()))
        if not common:
            continue

        acc_cand = [float(cand_by_seed[seed]["accuracy"]) for seed in common]
        acc_ref = [float(ref_by_seed[seed]["accuracy"]) for seed in common]
        diff = paired_diff(acc_cand, acc_ref)

        w_cand = [float(cand_by_seed[seed]["update_cost"]["wall_clock_total"]) for seed in common]
        w_ref = [float(ref_by_seed[seed]["update_cost"]["wall_clock_total"]) for seed in common]
        wall_cand_all.extend(w_cand)
        wall_ref_all.extend(w_ref)

        bed_evals[b] = {
            "paired_diff": diff,
            "mean_wall_cand": float(np.mean(w_cand)),
            "mean_wall_ref": float(np.mean(w_ref)),
        }

    if not bed_evals:
        return {"adopted": False, "status": "NOT APPLICABLE", "reason": f"No common runs for {ref_name} vs {cand_name}"}

    # Accuracy not below probe by more than 0.5 points on any bed: mean_diff >= -0.005
    cond_acc_not_drop = all(
        d["paired_diff"]["mean_diff"] >= -THRESHOLD_GAP_GROWTH_MAX_ACC_DROP
        for d in bed_evals.values()
    )

    tot_cand_wall = sum(wall_cand_all)
    tot_ref_wall = sum(wall_ref_all)
    wall_drop = (tot_cand_wall - tot_ref_wall) / tot_ref_wall if tot_ref_wall > 0 else 0.0
    # Whole-run wall-clock is at least 15% lower: drop <= -0.15
    cond_wall_drop = wall_drop <= -THRESHOLD_GAP_GROWTH_MIN_WALL_CLOCK_DROP

    adopted = cond_acc_not_drop and cond_wall_drop
    return {
        "adopted": adopted,
        "status": "PASS" if adopted else "FAIL",
        "cond_acc_not_drop": cond_acc_not_drop,
        "cond_wall_drop": cond_wall_drop,
        "wall_relative_change": wall_drop,
        "beds": bed_evals,
    }


def evaluate_rule_6_2_skip(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate adoption rule for skip_known (shipped vs shipped+skip).

    Rule: Default if accuracy stays within 0.3 points (0.003) on every bed
    and observe wall-clock falls by at least 10% (-0.10) on B2 or B3.
    """
    ref_name, cand_name = HELD_DEFAULT_PAIRS["skip"]
    grouped = group_records(records)
    beds = sorted(grouped.keys())
    if not beds:
        return {"adopted": False, "status": "NOT APPLICABLE", "reason": "No beds found"}

    bed_evals: Dict[str, Any] = {}
    obs_drop_b2_or_b3 = False

    for b in beds:
        ref_recs = _find_candidate_records(records, b, ref_name)
        cand_recs = _find_candidate_records(records, b, cand_name)
        if not ref_recs or not cand_recs:
            continue
        ref_by_seed = {r["seed"]: r for r in ref_recs if "seed" in r}
        cand_by_seed = {r["seed"]: r for r in cand_recs if "seed" in r}
        common = sorted(set(ref_by_seed.keys()) & set(cand_by_seed.keys()))
        if not common:
            continue

        acc_cand = [float(cand_by_seed[seed]["accuracy"]) for seed in common]
        acc_ref = [float(ref_by_seed[seed]["accuracy"]) for seed in common]
        diff = paired_diff(acc_cand, acc_ref)

        obs_cand = [float(cand_by_seed[seed]["update_cost"]["wall_clock_total"]) for seed in common]
        obs_ref = [float(ref_by_seed[seed]["update_cost"]["wall_clock_total"]) for seed in common]
        mean_obs_cand = float(np.mean(obs_cand)) if obs_cand else 0.0
        mean_obs_ref = float(np.mean(obs_ref)) if obs_ref else 0.0
        obs_change = (mean_obs_cand - mean_obs_ref) / mean_obs_ref if mean_obs_ref > 0 else 0.0

        if b.startswith("B2") or b.startswith("B3"):
            if obs_change <= -THRESHOLD_SKIP_KNOWN_MIN_OBS_DROP:
                obs_drop_b2_or_b3 = True

        bed_evals[b] = {
            "paired_diff": diff,
            "obs_change": obs_change,
        }

    if not bed_evals:
        return {"adopted": False, "status": "NOT APPLICABLE", "reason": f"No common runs for {ref_name} vs {cand_name}"}

    # Accuracy stays within 0.3 points on every bed: mean_diff >= -0.003
    cond_acc_ok = all(
        d["paired_diff"]["mean_diff"] >= -THRESHOLD_SKIP_KNOWN_MAX_ACC_DROP
        for d in bed_evals.values()
    )

    adopted = cond_acc_ok and obs_drop_b2_or_b3
    return {
        "adopted": adopted,
        "status": "PASS" if adopted else "FAIL",
        "cond_acc_ok": cond_acc_ok,
        "obs_drop_b2_or_b3": obs_drop_b2_or_b3,
        "beds": bed_evals,
    }


def evaluate_rule_6_2_chance(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate adoption rule for transfer_chance (shipped vs shipped+chance).

    Rule: Default if it gains at least 1 point (+0.01) on one bed and costs
    no more than 0.5 points (-0.005) on B2 and B3.
    """
    ref_name, cand_name = HELD_DEFAULT_PAIRS["chance"]
    grouped = group_records(records)
    beds = sorted(grouped.keys())
    if not beds:
        return {"adopted": False, "status": "NOT APPLICABLE", "reason": "No beds found"}

    bed_evals: Dict[str, Any] = {}
    gain_one_bed = False
    cost_ok_b2_b3 = True

    for b in beds:
        ref_recs = _find_candidate_records(records, b, ref_name)
        cand_recs = _find_candidate_records(records, b, cand_name)
        if not ref_recs or not cand_recs:
            continue
        ref_by_seed = {r["seed"]: r for r in ref_recs if "seed" in r}
        cand_by_seed = {r["seed"]: r for r in cand_recs if "seed" in r}
        common = sorted(set(ref_by_seed.keys()) & set(cand_by_seed.keys()))
        if not common:
            continue

        acc_cand = [float(cand_by_seed[seed]["accuracy"]) for seed in common]
        acc_ref = [float(ref_by_seed[seed]["accuracy"]) for seed in common]
        diff = paired_diff(acc_cand, acc_ref)

        if diff["mean_diff"] >= THRESHOLD_TRANSFER_CHANCE_MIN_GAIN_ONE_BED:
            gain_one_bed = True

        if b.startswith("B2") or b.startswith("B3"):
            if diff["mean_diff"] < -THRESHOLD_TRANSFER_CHANCE_MAX_COST_B2_B3:
                cost_ok_b2_b3 = False

        bed_evals[b] = {
            "paired_diff": diff,
        }

    if not bed_evals:
        return {"adopted": False, "status": "NOT APPLICABLE", "reason": f"No common runs for {ref_name} vs {cand_name}"}

    adopted = gain_one_bed and cost_ok_b2_b3
    return {
        "adopted": adopted,
        "status": "PASS" if adopted else "FAIL",
        "gain_one_bed": gain_one_bed,
        "cost_ok_b2_b3": cost_ok_b2_b3,
        "beds": bed_evals,
    }


def evaluate_rule_6_2_all(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate all four held-default adoption rules."""
    return {
        "lda_head": evaluate_rule_6_2_lda_head(records),
        "growth": evaluate_rule_6_2_growth(records),
        "skip": evaluate_rule_6_2_skip(records),
        "chance": evaluate_rule_6_2_chance(records),
    }


# -----------------------------------------------------------------------------
# Section 7 Predictions M1-1 to M1-9
# -----------------------------------------------------------------------------


def evaluate_prediction_m1_1(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """M1-1: On B1 N=100, batch 1 CPU: pool median latency >= 1.0 * replay, counted MACs <= 0.25 * replay.

    Pool is A2 (A1 printed beside it); best replay is A3-2.
    """
    b100_recs = [r for r in records if record_bed_key(r) == "B1_n100"]
    a2_recs = [r for r in b100_recs if r.get("arm") == "A2"]
    a3_2_recs = [r for r in b100_recs if r.get("arm") == "A3-2"]
    a1_recs = [r for r in b100_recs if r.get("arm") == "A1"]

    if not a2_recs or not a3_2_recs:
        return {
            "prediction": "M1-1",
            "status": "NOT APPLICABLE",
            "reason": "Missing A2 or A3-2 records on B1 N=100",
            "numbers": {},
        }

    m_a2 = compute_arm_summary(a2_recs)
    m_a3_2 = compute_arm_summary(a3_2_recs)
    m_a1 = compute_arm_summary(a1_recs) if a1_recs else {}

    lat_pool = m_a2["mean_answer_latency_b1"]
    lat_replay = m_a3_2["mean_answer_latency_b1"]
    lat_ratio = lat_pool / lat_replay if lat_replay > 0 else 0.0

    macs_pool = m_a2["mean_answer_macs_b1"]
    macs_replay = m_a3_2["mean_answer_macs_b1"]
    macs_ratio = macs_pool / macs_replay if macs_replay > 0 else 0.0

    cond_lat = lat_ratio >= THRESHOLD_M1_1_LATENCY_RATIO_MIN
    cond_macs = macs_ratio <= THRESHOLD_M1_1_MACS_RATIO_MAX
    passed = cond_lat and cond_macs

    return {
        "prediction": "M1-1",
        "status": "PASS" if passed else "FAIL",
        "cond_lat": cond_lat,
        "cond_macs": cond_macs,
        "numbers": {
            "pool_latency_s": lat_pool,
            "replay_latency_s": lat_replay,
            "latency_ratio": lat_ratio,
            "pool_macs": macs_pool,
            "replay_macs": macs_replay,
            "macs_ratio": macs_ratio,
            "a1_latency_s": m_a1.get("mean_answer_latency_b1"),
            "a1_macs": m_a1.get("mean_answer_macs_b1"),
        },
    }


def evaluate_prediction_m1_2(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """M1-2: On B1 N=100, pool-to-replay latency ratio at batch 32 is lower than at batch 1."""
    b100_recs = [r for r in records if record_bed_key(r) == "B1_n100"]
    a2_recs = [r for r in b100_recs if r.get("arm") == "A2"]
    a3_2_recs = [r for r in b100_recs if r.get("arm") == "A3-2"]
    a1_recs = [r for r in b100_recs if r.get("arm") == "A1"]

    if not a2_recs or not a3_2_recs:
        return {
            "prediction": "M1-2",
            "status": "NOT APPLICABLE",
            "reason": "Missing A2 or A3-2 records on B1 N=100",
            "numbers": {},
        }

    m_a2 = compute_arm_summary(a2_recs)
    m_a3_2 = compute_arm_summary(a3_2_recs)
    m_a1 = compute_arm_summary(a1_recs) if a1_recs else {}

    lat_b1_pool = m_a2["mean_answer_latency_b1"]
    lat_b1_replay = m_a3_2["mean_answer_latency_b1"]
    ratio_b1 = lat_b1_pool / lat_b1_replay if lat_b1_replay > 0 else 0.0

    lat_b32_pool = m_a2["mean_answer_latency_b32"]
    lat_b32_replay = m_a3_2["mean_answer_latency_b32"]
    ratio_b32 = lat_b32_pool / lat_b32_replay if lat_b32_replay > 0 else 0.0

    passed = ratio_b32 < ratio_b1

    return {
        "prediction": "M1-2",
        "status": "PASS" if passed else "FAIL",
        "numbers": {
            "ratio_batch_1": ratio_b1,
            "ratio_batch_32": ratio_b32,
            "a1_latency_b1": m_a1.get("mean_answer_latency_b1"),
            "a1_latency_b32": m_a1.get("mean_answer_latency_b32"),
        },
    }


def evaluate_prediction_m1_3(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """M1-3: Whole-stream training wall-clock of pool <= 0.5 * best same-memory replay on B1 N=30 and N=100."""
    b30_recs = [r for r in records if record_bed_key(r) == "B1_n30"]
    b100_recs = [r for r in records if record_bed_key(r) == "B1_n100"]

    a2_30 = [r for r in b30_recs if r.get("arm") == "A2"]
    a3_2_30 = [r for r in b30_recs if r.get("arm") == "A3-2"]
    a2_100 = [r for r in b100_recs if r.get("arm") == "A2"]
    a3_2_100 = [r for r in b100_recs if r.get("arm") == "A3-2"]

    if not a2_30 or not a3_2_30 or not a2_100 or not a3_2_100:
        return {
            "prediction": "M1-3",
            "status": "NOT APPLICABLE",
            "reason": "Missing A2 or A3-2 records on B1 N=30 or N=100",
            "numbers": {},
        }

    m_a2_30 = compute_arm_summary(a2_30)
    m_a3_2_30 = compute_arm_summary(a3_2_30)
    m_a2_100 = compute_arm_summary(a2_100)
    m_a3_2_100 = compute_arm_summary(a3_2_100)

    ratio_30 = (
        m_a2_30["mean_update_wall_clock"] / m_a3_2_30["mean_update_wall_clock"]
        if m_a3_2_30["mean_update_wall_clock"] > 0
        else 0.0
    )
    ratio_100 = (
        m_a2_100["mean_update_wall_clock"] / m_a3_2_100["mean_update_wall_clock"]
        if m_a3_2_100["mean_update_wall_clock"] > 0
        else 0.0
    )

    cond_30 = ratio_30 <= THRESHOLD_M1_3_TRAIN_RATIO_MAX
    cond_100 = ratio_100 <= THRESHOLD_M1_3_TRAIN_RATIO_MAX
    passed = cond_30 and cond_100

    return {
        "prediction": "M1-3",
        "status": "PASS" if passed else "FAIL",
        "cond_n30": cond_30,
        "cond_n100": cond_100,
        "numbers": {
            "ratio_n30": ratio_30,
            "ratio_n100": ratio_100,
            "pool_wall_n30": m_a2_30["mean_update_wall_clock"],
            "replay_wall_n30": m_a3_2_30["mean_update_wall_clock"],
            "pool_wall_n100": m_a2_100["mean_update_wall_clock"],
            "replay_wall_n100": m_a3_2_100["mean_update_wall_clock"],
        },
    }


def evaluate_prediction_m1_4(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """M1-4: Gate, router and bookkeeping take more than THRESHOLD_M1_4_SHARE_MIN of the pool's observe wall-clock on B2 and B3.

    share = (observe time - train_expert time + router refit time) / (observe time + router refit time); the pool is A2, A1 is printed beside it.
    """
    def _calc_share(recs: Sequence[Dict[str, Any]]) -> Optional[float]:
        shares = []
        for r in recs:
            split = r.get("pool_time_split")
            if not split:
                continue
            obs = float(split.get("observe", 0.0))
            train = float(split.get("train_expert", 0.0))
            refit = float(split.get("router_refit", 0.0))
            denom = obs + refit
            if denom > 0:
                shares.append((obs - train + refit) / denom)
        return float(np.mean(shares)) if shares else None

    def _recs(arm: str, b2: bool) -> List[Dict[str, Any]]:
        return [r for r in records if r.get("arm") == arm and ((r.get("bed") == "B2") if b2 else str(r.get("bed", "")).startswith("B3"))]

    share_b2, share_b3 = _calc_share(_recs("A2", True)), _calc_share(_recs("A2", False))
    a1_b2, a1_b3 = _calc_share(_recs("A1", True)), _calc_share(_recs("A1", False))

    if share_b2 is None or share_b3 is None:
        return {
            "prediction": "M1-4",
            "status": "NOT APPLICABLE",
            "reason": "Missing pool_time_split on A2 records of B2 or B3",
            "numbers": {"share_b2": share_b2, "share_b3": share_b3, "a1_share_b2": a1_b2, "a1_share_b3": a1_b3},
        }

    cond_b2 = share_b2 > THRESHOLD_M1_4_SHARE_MIN
    cond_b3 = share_b3 > THRESHOLD_M1_4_SHARE_MIN
    passed = cond_b2 and cond_b3

    return {
        "prediction": "M1-4",
        "status": "PASS" if passed else "FAIL",
        "cond_b2": cond_b2,
        "cond_b3": cond_b3,
        "numbers": {"share_b2": share_b2, "share_b3": share_b3, "a1_share_b2": a1_b2, "a1_share_b3": a1_b3},
    }


def evaluate_prediction_m1_5(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """M1-5: On B2 the better of kNN (each kNN arm on its own) and streaming LDA reaches at least the pool's final accuracy; the pool is A2 (A1 if there is no A2)."""
    b2_recs = [r for r in records if r.get("bed") == "B2"]
    pool_arm = "A2" if any(r.get("arm") == "A2" for r in b2_recs) else "A1"
    pool_recs = [r for r in b2_recs if r.get("arm") == pool_arm]
    baseline_acc: Dict[str, float] = {}
    for arm in ("A7-2", "A7", "A7all", "A5a"):
        arm_recs = [r for r in b2_recs if r.get("arm") == arm]
        if arm_recs:
            baseline_acc[arm] = compute_arm_summary(arm_recs)["mean_accuracy"]

    if not pool_recs or not baseline_acc:
        return {
            "prediction": "M1-5",
            "status": "NOT APPLICABLE",
            "reason": "Missing B2 pool or baseline records",
            "numbers": {},
        }

    pool_acc = compute_arm_summary(pool_recs)["mean_accuracy"]
    best_arm = max(baseline_acc, key=lambda a: baseline_acc[a])
    best_baseline = baseline_acc[best_arm]

    passed = best_baseline >= pool_acc

    return {
        "prediction": "M1-5",
        "status": "PASS" if passed else "FAIL",
        "numbers": {
            "pool_arm": pool_arm,
            "pool_acc": pool_acc,
            "baseline_acc": baseline_acc,
            "best_baseline_arm": best_arm,
            "best_baseline": best_baseline,
        },
    }


def evaluate_prediction_m1_6(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """M1-6: On B3, streaming LDA (A5a) is within THRESHOLD_M1_6_LDA_GAP_MAX (either side) of the LDA-head pool (cell H_ULDA) in at least 2 of 3 schedules."""
    schedules = ("B3seq", "B3mix", "B3drift")
    sched_checks: Dict[str, Any] = {}

    for s in schedules:
        s_recs = [r for r in records if r.get("bed") == s]
        lda_recs = [r for r in s_recs if r.get("arm") == "A5a"]
        pool_recs = _find_candidate_records(s_recs, s, "ULDA")

        if not lda_recs or not pool_recs:
            continue

        acc_lda = compute_arm_summary(lda_recs)["mean_accuracy"]
        acc_pool = compute_arm_summary(pool_recs)["mean_accuracy"]
        diff = abs(acc_lda - acc_pool)
        within = diff <= THRESHOLD_M1_6_LDA_GAP_MAX
        sched_checks[s] = {
            "acc_lda": acc_lda,
            "acc_pool": acc_pool,
            "diff": diff,
            "within_02": within,
        }

    if len(sched_checks) < 2:
        return {
            "prediction": "M1-6",
            "status": "NOT APPLICABLE",
            "reason": "Missing B3 schedule records for A5a or for the LDA-head pool (H_ULDA)",
            "numbers": sched_checks,
        }

    n_within = sum(1 for v in sched_checks.values() if v["within_02"])
    passed = n_within >= 2

    return {
        "prediction": "M1-6",
        "status": "PASS" if passed else "FAIL",
        "matches": n_within,
        "numbers": sched_checks,
    }


def evaluate_prediction_m1_7(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """M1-7: Rule 6.1 fires on B2 and B3 and does NOT fire on B1."""
    all_6_1 = evaluate_rule_6_1_all(records)
    if not all_6_1:
        return {
            "prediction": "M1-7",
            "status": "NOT APPLICABLE",
            "reason": "No bed evaluations available for Rule 6.1",
            "numbers": {},
        }

    # B1 beds: check whether any fired (using primary reading of R2 or R1)
    b1_keys = [k for k in all_6_1 if k.startswith("B1")]
    b2_keys = [k for k in all_6_1 if k == "B2"]
    b3_keys = [k for k in all_6_1 if k.startswith("B3")]

    if not b1_keys or not b2_keys or not b3_keys:
        return {
            "prediction": "M1-7",
            "status": "NOT APPLICABLE",
            "reason": "Missing beds among B1, B2, B3 for Rule 6.1",
            "numbers": {},
        }

    b1_fired = any(all_6_1[k]["R2_primary"]["fired"] for k in b1_keys)
    b2_fired = any(all_6_1[k]["R2_primary"]["fired"] for k in b2_keys)
    b3_fired = any(all_6_1[k]["R2_primary"]["fired"] for k in b3_keys)

    cond_b1_not = not b1_fired
    passed = cond_b1_not and b2_fired and b3_fired

    return {
        "prediction": "M1-7",
        "status": "PASS" if passed else "FAIL",
        "cond_b1_not_fired": cond_b1_not,
        "cond_b2_fired": b2_fired,
        "cond_b3_fired": b3_fired,
        "numbers": {
            "b1_fired": b1_fired,
            "b2_fired": b2_fired,
            "b3_fired": b3_fired,
        },
    }


def evaluate_prediction_m1_8(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """M1-8: Pool memory grows linearly with regimes on B1 (R^2 >= 0.95, slope within 25% of final bytes per expert)."""
    # Look for B1 records with memory_bytes_by_regime, prefer N=100 or N=30
    b1_recs = [
        r
        for r in records
        if str(r.get("bed", "")).startswith("B1")
        and r.get("arm") in ("A2", "A1")
        and "memory_bytes_by_regime" in r
        and r["memory_bytes_by_regime"]
    ]
    if not b1_recs:
        return {
            "prediction": "M1-8",
            "status": "NOT APPLICABLE",
            "reason": "No B1 pool records with memory_bytes_by_regime found",
            "numbers": {},
        }

    # Pick the record with the most regime entries
    pool_choice = [r for r in b1_recs if r.get("arm") == "A2"] or b1_recs               # A2 is the pool of the prediction; A1 only if there is no A2
    rec = max(pool_choice, key=lambda r: len(r["memory_bytes_by_regime"]))
    mem_dict = rec["memory_bytes_by_regime"]
    regimes = sorted([int(k) for k in mem_dict.keys()])
    if len(regimes) < 3:
        return {
            "prediction": "M1-8",
            "status": "NOT APPLICABLE",
            "reason": f"Too few regimes ({len(regimes)}) for linear regression",
            "numbers": {},
        }

    x = np.array([r + 1 for r in regimes], dtype=np.float64)
    y = np.array([mem_dict[str(r)] for r in regimes], dtype=np.float64)

    res = stats.linregress(x, y)
    r_squared = float(res.rvalue ** 2)
    slope = float(res.slope)

    final_bytes = float(y[-1])
    n_experts = float(rec.get("describe", {}).get("experts", len(regimes)))
    bytes_per_expert = final_bytes / n_experts if n_experts > 0 else final_bytes

    slope_rel_err = abs(slope - bytes_per_expert) / bytes_per_expert if bytes_per_expert > 0 else 0.0

    cond_r2 = r_squared >= THRESHOLD_M1_8_R_SQUARED_MIN
    cond_slope = slope_rel_err <= THRESHOLD_M1_8_SLOPE_REL_ERR_MAX
    passed = cond_r2 and cond_slope

    return {
        "prediction": "M1-8",
        "status": "PASS" if passed else "FAIL",
        "cond_r2": cond_r2,
        "cond_slope": cond_slope,
        "numbers": {
            "r_squared": r_squared,
            "slope": slope,
            "bytes_per_expert": bytes_per_expert,
            "slope_rel_error": slope_rel_err,
        },
    }


def evaluate_prediction_m1_9(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """M1-9: Rules 6.2 adopt LDA head and growth_rule='gap', but not skip_known and not transfer_chance."""
    r62 = evaluate_rule_6_2_all(records)

    # Check applicability
    for k in ("lda_head", "growth", "skip", "chance"):
        if r62[k].get("status") == "NOT APPLICABLE":
            return {
                "prediction": "M1-9",
                "status": "NOT APPLICABLE",
                "reason": f"Rule 6.2 for {k} is NOT APPLICABLE",
                "numbers": r62,
            }

    lda_adopted = r62["lda_head"]["adopted"]
    growth_adopted = r62["growth"]["adopted"]
    skip_adopted = r62["skip"]["adopted"]
    chance_adopted = r62["chance"]["adopted"]

    passed = (
        lda_adopted is True
        and growth_adopted is True
        and skip_adopted is False
        and chance_adopted is False
    )

    return {
        "prediction": "M1-9",
        "status": "PASS" if passed else "FAIL",
        "numbers": {
            "lda_adopted": lda_adopted,
            "growth_adopted": growth_adopted,
            "skip_adopted": skip_adopted,
            "chance_adopted": chance_adopted,
        },
    }


def evaluate_all_predictions(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Evaluate all predictions M1-1 through M1-9."""
    return {
        "M1-1": evaluate_prediction_m1_1(records),
        "M1-2": evaluate_prediction_m1_2(records),
        "M1-3": evaluate_prediction_m1_3(records),
        "M1-4": evaluate_prediction_m1_4(records),
        "M1-5": evaluate_prediction_m1_5(records),
        "M1-6": evaluate_prediction_m1_6(records),
        "M1-7": evaluate_prediction_m1_7(records),
        "M1-8": evaluate_prediction_m1_8(records),
        "M1-9": evaluate_prediction_m1_9(records),
    }
