"""In-memory mutation testing for m1.verdict.

Applies >= 20 single-edit mutants (flipping comparisons, changing thresholds, swapping R1/R2)
to m1.verdict source code entirely in memory and verifies that unit tests in
tests.m1.test_verdict catch and kill every mutant.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import tests.m1.test_verdict as test_module

VERDICT_PATH = Path(__file__).parent / "verdict.py"


@dataclass
class Mutant:
    id: str
    description: str
    target: str
    replacement: str


# 28 single-edit mutants spanning all verdict calculation logic and thresholds
MUTANTS: List[Mutant] = [
    # Rule 6.1 mutants
    Mutant(
        id="MUTANT_01_R61_ACC_FLIP",
        description="Rule 6.1: Flip accuracy margin check (>= to <)",
        target="acc_ok = b_acc >= (pool_acc - acc_margin)",
        replacement="acc_ok = b_acc < (pool_acc - acc_margin)",
    ),
    Mutant(
        id="MUTANT_02_R61_ACC_THRESHOLD",
        description="Rule 6.1: Shift margin threshold from -margin to +margin",
        target="acc_ok = b_acc >= (pool_acc - acc_margin)",
        replacement="acc_ok = b_acc >= (pool_acc + acc_margin)",
    ),
    Mutant(
        id="MUTANT_03_R61_PRIM_UPDATE_COST",
        description="Rule 6.1: Flip primary update wall-clock check (< to >)",
        target='up_cost_ok = m["mean_update_wall_clock"] < pool_up_wall',
        replacement='up_cost_ok = m["mean_update_wall_clock"] > pool_up_wall',
    ),
    Mutant(
        id="MUTANT_04_R61_PRIM_ANSWER_COST",
        description="Rule 6.1: Flip primary answer latency check (< to >)",
        target='ans_cost_ok = m["mean_answer_latency_b1"] < pool_ans_lat',
        replacement='ans_cost_ok = m["mean_answer_latency_b1"] > pool_ans_lat',
    ),
    Mutant(
        id="MUTANT_05_R61_SEC_UPDATE_COST",
        description="Rule 6.1: Flip secondary update MACs check (< to >)",
        target='up_cost_ok = m["mean_update_macs"] < pool_up_macs',
        replacement='up_cost_ok = m["mean_update_macs"] > pool_up_macs',
    ),
    Mutant(
        id="MUTANT_06_R61_SEC_ANSWER_COST",
        description="Rule 6.1: Flip secondary answer MACs check (< to >)",
        target='ans_cost_ok = m["mean_answer_macs_b1"] < pool_ans_macs',
        replacement='ans_cost_ok = m["mean_answer_macs_b1"] > pool_ans_macs',
    ),
    Mutant(
        id="MUTANT_07_R61_ALLOW_A8",
        description="Rule 6.1: Allow A8 into eligible baselines (omit exclusion filter)",
        target='eligible = [arm for arm in eligible if arm != "A8"]',
        replacement="eligible = [arm for arm in eligible]",
    ),
    Mutant(
        id="MUTANT_08_R61_SWAP_R1_R2",
        description="Rule 6.1: Swap R1 and R2 reference pool defaults",
        target='if reference_pool == "A1":',
        replacement='if reference_pool == "A2":',
    ),
    # Rule 6.2 LDA head mutants
    Mutant(
        id="MUTANT_09_R62_LDA_GAIN_FLIP",
        description="Rule 6.2: Flip LDA head paired gain threshold check (>= to <)",
        target='d["paired_diff"]["mean_diff"] >= THRESHOLD_LDA_HEAD_PAIRED_GAIN',
        replacement='d["paired_diff"]["mean_diff"] < THRESHOLD_LDA_HEAD_PAIRED_GAIN',
    ),
    Mutant(
        id="MUTANT_10_R62_LDA_CI_TWO",
        description="Rule 6.2: Require all 3 schedules to have CI > 0 instead of 2",
        target='sum(1 for d in sched_diffs.values() if d["paired_diff"]["ci_lo"] > 0.0) >= 2',
        replacement='sum(1 for d in sched_diffs.values() if d["paired_diff"]["ci_lo"] > 0.0) >= 3',
    ),
    Mutant(
        id="MUTANT_11_R62_LDA_LAT_RISE",
        description="Rule 6.2: Flip LDA head latency rise check (<= 10% to > 10%)",
        target="cond_lat_rise = max_lat_rise <= THRESHOLD_LDA_HEAD_WALL_CLOCK_MAX_RISE",
        replacement="cond_lat_rise = max_lat_rise > THRESHOLD_LDA_HEAD_WALL_CLOCK_MAX_RISE",
    ),
    # Rule 6.2 Growth mutants
    Mutant(
        id="MUTANT_12_R62_GAP_ACC_DROP",
        description="Rule 6.2: Flip growth gap accuracy drop check (>= -0.005 to < -0.005)",
        target='d["paired_diff"]["mean_diff"] >= -THRESHOLD_GAP_GROWTH_MAX_ACC_DROP',
        replacement='d["paired_diff"]["mean_diff"] < -THRESHOLD_GAP_GROWTH_MAX_ACC_DROP',
    ),
    Mutant(
        id="MUTANT_13_R62_GAP_WALL_DROP",
        description="Rule 6.2: Flip growth gap wall-clock drop check (<= -15% to >= -15%)",
        target="cond_wall_drop = wall_drop <= -THRESHOLD_GAP_GROWTH_MIN_WALL_CLOCK_DROP",
        replacement="cond_wall_drop = wall_drop >= -THRESHOLD_GAP_GROWTH_MIN_WALL_CLOCK_DROP",
    ),
    # Rule 6.2 Skip mutants
    Mutant(
        id="MUTANT_14_R62_SKIP_ACC_DROP",
        description="Rule 6.2: Flip skip_known accuracy drop check (>= -0.003 to < -0.003)",
        target='d["paired_diff"]["mean_diff"] >= -THRESHOLD_SKIP_KNOWN_MAX_ACC_DROP',
        replacement='d["paired_diff"]["mean_diff"] < -THRESHOLD_SKIP_KNOWN_MAX_ACC_DROP',
    ),
    Mutant(
        id="MUTANT_15_R62_SKIP_OBS_DROP",
        description="Rule 6.2: Flip skip_known observe drop check (<= -10% to >= -10%)",
        target="if obs_change <= -THRESHOLD_SKIP_KNOWN_MIN_OBS_DROP:",
        replacement="if obs_change >= -THRESHOLD_SKIP_KNOWN_MIN_OBS_DROP:",
    ),
    # Rule 6.2 Chance mutants
    Mutant(
        id="MUTANT_16_R62_CHANCE_GAIN",
        description="Rule 6.2: Flip transfer_chance gain check (>= 0.01 to < 0.01)",
        target='if diff["mean_diff"] >= THRESHOLD_TRANSFER_CHANCE_MIN_GAIN_ONE_BED:',
        replacement='if diff["mean_diff"] < THRESHOLD_TRANSFER_CHANCE_MIN_GAIN_ONE_BED:',
    ),
    Mutant(
        id="MUTANT_17_R62_CHANCE_COST",
        description="Rule 6.2: Flip transfer_chance B2/B3 cost check (< -0.005 to > -0.005)",
        target='if diff["mean_diff"] < -THRESHOLD_TRANSFER_CHANCE_MAX_COST_B2_B3:',
        replacement='if diff["mean_diff"] > -THRESHOLD_TRANSFER_CHANCE_MAX_COST_B2_B3:',
    ),
    # Predictions M1-1 to M1-9 mutants
    Mutant(
        id="MUTANT_18_M1_1_LAT_RATIO",
        description="M1-1: Flip latency ratio check (>= 1.0 to < 1.0)",
        target="cond_lat = lat_ratio >= THRESHOLD_M1_1_LATENCY_RATIO_MIN",
        replacement="cond_lat = lat_ratio < THRESHOLD_M1_1_LATENCY_RATIO_MIN",
    ),
    Mutant(
        id="MUTANT_19_M1_1_MACS_RATIO",
        description="M1-1: Flip counted MACs ratio check (<= 0.25 to > 0.25)",
        target="cond_macs = macs_ratio <= THRESHOLD_M1_1_MACS_RATIO_MAX",
        replacement="cond_macs = macs_ratio > THRESHOLD_M1_1_MACS_RATIO_MAX",
    ),
    Mutant(
        id="MUTANT_20_M1_2_RATIO_FLIP",
        description="M1-2: Flip batch latency ratio comparison (< to >)",
        target="passed = ratio_b32 < ratio_b1",
        replacement="passed = ratio_b32 > ratio_b1",
    ),
    Mutant(
        id="MUTANT_21_M1_3_RATIO_N30",
        description="M1-3: Flip wall-clock ratio check on N=30 (<= 0.5 to > 0.5)",
        target="cond_30 = ratio_30 <= THRESHOLD_M1_3_TRAIN_RATIO_MAX",
        replacement="cond_30 = ratio_30 > THRESHOLD_M1_3_TRAIN_RATIO_MAX",
    ),
    Mutant(
        id="MUTANT_22_M1_4_SHARE_B2",
        description="M1-4: Flip observe time share check (> 0.60 to < 0.60)",
        target="cond_b2 = share_b2 > THRESHOLD_M1_4_SHARE_MIN",
        replacement="cond_b2 = share_b2 < THRESHOLD_M1_4_SHARE_MIN",
    ),
    Mutant(
        id="MUTANT_23_M1_5_BETTER",
        description="M1-5: Flip better of kNN/LDA accuracy check (>= pool to < pool)",
        target="passed = best_baseline >= pool_acc",
        replacement="passed = best_baseline < pool_acc",
    ),
    Mutant(
        id="MUTANT_24_M1_6_MATCHES",
        description="M1-6: Require 3 schedules within 0.02 instead of 2",
        target="passed = n_within >= 2",
        replacement="passed = n_within >= 3",
    ),
    Mutant(
        id="MUTANT_25_M1_7_B1_NOT",
        description="M1-7: Flip condition that B1 must NOT fire",
        target="cond_b1_not = not b1_fired",
        replacement="cond_b1_not = b1_fired",
    ),
    Mutant(
        id="MUTANT_26_M1_8_R2",
        description="M1-8: Flip R^2 linearity check (>= 0.95 to < 0.95)",
        target="cond_r2 = r_squared >= THRESHOLD_M1_8_R_SQUARED_MIN",
        replacement="cond_r2 = r_squared < THRESHOLD_M1_8_R_SQUARED_MIN",
    ),
    Mutant(
        id="MUTANT_27_M1_8_SLOPE",
        description="M1-8: Flip slope relative error check (<= 0.25 to > 0.25)",
        target="cond_slope = slope_rel_err <= THRESHOLD_M1_8_SLOPE_REL_ERR_MAX",
        replacement="cond_slope = slope_rel_err > THRESHOLD_M1_8_SLOPE_REL_ERR_MAX",
    ),
    Mutant(
        id="MUTANT_29_M1_6_ONE_SIDED",
        description="M1-6: Accept streaming LDA far above the LDA-head pool (one-sided distance)",
        target="within = diff <= THRESHOLD_M1_6_LDA_GAP_MAX",
        replacement="within = acc_lda >= acc_pool - THRESHOLD_M1_6_LDA_GAP_MAX",
    ),
    Mutant(
        id="MUTANT_30_PAIR_USES_A1_A2",
        description="Rule 6.2 pairs: also accept A1 and A2 cells as pair cells",
        target='and r.get("arm") == f"H_{candidate_name}"',
        replacement='and r.get("arm") in ("A1", "A2", f"H_{candidate_name}")',
    ),
    Mutant(
        id="MUTANT_31_LDA_LAT_MEAN",
        description="Rule 6.2 LDA head: judge the latency rise on the mean over schedules instead of the worst one",
        target="cond_lat_rise = max_lat_rise <= THRESHOLD_LDA_HEAD_WALL_CLOCK_MAX_RISE",
        replacement="cond_lat_rise = mean_lat_rise <= THRESHOLD_LDA_HEAD_WALL_CLOCK_MAX_RISE",
    ),
    Mutant(
        id="MUTANT_32_M1_5_KNN_AVERAGED",
        description="M1-5: take the last kNN arm instead of the best baseline",
        target="best_arm = max(baseline_acc, key=lambda a: baseline_acc[a])",
        replacement="best_arm = list(baseline_acc)[0]",
    ),
    Mutant(
        id="MUTANT_33_M1_4_MIXES_A1",
        description="M1-4: use A1 records as the pool",
        target='share_b2, share_b3 = _calc_share(_recs("A2", True)), _calc_share(_recs("A2", False))',
        replacement='share_b2, share_b3 = _calc_share(_recs("A1", True)), _calc_share(_recs("A1", False))',
    ),
    Mutant(
        id="MUTANT_28_M1_9_CHANCE",
        description="M1-9: Invert expected transfer_chance outcome (expect True)",
        target="and chance_adopted is False",
        replacement="and chance_adopted is True",
    ),
]


def run_mutation_tests() -> Tuple[int, int, List[str]]:
    """Apply all mutants in memory and run test suite against each.

    Returns:
        (n_killed, n_survived, list_of_survivor_ids)
    """
    with open(VERDICT_PATH, "r", encoding="utf-8") as f:
        original_source = f.read()

    # Discover test functions in test_module
    test_funcs = [
        getattr(test_module, name)
        for name in dir(test_module)
        if name.startswith("test_") and callable(getattr(test_module, name))
    ]

    killed = 0
    survivors: List[str] = []

    print(f"[MUTATE] Starting mutation analysis with {len(MUTANTS)} mutants...")
    print(f"[MUTATE] Discovered {len(test_funcs)} unit test functions in {test_module.__name__}\n")

    orig_verdict_mod = sys.modules.get("m1.verdict")

    for m in MUTANTS:
        if m.target not in original_source:
            raise ValueError(f"Mutant {m.id} target string not found in {VERDICT_PATH}:\n{m.target!r}")

        # Apply single edit in memory
        mutated_source = original_source.replace(m.target, m.replacement, 1)

        # Create temporary module in memory
        mut_mod = types.ModuleType("m1.verdict")
        mut_mod.__file__ = str(VERDICT_PATH)
        code = compile(mutated_source, str(VERDICT_PATH), "exec")
        exec(code, mut_mod.__dict__)

        # Point sys.modules and test_module functions to mutated module
        sys.modules["m1.verdict"] = mut_mod
        for attr_name, attr_val in mut_mod.__dict__.items():
            if hasattr(test_module, attr_name):
                setattr(test_module, attr_name, attr_val)

        # Run test suite
        killed_by: Optional[str] = None
        for tf in test_funcs:
            try:
                sig = inspect.signature(tf)
                if len(sig.parameters) == 0:
                    tf()
                else:
                    import tempfile
                    with tempfile.TemporaryDirectory() as td:
                        kwargs = {}
                        if "tmp_path" in sig.parameters:
                            kwargs["tmp_path"] = Path(td)
                        if "capsys" in sig.parameters:
                            class DummyCapsys:
                                def readouterr(self):
                                    return types.SimpleNamespace(out="", err="")
                            kwargs["capsys"] = DummyCapsys()
                        tf(**kwargs)
            except (AssertionError, Exception) as exc:
                killed_by = f"{tf.__name__} ({type(exc).__name__})"
                break

        if killed_by:
            killed += 1
            print(f"  [KILLED]   {m.id:<28} | Killed by: {killed_by}")
        else:
            survivors.append(m.id)
            print(f"  [SURVIVED] {m.id:<28} | {m.description}")

    # Restore original module and namespace
    if orig_verdict_mod is not None:
        sys.modules["m1.verdict"] = orig_verdict_mod
        for attr_name, attr_val in orig_verdict_mod.__dict__.items():
            if hasattr(test_module, attr_name):
                setattr(test_module, attr_name, attr_val)

    print("\n" + "=" * 70)
    print(f"[MUTATE] Results: {killed}/{len(MUTANTS)} killed ({killed/len(MUTANTS)*100:.1f}%), {len(survivors)} survivors")
    print("=" * 70)

    return killed, len(survivors), survivors


def main() -> None:
    killed, n_survivors, survivors = run_mutation_tests()
    if n_survivors > 0:
        print(f"[MUTATE] FAILED: {n_survivors} mutants survived: {survivors}")
        sys.exit(1)
    else:
        print("[MUTATE] SUCCESS: All mutants killed. Verdict test suite is robust.")
        sys.exit(0)


if __name__ == "__main__":
    main()
