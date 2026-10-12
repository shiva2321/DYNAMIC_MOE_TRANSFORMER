import math

import numpy as np
import pytest
import torch

from growmoe.metrics import (accuracy_stats, auroc, false_novelty_rate, mean_ci, novel_detection_rate, paired_diff,
                             regime_owner, routing_purity, spawn_stats)
from growmoe.telemetry import EventLog


# ----------------------------------------------------------------------------------------------- AUROC
def test_auroc_perfect_reversed_and_chance():
    labels = [0, 0, 0, 1, 1, 1]
    assert auroc([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], labels) == 1.0
    assert auroc([0.9, 0.8, 0.7, 0.3, 0.2, 0.1], labels) == 0.0
    assert auroc([0.5] * 6, labels) == 0.5                                       # all ties


def test_auroc_textbook_example_with_a_mixed_ranking():
    assert auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75


def test_auroc_counts_ties_as_half():
    # positives {0.5, 0.9}, negatives {0.5, 0.1}: pairs (0.5,0.5)=0.5, (0.5,0.1)=1, (0.9,0.5)=1, (0.9,0.1)=1 -> 3.5/4
    assert auroc([0.5, 0.5, 0.9, 0.1], [1, 0, 1, 0]) == pytest.approx(0.875)


def test_auroc_accepts_tensors_and_rejects_a_single_class():
    assert auroc(torch.tensor([0.2, 0.9]), torch.tensor([0, 1])) == 1.0
    with pytest.raises(ValueError):
        auroc([0.1, 0.2], [1, 1])


# ------------------------------------------------------------------------------------ novelty rates
def test_false_novelty_rate_and_detection_rate_on_a_hand_example():
    novel = np.array([1, 0, 1, 0, 0, 0], dtype=bool)
    known = np.array([1, 1, 0, 1, 1, 0], dtype=bool)                             # truly novel: indices 2 and 5
    assert false_novelty_rate(novel, known) == pytest.approx(0.25)               # 1 of the 4 known samples was flagged
    assert novel_detection_rate(novel, known) == pytest.approx(0.5)              # 1 of the 2 novel samples was flagged


def test_novelty_rates_need_samples_of_the_relevant_kind():
    with pytest.raises(ValueError):
        false_novelty_rate(np.array([True]), np.array([False]))
    with pytest.raises(ValueError):
        novel_detection_rate(np.array([True]), np.array([True]))


# ---------------------------------------------------------------------------------------- ownership
def test_regime_owner_is_the_plurality_regime_and_ignores_unassigned_samples():
    assigned = [0, 0, 0, 1, 1, 2, -1, -1]
    regimes = [5, 5, 6, 6, 6, 7, 7, 7]
    assert regime_owner(assigned, regimes) == {0: 5, 1: 6, 2: 7}


def test_regime_owner_breaks_ties_towards_the_smaller_regime_id():
    assert regime_owner([0, 0, 0, 0], [9, 3, 9, 3]) == {0: 3}


def test_routing_purity_is_cluster_purity_over_assigned_samples():
    assigned = [0, 0, 0, 1, 1, 2, -1, -1]
    regimes = [5, 5, 6, 6, 6, 7, 7, 7]
    assert routing_purity(assigned, regimes) == pytest.approx(5 / 6)             # (2 + 2 + 1) of 6 assigned
    assert routing_purity([0, 0, 1, 1], [1, 1, 2, 2]) == 1.0
    assert math.isnan(routing_purity([-1, -1], [0, 1]))


# ----------------------------------------------------------------------------------------- spawn stats
def _log(*spawns):
    log = EventLog()
    for step, expert in spawns:
        log.add("spawn", step, expert=expert)
    log.add("grow", 5, expert=0)                                                   # other kinds must be ignored
    return log


def test_spawn_stats_on_a_constructed_event_list():
    onsets = {0: 0, 1: 100, 2: 200}
    owner_of = {0: 0, 1: 1, 2: 1, 3: 2}                                            # expert 2 duplicates regime 1; expert 4 owns nothing
    events = _log((0, 0), (130, 1), (150, 2), (260, 3), (300, 4))
    s = spawn_stats(events, owner_of, onsets)
    assert s["n_spawns"] == 5 and s["n_true_positive"] == 3
    assert s["precision"] == pytest.approx(3 / 5)
    assert s["recall"] == pytest.approx(1.0)
    assert s["ratio"] == pytest.approx(5 / 3)
    assert s["time_to_spawn"] == pytest.approx((0 + 30 + 60) / 3)
    assert s["time_to_spawn_median"] == pytest.approx(30)


def test_spawn_stats_missed_regimes_lower_recall_and_premature_spawns_do_not_count():
    onsets = {0: 0, 1: 100, 2: 200}
    owner_of = {0: 0, 1: 1}
    events = _log((0, 0), (50, 1))                                                  # expert 1 appeared before regime 1 began
    s = spawn_stats(events, owner_of, onsets)
    assert s["n_true_positive"] == 1 and s["precision"] == pytest.approx(0.5) and s["recall"] == pytest.approx(1 / 3)


def test_spawn_stats_with_no_spawns():
    s = spawn_stats(_log(), {}, {0: 0, 1: 10})
    assert s["n_spawns"] == 0 and s["recall"] == 0.0 and s["ratio"] == 0.0 and math.isnan(s["precision"])
    assert math.isnan(s["time_to_spawn"])


def test_spawn_stats_orders_events_by_step_not_by_log_order():
    onsets = {0: 0}
    owner_of = {7: 0, 8: 0}
    log = EventLog()
    log.add("spawn", 40, expert=8)                                                  # logged first but later in time
    log.add("spawn", 10, expert=7)
    s = spawn_stats(log, owner_of, onsets)
    assert s["n_true_positive"] == 1 and s["time_to_spawn"] == pytest.approx(10)


# ------------------------------------------------------------------------------------- accuracy matrix
R = np.array([[0.9, 0.1, 0.1],
              [0.7, 0.8, 0.1],
              [0.5, 0.6, 0.9]])                       # row i: after learning regime i; column j: accuracy on regime j


def test_accuracy_stats_on_a_hand_built_matrix():
    s = accuracy_stats(R, chance=0.1)
    assert s["avg"] == pytest.approx((0.5 + 0.6 + 0.9) / 3)
    assert s["forgetting"] == pytest.approx(((0.9 - 0.5) + (0.8 - 0.6)) / 2)       # best earlier accuracy minus final
    assert s["backward"] == pytest.approx(((0.5 - 0.9) + (0.6 - 0.8)) / 2)         # final minus right-after-learning
    assert s["forward"] == pytest.approx(((0.1 - 0.1) + (0.1 - 0.1)) / 2)          # accuracy before training minus chance


def test_accuracy_stats_forward_transfer_rewards_zero_shot_accuracy():
    r = np.array([[0.9, 0.5], [0.8, 0.9]])
    assert accuracy_stats(r, chance=0.1)["forward"] == pytest.approx(0.4)


def test_accuracy_stats_no_forgetting_when_nothing_drops():
    r = np.array([[0.9, 0.0], [0.9, 0.8]])
    s = accuracy_stats(r)
    assert s["forgetting"] == 0.0 and s["backward"] == 0.0


def test_accuracy_stats_supports_revisit_rows_via_learned_at():
    # 4 evaluation rows for 2 regimes: regime 0 learned after row 0, regime 1 after row 1, then two revisit rows
    r = np.array([[0.9, 0.1], [0.6, 0.8], [0.7, 0.8], [0.85, 0.75]])
    s = accuracy_stats(r, learned_at=[0, 1])
    assert s["avg"] == pytest.approx((0.85 + 0.75) / 2)
    assert s["forgetting"] == pytest.approx(((0.9 - 0.85) + (0.8 - 0.75)) / 2)     # best accuracy since learning, minus final
    assert s["backward"] == pytest.approx(((0.85 - 0.9) + (0.75 - 0.8)) / 2)


def test_accuracy_stats_rejects_malformed_input():
    with pytest.raises(ValueError):
        accuracy_stats(np.zeros((3,)))
    with pytest.raises(ValueError):
        accuracy_stats(R, learned_at=[0, 1])


# ------------------------------------------------------------------------------------- seeds / CI / paired
def test_mean_ci_matches_the_t_interval_by_hand():
    m, lo, hi = mean_ci([1, 2, 3, 4, 5])
    half = 2.7764451 * math.sqrt(2.5) / math.sqrt(5)                                 # t(4, .975) * sd / sqrt(n)
    assert m == 3.0 and lo == pytest.approx(3 - half, abs=1e-6) and hi == pytest.approx(3 + half, abs=1e-6)
    assert mean_ci([2.0]) == (2.0, float("-inf"), float("inf"))                      # a single run says nothing about spread


def test_paired_diff_by_hand():
    a, b = [1, 2, 3, 4, 5], [1, 1, 2, 2, 3]                                           # differences 0, 1, 1, 2, 2
    r = paired_diff(a, b)
    se = math.sqrt(0.7) / math.sqrt(5)                                                # sd of differences = sqrt(0.7)
    assert r["n"] == 5 and r["mean_diff"] == pytest.approx(1.2)
    assert r["ci_lo"] == pytest.approx(1.2 - 2.7764451 * se, abs=1e-6) and r["ci_hi"] == pytest.approx(1.2 + 2.7764451 * se, abs=1e-6)
    assert r["p_t"] == pytest.approx(0.0328, abs=2e-3)                                # t = 1.2 / se = 3.207, df = 4, two-sided


def test_paired_diff_identical_samples_are_not_significant():
    r = paired_diff([1, 2, 3], [1, 2, 3])
    assert r["mean_diff"] == 0.0 and r["p_t"] == 1.0 and r["p_wilcoxon"] == 1.0


def test_paired_diff_requires_equal_lengths():
    with pytest.raises(ValueError):
        paired_diff([1, 2, 3], [1, 2])
