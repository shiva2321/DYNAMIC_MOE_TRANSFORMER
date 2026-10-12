"""M0: the fast evaluation runner and the budget-tuning helper.

``run_fast`` is only acceptable if it reproduces the benchmark's own accuracy matrix exactly; it is faster because it scores all
regimes of a test point in one batched call and skips the per-batch predict that only the novelty metrics need.
"""

import numpy as np
import pytest

from growmoe import streams as S
from growmoe.baselines import OracleRouted, ReplayNet
from growmoe.bench import make_regime_of, run_stream
from growmoe.config import CommonCfg, PhyloCfg
from growmoe.fairfight import run_fast, tune_steps
from growmoe.memory_baselines import ClassMean, KnnMemory
from growmoe.pools.phylo import PhyloMoE


@pytest.fixture(scope="module")
def stream():
    return S.s2_permuted(0, n_regimes=3, schedule=[0, 1, 2, 0], n_train=128, n_test=40)


def _makers(stream):
    c = CommonCfg(seed=0, n_classes=stream.n_classes, inner_steps=2)
    return {
        "replay": lambda: ReplayNet(c),
        "knn": lambda: KnnMemory(c),
        "knn_capped": lambda: KnnMemory(c, capacity=100),
        "class_mean": lambda: ClassMean(c),
        "phylo": lambda: PhyloMoE(PhyloCfg(seed=0, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf")),
        "oracle": lambda: OracleRouted(make_regime_of(stream, 64), c),
    }


@pytest.mark.parametrize("name", ["replay", "knn", "knn_capped", "class_mean", "phylo", "oracle"])
def test_fast_runner_reproduces_the_benchmark_accuracy_matrix_exactly(stream, name):
    make = _makers(stream)[name]
    old = run_stream(make, stream, seed=0)
    new = run_fast(make, stream, seed=0, eval_every="segments")
    assert new.R.shape == old.R.shape
    np.testing.assert_allclose(new.R, old.R, atol=1e-12, rtol=0)
    assert new.acc == pytest.approx(old.accuracy_stats["avg"], abs=1e-12)
    assert new.stats["forgetting"] == pytest.approx(old.accuracy_stats["forgetting"], abs=1e-12)
    assert new.events == list(old.events)                                               # dropping the per-batch predict changed no decision
    assert new.summary.get("n_experts") == old.pool_summary.get("n_experts")


def test_final_only_mode_gives_the_same_final_accuracy_and_skips_the_intermediate_rows(stream):
    make = _makers(stream)["phylo"]
    full = run_fast(make, stream, seed=0, eval_every="segments")
    final = run_fast(make, stream, seed=0, eval_every="final")
    assert final.R.shape[0] == 1 and full.R.shape[0] == len(stream.segments)
    np.testing.assert_allclose(final.R[-1], full.R[-1], atol=1e-12, rtol=0)
    assert final.acc == pytest.approx(full.acc, abs=1e-12)


def test_result_carries_resources_and_a_per_regime_breakdown(stream):
    res = run_fast(_makers(stream)["knn"], stream, seed=0, d=stream.d)
    assert res.per_regime.shape == (len(stream.regimes),)
    assert res.report.stored_vectors == 128 * len(stream.segments)
    assert res.report.memory_bytes == 4 * stream.d * res.report.stored_vectors
    assert res.wall_clock > 0 and res.n_experts == 1


def test_run_is_deterministic_in_the_seed(stream):
    make = _makers(stream)["phylo"]
    a, b = run_fast(make, stream, seed=0), run_fast(make, stream, seed=0)
    np.testing.assert_array_equal(a.per_regime, b.per_regime)
    assert a.events == b.events


# --------------------------------------------------------------------------------------------- tune_steps
def test_tune_steps_picks_the_best_interior_grid_value():
    res = tune_steps(lambda s: -abs(s - 5), grid=(1, 5, 20, 50))
    assert res.best == 5 and not res.at_edge and res.table[5] == 0


def test_tune_steps_extends_the_grid_when_the_optimum_is_on_the_upper_edge():
    calls = []

    def score(s):
        calls.append(s)
        return -(s - 60.0) ** 2                                                         # best near 60, above the grid's top (50)

    res = tune_steps(score, grid=(1, 5, 20, 50), factor=2.5, cap=200)
    assert 125 in calls                                                                 # 50 * 2.5, evaluated because 50 was the edge
    assert res.best == 50 and not res.at_edge                                           # 125 is worse, so 50 is now interior


def test_tune_steps_stops_at_the_cap_and_says_so_when_more_is_always_better():
    res = tune_steps(lambda s: float(s), grid=(1, 5, 20, 50), factor=2.5, cap=200)
    assert res.best == 200 and res.at_edge


def test_tune_steps_prefers_the_cheaper_setting_on_a_tie():
    res = tune_steps(lambda s: 1.0, grid=(1, 5, 20, 50))
    assert res.best == 1


# --------------------------------------------------------------------------------------------- records and matched memory
def test_matched_capacity_gives_a_lookup_the_same_bytes_as_the_pool():
    from growmoe.accounting import PoolReport
    from growmoe.fairfight import matched_capacity
    rep = PoolReport(n_experts=3, params=1000, stored_vectors=500, memory_bytes=4 * 1000 + 4 * 128 * 500, gate_macs=0, expert_macs=0,
                     predict_macs=0)
    assert matched_capacity(rep, d=128) == (4 * 1000 + 4 * 128 * 500) // (4 * 128)
    tiny = PoolReport(n_experts=0, params=0, stored_vectors=0, memory_bytes=100, gate_macs=0, expert_macs=0, predict_macs=0)
    assert matched_capacity(tiny, d=128) == 16                                            # never smaller than 16 vectors


def test_run_record_is_plain_json_and_carries_the_resources(stream):
    import json
    from growmoe.fairfight import run_record
    res = run_fast(_makers(stream)["phylo"], stream, seed=0, eval_every="segments")
    rec = run_record("S2-3", "A", 7, res, steps=20, width=128)
    json.dumps(rec)                                                                       # plain data only
    assert (rec["stream"], rec["method"], rec["seed"], rec["steps"], rec["width"]) == ("S2-3", "A", 7, 20, 128)
    assert rec["acc"] == res.acc and rec["per_regime"] == res.per_regime.tolist()
    assert rec["memory_bytes"] == res.report.memory_bytes and rec["predict_macs"] == res.report.predict_macs
    assert rec["params"] == res.report.params and rec["naive_params"] == res.report.naive_params
    assert rec["n_experts"] == res.n_experts and rec["wall_clock"] == res.wall_clock
    assert rec["forgetting"] == res.stats["forgetting"]
    assert sum(rec["event_counts"].values()) == len(res.events)
    final_only = run_fast(_makers(stream)["knn"], stream, seed=0, eval_every="final")
    assert run_record("S2-3", "kNN_all", 0, final_only)["forgetting"] is None             # not measured in final-only mode
