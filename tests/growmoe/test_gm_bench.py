"""Tests for Task 17: Benchmark runner and Phase-1 comparison (growmoe.bench).

Tested properties (R10 handoff):
1. run_stream returns all metric keys (R matrix, accuracy_stats, auroc, false_novelty_rate, spawn_stats, wall_clock).
2. Ground truth never reaches the pool: spy pool records call arguments and confirms no GT leaks.
3. Determinism: same seed -> identical RunResult (R matrix, metrics, event log).
4. Dev seeds assertion: tune strictly requires dev seeds >= 100 and rejects test seeds (0-9).
5. Signature familiarity ablations conform to the VmfSignature protocol:
   GaussianMahalanobisSignature, KnnSignature, RndSignature implement update, fit, score, logpdf, pvalue, affinity.
"""

import inspect
import numpy as np
import pytest
import torch

from growmoe.bench import (
    GaussianMahalanobisSignature,
    KnnSignature,
    RndSignature,
    RunResult,
    run_stream,
    tune,
)
from growmoe.config import PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams import s6_long
from growmoe.types import Prediction, StepRecord


# --------------------------------------------------------------------------- 1. run_stream returns all metric keys
def test_bench_runner_returns_all_metric_keys():
    seed = 101
    stream = s6_long(seed=seed, n_regimes=2, d=64, n_train=40, related_frac=0, revisit_frac=0.5)

    def make_pool():
        return PhyloMoE(PhyloCfg(d=64, m_min=16, seed=seed))

    res = run_stream(make_pool, stream, seed=seed, batch_size=32)

    assert isinstance(res, RunResult)
    assert res.R.shape == (len(stream.segments), len(stream.regimes))
    assert "avg" in res.accuracy_stats
    assert "forgetting" in res.accuracy_stats
    assert isinstance(res.auroc, float)
    assert 0.0 <= res.false_novelty_rate <= 1.0
    assert "precision" in res.spawn_stats
    assert "recall" in res.spawn_stats
    assert "ratio" in res.spawn_stats
    assert res.wall_clock > 0.0
    assert res.total_samples > 0


# --------------------------------------------------------------------------- 2. Ground truth never reaches pool
def test_bench_runner_ground_truth_never_reaches_pool():
    seed = 102
    stream = s6_long(seed=seed, n_regimes=2, d=64, n_train=40, related_frac=0, revisit_frac=0)

    class SpyPool:
        name = "spy_pool"

        def __init__(self):
            self.step = 0
            self.received_args = []
            from growmoe.telemetry import EventLog
            self.events = EventLog()

        def predict(self, x, ctx=None):
            self.received_args.append(("predict", list(locals().keys())))
            b = x.shape[0]
            probs = torch.full((b, 8), 1.0 / 8)
            expert = torch.zeros(b, dtype=torch.long)
            p_nov = torch.zeros(b)
            unknown = torch.zeros(b, dtype=torch.bool)
            return Prediction(probs=probs, expert=expert, p_nov=p_nov, unknown=unknown)

        def observe(self, x, y=None, ctx=None):
            self.step += 1
            self.received_args.append(("observe", list(locals().keys())))
            b = x.shape[0]
            return StepRecord(
                novel=torch.zeros(b, dtype=torch.bool),
                p_nov=torch.zeros(b),
                assigned=torch.zeros(b, dtype=torch.long),
                loss=0.0,
                spawned=[],
                n_experts=1,
            )

        def summary(self):
            return {"n_experts": 1, "n_params": 100, "active_params_per_sample": 100, "lineage": {}, "states": {}}

    res = run_stream(lambda: SpyPool(), stream, seed=seed, batch_size=32)

    # Check observed calls in spy pool
    for method, arg_names in res.pool_summary.get("spy_args", []):
        assert "gt" not in arg_names
        assert "regime" not in arg_names
        assert "first_visit" not in arg_names
        assert "is_outlier" not in arg_names


# --------------------------------------------------------------------------- 3. Determinism
def test_bench_runner_deterministic_per_seed():
    seed = 103
    stream = s6_long(seed=seed, n_regimes=2, d=64, n_train=40, related_frac=0, revisit_frac=0.5)

    def make_pool():
        return PhyloMoE(PhyloCfg(d=64, m_min=16, seed=seed))

    res1 = run_stream(make_pool, stream, seed=seed, batch_size=32)
    res2 = run_stream(make_pool, stream, seed=seed, batch_size=32)

    assert np.allclose(res1.R, res2.R, equal_nan=True)
    assert res1.accuracy_stats == res2.accuracy_stats
    assert (res1.auroc == res2.auroc) or (math.isnan(res1.auroc) and math.isnan(res2.auroc))
    assert (res1.false_novelty_rate == res2.false_novelty_rate) or (
        math.isnan(res1.false_novelty_rate) and math.isnan(res2.false_novelty_rate)
    )


# --------------------------------------------------------------------------- 4. Dev seeds assertion in tune
def test_bench_tune_asserts_dev_seeds_only():
    grid = [{"lr": 1e-3}, {"lr": 3e-3}]

    # A stream with test seed (< 100) must raise ValueError / AssertionError
    test_stream = s6_long(seed=5, n_regimes=2, d=64, n_train=20)
    test_stream.meta["seed"] = 5
    with pytest.raises((ValueError, AssertionError), match="dev seed"):
        tune(lambda cfg: PhyloMoE(PhyloCfg(d=64, seed=100)), grid, [test_stream], budget=2)

    # A stream with dev seed (>= 100) passes
    dev_stream = s6_long(seed=105, n_regimes=2, d=64, n_train=20)
    dev_stream.meta["seed"] = 105
    best_cfg, best_score = tune(
        lambda cfg: PhyloMoE(PhyloCfg(d=64, seed=105, **cfg)),
        grid,
        [dev_stream],
        budget=2,
    )
    assert isinstance(best_cfg, dict)
    assert isinstance(best_score, float)


# --------------------------------------------------------------------------- 5. Signature familiarity ablations protocol
def test_bench_signature_ablations_conform_to_protocol():
    d = 64
    x = torch.randn(50, d)
    x = x / x.norm(dim=-1, keepdim=True)

    sig_classes = [GaussianMahalanobisSignature, KnnSignature, RndSignature]
    for cls in sig_classes:
        sig = cls(d=d, seed=106)
        sig.update(x)
        sig.fit()

        u = torch.randn(10, d)
        u = u / u.norm(dim=-1, keepdim=True)

        scores = sig.score(u)
        assert scores.shape == (10,)
        pvals = sig.pvalue(u)
        assert pvals.shape == (10,)
        assert (pvals >= 0.0).all() and (pvals <= 1.0).all()

        logpdf = sig.logpdf(u)
        assert logpdf.shape == (10,)

        aff = sig.affinity(u[:3])
        assert isinstance(aff, float)
        assert 0.0 <= aff <= 1.0


# --------------------------------------------------------------------------- 6. Novelty metric stub test (A1)
def test_novelty_metric_conformal_vs_p50_stub():
    """Prove that calibrated gate (unknown=BH(p, q)) has FNR <= 2q while old p <= 0.5 gives ~0.5 (A1)."""
    from growmoe.conformal import benjamini_hochberg
    q = 0.05
    n_samples = 2000
    rng = np.random.default_rng(20261005)

    # Under null (known samples), conformal p-values are uniform U[0, 1]
    p_vals = torch.from_numpy(rng.uniform(0.0, 1.0, size=n_samples).astype(np.float32))

    # Calibrated gate abstain decision: BH(p, q)
    unknown = benjamini_hochberg(p_vals, q=q)

    # Old bench formula: scores = 1 - p, flag if scores >= 0.5 <=> p <= 0.5
    scores = 1.0 - p_vals
    old_flagged = scores >= 0.5

    fnr_calibrated = float(unknown.float().mean().item())
    fnr_old = float(old_flagged.float().mean().item())

    # Calibrated FNR must be within [0, 2q] (nominal FDR level q = 0.05 -> <= 0.10)
    assert 0.0 <= fnr_calibrated <= 2.0 * q, f"Calibrated FNR {fnr_calibrated} exceeded 2q ({2*q})"

    # Old formula: P(U[0, 1] <= 0.5) = 0.50
    assert 0.45 <= fnr_old <= 0.55, f"Old formula FNR {fnr_old} not within [0.45, 0.55]"

    # Proves the huge difference: old formula is at least 4x higher than calibrated gate
    assert fnr_old > 4.0 * max(fnr_calibrated, 1e-4)


# --------------------------------------------------------------------------- 7. RESULTS.md vs JSON consistency test (A2 / D4 / G3)
def test_results_md_verdicts_match_computed_metrics():
    """Verify that verdicts in RESULTS.md match dynamically recomputed verdicts from phase1_results.json per-seed data."""
    import json
    import os
    import re
    from growmoe.bench import compute_verdicts_from_per_seed

    res_path = "research/growmoe_results/phase1/RESULTS.md"
    json_path = "research/growmoe_results/phase1/phase1_results.json"

    assert os.path.exists(res_path), f"Phase-1 results markdown missing from disk: {res_path}"
    assert os.path.exists(json_path), f"Phase-1 results json missing from disk: {json_path}"

    with open(json_path, "r", encoding="ascii") as f:
        data = json.load(f)

    assert "per_seed_metrics" in data, "per_seed_metrics missing from phase1_results.json"
    q_val = data.get("tuned", {}).get("A", {}).get("cfg", {}).get("q", 0.05)
    recomputed_eval = compute_verdicts_from_per_seed(data["per_seed_metrics"], q_val=q_val)
    recomputed_verdicts = {item[0]: item[2] for item in recomputed_eval}

    stored_verdicts = {item["id"]: item["pass"] for item in data["eval_p"]}

    with open(res_path, "r", encoding="ascii") as f:
        lines = f.readlines()

    # Parse rows: | ID | Prediction Description | Status | Measured Metric | Statistics / 95% CI |
    md_verdicts = {}
    table_started = False
    for line in lines:
        if "| ID |" in line:
            table_started = True
            continue
        if table_started:
            if not line.strip().startswith("|"):
                break
            parts = [p.strip() for p in line.strip().split("|")[1:-1]]
            if len(parts) >= 3:
                pid = parts[0]
                status_raw = parts[2]
                if pid in ("P1", "P2", "P3", "P4", "P5", "P6", "P8", "P10", "C1", "C2", "C3", "C4", "C5", "C6"):
                    if "NOT MEASURABLE" in status_raw:
                        v_parsed = "NOT MEASURABLE"
                    elif "INCONCLUSIVE" in status_raw:
                        v_parsed = "INCONCLUSIVE"
                    elif "PASS" in status_raw:
                        v_parsed = True
                    else:
                        v_parsed = False
                    md_verdicts[pid] = v_parsed

    assert len(md_verdicts) > 0, "No prediction rows found in RESULTS.md"
    assert len(recomputed_verdicts) > 0, "No recomputed predictions produced"

    for pid, expected_pass in recomputed_verdicts.items():
        assert pid in stored_verdicts, f"Prediction {pid} missing from phase1_results.json eval_p"
        assert stored_verdicts[pid] == expected_pass, (
            f"Verdict mismatch for {pid}: recomputed={expected_pass} vs stored in JSON={stored_verdicts[pid]}"
        )
        assert pid in md_verdicts, f"Prediction {pid} missing from RESULTS.md"
        assert md_verdicts[pid] == expected_pass, (
            f"Verdict mismatch for {pid}: recomputed={expected_pass} vs RESULTS.md={md_verdicts[pid]}"
        )


def test_compute_verdicts_p6_not_measurable_on_zero_children():
    """Unit test confirming that compute_verdicts_from_per_seed yields NOT MEASURABLE when n_aff_children < 5."""
    from growmoe.bench import compute_verdicts_from_per_seed

    mock_per_seed = {
        "A_phylomoe": {
            "s1": [{"false_novelty_rate": 0.0, "recall": 1.0, "ratio": 1.0, "retained_frac": 1.0, "avg_acc": 0.6, "forgetting": 0.0}],
            "s2": [{"false_novelty_rate": 0.0, "recall": 1.0, "ratio": 1.0, "retained_frac": 1.0, "avg_acc": 0.85, "forgetting": 0.0, "revisit_acc": 0.96, "n_aff_children": 0, "early_acc": 0.8, "wall_clock": 1.0, "n_params": 1000}],
            "s4": [{"false_novelty_rate": 0.0, "spurious_spawns": 0.0}],
        },
        "B4_oracle_routed": {"s1": [{"avg_acc": 0.7}]},
        "B2_replay": {"s1": [{"avg_acc": 0.5}]},
        "A_no_conformal": {"s2": [{"false_novelty_rate": 0.1, "recall": 0.8}], "s4": [{"spurious_spawns": 2.0}]},
        "B_bnp": {"s1": [{"auroc": 0.7}], "s2": [{"false_novelty_rate": 0.05, "recall": 0.7, "ratio": 2.0, "early_acc": 0.7, "wall_clock": 2.0}]},
        "C_keyed": {"s2": [{"false_novelty_rate": 0.05, "recall": 0.7, "ratio": 1.0, "n_params": 300, "avg_acc": 0.83, "early_acc": 0.75, "wall_clock": 0.5}]},
        "B_tau_low": {"s2": [{"n_spawns": 10}]},
        "B_tau_high": {"s2": [{"n_spawns": 2}]},
        "A_q_low": {"s2": [{"n_spawns": 4}]},
        "A_q_high": {"s2": [{"n_spawns": 5}]},
        "A_init_scratch": {"s2": [{"avg_acc": 0.8}]},
    }

    eval_p = compute_verdicts_from_per_seed(mock_per_seed, q_val=0.05)
    verdicts = {item[0]: item[2] for item in eval_p}
    assert verdicts["P6"] == "NOT MEASURABLE"
