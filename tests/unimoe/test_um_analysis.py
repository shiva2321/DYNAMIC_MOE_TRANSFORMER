"""The analysis layer (unimoe/analysis.py): every metric of the evaluation battery, checked against logs small enough to compute by hand."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unimoe import analysis as AN
from unimoe.runner import RunRecord


def regimes():
    mk = lambda i, dom, kind="text", meta=None: SimpleNamespace(id=i, name=f"r{i}", domain=dom, kind=kind, meta=meta or {})
    return {0: mk(0, "A"), 1: mk(1, "A"), 2: mk(2, "B")}


def record(R, train=None, test=None, ood=None, extra=None, per_phase=None, report=None, events=None):
    arrays = {"R": np.asarray(R)}
    arrays.update({f"train_{k}": np.asarray(v) for k, v in (train or {}).items()})
    arrays.update({f"test_{k}": np.asarray(v) for k, v in (test or {}).items()})
    return RunRecord("sys", "strm", 0, np.asarray(R), per_phase or [], ood or [], report or {}, {}, events or [], extra or {}, 1.0, arrays)


def test_final_accuracy_is_the_macro_mean_over_regimes_and_per_domain():
    rec = record([[0.9, 0.0, 0.0], [0.6, 0.8, 0.0], [0.5, 0.7, 0.9]])
    out = AN.final_accuracy(rec, regimes())
    assert out["mean"] == pytest.approx((0.5 + 0.7 + 0.9) / 3)
    assert out["per_domain"] == {"A": pytest.approx(0.6), "B": pytest.approx(0.9)}
    assert out["per_regime"] == [0.5, 0.7, 0.9]


def test_forgetting_backward_transfer_and_learning_accuracy_follow_the_first_visit_of_each_regime():
    R = [[0.9, 0.1, 0.0], [0.5, 0.8, 0.1], [0.4, 0.7, 0.9]]
    train = {"phase": [0, 0, 1, 1, 2, 2], "regime": [0, 0, 1, 1, 2, 2], "y": [0] * 6, "pred": [0, 1, 0, 0, 1, 1]}
    c = AN.continual(record(R, train=train), regimes())
    assert c["forgetting"] == [pytest.approx(0.5), pytest.approx(0.1), pytest.approx(0.0)]
    assert c["mean_forgetting"] == pytest.approx(0.2)
    assert c["bwt"] == pytest.approx(np.mean([0.4 - 0.9, 0.7 - 0.8, 0.9 - 0.9]))
    assert c["learning_acc"] == [0.9, 0.8, 0.9]
    assert c["first_visit_prequential"] == [pytest.approx(0.5), pytest.approx(1.0), pytest.approx(0.0)]    # the first batch of each regime, predicted before it was learned


def test_structure_counts_owners_merged_regimes_cross_domain_experts_and_purity():
    # expert 0 trained regimes 0 and 1 (both domain A): a merge; expert 1 trained regime 2 alone; expert 2 split regime 2 with expert 1
    train = {"regime": [0] * 10 + [1] * 10 + [2] * 10, "assigned": [0] * 10 + [0] * 10 + [1] * 6 + [2] * 4}
    s = AN.structure(record([[0, 0, 0]], train=train, per_phase=[{"n_experts": 3}]), regimes())
    assert s["n_experts"] == 3 and s["n_regimes"] == 3
    assert s["owner"] == {0: 0, 1: 0, 2: 1}
    assert s["regimes_with_exclusive_expert"] == 1 and s["merged_groups"] == [[0, 1]] and s["cross_domain_experts"] == 0
    assert s["split_regimes"] == [2]                                                             # regime 2's samples went to two experts, each with a real share
    assert s["purity"] == pytest.approx((10 + 6 + 4) / 30)                                       # each expert's largest single regime, over all samples
    assert 0.0 < s["nmi_regime"] <= 1.0
    mixed = {"regime": [0] * 10 + [2] * 10, "assigned": [0] * 20}
    assert AN.structure(record([[0, 0, 0]], train=mixed, per_phase=[{"n_experts": 1}]), regimes())["cross_domain_experts"] == 1


def test_routing_compares_the_routed_answer_with_the_owner_experts_answer_on_the_same_samples():
    test = {"regime": [0, 0, 0, 0, 2, 2], "y": [1, 1, 1, 1, 5, 5], "pred": [1, 0, 0, 1, 5, 4], "oracle_pred": [1, 1, 1, 1, 5, -1], "expert": [0, 1, 1, 0, 3, 3]}
    r = AN.routing(record([[0, 0, 0]], test=test, extra={"owner_of_regime": {"0": 0, "2": 3}}), regimes())
    assert r["routed_acc"] == pytest.approx(3 / 6) and r["n_oracle"] == 5
    assert r["routed_acc_on_oracle_samples"] == pytest.approx(3 / 5) and r["oracle_acc"] == pytest.approx(1.0) and r["gap"] == pytest.approx(0.4)
    assert r["owner_route_rate"] == pytest.approx(4 / 6)                                         # sent to its regime's owner: 2 of 4 for regime 0, 2 of 2 for regime 2


def test_failure_taxonomy_splits_the_errors_into_routing_expert_and_unowned_and_counts_false_refusals():
    test = {"regime": [0, 0, 0, 0, 1, 2], "y": [1, 1, 1, 1, 1, 5], "pred": [0, 0, 1, 0, 0, 4], "oracle_pred": [1, 0, 1, 1, 0, -1],
            "expert": [1, 0, 0, 1, 0, 9], "unknown": [False, False, False, True, False, False]}
    f = AN.failures(record([[0, 0, 0]], test=test, extra={"owner_of_regime": {"0": 0, "1": 0}}), regimes())
    assert f["counts"] == {"correct": 1, "routing_error": 2, "expert_error": 2, "unowned": 1}      # sample 4 (regime 1): routed to its owner, wrong, the owner is wrong too
    assert f["false_refusals"] == 1 and f["n"] == 6
    assert f["per_domain"]["A"]["routing_error"] == 2 and f["per_domain"]["B"]["unowned"] == 1


def test_novelty_reads_the_last_phase_and_the_false_novel_rate_over_the_run():
    ood = [{"phase": 0, "known_unknown_rate": 0.0, "stories_unknown_rate": 0.1, "stories_auroc": 0.7}, {"phase": 1, "known_unknown_rate": 0.04, "stories_unknown_rate": 0.3, "stories_auroc": 0.9}]
    n = AN.novelty(record([[0]], ood=ood))
    assert n["stories"] == {"unknown_rate": 0.3, "auroc": 0.9} and n["false_novel_last"] == 0.04 and n["false_novel_max"] == 0.04


def test_exact_match_counts_a_problem_only_when_every_one_of_its_slots_is_right():
    rg = {0: SimpleNamespace(id=0, name="add", domain="arith", kind="digits", meta={"problem_id": torch.tensor([0, 0, 0, 1, 1, 2])}), 1: regimes()[2]}
    test = {"regime": [0] * 6 + [1] * 2, "y": [1, 2, 3, 4, 5, 6, 0, 0], "pred": [1, 2, 3, 4, 0, 6, 0, 0]}
    e = AN.exact_match(record([[0, 0]], test=test), rg)
    assert e == {"add": pytest.approx(2 / 3)}                                                    # problems 0 and 2 are right, problem 1 has a wrong slot; text regimes are skipped


def test_representation_diagnostics_separate_regimes_and_domains_on_the_sphere():
    g = torch.Generator().manual_seed(0)
    mk = lambda c, n: torch.nn.functional.normalize(c + 0.1 * torch.randn(n, 8, generator=g), dim=1)
    ca, cb, cc = torch.eye(8)[0], torch.eye(8)[1], torch.eye(8)[2]
    x = torch.cat([mk(ca, 40), mk(cb, 40), mk(cc, 40)])
    regime = torch.tensor([0] * 40 + [1] * 40 + [2] * 40)
    d = AN.representation(x, regime, regimes())
    assert d["within_regime_cos"] > 0.9 and abs(d["between_regimes_same_domain_cos"]) < 0.2 and abs(d["between_domains_cos"]) < 0.2
    assert 1.0 <= d["effective_dim"]["A"] <= 8.0


def test_the_interval_over_seeds_is_the_mean_and_a_t_interval():
    m, lo, hi = AN.over_seeds([0.5, 0.6, 0.7])
    assert m == pytest.approx(0.6) and lo < m < hi
    assert AN.over_seeds([0.5])[0] == 0.5
