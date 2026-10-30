"""The runner (unimoe/runner.py): strict test-then-train, an accuracy matrix with no gaps, novelty on out-of-distribution sets, per-sample logs, and cached cells."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_um_suite as TS  # noqa: E402

from growmoe.baselines import FineTune, _Base  # noqa: E402
from growmoe.config import CommonCfg  # noqa: E402
from unimoe import runner as RN  # noqa: E402
from unimoe import suite as SU  # noqa: E402

C = 25


def stream(schedule="seq", with_ood=True):
    s = SU.build_suite(schedule, TS.toy_domains(), 0, TS.toy_ood(), n_visit=100, n_visit_arith=200, n_test=40, revisit_k=1, n_phases_mix=3, n_phases_drift=4)
    if with_ood:
        e0 = torch.zeros(30, 128)
        e0[:, 0] = 1.0
        s.ood = {"stories": e0}                                                                  # a fixed direction no domain vector points to
    return s


class SpyPool(_Base):
    """Records the order of calls; novelty = closeness of the input to the fixed direction e0."""
    name = "spy"

    def __init__(self, uses_regime=False, flag_known=False):
        super().__init__(CommonCfg(d=128, n_classes=C))
        self.calls = []
        self.uses_regime = uses_regime
        self.flag_known = flag_known                                                             # also flag about half of the ordinary inputs as unknown

    def predict(self, x, ctx=None):
        self.calls.append(("predict", ctx is not None, float(x.sum())))
        far = x[:, 0] < 0.9
        flagged = ~far | ((x[:, 1] > 0) if self.flag_known else torch.zeros_like(far))
        return self._prediction(torch.full((x.shape[0], C), 1.0 / C), -1, p_nov=(~flagged).float(), unknown=flagged)

    def observe(self, x, y=None, ctx=None):
        self.calls.append(("observe", ctx is not None, y is not None, float(x.sum())))
        self.step += 1
        return self._record(x.shape[0], -1, n_experts=1)

    def summary(self):
        return self._summary([], 0, n_experts=1)


def test_every_batch_is_predicted_before_it_is_learned_and_the_pool_gets_no_ground_truth():
    s, pool = stream(), SpyPool()
    RN.run_system(lambda: pool, s, seed=0, batch_size=64)
    n_batches = sum(len(list(s.algo_batches(k, 64, 0))) for k in range(len(s.phases)))
    train = []
    for i, c in enumerate(pool.calls):
        if c[0] == "observe":
            assert i >= 1 and pool.calls[i - 1][0] == "predict" and pool.calls[i - 1][2] == c[3]   # the prediction of THIS batch came first
            train.append(c)
    assert len(train) == n_batches and all(c[1] is False and c[2] is True for c in train)         # no regime ids; labels only to observe
    assert all(c[1] is False for c in pool.calls if c[0] == "predict")
    told = SpyPool(uses_regime=True)
    RN.run_system(lambda: told, s, seed=0, batch_size=64)
    assert all(c[1] is True for c in told.calls)                                                 # an oracle baseline is handed the regime, everywhere


def test_the_matrix_has_a_row_per_phase_a_column_per_regime_and_no_gaps():
    s = stream()
    rec = RN.run_system(lambda: FineTune(CommonCfg(d=128, n_classes=s.n_classes, inner_steps=3, seed=0)), s, seed=0)
    assert rec.R.shape == (len(s.phases), len(s.regimes)) and np.isfinite(rec.R).all() and rec.R.min() >= 0.0 and rec.R.max() <= 1.0
    assert len(rec.per_phase) == len(s.phases) and rec.per_phase[-1]["n_experts"] == 1
    assert rec.seconds > 0 and rec.report["params"] > 0


def test_novelty_is_read_on_every_out_of_distribution_set_and_on_the_known_test_data_at_every_phase():
    s = stream()
    rec = RN.run_system(lambda: SpyPool(), s, seed=0)
    for o in rec.ood:
        assert o["stories_unknown_rate"] == 1.0 and o["known_unknown_rate"] == 0.0 and o["stories_auroc"] == pytest.approx(1.0)
    assert len(rec.ood) == len(s.phases)


def test_the_false_novelty_rate_on_known_test_data_is_read_when_the_pool_flags_known_inputs():
    s = stream()
    rec = RN.run_system(lambda: SpyPool(flag_known=True), s, seed=0)
    assert all(0.3 < o["known_unknown_rate"] < 0.7 for o in rec.ood)
    assert all(o["stories_unknown_rate"] == 1.0 for o in rec.ood)                               # and the foreign direction is still caught


def test_the_logs_cover_every_training_sample_and_every_test_sample_once():
    s = stream()
    rec = RN.run_system(lambda: SpyPool(), s, seed=0)
    total = sum(len(p.y) for p in s.phases)
    for k in ("phase", "regime", "first", "y", "pred", "expert", "unknown", "p_nov", "assigned"):
        assert len(rec.arrays["train_" + k]) == total, k
    n_test = sum(len(r.y_te) for r in s.regimes.values())
    for k in ("regime", "y", "pred", "expert", "unknown", "p_nov"):
        assert len(rec.arrays["test_" + k]) == n_test, k
    assert set(rec.arrays["train_regime"].tolist()) == set(s.regimes)


def test_the_owner_of_a_regime_is_the_expert_that_trained_on_most_of_it():
    owner = RN.owner_of_regime(np.array([0, 0, 1, -1, 1, 1, -1]), np.array([0, 0, 0, 0, 1, 1, 2]))
    assert owner == {0: 0, 1: 1}                                                                # regime 2 was never assigned: no owner


def test_a_cell_is_computed_once_and_recomputed_when_the_code_stamp_changes(tmp_path):
    s, calls = stream(), []

    def make():
        calls.append(1)
        return SpyPool()

    a = RN.run_cell(tmp_path, "spy", make, s, seed=0, stamp="v1")
    b = RN.run_cell(tmp_path, "spy", make, s, seed=0, stamp="v1")
    assert len(calls) == 1 and np.allclose(a.R, b.R) and b.per_phase == a.per_phase
    assert np.array_equal(a.arrays["test_pred"], b.arrays["test_pred"])
    RN.run_cell(tmp_path, "spy", make, s, seed=0, stamp="v2")
    assert len(calls) == 2
    RN.run_cell(tmp_path, "spy", make, s, seed=1, stamp="v2")
    assert len(calls) == 3                                                                       # another seed is another cell
    assert any(tmp_path.rglob("*.json")) and any(tmp_path.rglob("*.npz"))


def test_a_real_pool_runs_a_short_stream_and_reports_its_structure_and_the_oracle_routed_accuracy():
    from unimoe import systems as SY
    s = stream()
    rec = RN.run_system(lambda: SY.make_system("U-real", s.n_classes, seed=0, recipe={"inner_steps": 2}), s, seed=0)
    assert rec.per_phase[-1]["n_experts"] >= 1 and "events" in rec.per_phase[-1]
    assert set(rec.extra["owner_of_regime"]) <= set(map(str, s.regimes))
    assert len(rec.arrays["test_oracle_pred"]) == len(rec.arrays["test_pred"])
    assert any(e["kind"] == "spawn" for e in rec.events) or rec.per_phase[-1]["n_experts"] == 1
