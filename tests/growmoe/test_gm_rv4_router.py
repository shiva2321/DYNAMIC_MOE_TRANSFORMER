"""RV4 stage 4a, pure functions: candidates, cross-validation, selection with hysteresis (spec section 4.1)."""

import numpy as np
import pytest
import torch

from growmoe.router import CANDIDATES, FITS, RouteRule, cv_owner_accuracy, fit_softmax, fold_assignment, select_rule
from growmoe.settest import generator
from rv4_helpers import aniso, antipodal, unit


def test_cross_validation_prefers_lda_when_a_shared_covariance_removes_the_noise():
    for seed in range(3):
        X, y = aniso(seed=seed)
        cv = cv_owner_accuracy(X, y, 6, 5, generator(1, seed))
        assert cv["lda"] > cv["density"] + 0.2 and cv["lda"] > 0.9
        assert select_rule(cv, "density", 0.01) == "lda"


def test_cross_validation_keeps_density_for_experts_that_hold_two_opposite_clusters():
    for seed in range(3):
        X, y = antipodal(seed=seed)
        cv = cv_owner_accuracy(X, y, 4, 5, generator(2, seed))
        assert cv["density"] > 0.95 and cv["lda"] < 0.5 and cv["softmax"] < 0.6
        assert select_rule(cv, "density", 0.01) == "density"


def test_every_candidate_fits_well_separated_classes():
    g = np.random.default_rng(0)
    X = np.concatenate([5 * np.eye(12)[c] + g.standard_normal((20, 12)) for c in range(3)])          # three clusters along three axes
    X, y = torch.from_numpy(unit(X)), torch.arange(3).repeat_interleave(20)
    for name, fit in FITS.items():
        scores = fit(X, y, 3, generator(3))(X)
        assert scores.shape == (60, 3), name
        assert (scores.argmax(1) == y).float().mean() > 0.95, name


def test_a_class_without_rows_can_never_be_chosen():
    X, y = aniso(k=3, n=16)
    for name in ("density", "lda", "softmax"):                                    # class 3 exists in the label space but holds nothing
        s = FITS[name](X, y, 4, generator(4))(X)
        assert s.shape == (len(X), 4), name
        assert bool((s[:, 3] == -float("inf")).all()) and bool((s.argmax(1) != 3).all()), name
    s = FITS["knn5"](X, y, 4, generator(4))(X)
    assert bool((s.argmax(1) != 3).all())


def test_softmax_is_class_balanced():
    g = np.random.default_rng(1)                                                  # 90 % of the rows belong to class 0; an unbalanced fit would ignore class 1
    a = g.standard_normal((180, 12)) + 3.0 * np.eye(12)[0]
    b = g.standard_normal((20, 12)) + 3.0 * np.eye(12)[1]
    X, y = torch.from_numpy(unit(np.concatenate([a, b]))), torch.tensor([0] * 180 + [1] * 20)
    s = fit_softmax(X, y, 2, generator(5))(X)
    assert float((s[y == 1].argmax(1) == 1).float().mean()) > 0.8


def test_folds_cover_every_row_once_and_spread_every_class():
    y = torch.tensor([0] * 23 + [1] * 17 + [2] * 11)
    f = fold_assignment(y, 3, 5, generator(6))
    assert f.shape == y.shape and set(f.tolist()) == set(range(5))
    for c in range(3):
        counts = torch.bincount(f[y == c], minlength=5)
        assert int(counts.max() - counts.min()) <= 1


def test_cross_validation_is_a_function_of_its_inputs_and_generator():
    X, y = aniso()
    a = cv_owner_accuracy(X, y, 6, 5, generator(7))
    assert a == cv_owner_accuracy(X, y, 6, 5, generator(7))
    assert a != cv_owner_accuracy(X, y, 6, 5, generator(8))                        # another generator, another fold split
    assert set(a) == set(CANDIDATES)


def test_cross_validation_runs_only_the_requested_candidates():
    X, y = aniso()
    assert set(cv_owner_accuracy(X, y, 6, 5, generator(9), ("density", "lda"))) == {"density", "lda"}


def test_select_rule_keeps_density_unless_another_rule_wins_by_more_than_the_margin():
    base = {"density": 0.50, "lda": 0.75, "softmax": 0.40, "knn5": 0.30}
    assert select_rule(base, "density", 0.25) == "density"                         # exactly the margin is not MORE than the margin
    assert select_rule({**base, "lda": 0.7501}, "density", 0.25) == "lda"
    assert select_rule({**base, "lda": 0.50, "softmax": 0.50}, "density", 0.0) == "density"       # a tie goes to density


def test_select_rule_picks_the_best_of_several_winners():
    assert select_rule({"density": 0.5, "lda": 0.8, "softmax": 0.9, "knn5": 0.7}, "density", 0.01) == "softmax"


def test_hysteresis_keeps_the_current_rule_inside_the_dead_band_and_switches_outside_it():
    dead = {"density": 0.935, "lda": 0.93, "softmax": 0.5, "knn5": 0.5}
    assert select_rule(dead, "lda", 0.01) == "lda"                                 # density leads by 0.005 only: stay
    assert select_rule(dead, "density", 0.01) == "density"
    back = {"density": 0.95, "lda": 0.93, "softmax": 0.5, "knn5": 0.5}
    assert select_rule(back, "lda", 0.01) == "density"                             # density leads by 0.02: go back
    close = {"density": 0.5, "lda": 0.9, "softmax": 0.905, "knn5": 0.5}
    assert select_rule(close, "lda", 0.01) == "lda"                                # softmax leads lda by 0.005 only: stay
    clear = {"density": 0.5, "lda": 0.9, "softmax": 0.915, "knn5": 0.5}
    assert select_rule(clear, "lda", 0.01) == "softmax"                            # softmax leads by 0.015: switch
    assert select_rule(clear, "knn5", 0.01) == "softmax"                           # a current rule that is not even better than density is simply left


def test_a_route_rule_reports_its_cost():
    rule = RouteRule()
    assert rule.name == "density" and rule.cost(12) == (0, 0, 0)
    rule.set("lda", object(), [0, 1, 2], {}, 48)
    assert rule.cost(12) == (3 * 12 + 3, 3 * 12, 0)
    rule.set("softmax", object(), [0, 1, 2], {}, 48)
    assert rule.cost(12) == (3 * 12 + 3, 3 * 12, 0)
    rule.set("knn5", object(), [0, 1, 2], {}, 48)
    assert rule.cost(12) == (0, 48 * 12, 48 * 12)
    rule.set("density", None, [0, 1, 2], {}, 48)
    assert rule.cost(12) == (0, 0, 0)
