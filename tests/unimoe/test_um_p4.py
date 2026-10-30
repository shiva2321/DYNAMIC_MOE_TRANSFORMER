"""P4 pieces: the capacity-hungry stream (unimoe/capacity.py), the refusal layer's decisions and selective-answer metrics (unimoe/refusal.py), the registry of the new variants (unimoe/systems2.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unimoe import capacity as CP
from unimoe import refusal as RF
from unimoe import systems2 as S2


class FakeFE:
    d = 128

    def digits(self, codes):
        x = np.zeros((len(codes), 128), dtype=np.float32)
        x[:, :codes.shape[1]] = codes
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return torch.from_numpy(np.where(n > 0, x / np.maximum(n, 1e-12), x))

    def text(self, texts):                                                                       # never called: the capacity stream is arithmetic only
        raise AssertionError("the capacity stream must not encode text")


# ---------------------------------------------------------------------------------------------------------------------------------- the capacity-hungry stream
def test_the_capacity_stream_is_six_arithmetic_regimes_with_a_lot_of_data_each_and_no_ood_text():
    s = CP.build_capacity_suite("seq", seed=0, fe=FakeFE(), n_per_op=600, n_test=100)
    assert len(s.regimes) == 6 and s.n_classes == 12 and set(s.label_blocks) == {"arith"} and s.ood == {}
    assert len(s.phases) == 6 and all(len(ph.regime.unique()) == 1 and len(ph.y) == 600 for ph in s.phases)         # no revisits; a visit is the full n_per_op
    assert all(0 < len(r.y_te) <= 100 and "problem_id" in r.meta for r in s.regimes.values())
    m = CP.build_capacity_suite("mix", seed=0, fe=FakeFE(), n_per_op=600, n_test=100)
    assert sum(len(p.y) for p in m.phases) == 6 * 600 and all(len(p.regime.unique()) == 6 for p in m.phases)


# ---------------------------------------------------------------------------------------------------------------------------------- refusal
def two_readers(n=400, shift=3.0, seed=0):
    rng = np.random.default_rng(seed)
    known = {"pool": rng.normal(0, 1, n), "reader": rng.normal(0, 1, n)}
    ood = {"pool": rng.normal(shift, 1, n), "reader": rng.normal(shift, 1, n)}
    return known, ood


def test_calibrated_refusal_keeps_known_messages_and_refuses_foreign_ones_and_is_silent_without_enough_outcomes():
    known, ood = two_readers()
    cal = {k: v[:60] for k, v in known.items()}
    test_known = {k: v[60:] for k, v in known.items()}
    r_known = RF.refusal_decision(cal, test_known, ["pool", "reader"], q=0.05)
    r_ood = RF.refusal_decision(cal, ood, ["pool", "reader"], q=0.05)
    assert r_known.mean() <= 0.12 and r_ood.mean() > 0.8
    tiny = {k: v[:10] for k, v in known.items()}
    assert not RF.refusal_decision(tiny, ood, ["pool", "reader"], q=0.05).any()                  # fewer than 19 verified messages: it refuses nothing, and says so by silence
    with pytest.raises(ValueError):
        RF.refusal_decision({}, ood, ["pool"], q=0.05)


def test_either_catches_what_one_reader_misses_at_a_comparable_false_refusal_rate():
    rng = np.random.default_rng(1)
    n = 3000
    known = {"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)}
    ood = {"a": np.concatenate([rng.normal(4, 1, n // 2), rng.normal(0, 1, n // 2)]), "b": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(4, 1, n // 2)])}   # each reader sees one half
    cal = {k: v[:100] for k, v in known.items()}
    held = {k: v[100:] for k, v in known.items()}
    either = RF.refusal_decision(cal, ood, ["a", "b"], q=0.05)
    only_a = RF.refusal_decision(cal, ood, ["a"], q=0.05)
    assert either.mean() > only_a.mean() + 0.3
    assert RF.refusal_decision(cal, held, ["a", "b"], q=0.05).mean() < 0.2


def test_selective_metrics_report_coverage_accuracy_on_the_answered_and_the_foreign_answers():
    correct = np.array([True, True, False, True, False, False, True, True])
    refuse = np.array([False, False, True, False, True, False, False, False])
    m = RF.selective_metrics(correct, refuse, {"stories": np.array([True, True, False, True])})
    assert m["false_refusal"] == pytest.approx(2 / 8) and m["coverage"] == pytest.approx(6 / 8)
    assert m["accuracy_all"] == pytest.approx(5 / 8) and m["accuracy_answered"] == pytest.approx(5 / 6)         # answered: messages 0, 1, 3, 5, 6, 7; all right but the 6th
    assert m["ood_answered"] == {"stories": pytest.approx(1 / 4)} and m["ood_refused"] == {"stories": pytest.approx(3 / 4)}


# ---------------------------------------------------------------------------------------------------------------------------------- the variants
def test_each_new_variant_switches_on_exactly_its_own_improvement():
    blocks = ((0, 6), (6, 6))
    base = S2.make_system2("UG0", 12, seed=1, recipe={}, blocks=blocks).cfg
    assert base.gap_mode == "recipe" and base.domain_head == "off" and base.lda_blend == 0.0 and base.set_gate                            # U-real, nothing added
    g1 = S2.make_system2("UG1", 12, seed=1, recipe={"grow_gap": 0.3}, blocks=blocks).cfg
    assert g1.gap_mode == "one_step" and g1.growth_rule == "gap" and g1.grow_gap == 0.3 and g1.grow_min_age == 30 and g1.grow_window == 20
    dom = S2.make_system2("UDOM", 12, seed=1, recipe={}, blocks=blocks).cfg
    assert dom.domain_head == "hard" and dom.domain_blocks == blocks and S2.make_system2("UDOMS", 12, seed=1, recipe={"domain_beta": 2.0}, blocks=blocks).cfg.domain_head == "soft"
    lda = S2.make_system2("ULDA", 12, seed=1, recipe={"lda_blend": 0.4}, blocks=blocks).cfg
    assert lda.lda_blend == 0.4 and lda.gap_mode == "recipe" and lda.domain_head == "off"
    plus = S2.make_system2("UPLUS", 12, seed=1, recipe={"lda_blend": 0.4, "grow_gap": 0.3}, blocks=blocks).cfg
    assert plus.gap_mode == "one_step" and plus.domain_head == "hard" and plus.lda_blend == 0.4
    for c in (base, g1, dom, lda, plus):
        assert c.seed == 1 + 7 and c.n_classes == 12 and c.route_rule == "accept_then_logf"
    with pytest.raises(ValueError):
        S2.make_system2("nonsense", 12, seed=1, recipe={}, blocks=blocks)


def test_the_oracle_lda_experts_are_told_the_regime_and_blend_an_lda_head_into_each_expert():
    from test_um_systems import toy, C
    x, y = toy(512, seed=3)
    regime = (torch.arange(512) % 2).long()
    y2 = torch.where(regime == 1, (y + 5) % C, y)
    plain = S2.make_system2("oracle_lda", C, seed=0, recipe={"inner_steps": 1, "lda_blend": 0.0}, blocks=())
    blended = S2.make_system2("oracle_lda", C, seed=0, recipe={"inner_steps": 1, "lda_blend": 0.7}, blocks=())
    assert plain.uses_regime and blended.uses_regime
    for lo in range(0, 384, 64):
        for pool in (plain, blended):
            pool.observe(x[lo:lo + 64], y2[lo:lo + 64], ctx=regime[lo:lo + 64])
    a = lambda p: float((p.predict(x[384:], ctx=regime[384:]).probs.argmax(1) == y2[384:]).float().mean())
    assert a(blended) >= a(plain) - 1e-6 and not torch.allclose(blended.predict(x[384:], ctx=regime[384:]).probs, plain.predict(x[384:], ctx=regime[384:]).probs)


# ---------------------------------------------------------------------------------------------------------------------------------- the refusal experiment's bookkeeping
def synthetic_scores(seed=0, n=900):
    from unimoe import p4_refusal as PR
    rng = np.random.default_rng(seed)
    z = {"known_pool_nov": rng.normal(0, 1, n), "known_reader_unclaimed": rng.normal(0, 1, n), "known_pool_flag": np.zeros(n, dtype=bool), "known_reader_none": np.zeros(n, dtype=bool),
         "known_y": rng.integers(0, 5, n), "known_regime": np.repeat([0, 1, 2], n // 3)}
    z["known_pred"] = np.where(rng.random(n) < 0.8, z["known_y"], (z["known_y"] + 1) % 5)
    for nm, (pool_shift, reader_shift) in {"stories": (4.0, 0.0), "wiki": (0.0, 4.0)}.items():                  # each reader sees only one of the two foreign registers
        z[f"ood_{nm}_pool_nov"] = rng.normal(pool_shift, 1, 300)
        z[f"ood_{nm}_reader_unclaimed"] = rng.normal(reader_shift, 1, 300)
        z[f"ood_{nm}_pool_flag"] = np.zeros(300, dtype=bool)
        z[f"ood_{nm}_reader_none"] = np.zeros(300, dtype=bool)
    return PR, z


def test_the_refusal_experiment_calibrates_on_a_sample_and_keeps_calibration_messages_out_of_the_evaluation():
    PR, z = synthetic_scores()
    dec, ev, names = PR.decisions(z, draw=0, cal_n=60, q=0.05)
    assert names == ["stories", "wiki"] and ev.sum() == 3 * PR.EVAL_CAP                                  # 300 messages per regime, the cap keeps 150 of the non-calibration ones
    small_ev = PR.decisions(synthetic_scores(2, n=240)[1], draw=0, cal_n=60, q=0.05)[1]
    assert small_ev.sum() == 240 - 60                                                                    # under the cap, exactly the 60 calibration messages are left out
    kn, oo = dec[PR.SYSTEMS[4]]
    assert len(kn) == ev.sum() and kn.mean() < 0.2
    assert oo["stories"].mean() > 0.7 and oo["wiki"].mean() > 0.7                                        # either: whichever reader sees the foreign register is enough
    only_pool = dec[PR.SYSTEMS[1]][1]
    assert only_pool["stories"].mean() > 0.7 and only_pool["wiki"].mean() < 0.3                          # a single reader misses what it cannot see
    assert not dec[PR.SYSTEMS[0]][0].any() and not dec[PR.SYSTEMS[2]][1]["wiki"].any()                   # the uncalibrated rules are taken as the readers gave them


def test_the_refusal_experiment_reports_accuracy_on_the_answered_messages_and_averages_over_draws():
    PR, z = synthetic_scores(1)
    z["known_pred"] = z["known_y"].copy()
    wrong = z["known_pool_nov"] > 1.0                                                                       # the pool is wrong exactly where it is least familiar
    z["known_pred"][wrong] = (z["known_y"][wrong] + 1) % 5
    res = PR.evaluate(z, cal_n=60, q=0.05)
    pool = res[PR.SYSTEMS[1]]
    assert pool["accuracy_answered"] > pool["accuracy_all"] and abs(pool["coverage"] + pool["false_refusal"] - 1.0) < 1e-9
    assert set(res[PR.SYSTEMS[4]]["ood_refused"]) == {"stories", "wiki"}
