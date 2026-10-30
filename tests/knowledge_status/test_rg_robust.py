"""Robust calibration against wrongly verified messages (routing_general/robust.py and lab_robust.py, lab G4c): trimming the sample, the second opinion, the contamination modes, the verdicts."""

import sys
from pathlib import Path

import numpy as np
import pytest

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
for p in (_KS, _KS / "routing_general", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import lab_outcome as LO  # noqa: E402
import lab_robust as LR  # noqa: E402
import outcome as OC  # noqa: E402
import robust as RB  # noqa: E402
import test_rg_lab as TL  # noqa: E402
import test_rg_lab_outcome as TLO  # noqa: E402


def test_trim_top_drops_the_k_largest_scores_of_every_reader_and_nothing_else():
    cal = {"a": [5.0, 1.0, 3.0, 2.0, 4.0], "b": [10.0, 30.0, 20.0, 0.0, 40.0]}
    out = RB.trim_top(cal, 2)
    assert sorted(out["a"]) == [1.0, 2.0, 3.0] and sorted(out["b"]) == [0.0, 10.0, 20.0]
    assert sorted(RB.trim_top(cal, 0)["a"]) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert cal["a"] == [5.0, 1.0, 3.0, 2.0, 4.0]                                                  # the input is not changed
    with pytest.raises(ValueError):
        RB.trim_top(cal, 5)                                                                       # nothing would be left to calibrate on
    with pytest.raises(ValueError):
        RB.trim_top(cal, -1)


def test_the_number_of_messages_injected_makes_the_expected_share_of_the_sample():
    assert RB.injected_count(90, 0.10) == 10 and RB.injected_count(40, 0.20) == 10 and RB.injected_count(33, 0.0) == 0
    assert RB.injected_count(0, 0.10) == 0
    with pytest.raises(ValueError):
        RB.injected_count(10, 1.0)


def test_owner_agrees_needs_the_readers_top_card_to_be_one_the_outcome_names():
    top = [0, 1, -1, 2, 1]
    owners = [frozenset({0}), frozenset({0, 1}), frozenset({0}), frozenset({1}), frozenset()]
    assert RB.owner_agrees(top, owners).tolist() == [True, True, False, False, False]            # -1 = the reader said none of these; no named expert = nothing to agree with
    with pytest.raises(ValueError):
        RB.owner_agrees([0], owners)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_a_trimmed_cut_flags_k_plus_one_over_n_plus_one_of_exchangeable_known_messages(k):
    rng = np.random.default_rng(10 + k)
    n, rates = 30, []
    for _ in range(300):
        cal = {"r": rng.normal(0, 1, n)}
        ev = {"known": {"r": rng.normal(0, 1, 2000)}, "unknown": {"r": rng.normal(3, 1, 10)}, "foreign": {"r": rng.normal(6, 1, 10)}}
        rates.append(OC.system_metrics(RB.trim_top(cal, k), ev, ["r"], "either", 0.05)["known_flagged"])
    assert np.mean(rates) == pytest.approx((k + 1) / (n + 1), abs=0.012)


def test_the_either_fusion_of_two_independent_readers_flags_about_twice_what_one_reader_does():
    """After leave-one-out recalibration the fused cut is 'above the best score of some reader', so two independent readers flag about the sum of their rates (a property of G4's fusion, F-53)."""
    rng = np.random.default_rng(4)
    one, two = [], []
    for _ in range(300):
        cal = {"a": rng.normal(0, 1, 30), "b": rng.normal(0, 1, 30)}
        ev = {"known": {"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)}, "unknown": {"a": rng.normal(3, 1, 10), "b": rng.normal(3, 1, 10)}, "foreign": {"a": rng.normal(6, 1, 10), "b": rng.normal(6, 1, 10)}}
        one.append(OC.system_metrics(cal, ev, ["a"], "either", 0.05)["known_flagged"])
        two.append(OC.system_metrics(cal, ev, ["a", "b"], "either", 0.05)["known_flagged"])
    assert np.mean(two) == pytest.approx(1 - (30 / 31) ** 2, abs=0.012) and np.mean(two) > 1.7 * np.mean(one)


def test_trimming_buys_back_the_power_that_wrongly_verified_messages_cost_and_spends_false_flags_for_it():
    rng = np.random.default_rng(3)
    found, kf = {"plain": [], "trim3": []}, {"plain": [], "trim3": []}
    for _ in range(60):
        sample = {"r": np.concatenate([rng.normal(0, 1, 27), rng.normal(3, 1, 3)])}                # three never-taught messages the check wrongly vouched for score like novel ones
        ev = {"known": {"r": rng.normal(0, 1, 400)}, "unknown": {"r": rng.normal(3, 1, 400)}, "foreign": {"r": rng.normal(6, 1, 50)}}
        for name, cal in (("plain", sample), ("trim3", RB.trim_top(sample, 3))):
            m = OC.system_metrics(cal, ev, ["r"], "either", 0.05)
            found[name].append(m["unknown_found"])
            kf[name].append(m["known_flagged"])
    assert np.mean(found["trim3"]) > np.mean(found["plain"]) + 0.3
    assert np.mean(kf["trim3"]) > np.mean(kf["plain"]) + 0.01                                    # the price: the cut is lower, so more known messages cross it


def test_owner_sets_name_the_experts_that_reproduce_the_stated_answer_or_the_label():
    experts = [lambda f: f[0] + f[1], lambda f: f[0] * f[1]]
    v = LO.AnswerVerifier(experts, {"3 plus 4": 7, "3 times 4": 12, "2 times 2": 4, "3 minus 1": 99})
    assert LR.owner_sets(v, ["3 plus 4", "3 times 4", "2 times 2", "3 minus 1", "unseen text"]) == [frozenset({0}), frozenset({1}), frozenset({0, 1}), frozenset(), frozenset()]
    assert LR.owner_sets(LO.TextVerifier(), ["a", "b"], owner=[2, 0]) == [frozenset({2}), frozenset({0})]
    assert LR.owner_sets(LO.TextVerifier(), ["a", "b"]) == [frozenset(), frozenset()]


def synth(seed=0, n_cal=20, n_known=60, n_unk=60, n_for=30, verified_share=0.8, vouch=0.3):
    rng = np.random.default_rng(seed)
    S = {r: {"cal": rng.normal(0, 1, n_cal), "known": rng.normal(0.3, 1, n_known), "unknown": rng.normal(1.5, 1, n_unk), "foreign": rng.normal(3, 1, n_for)} for r in LO.READERS}
    owner = np.repeat([0, 1, 2], n_known // 3)
    verified = rng.random(n_known) < verified_share
    v_unk = rng.random(n_unk) < vouch
    top = {"known": owner.copy(), "unknown": np.full(n_unk, -1)}                                  # the reader is right on known messages and says "none of these" to the never-taught ones
    ko = [frozenset({int(o)}) for o in owner]
    uo = [frozenset() for _ in range(n_unk)]
    return S, owner, verified, v_unk, top, ko, uo


def test_the_plain_calibrator_in_the_real_mode_reproduces_the_g4b_numbers_exactly():
    S, owner, verified, v_unk, top, ko, uo = synth()
    old = LO.contamination_arms(S, owner, verified, v_unk, 0.25, 7, 4, (10, 20), 0.05)
    new = LR.robust_arms(S, owner, verified, v_unk, top, ko, uo, 3, 0.25, 7, 4, (10, 20), 0.05, modes=("real",))
    for nv in ("10", "20"):
        for s in LR.SYSTEMS_C:
            assert new["real"]["results"]["plain"][nv][s]["contaminated"] == old["contaminated"]["verified"][nv][s]
            assert new["real"]["results"]["plain"][nv][s]["clean"] == old["clean"]["verified"][nv][s]
    assert new["real"]["share"]["plain"]["20"] == pytest.approx(old["share_unknown_in_sample"]["20"])


def test_when_nothing_is_wrongly_verified_the_contaminated_and_clean_samples_are_the_same_for_every_calibrator():
    S, owner, verified, _, top, ko, uo = synth()
    new = LR.robust_arms(S, owner, verified, np.zeros(60, dtype=bool), top, ko, uo, 3, 0.25, 7, 4, (10, 20), 0.05, modes=("real",))
    for c in LR.CALIBRATORS:
        for nv in ("10", "20"):
            for s in LR.SYSTEMS_C:
                cell = new["real"]["results"][c][nv].get(s)
                assert cell is None or cell["contaminated"] == cell["clean"]
    assert all(v == 0.0 for per in new["real"]["share"].values() for v in per.values())


def test_the_injected_modes_make_the_asked_share_of_the_sample_wrong_in_every_domain_and_the_second_opinion_keeps_them_out():
    S, owner, verified, v_unk, top, ko, uo = synth(seed=2, n_known=240, n_unk=240, verified_share=1.0, vouch=0.0)
    res = LR.robust_arms(S, owner, verified, v_unk, top, ko, uo, 3, 0.25, 1, 10, (40,), 0.05, modes=("inj10", "inj20"))
    assert res["inj10"]["share"]["plain"]["40"] == pytest.approx(0.10, abs=0.04)
    assert res["inj20"]["share"]["plain"]["40"] == pytest.approx(0.20, abs=0.05)
    assert res["inj20"]["share"]["agree"]["40"] == 0.0                                            # the reader said "none" to every wrong message, so none passes the second opinion
    assert res["inj20"]["n_actual"]["agree"]["40"] == 40.0                                       # and the known messages fill the sample


def test_each_trim_level_lowers_the_cut_of_the_same_sample_so_known_flagged_never_falls_as_k_grows():
    S, owner, verified, v_unk, top, ko, uo = synth(seed=5, n_known=120, n_unk=120, verified_share=1.0, vouch=0.0)
    res = LR.robust_arms(S, owner, verified, v_unk, top, ko, uo, 3, 0.25, 2, 8, (30,), 0.05, modes=("inj10",))["inj10"]["results"]
    for s in LR.SYSTEMS_C:
        kf = [res[c]["30"][s]["clean"]["known_flagged"] for c in ("plain", "trim1", "trim2", "trim3", "trim6")]
        assert kf == sorted(kf) and kf[3] > kf[0]                                                 # the same sample, a lower cut at each level
        found = [res[c]["30"][s]["contaminated"]["unknown_found"] for c in ("plain", "trim1", "trim2", "trim3", "trim6")]
        assert found == sorted(found)


def test_the_second_opinion_is_circular_it_removes_known_messages_the_reader_refuses():
    S, owner, verified, v_unk, top, ko, uo = synth(seed=3, n_known=120, n_unk=120, verified_share=1.0, vouch=0.0)
    top["known"] = np.where(np.arange(120) % 2 == 0, owner, -1)                                   # the reader refuses half of the known messages ...
    S["ask_exemplars"]["known"][1::2] += 3.0                                                      # ... and those are the ones it finds most novel
    res =LR.robust_arms(S, owner, verified, v_unk, top, ko, uo, 3, 0.25, 1, 5, (20,), 0.05, modes=("inj10",))
    assert res["inj10"]["n_actual"]["plain"]["20"] == 20.0 and res["inj10"]["n_actual"]["agree"]["20"] == 20.0
    assert res["inj10"]["results"]["agree"]["20"]["either_pair"]["clean"]["known_flagged"] > res["inj10"]["results"]["plain"]["20"]["either_pair"]["clean"]["known_flagged"]


def test_run_cell_reports_every_mode_calibrator_sample_size_and_system(tmp_path):
    w = TL.make_world()
    rec = LR.run_cell(w, TL.FOREIGN, TLO.Coincidence(True), ask=TL.fake_ask_for(w), encode=TL.fake_encode, cache_dir=tmp_path, seed=1, draws=6, nvs=(6,))
    assert set(rec["modes"]) == set(LR.MODES)
    for mode, m in rec["modes"].items():
        assert set(m["results"]) == set(LR.CALIBRATORS) and set(m["share"]) == set(LR.CALIBRATORS)
        assert set(m["results"]["plain"]["6"]) == set(LR.SYSTEMS_C)
        for s in LR.SYSTEMS_C:
            assert set(m["results"]["plain"]["6"][s]) == {"contaminated", "clean"} and m["results"]["plain"]["6"][s]["clean"]["draws"] == 6
        assert m["results"]["trim6"]["6"] == {}                                                   # a sample of six cannot lose six scores: the cell is empty, not an error
    assert rec["verification"]["coverage"] == 1.0
    assert rec["modes"]["real"]["share"]["plain"]["6"] > 0.0                                      # this check vouches for every never-taught message
    assert rec["modes"]["inj10"]["share"]["plain"]["6"] > 0.0


def cell_metrics(kf, found):
    return {"known_flagged": kf, "unknown_found": found, "foreign_found": 1.0, "auroc_unknown": 0.9, "draws": 20}


def make_row(mode_entries, nvs=("30",)):
    """mode_entries: {mode: {calibrator: {system: (clean_kf, clean_found, cont_kf, cont_found)}}} applied to every nv in nvs."""
    modes = {}
    for mode, cals in mode_entries.items():
        results = {c: {nv: {s: {"clean": cell_metrics(v[0], v[1]), "contaminated": cell_metrics(v[2], v[3])} for s, v in per.items()} for nv in nvs} for c, per in cals.items()}
        modes[mode] = {"results": results, "share": {c: {nv: 0.09 for nv in nvs} for c in cals}, "n_actual": {c: {nv: float(nv) for nv in nvs} for c in cals}}
    return {"modes": modes, "verification": {"coverage": 0.5, "false_verified_unknown": 0.1}, "n": {"known": 1, "unknown": 1, "foreign": 1, "cal": 1}}


def arith_row(trim1=(0.09, 0.40, 0.06, 0.36), trim2=(0.12, 0.45, 0.09, 0.40), agree=(0.13, 0.50, 0.12, 0.49)):
    e = lambda t: {"either_pair": t}
    return make_row({"real": {"plain": e((0.05, 0.25, 0.04, 0.17)), "trim1": e(trim1), "trim2": e(trim2), "trim3": e((0.16, 0.5, 0.12, 0.45)), "agree": e(agree)}})


def test_c11_needs_both_false_flag_limits_and_85_percent_of_the_calibrators_own_power():
    v = LR.c11([arith_row()])
    assert v["pass"] and v["cells"]["trim1"]["pass"] and not v["cells"]["trim2"]["pass"]
    assert v["cells"]["trim1"]["retention"] == pytest.approx(0.9) and v["plain_retention"] == pytest.approx(0.17 / 0.25)
    assert not LR.c11([arith_row(trim1=(0.09, 0.40, 0.06, 0.32))])["pass"]                           # keeps only 0.80 of its power, agree flags too many
    assert not LR.c11([arith_row(trim1=(0.11, 0.40, 0.06, 0.38))])["pass"]                           # clean false flags above 0.10
    assert not LR.c11([arith_row(trim1=(0.09, 0.40, 0.11, 0.38))])["pass"]                           # contaminated false flags above 0.10
    assert LR.c11([arith_row(trim1=(0.12, 0.4, 0.06, 0.38), agree=(0.09, 0.50, 0.08, 0.47))])["pass"]      # agree can carry it


def text_row(kf_by_k, n="30", singles=("ask_exemplars", "vmf_conformal")):
    cals = {LR.CAL_FOR_K[k]: {s: (kf, 0.3, kf, 0.3) for s in singles} for k, kf in kf_by_k.items()}
    return make_row({"real": cals}, nvs=(n,))


def test_c12_compares_the_single_readers_false_flags_with_k_plus_one_over_n_plus_one():
    theory30 = {k: (k + 1) / 31 for k in range(4)}
    rows = {d: [text_row(theory30)] for d in ("banking", "agnews", "newsgroups")}
    assert LR.c12(rows)["pass"]
    off = {d: [text_row({**theory30, 2: 0.2})] for d in ("banking", "agnews", "newsgroups")}                  # k = 2 flags 0.2 instead of 0.097
    v = LR.c12(off)
    assert not v["pass"] and not v["cells"]["30,2"]["ok"] and v["cells"]["30,1"]["ok"]
    assert LR.c12({"arithmetic": [text_row(theory30)]})["cells"] == {}                                        # arithmetic is not a text domain: nothing to judge


def test_c13_needs_one_calibrator_to_pass_the_c11_test_in_every_domain():
    ok = {"plain": {"either_pair": (0.05, 0.6, 0.04, 0.3)}, "trim1": {"either_pair": (0.09, 0.6, 0.08, 0.55)}}
    bad = {"plain": {"either_pair": (0.05, 0.6, 0.04, 0.3)}, "trim1": {"either_pair": (0.13, 0.6, 0.12, 0.58)}}
    fine = lambda c: make_row({"inj10": c})
    assert LR.c13({"a": [fine(ok)], "b": [fine(ok)]})["pass"]
    v = LR.c13({"a": [fine(ok)], "b": [fine(bad)]})
    assert not v["pass"] and v["cells"]["trim1"] == {"a": True, "b": False} and v["cells"]["plain"] == {"a": False, "b": False}


def test_the_report_has_every_domain_the_three_verdicts_and_the_calibrator_rows():
    per = {"arithmetic": [arith_row()], "banking": [make_row({"inj10": {"plain": {"either_pair": (0.05, 0.7, 0.04, 0.4)}}})]}
    text = LR.report(per, [150])
    assert "## arithmetic" in text and "## banking" in text and "| trim1 |" in text and "| agree |" in text
    assert "C11" in text and "C12" in text and "C13" in text
