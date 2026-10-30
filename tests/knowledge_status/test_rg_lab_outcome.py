"""The outcome lab (routing_general/lab_outcome.py): verifiers, the cell with stand-ins for the model and the encoder, and the verdict rules."""

import sys
from pathlib import Path

import numpy as np
import pytest

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
for p in (_KS, _KS / "routing_general", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import lab_outcome as LO  # noqa: E402
import test_rg_lab as TL  # noqa: E402


def test_text_verifier_trusts_labels_and_answer_verifier_trusts_a_taught_expert_reproducing_the_stated_answer():
    t = LO.TextVerifier()
    assert t.known(["a", "b"]).tolist() == [True, True] and t.unowned(["a", "b"]).tolist() == [False, False]
    experts = [lambda f: f[0] + f[1], lambda f: f[0] * f[1]]
    stated = {"3 plus 4": 7, "3 times 4": 12, "3 minus 1": 99, "no numbers here": 5}
    v = LO.AnswerVerifier(experts, stated)
    assert v.known(["3 plus 4", "3 times 4", "3 minus 1", "no numbers here", "unknown text"]).tolist() == [True, True, False, False, False]
    assert v.unowned(["3 plus 4"]).tolist() == [True]                                         # the same rule: a coincidence would vouch for a message nobody owns


def test_run_cell_reports_every_arm_system_and_sample_size_and_the_verification_rates(tmp_path):
    w = TL.make_world()
    rec = LO.run_cell(w, TL.FOREIGN, LO.TextVerifier(), ask=TL.fake_ask_for(w), encode=TL.fake_encode, cache_dir=tmp_path, seed=1, draws=2, nvs=(4, 8))
    assert set(rec["results"]) == set(LO.ARMS)
    for arm in LO.ARMS:
        assert set(rec["results"][arm]) == {"4", "8"}
        for nv in ("4", "8"):
            assert set(rec["results"][arm][nv]) == set(LO.SYSTEMS)
            assert all(0.0 <= m["known_flagged"] <= 1.0 and m["draws"] == 2 for m in rec["results"][arm][nv].values())
    assert rec["verification"]["coverage"] == 1.0 and rec["verification"]["false_verified_unknown"] == 0.0
    assert rec["n_actual"] == {"4": 4.0, "8": 8.0}                                            # ten messages in the pool: both sizes are available


def test_a_pool_with_nothing_verified_skips_the_calibrated_arms_and_says_so(tmp_path):
    class Nothing:
        def known(self, texts):
            return np.zeros(len(texts), dtype=bool)

        def unowned(self, texts):
            return np.zeros(len(texts), dtype=bool)
    w = TL.make_world()
    rec = LO.run_cell(w, TL.FOREIGN, Nothing(), ask=TL.fake_ask_for(w), encode=TL.fake_encode, cache_dir=tmp_path, seed=1, draws=2, nvs=(4,))
    assert rec["n_actual"] == {"4": 0.0} and rec["results"]["verified"]["4"] == {} and rec["results"]["union"]["4"] == {} and rec["results"]["fixed"]["4"]


def cell(fixed_kf, ver_kf, ver_found, singles):
    m = lambda kf, f: {"known_flagged": kf, "unknown_found": f, "auroc_unknown": 0.9, "foreign_found": 1.0}
    per = lambda arm_kf, arm_found: {s: m(arm_kf, singles.get(s, arm_found)) for s in LO.SYSTEMS}
    ver = per(ver_kf, ver_found)
    return {"results": {"fixed": {"30": per(fixed_kf, 0.8)}, "verified": {"30": ver}, "union": {"30": ver}}, "n_actual": {"30": 30.0},
            "verification": {k: 0.5 for k in ("coverage", "coverage_solid", "coverage_thin", "false_verified_unknown", "false_verified_other")}, "n": {"known": 1, "unknown": 1, "foreign": 1, "cal": 1}}


def test_the_verdicts_pass_when_every_domain_keeps_its_false_flags_and_its_power_and_fail_for_the_declared_reasons():
    good = {"arithmetic": [cell(0.36, 0.06, 0.55, {"ask_exemplars": 0.58})], "banking": [cell(0.07, 0.05, 0.80, {"vmf_conformal": 0.84})]}
    v = LO.verdicts(good)
    assert v["C8_pass"] and v["C9"]["pass"]
    leaky = {"arithmetic": [cell(0.36, 0.14, 0.55, {})]}                                         # still flags 14 % of known messages
    assert not LO.verdicts(leaky)["C8"]["arithmetic"]["pass"] and not LO.verdicts(leaky)["C9"]["pass"]
    weak = {"banking": [cell(0.07, 0.05, 0.50, {"vmf_conformal": 0.84})]}                        # the fusion finds 0.34 less than the best reader alone
    assert not LO.verdicts(weak)["C8_pass"]
    assert LO.verdicts({"banking": good["banking"]})["C9"] is None


def test_the_report_has_every_block_and_the_verdict_lines():
    text = LO.report({"banking": [cell(0.07, 0.05, 0.8, {})], "arithmetic": [cell(0.36, 0.06, 0.5, {})]}, [150])
    assert "## banking" in text and "## arithmetic" in text and "How many verified messages are needed" in text and "C8 overall" in text and "C9 (sanity" in text
    assert "| either_pair |" in text and "| ask_exemplars |" in text


class Coincidence:
    """Vouches for every never-taught message (the worst case of the answer check) or for none."""

    def __init__(self, vouch):
        self.vouch = vouch

    def known(self, texts):
        return np.ones(len(texts), dtype=bool)

    def unowned(self, texts):
        return np.full(len(texts), self.vouch, dtype=bool)


def test_contamination_puts_wrongly_verified_never_taught_messages_into_the_calibration_sample(tmp_path):
    w = TL.make_world()
    kw = dict(ask=TL.fake_ask_for(w), encode=TL.fake_encode, cache_dir=tmp_path, seed=1, draws=3, nvs=(6,), contaminate=0.25)
    bad = LO.run_cell(w, TL.FOREIGN, Coincidence(True), **kw)
    good = LO.run_cell(w, TL.FOREIGN, Coincidence(False), **kw)
    assert bad["contaminated"]["u"] == 0.25 and bad["contaminated"]["share_unknown_in_sample"]["6"] > 0.1
    assert good["contaminated"]["share_unknown_in_sample"]["6"] == 0.0
    for mode in ("contaminated", "clean"):
        assert set(bad["contaminated"][mode]) == {"verified", "union"}
        assert set(bad["contaminated"][mode]["verified"]["6"]) == set(LO.SYSTEMS)
    assert good["contaminated"]["contaminated"] == good["contaminated"]["clean"]                        # nothing wrongly verified: the paired samples are the same samples, so the numbers are equal
    assert bad["contaminated"]["contaminated"] != bad["contaminated"]["clean"]
    plain = LO.run_cell(w, TL.FOREIGN, Coincidence(True), **{**kw, "contaminate": 0.0})
    assert "contaminated" not in plain                                                          # the default run is the one of G4, unchanged


def contaminated(clean_found, dirty_found, dirty_kf=0.04):
    m = lambda f, kf: {"known_flagged": kf, "unknown_found": f, "auroc_unknown": 0.9, "foreign_found": 1.0, "draws": 5}
    one = lambda f, kf: {"verified": {"30": {s: m(f, kf) for s in LO.SYSTEMS}}, "union": {"30": {s: m(f, kf) for s in LO.SYSTEMS}}}
    return {"contaminated": {"u": 0.25, "share_unknown_in_sample": {"30": 0.09}, "contaminated": one(dirty_found, dirty_kf), "clean": one(clean_found, 0.05)}}


def test_the_contamination_verdict_needs_two_thirds_of_the_power_and_the_false_flag_limit():
    assert LO.contamination_verdict([contaminated(0.30, 0.25)])["C10_pass"]
    assert not LO.contamination_verdict([contaminated(0.30, 0.15)])["C10_pass"]                  # lost more than a third of the power
    assert not LO.contamination_verdict([contaminated(0.30, 0.25, dirty_kf=0.14)])["C10_pass"]    # or flags too many known messages
    v = LO.contamination_verdict([contaminated(0.30, 0.25), contaminated(0.40, 0.20)])
    assert v["clean_found"] == pytest.approx(0.35) and v["contaminated_found"] == pytest.approx(0.225) and v["share_unknown_in_sample"] == pytest.approx(0.09)


def test_the_report_shows_the_contamination_table_and_the_c10_line_when_the_cells_have_it():
    arith = {**cell(0.36, 0.06, 0.5, {}), **contaminated(0.30, 0.25)}
    text = LO.report({"arithmetic": [arith]}, [150])
    assert "Contamination (G4b)" in text and "C10 (arithmetic, contaminated stream)" in text and "| either_pair | 0.050 | 0.300 | 0.040 | 0.250 |" in text
