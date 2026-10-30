"""The router (routing_general/router.py): the domain-free routing and novelty layer as one object a pool can use, with stand-ins for the language model and the encoder."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
for p in (_KS, _KS / "routing_general", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import domains as DM  # noqa: E402
import outcome as OC  # noqa: E402
import router as RO  # noqa: E402
import test_rg_lab as TL  # noqa: E402


class NameReader:
    """A stand-in for the card reader: certain of the card whose underscored name occurs in the message, else certain of "none of these"."""

    def __init__(self, cards):
        self.keys = [c.name.replace(" ", "_") for c in cards]

    def probabilities(self, texts):
        k = len(self.keys)
        out = np.full((len(texts), k + 1), 0.01)
        for i, t in enumerate(texts):
            hit = [j for j, key in enumerate(self.keys) if key in t]
            out[i, hit[0] if hit else k] = 0.98
        return out / out.sum(axis=1, keepdims=True)


class NoisyReader:
    """Deterministic pseudo-random probabilities per message (so that scores vary), with a lean towards the card named in the message."""

    def __init__(self, cards):
        self.keys = [c.name.replace(" ", "_") for c in cards]

    def probabilities(self, texts):
        k = len(self.keys)
        out = np.zeros((len(texts), k + 1))
        for i, t in enumerate(texts):
            rng = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
            logits = rng.normal(0, 1, k + 1)
            hit = [j for j, key in enumerate(self.keys) if key in t]
            logits[hit[0] if hit else k] += 2.0
            e = np.exp(logits - logits.max())
            out[i] = e / e.sum()
        return out


def stream_world():
    """Two taught classes, 40 known messages in the stream's own wording (``test``; the teaching wording is ``train``), 20 never-taught ones."""
    return DM.build_labelled_world("toy", TL.table(40, "train"), TL.table(20, "test"), TL.NAMES, n_taught=2, n_teach=10, n_cal=10, n_known=40, n_unknown=20, seed=0)


def make_router(reader=NameReader, **kw):
    w = stream_world()
    return w, RO.Router(w.taught, w.teach, TL.fake_encode, reader, **kw)


def texts(pairs):
    return [t for t, _ in pairs]


def test_before_any_outcome_the_router_uses_the_readers_own_claim_rule_and_says_so():
    w, r = make_router()
    out = r.route(texts(w.known_test) + list(w.unknown_test))
    n = len(w.known_test)
    assert all(x.status == "claimed" and x.mode == "uncalibrated" for x in out[:n])
    assert all(x.status == "unclaimed" and x.mode == "uncalibrated" and x.flagged_by == ("claim rule",) for x in out[n:])
    assert [x.expert for x in out[:n]] == [i for _, i in w.known_test] and all(x.name == w.taught[x.expert] and x.team[0] == x.expert and len(x.team) <= 3 for x in out[:n])
    assert r.status() == {"experts": 2, "verified": 0, "needed": 19, "mode": "uncalibrated"}
    assert r.route([]) == []


def test_nineteen_verified_messages_switch_it_to_the_calibrated_either_fusion_and_eighteen_do_not():
    w, r = make_router()
    stream = texts(w.known_test)
    r.observe(stream[:18])
    assert r.status()["mode"] == "uncalibrated" and r.status()["verified"] == 18
    r.observe(stream[18:25])
    assert r.status()["mode"] == "calibrated" and r.status()["verified"] == 25
    held = stream[25:]
    out = r.route(held + list(w.unknown_test) + TL.FOREIGN)
    assert all(x.mode == "calibrated" for x in out)
    n = len(held)
    assert np.mean([x.status == "unclaimed" for x in out[:n]]) <= 0.10
    assert all(x.status == "unclaimed" and x.flagged_by for x in out[n:])                                  # never-taught and foreign messages are refused, and the readers that said so are named


def test_the_wording_shift_the_toy_world_has_is_what_calibrating_on_the_streams_outcomes_fixes():
    w, r = make_router(initial_cal=texts(w_cal := stream_world().known_cal))                            # held-out messages in the TEACHING wording
    stream = texts(w.known_test)
    shifted = np.mean([x.status == "unclaimed" for x in r.route(stream)])
    r.observe(stream[:25])
    adapted = np.mean([x.status == "unclaimed" for x in r.route(stream[25:])])
    assert shifted > 0.3 and adapted <= 0.10                                                              # the teaching-wording calibration refuses the stream; the stream's own outcomes do not


def test_the_router_does_exactly_what_the_lab_measured_the_calibrated_either_fusion_of_the_asking_reader_and_the_density_gate():
    w, r = make_router(reader=NoisyReader)
    known = texts(w.known_test)
    sample, held = known[:25], known[25:]
    r.observe(sample)
    cal = {n: r.scores(sample)[n] for n in RO.READERS}
    ev = {"known": r.scores(held), "unknown": r.scores(list(w.unknown_test)), "foreign": r.scores(TL.FOREIGN)}
    lab = OC.system_metrics(cal, ev, list(RO.READERS), "either", r.q)
    flagged = lambda ts: float(np.mean([x.status == "unclaimed" for x in r.route(ts)]))
    assert flagged(held) == pytest.approx(lab["known_flagged"]) and flagged(list(w.unknown_test)) == pytest.approx(lab["unknown_found"])
    assert flagged(TL.FOREIGN) == pytest.approx(lab["foreign_found"])
    assert 0.0 < lab["unknown_found"] < 1.0                                                              # the noisy reader makes the comparison non-trivial


def test_an_initial_calibration_in_the_teaching_wording_serves_until_verified_outcomes_replace_it():
    w, r = make_router(initial_cal=texts(w_cal := stream_world().known_cal))
    assert r.status()["mode"] == "initial"
    assert {x.mode for x in r.route(texts(w_cal))} == {"initial"}
    r.observe(texts(w_cal)[:19])
    assert r.status()["mode"] == "calibrated"                                                            # the stream's own outcomes take over completely
    _, small = make_router(initial_cal=texts(w_cal)[:5])
    assert small.status()["mode"] == "uncalibrated"                                                      # too few to flag anything: the claim rule is used instead of silence


def test_the_verified_pool_is_a_window_of_the_latest_outcomes():
    w, r = make_router(window=25)
    r.observe(texts(w.known_cal) + texts(w.known_test))
    assert r.status()["verified"] == 25
    assert r._verified == (texts(w.known_cal) + texts(w.known_test))[-25:]


def test_a_claim_carries_the_state_of_the_expert_so_a_thin_experts_answer_can_be_told_from_a_solid_ones():
    w = stream_world()
    r = RO.Router(w.taught, w.teach, TL.fake_encode, NameReader, states=["solid", "thin"])
    out = r.route(texts(w.known_test) + [w.unknown_test[0]])
    assert {x.state for x in out[:-1] if x.expert == 0} == {"solid"} and {x.state for x in out[:-1] if x.expert == 1} == {"thin"}
    assert out[-1].status == "unclaimed" and out[-1].state is None
    assert {x.state for x in make_router()[1].route(texts(w.known_test))} == {"solid"}                   # the default is solid
    with pytest.raises(ValueError):
        RO.Router(w.taught, w.teach, TL.fake_encode, NameReader, states=["solid"])                       # one state per expert


def test_respond_gives_the_claimed_experts_answer_and_says_not_mine_otherwise():
    w = TL.make_world()
    experts = [lambda f, i=i: 100 + i + f[0] for i in range(2)]
    featurize = lambda t: None if "unreadable" in t else (len(t) % 7,)
    r = RO.Router(w.taught, w.teach, TL.fake_encode, NameReader, experts=experts, featurize=featurize)
    known = texts(w.known_test)[:3]
    out = r.respond(known + [w.unknown_test[0], known[0] + " unreadable"])
    for t, x, (_, i) in zip(known, out[:3], w.known_test[:3]):
        assert x.reason == "answered" and x.value == 100 + i + len(t) % 7 and x.route.expert == i
    assert out[3].reason == "not mine" and out[3].value is None and out[3].route.status == "unclaimed"
    assert out[4].reason == "unreadable" and out[4].value is None and out[4].route.status == "claimed"
    with pytest.raises(ValueError):
        make_router()[1].respond(["x"])                                                                  # no experts attached


def test_a_new_expert_changes_the_roster_keeps_the_verified_outcomes_and_takes_its_messages():
    w, r = make_router()
    r.observe(texts(w.known_test)[:25])
    new = [f"test message {j} about lost_card please" for j in range(10)]
    ask = "test message 11 about lost_card please"
    before = r.route([ask])[0]
    assert before.status == "unclaimed"                                                                  # nobody taught it yet
    r.add_expert("lost card", new)
    assert r.status() == {"experts": 3, "verified": 25, "needed": 19, "mode": "calibrated"}
    after = r.route([ask])[0]
    assert after.status == "claimed" and after.expert == 2 and after.name == "lost card"
    with pytest.raises(ValueError):
        r.add_expert("lost card", new)                                                                   # names stay distinct


def test_the_router_refuses_a_roster_that_does_not_match_its_teaching_messages():
    w = TL.make_world()
    with pytest.raises(ValueError):
        RO.Router(w.taught + ["not taught"], w.teach, TL.fake_encode, NameReader)
    with pytest.raises(ValueError):
        RO.Router(w.taught, {n: v[:1] for n, v in w.teach.items()}, TL.fake_encode, NameReader)           # a card needs its exemplars and the density gate needs messages
