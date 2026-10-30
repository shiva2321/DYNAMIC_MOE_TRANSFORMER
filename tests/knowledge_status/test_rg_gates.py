"""Gates and the gated pool (routing_general/gates.py, pool.py): the pool's own vMF gate, the general layer's asking gate, a mix, and the container that puts experts behind a gate."""

import sys
from pathlib import Path

import numpy as np
import pytest

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
for p in (_KS, _KS / "routing_general", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import cards as CD  # noqa: E402
import gates as GT  # noqa: E402
import pool as PL  # noqa: E402


def clusters(n_per=60, d=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.eye(d)[:3] * 5.0
    X = np.vstack([c + rng.normal(0, 0.5, (n_per, d)) for c in centers])
    return X, np.repeat(np.arange(3), n_per), rng


def test_vmf_gate_routes_to_the_nearest_taught_cluster_and_abstains_far_away():
    X, y, rng = clusters()
    cal = np.vstack([c + rng.normal(0, 0.5, (30, 16)) for c in np.eye(16)[:3] * 5.0])
    g = GT.VmfGate(q=0.05).fit(X, y, cal)
    test = np.vstack([np.eye(16)[1] * 5.0 + rng.normal(0, 0.5, (20, 16)), np.eye(16)[9] * 5.0 + rng.normal(0, 0.5, (20, 16))])      # a taught cluster, then an untaught direction
    route, abstain = g.decide_vectors(test)
    assert (route[:20] == 1).mean() >= 0.9 and abstain[:20].mean() <= 0.2
    assert abstain[20:].mean() >= 0.9


class FakeReader:
    def __init__(self, rows):
        self.rows, self.seen = rows, []

    def probabilities(self, texts):
        self.seen.append(list(texts))
        return np.stack([self.rows[t] for t in texts])


def test_ask_gate_claims_the_best_card_and_abstains_when_unclaimed():
    rows = {"a": np.array([0.9, 0.05, 0.05]), "b": np.array([0.1, 0.8, 0.1]), "c": np.array([0.2, 0.2, 0.6]), "d": np.array([0.4, 0.3, 0.3])}
    g = GT.AskGate(FakeReader(rows), states=["solid", "solid"], tau=0.5)
    route, abstain = g.decide(["a", "b", "c", "d"])
    assert route.tolist()[:2] == [0, 1] and abstain.tolist() == [False, False, True, True]                  # c: "none" on top, d: best card below tau
    assert g.decide([])[0].shape == (0,)


def test_mix_gate_abstains_when_either_gate_does_and_routes_by_the_first():
    class G:
        def __init__(self, route, abstain):
            self.r, self.a = np.array(route), np.array(abstain)

        def decide(self, texts):
            return self.r, self.a
    mix = GT.MixGate(G([0, 1, 2, 0], [False, False, True, False]), G([2, 2, 2, 2], [False, True, False, False]))
    route, abstain = mix.decide(["w", "x", "y", "z"])
    assert route.tolist() == [0, 1, 2, 0] and abstain.tolist() == [False, True, True, False]


class Scripted:
    def __init__(self, route, abstain):
        self.r, self.a = np.array(route), np.array(abstain)

    def decide(self, texts):
        return self.r, self.a


def test_pool_answers_through_the_gate_and_says_why_it_did_not():
    experts = [lambda f: f[0] + f[1], lambda f: f[0] - f[1]]
    feats = {"t1": (7, 3), "t2": (7, 3), "t3": (7, 3), "t4": None}
    pool = PL.GatedPool(["add", "sub"], experts, lambda t: feats[t], Scripted([0, 1, 0, 1], [False, False, True, False]))
    out = pool.predict(["t1", "t2", "t3", "t4"])
    assert [(a.value, a.expert, a.reason) for a in out] == [(10, 0, "answered"), (4, 1, "answered"), (None, None, "abstained"), (None, 1, "unreadable")]
    assert pool.summary() == {"n_experts": 2, "names": ["add", "sub"], "gate": "Scripted"}
    with pytest.raises(ValueError):
        PL.GatedPool(["add"], experts, lambda t: None, Scripted([0], [False]))


def test_the_roster_of_a_pool_is_the_deck_of_cards_the_asking_gate_needs():
    pool = PL.GatedPool(["add", "sub"], [lambda f: 0, lambda f: 0], lambda t: None, Scripted([0], [False]))
    cards = pool.cards({"add": ["1 + 2", "3 + 4", "5 + 6"], "sub": ["9 - 1"]})
    assert [c.name for c in cards] == ["add", "sub"] and cards[0].exemplars == ("1 + 2", "3 + 4") and cards[1].exemplars == ("9 - 1",)
    assert isinstance(cards[0], CD.Card)
