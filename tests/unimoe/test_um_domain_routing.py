"""Domain-level routing (unimoe/domain_routing.py): which domain does a message belong to, by the pool's own routing, by a centroid rule, by the asking reader of the Router layer."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_um_suite as TS  # noqa: E402

from unimoe import domain_routing as DR  # noqa: E402
from unimoe import runner as RN  # noqa: E402
from unimoe import suite as SU  # noqa: E402

TEXT = ("banking", "agnews", "svamp")


def world():
    doms = TS.toy_domains()
    for n in TEXT:
        d = doms[n]
        d.meta["texts_train"] = [f"{n} train {i}" for i in range(len(d.x_train))]
        d.meta["texts_test"] = [f"{n} test {i}" for i in range(len(d.x_test))]
    s = SU.build_suite("seq", doms, 0, TS.toy_ood(), n_visit=100, n_visit_arith=300, n_test=40, revisit_k=1)
    return doms, s


class FakeReader:
    """Certain of the card whose domain name occurs in the message; otherwise certain of 'none of these'."""

    def __init__(self, names):
        self.names = names

    def probabilities(self, texts):
        out = np.full((len(texts), len(self.names) + 1), 0.01)
        for i, t in enumerate(texts):
            hit = [j for j, n in enumerate(self.names) if n in t]
            out[i, hit[0] if hit else len(self.names)] = 0.9
        return out / out.sum(axis=1, keepdims=True)


def test_an_expert_belongs_to_the_domain_it_trained_on_most_and_the_pool_is_scored_by_it():
    doms, s = world()
    regimes = s.regimes
    dom_of = {r: reg.domain for r, reg in regimes.items()}
    text_regimes = [r for r in regimes if dom_of[r] in TEXT]
    train_reg = np.concatenate([[r] * 10 for r in text_regimes])
    assigned = np.concatenate([[0] * 10 if dom_of[r] == "banking" else [1] * 10 for r in text_regimes])
    test_reg = np.concatenate([[r] * 3 for r in text_regimes])
    test_exp = np.concatenate([[0] * 3 if dom_of[r] == "banking" else [1] * 3 for r in text_regimes])
    from test_um_analysis import record
    rec = record([[0]], train={"regime": train_reg, "assigned": assigned}, test={"regime": test_reg, "expert": test_exp})
    ed = DR.expert_domains(rec, regimes)
    assert ed[0] == "banking" and ed[1] in ("agnews", "svamp")
    acc = DR.pool_domain_routing(rec, regimes, TEXT)
    assert acc["per_domain"]["banking"] == 1.0 and acc["per_domain"]["agnews"] + acc["per_domain"]["svamp"] == 1.0         # expert 1 serves one of the two other domains
    assert 0.0 < acc["accuracy"] < 1.0


def test_the_centroid_rule_sends_a_message_to_the_nearest_domain_centroid_of_the_training_data():
    doms, s = world()
    acc = DR.centroid_domain_routing(s, TEXT)
    assert set(acc["per_domain"]) == set(TEXT) and 0.0 <= acc["accuracy"] <= 1.0
    for ph in s.phases:                                                                        # make the domains separable: every vector of a domain points to its own axis
        for i, r in enumerate(ph.regime.tolist()):
            ph.x[i] = torch.nn.functional.one_hot(torch.tensor(list(doms).index(s.regimes[r].domain)), 128).float()
    for reg in s.regimes.values():
        reg.x_te = torch.stack([torch.nn.functional.one_hot(torch.tensor(list(doms).index(reg.domain)), 128).float()] * len(reg.y_te))
    assert DR.centroid_domain_routing(s, TEXT)["accuracy"] == 1.0


def test_the_asking_reader_picks_a_domain_card_from_the_message_and_a_none_answer_counts_as_a_miss():
    doms, s = world()
    names = list(TEXT)
    cards_reader = FakeReader(names)
    acc = DR.asking_domain_routing(s, doms, cards_reader, names, per_regime=20)
    assert acc["accuracy"] == 1.0 and acc["refused"] == 0.0 and set(acc["per_domain"]) == set(names)
    bad = DR.asking_domain_routing(s, doms, FakeReader(["nothing", "here", "matches"]), names, per_regime=20)
    assert bad["accuracy"] == 0.0 and bad["refused"] == 1.0                                    # a reader that says none to everything routes nothing


def test_the_cards_are_built_from_the_teachers_names_and_two_training_messages():
    doms, _ = world()
    cards = DR.domain_cards(doms, {"banking": "bank customers' requests", "agnews": "news stories", "svamp": "word problems"}, TEXT)
    assert [c.name for c in cards] == ["bank customers' requests", "news stories", "word problems"]
    assert all(len(c.exemplars) == 2 for c in cards) and cards[0].exemplars[0] == "banking train 0"


def test_the_ood_texts_are_exactly_the_messages_whose_vectors_the_suite_uses():
    from pathlib import Path
    from unimoe import domains as DM
    if not (Path.home() / ".cache" / "huggingface" / "hub" / "datasets--roneneldan--TinyStories").exists():
        pytest.skip("local Hugging Face cache not present")
    from test_um_domains import FakeFE
    fe = FakeFE()
    texts = DR.ood_texts(seed=3, n=40)
    vecs = DM.ood_sets(fe, seed=3, n=40)
    assert set(texts) == set(vecs)
    for name in texts:
        assert len(texts[name]) == 40 and torch.equal(fe.text(texts[name]), vecs[name])


def test_a_logistic_regression_domain_classifier_is_scored_on_the_text_domains_test_sets():
    doms, s = world()
    acc = DR.classifier_domain_routing(s, TEXT)
    assert set(acc["per_domain"]) == set(TEXT) and 0.0 <= acc["accuracy"] <= 1.0
    for ph in s.phases:
        for i, r in enumerate(ph.regime.tolist()):
            ph.x[i] = torch.nn.functional.one_hot(torch.tensor(list(doms).index(s.regimes[r].domain)), 128).float()
    for reg in s.regimes.values():
        reg.x_te = torch.stack([torch.nn.functional.one_hot(torch.tensor(list(doms).index(reg.domain)), 128).float()] * len(reg.y_te))
    assert DR.classifier_domain_routing(s, TEXT)["accuracy"] == 1.0
