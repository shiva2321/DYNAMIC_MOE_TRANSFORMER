"""Domain worlds for the general routing lab (routing_general/domains.py): a pure builder over labelled messages, and loaders of the cached datasets."""

import sys
from pathlib import Path

import pytest

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
sys.path.insert(0, str(_KS / "routing_general"))

import domains as DM  # noqa: E402

NAMES = ["card_arrival", "age_limit", "atm_support", "top_up", "exchange_rate", "lost_card"]


def table(n_each, tag):
    texts, labels = [], []
    for c in range(len(NAMES)):
        for j in range(n_each):
            texts.append(f"{tag} message {j} about {NAMES[c]}")
            labels.append(c)
    return texts, labels


def world(seed=0, **kw):
    tr = table(30, "train")
    te = table(20, "test")
    args = dict(name="toy", teach=tr, test=te, label_names=NAMES, n_taught=2, n_teach=10, n_cal=5, n_known=12, n_unknown=15, seed=seed)
    args.update(kw)
    return DM.build_labelled_world(**args)


def test_taught_unknown_known_are_consistent_and_disjoint():
    w = world()
    assert len(w.taught) == 2 and len(set(w.taught)) == 2
    assert all(len(w.teach[n]) == 10 for n in w.taught)
    assert all(t.startswith("train") for n in w.taught for t in w.teach[n])
    taught_labels = {n.replace(" ", "_") for n in w.taught}
    assert all(any(f"about {n}" in t for n in taught_labels) for t, _ in w.known_test)
    assert len(w.known_test) == 12 and all(0 <= i < 2 for _, i in w.known_test)
    assert all(t.startswith("test") for t, _ in w.known_test)
    assert len(w.unknown_test) == 15
    assert not any(f"about {n}" in t for t in w.unknown_test for n in taught_labels)          # a never-taught message is never of a taught class
    assert w.states == ["solid", "solid"]


def test_calibration_messages_are_held_out_from_teaching_and_test():
    w = world()
    cal = [t for t, _ in w.known_cal]
    assert len(cal) == 2 * 5
    teach = {t for n in w.taught for t in w.teach[n]}
    assert not teach & set(cal)
    assert not set(cal) & {t for t, _ in w.known_test}


def test_exemplars_are_the_first_teaching_messages_and_names_are_readable():
    w = world()
    for i, n in enumerate(w.taught):
        assert "_" not in n
        assert w.cards[i].name == n
        assert w.cards[i].exemplars == tuple(w.teach[n][:DM.N_EXEMPLARS])


def test_same_seed_same_world_other_seed_other_world():
    a, b, c = world(seed=1), world(seed=1), world(seed=2)
    assert a.taught == b.taught and a.unknown_test == b.unknown_test and a.known_test == b.known_test
    assert (a.taught, a.unknown_test) != (c.taught, c.unknown_test)


def test_roster_order_is_shuffled_by_seed_not_alphabetical_by_construction():
    orders = {tuple(world(seed=s, n_taught=4).taught) for s in range(8)}
    assert len(orders) > 1


def test_classes_too_small_to_teach_are_never_drawn_as_taught():
    texts, labels = table(30, "train")
    keep = [i for i, c in enumerate(labels) if c != 0 or i % 30 < 5]                          # class 0 keeps only 5 teaching messages
    small = ([texts[i] for i in keep], [labels[i] for i in keep])
    for seed in range(12):
        w = world(seed=seed, teach=small, n_taught=3)
        assert "card arrival" not in w.taught
        assert not any("about card_arrival" in t for t, _ in w.known_test)
    # its test messages are still legitimate never-taught messages
    assert any("about card_arrival" in t for s in range(12) for t in world(seed=s, teach=small, n_taught=3, n_unknown=60).unknown_test)
    assert len(world(teach=small, n_taught=len(NAMES) - 1).taught) == len(NAMES) - 1         # exactly five classes can be taught, and they are
    keep2 = [i for i, c in enumerate(labels) if c > 1 or i % 30 < 5]                         # classes 0 and 1 are both too small
    with pytest.raises(ValueError):
        world(teach=([texts[i] for i in keep2], [labels[i] for i in keep2]), n_taught=len(NAMES) - 1)


def test_loud_errors():
    with pytest.raises(ValueError):
        world(n_taught=len(NAMES))                                                           # nothing left to be never-taught
    with pytest.raises(ValueError):
        world(n_teach=30)                                                                    # no messages left for calibration
    with pytest.raises(ValueError):
        world(n_unknown=10 ** 6)


def test_loaders_read_the_local_cache_only_and_give_label_names():
    for loader, n_names in ((DM.load_banking, 77), (DM.load_agnews, 4), (DM.load_newsgroups, 20)):
        try:
            teach, test, names = loader()
        except FileNotFoundError:
            pytest.skip("dataset not in the local cache")
        assert len(names) == n_names
        assert len(teach[0]) == len(teach[1]) > 100 and len(test[0]) == len(test[1]) > 100
        assert set(teach[1]) <= set(range(n_names))
