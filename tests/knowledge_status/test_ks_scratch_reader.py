"""The from-scratch reader arms (Task 11b): features, the guarded map, the label-free pre-training and its cache. Tiny CPU sizes."""

import numpy as np
import pytest

import ceiling
import scratch_reader as SR


def test_the_bag_representation_is_deterministic_unit_length_and_sees_wording():
    rep = SR.BagRep()
    a = rep.encode(["17 plus 5", "17 plus 5", "seventeen times five", "17 plus 6"])
    assert a.shape == (4, 4096) and np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.allclose(a[0], a[1]) and np.allclose(SR.BagRep().encode(["17 plus 5"])[0], a[0])
    assert a[0] @ a[3] > a[0] @ a[2]                                                     # a close wording is closer than a far one
    assert rep.encode([]).shape == (0, 4096)


def toy(n_per=24, seed=0):
    """Two taught skills in two surfaces S1 and S2 with 'wordings' as disjoint word sets, plus none-messages; items shaped like ceiling.Item."""
    rng = np.random.default_rng(seed)
    words = {("add", "S1"): ["plus", "sum", "added"], ("add", "S2"): ["combine", "together", "total"], ("sub", "S1"): ["minus", "less", "take"], ("sub", "S2"): ["remove", "drop", "leftover"]}
    items = []
    for (skill, fam), ws in words.items():
        for k in range(n_per):
            text = f"{rng.choice(ws)} {rng.integers(10, 99)} {rng.choice(ws)} {rng.integers(2, 9)}"
            items.append(ceiling.Item(text, text, skill, fam, (f"{fam}.f{k % 3}", f"{fam}.{skill}.c{k % 3}"), "teach", (1, 2)))
    for k in range(n_per):
        text = f"the weather in town {int(rng.integers(10, 99))} was calm and quiet"
        items.append(ceiling.Item(text, text, "none", "other", ("other",), "teach", None))
    return items


def test_the_strict_arm_learns_to_group_by_skill_not_by_wording_and_refuses_forbidden_data():
    items = toy()
    X = SR.BagRep().encode([i.text for i in items])
    f = SR.make_mapper("R5_FS_strict", seed=0, epochs=150, hidden=64)(X, items, ["add", "sub"])
    Z = f(X)
    assert Z.shape == (len(items), 64) and np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-4)
    add = [k for k, i in enumerate(items) if i.label == "add"]
    sub = [k for k, i in enumerate(items) if i.label == "sub"]
    within = np.mean([Z[a] @ Z[b] for a in add[:12] for b in add[24:36]])             # add in S1 against add in S2
    across = np.mean([Z[a] @ Z[b] for a in add[:12] for b in sub[:12]])
    assert within > across + 0.2
    bad = [ceiling.Item("x", "x", "add", "S4", ("S4.f0", "S4.add.c0"), "teach", (1, 2))] + items[:10]
    with pytest.raises(ValueError, match="teaching wordings"):
        SR.make_mapper("R5_FS_strict", seed=0, epochs=2, hidden=8)(SR.BagRep().encode([i.text for i in bad]), bad, ["add", "sub"])
    untaught = [ceiling.Item("x", "x", "mul", "S1", ("S1.f0", "S1.mul.c0"), "teach", (1, 2))] + items[:10]
    with pytest.raises(ValueError, match="taught skills"):
        SR.make_mapper("R5_FS_strict", seed=0, epochs=2, hidden=8)(SR.BagRep().encode([i.text for i in untaught]), untaught, ["add", "sub"])
    with pytest.raises(KeyError):
        SR.make_mapper("R5_nonsense", seed=0)


SENTENCES = [f"customer {i} asked about the {w} of the card on day {i % 7}" for i, w in enumerate(["limit", "balance", "fee", "expiry", "pin", "statement", "rate", "delivery"] * 10)]


def test_pretraining_pulls_two_views_of_a_sentence_together_and_is_deterministic():
    hyper = dict(epochs=2, batch=16, hidden=32, out_dim=16, seed=1)
    state, mu, sd = SR.pretrain(SENTENCES, hyper)
    state2, mu2, sd2 = SR.pretrain(SENTENCES, hyper)
    assert all(np.allclose(state[k].numpy(), state2[k].numpy()) for k in state) and np.allclose(mu, mu2)
    import torch
    rep = SR.BagRep()
    net = SR._net(4096, 32, 16)
    net.load_state_dict(state)

    def embed(texts):
        with torch.no_grad():
            return torch.nn.functional.normalize(net(torch.as_tensor((rep.encode(texts) - mu) / sd, dtype=torch.float32)), dim=1).numpy()

    a = embed(SENTENCES[:20])
    b = embed([" ".join(s.split()[:-2]) for s in SENTENCES[:20]])                  # a shortened view of the same sentences
    same = np.mean(np.sum(a * b, axis=1))
    rolled = np.mean(np.sum(a * np.roll(b, 1, axis=0), axis=1))                       # views of other sentences
    assert same > rolled


def test_the_pretrained_network_is_cached_and_a_broken_cache_file_is_rebuilt(tmp_path):
    hyper = dict(epochs=1, batch=16, hidden=16, out_dim=8, seed=2)
    first = SR.pretrained(tmp_path, hyper, SENTENCES, device="cpu")
    files = list(tmp_path.glob("fs_pre_*.pt"))
    assert len(files) == 1
    again = SR.pretrained(tmp_path, hyper, SENTENCES, device="cpu")
    assert all(np.allclose(first[0][k].numpy(), again[0][k].numpy()) for k in first[0])
    files[0].write_bytes(b"not a torch file")
    rebuilt = SR.pretrained(tmp_path, hyper, SENTENCES, device="cpu")
    assert all(np.allclose(first[0][k].numpy(), rebuilt[0][k].numpy()) for k in first[0])
    SR.pretrained(tmp_path, {**hyper, "seed": 3}, SENTENCES, device="cpu")
    assert len(list(tmp_path.glob("fs_pre_*.pt"))) == 2                               # another hyper-parameter, another file


def test_the_pre_trained_arm_starts_from_the_pre_trained_network():
    items = toy(n_per=12)
    X = SR.BagRep().encode([i.text for i in items])
    init = SR.pretrain(SENTENCES, dict(epochs=1, batch=16, hidden=32, out_dim=16, seed=4))
    m = SR.ScratchMap(["add", "sub"], init=init, hidden=32, out_dim=16, epochs=0, seed=0)
    m.fit(X, [i.label for i in items], [i.family for i in items], [i.template_ids[0] for i in items])
    import torch
    with torch.no_grad():
        got = m._net[0].weight.detach().cpu().numpy()
    assert np.allclose(got, init[0]["0.weight"].numpy())                                 # with no fine-tuning the weights are exactly the pre-trained ones
    assert np.allclose(m._mu, init[1])                                                   # and the input is standardised with the pre-training sentences


def test_the_pre_trained_network_has_the_shape_of_the_fine_tuned_one(monkeypatch):
    seen = {}
    monkeypatch.setattr(SR, "pretrained", lambda cache_dir=None, hyper=None, sentences=None, device="auto": seen.update(hyper=hyper) or SR.pretrain(SENTENCES[:40], dict(epochs=1, batch=16, **hyper)))
    items = toy(n_per=12)
    X = SR.BagRep().encode([i.text for i in items])
    f = SR.make_mapper("R5_FS_pre", seed=0, hidden=48, out_dim=12, epochs=3)(X, items, ["add", "sub"])
    assert seen["hyper"] == {"hidden": 48, "out_dim": 12} and f(X).shape == (len(items), 12)
    seen.clear()
    SR.make_mapper("R5_FS_strict", seed=0, hidden=48, out_dim=12, epochs=3)(X, items, ["add", "sub"])
    assert seen == {}                                                                       # the strict arm never touches the pre-training


def test_the_stamp_covers_this_file_and_the_learned_map():
    import run_e2
    import runlib
    assert SR.stamp() == runlib.code_stamp(run_e2.STAMP_FILES + [SR.HERE / "scratch_reader.py", SR.HERE / "learned_map.py"])


# ------------------------------------------------------------------------------------------------------------------------------------ gaps found by the planted-bug sweep
def test_the_strict_arm_is_exactly_the_learned_map_with_a_hidden_layer():
    """The training loop of ScratchMap is a copy of LearnedMap's; with no pre-trained start and the same seed the two give the same map, so every detail of the copy is pinned."""
    import learned_map as LM
    items = toy(n_per=14)
    X = SR.BagRep().encode([i.text for i in items])
    args = ([i.label for i in items], [i.family for i in items], [i.template_ids[0] for i in items])
    kw = dict(out_dim=16, hidden=32, tau=0.1, twin_weight=4.0, epochs=40, lr=1e-3, weight_decay=1e-2, seed=3)
    a = SR.ScratchMap(["add", "sub"], **kw)
    b = LM.LearnedMap(["add", "sub"], **kw)
    a.fit(X, *args)
    b.fit(X, *args)
    assert np.allclose(a.transform(X), b.transform(X), atol=1e-5)
    c = SR.ScratchMap(["add", "sub"], **{**kw, "twin_weight": 1.0})
    c.fit(X, *args)
    assert not np.allclose(a.transform(X), c.transform(X), atol=1e-3)                  # the near-twin weight matters, so the golden comparison is not vacuous


def test_one_surface_only_or_one_skill_only_is_refused():
    items = [i for i in toy(n_per=8) if i.label in ("add", "none")]
    X = SR.BagRep().encode([i.text for i in items])
    with pytest.raises(ValueError, match="at least two taught skills"):
        SR.ScratchMap(["add", "sub"], hidden=8, out_dim=4, epochs=2).fit(X, [i.label for i in items], [i.family for i in items], [i.template_ids[0] for i in items])


def test_the_strict_arm_never_uses_a_pre_trained_start_even_if_one_is_passed():
    items = toy(n_per=12)
    X = SR.BagRep().encode([i.text for i in items])
    init = SR.pretrain(SENTENCES, dict(epochs=1, batch=16, hidden=32, out_dim=16, seed=5))
    with_init = SR.make_mapper("R5_FS_strict", seed=0, init=init, hidden=32, out_dim=16, epochs=20)(X, items, ["add", "sub"])(X)
    without = SR.make_mapper("R5_FS_strict", seed=0, hidden=32, out_dim=16, epochs=20)(X, items, ["add", "sub"])(X)
    assert np.allclose(with_init, without, atol=1e-6)


def record_encodes(monkeypatch):
    calls = []
    real = SR.BagRep.encode

    def spy(self, texts):
        calls.append(list(texts))
        return real(self, texts)

    monkeypatch.setattr(SR.BagRep, "encode", spy)
    return calls


def test_word_dropout_keeps_every_word_when_nothing_is_dropped_and_the_first_word_when_everything_is(monkeypatch):
    calls = record_encodes(monkeypatch)
    SR.pretrain(SENTENCES[:32], dict(epochs=1, batch=16, hidden=8, out_dim=4, seed=1, drop=0.0))
    views = [t for c in calls[1:] for t in c]
    assert views and all(v in SENTENCES[:32] for v in views)                          # nothing dropped: the views are the sentences
    calls.clear()
    SR.pretrain(SENTENCES[:32], dict(epochs=1, batch=16, hidden=8, out_dim=4, seed=1, drop=1.0))
    assert all(len(v.split()) == 1 and v == v.split()[0] for c in calls[1:] for v in c) and calls[1]      # everything dropped: one word is kept, never an empty text
    calls.clear()
    SR.pretrain(SENTENCES[:64], dict(epochs=1, batch=64, hidden=8, out_dim=4, seed=1, drop=0.5))
    lens = [len(v.split()) for v in calls[1]]
    assert 0.2 < np.mean(lens) / np.mean([len(s.split()) for s in SENTENCES[:64]]) < 0.8     # about half of the words survive


def test_a_final_batch_of_one_sentence_is_skipped_and_training_changes_the_weights(monkeypatch):
    import torch
    calls = record_encodes(monkeypatch)
    hyper = dict(epochs=2, batch=16, hidden=8, out_dim=4, seed=2)
    state, mu, sd = SR.pretrain(SENTENCES[:17], hyper)                                  # 17 sentences, batches of 16: the last batch has one sentence
    assert all(len(c) >= 2 for c in calls)                                              # a batch of one has no negatives and is not used
    torch.manual_seed(2)
    fresh = SR._net(4096, 8, 4)
    assert not np.allclose(state["0.weight"].numpy(), fresh.state_dict()["0.weight"].numpy())      # the network moved away from its random start


def test_the_default_pre_training_corpus_is_a_fixed_draw_of_the_registered_size(monkeypatch, tmp_path):
    import corpora
    asked = []
    monkeypatch.setattr(corpora, "load_other_sentences", lambda n, seed: asked.append((n, seed)) or SENTENCES[:20])
    monkeypatch.setattr(SR, "pretrain", lambda texts, h, dev: (asked.append(("device", dev)) or ({"w": __import__("torch").zeros(1)}, np.zeros(3), np.ones(3))))
    SR.pretrained(tmp_path, dict(epochs=1), device="cpu")
    assert asked[0] == (SR.PRETRAIN["n_sentences"], 0x5C4A7C4) and asked[1] == ("device", "cpu")
    asked.clear()
    SR.pretrained(tmp_path, dict(epochs=2), sentences=SENTENCES[:20], device="cpu")
    assert not [a for a in asked if a[0] != "device"]                                       # given sentences bypass the corpus
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    asked.clear()
    SR.pretrained(tmp_path, dict(epochs=3), sentences=SENTENCES[:20], device="auto")
    assert asked == [("device", "cpu")]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    SR.pretrained(tmp_path, dict(epochs=4), sentences=SENTENCES[:20], device="auto")
    assert asked[-1] == ("device", "cuda")


def test_two_skills_each_in_a_single_surface_give_nothing_to_pull_together_and_are_refused():
    base = toy(n_per=10)
    items = [ceiling.Item(i.text, i.span, i.label, "S1", ("S1.f0", "S1.c0"), "teach", None) for i in base if i.label in ("add", "sub")]       # every row of a skill in the same surface
    X = SR.BagRep().encode([i.text for i in items])
    with pytest.raises(ValueError, match="at least two taught skills"):
        SR.ScratchMap(["add", "sub"], hidden=8, out_dim=4, epochs=2).fit(X, [i.label for i in items], [i.family for i in items], [i.template_ids[0] for i in items])
