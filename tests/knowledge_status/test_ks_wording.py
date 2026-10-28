"""Mitigations for the wording shift (new module, subclass of the status machine, so the results already on disk stay valid): text clean-up before reading, calibration on
leave-one-wording-out scores, and a margin gate. Toy von Mises-Fisher worlds in which a wording is a shift of a skill's cloud."""

import numpy as np
import pytest

import reps
import status as ST
import wording as W
from growmoe.vmf import sample_vmf

D = 16


def vmf(mu, kappa, n, rng):
    return sample_vmf(mu, kappa, n, rng).numpy().astype(np.float64)


def unit(v):
    return v / np.linalg.norm(v)


def make_world(seed=0, shift=0.5, kappa=300.0, n_proto=120, n_cal=40, skills=("A", "B")):
    """Skills A, B (taught) and U (never taught). A skill's cloud is centred on a mean; each taught wording f1..f3 moves it by a random offset of size ``shift``, and the held-out wording f4
    moves it by another offset of the same size. U sits between A and B (an ambiguous part)."""
    rng = np.random.default_rng(seed)
    mu = {s: unit(rng.standard_normal(D)) for s in (*skills, "O")}
    mu["U"] = unit(mu["A"] + mu["B"])
    off = {(s, f): unit(rng.standard_normal(D)) for s in (*skills, "U") for f in ("f1", "f2", "f3", "f4")}
    mean = lambda s, f: unit(mu[s] + shift * off[(s, f)])
    data = {"protos": {}, "proto_fam": {}, "cal": [], "cal_grp": [], "cal_fam": []}
    for s in skills:
        data["protos"][s] = np.vstack([vmf(mean(s, f), kappa, n_proto, rng) for f in ("f1", "f2", "f3")])
        data["proto_fam"][s] = np.array([f for f in ("f1", "f2", "f3") for _ in range(n_proto)])
        for f in ("f1", "f2", "f3"):
            data["cal"].append(vmf(mean(s, f), kappa, n_cal, rng))
            data["cal_grp"] += [s] * n_cal
            data["cal_fam"] += [f] * n_cal
    data["protos"]["O"] = vmf(mu["O"], 20.0, 300, rng)
    data["cal"].append(vmf(mu["O"], 20.0, 120, rng))
    data["cal_grp"] += ["O"] * 120
    data["cal_fam"] += ["other"] * 120
    data["cal"], data["cal_grp"], data["cal_fam"] = np.vstack(data["cal"]), np.array(data["cal_grp"]), np.array(data["cal_fam"])
    data["fresh"] = lambda s, f, n, k=1: vmf(mean(s, f), kappa, n, np.random.default_rng([seed, k, ord(s[0]), ord(f[1])]))
    data["groups"] = {s: ST.SkillGroup(s, data["protos"][s], 2000, None, 0.97) for s in skills}
    data["other"] = ST.SkillGroup("O", data["protos"]["O"], 500, None, 1.0)
    return data


def machine(w, cls=ST.StatusMachine, **kw):
    if cls is ST.StatusMachine:
        return cls(w["groups"], w["other"], w["cal"], w["cal_grp"], q=0.05, **kw)
    return cls(w["groups"], w["other"], w["cal"], w["cal_grp"], q=0.05, proto_families=w["proto_fam"], cal_families=w["cal_fam"], **kw)


def rate(out, status="unseen"):
    return float(np.mean([a.status == status for a in out]))


# ------------------------------------------------------------------------------------------------------------------------------ clean-up and the reader wrapper
def test_clean_text_lowercases_collapses_spaces_and_strips():
    assert W.clean_text("  What IS  17   Plus 5 ?  ") == "what is 17 plus 5 ?"
    assert W.clean_text("") == "" and W.clean_text("\tA\nB ") == "a b"
    assert W.clean_text("17 + 5") == "17 + 5"                                                  # digits and symbols are untouched
    assert W.clean_text(W.clean_text("MiXeD  Case")) == W.clean_text("MiXeD  Case")             # idempotent


class Stub(reps.Representation):
    def __init__(self):
        super().__init__("stub", 4, tag="stub@x")
        self.seen = []

    def _encode(self, texts):
        self.seen.extend(texts)
        return np.array([[len(t), t.count("a"), t.count("b"), 1.0] for t in texts], dtype=np.float32)


def test_clean_rep_reads_the_cleaned_text_but_keeps_its_own_cache_identity():
    base = Stub()
    rep = W.CleanRep(base)
    out = rep.encode(["AB  a", "ab a"])
    assert base.seen == ["ab a", "ab a"] and np.allclose(out[0], out[1])
    assert rep.id != base.id and rep.tag != base.tag and rep.dim == base.dim


# ------------------------------------------------------------------------------------------------------------------------------ leave-one-wording-out calibration
def test_without_the_new_options_the_wording_machine_is_the_status_machine():
    w = make_world()
    pts = np.vstack([w["fresh"]("A", "f4", 80), w["fresh"]("U", "f4", 80), w["fresh"]("A", "f1", 80)])
    assert machine(w, W.WordingMachine).assess_many(pts) == machine(w).assess_many(pts)


def test_leave_one_wording_out_calibration_expects_a_wording_shift():
    w = make_world()
    base, lofo = machine(w), machine(w, W.WordingMachine, lofo=True)
    new_wording = w["fresh"]("A", "f4", 400)
    assert rate(base.assess_many(new_wording)) >= 0.4, "the toy shift must be large enough to set off the ordinary alarm"
    assert rate(lofo.assess_many(new_wording)) <= rate(base.assess_many(new_wording)) - 0.15
    # taught wordings stay calibrated (the alarm gets more careful, not more trigger-happy) and a skill that was never taught is still found
    seen_wording = np.vstack([w["fresh"]("A", f, 200, k=7) for f in ("f1", "f2", "f3")])
    assert rate(lofo.assess_many(seen_wording)) <= 0.07
    far = np.vstack([w["fresh"]("U", f, 150) for f in ("f1", "f2", "f3", "f4")])
    assert rate(lofo.assess_many(far)) >= 0.9


def test_lofo_scores_use_the_kernel_width_of_the_full_signature():
    w = make_world()
    base, lofo = machine(w), machine(w, W.WordingMachine, lofo=True)
    for j in range(len(base.names)):
        assert lofo._sig[j].kappa == base._sig[j].kappa                                          # one scale for the calibration scores and for the parts judged against them
    assert not np.allclose(lofo._cal[0], base._cal[0])                                           # but the calibration scores themselves are the cross-wording ones
    assert np.allclose(lofo._cal[-1], base._cal[-1])                                             # the Other group has no wordings: standard calibration


def test_lofo_needs_wording_labels_and_two_wordings():
    w = make_world()
    with pytest.raises(ValueError, match="proto_families"):
        W.WordingMachine(w["groups"], w["other"], w["cal"], w["cal_grp"], lofo=True, cal_families=w["cal_fam"])
    with pytest.raises(ValueError, match="one entry per"):
        W.WordingMachine(w["groups"], w["other"], w["cal"], w["cal_grp"], lofo=True, proto_families={**w["proto_fam"], "A": w["proto_fam"]["A"][:-1]}, cal_families=w["cal_fam"])
    with pytest.raises(ValueError, match="cal_families"):
        W.WordingMachine(w["groups"], w["other"], w["cal"], w["cal_grp"], lofo=True, proto_families=w["proto_fam"], cal_families=w["cal_fam"][:-1])
    one = {**w["proto_fam"], "A": np.array(["f1"] * len(w["proto_fam"]["A"]))}
    m = W.WordingMachine(w["groups"], w["other"], w["cal"], w["cal_grp"], lofo=True, proto_families=one, cal_families=w["cal_fam"])
    assert np.allclose(m._cal[0], machine(w)._cal[0])                                            # a group with a single wording falls back to the standard calibration


# ------------------------------------------------------------------------------------------------------------------------------ margin gate
def test_margin_gate_keeps_a_known_skill_in_a_new_wording_and_flags_an_ambiguous_part():
    w = make_world(shift=0.55)
    plain, gated = machine(w), machine(w, W.WordingMachine, margin_q=0.25)
    known = w["fresh"]("A", "f4", 400)
    ambiguous = np.vstack([w["fresh"]("U", f, 100) for f in ("f1", "f2", "f3", "f4")])
    assert rate(plain.assess_many(known)) >= 0.4
    assert rate(gated.assess_many(known)) <= rate(plain.assess_many(known)) - 0.25
    assert rate(gated.assess_many(ambiguous)) >= 0.7
    kept = [a for a in gated.assess_many(known) if a.status != "unseen"]
    assert np.mean([a.skill == "A" for a in kept]) >= 0.9                                        # the parts the gate lets through go to their own skill
    assert gated.last_overridden.shape == (len(known),) and gated.last_overridden.sum() > 0              # the last call was on the known parts: the gate let some of them through


def test_margin_gate_never_adds_an_unseen_call_and_agrees_where_the_alarm_is_quiet():
    w = make_world()
    plain, gated = machine(w), machine(w, W.WordingMachine, margin_q=0.25)
    pts = np.vstack([w["fresh"]("A", "f1", 200), w["fresh"]("B", "f4", 200), w["fresh"]("U", "f2", 200)])
    a, b = plain.assess_many(pts), gated.assess_many(pts)
    assert all(not (y.status == "unseen" and x.status != "unseen") for x, y in zip(a, b))
    assert all(x == y for x, y in zip(a, b) if x.status != "unseen")


def test_margin_gate_is_valid_on_taught_wordings():
    w = make_world()
    gated = machine(w, W.WordingMachine, margin_q=0.25)
    assert rate(gated.assess_many(np.vstack([w["fresh"]("A", f, 250, k=9) for f in ("f1", "f2", "f3")]))) <= 0.07


def test_margin_q_is_checked():
    w = make_world()
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError, match="margin_q"):
            machine(w, W.WordingMachine, margin_q=bad)


def test_both_mitigations_together_still_work():
    w = make_world(shift=0.55)
    both = machine(w, W.WordingMachine, lofo=True, margin_q=0.25)
    assert rate(both.assess_many(w["fresh"]("A", "f4", 300))) <= rate(machine(w).assess_many(w["fresh"]("A", "f4", 300))) - 0.25
    assert rate(both.assess_many(np.vstack([w["fresh"]("U", f, 100) for f in ("f1", "f2", "f3", "f4")]))) >= 0.6


# ------------------------------------------------------------------------------------------------------------------------------ a reader with another closing prompt
def test_prompted_rep_changes_only_the_closing_prompt():
    plain = reps.get_rep("R4_gpt2_L6")
    same = W.PromptedRep("R4_gpt2_L6_same", "gpt2", 6, False, reps.SUFFIX)
    other = W.PromptedRep("R4_gpt2_L6_p1", "gpt2", 6, False, reps.SUFFIX.replace("The operation requested", "Ignoring any extra words, the calculation asked for"))
    texts = ["17 plus 5", "I had 17 apples and got 5 more, how many now?"]
    np.testing.assert_allclose(same.encode(texts), plain.encode(texts), atol=1e-5)                 # the same prompt gives the same vectors as the ordinary reader
    assert not np.allclose(other.encode(texts), plain.encode(texts), atol=1e-3)
    assert other.id != plain.id and other.tag != same.tag and other.dim == plain.dim                # its cache files never mix with another prompt's
    np.testing.assert_array_equal(other.encode(texts), other.encode(texts))
    assert other.suffix_ids != plain.suffix_ids


# ------------------------------------------------------------------------------------------------------------------------------ the lab driver
import assignments  # noqa: E402
import lab_wording as LAB  # noqa: E402
import run_e2 as R  # noqa: E402

TINY = R.Sizes(counts=(("solid", 60), ("thin", 8)), other_teach=45, cal=30, adapt_per_family=2, test_heldout=12, test_taught=8, twins=12, traps=12, others=15,
               unseen_content=10, tangled=6, batches=4, batch_sizes=(1, 4), background=40, map_cap=40, qs=(0.05,))


@pytest.fixture
def tiny_thresholds(monkeypatch):
    for k, v in dict(solid_min_count=50, thin_max_count=20).items():
        monkeypatch.setitem(assignments.THRESHOLDS, k, v)


def fake_competence(pool, skill, count, seed):
    return {"accuracy": 0.97 if count >= 60 else 0.5, "chance": 0.1}


def test_the_wording_of_every_calibration_part_is_read_back_exactly(tiny_thresholds):
    world = R.build_world(100, "dev", TINY, fake_competence)
    fam = LAB.cal_family_labels(world)
    assert set(fam) == set(world.taught)
    for s in world.taught:
        assert len(fam[s]) == len(world.cal[s]) == TINY.cal and set(fam[s].tolist()) <= set(assignments.TEACH_FAMILIES)
    rows = LAB.family_arrays(world)
    assert len(rows["cal"]) == sum(len(v) for v in world.cal.values())
    assert set(rows["proto"]) == set(world.taught) and all(len(rows["proto"][s]) == len(world.teach[s]) for s in world.taught)


def row(name, false_unseen, found, f1, rank):
    return {"name": name, "false_unseen": false_unseen, "found": found, "macro_f1": f1, "rank": rank}


def test_the_winner_is_the_lowest_false_alarm_among_variants_that_still_find_new_skills():
    rows = [row("plain", 0.28, 0.89, 0.85, 0), row("clean", 0.24, 0.88, 0.85, 1), row("p1", 0.15, 0.70, 0.80, 2), row("p2", 0.16, 0.85, 0.79, 3), row("clean+p1", 0.17, 0.84, 0.82, 4)]
    assert LAB.choose_winner(rows)["name"] == "clean+p1"                                  # p1 finds too few, p2's score is too low
    rows[1]["false_unseen"] = 0.171
    assert LAB.choose_winner(rows)["name"] == "clean"                                      # clean is within 0.01 of the best allowed one and simpler
    assert LAB.choose_winner([row("plain", 0.28, 0.5, 0.6, 0), row("clean", 0.1, 0.5, 0.6, 1)])["name"] == "plain"        # nothing qualifies: no change
    with pytest.raises(ValueError, match="no variants"):
        LAB.choose_winner([])


def test_lofo_calibration_scores_are_exactly_the_cross_wording_scores():
    """Rebuilt by hand: for each wording, a signature without that wording's prototypes (kernel width of the full signature) scores that wording's held-out parts."""
    from growmoe.signature import VmfSignature
    w = make_world()
    lofo = machine(w, W.WordingMachine, lofo=True)
    j, name = 0, lofo.names[0]
    P = lofo._P[j]
    fam = w["proto_fam"][name]
    cal = w["cal"][w["cal_grp"] == name]
    cal = (cal / np.linalg.norm(cal, axis=1, keepdims=True)).astype(np.float32)
    cal_fam = w["cal_fam"][w["cal_grp"] == name]
    expected = []
    for f in sorted(set(cal_fam.tolist())):
        sig = VmfSignature(lofo.d, max_protos=256, seed=0, dedupe=True)
        sig.update(P[fam != f])
        sig.fit()
        sig._kappa = lofo._sig[j].kappa
        expected.append(sig.score(cal[cal_fam == f]).double().numpy())
    np.testing.assert_allclose(lofo._cal[j], np.sort(np.concatenate(expected)), atol=1e-5)


def test_margin_gate_never_overrides_a_part_whose_nearest_group_is_other_content():
    w = make_world()
    plain, gated = machine(w), machine(w, W.WordingMachine, margin_q=0.9)
    broad = vmf(unit(np.random.default_rng(3).standard_normal(D)) * 0 + (w["protos"]["O"].mean(axis=0) / np.linalg.norm(w["protos"]["O"].mean(axis=0))), 3.0, 600, np.random.default_rng(4))
    a, b = plain.assess_many(broad), gated.assess_many(broad)
    nearest_other = [x.nearest == "O" for x in a]
    assert any(nearest_other) and sum(x.status == "unseen" for x in a) > 20
    assert all(x == y for x, y, o in zip(a, b, nearest_other) if o)                                 # parts nearest to Other content: identical to the machine without the gate
    assert not gated.last_overridden[np.array(nearest_other)].any()


def test_the_margin_calibration_uses_the_skill_parts_only():
    w = make_world()
    gated = machine(w, W.WordingMachine, margin_q=0.25)
    assert len(gated._margins) == int((w["cal_grp"] != "O").sum()) and np.all(np.diff(gated._margins) >= 0)


def test_lofo_scores_use_the_full_kernel_width_even_when_leaving_a_wording_out_would_pick_another():
    from growmoe.signature import VmfSignature
    rng = np.random.default_rng(11)
    mu = unit(rng.standard_normal(D))
    mean = lambda off: unit(mu + 0.6 * off)
    offs = {f: unit(rng.standard_normal(D)) for f in ("f1", "f2", "f3")}
    kap = {"f1": 2000.0, "f2": 12.0, "f3": 12.0}                                    # one very tight wording and two broad ones: dropping the tight one changes the best kernel width
    protos = np.vstack([vmf(mean(offs[f]), kap[f], 150, rng) for f in offs])
    fam = np.array([f for f in offs for _ in range(150)])
    cal = np.vstack([vmf(mean(offs[f]), kap[f], 40, rng) for f in offs])
    cal_fam = np.array([f for f in offs for _ in range(40)])
    other_p = vmf(unit(rng.standard_normal(D)), 20.0, 200, rng)
    other_c = vmf(unit(rng.standard_normal(D)), 20.0, 100, rng)
    groups = {"A": ST.SkillGroup("A", protos, 2000, None, 0.97)}
    m = W.WordingMachine(groups, ST.SkillGroup("O", other_p, 300, None, 1.0), np.vstack([cal, other_c]), np.array(["A"] * 120 + ["O"] * 100), q=0.05,
                         proto_families={"A": fam}, cal_families=np.r_[cal_fam, ["other"] * 100], lofo=True, max_protos=512)
    P, calu = m._P[0], (cal / np.linalg.norm(cal, axis=1, keepdims=True)).astype(np.float32)
    full_kappa = m._sig[0].kappa
    own_kappas, with_full, with_own = [], [], []
    for f in offs:
        sig = VmfSignature(D, max_protos=512, seed=0, dedupe=True)
        sig.update(P[fam != f])
        sig.fit()
        own_kappas.append(sig.kappa)
        with_own.append(sig.score(calu[cal_fam == f]).double().numpy())
        sig._kappa = full_kappa
        with_full.append(sig.score(calu[cal_fam == f]).double().numpy())
    assert any(k != full_kappa for k in own_kappas), "the toy must make the leave-one-out kernel width differ"
    np.testing.assert_allclose(m._cal[0], np.sort(np.concatenate(with_full)), atol=1e-5)
    assert not np.allclose(np.sort(np.concatenate(with_own)), np.sort(np.concatenate(with_full)), atol=1e-3)
