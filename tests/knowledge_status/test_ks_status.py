"""Task 10: the status machine and the baselines it is compared with, on small synthetic von Mises-Fisher worlds (unit vectors that cluster around a direction).

The toy world: taught skills A, B, C and an Other-content group O live around different directions of a 16-dimensional sphere; U is a skill that was never taught and R is a
content type that was never taught. Nothing here needs a language model, so the whole file runs in seconds.
"""

import inspect
import math

import numpy as np
import pytest

import assignments
import baselines as BL
import status as ST
from growmoe.conformal import conformal_pvalues
from growmoe.vmf import sample_vmf

D = 16
KAPPA = {"A": 80.0, "B": 80.0, "C": 80.0, "O": 25.0, "U": 80.0, "R": 25.0}
TAUGHT = ("A", "B", "C")


def vmf(mu, kappa, n, rng):
    return sample_vmf(mu, kappa, n, rng).numpy().astype(np.float64)


def make_toy(seed=0, n_proto=400, n_cal=600, n_examples=2000, competence=0.97):
    """Returns the SkillGroups, calibration data and a function that draws fresh held-out points of any region of the toy world."""
    rng = np.random.default_rng(seed)
    mus = {}
    for name in KAPPA:
        v = rng.standard_normal(D)
        mus[name] = v / np.linalg.norm(v)
    protos = {n: vmf(mus[n], KAPPA[n], n_proto, rng) for n in (*TAUGHT, "O")}
    cal = {n: vmf(mus[n], KAPPA[n], n_cal, rng) for n in (*TAUGHT, "O")}
    groups = {n: ST.SkillGroup(n, protos[n], n_examples, None, competence) for n in TAUGHT}
    other = ST.SkillGroup("O", protos["O"], 500, None, 1.0)
    cal_vectors = np.vstack([cal[n] for n in (*TAUGHT, "O")])
    cal_groups = np.array([n for n in (*TAUGHT, "O") for _ in range(n_cal)])

    def fresh(name, n, seed_=1):
        return vmf(mus[name], KAPPA[name], n, np.random.default_rng([seed, seed_, ord(name)]))

    return dict(groups=groups, other=other, cal_vectors=cal_vectors, cal_groups=cal_groups, fresh=fresh, mus=mus, protos=protos, cal=cal)


@pytest.fixture(scope="module")
def toy():
    return make_toy()


def machine_of(toy, q=0.05, **kw):
    return ST.StatusMachine(toy["groups"], toy["other"], toy["cal_vectors"], toy["cal_groups"], q=q, **kw)


@pytest.fixture(scope="module")
def machine(toy):
    return machine_of(toy)


def rate(assessments, status):
    return float(np.mean([a.status == status for a in assessments]))


# ------------------------------------------------------------------------------------------------------------------------------ the machine
def test_false_novelty_is_at_most_q_on_exchangeable_data(toy, machine):
    out = machine.assess_many(toy["fresh"]("A", 5000))
    assert rate(out, "unseen") <= 0.07, rate(out, "unseen")
    assert rate(out, "unseen") >= 0.02, "a calibrated test should flag about q of exchangeable data, not nothing"


def test_unseen_group_is_flagged_when_far(toy, machine):
    out = machine.assess_many(toy["fresh"]("U", 2000))
    assert rate(out, "unseen") >= 0.9, rate(out, "unseen")


def test_thin_is_reported_for_a_group_with_few_examples(toy):
    few, many = make_toy(n_examples=20), make_toy(n_examples=2000)
    pts = few["fresh"]("A", 400)
    a_few, a_many = machine_of(few).assess_many(pts), machine_of(many).assess_many(pts)
    covered = [(f, m) for f, m in zip(a_few, a_many) if f.status != "unseen"]
    assert len(covered) >= 0.9 * len(pts)
    assert all(f.status == "thin" and m.status == "solid" for f, m in covered)
    assert all(f.skill == "A" and f.group_size == 1 for f, _ in covered)


def test_thin_follows_the_real_number_of_prototypes_not_only_the_count_field():
    """The same cloud, but 20 real taught examples against 2,000: the count that matters is the one the machine is told, and it is told it by SkillGroup.n_examples."""
    big, small = make_toy(n_proto=400, n_examples=2000), make_toy(n_proto=400, n_examples=20)
    assert machine_of(big).assess(big["fresh"]("B", 1)[0]).status in ("solid", "unseen")
    assert machine_of(small).assess(small["fresh"]("B", 1)[0]).status in ("thin", "unseen")


def test_low_competence_makes_a_big_group_thin(toy):
    weak = make_toy(competence=0.55)
    out = machine_of(weak).assess_many(weak["fresh"]("A", 300))
    assert rate(out, "solid") == 0.0 and rate(out, "thin") >= 0.9


def test_other_content_is_never_reported_as_unseen_arithmetic_on_toy(toy, machine):
    out = machine.assess_many(toy["fresh"]("O", 3000))
    assert rate(out, "unseen") <= 0.07                                 # same calibration bound as any taught group
    assert rate(out, "other") >= 0.9
    assert rate(out, "solid") == 0.0 and rate(out, "thin") == 0.0      # never mistaken for a taught arithmetic skill
    assert all(a.skill is None for a in out)
    # the control that is meant to be Unseen: a content type nobody taught (a recipe sentence)
    recipe = machine.assess_many(toy["fresh"]("R", 2000))
    assert rate(recipe, "unseen") >= 0.9


def test_covered_parts_are_assigned_to_their_own_skill(toy, machine):
    for name in TAUGHT:
        out = [a for a in machine.assess_many(toy["fresh"](name, 800)) if a.status != "unseen"]
        assert np.mean([a.skill == name for a in out]) >= 0.98
        assert all(a.nearest == name or a.skill is None or a.skill == a.nearest for a in out)


def test_machine_needs_only_skill_groups_and_a_vector(toy, machine):
    groups = {n: ST.SkillGroup(name=n, protos=toy["protos"][n], n_examples=2000, solver=None, competence=0.97) for n in TAUGHT}
    m = ST.StatusMachine(groups, toy["other"], toy["cal_vectors"], toy["cal_groups"])
    a = m.assess(toy["fresh"]("A", 1)[0])
    assert isinstance(a, ST.Assessment) and a.status in ST.STATUSES
    forbidden = {"label", "labels", "y", "skill", "skills", "truth", "family", "template", "gold"}
    for fn in (ST.StatusMachine.assess, ST.StatusMachine.assess_many, ST.StatusMachine.assess_group, ST.StatusMachine.flag_conflict):
        names = set(inspect.signature(fn).parameters) - {"self"}
        assert not (names & forbidden), (fn.__name__, names)
    assert set(inspect.signature(ST.StatusMachine.assess).parameters) == {"self", "part_vec"}
    assert set(inspect.signature(ST.StatusMachine.assess_group).parameters) == {"self", "part_vecs"}
    assert set(inspect.signature(ST.StatusMachine.flag_conflict).parameters) == {"self", "part_vec", "known_answer_disagreement"}


def make_overlap(cos_xy, kx, ky, seed=0, n_proto=400, n_cal=600):
    """Two taught groups X and Y whose clouds overlap (the means are ``cos_xy`` apart), plus an Other group far away. Returns the machine and held-out points of X and of Y."""
    rng = np.random.default_rng(seed)
    e, f = np.eye(D)[0], np.eye(D)[1]
    mu_x, mu_y = e, cos_xy * e + math.sqrt(1 - cos_xy ** 2) * f
    px, py, po = vmf(mu_x, kx, n_proto, rng), vmf(mu_y, ky, n_proto, rng), vmf(np.eye(D)[5], 20.0, n_proto, rng)
    cal = np.vstack([vmf(mu_x, kx, n_cal, rng), vmf(mu_y, ky, n_cal, rng), vmf(np.eye(D)[5], 20.0, n_cal, rng)])
    names = np.array(["X"] * n_cal + ["Y"] * n_cal + ["O"] * n_cal)
    groups = {"X": ST.SkillGroup("X", px, 2000, None, 0.97), "Y": ST.SkillGroup("Y", py, 2000, None, 0.97)}
    return ST.StatusMachine(groups, ST.SkillGroup("O", po, 500, None, 1.0), cal, names), vmf(mu_x, kx, 3000, rng), vmf(mu_y, ky, 3000, rng)


def test_a_part_covered_by_two_groups_goes_to_the_one_with_the_nearer_prototype():
    m, tx, ty = make_overlap(0.9, 40.0, 40.0)
    from_x = [a for a in m.assess_many(tx) if a.status != "unseen"]
    from_y = [a for a in m.assess_many(ty) if a.status != "unseen"]
    assert np.mean([a.skill == "X" for a in from_x]) >= 0.75 and np.mean([a.skill == "Y" for a in from_y]) >= 0.75


def test_false_unseen_stays_at_most_q_when_a_narrow_neighbour_sits_beside_a_wide_group():
    """Some of X's held-out points have their nearest taught prototype in the narrow group Y, which flags them. Judging a part against the nearest group alone would call those unseen
    (0.18 here, measured while designing this); a part is unseen only when EVERY group flags it, which keeps the rate at q."""
    m, tx, _ = make_overlap(0.8, 20.0, 80.0)
    out = m.assess_many(tx)
    assert np.mean([a.nearest != "X" for a in out]) >= 0.1                 # the situation really arises
    assert rate(out, "unseen") <= 0.07
    assert np.mean([a.skill == "X" for a in out if a.status != "unseen"]) >= 0.95


def test_assess_never_returns_related_or_conflict(toy, machine):
    pts = np.vstack([toy["fresh"](n, 300) for n in (*TAUGHT, "O", "U", "R")])
    assert {a.status for a in machine.assess_many(pts)} <= {"solid", "thin", "unseen", "other"}


def test_assess_equals_assess_many_row_by_row(toy, machine):
    pts = toy["fresh"]("B", 30)
    assert machine.assess_many(pts) == [machine.assess(p) for p in pts]


def test_evidence_counts_taught_examples_near_the_part_and_scales_with_the_count(toy):
    m_big, m_small = machine_of(make_toy(n_examples=2000)), machine_of(make_toy(n_examples=200))
    pts = make_toy()["fresh"]("A", 300)
    e_big = np.array([a.evidence for a in m_big.assess_many(pts) if a.status != "unseen"])
    e_small = np.array([a.evidence for a in m_small.assess_many(pts) if a.status != "unseen"])
    assert (e_big >= 1).mean() >= 0.9                                    # a covered part can sit just outside every taught neighbour's radius: zero is allowed, but rare
    assert e_big.mean() == pytest.approx(10 * e_small.mean(), rel=0.1)  # same prototypes, ten times the taught examples: ten times the local evidence
    unseen = [a for a in m_big.assess_many(make_toy()["fresh"]("U", 100)) if a.status == "unseen"]
    assert unseen and all(a.evidence == 0 and math.isnan(a.competence) and a.skill is None for a in unseen)


def test_group_state_rule_matches_the_status_truth_rule():
    for count in (1, 5, 20, 99, 100, 101, 250, 499, 500, 501, 2000):
        for acc in (0.0, 0.2, 0.3, 0.5, 0.6, 0.61, 0.8, 0.89, 0.9, 0.97, 1.0):
            truth = assignments.status_truth(count, acc, 0.2)
            assert ST.group_state(count, acc) == ("solid" if truth == "solid" else "thin"), (count, acc, truth)


def test_the_thresholds_are_the_machines_own_not_a_global(toy):
    strict = dict(assignments.THRESHOLDS, solid_min_count=5000)
    m = machine_of(toy, thresholds=strict)
    out = [a for a in m.assess_many(toy["fresh"]("A", 200)) if a.status != "unseen"]
    assert out and all(a.status == "thin" for a in out)
    assert ST.group_state(2000, 0.97, strict) == "thin" and ST.group_state(2000, 0.97) == "solid"


def test_the_machine_p_value_is_the_repository_conformal_p_value():
    cal = np.random.default_rng(0).standard_normal(57)
    test = np.random.default_rng(1).standard_normal(40)
    ours = ST.conformal_p(np.sort(cal), test)
    assert ours.dtype == np.float64
    np.testing.assert_allclose(ours, conformal_pvalues(__import__("torch").from_numpy(cal), __import__("torch").from_numpy(test)).numpy(), rtol=1e-6)


def test_a_tie_at_the_level_is_novel_exactly_when_the_p_value_is_at_most_q(toy):
    """19 calibration points is the smallest set that can ever flag at q = 0.05 (p = 1/20); 18 cannot."""
    small = dict(toy)
    keep = np.concatenate([np.flatnonzero(toy["cal_groups"] == n)[:19] for n in (*TAUGHT, "O")])
    ST.StatusMachine(toy["groups"], toy["other"], toy["cal_vectors"][keep], toy["cal_groups"][keep], q=0.05)
    keep18 = np.concatenate([np.flatnonzero(toy["cal_groups"] == n)[:18] for n in (*TAUGHT, "O")])
    with pytest.raises(ValueError, match="calibration"):
        ST.StatusMachine(small["groups"], small["other"], toy["cal_vectors"][keep18], toy["cal_groups"][keep18], q=0.05)


def test_machine_validates_its_inputs(toy):
    g, o, cv, cg = toy["groups"], toy["other"], toy["cal_vectors"], toy["cal_groups"]
    with pytest.raises(ValueError, match="at least one skill"):
        ST.StatusMachine({}, o, cv, cg)
    with pytest.raises(ValueError, match="same name"):
        ST.StatusMachine({**g, "O": ST.SkillGroup("O", toy["protos"]["A"], 100, None, 0.9)}, o, cv, cg)
    with pytest.raises(ValueError, match="key"):
        ST.StatusMachine({"Z": g["A"]}, o, cv, cg)
    with pytest.raises(ValueError, match="unknown group"):
        ST.StatusMachine(g, o, cv, np.where(cg == "A", "Q", cg))
    with pytest.raises(ValueError, match="calibration"):
        ST.StatusMachine(g, o, cv[:-1], cg)
    with pytest.raises(ValueError, match="dimension"):
        ST.StatusMachine(g, o, cv[:, :8], cg)
    with pytest.raises(ValueError, match="q must"):
        ST.StatusMachine(g, o, cv, cg, q=0.0)
    with pytest.raises(ValueError, match="finite"):
        ST.SkillGroup("X", np.full((5, D), np.nan), 10, None, 0.5)
    with pytest.raises(ValueError, match="competence"):
        ST.SkillGroup("X", toy["protos"]["A"], 10, None, 1.5)
    with pytest.raises(ValueError, match="at least two"):
        ST.SkillGroup("X", toy["protos"]["A"][:1], 10, None, 0.5)


def test_bad_part_vectors_are_refused(machine):
    with pytest.raises(ValueError, match="dimension"):
        machine.assess(np.ones(D + 1))
    with pytest.raises(ValueError, match="finite"):
        machine.assess(np.full(D, np.nan))
    with pytest.raises(ValueError, match="zero"):
        machine.assess(np.zeros(D))


def test_decisions_do_not_depend_on_call_order_or_instance(toy):
    a, b = machine_of(toy), machine_of(toy)
    pts = toy["fresh"]("A", 40)
    assert a.assess_many(pts) == b.assess_many(pts[::-1])[::-1]
    batch = toy["fresh"]("U", 8)
    first = a.assess_group(batch)
    a.assess_group(toy["fresh"]("A", 8))
    assert a.assess_group(batch) == first == b.assess_group(batch)


def test_group_test_flags_a_far_batch_and_passes_a_near_batch(toy, machine):
    for size in (1, 4, 8, 16):
        far = [machine.assess_group(toy["fresh"]("U", size, seed_=100 + i)) for i in range(40)]
        near = [machine.assess_group(toy["fresh"]("A", size, seed_=200 + i)) for i in range(60)]
        assert all(a.group_size == size for a in far + near)
        assert rate(far, "unseen") >= (0.9 if size >= 4 else 0.6), (size, rate(far, "unseen"))
        assert rate(near, "unseen") <= 0.15, (size, rate(near, "unseen"))
    covered = [a for a in near if a.status != "unseen"]
    assert covered and all(a.skill == "A" for a in covered)


def test_flag_conflict_changes_only_a_covered_part(toy, machine):
    covered, far = toy["fresh"]("A", 1)[0], toy["fresh"]("U", 1)[0]
    base = machine.assess(covered)
    assert base.status in ("solid", "thin")
    c = machine.flag_conflict(covered, True)
    assert c.status == "conflict" and c.skill == base.skill and c.p_value == base.p_value
    assert machine.flag_conflict(covered, False) == base
    assert machine.assess(far).status == "unseen"
    assert machine.flag_conflict(far, True) == machine.assess(far)             # nothing known covers it, so there is nothing to contradict


# ------------------------------------------------------------------------------------------------------------------------------ macro F1
def test_macro_f1_ignores_an_absent_class_and_reports_it():
    truth = ["solid"] * 6 + ["unseen"] * 4
    pred = ["solid"] * 5 + ["unseen"] + ["unseen"] * 4
    r = ST.macro_f1(truth, pred)
    f_solid, f_unseen = 2 * 5 / (2 * 5 + 0 + 1), 2 * 4 / (2 * 4 + 1 + 0)
    assert r["absent"] == ["thin"] and set(r["per_class"]) == {"solid", "unseen"}
    assert r["macro_f1"] == pytest.approx((f_solid + f_unseen) / 2)
    # a class that never occurs in the truth but IS predicted is not absent: the machine invented it, and it scores zero
    r2 = ST.macro_f1(["solid"] * 4, ["solid", "solid", "solid", "thin"])
    assert r2["absent"] == ["unseen"] and r2["per_class"]["thin"] == 0.0
    assert r2["macro_f1"] == pytest.approx((2 * 3 / (2 * 3 + 1 + 0) + 0.0) / 2)


def test_macro_f1_refuses_labels_it_does_not_know_and_empty_input():
    with pytest.raises(ValueError, match="unknown"):
        ST.macro_f1(["solid", "grey"], ["solid", "solid"])
    with pytest.raises(ValueError, match="empty"):
        ST.macro_f1([], [])
    with pytest.raises(ValueError, match="same length"):
        ST.macro_f1(["solid"], ["solid", "thin"])


# ------------------------------------------------------------------------------------------------------------------------------ baselines
def fit_world(seed=0, n=300):
    w = make_toy(seed=seed, n_proto=n, n_cal=600)
    X = np.vstack([w["protos"][k] for k in TAUGHT])
    y = np.array([k for k in TAUGHT for _ in range(n)])
    return w, X, y


def baseline_of(name, w):
    background = vmf(w["mus"]["A"], 0.0, 2000, np.random.default_rng(5))                # general content: uniform on the sphere
    return BL.make_baseline(name, background=background)


def auroc(far, near):
    far, near = np.asarray(far), np.asarray(near)
    return float((far[:, None] > near[None, :]).mean() + 0.5 * (far[:, None] == near[None, :]).mean())


def test_there_are_seven_baselines_with_the_names_of_the_spec():
    assert BL.BASELINE_NAMES == ("vmf_conformal", "knn_distance", "mahalanobis", "relative_mahalanobis", "classifier_confidence", "energy", "autoencoder_descriptor")


@pytest.mark.parametrize("name", BL.BASELINE_NAMES)
def test_every_baseline_scores_higher_for_far_points(name):
    w, X, y = fit_world()
    b = baseline_of(name, w).fit(X, y)
    near, far = b.novelty_score(w["cal"]["A"]), b.novelty_score(w["fresh"]("U", 500))
    assert near.shape == (600,) and np.isfinite(near).all() and np.isfinite(far).all()
    assert far.mean() > near.mean()
    assert auroc(far, near) >= 0.95, (name, auroc(far, near))


@pytest.mark.parametrize("name", BL.BASELINE_NAMES)
def test_every_baseline_flags_about_q_of_held_out_taught_data_after_the_shared_calibration(name):
    w, X, y = fit_world()
    b = baseline_of(name, w).fit(X, y)
    cal = np.vstack([w["cal"][k] for k in TAUGHT])
    thr = BL.conformal_threshold(b.novelty_score(cal), 0.05)
    fresh = np.vstack([w["fresh"](k, 1000) for k in TAUGHT])
    assert float((b.novelty_score(fresh) > thr).mean()) <= 0.08
    far = w["fresh"]("U", 500)
    assert float((b.novelty_score(far) > thr).mean()) >= 0.8


def test_conformal_threshold_cannot_flag_with_too_few_calibration_scores():
    assert BL.conformal_threshold(np.arange(18.0), 0.05) == math.inf
    assert BL.conformal_threshold(np.arange(19.0), 0.05) == 18.0
    with pytest.raises(ValueError):
        BL.conformal_threshold(np.array([]), 0.05)
    with pytest.raises(ValueError, match="q must"):
        BL.conformal_threshold(np.arange(30.0), 1.0)


def test_baselines_refuse_what_they_cannot_fit():
    w, X, y = fit_world()
    with pytest.raises(ValueError, match="same number"):
        BL.make_baseline("knn_distance").fit(X, y[:-1])
    with pytest.raises(ValueError, match="at least two classes"):
        BL.make_baseline("classifier_confidence").fit(X, np.array(["A"] * len(X)))
    with pytest.raises(ValueError, match="before"):
        BL.make_baseline("energy").novelty_score(X)
    with pytest.raises(ValueError, match="background"):
        BL.make_baseline("relative_mahalanobis")
    with pytest.raises(ValueError, match="unknown baseline"):
        BL.make_baseline("nope")
    with pytest.raises(ValueError, match="finite"):
        BL.make_baseline("knn_distance").fit(np.full_like(X, np.nan), y)


def test_baselines_are_deterministic():
    w, X, y = fit_world()
    pts = w["fresh"]("U", 50)
    for name in BL.BASELINE_NAMES:
        a, b = baseline_of(name, w).fit(X, y).novelty_score(pts), baseline_of(name, w).fit(X, y).novelty_score(pts)
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("size", (1, 4, 8, 16))
def test_group_test_gives_a_small_p_value_for_a_far_batch_of_every_size(size):
    w, X, _ = fit_world()
    ref = w["protos"]["A"]
    far = [BL.group_test(ref, w["fresh"]("U", 16, seed_=300 + i), size) for i in range(20)]
    near = [BL.group_test(ref, w["fresh"]("A", 16, seed_=400 + i), size) for i in range(40)]
    assert all(0.0 < p <= 1.0 for p in far + near)
    assert np.median(far) < np.median(near)
    assert np.mean([p <= 0.05 for p in near]) <= 0.2
    assert np.mean([p <= 0.05 for p in far]) >= (0.5 if size == 1 else 0.9)


def test_group_test_uses_only_the_first_size_rows():
    w, _, _ = fit_world()
    ref, near, far = w["protos"]["A"], w["fresh"]("A", 4), w["fresh"]("U", 12)
    assert BL.group_test(ref, np.vstack([near, far]), 4) == BL.group_test(ref, near, 4)
    assert BL.group_test(ref, np.vstack([near, far]), 16) < BL.group_test(ref, np.vstack([near, far]), 4)


def test_group_test_needs_enough_rows_and_a_valid_size():
    w, _, _ = fit_world()
    ref, batch = w["protos"]["A"], w["fresh"]("A", 4)
    with pytest.raises(ValueError, match="size"):
        BL.group_test(ref, batch, 8)
    with pytest.raises(ValueError, match="size"):
        BL.group_test(ref, batch, 0)


def test_centred_unit_removes_the_shared_offset_that_makes_cosines_degenerate():
    rng = np.random.default_rng(0)
    offset = 8.0 * rng.standard_normal(D)
    X = offset + rng.standard_normal((400, D))
    unit = lambda z: z / np.linalg.norm(z, axis=1, keepdims=True)
    raw_cos = (unit(X) @ unit(X).T)[np.triu_indices(400, 1)].mean()
    f = BL.centred_unit(X)
    Z = f(X)
    assert raw_cos > 0.9
    assert (Z @ Z.T)[np.triu_indices(400, 1)].mean() < 0.1
    np.testing.assert_allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(f(X[:5]), Z[:5], atol=1e-9)                    # the centre is fixed at fit time, not re-estimated on what is passed in
    with pytest.raises(ValueError, match="centre"):
        f(X.mean(axis=0, keepdims=True))
