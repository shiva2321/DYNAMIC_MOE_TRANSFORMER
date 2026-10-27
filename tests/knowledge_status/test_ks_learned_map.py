import numpy as np
import pytest

import ceiling
import learned_map as LM
import run_e1

TAUGHT = ("add", "sub", "mul")


def toy(n_per_cell=30, wording_scale=3.0, seed=0, templates=(0, 1, 2), skills_=TAUGHT):
    """Embeddings in which the skill lives in dimensions 0-3 and the wording in dimensions 4-11, and the wording is the louder of the two (a representation that sorts
    by wording). Template t belongs to the family S<t+1>; every item also has its own random wording noise inside the wording dimensions."""
    rng = np.random.default_rng(seed)
    skill_dir = np.eye(4)[: len(skills_)]
    template_dir = rng.standard_normal((6, 8))
    X, skill, family, template = [], [], [], []
    for k, s in enumerate(skills_):
        for t in templates:
            for _ in range(n_per_cell):
                wording = template_dir[t] + 0.7 * rng.standard_normal(8)
                X.append(np.r_[skill_dir[k] + 0.1 * rng.standard_normal(4), wording_scale * wording / np.linalg.norm(wording)])
                skill.append(s)
                family.append(f"S{t + 1}" if t < 3 else "S4")
                template.append(f"t{t}")
    return np.array(X, dtype=np.float32), np.array(skill), np.array(family), np.array(template)


def test_fit_refuses_data_from_held_out_families():
    X, skill, family, template = toy()
    for bad in ("S4", "S5", "S6", "S7", "S8", "unseen", "made_up"):
        f = family.astype("<U8")
        f[0] = bad
        with pytest.raises(ValueError, match="famil"):
            LM.LearnedMap(TAUGHT, epochs=2).fit(X, skill, f, template)
    LM.LearnedMap(TAUGHT, epochs=2).fit(X, skill, family, template)                      # the teaching wordings alone are fine


def test_fit_refuses_untaught_skills():
    X, skill, family, template = toy(skills_=("add", "sub", "mul", "div"))
    with pytest.raises(ValueError, match="div"):
        LM.LearnedMap(TAUGHT, epochs=2).fit(X, skill, family, template)
    keep = skill != "div"
    LM.LearnedMap(TAUGHT, epochs=2).fit(X[keep], skill[keep], family[keep], template[keep])


def test_none_rows_are_allowed_and_taught_set_must_be_known_skills():
    X, skill, family, template = toy()
    skill, family = skill.astype("<U8"), family.astype("<U8")                              # wide enough for "none" and "other"
    skill[:5] = "none"
    family[:5] = "other"
    LM.LearnedMap(TAUGHT, epochs=2).fit(X, skill, family, template)                      # "none" is not a skill, it is the class of everything else
    with pytest.raises(ValueError, match="skill"):
        LM.LearnedMap(("add", "teleport"))
    with pytest.raises(ValueError, match="at least two"):
        LM.LearnedMap(("add",), epochs=2).fit(X[skill == "add"], skill[skill == "add"], family[skill == "add"], template[skill == "add"])


def test_map_separates_a_constructed_toy_with_nuisance_directions():
    X, skill, family, template = toy(seed=1)
    m = LM.LearnedMap(TAUGHT, out_dim=8, epochs=300, seed=0)
    m.fit(X, skill, family, template)
    # fresh items in a wording nobody was fitted on (template 4: a new direction inside the wording dimensions, family S4)
    Xn, sn, fn, tn = toy(n_per_cell=10, seed=2, templates=(4,))
    Xa, sa, fa, ta = toy(n_per_cell=10, seed=3, templates=(0,))
    A, B = np.vstack([Xa, Xn]), np.r_[sa, sn]
    wording = np.r_[ta, tn]

    def auroc(Z):
        same, other = [], []
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                if B[i] == B[j] and wording[i] != wording[j]:
                    same.append((i, j))                                                # the same skill reworded
                elif B[i] != B[j] and wording[i] == wording[j]:
                    other.append((i, j))                                               # another skill, same wording
        return ceiling.pair_auroc(Z, same, other)

    assert auroc(A) < 0.2                                                              # before: wording wins, the twin is closer
    assert auroc(m.transform(A)) > 0.95                                                # after: the skill direction dominates, also in a wording never seen


def test_transform_is_unit_length_deterministic_and_needs_a_fit():
    X, skill, family, template = toy()
    with pytest.raises(RuntimeError):
        LM.LearnedMap(TAUGHT).transform(X)
    a, b = (LM.LearnedMap(TAUGHT, out_dim=6, epochs=20, seed=5) for _ in range(2))
    for m in (a, b):
        m.fit(X, skill, family, template)
    Za, Zb = a.transform(X), b.transform(X)
    assert Za.dtype == np.float32 and Za.shape == (len(X), 6)
    assert np.allclose(np.linalg.norm(Za, axis=1), 1.0, atol=1e-5)
    assert np.allclose(Za, Zb, atol=1e-5)                                              # the same seed gives the same map
    c = LM.LearnedMap(TAUGHT, out_dim=6, epochs=20, seed=6)
    c.fit(X, skill, family, template)
    assert not np.allclose(Za, c.transform(X), atol=1e-3)


def test_make_mapper_follows_the_ceiling_protocol_and_inherits_the_refusals():
    items = ceiling.build_items(100, "dev", ceiling.Sizes(teach=20, test=10, pairs=10, assignments=2))
    teach = [i for i in items.role("teach")]
    rng = np.random.default_rng(0)
    X = rng.standard_normal((len(teach), 16)).astype(np.float32)
    mapper = LM.make_mapper(7, epochs=3, out_dim=4)
    f = mapper(X, teach, ("add", "sub", "mul", "div", "mod", "max"))
    Z = f(X[:5])
    assert Z.shape == (5, 4) and np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-5)
    stories = [i for i in items.role("test") if i.family == "S4"][:3]
    with pytest.raises(ValueError, match="famil"):
        mapper(np.vstack([X[:10], rng.standard_normal((3, 16)).astype(np.float32)]), teach[:10] + stories, ("add", "sub", "mul", "div", "mod", "max"))


def test_an_r5_cell_runs_end_to_end_and_has_its_own_code_stamp(tmp_path):
    small = ceiling.Sizes(teach=30, test=20, pairs=30, assignments=3)
    stamp = run_e1.stamp_for("R5_R0_hash")
    assert stamp != run_e1.stamp_for("R0_hash")                                        # the learned map's code is part of an R5 result, not of a plain one
    r = run_e1.run_cell("R5_R0_hash", 100, "dev", tmp_path, stamp, sizes=small)
    assert r["rep"] == "R5_R0_hash" and r["code_stamp"] == stamp
    for k in ("b1a", "b1b", "b1c", "b1b_taught", "b1b_story"):
        assert 0.0 <= r[k] <= 1.0
    assert (tmp_path / "R5_R0_hash__s100.json").exists()
