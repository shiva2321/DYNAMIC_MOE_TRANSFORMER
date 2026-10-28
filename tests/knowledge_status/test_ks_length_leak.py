"""The length-only leak check of anti-goal A1: deterministic, bounded, and able to see a leak when there is one."""

import numpy as np
import pytest

import check_length_leak as CL
import pools


def test_the_check_is_deterministic_and_reports_every_family_with_the_right_chance():
    a, b = CL.length_leak("dev", n=120, seed=3), CL.length_leak("dev", n=120, seed=3)
    assert a == b and list(a) == list(CL.FAMILIES)
    for fam, (req, whole, chance) in a.items():
        assert 0.0 <= req <= 1.0 and 0.0 <= whole <= 1.0
        assert chance == (0.333 if fam == "S5" else 0.167)                           # S5 has three skills, the others six


def test_the_check_sees_a_planted_length_leak(monkeypatch):
    """A pool where each skill has its own request length must read near 1.0 from the request length."""
    import generator as G
    import skills

    def fake_single(rng, pool, skill, family, *a, **k):
        text = "x" * (10 + 8 * skills.SKILLS.index(skill))
        return G.Message(text, (G.Part(0, len(text), "skill", skill, 1, 1, 2),), family, ("f", "c"), pool)

    monkeypatch.setattr(G, "make_single", fake_single)
    out = CL.length_leak("dev", n=300, seed=1)
    assert all(req >= 0.95 for req, _, _ in out.values())


def test_the_confirmatory_pool_is_not_much_leakier_than_the_dev_pool(monkeypatch):
    monkeypatch.setenv("KS_ALLOW_CONFIRM", "1")
    pools.load_pool.cache_clear()
    try:
        dev, con = CL.length_leak("dev", n=400), CL.length_leak("confirm", n=400)
    finally:
        pools.load_pool.cache_clear()
    for fam in CL.FAMILIES:
        assert con[fam][1] <= dev[fam][1] + 0.20, (fam, dev[fam], con[fam])          # whole-message length, which is what a surface-cue model would see
        assert con[fam][0] <= 0.65, (fam, con[fam])                                  # the request alone never gives the skill away
