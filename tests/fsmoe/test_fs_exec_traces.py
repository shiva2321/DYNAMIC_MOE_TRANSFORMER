import numpy as np
import pytest

from fsmoe import exec_traces as EX
from fsmoe.domains import code as CO
from fsmoe.domains.base import rng_for


def _out(fn):
    return np.asarray([float(fn(list(p))) for p in CO.PROBES["list"]])


def test_every_library_idiom_computes_its_own_family():
    """The labels of the library snippets are right: each matches its family's reference outputs (taught and near-twin families) on the probe lists."""
    for fam in EX.TAUGHT + EX.NEAR:
        ref = np.asarray([float(CO._ref(fam, list(p))) for p in CO.PROBES["list"]])
        for k, src in enumerate(EX.LIB[fam]):
            fn = EX.load_fn(src.format(f="probe_fn"), "probe_fn")
            assert np.allclose(_out(fn), ref), (fam, k)


def test_far_library_functions_match_no_taught_family():
    refs = {f: np.asarray([float(CO._ref(f, list(p))) for p in CO.PROBES["list"]]) for f in EX.TAUGHT}
    for fam in EX.FAR_LIB:
        for src in EX.LIB[fam]:
            fn = EX.load_fn(src.format(f="probe_fn"), "probe_fn")
            assert EX.fingerprint_distance(fn, refs) > 0.05, fam


def test_the_fingerprint_is_zero_for_known_code_and_large_for_failures():
    refs = {f: np.asarray([float(CO._ref(f, list(p))) for p in CO.PROBES["list"]]) for f in EX.TAUGHT}
    rng = rng_for(1, "t")
    for style in CO.TEACH + CO.HELDOUT:
        text, fn = CO.render("max_list", style, rng)
        assert EX.fingerprint_distance(EX.load_fn(text, fn), refs) < 1e-9
    assert EX.fingerprint_distance(lambda x: x[10], refs) == 1e6


def test_prefix_steps_are_the_recurrences_of_the_functions_in_every_style():
    rng = rng_for(2, "t")
    lists = EX.draw_lists(rng, 8)
    rules = {"sum_list": lambda p, e: p + e, "product_list": lambda p, e: p * e, "max_list": lambda p, e: max(p, e), "min_list": lambda p, e: min(p, e),
             "count_even": lambda p, e: p + (e % 2 == 0), "count_odd": lambda p, e: p + (e % 2 == 1), "count_positive": lambda p, e: p + (e > 0)}
    for fam, rule in rules.items():
        for style in CO.TEACH + CO.HELDOUT:
            text, fn = CO.render(fam, style, rng)
            outs = EX.prefix_outcomes(EX.load_fn(text, fn), lists)
            prev, el, nxt = EX.steps(lists, outs)
            assert np.allclose([rule(p, e) for p, e in zip(prev, el)], nxt), (fam, style)
    assert EX.prefix_outcomes(lambda x: x[10], lists) is None


def test_steps_count_and_snippet_set_are_well_formed():
    lists = [[1, 2, 3, 4, 5], [2, 2, 2, 2, 2, 2]]
    outs = [[3.0, 6.0, 10.0, 15.0], [4.0, 6.0, 8.0, 10.0, 12.0]]
    prev, el, nxt = EX.steps(lists, outs)
    assert len(prev) == 3 + 4 and prev[0] == 3.0 and el[0] == 3 and nxt[0] == 6.0
    snips = EX.build_snippets(3)
    roles = {s.role for s in snips}
    assert roles == {"known_tw", "known_ho", "near", "far"}
    assert all(s.family in EX.TAUGHT for s in snips if s.role.startswith("known")) and all(s.family in EX.NEAR for s in snips if s.role == "near")
    assert {s.kind for s in snips} == {"template", "library"}


def test_a_step_expert_learns_the_function_it_was_taught_and_scores_a_different_one_high():
    prev, el, nxt = EX.teaching_steps(5, "sum_list", 30)
    sumx = EX.StepExpert(0, epochs=40).fit(prev, el, nxt)
    rng = rng_for(5, "eval")
    lists = EX.draw_lists(rng, 6)
    ok = EX.steps(lists, EX.prefix_outcomes(EX.load_fn("def f(x):\n    return sum(x)\n", "f"), lists))
    twin = EX.steps(lists, EX.prefix_outcomes(EX.load_fn("import math\ndef f(x):\n    return math.prod(x)\n", "f"), lists))
    assert sumx.score(*ok) < 5.0 and sumx.score(*twin) > 10 * sumx.score(*ok)
