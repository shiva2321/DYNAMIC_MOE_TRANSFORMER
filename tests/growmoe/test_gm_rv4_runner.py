"""RV4 runner: cell identity, arm flags and the declared selection rules of the dev constants (spec sections 6.7 and 8). The runner files live in research/growmoe_results/rv2d/."""

import sys
from pathlib import Path

import pytest

RV2D = Path(__file__).resolve().parents[2] / "research" / "growmoe_results" / "rv2d"
sys.path.insert(0, str(RV2D))

import rv2d_lib as R        # noqa: E402
import rv2d_select as S     # noqa: E402


def test_cell_ids_are_unique_file_safe_and_dot_free():
    cells = [R.Cell("s3", s, a, seed, gap=g, tmargin=t) for s in ("OV-0.6", "OV-0.85", "F-15") for a, g, t in (("S3", 0, 0), ("S3+G", 0.2, 0), ("S3+G", 0.23, 0), ("S3+T", 0, 0.25), ("S3+T+G", 0.2, 0.25))
             for seed in (100, 101)]
    ids = [c.cid for c in cells]
    assert len(set(ids)) == len(ids) and all("." not in i and "+" not in i and "/" not in i for i in ids)


def test_arm_flags_follow_the_arm_and_refuse_a_missing_constant():
    off = {"router": False, "spawn_rule": "probe", "growth_rule": "probe"}      # the arms are defined against the pool before stage 4 (the default since the adoption of 2026-10-21): an arm lists what it turns ON
    assert R.arm_flags(R.Cell("s3", "F-15", "S3")) == off
    assert R.arm_flags(R.Cell("s3", "F-15", "S3+G", gap=0.2)) == {**off, "growth_rule": "gap", "grow_gap": 0.2}
    assert R.arm_flags(R.Cell("s3", "F-15", "S3+T", tmargin=0.25)) == {**off, "spawn_rule": "transfer", "transfer_margin": 0.25}
    assert R.arm_flags(R.Cell("s3", "F-15", "S3+T+G", gap=0.23, tmargin=0.15)) == {**off, "spawn_rule": "transfer", "growth_rule": "gap", "grow_gap": 0.23, "transfer_margin": 0.15}
    with pytest.raises(ValueError):
        R.arm_flags(R.Cell("s3", "F-15", "S3+G"))                      # a growth arm without its threshold
    with pytest.raises(ValueError):
        R.arm_flags(R.Cell("s3", "F-15", "S3+T"))                      # a transfer arm without its margin


def test_function_regime_cells_get_a_pool_seed_that_differs_from_the_stream_seed():
    """E-035: with pool seed == stream seed the expert founded for regime r starts with regime r's teacher units. Amendment A4-2: the function-regime family uses stream seed + 7."""
    assert R.POOL_SEED_OFFSET == 7
    for stream in ("F-15", "OV-0.6", "S1", "S2-5"):
        assert R.pool_seed(R.Cell("s3", stream, "S3", 100)) == 107
    for family in ("arith-seq", "arith-mixed16", "arith-mixed128", "banking"):                 # no teacher in these streams: their seeds are untouched
        assert R.pool_seed(R.Cell(family, "-", "S3", 100)) == 100


def _row(family, stream, arm, seed, acc, params=1000, n_grow=0, gap=0.0, tmargin=0.0, experts=4, router=None):
    return {"family": family, "stream": stream, "arm": arm, "seed": seed, "gap": gap, "tmargin": tmargin, "acc": acc, "params": params, "n_grow": n_grow, "n_experts": experts,
            "router": router or {}}


def test_choose_grow_gap_maximises_the_declared_score_under_the_guards():
    rows = []
    for seed in (100, 101):
        for fam, stream in (("arith-mixed16", "-"), ("arith-mixed128", "-"), ("s3", "F-15"), ("s3", "OV-0.6")):
            rows.append(_row(fam, stream, "S3", seed, 0.30))
            for g, gain, grow, relp in ((0.14, 0.02, 1, 0.1), (0.17, 0.05, 1, 0.2), (0.20, 0.10, 1, 0.2), (0.23, 0.11, 3, 0.9), (0.26, 0.11, 3, 0.9)):
                if fam == "arith-mixed16":
                    gain += 0.10 * (g - 0.14) / 0.12                    # the starved expert profits from the larger thresholds
                rows.append(_row(fam, stream, "S3+G", seed, 0.30 + gain, params=int(1000 * (1 + relp)), n_grow=grow, gap=g))
    out = S.choose_grow_gap(rows)
    assert out["chosen"] == 0.20                                         # 0.23 and 0.26 break the guard (3 growth events, +90 % parameters); 0.20 beats 0.17 on the score
    assert set(out["curve"]) == {0.14, 0.17, 0.20, 0.23, 0.26} and out["curve"][0.23]["guards_ok"] is False


def test_choose_grow_gap_reports_no_choice_when_no_value_satisfies_the_guards():
    rows = []
    for seed in (100, 101):
        for fam, stream in (("arith-mixed16", "-"), ("arith-mixed128", "-"), ("s3", "F-15"), ("s3", "OV-0.6")):
            rows.append(_row(fam, stream, "S3", seed, 0.30))
            for g in (0.14, 0.17, 0.20, 0.23, 0.26):
                rows.append(_row(fam, stream, "S3+G", seed, 0.20, n_grow=5, gap=g))              # every value loses 10 points and grows 5 times
    out = S.choose_grow_gap(rows)
    assert out["chosen"] is None and "per-family" in out["reason"]


def test_choose_router_setting_takes_the_best_mean_gain_among_the_settings_that_hurt_no_guard_stream():
    gains = {                                                              # setting -> gain over no router, per stream (fractions of accuracy)
        "restrict-0.01": {("arith-seq", "-"): 0.12, ("s3", "OV-0.85"): 0.08, ("s3", "S1"): 0.0, ("s3", "F-15"): 0.0},
        "restrict-0.03": {("arith-seq", "-"): 0.12, ("s3", "OV-0.85"): 0.09, ("s3", "S1"): 0.0, ("s3", "F-15"): -0.02},       # loses 2 points on a guard stream
        "free-0.01": {("arith-seq", "-"): 0.13, ("s3", "OV-0.85"): 0.10, ("s3", "S1"): -0.01, ("s3", "F-15"): 0.0},          # loses 1 point on a guard stream
        "free-0.03": {("arith-seq", "-"): 0.05, ("s3", "OV-0.85"): 0.02, ("s3", "S1"): 0.0, ("s3", "F-15"): 0.0},
    }
    rows = []
    for fam, stream in gains["restrict-0.01"]:
        for seed in (100, 101):
            router = {"off": {"acc": 0.40}}
            router.update({s: {"acc": 0.40 + g[(fam, stream)], "selected": "lda"} for s, g in gains.items()})
            rows.append(_row(fam, stream, "S3", seed, 0.40, router=router))
    out = S.choose_router_setting(rows)
    assert out["chosen"] == "restrict-0.01"                                    # restrict-0.03 and free-0.01 break the -0.5 point guard; free-0.03 is feasible but gains less
    assert out["curve"]["free-0.01"]["guards_ok"] is False and out["curve"]["free-0.03"]["guards_ok"] is True


def test_choose_transfer_margin_prefers_the_default_on_a_tie_and_respects_the_expert_guard():
    rows = []
    for seed in (100, 101):
        for stream, fam in (("-", "arith-seq"), ("S1", "s3"), ("S2-5", "s3")):
            rows.append(_row(fam, stream, "S3", seed, 0.30, experts=4))
            for t, gain, exp in ((0.15, 0.10, 4), (0.25, 0.10, 4), (0.35, 0.11, 9)):
                rows.append(_row(fam, stream, "S3+T", seed, 0.30 + gain, tmargin=t, experts=exp))
    out = S.choose_transfer_margin(rows)
    assert out["chosen"] == 0.25                                               # 0.35 breaks the 1.2x expert guard; 0.15 and 0.25 tie, the default wins


def test_choose_transfer_margin_counts_experts_against_the_true_functions_when_the_rows_have_them():
    """Amendment A4-1: S3 under-spawns (3 experts for 5 functions); the transfer rule's 5 are right and must not break the expert guard, 9 are a cascade."""
    rows = []
    for seed in (100, 101):
        rows.append(_row("arith-seq", "-", "S3", seed, 0.30, experts=4))
        rows.append(dict(_row("s3", "S2-5", "S3", seed, 0.30, experts=3), regimes=5))
        for t, exp in ((0.15, 5), (0.25, 5), (0.35, 9)):
            rows.append(_row("arith-seq", "-", "S3+T", seed, 0.40, tmargin=t, experts=6))
            rows.append(dict(_row("s3", "S2-5", "S3+T", seed, 0.31, tmargin=t, experts=exp), regimes=5))
    out = S.choose_transfer_margin(rows)
    assert out["curve"][0.25]["guards_ok"] is True and out["curve"][0.15]["guards_ok"] is True and out["curve"][0.35]["guards_ok"] is False
    assert out["chosen"] == 0.25


def test_the_code_stamp_changes_with_any_stamped_file_and_ignores_line_endings(tmp_path):
    """A cell result is valid only for the code that produced it: the stamp is a hash of the source files, with CRLF and LF alike."""
    (tmp_path / "a.py").write_bytes(b"x = 1\r\n")
    (tmp_path / "b.py").write_bytes(b"y = 1\n")
    first = R.code_stamp(tmp_path, ["a.py", "b.py"])
    (tmp_path / "a.py").write_bytes(b"x = 1\n")
    assert R.code_stamp(tmp_path, ["a.py", "b.py"]) == first
    (tmp_path / "b.py").write_bytes(b"y = 2\n")
    assert R.code_stamp(tmp_path, ["a.py", "b.py"]) != first
    assert R.code_stamp(tmp_path, ["a.py"]) != R.code_stamp(tmp_path, ["a.py", "b.py"])
    assert len(R.code_stamp()) == 16                                                   # the default: every source file of the growmoe package and this module


def test_the_selection_cli_can_write_the_router_curve_whose_keys_are_tuples():
    """The router curve is keyed by (family, stream); the first real dev1 run crashed on json.dumps with 'keys must be str ... not tuple'."""
    import json
    import rv2d_select_cli as C
    out = C._plain({("s3", "OV-0.85"): {"gain": 0.1}, "a": [{("x", "y"): 1}], 3: "n"})
    assert out == {"s3:OV-0.85": {"gain": 0.1}, "a": [{"x:y": 1}], "3": "n"}
    json.dumps(out)
    rows = [{"family": "s3", "stream": "OV-0.85", "arm": "S3", "seed": 100, "router": {"off": {"acc": 0.3}, "restrict-0.01": {"acc": 0.4}, "restrict-0.03": {"acc": 0.4}, "free-0.01": {"acc": 0.4}, "free-0.03": {"acc": 0.4}}}]
    json.dumps(C._plain(S.choose_router_setting(rows)))
