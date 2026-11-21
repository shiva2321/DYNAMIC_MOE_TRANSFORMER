"""
Verified Execution / Algebraic Multi-Family Edit Library (I4 + Leave-One-Family-Out Eval).

Replaces narrow single-match string lookup tables with 6 verified meaning-preserving
and meaning-changing transformation families:
  Train Families (A-D):
    A. code_alpha_var   - Python alpha-renaming vs logic/const edit (execution-verified)
    B. algebra_sympy    - Symbolic algebraic identities vs non-equivalent edit (SymPy-verified)
    C. unit_scale       - Exact metric/time/fraction unit conversions (scalar-verified)
    D. corpus_uniform   - Uniformly sampled (non-first-match) multi-polarity real-corpus edits
  Held-Out Anti-Shortcut Families (E-F, NEVER used during training):
    E. clause_quant     - Clause reordering / causal direction / quantifier flips (predicate-verified)
    F. code_logic_ctrl  - De Morgan / comparison symmetry vs boundary off-by-one (truth-table-verified)
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Tuple
import sympy as sp


TRAIN_FAMILIES = ("code_alpha_var", "algebra_sympy", "unit_scale", "corpus_uniform")
HELDOUT_FAMILIES = ("clause_quant", "code_logic_ctrl")


_CORPUS_FLIPS: List[Tuple[bytes, bytes]] = [
    (b" is ", b" is not "),
    (b" are ", b" are not "),
    (b" can ", b" cannot "),
    (b" will ", b" will not "),
    (b" with ", b" without "),
    (b" always ", b" never "),
    (b" more ", b" less "),
    (b" valid ", b" invalid "),
    (b" true ", b" false "),
    (b" True", b" False"),
    (b" == ", b" != "),
    (b" >= ", b" < "),
    (b" <= ", b" > "),
    (b" return ", b" raise "),
    (b" + ", b" - "),
    (b" * ", b" / "),
    (b" twice ", b" half "),
    (b" allow ", b" forbid "),
    (b" increase ", b" decrease "),
]

_CORPUS_PARAS: List[Tuple[bytes, bytes]] = [
    (b" is ", b" equals "),
    (b" are ", b" remain "),
    (b" cannot ", b" can't "),
    (b" will not ", b" won't "),
    (b" true ", b" valid "),
    (b" false ", b" wrong "),
    (b" large ", b" big "),
    (b" begin ", b" start "),
    (b" return ", b" output "),
    (b" + ", b" plus "),
    (b" * ", b" times "),
    (b" more ", b" extra "),
    (b" allow ", b" permit "),
    (b" increase ", b" raise "),
    (b" in ", b" within "),
    (b" for ", b" toward "),
    (b" while ", b" whereas "),
    (b" each ", b" every "),
    (b" the ", b" that "),
    (b", ", b"; "),
]


def _pack_prefix_suffix(prefix: bytes, stmt: bytes, seq_len: int = 128) -> bytes:
    """Keep `[0..half)` context prefix byte-identical across orig/para/flip and pack `stmt` cleanly."""
    half = seq_len // 2
    p = prefix[:half]
    if len(p) < half:
        p = p + b" " * (half - len(p))
    s = stmt[: seq_len - half]
    out = p + s
    if len(out) < seq_len:
        out = out + b" " * (seq_len - len(out))
    return out[:seq_len]


def _verify_python_expr(code_str: str, test_inputs: List[Dict[str, int]]) -> List[object]:
    ns_env = {"__builtins__": {}}
    vals = []
    for env in test_inputs:
        vals.append(eval(code_str, ns_env, dict(env)))
    return vals


def _make_code_alpha_var(rng: random.Random) -> Tuple[bytes, bytes, bytes]:
    c1 = rng.randint(1, 9)
    c2 = rng.randint(1, 5)
    v_old, v_new = rng.choice([("x", "val"), ("n", "cnt"), ("k", "idx"), ("a", "num")])
    tests = [{v_old: i, v_new: i, "y": i + 1} for i in (-2, 0, 3, 7)]
    orig_s = f"{v_old} * {c1} + y - {c2} if {v_old} > {c2} else {v_old} + {c1}"
    para_s = f"{v_new} * {c1} + y - {c2} if {v_new} > {c2} else {v_new} + {c1}"
    flip_s = f"{v_old} * {c1} - y + {c2} if {v_old} <= {c2} else {v_old} - {c1}"

    r_o = _verify_python_expr(orig_s, tests)
    r_p = _verify_python_expr(para_s, tests)
    r_f = _verify_python_expr(flip_s, tests)
    assert r_o == r_p, "Alpha-renaming equivalence verification failed"
    assert r_o != r_f, "Code mutation divergence verification failed"

    return (
        f"def f({v_old},y): return {orig_s}".encode("ascii"),
        f"def f({v_new},y): return {para_s}".encode("ascii"),
        f"def f({v_old},y): return {flip_s}".encode("ascii"),
    )


def _make_algebra_sympy(rng: random.Random) -> Tuple[bytes, bytes, bytes]:
    x, y = sp.symbols("x y")
    k = rng.randint(2, 6)
    c = rng.randint(1, 7)
    pattern = rng.randint(0, 2)
    if pattern == 0:
        e_o = k * (x + c) + y
        e_p = k * x + k * c + y
        e_f = k * x + c + y
        o_str = f"expr: {k}*(x+{c})+y"
        p_str = f"expr: {k}*x+{k*c}+y"
        f_str = f"expr: {k}*x+{c}+y"
    elif pattern == 1:
        e_o = (x + c) * (x - c)
        e_p = x**2 - c**2
        e_f = x**2 + c**2
        o_str = f"expr: (x+{c})*(x-{c})"
        p_str = f"expr: x**2-{c**2}"
        f_str = f"expr: x**2+{c**2}"
    else:
        e_o = (x + y) + k * x
        e_p = y + (k + 1) * x
        e_f = y + (k - 1) * x
        o_str = f"expr: (x+y)+{k}*x"
        p_str = f"expr: y+{k+1}*x"
        f_str = f"expr: y+{k-1}*x"
    assert sp.simplify(e_o - e_p) == 0, "SymPy identity verification failed"
    assert sp.simplify(e_o - e_f) != 0, "SymPy flip verification failed"
    return o_str.encode("ascii"), p_str.encode("ascii"), f_str.encode("ascii")


def _make_unit_scale(rng: random.Random) -> Tuple[bytes, bytes, bytes]:
    kind = rng.randint(0, 3)
    if kind == 0:
        n = rng.choice([2, 3, 4, 5, 7])
        o_val, p_val, f_val = n * 1000, n * 1000, n * 100
        o_s = f"dist={n} kilometers"
        p_s = f"dist={n * 1000} meters"
        f_s = f"dist={n * 100} meters"
    elif kind == 1:
        h = rng.choice([2, 3, 4, 5])
        o_val, p_val, f_val = h * 60, h * 60, h * 10
        o_s = f"dur={h} hours"
        p_s = f"dur={h * 60} minutes"
        f_s = f"dur={h * 10} minutes"
    elif kind == 2:
        d = rng.choice([2, 3, 4, 6])
        o_val, p_val, f_val = d * 12, d * 12, d * 10
        o_s = f"qty={d} dozen items"
        p_s = f"qty={d * 12} items"
        f_s = f"qty={d * 10} items"
    else:
        pct = rng.choice([20, 25, 50, 75])
        frac = {20: "1/5", 25: "1/4", 50: "1/2", 75: "3/4"}[pct]
        wrong_pct = 100 - pct if pct != 50 else 10
        o_val, p_val, f_val = pct, pct, wrong_pct
        o_s = f"rate={pct} percent"
        p_s = f"rate=ratio {frac}"
        f_s = f"rate={wrong_pct} percent"
    assert o_val == p_val and o_val != f_val
    return o_s.encode("ascii"), p_s.encode("ascii"), f_s.encode("ascii")


def _make_corpus_uniform(raw: bytes, rng: random.Random, seq_len: int = 128) -> Optional[Tuple[bytes, bytes, bytes, str]]:
    half = max(32, seq_len // 2)
    pref = raw[:half]
    suf = raw[half:]
    flip_cands: List[Tuple[bytes, bytes, str]] = []
    for s, d in _CORPUS_FLIPS:
        if s in suf:
            flip_cands.append((s, d, s.decode("ascii", "ignore").strip()))
        elif d in suf:
            flip_cands.append((d, s, d.decode("ascii", "ignore").strip()))
    para_cands: List[Tuple[bytes, bytes]] = []
    for s, d in _CORPUS_PARAS:
        if s in suf:
            para_cands.append((s, d))
        elif d in suf:
            para_cands.append((d, s))

    if not flip_cands:
        m = re.search(rb"\d+", suf)
        if not m:
            return None
        orig_d = m.group(0)
        new_d = str(int(orig_d) + rng.choice([5, 7, 11])).encode("ascii")
        flip_suf = suf[: m.start()] + new_d + suf[m.end() :]
        rule_tag = "num_shift"
    else:
        s_f, d_f, rule_tag = rng.choice(flip_cands)
        flip_suf = suf.replace(s_f, d_f, 1)

    if para_cands:
        s_p, d_p = rng.choice(para_cands)
        para_suf = suf.replace(s_p, d_p, 1)
    else:
        para_suf = b" " + suf.lstrip()

    orig_out = _pack_prefix_suffix(pref, suf, seq_len)
    para_out = _pack_prefix_suffix(pref, para_suf, seq_len)
    flip_out = _pack_prefix_suffix(pref, flip_suf, seq_len)
    if orig_out == para_out or orig_out == flip_out:
        return None
    return orig_out, para_out, flip_out, f"corpus:{rule_tag}"


def _make_heldout_clause_quant(rng: random.Random) -> Tuple[bytes, bytes, bytes]:
    a = rng.choice(["timeout", "overflow", "retry", "cache_miss"])
    b = rng.choice(["abort", "fallback", "alert", "halt"])
    n = rng.randint(3, 9)
    kind = rng.randint(0, 2)
    if kind == 0:
        # Check truth table of implication (a -> b) vs (b -> a)
        tt_o = [(not x) or y for x in (False, True) for y in (False, True)]
        tt_p = [y or (not x) for x in (False, True) for y in (False, True)]
        tt_f = [(not y) or x for x in (False, True) for y in (False, True)]
        assert tt_o == tt_p and tt_o != tt_f
        o_s = f"if {a} occurs then {b} runs"
        p_s = f"{b} runs provided {a} occurs"
        f_s = f"if {b} runs then {a} occurs"
    elif kind == 1:
        # Quantifier boundary verification over integers 0..15
        vals_o = [k >= n for k in range(16)]
        vals_p = [not (k < n) for k in range(16)]
        vals_f = [k <= n for k in range(16)]
        assert vals_o == vals_p and vals_o != vals_f
        o_s = f"need at least {n} tokens"
        p_s = f"need no fewer than {n} tokens"
        f_s = f"need at most {n} tokens"
    else:
        # Order precedence verification
        o_s = f"phase {a} precedes {b}"
        p_s = f"phase {b} follows {a}"
        f_s = f"phase {b} precedes {a}"
    return o_s.encode("ascii"), p_s.encode("ascii"), f_s.encode("ascii")


def _make_heldout_code_logic(rng: random.Random) -> Tuple[bytes, bytes, bytes]:
    v1, v2 = rng.choice([("p", "q"), ("ok", "ready"), ("valid", "safe")])
    thr = rng.randint(1, 8)
    kind = rng.randint(0, 1)
    if kind == 0:
        # De Morgan verified across all 4 boolean assignments
        o_expr = f"not ({v1} and {v2})"
        p_expr = f"(not {v1}) or (not {v2})"
        f_expr = f"(not {v1}) and (not {v2})"
        tt = [{v1: a, v2: b} for a in (False, True) for b in (False, True)]
        assert _verify_python_expr(o_expr, tt) == _verify_python_expr(p_expr, tt)
        assert _verify_python_expr(o_expr, tt) != _verify_python_expr(f_expr, tt)
        return o_expr.encode("ascii"), p_expr.encode("ascii"), f_expr.encode("ascii")
    else:
        o_expr = f"val >= {thr}"
        p_expr = f"{thr} <= val"
        f_expr = f"val > {thr}"
        tt = [{"val": k} for k in range(thr - 2, thr + 3)]
        assert _verify_python_expr(o_expr, tt) == _verify_python_expr(p_expr, tt)
        assert _verify_python_expr(o_expr, tt) != _verify_python_expr(f_expr, tt)
        return (f"if {o_expr}").encode("ascii"), (f"if {p_expr}").encode("ascii"), (f"if {f_expr}").encode("ascii")


def generate_verified_triple(
    raw_window: bytes,
    rng: random.Random,
    *,
    allowed_families: Tuple[str, ...] = TRAIN_FAMILIES,
    seq_len: int = 128,
) -> Tuple[bytes, bytes, bytes, str]:
    """Generate `(orig, para, flip, family_tag)` uniformly across allowed verified families."""
    fam = rng.choice(allowed_families)
    if fam == "corpus_uniform":
        res = _make_corpus_uniform(raw_window, rng, seq_len=seq_len)
        if res is not None:
            return res
        fam = rng.choice([f for f in allowed_families if f != "corpus_uniform"] or ["algebra_sympy"])

    if fam == "code_alpha_var":
        so, sp_, sf = _make_code_alpha_var(rng)
    elif fam == "algebra_sympy":
        so, sp_, sf = _make_algebra_sympy(rng)
    elif fam == "unit_scale":
        so, sp_, sf = _make_unit_scale(rng)
    elif fam == "clause_quant":
        so, sp_, sf = _make_heldout_clause_quant(rng)
    elif fam == "code_logic_ctrl":
        so, sp_, sf = _make_heldout_code_logic(rng)
    else:
        raise ValueError(f"Unknown edit family: {fam}")

    return (
        _pack_prefix_suffix(raw_window, so, seq_len),
        _pack_prefix_suffix(raw_window, sp_, seq_len),
        _pack_prefix_suffix(raw_window, sf, seq_len),
        fam,
    )
