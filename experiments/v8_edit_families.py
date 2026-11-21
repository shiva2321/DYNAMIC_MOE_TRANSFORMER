"""
v8 Edit Families & I7 Procedural Structural Pre-Training Generator.

Strict anti-shortcut separation (`test_v8_edit_families_disjoint`):
  - Re-exports the EXACT frozen `HELDOUT_FAMILIES` (`clause_quant`, `code_logic_ctrl`)
    and evaluators from `experiments/v7_edit_families.py` so F4b comparison vs v7 is
    strictly apples-to-apples with zero shifted goalposts.
  - Enriches `TRAIN_FAMILIES` (`code_alpha_var`, `algebra_sympy`, `unit_scale`, `corpus_uniform`)
    with execution- / SymPy-verified asymmetric order, grouping, and identifier invariance
    WITHOUT any template or phrase overlap with `HELDOUT_FAMILIES`.
  - Adds `generate_i7_procedural_batch` (pure symbolic Dyck bracket nesting, commutative vs
    non-commutative argument swaps, and cardinality boundary sensitivity) for `P0` pre-training.
"""

from __future__ import annotations

import random
from typing import List, Tuple
import sympy as sp
import torch

from experiments.v7_edit_families import (
    HELDOUT_FAMILIES,
    TRAIN_FAMILIES,
    _CORPUS_FLIPS,
    _CORPUS_PARAS,
    _make_corpus_uniform,
    _make_unit_scale,
    _pack_prefix_suffix,
    _verify_python_expr,
    generate_verified_triple as generate_v7_triple,
)

# Forbidden held-out signature phrases that must NEVER appear inside v8 TRAIN/I7 templates
HELDOUT_FORBIDDEN_PHRASES = (
    "occurs",
    "provided",
    "runs",
    "precedes",
    "follows",
    "tokens",
    "no fewer",
    "not (",
)


def _make_v8_code_alpha_var(rng: random.Random) -> Tuple[bytes, bytes, bytes]:
    mode = rng.randint(0, 2)
    c1 = rng.randint(2, 9)
    c2 = rng.randint(1, 6)
    v_old, v_new = rng.choice([("x", "u_val"), ("n", "item_n"), ("k", "idx_k"), ("a", "num_a")])
    v2_old, v2_new = rng.choice([("y", "w_val"), ("m", "rhs_m")])
    if mode == 0:
        tests = [{v_old: i, v_new: i, v2_old: i + 1} for i in (-2, 0, 3, 7)]
        orig_s = f"{v_old} * {c1} + {v2_old} - {c2} if {v_old} > {c2} else {v_old} + {c1}"
        para_s = f"{v_new} * {c1} + {v2_old} - {c2} if {v_new} > {c2} else {v_new} + {c1}"
        flip_s = f"{v_old} * {c1} - {v2_old} + {c2} if {v_old} <= {c2} else {v_old} - {c1}"
    elif mode == 1:
        # Non-commutative subtraction & sign-compensated swap (execution-verified)
        tests = [{v_old: i, v_new: i, v2_old: i + 3, v2_new: i + 3} for i in (-3, 1, 5)]
        orig_s = f"({v_old} - {v2_old}) * {c1} + {c2}"
        para_s = f"-({v2_new} - {v_new}) * {c1} + {c2}"
        flip_s = f"({v2_old} - {v_old}) * {c1} + {c2}"
    else:
        # Chained non-commutative functional composition vs variable rename (execution-verified)
        tests = [{v_old: i, v_new: i, v2_old: i + 2, v2_new: i + 2} for i in (-1, 2, 4)]
        orig_s = f"({v_old} + {c1}) * {v2_old} - {c2}"
        para_s = f"{v2_new} * ({c1} + {v_new}) - {c2}"
        flip_s = f"({v_old} + {c1}) * {c2} - {v2_old}"
    assert _verify_python_expr(orig_s, tests) == _verify_python_expr(para_s, tests)
    assert _verify_python_expr(orig_s, tests) != _verify_python_expr(flip_s, tests)
    return (
        f"fn_{v_old}({v_old},{v2_old}) => {orig_s}".encode("ascii"),
        f"fn_{v_new}({v_new},{v2_new if mode >= 1 else v2_old}) => {para_s}".encode("ascii"),
        f"fn_{v_old}({v_old},{v2_old}) => {flip_s}".encode("ascii"),
    )


def _make_v8_algebra_sympy(rng: random.Random) -> Tuple[bytes, bytes, bytes]:
    x, y, z = sp.symbols("x y z")
    k = rng.randint(2, 6)
    c = rng.randint(1, 7)
    pattern = rng.randint(0, 3)
    if pattern == 0:
        e_o, e_p, e_f = k * (x + c) + y, k * x + k * c + y, k * x + c + y
        o_str, p_str, f_str = f"expr: {k}*(x+{c})+y", f"expr: {k}*x+{k*c}+y", f"expr: {k}*x+{c}+y"
    elif pattern == 1:
        e_o, e_p, e_f = (x + c) * (x - c), x**2 - c**2, x**2 + c**2
        o_str, p_str, f_str = f"expr: (x+{c})*(x-{c})", f"expr: x**2-{c**2}", f"expr: x**2+{c**2}"
    elif pattern == 2:
        # Commutative reordering with subtraction term preserved vs broken order flip
        e_o, e_p, e_f = (x - y) + k * z, k * z + (x - y), k * z + (y - x)
        o_str, p_str, f_str = f"calc: (x-y)+{k}*z", f"calc: {k}*z+(x-y)", f"calc: {k}*z+(y-x)"
    else:
        e_o, e_p, e_f = (x + y) + k * x, y + (k + 1) * x, y + (k - 1) * x
        o_str, p_str, f_str = f"expr: (x+y)+{k}*x", f"expr: y+{k+1}*x", f"expr: y+{k-1}*x"
    assert sp.simplify(e_o - e_p) == 0
    assert sp.simplify(e_o - e_f) != 0
    return o_str.encode("ascii"), p_str.encode("ascii"), f_str.encode("ascii")


def generate_v8_train_triple(
    raw_window: bytes,
    rng: random.Random,
    *,
    allowed_families: Tuple[str, ...] = TRAIN_FAMILIES,
    seq_len: int = 128,
) -> Tuple[bytes, bytes, bytes, str]:
    """Generate enriched verified triples strictly within `TRAIN_FAMILIES` (`A-D`)."""
    fam = rng.choice(allowed_families)
    if fam == "corpus_uniform":
        res = _make_corpus_uniform(raw_window, rng, seq_len=seq_len)
        if res is not None:
            return res
        fam = rng.choice([f for f in allowed_families if f != "corpus_uniform"] or ["algebra_sympy"])

    if fam == "code_alpha_var":
        so, sp_, sf = _make_v8_code_alpha_var(rng)
    elif fam == "algebra_sympy":
        so, sp_, sf = _make_v8_algebra_sympy(rng)
    elif fam == "unit_scale":
        so, sp_, sf = _make_unit_scale(rng)
    else:
        return generate_v7_triple(raw_window, rng, allowed_families=allowed_families, seq_len=seq_len)

    return (
        _pack_prefix_suffix(raw_window, so, seq_len),
        _pack_prefix_suffix(raw_window, sp_, seq_len),
        _pack_prefix_suffix(raw_window, sf, seq_len),
        fam,
    )


def generate_i7_procedural_batch(
    batch_size: int,
    rng: random.Random,
    *,
    seq_len: int = 128,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    I7 Procedural Structural Pre-Training batch (pure symbolic syntax/symmetry/order/count):
      - Teaches Dyck bracket nesting sensitivity, symmetric vs directed-order invariants,
        associative regrouping vs meaning flip, with ZERO overlap with `HELDOUT_FAMILIES`.
      - Returns `(x [B, L] BYTE_PAD_ID=257 padded, equiv_labels, (ca, cb), nuance_t, struct_y)`.
    """
    n_pairs = max(2, batch_size // 4)
    seqs: List[bytes] = []
    eq_labels: List[int] = []
    contra_a: List[int] = []
    contra_b: List[int] = []
    nuance_t: List[float] = []
    struct_y: List[int] = []

    syms = ["alpha", "beta", "gamma", "delta", "sigma", "omega", "tau", "kappa"]
    lbl_id = 0
    for _ in range(n_pairs):
        u, v, w = rng.sample(syms, 3)
        kind = rng.randint(0, 2)
        if kind == 0:
            # Ordered sequential flow sensitivity: (u -> v) != (v -> u), while whitespace/delimiters preserve meaning
            o_b = f"seq: [ {u} -> {v} ] | {w}".encode("ascii")
            p_b = f"seq: [ {u} -> {v} ] ; {w}".encode("ascii")
            f_b = f"seq: [ {v} -> {u} ] | {w}".encode("ascii")
        elif kind == 1:
            # Dyck scope + associative bracketing invariance vs scope-moving error
            o_b = f"nest: ( ( {u} + {v} ) + {w} )".encode("ascii")
            p_b = f"nest: ( {u} + ( {v} + {w} ) )".encode("ascii")
            f_b = f"nest: ( ( {u} + {v} ) - {w} )".encode("ascii")
        else:
            # Commutative multiset reorder vs cardinality/symbol multiplicity change
            o_b = f"set: {u} , {v} , {w}".encode("ascii")
            p_b = f"set: {w} , {u} , {v}".encode("ascii")
            f_b = f"set: {u} , {v} , {v}".encode("ascii")

        idx_o = len(seqs)
        seqs.extend([o_b, p_b, o_b, f_b])
        eq_labels.extend([lbl_id, lbl_id, lbl_id + 1000, lbl_id + 2000])
        contra_a.append(idx_o + 2)
        contra_b.append(idx_o + 3)
        nuance_t.append(1.0)
        struct_y.extend([1, 1, 0, 2])
        lbl_id += 1

    out = torch.full((len(seqs), seq_len), 257, dtype=torch.long)
    for i, b in enumerate(seqs):
        n = min(len(b), seq_len)
        if n > 0:
            out[i, :n] = torch.tensor(list(b[:n]), dtype=torch.long)
    return (
        out.to(device),
        torch.tensor(eq_labels, dtype=torch.long, device=device),
        (torch.tensor(contra_a, dtype=torch.long, device=device), torch.tensor(contra_b, dtype=torch.long, device=device)),
        torch.tensor(nuance_t, dtype=torch.float32, device=device),
        torch.tensor(struct_y, dtype=torch.long, device=device),
    )
