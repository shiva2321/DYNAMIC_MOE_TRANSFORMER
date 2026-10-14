"""The arithmetic benchmark's data format (growmoe/arith.py): the ground truth is right, the encoding loses nothing, the splits are disjoint, the scoring is exact."""

import numpy as np
import pytest

from growmoe import arith as A


def _py_answer(op: str, a: int, b: int) -> int:
    """An independent, deliberately plain implementation (python ints, no numpy) to check the vectorised ground truth against."""
    return {"add": a + b, "sub": a - b, "mul": a * b, "mod": a % b if b else 0, "div": a // b if b else 0, "max": max(a, b)}[op]


def _problems(n=400, seed=0, split="train", lens=(1, 2, 3, 4, 5)):
    rng = np.random.default_rng(seed)
    return A.ProblemSet(**A.sample_problems(rng, n, tuple(range(A.N_OPS)), lens, lens, split))


def _read_number(block: np.ndarray) -> int:
    """Decode one operand block (5 slots x 11 symbols) back to an integer; a one-hot per slot, PAD = 10."""
    digits = [int(np.argmax(block[s * A.N_SYM:(s + 1) * A.N_SYM])) for s in range(A.N_SLOTS)]
    return int("".join(str(d) for d in digits if d != A.PAD))


def test_the_ground_truth_equals_plain_python_arithmetic():
    ps = _problems(500)
    for i in range(len(ps)):
        assert int(ps.ans[i]) == _py_answer(A.OPS[int(ps.op[i])], int(ps.a[i]), int(ps.b[i]))


def test_the_encoding_of_the_operands_and_the_operator_is_lossless():
    ps = _problems(300, seed=1)
    for i in range(len(ps)):
        row = ps.xprob[i]
        assert _read_number(row[:A.N_SLOTS * A.N_SYM]) == int(ps.a[i])
        assert _read_number(row[A.N_SLOTS * A.N_SYM:2 * A.N_SLOTS * A.N_SYM]) == int(ps.b[i])
        assert int(np.argmax(row[2 * A.N_SLOTS * A.N_SYM:])) == int(ps.op[i])


def test_every_example_is_a_twelve_hot_vector_of_the_model_width_and_the_slot_is_marked():
    ps = _problems(50, seed=2)
    x = ps.x()
    assert x.shape == (ps.n_samples, A.D_MODEL) and np.all(x.sum(axis=1) == 12) and np.all(x[:, A.D_IN:] == 0)
    slot = np.argmax(x[:, A.D_PROB:A.D_IN], axis=1)
    assert np.array_equal(slot, ps.pos)


def test_the_targets_rebuild_the_answer_and_every_answer_fits_its_operators_format():
    ps = _problems(600, seed=3)
    for i in range(len(ps)):
        t = ps.target[i]
        digits = [int(t[p]) for p in range(10) if t[p] != A.PAD]
        value = int("".join(str(d) for d in reversed(digits)))
        sign = -1 if t[A.SIGN_POS] == A.NEG else 1
        assert sign * value == int(ps.ans[i])
        valid = A.VALID[int(ps.op[i])]
        assert all(valid[p] for p in range(10) if t[p] != A.PAD), "an answer digit would fall outside the operator's slots"
        assert t[A.SIGN_POS] == A.PAD or valid[A.SIGN_POS], "only subtraction has a sign slot"


def test_the_extreme_answers_still_fit():
    big = A.draw_numbers(np.random.default_rng(0), np.array([5]))
    assert big.shape == (1,)
    cases = [("add", 99999, 99999, 6), ("mul", 99999, 99999, 10), ("sub", 0, 99999, 5), ("div", 99999, 1, 5), ("mod", 99998, 99999, 5), ("max", 99999, 0, 5)]
    for name, a, b, digits in cases:
        o = A.OPS.index(name)
        ps = A.ProblemSet(np.array([o]), np.array([a]), np.array([b]), np.array([5]), np.array([5 if b > 9 else len(str(b))]))
        assert int(ps.ans[0]) == _py_answer(name, a, b)
        assert int(A.n_digits(ps.ans)[0]) == digits
        assert all(A.VALID[o][p] for p in range(10) if ps.target[0][p] != A.PAD)
    neg = A.ProblemSet(np.array([A.OPS.index("sub")]), np.array([0]), np.array([99999]), np.array([1]), np.array([5]))
    assert neg.target[0][A.SIGN_POS] == A.NEG


def test_numbers_have_exactly_the_requested_number_of_digits_and_divisors_are_never_zero():
    ps = _problems(800, seed=4)
    assert np.array_equal(A.n_digits(ps.a), ps.la) and np.array_equal(A.n_digits(ps.b), ps.lb)
    dm = np.isin(ps.op, [A.OPS.index("mod"), A.OPS.index("div")])
    assert (ps.b[dm] >= 1).all()


def test_train_and_test_problems_are_disjoint_and_the_test_split_is_the_heldout_tenth():
    tr, te = _problems(2000, seed=5, split="train"), _problems(2000, seed=6, split="test")
    assert not A.heldout(tr.op, tr.a, tr.b).any() and A.heldout(te.op, te.a, te.b).all()
    key = lambda p: set(zip(p.op.tolist(), p.a.tolist(), p.b.tolist()))          # noqa: E731
    assert key(tr).isdisjoint(key(te))
    rng = np.random.default_rng(7)
    la = rng.integers(1, 6, 200000)
    a, b, o = A.draw_numbers(rng, la), A.draw_numbers(rng, la), rng.integers(0, A.N_OPS, 200000)
    share = float(A.heldout(o, a, b).mean())
    assert abs(share - A.HELDOUT_PCT / 100) < 0.01


def test_even_one_digit_problems_have_unseen_test_cases():
    rng = np.random.default_rng(8)
    tr = A.ProblemSet(**A.sample_problems(rng, 3000, (0, 1, 2), (1,), (1,), "train"))
    te = A.ProblemSet(**A.sample_problems(rng, 100, (0, 1, 2), (1,), (1,), "test"))
    seen = set(zip(tr.op.tolist(), tr.a.tolist(), tr.b.tolist()))
    assert all((o, x, y) not in seen for o, x, y in zip(te.op.tolist(), te.a.tolist(), te.b.tolist()))


def test_scoring_is_exact_match_over_all_slots_and_digit_accuracy_over_slots():
    ps = _problems(60, seed=9)
    pred = ps.y.copy()
    assert ps.exact(pred).all() and np.allclose(ps.digit_acc(pred), 1.0)
    i = 7
    first = int(np.flatnonzero(ps.pi == i)[0])
    pred[first] = (pred[first] + 1) % A.N_CLASS
    ex, da = ps.exact(pred), ps.digit_acc(pred)
    assert not ex[i] and ex.sum() == len(ps) - 1
    assert abs(da[i] - (1 - 1 / int((ps.pi == i).sum()))) < 1e-9


def test_the_sequence_view_carries_the_same_information():
    ps = _problems(200, seed=10)
    inp, out = A.sequence_tokens(ps)
    assert inp.shape == (len(ps), 13) and out.shape == (len(ps), 11)
    assert np.array_equal(inp[:, 0] - 12, ps.op) and (inp[:, 6] == 18).all() and (inp[:, 12] == 18).all()
    for i in range(len(ps)):
        a = int("".join(str(t) for t in inp[i, 1:6] if t != A.PAD))
        b = int("".join(str(t) for t in inp[i, 7:12] if t != A.PAD))
        assert (a, b) == (int(ps.a[i]), int(ps.b[i]))
        digits = [int(t) for t in out[i, 1:] if t != A.PAD]
        value = int("".join(str(d) for d in digits))
        assert (-1 if out[i, 0] == A.NEG else 1) * value == int(ps.ans[i])
    v = A.valid_output_tokens(ps)
    assert v.shape == (len(ps), 11) and v.sum() == ps.n_samples


@pytest.mark.parametrize("lens", [(1, 2, 3), (1, 2, 4, 5)])
def test_a_length_restricted_training_set_only_contains_those_lengths(lens):
    ps = _problems(500, seed=11, lens=lens)
    assert set(np.unique(ps.la)) <= set(lens) and set(np.unique(ps.lb)) <= set(lens)
