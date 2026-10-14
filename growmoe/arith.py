"""Arithmetic on 1-5 digit numbers as a LEARNING problem: the data format only, no arithmetic knowledge of any kind.

Nothing here knows how to add, carry, borrow, multiply or divide. A problem is two numbers (1 to 5 digits each) and an operator; the answer is a string of
digits (and a sign). The learners are shown examples and must find whatever rule they can. The only things fixed by the format are bookkeeping:

* a number is written as 5 right-aligned digit slots, unused leading slots hold the symbol PAD (0 is a real digit);
* the answer is written as digit slots counted from the right (position 0 = units, 1 = tens, ...) plus one sign slot (only subtraction can be negative);
  a slot the answer does not use holds PAD; how many slots an operator can ever need is a property of the operator's FORMAT (add 6, sub 5 + sign, mul 10, mod/div/max 5);
* every (problem, output slot) is one training example: input = [digits of A | digits of B | operator | slot] as one-hot blocks (127 numbers, padded to 128),
  label = the symbol in that slot (0-9, PAD, NEG: 12 classes). A model answers a problem by answering all its slots; the answer is right only if EVERY slot is right (exact match).

Held-out problems are held out by a hash of (operator, a, b): about HELDOUT_PCT % of all problems of every length can never be trained on, so "unseen" is exact even where
the universe is tiny (1-digit numbers have only 100 pairs per operator).
"""

from dataclasses import dataclass

import numpy as np

OPS = ("add", "sub", "mul", "mod", "div", "max")
SYMBOL = {"add": "+", "sub": "-", "mul": "*", "mod": "%", "div": "//", "max": "max"}
N_OPS, N_SLOTS, N_SYM = len(OPS), 5, 11
PAD, NEG, N_CLASS = 10, 11, 12
N_POS, SIGN_POS = 11, 10
D_PROB = 2 * N_SLOTS * N_SYM + N_OPS                      # 116: the part of an input that describes the problem
D_IN = D_PROB + N_POS                                     # 127
D_MODEL = 128                                             # inputs are zero-padded to 128 numbers (the pool's frontend size)
WIDTH = {"add": 6, "sub": 5, "mul": 10, "mod": 5, "div": 5, "max": 5}   # digit slots an operator's answer can ever need (sub also has the sign slot)
HELDOUT_PCT = 10

VALID = np.zeros((N_OPS, N_POS), dtype=bool)               # which output slots exist for which operator
for _i, _op in enumerate(OPS):
    VALID[_i, :WIDTH[_op]] = True
VALID[OPS.index("sub"), SIGN_POS] = True


# ------------------------------------------------------------------------------------------------ problems
def draw_numbers(rng: np.random.Generator, lengths: np.ndarray, nonzero: bool = False) -> np.ndarray:
    """Numbers with exactly ``lengths`` digits (1-digit numbers are 0-9, or 1-9 when ``nonzero``)."""
    lengths = np.asarray(lengths)
    lo = np.where(lengths == 1, 1 if nonzero else 0, 10 ** (lengths - 1))
    hi = 10 ** lengths
    return rng.integers(lo, hi)


def answers(op: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The ground truth (used ONLY to make labels and to score; no learner sees this function)."""
    bb = np.where(b == 0, 1, b)                            # b is never 0 for mod / div; this only keeps the unused branches finite
    return np.select([op == 0, op == 1, op == 2, op == 3, op == 4, op == 5], [a + b, a - b, a * b, a % bb, a // bb, np.maximum(a, b)])


def _mix(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def heldout(op: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """True for the problems that are reserved for testing (a fixed hash of operator, a, b)."""
    key = op.astype(np.uint64) * np.uint64(10 ** 10) + a.astype(np.uint64) * np.uint64(100000) + b.astype(np.uint64)
    return (_mix(key) % np.uint64(100)) < np.uint64(HELDOUT_PCT)


def sample_problems(rng: np.random.Generator, n: int, ops: tuple, lens_a: tuple, lens_b: tuple, split: str) -> dict:
    """``n`` problems per operator in ``ops`` with operand lengths drawn uniformly from ``lens_a`` / ``lens_b``.
    ``split`` = "train" (never a held-out problem) or "test" (only held-out problems)."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    cols = {k: [] for k in ("op", "a", "b", "la", "lb")}
    for oi in ops:
        got, need = 0, n
        while got < n:
            m = int(need * (1.25 if split == "train" else 12)) + 64
            la = rng.choice(np.asarray(lens_a), size=m)
            lb = rng.choice(np.asarray(lens_b), size=m)
            a = draw_numbers(rng, la)
            b = draw_numbers(rng, lb, nonzero=OPS[oi] in ("mod", "div"))
            o = np.full(m, oi)
            keep = heldout(o, a, b) if split == "test" else ~heldout(o, a, b)
            take = np.flatnonzero(keep)[:need]
            for k, v in (("op", o), ("a", a), ("b", b), ("la", la), ("lb", lb)):
                cols[k].append(v[take])
            got += len(take)
            need = n - got
    return {k: np.concatenate(v) for k, v in cols.items()}


# ------------------------------------------------------------------------------------------------ encoding
def n_digits(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    out = np.ones(x.shape, dtype=np.int64)
    for k in range(1, 12):
        out += (x >= 10 ** k)
    return out


def encode_problems(op, a, b, la, lb) -> np.ndarray:
    """(n, D_PROB) float32 one-hot description of the problems: digits of A, digits of B (right-aligned, PAD in unused slots), operator."""
    n = len(op)
    x = np.zeros((n, D_PROB), dtype=np.float32)
    rows = np.arange(n)
    for side, (num, ln) in enumerate(((a, la), (b, lb))):
        for s in range(N_SLOTS):
            k = N_SLOTS - 1 - s                                  # digit index counted from the right
            sym = np.where(k < ln, (num // 10 ** k) % 10, PAD)
            x[rows, side * N_SLOTS * N_SYM + s * N_SYM + sym] = 1.0
    x[rows, 2 * N_SLOTS * N_SYM + op] = 1.0
    return x


def target_symbols(ans: np.ndarray) -> np.ndarray:
    """(n, N_POS) the symbol every output slot should hold: the digit counted from the right, PAD where the answer has fewer digits, NEG/PAD in the sign slot."""
    absn = np.abs(ans)
    nd = n_digits(absn)
    t = np.full((len(ans), N_POS), PAD, dtype=np.int64)
    for p in range(10):
        t[:, p] = np.where(p < nd, (absn // 10 ** p) % 10, PAD)
    t[:, SIGN_POS] = np.where(ans < 0, NEG, PAD)
    return t


@dataclass
class ProblemSet:
    """A set of problems and, for each of them, one example per output slot the operator has."""
    op: np.ndarray
    a: np.ndarray
    b: np.ndarray
    la: np.ndarray
    lb: np.ndarray

    def __post_init__(self) -> None:
        self.ans = answers(self.op, self.a, self.b)
        self.target = target_symbols(self.ans)                   # (n, N_POS)
        self.xprob = encode_problems(self.op, self.a, self.b, self.la, self.lb)
        pi, pos = np.nonzero(VALID[self.op])                     # every (problem, slot) the operator has, problem-major
        self.pi, self.pos = pi, pos
        self.y = self.target[pi, pos]

    def __len__(self) -> int:
        return len(self.op)

    @property
    def n_samples(self) -> int:
        return len(self.pi)

    def x(self, sample_idx: np.ndarray = None) -> np.ndarray:
        """(m, 128) float32 inputs of the chosen examples (all if None): problem block, slot one-hot, one zero pad."""
        idx = np.arange(self.n_samples) if sample_idx is None else np.asarray(sample_idx)
        out = np.zeros((len(idx), D_MODEL), dtype=np.float32)
        out[:, :D_PROB] = self.xprob[self.pi[idx]]
        out[np.arange(len(idx)), D_PROB + self.pos[idx]] = 1.0
        return out

    def subset(self, mask: np.ndarray) -> "ProblemSet":
        return ProblemSet(self.op[mask], self.a[mask], self.b[mask], self.la[mask], self.lb[mask])

    # ---- scoring: a prediction is one class per example
    def exact(self, pred: np.ndarray) -> np.ndarray:
        """Per problem: True iff every output slot of the problem was predicted right."""
        wrong = np.bincount(self.pi, weights=(pred != self.y).astype(np.float64), minlength=len(self))
        return wrong == 0

    def digit_acc(self, pred: np.ndarray) -> np.ndarray:
        """Per problem: share of its output slots predicted right."""
        right = np.bincount(self.pi, weights=(pred == self.y).astype(np.float64), minlength=len(self))
        return right / np.bincount(self.pi, minlength=len(self))


def sequence_tokens(ps: ProblemSet):
    """Token view for a sequence model (same information, same targets): input (n, 13) = [operator token, 5 slots of A, SEP, 5 slots of B, SEP];
    output (n, 11) = [sign, digit 9, ..., digit 0] (most significant first). Tokens: 0-9 digits, 10 PAD, 11 NEG, 12-17 operators, 18 SEP."""
    n = len(ps)
    inp = np.zeros((n, 13), dtype=np.int64)
    inp[:, 0] = 12 + ps.op
    for side, (num, ln) in enumerate(((ps.a, ps.la), (ps.b, ps.lb))):
        for s in range(N_SLOTS):
            k = N_SLOTS - 1 - s
            inp[:, 1 + side * 6 + s] = np.where(k < ln, (num // 10 ** k) % 10, PAD)
        inp[:, 6 + side * 6] = 18
    out = np.concatenate([ps.target[:, SIGN_POS:SIGN_POS + 1], ps.target[:, :10][:, ::-1]], axis=1)
    return inp, out


def valid_output_tokens(ps: ProblemSet) -> np.ndarray:
    """(n, 11) bool: which of the 11 output tokens of ``sequence_tokens`` belong to the operator's format (sign slot, then digit slots 9..0)."""
    v = VALID[ps.op]
    return np.concatenate([v[:, SIGN_POS:SIGN_POS + 1], v[:, :10][:, ::-1]], axis=1)
