"""Continual-learning streams S1-S7. Local data only (scikit-learn bundled sets, Python stdlib text and code,
synthetic generators), deterministic in the seed. Ground truth (regime ids, first-visit flags, outlier flags,
clean labels) lives in ``gt`` and is read only by the benchmark runner; ``algo_batches`` strips it.

A ``Stream`` is a list of materialised ``Segment`` s (one visit of one regime) plus per-regime test sets.
S1-S4 pass raw data through a frontend (so the algorithms never see the data type); S5-S7 are generated
directly on the unit sphere and ignore any frontend. The text and code corpora come from the installed
Python 3.12 standard library, so S1 is reproducible on a given interpreter version, not across versions.
"""

import os
import sysconfig
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from torch import Tensor

from .experts import MLPExpert
from .frontend import HashFrontend

S2_SCHEDULE = list(range(15)) + [3, 7, 0, 12] + list(range(15, 30)) + [5, 20, 1]
_ENGLISH_TOPICS = ("specialnames", "compound", "debugger", "formatstrings")
_CODE_MODULES = ("argparse.py", "statistics.py", "fractions.py", "textwrap.py")
_TEXT_WINDOW = 160
_JITTER = 0.1                      # std of the bootstrap jitter, in z-scored feature units


# ------------------------------------------------------------------------------------------- data model
@dataclass
class Segment:
    """One visit of one regime."""
    regime: int
    first_visit: bool
    x: Tensor
    y: Tensor
    src_ids: Tensor
    y_clean: Tensor
    is_outlier: Tensor
    ctx: Optional[Tensor] = None


@dataclass
class Regime:
    id: int
    name: str
    x_te: Tensor
    y_te: Tensor
    te_ids: Tensor
    parent: Optional[int] = None            # ground-truth lineage (analysis only)
    ctx: Optional[Tensor] = None
    meta: dict = field(default_factory=dict)


@dataclass
class Stream:
    name: str
    d: int
    n_classes: int
    regimes: Dict[int, Regime]
    segments: List[Segment]
    meta: dict = field(default_factory=dict)

    def _walk(self, batch_size: int, seed: int) -> Iterator[Tuple[int, Segment, Tensor]]:
        for k, seg in enumerate(self.segments):
            order = torch.from_numpy(np.random.default_rng([seed, 0x42415443, k]).permutation(len(seg.y)))
            for lo in range(0, len(order), batch_size):
                yield k, seg, order[lo:lo + batch_size]

    def batches(self, batch_size: int, seed: int = 0) -> Iterator[Tuple[Tensor, Tensor, dict]]:
        """(x, y, gt) batches; batches never cross a segment boundary; order within a segment depends on ``seed``."""
        for k, seg, idx in self._walk(batch_size, seed):
            gt = {"regime": seg.regime, "segment": k, "first_visit": seg.first_visit,
                  "is_outlier": seg.is_outlier[idx], "y_clean": seg.y_clean[idx]}
            yield seg.x[idx], seg.y[idx], gt

    def algo_batches(self, batch_size: int, seed: int = 0) -> Iterator[Tuple[Tensor, Tensor, Optional[Tensor]]]:
        """What an algorithm may see: (x, y, ctx). No regime ids, flags or clean labels."""
        for _, seg, idx in self._walk(batch_size, seed):
            yield seg.x[idx], seg.y[idx], (seg.ctx[idx] if seg.ctx is not None else None)

    def test_sets(self) -> Dict[int, Tuple[Tensor, Tensor]]:
        return {r: (reg.x_te, reg.y_te) for r, reg in self.regimes.items()}

    def test_ctx(self) -> Dict[int, Optional[Tensor]]:
        return {r: reg.ctx for r, reg in self.regimes.items()}


# ------------------------------------------------------------------------------------------------ helpers
def _rng(*tags: int) -> np.random.Generator:
    return np.random.default_rng([int(t) for t in tags])


def _tensor(x) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def _long(x) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.int64))


def _ix(ids) -> Tensor:
    return torch.from_numpy(np.asarray(ids, dtype=np.int64))


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _segment(regime: int, first: bool, x: Tensor, y: Tensor, ids, ctx: Tensor = None) -> Segment:
    return Segment(regime, first, x, y, _long(ids), y.clone(), torch.zeros(len(y), dtype=torch.bool), ctx)


def _take(pool: np.ndarray, visit: int, n: int) -> np.ndarray:
    """The ``visit``-th chunk of n pool positions, wrapping around when the pool is exhausted."""
    return pool[(visit * n + np.arange(n)) % len(pool)]


def _frontend(frontend) -> HashFrontend:
    return frontend if frontend is not None else HashFrontend()


# ----------------------------------------------------------------------------- digits-based (S2, S3, S4)
def _digits():
    d = load_digits()
    return d.data.astype(np.float64), d.target.astype(np.int64)


def _digit_split(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    perm = _rng(seed, 0x53504C).permutation(1797)
    return np.sort(perm[:1258]), np.sort(perm[1258:])


def _pixel_perms(seed: int, n: int) -> List[np.ndarray]:
    return [np.arange(64) if r == 0 else _rng(seed, 0x504552, r).permutation(64) for r in range(n)]


def s2_permuted(seed: int = 0, n_regimes: int = 30, frontend=None, n_train: int = 1000, n_test: int = 500,
                schedule: Sequence[int] = None) -> Stream:
    """Digits under ``n_regimes`` fixed pixel permutations (domain-incremental, shared 10 labels) with revisits."""
    fe = _frontend(frontend)
    X, Y = _digits()
    tr, te = _digit_split(seed)
    if schedule is None:
        schedule = S2_SCHEDULE if n_regimes == 30 else list(range(n_regimes))
    regimes, pools = {}, {}
    for r, perm in enumerate(_pixel_perms(seed, n_regimes)):
        z = fe(X[:, perm])
        te_r = te[_rng(seed, 0x5445, r).permutation(len(te))[:n_test]]
        regimes[r] = Regime(r, f"perm{r}", z[_ix(te_r)], _long(Y[te_r]), _long(te_r), meta={"type": "image"})
        pools[r] = (z, tr[_rng(seed, 0x5452, r).permutation(len(tr))])
    segments, visits = [], {}
    for r in schedule:
        v = visits.get(r, 0)
        visits[r] = v + 1
        ids = _take(pools[r][1], v, n_train)
        segments.append(_segment(r, v == 0, pools[r][0][_ix(ids)], _long(Y[ids]), ids))
    return Stream("s2_permuted", fe.d, 10, regimes, segments, {"schedule": list(schedule)})


def s3_drift(seed: int = 0, frontend=None, n_samples: int = 4000, n_swaps: int = 64, n_segments: int = 40,
             n_test: int = 500) -> Stream:
    """Digits whose pixel order drifts from the identity towards a scrambled one, one random swap at a time."""
    fe = _frontend(frontend)
    X, Y = _digits()
    tr, te = _digit_split(seed)
    rng = _rng(seed, 0x445246)
    perms, p = [np.arange(64)], np.arange(64)
    for _ in range(n_swaps):
        i, j = rng.choice(64, size=2, replace=False)
        p = p.copy()
        p[i], p[j] = p[j], p[i]
        perms.append(p)
    per_seg = n_samples // n_segments
    regimes, segments, seen = {}, [], set()
    for s in range(5):
        k = int(round(s / 4 * n_swaps))
        te_s = te[_rng(seed, 0x5454, s).permutation(len(te))[:n_test]]
        regimes[s] = Regime(s, f"phi{s / 4:.2f}", fe(X[te_s][:, perms[k]]), _long(Y[te_s]), _long(te_s),
                            meta={"type": "image", "phi": s / 4})
    for j in range(n_segments):
        phi = (j + 0.5) / n_segments
        stage = min(4, int(4 * phi + 0.5))
        ids = tr[_rng(seed, 0x5347, j).permutation(len(tr))[:per_seg]]
        segments.append(_segment(stage, stage not in seen, fe(X[ids][:, perms[int(round(phi * n_swaps))]]),
                                 _long(Y[ids]), ids))
        seen.add(stage)
    return Stream("s3_drift", fe.d, 10, regimes, segments, {"n_swaps": n_swaps})


def s4_outliers(seed: int = 0, outlier_rate: float = 0.01, label_noise: float = 0.1, n_regimes: int = 10,
                frontend=None, n_train: int = 1000, n_test: int = 500) -> Stream:
    """A prefix of S2 whose training data contain isolated outliers (replaced inputs) and flipped labels.

    Outliers are Gaussian-noise vectors or text from a foreign type; they get random labels and ``y_clean = -1``.
    Label noise flips a non-outlier's label to a uniformly random other label. Test sets stay clean."""
    fe = _frontend(frontend)
    base = s2_permuted(seed, n_regimes=n_regimes, frontend=fe, n_train=n_train, n_test=n_test,
                       schedule=list(range(n_regimes)))
    from pydoc_data.topics import topics
    foreign = topics["types"]
    rng = _rng(seed, 0x4F5554)
    next_id = 0
    segments = []
    for s in base.segments:
        n = len(s.y)
        x, y, clean = s.x.clone(), s.y.clone(), s.y.clone()
        ids = s.src_ids.clone()
        is_out = torch.from_numpy(rng.random(n) < outlier_rate)
        flip = torch.from_numpy((rng.random(n) < label_noise)) & ~is_out
        offset = torch.from_numpy(rng.integers(1, 10, size=n))
        y = torch.where(flip, (y + offset) % 10, y)
        idx = torch.nonzero(is_out).flatten()
        for pos, i in enumerate(idx.tolist()):
            if pos % 2 == 0:
                x[i] = fe(rng.standard_normal((1, 64)) * 6.0)[0]
            else:
                o = int(rng.integers(0, len(foreign) - _TEXT_WINDOW))
                x[i] = fe([foreign[o:o + _TEXT_WINDOW]])[0]
            y[i] = int(rng.integers(0, 10))
            clean[i] = -1
            ids[i] = -(1 + next_id)
            next_id += 1
        seg = Segment(s.regime, s.first_visit, x, y, ids, clean, is_out)
        segments.append(seg)
    return Stream("s4_outliers", fe.d, 10, base.regimes, segments,
                  {"outlier_rate": outlier_rate, "label_noise": label_noise})


# -------------------------------------------------------------------------------------- S1: mixed types
def _tabular(loader, seed, tag, fe, n_train, n_test, label_offset):
    d = loader()
    X, Y = d.data.astype(np.float64), d.target.astype(np.int64)
    rng = _rng(seed, tag)
    perm = rng.permutation(len(Y))
    cut = int(round(0.7 * len(Y)))
    tr, te = perm[:cut], perm[cut:]
    mu, sd = X[tr].mean(0), np.maximum(X[tr].std(0), 1e-9)
    Z = (X - mu) / sd

    def boot(rows, n):
        pick = rows[rng.integers(0, len(rows), size=n)]
        return pick, Z[pick] + _JITTER * rng.standard_normal((n, Z.shape[1]))

    tr_ids, x_tr = boot(tr, 2 * n_train)
    te_ids, x_te = boot(te, n_test)
    return (fe(x_tr), _long(Y[tr_ids] + label_offset), tr_ids,
            fe(x_te), _long(Y[te_ids] + label_offset), te_ids, list(range(label_offset, label_offset + len(set(Y)))))


def _sinusoids(n, rng, offset=0):
    f = np.array([2, 4, 8, 16])
    y = rng.integers(0, 4, size=n)
    t = np.arange(64)[None, :]
    x = (rng.uniform(0.7, 1.3, (n, 1)) * np.sin(2 * np.pi * f[y][:, None] * t / 64 + rng.uniform(0, 2 * np.pi, (n, 1)))
         + 0.3 * rng.standard_normal((n, 64)))
    return x, y + offset


def _ar2(n, rng, offset=0):
    coef = np.array([[1.2, -0.6], [0.5, 0.3], [-0.8, -0.1]])
    y = rng.integers(0, 3, size=n)
    x = np.zeros((n, 164))
    e = rng.standard_normal((n, 164))
    for t in range(2, 164):
        x[:, t] = coef[y, 0] * x[:, t - 1] + coef[y, 1] * x[:, t - 2] + e[:, t]
    return x[:, 100:], y + offset


def _series(gen, seed, tag, fe, n_train, n_test, offset, n_classes):
    rng = _rng(seed, tag)
    x_tr, y_tr = gen(2 * n_train, rng, offset)
    x_te, y_te = gen(n_test, rng, offset)
    n = len(y_tr)
    return (fe(x_tr), _long(y_tr), np.arange(n), fe(x_te), _long(y_te), n + np.arange(n_test),
            list(range(offset, offset + n_classes)))


def _corpus_windows(texts, seed, tag, fe, n_train, n_test, offset):
    rng = _rng(seed, tag)

    def draw(n, lo_frac, hi_frac):
        cls = rng.integers(0, len(texts), size=n)
        offs = np.array([rng.integers(int(lo_frac * len(texts[c])), int(hi_frac * len(texts[c])) - _TEXT_WINDOW) for c in cls])
        return [texts[c][o:o + _TEXT_WINDOW] for c, o in zip(cls, offs)], cls, cls * 10_000_000 + offs

    w_tr, c_tr, i_tr = draw(2 * n_train, 0.0, 0.75)            # train windows from the first 75 % of each text,
    w_te, c_te, i_te = draw(n_test, 0.75, 1.0)                 # test windows from the last 25 %: disjoint regions
    return (fe(w_tr), _long(c_tr + offset), i_tr, fe(w_te), _long(c_te + offset), i_te,
            list(range(offset, offset + len(texts))))


def _english():
    from pydoc_data.topics import topics
    return [topics[k] for k in _ENGLISH_TOPICS]


def _code():
    root = sysconfig.get_paths()["stdlib"]
    out = []
    for name in _CODE_MODULES:
        with open(os.path.join(root, name), encoding="utf-8", errors="replace") as f:
            out.append(f.read())
    return out


def s1_types(seed: int = 0, frontend=None, n_train: int = 600, n_test: int = 200, revisit: Sequence[int] = (3, 0, 5),
             revisit_size: int = 200) -> Stream:
    """Seven regimes of five data types (image, tabular x2, series x2, text, code) with disjoint label blocks
    (class-incremental, 30 labels), followed by revisits of some earlier types. Small sets are bootstrapped with jitter."""
    fe = _frontend(frontend)
    builders = [
        ("digits", "image", None),
        ("wine", "tabular", lambda off: _tabular(load_wine, seed, 0x57494E, fe, n_train, n_test, off)),
        ("breast_cancer", "tabular", lambda off: _tabular(load_breast_cancer, seed, 0x4243, fe, n_train, n_test, off)),
        ("sinusoid", "series", lambda off: _series(_sinusoids, seed, 0x534E, fe, n_train, n_test, off, 4)),
        ("ar2", "series", lambda off: _series(_ar2, seed, 0x4152, fe, n_train, n_test, off, 3)),
        ("pydoc_text", "text", lambda off: _corpus_windows(_english(), seed, 0x454E, fe, n_train, n_test, off)),
        ("stdlib_code", "code", lambda off: _corpus_windows(_code(), seed, 0x5059, fe, n_train, n_test, off)),
    ]
    regimes, pools, offset = {}, {}, 0
    for r, (name, kind, make) in enumerate(builders):
        if make is None:                                     # digits: real images, real train/test split
            digits = load_digits()
            X, Y = digits.images.astype(np.float64), digits.target.astype(np.int64)
            tr, te = _digit_split(seed)
            te = te[_rng(seed, 0x4454).permutation(len(te))[:n_test]]
            x_all = fe(X)
            parts = (x_all[_ix(tr)], _long(Y[tr] + offset), tr, x_all[_ix(te)], _long(Y[te] + offset), te,
                     list(range(offset, offset + 10)))
        else:
            parts = make(offset)
        x_tr, y_tr, ids_tr, x_te, y_te, ids_te, labels = parts
        regimes[r] = Regime(r, name, x_te, y_te, _long(ids_te), meta={"type": kind, "labels": labels})
        pools[r] = (x_tr, y_tr, np.asarray(ids_tr), _rng(seed, 0x504F, r).permutation(len(y_tr)))
        offset += len(labels)
    schedule = list(range(len(builders))) + list(revisit)
    segments, visits = [], {}
    for r in schedule:
        v = visits.get(r, 0)
        visits[r] = v + 1
        n = n_train if v == 0 else revisit_size
        start = 0 if v == 0 else n_train + (v - 1) * revisit_size
        x_tr, y_tr, ids_tr, order = pools[r]
        pick = order[(start + np.arange(n)) % len(order)]
        segments.append(_segment(r, v == 0, x_tr[_ix(pick)], y_tr[_ix(pick)], ids_tr[pick]))
    return Stream("s1_types", fe.d, offset, regimes, segments, {"schedule": schedule})


# ------------------------------------------------------------------------------ latent-space (S5, S6, S7)
def _balanced_teacher(width: int, depth: int, d: int, n_classes: int, seed: int) -> MLPExpert:
    """Random teacher whose head bias is tuned until its argmax labels are roughly class-balanced."""
    teacher = MLPExpert(d_in=d, hidden=width, depth=depth, n_classes=n_classes, seed=seed)
    teacher.mark_seen(torch.arange(n_classes))
    ref = _tensor(_unit(_rng(seed, 0x424C).standard_normal((20000, d))))
    with torch.no_grad():
        for _ in range(80):
            freq = torch.bincount(teacher(ref).argmax(1), minlength=n_classes).float() / len(ref)
            teacher.head.bias -= 2.0 * (freq - 1.0 / n_classes)
    return teacher


def s5_ladder(seed: int = 0, width: int = 32, depth: int = 1, d: int = 128, n_train: int = 20000,
              n_test: int = 1000, epochs: int = 1, frontend=None) -> Stream:
    """One regime whose labels are the argmax of a random ``width`` x ``depth`` teacher on uniform unit vectors.

    Deviation from the spec's 2,000 training samples, measured: at 2,000 (d = 128) every teacher is data-limited
    (best student accuracy 0.36-0.57 whatever the student width). At 20,000 the ladder is informative: for the
    w32 d2 teacher students of width 8 / 32 / 128 reach 0.53 / 0.56 / 0.66, for w8 d1 they saturate (0.78 / 0.82 / 0.82).
    ``epochs`` repeats the training segment (revisits of the same data) for runs that need several passes."""
    teacher = _balanced_teacher(width, depth, d, 8, 7919 * seed + 31 * width + depth)
    rng = _rng(seed, 0x4535, width, depth)
    x_tr, x_te = _tensor(_unit(rng.standard_normal((n_train, d)))), _tensor(_unit(rng.standard_normal((n_test, d))))
    with torch.no_grad():
        y_tr, y_te = teacher(x_tr).argmax(1), teacher(x_te).argmax(1)
    regime = Regime(0, f"teacher_w{width}_d{depth}", x_te, y_te, n_train + torch.arange(n_test),
                    meta={"type": "latent", "teacher": teacher, "width": width, "depth": depth})
    segments = [_segment(0, e == 0, x_tr, y_tr, np.arange(n_train)) for e in range(epochs)]
    return Stream(f"s5_w{width}_d{depth}", d, 8, {0: regime}, segments, {"width": width, "depth": depth})


def s5_grid(seed: int = 0, **kw) -> Dict[Tuple[int, int], Stream]:
    return {(w, dep): s5_ladder(seed, width=w, depth=dep, **kw) for w in (8, 32, 128) for dep in (1, 2, 3)}


def _haar(d: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def _small_rotation(means: np.ndarray, target_cos: float, rng: np.random.Generator) -> np.ndarray:
    """Orthogonal Q with mean_c <Q mu_c, mu_c> = target_cos, found by bisection on the scale of a random skew matrix."""
    d = means.shape[1]
    g = rng.standard_normal((d, d))
    a = (g - g.T) / (2.0 * np.sqrt(d))

    def mean_cos(eps):
        return float(np.mean(np.sum(means * expm_multiply(eps * a, means.T).T, axis=1)))

    lo, hi = 0.0, 0.05
    while mean_cos(hi) > target_cos:
        lo, hi = hi, 2.0 * hi
        if hi > 64.0:
            raise RuntimeError("could not reach the requested rotation")
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mean_cos(mid) > target_cos else (lo, mid)
    return expm(0.5 * (lo + hi) * a)


def _mixture(means: np.ndarray, n: int, sigma: float, rng: np.random.Generator):
    y = rng.integers(0, means.shape[0], size=n)
    x = means[y] + sigma * rng.standard_normal((n, means.shape[1])) / np.sqrt(means.shape[1])
    return x, y


def s6_long(seed: int = 0, n_regimes: int = 60, related_frac: float = 0.3, d: int = 128, n_train: int = 500,
            n_test: int = 200, sigma: float = 0.6, rotation_cos: float = 0.9, revisit_frac: float = 0.1,
            frontend=None) -> Stream:
    """Many regimes of one 8-class Gaussian-mixture family, each in its own random orientation. A ``related_frac``
    share are small rotations (class-mean cosine ``rotation_cos``) of an earlier regime; the rest are unrelated."""
    rng = _rng(seed, 0x4536)
    means = np.linalg.qr(rng.standard_normal((d, 8)))[0].T                     # 8 orthonormal class directions
    n_related = int(round(related_frac * (n_regimes - 1)))
    related = set(rng.choice(np.arange(1, n_regimes), size=n_related, replace=False).tolist()) if n_related else set()
    rots, regimes, pools = {}, {}, {}
    for r in range(n_regimes):
        parent = int(rng.integers(0, r)) if r in related else None
        rots[r] = rots[parent] @ _small_rotation(means, rotation_cos, rng) if parent is not None else _haar(d, rng)
        r_rng = _rng(seed, 0x4533, r)
        x_te, y_te = _mixture(means, n_test, sigma, r_rng)
        x_tr, y_tr = _mixture(means, 2 * n_train, sigma, r_rng)
        regimes[r] = Regime(r, f"rot{r}", _tensor(_unit(x_te @ rots[r].T)), _long(y_te),
                            2 * n_train + torch.arange(n_test), parent=parent, meta={"type": "latent"})
        pools[r] = (_tensor(_unit(x_tr @ rots[r].T)), _long(y_tr))
    schedule = list(range(n_regimes))
    n_rev = int(round(revisit_frac * n_regimes))
    for _ in range(n_rev):
        at = int(rng.integers(min(10, n_regimes // 2), len(schedule) + 1))
        schedule.insert(at, int(rng.integers(0, len(set(schedule[:at])))))       # regimes are introduced in order 0, 1, ...
    segments, visits = [], {}
    for r in schedule:
        v = visits.get(r, 0)
        visits[r] = v + 1
        x, y = pools[r]
        ids = _take(np.arange(len(y)), v, n_train)
        segments.append(_segment(r, v == 0, x[ids], y[ids], ids))
    return Stream("s6_long", d, 8, regimes, segments, {"schedule": schedule, "rotation_cos": rotation_cos})


def s7_mapping(seed: int = 0, with_ctx: bool = False, n_regimes: int = 4, d: int = 128, n_train: int = 600,
               n_test: int = 200, ctx_dim: int = 16, sigma: float = 0.6, frontend=None) -> Stream:
    """The same inputs under ``n_regimes`` different label permutations, optionally with a context code per regime."""
    rng = _rng(seed, 0x4537)
    means = np.linalg.qr(rng.standard_normal((d, 8)))[0].T
    x_tr, y_tr = _mixture(means, n_train, sigma, rng)
    x_te, y_te = _mixture(means, n_test, sigma, rng)
    x_tr, x_te = _tensor(_unit(x_tr)), _tensor(_unit(x_te))
    perms = [np.arange(8)]
    while len(perms) < n_regimes:
        p = rng.permutation(8)
        if all(not np.array_equal(p, q) for q in perms):
            perms.append(p)
    regimes, segments = {}, []
    for r, p in enumerate(perms):
        ctx = _tensor(_unit(rng.standard_normal((1, ctx_dim)))[0]) if with_ctx else None
        regimes[r] = Regime(r, f"map{r}", x_te, _long(p[y_te]), n_train + torch.arange(n_test), ctx=ctx,
                            meta={"type": "latent", "perm": p.tolist()})
    schedule = list(range(n_regimes)) + [1, 0]
    seen = set()
    for r in schedule:
        ctx = regimes[r].ctx
        segments.append(_segment(r, r not in seen, x_tr, _long(perms[r][y_tr]), np.arange(n_train),
                                 ctx=ctx.expand(n_train, -1).clone() if ctx is not None else None))
        seen.add(r)
    return Stream("s7_mapping", d, 8, regimes, segments, {"schedule": schedule, "with_ctx": with_ctx})
