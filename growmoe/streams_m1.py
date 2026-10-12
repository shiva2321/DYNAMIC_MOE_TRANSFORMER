"""M1 stream: function regimes - a continual stream where the answer has to be computed, not looked up.

Regime ``r`` lives in its own region of the unit sphere (inputs ``unit(centre_scale * centre_r + noise)``). Its label is the argmax of
a hidden random teacher network (``teacher_width`` x ``teacher_depth``, class-balanced) applied to the unit-length within-region
noise. Two inputs that are close on the sphere therefore do NOT share labels (a kNN lookup stays near 22 %, chance 12.5 %), yet each
regime is learnable by a small network (45-60 % at 300-1000 samples; research/moe_diligence_2026-10/spike_lookup_fails_out.txt).

Schedule: every regime once, then revisits of regime 0 and regime ``n_regimes // 2`` on fresh samples. Generated directly on the sphere,
deterministic in the seed, no downloads.
"""

import numpy as np

from .streams import Regime, Stream, _balanced_teacher, _long, _rng, _segment, _tensor, _unit


def function_regimes(seed: int = 0, n_regimes: int = 15, d: int = 128, n_classes: int = 8, n_train: int = 800, n_test: int = 400,
                     centre_scale: float = 8.0, teacher_width: int = 8, teacher_depth: int = 1,
                     overlap: float = 0.0, mixed_tail: int = 0, minor_share: float = 0.3) -> Stream:
    """``overlap`` is the target cosine between regime centres (0 = the pilot stream, bit for bit). ``mixed_tail`` appends that many segments
    in which a regime the pool already knows arrives mixed with a second known one (``1 - minor_share`` / ``minor_share``), built from the
    unused second halves of regimes that no revisit uses; ``meta["mixed_pairs"]`` and ``meta["minority_regimes"]`` name them."""
    if n_regimes < 2:
        raise ValueError("a function-regime stream needs at least 2 regimes")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap (the target cosine between regime centres) must lie in [0, 1)")
    if not 0.0 < minor_share <= 0.5:
        raise ValueError("minor_share must lie in (0, 0.5]")
    free = [r for r in range(n_regimes) if r not in (0, n_regimes // 2)]                 # regimes whose second half no revisit uses
    pairs = [(free[i], free[i + 1]) for i in range(0, len(free) - 1, 2)]
    if not 0 <= mixed_tail <= len(pairs):
        raise ValueError(f"mixed_tail must lie in [0, {len(pairs)}] for {n_regimes} regimes")
    rng = _rng(seed, 0xF5)
    g = rng.standard_normal((n_regimes, d))
    if overlap == 0.0:
        centres = _unit(g)
    else:                                                                                 # the shared direction has its own generator: the main stream is untouched
        shared = _rng(seed, 0xF6).standard_normal((1, d))
        centres = _unit(np.sqrt(1.0 - overlap) * _unit(g) + np.sqrt(overlap) * _unit(shared))
    regimes, pools = {}, {}
    for r in range(n_regimes):
        teacher = _balanced_teacher(teacher_width, teacher_depth, d, n_classes, seed=seed * 1000 + r)

        def draw(n: int):
            z = rng.standard_normal((n, d))
            y = teacher(_tensor(_unit(z))).argmax(1).detach().numpy()
            return _unit(centre_scale * centres[r] + z), y

        x_tr, y_tr = draw(2 * n_train)                                                  # two chunks: first visit and one revisit
        x_te, y_te = draw(n_test)
        regimes[r] = Regime(r, f"fn{r}", _tensor(x_te), _long(y_te), _long(np.arange(n_test)), meta={"type": "function"})
        pools[r] = (x_tr, y_tr)
    schedule = list(range(n_regimes)) + [0, n_regimes // 2]
    segments, visits = [], {}
    for r in schedule:
        v = visits.get(r, 0)
        visits[r] = v + 1
        sl = slice(v * n_train, (v + 1) * n_train)
        segments.append(_segment(r, v == 0, _tensor(pools[r][0][sl]), _long(pools[r][1][sl]), np.arange(sl.start, sl.stop)))
    n_minor = int(round(minor_share * n_train))
    n_major = n_train - n_minor
    for a, b in pairs[:mixed_tail]:
        xa, ya = pools[a][0][n_train:n_train + n_major], pools[a][1][n_train:n_train + n_major]
        xb, yb = pools[b][0][n_train:n_train + n_minor], pools[b][1][n_train:n_train + n_minor]
        segments.append(_segment(a, False, _tensor(np.concatenate([xa, xb])), _long(np.concatenate([ya, yb])), np.arange(n_train, 2 * n_train)))
        schedule.append(a)
    meta = {"schedule": schedule, "centres": centres, "overlap": overlap, "mixed_pairs": pairs[:mixed_tail],
            "minority_regimes": [b for _, b in pairs[:mixed_tail]]}
    return Stream("function_regimes", d, n_classes, regimes, segments, meta)
