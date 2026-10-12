"""THROWAWAY spike on DEV seeds (100-102): does the plain batch vote hurt REGIMES THE POOL ALREADY KNOWS when they later arrive mixed?

spike_batch_guard's mixed streams are a poor test: the baseline already merges the paired regimes there. Here every regime is first learned in a pure block (as in
overlap_stream), then a tail of 70/30 mixed segments of known regimes (1+2, 3+4, 6+7, 8+9) is appended from their unused second halves. A vote that pushes the 30 % minority
into the majority's expert would damage the minority regimes; the per-sample baseline should not. Variants as in spike_vote_variants.py.

    python research/growmoe_results/rv2/spike_mixphase.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "m2a"))
sys.path.insert(0, str(HERE.parent / "m0"))
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np                                                                    # noqa: E402
import torch                                                                            # noqa: E402

import run_m2a as R                                                                     # noqa: E402
from growmoe.streams import Regime, Stream, _balanced_teacher, _long, _rng, _segment, _tensor, _unit  # noqa: E402
from spike_overlap import C, D, N_REG                                                   # noqa: E402
from spike_vote_variants import MODES, VotePool                                         # noqa: E402


def mixphase_stream(seed, rho, n_train=800, n_test=400, scale=8.0, major_n=560, minor_n=240):
    rng = _rng(seed, 0xF5)
    g = rng.standard_normal((N_REG, D))
    shared = rng.standard_normal((1, D))
    centres = _unit(np.sqrt(1.0 - rho) * _unit(g) + np.sqrt(rho) * _unit(shared))
    regimes, pools = {}, {}
    for r in range(N_REG):
        teacher = _balanced_teacher(8, 1, D, C, seed=seed * 1000 + r)

        def draw(n):
            z = rng.standard_normal((n, D))
            y = teacher(_tensor(_unit(z))).argmax(1).detach().numpy()
            return _unit(scale * centres[r] + z), y

        x_tr, y_tr = draw(2 * n_train)
        x_te, y_te = draw(n_test)
        regimes[r] = Regime(r, f"fn{r}", _tensor(x_te), _long(y_te), _long(np.arange(n_test)), meta={"type": "function"})
        pools[r] = (x_tr, y_tr)
    segments, visits = [], {}
    for r in list(range(N_REG)) + [0, N_REG // 2]:
        v = visits.get(r, 0)
        visits[r] = v + 1
        sl = slice(v * n_train, (v + 1) * n_train)
        segments.append(_segment(r, v == 0, _tensor(pools[r][0][sl]), _long(pools[r][1][sl]), np.arange(sl.start, sl.stop)))
    for a, b in [(1, 2), (3, 4), (6, 7), (8, 9)]:
        xa, ya = pools[a][0][n_train:n_train + major_n], pools[a][1][n_train:n_train + major_n]
        xb, yb = pools[b][0][n_train:n_train + minor_n], pools[b][1][n_train:n_train + minor_n]
        segments.append(_segment(a, False, _tensor(np.concatenate([xa, xb])), _long(np.concatenate([ya, yb])), np.arange(major_n + minor_n)))
    return Stream("function_mixphase", D, C, regimes, segments, {"rho": rho})


def run(rho, seed, variant):
    st = mixphase_stream(seed, rho)
    pool = R.factory("A_new", st, seed, {"steps": 5})()
    pool = VotePool(pool.cfg, vote_mode=MODES[variant])
    n_pure = 12
    snap = None
    for bi, ((x, y, ctx), k) in enumerate(zip(st.algo_batches(64, seed), [k for k, _, _ in st._walk(64, seed)])):
        if k == n_pure and snap is None:                                               # accuracy just before the mixed tail begins
            xs = torch.cat([r.x_te for r in st.regimes.values()])
            ys = torch.cat([r.y_te for r in st.regimes.values()])
            snap = float((pool.predict(xs).probs.argmax(1) == ys).float().mean())
        pool.observe(x, y, ctx)
    xs = torch.cat([r.x_te for r in st.regimes.values()])
    ys = torch.cat([r.y_te for r in st.regimes.values()])
    minor = torch.cat([st.regimes[b].x_te for _, b in [(1, 2), (3, 4), (6, 7), (8, 9)]])
    minor_y = torch.cat([st.regimes[b].y_te for _, b in [(1, 2), (3, 4), (6, 7), (8, 9)]])
    return {"rho": rho, "seed": seed, "variant": variant, "before_tail": snap, "after_tail": float((pool.predict(xs).probs.argmax(1) == ys).float().mean()),
            "minority_regimes_after": float((pool.predict(minor).probs.argmax(1) == minor_y).float().mean()), "experts": len(pool.experts)}


if __name__ == "__main__":
    rows = []
    for rho in (0.0, 0.3):
        for variant in MODES:
            for seed in (100, 101, 102):
                rows.append(run(rho, seed, variant))
                print(rows[-1], flush=True)
    (HERE / "spike_mixphase.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\nrho  variant   before tail %   after tail %   minority regimes (2,4,7,9) after %   experts")
    for rho in (0.0, 0.3):
        for variant in MODES:
            rr = [r for r in rows if r["rho"] == rho and r["variant"] == variant]
            m = lambda k: np.mean([r[k] for r in rr])                                      # noqa: E731
            print(f"{rho:3.1f} {variant:8s} {100 * m('before_tail'):12.1f} {100 * m('after_tail'):13.1f} {100 * m('minority_regimes_after'):28.1f} {m('experts'):13.1f}")
