"""THROWAWAY spike on DEV seeds (100-102): what happens to the repaired pool when regimes OVERLAP in input space?

function_regimes puts every regime in its own region (centres nearly orthogonal). Here the centres share a common component so that
their pairwise cosine is rho (0 = as in M0 / M2-A). The labels stay a different hidden function per regime, so at high rho the
inputs of different regimes look alike and only the function differs.

Reported per rho: accuracy of the repaired pool (A_new, 5 steps), of oracle-routed experts (B4, 5 steps), the owner-routed bound for
the pool's own experts, number of experts for 10 regimes, and the lifecycle events.

    python research/growmoe_results/rv2/spike_overlap.py
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "m2a"))
sys.path.insert(0, str(HERE.parent / "m0"))
sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402

import run_m2a as R                                                                     # noqa: E402
from growmoe import bench                                                               # noqa: E402
from growmoe.baselines import OracleRouted                                              # noqa: E402
from growmoe.config import CommonCfg                                                    # noqa: E402
from growmoe.fairfight import run_fast                                                  # noqa: E402
from growmoe.streams import Regime, Stream, _balanced_teacher, _long, _rng, _segment, _tensor, _unit  # noqa: E402

torch.set_num_threads(3)
D, C, N_REG = 128, 8, 10


def overlap_stream(seed, rho, n_train=800, n_test=400, scale=8.0):
    """Copy of growmoe.streams_m1.function_regimes with one change: centres share a component so that their cosine is about rho."""
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
    schedule = list(range(N_REG)) + [0, N_REG // 2]
    segments, visits = [], {}
    for r in schedule:
        v = visits.get(r, 0)
        visits[r] = v + 1
        sl = slice(v * n_train, (v + 1) * n_train)
        segments.append(_segment(r, v == 0, _tensor(pools[r][0][sl]), _long(pools[r][1][sl]), np.arange(sl.start, sl.stop)))
    return Stream("function_overlap", D, C, regimes, segments, {"rho": rho}), float(np.mean((centres @ centres.T)[np.triu_indices(N_REG, 1)]))


def run(rho, seed):
    st, mean_cos = overlap_stream(seed, rho)
    make = R.factory("A_new", st, seed, {"steps": 5})
    pool = make()
    for x, y, ctx in st.algo_batches(64, seed):
        pool.observe(x, y, ctx)
    eids = list(pool.experts)
    x = torch.cat([r.x_te for r in st.regimes.values()])
    y = torch.cat([r.y_te for r in st.regimes.values()])
    regime = torch.cat([torch.full((len(r.y_te),), i, dtype=torch.long) for i, r in enumerate(st.regimes.values())])
    pred = pool.predict(x)
    acc = float((pred.probs.argmax(1) == y).float().mean())
    xn = pool._normalize(x.float())
    with torch.no_grad():
        per_e = torch.stack([pool.experts[e].module.proba(xn).argmax(1) for e in eids])
    idx = torch.tensor([eids.index(int(e)) for e in pred.expert])
    owner = torch.stack([torch.mode(idx[regime == r]).values for r in range(N_REG)])
    acc_owner = float((per_e[owner[regime], torch.arange(len(y))] == y).float().mean())
    routed_to_owner = float((idx == owner[regime]).float().mean())
    # the best the pool's own experts could do if every input were sent to the expert that is best for ITS regime
    best_per_regime = torch.stack([(per_e[:, regime == r] == y[regime == r][None]).float().mean(1).max() for r in range(N_REG)])
    b4 = run_fast(lambda: OracleRouted(bench.make_regime_of(st, 64), CommonCfg(seed=seed, d=D, n_classes=C, inner_steps=5)), st, seed, eval_every="final").acc
    ev = {k: sum(1 for e in pool.events if e.kind == k) for k in ("spawn", "extend", "alarm")}
    return {"rho": rho, "seed": seed, "mean_centre_cos": mean_cos, "acc": acc, "acc_owner_bound": acc_owner, "acc_best_expert_per_regime": float(best_per_regime.mean()),
            "routed_to_owner": routed_to_owner, "b4": b4, "experts": len(eids), **ev}


if __name__ == "__main__":
    rows = []
    for rho in (0.0, 0.3, 0.6, 0.85):
        for seed in (100, 101, 102):
            rows.append(run(rho, seed))
            print(rows[-1], flush=True)
    (HERE / "spike_overlap.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\nrho   centre cos   pool acc   owner bound   best-expert bound   routed to owner   oracle B4   experts (10 regimes)   spawn extend alarm")
    for rho in (0.0, 0.3, 0.6, 0.85):
        rr = [r for r in rows if r["rho"] == rho]
        m = lambda k: np.mean([r[k] for r in rr])                                          # noqa: E731
        print(f"{rho:4.2f} {m('mean_centre_cos'):10.2f} {100 * m('acc'):10.1f} {100 * m('acc_owner_bound'):13.1f} {100 * m('acc_best_expert_per_regime'):19.1f} "
              f"{100 * m('routed_to_owner'):17.1f} {100 * m('b4'):11.1f} {m('experts'):15.1f}  {m('spawn'):9.1f} {m('extend'):6.1f} {m('alarm'):5.1f}")
