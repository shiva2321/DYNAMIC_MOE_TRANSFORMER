"""Why do the repair tests heal so few duplicates? (throwaway)  Runs the 'none' arm (no repair) to the end, then asks every repair test about every (natural, injected) twin pair of the
same regime and prints the numbers each test had to work with. The same stream with the 'rival' and 'hold' arms is run alongside so their evidence stores can be read.

    python healing2_diag.py [seed] [inject]
"""

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import healing2_study as H  # noqa: E402
from growmoe.streams_m1 import function_regimes  # noqa: E402
from growmoe.transfer import cluster_conflict  # noqa: E402
from growmoe.settest import generator  # noqa: E402


def trained(arm: str, seed: int, inject_at=(3,)):
    """Re-run H.run's loop and return the pool plus the ids (H.run only returns summaries)."""
    st = function_regimes(seed, n_regimes=H.REGIMES, d=H.D, n_classes=H.C, n_train=640, n_test=200)
    pool = H.make_pool(arm, seed, None)
    pool.cfg.merge_every = 10 ** 9                                                       # look at the evidence, do not act on it
    natural, injected, last_k, within, seg_natural = [], [], None, 0, 0
    for k, seg, idx in st._walk(64, seed):
        if k != last_k:
            last_k, within, seg_natural = k, 0, 0
        within += 1
        x, y = seg.x[idx], seg.y[idx]
        x_ho = y_ho = None
        if arm in H.HOLD_ARMS and len(pool.experts) > 0:
            mask = torch.zeros(len(y), dtype=torch.bool)
            mask[::8] = True
            x, y, x_ho, y_ho = x[~mask], y[~mask], x[mask], y[mask]
        before = set(pool.experts)
        if arm in H.RIVAL_ARMS:
            pool.rival_observe(x, y)
        pool.observe(x, y)
        new = sorted(set(pool.experts) - before)
        natural += new
        for e in new:
            pool.regime_of[e] = seg.regime
        seg_natural += len(new)
        if x_ho is not None:
            pool.add_holdout(x_ho, y_ho)
        if seg.first_visit and within in inject_at and seg_natural:
            new = H.inject(pool, x, y)
            injected += new
            for e in new:
                pool.regime_of[e] = seg.regime
    return st, pool, natural, injected


def main() -> None:
    torch.set_num_threads(2)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    inject_at = tuple(int(v) for v in sys.argv[2].split(",")) if len(sys.argv) > 2 else (3,)
    st, hp, nat, inj = trained("hold", seed, inject_at)
    _, rp, nat_r, inj_r = trained("rival", seed, inject_at)
    _, pp, nat_p, inj_p = trained("preq", seed, inject_at)
    cfg = pp.cfg
    print(f"seed {seed} injections at batches {inject_at}: pool sizes hold {len(hp.experts)}, rival {len(rp.experts)}, preq {len(pp.experts)}")
    tests = {r: st for r, st in st.test_sets().items()}
    for D in inj_p[:12]:
        reg = pp.regime_of[D]
        N = next(e for e in nat_p if pp.regime_of.get(e) == reg and e not in inj_p)
        rd, rn = pp.experts[D], pp.experts[N]
        xt, yt = tests[reg]
        with torch.no_grad():
            accD = float((rd.module(pp._normalize(xt.float())).argmax(1) == yt).float().mean())
            accN = float((rn.module(pp._normalize(xt.float())).argmax(1) == yt).float().mean())
        ov = (float(rd.meta["sig"].pvalue(rn.meta["sig"].protos).mean()), float(rn.meta["sig"].pvalue(rd.meta["sig"].protos).mean()))
        X = torch.cat([rd.meta["sig"].protos, rn.meta["sig"].protos])
        with torch.no_grad():
            agree = float((rd.module(X).argmax(1) == rn.module(X).argmax(1)).float().mean())
        line = [f"regime {reg:2d} N={N:2d} D={D:2d} acc N {100 * accN:4.1f} D {100 * accD:4.1f} | overlap {ov[0]:.2f}/{ov[1]:.2f} agree {agree:.2f} | assigned N {rn.n_assigned} D {rd.n_assigned}"]
        # preq: keeper i, retired j: i's loss on j's replay vs j's honest reference
        for keeper, retired, name in ((D, N, "retire N"), (N, D, "retire D")):
            ri, rj = pp.experts[keeper], pp.experts[retired]
            ref = list(rj.meta.get("ref", ()))
            if rj.replay is None or rj.replay.n < 64 or len(ref) < cfg.transfer_min_ref:
                line.append(f"preq {name}: n/a (replay {None if rj.replay is None else rj.replay.n}, ref {len(ref)})")
                continue
            rx, ry = rj.replay.data
            with torch.no_grad():
                losses = F.cross_entropy(ri.module(rx), ry, reduction="none")
            margin = 0.10 * math.log(max(int(rj.module.seen.sum()), 2))
            conflict, p, diff = cluster_conflict(losses, torch.tensor(ref), cfg.transfer_alpha, cfg.transfer_perms, margin, generator(cfg.seed, 0x4D47, 0, keeper, retired))
            line.append(f"preq {name}: diff {diff:+.2f} p {p:.3f} margin {margin:.2f} -> {'REFUSE' if conflict else 'pass'} (replay {rx.shape[0]}, ref {len(ref)})")
        # hold
        for keeper, retired, name in ((D, N, "retire N"), (N, D, "retire D")):
            r = hp.hold.get(retired)
            if r is None or r.n < 2:
                line.append(f"hold {name}: n={0 if r is None else r.n}")
                continue
            hx, hy = r.data
            with torch.no_grad():
                d = F.cross_entropy(hp.experts[keeper].module(hx), hy, reduction="none") - F.cross_entropy(hp.experts[retired].module(hx), hy, reduction="none")
            m, se = float(d.mean()), float(d.std() / math.sqrt(len(d)))
            margin = 0.10 * math.log(max(int(hp.experts[retired].module.seen.sum()), 2))
            line.append(f"hold {name}: n={len(d)} mean {m:+.2f} se {se:.2f} ucb {m + 2.33 * se:+.2f} margin {margin:.2f} -> {'pass' if m + 2.33 * se < margin else 'refuse'}")
        for keeper, retired, name in ((D, N, "retire N"), (N, D, "retire D")):
            dq = rp.rival_stat.get((retired, keeper))
            if not dq or len(dq) < 2:
                line.append(f"rival {name}: n={0 if not dq else len(dq)}")
                continue
            d = np.asarray(dq)
            m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
            line.append(f"rival {name}: n={len(d)} mean {m:+.2f} se {se:.2f} ucb {m + 2.33 * se:+.2f}")
        print("\n  ".join(line))


if __name__ == "__main__":
    main()
