"""Stage-4 spike C: can capacity need be seen on the LIVE stream, quickly, on an honest yardstick?  (throwaway; no product file touched)

Question. The shipped growth policy judged candidates on a fixed stored set (held-out cross-entropy after a 480-step probe) and did not grow a 16-unit expert that gains 18 points when
widened (E-030). This spike tests the alternative of the stage-4 design: a SHADOW trial on the incoming stream. At a chosen point of the stream the expert is cloned, the clone is
widened to twice its width (function-preserving, two initialisations: the current "random in, zero out" and a GradMax-style one), and control and shadow are trained on the SAME next
batches with the pool's recipe; per batch the honest prequential loss of each (cross-entropy on the new samples BEFORE any update) is recorded, and the paired difference is tested
after T = 10, 20, 40, 80 batches. Also recorded at the trial start: cheap diagnosis signals (prequential loss, its slope, the gap between prequential and replay-fit loss) and a cheap
natural-expansion-style score (how much of the residual output gradient new random units could explain, minus a shuffled-gradient null).

    python spike_growth_signal.py --seed 0
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent / "diagnosis"
sys.path.insert(0, str(DIAG))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy import stats  # noqa: E402

import diag_lib as D  # noqa: E402
from diag_lib import A, L  # noqa: E402
from growmoe.config import CommonCfg  # noqa: E402
from growmoe.experts import MASK_VALUE  # noqa: E402
from growmoe.pools.base import BasePool  # noqa: E402

OUT = HERE / "growth_signal"
WIDTHS = (16, 32, 64, 128, 512)
CHECKPOINTS = (900, 2300)
TRIAL_T = (10, 20, 40, 80)


# ------------------------------------------------------------------------------------------------ growth variants (spike-local copies)
def widen_random(expert, k, seed):
    """The product's widen: random incoming weights, zero outgoing weights (function-preserving)."""
    return expert.clone().widen(k, np.random.default_rng(seed))


@torch.no_grad()
def block_moments(expert, x, y):
    """Input activations a = LN(x), output gradients delta = dL/d(block output) per sample, and hidden activations, for the first block of a depth-1 expert."""
    expert.eval()
    with torch.enable_grad():
        b = expert.blocks[0]
        h0 = x.clone()
        a = b.ln(h0)
        z = F.linear(a, b.w1, b.b1)
        act = F.gelu(z)
        out = h0 + F.linear(act, b.w2, b.b2)
        out.retain_grad()
        logits = expert.head(expert.norm(out)).masked_fill(~expert.seen, MASK_VALUE)
        loss = F.cross_entropy(logits, y.long())
        loss.backward()
        delta = out.grad * len(x)
    return a.detach(), delta.detach(), act.detach()


@torch.no_grad()
def widen_gradmax(expert, k, x, y):
    """GradMax-style growth (Evci et al., ICLR 2022): new units get ZERO incoming weights and outgoing weights = the top-k left singular vectors of E[delta a^T], scaled to the mean
    norm of the existing outgoing columns. Function-preserving (zero incoming weights and gelu(0) = 0)."""
    child = expert.clone()
    a, delta, _ = block_moments(child, x, y)
    M = delta.T @ a / len(x)                                           # (d_out, d_in)
    U, S, Vt = torch.linalg.svd(M)
    b = child.blocks[0]
    d, h = b.w2.shape
    scale = float(b.w2.norm(dim=0).mean()) if h > 0 else 1.0
    cols = U[:, :k] * scale
    if cols.shape[1] < k:                                              # more new units than singular vectors: pad with small random columns
        cols = torch.cat([cols, 0.01 * scale * torch.randn(d, k - cols.shape[1])], dim=1)
    b.w1 = nn.Parameter(torch.cat([b.w1.data, torch.zeros(k, b.w1.shape[1])]))
    b.b1 = nn.Parameter(torch.cat([b.b1.data, torch.zeros(k)]))
    b.w2 = nn.Parameter(torch.cat([b.w2.data, cols], dim=1))
    return child


@torch.no_grad()
def expansion_score(expert, x, y, k, seed):
    """Share of the residual output-gradient energy that k NEW random units could explain beyond the existing units, minus the same quantity for shuffled gradients (the null)."""
    a, delta, H = block_moments(expert, x, y)
    n = len(x)
    g = torch.Generator().manual_seed(seed)
    Wp = torch.randn(k, a.shape[1], generator=g) / math.sqrt(a.shape[1])
    Hp = F.gelu(a @ Wp.T)

    def share(dl):
        lam = 1e-3 * n
        Bm = torch.linalg.solve(H.T @ H + lam * torch.eye(H.shape[1]), H.T @ dl)         # existing units explain part of the gradient
        dr = dl - H @ Bm
        Cm = torch.linalg.solve(H.T @ H + lam * torch.eye(H.shape[1]), H.T @ Hp)
        hr = Hp - H @ Cm                                                                   # new features, with what existing units already span removed
        coef = torch.linalg.solve(hr.T @ hr + lam * torch.eye(k), hr.T @ dr)
        return float(((hr @ coef) ** 2).sum() / (delta ** 2).sum())

    perm = torch.randperm(n, generator=g)
    return {"score": share(delta), "null": share(delta[perm])}


# ------------------------------------------------------------------------------------------------ training pieces
def prequential(net, xb, yb):
    with torch.no_grad():
        return float(F.cross_entropy(net(xb), yb))


def paired_train_step(models, opts, xb, yb, rx, ry):
    for net, opt in zip(models, opts):
        net.train_step(xb, yb, opt, replay=(rx, ry))


def run_trial(rec, pool, X, Y, order, start_batch, shadow, T_max):
    """Control = clone of the live expert; both trained with fresh Adam on the next batches with SHARED replay samples; returns per-batch honest prequential losses."""
    control = rec.module.clone()
    mods = [control, shadow]
    opts = [torch.optim.Adam(m.parameters(), lr=pool.cfg.lr) for m in mods]
    reservoir = copy.deepcopy(rec.replay)
    lc, ls = [], []
    for k in range(T_max):
        i = (start_batch + k) * 64
        b = torch.from_numpy(order[i:i + 64])
        xb, yb = X[b], Y[b]
        lc.append(prequential(control, xb, yb))
        ls.append(prequential(shadow, xb, yb))
        for _ in range(pool.cfg.inner_steps):
            rx, ry = reservoir.sample(len(xb))
            paired_train_step(mods, opts, xb, yb, rx, ry)
        reservoir.add(xb, yb)
    return np.array(lc), np.array(ls)


def summarise_trial(lc, ls):
    out = {}
    for T in TRIAL_T:
        d = lc[:T] - ls[:T]                                              # positive = the shadow is better
        t, p = stats.ttest_1samp(d, 0.0, alternative="greater") if len(d) > 1 else (float("nan"), float("nan"))
        out[str(T)] = {"mean_gain_nats": float(d.mean()), "rel_gain": float(d.mean() / max(lc[:T].mean(), 1e-9)), "t": float(t), "p_greater": float(p), "frac_better": float((d > 0).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    tr = L.build_train("FULL", a.seed, 8000)
    X, Y = torch.from_numpy(D.unit(tr.x())), torch.from_numpy(tr.y).long()
    order = np.random.default_rng([a.seed, 0x5F1]).permutation(len(X))
    for w in WIDTHS:
        stem = OUT / f"w{w}_s{a.seed}"
        if D.valid_json(stem.with_suffix(".json")):
            continue
        t0 = time.time()
        cfg = CommonCfg(d=A.D_MODEL, n_classes=A.N_CLASS, lr=3e-3, inner_steps=5, replay_size=20000, seed=a.seed)
        pool = BasePool(cfg)
        D.ARCHS["_tmp"] = ("expert", w, 1, True)
        eid = pool.add_expert(D.build_net("_tmp", a.seed), parent=None, state="active").id
        rec = pool.experts[eid]
        preq, fit = [], []
        res = {"width": w, "seed": a.seed, "checkpoints": {}}
        for k in range(max(CHECKPOINTS)):
            b = torch.from_numpy(order[k * 64:(k + 1) * 64])
            xb, yb = X[b], Y[b]
            preq.append(prequential(rec.module, xb, yb))
            if rec.replay.n >= 64:
                rx, ry = rec.replay.sample(64)
                fit.append(prequential(rec.module, rx, ry))
            pool.train_expert(eid, xb, yb, replay=True)
            if k + 1 in CHECKPOINTS:
                n0 = k + 1
                lp = np.array(preq)
                diag = {"preq_loss_last100": float(lp[-100:].mean()), "preq_loss_prev100": float(lp[-200:-100].mean()), "fit_loss_last100": float(np.mean(fit[-100:])),
                        "slope_ratio": float(lp[-50:].mean() / max(lp[-100:-50].mean(), 1e-9))}
                diag["gap"] = (diag["preq_loss_last100"] - diag["fit_loss_last100"]) / max(diag["preq_loss_last100"], 1e-9)
                rx, ry = rec.replay.sample(2000)
                diag["senn_lite"] = expansion_score(rec.module, rx, ry, k=max(w, 16), seed=a.seed)
                trials = {}
                for init in ("random", "gradmax"):
                    if init == "random":
                        shadow = widen_random(rec.module, w, seed=1000 + a.seed)
                    else:
                        shadow = widen_gradmax(rec.module, w, rx, ry)
                    with torch.no_grad():                                      # function preservation check
                        err = float((shadow(rx[:256]) - rec.module(rx[:256])).abs().max())
                    lc, ls = run_trial(rec, pool, X, Y, order, n0, shadow, max(TRIAL_T))
                    trials[init] = {"preservation_error": err, "summary": summarise_trial(lc, ls), "control_curve": [float(v) for v in lc], "shadow_curve": [float(v) for v in ls]}
                res["checkpoints"][str(n0)] = {"diagnosis": diag, "trials": trials}
                print(f"[w{w} s{a.seed} @{n0}] preq {diag['preq_loss_last100']:.3f} gap {diag['gap']:.2f} slope {diag['slope_ratio']:.3f} senn {diag['senn_lite']['score']:.3f}/{diag['senn_lite']['null']:.3f} | "
                      + " | ".join(f"{init}: T=20 gain {100 * t['summary']['20']['rel_gain']:+.1f}% p {t['summary']['20']['p_greater']:.3f}, T=80 {100 * t['summary']['80']['rel_gain']:+.1f}% p {t['summary']['80']['p_greater']:.3f}" for init, t in trials.items()), flush=True)
        res["seconds"] = time.time() - t0
        D.atomic_write_text(stem.with_suffix(".json"), json.dumps(res))


if __name__ == "__main__":
    main()
