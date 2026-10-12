"""SPIKE 2 (throwaway, labelled) -- how should a child expert be initialised from its parent?

Protocol. A parent expert is trained on S6 regime 0. The child regime is either unrelated (fresh random
orientation) or a small rotation of the parent with class-mean cosine c in {0.97, 0.9, 0.7, 0.5}. The child
expert starts from one of: scratch, clone, inherit(rho, sigma = 0.05) for rho in {0.25, 0.5, 0.75}, or the
affinity-scaled rule rho = rho_max * (1 - affinity) with rho_max = 0.75, where

    affinity = clip((s_child - s_floor) / (s_self - s_floor), 0, 1)

s_child = mean over child samples of the max cosine to the parent's prototypes, s_self = mean nearest-neighbour
cosine among the parent's own prototypes, s_floor = the same for uniformly random unit vectors.

The child then trains in the LOW-SAMPLE regime that a freshly spawned expert actually faces: batches of 8, 60
steps, held-out accuracy (200 test samples) after EVERY step starting from step 0 (= zero-shot, forward transfer).
Reported: accuracy at steps 0 / 4 / 16 / 60, the area under the learning curve (mean accuracy over steps 1..60),
steps to 90 % (censored at 61), and the paired difference to scratch over seeds.

First attempt (recorded, not reported as a result): at the default S6 noise (sigma 0.6) EVERY arm reached 100 %
accuracy within 10 steps, so that task could not discriminate initialisations. The noise is therefore raised to
sigma = 1.0 and 2.0 and both settings are reported; the protocol above is otherwise unchanged.

Run from the repository root:  python research/spikes/spike_init_modes.py --seeds 10
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

from growmoe import streams as S                      # noqa: E402
from growmoe.experts import MLPExpert                 # noqa: E402
from growmoe.metrics import mean_ci, paired_diff      # noqa: E402
from growmoe.signature import VmfSignature            # noqa: E402

D, HIDDEN, BATCH, STEPS, RHO_MAX, SIGMA_INH = 128, 64, 8, 60, 0.75, 0.05
LEVELS = {"unrelated": None, "cos 0.5": 0.5, "cos 0.7": 0.7, "cos 0.9": 0.9, "cos 0.97": 0.97}
MODES = ["scratch", "clone", "clone+fresh head", "rho 0.75", "affinity-scaled", "shrink-perturb"]
NOISES = (1.0, 2.0)


def curve(model, x, y, xt, yt, seed):
    """Held-out accuracy at step 0 (zero-shot) and after every training step."""
    g = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    def acc():
        with torch.no_grad():
            return (model(xt).argmax(1) == yt).float().mean().item()

    model.mark_seen(torch.arange(8))                       # all 8 labels are shared across regimes (domain-incremental)
    out = [acc()]
    for _ in range(STEPS):
        i = torch.from_numpy(g.integers(0, len(x), BATCH))
        model.train_step(x[i], y[i], opt)
        out.append(acc())
    return np.array(out)


def affinity(sig, x_child, rng):
    protos = sig.protos
    s_child = (x_child @ protos.T).max(1).values.mean().item()
    s_self = sig.nn_similarity().mean().item()
    rand = torch.nn.functional.normalize(torch.from_numpy(rng.standard_normal((500, D))).float(), dim=1)
    s_floor = (rand @ protos.T).max(1).values.mean().item()
    return float(np.clip((s_child - s_floor) / max(s_self - s_floor, 1e-9), 0.0, 1.0))


def shrink_perturb(parent, lam, seed):
    """child = lam * parent + (1 - lam) * fresh, over ALL parameters; lam = 0 is exactly the scratch initialisation
    (same seed as the scratch arm), lam = 1 is a clone (Ash and Adams, 2020, with the shrink factor set by affinity)."""
    fresh = MLPExpert(D, HIDDEN, 1, 8, seed=seed + 1000)
    child = parent.clone()
    with torch.no_grad():
        for pc, pf in zip(child.parameters(), fresh.parameters()):
            pc.mul_(lam).add_((1.0 - lam) * pf)
    return child


def one_run(seed, sigma, cos):
    st = S.s6_long(seed, n_regimes=2, related_frac=0.0 if cos is None else 1.0, rotation_cos=cos or 0.9,
                   d=D, n_train=500, n_test=200, sigma=sigma)
    xp, yp = st.segments[0].x, st.segments[0].y
    xc, yc = st.segments[1].x, st.segments[1].y
    xt, yt = st.test_sets()[1]
    parent = MLPExpert(D, HIDDEN, 1, 8, seed=seed)
    parent.mark_seen(torch.arange(8))
    g, opt = np.random.default_rng(seed + 11), None
    opt = torch.optim.Adam(parent.parameters(), lr=3e-3)
    for _ in range(400):
        i = torch.from_numpy(g.integers(0, len(xp), 32))
        parent.train_step(xp[i], yp[i], opt)
    sig = VmfSignature(D, max_protos=256, seed=seed)
    sig.update(xp)
    aff = affinity(sig, xc, np.random.default_rng(seed + 5))
    out = {}
    for mode in MODES:
        rng = np.random.default_rng([seed, 77])
        if mode == "scratch":
            child = MLPExpert(D, HIDDEN, 1, 8, seed=seed + 1000)
        elif mode == "clone":
            child = parent.clone()
        elif mode == "clone+fresh head":
            child = parent.clone()
            fresh = MLPExpert(D, HIDDEN, 1, 8, seed=seed + 1000)
            with torch.no_grad():
                child.head.weight.copy_(fresh.head.weight)
                child.head.bias.copy_(fresh.head.bias)
        elif mode == "shrink-perturb":
            child = shrink_perturb(parent, aff, seed)
        elif mode == "affinity-scaled":
            child = parent.inherit(RHO_MAX * (1.0 - aff), SIGMA_INH, rng)
        else:
            child = parent.inherit(float(mode.split()[1]), SIGMA_INH, rng)
        a = curve(child, xc, yc, xt, yt, seed)
        hit = np.nonzero(a >= 0.9)[0]
        out[mode] = {"acc0": float(a[0]), "acc4": float(a[4]), "acc16": float(a[16]), "acc60": float(a[60]),
                     "aulc": float(a[1:].mean()), "steps90": int(hit[0]) if len(hit) else STEPS + 1, "affinity": aff}
    out["parent_acc"] = float(((parent(xp[:200]).argmax(1) == yp[:200]).float().mean()))
    return out


def report(results, sigma):
    print(f"\n=== noise sigma = {sigma} ===")
    for metric, label in (("aulc", "area under the learning curve (mean accuracy over steps 1-60; higher is better)"),
                          ("acc0", "zero-shot accuracy at step 0 (forward transfer; chance = 0.125)"),
                          ("acc16", "accuracy after 16 steps (128 samples)"),
                          ("steps90", "steps to 90 % accuracy (lower is better; censored at 61)")):
        print(f"\n{label}\n  mean over seeds; d = paired difference vs scratch (p = paired t)")
        print(f"{'':11s}" + "".join(f"{m:>26s}" for m in MODES))
        for level in LEVELS:
            cells = []
            for mode in MODES:
                v = [r[mode][metric] for r in results[level]]
                if mode == "scratch":
                    cells.append(f"{np.mean(v):.3f}")
                else:
                    d = paired_diff(v, [r["scratch"][metric] for r in results[level]])
                    cells.append(f"{np.mean(v):.3f} d{d['mean_diff']:+.3f} p={d['p_t']:.3f}")
            print(f"{level:11s}" + "".join(f"{c:>26s}" for c in cells))
    print("\nmeasured affinity by level: " + ", ".join(f"{lv} {np.mean([r['scratch']['affinity'] for r in results[lv]]):.2f}" for lv in LEVELS))
    print("parent accuracy on its own data: " + f"{np.mean([r['parent_acc'] for lv in LEVELS for r in results[lv]]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="research/growmoe_results/phase0/spike2_init_modes.json")
    ap.add_argument("--summarise", action="store_true", help="only print the tables of an existing --out file")
    args = ap.parse_args()
    t0 = time.time()
    if args.summarise:
        allres = json.loads(Path(args.out).read_text())
    else:
        allres = {}
        for sigma in NOISES:
            res = {}
            for seed in range(args.seeds):
                for level, cos in LEVELS.items():
                    res.setdefault(level, []).append(one_run(seed, sigma, cos))
                print(f"sigma {sigma} seed {seed} done [{time.time() - t0:.0f}s]", flush=True)
            allres[str(sigma)] = res
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(allres))
    for sigma, res in allres.items():
        report(res, sigma)


if __name__ == "__main__":
    main()
