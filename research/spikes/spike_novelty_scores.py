"""SPIKE 1 (throwaway, labelled) -- sample-level novelty AUROC of six detectors on S1 / S2 / S6 latents.

Protocol (fixed before looking at any result). At every regime's FIRST appearance, fit each detector on the
training data seen before it, then score held-out TEST samples: positives = the new regime's test set, negatives =
test samples of regimes seen before. AUROC per transition; mean over transitions; mean +- 95 % CI over seeds.
Novelty scores are high for novel samples. No hyper-parameter is tuned here.

Detectors: Gaussian-Mahalanobis (per-regime Gaussian, Ledoit-Wolf), kNN (5th nearest cosine), vMF-KDE (the
signature of growmoe.signature, raw score and conformal p-value, max over per-regime signatures), VAE-ELBO,
key-activation (cosine to the regime mean direction, z-scored on that regime), RND.

Run from the repository root:  python research/spikes/spike_novelty_scores.py --seeds 5
"""

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import torch.nn as nn                                 # noqa: E402
import torch.nn.functional as F                       # noqa: E402
from sklearn.covariance import LedoitWolf             # noqa: E402

from growmoe import streams as S                      # noqa: E402
from growmoe.frontend import HashFrontend             # noqa: E402
from growmoe.metrics import auroc, mean_ci            # noqa: E402
from growmoe.signature import VmfSignature            # noqa: E402

D = 128


# ------------------------------------------------------------------------------------------ detectors
class GaussianMaha:
    name = "Gaussian-Mahalanobis"

    def fit(self, train, seed):
        self.params = []
        for X in train.values():
            lw = LedoitWolf().fit(X.numpy().astype(np.float64))
            self.params.append((lw.location_, lw.precision_))

    def score(self, x):
        X = x.numpy().astype(np.float64)
        d2 = np.stack([np.einsum("ij,jk,ik->i", X - mu, P, X - mu) for mu, P in self.params])
        return d2.min(axis=0)


class MahaConformal:
    """Per-regime Gaussian (Ledoit-Wolf on 70 % of the regime) with split-conformal p-values on the other 30 %."""
    name = "Gaussian-Mahalanobis conformal"

    def fit(self, train, seed):
        self.models = []
        for r, X in train.items():
            perm = np.random.default_rng([seed, int(r), 9]).permutation(len(X))
            cut = int(0.7 * len(X))
            lw = LedoitWolf().fit(X[perm[:cut]].numpy().astype(np.float64))
            cal = self._d2(X[perm[cut:]], lw.location_, lw.precision_)
            self.models.append((lw.location_, lw.precision_, np.sort(cal)))

    @staticmethod
    def _d2(x, mu, P):
        X = x.numpy().astype(np.float64)
        return np.einsum("ij,jk,ik->i", X - mu, P, X - mu)

    def pvalues(self, x):
        out = []
        for mu, P, cal in self.models:
            d2 = self._d2(x, mu, P)
            n_ge = len(cal) - np.searchsorted(cal, d2, side="left")           # calibration scores >= d2
            out.append((1.0 + n_ge) / (len(cal) + 1.0))
        return torch.from_numpy(np.stack(out)).float()

    def score(self, x):
        return (-self.pvalues(x).max(0).values).numpy()


class UnionKdeMaha:
    """Flag a sample if EITHER detector rejects it for a regime (Bonferroni over the two), then max over regimes."""
    name = "KDE u Maha (Bonferroni)"

    def __init__(self):
        self.kde, self.maha = VmfKDE(True, 256, GRID_EXT), MahaConformal()

    def fit(self, train, seed):
        self.kde.fit(train, seed)
        self.maha.fit(train, seed)

    def score(self, x):
        p = torch.clamp(2.0 * torch.minimum(self.kde.pvalues(x), self.maha.pvalues(x)), max=1.0)
        return (-p.max(0).values).numpy()


class KNN:
    name = "kNN(k=5)"

    def fit(self, train, seed):
        self.ref = torch.cat(list(train.values()))

    def score(self, x):
        return (-(x @ self.ref.T).topk(5, dim=1).values[:, -1]).numpy()


GRID_EXT = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0)


class VmfKDE:
    def __init__(self, conformal, max_protos=256, grid=None):
        self.conformal, self.max_protos, self.grid = conformal, max_protos, grid
        tag = "" if grid is None else f", kappa<={int(max(grid))}"
        self.name = f"vMF-KDE {'conformal' if conformal else 'raw'} (protos={max_protos}{tag})"

    def pvalues(self, x):
        return torch.stack([s.pvalue(x) for s in self.sigs])                  # (n_regimes, B)

    def fit(self, train, seed):
        self.sigs, self.kappas = [], []
        for r, X in train.items():
            kw = {} if self.grid is None else {"kappa_grid": self.grid}
            s = VmfSignature(D, max_protos=self.max_protos, seed=seed * 100 + int(r), **kw)
            s.update(X)
            s.fit()
            self.sigs.append(s)
            self.kappas.append(s.kappa)

    def score(self, x):
        if self.conformal:
            return (-torch.stack([s.pvalue(x) for s in self.sigs]).max(0).values).numpy()
        return (-torch.stack([s.score(x) for s in self.sigs]).max(0).values).numpy()


class KeyActivation:
    name = "key-activation"

    def fit(self, train, seed):
        self.keys, self.stats = [], []
        for X in train.values():
            k = F.normalize(X.mean(0), dim=0)
            c = X @ k
            self.keys.append(k)
            self.stats.append((c.mean(), c.std().clamp_min(1e-6)))

    def score(self, x):
        z = torch.stack([((x @ k) - m) / s for k, (m, s) in zip(self.keys, self.stats)])
        return (-z.max(0).values).numpy()


class _VAE(nn.Module):
    def __init__(self, d=D, h=64, z=16):
        super().__init__()
        self.enc, self.mu, self.lv = nn.Linear(d, h), nn.Linear(h, z), nn.Linear(h, z)
        self.dec, self.xmu, self.xlv = nn.Linear(z, h), nn.Linear(h, d), nn.Linear(h, d)

    def neg_elbo(self, x, sample):
        e = torch.tanh(self.enc(x))
        mu, lv = self.mu(e), self.lv(e)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if sample else mu
        h = torch.tanh(self.dec(z))
        xm, xl = self.xmu(h), self.xlv(h).clamp(-8, 4)
        nll = 0.5 * (((x - xm) ** 2) / torch.exp(xl) + xl + math.log(2 * math.pi)).sum(1)
        kl = 0.5 * (torch.exp(lv) + mu ** 2 - 1.0 - lv).sum(1)
        return nll + kl


class VaeElbo:
    name = "VAE-ELBO"

    def fit(self, train, seed, steps=300):
        torch.manual_seed(seed)
        X = torch.cat(list(train.values()))
        self.m = _VAE()
        opt = torch.optim.Adam(self.m.parameters(), lr=2e-3)
        for _ in range(steps):
            xb = X[torch.randint(0, len(X), (128,))]
            loss = self.m.neg_elbo(xb, True).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def score(self, x):
        with torch.no_grad():
            return self.m.neg_elbo(x, False).numpy()


class RND:
    name = "RND"

    def fit(self, train, seed, steps=300):
        torch.manual_seed(seed + 7)
        X = torch.cat(list(train.values()))
        mk = lambda: nn.Sequential(nn.Linear(D, 128), nn.Tanh(), nn.Linear(128, 64))
        self.target, self.pred = mk(), mk()
        for p in self.target.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam(self.pred.parameters(), lr=2e-3)
        for _ in range(steps):
            xb = X[torch.randint(0, len(X), (128,))]
            loss = ((self.pred(xb) - self.target(xb)) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def score(self, x):
        with torch.no_grad():
            return ((self.pred(x) - self.target(x)) ** 2).sum(1).numpy()


def detectors():
    return [GaussianMaha(), MahaConformal(), KNN(), VmfKDE(True), VmfKDE(False), VmfKDE(True, 64), VmfKDE(True, 1024),
            VmfKDE(True, 256, GRID_EXT), VmfKDE(True, 1024, GRID_EXT), UnionKdeMaha(),
            VaeElbo(), KeyActivation(), RND()]


# -------------------------------------------------------------------------------------- the protocol
def first_visits(stream):
    order, seen = [], set()
    for i, seg in enumerate(stream.segments):
        if seg.regime not in seen:
            seen.add(seg.regime)
            order.append((seg.regime, i))
    return order


def run_stream(stream, seed, stream_name, rows, max_known=100, max_novel=200):
    order = first_visits(stream)
    tests = stream.test_sets()
    for r, seg_idx in order[1:]:
        before = [s for s in stream.segments[:seg_idx]]
        train = {}
        for s in before:
            train.setdefault(s.regime, []).append(s.x)
        train = {k: torch.cat(v) for k, v in train.items()}
        xs_known = torch.cat([tests[q][0][:max_known] for q in train])
        x_new = tests[r][0][:max_novel]
        x_all = torch.cat([xs_known, x_new])
        y_all = np.r_[np.zeros(len(xs_known)), np.ones(len(x_new))]
        meta = stream.regimes[r]
        for det in detectors():
            det.fit(train, seed * 1000 + r)
            rows.append({"stream": stream_name, "seed": seed, "regime": r, "name": meta.name,
                         "type": meta.meta.get("type"), "related": meta.parent is not None, "n_seen": len(train),
                         "detector": det.name, "auroc": auroc(det.score(x_all), y_all),
                         "kappa": float(np.mean(det.kappas)) if hasattr(det, "kappas") else None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="research/growmoe_results/phase0/spike1_novelty.json")
    ap.add_argument("--s2_regimes", type=int, default=10)
    ap.add_argument("--s6_regimes", type=int, default=12)
    ap.add_argument("--pairs", action="store_true", help="run only the S1 ordered type-pair analysis")
    args = ap.parse_args()
    rows, t0 = [], time.time()
    if args.pairs:
        for seed in range(args.seeds):
            run_s1_pairs(S.s1_types(seed, frontend=HashFrontend(d=D, seed=seed)), seed, rows)
            print(f"pairs seed {seed} done [{time.time() - t0:.0f}s]", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows))
        pairs_summary(rows, args.seeds)
        return
    for seed in range(args.seeds):
        fe = HashFrontend(d=D, seed=seed)
        run_stream(S.s1_types(seed, frontend=fe), seed, "S1", rows)
        run_stream(S.s2_permuted(seed, n_regimes=args.s2_regimes, frontend=fe), seed, "S2", rows)
        run_stream(S.s6_long(seed, n_regimes=args.s6_regimes), seed, "S6", rows)
        print(f"seed {seed} done  [{time.time() - t0:.0f}s]", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows))

    names = list(dict.fromkeys(r["detector"] for r in rows))
    print("\nAUROC, first-visit regime vs seen regimes (mean over transitions, then mean [95% CI] over seeds)")
    print(f"{'detector':38s}" + "".join(f"{s:>26s}" for s in ("S1", "S2", "S6")))
    for n in names:
        cells = []
        for st in ("S1", "S2", "S6"):
            per_seed = [np.mean([r["auroc"] for r in rows if r["stream"] == st and r["detector"] == n and r["seed"] == s])
                        for s in range(args.seeds)]
            m, lo, hi = mean_ci(per_seed)
            cells.append(f"{m:6.3f} [{lo:5.3f},{hi:5.3f}]")
        print(f"{n:38s}" + "".join(f"{c:>26s}" for c in cells))
    breakdown(rows, args.seeds)


def pair_detectors():
    return [GaussianMaha(), MahaConformal(), KNN(), VmfKDE(True), VmfKDE(False), VaeElbo(), KeyActivation(), RND()]


def run_s1_pairs(stream, seed, rows, max_n=200):
    """Ordered type pairs: fit on ONE regime, then score every other regime's test set against that regime's test set."""
    tests = stream.test_sets()
    train = {r: torch.cat([s.x for s in stream.segments if s.regime == r and s.first_visit]) for r in stream.regimes}
    for a in stream.regimes:
        for det in pair_detectors():
            det.fit({a: train[a]}, seed * 1000 + a)
            for b in stream.regimes:
                if b == a:
                    continue
                x = torch.cat([tests[a][0][:max_n], tests[b][0][:max_n]])
                y = np.r_[np.zeros(len(tests[a][0][:max_n])), np.ones(len(tests[b][0][:max_n]))]
                rows.append({"stream": "S1pair", "seed": seed, "fit": stream.regimes[a].name, "new": stream.regimes[b].name,
                             "detector": det.name, "auroc": auroc(det.score(x), y)})


def pairs_summary(rows, seeds):
    names = list(dict.fromkeys(r["detector"] for r in rows))
    w = max(len(n) for n in names) + 2
    print("S1 ordered type pairs (fit on one type, test every other type): 42 pairs x seeds")
    print(f"{'detector':{w}s}{'mean':>8s}{'min':>8s}{'#pairs<0.8':>12s}{'#pairs<0.5':>12s}")
    for n in names:
        per_pair = {}
        for r in rows:
            if r["detector"] == n:
                per_pair.setdefault((r["fit"], r["new"]), []).append(r["auroc"])
        m = np.array([np.mean(v) for v in per_pair.values()])
        print(f"{n:{w}s}{m.mean():8.3f}{m.min():8.3f}{int((m < 0.8).sum()):12d}{int((m < 0.5).sum()):12d}")
    worst = {}
    for r in rows:
        if r["detector"] in ("VAE-ELBO", "vMF-KDE conformal (protos=256)"):
            worst.setdefault((r["detector"], r["fit"], r["new"]), []).append(r["auroc"])
    for det in ("VAE-ELBO", "vMF-KDE conformal (protos=256)"):
        bad = sorted(((np.mean(v), a, b) for (d, a, b), v in worst.items() if d == det and np.mean(v) < 0.8))
        print(f"{det}: pairs below 0.8 (mean AUROC, fit -> new): " + ("none" if not bad else "; ".join(f"{a}->{b} {m:.2f}" for m, a, b in bad)))


def breakdown(rows, seeds):
    names = list(dict.fromkeys(r["detector"] for r in rows))
    w = max(len(n) for n in names) + 2
    print()
    print("S1 by arriving regime (mean over seeds)")
    regs = ["wine", "breast_cancer", "sinusoid", "ar2", "pydoc_text", "stdlib_code"]
    print(f"{'detector':{w}s}" + "".join(f"{r[:12]:>13s}" for r in regs))
    for n in names:
        print(f"{n:{w}s}" + "".join(f"{np.mean([r['auroc'] for r in rows if r['stream'] == 'S1' and r['name'] == g and r['detector'] == n]):13.3f}" for g in regs))
    print()
    print("S6 columns: unrelated arrivals, related arrivals (small rotation of an earlier regime)")
    for n in names:
        print(f"{n:{w}s}" + "".join(f"{np.mean([r['auroc'] for r in rows if r['stream'] == 'S6' and r['related'] == rel and r['detector'] == n]):13.3f}" for rel in (False, True)))


if __name__ == "__main__":
    main()
