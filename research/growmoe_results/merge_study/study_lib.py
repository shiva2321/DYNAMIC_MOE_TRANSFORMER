"""Merge study (throwaway; no product file touched): when a wrong expert was founded, how can the pool recognise it and give the capacity back?

Experts are trained the way ``BasePool.train_expert`` trains them (batch + equal replay, 5 inner steps, Adam 3e-3, a 256-slot reservoir), and they keep what the pool keeps: a signature
of their inputs (prototypes), a replay reservoir (labelled, TRAINED ON) and the honest per-sample losses taken before training (the transfer test's reference). Pairs are built by
construction so that the right answer is known:

    dup_ancestry      B starts as a shrink-perturbed copy of A (a false spawn), both then keep learning the same function on different batches   -> merge is right
    dup_independent   A and B trained from different random seeds on the same function                                                           -> merge is right
    dominated         A knows two regions (two functions), B only one of them                                                                    -> retiring B is right, retiring A is wrong
    distinct_region   A and B live in different regions with different functions                                                                 -> merging is wrong
    distinct_perm     the same inputs under shifted labels                                                                                       -> merging is wrong
    partial20/partial5  B equals A except that the labels are shifted inside a cap holding 20 % / 5 % of the inputs                                -> merging is wrong (subtly)

Every candidate test is applied to every pair. Honest data = fresh samples of each regime that no expert has seen (what a held-out reservoir or the deferred stream would provide).
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from growmoe.experts import MLPExpert  # noqa: E402
from growmoe.replay import Reservoir  # noqa: E402
from growmoe.settest import generator  # noqa: E402
from growmoe.signature import VmfSignature  # noqa: E402
from growmoe.transfer import cluster_conflict  # noqa: E402

D, C = 16, 4


# ------------------------------------------------------------------------------------------------ regimes
class Regime:
    """Inputs on the unit sphere (around ``centre`` when given), labelled by a linear teacher; ``perm`` shifts every label, ``cap`` = (direction, threshold, shift) shifts the labels of
    the inputs inside the cap {x . direction > threshold}."""

    def __init__(self, W, centre=None, spread=4.0, perm=0, cap=None):
        self.W, self.centre, self.spread, self.perm, self.cap = W, centre, spread, perm, cap

    def batch(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(n, D, generator=g)
        x = F.normalize(z if self.centre is None else self.spread * self.centre[None, :] + z, dim=1)
        y = ((x @ self.W).argmax(1) + self.perm) % C
        if self.cap is not None:
            v, t, shift = self.cap
            m = (x @ v) > t
            y = torch.where(m, (y + shift) % C, y)
        return x, y


def teacher(seed):
    return torch.randn(D, C, generator=torch.Generator().manual_seed(seed))


def cap_threshold(v, share, seed=0):
    """Threshold t such that a share of uniform sphere inputs has x . v > t."""
    z = F.normalize(torch.randn(20000, D, generator=torch.Generator().manual_seed(seed)), dim=1)
    return float(torch.quantile(z @ v, 1.0 - share))


# ------------------------------------------------------------------------------------------------ an expert as the pool keeps it
class Solo:
    def __init__(self, seed, module=None, hidden=32, replay=256, inner=5, lr=3e-3):
        self.seed, self.inner = seed, inner
        self.module = module if module is not None else MLPExpert(D, hidden, 1, C, seed=seed)
        self.module.mark_seen(torch.arange(C))
        self.opt = torch.optim.Adam(self.module.parameters(), lr=lr)
        self.replay = Reservoir(replay, D, np.random.default_rng([seed, 0x52]))
        self.sig = VmfSignature(D, 256, seed=seed)
        self.ref = deque(maxlen=512)                                    # honest per-sample losses (before training), as the transfer rule keeps them

    def train(self, x, y, record=True):
        if record:
            with torch.no_grad():
                self.ref.extend(F.cross_entropy(self.module(x), y, reduction="none").tolist())
        n_rep = len(x) if self.replay.n else 0
        for _ in range(self.inner):
            self.module.train_step(x, y, self.opt, replay=self.replay.sample(n_rep) if n_rep else None)
        self.replay.add(x, y)
        self.sig.update(x)

    def fit(self, regimes, n_batches, seed0):
        for k in range(n_batches):
            r = regimes[k % len(regimes)]
            self.train(*r.batch(64, seed0 + k))
        return self


def acc(module, x, y):
    with torch.no_grad():
        return float((module(x).argmax(1) == y).float().mean())


def loss_each(module, x, y):
    with torch.no_grad():
        return F.cross_entropy(module(x), y, reduction="none")


# ------------------------------------------------------------------------------------------------ the pairs
def make_pair(kind: str, seed: int):
    """-> dict(A, B, held={'A': (x, y), 'B': (x, y)}, truth={'merge_ok': bool, 'retire_B_ok': bool, 'retire_A_ok': bool}, regimes)"""
    W1, W2 = teacher(10 + seed), teacher(500 + seed)
    c1, c2 = F.normalize(torch.randn(D, generator=torch.Generator().manual_seed(900 + seed)), dim=0), F.normalize(torch.randn(D, generator=torch.Generator().manual_seed(901 + seed)), dim=0)
    base = Regime(W1)
    s0 = 100_000 * (seed + 1)
    if kind == "dup_ancestry":
        A = Solo(seed).fit([base], 30, s0)
        B = Solo(seed + 7, module=A.module.shrink_perturb(lam=0.9, seed=seed))
        B.replay.add(*base.batch(48, s0 + 5000)); B.sig.update(base.batch(48, s0 + 5000)[0])
        A.fit([base], 30, s0 + 1000)
        B.fit([base], 30, s0 + 2000)
        rA, rB, truth = base, base, dict(merge_ok=True, retire_B_ok=True, retire_A_ok=True)
    elif kind == "dup_independent":
        A, B = Solo(seed).fit([base], 40, s0), Solo(seed + 7).fit([base], 40, s0 + 1000)
        rA, rB, truth = base, base, dict(merge_ok=True, retire_B_ok=True, retire_A_ok=True)
    elif kind == "dominated":
        r1, r2 = Regime(W1, centre=c1, spread=6.0), Regime(W2, centre=c2, spread=6.0)
        A, B = Solo(seed).fit([r1, r2], 60, s0), Solo(seed + 7).fit([r2], 40, s0 + 1000)
        rA, rB, truth = r1, r2, dict(merge_ok=True, retire_B_ok=True, retire_A_ok=False)          # A knows both regions, B only r2 ("A" is held out on BOTH regions below)
    elif kind == "distinct_region":
        r1, r2 = Regime(W1, centre=c1, spread=6.0), Regime(W2, centre=c2, spread=6.0)
        A, B = Solo(seed).fit([r1], 40, s0), Solo(seed + 7).fit([r2], 40, s0 + 1000)
        rA, rB, truth = r1, r2, dict(merge_ok=False, retire_B_ok=False, retire_A_ok=False)
    elif kind == "distinct_perm":
        rB_ = Regime(W1, perm=1)
        A, B = Solo(seed).fit([base], 40, s0), Solo(seed + 7).fit([rB_], 40, s0 + 1000)
        rA, rB, truth = base, rB_, dict(merge_ok=False, retire_B_ok=False, retire_A_ok=False)
    elif kind in ("partial20", "partial5"):
        share = 0.20 if kind == "partial20" else 0.05
        v = F.normalize(torch.randn(D, generator=torch.Generator().manual_seed(77 + seed)), dim=0)
        rB_ = Regime(W1, cap=(v, cap_threshold(v, share), 1))
        A, B = Solo(seed).fit([base], 40, s0), Solo(seed + 7).fit([rB_], 40, s0 + 1000)
        rA, rB, truth = base, rB_, dict(merge_ok=False, retire_B_ok=False, retire_A_ok=False)
    else:
        raise ValueError(kind)
    held = {"A": rA.batch(600, s0 + 90_000), "B": rB.batch(600, s0 + 91_000)}
    if kind == "dominated":
        held = {"A": (torch.cat([r1.batch(300, s0 + 90_000)[0], r2.batch(300, s0 + 90_500)[0]]), torch.cat([r1.batch(300, s0 + 90_000)[1], r2.batch(300, s0 + 90_500)[1]])),
                "B": r2.batch(600, s0 + 91_000)}
    return {"A": A, "B": B, "held": held, "truth": truth, "kind": kind}


# ------------------------------------------------------------------------------------------------ the candidate tests
def clone_merge(A: Solo, B: Solo, val_A, val_B, steps=60, seed=0):
    """What ``PhyloMoE._attempt_merges`` does: a clone of A is trained ``steps`` steps on the fit halves of both replays; drops are measured on the given validation sets."""
    rx_i, ry_i = A.replay.data
    rx_j, ry_j = B.replay.data
    hi, hj = len(rx_i) // 2, len(rx_j) // 2
    ux, uy = torch.cat([rx_i[:hi], rx_j[:hj]]), torch.cat([ry_i[:hi], ry_j[:hj]])
    clone = A.module.clone()
    opt = torch.optim.Adam(clone.parameters(), lr=3e-3)
    bs = min(64, len(ux))
    for s in range(steps):
        gen = torch.Generator().manual_seed(seed + s)
        sel = torch.randint(0, len(ux), (bs,), generator=gen)
        clone.train_step(ux[sel], uy[sel], opt, replay=None)
    va, vb = (val_A, val_B)
    return clone, (acc(A.module, *va) - acc(clone, *va)), (acc(B.module, *vb) - acc(clone, *vb))


def t0_existing(A, B):
    """The shipped test: validation halves of the replays (which the experts MEMORISED), tolerance 0.02."""
    rx_i, ry_i = A.replay.data
    rx_j, ry_j = B.replay.data
    hi, hj = len(rx_i) // 2, len(rx_j) // 2
    _, di, dj = clone_merge(A, B, (rx_i[hi:], ry_i[hi:]), (rx_j[hj:], ry_j[hj:]))
    return {"accept": di <= 0.02 and dj <= 0.02, "drop_A": di, "drop_B": dj}


def t1_honest_clone(A, B, held, tol=0.03):
    """The same clone, but judged on honest fresh samples of both regimes (a held-out reservoir or the deferred stream would provide them)."""
    _, di, dj = clone_merge(A, B, held["A"], held["B"])
    return {"accept": di <= tol and dj <= tol, "drop_A": di, "drop_B": dj}


def agreement(A, B, x):
    with torch.no_grad():
        return float((A.module(x).argmax(1) == B.module(x).argmax(1)).float().mean())


def t2_agreement(A, B, held, tau=0.80):
    """Label-free: do the two experts give the same answer? On the union of their stored prototypes (what the pool has) and on honest fresh inputs (upper bound)."""
    protos = torch.cat([A.sig.protos, B.sig.protos])
    fresh = torch.cat([held["A"][0], held["B"][0]])
    a_p, a_f = agreement(A, B, protos), agreement(A, B, fresh)
    return {"accept": a_p >= tau, "agree_protos": a_p, "agree_fresh": a_f}


def transfer_direction(src: Solo, dst: Solo, alpha=0.01, margin_k=0.25, perms=999, seed=0):
    """Does ``src`` already predict ``dst``'s stored labelled samples as well as ``dst`` itself did honestly? (the stage-4 transfer test, run between two existing experts).
    src has never seen dst's replay, so its losses on it are honest; dst's honest losses are its reference."""
    rx, ry = dst.replay.data
    losses = loss_each(src.module, rx, ry)
    margin = margin_k * float(np.log(max(C, 2)))
    ref = torch.tensor(list(dst.ref))
    conflict, p, diff = cluster_conflict(losses, ref, alpha, perms, margin, generator(4242, seed))
    se = float(torch.sqrt(losses.var() / len(losses) + ref.var() / len(ref)))
    return {"conflict": bool(conflict), "p": p, "diff": diff, "se": se, "ucb": diff + 1.645 * se, "loss_src_on_dst": float(losses.mean()), "ref_dst": float(np.mean(dst.ref))}


def t5_mutual_transfer(A, B, seed=0):
    ab = transfer_direction(A, B, seed=seed)       # A on B's data
    ba = transfer_direction(B, A, seed=seed + 1)   # B on A's data
    return {"retire_B_accept": not ab["conflict"], "retire_A_accept": not ba["conflict"], "accept": (not ab["conflict"]) and (not ba["conflict"]), "A_on_B": ab, "B_on_A": ba}


# ------------------------------------------------------------------------------------------------ what executing a merge would cost
def soup(A: Solo, B: Solo):
    m = A.module.clone()
    with torch.no_grad():
        for pm, pa, pb in zip(m.parameters(), A.module.parameters(), B.module.parameters()):
            pm.copy_(0.5 * (pa + pb))
    return m


def distill(A: Solo, B: Solo, steps=100, seed=0):
    """A student (a copy of A) trained on the union of the two experts' stored PROTOTYPES (inputs only) to reproduce the average of the experts' answers."""
    xs = torch.cat([A.sig.protos, B.sig.protos])
    with torch.no_grad():
        target = 0.5 * (A.module.proba(xs) + B.module.proba(xs))
    st = A.module.clone()
    opt = torch.optim.Adam(st.parameters(), lr=3e-3)
    for s in range(steps):
        gen = torch.Generator().manual_seed(seed + s)
        sel = torch.randint(0, len(xs), (min(64, len(xs)),), generator=gen)
        logp = F.log_softmax(st(xs[sel]), dim=-1)
        loss = F.kl_div(logp, target[sel], reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
    return st


def ensemble_acc(A, B, x, y):
    with torch.no_grad():
        return float(((A.module.proba(x) + B.module.proba(x)).argmax(1) == y).float().mean())
