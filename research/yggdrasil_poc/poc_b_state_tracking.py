"""PoC-B: does the algebra of the "fibre" decide what state a model can track?

Task (the classic group word problem): read a stream of group elements g1, g2, ..., gT and output the running
product p_t = g_t o ... o g_1 after every symbol. Two groups of the SAME size (6) and the SAME input format:
  * Z6 - the cyclic group of order 6, ABELIAN  (the product does not depend on the order of the symbols)
  * S3 - permutations of three objects, NON-ABELIAN (the order matters: g1 g2 != g2 g1 in general)
Three recurrent state machines, each with a token-dependent transition applied to the previous state:
  * torus : state = angles in T^k, the token ADDS a learned phase vector (this is exactly the Yggdrasil torus fibre:
            bind = angle addition; abelian by construction), read-out MLP on (cos, sin)
  * so3   : state = a vector in R^3, the token ROTATES it by a learned rotation in SO(3) (non-abelian)
  * gru   : a standard GRU (state-dependent, non-linear transition), the usual baseline
Trained on lengths <= 16 (short-to-long curriculum), tested on length 16 (in distribution) and 32..256 (length
generalisation). Metric: accuracy of the product after the LAST symbol; chance is 1/6.

Protocol: learning rate chosen on dev seeds 100.., results on test seeds 0..9, mean +- 95% CI (t-interval).
Run:  python research/yggdrasil_poc/poc_b_state_tracking.py tune   |   ... main        (CPU, a few minutes, < 300 MB)
"""

import itertools
import json
import math
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

import os
torch.set_num_threads(int(os.environ.get("POC_THREADS", "4")))
LENGTHS = (16, 32, 64, 128, 256)
SCHEDULE = ((200, 4), (200, 8), (300, 16))          # (steps, sequence length)
BATCH = 256
N_SYM = 6


def make_group(name):
    if name == "z6":
        return torch.tensor([[(a + b) % 6 for b in range(6)] for a in range(6)])
    if name == "s5":
        perms = list(itertools.permutations(range(5)))      # order 120
        idx = {p: i for i, p in enumerate(perms)}
        mul = torch.zeros(120, 120, dtype=torch.long)
        for a, pa in enumerate(perms):
            for b, pb in enumerate(perms):
                mul[a, b] = idx[tuple(pa[pb[i]] for i in range(5))]
        return mul
    perms = list(itertools.permutations(range(3)))      # perms[0] is the identity
    idx = {p: i for i, p in enumerate(perms)}
    mul = torch.zeros(6, 6, dtype=torch.long)
    for a, pa in enumerate(perms):
        for b, pb in enumerate(perms):
            mul[a, b] = idx[tuple(pa[pb[i]] for i in range(3))]       # a after b
    return mul


def prefix_products(x, mul):
    p = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    out = torch.empty_like(x)
    for t in range(x.shape[1]):
        p = mul[x[:, t], p]
        out[:, t] = p
    return out


class TorusModel(nn.Module):
    def __init__(self, k=8, hid=64, n_sym=6):
        super().__init__()
        self.n_sym = n_sym
        self.phi = nn.Parameter(torch.rand(n_sym, k) * 2 * math.pi)
        self.head = nn.Sequential(nn.Linear(2 * k, hid), nn.ReLU(), nn.Linear(hid, n_sym))

    def forward(self, x):
        theta = torch.cumsum(self.phi[x], dim=1)          # a pure sum: the ORDER of the tokens cannot matter
        return self.head(torch.cat([theta.cos(), theta.sin()], -1))


def skew(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1),
                        torch.stack([w[..., 2], z, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)


class SO3Model(nn.Module):
    def __init__(self, hid=64, n_sym=6):
        super().__init__()
        self.n_sym = n_sym
        self.w = nn.Parameter(torch.randn(n_sym, 3))
        self.v0 = nn.Parameter(F.normalize(torch.randn(3), dim=0))
        self.head = nn.Sequential(nn.Linear(3, hid), nn.ReLU(), nn.Linear(hid, n_sym))

    def forward(self, x):
        R = torch.matrix_exp(skew(self.w))                # (n_sym, 3, 3), exactly orthogonal
        v = self.v0.expand(x.shape[0], 3)
        outs = []
        for t in range(x.shape[1]):
            v = torch.einsum("bij,bj->bi", R[x[:, t]], v)
            outs.append(v)
        return self.head(torch.stack(outs, 1))


class GRUModel(nn.Module):
    def __init__(self, emb=16, hid=32, n_sym=6):
        super().__init__()
        self.n_sym = n_sym
        self.emb = nn.Embedding(n_sym, emb)
        self.rnn = nn.GRU(emb, hid, batch_first=True)
        self.head = nn.Linear(hid, n_sym)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h)


class TorusLatticeModel(nn.Module):
    """Torus whose phases are restricted to a lattice of N points (straight-through rounding): sums of lattice phases
    are exact, so the state cannot drift however long the sequence is."""

    def __init__(self, k=8, n_lat=12, hid=64, n_sym=6):
        super().__init__()
        self.n_sym = n_sym
        self.n_lat = n_lat
        self.a = nn.Parameter(torch.rand(n_sym, k) * n_lat)
        self.head = nn.Sequential(nn.Linear(2 * k, hid), nn.ReLU(), nn.Linear(hid, n_sym))

    def forward(self, x):
        a = self.a + (self.a.round() - self.a).detach()                 # integer lattice index, straight-through gradient
        idx = torch.remainder(torch.cumsum(a[x], dim=1), self.n_lat)    # exact integer arithmetic modulo N
        theta = idx * (2 * math.pi / self.n_lat)
        return self.head(torch.cat([theta.cos(), theta.sin()], -1))


class RotationModel(nn.Module):
    """Over-parameterised non-abelian fibre: state in R^k, every token applies exp(skew) in SO(k)."""

    def __init__(self, k=8, hid=64, init_scale=0.5, n_sym=6):
        super().__init__()
        self.n_sym = n_sym
        self.k = k
        self.iu = torch.triu_indices(k, k, 1)
        self.w = nn.Parameter(torch.randn(n_sym, k * (k - 1) // 2) * init_scale)
        self.v0 = nn.Parameter(F.normalize(torch.randn(k), dim=0))
        self.head = nn.Sequential(nn.Linear(k, hid), nn.ReLU(), nn.Linear(hid, n_sym))

    def forward(self, x):
        K = torch.zeros(self.n_sym, self.k, self.k, device=x.device)
        K = K.index_put((torch.arange(self.n_sym, device=x.device)[:, None], self.iu[0][None], self.iu[1][None]), self.w)
        R = torch.matrix_exp(K - K.transpose(-1, -2))
        v = self.v0.expand(x.shape[0], self.k)
        outs = []
        for t in range(x.shape[1]):
            v = torch.einsum("bij,bj->bi", R[x[:, t]], v)
            outs.append(v)
        return self.head(torch.stack(outs, 1))


class HouseholderModel(nn.Module):
    """DeltaProduct-style fibre: each token is a product of n_h Householder reflections (I - 2uu^T) of R^k."""

    def __init__(self, k=8, n_h=2, hid=64, n_sym=6):
        super().__init__()
        self.n_sym = n_sym
        self.k = k
        self.u = nn.Parameter(torch.randn(n_sym, n_h, k))
        self.v0 = nn.Parameter(F.normalize(torch.randn(k), dim=0))
        self.head = nn.Sequential(nn.Linear(k, hid), nn.ReLU(), nn.Linear(hid, n_sym))

    def forward(self, x):
        u = F.normalize(self.u, dim=-1)
        eye = torch.eye(self.k, device=x.device)
        R = eye.expand(self.n_sym, self.k, self.k)
        for j in range(u.shape[1]):
            R = (eye - 2 * u[:, j, :, None] * u[:, j, None, :]) @ R
        v = self.v0.expand(x.shape[0], self.k)
        outs = []
        for t in range(x.shape[1]):
            v = torch.einsum("bij,bj->bi", R[x[:, t]], v)
            outs.append(v)
        return self.head(torch.stack(outs, 1))


class VQRotationModel(nn.Module):
    """
    (1) VQ-snapped rotation state:
    State in R^k, every token applies R in SO(k).
    After every token, state vector is replaced by its nearest entry of a learned codebook (K = 4 * |G|).
    Straight-through gradient, commitment loss, dead-code revival.
    """
    def __init__(self, k=8, hid=64, init_scale=0.5, n_sym=6):
        super().__init__()
        self.n_sym = n_sym
        self.k = k
        self.num_codes = 4 * n_sym
        self.iu = torch.triu_indices(k, k, 1)
        self.w = nn.Parameter(torch.randn(n_sym, k * (k - 1) // 2) * init_scale)
        self.v0 = nn.Parameter(F.normalize(torch.randn(k), dim=0))
        self.codebook = nn.Parameter(F.normalize(torch.randn(self.num_codes, k), dim=-1))
        self.head = nn.Sequential(nn.Linear(k, hid), nn.ReLU(), nn.Linear(hid, n_sym))
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        K = torch.zeros(self.n_sym, self.k, self.k, device=x.device)
        K = K.index_put((torch.arange(self.n_sym, device=x.device)[:, None], self.iu[0][None], self.iu[1][None]), self.w)
        R = torch.matrix_exp(K - K.transpose(-1, -2))
        v = self.v0.expand(x.shape[0], self.k)
        outs = []
        commit_losses = []
        C = F.normalize(self.codebook, dim=-1)

        for t in range(x.shape[1]):
            v_cont = torch.einsum("bij,bj->bi", R[x[:, t]], v)
            v_cont_norm = F.normalize(v_cont, dim=-1)

            sim = torch.matmul(v_cont_norm, C.t())
            idx = sim.argmax(dim=-1)
            v_quant = C[idx]

            c_loss = 0.25 * ((v_cont - v_quant.detach()) ** 2).mean() + ((v_cont.detach() - v_quant) ** 2).mean()
            commit_losses.append(c_loss)

            v = v_cont + (v_quant - v_cont).detach()
            outs.append(v)

        self.aux_loss = torch.stack(commit_losses).mean() if commit_losses else torch.tensor(0.0, device=x.device)
        return self.head(torch.stack(outs, 1))


class GroupLookupModel(nn.Module):
    """
    (2) Finite-group lookup state:
    Soft permutation / Sinkhorn representation of group elements with exact multiplication table update.
    """
    def __init__(self, n_sym=6, n_sinkhorn=5, hid=64):
        super().__init__()
        self.n_sym = n_sym
        self.n_sinkhorn = n_sinkhorn
        self.logits = nn.Parameter(torch.randn(n_sym, n_sym, n_sym) * 0.1)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        P_log = self.logits
        for _ in range(self.n_sinkhorn):
            P_log = P_log - torch.logsumexp(P_log, dim=-1, keepdim=True)
            P_log = P_log - torch.logsumexp(P_log, dim=-2, keepdim=True)
        P_soft = torch.exp(P_log)

        hard_idx = P_soft.argmax(dim=-2)
        P_hard = F.one_hot(hard_idx, num_classes=self.n_sym).float().permute(0, 2, 1)

        P = P_soft + (P_hard - P_soft).detach()

        B, T = x.shape
        s = torch.zeros(B, self.n_sym, device=x.device)
        s[:, 0] = 1.0

        outs = []
        for t in range(T):
            P_t = P[x[:, t]]
            s = torch.einsum("bij,bj->bi", P_t, s)
            outs.append(s)

        out_probs = torch.stack(outs, 1)
        logits = torch.log(out_probs.clamp(min=1e-8))
        return logits


MODELS = {"torus": TorusModel, "so3": SO3Model, "gru": GRUModel,
          "torus_lat": TorusLatticeModel, "so8": RotationModel, "hh2": HouseholderModel,
          "vq_rot": VQRotationModel, "grp_lookup": GroupLookupModel}


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_and_test(model_name, group, seed, lr, n_test=1000):
    mul = make_group(group).to(DEVICE)
    n_sym = mul.shape[0]
    torch.manual_seed(seed)
    model = MODELS[model_name](n_sym=n_sym).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    sched = ((200, 6), (200, 12), (300, 24)) if group == "s5" else SCHEDULE
    for steps, T in sched:
        for _ in range(steps):
            x = torch.randint(0, n_sym, (BATCH, T), generator=g).to(DEVICE)
            out = model(x)
            loss = F.cross_entropy(out.reshape(-1, n_sym), prefix_products(x, mul).reshape(-1))
            if hasattr(model, "aux_loss") and model.aux_loss > 0:
                loss = loss + model.aux_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    accs = {}
    gt = torch.Generator(device="cpu").manual_seed(10_000 + seed)
    with torch.no_grad():
        for T in LENGTHS:
            x = torch.randint(0, n_sym, (n_test, T), generator=gt).to(DEVICE)
            pred = model(x)[:, -1].argmax(-1)
            accs[T] = (pred == prefix_products(x, mul)[:, -1]).float().mean().item()
    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return accs


def ci95(a):
    a = np.asarray(a, dtype=float)
    if len(a) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / math.sqrt(len(a)))


OUT = Path("research/yggdrasil_results/poc_b")
LR_GRID = (3e-3, 1e-2, 3e-2, 1e-1)


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def tune(names, groups=("z6", "s3", "s5")):
    """Learning rate per model, chosen on dev seeds 100, 101 as the mean over all tasks of the accuracy at lengths 16 and 64."""
    print("tuning the learning rate per model on dev seeds 100, 101, mean over tasks of acc at length 16 and 64", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for m in names:
        scores = {}
        for lr in LR_GRID:
            vals = [(a[16] + a[64]) / 2 for grp in groups for a in (train_and_test(m, grp, s, lr) for s in (100, 101))]
            scores[lr] = float(np.mean(vals))
            print(f"  {m:10s} lr {lr:g}: {scores[lr]:.3f}", flush=True)
        best = max(scores, key=scores.get)
        (OUT / f"lr_{m}.json").write_text(json.dumps({"lr": best, "scores": {str(k): v for k, v in scores.items()}}))
        print(f"  -> {m}: lr {best:g}", flush=True)


def main(names, groups=("z6", "s3", "s5")):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for m in names:
        lr = load(OUT / f"lr_{m}.json")["lr"]
        res = load(OUT / f"res_{m}.json") or {"lr": lr}
        res["lr"] = lr
        for group in groups:
            runs = [train_and_test(m, group, s, lr) for s in range(10)]
            res[group] = {str(T): [r[T] for r in runs] for T in LENGTHS}
            row = "  ".join(f"T={T}: {100 * np.mean([r[T] for r in runs]):5.1f}+-{100 * ci95([r[T] for r in runs]):4.1f}" for T in LENGTHS)
            print(f"{group:4s} {m:10s} (lr {lr:g}) {row}   [{time.time() - t0:.0f}s]", flush=True)
        (OUT / f"res_{m}.json").write_text(json.dumps(res))


def report():
    print("accuracy (%) of the product after the LAST symbol; mean +- 95% CI over 10 seeds; trained on lengths <= 16 (<=24 for S5)")
    print(f"{'task':4s} {'model':10s} {'lr':>6s} | " + " ".join(f"{'T=' + str(T):>12s}" for T in LENGTHS) + " | seeds >=95% at T=256")
    for group in ("z6", "s3", "s5"):
        for m in MODELS:
            r = load(OUT / f"res_{m}.json")
            if group not in r:
                continue
            cells = " ".join(f"{100 * np.mean(r[group][str(T)]):6.1f}+-{100 * ci95(r[group][str(T)]):4.1f}" for T in LENGTHS)
            solved = sum(v >= 0.95 for v in r[group]["256"])
            print(f"{group:4s} {m:10s} {r['lr']:6g} | {cells} | {solved}/10")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    chosen = sys.argv[2:] or list(MODELS)
    if mode == "report":
        report()
    else:
        {"tune": tune, "main": main}[mode](chosen)
