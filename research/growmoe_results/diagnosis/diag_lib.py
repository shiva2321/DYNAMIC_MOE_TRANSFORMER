"""Tools for the GrowMoE diagnosis study (why does it fall behind, why does it not work as intended).

Nothing here changes product code. The learners reuse the pieces the pool itself uses (``MLPExpert``, ``BasePool.train_expert``, ``Reservoir``) so that "the pool's
training recipe" is the real thing, only without the gate, the probation buffer and the lifecycle. Data, test grid and scoring are the arithmetic benchmark's
(``research/growmoe_results/arith``): the same held-out problems, exact match (every answer slot right).
"""

import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARITH = HERE.parent / "arith"
sys.path.insert(0, str(ARITH))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import arith_lib as L  # noqa: E402
from growmoe import arith as A  # noqa: E402
from growmoe.config import CommonCfg  # noqa: E402
from growmoe.experts import MLPExpert  # noqa: E402
from growmoe.pools.base import BasePool  # noqa: E402
from growmoe.replay import Reservoir  # noqa: E402

DEV = L.DEV


# ---------------------------------------------------------------------------------------------------------------- outage-proof writing
# The machine loses power now and then (see memory note plan-for-daily-power-outages): a result file must never be left half written, and a half-written file must never count as done.
def atomic_write_bytes(path, data: bytes) -> None:
    """Write ``data`` to a temp file in the same folder, flush it to disk, then rename it over ``path`` (the rename is atomic on the same volume)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_text(path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_savez(path, **arrays) -> None:
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    atomic_write_bytes(path, buf.getvalue())


def atomic_pickle(path, obj) -> None:
    """Pickle an object this project created itself (checkpoints); never used on files from outside."""
    import pickle
    atomic_write_bytes(path, pickle.dumps(obj, protocol=4))


def valid_json(path) -> bool:
    """True when the file exists and parses (a truncated file counts as missing)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            json.load(fh)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------------------------------------------- architectures
ARCHS = {
    # name: (kind, width, depth, normalise inputs to unit length like the pool does)
    "E128": ("expert", 128, 1, True),        # the shipped GrowMoE expert (residual block, hidden 128): 35,084 parameters
    "E512": ("expert", 512, 1, True),        # the tuned GrowMoE expert (hidden 512): 133,772 parameters
    "S256x2": ("mlp", 256, 2, False),        # MLP-S of the benchmark: 101,900 parameters
    "S256x2n": ("mlp", 256, 2, True),        # the same network on unit-length inputs (control: does the normalisation matter?)
    "L1024x3": ("mlp", 1024, 3, False),      # MLP-L of the benchmark: 2,243,596 parameters
}


def build_net(arch: str, seed: int) -> nn.Module:
    kind, width, depth, _ = ARCHS[arch]
    if kind == "expert":
        return MLPExpert(d_in=A.D_MODEL, hidden=width, depth=depth, n_classes=A.N_CLASS, seed=seed * 1000)
    torch.manual_seed(seed)
    return L._mlp(width, depth)


def unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)).astype(np.float32)


class PlainNet(nn.Module):
    """Gives an ordinary network the two methods ``BasePool.train_expert`` needs; the step is the one ``MLPExpert.train_step`` takes (no class mask)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def mark_seen(self, y):
        pass

    def train_step(self, u, y, opt, replay=None, project=False):
        self.train()
        y = y.long()
        if replay is not None:
            x_all, y_all = torch.cat([u, replay[0]]), torch.cat([y, replay[1].long()])
        else:
            x_all, y_all = u, y
        logits = self(x_all)
        loss = F.cross_entropy(logits, y_all)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(F.cross_entropy(logits[: u.shape[0]], y).detach())


class NetLearner(L.Learner):
    """Common prediction for a single trained network."""
    net = None
    arch = ""
    device = DEV

    def predict(self, ps):
        x = ps.x()
        if ARCHS[self.arch][3]:
            x = unit(x)
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 65536):
                out.append(self.net(torch.from_numpy(np.ascontiguousarray(x[i:i + 65536])).float().to(self.device)).argmax(-1).cpu().numpy())
        return {"pred": np.concatenate(out)}

    def train_loss(self, tr: A.ProblemSet, n: int = 30000) -> float:
        """Cross-entropy of the trained network on a random sample of its own training examples (how well it FITS what it was shown)."""
        x, y = tr.x(), tr.y
        idx = np.random.default_rng(1).choice(len(x), size=min(n, len(x)), replace=False)
        xs = unit(x[idx]) if ARCHS[self.arch][3] else x[idx].astype(np.float32)
        self.net.eval()
        with torch.no_grad():
            lg = self.net(torch.from_numpy(xs).to(self.device))
            return float(F.cross_entropy(lg, torch.from_numpy(y[idx]).long().to(self.device)))


class Offline(NetLearner):
    """The dense recipe of the benchmark: shuffled mini-batches, many epochs, AdamW with a one-cycle cosine schedule (``L._train_net``)."""

    def __init__(self, arch: str, epochs: int, lr: float = 2e-3, bs: int = 512):
        self.arch, self.epochs, self.lr, self.bs = arch, epochs, lr, bs
        self.name = f"{arch}|dense-{epochs}ep"

    def fit(self, tr, seed):
        net = build_net(self.arch, seed)
        net.to(self.device)
        x, y = tr.x(), tr.y
        if ARCHS[self.arch][3]:
            x = unit(x)
        if hasattr(net, "mark_seen"):
            net.mark_seen(torch.from_numpy(np.unique(y)))
        torch.manual_seed(seed)
        t0 = time.time()
        L._train_net(net, torch.from_numpy(x).float().to(self.device), torch.from_numpy(y).long().to(self.device), self.epochs, self.lr, self.bs)
        self.net = net
        return {"seconds": time.time() - t0, "params": sum(p.numel() for p in net.parameters()), "stored_vectors": 0, "train_loss": self.train_loss(tr),
                "example_gradients": int(self.epochs * len(x))}


class OfflineSubset(Offline):
    """The dense recipe on a random subset of ``n_keep`` training examples: what an offline learner can do with the SAME number of stored examples as a pool whose replay buffer holds ``n_keep``."""

    def __init__(self, arch: str, n_keep: int, steps: int = 23000, lr: float = 2e-3, bs: int = 512):
        epochs = max(1, math.ceil(steps / math.ceil(n_keep / bs)))                  # as many optimiser steps as the full-data dense recipe takes (about 23,000), whatever the subset size
        super().__init__(arch, epochs, lr, bs)
        self.n_keep, self.steps = n_keep, steps
        self.name = f"{arch}|dense-{epochs}ep-on-{n_keep}"

    def fit(self, tr, seed):
        keep = np.sort(np.random.default_rng([seed, 0x5B5]).choice(len(tr.y), size=min(self.n_keep, len(tr.y)), replace=False))

        class _View:                                                  # exposes just the examples we keep
            def x(_, idx=None):
                return tr.x()[keep]

            y = tr.y[keep]

        info = super().fit(_View(), seed)
        info["train_loss"] = self.train_loss(tr)
        return info


class Online(NetLearner):
    """The streaming recipe of the pool, without the pool: batches of 64 in random order, one pass, ``steps`` optimiser steps per batch on the SAME batch mixed with an
    equal number of samples drawn from a reservoir of ``replay`` stored examples, Adam at a constant 3e-3 (``BasePool.train_expert``, the code the pool calls).
    ``anneal`` decays the rate linearly to zero over the stream (a test of what the missing schedule costs). Runs on the CPU, like the pool."""
    device = torch.device("cpu")

    def __init__(self, arch: str, steps: int, replay: int, lr: float = 3e-3, anneal: bool = False, bs: int = 64):
        self.arch, self.steps, self.replay, self.lr, self.anneal, self.bs = arch, steps, replay, lr, anneal, bs
        self.name = f"{arch}|online-s{steps}-r{replay}{'-anneal' if anneal else ''}{'' if bs == 64 else f'-b{bs}'}"

    def fit(self, tr, seed):
        torch.set_num_threads(3)
        cfg = CommonCfg(d=A.D_MODEL, n_classes=A.N_CLASS, lr=self.lr, inner_steps=self.steps, replay_size=self.replay, seed=seed)
        pool = BasePool(cfg)
        net = build_net(self.arch, seed)
        if not hasattr(net, "train_step"):
            net = PlainNet(net)
        eid = pool.add_expert(net, parent=None, state="active").id
        x, y = tr.x(), tr.y
        x = unit(x) if ARCHS[self.arch][3] else x.astype(np.float32)
        X, Y = torch.from_numpy(x), torch.from_numpy(y).long()
        order = np.random.default_rng([seed, 0x5F1]).permutation(len(x))                  # the order the pool's fit uses
        nb = math.ceil(len(order) / self.bs)
        hist, t0 = [], time.time()
        for k, i in enumerate(range(0, len(order), self.bs)):
            b = torch.from_numpy(order[i:i + self.bs])
            if self.anneal:
                for g in pool.experts[eid].opt.param_groups:
                    g["lr"] = self.lr * (1.0 - k / nb)
            loss = pool.train_expert(eid, X[b], Y[b], replay=True)
            hist.append(loss)
        self.net = pool.experts[eid].module
        curve = [float(np.mean(hist[j:j + 100])) for j in range(0, len(hist), 100)]
        return {"seconds": time.time() - t0, "params": sum(p.numel() for p in net.parameters()), "stored_vectors": int(pool.experts[eid].replay.n), "train_loss": self.train_loss(tr),
                "example_gradients": int(nb * self.steps * 2 * self.bs), "loss_curve_per_100_batches": curve}


# ---------------------------------------------------------------------------------------------------------------- summaries
def summarise(res: dict) -> dict:
    ex = np.array(res["grid_exact"])
    out = {"mean_exact": float(ex.mean()), "exact_by_op": {A.OPS[i]: float(ex[i].mean()) for i in range(A.N_OPS)},
           "seen_exact": float(np.mean(res["seen_exact_by_op"])), "digit_exact": float(np.array(res["grid_digit"]).mean())}
    return out
