"""Study 5d: why does the growth policy miss a capacity shortage? Probe length against measured gain.

The policy trains a control and each widened candidate for 480 steps and asks for a held-out loss gain of at least 2 %. Here a capacity-starved expert (16 hidden units) is trained on the stream
for 900 batches with the pool's recipe; then, on the SAME state, control and candidates are trained for 480 / 1,000 / 2,000 / 5,000 steps (identical mini-batches, fresh Adam, like the policy's
probe) and compared on 4,000 held-out examples of the stream. How large is the gain the policy would see, and how large does it become when the probe is allowed to run as long as the
growth would run in reality?

    python probe_horizon.py --seed 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import diag_lib as D  # noqa: E402
from diag_lib import A, L  # noqa: E402
from growmoe.config import CommonCfg  # noqa: E402
from growmoe.pools.base import BasePool  # noqa: E402

OUT = HERE / "horizon"


def train_probe(model, x, y, batches, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for idx in batches:
        model.train_step(x[idx], y[idx], opt)
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm", type=int, default=900)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    stem = OUT / f"s{a.seed}"
    if stem.with_suffix(".json").exists():
        return
    torch.set_num_threads(3)
    tr = L.build_train("FULL", a.seed, 8000)
    x, y = D.unit(tr.x()), tr.y
    X, Y = torch.from_numpy(x), torch.from_numpy(y).long()
    order = np.random.default_rng([a.seed, 0x5F1]).permutation(len(x))
    cfg = CommonCfg(d=A.D_MODEL, n_classes=A.N_CLASS, lr=3e-3, inner_steps=5, replay_size=20000, seed=a.seed)
    pool = BasePool(cfg)
    D.ARCHS["_tmp"] = ("expert", 16, 1, True)
    eid = pool.add_expert(D.build_net("_tmp", a.seed), parent=None, state="active").id
    for k, i in enumerate(range(0, a.warm * 64, 64)):
        b = torch.from_numpy(order[i:i + 64])
        pool.train_expert(eid, X[b], Y[b], replay=True)
    expert = pool.experts[eid].module
    rx, ry = pool.experts[eid].replay.data
    n_tr = min(16000, rx.shape[0])
    x_tr, y_tr = rx[:n_tr].clone(), ry[:n_tr].clone()
    held = torch.from_numpy(order[a.warm * 64:a.warm * 64 + 4000])           # examples the stream has not shown yet
    x_va, y_va = X[held], Y[held]
    expert.eval()
    with torch.no_grad():
        status = F.cross_entropy(expert(x_va), y_va, reduction="none").numpy().astype(np.float64)
    print(f"after {a.warm} batches the 16-wide expert has held-out CE {status.mean():.3f}; probing on {n_tr} stored and {len(held)} held-out examples", flush=True)
    out = {"seed": a.seed, "warm_batches": a.warm, "status_ce": float(status.mean()), "rows": []}
    rng = np.random.default_rng(a.seed)
    for steps in (480, 1000, 2000, 5000):
        batches = [torch.from_numpy(rng.integers(0, n_tr, size=64)) for _ in range(steps)]
        t0 = time.time()
        control = expert.clone()
        train_probe(control, x_tr, y_tr, batches)
        with torch.no_grad():
            c = F.cross_entropy(control(x_va), y_va, reduction="none").numpy().astype(np.float64)
        row = {"steps": steps, "control_ce": float(c.mean()), "candidates": {}}
        for k in (8, 48, 112, 496):
            shadow = expert.clone().widen(k, np.random.default_rng(1000 + k))
            train_probe(shadow, x_tr, y_tr, batches)
            with torch.no_grad():
                s = F.cross_entropy(shadow(x_va), y_va, reduction="none").numpy().astype(np.float64)
            delta = c - s
            idx = np.random.default_rng(7).integers(0, len(delta), size=(1000, len(delta)))
            p = float((1 + (delta[idx].mean(axis=1) <= 0).sum()) / 1001)
            row["candidates"][f"widen+{k}"] = {"gain": float(delta.mean() / c.mean()), "p": p, "ce": float(s.mean()), "acc": float((shadow(x_va).argmax(1) == y_va).float().mean()),
                                              "control_acc": float((control(x_va).argmax(1) == y_va).float().mean())}
        out["rows"].append(row)
        print(f"probe of {steps} steps: control CE {c.mean():.3f}  " + "  ".join(f"+{k}: gain {100 * v['gain']:+.1f}% (p {v['p']:.3f})" for k, v in ((k.split('+')[1], v) for k, v in row['candidates'].items())) + f"   ({time.time() - t0:.0f}s)", flush=True)
    D.atomic_write_text(stem.with_suffix(".json"), json.dumps(out))


if __name__ == "__main__":
    main()
