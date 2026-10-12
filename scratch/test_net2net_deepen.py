import sys
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
from growmoe.experts import MLPExpert

def units(n, d, seed):
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, d))
    return torch.from_numpy(x / np.linalg.norm(x, axis=1, keepdims=True)).float()

def test_deepen_continuation(sd=100, d=32, c=4, n_tr=2000, n_va=500):
    rng = np.random.default_rng(sd)
    x_tr = units(n_tr, d, sd * 10 + 1)
    x_va = units(n_va, d, sd * 10 + 2)

    # Teacher: 3 layers, non-linear
    t = MLPExpert(d_in=d, hidden=32, depth=3, n_classes=c, seed=sd * 10 + 3)
    t.mark_seen(torch.arange(c))
    with torch.no_grad():
        y_tr = t(x_tr).argmax(dim=1)
        y_va = t(x_va).argmax(dim=1)

    # Train depth-1 student to convergence (e.g. 300 steps)
    s = MLPExpert(d_in=d, hidden=32, depth=1, n_classes=c, seed=sd * 10 + 4)
    s.mark_seen(torch.arange(c))
    opt = torch.optim.Adam(s.parameters(), lr=3e-3)
    for _ in range(300):
        idx = torch.from_numpy(rng.integers(0, len(x_tr), 64))
        s.train_step(x_tr[idx], y_tr[idx], opt)

    with torch.no_grad():
        loss_base = F.cross_entropy(s(x_va), y_va).item()
        print(f"Seed {sd}: Depth-1 student at step 300: val loss = {loss_base:.4f}")

    # Arm 1: Control (continue training depth-1 for 100 more steps)
    s_ctrl = s.clone()
    opt_ctrl = torch.optim.Adam(s_ctrl.parameters(), lr=3e-3)
    for _ in range(100):
        idx = torch.from_numpy(rng.integers(0, len(x_tr), 64))
        s_ctrl.train_step(x_tr[idx], y_tr[idx], opt_ctrl)
    with torch.no_grad():
        loss_ctrl = F.cross_entropy(s_ctrl(x_va), y_va).item()

    # Arm 2: Deepened (deepen, then train for 100 more steps)
    s_deep = s.clone()
    s_deep.deepen(np.random.default_rng(sd * 10 + 5))
    opt_deep = torch.optim.Adam(s_deep.parameters(), lr=3e-3)
    for _ in range(100):
        idx = torch.from_numpy(rng.integers(0, len(x_tr), 64))
        s_deep.train_step(x_tr[idx], y_tr[idx], opt_deep)
    with torch.no_grad():
        loss_deep = F.cross_entropy(s_deep(x_va), y_va).item()

    # Arm 3: Deepened with 300 steps
    s_deep300 = s.clone()
    s_deep300.deepen(np.random.default_rng(sd * 10 + 5))
    opt_deep300 = torch.optim.Adam(s_deep300.parameters(), lr=3e-3)
    for _ in range(300):
        idx = torch.from_numpy(rng.integers(0, len(x_tr), 64))
        s_deep300.train_step(x_tr[idx], y_tr[idx], opt_deep300)
    with torch.no_grad():
        loss_deep300 = F.cross_entropy(s_deep300(x_va), y_va).item()

    print(f"   Control (+100 steps): val loss = {loss_ctrl:.4f}")
    print(f"   Deepened (+100 steps): val loss = {loss_deep:.4f} (diff = {loss_ctrl - loss_deep:+.4f})")
    print(f"   Deepened (+300 steps): val loss = {loss_deep300:.4f} (diff = {loss_ctrl - loss_deep300:+.4f})")

if __name__ == "__main__":
    for s in [100, 101, 102, 103, 104]:
        test_deepen_continuation(s)
