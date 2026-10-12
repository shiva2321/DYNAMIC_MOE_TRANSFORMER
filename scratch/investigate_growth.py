import sys
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
from growmoe.config import GrowthCfg
from growmoe.experts import MLPExpert
from growmoe.stack import StackGrowthPolicy

def units(n, d, seed):
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, d))
    return torch.from_numpy(x / np.linalg.norm(x, axis=1, keepdims=True)).float()

from growmoe.streams import s5_ladder

def test_s5_ladder(sd=100, width=32, depth_t=3, n_train=10000):
    stream = s5_ladder(seed=sd, width=width, depth=depth_t, d=128, n_train=n_train, n_test=1000)
    seg = stream.segments[0]
    x_tr, y_tr = seg.x, seg.y
    reg = stream.regimes[0]
    x_va, y_va = reg.x_te, reg.y_te
    c = 8
    d = 128
    
    print(f"\n--- S5 Ladder: Teacher width={width}, depth={depth_t}, n_train={n_train} ---")
    rng = np.random.default_rng(sd)
    for s_dep in [1, 2, 3]:
        s = MLPExpert(d_in=d, hidden=32, depth=s_dep, n_classes=c, seed=sd * 10 + 6)
        s.mark_seen(torch.arange(c))
        opt = torch.optim.Adam(s.parameters(), lr=3e-3)
        for _ in range(1000):
            idx = torch.from_numpy(rng.integers(0, len(x_tr), 64))
            s.train_step(x_tr[idx], y_tr[idx], opt)
        with torch.no_grad():
            loss = F.cross_entropy(s(x_va), y_va).item()
            acc = (s(x_va).argmax(1) == y_va).float().mean().item()
            print(f"Student depth {s_dep} (1000 steps): val loss = {loss:.4f}, acc = {acc:.3f}")

if __name__ == "__main__":
    test_s5_ladder(100, width=32, depth_t=3, n_train=10000)


