"""Spike N (PILOT, not pre-registered): a task family where memorising fails. Labels are a hidden function (a random teacher
network) of random unit vectors in 128 dimensions. Same training data for both: kNN lookup vs a small trained network.
Throwaway diligence script. Not the repo's S5/S6 evaluation."""
import time

from dil_common import *  # noqa

from growmoe.experts import MLPExpert
from growmoe.streams import _balanced_teacher

D, C = 128, 8
t0 = time.time()
rows = {}
for (tw, td) in ((8, 1), (32, 2)):
    for n_train in (300, 1000, 3000, 10000):
        for sd in range(5):
            g = np.random.default_rng([sd, tw, td, 0x4C46])
            z = g.standard_normal((n_train + 2000, D))
            X = torch.from_numpy((z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32))
            teacher = _balanced_teacher(tw, td, D, C, seed=sd)
            with torch.no_grad():
                y = teacher(X).argmax(1)
            Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
            # kNN (cosine)
            sims = Xte @ Xtr.T
            idx = sims.topk(5, dim=1).indices
            votes = torch.zeros(Xte.shape[0], C)
            votes.scatter_add_(1, ytr[idx], torch.ones_like(idx, dtype=torch.float32))
            acc_knn = float((votes.argmax(1) == yte).float().mean())
            # small trained net, same data, fixed budget (no tuning on test)
            m = MLPExpert(D, 64, 1, C, seed=sd)
            m.mark_seen(torch.arange(C))
            opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            gen = np.random.default_rng([sd, 7])
            for _ in range(1500):
                b = torch.from_numpy(gen.integers(0, n_train, 64))
                m.train_step(Xtr[b], ytr[b], opt)
            m.eval()
            with torch.no_grad():
                acc_mlp = float((m(Xte).argmax(1) == yte).float().mean())
            rows.setdefault((tw, td, n_train), []).append((acc_knn, acc_mlp))
    print(f"teacher w{tw} d{td} done ({time.time()-t0:.0f}s)", flush=True)

print("\n=== Hidden-function labels (8 classes, chance 12.5%), accuracy % on 2000 fresh test points, 5 seeds, mean [95% CI] ===")
print(f"{'teacher':12s} {'train N':>8s} {'kNN lookup':>22s} {'small trained net':>22s} {'net minus kNN':>16s}")
for (tw, td, n), r in rows.items():
    a = [x[0] for x in r]
    b = [x[1] for x in r]
    d = [y - x for x, y in zip(a, b)]
    print(f"w{tw:<2d} depth {td:<2d}  {n:8d} {fmt(a):>22s} {fmt(b):>22s} {np.mean(d)*100:+15.1f}")
