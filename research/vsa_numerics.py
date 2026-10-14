"""
vsa_numerics.py - reproducible numerical checks of the geometric claims used in
research/VSA_GEOMETRY_TO_INTELLIGENCE.md.

Pure numpy, CPU, seeded.  Run:  python research/vsa_numerics.py [--quick] [--seed 0]
Writes research/results/vsa_numerics.json.

E1  quasi-orthogonality of N random unit vectors vs dimension D   (concentration of measure)
E2  bundling (superposition) capacity: recover M members of a sum from a codebook of N
E3  binding schemes inside a role-filler record, incl. the exact scheme used by
    ishmn.neuro.global_workspace.HyperspaceGlobalBus (real Hadamard, unit-L2 gaussian keys)
E4  fractional power encoding: homomorphism phi(x)*phi(y)=phi(x+y) and kernel = Fourier transform of freq. law
E5  resonator network factorisation of a 3-factor product
E6  nearest-pattern retrieval on S^{D-1} under angular noise (what ISHMN's D=8 lattice can/can't do)
"""
import argparse
import json
import math
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
Q = args.quick
RES = {}


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def hdr(s):
    print("\n" + "=" * 92 + "\n" + s + "\n" + "=" * 92, flush=True)


# --------------------------------------------------------------------------------------------
def e1():
    hdr("E1  Quasi-orthogonality: N=5295 random unit vectors on S^{D-1} (5295 = ISHMN concept count)")
    N = 5295
    rows = []
    print(f"{'D':>6} {'mean|cos|':>10} {'theory':>8} {'max|cos|':>9} {'theory':>8} {'min angle':>10}")
    for D in (8, 16, 32, 64, 128, 256, 1024, 4096, 10000):
        X = unit(rng.standard_normal((N, D), dtype=np.float32))
        G = X @ X.T
        np.fill_diagonal(G, 0.0)
        A = np.abs(G)
        mean_abs = float(A.sum() / (N * (N - 1)))
        mx = float(A.max())
        th_mean = math.sqrt(2 / (math.pi * D))
        th_max = min(1.0, math.sqrt(2 * math.log(N * (N - 1) / 2) / D))
        ang = math.degrees(math.acos(min(1.0, mx)))
        rows.append(dict(D=D, mean_abs=mean_abs, theory_mean=th_mean, max_abs=mx, theory_max=th_max, min_angle_deg=ang))
        print(f"{D:>6} {mean_abs:>10.4f} {th_mean:>8.4f} {mx:>9.4f} {th_max:>8.4f} {ang:>9.1f}d", flush=True)
        del X, G, A
    RES["E1"] = rows


# --------------------------------------------------------------------------------------------
def e2():
    hdr("E2  Bundling capacity (bipolar MAP): exact recovery of all M members of sum(M items) from codebook N=5295")
    N = 5295
    trials = 6 if Q else 20
    Ms = [4, 8, 16, 32, 64, 128, 256, 512]
    out = {}
    print(f"{'D':>6} | " + " ".join(f"M={m:<4}" for m in Ms) + "   <- P(exact recovery of the whole set)")
    for D in (256, 1024, 4096, 10000):
        C = rng.choice(np.float32([-1, 1]), size=(N, D))
        row = []
        for M in Ms:
            exact = 0
            for _ in range(trials):
                idx = rng.choice(N, M, replace=False)
                b = C[idx].sum(0)
                s = C @ b
                top = np.argpartition(-s, M - 1)[:M]
                exact += set(top.tolist()) == set(idx.tolist())
            row.append(exact / trials)
        out[D] = dict(zip(Ms, row))
        print(f"{D:>6} | " + " ".join(f"{p:<6.2f}" for p in row), flush=True)
    # empirical capacity = largest M with P>=0.95 ; compare with D / (M ln N)
    cap = {}
    for D, d in out.items():
        ok = [m for m, p in d.items() if p >= 0.95]
        cap[D] = max(ok) if ok else 0
    print("capacity (largest tested M with P>=0.95):", cap)
    print("D / (M ln N) at capacity:", {D: (round(D / (m * math.log(N)), 2) if m else None) for D, m in cap.items()})
    RES["E2"] = dict(p_exact={str(k): v for k, v in out.items()}, capacity={str(k): v for k, v in cap.items()})


# --------------------------------------------------------------------------------------------
class Scheme:
    def __init__(self, name, D):
        self.name, self.D = name, D

    def gen(self, n):
        D = self.D
        if self.name == "MAP":
            return rng.choice(np.float32([-1, 1]), size=(n, D))
        if self.name == "FHRR":
            return np.exp(1j * rng.uniform(0, 2 * np.pi, (n, D))).astype(np.complex64)
        if self.name == "HRR":
            return (rng.standard_normal((n, D)) / math.sqrt(D)).astype(np.float32)
        if self.name == "BUS":  # HyperspaceGlobalBus: F.normalize(torch.randn(..., d_hyper), dim=-1)
            return unit(rng.standard_normal((n, D))).astype(np.float32)
        raise ValueError

    def bind(self, a, b):
        if self.name == "HRR":
            return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real
        return a * b

    def unbind(self, r, a):
        if self.name == "FHRR":
            return r * np.conj(a)
        if self.name == "HRR":
            inv = np.roll(a[::-1], 1)  # involution a*[i] = a[-i mod D]
            return self.bind(r, inv)
        return r * a  # MAP: self-inverse.  BUS: multiplies by key AGAIN (key^2, not 1)

    def sim(self, X, y):
        if self.name == "FHRR":
            return (np.conj(X) @ y).real
        return X @ y


def e3():
    hdr("E3  Role-filler record: P(correct filler | unbind role, cleanup over 1000 fillers).  BUS = ISHMN HyperspaceGlobalBus scheme")
    Fn = 1000
    Ks = [1, 2, 4, 8, 16]
    trials = 100 if Q else 300
    out = {}
    print(f"{'scheme':>6} {'D':>5} | " + " ".join(f"K={k:<3}" for k in Ks))
    for name in ("MAP", "FHRR", "HRR", "BUS"):
        for D in (64, 256, 1024):
            sch = Scheme(name, D)
            Fcb = sch.gen(Fn)
            row = []
            for K in Ks:
                ok = tot = 0
                for _ in range(trials):
                    roles = sch.gen(K)
                    fi = rng.choice(Fn, K, replace=False)
                    rec = sum(sch.bind(roles[k], Fcb[fi[k]]) for k in range(K))
                    for k in range(K):
                        est = sch.unbind(rec, roles[k])
                        ok += int(np.argmax(sch.sim(Fcb, est)) == fi[k])
                        tot += 1
                row.append(ok / tot)
            out[f"{name}-{D}"] = dict(zip(Ks, row))
            print(f"{name:>6} {D:>5} | " + " ".join(f"{p:<5.2f}" for p in row), flush=True)
    RES["E3_record"] = out

    # isometry: does binding preserve the geometry (cosine, norm)?
    print("\nIsometry check at D=1024: mean |cos(bind(a,x),bind(a,y)) - cos(x,y)| and mean ||bind(a,x)||/||x||")
    iso = {}
    for name in ("MAP", "FHRR", "HRR", "BUS"):
        sch = Scheme(name, 1024)
        dev, nr = [], []
        for _ in range(200):
            a = sch.gen(1)[0]
            xy = sch.gen(2)
            x = xy[0]
            y = 0.6 * xy[0] + 0.8 * xy[1] if name != "FHRR" else xy[0]  # correlated pair
            if name == "FHRR":  # correlated phasors: keep half the phases
                y = xy[0].copy()
                m = rng.random(1024) < 0.5
                y[m] = xy[1][m]

            def cs(u, v):
                return float((np.conj(u) @ v).real / (np.linalg.norm(u) * np.linalg.norm(v)))

            dev.append(abs(cs(sch.bind(a, x), sch.bind(a, y)) - cs(x, y)))
            nr.append(float(np.linalg.norm(sch.bind(a, x)) / np.linalg.norm(x)))
        iso[name] = dict(mean_cos_dev=float(np.mean(dev)), mean_norm_ratio=float(np.mean(nr)), std_norm_ratio=float(np.std(nr)))
        print(f"  {name:>5}: cos deviation {np.mean(dev):.5f}   norm ratio {np.mean(nr):.4f} +/- {np.std(nr):.4f}")
    RES["E3_isometry"] = iso


# --------------------------------------------------------------------------------------------
def e4():
    hdr("E4  Fractional power encoding: phi(x)_j = exp(i w_j x).  Bind = add the encoded numbers; similarity = Fourier transform of w-law")
    D = 4096
    w = rng.uniform(-np.pi, np.pi, D)
    x, y = 0.37, 1.21
    err = float(np.abs(np.exp(1j * w * x) * np.exp(1j * w * y) - np.exp(1j * w * (x + y))).max())
    print(f"homomorphism  phi(x)*phi(y) vs phi(x+y): max abs err = {err:.2e}")
    RES["E4_homomorphism_err"] = err
    a, sig = 2.0, 1.5
    deltas = np.linspace(0.0, 6.0, 61)
    rows = []
    print(f"{'D':>6} {'sinc-kernel err':>16} {'gauss-kernel err':>17}   (max over delta in [0,6], mean of 10 draws)")
    for D in (128, 512, 2048, 8192, 32768):
        e_s, e_g = [], []
        for _ in range(10 if not Q else 4):
            ws = rng.uniform(-a, a, D)
            wg = rng.normal(0, sig, D)
            ks = np.array([np.cos(ws * d).mean() for d in deltas])
            kg = np.array([np.cos(wg * d).mean() for d in deltas])
            ts = np.where(deltas == 0, 1.0, np.sin(a * deltas) / np.maximum(a * deltas, 1e-12))
            tg = np.exp(-(sig * deltas) ** 2 / 2)
            e_s.append(np.abs(ks - ts).max())
            e_g.append(np.abs(kg - tg).max())
        rows.append(dict(D=D, sinc_err=float(np.mean(e_s)), gauss_err=float(np.mean(e_g))))
        print(f"{D:>6} {np.mean(e_s):>16.4f} {np.mean(e_g):>17.4f}", flush=True)
    sl = np.polyfit(np.log([r["D"] for r in rows]), np.log([r["sinc_err"] for r in rows]), 1)[0]
    print(f"log-log slope of error vs D = {sl:.2f}   (random-features theory: -0.5)")
    RES["E4_kernel"] = dict(rows=rows, slope=float(sl))


# --------------------------------------------------------------------------------------------
def resonator_trial(D, n, F=3, maxit=60):
    books = [rng.choice(np.float32([-1, 1]), size=(n, D)) for _ in range(F)]
    truth = [int(rng.integers(n)) for _ in range(F)]
    s = np.prod([books[f][truth[f]] for f in range(F)], axis=0)
    est = []
    for b in books:
        e = np.sign(b.sum(0))
        e[e == 0] = 1
        est.append(e)
    it = 0
    for it in range(1, maxit + 1):
        changed = False
        for f in range(F):
            u = s.copy()
            for g in range(F):
                if g != f:
                    u = u * est[g]
            a = books[f] @ u
            new = np.sign(books[f].T @ a)
            new[new == 0] = 1
            changed |= not np.array_equal(new, est[f])
            est[f] = new
        if not changed:
            break
    ok = all(int(np.argmax(books[f] @ est[f])) == truth[f] for f in range(F))
    return ok, it


def e5():
    hdr("E5  Resonator network: factor s = a*b*c (bipolar), each factor from a codebook of n.  Search space = n^3")
    trials = 20 if Q else 60
    out = {}
    print(f"{'n':>4} {'n^3':>9} | " + " ".join(f"D={d:<5}" for d in (256, 512, 1024, 2048, 4096)) + "   (P(success), mean iters)")
    for n in (10, 30, 100):
        row = []
        for D in (256, 512, 1024, 2048, 4096):
            oks, its = 0, []
            for _ in range(trials):
                ok, it = resonator_trial(D, n)
                oks += ok
                its.append(it)
            row.append((oks / trials, float(np.mean(its))))
            out[f"n{n}-D{D}"] = dict(p=oks / trials, iters=float(np.mean(its)))
        print(f"{n:>4} {n**3:>9} | " + " ".join(f"{p:.2f}/{i:<4.1f}" for p, i in row), flush=True)
    RES["E5"] = out


# --------------------------------------------------------------------------------------------
def e6():
    hdr("E6  Nearest-pattern retrieval on S^{D-1}: N=5295 random patterns, query = pattern rotated by angle theta in a random direction")
    N = 5295
    nq = 400 if not Q else 100
    angs = (10, 20, 40, 60)
    out = {}
    print(f"{'D':>6} | " + " ".join(f"th={a:<3}d" for a in angs) + "   median angle to NEAREST OTHER pattern")
    for D in (8, 16, 32, 64, 128, 512, 2048):
        P = unit(rng.standard_normal((N, D), dtype=np.float32))
        idx = rng.integers(N, size=nq)
        accs = []
        for ang in angs:
            th = math.radians(ang)
            eps = rng.standard_normal((nq, D), dtype=np.float32)
            eps -= (eps * P[idx]).sum(1, keepdims=True) * P[idx]
            eps = unit(eps)
            q = math.cos(th) * P[idx] + math.sin(th) * eps
            accs.append(float((np.argmax(q @ P.T, 1) == idx).mean()))
        sub = P[:1500]
        G = sub @ sub.T
        np.fill_diagonal(G, -1)
        nn = float(np.median(np.degrees(np.arccos(np.clip(G.max(1), -1, 1)))))
        out[D] = dict(acc=dict(zip(angs, accs)), nn_angle_deg_1500=nn)
        print(f"{D:>6} | " + " ".join(f"{a:<7.2f}" for a in accs) + f"   {nn:.1f} deg (1500 patterns)", flush=True)
    RES["E6"] = {str(k): v for k, v in out.items()}


if __name__ == "__main__":
    t0 = time.time()
    for fn in (e1, e2, e3, e4, e5, e6):
        t = time.time()
        fn()
        print(f"  [{fn.__name__} {time.time() - t:.1f}s]", flush=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results", "vsa_numerics.json"), "w") as f:
        json.dump(RES, f, indent=1)
    print(f"\nTotal {time.time() - t0:.1f}s.  Wrote research/results/vsa_numerics.json")
