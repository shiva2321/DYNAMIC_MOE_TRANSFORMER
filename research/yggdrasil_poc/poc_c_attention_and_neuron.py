"""PoC-C: (1) one linear layer on the hyperboloid contains three kinds of neuron; (2) Yggdrasil attention is ONE fused dot
product; (3) how much numerical precision does it need; (4) what does it cost compared with the current approach.

Conventions: hyperboloid L = {x in R^(D+1): <x,x>_L = -1/k, x0 > 0}, <x,y>_L = -x0 y0 + x_s . y_s, curvature -k.
A point is stored by its space-like part x_s; the time part is DERIVED: x0 = sqrt(1/k + |x_s|^2).

Run: python research/yggdrasil_poc/poc_c_attention_and_neuron.py        (small GPU tensors, < 1.5 GB peak, ~1 minute)
"""

import math
import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)


# ------------------------------------------------------------------------------------------------ helpers
def lift(xs, k=1.0):
    """space-like part -> full hyperboloid point (x0, x_s)."""
    return torch.cat([torch.sqrt(1.0 / k + (xs * xs).sum(-1, keepdim=True)), xs], -1)


def mink(x, y):
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(-1)


def poincare_to_lorentz(u):
    n2 = (u * u).sum(-1, keepdim=True)
    return torch.cat([(1 + n2) / (1 - n2), 2 * u / (1 - n2)], -1)


def poincare_distance(u, v):
    uu, vv = (u * u).sum(-1), (v * v).sum(-1)
    return torch.acosh(1 + 2 * ((u - v) ** 2).sum(-1) / ((1 - uu) * (1 - vv)))


# -------------------------------------------------------------- 1. the three neuron types are ONE linear functional
def neuron_identities():
    print("=" * 100)
    print("1. One linear functional <w,x>_L on the hyperboloid = three neuron types, decided by the sign of <w,w>_L")
    print("=" * 100)
    g = torch.Generator().manual_seed(1)
    u = (torch.rand(2000, 2, generator=g, dtype=torch.float64) * 2 - 1)
    u = u / u.norm(dim=-1, keepdim=True) * (torch.rand(2000, 1, generator=g, dtype=torch.float64) ** 0.5) * 0.97
    x = poincare_to_lorentz(u)                                   # points of H^2 as hyperboloid vectors

    # (i) time-like w = p: distance-to-a-point neuron (the prototype / RBF neuron of the current memory cortex)
    pu = torch.tensor([0.3, -0.4], dtype=torch.float64)
    p = poincare_to_lorentz(pu)
    err = (mink(p, x) + torch.cosh(poincare_distance(pu.expand_as(u), u))).abs().max().item()
    print(f"(i)  time-like  : <p,x>_L = -cosh d(p,x)              max |error| over 2000 points = {err:.2e}")

    # (ii) space-like w, <w,w>=1: signed distance to a geodesic hyperplane, sinh(d) = <w,x>_L
    w = torch.tensor([0.4, 1.0, -0.7], dtype=torch.float64)
    w = w / torch.sqrt(mink(w, w))
    c = torch.tensor([1.0, 0, 0], dtype=torch.float64)
    a = c - mink(c, w) * w
    a = a / torch.sqrt(-mink(a, a))
    e = torch.tensor([0, 1.0, 0], dtype=torch.float64)
    b = e - mink(e, w) * w + mink(e, a) * a
    b = b / torch.sqrt(mink(b, b))
    s = torch.linspace(-12, 12, 80001, dtype=torch.float64)
    line = torch.cosh(s)[:, None] * a + torch.sinh(s)[:, None] * b          # the geodesic line {<w,y>_L = 0}
    sub = x[:300]
    d = torch.acosh((-(sub @ torch.diag(torch.tensor([-1.0, 1, 1], dtype=torch.float64)) @ line.T)).clamp(min=1.0)).min(dim=1).values
    sgn = torch.sign(mink(w.expand_as(sub), sub))
    err = (torch.sinh(sgn * d) - mink(w.expand_as(sub), sub)).abs().max().item()
    print(f"(ii) space-like : <w,x>_L = sinh(signed d(x, H_w))     max |error| over 300 points  = {err:.2e}  (distance found by brute-force search on the line)")

    # (iii) light-like w = (1, omega), |omega|=1: horosphere / Busemann function
    om = torch.tensor([math.cos(1.1), math.sin(1.1)], dtype=torch.float64)
    wl = torch.cat([torch.ones(1, dtype=torch.float64), om])
    busemann = torch.log(((om - u) ** 2).sum(-1) / (1 - (u * u).sum(-1)))
    err = (torch.log(-mink(wl.expand_as(x), x)) - busemann).abs().max().item()
    print(f"(iii) light-like: -<w,x>_L = exp(Busemann_omega(x))    max |error| of the log      = {err:.2e}  (Busemann computed in the Poincare ball)")
    print("-> a ReLU(<w,x>_L + b) neuron is a ball-around-a-point detector, a geodesic half-space, or a horosphere detector,")
    print("   depending only on whether its weight vector is time-like, space-like or light-like. The model can learn which.")


# ------------------------------------------------------------------------------ 2. attention = ONE dot product
def explicit_attention(qs, ks, vs, qth, kth, beta, k, tau, kappa, causal=True):
    """Reference: spell out every pair (T, T, D) like the pairwise-distance code does."""
    q0, k0 = lift(qs, k)[..., :1], lift(ks, k)[..., :1]
    sq_space = ((qs[..., :, None, :] - ks[..., None, :, :]) ** 2).sum(-1)
    dt = q0[..., :, None, 0] - k0[..., None, :, 0]
    sqdist = sq_space - dt ** 2                                           # squared Lorentzian distance = -2/k - 2<q,k>_L
    logit = -sqdist / (2 * tau) + kappa * torch.cos(qth[..., :, None, :] - kth[..., None, :, :]).sum(-1) + beta[..., None, :]
    if causal:
        T = qs.shape[-2]
        logit = logit.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=qs.device)), float("-inf"))
    out = torch.softmax(logit, -1) @ lift(vs, k)
    return out / torch.sqrt(k * (-mink(out, out)).clamp(min=1e-9))[..., None]          # Lorentz centroid: back onto the hyperboloid


def augmented(qs, ks, vs, qth, kth, beta, k, tau, kappa, pad_to=8):
    qa = torch.cat([-lift(qs, k)[..., :1], qs, math.sqrt(kappa * tau) * torch.cos(qth), math.sqrt(kappa * tau) * torch.sin(qth),
                    torch.ones_like(qs[..., :1])], -1)
    ka = torch.cat([lift(ks, k)[..., :1], ks, math.sqrt(kappa * tau) * torch.cos(kth), math.sqrt(kappa * tau) * torch.sin(kth),
                    tau * beta[..., None]], -1)
    va = lift(vs, k)
    da = qa.shape[-1]
    dpad = (-da) % pad_to
    qa, ka = F.pad(qa, (0, dpad)), F.pad(ka, (0, dpad))
    va = F.pad(va, (0, qa.shape[-1] - va.shape[-1]))
    return qa, ka, va


def fused_attention(qs, ks, vs, qth, kth, beta, k, tau, kappa, causal=True):
    qa, ka, va = augmented(qs, ks, vs, qth, kth, beta, k, tau, kappa)
    out = F.scaled_dot_product_attention(qa, ka, va, is_causal=causal, scale=1.0 / tau)[..., : vs.shape[-1] + 1]
    return out / torch.sqrt(k * (-mink(out, out)).clamp(min=1e-9))[..., None]


def attention_equivalence():
    print("\n" + "=" * 100)
    print("2. Yggdrasil attention (hyperbolic + torus phase + salience) as a single scaled dot product")
    print("=" * 100)
    for dtype, name in [(torch.float64, "float64"), (torch.float32, "float32")]:
        B, H, T, D, M, Dv, k, tau, kappa = 1, 2, 48, 16, 8, 16, 1.0, 0.7, 1.3
        g = torch.Generator().manual_seed(3)
        qs, ks, vs = (torch.randn(B, H, T, d, generator=g, dtype=dtype) * 0.8 for d in (D, D, Dv))
        qth, kth = (torch.rand(B, H, T, M, generator=g, dtype=dtype) * 2 * math.pi for _ in range(2))
        beta = torch.randn(B, H, T, generator=g, dtype=dtype)
        ref = explicit_attention(qs, ks, vs, qth, kth, beta, k, tau, kappa)
        fus = fused_attention(qs, ks, vs, qth, kth, beta, k, tau, kappa)
        closure = (mink(fus, fus) + 1.0 / k).abs().max().item()
        print(f"{name}: max |explicit pairwise - fused dot product| = {(ref - fus).abs().max().item():.2e};  "
              f"output stays on the hyperboloid: |<c,c>_L + 1/k| = {closure:.2e}")
    # rank-vs-weights remark: the Lorentz inner product is a monotone function of the geodesic distance but NOT the same kernel
    d = torch.linspace(0, 6, 7, dtype=torch.float64)
    print("squared-Lorentz logit  -(cosh d - 1)  vs  Gaussian-in-geodesic-distance logit  -d^2/2   for geodesic distance d = 0..6:")
    print("  -(cosh d - 1):", np.round(-(torch.cosh(d) - 1).numpy(), 1), "\n  -d^2/2       :", np.round(-(d ** 2 / 2).numpy(), 1))
    print("-> same ordering of neighbours, but the Lorentz form falls off exponentially in d (a much sharper, 'only the closest matter' kernel).")


# ----------------------------------------------------------------------- 3. precision needed (the cancellation)
def precision_vs_radius():
    print("\n" + "=" * 100)
    print("3. Precision: the logit <q,k>_L = -q0 k0 + q.k subtracts two numbers of size ~R^2 to get an O(1) answer")
    print("=" * 100)
    if not torch.cuda.is_available():
        print("no GPU: skipped")
        return
    D, Q, Kc, nclu = 32, 128, 64, 8
    print(f"{'R=|x_s|':>8s} {'hyp. radius':>11s} | " + " ".join(f"{n:>13s}" for n in ["fp32", "tf32", "fp16", "bf16"]) + "     (error in nats of the neighbour logits; <0.05 is safe)")
    for R in [0.5, 1, 2, 4, 8, 16, 32, 64]:
        g = torch.Generator().manual_seed(5)
        centre = F.normalize(torch.randn(Q, D, generator=g, dtype=torch.float64), dim=-1) * R
        delta = torch.randn(Q, nclu, D, generator=g, dtype=torch.float64)
        delta = delta - (delta * centre[:, None] / R).sum(-1, keepdim=True) * centre[:, None] / R           # tangential step
        delta = F.normalize(delta, dim=-1) * (0.3 + 0.7 * torch.rand(Q, nclu, 1, generator=g, dtype=torch.float64))
        keys = (centre[:, None] + delta).reshape(-1, D)
        qa64 = torch.cat([-lift(centre)[..., :1], centre], -1)
        ka64 = torch.cat([lift(keys)[..., :1], keys], -1)
        truth = (qa64 @ ka64.T).reshape(Q, Q, nclu)[torch.arange(Q), torch.arange(Q)]            # each query vs its own cluster
        row = []
        for name in ["fp32", "tf32", "fp16", "bf16"]:
            dt = {"fp32": torch.float32, "tf32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]
            torch.backends.cuda.matmul.allow_tf32 = name == "tf32"
            qa, ka = qa64.to("cuda", dt), ka64.to("cuda", dt)
            est = (qa @ ka.T).float().cpu().double().reshape(Q, Q, nclu)[torch.arange(Q), torch.arange(Q)]
            row.append((est - truth).abs().max().item() if torch.isfinite(est).all() else float("inf"))
        torch.backends.cuda.matmul.allow_tf32 = False
        print(f"{R:8.1f} {math.asinh(R):11.2f} | " + " ".join(f"{v:13.4f}" for v in row))
    print("-> with half-precision inputs the logit error grows like R^2: bf16 is unusable beyond R ~ 2 (hyperbolic radius ~1.4),")
    print("   fp16/tf32 beyond R ~ 8. Keep the manifold part of the QK product in fp32, or keep the radius small (local frames).")


# -------------------------------------------------------------------------------------- 4. speed and memory
def bench(fn, reps=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), (torch.cuda.max_memory_allocated() - base) / 1e6


def speed_and_memory():
    print("\n" + "=" * 100)
    print("4. Cost (forward + backward, causal, B=1, H=4, D=32 space dims + 8 phases): ms and extra peak MB")
    print("=" * 100)
    if not torch.cuda.is_available():
        print("no GPU: skipped")
        return
    B, H, D, M, k, tau, kappa = 1, 4, 32, 8, 1.0, 0.7, 1.0
    print(f"{'T':>5s} | {'naive (T,T,D) fp32':>20s} | {'manual matmul fp32':>20s} | {'fused SDPA fp32':>20s} | {'fused SDPA fp16':>20s} | {'plain Euclid SDPA fp16':>22s}")
    for T in [256, 512, 1024, 2048]:
        def make(dtype):
            g = torch.Generator(device="cuda").manual_seed(0)
            qs, ks, vs = (torch.randn(B, H, T, d, generator=g, device="cuda", dtype=dtype).mul_(0.8).requires_grad_(True) for d in (D, D, D))
            qth, kth = (torch.rand(B, H, T, M, generator=g, device="cuda", dtype=dtype) * 2 * math.pi for _ in range(2))
            return qs, ks, vs, qth, kth, torch.randn(B, H, T, generator=g, device="cuda", dtype=dtype)

        res = {}
        qs, ks, vs, qth, kth, beta = make(torch.float32)

        def run_naive():
            explicit_attention(qs, ks, vs, qth, kth, beta, k, tau, kappa).sum().backward()

        def run_manual():
            qa, ka, va = augmented(qs, ks, vs, qth, kth, beta, k, tau, kappa)
            logit = (qa @ ka.transpose(-1, -2)) / tau
            logit = logit.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device="cuda")), float("-inf"))
            (torch.softmax(logit, -1) @ va).sum().backward()

        def run_fused32():
            fused_attention(qs, ks, vs, qth, kth, beta, k, tau, kappa).sum().backward()

        res["naive"] = bench(run_naive) if T <= 512 else None
        res["manual"] = bench(run_manual)
        res["fused32"] = bench(run_fused32)
        qs16, ks16, vs16, qth16, kth16, beta16 = make(torch.float16)

        def run_fused16():
            fused_attention(qs16, ks16, vs16, qth16, kth16, beta16, k, tau, kappa).sum().backward()

        def run_plain():
            F.scaled_dot_product_attention(qs16, ks16, vs16, is_causal=True).sum().backward()

        res["fused16"] = bench(run_fused16)
        res["plain"] = bench(run_plain)
        cell = lambda r: "(skipped: memory)".rjust(20) if r is None else f"{r[0]:8.2f} ms {r[1]:7.0f} MB"
        print(f"{T:5d} | {cell(res['naive'])} | {cell(res['manual'])} | {cell(res['fused32'])} | {cell(res['fused16'])} | {cell(res['plain'])}")
        del qs, ks, vs, qs16, ks16, vs16
        torch.cuda.empty_cache()


def kernel_vs_gemm_floor():
    print("\n" + "=" * 100)
    print("5. The current Yggdrasil pairwise-distance kernel versus a GEMM (tensor-core style) formulation, 4096 x 4096 pairs")
    print("=" * 100)
    if not torch.cuda.is_available():
        print("no GPU: skipped")
        return
    try:
        from geometry.gpu_kernel import YggdrasilKernel

        kern = YggdrasilKernel(0, require_gpu=True)
    except Exception as exc:                                                    # pragma: no cover
        print("kernel not loadable here:", exc)
        return
    N, dh, dt = 4096, 16, 16
    g = torch.Generator().manual_seed(2)
    h = F.normalize(torch.randn(N, dh, generator=g), dim=-1) * 0.5 * torch.rand(N, 1, generator=g)
    th = (torch.rand(N, dt, generator=g) * 2 - 1) * math.pi
    A = torch.cat([h, th, torch.exp(torch.randn(N, 1, generator=g))], -1).cuda()
    B = A.flip(0).contiguous()
    t_kernel, _ = bench(lambda: kern.pairwise_distance(A, B, dh, dt), reps=20, warm=5)
    Aa, Bb = torch.randn(N, 34, device="cuda"), torch.randn(N, 34, device="cuda")
    t_gemm, _ = bench(lambda: (Aa @ Bb.T), reps=20, warm=5)
    print(f"current kernel (hyperbolic + wrapped torus + log-radius, one thread per output): {t_kernel:6.2f} ms")
    print(f"one fp32 GEMM of the same shape (4096x34 @ 34x4096)  [floor of the dot-product formulation]  : {t_gemm:6.2f} ms")
    print("-> the two compute DIFFERENT functions; this only bounds what a dot-product formulation can reach on this GPU.")


if __name__ == "__main__":
    neuron_identities()
    attention_equivalence()
    precision_vs_radius()
    speed_and_memory()
    kernel_vs_gemm_floor()
