"""PoC-C2: where do the ~3 ms of the 'fused' path in PoC-C go? Time the pieces separately (forward only, B=1, H=4)."""
import math, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch, torch.nn.functional as F
import poc_c_attention_and_neuron as C

B, H, D, M, k, tau, kappa = 1, 4, 32, 8, 1.0, 0.7, 1.0
print(f"{'T':>5s} | {'prep (cat, cos, sin, pad)':>26s} | {'SDPA core fp16':>15s} | {'SDPA core fp32':>15s} | {'renorm':>8s} | {'plain Euclid fp16':>18s}")
for T in [256, 1024, 2048, 4096]:
    g = torch.Generator(device="cuda").manual_seed(0)
    qs, ks, vs = (torch.randn(B, H, T, d, generator=g, device="cuda") * 0.8 for d in (D, D, D))
    qth, kth = (torch.rand(B, H, T, M, generator=g, device="cuda") * 2 * math.pi for _ in range(2))
    beta = torch.randn(B, H, T, generator=g, device="cuda")
    prep = lambda: C.augmented(qs, ks, vs, qth, kth, beta, k, tau, kappa)
    t_prep, _ = C.bench(prep, reps=30, warm=5)
    qa, ka, va = prep()
    core16 = lambda: F.scaled_dot_product_attention(qa.half(), ka.half(), va.half(), is_causal=True, scale=1 / tau)
    core32 = lambda: F.scaled_dot_product_attention(qa, ka, va, is_causal=True, scale=1 / tau)
    out = core32()
    renorm = lambda: out / torch.sqrt(k * (-C.mink(out, out)).clamp(min=1e-9))[..., None]
    plain = lambda: F.scaled_dot_product_attention(qs.half(), ks.half(), vs.half(), is_causal=True)
    r = [C.bench(f, reps=30, warm=5)[0] for f in (core16, core32, renorm, plain)]
    print(f"{T:5d} | {t_prep:23.3f} ms | {r[0]:12.3f} ms | {r[1]:12.3f} ms | {r[2]:5.3f} ms | {r[3]:15.3f} ms")
