"""Tests for the Lorentz model, the fused Yggdrasil attention and the norm-gated neuron (geometry/lorentz.py, attention.py, neuron.py).

Everything here is small (T <= 64) except one GPU memory check. The claims pinned down:
* the hyperboloid and the Poincare ball describe the same space (same distance);
* the fused dot-product attention equals the pair-by-pair definition (hyperbolic + torus phase + salience + causal + per-head c/tau);
* the output is the Lorentz centroid and stays a valid point; the flat limit c -> 0 is Gaussian (L2) attention;
* half-precision logits lose accuracy like R^2 (the reason the product is evaluated in float32 by default);
* one ambient-linear neuron is a prototype / hyperplane / horosphere neuron depending on the sign of <w, w>_L.
"""

import math

import pytest
import torch

from geometry import LorentzModel, PoincareBall
from geometry.attention import clip_radius, reference_attention, safe_radius, yggdrasil_attention
from geometry.neuron import NormGatedLinear

CUDA = torch.cuda.is_available()
D64 = dict(dtype=torch.float64)


def _ball_points(n, d, c, seed, r=0.6):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, d, generator=g, **D64)
    return u / u.norm(dim=-1, keepdim=True) * (r / math.sqrt(c)) * torch.rand(n, 1, generator=g, **D64)


# ------------------------------------------------------------------------------------------------ the model itself
@pytest.mark.parametrize("c", [1.0, 0.5, 2.0])
def test_hyperboloid_and_poincare_ball_are_the_same_space(c):
    lm, pb = LorentzModel(5, c), PoincareBall(5, c)
    u, v = _ball_points(30, 5, c, 0), _ball_points(30, 5, c, 1)
    x, y = lm.from_poincare(u), lm.from_poincare(v)
    assert torch.allclose(lm.inner(x, x), torch.full((30,), -1.0 / c, **D64), atol=1e-9)
    assert torch.allclose(lm.to_poincare(x), u, atol=1e-9)
    assert torch.allclose(lm.distance(x, y), pb.distance(u, v), atol=1e-5)


@pytest.mark.parametrize("c", [1.0, 0.7])
def test_exp_and_log_are_inverse_and_stay_on_the_hyperboloid(c):
    lm = LorentzModel(6, c)
    g = torch.Generator().manual_seed(2)
    v = torch.randn(20, 6, generator=g, **D64) * 0.7
    x = lm.exp_map_zero(v)
    assert torch.allclose(lm.inner(x, x), torch.full((20,), -1.0 / c, **D64), atol=1e-8)
    assert torch.allclose(lm.log_map_zero(x), v, atol=1e-8)
    base = lm.exp_map_zero(torch.randn(20, 6, generator=g, **D64) * 0.5)
    target = lm.exp_map_zero(torch.randn(20, 6, generator=g, **D64) * 0.5)
    tangent = lm.log_map(base, target)
    assert torch.allclose(lm.inner(base, tangent), torch.zeros(20, **D64), atol=1e-8)          # the log lies in the tangent space
    assert torch.allclose(lm.exp_map(base, tangent), target, atol=1e-7)
    assert torch.allclose(lm.inner(tangent, tangent).sqrt(), lm.distance(base, target), atol=1e-7)


def test_log_map_is_accurate_close_to_the_base_point():
    lm = LorentzModel(4, 1.0)
    x = lm.exp_map_zero(torch.tensor([[0.3, -0.2, 0.1, 0.4]], **D64))
    t = torch.tensor([[0.2, 1.0, -2.0, 0.3, 0.5]], **D64)
    v = t + lm.inner(x, t).unsqueeze(-1) * x                       # tangent at x for c = 1: <x, v>_L = 0
    v = v / torch.sqrt(lm.inner(v, v)).unsqueeze(-1) * 1e-4
    y = lm.exp_map(x, v)
    assert torch.allclose(lm.log_map(x, y), v, atol=1e-9)


def test_centroid_of_one_point_is_that_point_and_always_lies_on_the_hyperboloid():
    lm = LorentzModel(5, 1.3)
    pts = lm.exp_map_zero(torch.randn(8, 5, generator=torch.Generator().manual_seed(3), **D64))
    w = torch.rand(8, generator=torch.Generator().manual_seed(4), **D64)
    w = w / w.sum()
    cen = lm.centroid(w, pts)
    assert abs(lm.inner(cen, cen).item() + 1.0 / 1.3) < 1e-9
    single = lm.centroid(torch.ones(1, **D64), pts[:1])
    assert torch.allclose(single, pts[:1].squeeze(0), atol=1e-9)
    tm = lm.tangent_mean(w, pts)
    assert abs(lm.inner(tm, tm).item() + 1.0 / 1.3) < 1e-8


# ------------------------------------------------------------------------------------------- the fused attention
def _inputs(B=2, H=3, T=11, D=5, Dv=4, M=3, seed=0, scale=0.9):
    g = torch.Generator().manual_seed(seed)
    q, k = (torch.randn(B, H, T, D, generator=g, **D64) * scale for _ in range(2))
    v = torch.randn(B, H, T, Dv, generator=g, **D64) * scale
    tq, tk = (torch.rand(B, H, T, M, generator=g, **D64) * 2 * math.pi for _ in range(2))
    bias = torch.randn(B, H, T, generator=g, **D64)
    return q, k, v, tq, tk, bias


@pytest.mark.parametrize("aggregate", ["lorentz", "tangent", "euclid"])
@pytest.mark.parametrize("causal", [False, True])
def test_fused_attention_equals_the_pairwise_definition(aggregate, causal):
    q, k, v, tq, tk, bias = _inputs()
    c = torch.tensor([0.5, 1.0, 1.7], **D64).view(1, 3, 1, 1)                  # per-head curvature
    tau = torch.tensor([0.6, 1.0, 2.0], **D64).view(1, 3, 1, 1)                # per-head temperature
    kappa = torch.tensor([0.0, 0.8, 1.5], **D64).view(1, 3, 1, 1)               # per-head phase strength
    kw = dict(c=c, tau=tau, theta_q=tq, theta_k=tk, kappa=kappa, key_bias=bias, is_causal=causal, aggregate=aggregate)
    fused = yggdrasil_attention(q, k, v, **kw)
    ref = reference_attention(q, k, v, **kw)
    assert torch.allclose(fused, ref, atol=1e-10), f"max diff {(fused - ref).abs().max().item():.2e}"


def test_each_term_can_be_switched_off_independently():
    q, k, v, tq, tk, bias = _inputs()
    for kw in [dict(), dict(theta_q=tq, theta_k=tk), dict(key_bias=bias), dict(theta_q=tq, theta_k=tk, key_bias=bias)]:
        assert torch.allclose(yggdrasil_attention(q, k, v, **kw), reference_attention(q, k, v, **kw), atol=1e-10)


def test_lorentz_aggregate_is_the_lorentz_centroid_of_the_values():
    q, k, v, *_ = _inputs(B=1, H=1, T=7)
    lm = LorentzModel(4, 1.0)
    d2 = ((q[..., :, None, :] - k[..., None, :, :]) ** 2).sum(-1) - (lm.time(q)[..., :, None, 0] - lm.time(k)[..., None, :, 0]) ** 2
    a = torch.softmax(-d2 / 2.0, dim=-1)
    expected = lm.centroid(a, lm.lift(v).unsqueeze(-3).expand(1, 1, 7, 7, 5))[..., 1:]
    assert torch.allclose(yggdrasil_attention(q, k, v), expected, atol=1e-10)


def test_flat_limit_is_gaussian_l2_attention():
    q, k, v, *_ = _inputs(B=1, H=2, T=9)
    out = yggdrasil_attention(q, k, v, c=1e-6, aggregate="euclid")
    gaussian = torch.softmax(-((q[..., :, None, :] - k[..., None, :, :]) ** 2).sum(-1) / 2.0, dim=-1) @ v
    assert torch.allclose(out, gaussian, atol=2e-4)


def test_gradients_are_correct():
    q, k, v, tq, tk, bias = _inputs(B=1, H=1, T=4, D=3, Dv=2, M=2, scale=0.6)
    ins = [t.clone().requires_grad_(True) for t in (q, k, v, tq, tk, bias)]
    f = lambda q_, k_, v_, tq_, tk_, b_: yggdrasil_attention(q_, k_, v_, c=1.2, tau=0.8, theta_q=tq_, theta_k=tk_, kappa=0.9, key_bias=b_, is_causal=True)
    assert torch.autograd.gradcheck(f, tuple(ins), eps=1e-6, atol=1e-6)


def test_float32_fused_matches_float64_reference():
    q, k, v, tq, tk, bias = _inputs(T=32)
    out = yggdrasil_attention(*(t.float() for t in (q, k, v)), theta_q=tq.float(), theta_k=tk.float(), key_bias=bias.float(), is_causal=True)
    ref = reference_attention(q, k, v, theta_q=tq, theta_k=tk, key_bias=bias, is_causal=True)
    assert (out.double() - ref).abs().max().item() < 5e-5


# -------------------------------------------------------------------------------------------------- the precision rule
def test_half_precision_logit_error_grows_like_the_radius_squared():
    """<q,k>_L subtracts two numbers of size ~R^2; with a 7-bit mantissa (bfloat16) the absolute logit error is ~2^-8 R^2."""
    lm = LorentzModel(16, 1.0)
    errs = {}
    for r in (1.0, 8.0):
        g = torch.Generator().manual_seed(5)
        q = torch.nn.functional.normalize(torch.randn(64, 16, generator=g, **D64), dim=-1) * r
        k = q + 0.4 * torch.nn.functional.normalize(torch.randn(64, 16, generator=g, **D64), dim=-1)
        qa = torch.cat([-lm.time(q), q], -1)
        ka = torch.cat([lm.time(k), k], -1)
        truth = (qa * ka).sum(-1)
        low = (qa.bfloat16().float() * ka.bfloat16().float()).sum(-1)
        errs[r] = (low - truth).abs().max().item()
    assert errs[1.0] < 0.05 < errs[8.0] and errs[8.0] > 8 * errs[1.0]


def test_safe_radius_and_clip_radius():
    assert safe_radius(torch.bfloat16) < safe_radius(torch.float16) < safe_radius(torch.float32)
    x = torch.randn(10, 8) * 5
    y = clip_radius(x, 3.0)
    assert (y.norm(dim=-1) <= 3.0 + 1e-5).all()
    small = torch.randn(4, 8) * 0.01
    assert torch.allclose(clip_radius(small, 3.0), small)


@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
def test_gpu_matches_cpu_and_never_materialises_the_t_by_t_matrix():
    q, k, v, tq, tk, bias = _inputs(B=1, H=2, T=48)
    kw = dict(theta_q=tq.float(), theta_k=tk.float(), key_bias=bias.float(), is_causal=True)
    cpu = yggdrasil_attention(q.float(), k.float(), v.float(), **kw)
    gpu = yggdrasil_attention(q.float().cuda(), k.float().cuda(), v.float().cuda(), **{n: t.cuda() if torch.is_tensor(t) else t for n, t in kw.items()})
    assert (gpu.cpu() - cpu).abs().max().item() < 2e-4
    B, H, T, D = 1, 4, 2048, 32
    qq, kk, vv = (torch.randn(B, H, T, D, device="cuda") * 0.8 for _ in range(3))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        yggdrasil_attention(qq, kk, vv, is_causal=True)
    torch.cuda.synchronize()
    extra_mb = (torch.cuda.max_memory_allocated() - base) / 1e6
    assert extra_mb < 30, f"peak {extra_mb:.0f} MB; one (B, H, T, T) float32 matrix alone would be {B * H * T * T * 4 / 1e6:.0f} MB"


# ------------------------------------------------------------------------------------------------------- the neuron
def _layer(w, w0, d, bias=False):
    layer = NormGatedLinear(d, w.shape[0], curvature=1.0, bias=bias).double()
    with torch.no_grad():
        layer.weight.copy_(w)
        layer.time_weight.copy_(w0)
    return layer


def test_zero_time_weight_is_an_ordinary_linear_neuron():
    layer = NormGatedLinear(7, 5)
    x = torch.randn(3, 7)
    assert torch.allclose(layer(x), torch.nn.functional.linear(x, layer.weight, layer.bias), atol=1e-6)


def test_time_like_neuron_is_a_distance_to_a_prototype_neuron():
    lm = LorentzModel(3, 1.0)
    g = torch.Generator().manual_seed(6)
    w = torch.randn(4, 3, generator=g, **D64) * 0.4
    w0 = (w.norm(dim=-1) + 0.2 + torch.rand(4, generator=g, **D64))              # w0 > |w_s|  =>  <w, w>_L < 0
    layer = _layer(w, w0, 3)
    assert (layer.causal_character() < 0).all()
    x = torch.randn(25, 3, generator=g, **D64)
    mass = torch.sqrt(w0 ** 2 - (w ** 2).sum(-1))
    protos = torch.cat([w0[:, None], w], -1) / mass[:, None]                      # unit time-like vectors = hyperboloid points
    d = lm.distance(protos[None, :, :].expand(25, 4, 4), lm.lift(x)[:, None, :].expand(25, 4, 4))
    assert torch.allclose(layer(x), -mass[None, :] * torch.cosh(d), atol=1e-7)


def test_space_like_neuron_is_a_signed_distance_to_a_geodesic_hyperplane():
    lm = LorentzModel(2, 1.0)
    w = torch.tensor([[0.8, -0.6]], **D64) * 1.7
    layer = _layer(w, torch.zeros(1, **D64), 2)                                   # w0 = 0: the hyperplane passes through the origin
    assert (layer.causal_character() > 0).all()
    x = torch.randn(12, 2, generator=torch.Generator().manual_seed(7), **D64) * 1.2
    t = torch.tensor([0.6, 0.8], **D64)                                           # unit direction of the geodesic line {w_s . x_s = 0}
    s = torch.linspace(-10, 10, 20001, **D64)
    line = lm.exp_map_zero(s[:, None] * t[None, :])
    dist = lm.distance(lm.lift(x)[:, None, :].expand(12, 20001, 3), line[None].expand(12, 20001, 3)).min(dim=1).values
    signed = torch.sign(layer(x).squeeze(-1)) * dist
    assert torch.allclose(layer(x).squeeze(-1), w.norm() * torch.sinh(signed), atol=1e-4)      # the brute-force line search has ~1e-6 resolution


def test_light_like_neuron_is_a_horosphere_busemann_neuron():
    lm = LorentzModel(2, 1.0)
    omega = torch.tensor([math.cos(1.1), math.sin(1.1)], **D64)
    layer = _layer(omega[None, :], torch.ones(1, **D64), 2)                      # <w, w>_L = |omega|^2 - 1 = 0
    assert abs(layer.causal_character().item()) < 1e-12
    u = _ball_points(40, 2, 1.0, 8, r=0.95)
    x = lm.from_poincare(u)[..., 1:]
    busemann = torch.log(((omega - u) ** 2).sum(-1) / (1 - (u * u).sum(-1)))
    assert torch.allclose(torch.log(-layer(x).squeeze(-1)), busemann, atol=1e-8)
