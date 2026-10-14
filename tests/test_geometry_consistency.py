"""Consistency tests added after an independent review of the geometry package and its GPU kernel.

Each test pins a property that the first version violated or never checked:
* the GPU/CPU engine must compute the SAME metric as the reference manifold (it used a linear radial distance,
  the reference a log-ratio one);
* the Riemannian retraction must be correct for every dim_h (it silently left dimensions >= 64 uninitialised);
* the custom backward must equal the autograd gradient of the reference metric and must not materialise
  (N, M, D) tensors;
* the hyperbolic base is NOT associative and has a finite exact-inversion budget - recorded as executable facts.
"""

import math

import pytest
import torch

from geometry import YggdrasilManifold
from geometry.gpu_kernel import YggdrasilKernel
from geometry.hybrid import YggdrasilState
from geometry.layers import YggdrasilDistanceFunction
from geometry.poincare import PoincareBall

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


def _states(n, dh, dt, seed, r_lo=0.05, r_hi=8.0):
    """Random valid points: [hyperbolic | torus | intensity], intensities spread over two decades."""
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, dh, generator=g) * (0.15 / math.sqrt(dh)) * math.sqrt(dh)      # radius ~0.15 * sqrt(dh) / sqrt(dh) * sqrt(dh)
    h = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-6) * (0.1 + 0.6 * torch.rand(n, 1, generator=g))
    th = (torch.rand(n, dt, generator=g) * 2 - 1) * math.pi
    log_r = math.log(r_lo) + (math.log(r_hi) - math.log(r_lo)) * torch.rand(n, 1, generator=g)
    return torch.cat([h, th, torch.exp(log_r)], dim=-1)


def _reference_pairwise(A, B, dh, dt, c=1.0, wh=1.0, wt=1.0, wr=1.0):
    """Differentiable PyTorch implementation of the manifold metric defined in geometry/hybrid.py."""
    ym = YggdrasilManifold(dh, dt, c)
    sa = YggdrasilState(h=A[:, None, :dh], theta=A[:, None, dh:dh + dt], r=A[:, None, -1:])
    sb = YggdrasilState(h=B[None, :, :dh], theta=B[None, :, dh:dh + dt], r=B[None, :, -1:])
    return ym.distance(sa, sb, (wh, wt, wr))


# ------------------------------------------------------------------------- engine == reference manifold
@pytest.mark.parametrize("device", DEVICES)
def test_engine_distance_equals_the_reference_manifold_metric(device):
    dh, dt = 8, 8
    A, B = _states(40, dh, dt, 0), _states(25, dh, dt, 1)
    k = YggdrasilKernel(0, require_gpu=device == "cuda")
    got = k.pairwise_distance(A.to(device), B.to(device), dh, dt).cpu()
    ref = _reference_pairwise(A, B, dh, dt)
    assert torch.allclose(got, ref, rtol=2e-4, atol=2e-4), f"max diff {(got - ref).abs().max().item():.2e}"


def test_intensity_is_a_multiplicative_coordinate():
    """Scaling BOTH intensities by the same factor must not change the distance (ln r_a - ln r_b is invariant)."""
    dh, dt = 4, 4
    A, B = _states(10, dh, dt, 2), _states(10, dh, dt, 3)
    k = YggdrasilKernel(0)
    d1 = k.pairwise_distance(A, B, dh, dt)
    A2, B2 = A.clone(), B.clone()
    A2[:, -1] *= 7.0
    B2[:, -1] *= 7.0
    assert torch.allclose(d1, k.pairwise_distance(A2, B2, dh, dt), atol=2e-4)


# ------------------------------------------------------------------------------------ Riemannian retraction
def _reference_retract(p, g, lr, c=1.0, eps=1e-5):
    pb = PoincareBall(p.shape[1], c)
    factor = (1.0 - c * (p * p).sum(-1, keepdim=True)).clamp(min=1e-7)
    v = -lr * (factor ** 2 / 4.0) * g                       # Riemannian gradient step, inverse metric (1 - c|x|^2)^2 / 4
    out = pb.exp_map(p, v)
    return pb.project(out, eps)


@pytest.mark.skipif(not CUDA, reason="the retraction kernel exists on the GPU only")
@pytest.mark.parametrize("dim_h", [4, 16, 63, 64, 65, 80, 128, 200])
def test_riemannian_retract_matches_the_torch_reference_for_every_dimension(dim_h):
    torch.manual_seed(dim_h)
    p = torch.randn(32, dim_h) * (0.3 / math.sqrt(dim_h))
    g = torch.randn(32, dim_h)
    got = YggdrasilKernel(0, require_gpu=True).riemannian_retract(p.cuda(), g.cuda(), lr=0.05).cpu()
    ref = _reference_retract(p, g, 0.05)
    assert torch.allclose(got, ref, atol=2e-5), f"dim_h={dim_h}: max diff {(got - ref).abs().max().item():.2e}"


@pytest.mark.skipif(not CUDA, reason="the retraction kernel exists on the GPU only")
def test_riemannian_retract_never_leaves_the_ball_even_for_a_huge_step():
    p = torch.randn(16, 96).cuda() * 0.05
    g = torch.randn(16, 96).cuda() * 1e3
    out = YggdrasilKernel(0, require_gpu=True).riemannian_retract(p, g, lr=10.0)
    assert torch.isfinite(out).all() and (out.norm(dim=-1) <= 1.0 - 1e-5 + 1e-6).all()


def test_riemannian_retract_has_a_cpu_path():
    p = torch.randn(8, 12) * 0.1
    g = torch.randn(8, 12)
    got = YggdrasilKernel(0).riemannian_retract(p, g, lr=0.05)
    assert torch.allclose(got, _reference_retract(p, g, 0.05), atol=2e-5)


# ------------------------------------------------------------------------------------------------- backward
def _custom_grads(A, B, dh, dt, W, kernel):
    A, B = A.clone().requires_grad_(True), B.clone().requires_grad_(True)
    D = YggdrasilDistanceFunction.apply(A, B, dh, dt, 1.0, 1e-5, 1.0, 1.0, 1.0, kernel)
    (D * W).sum().backward()
    return A.grad, B.grad


@pytest.mark.parametrize("device", DEVICES)
def test_custom_backward_equals_autograd_of_the_reference_metric(device):
    dh, dt = 8, 6
    A, B = _states(30, dh, dt, 4), _states(20, dh, dt, 5)
    W = torch.randn(30, 20, generator=torch.Generator().manual_seed(6))
    gA, gB = _custom_grads(A.to(device), B.to(device), dh, dt, W.to(device), YggdrasilKernel(0, require_gpu=device == "cuda"))
    A_ref, B_ref = A.clone().requires_grad_(True), B.clone().requires_grad_(True)
    (_reference_pairwise(A_ref, B_ref, dh, dt) * W).sum().backward()
    scale = A_ref.grad.abs().max().item()
    assert (gA.cpu() - A_ref.grad).abs().max().item() < 2e-3 * max(1.0, scale)
    assert (gB.cpu() - B_ref.grad).abs().max().item() < 2e-3 * max(1.0, B_ref.grad.abs().max().item())


@pytest.mark.skipif(not CUDA, reason="memory behaviour is checked on the GPU")
def test_backward_does_not_materialise_n_by_m_by_d_tensors():
    dh, dt, n = 16, 16, 2048
    A, B = _states(n, dh, dt, 7).cuda(), _states(n, dh, dt, 8).cuda()
    W = torch.randn(n, n, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _custom_grads(A, B, dh, dt, W, YggdrasilKernel(0, require_gpu=True))
    torch.cuda.synchronize()
    extra_gb = (torch.cuda.max_memory_allocated() - base) / 1e9
    # one (N, M, D) float32 tensor would already be 2048*2048*33*4 = 0.55 GB; the old code held several at once
    assert extra_gb < 0.8, f"backward peaked at {extra_gb:.2f} GB above its inputs"


# ------------------------------------------------ executable facts about the hyperbolic base (not bugs)
def test_mobius_addition_is_not_associative_and_not_commutative():
    pb = PoincareBall(16, 1.0)
    torch.manual_seed(0)
    a, b, c = (pb.exp_map_zero(torch.randn(1, 16, dtype=torch.float64) * 0.1) for _ in range(3))
    assoc = pb.distance(pb.mobius_add(pb.mobius_add(a, b), c), pb.mobius_add(a, pb.mobius_add(b, c))).item()
    comm = pb.distance(pb.mobius_add(a, b), pb.mobius_add(b, a)).item()
    assert assoc > 1e-4 and comm > 1e-4          # a gyrogroup: composing relations into one action is NOT exact


def test_sequential_application_is_exactly_invertible_inside_the_depth_budget():
    pb = PoincareBall(16, 1.0)
    torch.manual_seed(1)
    x = pb.exp_map_zero(torch.randn(1, 16) * 0.1)
    deltas = [pb.exp_map_zero(torch.randn(1, 16) * 0.05) for _ in range(20)]       # action radius ~0.19
    y = x.clone()
    for d in deltas:
        y = pb.mobius_add(d, y)
    for d in reversed(deltas):
        y = pb.mobius_add(-d, y)
    assert pb.distance(x, y).item() < 1e-4


def test_inversion_fails_once_a_chain_pushes_points_into_the_boundary_clamp():
    """The ball clamps at radius 1 - 1e-4 (hyperbolic radius ~9.9); beyond it information is destroyed."""
    pb = PoincareBall(16, 1.0)
    torch.manual_seed(2)
    x = pb.exp_map_zero(torch.randn(1, 16, dtype=torch.float64) * 0.1)
    deltas = [pb.exp_map_zero(torch.randn(1, 16, dtype=torch.float64) * 0.4) for _ in range(100)]    # radius ~0.92
    y = x.clone()
    for d in deltas:
        y = pb.mobius_add(d, y)
    for d in reversed(deltas):
        y = pb.mobius_add(-d, y)
    assert pb.distance(x, y).item() > 1.0


@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
def test_the_gpu_engine_really_loads():
    """A broken PTX (for example one without its NUL terminator, CUDA error 218) used to fall back to the CPU path with
    only a printed note, so every 'GPU' test passed on the CPU. require_gpu=True turns that into an error."""
    k = YggdrasilKernel(0, require_gpu=True)
    assert k._gpu_engine


@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
def test_gpu_and_cpu_paths_agree_on_the_same_inputs():
    dh, dt = 16, 16
    A, B = _states(300, dh, dt, 11), _states(200, dh, dt, 12)
    k = YggdrasilKernel(0, require_gpu=True)
    gpu = k.pairwise_distance(A.cuda(), B.cuda(), dh, dt).cpu()
    k_cpu = YggdrasilKernel(0, require_gpu=False)
    cpu = k_cpu.pairwise_distance(A, B, dh, dt)
    assert (gpu - cpu).abs().max().item() < 2e-5


def test_hypersphere_antipodal_accuracy():
    """Verify Hypersphere.distance gives exactly pi without NaN or clipping loss at antipodal points."""
    from geometry.sphere import Hypersphere
    D = 16
    sphere = Hypersphere(D)
    g = torch.Generator().manual_seed(42)
    u = torch.randn(50, D, generator=g)
    u = u / u.norm(dim=-1, keepdim=True)
    v_antipodal = -u
    dist = sphere.distance(u, v_antipodal)
    expected = torch.full_like(dist, math.pi)
    assert not torch.isnan(dist).any(), "Antipodal distance must not contain NaNs"
    assert torch.allclose(dist, expected, atol=1e-5), f"Antipodal error: {(dist - expected).abs().max().item()}"

    # Coincident points must have distance 0.0
    dist_zero = sphere.distance(u, u)
    assert torch.allclose(dist_zero, torch.zeros_like(dist_zero), atol=1e-6)
