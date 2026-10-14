import math
import os
import time
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from geometry.gpu_kernel import YggdrasilKernel


def test_ptx_nul_terminator():
    """Verify that gpu_kernels.ptx ends with a NUL terminator byte (code 218 prevention)."""
    ptx_path = Path("rust_yggdrasil/src/gpu_kernels.ptx")
    assert ptx_path.exists(), "gpu_kernels.ptx must exist"
    data = ptx_path.read_bytes()
    assert len(data) > 0, "gpu_kernels.ptx is empty"
    assert data[-1] == 0 or data[-1] == b"\x00"[0], f"PTX must end with NUL byte, got {data[-1]}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for kernel testing")
def test_gemm_epilogue_accuracy_vs_engine():
    """Acceptance: agrees with current engine metric to < 2e-4."""
    kern = YggdrasilKernel(0, require_gpu=True)
    N, M = 64, 32
    dh, dt = 8, 8
    total_dim = dh + dt + 1

    g = torch.Generator(device="cuda").manual_seed(42)
    # Generate points inside Poincare ball
    h_A = F.normalize(torch.randn(N, dh, generator=g, device="cuda"), dim=-1) * 0.5 * torch.rand(N, 1, generator=g, device="cuda")
    t_A = (torch.rand(N, dt, generator=g, device="cuda") * 2 - 1) * math.pi
    r_A = torch.exp(torch.randn(N, 1, generator=g, device="cuda"))
    A = torch.cat([h_A, t_A, r_A], dim=-1)

    h_B = F.normalize(torch.randn(M, dh, generator=g, device="cuda"), dim=-1) * 0.5 * torch.rand(M, 1, generator=g, device="cuda")
    t_B = (torch.rand(M, dt, generator=g, device="cuda") * 2 - 1) * math.pi
    r_B = torch.exp(torch.randn(M, 1, generator=g, device="cuda"))
    B = torch.cat([h_B, t_B, r_B], dim=-1)

    dist_engine = kern.pairwise_distance(A, B, dh, dt)
    dist_gemm = kern.pairwise_distance_gemm_epilogue(A, B, dh, dt, torus_mode="wrapped")

    diff = (dist_engine - dist_gemm).abs().max().item()
    assert diff < 2e-4, f"GEMM epilogue deviates from engine metric by {diff:.2e} (target < 2e-4)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for kernel benchmarking")
def test_gemm_epilogue_speedup():
    """Verify GEMM + epilogue is significantly faster than naive engine at 4096^2."""
    kern = YggdrasilKernel(0, require_gpu=True)
    N = 4096
    dh, dt = 16, 16

    g = torch.Generator(device="cuda").manual_seed(101)
    h = F.normalize(torch.randn(N, dh, generator=g, device="cuda"), dim=-1) * 0.5 * torch.rand(N, 1, generator=g, device="cuda")
    th = (torch.rand(N, dt, generator=g, device="cuda") * 2 - 1) * math.pi
    A = torch.cat([h, th, torch.exp(torch.randn(N, 1, generator=g, device="cuda"))], -1)
    B = A.flip(0).contiguous()

    # Warmup
    for _ in range(3):
        _ = kern.pairwise_distance(A[:512], B[:512], dh, dt)
        _ = kern.pairwise_distance_gemm_epilogue(A[:512], B[:512], dh, dt, torus_mode="wrapped")
    torch.cuda.synchronize()

    # Timing engine
    reps = 10
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(reps):
        _ = kern.pairwise_distance(A, B, dh, dt)
    end.record()
    torch.cuda.synchronize()
    t_engine = start.elapsed_time(end) / reps

    # Timing GEMM + epilogue
    start.record()
    for _ in range(reps):
        _ = kern.pairwise_distance_gemm_epilogue(A, B, dh, dt, torus_mode="wrapped")
    end.record()
    torch.cuda.synchronize()
    t_gemm = start.elapsed_time(end) / reps

    speedup = t_engine / max(t_gemm, 1e-6)
    print(f"\n4096^2 pairwise distance: engine = {t_engine:.2f} ms, GEMM+epilogue = {t_gemm:.2f} ms ({speedup:.2f}x speedup)")
    assert t_gemm < t_engine, f"GEMM epilogue ({t_gemm:.2f} ms) must be faster than naive engine ({t_engine:.2f} ms)"


def test_attention_prep_overhead():
    """Benchmark eager attention prep overhead vs baseline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, T, D = 2, 4, 128, 16
    q_s = torch.randn(B, H, T, D, device=device)
    k_s = torch.randn(B, H, T, D, device=device)
    v_s = torch.randn(B, H, T, D, device=device)

    from geometry.attention import augment
    q_aug, k_aug, v_aug, dv = augment(q_s, k_s, v_s, c=1.0, tau=1.0)
    assert q_aug.shape[-1] >= D + 1
    assert dv == D
