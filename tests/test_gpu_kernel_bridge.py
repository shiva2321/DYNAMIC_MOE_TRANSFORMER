import pytest
import torch
from geometry.gpu_kernel import YggdrasilKernel

def test_kernel_initialization():
    kernel = YggdrasilKernel(device_ordinal=0)
    assert kernel is not None

def test_kernel_cpu_gpu_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel = YggdrasilKernel(device_ordinal=0)

    dim_h, dim_t = 8, 8
    total_dim = dim_h + dim_t + 1
    N, M = 32, 16

    A_cpu = torch.randn(N, total_dim, dtype=torch.float32) * 0.1
    B_cpu = torch.randn(M, total_dim, dtype=torch.float32) * 0.1
    A_cpu[:, -1] = A_cpu[:, -1].abs() + 0.1
    B_cpu[:, -1] = B_cpu[:, -1].abs() + 0.1

    A_cpu = kernel.project(A_cpu, dim_h, dim_t)
    B_cpu = kernel.project(B_cpu, dim_h, dim_t)

    dist_cpu = kernel.pairwise_distance(A_cpu, B_cpu, dim_h, dim_t)
    assert not torch.isnan(dist_cpu).any()

    if device.type == "cuda":
        A_gpu = A_cpu.to(device)
        B_gpu = B_cpu.to(device)
        dist_gpu = kernel.pairwise_distance(A_gpu, B_gpu, dim_h, dim_t)
        err = (dist_cpu - dist_gpu.cpu()).abs().max().item()
        assert err < 1e-5, f"GPU distance deviates from CPU by {err}"

def test_kernel_boundary_retraction():
    kernel = YggdrasilKernel(device_ordinal=0)
    dim_h, dim_t = 4, 4
    total_dim = dim_h + dim_t + 1

    wild = torch.randn(20, total_dim, dtype=torch.float32) * 20.0
    wild[:, -1] = -5.0
    projected = kernel.project(wild, dim_h, dim_t)

    h_norms = projected[:, :dim_h].norm(dim=-1)
    assert (h_norms <= 0.99999 + 1e-4).all()
    assert (projected[:, -1] >= 1e-5).all()
