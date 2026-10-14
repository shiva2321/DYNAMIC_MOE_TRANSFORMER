"""
bench_y7_kernels.py - Benchmark runner for Round Y7 (Kernels & GEMM + Epilogue).
Evaluates:
1. PTX compilation and NUL terminator verification across architectures.
2. Accuracy vs current Yggdrasil engine metric (target < 2e-4).
3. 4096^2 Pairwise distance latency (engine vs GEMM + epilogue vs GEMM floor).
4. Attention preparation eager overhead vs fused operation.
ASCII output only.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, ".")

import json
import time
import math
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F

from geometry.gpu_kernel import YggdrasilKernel
from geometry.attention import augment, _time, _log0, _exp0


def run_ptx_check():
    print("--- 1. PTX & NUL TERMINATOR CHECK ---", flush=True)
    ptx_path = Path("rust_yggdrasil/src/gpu_kernels.ptx")
    assert ptx_path.exists(), "gpu_kernels.ptx does not exist"
    data = ptx_path.read_bytes()
    size = len(data)
    last_byte = data[-1]
    is_nul = (last_byte == 0 or last_byte == b"\x00"[0])
    print(f"PTX size: {size} bytes | Last byte: {last_byte} | NUL terminated: {is_nul}", flush=True)
    assert is_nul, "gpu_kernels.ptx must have NUL terminator"
    return {"ptx_size_bytes": size, "nul_terminated": is_nul}


def run_accuracy_benchmark(kern: YggdrasilKernel):
    print("\n--- 2. ACCURACY VS ENGINE METRIC ---", flush=True)
    N, M = 128, 64
    dh, dt = 16, 16
    g = torch.Generator(device="cuda").manual_seed(42)

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

    diff = (dist_engine - dist_gemm).abs()
    max_err = float(diff.max().item())
    mean_err = float(diff.mean().item())

    print(f"Max error vs engine: {max_err:.2e} (acceptance threshold < 2.00e-4)", flush=True)
    print(f"Mean error vs engine: {mean_err:.2e}", flush=True)
    assert max_err < 2e-4, f"Failed acceptance threshold: max error {max_err:.2e} >= 2e-4"
    return {"max_error": max_err, "mean_error": mean_err, "accepted": max_err < 2e-4}


def run_latency_benchmark(kern: YggdrasilKernel):
    print("\n--- 3. 4096^2 PAIRWISE DISTANCE LATENCY BENCHMARK ---", flush=True)
    N = 4096
    dh, dt = 16, 16

    g = torch.Generator(device="cuda").manual_seed(101)
    h = F.normalize(torch.randn(N, dh, generator=g, device="cuda"), dim=-1) * 0.5 * torch.rand(N, 1, generator=g, device="cuda")
    th = (torch.rand(N, dt, generator=g, device="cuda") * 2 - 1) * math.pi
    A = torch.cat([h, th, torch.exp(torch.randn(N, 1, generator=g, device="cuda"))], -1)
    B = A.flip(0).contiguous()

    # GEMM floor baseline (one 4096x33 @ 33x4096 matmul)
    Aa = torch.randn(N, dh + dt + 1, device="cuda")
    Bb = torch.randn(N, dh + dt + 1, device="cuda")

    # Warmup
    for _ in range(5):
        _ = kern.pairwise_distance(A[:512], B[:512], dh, dt)
        _ = kern.pairwise_distance_gemm_epilogue(A[:512], B[:512], dh, dt, torus_mode="wrapped")
        _ = torch.matmul(Aa[:512], Bb[:512].t())
    torch.cuda.synchronize()

    reps = 30
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 1. Plain GEMM Floor
    start.record()
    for _ in range(reps):
        _ = torch.matmul(Aa, Bb.t())
    end.record()
    torch.cuda.synchronize()
    t_gemm_floor = start.elapsed_time(end) / reps

    # 2. Naive Engine Kernel
    start.record()
    for _ in range(reps):
        _ = kern.pairwise_distance(A, B, dh, dt)
    end.record()
    torch.cuda.synchronize()
    t_engine = start.elapsed_time(end) / reps

    # 3. Fast CUDA Kernel (64x64 tiled FMA + epilogue)
    start.record()
    for _ in range(reps):
        _ = kern.pairwise_distance_fast(A, B, dh, dt)
    end.record()
    torch.cuda.synchronize()
    t_cuda_fast = start.elapsed_time(end) / reps

    # 4. PyTorch Eager Reference (wrapped metric)
    start.record()
    for _ in range(5):
        _ = kern.pairwise_distance_gemm_epilogue(A, B, dh, dt, torus_mode="wrapped", use_cuda_kernel=False)
    end.record()
    torch.cuda.synchronize()
    t_torch_wrapped = start.elapsed_time(end) / 5

    # 5. PyTorch Eager Reference (smooth surrogate)
    start.record()
    for _ in range(10):
        _ = kern.pairwise_distance_gemm_epilogue(A, B, dh, dt, torus_mode="smooth", use_cuda_kernel=False)
    end.record()
    torch.cuda.synchronize()
    t_torch_smooth = start.elapsed_time(end) / 10

    speedup_fast = t_engine / max(t_cuda_fast, 1e-6)

    print(f"Plain GEMM floor (4096x33 @ 33x4096):    {t_gemm_floor:6.2f} ms", flush=True)
    print(f"Current naive engine kernel (16x16 tile): {t_engine:6.2f} ms", flush=True)
    print(f"Fast CUDA kernel (64x64 tiled + FMA):     {t_cuda_fast:6.2f} ms ({speedup_fast:.2f}x speedup vs naive)", flush=True)
    print(f"PyTorch eager reference (wrapped metric): {t_torch_wrapped:6.2f} ms", flush=True)
    print(f"PyTorch eager reference (smooth metric):  {t_torch_smooth:6.2f} ms", flush=True)

    target_speedup = 4.0
    target_4x_met = bool(speedup_fast >= target_speedup)

    return {
        "gemm_floor_ms": t_gemm_floor,
        "engine_kernel_ms": t_engine,
        "cuda_fast_ms": t_cuda_fast,
        "cuda_fast_speedup_vs_engine": speedup_fast,
        "target_4x_met": target_4x_met,
        "torch_wrapped_ms": t_torch_wrapped,
        "torch_smooth_ms": t_torch_smooth,
        "speedup_vs_torch_wrapped": float(t_torch_wrapped / max(t_cuda_fast, 1e-6)),
    }


def run_attention_prep_benchmark():
    print("\n--- 4. ATTENTION PREPARATION OVERHEAD ---", flush=True)
    B, H, T, D = 4, 8, 256, 16
    q_s = torch.randn(B, H, T, D, device="cuda")
    k_s = torch.randn(B, H, T, D, device="cuda")
    v_s = torch.randn(B, H, T, D, device="cuda")
    theta_q = torch.randn(B, H, T, 4, device="cuda")
    theta_k = torch.randn(B, H, T, 4, device="cuda")
    key_bias = torch.randn(B, H, T, device="cuda")

    # Warmup
    for _ in range(10):
        _ = augment(q_s, k_s, v_s, theta_q=theta_q, theta_k=theta_k, key_bias=key_bias)
    torch.cuda.synchronize()

    reps = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(reps):
        _ = augment(q_s, k_s, v_s, theta_q=theta_q, theta_k=theta_k, key_bias=key_bias)
    end.record()
    torch.cuda.synchronize()
    t_prep = start.elapsed_time(end) / reps

    # Centroid renormalisation timing
    out_pt = torch.randn(B, H, T, D + 1, device="cuda")
    start.record()
    for _ in range(reps):
        norm = torch.sqrt((out_pt[..., :1] ** 2 - (out_pt[..., 1:] ** 2).sum(-1, keepdim=True)).clamp(min=1e-7))
        _ = out_pt[..., 1:] / norm
    end.record()
    torch.cuda.synchronize()
    t_renorm = start.elapsed_time(end) / reps

    total_overhead = t_prep + t_renorm
    print(f"Eager augment preparation: {t_prep:.3f} ms", flush=True)
    print(f"Centroid renormalisation:  {t_renorm:.3f} ms", flush=True)
    print(f"Total prep + renorm:       {total_overhead:.3f} ms", flush=True)
    print(f"Overhead > 0.3 ms threshold: {total_overhead > 0.3}", flush=True)

    return {
        "prep_ms": t_prep,
        "renorm_ms": t_renorm,
        "total_overhead_ms": total_overhead,
        "exceeds_threshold": total_overhead > 0.3,
        "fused_kernel_status": "not done (stretch / overhead measured)",
    }


def main():
    print("=" * 70, flush=True)
    print("ROUND Y7 BENCHMARK: KERNELS, GEMM EPILOGUE & SPEEDUP", flush=True)
    print("=" * 70, flush=True)

    kern = YggdrasilKernel(0, require_gpu=True)
    res_ptx = run_ptx_check()
    res_acc = run_accuracy_benchmark(kern)
    res_lat = run_latency_benchmark(kern)
    res_prep = run_attention_prep_benchmark()

    results = {
        "ptx": res_ptx,
        "accuracy": res_acc,
        "latency": res_lat,
        "prep": res_prep,
    }

    os.makedirs("research/yggformer_results", exist_ok=True)
    with open("research/yggformer_results/e7_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved Round Y7 results to research/yggformer_results/e7_results.json", flush=True)


if __name__ == "__main__":
    main()
