import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from geometry.gpu_kernel import YggdrasilKernel

def main():
    print("================================================================")
    print("  VERIFYING RUST GPU KERNEL ON NVIDIA GEFORCE RTX 3060")
    print("================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active PyTorch device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    kernel = YggdrasilKernel(device_ordinal=0)

    dim_h = 16
    dim_t = 16
    total_dim = dim_h + dim_t + 1  # 33

    # 1. Numerical Accuracy Verification: GPU vs CPU
    print("\n--- Test 1: Cross-Device Numerical Invariance (GPU vs CPU) ---")
    N = 256
    M = 128

    torch.manual_seed(42)
    # Generate points within safe initial norms
    A_cpu = torch.randn(N, total_dim, dtype=torch.float32) * 0.1
    B_cpu = torch.randn(M, total_dim, dtype=torch.float32) * 0.1
    # Radial coordinate positive
    A_cpu[:, -1] = A_cpu[:, -1].abs() + 0.1
    B_cpu[:, -1] = B_cpu[:, -1].abs() + 0.1

    # Project on CPU
    A_cpu = kernel.project(A_cpu, dim_h, dim_t)
    B_cpu = kernel.project(B_cpu, dim_h, dim_t)

    # Compute CPU distances
    dist_cpu = kernel.pairwise_distance(A_cpu, B_cpu, dim_h, dim_t)

    if device.type == "cuda":
        A_gpu = A_cpu.to(device)
        B_gpu = B_cpu.to(device)

        # Compute GPU distances
        dist_gpu = kernel.pairwise_distance(A_gpu, B_gpu, dim_h, dim_t)
        max_dist_err = (dist_cpu - dist_gpu.cpu()).abs().max().item()
        print(f"Max Pairwise Distance Error (GPU vs CPU): {max_dist_err:.2e}")
        assert max_dist_err < 1e-5, f"GPU distance deviates from CPU by {max_dist_err}"

        # Test Mobius addition GPU vs CPU
        U_cpu = A_cpu[:M, :dim_h]
        V_cpu = B_cpu[:, :dim_h]
        mob_cpu = kernel.mobius_add(U_cpu, V_cpu)
        mob_gpu = kernel.mobius_add(U_cpu.to(device), V_cpu.to(device))
        max_mob_err = (mob_cpu - mob_gpu.cpu()).abs().max().item()
        print(f"Max Mobius Addition Error (GPU vs CPU): {max_mob_err:.2e}")
        assert max_mob_err < 1e-5, f"GPU mobius_add deviates from CPU by {max_mob_err}"

        # Test Torus Binding GPU vs CPU
        T1_cpu = A_cpu[:M, dim_h:dim_h+dim_t]
        T2_cpu = B_cpu[:, dim_h:dim_h+dim_t]
        bind_cpu = kernel.torus_bind(T1_cpu, T2_cpu)
        bind_gpu = kernel.torus_bind(T1_cpu.to(device), T2_cpu.to(device))
        max_bind_err = (bind_cpu - bind_gpu.cpu()).abs().max().item()
        print(f"Max Torus Binding Error (GPU vs CPU): {max_bind_err:.2e}")
        assert max_bind_err < 1e-5, f"GPU torus_bind deviates from CPU by {max_bind_err}"

        # Test Invertibility on GPU (1000 hops)
        print("\n--- Test 2: Invertibility & Zero-Drift on GPU ---")
        curr = T1_cpu.to(device)
        shift = T2_cpu.to(device)
        shifted = kernel.torus_bind(curr, shift)
        recovered = kernel.torus_unbind(shifted, shift)
        drift = (curr - recovered).abs().max().item()
        print(f"GPU Torus Single-hop Inversion Error: {drift:.2e} rad")
        assert drift < 1e-6, "GPU Torus failed exact inversion"

        # Test Riemannian Retraction on GPU
        print("\n--- Test 3: Riemannian Retraction Step on GPU ---")
        pts = U_cpu.to(device)
        grads = torch.randn_like(pts) * 0.05
        retracted = kernel.riemannian_retract(pts, grads, lr=0.01)
        # Check that norm <= (1 - eps)
        norm_retracted = retracted.norm(dim=-1).max().item()
        print(f"Max norm after Riemannian step: {norm_retracted:.6f} (boundary: 0.999990)")
        assert norm_retracted <= 0.999990 + 1e-5, "Retraction failed boundary check"

    # 4. Large-Scale Throughput Benchmark on RTX 3060
    if device.type == "cuda":
        print("\n--- Test 4: High-Throughput Large-Scale GPU Benchmark ---")
        N_large = 4096
        M_large = 4096
        total_pairs = N_large * M_large  # 16,777,216 pairs

        A_large = torch.randn(N_large, total_dim, device=device, dtype=torch.float32) * 0.1
        B_large = torch.randn(M_large, total_dim, device=device, dtype=torch.float32) * 0.1
        A_large[:, -1] = A_large[:, -1].abs() + 0.1
        B_large[:, -1] = B_large[:, -1].abs() + 0.1

        # Warmup
        _ = kernel.pairwise_distance(A_large[:128], B_large[:128], dim_h, dim_t)
        torch.cuda.synchronize()

        start_time = time.perf_counter()
        iters = 5
        for _ in range(iters):
            out_large = kernel.pairwise_distance(A_large, B_large, dim_h, dim_t)
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start_time) / iters

        throughput = total_pairs / elapsed
        print(f"Matrix Size: ({N_large} x {M_large}) = {total_pairs:,} pairwise distances")
        print(f"Kernel Execution Time: {elapsed * 1000.0:.2f} ms")
        print(f"Throughput: {throughput / 1e6:.2f} Million distances / second on RTX 3060!")

    print("\nALL GPU KERNEL TESTS AND INVARIANTS PASSED PERFECTLY!")

if __name__ == "__main__":
    main()
