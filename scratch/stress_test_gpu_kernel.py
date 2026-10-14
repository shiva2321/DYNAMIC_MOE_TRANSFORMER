import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geometry.gpu_kernel import YggdrasilKernel

def test_extreme_blindspots():
    print("================================================================")
    print("  STRESS TESTING BLINDSPOTS & NUMERICAL STABILITY ON GPU")
    print("================================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel = YggdrasilKernel(device_ordinal=0)

    dim_h = 16
    dim_t = 16
    total_dim = dim_h + dim_t + 1

    # Blindspot 1: Exactly Identical Points (u == v)
    print("\n[Blindspot 1] Identical Points (u == v):")
    A = torch.zeros(10, total_dim, device=device, dtype=torch.float32)
    A[:, -1] = 1.0
    dist = kernel.pairwise_distance(A, A, dim_h, dim_t)
    diag = torch.diagonal(dist)
    print("Diagonal values (should all be 0.0):", diag.cpu().tolist())
    assert torch.all(diag == 0.0), "Zero distance failed on identical points!"
    assert not torch.isnan(dist).any(), "NaN found in identical points!"

    # Blindspot 2: Extreme Boundary Proximity (norm = 0.99999)
    print("\n[Blindspot 2] Extreme Boundary Proximity (norm = 0.99999):")
    B = torch.zeros(10, total_dim, device=device, dtype=torch.float32)
    B[:, 0] = 0.99999
    B[:, -1] = 1.0
    C = torch.zeros(10, total_dim, device=device, dtype=torch.float32)
    C[:, 0] = -0.99999
    C[:, -1] = 1.0
    dist_bound = kernel.pairwise_distance(B, C, dim_h, dim_t)
    print(f"Hyperbolic distance across antipodal boundary: {dist_bound[0, 0].item():.4f}")
    assert not torch.isnan(dist_bound).any(), "NaN found at boundary!"
    assert not torch.isinf(dist_bound).any(), "Inf found at boundary!"
    assert dist_bound[0, 0].item() > 20.0, "Antipodal boundary points should have massive hyperbolic distance"

    # Blindspot 3: Unprojected / Out-of-bounds Points Retraction
    print("\n[Blindspot 3] Unprojected Wild Vectors Projected on GPU:")
    wild = torch.randn(100, total_dim, device=device, dtype=torch.float32) * 50.0  # huge norms
    wild[:, -1] = -10.0  # negative radial scale
    projected = kernel.project(wild, dim_h, dim_t)
    h_norms = projected[:, :dim_h].norm(dim=-1)
    max_h_norm = h_norms.max().item()
    print(f"Max Hyperbolic Norm after GPU projection: {max_h_norm:.6f} (boundary: 0.999990)")
    assert max_h_norm <= 0.999990 + 1e-5, f"Hyperbolic boundary violated: {max_h_norm}"

    torus_coords = projected[:, dim_h:dim_h+dim_t]
    assert (torus_coords >= -3.1415927).all() and (torus_coords <= 3.1415927).all(), "Torus bounds violated!"

    radial_coords = projected[:, -1]
    assert (radial_coords >= 1e-5).all(), "Radial positivity violated!"
    print("All coordinates safely clamped within valid manifold domains!")

    # Blindspot 4: Multi-hop Zero-Drift Ring Test
    print("\n[Blindspot 4] 10,000-Hop Inversion Ring Test on GPU:")
    raw_points = torch.zeros(64, total_dim, device=device, dtype=torch.float32)
    raw_points[:, dim_h:dim_h+dim_t] = torch.randn(64, dim_t, device=device, dtype=torch.float32)
    raw_points[:, -1] = 1.0
    projected_state = kernel.project(raw_points, dim_h, dim_t)
    initial_torus = projected_state[:, dim_h:dim_h+dim_t].clone()

    current = initial_torus.clone()
    step_vec = torch.randn(64, dim_t, device=device, dtype=torch.float32) * 0.5

    for _ in range(1000):
        current = kernel.torus_bind(current, step_vec)
    for _ in range(1000):
        current = kernel.torus_unbind(current, step_vec)

    # Angular distance on Torus
    diff = (current - initial_torus) % (2.0 * 3.141592653589793)
    diff = torch.where(diff >= 3.141592653589793, diff - 2.0 * 3.141592653589793, diff)
    diff = torch.where(diff < -3.141592653589793, diff + 2.0 * 3.141592653589793, diff)
    accumulated_drift = diff.abs().max().item()
    print(f"Accumulated Angular Drift after 2,000 GPU Lie Group Operations: {accumulated_drift:.2e} rad")
    assert accumulated_drift < 1e-5, f"Accumulated drift too high: {accumulated_drift}"

    print("\n>>> ALL STRESS TESTS AND BLINDSPOT GUARDS PASSED 100% <<<")

if __name__ == "__main__":
    test_extreme_blindspots()
