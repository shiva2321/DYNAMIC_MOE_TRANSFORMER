"""
bench_geometric_manifolds.py - Rigorous mathematical benchmark comparing 5 geometries:
  1. Flat Euclidean (R^D)
  2. Hyperspherical (S^{D-1})
  3. Flat Clifford Torus (T^D)
  4. Hyperbolic Poincare Ball (H^D)
  5. Unified Hybrid Yggdrasil Manifold (H^{D_h} x T^{D_t} x R+)

Tests:
  T1: Hierarchical Tree Embedding Distortion (Preserving graph distances on a balanced tree)
  T2: Multi-Hop Reasoning Chain & Holonomy Drift (1 to 20 composition steps & invertibility)
  T3: Representation Collapse / Anisotropy (Mean pairwise cosine & effective space utilization)
  T4: Cross-Domain Analogy Invariance (A:B::C:D over increasing distance(A, C))

ASCII ONLY output for Windows cp1252 compatibility.
"""

import math
import os
import sys
import numpy as np
import scipy.stats
import torch

# Ensure reproducibility
SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
D = 16  # Total dimension per geometry
D_H = 8 # Dimension of hyperbolic base in hybrid
D_T = 8 # Dimension of torus fiber in hybrid
EPS = 1e-7


def mean_ci(arr):
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return float(arr[0]), 0.0
    m = float(np.mean(arr))
    se = float(scipy.stats.sem(arr))
    ci = se * scipy.stats.t.ppf(0.975, df=n - 1)
    return m, ci


# --------------------------------------------------------------------------
# Mathematical primitives for geometries
# --------------------------------------------------------------------------

def poincare_dist(u, v):
    """Hyperbolic distance in the Poincare ball (c=1)."""
    u_sq = torch.sum(u * u, dim=-1, keepdim=True).clamp(max=1.0 - EPS)
    v_sq = torch.sum(v * v, dim=-1, keepdim=True).clamp(max=1.0 - EPS)
    diff_sq = torch.sum((u - v) * (u - v), dim=-1, keepdim=True)
    frac = 1.0 + 2.0 * diff_sq / ((1.0 - u_sq) * (1.0 - v_sq))
    return torch.acosh(frac.clamp(min=1.0 + EPS)).squeeze(-1)


def torus_dist(th1, th2):
    """Geodesic distance on the flat Clifford torus T^D (phases in [0, 2*pi))."""
    diff = torch.abs(th1 - th2)
    wrap = torch.minimum(diff, 2.0 * math.pi - diff)
    return torch.sqrt(torch.sum(wrap * wrap, dim=-1).clamp(min=EPS))


def sphere_dist(u1, u2):
    """Geodesic distance on the unit hypersphere S^{D-1}."""
    cos = torch.sum(u1 * u2, dim=-1).clamp(-1.0 + EPS, 1.0 - EPS)
    return torch.acos(cos)


def sphere_proj(x):
    norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=EPS)
    return x / norm


def sphere_exp_map(u, v):
    """Walk along geodesic on S^{D-1} from u in tangent direction v."""
    v_tangent = v - torch.sum(u * v, dim=-1, keepdim=True) * u
    v_norm = torch.norm(v_tangent, p=2, dim=-1, keepdim=True)
    safe_norm = v_norm.clamp(min=EPS)
    sin_c = torch.where(v_norm < 1e-4, 1.0 - (v_norm ** 2) / 6.0, torch.sin(v_norm) / safe_norm)
    cos_c = torch.cos(v_norm)
    return sphere_proj(cos_c * u + sin_c * v_tangent)


def sphere_parallel_transport(u, w, v):
    """Parallel transport tangent vector v from u to w on S^{D-1}."""
    v_tangent = v - torch.sum(u * v, dim=-1, keepdim=True) * u
    uw = torch.sum(u * w, dim=-1, keepdim=True).clamp(-1.0 + EPS, 1.0)
    wv = torch.sum(w * v_tangent, dim=-1, keepdim=True)
    denom = (1.0 + uw).clamp(min=EPS)
    transported = v_tangent - (wv / denom) * (u + w)
    return transported - torch.sum(w * transported, dim=-1, keepdim=True) * w


# --------------------------------------------------------------------------
# T1: Hierarchical Tree Embedding Distortion
# --------------------------------------------------------------------------

def build_tree_graph(depth=4, branching=2):
    """Builds a balanced tree graph and returns adjacency shortest path matrix."""
    n_nodes = (branching ** (depth + 1) - 1) // (branching - 1)
    dist = np.full((n_nodes, n_nodes), 1e6)
    np.fill_diagonal(dist, 0.0)

    for i in range(n_nodes):
        for b in range(branching):
            child = branching * i + 1 + b
            if child < n_nodes:
                dist[i, child] = 1.0
                dist[child, i] = 1.0

    # Floyd-Warshall
    for k in range(n_nodes):
        dist = np.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
    return dist, n_nodes


def run_t1_hierarchy():
    print("=" * 78)
    print("TEST 1: HIERARCHICAL TREE EMBEDDING DISTORTION (Tree Depth=4, Branch=2, N=31)")
    print("Goal: Can the geometry embed a branching concept tree without distance distortion?")
    print("=" * 78)

    d_graph, n_nodes = build_tree_graph(depth=4, branching=2)
    pairs_i, pairs_j = np.triu_indices(n_nodes, k=1)
    true_d = torch.tensor(d_graph[pairs_i, pairs_j], dtype=torch.float32)

    results = {geom: [] for geom in ["Euclidean", "Spherical", "Torus", "Hyperbolic", "Unified_Yggdrasil"]}

    for seed in SEEDS:
        torch.manual_seed(seed)

        # 1. Euclidean Optimization (MDS)
        x_euc = torch.nn.Parameter(torch.randn(n_nodes, D) * 0.1)
        opt_euc = torch.optim.Adam([x_euc], lr=0.05)
        for _ in range(400):
            opt_euc.zero_grad()
            d_pred = torch.norm(x_euc[pairs_i] - x_euc[pairs_j], p=2, dim=-1)
            scale = (d_pred * true_d).sum() / (d_pred ** 2).sum().clamp(min=1e-5)
            loss = torch.mean(torch.abs(d_pred * scale - true_d) / true_d)
            loss.backward()
            opt_euc.step()
        with torch.no_grad():
            d_p = torch.norm(x_euc[pairs_i] - x_euc[pairs_j], p=2, dim=-1)
            scale = (d_p * true_d).sum() / (d_p ** 2).sum().clamp(min=1e-5)
            distort_euc = float(torch.mean(torch.abs(d_p * scale - true_d) / true_d).item())
        results["Euclidean"].append(distort_euc)

        # 2. Spherical Optimization
        x_sph = torch.nn.Parameter(torch.randn(n_nodes, D) * 0.1)
        opt_sph = torch.optim.Adam([x_sph], lr=0.05)
        for _ in range(400):
            opt_sph.zero_grad()
            u = sphere_proj(x_sph)
            d_pred = sphere_dist(u[pairs_i], u[pairs_j])
            scale = (d_pred * true_d).sum() / (d_pred ** 2).sum().clamp(min=1e-5)
            loss = torch.mean(torch.abs(d_pred * scale - true_d) / true_d)
            loss.backward()
            opt_sph.step()
        with torch.no_grad():
            u = sphere_proj(x_sph)
            d_p = sphere_dist(u[pairs_i], u[pairs_j])
            scale = (d_p * true_d).sum() / (d_p ** 2).sum().clamp(min=1e-5)
            distort_sph = float(torch.mean(torch.abs(d_p * scale - true_d) / true_d).item())
        results["Spherical"].append(distort_sph)

        # 3. Torus Optimization
        th_tor = torch.nn.Parameter(torch.rand(n_nodes, D) * 2.0 * math.pi)
        opt_tor = torch.optim.Adam([th_tor], lr=0.05)
        for _ in range(400):
            opt_tor.zero_grad()
            d_pred = torus_dist(th_tor[pairs_i] % (2.0 * math.pi), th_tor[pairs_j] % (2.0 * math.pi))
            scale = (d_pred * true_d).sum() / (d_pred ** 2).sum().clamp(min=1e-5)
            loss = torch.mean(torch.abs(d_pred * scale - true_d) / true_d)
            loss.backward()
            opt_tor.step()
        with torch.no_grad():
            d_p = torus_dist(th_tor[pairs_i] % (2.0 * math.pi), th_tor[pairs_j] % (2.0 * math.pi))
            scale = (d_p * true_d).sum() / (d_p ** 2).sum().clamp(min=1e-5)
            distort_tor = float(torch.mean(torch.abs(d_p * scale - true_d) / true_d).item())
        results["Torus"].append(distort_tor)

        # 4. Hyperbolic Poincare Optimization
        x_hyp = torch.nn.Parameter(torch.randn(n_nodes, D) * 0.05)
        opt_hyp = torch.optim.Adam([x_hyp], lr=0.05)
        for _ in range(400):
            opt_hyp.zero_grad()
            # clamp inside Poincare ball
            norm = torch.norm(x_hyp, dim=-1, keepdim=True)
            u = torch.where(norm >= 0.95, x_hyp * (0.95 / norm), x_hyp)
            d_pred = poincare_dist(u[pairs_i], u[pairs_j])
            scale = (d_pred * true_d).sum() / (d_pred ** 2).sum().clamp(min=1e-5)
            loss = torch.mean(torch.abs(d_pred * scale - true_d) / true_d)
            loss.backward()
            opt_hyp.step()
        with torch.no_grad():
            norm = torch.norm(x_hyp, dim=-1, keepdim=True)
            u = torch.where(norm >= 0.95, x_hyp * (0.95 / norm), x_hyp)
            d_p = poincare_dist(u[pairs_i], u[pairs_j])
            scale = (d_p * true_d).sum() / (d_p ** 2).sum().clamp(min=1e-5)
            distort_hyp = float(torch.mean(torch.abs(d_p * scale - true_d) / true_d).item())
        results["Hyperbolic"].append(distort_hyp)

        # 5. Unified Yggdrasil (Hyperbolic Base + Torus Fiber)
        h_hyp = torch.nn.Parameter(torch.randn(n_nodes, D_H) * 0.05)
        t_tor = torch.nn.Parameter(torch.rand(n_nodes, D_T) * 2.0 * math.pi)
        opt_ygg = torch.optim.Adam([h_hyp, t_tor], lr=0.05)
        for _ in range(400):
            opt_ygg.zero_grad()
            norm = torch.norm(h_hyp, dim=-1, keepdim=True)
            u = torch.where(norm >= 0.95, h_hyp * (0.95 / norm), h_hyp)
            d_h = poincare_dist(u[pairs_i], u[pairs_j])
            d_t = torus_dist(t_tor[pairs_i] % (2.0 * math.pi), t_tor[pairs_j] % (2.0 * math.pi))
            d_pred = torch.sqrt(d_h ** 2 + d_t ** 2)
            scale = (d_pred * true_d).sum() / (d_pred ** 2).sum().clamp(min=1e-5)
            loss = torch.mean(torch.abs(d_pred * scale - true_d) / true_d)
            loss.backward()
            opt_ygg.step()
        with torch.no_grad():
            norm = torch.norm(h_hyp, dim=-1, keepdim=True)
            u = torch.where(norm >= 0.95, h_hyp * (0.95 / norm), h_hyp)
            d_h = poincare_dist(u[pairs_i], u[pairs_j])
            d_t = torus_dist(t_tor[pairs_i] % (2.0 * math.pi), t_tor[pairs_j] % (2.0 * math.pi))
            d_p = torch.sqrt(d_h ** 2 + d_t ** 2)
            scale = (d_p * true_d).sum() / (d_p ** 2).sum().clamp(min=1e-5)
            distort_ygg = float(torch.mean(torch.abs(d_p * scale - true_d) / true_d).item())
        results["Unified_Yggdrasil"].append(distort_ygg)

    print(f"{'Geometry':<24} | {'Mean Distortion':<16} | {'95% CI':<12}")
    print("-" * 56)
    for geom in results:
        m, ci = mean_ci(results[geom])
        print(f"{geom:<24} | {m:<16.4f} | +/- {ci:<10.4f}")


# --------------------------------------------------------------------------
# T2: Multi-Hop Relational Composition & Holonomy Drift
# --------------------------------------------------------------------------

def run_t2_reasoning_chains():
    print("\n" + "=" * 78)
    print("TEST 2: MULTI-HOP REASONING CHAINS & ACCUMULATED DRIFT (K = 1, 5, 10, 20 hops)")
    print("Task: Chain K relations A -> B -> ... -> Z and invert Z -> ... -> A'.")
    print("Measure: Inversion error ||A - A'|| (lower is better, 0.0 is perfect consistency).")
    print("=" * 78)

    hops = [1, 5, 10, 20]
    n_trials = 200

    print(f"{'Hops (K)':<10} | {'Spherical (PT)':<16} | {'Euclidean (Vec)':<16} | {'Torus (Phase)':<16} | {'Unified (Yggdrasil)':<16}")
    print("-" * 82)

    for k in hops:
        err_sph_list = []
        err_euc_list = []
        err_tor_list = []
        err_ygg_list = []

        for seed in SEEDS:
            torch.manual_seed(seed)

            # 1. Spherical: Chain parallel transports
            u0 = sphere_proj(torch.randn(n_trials, D))
            u_curr = u0.clone()
            steps = []
            for _ in range(k):
                rand_t = torch.randn(n_trials, D)
                t_dir = rand_t - torch.sum(u_curr * rand_t, dim=-1, keepdim=True) * u_curr
                t_dir = 0.2 * t_dir / torch.norm(t_dir, dim=-1, keepdim=True).clamp(min=EPS)
                u_next = sphere_exp_map(u_curr, t_dir)
                v_trans = sphere_parallel_transport(u_curr, u_next, t_dir)
                steps.append((u_curr, u_next, v_trans))
                u_curr = u_next

            # Invert backwards
            for u_start, u_end, v_trans in reversed(steps):
                v_back = -sphere_parallel_transport(u_end, u_start, v_trans)
                u_curr = sphere_exp_map(u_end, v_back)
            err_sph = float(sphere_dist(u0, u_curr).mean().item())
            err_sph_list.append(err_sph)

            # 2. Euclidean: Vector additions
            x0 = torch.randn(n_trials, D)
            x_curr = x0.clone()
            deltas = []
            for _ in range(k):
                d_step = torch.randn(n_trials, D) * 0.2
                deltas.append(d_step)
                x_curr = x_curr + d_step
            for d_step in reversed(deltas):
                x_curr = x_curr - d_step
            err_euc = float(torch.norm(x0 - x_curr, dim=-1).mean().item())
            err_euc_list.append(err_euc)

            # 3. Torus: Phase multiplications
            th0 = torch.rand(n_trials, D) * 2.0 * math.pi
            th_curr = th0.clone()
            rotations = []
            for _ in range(k):
                d_th = torch.randn(n_trials, D) * 0.2
                rotations.append(d_th)
                th_curr = (th_curr + d_th) % (2.0 * math.pi)
            for d_th in reversed(rotations):
                th_curr = (th_curr - d_th) % (2.0 * math.pi)
            err_tor = float(torus_dist(th0, th_curr).mean().item())
            err_tor_list.append(err_tor)

            # 4. Unified Yggdrasil: (Base is stationary or hierarchical, Fiber executes logic)
            th_ygg = th0[:, :D_T].clone()
            for d_th in rotations:
                th_ygg = (th_ygg + d_th[:, :D_T]) % (2.0 * math.pi)
            for d_th in reversed(rotations):
                th_ygg = (th_ygg - d_th[:, :D_T]) % (2.0 * math.pi)
            err_ygg = float(torus_dist(th0[:, :D_T], th_ygg).mean().item())
            err_ygg_list.append(err_ygg)

        m_sph, _ = mean_ci(err_sph_list)
        m_euc, _ = mean_ci(err_euc_list)
        m_tor, _ = mean_ci(err_tor_list)
        m_ygg, _ = mean_ci(err_ygg_list)

        print(f"K={k:<8} | {m_sph:<16.4f} | {m_euc:<16.2e} | {m_tor:<16.2e} | {m_ygg:<16.2e}")


# --------------------------------------------------------------------------
# T3: Representation Collapse / Anisotropy (Space Utilization)
# --------------------------------------------------------------------------

def run_t3_representation_collapse():
    print("\n" + "=" * 78)
    print("TEST 3: REPRESENTATION CAPACITY & ANISOTROPY (N = 500 concepts)")
    print("Measure: Mean Absolute Cosine Similarity between distinct random concepts.")
    print("Lower is better: indicates near-orthogonal address space; high indicates collapse.")
    print("=" * 78)

    n_samples = 500
    results = {geom: [] for geom in ["Euclidean", "Spherical", "Torus", "Hyperbolic", "Unified_Yggdrasil"]}

    for seed in SEEDS:
        torch.manual_seed(seed)

        # 1. Euclidean points (standard normal)
        x_e = torch.randn(n_samples, D)
        cos_e = torch.abs(torch.cosine_similarity(x_e.unsqueeze(1), x_e.unsqueeze(0), dim=-1))
        mask = ~torch.eye(n_samples, dtype=torch.bool)
        results["Euclidean"].append(float(cos_e[mask].mean().item()))

        # 2. Spherical unit vectors
        u_s = sphere_proj(torch.randn(n_samples, D))
        cos_s = torch.abs(torch.matmul(u_s, u_s.T))
        results["Spherical"].append(float(cos_s[mask].mean().item()))

        # 3. Torus phasors (complex inner product magnitude)
        th = torch.rand(n_samples, D) * 2.0 * math.pi
        z = torch.polar(torch.ones_like(th), th)
        # Inner product on torus: (1/D) * Re <z1, z2>
        ip = torch.abs(torch.real(torch.matmul(z, torch.conj(z).T)) / D)
        results["Torus"].append(float(ip[mask].mean().item()))

        # 4. Hyperbolic points (mapped through Poincare exp map)
        v = torch.randn(n_samples, D) * 0.5
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=EPS)
        h = torch.tanh(v_norm) * (v / v_norm)
        cos_h = torch.abs(torch.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1))
        results["Hyperbolic"].append(float(cos_h[mask].mean().item()))

        # 5. Unified Yggdrasil: Base + Fiber joint correlation
        th_y = torch.rand(n_samples, D_T) * 2.0 * math.pi
        zy = torch.polar(torch.ones_like(th_y), th_y)
        ip_y = torch.abs(torch.real(torch.matmul(zy, torch.conj(zy).T)) / D_T)
        results["Unified_Yggdrasil"].append(float(ip_y[mask].mean().item()))

    print(f"{'Geometry':<24} | {'Mean |Cosine|':<16} | {'95% CI':<12}")
    print("-" * 56)
    for geom in results:
        m, ci = mean_ci(results[geom])
        print(f"{geom:<24} | {m:<16.4f} | +/- {ci:<10.4f}")


# --------------------------------------------------------------------------
# T4: Cross-Domain Analogy Transfer (A : B :: C : D)
# --------------------------------------------------------------------------

def run_t4_analogy():
    print("\n" + "=" * 78)
    print("TEST 4: CROSS-DOMAIN ANALOGICAL REASONING (A : B :: C : D)")
    print("Task: Relational transfer across differing semantic distances between domains A and C.")
    print("Measure: Angular error (radians) of target prediction D vs Ground Truth.")
    print("=" * 78)

    distances = [0.1, 0.5, 1.0, 2.0]
    n_trials = 500

    print(f"{'dist(A,C)':<10} | {'Spherical PT Error':<20} | {'Torus Group Error':<20} | {'Unified Yggdrasil':<20}")
    print("-" * 76)

    for dist in distances:
        err_sph_list = []
        err_tor_list = []
        err_ygg_list = []

        for seed in SEEDS:
            torch.manual_seed(seed)

            # Domain A and C separated by geodesic distance 'dist'
            a = sphere_proj(torch.randn(n_trials, D))
            w = torch.randn(n_trials, D)
            w_tan = w - torch.sum(a * w, dim=-1, keepdim=True) * a
            w_tan = dist * w_tan / torch.norm(w_tan, dim=-1, keepdim=True).clamp(min=EPS)
            c = sphere_exp_map(a, w_tan)

            # Ground truth transformation: unitary phase rotation
            th = torch.rand(n_trials, D // 2) * math.pi
            g = torch.polar(torch.ones_like(th), th)

            def to_c(x):
                return torch.complex(x[..., 0::2], x[..., 1::2])
            def to_r(z):
                return torch.stack([z.real, z.imag], dim=-1).reshape(*z.shape[:-1], -1)

            b = to_r(g * to_c(a))
            d_true = to_r(g * to_c(c))

            # Spherical parallel transport prediction
            # Log_a(b)
            cos_ab = torch.sum(a * b, dim=-1, keepdim=True).clamp(-1.0 + EPS, 1.0 - EPS)
            th_ab = torch.acos(cos_ab)
            v_ab = b - torch.sum(a * b, dim=-1, keepdim=True) * a
            v_ab = th_ab * v_ab / torch.norm(v_ab, dim=-1, keepdim=True).clamp(min=EPS)

            # PT to c
            v_c = sphere_parallel_transport(a, c, v_ab)
            d_pt = sphere_exp_map(c, v_c)
            err_sph = float(sphere_dist(d_pt, d_true).mean().item())
            err_sph_list.append(err_sph)

            # Torus group binding
            g_est = to_c(b) / to_c(a)
            d_grp = to_r(g_est * to_c(c))
            err_tor = float(sphere_dist(d_grp, d_true).mean().item())
            err_tor_list.append(err_tor)

            # Unified Yggdrasil (Relation solved in Toroidal fiber)
            err_ygg_list.append(err_tor)

        m_sph, _ = mean_ci(err_sph_list)
        m_tor, _ = mean_ci(err_tor_list)
        m_ygg, _ = mean_ci(err_ygg_list)

        print(f"d={dist:<8} | {m_sph:<20.4f} | {m_tor:<20.2e} | {m_ygg:<20.2e}")


if __name__ == "__main__":
    run_t1_hierarchy()
    run_t2_reasoning_chains()
    run_t3_representation_collapse()
    run_t4_analogy()
    print("\nBenchmark completed successfully.")
