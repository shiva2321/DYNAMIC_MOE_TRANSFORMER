"""
Riemannian Geometry and Infinite Spherical Hyperspace Utilities.

Mathematical Framework:
- Points in Infinite Spherical Hyperspace H_S^D reside in S^{D-1} x [0, inf).
- Any point x in R^D can be represented as x = r * u, where:
    u in S^{D-1} (||u|| = 1) is the directional unit vector.
    r in [0, inf) is the unbounded radial coordinate (infinite scale / shell).
- The tangent space at u is:
    T_u S^{D-1} = { v in R^D : <u, v> = 0 }
- The Riemannian Exponential Map exp_u(v) transports u along the geodesic in direction v:
    exp_u(v) = cos(||v||) * u + sin(||v||) * (v / ||v||)
- The Geodesic Distance between u1, u2 on S^{D-1}:
    d_S(u1, u2) = arccos(clamp(<u1, u2>, -1.0, 1.0))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-7

def project_to_sphere(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Projects any non-zero vector onto the unit hypersphere S^{D-1}."""
    norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=EPS)
    return x / norm

def project_to_tangent_space(u: torch.Tensor, v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Projects vector v onto the tangent space T_u S^{D-1}.
    Formula: proj_{T_u}(v) = v - <u, v> * u
    Assumes ||u|| == 1.
    """
    inner_prod = torch.sum(u * v, dim=dim, keepdim=True)
    return v - inner_prod * u

def exp_map_sphere(u: torch.Tensor, v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Riemannian Exponential Map on S^{D-1}:
    Walks along the geodesic starting at u in tangent direction v by distance ||v||.
    exp_u(v) = cos(||v||) * u + sin(||v||) * (v / ||v||)
    Gracefully handles ||v|| -> 0 using Taylor series or sinc.
    """
    # Ensure v is in tangent space
    v_tangent = project_to_tangent_space(u, v, dim=dim)
    v_norm = torch.norm(v_tangent, p=2, dim=dim, keepdim=True)
    
    # Safe sinc(theta) = sin(theta) / theta
    # For small theta: sin(theta)/theta ~ 1 - theta^2/6
    safe_v_norm = v_norm.clamp(min=EPS)
    sin_coeff = torch.where(
        v_norm < 1e-4,
        1.0 - (v_norm ** 2) / 6.0,
        torch.sin(v_norm) / safe_v_norm
    )
    cos_coeff = torch.cos(v_norm)
    
    u_next = cos_coeff * u + sin_coeff * v_tangent
    # Re-normalize for numerical stability
    return project_to_sphere(u_next, dim=dim)

def log_map_sphere(u: torch.Tensor, w: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Riemannian Logarithmic Map on S^{D-1}:
    Computes the initial tangent vector v in T_u S^{D-1} such that exp_u(v) = w.
    log_u(w) = (d / sin(d)) * (w - <u, w> * u) where d = arccos(<u, w>)
    """
    cos_dist = torch.sum(u * w, dim=dim, keepdim=True).clamp(-1.0 + EPS, 1.0 - EPS)
    theta = torch.acos(cos_dist)
    
    diff = project_to_tangent_space(u, w, dim=dim)
    diff_norm = torch.norm(diff, p=2, dim=dim, keepdim=True).clamp(min=EPS)
    
    direction = diff / diff_norm
    return theta * direction

def geodesic_distance_sphere(u1: torch.Tensor, u2: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Geodesic angle distance on S^{D-1} in radians [0, pi]."""
    cos_sim = torch.sum(u1 * u2, dim=dim, keepdim=True).clamp(-1.0 + EPS, 1.0 - EPS)
    return torch.acos(cos_sim)

def spherical_to_cartesian(u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Converts spherical coordinates (u, r) in S^{D-1} x [0, inf) to Cartesian x in R^D."""
    return r * u

def cartesian_to_spherical(x: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts Cartesian x in R^D to spherical coordinates (u, r)."""
    r = torch.norm(x, p=2, dim=dim, keepdim=True)
    u = torch.where(r < EPS, torch.zeros_like(x), x / r.clamp(min=EPS))
    # If r is near 0, default u to first standard basis
    if (r < EPS).any():
        u[..., 0] = torch.where(r.squeeze(dim) < EPS, torch.tensor(1.0, device=x.device), u[..., 0])
    return u, r

def parallel_transport_sphere(u: torch.Tensor, w: torch.Tensor, v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Riemannian Parallel Transport on S^{D-1}:
    Transports a tangent vector v in T_u S^{D-1} along the unique minimizing geodesic to T_w S^{D-1}.
    Formula:
        P_{u -> w}(v) = v - (<w, v> / (1 + <u, w>)) * (u + w)
    Preserves vector norm ||v|| and inner products (isometry).
    """
    # Ensure v is tangent at u
    v_tangent = project_to_tangent_space(u, v, dim=dim)
    
    uw_dot = torch.sum(u * w, dim=dim, keepdim=True).clamp(-1.0 + EPS, 1.0)
    wv_dot = torch.sum(w * v_tangent, dim=dim, keepdim=True)
    
    denominator = (1.0 + uw_dot).clamp(min=EPS)
    correction = (wv_dot / denominator) * (u + w)
    transported = v_tangent - correction
    
    # Project strictly to T_w S^{D-1} for numerical safety
    return project_to_tangent_space(w, transported, dim=dim)

def analogical_transfer_sphere(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Riemannian Analogy Functor on S^{D-1}:
    Solves A : B :: C : D (i.e. 'A is to B as C is to D').
    Transports the relational transformation v = Log_A(B) along the geodesic from A to C,
    then evaluates D = Exp_C(P_{A -> C}(v)).
    """
    # 1. Relational tangent direction from A to B
    v_rel = log_map_sphere(a, b, dim=dim)
    
    # 2. Parallel transport the relational vector to C's tangent space
    v_transported = parallel_transport_sphere(a, c, v_rel, dim=dim)
    
    # 3. Apply the transported relation to C
    d = exp_map_sphere(c, v_transported, dim=dim)
    return project_to_sphere(d, dim=dim)
