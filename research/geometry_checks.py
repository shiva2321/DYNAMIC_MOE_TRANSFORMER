"""
geometry_checks.py - tests the *assumptions* behind two ways of solving A:B::C:D, using the repo's own
ishmn/geometry.py (loaded by file path, so the heavy ishmn package __init__ is not imported).

  PT   = ishmn.geometry.analogical_transfer_sphere  (Log_A(B) parallel-transported to C, then Exp_C)
  GRP  = group-binding analogy: estimate the group element g = B * A^-1, apply to C   (VSA / FHRR mapping vector)

Run: python research/geometry_checks.py
Writes research/results/geometry_checks.json

H1  holonomy: parallel transport around a closed geodesic triangle on S^{D-1} rotates the vector by the
    enclosed area (curvature +1).  A flat torus (abelian group action) has zero holonomy.
H2  analogy under a TRUE symmetry relation (diagonal complex phases, T^4 acting on R^8 = C^4):
    GRP is exact for every (A, C); PT error grows with the distance between A and C.
H3  analogy under a RELATION that is a rotation in a random 2-plane of R^8 (SO(8) element):
    one (A, B) pair does not determine the rotation -> both methods are under-determined (expected failure).
"""
import importlib.util
import json
import math
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("ishmn_geometry", os.path.join(ROOT, "ishmn", "geometry.py"))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

torch.manual_seed(0)
rng = np.random.default_rng(0)
D = 8
DT = torch.float64
OUT = {}


def rand_unit(n):
    return G.project_to_sphere(torch.randn(n, D, dtype=DT))


def geo(a, b):
    return G.geodesic_distance_sphere(a, b).squeeze(-1)


def angle_between(u, v):
    c = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1))
    return torch.acos(c.clamp(-1, 1))


# ---------------------------------------------------------------- H1 holonomy
print("H1  holonomy of parallel transport around a closed geodesic triangle p->q->r->p on S^7")
print(f"{'eps':>6} {'mean triangle area':>20} {'mean holonomy angle':>21} {'max |angle-area|':>18}")
rows = []
for eps in (0.05, 0.1, 0.2, 0.4, 0.8):
    n = 400
    p = rand_unit(n)
    t1 = G.project_to_tangent_space(p, torch.randn(n, D, dtype=DT))
    t2 = G.project_to_tangent_space(p, torch.randn(n, D, dtype=DT))
    t1 = t1 / t1.norm(dim=-1, keepdim=True)
    t2 = t2 / t2.norm(dim=-1, keepdim=True)
    q = G.exp_map_sphere(p, eps * t1)
    r = G.exp_map_sphere(p, eps * t2)
    v = G.log_map_sphere(p, q)  # lies in the triangle's plane: maximally rotated by holonomy
    v1 = G.parallel_transport_sphere(p, q, v)
    v2 = G.parallel_transport_sphere(q, r, v1)
    v3 = G.parallel_transport_sphere(r, p, v2)
    hol = angle_between(v, v3)
    # spherical excess = area (unit sphere)
    a_, b_, c_ = geo(q, r), geo(p, r), geo(p, q)
    s = (a_ + b_ + c_) / 2
    tn = torch.sqrt(torch.clamp(torch.tan(s / 2) * torch.tan((s - a_) / 2) * torch.tan((s - b_) / 2) * torch.tan((s - c_) / 2), min=0))
    area = 4 * torch.atan(tn)  # L'Huilier
    rows.append(dict(eps=eps, area=float(area.mean()), holonomy=float(hol.mean()), max_dev=float((hol - area).abs().max())))
    print(f"{eps:>6} {area.mean():>20.6f} {hol.mean():>21.6f} {(hol - area).abs().max():>18.2e}")
OUT["H1"] = rows
print("  -> holonomy angle == enclosed area (curvature +1): composition of transported relations is PATH-DEPENDENT.")
print("  -> for an abelian group action (phasor binding) the holonomy is identically 0: g_pq g_qr g_rp = 1.")

# ---------------------------------------------------------------- H2 torus-action relation
print("\nH2  relation = element of T^4 acting on R^8=C^4 (a true symmetry).  error = geodesic dist(prediction, truth)")


def to_c(x):
    return torch.complex(x[..., 0::2], x[..., 1::2])


def to_r(z):
    out = torch.stack([z.real, z.imag], dim=-1)
    return out.reshape(*z.shape[:-1], -1)


print(f"{'dist(A,C)':>10} {'PT error':>10} {'GRP error':>11}   (rel. size ||phases|| ~ U[0,pi])")
rows = []
for phi in (0.05, 0.2, 0.5, 1.0, 1.5, 2.5):
    n = 1000
    a = rand_unit(n)
    w = G.project_to_tangent_space(a, torch.randn(n, D, dtype=DT))
    w = w / w.norm(dim=-1, keepdim=True)
    c = G.exp_map_sphere(a, phi * w)
    th = torch.rand(n, D // 2, dtype=DT) * math.pi
    g = torch.polar(torch.ones_like(th), th)
    b = to_r(g * to_c(a))
    d_true = to_r(g * to_c(c))
    d_pt = G.analogical_transfer_sphere(a, b, c)
    g_hat = to_c(b) / to_c(a)  # exact unbinding
    d_grp = to_r(g_hat * to_c(c))
    e_pt, e_grp = geo(d_pt, d_true).mean(), geo(d_grp, d_true).mean()
    rows.append(dict(dist_AC=phi, pt=float(e_pt), grp=float(e_grp)))
    print(f"{phi:>10} {e_pt:>10.4f} {e_grp:>11.2e}")
OUT["H2"] = rows

# ---------------------------------------------------------------- H3 random-plane rotation relation
print("\nH3  relation = rotation by angle t in a random 2-plane of R^8 (SO(8) element), single (A,B) example")
rows = []
for phi in (0.2, 1.0, 2.0):
    n = 1000
    a = rand_unit(n)
    w = G.project_to_tangent_space(a, torch.randn(n, D, dtype=DT))
    w = w / w.norm(dim=-1, keepdim=True)
    c = G.exp_map_sphere(a, phi * w)
    # random plane via two orthonormal vectors, rotation angle 1.0 rad
    u1 = rand_unit(n)
    u2 = G.project_to_tangent_space(u1, torch.randn(n, D, dtype=DT))
    u2 = u2 / u2.norm(dim=-1, keepdim=True)
    t = 1.0

    def rot(x):
        x1, x2 = (x * u1).sum(-1, keepdim=True), (x * u2).sum(-1, keepdim=True)
        return x + (math.cos(t) - 1) * (x1 * u1 + x2 * u2) + math.sin(t) * (x1 * u2 - x2 * u1)

    b, d_true = rot(a), rot(c)
    d_pt = G.analogical_transfer_sphere(a, b, c)
    e_pt = geo(d_pt, d_true).mean()
    e_null = geo(c, d_true).mean()  # predict "no change"
    rows.append(dict(dist_AC=phi, pt=float(e_pt), identity=float(e_null)))
    print(f"dist(A,C)={phi:>4}: PT error {e_pt:.4f}   vs  'do nothing' baseline {e_null:.4f}")
OUT["H3"] = rows
print("  -> with an SO(8)-sized hypothesis class one example does not identify the relation; the inductive bias is the GROUP.")

os.makedirs(os.path.join(ROOT, "research", "results"), exist_ok=True)
with open(os.path.join(ROOT, "research", "results", "geometry_checks.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("\nwrote research/results/geometry_checks.json")
