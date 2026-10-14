"""
Unit tests for the isolated geometry package:
  - PoincareBall (H^D)
  - CliffordTorus (T^D)
  - Hypersphere (S^{D-1})
  - YggdrasilManifold (H^{D_h} x T^{D_t} x R+)
"""

import math
import pytest
import torch
import numpy as np

from geometry import (
    PoincareBall,
    CliffordTorus,
    Hypersphere,
    YggdrasilManifold,
    YggdrasilState,
    YggdrasilAction,
)

EPS = 1e-6


# --------------------------------------------------------------------------
# Poincare Ball Tests
# --------------------------------------------------------------------------

def test_poincare_projection_and_distance():
    pb = PoincareBall(dim=8, c=1.0)
    # Outside ball points must be projected inside
    outside = torch.randn(10, 8) * 5.0
    projected = pb.project(outside)
    norms = torch.norm(projected, dim=-1)
    assert torch.all(norms <= pb.max_norm() + 1e-6)

    # Distance to self must be exact zero
    d_self = pb.distance(projected, projected)
    assert torch.allclose(d_self, torch.zeros_like(d_self), atol=1e-6)

    # Symmetry: d(u, v) == d(v, u)
    u = pb.project(torch.randn(10, 8) * 0.5)
    v = pb.project(torch.randn(10, 8) * 0.5)
    d_uv = pb.distance(u, v)
    d_vu = pb.distance(v, u)
    assert torch.allclose(d_uv, d_vu, atol=1e-5)
    assert torch.all(d_uv >= 0.0)


def test_poincare_mobius_addition():
    pb = PoincareBall(dim=8, c=1.0)
    # Identity: u (+) 0 == u
    u = pb.project(torch.randn(5, 8) * 0.4)
    zero = torch.zeros_like(u)
    res = pb.mobius_add(u, zero)
    assert torch.allclose(res, u, atol=1e-5)

    # Inverse: u (+) (-u) == 0
    inv_res = pb.mobius_add(u, -u)
    assert torch.allclose(inv_res, zero, atol=1e-5)


def test_poincare_exp_log_maps():
    pb = PoincareBall(dim=8, c=1.0)
    # At origin
    v = torch.randn(5, 8) * 0.3
    x = pb.exp_map_zero(v)
    v_rec = pb.log_map_zero(x)
    assert torch.allclose(v, v_rec, atol=1e-4)

    # At general interior point
    p = pb.project(torch.randn(5, 8) * 0.1)
    tan_v = torch.randn(5, 8) * 0.05
    y = pb.exp_map(p, tan_v)
    tan_v_rec = pb.log_map(p, y)
    assert torch.allclose(tan_v, tan_v_rec, atol=1e-4)


# --------------------------------------------------------------------------
# Clifford Torus Tests
# --------------------------------------------------------------------------

def test_torus_wrap_and_distance():
    torus = CliffordTorus(dim=8)
    angles = torch.tensor([0.0, math.pi, 2 * math.pi, 3 * math.pi, -0.5])
    wrapped = torus.wrap(angles)
    assert torch.all(wrapped >= 0.0)
    assert torch.all(wrapped < 2 * math.pi)

    # Distance to self is exactly 0
    th = torch.rand(10, 8) * 2 * math.pi
    d_self = torus.distance(th, th)
    assert torch.allclose(d_self, torch.zeros_like(d_self), atol=1e-6)

    # Wrap-around distance: distance between 0.1 and 2*pi - 0.1 is 0.2
    th1 = torch.full((1, 8), 0.1)
    th2 = torch.full((1, 8), 2 * math.pi - 0.1)
    d = torus.distance(th1, th2)
    expected = math.sqrt(8 * (0.2 ** 2))
    assert math.isclose(d.item(), expected, rel_tol=1e-5)


def test_torus_binding_and_inversion():
    torus = CliffordTorus(dim=8)
    a = torch.rand(10, 8) * 2 * math.pi
    b = torch.rand(10, 8) * 2 * math.pi

    # Exact unbinding: (a (+) b) (-) b == a
    compound = torus.bind(a, b)
    recovered_a = torus.unbind(compound, b)
    d_err = torus.distance(a, recovered_a)
    assert torch.all(d_err < 1e-5)

    # Invert property: a (+) (-a) == 0 (distance on torus is 0)
    inv_a = torus.invert(a)
    zero_approx = torus.bind(a, inv_a)
    d_zero = torus.distance(zero_approx, torch.zeros_like(zero_approx))
    assert torch.all(d_zero < 1e-5)


def test_torus_zero_holonomy_over_long_chains():
    torus = CliffordTorus(dim=8)
    th0 = torch.rand(20, 8) * 2 * math.pi
    curr = th0.clone()

    # Apply 100 random rotations
    rotations = [torch.randn(20, 8) for _ in range(100)]
    for r in rotations:
        curr = torus.bind(curr, r)

    # Revert all 100 rotations in reverse order
    for r in reversed(rotations):
        curr = torus.unbind(curr, r)

    d_drift = torus.distance(th0, curr)
    # Must have zero drift (within numerical float precision)
    assert torch.all(d_drift < 1e-5)


# --------------------------------------------------------------------------
# Hypersphere Tests
# --------------------------------------------------------------------------

def test_hypersphere_projection_and_distance():
    hs = Hypersphere(dim=8)
    x = torch.randn(10, 8) * 3.0
    u = hs.project(x)
    norms = torch.norm(u, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    d_self = hs.distance(u, u)
    assert torch.allclose(d_self, torch.zeros_like(d_self), atol=1e-6)

    # Opposite vectors have distance pi
    u_opp = -u
    d_pi = hs.distance(u, u_opp)
    assert torch.allclose(d_pi, torch.full_like(d_pi, math.pi), atol=1e-4)


# --------------------------------------------------------------------------
# Yggdrasil Manifold Tests
# --------------------------------------------------------------------------

def test_yggdrasil_state_and_distance():
    ym = YggdrasilManifold(d_h=8, d_t=8, c=1.0)
    h = torch.randn(5, 8) * 0.4
    theta = torch.rand(5, 8) * 2 * math.pi
    r = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

    state = ym.project_state(h, theta, r)
    assert state.h.shape == (5, 8)
    assert state.theta.shape == (5, 8)
    assert state.r.shape == (5, 1)

    d_self = ym.distance(state, state)
    assert torch.allclose(d_self, torch.zeros_like(d_self), atol=1e-6)


def test_yggdrasil_exact_reversibility():
    ym = YggdrasilManifold(d_h=8, d_t=8, c=1.0)
    h0 = torch.randn(10, 8) * 0.3
    th0 = torch.rand(10, 8) * 2 * math.pi
    r0 = torch.full((10, 1), 2.0)
    s0 = ym.project_state(h0, th0, r0)

    # Multi-hop action chain: 10 actions
    actions = [
        YggdrasilAction(
            delta_theta=torch.randn(10, 8) * 0.5,
            scale_factor=torch.tensor(1.1)
        )
        for _ in range(10)
    ]

    # Step forward
    s_curr = s0
    for a in actions:
        s_curr = ym.compose_action(s_curr, a)

    # Step backwards
    for a in reversed(actions):
        s_curr = ym.invert_action(s_curr, a)

    d_err = ym.distance(s0, s_curr)
    assert torch.all(d_err < 1e-5), f"Max reversibility drift was {d_err.max().item()}"


def test_yggdrasil_analogy_solving():
    ym = YggdrasilManifold(d_h=8, d_t=8, c=1.0)
    n = 20
    # Domains A and C
    s_a = ym.project_state(torch.randn(n, 8) * 0.3, torch.rand(n, 8) * 2 * math.pi, torch.ones(n, 1))
    s_c = ym.project_state(torch.randn(n, 8) * 0.3, torch.rand(n, 8) * 2 * math.pi, torch.ones(n, 1))

    # Known relation transforms A to B on Torus fiber
    true_rel = YggdrasilAction(
        delta_theta=torch.randn(n, 8) * 0.5,
        scale_factor=torch.tensor(1.5)
    )
    s_b = ym.compose_action(s_a, true_rel)
    s_d_true = ym.compose_action(s_c, true_rel)

    # Solve A : B :: C : D
    s_d_pred = ym.analogy(s_a, s_b, s_c)

    d_error = ym.distance(s_d_true, s_d_pred)
    assert torch.all(d_error < 1e-5), f"Analogy error was {d_error.max().item()}"
