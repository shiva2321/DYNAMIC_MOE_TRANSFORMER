"""
Unit Tests for Sliced Isotropic Gaussian Regularization (SIGReg) - LeJEPA
Validates the Stage 0 Anti-Collapse Diagnostic Gate:
- Empirical characteristic function test on S^{D-1} slices
- Recovery from representation collapse (EXP-45 failure mode)
- Neural network encoder anti-collapse verification
"""

import pytest
import torch
import torch.nn as nn
from ishmn.sigreg import SIGReg, epps_pulley_statistic


def test_epps_pulley_statistic_properties():
    """Verify that standard Gaussian has near-zero discrepancy, while collapsed features have high loss."""
    torch.manual_seed(42)
    B, K = 512, 64

    # 1. Standard normal samples
    y_normal = torch.randn(B, K)
    loss_normal = epps_pulley_statistic(y_normal)
    assert loss_normal.item() < 0.01, f"Expected near-zero loss for normal, got {loss_normal.item()}"

    # 2. Fully collapsed constant samples
    y_collapsed = torch.zeros(B, K)
    loss_collapsed = epps_pulley_statistic(y_collapsed)
    assert loss_collapsed.item() > 0.15, f"Expected high loss for collapsed constant, got {loss_collapsed.item()}"

    # 3. EXP-45 style low-variance samples (std = 0.15)
    y_low_var = torch.randn(B, K) * 0.15
    loss_low_var = epps_pulley_statistic(y_low_var)
    assert loss_low_var.item() > 0.10, f"Expected elevated loss for low variance, got {loss_low_var.item()}"
    assert loss_low_var.item() > loss_normal.item()


def test_sigreg_module_forward_and_diagnostics():
    """Verify SIGReg module forward pass on [B, D] and [B, S, D] shapes."""
    sigreg = SIGReg(num_slices=32, lambda_reg=1.5)
    z_2d = torch.randn(64, 16, requires_grad=True)
    loss_2d, diag_2d = sigreg(z_2d)

    assert loss_2d.requires_grad
    assert "mean_var" in diag_2d
    assert "sigreg_loss" in diag_2d
    assert diag_2d["mean_var"] > 0.5

    loss_2d.backward()
    assert z_2d.grad is not None
    assert z_2d.grad.norm().item() > 0.0

    # 3D Sequence input
    z_3d = torch.randn(8, 16, 16)
    loss_3d, diag_3d = sigreg(z_3d)
    assert loss_3d.item() >= 0.0


def test_sigreg_anti_collapse_gate_recovery():
    """
    Gate 0 Falsification Gate:
    Replicates the EXP-45 representation collapse failure mode (variance 0.11 - 0.18)
    and validates that SIGReg recovers normalized variance >= 0.85 without heuristic tuning.
    """
    torch.manual_seed(1337)
    B, D = 128, 16

    # Start with a collapsed representation (variance = 0.0225, std = 0.15)
    z = (torch.randn(B, D) * 0.15).requires_grad_(True)
    initial_var = z.var(dim=0).mean().item()
    assert initial_var < 0.05, f"Initial state not properly collapsed: var={initial_var}"

    sigreg = SIGReg(num_slices=64, lambda_reg=1.0)
    optimizer = torch.optim.Adam([z], lr=0.05)

    for step in range(100):
        optimizer.zero_grad()
        loss, diag = sigreg(z)
        loss.backward()
        optimizer.step()

    final_var = z.var(dim=0).mean().item()
    final_std = z.std(dim=0).mean().item()
    final_mean = z.mean().abs().item()

    # Verify Gate 0 pass criteria
    assert final_var >= 0.85, f"Gate 0 Failed: Expected variance >= 0.85, got {final_var:.4f}"
    assert abs(final_std - 1.0) < 0.15, f"Expected standard deviation ~ 1.0, got {final_std:.4f}"
    assert final_mean < 0.10, f"Expected centered mean near 0, got {final_mean:.4f}"


def test_encoder_anti_collapse_training():
    """Verify that an encoder trained with SIGReg preserves high-variance isotropic features."""
    torch.manual_seed(42)
    B, in_dim, out_dim = 128, 32, 16

    # Simple 2-layer MLP encoder
    encoder = nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Linear(64, out_dim)
    )
    sigreg = SIGReg(num_slices=64, lambda_reg=1.0)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

    for _ in range(80):
        x = torch.randn(B, in_dim)
        optimizer.zero_grad()
        z = encoder(x)
        loss, diag = sigreg(z)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_x = torch.randn(B, in_dim)
        test_z = encoder(test_x)
        test_var = test_z.var(dim=0).mean().item()
        test_std = test_z.std(dim=0).mean().item()

    assert test_var >= 0.85, f"Encoder representation collapsed: variance = {test_var:.4f}"
    assert abs(test_std - 1.0) < 0.20, f"Encoder std deviates from 1.0: {test_std:.4f}"


def test_admit_concept_anti_crowding_filter():
    """Verify admit_concept enforces tau_sep threshold and merges crowding concepts."""
    from ishmn.memory import InfiniteSphericalMemory
    from ishmn.geometry import project_to_sphere, geodesic_distance_sphere

    mem = InfiniteSphericalMemory(num_slots=8, hyperspace_dim=8, payload_dim=16)
    assert len(mem.semantic_concepts) == 0

    # Concept 1: Base vector
    u1 = project_to_sphere(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    v1 = torch.ones(1, 16) * 1.0
    r1 = 1.5
    res1 = mem.admit_concept(u1, r1, v1, metadata={"name": "c1"}, tau_sep=0.35)
    assert res1["action"] == "admitted"
    assert len(mem.semantic_concepts) == 1

    # Concept 2: Far away orthogonal vector (dist = pi/2 ~ 1.57 rad >= 0.35)
    u2 = project_to_sphere(torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    v2 = torch.ones(1, 16) * 2.0
    r2 = 2.5
    res2 = mem.admit_concept(u2, r2, v2, metadata={"name": "c2"}, tau_sep=0.35)
    assert res2["action"] == "admitted"
    assert len(mem.semantic_concepts) == 2

    # Concept 3: Very close to Concept 2 (dist ~ 0.10 rad < 0.35)
    u3 = project_to_sphere(torch.tensor([0.0, 0.995, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0]))
    v3 = torch.ones(1, 16) * 3.0
    r3 = 2.7
    res3 = mem.admit_concept(u3, r3, v3, metadata={"name": "c3"}, tau_sep=0.35)
    assert res3["action"] == "merged"
    assert len(mem.semantic_concepts) == 2  # Not expanded, merged into concept 2!
    assert mem.semantic_concepts[1]["exemplar_count"] == 2
    # Verify Fréchet midpoint
    dist_to_u2 = geodesic_distance_sphere(mem.semantic_concepts[1]["u"], u2).item()
    assert dist_to_u2 < 0.10

