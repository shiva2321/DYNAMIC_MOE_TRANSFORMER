"""
Unit tests for Hierarchical Riemannian Predictive Coding (HRPC).
"""

import math
import pytest
import torch
import torch.nn.functional as F

from ishmn.predictive_coding import (
    DendriticCompartment,
    RiemannianPredictiveCodingLayer,
    HierarchicalRiemannianPredictiveCoding
)
from ishmn.geometry import project_to_sphere, geodesic_distance_sphere


def test_dendritic_compartment():
    d_model = 16
    dendrite = DendriticCompartment(d_model=d_model, d_apical=d_model)
    basal_in = torch.randn(4, d_model)
    apical_in = torch.randn(4, d_model)

    soma_out, discrepancy = dendrite(basal_in, apical_in)

    assert soma_out.shape == (4, d_model)
    assert discrepancy.shape == (4, d_model)
    # With no apical input, should still function smoothly
    soma_no_apical, disc_no_apical = dendrite(basal_in, None)
    assert soma_no_apical.shape == (4, d_model)


def test_riemannian_predictive_coding_layer():
    d_model = 16
    hyperspace_dim = 8
    layer = RiemannianPredictiveCodingLayer(
        level_idx=1,
        d_model=d_model,
        hyperspace_dim=hyperspace_dim
    )

    h = torch.randn(4, d_model)
    u, r = layer.project_manifold(h)
    assert u.shape == (4, hyperspace_dim)
    assert r.shape == (4, 1)
    # Unit norm check
    norms = torch.norm(u, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert torch.all(r > 1.0)

    # Top-down generative prediction
    mu_u, mu_r, mu_h = layer.generate_child_prediction(h)
    assert mu_u.shape == (4, hyperspace_dim)
    assert mu_r.shape == (4, 1)
    assert mu_h.shape == (4, d_model)

    # Prediction error
    energy, d_ang, tang_err, rad_err = layer.compute_layer_prediction_error(
        u_actual=u,
        r_actual=r,
        u_pred=mu_u,
        r_pred=mu_r
    )
    assert energy.dim() == 0  # scalar
    assert d_ang.shape == (4,)
    assert tang_err.shape == (4, hyperspace_dim)
    assert rad_err.shape == (4,)
    assert torch.isfinite(energy)


def test_hrpc_free_energy_reduction():
    """
    Core verification: Fast state relaxation must decrease total Variational Free Energy.
    """
    torch.manual_seed(42)
    model = HierarchicalRiemannianPredictiveCoding(
        num_layers=3,
        d_model=16,
        hyperspace_dim=6,
        input_dim=12
    )

    x = torch.randn(8, 12)
    # Perform 15 steps of fast continuous state relaxation
    states, trajectory = model.infer_states(
        sensory_input=x,
        num_steps=15,
        step_size=0.08
    )

    f_init = trajectory[0]["total_free_energy"]
    f_final = trajectory[-1]["total_free_energy"]

    # Free energy must be reduced via relaxation down the manifold gradient
    assert f_final < f_init, f"Free energy should decrease: init={f_init}, final={f_final}"
    # Verify state dimensions
    assert len(states) == 3
    for s in states:
        assert s.shape == (8, 16)


def test_manifold_vicreg_regularization():
    """
    Test that VICReg penalizes dimensional collapse.
    """
    model = HierarchicalRiemannianPredictiveCoding(
        num_layers=2,
        d_model=16,
        hyperspace_dim=8,
        input_dim=16
    )

    # 1. Collapsed state: identical vectors
    collapsed_u = project_to_sphere(torch.ones(8, 8))
    loss_collapsed = model.compute_vicreg_loss([collapsed_u])

    # 2. Diverse state: uniformly spread vectors
    diverse_u = project_to_sphere(torch.randn(8, 8))
    loss_diverse = model.compute_vicreg_loss([diverse_u])

    # Collapsed representations should incur significantly higher VICReg penalty
    assert loss_collapsed.item() > loss_diverse.item()


def test_theta_cycle_dynamics():
    """
    Verify theta rhythm modulation of sensory vs prior precision.
    """
    model = HierarchicalRiemannianPredictiveCoding(
        num_layers=3,
        d_model=16,
        hyperspace_dim=6,
        input_dim=12
    )
    x = torch.randn(4, 12)

    # Valley of theta (phase = 0): high sensory weighting
    states_valley, meta_valley = model.theta_cycle_step(x, phase=0.0, num_fast_steps=4)
    assert meta_valley["sensory_weight"] > meta_valley["prior_weight"]

    # Peak of theta (phase = pi): high prior/consolidation weighting
    states_peak, meta_peak = model.theta_cycle_step(x, phase=math.pi, num_fast_steps=4)
    assert meta_peak["prior_weight"] > meta_peak["sensory_weight"]


def test_training_step_parameter_update():
    """
    Verify that slow parameter learning properly updates synaptic weights.
    """
    model = HierarchicalRiemannianPredictiveCoding(
        num_layers=3,
        d_model=16,
        hyperspace_dim=6,
        input_dim=12
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = torch.randn(4, 12)
    meta = model.train_predictive_step(x, optimizer, inference_steps=4)

    assert "total_free_energy" in meta
    assert "energy_reduction" in meta
    assert meta["energy_reduction"] >= 0.0


def test_hrpc_forward_relaxation_step():
    """
    Verify forward-only autograd-free dendritic relaxation:
    - Runs cleanly without requiring gradients.
    - Yields valid bounded latent states across all hierarchical layers.
    """
    model = HierarchicalRiemannianPredictiveCoding(
        num_layers=3,
        d_model=16,
        hyperspace_dim=6,
        input_dim=12
    )
    x = torch.randn(4, 12)

    with torch.no_grad():
        relaxed_states = model.forward_relaxation_step(x, num_steps=3, alpha=0.15)

    assert len(relaxed_states) == 3
    for s in relaxed_states:
        assert s.shape == (4, 16)
        assert torch.isfinite(s).all()

