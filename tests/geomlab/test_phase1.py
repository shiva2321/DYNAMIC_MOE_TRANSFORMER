"""
Unit tests for Phase 1: Real Latent Geometry Health and Capacity.
"""

import pytest
import torch
from geomlab.phase1_capacity import (
    load_banking_dataset,
    EuclideanModel,
    SphericalModel,
    TorusModel,
    evaluate_bundle_capacity,
    compute_mutual_coherence,
)


def test_banking_dataset_loader():
    train_x, train_y, val_x, val_y, cb_texts, cb_labels = load_banking_dataset(seed=42)
    assert train_x.shape[0] > 8000
    assert train_x.shape[1] == 1024
    assert len(train_y) == train_x.shape[0]
    assert len(cb_texts) == 1000
    assert len(cb_labels) == 1000
    assert train_y.max().item() < 77
    assert train_y.min().item() >= 0


def test_models_forward_and_shapes():
    B = 16
    in_dim = 1024
    d_model = 128
    num_classes = 77
    x = torch.randn(B, in_dim)

    # Euclidean
    e_model = EuclideanModel(in_dim=in_dim, d_model=d_model, num_classes=num_classes)
    e_logits, e_z = e_model(x)
    assert e_logits.shape == (B, num_classes)
    assert e_z.shape == (B, d_model)

    # Spherical
    s_model = SphericalModel(in_dim=in_dim, d_model=d_model, num_classes=num_classes)
    s_logits, s_z = s_model(x)
    assert s_logits.shape == (B, num_classes)
    assert s_z.shape == (B, d_model)
    # Check unit norm
    norms = torch.norm(s_z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Torus
    t_model = TorusModel(in_dim=in_dim, d_model=d_model, num_classes=num_classes)
    t_logits, t_z = t_model(x)
    assert t_logits.shape == (B, num_classes)
    assert t_z.shape == (B, d_model)
    # Check phase angles in [-pi, pi]
    assert t_z.max().item() <= 3.1416
    assert t_z.min().item() >= -3.1416


def test_bundle_capacity_random_vectors():
    N = 500
    D = 256
    torch.manual_seed(42)
    # Haar-random unit vectors on sphere
    vectors = torch.randn(N, D)
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)

    res = evaluate_bundle_capacity(vectors, is_torus=False, bundle_sizes=(1, 2, 4), num_trials=30, seed=42)
    # For M=1, single vector retrieval should be 100%
    assert res[1] == 100.0, f"Expected 100% for M=1, got {res[1]}"
    # For M=2, should be very high (> 90%)
    assert res[2] > 85.0, f"Expected > 85% for M=2, got {res[2]}"

    # Torus random phasors
    phases = torch.empty(N, D).uniform_(-3.1415, 3.1415)
    res_torus = evaluate_bundle_capacity(phases, is_torus=True, bundle_sizes=(1, 2, 4), num_trials=30, seed=42)
    assert res_torus[1] == 100.0
    assert res_torus[2] > 85.0


def test_mutual_coherence():
    N, D = 100, 64
    vectors = torch.randn(N, D)
    mc = compute_mutual_coherence(vectors)
    assert 0.0 <= mc <= 1.0
