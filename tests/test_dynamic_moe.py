"""
test_dynamic_moe.py - Verification tests for DynamicYggdrasilMoE architecture.
Tests spawning, geodesic routing, freezing, and backward autograd gradients.
"""

import pytest
import torch
import torch.nn.functional as F

from geometry.dynamic_moe import DynamicYggdrasilMoE, GeometricExpertBranch


def test_dynamic_moe_spawning_and_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicYggdrasilMoE(
        in_features=16,
        hidden_dim=32,
        out_features=5,
        dim_h=4,
        dim_t=4,
        num_prototypes_per_expert=8,
    ).to(device)

    assert model.num_experts == 0

    # Spawning first expert
    e0 = model.spawn_expert("task_add")
    assert model.num_experts == 1
    assert isinstance(e0, GeometricExpertBranch)

    # Test forward pass with single expert
    x = torch.randn(8, 16, device=device)
    out = model(x)
    assert out.shape == (8, 5)

    # Spawning second expert
    e1 = model.spawn_expert("task_sub")
    assert model.num_experts == 2

    # Forward with autonomous geodesic routing
    out2, routing = model(x, return_routing=True)
    assert out2.shape == (8, 5)
    assert routing.shape == (8, 2)
    # Routing probabilities sum to 1
    assert torch.allclose(routing.sum(dim=-1), torch.ones(8, device=device), atol=1e-4)


def test_dynamic_moe_freezing_and_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicYggdrasilMoE(
        in_features=8,
        hidden_dim=16,
        out_features=2,
        dim_h=4,
        dim_t=4,
        num_prototypes_per_expert=4,
    ).to(device)

    # Task 1: Add
    e0 = model.spawn_expert("task_0")
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    x1 = torch.randn(10, 8, device=device)
    y1 = torch.randn(10, 2, device=device)

    # Train Task 1
    for _ in range(5):
        opt.zero_grad()
        loss = F.mse_loss(model(x1, expert_id=0), y1)
        loss.backward()
        opt.step()

    # Freeze Task 0
    w0_before = [p.clone() for p in e0.parameters()]
    model.freeze_expert(0)

    # Task 2: Sub
    e1 = model.spawn_expert("task_1")
    # Only train e1 and encoder
    opt2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

    x2 = torch.randn(10, 8, device=device)
    y2 = torch.randn(10, 2, device=device)

    for _ in range(5):
        opt2.zero_grad()
        loss = F.mse_loss(model(x2, expert_id=1), y2)
        loss.backward()
        opt2.step()

    # Assert that expert 0 parameters did NOT change at all (exact preservation)
    for p_before, p_after in zip(w0_before, e0.parameters()):
        assert torch.equal(p_before, p_after)
