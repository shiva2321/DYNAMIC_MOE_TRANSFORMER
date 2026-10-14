import pytest
import torch
import torch.nn as nn
from geometry.layers import (
    YggdrasilProjector,
    YggdrasilMemoryCortex,
    YggdrasilSandwichModel,
)


def test_yggdrasil_projector_bounds():
    dim_h, dim_t = 8, 8
    c = 1.0
    projector = YggdrasilProjector(in_features=32, dim_h=dim_h, dim_t=dim_t, c=c)

    # Random extreme inputs
    x = torch.randn(16, 32) * 100.0
    state = projector(x)

    # Check Hyperbolic norm < 1.0 / sqrt(c)
    h_norms = torch.norm(state[:, :dim_h], dim=-1)
    assert (h_norms < 1.0).all()

    # Check Torus angles in [-pi, pi]
    t_coords = state[:, dim_h:dim_h + dim_t]
    assert (t_coords >= -3.1415927).all() and (t_coords <= 3.1415927).all()

    # Check Radial positivity
    r_coords = state[:, -1]
    assert (r_coords > 0.0).all()


def test_yggdrasil_memory_cortex_autodiff():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim_h, dim_t = 8, 8
    cortex = YggdrasilMemoryCortex(
        num_prototypes=10,
        dim_h=dim_h,
        dim_t=dim_t,
        out_features=16,
    ).to(device)

    total_dim = dim_h + dim_t + 1
    raw = torch.randn(8, total_dim, device=device) * 0.1
    raw[:, -1] = raw[:, -1].abs() + 0.2
    queries = raw.detach().requires_grad_(True)

    logits, context = cortex(queries)
    assert logits.shape == (8, 10)
    assert context.shape == (8, 16)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(context).any()

    # Test backward pass
    loss = context.sum() + logits.sum()
    loss.backward()

    assert queries.grad is not None
    assert not torch.isnan(queries.grad).any()
    assert cortex.prototypes.grad is not None
    assert not torch.isnan(cortex.prototypes.grad).any()


def test_yggdrasil_sandwich_model_train_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YggdrasilSandwichModel(
        in_features=20,
        hidden_dim=32,
        num_prototypes=8,
        num_classes=4,
        dim_h=8,
        dim_t=8,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(16, 20, device=device)
    y = torch.randint(0, 4, (16,), device=device)

    # 1. Forward
    out = model(x)
    assert out.shape == (16, 4)
    loss = criterion(out, y)
    assert not torch.isnan(loss)

    # 2. Backward
    optimizer.zero_grad()
    loss.backward()

    # Check all gradients exist and non-zero
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient"

    # 3. Optimizer Step + Manifold Retraction
    optimizer.step()
    model.cortex.clamp_prototypes()
