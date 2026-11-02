"""Tests for costlab.bytes: parameter footprint and active bytes tracking."""

import torch
import torch.nn as nn

from costlab.bytes import ActiveBytes, active_param_bytes, total_param_bytes


def test_total_param_bytes_linear():
    """Verify total_param_bytes on single Linear layer."""
    in_features, out_features = 32, 64
    lin = nn.Linear(in_features, out_features, bias=True)
    # Weights: 32*64 float32 (4 bytes), Bias: 64 float32
    expected = (32 * 64 + 64) * 4
    assert total_param_bytes(lin) == expected


def test_total_param_bytes_tied_weights():
    """Verify parameters sharing storage (tied weights) are counted only once."""
    lin1 = nn.Linear(32, 32, bias=False)
    lin2 = nn.Linear(32, 32, bias=False)
    lin2.weight = lin1.weight  # tie weights

    model = nn.Sequential(lin1, lin2)
    expected = 32 * 32 * 4
    assert total_param_bytes(model) == expected


def test_active_param_bytes_mlp():
    """Verify active_param_bytes on dense MLP equals total_param_bytes."""
    mlp = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )
    x = torch.randn(4, 16)
    res = active_param_bytes(mlp, mlp, x)

    assert isinstance(res, ActiveBytes)
    # Tuple unpacking
    tot_unpacked, by_m_unpacked = res
    assert tot_unpacked == res.total
    assert by_m_unpacked == res.by_module

    tot = total_param_bytes(mlp)
    assert res.total == tot
    assert len(res.by_module) > 0


def test_active_param_bytes_conditional_branch():
    """Verify modules in unexecuted branches are not counted in active_param_bytes."""
    class ConditionalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.branch_a = nn.Linear(16, 16, bias=False)
            self.branch_b = nn.Linear(16, 16, bias=False)

        def forward(self, x, take_b=False):
            if take_b:
                return self.branch_b(x)
            return self.branch_a(x)

    model = ConditionalModel()
    x = torch.randn(2, 16)

    # Take branch A
    res_a = active_param_bytes(model, lambda: model(x, take_b=False))
    assert res_a.total == 16 * 16 * 4
    assert "branch_a" in res_a.by_module
    assert "branch_b" not in res_a.by_module

    # Take branch B
    res_b = active_param_bytes(model, lambda: model(x, take_b=True))
    assert res_b.total == 16 * 16 * 4
    assert "branch_b" in res_b.by_module
    assert "branch_a" not in res_b.by_module
