import torch

from dmoe.layer import DynamicMorphicTransformerLayer


def test_forward_shape_stable():
    layer = DynamicMorphicTransformerLayer(d_model=8, num_heads=2, d_ff=16, initial_experts=2)
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    out_r, out_i, entropy = layer(z_r, z_i)
    assert out_r.shape == (2, 3, 8)
    assert out_i.shape == (2, 3, 8)
    assert entropy.dim() == 0


def test_salience_updates_in_training_mode_and_moves_in_expected_direction():
    layer = DynamicMorphicTransformerLayer(d_model=8, num_heads=2, d_ff=16, initial_experts=2)
    layer.train()
    initial = layer.layer_salience.item()
    z_r = torch.randn(2, 3, 8) + 1.0
    z_i = torch.randn(2, 3, 8) + 1.0

    # Zero residual contribution -> salience moves down below initial (1.0)
    with torch.no_grad():
        layer.alpha_residual.fill_(0.0)
    layer(z_r, z_i)
    assert layer.layer_salience.item() < initial

    # Large residual contribution -> salience moves up above initial (1.0)
    with torch.no_grad():
        layer.layer_salience.fill_(initial)
        layer.alpha_residual.fill_(50.0)
    layer(z_r, z_i)
    assert layer.layer_salience.item() > initial


def test_salience_does_not_update_in_eval_mode():
    layer = DynamicMorphicTransformerLayer(d_model=8, num_heads=2, d_ff=16, initial_experts=2)
    layer.eval()
    initial = layer.layer_salience.item()
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    layer(z_r, z_i)
    assert layer.layer_salience.item() == initial


def test_gradients_flow():
    layer = DynamicMorphicTransformerLayer(d_model=8, num_heads=2, d_ff=16, initial_experts=2)
    z_r = torch.randn(2, 3, 8, requires_grad=True)
    z_i = torch.randn(2, 3, 8, requires_grad=True)
    out_r, out_i, _ = layer(z_r, z_i)
    (out_r.sum() + out_i.sum()).backward()
    assert z_r.grad is not None
    assert not torch.isnan(z_r.grad).any()
