import torch

from dmoe.manifold import GeometricCorrectiveManifold


def test_output_shape_matches_input():
    m = GeometricCorrectiveManifold(d_model=8)
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    out_r, out_i, err = m(z_r, z_i)
    assert out_r.shape == z_r.shape
    assert out_i.shape == z_i.shape
    assert err.dim() == 0


def test_running_error_updates_only_in_training_mode():
    m = GeometricCorrectiveManifold(d_model=8)
    initial = m.running_error_magnitude.item()

    m.eval()
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    m(z_r, z_i)
    assert m.running_error_magnitude.item() == initial

    m.train()
    m(z_r, z_i)
    assert m.running_error_magnitude.item() != initial


def test_error_is_non_negative():
    m = GeometricCorrectiveManifold(d_model=8)
    z_r = torch.randn(4, 2, 8) * 100
    z_i = torch.randn(4, 2, 8) * 100
    _, _, err = m(z_r, z_i)
    assert err.item() >= 0.0


def test_gradients_flow_through_corrective_path():
    m = GeometricCorrectiveManifold(d_model=8)
    z_r = torch.randn(2, 3, 8, requires_grad=True)
    z_i = torch.randn(2, 3, 8, requires_grad=True)
    out_r, out_i, err = m(z_r, z_i)
    (out_r.sum() + out_i.sum() + err).backward()
    assert z_r.grad is not None
    assert not torch.isnan(z_r.grad).any()
    assert m.predictor_stage1.weight_real.grad is not None
    assert not torch.isnan(m.predictor_stage1.weight_real.grad).any()
    assert m.corrector_linear1.weight_real.grad is not None
    assert not torch.isnan(m.corrector_linear1.weight_real.grad).any()


def test_predictor_stages_are_named_clearly():
    m = GeometricCorrectiveManifold(d_model=8)
    assert hasattr(m, "predictor_stage1")
    assert hasattr(m, "predictor_stage2")
