import math

import torch

from dmoe.manifold import GeometricCorrectiveManifold


def _zero_predictor(m: GeometricCorrectiveManifold) -> None:
    with torch.no_grad():
        for p in [
            m.predictor_stage1.weight_real, m.predictor_stage1.weight_imag,
            m.predictor_stage1.bias_real, m.predictor_stage1.bias_imag,
            m.predictor_stage2.weight_real, m.predictor_stage2.weight_imag,
            m.predictor_stage2.bias_real, m.predictor_stage2.bias_imag,
        ]:
            p.zero_()


def test_output_shape_matches_input():
    m = GeometricCorrectiveManifold(d_model=8)
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    out_r, out_i, err, novelty = m(z_r, z_i)
    assert out_r.shape == z_r.shape
    assert out_i.shape == z_i.shape
    assert err.dim() == 0
    assert novelty.dim() == 0


def test_running_error_updates_only_in_training_mode():
    m = GeometricCorrectiveManifold(d_model=8)
    initial_mean = m.running_error_magnitude.item()
    initial_var = m.running_error_variance.item()

    m.eval()
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    m(z_r, z_i)
    assert m.running_error_magnitude.item() == initial_mean
    assert m.running_error_variance.item() == initial_var

    m.train()
    m(z_r, z_i)
    assert m.running_error_magnitude.item() != initial_mean
    assert m.running_error_variance.item() != initial_var


def test_error_is_non_negative():
    m = GeometricCorrectiveManifold(d_model=8)
    z_r = torch.randn(4, 2, 8) * 100
    z_i = torch.randn(4, 2, 8) * 100
    _, _, err, _ = m(z_r, z_i)
    assert err.item() >= 0.0


def test_gradients_flow_through_corrective_path():
    m = GeometricCorrectiveManifold(d_model=8)
    z_r = torch.randn(2, 3, 8, requires_grad=True)
    z_i = torch.randn(2, 3, 8, requires_grad=True)
    out_r, out_i, err, novelty = m(z_r, z_i)
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


def test_novelty_score_is_detached_from_computation_graph():
    m = GeometricCorrectiveManifold(d_model=8)
    z_r = torch.randn(2, 3, 8, requires_grad=True)
    z_i = torch.randn(2, 3, 8, requires_grad=True)
    _, _, _, novelty = m(z_r, z_i)
    assert novelty.requires_grad is False


def test_novelty_score_matches_z_score_formula():
    m = GeometricCorrectiveManifold(d_model=4)
    _zero_predictor(m)
    with torch.no_grad():
        m.running_error_magnitude.fill_(0.5)
        m.running_error_variance.fill_(0.04)

    z_r = torch.full((1, 1, 4), 0.6)
    z_i = torch.zeros(1, 1, 4)

    _, _, batch_mean_error, novelty_score = m(z_r, z_i)

    expected_novelty = (batch_mean_error.detach() - 0.5) / (math.sqrt(0.04) + 1e-8)
    assert torch.allclose(novelty_score, expected_novelty, atol=1e-4)


def test_novelty_score_is_near_zero_when_batch_matches_running_mean():
    m = GeometricCorrectiveManifold(d_model=4)
    _zero_predictor(m)
    with torch.no_grad():
        error_norm_value = math.sqrt(0.6 ** 2 + 1e-8)
        m.running_error_magnitude.fill_(error_norm_value)
        m.running_error_variance.fill_(0.04)

    z_r = torch.full((1, 1, 4), 0.6)
    z_i = torch.zeros(1, 1, 4)

    _, _, _, novelty_score = m(z_r, z_i)
    assert abs(novelty_score.item()) < 1e-3


def test_novelty_score_is_large_when_batch_far_from_running_mean():
    m = GeometricCorrectiveManifold(d_model=4)
    _zero_predictor(m)
    with torch.no_grad():
        m.running_error_magnitude.fill_(0.5)
        m.running_error_variance.fill_(0.01)

    z_r = torch.full((1, 1, 4), 10.0)
    z_i = torch.zeros(1, 1, 4)

    _, _, _, novelty_score = m(z_r, z_i)
    assert novelty_score.item() > 50.0

