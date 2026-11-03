import torch

from dmoe.complex_ops import ComplexLinear, ModReLU, ComplexLayerNorm


def test_complex_linear_matches_manual_computation():
    layer = ComplexLinear(2, 2, bias=True)
    with torch.no_grad():
        layer.weight_real.copy_(torch.tensor([[1.0, 2.0], [0.5, -1.0]]))
        layer.weight_imag.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        layer.bias_real.copy_(torch.tensor([0.1, 0.2]))
        layer.bias_imag.copy_(torch.tensor([-0.1, 0.0]))

    z_r = torch.tensor([[1.0, 2.0]])
    z_i = torch.tensor([[3.0, 4.0]])

    out_r, out_i = layer(z_r, z_i)

    # Hand-computed complex product:
    # row 0: (1+0i)(1+3i) + (2+1i)(2+4i) + (0.1-0.1i) = (1+3i) + (0+10i) + (0.1-0.1i) = 1.1 + 12.9i
    # row 1: (0.5+1i)(1+3i) + (-1+0i)(2+4i) + (0.2+0i) = (-2.5+2.5i) + (-2-4i) + 0.2 = -4.3 - 1.5i
    expected_r = torch.tensor([[1.1, -4.3]])
    expected_i = torch.tensor([[12.9, -1.5]])

    assert torch.allclose(out_r, expected_r, atol=1e-6)
    assert torch.allclose(out_i, expected_i, atol=1e-6)


def test_complex_linear_gradients_flow():
    layer = ComplexLinear(4, 4)
    z_r = torch.randn(2, 4, requires_grad=True)
    z_i = torch.randn(2, 4, requires_grad=True)
    out_r, out_i = layer(z_r, z_i)
    (out_r.sum() + out_i.sum()).backward()
    assert layer.weight_real.grad is not None
    assert not torch.isnan(layer.weight_real.grad).any()
    assert layer.weight_imag.grad is not None
    assert not torch.isnan(layer.weight_imag.grad).any()
    assert z_r.grad is not None
    assert z_i.grad is not None


def test_modrelu_zeroes_small_magnitude():
    act = ModReLU(1)
    with torch.no_grad():
        act.b.fill_(-5.0)
    z_r = torch.tensor([[3.0]])
    z_i = torch.tensor([[4.0]])
    out_r, out_i = act(z_r, z_i)
    assert torch.allclose(out_r, torch.zeros_like(out_r), atol=1e-5)
    assert torch.allclose(out_i, torch.zeros_like(out_i), atol=1e-5)


def test_modrelu_preserves_phase_direction_above_threshold():
    act = ModReLU(1)
    with torch.no_grad():
        act.b.fill_(0.0)
    z_r = torch.tensor([[3.0]])
    z_i = torch.tensor([[4.0]])
    out_r, out_i = act(z_r, z_i)
    mag_in = torch.sqrt(z_r ** 2 + z_i ** 2)
    mag_out = torch.sqrt(out_r ** 2 + out_i ** 2)
    assert torch.allclose(mag_out, mag_in, atol=1e-4)
    assert torch.allclose(out_r / z_r, out_i / z_i, atol=1e-4)


def test_modrelu_gradients_flow():
    act = ModReLU(4)
    z_r = torch.randn(2, 4, requires_grad=True)
    z_i = torch.randn(2, 4, requires_grad=True)
    out_r, out_i = act(z_r, z_i)
    (out_r.sum() + out_i.sum()).backward()
    assert act.b.grad is not None
    assert not torch.isnan(act.b.grad).any()
    assert z_r.grad is not None
    assert z_i.grad is not None


def test_complex_layer_norm_output_shape_zero_mean_unit_variance_and_gradients():
    norm = ComplexLayerNorm(8)
    z_r = (torch.randn(3, 8) * 10 + 5).detach().requires_grad_(True)
    z_i = (torch.randn(3, 8) * 10 - 5).detach().requires_grad_(True)
    out_r, out_i = norm(z_r, z_i)
    assert out_r.shape == z_r.shape
    assert out_i.shape == z_i.shape
    assert not torch.isnan(out_r).any()
    assert not torch.isnan(out_i).any()

    # With default gamma=(1+0i), beta=(0+0i), output has zero mean and unit complex variance
    assert torch.allclose(out_r.mean(dim=-1), torch.zeros(3), atol=1e-5)
    assert torch.allclose(out_i.mean(dim=-1), torch.zeros(3), atol=1e-5)
    complex_var = (out_r ** 2 + out_i ** 2).mean(dim=-1)
    assert torch.allclose(complex_var, torch.ones(3), atol=1e-4)

    (out_r.sum() + out_i.sum()).backward()
    assert norm.gamma_real.grad is not None
    assert not torch.isnan(norm.gamma_real.grad).any()
    assert z_r.grad is not None
    assert z_i.grad is not None
