import pytest
import torch

from dmoe.attention import ComplexMultiHeadAttention


def test_output_shape():
    attn = ComplexMultiHeadAttention(d_model=16, num_heads=4)
    z_r = torch.randn(2, 5, 16)
    z_i = torch.randn(2, 5, 16)
    out_r, out_i = attn(z_r, z_i)
    assert out_r.shape == (2, 5, 16)
    assert out_i.shape == (2, 5, 16)


def test_attention_weights_sum_to_one_along_key_dimension():
    attn = ComplexMultiHeadAttention(d_model=16, num_heads=4)
    z_r = torch.randn(2, 5, 16)
    z_i = torch.randn(2, 5, 16)
    attn_weights = attn.compute_attention_weights(z_r, z_i)
    assert attn_weights.shape == (2, 4, 5, 5)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-6)


def test_rejects_non_divisible_heads():
    with pytest.raises(AssertionError):
        ComplexMultiHeadAttention(d_model=10, num_heads=3)


def test_gradients_flow():
    attn = ComplexMultiHeadAttention(d_model=8, num_heads=2)
    z_r = torch.randn(1, 3, 8, requires_grad=True)
    z_i = torch.randn(1, 3, 8, requires_grad=True)
    out_r, out_i = attn(z_r, z_i)
    (out_r.sum() + out_i.sum()).backward()
    assert z_r.grad is not None
    assert not torch.isnan(z_r.grad).any()


def test_multiple_head_configurations_preserve_shape():
    for d_model, num_heads in [(8, 1), (8, 2), (8, 4), (32, 8)]:
        attn = ComplexMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        z_r = torch.randn(2, 4, d_model)
        z_i = torch.randn(2, 4, d_model)
        out_r, out_i = attn(z_r, z_i)
        assert out_r.shape == (2, 4, d_model)
        assert out_i.shape == (2, 4, d_model)

