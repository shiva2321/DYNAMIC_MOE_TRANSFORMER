import pytest
import torch

from dmoe.decoder import UniversalProjectiveDecoder


def test_token_head_shape():
    dec = UniversalProjectiveDecoder(d_model=8, vocab_size=100, continuous_dim=4)
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    out = dec(z_r, z_i, head="token")
    assert out.shape == (2, 3, 100)


def test_continuous_head_shape():
    dec = UniversalProjectiveDecoder(d_model=8, vocab_size=100, continuous_dim=4)
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    out = dec(z_r, z_i, head="continuous")
    assert out.shape == (2, 3, 4)


def test_latent_head_shape():
    dec = UniversalProjectiveDecoder(d_model=8)
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    out = dec(z_r, z_i, head="latent")
    assert out.shape == (2, 3, 8)


def test_unknown_head_raises():
    dec = UniversalProjectiveDecoder(d_model=8)
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    with pytest.raises(ValueError):
        dec(z_r, z_i, head="nonsense")
