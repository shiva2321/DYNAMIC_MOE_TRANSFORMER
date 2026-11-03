import pytest
import torch

from dmoe.encoder import UniversalMultimodalEncoder


def test_text_input_shape():
    enc = UniversalMultimodalEncoder(d_model=16)
    x = torch.randint(0, 1000, (2, 5), dtype=torch.long)
    r, i = enc(x)
    assert r.shape == (2, 5, 16)
    assert i.shape == (2, 5, 16)


def test_image_input_shape():
    enc = UniversalMultimodalEncoder(d_model=16)
    x = torch.randn(2, 3, 8, 8)
    r, i = enc(x)
    assert r.shape[0] == 2
    assert r.shape[-1] == 16
    assert i.shape == r.shape


def test_numeric_input_shape_and_lazy_projector_reuse():
    enc = UniversalMultimodalEncoder(d_model=16)
    x1 = torch.randn(2, 5, 7)
    r1, i1 = enc(x1)
    assert r1.shape == (2, 5, 16)
    assert "7" in enc.numeric_projectors

    projector_before = enc.numeric_projectors["7"]
    x2 = torch.randn(3, 4, 7)
    r2, i2 = enc(x2)
    assert r2.shape == (3, 4, 16)
    assert enc.numeric_projectors["7"] is projector_before


def test_numeric_input_varying_feature_dims():
    enc = UniversalMultimodalEncoder(d_model=16)
    x1 = torch.randn(2, 5, 7)
    x2 = torch.randn(2, 5, 11)
    r1, i1 = enc(x1)
    r2, i2 = enc(x2)
    assert r1.shape == (2, 5, 16)
    assert r2.shape == (2, 5, 16)
    assert "7" in enc.numeric_projectors
    assert "11" in enc.numeric_projectors


def test_numeric_projection_distinguishes_feature_position():
    # Regression test for the original bug: a fixed all-ones projection made
    # output identical regardless of WHICH feature index held a signal.
    enc = UniversalMultimodalEncoder(d_model=16)
    torch.manual_seed(0)
    x_a = torch.zeros(1, 1, 5)
    x_a[0, 0, 0] = 10.0
    x_b = torch.zeros(1, 1, 5)
    x_b[0, 0, 4] = 10.0
    r_a, _ = enc(x_a)
    r_b, _ = enc(x_b)
    assert not torch.allclose(r_a, r_b)


def test_text_detection_works_for_int32_dtype():
    enc = UniversalMultimodalEncoder(d_model=16)
    x = torch.randint(0, 1000, (2, 5), dtype=torch.int32)
    r, i = enc(x)
    assert r.shape == (2, 5, 16)


def test_rejects_unsupported_shape():
    enc = UniversalMultimodalEncoder(d_model=16)
    x = torch.randn(2, 2, 2, 2, 2)
    with pytest.raises(ValueError):
        enc(x)

