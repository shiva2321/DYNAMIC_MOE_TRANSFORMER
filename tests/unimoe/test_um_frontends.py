"""The shared unit sphere (unimoe/frontends.py): text through a frozen encoder and a fixed random projection, digit codes padded and normalised, embeddings cached on disk."""

import numpy as np
import pytest
import torch

from unimoe.frontends import SphereFrontend


class CountingEncoder:
    """A stand-in for the frozen sentence encoder: a clustered, normalised 384-d vector per text, derived from the text's first word; counts its calls."""

    def __init__(self):
        self.calls = 0
        self.centers = np.random.default_rng(0).standard_normal((50, 384))

    def __call__(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            c = self.centers[abs(hash_word(t)) % 50]
            v = c + 0.3 * np.random.default_rng(abs(hash_word(t + "x")) % (2 ** 31)).standard_normal(384)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, dtype=np.float32)


def hash_word(t):
    return sum((i + 1) * ord(ch) for i, ch in enumerate(t.split()[0]))


def test_text_gives_unit_vectors_of_the_asked_size_and_is_deterministic():
    fe = SphereFrontend(d=128, encoder=CountingEncoder())
    x = fe.text(["alpha one", "beta two", "alpha three"])
    assert x.shape == (3, 128) and x.dtype == torch.float32
    assert torch.allclose(x.norm(dim=1), torch.ones(3), atol=1e-5)
    assert torch.equal(x, SphereFrontend(d=128, encoder=CountingEncoder()).text(["alpha one", "beta two", "alpha three"]))
    assert not torch.equal(x, SphereFrontend(d=128, seed=1, encoder=CountingEncoder()).text(["alpha one", "beta two", "alpha three"]))     # the projection depends on the seed


def test_the_projection_keeps_what_the_encoder_separates():
    fe = SphereFrontend(d=128, encoder=CountingEncoder())
    texts = [f"{w} item {i}" for w in ("alpha", "beta", "gamma", "delta") for i in range(8)]
    x = fe.text(texts)
    cos = (x @ x.T).numpy()
    same = np.kron(np.eye(4), np.ones((8, 8))).astype(bool) & ~np.eye(32, dtype=bool)
    assert cos[same].mean() - cos[~np.kron(np.eye(4), np.ones((8, 8))).astype(bool)].mean() > 0.5


def test_a_second_request_for_the_same_texts_is_read_from_disk_not_encoded_again(tmp_path):
    enc = CountingEncoder()
    fe = SphereFrontend(d=128, cache_dir=tmp_path, encoder=enc)
    a = fe.text(["alpha one", "beta two"])
    assert enc.calls == 1
    enc2 = CountingEncoder()
    b = SphereFrontend(d=128, cache_dir=tmp_path, encoder=enc2).text(["alpha one", "beta two"])
    assert enc2.calls == 0 and torch.equal(a, b)
    SphereFrontend(d=128, cache_dir=tmp_path, encoder=enc2).text(["alpha one", "beta two", "gamma"])
    assert enc2.calls == 1                                                                         # different texts, different cache entry


def test_digit_codes_are_padded_to_the_sphere_size_and_normalised():
    fe = SphereFrontend(d=128, encoder=CountingEncoder())
    codes = np.zeros((4, 127), dtype=np.float32)
    codes[:, :3] = [[1, 1, 1], [2, 0, 0], [0, 0, 5], [0, 0, 0]]
    x = fe.digits(codes)
    assert x.shape == (4, 128)
    assert torch.allclose(x[:3].norm(dim=1), torch.ones(3), atol=1e-5)
    assert torch.isfinite(x).all() and x[3].norm() == 0                                            # an all-zero code stays zero instead of becoming NaN
    with pytest.raises(ValueError):
        fe.digits(np.zeros((2, 129), dtype=np.float32))                                            # wider than the sphere
