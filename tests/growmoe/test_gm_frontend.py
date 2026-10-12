import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from growmoe.frontend import HashFrontend

REPO = Path(__file__).resolve().parents[2]


def _norms(z):
    return z.norm(dim=-1)


def test_numeric_and_text_outputs_are_unit_vectors():
    fe = HashFrontend(d=128, seed=0)
    rng = np.random.default_rng(0)
    zn = fe(rng.normal(size=(10, 64)))
    zt = fe(["hello world", "def f(x): return x + 1", b"\x00\x01\x02 raw bytes"])
    assert zn.shape == (10, 128) and zt.shape == (3, 128)
    assert torch.allclose(_norms(zn), torch.ones(10), atol=1e-5)
    assert torch.allclose(_norms(zt), torch.ones(3), atol=1e-5)


def test_two_fresh_instances_agree_exactly():
    x = np.random.default_rng(1).normal(size=(5, 30))
    assert torch.equal(HashFrontend(seed=3)(x), HashFrontend(seed=3)(x))
    t = ["alpha beta gamma", "delta"]
    assert torch.equal(HashFrontend(seed=3)(t), HashFrontend(seed=3)(t))


def test_deterministic_across_processes_with_different_pythonhashseed():
    code = (
        "import numpy as np\n"
        "from growmoe.frontend import HashFrontend\n"
        "fe = HashFrontend(seed=5)\n"
        "x = np.arange(40, dtype=float).reshape(2, 20)\n"
        "z = fe(x); t = fe(['the quick brown fox', 'jumps over'])\n"
        "print(z.numpy().tobytes().hex()); print(t.numpy().tobytes().hex())\n"
    )
    outs = []
    for hs in ("1", "2", "random"):
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONWARNINGS="ignore")
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip().splitlines()[-2:])
    assert outs[0] == outs[1] == outs[2]


def test_accepts_many_input_shapes_and_ragged_lists():
    fe = HashFrontend()
    rng = np.random.default_rng(2)
    assert fe(rng.normal(size=(4, 8, 8))).shape == (4, 128)      # images
    assert fe(rng.normal(size=(4, 13))).shape == (4, 128)        # tabular
    assert fe(rng.normal(size=(4, 4096 + 10))).shape == (4, 128)  # longer than max_len folds
    ragged = [rng.normal(size=5), rng.normal(size=11), rng.normal(size=3)]
    assert fe(ragged).shape == (3, 128)
    assert fe(torch.randn(3, 20)).shape == (3, 128)


def test_cosine_preservation_for_sparse_vectors():
    fe = HashFrontend(d=128, seed=0)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 64)) * (rng.random((400, 64)) < 0.5)
    y = x + 0.8 * rng.normal(size=(400, 64)) * (rng.random((400, 64)) < 0.5)
    cos_x = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
    zx, zy = fe(x), fe(y)
    cos_z = (zx * zy).sum(1).numpy()
    err = np.abs(cos_z - cos_x)
    assert np.quantile(err, 0.95) < 0.2


def test_different_seeds_give_near_orthogonal_maps_of_the_same_input():
    x = np.random.default_rng(4).normal(size=(50, 64))
    za, zb = HashFrontend(seed=1)(x), HashFrontend(seed=2)(x)
    cos = (za * zb).sum(1).abs()
    assert cos.mean() < 0.25


def test_zero_padding_and_global_scale_do_not_change_encoding():
    fe = HashFrontend()
    v = np.random.default_rng(5).normal(size=(3, 13))
    padded = np.concatenate([v, np.zeros((3, 51))], axis=1)
    assert torch.allclose(fe(v), fe(padded), atol=1e-6)
    assert torch.allclose(fe(v), fe(7.5 * v), atol=1e-6)


def test_salt_by_shape_separates_shapes_only_when_asked():
    v = np.random.default_rng(6).normal(size=(3, 13))
    padded = np.concatenate([v, np.zeros((3, 51))], axis=1)
    salted = HashFrontend(salt_by_shape=True)
    assert (salted(v) * salted(padded)).sum(1).abs().max() < 0.6


def test_degenerate_inputs_map_to_a_fixed_unit_vector():
    fe = HashFrontend()
    z = fe(np.zeros((2, 10)))
    assert torch.allclose(_norms(z), torch.ones(2), atol=1e-6)
    assert torch.equal(z[0], z[1])
    e = fe([""])
    assert torch.allclose(_norms(e), torch.ones(1), atol=1e-6)
