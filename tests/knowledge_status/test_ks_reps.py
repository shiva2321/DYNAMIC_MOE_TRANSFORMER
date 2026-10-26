import socket

import numpy as np
import pytest

import reps
import runlib

SMALL = [("R0_hash", None), ("R1_minilm128", None), ("R4_gpt2_L6", "EleutherAI/pythia-70m")]   # the R4 stand-in (pythia-70m) is a 6-layer model, so layer 6 is its last
TEXTS = ["12 + 7", "Thanks for helping me out. What is twelve times five?", "How do I top up my card", "the sum of 4 and 9"]
HARD = ["", "a" * 5000, "café 三 plus 五", "wnhat is twleve tmies fvie ??  pLeAsE", "word " * 1500]


@pytest.fixture(scope="module", params=SMALL, ids=[p[0] for p in SMALL])
def rep(request):
    rid, model = request.param
    r = reps.get_rep(rid, model_name=model)
    yield r
    r.close()


def test_unit_length_finite_and_deterministic(rep):
    z = rep.encode(TEXTS)
    assert z.dtype == np.float32 and z.shape == (len(TEXTS), rep.dim)
    assert np.isfinite(z).all()
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(z, rep.encode(TEXTS))
    assert not np.allclose(z[0], z[1]) and not np.allclose(z[1], z[2])           # different sentences are not one point


def test_empty_very_long_and_non_ascii_texts_encode(rep):
    z = rep.encode(HARD)
    assert z.shape == (len(HARD), rep.dim) and np.isfinite(z).all()
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5)
    assert rep.encode([]).shape == (0, rep.dim)
    with pytest.raises(TypeError):
        rep.encode(["fine", 7])


class Counting(reps.Representation):
    """A real hashed representation that counts how many texts it was asked to encode."""

    def __init__(self):
        inner = reps.get_rep("R0_hash")
        super().__init__("R0_hash", inner.dim, tag="counting-test")
        self.inner, self.calls = inner, 0

    def _encode(self, texts):
        self.calls += len(texts)
        return self.inner._encode(texts)                                            # raw output: each path then normalises exactly once, like the real thing


def test_cache_hit_returns_identical_vectors_without_recomputing(tmp_path):
    c = Counting()
    first = reps.cached_encode(c, TEXTS, tmp_path)
    assert c.calls == len(TEXTS) and np.array_equal(first, c.inner.encode(TEXTS))
    again = reps.cached_encode(c, TEXTS, tmp_path)
    assert c.calls == len(TEXTS) and np.array_equal(first, again)                   # a hit: nothing was encoded again
    assert len(list(tmp_path.glob("*.npz"))) == 1 and not list(tmp_path.glob("*.tmp"))


def test_truncated_cache_file_is_ignored_and_recomputed(tmp_path):
    c = Counting()
    first = reps.cached_encode(c, TEXTS, tmp_path)
    (path,) = tmp_path.glob("*.npz")
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])                 # a power cut in the middle of a write
    again = reps.cached_encode(c, TEXTS, tmp_path)
    assert c.calls == 2 * len(TEXTS) and np.array_equal(first, again)
    reps.cached_encode(c, TEXTS, tmp_path)
    assert c.calls == 2 * len(TEXTS)                                                # the rewritten file is a good hit again
    path.write_bytes(b"")
    assert np.array_equal(first, reps.cached_encode(c, TEXTS, tmp_path))            # an empty file is just another miss


def test_cache_blocks_of_1000_keep_order(tmp_path):
    c = Counting()
    texts = [f"{i} plus {i + 1}" for i in range(2300)]
    z = reps.cached_encode(c, texts, tmp_path)
    assert z.shape == (2300, c.dim) and len(list(tmp_path.glob("*.npz"))) == 3
    assert np.array_equal(z, c.inner.encode(texts))
    c.calls = 0
    texts2 = texts[:1000] + ["a brand new sentence"] + texts[1001:]                 # only the second block changes
    z2 = reps.cached_encode(c, texts2, tmp_path)
    assert c.calls == 1000 and np.array_equal(z2[:1000], z[:1000]) and np.array_equal(z2[1001:], z[1001:])


def test_cache_files_depend_on_texts_rep_and_model(tmp_path):
    a, b = Counting(), Counting()
    b.tag = "a-different-model"
    reps.cached_encode(a, TEXTS, tmp_path)
    reps.cached_encode(b, TEXTS, tmp_path)
    reps.cached_encode(a, TEXTS[::-1], tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 3                                   # same texts under another model tag, or in another order, never collide


def test_atomic_write_bytes_leaves_no_temp_file(tmp_path):
    target = tmp_path / "x.bin"
    runlib.atomic_write_bytes(target, b"abc")
    runlib.atomic_write_bytes(target, b"abcdef")
    assert target.read_bytes() == b"abcdef" and [p.name for p in tmp_path.iterdir()] == ["x.bin"]


def test_rep_ids_and_layers():
    assert reps.REP_IDS == ("R0_hash", "R1_minilm384", "R1_minilm128", "R2_mpnet", "R3_bert",
                            "R4_gpt2_L4", "R4_gpt2_L6", "R4_gpt2_L8", "R4_qwen_L9", "R4_qwen_L14", "R4_qwen_L18")
    with pytest.raises(KeyError):
        reps.get_rep("R9_nothing")
    with pytest.raises(ValueError):
        reps.get_rep("R4_gpt2_L7", model_name="EleutherAI/pythia-70m")                          # the stand-in has only 6 layers
    assert reps.get_rep("R1_minilm384").dim == 384 and reps.get_rep("R1_minilm128").dim == 128 and reps.get_rep("R0_hash").dim == 128


def test_last_token_state_does_not_depend_on_padding():
    r = reps.get_rep("R4_gpt2_L6", model_name="EleutherAI/pythia-70m")
    alone = r.encode(["12 + 7"])[0]
    batched = r.encode(["12 + 7", "a much longer sentence about nothing in particular that forces padding of the short one"])[0]
    assert np.allclose(alone, batched, atol=1e-4)
    ids = r._input_ids("x" * 5000)
    assert ids[-len(r.suffix_ids):] == r.suffix_ids and len(ids) <= reps.MAX_TEXT_TOKENS + len(r.suffix_ids)   # the question survives a long text


def test_offline_mode_never_touches_the_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a network connection was attempted")
    reps.clear_models()
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for rid, model in SMALL[1:]:
        assert np.isfinite(reps.get_rep(rid, model_name=model).encode(TEXTS)).all()
    with pytest.raises(FileNotFoundError, match="no-such-org/missing-model"):
        reps.get_rep("R4_gpt2_L6", model_name="no-such-org/missing-model").encode(["x"])


def test_default_gpt2_ids_load_and_the_layers_differ():
    a, b = reps.get_rep("R4_gpt2_L4"), reps.get_rep("R4_gpt2_L8")
    assert a.dim == b.dim == 768 and a.tag != b.tag
    za, zb = a.encode(TEXTS), b.encode(TEXTS)
    assert np.isfinite(za).all() and np.allclose(np.linalg.norm(za, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(za, zb, atol=1e-3)                                           # layer 4 and layer 8 are different views of the same sentence
