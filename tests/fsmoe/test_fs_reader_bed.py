import hashlib

import numpy as np
import pytest

from fsmoe import bed as BED
from fsmoe.frontend import Frontend
from fsmoe.reader import DIM, CachedReader, QwenReader


def _fake_vec(text: str) -> np.ndarray:
    seed = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).standard_normal(DIM).astype(np.float16)


class StubReader(QwenReader):
    """The real cache logic with a fake model: counts how many blocks it had to compute."""
    def __init__(self, cache_dir):
        super().__init__(cache_dir)
        self.computed = 0

    def _encode_block(self, texts):
        self.computed += 1
        return np.stack([_fake_vec(t) for t in texts])


def test_reader_cache_is_block_wise_atomic_and_resumable(tmp_path):
    r = StubReader(tmp_path)
    texts = [f"message {i}" for i in range(1200)]
    a = r.encode(texts, "x")
    assert a.shape == (1200, DIM) and a.dtype == np.float16 and r.computed == 3
    b = r.encode(texts, "x")
    assert r.computed == 3 and np.array_equal(a, b)                                  # a rerun costs nothing
    c = r.encode(texts + ["one more"], "x")
    assert r.computed == 4 and np.array_equal(c[:1200], a)                           # only the new block is computed
    assert not list(tmp_path.rglob("*.tmp.npy"))
    assert r.encode([], "y").shape == (0, DIM)


def test_reader_cache_key_depends_on_the_texts_and_the_settings(tmp_path):
    r1, r2 = StubReader(tmp_path), StubReader(tmp_path)
    r2.suffix = "\nsomething else"
    r1.encode(["a", "b"], "x")
    r2.encode(["a", "b"], "x")
    assert r2.computed == 1 and r1.tag != r2.tag
    r1.encode(["a", "c"], "x")
    assert r1.computed == 2


class StubMini(CachedReader):
    dim = 384
    tag = "stub_mini"

    def _encode_block(self, texts):
        return np.stack([_fake_vec(t)[:384] for t in texts])


def test_frontend_only_accepts_teaching_rows_and_gives_unit_vectors():
    raw = np.random.default_rng(0).standard_normal((300, DIM)).astype(np.float16)
    with pytest.raises(ValueError):
        Frontend(dims=(DIM,)).fit(raw, np.array([True] * 299 + [False]))
    for mode in ("rp", "pca"):
        fe = Frontend(64, mode, dims=(DIM,)).fit(raw, np.ones(300, dtype=bool))
        z = fe(raw[:10])
        assert z.shape == (10, 64) and np.allclose(z.norm(dim=1).numpy(), 1.0, atol=1e-5)
        assert np.array_equal(z.numpy(), fe(raw[:10]).numpy())
    with pytest.raises(RuntimeError):
        Frontend(dims=(DIM,))(raw[:2])


def test_frontend_with_two_blocks_is_unit_length_and_checks_the_width():
    raw = np.random.default_rng(1).standard_normal((200, DIM + 384)).astype(np.float16)
    fe = Frontend(32).fit(raw, np.ones(200, dtype=bool))
    z = fe(raw[:7])
    assert fe.out_dim == 64 and z.shape == (7, 64) and np.allclose(z.norm(dim=1).numpy(), 1.0, atol=1e-5)
    with pytest.raises(ValueError):
        fe(raw[:7, :DIM])


def test_two_readers_are_concatenated_side_by_side(tmp_path):
    out = BED._encode([StubReader(tmp_path), StubMini(tmp_path)], ["a", "bb", "ccc"], "x")
    assert out.shape == (3, DIM + 384) and np.array_equal(out[:, :DIM], np.stack([_fake_vec(t) for t in ("a", "bb", "ccc")]))


def test_bed_masks_teaching_rows_and_withheld_skills(tmp_path):
    bed = BED.build_bed(5, [StubReader(tmp_path), StubMini(tmp_path)], sizes={"arith": (2, 1), "code": (2, 1), "lang": (2, 1)}, with_far=False)
    teach = bed.teaching()
    assert teach.sum() > 0
    assert bed.taught[teach].all() and not bed.heldout[teach].any() and (bed.split[teach] == "teach").all()
    assert not set(bed.skill[teach]) & set(bed.withheld)
    ev = bed.mask(split="eval")
    assert set(bed.skill[ev]) >= set(bed.withheld) and bed.heldout[ev].any()
    assert bed.raw.shape == (len(bed.text), DIM + 384) and bed.y.max() < len(bed.skills)
    assert all(t in bed.skills for t in bed.twin.values())
    assert bed.far_withheld <= bed.withheld
    for name in ("arith", "code", "lang"):
        assert bed.mask(domain=name).sum() > 0
