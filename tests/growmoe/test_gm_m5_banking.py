"""M5: class-incremental banking77 stream tests."""

from pathlib import Path
import pytest
import torch

# Skip tests if local cache is not available
cache_base = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--banking77"
if not cache_base.exists() or not list(cache_base.glob("snapshots/*/data/*.parquet")):
    pytest.skip("banking77 local cache missing", allow_module_level=True)

from growmoe.streams_real import banking_incremental


def test_shapes_and_unit_norm():
    st = banking_incremental(seed=0)
    assert st.n_classes == 77
    assert st.d == 128
    for seg in st.segments:
        assert seg.x.shape == (len(seg.y), st.d)
        norms = seg.x.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    for reg in st.regimes.values():
        assert reg.x_te.shape == (len(reg.y_te), st.d)
        norms = reg.x_te.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_deterministic_given_a_seed():
    st1 = banking_incremental(seed=42)
    st2 = banking_incremental(seed=42)
    assert st1.meta["schedule"] == st2.meta["schedule"]
    assert st1.meta["groups"] == st2.meta["groups"]
    assert torch.equal(st1.segments[0].x, st2.segments[0].x)
    assert torch.equal(st1.segments[0].y, st2.segments[0].y)
    assert torch.equal(st1.regimes[0].x_te, st2.regimes[0].x_te)
    assert torch.equal(st1.regimes[0].y_te, st2.regimes[0].y_te)


def test_different_seeds_differ():
    st1 = banking_incremental(seed=0)
    st2 = banking_incremental(seed=1)
    assert st1.meta["groups"] != st2.meta["groups"]
    assert not torch.equal(st1.segments[0].x, st2.segments[0].x)


def test_label_blocks_disjoint_and_cover_77_intents_once():
    st = banking_incremental(seed=0)
    all_labels = set()
    for r in range(len(st.regimes)):
        reg_labels = set(st.regimes[r].meta["labels"])
        assert len(reg_labels) > 0
        assert all_labels.isdisjoint(reg_labels), f"Regime {r} labels overlap with previous regimes"
        all_labels.update(reg_labels)
    assert all_labels == set(range(77))
    assert len(all_labels) == 77


def test_every_segment_contains_only_its_regimes_labels():
    st = banking_incremental(seed=0)
    for seg in st.segments:
        reg_labels = set(st.regimes[seg.regime].meta["labels"])
        seg_labels = set(seg.y.tolist())
        assert seg_labels.issubset(reg_labels), f"Segment in regime {seg.regime} has foreign labels {seg_labels - reg_labels}"


def _share_fraction(train_x: torch.Tensor, test_x: torch.Tensor, chunk_size: int = 512, tol: float = 1e-5) -> float:
    n_test = len(test_x)
    if n_test == 0:
        return 0.0
    close_count = 0
    for lo in range(0, n_test, chunk_size):
        te_chunk = test_x[lo:lo + chunk_size]
        dists = torch.cdist(te_chunk, train_x, compute_mode="donot_use_mm_for_euclid_dist")
        min_dists, _ = dists.min(dim=1)
        close_count += (min_dists <= tol).sum().item()
    return close_count / n_test


def test_share_fraction_helper():
    rng = torch.Generator().manual_seed(42)
    tr = torch.randn(100, 128, generator=rng)
    tr = tr / tr.norm(dim=1, keepdim=True)
    te_copies = tr[:50].clone()
    assert _share_fraction(tr, te_copies) == 1.0

    te_indep = torch.randn(100, 128, generator=rng)
    te_indep = te_indep / te_indep.norm(dim=1, keepdim=True)
    assert _share_fraction(tr, te_indep) == 0.0

    te_half = torch.cat([tr[:50].clone(), te_indep[:50]])
    assert _share_fraction(tr, te_half) == 0.5


def test_no_test_document_appears_in_any_training_segment():
    st = banking_incremental(seed=0)
    train_x = torch.cat([seg.x for seg in st.segments])
    test_x = torch.cat([reg.x_te for reg in st.regimes.values()])
    frac = _share_fraction(train_x, test_x)
    print(f"Measured share fraction: {frac:.6f}")
    assert frac <= 0.01


def test_schedule_is_as_specified():
    revisit = (2, 0, 4)
    st = banking_incremental(seed=0, n_regimes=6, n_train=400, revisit=revisit, revisit_size=200)
    expected_schedule = [0, 1, 2, 3, 4, 5, 2, 0, 4]
    assert st.meta["schedule"] == expected_schedule
    assert [seg.regime for seg in st.segments] == expected_schedule
    assert [seg.first_visit for seg in st.segments] == [True] * 6 + [False] * 3
    assert [len(seg.y) for seg in st.segments] == [400] * 6 + [200] * 3


def test_test_sets_contain_only_regimes_intents():
    st = banking_incremental(seed=0, n_test=150)
    for r, reg in st.regimes.items():
        assert len(reg.y_te) <= 150
        reg_labels = set(reg.meta["labels"])
        assert set(reg.y_te.tolist()) == reg_labels
        assert len(reg.meta["intents"]) == len(reg_labels)
