"""RV3: per-expert source windows (spec section 4.2)."""

import numpy as np
import pytest
import torch

from growmoe.settest import generator
from growmoe.sources import SourceWindows

D = 16


def _batch(n, axis, seed, spread=0.35):
    """n unit vectors around the coordinate axis ``axis``."""
    g = np.random.default_rng(seed)
    c = np.zeros(D)
    c[axis] = 1.0
    z = c[None, :] + spread * g.standard_normal((n, D)) / np.sqrt(D)
    return torch.nn.functional.normalize(torch.from_numpy(z.astype(np.float32)), dim=1)


def _feed(w, axis, n_batches, seed0, step0=0, b=16):
    opened = 0
    for k in range(n_batches):
        opened += w.update(_batch(b, axis, seed0 + k), generator(7, step0 + k), step=step0 + k)
    return opened


def test_a_fresh_store_has_no_references_and_the_first_batch_opens_a_window():
    w = SourceWindows()
    assert len(w) == 0 and w.n_samples() == 0 and w.refs() == []
    assert w.update(_batch(16, 0, 1), generator(0)) is False
    assert len(w) == 1 and w.n_samples() == 16
    assert w.refs() == []                                                                # 16 rows < min_ref 48: not yet a reference


def test_batches_of_one_source_share_one_window_and_it_keeps_only_the_last_five_batches():
    w = SourceWindows(max_batches=5)
    assert _feed(w, 0, 12, seed0=10) == 0
    assert len(w) == 1 and w.n_samples() == 5 * 16
    (ref,) = w.refs()
    assert ref.shape == (80, D)


def test_a_batch_from_another_source_opens_a_new_window_and_returning_to_the_first_opens_a_third():
    w = SourceWindows()
    _feed(w, 0, 5, seed0=10)                                                             # A: 80 rows
    assert _feed(w, 5, 1, seed0=100, step0=5) == 1                                      # B opens window 2
    _feed(w, 5, 4, seed0=101, step0=6)
    assert _feed(w, 0, 1, seed0=200, step0=10) == 1                                     # back to A opens window 3 (windows never merge back)
    assert len(w) == 3
    means = [r.mean(0) for r in w.refs()]
    assert len(means) == 2                                                               # the third window holds one batch (16 rows < 48)
    assert float(means[0][0]) > 0.8 and float(means[1][5]) > 0.8                         # window 1 is source A, window 2 is source B


def test_the_number_of_windows_is_capped_and_the_oldest_is_dropped():
    w = SourceWindows(max_windows=3)
    for i, axis in enumerate([0, 3, 6, 9, 12]):
        _feed(w, axis, 4, seed0=300 + 10 * i, step0=10 * i)
    assert len(w) == 3
    centres = [int(torch.cat(list(win.batches)).mean(0).argmax()) for win in w._wins]
    assert centres == [6, 9, 12]


def test_small_batches_and_young_windows_never_split():
    w = SourceWindows(min_ref=48, min_split=8)
    _feed(w, 0, 2, seed0=1)                                                              # 32 rows: below min_ref
    assert w.update(_batch(16, 7, 5), generator(0), step=9) is False and len(w) == 1
    w2 = SourceWindows(min_ref=48, min_split=8)
    _feed(w2, 0, 5, seed0=1)
    assert w2.update(_batch(7, 7, 6), generator(0), step=9) is False and len(w2) == 1   # 7 rows < min_split: joins the window, however foreign


def test_the_stored_rows_are_a_copy_and_an_empty_batch_is_ignored():
    w = SourceWindows()
    xb = _batch(16, 0, 1)
    before = xb.clone()
    w.update(xb, generator(0))
    xb.zero_()
    assert torch.equal(w._wins[0].cat(), before)
    assert w.update(torch.zeros(0, D), generator(0)) is False and w.n_samples() == 16


def test_a_huge_batch_is_stored_as_a_bounded_random_subset():
    w = SourceWindows(max_rows=64)
    w.update(_batch(1000, 0, 1), generator(0))
    assert w.n_samples() == 64


def test_merge_takes_the_union_by_recency_and_respects_the_cap():
    a, b = SourceWindows(max_windows=3), SourceWindows(max_windows=3)
    _feed(a, 0, 4, seed0=1, step0=0)
    _feed(a, 4, 4, seed0=20, step0=4)                                                    # a: sources 0 (last push 3) and 4 (last push 7)
    _feed(b, 8, 4, seed0=40, step0=10)
    _feed(b, 12, 4, seed0=60, step0=14)                                                  # b: sources 8 (last 13) and 12 (last 17)
    a.merge_from(b)
    assert len(a) == 3
    assert [int(torch.cat(list(win.batches)).mean(0).argmax()) for win in a._wins] == [4, 8, 12]
    n = a.n_samples()
    a.merge_from(a)
    a.merge_from(None)
    a.merge_from(SourceWindows())
    assert len(a) == 3 and a.n_samples() == n
    assert len(b) == 2                                                                   # the donor is untouched


def test_the_same_inputs_and_seeds_give_the_same_windows():
    def build():
        w = SourceWindows()
        _feed(w, 0, 5, seed0=1)
        _feed(w, 5, 5, seed0=50, step0=5)
        _feed(w, 0, 5, seed0=90, step0=10)
        return [x.sum().item() for x in w.refs()], len(w)
    assert build() == build()


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        SourceWindows(max_windows=0)
    with pytest.raises(ValueError):
        SourceWindows(split_alpha=1.5)
    with pytest.raises(ValueError):
        SourceWindows(stat="nope")


def test_reference_pairs_carry_a_cached_mean_that_is_dropped_when_the_window_changes():
    w = SourceWindows(min_ref=16)
    _feed(w, 0, 3, seed0=1)
    ((rows, mean),) = w.ref_pairs()
    assert torch.allclose(mean, rows.mean(0), atol=1e-6)
    ((rows2, mean2),) = w.ref_pairs()
    assert rows2 is rows and mean2 is mean                                               # cached: no recomputation while the window is unchanged
    _feed(w, 0, 1, seed0=50, step0=3)
    ((rows3, mean3),) = w.ref_pairs()
    assert rows3 is not rows and rows3.shape[0] == rows.shape[0] + 16 and not torch.equal(mean3, mean)
    assert [r.shape for r in w.refs()] == [rows3.shape]
