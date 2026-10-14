"""RV2-B: the stage-1 fingerprint of an expert (spec section 5): k unit sub-centroids of its stored prototypes."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from growmoe.fingerprint import Fingerprint, spherical_kmeans
from growmoe.signature import VmfSignature

D = 16


def _blobs(n_per=40, centres=3, seed=0):
    g = np.random.default_rng(seed)
    c = F.normalize(torch.from_numpy(g.standard_normal((centres, D)).astype(np.float32)), dim=1)
    pts = torch.cat([F.normalize(c[i] * 6 + torch.from_numpy(g.standard_normal((n_per, D)).astype(np.float32)), dim=1) for i in range(centres)])
    return pts, c


def _sig(points, seed=0):
    s = VmfSignature(D, 256, seed=seed)
    s.update(points)
    s.fit()
    return s


def test_kmeans_returns_unit_rows_and_is_deterministic():
    pts, _ = _blobs()
    a, b = spherical_kmeans(pts, 3, seed=1), spherical_kmeans(pts, 3, seed=1)
    assert a.shape == (3, D) and torch.equal(a, b)
    assert torch.allclose(a.norm(dim=1), torch.ones(3), atol=1e-5)


def test_kmeans_with_one_cluster_is_the_normalised_mean():
    pts, _ = _blobs()
    c = spherical_kmeans(pts, 1, seed=0)
    assert torch.allclose(c[0], F.normalize(pts.mean(0), dim=0), atol=1e-5)


def test_kmeans_finds_well_separated_blobs():
    pts, centres = _blobs(n_per=60, centres=3, seed=2)
    found = spherical_kmeans(pts, 3, iters=15, seed=0)
    assert float((found @ centres.T).max(dim=1).values.min()) > 0.9                     # every centroid sits on one blob


def test_k_is_capped_at_the_number_of_points():
    pts, _ = _blobs(n_per=1, centres=3)
    assert spherical_kmeans(pts, 8).shape == (3, D)


def test_a_fingerprint_is_built_from_the_signature_and_goes_stale_after_enough_new_points():
    pts, _ = _blobs()
    sig = _sig(pts)
    fp = Fingerprint.build(sig, 4, seed=0)
    assert fp.centroids.shape == (4, D) and fp.built_seen == sig.seen
    assert not fp.stale(sig, 128)
    sig.update(pts[:100])
    assert not fp.stale(sig, 128)
    sig.update(pts[:40])
    assert fp.stale(sig, 128)


def test_padded_repeats_rows_so_every_expert_has_exactly_k():
    pts, _ = _blobs(n_per=1, centres=2)
    fp = Fingerprint.build(_sig(torch.cat([pts, pts])), 4, seed=0)                      # only 2 distinct prototypes -> at most 4 rows after repeating
    assert fp.padded(4).shape == (4, D)
    assert fp.padded(1).shape == (1, D)
