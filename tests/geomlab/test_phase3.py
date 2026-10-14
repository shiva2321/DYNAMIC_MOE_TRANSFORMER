"""
Unit tests for Phase 3: Unified Heterogeneous Multi-Manifold Architecture (GeoCore).
"""

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from geomlab.phase3_geocore import (
    build_bound_tracking_dataset,
    BaselineTransformer,
    BaselineRecurrent,
    UncleanedGeoCore,
    FullGeoCore,
)


def test_bound_tracking_dataset_shapes_and_logic():
    data = build_bound_tracking_dataset(
        num_entities=32,
        num_roles=4,
        num_train=100,
        num_val=30,
        num_ood=20,
        train_lengths=(1, 2),
        ood_lengths=(4, 8),
        seed=42,
    )

    tr = data["train"]
    val = data["val"]
    ood = data["ood_dict"]
    feats = data["entity_features"]

    assert feats.shape == (32, 1024)
    assert tr["fillers"].shape == (100, 4)
    assert tr["trans"].shape == (100, 2)
    assert tr["queries"].shape == (100,)
    assert tr["targets"].shape == (100,)
    assert val["fillers"].shape == (30, 4)

    for k in (4, 8):
        assert ood[k]["fillers"].shape == (20, 4)
        assert ood[k]["trans"].shape == (20, k)
        assert ood[k]["targets"].shape == (20,)


def test_models_forward_and_shapes():
    num_entities = 32
    d_model = 64
    feats = torch.randn(num_entities, 1024)
    B = 8
    fillers = torch.randint(0, num_entities, (B, 4))
    trans = torch.randint(0, 4, (B, 2))
    queries = torch.randint(0, 4, (B,))
    lengths = torch.tensor([1, 2, 2, 1, 2, 1, 2, 1], dtype=torch.long)

    models = [
        BaselineTransformer(feats, num_roles=4, num_trans=4, d_model=d_model, n_heads=2, n_layers=1),
        BaselineRecurrent(feats, num_roles=4, num_trans=4, d_model=d_model),
        UncleanedGeoCore(feats, num_roles=4, num_trans=4, d_model=d_model, block_size=4),
        FullGeoCore(feats, num_roles=4, num_trans=4, d_model=d_model, block_size=4),
    ]

    for m in models:
        logits, rep = m(fillers, trans, queries, lengths)
        assert logits.shape == (B, num_entities), f"Model {m.__class__.__name__} failed logits shape"
        assert not torch.isnan(logits).any(), f"Model {m.__class__.__name__} produced NaNs"


def test_hopfield_cleanup_denoises():
    """
    Verifies that the Metric Hopfield Attractor takes a noisy phasor corrupted by
    superposition crosstalk and increases its cosine similarity to the ground-truth target.
    """
    N = 32
    D = 64
    torch.manual_seed(42)
    feats = torch.randn(N, 1024)
    model = FullGeoCore(feats, num_roles=4, num_trans=4, d_model=D, block_size=4, hopfield_beta=20.0)

    entity_phases = model._get_entity_phases()  # (N, D)
    cb_phasors = torch.exp(1j * entity_phases)

    # Pick target entity 5
    target_idx = 5
    clean_target = cb_phasors[target_idx]  # (D,)

    # Add crosstalk noise (superposition with 3 other entities)
    crosstalk = cb_phasors[1] + cb_phasors[12] + cb_phasors[20]
    noisy_phasor = clean_target + 0.3 * crosstalk
    noisy_unit = noisy_phasor / noisy_phasor.abs()

    # Similarity before cleanup
    sim_before = (noisy_unit * clean_target.conj()).real.mean().item()

    # Apply Hopfield cleanup
    cleaned_unit = model._hopfield_cleanup(noisy_unit.unsqueeze(0), entity_phases).squeeze(0)
    sim_after = (cleaned_unit * clean_target.conj()).real.mean().item()

    assert sim_after > sim_before, f"Hopfield failed to improve similarity: before={sim_before:.3f}, after={sim_after:.3f}"
    assert sim_after > 0.85, f"Hopfield failed to snap close to prototype: {sim_after:.3f}"
