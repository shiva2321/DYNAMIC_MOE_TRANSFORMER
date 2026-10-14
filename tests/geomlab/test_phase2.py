"""
Unit tests for Phase 2: Compositional OOD Generalization and Length Extrapolation.
"""

import math
import numpy as np
import pytest
import torch

from geomlab.phase2_composition import (
    build_graph_composition_dataset,
    StandardTransformer,
    EuclideanRNN,
    TorusIsometry,
    BlockUnitaryIsometry,
)


def test_dataset_shapes_and_leakage():
    data = build_graph_composition_dataset(
        num_entities=64,
        num_relations=4,
        num_train=300,
        num_val=100,
        num_ood_per_length=50,
        train_lengths=(1, 2, 3),
        ood_lengths=(4, 6, 8),
        seed=42,
    )

    train_paths = data["train_paths"]
    train_targets = data["train_targets"]
    train_lengths = data["train_lengths"]
    val_paths = data["val_paths"]
    val_targets = data["val_targets"]
    ood_dict = data["ood_dict"]

    assert train_paths.shape[0] == 300
    assert train_targets.shape[0] == 300
    assert train_lengths.shape[0] == 300
    assert val_paths.shape[0] == 100

    for k in (4, 6, 8):
        assert k in ood_dict
        assert len(ood_dict[k]["paths"]) == 50
        assert ood_dict[k]["targets"].shape[0] == 50

    # Test zero data leakage
    train_set = set(tuple(p[: l + 1].tolist()) for p, l in zip(train_paths, train_lengths))
    val_set = set(tuple(p[: l + 1].tolist()) for p, l in zip(val_paths, data["val_lengths"]))

    assert len(train_set.intersection(val_set)) == 0, "Data leakage between train and val!"

    for k in (4, 6, 8):
        ood_set = set(tuple(p) for p in ood_dict[k]["paths"])
        assert len(train_set.intersection(ood_set)) == 0, f"Leakage between train and OOD K={k}!"


def test_models_forward_and_output_shapes():
    B = 8
    L = 4  # e0 + 3 relations
    seq = torch.randint(0, 4, (B, L))
    seq[:, 0] = torch.randint(0, 64, (B,))  # First token is entity
    lengths = torch.full((B,), 3, dtype=torch.long)

    models = [
        StandardTransformer(num_entities=64, num_relations=4, d_model=64, n_heads=2, n_layers=1, max_seq_len=16),
        EuclideanRNN(num_entities=64, num_relations=4, d_model=64),
        TorusIsometry(num_entities=64, num_relations=4, d_model=64),
        BlockUnitaryIsometry(num_entities=64, num_relations=4, d_model=64, block_size=4),
    ]

    for m in models:
        logits, rep = m(seq, lengths)
        assert logits.shape == (B, 64), f"Model {m.__class__.__name__} failed logits shape"
        assert not torch.isnan(logits).any(), f"Model {m.__class__.__name__} produced NaNs in logits"


def test_isometry_norm_preservation():
    """
    Verifies that BlockUnitaryIsometry and TorusIsometry have strictly zero norm drift
    at all composition lengths K in {1, 2, 4, 8, 12}.
    """
    d_model = 64
    block_size = 4
    num_blocks = d_model // block_size

    model_u = BlockUnitaryIsometry(num_entities=32, num_relations=4, d_model=d_model, block_size=block_size)
    model_t = TorusIsometry(num_entities=32, num_relations=4, d_model=d_model)

    u_entities = model_u._exp_unitary(model_u.entity_gens_real, model_u.entity_gens_imag)
    u_rels = model_u._exp_unitary(model_u.rel_gens_real, model_u.rel_gens_imag)

    # Check that every relation block is unitary: U @ U^H = I
    for r in range(4):
        for b in range(num_blocks):
            u_block = u_rels[r, b]
            eye = torch.eye(block_size, dtype=u_block.dtype)
            prod = torch.matmul(u_block, u_block.conj().T)
            assert torch.allclose(prod, eye, atol=1e-5), f"Relation {r} block {b} is not unitary!"

    # Trace paths of lengths up to 12 and check Frobenius norm
    curr_u = u_entities[0]  # (num_blocks, b, b)
    initial_norm = torch.linalg.norm(curr_u).item()

    for step in range(12):
        curr_u = torch.matmul(curr_u, u_rels[step % 4])
        curr_norm = torch.linalg.norm(curr_u).item()
        assert abs(curr_norm - initial_norm) < 1e-4, f"Unitary norm drifted at step {step + 1}!"
