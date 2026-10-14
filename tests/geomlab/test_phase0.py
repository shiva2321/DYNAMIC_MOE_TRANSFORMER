"""
Unit tests for Phase 0: Group symmetries, algebraic models, and invariants.
"""

import math
import pytest
import torch
from geomlab.phase0_algebra import (
    build_z113_dataset,
    build_s5_dataset,
    BaselineMLP,
    BaselineTransformer,
    TorusPhaseModel,
    BlockUnitaryModel,
    compute_fourier_participation_ratio,
)


def test_z113_dataset():
    p = 113
    train_x, train_y, test_x, test_y = build_z113_dataset(p=p, train_fraction=0.3, seed=42)
    assert len(train_x) == int(p * p * 0.3)
    assert len(train_x) + len(test_x) == p * p
    assert train_x.shape == (len(train_x), 2)
    assert train_y.shape == (len(train_x),)

    # Check targets match (a + b) % p
    for i in range(min(100, len(train_x))):
        a, b = train_x[i].tolist()
        expected = (a + b) % p
        assert train_y[i].item() == expected

    # Check disjoint sets
    train_set = set(tuple(x.tolist()) for x in train_x)
    test_set = set(tuple(x.tolist()) for x in test_x)
    assert len(train_set.intersection(test_set)) == 0


def test_s5_dataset():
    train_x, train_y, test_x, test_y, perms = build_s5_dataset(train_fraction=0.3, seed=42)
    assert len(perms) == 120
    assert len(train_x) == int(120 * 120 * 0.3)
    assert len(train_x) + len(test_x) == 120 * 120

    # Verify group identity element
    identity_perm = (0, 1, 2, 3, 4)
    id_idx = perms.index(identity_perm)
    
    # Test random compositions
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    for i in range(min(100, len(train_x))):
        idx1, idx2 = train_x[i].tolist()
        p1, p2 = perms[idx1], perms[idx2]
        composed = tuple(p1[p2[k]] for k in range(5))
        assert train_y[i].item() == perm_to_idx[composed]


def test_models_forward_and_invariants():
    batch_size = 16
    num_tokens = 113
    d_model = 64
    x = torch.randint(0, num_tokens, (batch_size, 2))

    # Baseline MLP
    mlp = BaselineMLP(num_tokens=num_tokens, d_model=d_model)
    out_mlp = mlp(x)
    assert out_mlp.shape == (batch_size, num_tokens)

    # Baseline Transformer
    transformer = BaselineTransformer(num_tokens=num_tokens, d_model=d_model, n_heads=4)
    out_tf = transformer(x)
    assert out_tf.shape == (batch_size, num_tokens)

    # Torus Phase Model
    torus = TorusPhaseModel(num_tokens=num_tokens, d_model=d_model)
    out_torus = torus(x)
    assert out_torus.shape == (batch_size, num_tokens)

    # Test exact torus identity: if theta_a + theta_b == theta_c, cosine similarity is 1.0
    with torch.no_grad():
        torus.phases[2] = torus.phases[0] + torus.phases[1]
        test_pair = torch.tensor([[0, 1]])
        logits = torus(test_pair)
        # Token 2 should have maximum similarity
        assert torch.argmax(logits[0]).item() == 2


def test_block_unitary_invariants():
    batch_size = 8
    num_tokens = 120
    d_model = 64
    b = 4
    x = torch.randint(0, num_tokens, (batch_size, 2))

    model = BlockUnitaryModel(num_tokens=num_tokens, d_model=d_model, b=b)
    
    # Verify unitary matrix property: U^H @ U == I
    U_all = model.get_unitary()  # (120, M, 4, 4)
    num_blocks = d_model // b
    assert U_all.shape == (num_tokens, num_blocks, b, b)

    I_expected = torch.eye(b, dtype=torch.complex64).unsqueeze(0).unsqueeze(0)
    U_H = U_all.conj().transpose(-1, -2)
    prod = torch.matmul(U_H, U_all)
    diff = (prod - I_expected).abs().max().item()
    assert diff < 1e-5, f"Matrix is not unitary! Max deviation from identity: {diff}"

    # Verify forward shape
    logits = model(x)
    assert logits.shape == (batch_size, num_tokens)


def test_fourier_participation_ratio():
    p = 113
    d = 32
    # Case 1: single pure Fourier frequency -> participation ratio should be close to 1
    pure_phase = torch.zeros(p, d)
    for a in range(p):
        pure_phase[a, :] = 2 * math.pi * a * 3 / p  # freq k = 3
    pr_pure = compute_fourier_participation_ratio(pure_phase, p=p)
    assert pr_pure < 1.5, f"Expected PR near 1 for pure frequency, got {pr_pure}"

    # Case 2: random uniform phases on circle -> participation ratio should be close to p
    rng_phase = torch.empty(p, d).uniform_(-math.pi, math.pi)
    pr_noise = compute_fourier_participation_ratio(rng_phase, p=p)
    assert pr_noise > 80.0, f"Expected high PR for random noise, got {pr_noise}"
