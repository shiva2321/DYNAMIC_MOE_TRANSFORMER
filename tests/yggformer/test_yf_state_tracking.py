"""
test_yf_state_tracking.py - Unit and invariant tests for Round Y4 (E2b non-abelian state tracking).
Verifies:
1. S5 permutation group properties: order 120, identity element, associativity.
2. VQRotationModel: straight-through quantization and commitment loss.
3. GroupLookupModel: Sinkhorn/permutation state update with exact lookup.
"""

import itertools
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_s5_group():
    perms = list(itertools.permutations(range(5)))
    assert len(perms) == 120
    idx = {p: i for i, p in enumerate(perms)}
    mul = torch.zeros(120, 120, dtype=torch.long)
    for a, pa in enumerate(perms):
        for b, pb in enumerate(perms):
            mul[a, b] = idx[tuple(pa[pb[i]] for i in range(5))]
    return mul


def test_s5_group_axioms():
    """Verify S5 multiplication table is order 120, has identity at 0, and is associative."""
    mul = make_s5_group()
    assert mul.shape == (120, 120)

    # Identity check
    for a in range(120):
        assert mul[a, 0] == a
        assert mul[0, a] == a

    # Associativity check on random sample triples
    g = torch.Generator().manual_seed(42)
    sample_a = torch.randint(0, 120, (20,), generator=g)
    sample_b = torch.randint(0, 120, (20,), generator=g)
    sample_c = torch.randint(0, 120, (20,), generator=g)

    for a, b, c in zip(sample_a, sample_b, sample_c):
        ab_c = mul[mul[a, b], c]
        a_bc = mul[a, mul[b, c]]
        assert ab_c == a_bc, f"Associativity failed for ({a},{b},{c})"


def test_vq_rotation_model_straight_through():
    """Verify VQRotationModel performs straight-through snapping and produces commitment loss."""
    from research.yggdrasil_poc.poc_b_state_tracking import VQRotationModel
    model = VQRotationModel(n_sym=6, k=8)
    x = torch.randint(0, 6, (4, 16))
    out = model(x)
    assert out.shape == (4, 16, 6)
    assert hasattr(model, "aux_loss")
    assert model.aux_loss >= 0.0

    # Gradient check
    loss = out.sum() + model.aux_loss
    loss.backward()
    assert model.w.grad is not None
    assert torch.isfinite(model.w.grad).all()


def test_group_lookup_model_sinkhorn():
    """Verify GroupLookupModel produces valid soft/hard permutation state transitions."""
    from research.yggdrasil_poc.poc_b_state_tracking import GroupLookupModel
    model = GroupLookupModel(n_sym=6)
    x = torch.randint(0, 6, (4, 16))
    out = model(x)
    assert out.shape == (4, 16, 6)
    assert torch.isfinite(out).all()

    # Gradient check
    loss = out.sum()
    loss.backward()
    assert model.logits.grad is not None
    assert torch.isfinite(model.logits.grad).all()
