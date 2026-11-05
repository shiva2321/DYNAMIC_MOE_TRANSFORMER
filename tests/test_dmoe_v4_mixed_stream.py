"""
Unit tests for dMoE v4 mixed-stream continual learning:
1. SequenceContentRouter per-sequence signature extraction and routing accuracy.
2. Per-sequence MoE dispatch on interleaved batches (B2).
3. NullSpaceProjector preservation of prior domain subspaces (always-plastic).
4. ContinualLearner always-plastic training and mixed-stream inference.
"""

import math
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.run_dmoe_v2_easy_eval import generate_easy_domain_batch
from dmoe.continual import ContinualLearner
from dmoe.model import SelfStabilizingDynamicMorphicMoE
from dmoe.null_space import NullSpaceProjector
from dmoe.routing import SequenceContentRouter


def test_sequence_content_router_accuracy():
    """Verify SequenceContentRouter achieves >= 95% accuracy on interleaved sequences."""
    torch.manual_seed(42)
    router = SequenceContentRouter(d_model=32)

    model = SelfStabilizingDynamicMorphicMoE(
        d_model=32,
        num_heads=2,
        d_ff=128,
        initial_layers=2,
        initial_experts=1,
        tokenizer_free=True,
        chunk_size=4,
    )
    model.eval()

    # Collect domain signatures for 3 domains
    exemplars = []
    for d in range(3):
        x, _ = generate_easy_domain_batch(d, batch_size=32, seq_len=16, seed=100 + d)
        sig = router.extract_signature(x, model.encoder)
        assert sig.shape == (32, 32)
        assert torch.allclose(torch.norm(sig, dim=-1), torch.ones(32), atol=1e-5)
        exemplars.append(sig)

    # Interleave sequences from domains 0, 1, 2
    x0, _ = generate_easy_domain_batch(0, batch_size=32, seq_len=16, seed=200)
    x1, _ = generate_easy_domain_batch(1, batch_size=32, seq_len=16, seed=201)
    x2, _ = generate_easy_domain_batch(2, batch_size=32, seq_len=16, seed=202)

    x_mixed = torch.cat([x0, x1, x2], dim=0)  # [96, 16]
    labels_true = torch.tensor([0] * 32 + [1] * 32 + [2] * 32)

    # Shuffle
    perm = torch.randperm(96)
    x_mixed = x_mixed[perm]
    labels_true = labels_true[perm]

    sig_mixed = router.extract_signature(x_mixed, model.encoder)
    assigned, max_sims, is_novel = router.route_sequences(sig_mixed, exemplars)
    assert assigned.shape == (96,)
    assert max_sims.shape == (96,)

    acc = (assigned == labels_true).float().mean().item()
    assert acc >= 0.95, f"Expected routing accuracy >= 0.95, got {acc:.4f}"


def test_mixed_stream_per_sequence_dispatch():
    """Verify DynamicMorphicMoE and SelfStabilizingDynamicMorphicMoE dispatch per-sequence."""
    torch.manual_seed(42)
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=32,
        num_heads=2,
        d_ff=128,
        initial_layers=2,
        initial_experts=1,
        tokenizer_free=True,
        chunk_size=4,
        enable_local_lda=True,
        enable_setgate=True,
    )
    model.eval()

    # Populate 3 experts
    router = model.sequence_router
    exemplars = []
    for d in range(3):
        if d > 0:
            for layer in model.layers:
                layer.moe.spawn_expert()
        x, _ = generate_easy_domain_batch(d, batch_size=32, seq_len=16, seed=300 + d)
        sig = router.extract_signature(x, model.encoder)
        exemplars.append(sig)

    model.update_sequence_exemplars(exemplars)
    assert len([e for e in model.layers[0].moe.experts if getattr(e, "exemplars", None) is not None]) == 3

    # Mixed batch: 10 seqs from dom 0, 10 from dom 1, 10 from dom 2
    x0, _ = generate_easy_domain_batch(0, batch_size=10, seq_len=16, seed=400)
    x1, _ = generate_easy_domain_batch(1, batch_size=10, seq_len=16, seed=401)
    x2, _ = generate_easy_domain_batch(2, batch_size=10, seq_len=16, seed=402)
    x_mixed = torch.cat([x0, x1, x2], dim=0)

    out = model(x_mixed, output_head="byte_chunk", update_stats=False)
    assert "output" in out
    assert out["output"].shape == (30, 4, 4, 256)

    assigned = getattr(model, "last_assigned_experts", None)
    assert assigned is not None
    assert assigned.shape == (30,)
    # Sequences 0..9 should be routed to 0, 10..19 to 1, 20..29 to 2
    expected = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
    acc = (assigned == expected).float().mean().item()
    assert acc >= 0.95, f"Expected dispatch accuracy >= 0.95, got {acc:.4f}"


def test_null_space_projector_preserves_prior_subspace():
    """Verify NullSpaceProjector ensures Delta W @ U_prior = 0 while weights update."""
    torch.manual_seed(42)
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=32,
        num_heads=2,
        d_ff=128,
        initial_layers=2,
        initial_experts=1,
        tokenizer_free=True,
        chunk_size=4,
    )
    nsp = NullSpaceProjector(model, energy_threshold=0.98, max_rank_ratio=0.6)

    # 1. Domain 0 training activations
    x0, _ = generate_easy_domain_batch(0, batch_size=32, seq_len=16, seed=500)
    model.train()
    _ = model(x0, output_head="byte_chunk")
    ranks_added = nsp.update_basis_for_domain(domain_id=0)
    assert any(r > 0 for r in ranks_added.values())

    # Snapshot current weights and prior subspace activations
    tracked_layer = model.layers[0].attn.q_proj
    w_real_orig = tracked_layer.weight_real.clone().detach()
    U0 = nsp.bases["layers.0.attn.q_proj"]
    assert U0.size(0) == 32
    k0 = U0.size(1)
    assert k0 > 0

    # 2. Domain 1 forward and backward pass
    x1, _ = generate_easy_domain_batch(1, batch_size=32, seq_len=16, seed=501)
    model.zero_grad()
    out1 = model(x1, output_head="byte_chunk")
    loss1 = out1["output"].sum()
    loss1.backward()

    # Project gradients onto null space
    nsp.project_gradients()

    # Verify perturbation on prior subspace
    grad_real = tracked_layer.weight_real.grad.data
    # Delta W = -eta * grad_real
    # For any vector v in span(U0): Delta W @ v == 0
    perturbation = grad_real @ U0
    rel_pert = perturbation.norm() / (grad_real.norm() + 1e-9)
    assert rel_pert.item() < 1e-4, f"Relative perturbation on prior subspace too large: {rel_pert.item()}"

    # Also verify that gradients are NOT all zero (learning on new directions is active!)
    assert grad_real.abs().max().item() > 1e-4, "Gradients should not be completely zeroed!"


def test_continual_learner_always_plastic_mixed_stream():
    """Verify ContinualLearner trains with null-space projection without freezing backbone."""
    torch.manual_seed(42)
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=32,
        num_heads=2,
        d_ff=128,
        initial_layers=2,
        initial_experts=1,
        tokenizer_free=True,
        chunk_size=4,
        enable_local_lda=True,
        enable_setgate=True,
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=4e-3)
    learner = ContinualLearner(
        model=model,
        optimizer=opt,
        replay_buffer_capacity=0,
        enable_local_lda=True,
        enable_setgate=True,
        freeze_backbone_on_domain=False,  # ALWAYS-PLASTIC
        enable_null_space=True,
    )

    # Train on domain 0
    for st in range(5):
        x0, y0 = generate_easy_domain_batch(0, batch_size=16, seq_len=16, seed=600 + st)
        res0 = learner.step(x0, y0, output_head="byte_chunk")
        assert math.isfinite(res0["total_loss"])

    # Domain transition: should NOT freeze backbone
    learner.domain_transition(new_domain_id=1)
    assert model.encoder.unify_real.weight.requires_grad is True
    assert model.layers[0].attn.q_proj.weight_real.requires_grad is True

    # Train on domain 1
    for st in range(5):
        x1, y1 = generate_easy_domain_batch(1, batch_size=16, seq_len=16, seed=700 + st)
        res1 = learner.step(x1, y1, output_head="byte_chunk")
        assert math.isfinite(res1["total_loss"])

    # Evaluate on mixed-stream batch
    x0_test, _ = generate_easy_domain_batch(0, batch_size=16, seq_len=16, seed=800)
    x1_test, _ = generate_easy_domain_batch(1, batch_size=16, seq_len=16, seed=801)
    x_mix = torch.cat([x0_test, x1_test], dim=0)

    # Populate sequence router exemplars
    sig0 = model.sequence_router.extract_signature(x0_test, model.encoder)
    sig1 = model.sequence_router.extract_signature(x1_test, model.encoder)
    model.update_sequence_exemplars([sig0, sig1])

    logits = learner.predict_logits(x_mix, output_head="byte_chunk")
    assert logits.shape == (32, 4 * 4, 256)
    assert torch.isfinite(logits).all()


def test_pure_neural_continual_learning_no_lda():
    """Verify pure neural continual learning without LDA maintains modular expert heads."""
    torch.manual_seed(42)
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=32,
        num_heads=2,
        d_ff=128,
        initial_layers=2,
        initial_experts=1,
        tokenizer_free=True,
        chunk_size=4,
        enable_local_lda=False,
        enable_setgate=True,
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=4e-3)
    learner = ContinualLearner(
        model=model,
        optimizer=opt,
        replay_buffer_capacity=0,
        enable_local_lda=False,
        enable_setgate=True,
        freeze_backbone_on_domain=True,
        enable_null_space=False,
    )

    # Train on domain 0
    for st in range(10):
        x0, y0 = generate_easy_domain_batch(0, batch_size=16, seq_len=16, seed=900 + st)
        res0 = learner.step(x0, y0, output_head="byte_chunk")
        assert math.isfinite(res0["total_loss"])

    # Domain transition: freezes expert 0 and head 0, spawns expert 1 and head 1
    learner.domain_transition(new_domain_id=1)
    assert len(model.layers[0].moe.experts) == 2
    assert len(model.decoder.expert_heads) == 2
    assert model.layers[0].moe.expert_frozen[0].item() is True
    assert model.layers[0].moe.expert_frozen[1].item() is False

    # Check head 0 frozen and head 1 trainable
    assert all(not p.requires_grad for p in model.decoder.expert_heads[0].parameters())
    assert all(p.requires_grad for p in model.decoder.expert_heads[1].parameters())

    # Train on domain 1
    for st in range(10):
        x1, y1 = generate_easy_domain_batch(1, batch_size=16, seq_len=16, seed=950 + st)
        res1 = learner.step(x1, y1, output_head="byte_chunk")
        assert math.isfinite(res1["total_loss"])

    # Evaluate pure neural forward pass on mixed stream
    x0_test, _ = generate_easy_domain_batch(0, batch_size=16, seq_len=16, seed=1000)
    x1_test, _ = generate_easy_domain_batch(1, batch_size=16, seq_len=16, seed=1001)
    x_mix = torch.cat([x0_test, x1_test], dim=0)

    logits = learner.predict_logits(x_mix, output_head="byte_chunk")
    assert logits.shape == (32, 4, 4, 256)
    assert torch.isfinite(logits).all()

