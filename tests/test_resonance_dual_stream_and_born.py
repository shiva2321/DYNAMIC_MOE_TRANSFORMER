"""
Test Suite: Resonance X Synthesis & Dual-Stream Phasor Integrations for ISHMN.
Tests:
1. DualStreamRiemannianAttention & DualStreamRiemannianBlock
2. Rectified Born Rule & BornRuleLogitsProcessor
3. VSAEpisodicMemory (Differentiable Read, Hebbian Write, Vigilance Gate, Direct Edit)
4. SphericalCircularVarianceLoss (Circular Directional Statistics)
5. LoopedRiemannianTransformer with Dual-Stream
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ishmn.geometry import project_to_sphere, geodesic_distance_sphere
from ishmn.spherical_attention import (
    DualStreamRiemannianAttention,
    DualStreamRiemannianBlock,
    LoopedRiemannianTransformer
)
from ishmn.born_rule import rectified_born, BornRuleLogitsProcessor
from ishmn.vsa_memory import VSAEpisodicMemory
from ishmn.sigreg import SphericalCircularVarianceLoss


class TestDualStreamAttention:
    def test_dual_stream_attention_shapes_and_modulation(self):
        B, T, D = 2, 4, 32
        hyperspace_dim = 8
        num_heads = 4

        attn = DualStreamRiemannianAttention(
            d_model=D,
            num_heads=num_heads,
            hyperspace_dim=hyperspace_dim,
            initial_kappa=4.0
        )

        x = torch.randn(B, T, D, requires_grad=True)
        x_mag = torch.randn(B, T, D, requires_grad=True)

        out_fused, out_mag, meta = attn(x, x_mag=x_mag)

        assert out_fused.shape == (B, T, D)
        assert out_mag.shape == (B, T, D)
        assert "phase_weights" in meta
        assert "mean_kappa_eff" in meta
        assert meta["mean_kappa_eff"] >= 0.1

        # Test autograd flow through both streams
        loss = out_fused.sum() + out_mag.sum()
        loss.backward()

        assert x.grad is not None
        assert x_mag.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(x_mag.grad).any()

    def test_dual_stream_block_forward(self):
        B, T, D = 2, 6, 32
        block = DualStreamRiemannianBlock(
            d_model=D,
            num_heads=4,
            hyperspace_dim=8,
            d_ff=64
        )

        x = torch.randn(B, T, D)
        x_out, x_mag_out = block(x)

        assert x_out.shape == (B, T, D)
        assert x_mag_out.shape == (B, T, D)
        assert not torch.isnan(x_out).any()

    def test_looped_dual_stream_transformer(self):
        B, T, D = 2, 5, 32
        looped = LoopedRiemannianTransformer(
            d_model=D,
            num_heads=4,
            hyperspace_dim=8,
            d_ff=64,
            max_loops=4,
            use_dual_stream=True
        )

        x = torch.randn(B, T, D)
        out, meta = looped(x)

        assert out.shape == (B, T, D)
        assert meta["final_loops"] >= 1
        assert "ponder_cost" in meta
        assert not torch.isnan(out).any()


class TestBornRule:
    def test_rectified_born_basic(self):
        V = 100
        sims = torch.randn(V)
        result = rectified_born(sims, alpha=0.2, gate_threshold=0.01)

        assert result.probs.shape == (V,)
        assert math.isclose(float(result.probs.sum().item()), 1.0, rel_tol=1e-4)
        assert 0 <= result.best_idx < V
        assert result.best_prob > 0.0
        assert result.entropy >= 0.0

    def test_rectified_born_repetition_suppression(self):
        V = 50
        D = 16
        vocab = torch.randn(V, D)
        vocab = F.normalize(vocab, p=2, dim=-1)

        # Target token 5 is identical to recent token
        hist = vocab[5:6]  # (1, D)
        sims = torch.ones(V) * 0.5
        sims[5] = 0.9  # Token 5 had highest initial similarity

        result = rectified_born(
            similarity=sims,
            history_embeds=hist,
            vocab_matrix=vocab,
            alpha=0.8,
            gate_threshold=0.01
        )

        # Token 5 should suffer heavy destructive interference
        assert result.best_idx != 5
        assert result.probs[5] < result.probs[result.best_idx]

    def test_rectified_born_logits_shift_invariance(self):
        """Verify that rectified_born with is_logits=True is strictly shift-invariant (z -> z + C)."""
        V = 64
        torch.manual_seed(42)
        raw_logits = torch.randn(V) * 5.0
        shifted_logits = raw_logits + 100.0  # Large constant shift

        res1 = rectified_born(raw_logits, is_logits=True)
        res2 = rectified_born(shifted_logits, is_logits=True)

        assert torch.allclose(res1.probs, res2.probs, atol=1e-5), "Born probabilities changed under logit constant shift!"
        assert res1.best_idx == res2.best_idx
        assert math.isclose(res1.best_prob, res2.best_prob, rel_tol=1e-4)
        assert math.isclose(res1.margin, res2.margin, rel_tol=1e-4)

    def test_born_rule_logits_processor(self):
        V, D = 100, 32
        embed = nn.Embedding(V, D)
        processor = BornRuleLogitsProcessor(embed_tokens=embed, alpha=0.3, window_size=8)

        # Simulate input_ids with repetition: [1, 5, 5, 5]
        input_ids = torch.tensor([[1, 10, 5, 5, 5]], dtype=torch.long)
        scores = torch.randn(1, V)
        scores[0, 5] = 10.0  # Strongly favors repeating 5

        processed = processor(input_ids, scores)
        assert processed.shape == (1, V)
        # Processed logit for 5 should be reduced
        assert processed[0, 5] < 10.0


class TestVSAEpisodicMemory:
    def test_vsa_differentiable_read_empty(self):
        D, payload_dim = 8, 32
        mem = VSAEpisodicMemory(hyperspace_dim=D, payload_dim=payload_dim, capacity=64)

        u_q = project_to_sphere(torch.randn(2, 3, D))
        r_q = torch.ones(2, 3, 1)

        echo, conf = mem(u_q, r_q)
        assert echo.shape == (2, 3, payload_dim)
        assert conf.shape == (2, 3, 1)
        assert (echo == 0).all()
        assert (conf == 0).all()

    def test_vsa_hebbian_write_and_read(self):
        D, payload_dim = 8, 32
        mem = VSAEpisodicMemory(hyperspace_dim=D, payload_dim=payload_dim, capacity=64)

        u1 = project_to_sphere(torch.randn(D))
        r1 = torch.tensor([[1.5]])
        v1 = torch.randn(payload_dim)

        # Write first association
        alpha_used = mem.write(u1, r1, v1)
        assert alpha_used > 0.0
        assert mem.get_occupancy_ratio() == 1.0 / 64

        # Read back at exact coordinate
        echo, conf = mem(u1.unsqueeze(0), r1.unsqueeze(0))
        assert echo.shape == (1, payload_dim)
        cos_sim = F.cosine_similarity(echo, v1.unsqueeze(0)).item()
        assert cos_sim > 0.90  # Accurate retrieval

    def test_vsa_vigilance_gate_skips_redundancy(self):
        D, payload_dim = 8, 32
        mem = VSAEpisodicMemory(hyperspace_dim=D, payload_dim=payload_dim, capacity=64, vigilance=0.85)

        u = project_to_sphere(torch.randn(D))
        r = torch.tensor([[2.0]])
        v = torch.randn(payload_dim)

        # First write: completely novel, must write
        a1 = mem.write(u, r, v)
        assert a1 > 0.0

        # Second write with identical (u, r, v): memory already predicts it -> vigilance skips!
        a2 = mem.write(u, r, v)
        assert a2 == 0.0  # Skipped!
        assert mem.occupied.sum().item() == 1  # No duplicate slot consumed

    def test_vsa_direct_edit(self):
        D, payload_dim = 8, 32
        mem = VSAEpisodicMemory(hyperspace_dim=D, payload_dim=payload_dim, capacity=16)

        u = project_to_sphere(torch.randn(D))
        r = torch.tensor([[1.0]])
        v_override = torch.ones(payload_dim) * 5.0

        mem.direct_edit(slot_idx=0, u=u, r=r, v=v_override)

        echo, _ = mem(u.unsqueeze(0), r.unsqueeze(0))
        assert torch.allclose(echo.squeeze(0), v_override, atol=1e-4)


class TestSphericalCircularVarianceLoss:
    def test_circular_variance_uniform_vs_cone(self):
        D = 8
        loss_fn = SphericalCircularVarianceLoss(tau_cone=0.50, lambda_cone=2.0)

        # Case 1: Uniformly dispersed points around sphere
        u_uniform = project_to_sphere(torch.randn(1000, D))
        loss_uniform, diag_uniform = loss_fn(u_uniform)

        assert diag_uniform["cone_norm"] < 0.20  # Near origin
        assert diag_uniform["circular_variance"] > 0.80
        assert loss_uniform.item() == 0.0  # Zero penalty
        assert not diag_uniform["collapsed"]

        # Case 2: Highly clustered points (cone collapse)
        center = project_to_sphere(torch.randn(1, D))
        u_cone = project_to_sphere(center + 0.05 * torch.randn(50, D))
        loss_cone, diag_cone = loss_fn(u_cone)

        assert diag_cone["cone_norm"] > 0.90  # Highly clustered
        assert diag_cone["circular_variance"] < 0.10
        assert loss_cone.item() > 0.0  # Penalty triggered
        assert diag_cone["collapsed"]
