import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import sys

SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from dmoe.routing import SetGate, perm_test_exchangeability
from dmoe.experts import LocalLdaExpert, DynamicMorphicMoE
from dmoe.model import SelfStabilizingDynamicMorphicMoE


def test_perm_test_exchangeability_identical_distributions():
    """Identical distributions should be accepted (p > alpha, rejected=False)."""
    torch.manual_seed(42)
    ref = torch.randn(64, 32)
    sample = torch.randn(64, 32)  # drawn from same N(0, 1)

    rejected, p_val = perm_test_exchangeability(sample, ref, alpha=0.05, num_perms=100, seed=123)
    assert not rejected, f"Identical distributions rejected with p={p_val}"
    assert p_val > 0.05


def test_perm_test_exchangeability_distinct_distributions():
    """Shifted distributions should be rejected (p <= alpha, rejected=True)."""
    torch.manual_seed(42)
    ref = torch.randn(64, 32)
    sample = torch.randn(64, 32) + 2.5  # shifted centroid

    rejected, p_val = perm_test_exchangeability(sample, ref, alpha=0.05, num_perms=100, seed=123)
    assert rejected, f"Shifted distribution failed to reject: p={p_val}"
    assert p_val <= 0.05


def test_setgate_routing_known_vs_novel():
    """SetGate should route matching latents to owner and mark novel latents as 'novel'."""
    gen = torch.Generator().manual_seed(42)
    gate = SetGate(d_model=32, alpha=0.05, num_perms=100)

    # Expert 0: centered at 0
    ref0 = torch.randn(64, 32, generator=gen)
    # Expert 1: centered at +3.0
    ref1 = torch.randn(64, 32, generator=gen) + 3.0

    exemplars = [ref0, ref1]

    # Batch A matches Expert 0
    batch_a = torch.randn(32, 32, generator=gen)
    verdict_a, owner_a, p_a = gate.route_batch(batch_a, exemplars, step=0)
    assert verdict_a == "known"
    assert owner_a == 0

    # Batch B matches Expert 1
    batch_b = torch.randn(32, 32, generator=gen) + 3.0
    verdict_b, owner_b, p_b = gate.route_batch(batch_b, exemplars, step=1)
    assert verdict_b == "known"
    assert owner_b == 1

    # Batch C is novel (centered at -5.0)
    batch_c = torch.randn(32, 32, generator=gen) - 5.0
    verdict_c, owner_c, p_c = gate.route_batch(batch_c, exemplars, step=2)
    assert verdict_c == "novel"
    assert owner_c == -1


def test_local_lda_expert_zero_gradient_corruption():
    """Absorbing and optimizing one expert must not corrupt frozen sibling experts."""
    torch.manual_seed(42)
    exp0 = LocalLdaExpert(32, 64, enable_lda_head=True, num_classes=16, num_slots=2)
    exp1 = LocalLdaExpert(32, 64, enable_lda_head=True, num_classes=16, num_slots=2)

    # Train exp0 with 5 samples
    feats0 = torch.randn(8, 2, 2, 64)
    tgts0 = torch.randint(0, 16, (8, 4))
    exp0.absorb(feats0, tgts0)
    preds0_before = exp0.predict_logits(feats0).clone()
    weights0_before = {k: v.clone() for k, v in exp0.named_parameters()}

    # Freeze exp0
    for p in exp0.parameters():
        p.requires_grad_(False)

    # Train exp1 via gradient descent and absorb different targets
    opt1 = torch.optim.AdamW(exp1.parameters(), lr=1e-2)
    for _ in range(5):
        opt1.zero_grad()
        dummy_in_r = torch.randn(4, 2, 32)
        dummy_in_i = torch.randn(4, 2, 32)
        out_r, out_i = exp1(dummy_in_r, dummy_in_i)
        loss = out_r.sum() + out_i.sum()
        loss.backward()
        opt1.step()

    feats1 = torch.randn(8, 2, 2, 64) + 10.0
    tgts1 = torch.randint(0, 16, (8, 4))
    exp1.absorb(feats1, tgts1)

    # Verify exp0 weights and predictions are completely unaltered
    preds0_after = exp0.predict_logits(feats0)
    assert torch.allclose(preds0_before, preds0_after, atol=1e-6), "Exp0 LDA predictions drifted!"
    for k, v in exp0.named_parameters():
        assert torch.allclose(weights0_before[k], v, atol=1e-7), f"Exp0 parameter {k} changed!"


def test_dmoe_v3_model_level_setgate_routing_and_isolation():
    """End-to-end test: SetGate routes chunk 0 latents to domain experts with zero cross-talk."""
    from dmoe.continual import ContinualLearner
    import run_dmoe_v2_easy_eval as easy

    torch.manual_seed(42)
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=32,
        num_heads=2,
        d_ff=64,
        initial_layers=2,
        initial_experts=1,
        tokenizer_free=True,
        chunk_size=4,
        enable_local_lda=True,
        enable_setgate=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = ContinualLearner(
        model, opt, replay_buffer_capacity=0, enable_local_lda=True,
        enable_setgate=True, freeze_backbone_on_domain=True,
    )

    # Train 10 steps on Domain 0
    for st in range(10):
        x0, y0 = easy.generate_easy_domain_batch(0, batch_size=8, seq_len=16, seed=st)
        learner.step(x0, y0, output_head="byte_chunk")

    # Transition to Domain 1
    learner.domain_transition(new_domain_id=1)
    assert len(model.layers[0].moe.experts) == 2
    assert bool(model.layers[0].moe.expert_frozen[0]) is True
    assert bool(model.layers[0].moe.expert_frozen[1]) is False

    # Train 10 steps on Domain 1
    for st in range(10):
        x1, y1 = easy.generate_easy_domain_batch(1, batch_size=8, seq_len=16, seed=100 + st)
        learner.step(x1, y1, output_head="byte_chunk")

    # Evaluate routing on Domain 0 test batch
    x0_test, y0_test = easy.generate_easy_domain_batch(0, batch_size=16, seq_len=16, seed=999)
    model.eval()
    learner.predict_logits(x0_test, output_head="byte_chunk")
    assert model.last_routed_expert == 0, f"Domain 0 test batch routed to expert {model.last_routed_expert}"

    # Evaluate routing on Domain 1 test batch
    x1_test, y1_test = easy.generate_easy_domain_batch(1, batch_size=16, seq_len=16, seed=1000)
    learner.predict_logits(x1_test, output_head="byte_chunk")
    assert model.last_routed_expert == 1, f"Domain 1 test batch routed to expert {model.last_routed_expert}"

