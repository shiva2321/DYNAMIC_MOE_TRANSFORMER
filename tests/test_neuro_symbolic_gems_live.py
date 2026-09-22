"""
Unit tests for Harvested Neuro-Symbolic Gems Integration:
1. Gem 1: Continuous Refractory Satiation Field sigma(v) for loop suppression.
2. Gem 3: Global Workspace Theory (GWT) Coalition MoE Routing.
3. Strict greedy argmax decoding at temperature <= 0.0.
4. State dict backward compatibility across GWT routing additions.
"""

import pytest
import torch
import torch.nn.functional as F

from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.spherical_dynamic_moe import SphericalDynamicMoE
from ishmn.geometry import project_to_sphere


def test_refractory_satiation_field():
    device = torch.device("cpu")
    vocab_size = 50
    d_model = 32
    seq_len = 16

    model = NativeSphericalTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=2,
        num_heads=2,
        hyperspace_dim=6,
        d_ff=64,
        max_seq_len=seq_len + 16,
        num_experts=2,
        top_k=1,
        device=device
    ).to(device)
    model.eval()

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)

    # 1. Greedy decoding with satiation disabled (pure argmax baseline)
    gen_greedy = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.0,
        enable_born_rule=False,
        enable_satiation=False
    )
    assert gen_greedy.shape == (1, 13)

    # 2. Greedy decoding with Refractory Satiation Field enabled (Gem 1)
    gen_satiated = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.0,
        enable_born_rule=False,
        enable_satiation=True,
        satiation_alpha=5.0,
        satiation_decay=0.70,
        satiation_threshold=0.05
    )
    assert gen_satiated.shape == (1, 13)

    # Check for immediate repeat suppression
    tokens_satiated = gen_satiated[0].tolist()
    immediate_repeats = sum(1 for i in range(len(tokens_satiated) - 1) if tokens_satiated[i] == tokens_satiated[i + 1])
    # With satiation_alpha=5.0, immediate repeats should be heavily suppressed
    assert immediate_repeats <= 1


def test_gwt_coalition_moe_routing():
    device = torch.device("cpu")
    B = 2
    d_model = 16
    d_ff = 32
    d_context = 16
    hyperspace_dim = 6

    # Instantiate MoE with GWT Coalition enabled
    moe = SphericalDynamicMoE(
        d_model=d_model,
        d_ff=d_ff,
        d_context=d_context,
        hyperspace_dim=hyperspace_dim,
        initial_experts=3,
        top_k=2,
        enable_gwt_coalition=True,
        gwt_context_weight=0.5,
        device=device
    )

    x = torch.randn(B, d_model, requires_grad=True)
    u_x = project_to_sphere(torch.randn(B, hyperspace_dim))
    r_x = torch.ones(B, 1)

    # Context A vs Context B
    ctx_a = torch.randn(B, d_context, requires_grad=True)
    ctx_b = -ctx_a

    # Forward pass with Context A
    out_a, meta_a = moe(x, u_x, r_x, context=ctx_a)
    assert out_a.shape == (B, d_model)
    assert not torch.isnan(out_a).any()

    # Forward pass with Context B (should shift resonance and bids)
    out_b, meta_b = moe(x, u_x, r_x, context=ctx_b)
    assert out_b.shape == (B, d_model)
    assert not torch.isnan(out_b).any()

    # Verify that different contexts produce different routing bids / outputs
    assert not torch.allclose(out_a, out_b, atol=1e-4)

    # Verify clean gradient propagation to GWT projection weights
    loss = out_a.sum()
    loss.backward()
    assert moe.context_to_proposal.weight.grad is not None
    assert not torch.isnan(moe.context_to_proposal.weight.grad).any()


def test_gwt_moe_backward_compatibility():
    device = torch.device("cpu")
    # 1. MoE without GWT (standard baseline)
    moe_old = SphericalDynamicMoE(
        d_model=16,
        d_ff=32,
        d_context=16,
        hyperspace_dim=6,
        initial_experts=2,
        enable_gwt_coalition=False,
        device=device
    )
    state_dict_old = moe_old.state_dict()

    # 2. MoE with GWT enabled
    moe_new = SphericalDynamicMoE(
        d_model=16,
        d_ff=32,
        d_context=16,
        hyperspace_dim=6,
        initial_experts=2,
        enable_gwt_coalition=True,
        device=device
    )

    # Strict load should succeed thanks to backward-compatibility shim
    moe_new.load_state_dict(state_dict_old, strict=True)
    assert moe_new.context_to_proposal is not None
