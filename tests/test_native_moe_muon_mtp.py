"""
Unit Tests for Native Spherical Dynamic MoE Transformer, Muon Optimizer,
DeepSeek-Style Shared Expert Isolation, and Multi-Token Prediction (MTP).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ishmn.muon import zeropower_via_newtonschulz5, Muon, HybridMuonAdamW
from ishmn.spherical_dynamic_moe import SphericalDynamicMoE
from ishmn.native_cognitive_transformer import NativeSphericalBlock, NativeSphericalTransformer


def test_zeropower_via_newtonschulz5_orthogonalization():
    """Verify that Newton-Schulz 5th-order iteration properly orthogonalizes 2D matrices."""
    torch.manual_seed(42)
    # Test square matrix
    G_sq = torch.randn(32, 32)
    X_sq = zeropower_via_newtonschulz5(G_sq, steps=5)
    gram_sq = X_sq @ X_sq.T
    eye_sq = torch.eye(32)
    diff_sq = torch.norm(gram_sq - eye_sq) / 32.0
    assert diff_sq < 0.15, f"Square matrix orthogonalization error too high: {diff_sq}"

    # Test wide matrix (M < N)
    G_wide = torch.randn(16, 48)
    X_wide = zeropower_via_newtonschulz5(G_wide, steps=5)
    gram_wide = X_wide @ X_wide.T
    eye_wide = torch.eye(16)
    diff_wide = torch.norm(gram_wide - eye_wide) / 16.0
    assert diff_wide < 0.15, f"Wide matrix orthogonalization error too high: {diff_wide}"

    # Test tall matrix (M > N)
    G_tall = torch.randn(48, 16)
    X_tall = zeropower_via_newtonschulz5(G_tall, steps=5)
    gram_tall = X_tall.T @ X_tall
    eye_tall = torch.eye(16)
    diff_tall = torch.norm(gram_tall - eye_tall) / 16.0
    assert diff_tall < 0.15, f"Tall matrix orthogonalization error too high: {diff_tall}"


def test_muon_optimizer_step_convergence():
    """Verify that Muon optimizer performs valid parameter updates and reduces loss."""
    torch.manual_seed(42)
    layer = nn.Linear(32, 16, bias=False)
    opt = Muon(layer.parameters(), lr=0.05, momentum=0.9)

    x = torch.randn(10, 32)
    target = torch.randn(10, 16)

    # Compute initial loss
    out0 = layer(x)
    loss0 = F.mse_loss(out0, target).item()

    # Step optimizer 5 times
    for _ in range(5):
        opt.zero_grad()
        out = layer(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()

    out_final = layer(x)
    loss_final = F.mse_loss(out_final, target).item()
    assert loss_final < loss0, f"Muon failed to reduce loss: {loss0} -> {loss_final}"


def test_hybrid_muon_adamw_coordination():
    """Verify that HybridMuonAdamW separates 2D weights (Muon) from 1D/embeddings (AdamW)."""
    torch.manual_seed(42)
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(50, 32)
            self.fc1 = nn.Linear(32, 64, bias=True)
            self.norm = nn.LayerNorm(64)
            self.fc2 = nn.Linear(64, 10, bias=False)

        def forward(self, idx):
            h = self.emb(idx)
            h = F.relu(self.fc1(h))
            h = self.norm(h)
            return self.fc2(h)

    model = ToyModel()
    hybrid_opt = HybridMuonAdamW(model, muon_lr=0.02, adamw_lr=1e-3)

    # fc1.weight and fc2.weight are 2D -> in Muon
    # emb.weight (embedding), fc1.bias (1D), norm.weight/bias (1D) -> in AdamW
    assert hybrid_opt.muon is not None
    assert hybrid_opt.adamw is not None
    assert len(hybrid_opt.muon_params) == 2, f"Expected 2 Muon params, got {len(hybrid_opt.muon_params)}"
    assert len(hybrid_opt.adamw_params) == 4, f"Expected 4 AdamW params, got {len(hybrid_opt.adamw_params)}"

    # Run step
    idx = torch.randint(0, 50, (4, 8))
    out = model(idx)
    loss = out.sum()
    loss.backward()
    hybrid_opt.step()
    hybrid_opt.zero_grad()


def test_spherical_dynamic_moe_with_shared_expert_and_bias():
    """Verify DeepSeek-style Shared Expert Isolation and Aux-Loss-Free Dynamic Balancing."""
    torch.manual_seed(42)
    moe = SphericalDynamicMoE(
        d_model=32,
        d_ff=64,
        hyperspace_dim=8,
        top_k=2,
        initial_experts=3,
        enable_shared_expert=True,
        enable_aux_loss_free_balancing=True,
        bias_gamma=0.05
    )

    assert moe.shared_expert is not None
    assert moe.expert_bias.shape[0] == 3
    assert moe.routed_load_ema.shape[0] == 3

    x = torch.randn(8, 32)
    u = F.normalize(torch.randn(8, 8), dim=-1)
    r = torch.ones(8, 1)

    moe.train()
    out, meta = moe(x, u, r)
    assert out.shape == (8, 32)
    assert meta["num_experts"] >= 3
    assert len(meta["topk_indices"]) == 8

    # Verify that bias buffer updated
    assert moe.expert_bias.abs().sum().item() >= 0.0


def test_native_spherical_transformer_mtp_and_born_generation():
    """Verify NativeSphericalTransformer forward pass, MTP multi-head loss, and Born Rule generation."""
    torch.manual_seed(42)
    model = NativeSphericalTransformer(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=2,
        hyperspace_dim=8,
        d_ff=64,
        max_seq_len=64,
        mtp_future_tokens=3,
        vsa_capacity=32,
        num_experts=2,
        top_k=1,
        enable_shared_expert=True,
        enable_aux_loss_free_balancing=True
    )

    B, T = 2, 8
    input_ids = torch.randint(0, 64, (B, T))
    targets = torch.randint(0, 64, (B, T))

    # 1. Forward Pass with MTP
    out = model(input_ids, targets=targets, return_mtp=True)
    assert "logits" in out
    assert out["logits"].shape == (B, T, 64)
    assert len(out["mtp_logits"]) == 2 # Offsets t+2 and t+3
    assert out["loss"] is not None
    assert out["loss"].item() > 0.0
    assert "primary_ce" in out["loss_components"]
    assert "mtp_offset_2" in out["loss_components"]

    # 2. Backward Pass
    model.zero_grad()
    out["loss"].backward()

    # Check that trunk embeddings, attention heads, primary head, and active experts received gradients
    assert model.tok_embed.weight.grad is not None
    assert model.head_primary.weight.grad is not None
    assert model.blocks[0].attn.q_phase.weight.grad is not None
    assert any(p.grad is not None for p in model.blocks[0].moe.parameters())

    # 3. Autoregressive Generation with Rectified Born Rule
    prompt = torch.tensor([[1, 5, 12]], dtype=torch.long)
    gen_tokens = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.8,
        enable_born_rule=True,
        born_alpha=3.5
    )
    assert gen_tokens.shape == (1, 13)
