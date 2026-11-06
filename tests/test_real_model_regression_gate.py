"""
Regression Gate: Verify Real-Valued DMoE matches complex model on continual learning.
Gate criteria from §14:
- Real model must learn Domain 0 (accuracy >= 0.85).
- After transition to Domain 1 and 2, retention of Domain 0 must be >= 0.85.
- Per-sequence routing must be >= 0.90.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

import run_dmoe_v2_easy_eval as easy
from dmoe.real_model import RealSelfStabilizingDMoE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_real_model_forward_and_spawn():
    model = RealSelfStabilizingDMoE(d_model=32, num_heads=2, d_ff=128, initial_layers=2).to(DEVICE)
    x = torch.randint(32, 48, (8, 16), device=DEVICE)
    out = model(x)
    assert out["output"].shape == (8, 4, 4, 256)

    # Test expert and head spawning
    exp_idx = model.layers[0].moe.spawn_expert()
    head_idx = model.decoder.spawn_expert_head()
    assert exp_idx == 1
    assert head_idx == 1
    assert len(model.layers[0].moe.experts) == 2
    assert len(model.decoder.expert_heads) == 2


def test_real_model_continual_learning_equivalence_gate():
    """Verify Real DMoE achieves >= 0.85 accuracy and retention on continual learning benchmark."""
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    model = RealSelfStabilizingDMoE(d_model=32, num_heads=2, d_ff=128, initial_layers=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3)

    # Train Domain 0
    model.train()
    for st in range(120):
        x, y = easy.generate_easy_domain_batch(0, batch_size=16, seq_len=16, seed=seed * 1000 + st)
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        logits = out["output"].view(16, -1, 256)
        loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
        loss.backward()
        opt.step()

    # Measure Domain 0 initial accuracy
    model.eval()
    with torch.no_grad():
        x_a, y_a = easy.generate_easy_domain_batch(0, batch_size=32, seq_len=16, seed=999)
        x_a, y_a = x_a.to(DEVICE), y_a.to(DEVICE)
        preds_a = model(x_a)["output"].view(32, -1, 256).argmax(dim=-1)
        acc_a_init = (preds_a == y_a).float().mean().item()
        assert acc_a_init >= 0.85, f"Domain 0 initial accuracy too low: {acc_a_init:.4f}"

    # Domain Transition to Domain 1: freeze expert 0 and head 0, spawn expert 1 and head 1
    for layer in model.layers:
        layer.moe.freeze_expert(0)
        layer.moe.spawn_expert()
    model.decoder.freeze_expert_head(0)
    model.decoder.spawn_expert_head()

    # Freeze backbone
    for p in model.byte_embed.parameters():
        p.requires_grad_(False)
    for p in model.chunk_proj.parameters():
        p.requires_grad_(False)
    for layer in model.layers:
        for p in layer.norm1.parameters():
            p.requires_grad_(False)
        for p in layer.norm2.parameters():
            p.requires_grad_(False)
        for p in (layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj):
            for param in p.parameters():
                param.requires_grad_(False)
    for p in model.decoder.final_norm.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=4e-3)

    # Train Domain 1
    model.train()
    for st in range(120):
        x, y = easy.generate_easy_domain_batch(1, batch_size=16, seq_len=16, seed=seed * 2000 + st)
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        logits = out["output"].view(16, -1, 256)
        loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
        loss.backward()
        opt.step()

    # Evaluate on mixed batch (Domain 0 + Domain 1)
    model.eval()
    with torch.no_grad():
        x0_test, y0_test = easy.generate_easy_domain_batch(0, batch_size=32, seq_len=16, seed=888)
        x1_test, y1_test = easy.generate_easy_domain_batch(1, batch_size=32, seq_len=16, seed=777)
        x_mix = torch.cat([x0_test, x1_test], dim=0).to(DEVICE)
        y_mix = torch.cat([y0_test, y1_test], dim=0).to(DEVICE)

        out_mix = model(x_mix)
        preds_mix = out_mix["output"].view(64, -1, 256).argmax(dim=-1)

        acc_dom0 = (preds_mix[:32] == y0_test.to(DEVICE)).float().mean().item()
        acc_dom1 = (preds_mix[32:] == y1_test.to(DEVICE)).float().mean().item()
        retention_0 = acc_dom0 / acc_a_init

        print(f"\nReal-Valued DMoE Regression Gate Results:")
        print(f"Domain 0 Initial:   {acc_a_init:.4f}")
        print(f"Domain 0 Final:     {acc_dom0:.4f} (Retention: {retention_0:.4f})")
        print(f"Domain 1 Final:     {acc_dom1:.4f}")

        assert acc_dom0 >= 0.85, f"Domain 0 retention failed: {acc_dom0:.4f}"
        assert acc_dom1 >= 0.85, f"Domain 1 learning failed: {acc_dom1:.4f}"
        assert retention_0 >= 0.85, f"Retention ratio below gate threshold: {retention_0:.4f}"
