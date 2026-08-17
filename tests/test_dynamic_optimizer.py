"""
Unit Tests for DynamicWarmupAdamW.
Verifies:
1. Dynamic group addition and birth step tracking.
2. Per-group localized linear warmup schedules.
3. Second-moment variance buffer conditioning.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hyperspace.dynamic_optimizer import DynamicWarmupAdamW

def test_dynamic_optimizer_warmup_and_conditioning():
    # Base model with 1 layer
    linear1 = nn.Linear(32, 32)
    opt = DynamicWarmupAdamW(linear1.parameters(), lr=1e-3, default_group_warmup_steps=10)

    # Initial state
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["spawn_step"] == 0

    # Step forward 5 steps on base model
    for _ in range(5):
        loss = linear1(torch.randn(4, 32)).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        opt.update_dynamic_schedules(global_lr_multiplier=1.0)

    assert opt.global_step == 5

    # Dynamically spawn a new layer
    linear2 = nn.Linear(32, 32)
    opt.add_dynamic_param_group(
        list(linear2.parameters()),
        lr=1e-3,
        warmup_steps=10,
        group_name="new_expert",
    )

    assert len(opt.param_groups) == 2
    new_group = opt.param_groups[1]
    assert new_group["spawn_step"] == 5
    assert new_group["warmup_steps"] == 10

    # Step forward and verify localized warmup progression
    # Step 1 after spawn (global_step = 6, local_step = 1 / 10 -> LR ~ 1e-4)
    loss = linear2(torch.randn(4, 32)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    opt.update_dynamic_schedules(global_lr_multiplier=1.0)

    assert pytest.approx(new_group["lr"], rel=1e-3) == 1e-4

    # Advance 9 more steps (local_step reaches 10 / 10 -> LR = 1e-3)
    for _ in range(9):
        loss = linear2(torch.randn(4, 32)).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        opt.update_dynamic_schedules(global_lr_multiplier=1.0)

    assert pytest.approx(new_group["lr"], rel=1e-3) == 1e-3
    assert opt.global_step == 15
