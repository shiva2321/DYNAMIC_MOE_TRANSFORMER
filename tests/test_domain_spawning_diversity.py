"""
Unit Test: Multi-Domain Autonomous Spawning & Routing Diversity.
Guarantees that distinct domain streams trigger distinct expert spawns
and prevents any single expert from seizing a routing monopoly.
"""

import os
import sys
import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.hyper_moe import DynamicHyperMoE
from hyperspace.vsa import ComplexPhasorVSA

def test_autonomous_multi_domain_spawning_diversity():
    d_model = 128
    d_ff = 256
    d_hyper = 1024
    batch_size = 8
    seq_len = 32
    
    # 1. Initialize with 2 bootstrap generic slots
    moe = DynamicHyperMoE(
        d_model=d_model,
        d_ff=d_ff,
        d_hyper=d_hyper,
        top_k=2,
        max_experts=16,
        spawn_threshold=0.20,
    )
    moe.train()
    assert moe.num_experts == 2, "Model must start with exactly 2 bootstrap experts"

    # 2. Simulate 4 highly distinct domain embedding streams (Code, Physics, Medicine, Law)
    # Each domain has a distinct semantic subspace direction
    torch.manual_seed(42)
    domain_directions = torch.randn(4, d_model)
    domain_directions = F.normalize(domain_directions, p=2, dim=-1)

    domain_expert_winners = []
    
    for d_idx in range(4):
        # Generate domain-specific batch
        domain_center = domain_directions[d_idx]
        noise = torch.randn(batch_size, seq_len, d_model) * 0.2
        domain_batch = domain_center.unsqueeze(0).unsqueeze(0) + noise
        
        # Forward pass with allow_spawning=True
        out, telem = moe(domain_batch, allow_spawning=True)
        
        # Record primary winning expert for this domain
        top_idx = telem["top_indices"] # [B, S, K]
        # Most frequent expert in slot 0
        slot0_exps = top_idx[:, :, 0].flatten()
        winner_exp = torch.mode(slot0_exps).values.item()
        domain_expert_winners.append(winner_exp)
        
        # Update usage counts
        for exp_id in range(moe.num_experts):
            moe.expert_usage_counts[exp_id] += (top_idx == exp_id).sum().item()

    # 3. Assertions for True Multi-Expert Diversity
    print(f"Spawned Experts Count: {moe.num_experts}")
    print(f"Domain Winners: {domain_expert_winners}")
    
    # (a) Model must have spawned new experts on the fly (more than 2 bootstrap)
    assert moe.num_experts >= 3, f"Expected at least 3 experts, got {moe.num_experts}"
    
    # (b) Distinct domains must not all route to the same single expert
    unique_winners = len(set(domain_expert_winners))
    assert unique_winners >= 2, f"Expected multiple unique domain winning experts, got {unique_winners}"
    
    # (c) No single expert should have 100% total routing weight across all 4 domains
    total_usage = moe.expert_usage_counts[:moe.num_experts].float().sum()
    max_usage_fraction = (moe.expert_usage_counts[:moe.num_experts].float().max() / total_usage).item()
    assert max_usage_fraction < 0.75, f"Monopoly detected! Single expert captured {max_usage_fraction*100:.1f}% of total usage"
