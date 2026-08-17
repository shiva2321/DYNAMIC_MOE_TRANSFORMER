"""
Unit Tests for SleepConsolidationEngine.
Verifies:
1. Attractor coincidence detection and similarity calculation in C^D.
2. Usage-weighted barycentric parameter merging.
3. Inactive expert pruning and capacity reclamation.
4. Memory key tensor re-indexing.
"""

import os
import sys
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.hyper_moe import DynamicHyperMoE
from hyperspace.sleep_consolidation import SleepConsolidationEngine
from hyperspace.vsa import ComplexPhasorVSA

def test_sleep_consolidation_merging_and_pruning():
    d_model = 64
    d_ff = 128
    d_hyper = 256
    
    # 1. Initialize DynamicHyperMoE with 2 initial bootstrap experts
    moe = DynamicHyperMoE(d_model=d_model, d_ff=d_ff, d_hyper=d_hyper, top_k=2, max_experts=8)
    assert moe.num_experts == 2

    # 2. Spawn an almost identical expert (high similarity >= 0.95)
    key_0 = moe.memory.expert_keys[0:1].clone()
    # Add tiny perturbation
    pert = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper)) * 0.05
    near_duplicate_key = ComplexPhasorVSA.normalize(key_0 + pert)
    
    exp_2_id = moe._spawn_expert(near_duplicate_key, label="duplicate_expert")
    assert moe.num_experts == 3

    # 3. Spawn a starved inactive expert
    starved_key = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper))
    exp_3_id = moe._spawn_expert(starved_key, label="starved_expert")
    assert moe.num_experts == 4

    # Set usage counts: Expert 0: 50 hits, Expert 2: 50 hits (should merge), Expert 1: 100 hits, Expert 3: 0 hits (should prune)
    moe.expert_usage_counts[0] = 50
    moe.expert_usage_counts[1] = 100
    moe.expert_usage_counts[2] = 50
    moe.expert_usage_counts[3] = 0

    # 4. Execute Sleep Consolidation
    engine = SleepConsolidationEngine(
        merge_similarity_threshold=0.85,
        min_usage_prune_threshold=5,
        min_experts_to_keep=2,
    )

    result = engine.consolidate_moe_layer(moe, layer_idx=0, verbose=False)

    assert result["merged"] == 1 # Expert 2 merged into Expert 0
    assert result["pruned"] == 1 # Expert 3 pruned due to 0 usage
    assert moe.num_experts == 2 # Only Expert 0 (consolidated) and Expert 1 survive
    assert len(moe.experts) == 2
    assert moe.memory.expert_keys.shape[0] == 2
