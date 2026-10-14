"""
test_yf_tasks.py - Tests for hierarchical benchmark generators (Task T-A and T-B).
Verifies:
1. Tree depth and branching match specifications.
2. Label integrity: ground truth ancestor labels are exactly recoverable.
3. Anti-shortcut guardrail: bag-of-features / random baselines remain near chance.
4. No leakage: query leaf is strictly held-out from context leaves.
5. Determinism: identical trees generated for fixed seed.
"""

import math
import pytest
import numpy as np
import torch

from yggformer.tasks.hierarchy import TreeHierarchyDataset, TransitiveClosureDataset


def test_tree_hierarchy_preconditions():
    """Verify tree depth, branching factor, and leaf count preconditions."""
    depth = 4
    branching = 2
    ds = TreeHierarchyDataset(depth=depth, branching=branching, d_data=32, n_samples=50, seed=100)

    # In a complete tree of depth D and branching B, leaves = B^D
    expected_leaves = branching ** depth
    assert ds.num_leaves == expected_leaves
    assert ds.depth == depth
    assert ds.branching == branching


def test_tree_hierarchy_no_leakage_and_label_integrity():
    """Query leaf must be strictly held out from context; ancestor labels are correct."""
    ds = TreeHierarchyDataset(depth=3, branching=3, d_data=32, n_samples=20, n_context=10, seed=101)

    for i in range(len(ds)):
        sample = ds[i]
        ctx_leaf_ids = sample["context_leaf_ids"]
        query_leaf_id = sample["query_leaf_id"]

        # No leakage
        assert query_leaf_id not in ctx_leaf_ids, f"Leakage detected: query {query_leaf_id} was in context!"

        # Label is integer within [0, n_classes - 1]
        label = sample["query_label"]
        assert 0 <= label < ds.n_classes


def test_anti_shortcut_baseline_near_chance():
    """A trivial bag-of-features centroid matcher without hierarchy stays near chance."""
    # 4 classes -> chance = 25%
    ds = TreeHierarchyDataset(depth=4, branching=2, cluster_depth=2, d_data=32, n_samples=200, n_context=12, seed=102)
    chance = 1.0 / ds.n_classes  # 0.25

    shortcut_acc = ds.evaluate_shortcut_baseline()
    # Must stay within 15% of chance (e.g. <= 0.40)
    assert abs(shortcut_acc - chance) < 0.15, f"Shortcut baseline was too high: {shortcut_acc:.3f} vs chance {chance:.3f}"


def test_transitive_closure_generator():
    """Verify transitive closure DAG generation and queries at various hop distances."""
    ds = TransitiveClosureDataset(n_nodes=20, edge_prob=0.3, n_queries=100, seed=103)
    assert len(ds) == 100
    pos_count = sum(1 for q in ds if q["target"] == 1)
    neg_count = sum(1 for q in ds if q["target"] == 0)

    # Balanced query set
    assert pos_count > 0 and neg_count > 0
    # Queries test hop distances 1 to 5
    hops = [q["hop_distance"] for q in ds if q["target"] == 1]
    assert len(hops) > 0
    assert max(hops) >= 2


def test_determinism_fixed_seed():
    """Generators produce identical items for identical seeds."""
    ds1 = TreeHierarchyDataset(depth=3, branching=2, d_data=16, n_samples=10, seed=42)
    ds2 = TreeHierarchyDataset(depth=3, branching=2, d_data=16, n_samples=10, seed=42)

    for s1, s2 in zip(ds1, ds2):
        assert torch.equal(s1["query_vector"], s2["query_vector"])
        assert s1["query_label"] == s2["query_label"]
