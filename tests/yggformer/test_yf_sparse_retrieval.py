"""
test_yf_sparse_retrieval.py - Unit and invariant tests for Round Y5 (E5 sparse retrieval & tree routing).
Verifies:
1. Hyperbolic triangle inequality upper bound on Lorentz attention logits.
2. Lorentz Key Tree construction and radius bounds.
3. Branch-and-bound top-B retrieval correctness vs exact dense search.
4. Monotonicity of the logit kernel -(cosh(sqrt(c) d) - 1)/(c tau).
"""

import math
import pytest
import torch
import torch.nn.functional as F

from geometry.lorentz import LorentzModel
from yggformer.tree_search import LorentzKeyTree, branch_and_bound_top_b


def test_lorentz_logit_monotonicity_and_bound():
    """Verify logit kernel is strictly monotonically decreasing with geodesic distance,
    and triangle inequality provides an exact admissible upper bound."""
    c = 1.0
    tau = 1.0
    lorentz = LorentzModel(dim=8, c=c)

    def logit_fn(d):
        return -(torch.cosh(math.sqrt(c) * d) - 1.0) / (c * tau)

    # Monotonicity
    d_vals = torch.linspace(0.0, 5.0, 50)
    logits = logit_fn(d_vals)
    diffs = logits[1:] - logits[:-1]
    assert (diffs < 0).all(), "Logit function must be strictly monotonically decreasing"

    # Triangle inequality bound
    torch.manual_seed(42)
    center_s = torch.randn(8) * 0.3
    center = lorentz.exp_map_zero(center_s)
    # Generate points around center
    keys = torch.stack([lorentz.exp_map_zero(center_s + torch.randn(8) * 0.2) for _ in range(20)])

    # True distances from center
    d_center = torch.stack([lorentz.distance(k, center) for k in keys])
    r_max = d_center.max()

    # Arbitrary query
    query = lorentz.exp_map_zero(torch.randn(8) * 0.8)
    d_q_center = lorentz.distance(query, center)

    # Lower bound on distance from query to any key in subtree
    d_lb = torch.clamp(d_q_center - r_max, min=0.0)
    upper_bound = logit_fn(d_lb)

    # Exact logits
    d_q_k = torch.stack([lorentz.distance(query, k) for k in keys])
    exact_logits = logit_fn(d_q_k)

    # The bound must never be violated
    assert (upper_bound >= exact_logits - 1e-6).all(), "Admissible upper bound violated!"


def test_lorentz_key_tree_radius_invariant():
    """Verify all descendant keys lie within node radius of node centroid."""
    torch.manual_seed(43)
    dim = 8
    lorentz = LorentzModel(dim=dim, c=1.0)
    keys = torch.stack([lorentz.exp_map_zero(torch.randn(dim) * 0.5) for _ in range(64)])

    tree = LorentzKeyTree.build(keys, max_leaf_size=8, c=1.0)
    assert tree.root is not None

    def check_node(node):
        for leaf_idx in node.leaf_indices:
            dist = lorentz.distance(node.centroid, keys[leaf_idx]).item()
            assert dist <= node.radius + 1e-5, f"Key {leaf_idx} dist {dist} > radius {node.radius}"
        for child in node.children:
            check_node(child)

    check_node(tree.root)


def test_branch_and_bound_top_b_exactness():
    """Verify branch-and-bound retrieves exact top-B keys with fewer leaf evaluations than dense."""
    torch.manual_seed(44)
    dim = 8
    c = 1.0
    tau = 1.0
    N = 128
    B = 8
    lorentz = LorentzModel(dim=dim, c=c)
    keys = torch.stack([lorentz.exp_map_zero(torch.randn(dim) * 0.5) for _ in range(N)])
    query = lorentz.exp_map_zero(torch.randn(dim) * 0.5)

    tree = LorentzKeyTree.build(keys, max_leaf_size=8, c=c)

    # Exact dense top-B
    dists = torch.stack([lorentz.distance(query, k) for k in keys])
    logits = -(torch.cosh(math.sqrt(c) * dists) - 1.0) / (c * tau)
    exact_top_b = torch.topk(logits, k=B).indices.tolist()

    # Branch-and-bound top-B
    bb_indices, leaf_evals, node_visits = branch_and_bound_top_b(tree, query, B=B, c=c, tau=tau)

    # Must match exact top-B indices
    assert set(bb_indices) == set(exact_top_b)
    # Must evaluate strictly fewer leaves than total keys
    assert leaf_evals < N
