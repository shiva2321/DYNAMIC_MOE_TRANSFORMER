"""
tree_search.py - Hierarchical Lorentz Key Tree and Branch-and-Bound Top-B Retrieval (Round Y5 / E5).
Implements:
1. LorentzKeyTree: hierarchical clustering of keys on the hyperboloid using Lorentz centroids and geodesic radii.
2. branch_and_bound_top_b: admissible pruning using the hyperbolic triangle inequality upper bound:
   UB(q, node) = -(cosh(sqrt(c) * max(0, d(q, centroid) - radius)) - 1) / (c * tau)
"""

import math
import heapq
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F

from geometry.lorentz import LorentzModel


def lorentz_logit_fn(d: torch.Tensor, c: float = 1.0, tau: float = 1.0) -> torch.Tensor:
    """Exact hyperbolic attention logit as a function of geodesic distance d."""
    return -(torch.cosh(math.sqrt(c) * d) - 1.0) / (c * tau)


class LorentzKeyNode:
    def __init__(
        self,
        centroid: torch.Tensor,
        radius: float,
        leaf_indices: List[int],
        children: Optional[List["LorentzKeyNode"]] = None,
        is_leaf: bool = False,
    ):
        self.centroid = centroid # (D+1,)
        self.radius = radius # scalar float
        self.leaf_indices = leaf_indices
        self.children = children or []
        self.is_leaf = is_leaf


class LorentzKeyTree:
    def __init__(self, keys: torch.Tensor, root: LorentzKeyNode, c: float = 1.0):
        self.keys = keys # (N, D+1)
        self.root = root
        self.c = c
        self.dim = keys.shape[-1] - 1
        self.lorentz = LorentzModel(dim=self.dim, c=c)

    @classmethod
    def build(cls, keys: torch.Tensor, max_leaf_size: int = 16, c: float = 1.0) -> "LorentzKeyTree":
        dim = keys.shape[-1] - 1
        lorentz = LorentzModel(dim=dim, c=c)
        N = keys.shape[0]

        def _build_node(indices: List[int]) -> LorentzKeyNode:
            subset_keys = keys[indices] # (K, D+1)
            # Centroid on the hyperboloid: normalized sum of vectors
            s = subset_keys.sum(dim=0)
            inner = lorentz.inner(s, s).item()
            if inner >= 0:
                # Fallback to mean of space components
                centroid = lorentz.centroid(subset_keys)
            else:
                centroid = s / (math.sqrt(c) * math.sqrt(-inner))

            # Compute geodesic radius
            dists = [lorentz.distance(centroid, keys[idx]).item() for idx in indices]
            radius = max(dists) if dists else 0.0

            if len(indices) <= max_leaf_size:
                return LorentzKeyNode(
                    centroid=centroid,
                    radius=radius,
                    leaf_indices=indices,
                    is_leaf=True,
                )

            # 2-means split on hyperboloid
            # Pick first seed as farthest from centroid, second seed as farthest from first
            idx_a = indices[int(torch.tensor(dists).argmax().item())]
            seed_a = keys[idx_a]
            dists_a = [lorentz.distance(seed_a, keys[idx]).item() for idx in indices]
            idx_b = indices[int(torch.tensor(dists_a).argmax().item())]
            seed_b = keys[idx_b]

            group_a, group_b = [], []
            for idx in indices:
                da = lorentz.distance(seed_a, keys[idx]).item()
                db = lorentz.distance(seed_b, keys[idx]).item()
                if da <= db:
                    group_a.append(idx)
                else:
                    group_b.append(idx)

            # Prevent empty split
            if not group_a or not group_b:
                half = len(indices) // 2
                group_a = indices[:half]
                group_b = indices[half:]

            child_a = _build_node(group_a)
            child_b = _build_node(group_b)

            return LorentzKeyNode(
                centroid=centroid,
                radius=radius,
                leaf_indices=indices,
                children=[child_a, child_b],
                is_leaf=False,
            )

        root = _build_node(list(range(N)))
        return cls(keys=keys, root=root, c=c)


def branch_and_bound_top_b(
    tree: LorentzKeyTree,
    query: torch.Tensor,
    B: int = 8,
    c: float = 1.0,
    tau: float = 1.0,
) -> Tuple[List[int], int]:
    """
    Search for top-B keys with highest logits using branch-and-bound.
    Returns:
        (top_b_indices, total_visits)
    """
    lorentz = tree.lorentz
    keys = tree.keys

    def get_node_upper_bound(node: LorentzKeyNode) -> float:
        # Geodesic distance to node centroid
        d_center = lorentz.distance(query, node.centroid).item()
        d_min = max(0.0, d_center - node.radius)
        # Upper bound logit
        val = -(math.cosh(math.sqrt(c) * d_min) - 1.0) / (c * tau)
        return val

    # Min-heap of best candidates found so far: (logit, index)
    best_candidates = [] # size at most B
    node_visits = 0
    leaf_evals = 0

    # Max-heap for exploration priority: (-upper_bound, node)
    root_ub = get_node_upper_bound(tree.root)
    queue = [(-root_ub, 0, tree.root)] # counter to break ties
    counter = 1

    while queue:
        neg_ub, _, node = heapq.heappop(queue)
        ub = -neg_ub
        node_visits += 1

        # Pruning check: if the best possible logit in this node cannot beat
        # the worst of our current top-B candidates, skip!
        if len(best_candidates) == B and ub <= best_candidates[0][0]:
            continue

        if node.is_leaf:
            # Evaluate all leaf keys in this node
            for idx in node.leaf_indices:
                leaf_evals += 1
                dist = lorentz.distance(query, keys[idx]).item()
                logit = -(math.cosh(math.sqrt(c) * dist) - 1.0) / (c * tau)

                if len(best_candidates) < B:
                    heapq.heappush(best_candidates, (logit, idx))
                elif logit > best_candidates[0][0]:
                    heapq.heappushpop(best_candidates, (logit, idx))
        else:
            # Explore children, computing upper bound for each
            for child in node.children:
                child_ub = get_node_upper_bound(child)
                # Only push if it can potentially beat the worst in top-B
                if len(best_candidates) < B or child_ub > best_candidates[0][0]:
                    heapq.heappush(queue, (-child_ub, counter, child))
                    counter += 1

    # Extract top-B indices ordered descending by logit
    best_candidates.sort(key=lambda x: x[0], reverse=True)
    top_indices = [idx for _, idx in best_candidates]
    return top_indices, leaf_evals, node_visits
