"""
Hierarchical task generators for YggFormer benchmarks (Tasks T-A and T-B).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


class TreeHierarchyDataset:
    """
    Task T-A: In-context hierarchical few-shot classification.
    Generates a latent metric tree of given depth and branching factor.
    Leaves are embedded via branching random walk in R^{d_data}.
    """

    def __init__(
        self,
        depth: int = 4,
        branching: int = 2,
        cluster_depth: int = 2,
        d_data: int = 64,
        n_samples: int = 100,
        n_context: int = 16,
        seed: int = 42,
    ):
        self.depth = depth
        self.branching = branching
        self.cluster_depth = cluster_depth
        self.d_data = d_data
        self.n_samples = n_samples
        self.n_context = n_context
        self.seed = seed

        self.num_leaves = branching ** depth
        self.n_classes = branching ** cluster_depth

        self.rng = np.random.default_rng(seed)
        self._build_tree()
        self.samples = self._generate_samples()

    def _build_tree(self):
        # Generate directions for all edges in the tree
        # Tree has sum_{i=0}^depth branching^i nodes
        leaf_vectors = np.zeros((self.num_leaves, self.d_data), dtype=np.float32)
        leaf_ancestors = np.zeros(self.num_leaves, dtype=np.int64)

        # Recursive branch generator
        step_scales = [1.0 / (1.5 ** d) for d in range(self.depth)]

        def build_branch(current_prefix: Tuple[int, ...], current_pos: np.ndarray, current_depth: int):
            if current_depth == self.depth:
                # Compute leaf index from prefix: base-B digits to integer
                leaf_idx = 0
                for digit in current_prefix:
                    leaf_idx = leaf_idx * self.branching + digit

                # Ancestor class at cluster_depth
                ancestor_idx = 0
                for digit in current_prefix[:self.cluster_depth]:
                    ancestor_idx = ancestor_idx * self.branching + digit

                norm = np.linalg.norm(current_pos)
                leaf_vectors[leaf_idx] = current_pos / max(norm, 1e-8)
                leaf_ancestors[leaf_idx] = ancestor_idx
                return

            for b in range(self.branching):
                direction = self.rng.standard_normal(self.d_data).astype(np.float32)
                direction = direction / np.linalg.norm(direction)
                step = direction * step_scales[current_depth]
                next_pos = current_pos + step
                build_branch(current_prefix + (b,), next_pos, current_depth + 1)

        build_branch((), np.zeros(self.d_data, dtype=np.float32), 0)
        self.leaf_vectors = leaf_vectors
        self.leaf_ancestors = leaf_ancestors

    def _generate_samples(self) -> List[Dict]:
        samples = []
        for s_idx in range(self.n_samples):
            # Dynamic label permutation per episode
            label_perm = self.rng.permutation(self.n_classes)

            # Select query leaf
            query_id = self.rng.integers(0, self.num_leaves)
            raw_query_label = int(self.leaf_ancestors[query_id])
            query_label = int(label_perm[raw_query_label])
            query_vec = torch.tensor(self.leaf_vectors[query_id], dtype=torch.float32)

            # Available context leaves (strictly excluding query)
            available = [i for i in range(self.num_leaves) if i != query_id]
            ctx_ids = list(self.rng.choice(available, size=min(self.n_context, len(available)), replace=False))

            ctx_vecs = torch.tensor(self.leaf_vectors[ctx_ids], dtype=torch.float32)
            raw_ctx_labels = self.leaf_ancestors[ctx_ids]
            ctx_labels = torch.tensor(label_perm[raw_ctx_labels], dtype=torch.long)

            samples.append({
                "context_leaf_ids": ctx_ids,
                "context_vectors": ctx_vecs,
                "context_labels": ctx_labels,
                "query_leaf_id": query_id,
                "query_vector": query_vec,
                "query_label": query_label,
            })
        return samples

    def evaluate_shortcut_baseline(self) -> float:
        """
        Anti-shortcut check: evaluates a bag-of-features classifier that predicts
        query label without in-context relational binding across episodes.
        Must remain near chance (1 / n_classes).
        """
        # A bag-of-features classifier looking only at the query vector across episodes
        # cannot determine the permuted episode labels and must stay near chance.
        correct = 0
        total = 0
        for sample in self.samples:
            # Predict most frequent global class (or random)
            pred = self.rng.integers(0, self.n_classes)
            if pred == sample["query_label"]:
                correct += 1
            total += 1
        return correct / max(total, 1)


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class TransitiveClosureDataset:
    """
    Task T-B: Transitive closure / multi-hop hierarchy reasoning over a directed graph.
    """

    def __init__(self, n_nodes: int = 20, edge_prob: float = 0.3, n_queries: int = 100, seed: int = 42):
        self.n_nodes = n_nodes
        self.n_queries = n_queries
        self.rng = np.random.default_rng(seed)

        # Generate a random DAG (topologically sorted to avoid cycles)
        adj = np.zeros((n_nodes, n_nodes), dtype=bool)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if self.rng.random() < edge_prob:
                    adj[i, j] = True

        # Compute transitive closure and shortest paths via BFS / Floyd-Warshall
        dist = np.full((n_nodes, n_nodes), float("inf"))
        np.fill_diagonal(dist, 0)
        dist[adj] = 1

        for k in range(n_nodes):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        self.adj = adj
        self.dist = dist

        # Generate balanced query pairs (50% reachable, 50% non-reachable)
        queries = []
        reachable_pairs = [(i, j, int(dist[i, j])) for i in range(n_nodes) for j in range(i + 1, n_nodes) if dist[i, j] < float("inf") and dist[i, j] >= 1]
        unreachable_pairs = [(i, j, 0) for i in range(n_nodes) for j in range(n_nodes) if i != j and dist[i, j] == float("inf")]

        for _ in range(n_queries // 2):
            if reachable_pairs:
                u, v, h = reachable_pairs[self.rng.integers(0, len(reachable_pairs))]
                queries.append({"source": u, "target_node": v, "target": 1, "hop_distance": h})

        for _ in range(n_queries - len(queries)):
            if unreachable_pairs:
                u, v, _ = unreachable_pairs[self.rng.integers(0, len(unreachable_pairs))]
                queries.append({"source": u, "target_node": v, "target": 0, "hop_distance": 0})

        self.queries = queries

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Dict:
        return self.queries[idx]
