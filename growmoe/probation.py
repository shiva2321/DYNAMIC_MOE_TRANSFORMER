"""Probation buffer and sub-critical graph clustering for novel samples.

Core properties:
1. Bounded buffer of novel unit vectors with FIFO capacity eviction and arrival TTL expiry.
2. Sub-critical random-null cosine threshold:
       tau(d, n, edge_null) = norm.ppf(1 - edge_null / n) / sqrt(d)
   guaranteeing that random noise vectors link with expected degree <= edge_null.
3. DBSCAN-style graph clustering:
   - Adjacency A = (cos >= tau) without self-loops.
   - Core points: degree >= min_neighbors.
   - Connected components over core points; non-core points join if adjacent to a core.
   - Retains components with >= m_min points, returned sorted by size descending.
"""

from collections import deque
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.stats
import torch


class Probation:
    def __init__(self, cap: int, d: int):
        self.cap = int(cap)
        self.d = int(d)
        self.x = torch.empty((0, self.d), dtype=torch.float32)
        self.y = torch.empty((0,), dtype=torch.long)
        self.arrival = torch.empty((0,), dtype=torch.long)

    def __len__(self) -> int:
        return self.x.shape[0]

    def add(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Optional[Union[torch.Tensor, np.ndarray, int]] = None,
        arrival: Union[int, torch.Tensor, np.ndarray] = 0,
    ) -> None:
        """Add samples to the buffer. Evicts the oldest entries when capacity is exceeded."""
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            x_t = x.float()
        else:
            x_t = torch.tensor(x, dtype=torch.float32)

        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)

        b = x_t.shape[0]
        if b == 0:
            return

        # Ensure unit norm
        x_norm = x_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_t = x_t / x_norm

        # Labels (unlabeled stored as -1)
        if y is None:
            y_t = torch.full((b,), -1, dtype=torch.long)
        elif isinstance(y, (int, np.integer)):
            y_t = torch.full((b,), int(y), dtype=torch.long)
        elif isinstance(y, np.ndarray):
            y_t = torch.from_numpy(y).long()
        elif isinstance(y, torch.Tensor):
            y_t = y.long()
        else:
            y_t = torch.tensor(y, dtype=torch.long)

        if y_t.dim() == 0:
            y_t = y_t.unsqueeze(0)

        # Arrival indices
        if isinstance(arrival, (int, np.integer)):
            arr_t = torch.full((b,), int(arrival), dtype=torch.long)
        elif isinstance(arrival, np.ndarray):
            arr_t = torch.from_numpy(arrival).long()
        elif isinstance(arrival, torch.Tensor):
            arr_t = arrival.long()
        else:
            arr_t = torch.tensor(arrival, dtype=torch.long)

        if arr_t.dim() == 0:
            arr_t = arr_t.unsqueeze(0)

        self.x = torch.cat([self.x, x_t], dim=0)
        self.y = torch.cat([self.y, y_t], dim=0)
        self.arrival = torch.cat([self.arrival, arr_t], dim=0)

        # Evict oldest if beyond capacity
        if len(self) > self.cap:
            excess = len(self) - self.cap
            self.x = self.x[excess:]
            self.y = self.y[excess:]
            self.arrival = self.arrival[excess:]

    def expire(self, now: int, ttl: int) -> None:
        """Drop entries with now - arrival > ttl."""
        if len(self) == 0:
            return
        keep_mask = (int(now) - self.arrival) <= int(ttl)
        self.x = self.x[keep_mask]
        self.y = self.y[keep_mask]
        self.arrival = self.arrival[keep_mask]

    @staticmethod
    def tau(d: int, n: int, edge_null: float = 0.5) -> float:
        """Cosine threshold at which two random unit vectors link with probability edge_null / n.

        cos of random pairs on S^{d-1} ~ N(0, 1/d).
        tau = norm.ppf(1 - edge_null / n) / sqrt(d).
        """
        if n <= 1:
            return 1.0
        p_tail = float(edge_null) / float(n)
        p_tail = min(max(p_tail, 1e-12), 1.0 - 1e-12)
        z = scipy.stats.norm.ppf(1.0 - p_tail)
        return float(z / math.sqrt(d))

    def clusters(
        self,
        min_neighbors: int = 4,
        m_min: int = 32,
        edge_null: float = 0.5,
    ) -> List[np.ndarray]:
        """Find candidate regime clusters via DBSCAN on the cosine graph.

        Returns index arrays of cluster members, sorted by size descending.
        """
        n = len(self)
        if n < m_min:
            return []

        # Sub-critical threshold calibrated against capacity so sparse buffers do not overconnect
        tau_val = self.tau(self.d, max(n, self.cap), edge_null=edge_null)

        # Cosine matrix
        cos_matrix = self.x @ self.x.t()
        adj = (cos_matrix >= tau_val)
        adj.fill_diagonal_(False)

        degrees = adj.sum(dim=1)
        core_mask = (degrees >= min_neighbors)
        core_indices = torch.where(core_mask)[0].tolist()

        if len(core_indices) == 0:
            return []

        visited_cores = set()
        claimed_points = set()
        components = []

        # Connected components over core points
        adj_np = adj.cpu().numpy()
        for root in core_indices:
            if root in visited_cores:
                continue

            queue = deque([root])
            visited_cores.add(root)
            comp_cores = [root]

            while queue:
                curr = queue.popleft()
                # Neighbors of curr that are core points
                neighbors = np.where(adj_np[curr])[0]
                for nbr in neighbors:
                    if core_mask[nbr] and nbr not in visited_cores:
                        visited_cores.add(nbr)
                        comp_cores.append(nbr)
                        queue.append(nbr)

            # Find non-core border points adjacent to any core in this component
            comp_cores_set = set(comp_cores)
            comp_all = set(comp_cores)
            claimed_points.update(comp_cores)

            # Check all points adjacent to core set that are not yet claimed
            adj_to_comp = adj_np[comp_cores].any(axis=0)
            candidate_border = np.where(adj_to_comp)[0]
            for pt in candidate_border:
                if pt not in claimed_points:
                    comp_all.add(pt)
                    claimed_points.add(pt)

            if len(comp_all) >= m_min:
                components.append(np.array(sorted(comp_all), dtype=np.int64))

        components.sort(key=len, reverse=True)
        return components

    def take(self, idx: Union[List[int], np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove and return the entries at idx."""
        if len(self) == 0:
            return torch.empty((0, self.d)), torch.empty((0,), dtype=torch.long)

        idx_t = torch.as_tensor(idx, dtype=torch.long)
        if idx_t.numel() == 0:
            return torch.empty((0, self.d)), torch.empty((0,), dtype=torch.long)

        x_taken = self.x[idx_t].clone()
        y_taken = self.y[idx_t].clone()

        mask = torch.ones(len(self), dtype=torch.bool)
        mask[idx_t] = False

        self.x = self.x[mask]
        self.y = self.y[mask]
        self.arrival = self.arrival[mask]

        return x_taken, y_taken
