"""Uniform reservoir of (x, y) pairs (Vitter's Algorithm R), shared by the baselines and the pools."""

import numpy as np
import torch
from torch import Tensor


class Reservoir:
    def __init__(self, capacity: int, d: int, rng: np.random.Generator) -> None:
        self.capacity, self.rng, self.seen = capacity, rng, 0
        self.x, self.y = torch.zeros(capacity, d), torch.zeros(capacity, dtype=torch.long)
        self.n = 0

    def add(self, x: Tensor, y: Tensor) -> None:
        for i in range(x.shape[0]):
            if self.n < self.capacity:
                slot = self.n
                self.n += 1
            else:
                slot = int(self.rng.integers(0, self.seen + 1))
                if slot >= self.capacity:
                    self.seen += 1
                    continue
            self.x[slot], self.y[slot] = x[i], y[i]
            self.seen += 1

    def sample(self, n: int):
        idx = torch.from_numpy(self.rng.integers(0, self.n, size=n))
        return self.x[idx], self.y[idx]

    @property
    def data(self):
        """Everything currently held, as (x, y) views."""
        return self.x[: self.n], self.y[: self.n]
