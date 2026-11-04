import random
from typing import Any, List


class ReplayBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: List[Any] = []
        self._seen_count = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: Any) -> None:
        self._seen_count += 1
        if len(self._items) < self.capacity:
            self._items.append(item)
            return
        replace_idx = random.randrange(self._seen_count)
        if replace_idx < self.capacity:
            self._items[replace_idx] = item

    def sample(self, k: int) -> List[Any]:
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0 or not self._items:
            return []
        if k >= len(self._items):
            return list(self._items)
        return random.sample(self._items, k)
