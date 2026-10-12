"""Training-free, deterministic, type-agnostic frontend: anything -> unit vector in R^d.

Numeric arrays of any shape use signed feature hashing (a sparse random projection, so
Johnson-Lindenstrauss preserves geometry). Text/bytes use hashed 1-3-gram counts. No Python
``hash()`` is used anywhere, so results are identical across processes.

By default the numeric map does NOT depend on the array shape (``salt_by_shape=False``): a 13-vector
and the same values zero-padded to 64 encode identically, so the data type is not leaked through
the projection. Different types differ only through their statistics.
"""

import zlib
from typing import Sequence, Union

import numpy as np
import torch

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix64(h: np.ndarray) -> np.ndarray:
    """splitmix64-style finaliser (vectorised, wraps modulo 2**64)."""
    h = h ^ (h >> np.uint64(30))
    h = h * np.uint64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> np.uint64(27))
    h = h * np.uint64(0x94D049BB133111EB)
    h = h ^ (h >> np.uint64(31))
    return h


class HashFrontend:
    def __init__(self, d: int = 128, seed: int = 0, max_len: int = 4096,
                 ngram: Sequence[int] = (1, 2, 3), salt_by_shape: bool = False) -> None:
        self.d = d
        self.seed = seed
        self.max_len = max_len
        self.ngram = tuple(ngram)
        self.salt_by_shape = salt_by_shape
        self._tables: dict = {}
        rng = np.random.default_rng([seed, 0xB17E5])
        # one odd 64-bit multiplier per n-gram position
        self._mult = rng.integers(1, 2 ** 63 - 1, size=max(self.ngram) + 1, dtype=np.int64).astype(np.uint64) | np.uint64(1)

    # ------------------------------------------------------------------ numeric
    def _table(self, shape_key=None):
        key = shape_key if self.salt_by_shape else None
        if key not in self._tables:
            salt = 0 if key is None else zlib.crc32(repr(key).encode("utf-8"))
            rng = np.random.default_rng([self.seed, 0x4E554D, salt])
            bucket = rng.integers(0, self.d, size=self.max_len)
            sign = rng.choice(np.array([-1.0, 1.0]), size=self.max_len)
            self._tables[key] = (torch.from_numpy(bucket), torch.from_numpy(sign.astype(np.float32)))
        return self._tables[key]

    def _unit(self, z: torch.Tensor) -> torch.Tensor:
        n = z.norm(dim=-1, keepdim=True)
        fixed = torch.zeros_like(z)
        fixed[..., 0] = 1.0
        return torch.where(n > 1e-12, z / n.clamp_min(1e-12), fixed)

    def encode_numeric(self, items) -> torch.Tensor:
        if isinstance(items, torch.Tensor):
            items = items.detach().cpu().numpy()
        if isinstance(items, np.ndarray):
            arr = np.asarray(items, dtype=np.float64)
            batch = [arr[i] for i in range(arr.shape[0])] if arr.ndim and arr.dtype != object else list(arr)
        else:
            batch = [np.asarray(a, dtype=np.float64) for a in items]
        z = torch.zeros(len(batch), self.d)
        for i, a in enumerate(batch):
            flat = torch.from_numpy(np.ascontiguousarray(a.reshape(-1), dtype=np.float32))
            L = flat.numel()
            if L == 0:
                continue
            bucket, sign = self._table(tuple(a.shape))
            pos = torch.arange(L) % self.max_len
            z[i].index_add_(0, bucket[pos], flat * sign[pos])
        return self._unit(z)

    # --------------------------------------------------------------------- text
    def _ngram_hashes(self, b: np.ndarray, n: int) -> np.ndarray:
        m = len(b) - n + 1
        if m <= 0:
            return np.empty(0, dtype=np.uint64)
        acc = np.zeros(m, dtype=np.uint64)
        for j in range(n):
            acc ^= (b[j:j + m].astype(np.uint64) + np.uint64(1)) * self._mult[j]
        return _mix64(acc + np.uint64((n * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF))

    def encode_bytes(self, items) -> torch.Tensor:
        z = np.zeros((len(items), self.d), dtype=np.float64)
        for i, s in enumerate(items):
            raw = s.encode("utf-8") if isinstance(s, str) else bytes(s)
            b = np.frombuffer(raw, dtype=np.uint8)
            for n in self.ngram:
                h = self._ngram_hashes(b, n)
                if h.size == 0:
                    continue
                idx = (h % np.uint64(self.d)).astype(np.int64)
                sgn = np.where((h >> np.uint64(63)) == 1, -1.0, 1.0)
                np.add.at(z[i], idx, sgn)
            z[i] = np.sign(z[i]) * np.log1p(np.abs(z[i]))
        return self._unit(torch.from_numpy(z.astype(np.float32)))

    # ----------------------------------------------------------------- dispatch
    def __call__(self, items: Union[np.ndarray, torch.Tensor, Sequence]) -> torch.Tensor:
        if isinstance(items, (np.ndarray, torch.Tensor)):
            return self.encode_numeric(items)
        if len(items) and isinstance(items[0], (str, bytes, bytearray)):
            return self.encode_bytes(items)
        return self.encode_numeric(items)
