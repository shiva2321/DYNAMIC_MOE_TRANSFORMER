"""The shared representation space: every input becomes a unit vector on one sphere R^d.

Text goes through a frozen sentence encoder (all-MiniLM-L6-v2, local cache, offline) and a fixed random projection to d numbers (the same map as the hybrid frontend of the earlier growmoe
studies, so results compare), then unit length. Digit codes (arithmetic problems as one-hot slots) are zero-padded to d and normalised. Different kinds of input share the sphere and occupy
different regions of it; the pool never sees which kind a vector was. Encoder embeddings are cached on disk by a hash of the text list, so a power cut or a rerun costs nothing.
"""

import hashlib
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch

ENCODER_DIM = 384
PROJECTION_TAG = 0xE1B


class SphereFrontend:
    def __init__(self, d: int = 128, seed: int = 0, cache_dir=None, encoder: Optional[Callable[[Sequence[str]], np.ndarray]] = None, device: Optional[str] = None) -> None:
        self.d, self.seed = d, seed
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._encoder, self._device = encoder, device
        rng = np.random.default_rng([seed, PROJECTION_TAG])
        self.P = torch.from_numpy((rng.standard_normal((ENCODER_DIM, d)) / np.sqrt(d)).astype(np.float32))

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self._device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self._encoder = lambda ts: st.encode(list(ts), batch_size=256, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(self._encoder(texts), dtype=np.float32)

    def _embeddings(self, texts: Sequence[str]) -> np.ndarray:
        if self.cache_dir is None:
            return self._encode(texts)
        key = hashlib.sha1("\x1f".join(texts).encode("utf-8")).hexdigest()
        path = self.cache_dir / f"emb_{key}.npy"
        if path.exists():
            return np.load(path)
        emb = self._encode(texts)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, emb)
        tmp.replace(path)                                                                       # atomic: a power cut leaves no half-written cache entry
        return emb

    def text(self, texts: Sequence[str]) -> torch.Tensor:
        texts = [str(t) for t in texts]
        if not texts:
            return torch.zeros(0, self.d)
        z = torch.from_numpy(self._embeddings(texts)) @ self.P
        return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def digits(self, codes: np.ndarray) -> torch.Tensor:
        codes = np.asarray(codes, dtype=np.float32)
        if codes.ndim != 2 or codes.shape[1] > self.d:
            raise ValueError(f"digit codes must be (n, <= {self.d}), got {codes.shape}")
        x = torch.zeros(codes.shape[0], self.d)
        x[:, :codes.shape[1]] = torch.from_numpy(codes)
        n = x.norm(dim=1, keepdim=True)
        return torch.where(n > 0, x / n.clamp_min(1e-12), x)
