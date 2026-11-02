"""The shared readers: two frozen encoders turn a message, a snippet or an utterance into vectors, the same way for all three domains.

* ``QwenReader``  Qwen2.5-1.5B-Instruct (local cache only, never downloaded), the hidden state of the LAST token at a late layer, read after a fixed generic suffix (1536 numbers). Strong on code and
  arithmetic.
* ``MiniLMReader``  all-MiniLM-L6-v2 sentence embeddings (384 numbers). Strong on everyday language. The dev pilot (results/fsmoe/pilot_700.txt) measured why both are needed: a closed-form classifier
  on held-out wordings reads 0.99 / 0.79 / 0.24 (code / arithmetic / language) from Qwen alone and 0.78 / 0.83 / 0.81 from MiniLM alone.
Raw states are kept (float16); the frontend standardises and projects them. Every result is cached on disk in blocks of 500 texts, keyed by the block's texts and the reader's own settings, written
atomically, so a power cut costs one block at most and a rerun costs nothing.
"""

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
SUFFIX = "\nThe core subject of the above is"
LAYER = 26
BLOCK = 500
BATCH = 16
MAX_CHARS = 1500
DIM = 1536                                                  # the Qwen block, kept as the default width of a reader
MINI_DIM = 384


class CachedReader:
    """The block cache every reader shares; a subclass implements ``_encode_block`` and sets ``tag`` and ``dim``."""
    dim = DIM

    def __init__(self, cache_dir) -> None:
        self.cache_dir = Path(cache_dir)

    @property
    def tag(self) -> str:
        raise NotImplementedError

    def _encode_block(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def _block_path(self, name: str, k: int, texts: Sequence[str]) -> Path:
        key = hashlib.sha1((self.tag + "\x1e" + "\x1f".join(texts)).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / self.tag / name / f"b{k:05d}_{key}.npy"

    def encode(self, texts: Sequence[str], name: str) -> np.ndarray:
        """float16 (n, dim). ``name`` only names the cache folder; the block key is made of the texts."""
        texts = [str(t) for t in texts]
        blocks = []
        for k, lo in enumerate(range(0, len(texts), BLOCK)):
            chunk = texts[lo:lo + BLOCK]
            path = self._block_path(name, k, chunk)
            if path.exists():
                blocks.append(np.load(path))
                continue
            arr = self._encode_block(chunk)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, arr)
            tmp.replace(path)                                                                   # atomic: a power cut leaves no half-written block
            blocks.append(arr)
        return np.concatenate(blocks) if blocks else np.zeros((0, self.dim), dtype=np.float16)


class QwenReader(CachedReader):
    dim = DIM

    def __init__(self, cache_dir, layer: int = LAYER, suffix: str = SUFFIX, batch: int = BATCH, device: Optional[str] = None) -> None:
        super().__init__(cache_dir)
        self.layer, self.suffix, self.batch, self._device = layer, suffix, batch, device
        self._tok = self._model = None

    @property
    def tag(self) -> str:
        return f"qwen15_L{self.layer}_{hashlib.sha1(self.suffix.encode()).hexdigest()[:6]}"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        import transformers
        dev = self._device or ("cuda" if torch.cuda.is_available() and os.environ.get("KS_DEVICE") != "cpu" else "cpu")
        try:
            tok = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
            model = transformers.AutoModel.from_pretrained(MODEL, local_files_only=True, dtype=torch.float16 if dev == "cuda" else torch.float32)
        except OSError as err:
            raise FileNotFoundError(f"model {MODEL!r} is not in the local Hugging Face cache (offline; this project never downloads): {err}") from err
        tok.padding_side = "left"
        self._tok, self._model, self._dev = tok, model.eval().to(dev), dev

    def close(self) -> None:
        self._tok = self._model = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _forward(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        self._load()
        enc = self._tok([t[:MAX_CHARS] + self.suffix for t in texts], return_tensors="pt", padding=True).to(self._dev)
        with torch.inference_mode():
            hs = self._model(**enc, output_hidden_states=True).hidden_states[self.layer]
        out = hs[:, -1, :].float().cpu().numpy()
        if not np.isfinite(out).all():
            raise FloatingPointError("the reader produced non-finite values")
        return out

    def _encode_block(self, texts: List[str]) -> np.ndarray:
        order = np.argsort([len(t) for t in texts], kind="stable")
        parts = [self._forward([texts[int(i)] for i in order[lo:lo + self.batch]]) for lo in range(0, len(texts), self.batch)]
        out = np.empty((len(texts), self.dim), dtype=np.float16)
        out[order] = np.concatenate(parts).astype(np.float16)
        return out


class MiniLMReader(CachedReader):
    dim = MINI_DIM

    def __init__(self, cache_dir, device: Optional[str] = None) -> None:
        super().__init__(cache_dir)
        self._device, self._st = device, None

    @property
    def tag(self) -> str:
        return "minilm_l6"

    def _encode_block(self, texts: List[str]) -> np.ndarray:
        if self._st is None:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = self._device or ("cuda" if torch.cuda.is_available() and os.environ.get("KS_DEVICE") != "cpu" else "cpu")
            try:
                self._st = SentenceTransformer(MINILM, device=dev, local_files_only=True)
            except OSError as err:
                raise FileNotFoundError(f"model {MINILM!r} is not in the local Hugging Face cache (offline): {err}") from err
        return self._st.encode(texts, batch_size=256, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True).astype(np.float16)

    def close(self) -> None:
        self._st = None
