"""Representations: ways to turn a sentence into a unit vector so that sentences can be compared (spec section 6).

R0 hashed byte n-grams (the control: it sees only letters, no meaning), R1 MiniLM (384, and the pool's fixed random projection to 128), R2 mpnet, R3 bert (mean pool),
R4 the last-token hidden state of a causal language model at a middle layer, read after a fixed suffix that asks for the operation. Every model is loaded from the local
Hugging Face cache only; a missing model is a loud FileNotFoundError that names it. ``cached_encode`` keeps every result on disk in blocks of 1,000 texts, so a power
cut costs one block at most.
"""

import hashlib
import io
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import runlib

BLOCK = 1000
BATCH = 16
MAX_TEXT_TOKENS = 256
SUFFIX = "\nThe operation requested is"
_MINILM, _MPNET, _BERT = "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "bert-base-uncased"
_CAUSAL = {"gpt2": ("gpt2", (4, 6, 8)), "qwen": ("Qwen/Qwen2.5-1.5B-Instruct", (9, 14, 18))}      # layers {L//3, L//2, 2L//3} of 12 and 28
REP_IDS = ("R0_hash", "R1_minilm384", "R1_minilm128", "R2_mpnet", "R3_bert",
           *(f"R4_{k}_L{layer}" for k, (_, layers) in _CAUSAL.items() for layer in layers))
_R4_ID = re.compile(r"^R4_(gpt2|qwen)_L(\d+)$")
_MODELS: Dict[Tuple, object] = {}
_REPS: Dict[Tuple, "Representation"] = {}


def _gpu() -> bool:
    import torch
    return torch.cuda.is_available() and os.environ.get("KS_DEVICE") != "cpu"


def _from_cache(name: str, load):
    try:
        return load(name)
    except OSError as err:
        raise FileNotFoundError(f"model {name!r} is not in the local Hugging Face cache (offline mode; this project never downloads): {err}") from err


def clear_models() -> None:
    """Drop every loaded model and free GPU memory (one GPU job at a time)."""
    _MODELS.clear()
    _REPS.clear()
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sentence_model(name: str, gpu: bool):
    key = ("st", name, gpu)
    if key not in _MODELS:
        from sentence_transformers import SentenceTransformer
        _MODELS[key] = _from_cache(name, lambda n: SentenceTransformer(n, device="cuda" if gpu else "cpu", local_files_only=True))
    return _MODELS[key]


def _hf_model(name: str, auto: str, gpu: bool, fp16: bool):
    key = ("hf", name, gpu, fp16)
    if key not in _MODELS:
        import torch
        import transformers
        tok = _from_cache(name, lambda n: transformers.AutoTokenizer.from_pretrained(n, local_files_only=True))
        kwargs = {"torch_dtype": torch.float16} if fp16 else {}
        model = _from_cache(name, lambda n: getattr(transformers, auto).from_pretrained(n, local_files_only=True, **kwargs)).eval()
        _MODELS[key] = (tok, model.to("cuda") if gpu else model)
    return _MODELS[key]


class Representation:
    """``encode(texts)`` returns float32 (n, dim), unit length, finite, deterministic. Subclasses implement ``_encode``; ``tag`` names the model behind the id."""

    def __init__(self, rep_id: str, dim: int, tag: Optional[str] = None) -> None:
        self.id, self.dim, self.tag = rep_id, dim, tag or rep_id

    def _encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("texts must be strings")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        z = np.asarray(self._encode(texts), dtype=np.float32)
        if z.shape != (len(texts), self.dim):
            raise RuntimeError(f"{self.id} returned shape {z.shape}, expected {(len(texts), self.dim)}")
        if not np.isfinite(z).all():
            raise FloatingPointError(f"{self.id} produced non-finite values")
        norm = np.linalg.norm(z, axis=1, keepdims=True)
        fixed = np.zeros_like(z)
        fixed[:, 0] = 1.0
        return np.where(norm > 1e-12, z / np.maximum(norm, 1e-12), fixed).astype(np.float32)

    def close(self) -> None:
        """Release what this representation holds (models are shared; clear_models() frees them all)."""


class HashRep(Representation):
    """R0: signed hashed byte 1-3-grams (growmoe.frontend.HashFrontend), 128 numbers; the control that sees spelling only."""

    def __init__(self) -> None:
        from growmoe.frontend import HashFrontend
        super().__init__("R0_hash", 128)
        self._f = HashFrontend(d=128, seed=0)

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self._f(texts).numpy()


class SentenceRep(Representation):
    """R1 and R2: a sentence-embedding model, optionally followed by the pool's fixed random projection (seed 0, as research/growmoe_results/m0/run_m0.py::HybridFrontend)."""

    def __init__(self, rep_id: str, model_name: str, dim: int, project_to: Optional[int] = None, gpu: bool = False) -> None:
        super().__init__(rep_id, project_to or dim, tag=f"{rep_id}@{model_name}")
        self.model_name, self.gpu = model_name, gpu
        self._P = None
        if project_to:
            rng = np.random.default_rng([0, 0xE1B])
            self._P = (rng.standard_normal((dim, project_to)) / np.sqrt(project_to)).astype(np.float32)

    def _encode(self, texts: List[str]) -> np.ndarray:
        emb = _sentence_model(self.model_name, self.gpu).encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return emb @ self._P if self._P is not None else emb


class BertRep(Representation):
    """R3: bert-base-uncased, mean of the last-layer token vectors over the real (not padding) tokens."""

    def __init__(self, gpu: bool = False) -> None:
        super().__init__("R3_bert", 768, tag=f"R3_bert@{_BERT}")
        self.gpu = gpu

    def _encode(self, texts: List[str]) -> np.ndarray:
        import torch
        tok, model = _hf_model(_BERT, "AutoModel", self.gpu, False)
        out = []
        for s in range(0, len(texts), 32):
            enc = tok(texts[s:s + 32], return_tensors="pt", padding=True, truncation=True, max_length=MAX_TEXT_TOKENS)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
        return np.concatenate(out)


class LastTokenRep(Representation):
    """R4: feed ``text + SUFFIX`` to a causal language model and take the hidden state of the last token at ``layer`` (0 is the embeddings, the model depth is the top)."""

    def __init__(self, rep_id: str, model_name: str, layer: int, fp16_gpu: bool) -> None:
        from transformers import AutoConfig
        cfg = _from_cache(model_name, lambda n: AutoConfig.from_pretrained(n, local_files_only=True))
        if not 1 <= layer <= cfg.num_hidden_layers:
            raise ValueError(f"{rep_id}: layer {layer} is outside 1..{cfg.num_hidden_layers} for {model_name}")
        super().__init__(rep_id, int(cfg.hidden_size), tag=f"{rep_id}@{model_name}")
        self.model_name, self.layer, self.fp16_gpu = model_name, layer, fp16_gpu
        if fp16_gpu and not _gpu():
            raise RuntimeError(f"{rep_id} runs in fp16 on the GPU and no GPU is available (or KS_DEVICE=cpu is set); the model is too big to run on the CPU under this project's memory rule")

    def _model(self):
        return _hf_model(self.model_name, "AutoModelForCausalLM", self.fp16_gpu, self.fp16_gpu)

    @property
    def suffix_ids(self) -> List[int]:
        return self._model()[0](SUFFIX, add_special_tokens=False)["input_ids"]

    def _input_ids(self, text: str) -> List[int]:
        ids = self._model()[0](text, add_special_tokens=False, truncation=True, max_length=MAX_TEXT_TOKENS)["input_ids"]
        return ids + self.suffix_ids

    def _encode(self, texts: List[str]) -> np.ndarray:
        import torch
        tok, model = self._model()
        pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        out = []
        for s in range(0, len(texts), BATCH):
            ids = [self._input_ids(t) for t in texts[s:s + BATCH]]
            n = max(len(x) for x in ids)
            batch = torch.tensor([x + [pad] * (n - len(x)) for x in ids], device=model.device)       # padding on the right: a causal model never lets earlier tokens see it
            mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in ids], device=model.device)
            with torch.no_grad():
                h = model(input_ids=batch, attention_mask=mask, output_hidden_states=True).hidden_states[self.layer]
            last = torch.tensor([len(x) - 1 for x in ids], device=h.device)
            out.append(h[torch.arange(len(ids), device=h.device), last].float().cpu().numpy())
        return np.concatenate(out)


def get_rep(rep_id: str, model_name: Optional[str] = None) -> Representation:
    """The representation with this id. ``model_name`` swaps the language model behind an R4 id (used for small stand-ins in tests); R4 ids not in REP_IDS but of the
    form R4_gpt2_L<n> / R4_qwen_L<n> are accepted for other layers."""
    key = (rep_id, model_name)
    if key in _REPS:
        return _REPS[key]
    m = _R4_ID.match(rep_id)
    if model_name is not None and not m:
        raise ValueError("model_name can only replace the language model of an R4 representation")
    if rep_id == "R0_hash":
        rep: Representation = HashRep()
    elif rep_id == "R1_minilm384":
        rep = SentenceRep(rep_id, _MINILM, 384)
    elif rep_id == "R1_minilm128":
        rep = SentenceRep(rep_id, _MINILM, 384, project_to=128)
    elif rep_id == "R2_mpnet":
        rep = SentenceRep(rep_id, _MPNET, 768, gpu=_gpu())
    elif rep_id == "R3_bert":
        rep = BertRep(gpu=_gpu())
    elif m:
        default_name, _ = _CAUSAL[m.group(1)]
        rep = LastTokenRep(rep_id, model_name or default_name, int(m.group(2)), fp16_gpu=m.group(1) == "qwen")
    else:
        raise KeyError(f"unknown representation {rep_id!r}; known: {', '.join(REP_IDS)}")
    _REPS[key] = rep
    return rep


# ------------------------------------------------------------------------------------------------------------------------------------ cache
def _block_key(tag: str, block: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(tag.encode("utf-8") + b"\0")
    for t in block:
        raw = t.encode("utf-8", "surrogatepass")
        h.update(len(raw).to_bytes(8, "little") + raw)
    return h.hexdigest()


def _read_block(path: Path, n: int, dim: int) -> Optional[np.ndarray]:
    """The cached vectors, or None when the file is missing, cut short, damaged or the wrong shape (it is then recomputed and rewritten)."""
    try:
        with np.load(path) as f:
            z = f["vectors"]
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
        return None
    return z if z.dtype == np.float32 and z.shape == (n, dim) and np.isfinite(z).all() else None


def cached_encode(rep: Representation, texts: Sequence[str], cache_dir) -> np.ndarray:
    """``rep.encode(texts)``, kept on disk: one .npz per 1,000-text block, named by the representation id and a sha256 of the model tag and the block's texts."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts = list(texts)
    out = np.zeros((len(texts), rep.dim), dtype=np.float32)
    for s in range(0, len(texts), BLOCK):
        block = texts[s:s + BLOCK]
        path = cache_dir / f"{rep.id}_{_block_key(rep.tag, block)}.npz"
        z = _read_block(path, len(block), rep.dim)
        if z is None:
            z = rep.encode(block)
            buf = io.BytesIO()
            np.savez(buf, vectors=z)
            runlib.atomic_write_bytes(path, buf.getvalue())
        out[s:s + len(block)] = z
    return out
