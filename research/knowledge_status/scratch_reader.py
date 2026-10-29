"""Task 11b: the from-scratch reader arms (the user's decision D-40). No pretrained model: the input is a hashed bag of words and character pieces, the encoder is a small network
trained here, on the same contrastive objective as the learned map (same skill, other wording: pulled together; near-twin and other skills: pushed apart).

* ``R5_FS_strict``: the encoder is trained on the teaching data only (S1 to S3 and the none-messages), nothing else.
* ``R5_FS_pre``: the encoder is first pre-trained WITHOUT labels on everyday sentences from the local corpora (banking77, AG News: two word-dropout views of a sentence are pulled together),
  then trained on the teaching data exactly as the strict arm. Held-out wordings (S4 to S8) and test templates never enter either arm; ``fit`` refuses them as the learned map does.

The ids start with ``R5_`` so that the harness applies a map on top of a base reader; the base reader is passed in (``BagRep``), and the map factory is ``make_mapper``. Encoder side only; a decoder
is a separate later piece. Operating points per arm are chosen on the tuning seeds 150 to 154 (``tune_scratch.py``) before any dev or confirmatory seed is run.
"""

import hashlib
import pickle
import re
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import learned_map as LM
import reps
import runlib

HERE = Path(__file__).resolve().parent
ARMS = ("R5_FS_strict", "R5_FS_pre")
DIM = 4096
HYPER: Dict[str, Dict[str, object]] = {                              # copied by hand from results/FS_TUNING.md (tuning seeds 150 to 154): the winner of each arm
    "R5_FS_strict": dict(out_dim=64, hidden=1024, tau=0.1, twin_weight=4.0, epochs=300, lr=1e-3, weight_decay=1e-2),
    "R5_FS_pre": dict(out_dim=64, hidden=1024, tau=0.1, twin_weight=4.0, epochs=300, lr=1e-3, weight_decay=1e-2),
}
PRETRAIN = dict(n_sentences=20000, epochs=3, batch=256, tau=0.1, lr=1e-3, drop=0.2, hidden=512, out_dim=64, seed=0)
_WORD = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")


def _h(token: str) -> int:
    return zlib.crc32(token.encode("utf-8"))


class BagRep(reps.Representation):
    """Signed hashed bag of lower-cased word 1-, 2- and 3-grams and character 3-grams (4,096 numbers). Knows spelling only, as R0 does, and has more room."""

    def __init__(self) -> None:
        super().__init__("FS_bag", DIM)

    @staticmethod
    def features(text: str) -> Dict[int, float]:
        words = _WORD.findall(text.lower())
        grams = [" ".join(words[i:i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)]
        low = " " + " ".join(text.lower().split()) + " "
        grams += ["#" + low[i:i + 3] for i in range(len(low) - 2)]
        out: Dict[int, float] = {}
        for g in grams:
            h = _h(g)
            out[h % DIM] = out.get(h % DIM, 0.0) + (1.0 if (h >> 16) & 1 else -1.0)
        return out

    def _encode(self, texts: List[str]) -> np.ndarray:
        z = np.zeros((len(texts), DIM), dtype=np.float32)
        for k, t in enumerate(texts):
            for j, v in self.features(t).items():
                z[k, j] = v
        return z


def _net(d: int, hidden: int, out_dim: int):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, out_dim))


def pretrain(sentences: Sequence[str], hyper: Optional[dict] = None, device: str = "cpu"):
    """Label-free pre-training: two word-dropout views of each sentence are pulled together against the other sentences of the batch. Returns (state_dict, mu, sd) with the input
    standardisation of the pre-training sentences. Deterministic for a given (sentences, hyper)."""
    import torch
    h = {**PRETRAIN, **(hyper or {})}
    rep = BagRep()
    g = np.random.default_rng(h["seed"])
    torch.manual_seed(h["seed"])
    texts = list(sentences)
    base = rep.encode(texts)
    mu, sd = base.mean(axis=0), base.std(axis=0) + 1e-6

    def view(t: str) -> str:
        words = t.split()
        kept = [w for w in words if g.random() >= h["drop"]] or words[:1]
        return " ".join(kept)

    net = _net(DIM, h["hidden"], h["out_dim"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=h["lr"], weight_decay=1e-2)
    n = len(texts)
    for _ in range(h["epochs"]):
        order = g.permutation(n)
        for s in range(0, n - 1, h["batch"]):
            idx = order[s:s + h["batch"]]
            if len(idx) < 2:
                continue
            a = torch.as_tensor((rep.encode([view(texts[i]) for i in idx]) - mu) / sd, dtype=torch.float32, device=device)
            b = torch.as_tensor((rep.encode([view(texts[i]) for i in idx]) - mu) / sd, dtype=torch.float32, device=device)
            ua, ub = (torch.nn.functional.normalize(net(x), dim=1) for x in (a, b))
            logits = (ua @ ub.T) / h["tau"]
            labels = torch.arange(len(idx), device=device)
            loss = 0.5 * (torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.T, labels))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}, mu, sd


def pretrained(cache_dir: Optional[Path] = None, hyper: Optional[dict] = None, sentences: Optional[Sequence[str]] = None, device: str = "auto"):
    """``pretrain`` on ``n_sentences`` everyday sentences of the local corpora (a fixed draw), kept on disk under a key made of the hyper-parameters and the sentences."""
    import torch
    import corpora
    h = {**PRETRAIN, **(hyper or {})}
    texts = list(sentences) if sentences is not None else corpora.load_other_sentences(h["n_sentences"], 0x5C4A7C4)
    key = hashlib.sha256((repr(sorted(h.items())) + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
    path = Path(cache_dir or HERE / "_cache") / f"fs_pre_{key}.pt"
    if path.exists():
        try:
            d = torch.load(path, weights_only=True)                          # tensors only: nothing is unpickled
            return d["state"], d["mu"].numpy(), d["sd"].numpy()
        except (OSError, RuntimeError, EOFError, KeyError, ValueError, pickle.UnpicklingError):          # a half-written or unreadable file is rebuilt
            pass
    dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    out = pretrain(texts, h, dev)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"state": out[0], "mu": torch.from_numpy(np.asarray(out[1])), "sd": torch.from_numpy(np.asarray(out[2]))}, tmp)
    tmp.replace(path)
    return out


class ScratchMap(LM.LearnedMap):
    """The learned map with its own network: hidden layer always on, and optionally started from a pre-trained network (and its input standardisation)."""

    def __init__(self, taught: Sequence[str], init=None, **kw) -> None:
        super().__init__(taught, **kw)
        self.init = init

    def fit(self, X, skill, family, template) -> None:
        import torch
        X = np.asarray(X, dtype=np.float32)
        skill, family, template = (np.asarray(a) for a in (skill, family, template))
        if not (len(X) == len(skill) == len(family) == len(template)):
            raise ValueError("X, skill, family and template must have one entry per row")
        bad_family = sorted(set(family.tolist()) - LM.ALLOWED_FAMILIES)
        if bad_family:
            raise ValueError(f"the map may only be fitted on the teaching wordings {sorted(LM.ALLOWED_FAMILIES)}; got families {bad_family}")
        bad_skill = sorted(set(skill.tolist()) - self.taught - {"none"})
        if bad_skill:
            raise ValueError(f"the map may only be fitted on taught skills {sorted(self.taught)} (and 'none'); got {bad_skill}")
        n = len(X)
        is_none = skill == "none"
        same_skill = skill[:, None] == skill[None, :]
        same_surface = (family[:, None] == family[None, :]) & (template[:, None] == template[None, :])
        eye = np.eye(n, dtype=bool)
        positive = same_skill & ~same_surface & ~eye & ~is_none[:, None]
        neutral = (same_skill & same_surface & ~eye) | (is_none[:, None] & is_none[None, :] & ~eye)
        twin = ~same_skill & (template[:, None] == template[None, :]) & ~is_none[:, None] & ~is_none[None, :]
        if not (positive.sum(axis=1) > 0).any() or len({s for s in skill.tolist() if s != "none"}) < 2:
            raise ValueError("the map needs at least two taught skills, each in at least two surfaces")
        dev = self._dev()
        torch.manual_seed(self.seed)
        if self.init is not None:
            state, self._mu, self._sd = self.init[0], np.asarray(self.init[1], dtype=np.float32), np.asarray(self.init[2], dtype=np.float32)
        else:
            state, self._mu, self._sd = None, X.mean(axis=0), X.std(axis=0) + 1e-6
        Z = torch.as_tensor((X - self._mu) / self._sd, dtype=torch.float32, device=dev)
        net = _net(X.shape[1], self.hidden, self.out_dim).to(dev)
        if state is not None:
            net.load_state_dict(state)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos = torch.as_tensor(positive, device=dev)
        w = torch.ones((n, n), dtype=torch.float32, device=dev)
        w[torch.as_tensor(twin, device=dev)] = float(self.twin_weight)
        w[torch.as_tensor(neutral | eye, device=dev)] = 0.0
        log_w = torch.log(w.clamp_min(1e-30))
        anchors = pos.any(dim=1)
        n_pos = pos.sum(dim=1).clamp_min(1)
        for _ in range(self.epochs):
            opt.zero_grad()
            U = torch.nn.functional.normalize(net(Z), dim=1)
            s = (U @ U.T) / self.tau
            log_den = torch.logsumexp(s + log_w, dim=1, keepdim=True)
            loss = (-((s - log_den) * pos).sum(dim=1) / n_pos)[anchors].mean()
            loss.backward()
            opt.step()
        self._net = net.eval()


def make_mapper(arm: str, seed: int, init=None, **hyper) -> Callable:
    """The hook ``ceiling.run_ceiling`` and ``run_e2.run_cell`` expect, for one arm. ``init`` is the pre-trained network for ``R5_FS_pre`` (loaded from the cache when not given)."""
    if arm not in ARMS:
        raise KeyError(f"unknown from-scratch arm {arm!r}")
    params = {**HYPER[arm], **hyper}
    if arm == "R5_FS_pre" and init is None:
        init = pretrained(hyper={"hidden": params["hidden"], "out_dim": params["out_dim"]})          # the pre-trained network must have the shape of the one that is fine-tuned

    def mapper(X: np.ndarray, items: List, taught: Sequence[str]) -> Callable[[np.ndarray], np.ndarray]:
        m = ScratchMap(taught, init=init if arm == "R5_FS_pre" else None, seed=seed, **params)
        m.fit(X, [i.label for i in items], [i.family for i in items], [i.template_ids[0] for i in items])
        return m.transform

    return mapper


def stamp() -> str:
    """The code stamp of every from-scratch cell: the experiment's stamped files plus this file and the learned map it builds on."""
    import run_e2
    return runlib.code_stamp(run_e2.STAMP_FILES + [HERE / "scratch_reader.py", HERE / "learned_map.py"])


def run_e1_cell(arm: str, seed: int, pool: str, out_dir, stamp: str, **hyper) -> dict:
    """E1 (the information ceiling) for a from-scratch arm: ``run_e1.run_cell`` builds the Qwen learned map itself, so this does the same with this arm's mapper."""
    import json
    import ceiling
    import run_e2 as R
    path = Path(out_dir) / f"{arm}__s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    res = ceiling.run_ceiling("FS_bag", seed, pool, sizes=ceiling.SIZES, rep=BagRep(), cache_dir=R.CACHE, mapper=make_mapper(arm, seed, **hyper))
    res["rep"], res["code_stamp"] = arm, stamp
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(res))
    return res


def run_e2_cell(arm: str, seed: int, pool: str, out_dir, stamp: str, **hyper) -> dict:
    import run_e2 as R
    return R.run_cell(arm, seed, pool, out_dir, stamp, rep=BagRep(), mapper=make_mapper(arm, seed, **hyper))
