"""A stream that demands growth, with the answer key made by an oracle.

Inputs are directions on a sphere (d = 32). A *regime* is a patch of the sphere around a random centre with its own labelling rule over K = 6 classes. The rule is drawn from one of three families:
``lin`` (a linear rule: a tiny network suffices), ``wide`` (a maximum of 24 random linear pieces, 4 per class: it takes many units), ``deep`` (class scores built from products of four tanh features:
interactions that a narrow shallow network approximates badly). What a regime NEEDS is not assumed from its family: ``oracle_need`` trains three fixed networks offline on the regime's own data (small
8 units, wide 64 units, deep 3 blocks of 8 units) and calls the need ``width`` / ``depth`` / ``none`` by what they reach. The system never sees a regime id; the id is bookkeeping for scoring only.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

D, K = 32, 6
FAMILIES = ("lin", "wide", "deep")


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class Regime:
    def __init__(self, rid: int, kind: str, rng: np.random.Generator, d: int = D, k: int = K, spread: float = 1.5, pieces: int = 24, centre: Optional[np.ndarray] = None) -> None:
        if kind not in FAMILIES:
            raise ValueError(f"kind must be one of {list(FAMILIES)}, got {kind!r}")
        self.rid, self.kind, self.d, self.k, self.spread, self.pieces = rid, kind, d, k, spread, pieces
        self.centre = unit(rng.standard_normal(d)) if centre is None else centre
        self.offset = np.zeros(k)
        for _ in range(80):                                                  # redraw until every class holds at least 4 % of the regime
            self._draw(rng)
            x = self._inputs(4000, rng)
            if self.kind != "wide":
                self._balance(x)
            if np.bincount(self.label(x), minlength=k).min() >= 0.04 * 4000:
                break

    def _draw(self, rng: np.random.Generator) -> None:
        d, k = self.d, self.k
        self.offset = np.zeros(k)
        if self.kind == "lin":
            self.A = rng.standard_normal((k, d))
        elif self.kind == "wide":
            self.A, self.b, self.cls = rng.standard_normal((self.pieces, d)), rng.standard_normal(self.pieces) * 0.5, np.arange(self.pieces) % k
        else:
            self.A, self.C = rng.standard_normal((4, d)), rng.standard_normal((k, 10))

    def _balance(self, x: np.ndarray) -> None:
        for _ in range(100):
            freq = np.bincount(self.label(x), minlength=self.k) / len(x)
            self.offset += 0.5 * (1.0 / self.k - freq)

    def _inputs(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return unit(self.centre + self.spread * rng.standard_normal((n, self.d)) / np.sqrt(self.d)).astype(np.float32)

    def scores(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.centre) * np.sqrt(self.d) / self.spread
        if self.kind == "lin":
            return z @ self.A.T
        if self.kind == "wide":
            j = (z @ self.A.T + self.b).argmax(1)
            out = np.zeros((len(z), self.k))
            out[np.arange(len(z)), self.cls[j]] = 1.0
            return out
        u = np.tanh(2.0 * (z @ self.A.T))
        feats = np.stack([u[:, i] * u[:, j] for i in range(4) for j in range(i + 1, 4)] + [u[:, 0], u[:, 1], u[:, 2], u[:, 3]], axis=1)
        return feats @ self.C.T

    def label(self, x: np.ndarray) -> np.ndarray:
        return (self.scores(x) + self.offset).argmax(1)

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        x = self._inputs(n, rng)
        return x, self.label(x)


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    regime: int
    phase: int


class GrowthWorld:
    """``kinds`` lists the family of every regime in order of arrival; regimes arrive ``per_phase`` at a time; a phase lasts ``phase_batches`` batches, half of them from the regimes that just arrived and
    half from all earlier ones (so old regimes recur and forgetting would show)."""

    def __init__(self, seed: int, kinds: Sequence[str], per_phase: int = 4, phase_batches: int = 400, batch: int = 128, n_test: int = 500, min_sep: float = 0.3, n_groups: int = 0, share: float = 0.6, sibling_sep: float = 0.45) -> None:
        self.seed, self.kinds, self.per_phase, self.phase_batches, self.batch = seed, list(kinds), per_phase, phase_batches, batch
        rng = np.random.default_rng([seed, 0x47524F])
        centres: List[np.ndarray] = []                                      # regimes sit at least 0.3 apart in cosine: the growth lab isolates growth decisions from the (separate, known) near-twin problem
        supers: List[np.ndarray] = []                                       # with ``n_groups`` the regimes cluster: a regime's centre is ``share`` of its group's direction plus its own, so groups exist for a tier to find
        for _ in range(n_groups):
            for _try in range(2000):
                s = unit(rng.standard_normal(D))
                if all(abs(float(s @ o)) <= min_sep for o in supers):
                    break
            supers.append(s)
        self.group = [i % n_groups if n_groups else -1 for i in range(len(self.kinds))]
        for i, _ in enumerate(self.kinds):
            for _try in range(2000):
                own = unit(rng.standard_normal(D))
                c = unit(share * supers[self.group[i]] + np.sqrt(1.0 - share ** 2) * own) if n_groups else own
                if all(abs(float(c @ o)) <= (sibling_sep if self.group[j] == self.group[i] and n_groups else min_sep) for j, o in enumerate(centres)):
                    break
            centres.append(c)
        self.regimes: List[Regime] = [Regime(i, k, rng, centre=centres[i]) for i, k in enumerate(self.kinds)]
        self.test: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for r in self.regimes:
            x, y = r.sample(n_test, np.random.default_rng([seed, 0x54455354, r.rid]))
            self.test[r.rid] = (torch.from_numpy(x), torch.from_numpy(y))
        self.n_phases = int(np.ceil(len(self.kinds) / per_phase))

    def stream(self) -> List[Batch]:
        rng = np.random.default_rng([self.seed, 0x53545245])
        out: List[Batch] = []
        for p in range(self.n_phases):
            new = list(range(p * self.per_phase, min((p + 1) * self.per_phase, len(self.regimes))))
            old = list(range(0, p * self.per_phase))
            for _ in range(self.phase_batches):
                pool = new if (not old or rng.random() < 0.5) else old
                r = int(rng.choice(pool))
                x, y = self.regimes[r].sample(self.batch, rng)
                out.append(Batch(torch.from_numpy(x), torch.from_numpy(y), r, p))
        return out


def _fit_offline(model, x: torch.Tensor, y: torch.Tensor, steps: int, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        idx = torch.randint(0, x.shape[0], (128,), generator=g)
        model.train_step(x[idx], y[idx], opt)
    model.eval()


def oracle_need(regime: Regime, seed: int, n_train: int = 12000, n_test: int = 1500, steps: int = 1500, margin: float = 0.05) -> Dict[str, object]:
    """What the regime needs, by offline fixed networks on its own data: ``width`` when the wide net beats the small one by ``margin`` and the deep net; ``depth`` when the deep net beats the small
    one by ``margin`` and the wide net; otherwise ``none``."""
    from growmoe.experts import MLPExpert
    rng = np.random.default_rng([seed, 0x4F52, regime.rid])
    x, y = (torch.from_numpy(a) for a in regime.sample(n_train, rng))
    xt, yt = (torch.from_numpy(a) for a in regime.sample(n_test, rng))
    acc = {}
    for name, (h, depth) in {"small": (8, 1), "wide": (64, 1), "deep": (8, 3)}.items():
        m = MLPExpert(D, h, depth, regime.k, seed=seed + regime.rid)
        _fit_offline(m, x, y, steps, seed)
        with torch.no_grad():
            acc[name] = float((m(xt).argmax(1) == yt).float().mean())
    return {"need": need_from_acc(acc, margin), "acc": acc}


def need_from_acc(acc: Dict[str, float], margin: float = 0.05) -> str:
    """``width`` when the wide net beats the small one by ``margin`` and is not behind the deep net; ``depth`` when the deep net beats the small one by ``margin`` and the wide net; else ``none``."""
    if acc["wide"] - acc["small"] >= margin and acc["wide"] >= acc["deep"]:
        return "width"
    if acc["deep"] - acc["small"] >= margin and acc["deep"] > acc["wide"]:
        return "depth"
    return "none"


SCENARIOS = {
    "mixed": ["lin"] * 4 + ["wide"] * 8 + ["deep"] * 4,                      # 16 regimes with different needs
    "easy": ["lin"] * 8,                                                      # nothing needs growth beyond new experts
    "big": ["lin"] * 36 + ["wide"] * 6 + ["deep"] * 6,                        # 48 regimes: the flat gate becomes the cost
}


def oracle_table(world: GrowthWorld, scenario: str, out_dir, only: Sequence[str] = FAMILIES) -> Dict[int, Dict[str, object]]:
    """The oracle need of every regime of the families in ``only`` (the others are recorded as ``none`` without training: they are the ``lin`` rules of the big scenario). Cached in
    ``out_dir/oracle_<scenario>_s<seed>.json``, written atomically, so a power cut costs one regime at most."""
    import json
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"oracle_{scenario}_s{world.seed}.json"
    table: Dict[int, Dict[str, object]] = {}
    if path.exists():
        table = {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    for r in world.regimes:
        if r.rid in table:
            continue
        table[r.rid] = oracle_need(r, world.seed) if r.kind in only else {"need": "none", "acc": {}, "skipped": True}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(table), encoding="utf-8")
        tmp.replace(path)
        print(f"oracle {scenario} s{world.seed} regime {r.rid} {r.kind}: {table[r.rid]['need']} {table[r.rid]['acc']}", flush=True)
    return table
