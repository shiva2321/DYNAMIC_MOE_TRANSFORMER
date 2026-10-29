"""Experts that SHARE what matters (Lab C). The separate solver of assignments.py learns each skill alone; a skill taught 20 or 100 examples stays at chance. Here the experts share one number
encoder (the two layers that turn the digits of the operands into features: carries, borrows, place value) and keep their own small output head. Every training step draws the same number of
examples from every skill, so a skill with few examples is revisited often while the encoder is held up by the skills with many.

Nothing is told about the operations: the encoder only ever sees digits and the head of a skill only ever sees that skill's answers.
"""

import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

_KS = Path(__file__).resolve().parents[1]
for _p in (str(_KS.parents[1]), str(_KS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from assignments import Solver  # noqa: E402
from growmoe import arith  # noqa: E402


def decode(sym: np.ndarray) -> np.ndarray:
    """Integer answers from the argmax symbols of the output slots (the rule of assignments.Solver.predict: a PAD slot counts as 0, the sign slot makes the number negative)."""
    digits = np.where(sym[:, :10] < 10, sym[:, :10], 0).astype(np.int64)
    value = (digits * (10 ** np.arange(10, dtype=np.int64))).sum(axis=1)
    return np.where(sym[:, arith.SIGN_POS] == arith.NEG, -value, value)


class SharedExperts:
    def __init__(self, names: Sequence[str], seed: int = 0, hidden: int = 128, steps: int = 6000, lr: float = 2e-3, batch: int = 128, weight_decay: float = 1e-4, device: str = "auto") -> None:
        if len(set(names)) != len(names) or not names:
            raise ValueError("expert names must be distinct and non-empty")
        self.names, self.seed, self.hidden, self.steps, self.lr, self.batch, self.weight_decay = list(names), seed, hidden, steps, lr, batch, weight_decay
        self.device, self.trunk, self.heads = device, None, None

    def _dev(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, data: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> "SharedExperts":
        import torch
        if set(data) != set(self.names):
            raise ValueError(f"data for {sorted(data)}, experts {sorted(self.names)}")
        dev = self._dev()
        X = {k: torch.from_numpy(Solver._features(v[0], v[1])).to(dev) for k, v in data.items()}
        T = {k: torch.from_numpy(arith.target_symbols(np.asarray(v[2], dtype=np.int64))).to(dev) for k, v in data.items()}
        d_in = next(iter(X.values())).shape[1]
        per = max(1, self.batch // len(self.names))
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            trunk = torch.nn.Sequential(torch.nn.Linear(d_in, self.hidden), torch.nn.ReLU(), torch.nn.Linear(self.hidden, self.hidden), torch.nn.ReLU()).to(dev)
            heads = torch.nn.ModuleDict({k: torch.nn.Linear(self.hidden, arith.N_POS * arith.N_CLASS) for k in self.names}).to(dev)
            opt = torch.optim.Adam(list(trunk.parameters()) + list(heads.parameters()), lr=self.lr, weight_decay=self.weight_decay)
            gen = torch.Generator().manual_seed(self.seed)
            for _ in range(self.steps):
                loss = 0.0
                for k in self.names:
                    idx = torch.randint(0, len(X[k]), (per,), generator=gen).to(dev)
                    logits = heads[k](trunk(X[k][idx])).view(-1, arith.N_POS, arith.N_CLASS)
                    loss = loss + torch.nn.functional.cross_entropy(logits.reshape(-1, arith.N_CLASS), T[k][idx].reshape(-1))
                opt.zero_grad()
                (loss / len(self.names)).backward()
                opt.step()
        self.trunk, self.heads = trunk.eval(), heads.eval()
        return self

    def predict(self, name: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        import torch
        if self.trunk is None:
            raise RuntimeError("fit the experts before predicting")
        if name not in self.heads:
            raise KeyError(name)
        with torch.no_grad():
            x = torch.from_numpy(Solver._features(a, b)).to(next(self.trunk.parameters()).device)
            sym = self.heads[name](self.trunk(x)).view(-1, arith.N_POS, arith.N_CLASS).argmax(-1).cpu().numpy()
        return decode(sym)

    def solver(self, name: str):
        """The expert as an object with ``predict(a, b)``, like assignments.Solver, so the function check can use it."""
        outer = self

        class _One:
            def predict(self, a, b):
                return outer.predict(name, a, b)

        return _One()


def held_out_accuracy(predict, op_id: int, rng: np.random.Generator, n: int = 500, lens: Tuple[int, int] = (1, 2)) -> float:
    """Exact-answer accuracy on problems the experts never saw (arith's held-out split); ``predict(a, b)`` gives integers."""
    held = arith.sample_problems(rng, n, (op_id,), lens, lens, "test")
    return float((np.asarray(predict(held["a"], held["b"])) == arith.answers(held["op"], held["a"], held["b"])).mean())
