"""Gates: how a pool decides, for each message, which expert answers and whether anyone should.

A gate returns (route, abstain) for a list of messages: the index of the expert it would send the message to, and whether it refuses to answer at all. Three gates:
* ``VmfGate``: the pool's own gate, a density of the message vector per expert (``growmoe.signature.VmfSignature``), cut at the share q of held-out known messages;
* ``AskGate``: the general layer's gate, the card reader and the domain-free rule (claimed or not);
* ``MixGate``: abstains when either gate does and routes with the first.
"""

import sys
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_KS = _HERE.parent
for _p in (str(_KS.parents[1]), str(_KS), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import baselines as BL  # noqa: E402
import measures as MS  # noqa: E402
import route as RT  # noqa: E402


class VmfGate:
    def __init__(self, encode: Optional[Callable[[Sequence[str]], np.ndarray]] = None, q: float = 0.05, seed: int = 0) -> None:
        self.encode, self.q, self.seed = encode, q, seed

    def fit(self, X: np.ndarray, y: np.ndarray, cal_X: np.ndarray) -> "VmfGate":
        y = np.asarray(y)
        if sorted(set(y.tolist())) != list(range(int(y.max()) + 1)):
            raise ValueError("expert indices must be 0..k-1 with every expert present")
        self.det = BL.make_baseline("vmf_conformal", seed=self.seed).fit(X, y)
        self.cut = MS.conformal_cut(self.det.novelty_score(cal_X), self.q)
        return self

    def decide_vectors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        U = BL._unit(np.asarray(X, dtype=np.float64)).astype(np.float32)
        P = np.stack([s.pvalue(U).double().numpy() for s in self.det._sigs], axis=1)
        D = np.stack([s.score(U).double().numpy() for s in self.det._sigs], axis=1)
        return (P + 1e-7 * D).argmax(axis=1), self.det.novelty_score(X) > self.cut                 # p-value first, density as the tie-break (as in the detector)

    def decide(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        if self.encode is None:
            raise ValueError("this gate was built without an encoder; use decide_vectors")
        if not len(texts):
            return np.zeros(0, dtype=int), np.zeros(0, dtype=bool)
        return self.decide_vectors(self.encode(list(texts)))


class AskGate:
    def __init__(self, reader, states: Sequence[str], tau: float = 0.5) -> None:
        self.reader, self.states, self.tau = reader, list(states), tau

    def decide(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        if not len(texts):
            return np.zeros(0, dtype=int), np.zeros(0, dtype=bool)
        P = self.reader.probabilities(list(texts))
        dec = RT.decide(P, self.states, self.tau)
        return P[:, :-1].argmax(axis=1), np.array([d.status == "unclaimed" for d in dec], dtype=bool)


class MixGate:
    def __init__(self, first, second) -> None:
        self.first, self.second = first, second

    def decide(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        r1, a1 = self.first.decide(texts)
        _, a2 = self.second.decide(texts)
        return r1, np.asarray(a1, dtype=bool) | np.asarray(a2, dtype=bool)
