"""The router: the domain-free routing and novelty layer as one object a pool can use (design note section 7).

For every message it says who answers, whether anyone should, and which readers said "not mine". It is built from what labs G1-G4c measured and nothing else:
* a roster of cards (the teacher's name for each expert and two of its teaching messages), shown to a language model as a multiple-choice question (the asking reader, ``reader.py``);
* the pool's own density gate on message vectors (the vMF detector of the lab);
* every reader's cut calibrated on VERIFIED outcomes of the stream's own wording (``observe``: a message whose ownership a label or a stated answer confirmed); at least 19 are needed, below that
  the router falls back to the asking reader's own claim rule and says so (``mode``); an optional initial calibration in the teaching wording serves until the stream's outcomes take over;
* the two readers fused by "either": a message is "not mine" when either reader says so (answer-more; a stated policy, not a derived fact).
The experts are callables on features and the pool is not edited: ``add_expert`` is how a growing pool tells the router about a new one.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_KS = _HERE.parent
for _p in (str(_KS.parents[1]), str(_KS), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import baselines as BL  # noqa: E402
import cards as CD  # noqa: E402
import measures as MS  # noqa: E402
import route as RT  # noqa: E402

READERS = ("ask_exemplars", "vmf_conformal")          # the names the lab gives the two readers
N_EXEMPLARS = 2


@dataclass(frozen=True)
class Route:
    status: str                       # "claimed" | "unclaimed" (unclaimed = "not mine": noise, out of scope or a new skill; the router does not say which)
    expert: Optional[int]             # the expert to send it to (the asking reader's best card) when claimed
    name: Optional[str]
    team: Tuple[int, ...]             # experts by probability, when the reader itself claims; the single best one otherwise
    p_card: float                     # the probability the asking reader gives its best card
    mode: str                         # "calibrated" | "initial" | "uncalibrated"
    flagged_by: Tuple[str, ...]       # the readers that said "not mine" (or "claim rule" before any calibration)


@dataclass(frozen=True)
class Response:
    route: Route
    value: Optional[object]
    reason: str                       # "answered" | "not mine" | "unreadable"


class Router:
    def __init__(self, names: Sequence[str], teach: Mapping[str, Sequence[str]], encode: Callable[[Sequence[str]], np.ndarray], reader_factory: Callable[[Sequence[CD.Card]], object],
                 q: float = 0.05, tau: float = 0.5, window: int = 60, initial_cal: Sequence[str] = (), states: Optional[Sequence[str]] = None,
                 experts: Optional[Sequence[Callable]] = None, featurize: Optional[Callable[[str], Optional[tuple]]] = None, seed: int = 0) -> None:
        if not 0.0 < q < 1.0 or window < 1:
            raise ValueError(f"need 0 < q < 1 and a window of at least 1, got q={q}, window={window}")
        self.encode, self.reader_factory, self.q, self.tau, self.window, self.seed = encode, reader_factory, q, tau, window, seed
        self.names: List[str] = []
        self.teach: Dict[str, List[str]] = {}
        self.states: List[str] = []
        self.experts: Optional[List[Callable]] = None if experts is None else []
        self.featurize = featurize
        if experts is not None and len(experts) != len(names):
            raise ValueError(f"{len(experts)} experts for {len(names)} names")
        if states is not None and len(states) != len(names):
            raise ValueError(f"{len(states)} states for {len(names)} names")
        for i, n in enumerate(names):
            if n not in teach:
                raise ValueError(f"no teaching messages for expert {n!r}")
            self._append(n, teach[n], "solid" if states is None else states[i], None if experts is None else experts[i])
        self.needed = int(math.ceil((1.0 - q) / q - 1e-9))        # the fewest calibration messages for which a cut at q can exist
        self._verified: List[str] = []
        self._initial: List[str] = [str(t) for t in initial_cal]
        self._rebuild()

    # ------------------------------------------------------------------------------------------------------------------------------ roster
    def _append(self, name: str, teach: Sequence[str], state: str, expert: Optional[Callable]) -> None:
        if name in self.names:
            raise ValueError(f"expert names must be distinct, {name!r} is taken")
        msgs = [str(t) for t in teach]
        if len(msgs) < N_EXEMPLARS:
            raise ValueError(f"expert {name!r} needs at least {N_EXEMPLARS} teaching messages, got {len(msgs)}")
        self.names.append(name)
        self.teach[name] = msgs
        self.states.append(state)
        if self.experts is not None:
            if expert is None:
                raise ValueError("this router has experts attached, so a new expert needs its callable")
            self.experts.append(expert)

    def _rebuild(self) -> None:
        """Everything that depends on the roster: the cards and the asking reader, the density gate, and the calibration scores of the messages already kept."""
        self.cards = [CD.Card(n, tuple(self.teach[n][:N_EXEMPLARS])) for n in self.names]
        self.reader = self.reader_factory(self.cards)
        X, y = [], []
        for i, n in enumerate(self.names):
            X += self.teach[n]
            y += [i] * len(self.teach[n])
        self.det = BL.make_baseline("vmf_conformal", seed=self.seed).fit(self.encode(X), np.asarray(y))
        self._vscores = self._score(self._verified)[0] if self._verified else {r: np.zeros(0) for r in READERS}
        self._iscores = self._score(self._initial)[0] if self._initial else {r: np.zeros(0) for r in READERS}

    def add_expert(self, name: str, teach: Sequence[str], state: str = "solid", expert: Optional[Callable] = None) -> None:
        """A new expert joined the pool: the roster, the question, the density gate and the calibration scores are rebuilt; the verified outcomes are kept."""
        if len(self.names) + 1 > CD.MAX_CARDS:
            raise ValueError(f"a roster holds at most {CD.MAX_CARDS} cards")
        self._append(name, teach, state, expert)
        self._rebuild()

    # ------------------------------------------------------------------------------------------------------------------------------ scores
    def _score(self, texts: Sequence[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        texts = list(texts)
        P = np.asarray(self.reader.probabilities(texts), dtype=np.float64)
        return {"ask_exemplars": RT.unclaimed_score(P), "vmf_conformal": self.det.novelty_score(self.encode(texts))}, P

    def scores(self, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        """Each reader's novelty score of each message (larger = less like anything taught)."""
        return self._score(texts)[0]

    # ------------------------------------------------------------------------------------------------------------------------------ outcomes
    def observe(self, texts: Sequence[str]) -> None:
        """Messages whose ownership an outcome confirmed (a label, or a stated answer that a taught expert reproduced). They calibrate every reader's cut; only the latest ``window`` are kept."""
        new = [str(t) for t in texts]
        if not new:
            return
        s = self.scores(new)
        self._verified = (self._verified + new)[-self.window:]
        self._vscores = {r: np.concatenate([self._vscores[r], s[r]])[-self.window:] for r in READERS}

    def _calibration(self) -> Tuple[Optional[Dict[str, np.ndarray]], str]:
        if len(self._verified) >= self.needed:
            return self._vscores, "calibrated"
        if len(self._initial) >= self.needed:
            return self._iscores, "initial"
        return None, "uncalibrated"

    def status(self) -> Dict[str, object]:
        return {"experts": len(self.names), "verified": len(self._verified), "needed": self.needed, "mode": self._calibration()[1]}

    # ------------------------------------------------------------------------------------------------------------------------------ routing
    def route(self, texts: Sequence[str]) -> List[Route]:
        texts = list(texts)
        if not texts:
            return []
        sc, P = self._score(texts)
        k = len(self.names)
        claim = RT.decide(P, self.states, self.tau)
        cal, mode = self._calibration()
        if cal is None:
            flagged = np.array([d.status == "unclaimed" for d in claim])
            by = [("claim rule",) if f else () for f in flagged]
        else:
            fused = MS.fused_scores(cal, sc, READERS, "either")
            cut = MS.conformal_cut(MS.fused_scores(cal, cal, READERS, "either", leave_one_out=True), self.q)
            flagged = fused > cut
            pv = {r: MS.p_novel(cal[r], sc[r]) for r in READERS}
            by = [tuple(r for r in READERS if pv[r][i] < 1.0 - cut) if flagged[i] else () for i in range(len(texts))]
        out: List[Route] = []
        for i in range(len(texts)):
            best = int(P[i, :k].argmax())
            p = float(P[i, best])
            if flagged[i]:
                out.append(Route("unclaimed", None, None, (), p, mode, by[i]))
            else:
                team = claim[i].team if claim[i].status != "unclaimed" else (best,)
                out.append(Route("claimed", best, self.names[best], team, p, mode, ()))
        return out

    def respond(self, texts: Sequence[str]) -> List[Response]:
        """Route and answer: the claimed expert's answer on the message's features, nothing for a message that is not anyone's."""
        if self.experts is None or self.featurize is None:
            raise ValueError("respond needs experts and a featurizer; this router only routes")
        texts = list(texts)
        out: List[Response] = []
        for t, r in zip(texts, self.route(texts)):
            if r.status == "unclaimed":
                out.append(Response(r, None, "not mine"))
                continue
            f = self.featurize(t)
            out.append(Response(r, None, "unreadable") if f is None else Response(r, self.experts[r.expert](f), "answered"))
        return out
