"""A pool of experts behind a gate (inference time).

The pool does not know a domain: the experts are callables on features, the featurizer turns a message into those features (or None when it cannot), and the gate decides who answers and
whether anyone does. ``cards`` builds the roster the asking gate needs from the pool's own teaching messages. This is the container the general layer is attached to; the growmoe pool
(``growmoe/pools/phylo.py``) is not edited and not yet used as the container.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import cards as CD


@dataclass(frozen=True)
class Answer:
    value: Optional[int]
    expert: Optional[int]
    reason: str                       # "answered" | "abstained" | "unreadable"


class GatedPool:
    def __init__(self, names: Sequence[str], experts: Sequence[Callable], featurize: Callable[[str], Optional[tuple]], gate) -> None:
        if len(names) != len(experts):
            raise ValueError(f"{len(names)} names for {len(experts)} experts")
        self.names, self.experts, self.featurize, self.gate = list(names), list(experts), featurize, gate

    def predict(self, texts: Sequence[str]) -> List[Answer]:
        route, abstain = self.gate.decide(list(texts))
        out: List[Answer] = []
        for t, r, ab in zip(texts, route, abstain):
            if ab:
                out.append(Answer(None, None, "abstained"))
                continue
            f = self.featurize(t)
            out.append(Answer(None, int(r), "unreadable") if f is None else Answer(self.experts[int(r)](f), int(r), "answered"))
        return out

    def cards(self, teach: Mapping[str, Sequence[str]], n_exemplars: int = 2) -> List[CD.Card]:
        return [CD.Card(n, tuple(teach[n][:n_exemplars])) for n in self.names]

    def summary(self) -> Dict[str, object]:
        return {"n_experts": len(self.experts), "names": list(self.names), "gate": type(self.gate).__name__}
