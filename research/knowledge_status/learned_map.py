"""R5 (Task 9): a small learned map that sorts messages by the operation they ask for instead of by how they are worded.

The off-the-shelf representations put "17 plus 5" next to "17 times 5" (same wording, other operation) and far from "I had 17 apples and got 5 more" (same operation,
other wording). The map is trained contrastively to undo that, on the teaching messages only:

* pulled together: the same skill in a different surface (another family or another wrapper template);
* pushed apart: every other skill, and above all the near-twin (the same wrapper template, another skill) which gets ``twin_weight`` times the push;
* the "none" messages (everyday sentences, traps) are pushed away from every skill and are not pulled together with each other: "none" is everything else, not a topic.

What it may see is fixed by the constructor and checked in ``fit``: only skills that were taught (plus "none"), and only the teaching wordings S1 to S3 (plus the
"other" and "trap" none-messages). A held-out wording (S4 to S8, "unseen") or an untaught skill raises ``ValueError``, so no evaluation can borrow from them by accident.
"""

import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import assignments
import skills

ALLOWED_FAMILIES = frozenset(assignments.TEACH_FAMILIES) | {"other", "trap"}
HYPER: Dict[str, object] = dict(out_dim=64, hidden=0, tau=0.1, twin_weight=4.0, epochs=300, lr=3e-3, weight_decay=1e-2)       # the operating point; tuned on dev seeds 105 to 109


class LearnedMap:
    """``fit(X, skill, family, template)`` learns the map, ``transform(X)`` applies it and returns unit-length rows. ``hidden=0`` is one linear layer, otherwise one hidden
    layer of that width. The input is standardised with the mean and spread of the training rows first (raw language-model states are nearly degenerate, Task 6)."""

    def __init__(self, taught: Sequence[str], out_dim: int = 64, hidden: int = 0, tau: float = 0.1, twin_weight: float = 4.0, epochs: int = 300, lr: float = 3e-3,
                 weight_decay: float = 1e-2, seed: int = 0, device: str = "auto") -> None:
        unknown = sorted(set(taught) - set(skills.SKILLS))
        if unknown:
            raise ValueError(f"not skills: {unknown}")
        self.taught = frozenset(taught)
        self.out_dim, self.hidden, self.tau, self.twin_weight = out_dim, hidden, tau, twin_weight
        self.epochs, self.lr, self.weight_decay, self.seed, self.device = epochs, lr, weight_decay, seed, device
        self._net = None
        self._mu: Optional[np.ndarray] = None
        self._sd: Optional[np.ndarray] = None

    def _dev(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() and os.environ.get("KS_DEVICE") != "cpu" else "cpu"

    def fit(self, X: np.ndarray, skill: Sequence[str], family: Sequence[str], template: Sequence[str]) -> None:
        import torch
        X = np.asarray(X, dtype=np.float32)
        skill, family, template = (np.asarray(a) for a in (skill, family, template))
        if not (len(X) == len(skill) == len(family) == len(template)):
            raise ValueError("X, skill, family and template must have one entry per row")
        bad_family = sorted(set(family.tolist()) - ALLOWED_FAMILIES)
        if bad_family:
            raise ValueError(f"the map may only be fitted on the teaching wordings {sorted(ALLOWED_FAMILIES)}; got families {bad_family}")
        bad_skill = sorted(set(skill.tolist()) - self.taught - {"none"})
        if bad_skill:
            raise ValueError(f"the map may only be fitted on taught skills {sorted(self.taught)} (and 'none'); got {bad_skill}")
        n = len(X)
        is_none = skill == "none"
        same_skill = skill[:, None] == skill[None, :]
        same_surface = (family[:, None] == family[None, :]) & (template[:, None] == template[None, :])
        eye = np.eye(n, dtype=bool)
        positive = same_skill & ~same_surface & ~eye & ~is_none[:, None]                  # the same skill, reworded
        neutral = (same_skill & same_surface & ~eye) | (is_none[:, None] & is_none[None, :] & ~eye)     # says nothing either way
        twin = ~same_skill & (template[:, None] == template[None, :]) & ~is_none[:, None] & ~is_none[None, :]
        if not (positive.sum(axis=1) > 0).any() or len({s for s in skill.tolist() if s != "none"}) < 2:
            raise ValueError("the map needs at least two taught skills, each in at least two surfaces")

        dev = self._dev()
        torch.manual_seed(self.seed)
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0) + 1e-6
        Z = torch.as_tensor((X - self._mu) / self._sd, dtype=torch.float32, device=dev)
        d = X.shape[1]
        layers = [torch.nn.Linear(d, self.out_dim)] if self.hidden == 0 else [torch.nn.Linear(d, self.hidden), torch.nn.GELU(), torch.nn.Linear(self.hidden, self.out_dim)]
        net = torch.nn.Sequential(*layers).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos = torch.as_tensor(positive, device=dev)
        # the weight of every other row in an anchor's denominator: 0 for itself and for neutral rows, twin_weight for near-twins, 1 for the rest
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
            per_anchor = -((s - log_den) * pos).sum(dim=1) / n_pos
            loss = per_anchor[anchors].mean()
            loss.backward()
            opt.step()
        self._net = net.eval()

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("fit the map first")
        import torch
        dev = next(self._net.parameters()).device
        Z = torch.as_tensor((np.asarray(X, dtype=np.float32) - self._mu) / self._sd, dtype=torch.float32, device=dev)
        with torch.no_grad():
            U = torch.nn.functional.normalize(self._net(Z), dim=1)
        return U.cpu().numpy().astype(np.float32)


def make_mapper(seed: int, **hyper) -> Callable:
    """The hook ``ceiling.run_ceiling`` expects: given the embeddings of the training items, the items and the skills taught, learn a map and return its ``transform``."""
    params = {**HYPER, **hyper}

    def mapper(X: np.ndarray, items: List, taught: Sequence[str]) -> Callable[[np.ndarray], np.ndarray]:
        m = LearnedMap(taught, seed=seed, **params)
        m.fit(X, [i.label for i in items], [i.family for i in items], [i.template_ids[0] for i in items])
        return m.transform

    return mapper
