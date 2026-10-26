"""Who has been taught what, and how well it really works (spec section 5.6).

A run draws an ASSIGNMENT of the six skills to Solid / Thin / Unseen (rotated, so the state is never tied to a skill's identity). Taught skills get teaching examples
(surface families S1 to S3 only); a small solver is trained on each skill's examples and scored on problems it has never seen, which gives the COMPETENCE table. The
status truth ("solid", "thin" or "grey") is then read off the measured count and accuracy, not off the label.
"""

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

import numpy as np

import generator as G
import skills
from growmoe import arith

STATES = ("solid", "thin", "unseen")
COUNTS = {"solid": 2000, "thin": 20}
COUNT_SWEEP = (5, 20, 100, 500, 2000)
TEACH_FAMILIES = ("S1", "S2", "S3")
TEST_FAMILIES = ("S4", "S5", "S6", "S7", "S8")
# The draft rule of spec 5.6. Dev may change these numbers; the freeze records them. Read at call time, so a change takes effect at once.
THRESHOLDS = {"solid_min_count": 500, "solid_min_acc": 0.90, "thin_max_count": 100, "thin_acc_margin": 0.10, "thin_max_acc": 0.60}
_EPS = 1e-9


def draw_assignment(rng: np.random.Generator, need_solid: FrozenSet[str] = frozenset(), n_solid: int = 2, n_thin: int = 1) -> Dict[str, str]:
    """Skill -> "solid" | "thin" | "unseen". The skills in ``need_solid`` are always Solid (a mul trace needs add, a div trace needs sub); the rest are drawn at random."""
    unknown = set(need_solid) - set(skills.SKILLS)
    if unknown:
        raise ValueError(f"unknown skills in need_solid: {sorted(unknown)}")
    if len(need_solid) > n_solid:
        raise ValueError(f"{len(need_solid)} skills are forced Solid but only {n_solid} Solid slots exist")
    if n_solid + n_thin > len(skills.SKILLS):
        raise ValueError(f"{n_solid} Solid + {n_thin} Thin skills do not fit among {len(skills.SKILLS)} skills")
    rest = [s for s in skills.SKILLS if s not in need_solid]
    order = [rest[int(i)] for i in rng.permutation(len(rest))]
    free_solid = n_solid - len(need_solid)
    solid = set(need_solid) | set(order[:free_solid])
    thin = set(order[free_solid:free_solid + n_thin])
    return {s: "solid" if s in solid else "thin" if s in thin else "unseen" for s in skills.SKILLS}


def status_truth(count: int, accuracy: float, chance: float) -> str:
    """"solid", "thin" or "grey" from the MEASURED count and held-out accuracy (draft rule in THRESHOLDS). Count wins first: 100 examples or fewer is thin however well it
    scored, because that is little evidence. Grey is the zone in between, reported but left out of gated scores."""
    if count <= 0:
        raise ValueError("an untaught skill has no status truth other than 'unseen'; count must be positive")
    if not (0.0 <= accuracy <= 1.0 and 0.0 <= chance <= 1.0):
        raise ValueError(f"accuracy {accuracy} and chance {chance} must lie in [0, 1]")
    t = THRESHOLDS
    if count <= t["thin_max_count"]:
        return "thin"
    if count >= t["solid_min_count"] and accuracy >= t["solid_min_acc"] - _EPS:
        return "solid"
    if chance + t["thin_acc_margin"] - _EPS <= accuracy <= t["thin_max_acc"] + _EPS:
        return "thin"
    return "grey"


# ------------------------------------------------------------------------------------------------------------------------------------ examples
@dataclass(frozen=True)
class Example:
    text: str
    skill: str
    answer: int
    a: int
    b: int
    family: str
    template_ids: Tuple[str, ...]            # (frame id, request wording id)


def text_keys(examples) -> FrozenSet[str]:
    """Fingerprints of the texts, ignoring case and spacing, to prove that no text is shared between teaching and test."""
    return frozenset(hashlib.sha1(" ".join(e.text.lower().split()).encode("utf-8", "surrogatepass")).hexdigest() for e in examples)


def _draw(rng: np.random.Generator, pool: str, skill: str, n: int, families: Tuple[str, ...], held_out: bool, digits: Tuple[int, int], exclude: FrozenSet[str]) -> List[Example]:
    if skill not in skills.SKILLS:
        raise ValueError(f"unknown skill {skill!r}")
    fams = [f for f in families if f != "S5" or skill in skills.COMMUTATIVE]
    op = np.array([skills.ARITH_OP_ID[skill]])
    out: List[Example] = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > 500 * n + 1000:
            raise RuntimeError(f"could not draw {n} examples of {skill} (held_out={held_out}); the operand range is too small for that many distinct problems")
        msg = G.make_single(rng, pool, skill, fams[int(rng.integers(0, len(fams)))], digits)
        p = next(x for x in msg.parts if x.kind == "skill")
        if bool(arith.heldout(op, np.array([p.a]), np.array([p.b]))[0]) != held_out:
            continue
        ex = Example(msg.text, skill, p.answer, p.a, p.b, msg.family, msg.template_ids)
        if text_keys([ex]) & exclude:
            continue
        out.append(ex)
    return out


def draw_teaching(rng: np.random.Generator, pool: str, skill: str, n: int, digits: Tuple[int, int] = (1, 2)) -> List[Example]:
    """``n`` teaching examples of ``skill``: surface families S1 to S3 only, and never a problem that is reserved for testing."""
    return _draw(rng, pool, skill, n, TEACH_FAMILIES, False, digits, frozenset())


def draw_test(rng: np.random.Generator, pool: str, skill: str, n: int, taught_keys: FrozenSet[str] = frozenset(), digits: Tuple[int, int] = (1, 2)) -> List[Example]:
    """``n`` test examples of ``skill``: surface families S4 to S8 only, only reserved (never taught) problems, and no text whose fingerprint is in ``taught_keys``."""
    return _draw(rng, pool, skill, n, TEST_FAMILIES, True, digits, taught_keys)


# ------------------------------------------------------------------------------------------------------------------------------------ solver
class Solver:
    """A small network that learns one skill from (a, b, answer) alone: the operands as digit one-hots (as growmoe.arith.encode_problems), one output head per answer digit.
    The defaults were tuned once on dev seeds over 17 settings and are the same for every skill: width 64, weight decay 1e-4 (L2), 1,000 epochs. Weight decay is what makes it
    generalise to problems it has not seen (0.86 -> 0.93 for add at 2,000 examples); 1e-3 and above stops it learning."""

    def __init__(self, seed: int = 0, hidden: int = 64, epochs: int = 1000, lr: float = 2e-3, batch: int = 128, weight_decay: float = 1e-4, device: str = "auto") -> None:
        self.seed, self.hidden, self.epochs, self.lr, self.batch, self.weight_decay = seed, hidden, epochs, lr, batch, weight_decay
        self.device = device
        self.net = None

    def _dev(self) -> str:
        """"auto" is the GPU when there is one (and KS_DEVICE is not "cpu"); the network is tiny, so this is about keeping the CPU free, not about speed alone."""
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() and os.environ.get("KS_DEVICE") != "cpu" else "cpu"

    @staticmethod
    def _features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
        return arith.encode_problems(np.zeros(len(a), dtype=np.int64), a, b, arith.n_digits(a), arith.n_digits(b))[:, :arith.D_PROB - arith.N_OPS]

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> "Solver":
        import torch
        dev = self._dev()
        x = torch.from_numpy(self._features(a, b)).to(dev)
        t = torch.from_numpy(arith.target_symbols(np.asarray(y, dtype=np.int64))).to(dev)
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], self.hidden), torch.nn.ReLU(), torch.nn.Linear(self.hidden, self.hidden), torch.nn.ReLU(),
                                      torch.nn.Linear(self.hidden, arith.N_POS * arith.N_CLASS)).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            gen = torch.Generator().manual_seed(self.seed)
            for _ in range(self.epochs):
                order = torch.randperm(len(x), generator=gen).to(dev)       # the shuffle is drawn on the CPU so a seed means the same order on every device
                for s in range(0, len(x), self.batch):
                    idx = order[s:s + self.batch]
                    logits = net(x[idx]).view(-1, arith.N_POS, arith.N_CLASS)
                    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, arith.N_CLASS), t[idx].reshape(-1))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        self.net = net.eval()
        return self

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Integer answers. A digit slot that holds PAD counts as 0; the sign slot makes the number negative."""
        import torch
        if self.net is None:
            raise RuntimeError("fit the solver before predicting")
        with torch.no_grad():
            x = torch.from_numpy(self._features(a, b)).to(next(self.net.parameters()).device)
            sym = self.net(x).view(-1, arith.N_POS, arith.N_CLASS).argmax(-1).cpu().numpy()
        digits = np.where(sym[:, :10] < 10, sym[:, :10], 0).astype(np.int64)
        value = (digits * (10 ** np.arange(10, dtype=np.int64))).sum(axis=1)
        return np.where(sym[:, arith.SIGN_POS] == arith.NEG, -value, value)


# ------------------------------------------------------------------------------------------------------------------------------------ competence
def competence_table(assignment: Mapping[str, str], counts: Mapping[str, int], rng: np.random.Generator, pool: str, n_test: int = 500,
                     digits: Tuple[int, int] = (1, 2)) -> Dict[str, dict]:
    """For every TAUGHT skill: teach it ``counts[state]`` examples, train its solver, and score the solver on ``n_test`` problems that were never taught.
    ``chance`` is the accuracy of always giving the most common taught answer. Unseen skills are not measured."""
    lens = tuple(range(digits[0], digits[1] + 1))
    table: Dict[str, dict] = {}
    for skill, state in assignment.items():
        if state == "unseen":
            continue
        op = skills.ARITH_OP_ID[skill]
        teach = draw_teaching(rng, pool, skill, int(counts[state]), digits)
        a, b, y = (np.array([getattr(e, k) for e in teach], dtype=np.int64) for k in ("a", "b", "answer"))
        solver = Solver(seed=int(rng.integers(0, 2 ** 31 - 1))).fit(a, b, y)
        held = arith.sample_problems(rng, n_test, (op,), lens, lens, "test")
        truth = arith.answers(held["op"], held["a"], held["b"])
        values, freq = np.unique(y, return_counts=True)
        accuracy = float((solver.predict(held["a"], held["b"]) == truth).mean())
        chance = float((truth == values[int(freq.argmax())]).mean())
        table[skill] = {"state": state, "examples": len(teach), "accuracy": accuracy, "chance": chance, "status": status_truth(len(teach), accuracy, chance)}
    return table
