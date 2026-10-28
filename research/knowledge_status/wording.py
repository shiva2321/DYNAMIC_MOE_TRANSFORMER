"""Mitigations for the wording shift found in Task 11 (a known skill in a story or in noisy text sits farther from the taught examples than the calibration expects, although its nearest skill
is still the right one). A new module on purpose: ``status.py`` and ``run_e2.py`` stay as they were, so the results already on disk stay valid.

* ``clean_text`` / ``CleanRep``: read the text after a generic clean-up (lower case, one space between words). It does not look at any test wording; it removes a difference (case, spacing) that
  carries no meaning.
* ``WordingMachine(lofo=True)``: leave-one-wording-out calibration. The calibration scores of a taught group are computed for each taught wording against prototypes of the OTHER taught wordings
  (same kernel width as the full signature), so the alarm is calibrated on what a change of wording looks like. Only S1 to S3 data are used, as the calibration rule requires.
* ``WordingMachine(margin_q=...)``: a margin gate. A part that every group flags is still called "unseen" only if it is also AMBIGUOUS: the margin between its nearest and its second-nearest group
  is among the lowest ``margin_q`` of the calibration margins. A part with a clear nearest skill goes to that skill (it is a known skill in an unfamiliar wording). The gate only ever removes
  unseen calls, so the false-unseen guarantee on the taught wordings is kept; parts whose nearest group is Other content are never overridden.
Without these options ``WordingMachine`` is exactly ``StatusMachine``.
"""

import math
import re
from typing import Dict, List, Optional

import numpy as np

import reps
import status as ST
from growmoe.signature import VmfSignature

_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return _SPACES.sub(" ", text.lower()).strip()


class CleanRep(reps.Representation):
    """A representation that reads ``clean_text(text)``. Its id and tag differ from the wrapped one, so its cache files do not mix with the plain ones."""

    def __init__(self, base: reps.Representation) -> None:
        super().__init__(base.id + "_clean", base.dim, tag=base.tag + "+clean")
        self.base = base

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.base.encode([clean_text(t) for t in texts])


class PromptedRep(reps.LastTokenRep):
    """The last-token reader with another closing prompt (``suffix`` replaces ``reps.SUFFIX``). The prompt decides what the last-token state is about, so a prompt that tells the model to
    ignore extra words may make the state less sensitive to the story around a request. The prompt must not name the skills (that would give the reader the label set). The cache identity
    includes the prompt through the reader's id."""

    def __init__(self, rep_id: str, model_name: str, layer: int, fp16_gpu: bool, suffix: str) -> None:
        super().__init__(rep_id, model_name, layer, fp16_gpu)
        self.suffix = suffix
        self.tag = f"{self.tag}|{suffix!r}"

    @property
    def suffix_ids(self) -> List[int]:
        return self._model()[0](self.suffix, add_special_tokens=False)["input_ids"]


class WordingMachine(ST.StatusMachine):
    def __init__(self, groups, other, cal_vectors, cal_groups, q: float = 0.05, thresholds=ST.THRESHOLDS, seed: int = 0, max_protos: int = 256, n_perm: int = 999, ref_cap: int = 512, *,
                 proto_families: Optional[Dict[str, np.ndarray]] = None, cal_families: Optional[np.ndarray] = None, lofo: bool = False, margin_q: Optional[float] = None) -> None:
        if margin_q is not None and not 0.0 < margin_q < 1.0:
            raise ValueError(f"margin_q must lie strictly between 0 and 1, got {margin_q}")
        if lofo:
            if proto_families is None:
                raise ValueError("lofo needs proto_families: the wording of every prototype of every taught group")
            for name, g in groups.items():
                if name not in proto_families or len(proto_families[name]) != len(g.protos):
                    raise ValueError(f"proto_families[{name!r}] must have one entry per prototype ({len(g.protos)})")
            if cal_families is None or len(cal_families) != len(cal_groups):
                raise ValueError("cal_families must have one entry per calibration vector")
        super().__init__(groups, other, cal_vectors, cal_groups, q=q, thresholds=thresholds, seed=seed, max_protos=max_protos, n_perm=n_perm, ref_cap=ref_cap)
        self.margin_q, self.lofo = margin_q, lofo
        self.last_overridden = np.zeros(0, dtype=bool)
        cal = np.asarray(cal_vectors, dtype=np.float64)
        cg = np.asarray(cal_groups)
        cal_unit = (cal / np.linalg.norm(cal, axis=1, keepdims=True)).astype(np.float32)
        if lofo:
            self._leave_one_wording_out(groups, cal_unit, cg, np.asarray(cal_families), proto_families, max_protos)
        if margin_q is not None:
            skill_rows = cg != other.name
            margins = self._margin(self._cosmax(cal_unit[skill_rows]))[0]
            self._margins = np.sort(margins)

    # -------------------------------------------------------------------------------------------------------------------------- leave-one-wording-out
    def _leave_one_wording_out(self, groups, cal_unit, cg, cal_fam, proto_fam, max_protos) -> None:
        need = math.ceil(1.0 / self.q - 1e-9) - 1
        for j, name in enumerate(self.names):
            if name not in groups:
                continue                                                           # Other content has no wordings: standard calibration
            fam = np.asarray(proto_fam[name])
            if len(set(fam.tolist())) < 2:
                continue                                                           # one wording only: nothing to leave out
            rows_g = np.flatnonzero(cg == name)
            scores: List[np.ndarray] = []
            for k, f in enumerate(sorted(set(cal_fam[rows_g].tolist()) & set(fam.tolist()))):
                rows = rows_g[cal_fam[rows_g] == f]
                sig = VmfSignature(self.d, max_protos=max_protos, seed=self.seed + 1000 * (j + 1) + k, dedupe=True)
                sig.update(self._P[j][fam != f])
                if sig.n < 2:
                    continue
                sig.fit()
                sig._kappa = self._sig[j].kappa                                    # one scale for the calibration scores and for the parts judged against them
                scores.append(sig.score(cal_unit[rows]).double().numpy())
            pooled = np.concatenate(scores) if scores else np.zeros(0)
            if len(pooled) >= need:
                self._cal[j] = np.sort(pooled)

    # -------------------------------------------------------------------------------------------------------------------------- margin gate
    def _cosmax(self, U32: np.ndarray) -> np.ndarray:
        return np.stack([(U32 @ P.T).max(axis=1) for P in self._P], axis=1)

    @staticmethod
    def _margin(cosmax: np.ndarray):
        order = np.argsort(-cosmax, axis=1)
        top, second = cosmax[np.arange(len(cosmax)), order[:, 0]], cosmax[np.arange(len(cosmax)), order[:, 1]]
        return top - second, order[:, 0]

    def assess_many(self, part_vecs: np.ndarray) -> List[ST.Assessment]:
        out = super().assess_many(part_vecs)
        if self.margin_q is None:
            self.last_overridden = np.zeros(len(out), dtype=bool)
            return out
        U32 = ST._as_rows(part_vecs, self.d).astype(np.float32)
        cosmax = self._cosmax(U32)
        margin, nearest = self._margin(cosmax)
        p_margin = (1.0 + np.searchsorted(self._margins, margin, side="right")) / (len(self._margins) + 1.0)
        skill_nearest = np.array([self.names[g] != self.other_name for g in nearest])
        self.last_overridden = np.array([a.status == "unseen" for a in out]) & (p_margin > self.margin_q) & skill_nearest
        for i in np.flatnonzero(self.last_overridden):
            g = int(nearest[i])
            ev = int(self._evidence(g, U32[i:i + 1])[0])
            out[i] = self._build(g, out[i].p_value, ev, 1, self.names[g])
        return out
