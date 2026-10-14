"""Edits to growmoe/pools/phylo.py (stage 3). Each entry: (name, old, new). `old` must occur exactly once."""

FILE = "growmoe/pools/phylo.py"

EDITS = [
    ("phylo-imports",
     '''from ..fingerprint import Fingerprint
''',
     '''from ..fingerprint import Fingerprint
from ..groupvote import vote_groups
from ..setgate import SetGate
from ..settest import generator
from ..sources import SourceWindows
'''),

    ("phylo-init",
     '''        self.samples_seen = 0
        self.sig_cls = getattr(self.cfg, "sig_cls", None) or VmfSignature
        self.reset_gate_stats()
''',
     '''        self.samples_seen = 0
        self.sig_cls = getattr(self.cfg, "sig_cls", None) or VmfSignature
        self.reset_gate_stats()
        self._set = SetGate(self.cfg)                                  # routing v2 stage 3a; idle unless cfg.set_gate
'''),

    ("phylo-observe-gate",
     '''        if self.cfg.batch_vote:                                    # routing v2 stage 1; reads only `assigned`, so it holds in every gate mode
            assigned = self._batch_consistent(assigned)

        spawned = []

        # ------------------------------------------------ 13c Probation Buffer
        novel_indices = torch.where(novel)[0]
''',
     '''        if self.cfg.batch_vote:                                    # routing v2 stage 1; reads only `assigned`, so it holds in every gate mode
            assigned = self._batch_consistent(assigned)

        set_verdict, probation_mask = None, novel                  # routing v2 stage 3a (OFF by default): the accepted samples are tested as a SET
        if self.cfg.set_gate:
            assigned, set_novel, set_verdict = self._set_gate_step(x, y_t, assigned, eids)
            if set_novel is not None:
                probation_mask = novel | set_novel                 # called new as a set: they join the individually flagged ones; `novel` itself stays the per-sample gate's

        spawned = []

        # ------------------------------------------------ 13c Probation Buffer
        novel_indices = torch.where(probation_mask)[0]
'''),

    ("phylo-observe-windows",
     '''            rec.meta["sig"].update(xk)

            if y_t is not None:
                yk = y_t[mask_k]
''',
     '''            rec.meta["sig"].update(xk)
            if self.cfg.set_gate:                                  # routing v2 stage 3a: remember which source these samples came from
                self._update_windows(rec, xk)

            if y_t is not None:
                yk = y_t[mask_k]
'''),

    ("phylo-observe-return",
     '''            loss=mean_loss,
            spawned=spawned,
            n_experts=len(self.experts),
        )
''',
     '''            loss=mean_loss,
            spawned=spawned,
            n_experts=len(self.experts),
            meta={"set_verdict": set_verdict} if self.cfg.set_gate else {},
        )
'''),

    ("phylo-bootstrap-wait-meta",
     '''                    spawned=[],
                    n_experts=0,
                )
''',
     '''                    spawned=[],
                    n_experts=0,
                    meta={"set_verdict": None} if self.cfg.set_gate else {},
                )
'''),

    ("phylo-bootstrap-found-meta",
     '''                spawned=[rec.id],
                n_experts=len(self.experts),
            )
''',
     '''                spawned=[rec.id],
                n_experts=len(self.experts),
                meta={"set_verdict": None} if self.cfg.set_gate else {},
            )
'''),

    ("phylo-merge-windows",
     '''                        rec_i.meta["sig"].fit()
                        rec_i.replay.add(rx_j, ry_j)
''',
     '''                        rec_i.meta["sig"].fit()
                        rec_i.replay.add(rx_j, ry_j)
                        if rec_j.meta.get("windows") is not None:          # routing v2 stage 3a: the merged expert remembers the sources of both
                            self._windows_of(rec_i).merge_from(rec_j.meta["windows"])
'''),

    ("phylo-methods",
     '''    _FAST_RULES = ("accept_then_logf", "argmax_logf", "argmax_p", "zscore_logf", "nearest_proto")
''',
     '''    # ------------------------------------------------------------------------------------------ routing v2 stage 3a: the batch-level gate
    @property
    def set_stats(self) -> dict:
        """Counters of the batch-level gate (tests, early stops, permutations, verdicts, overrides, pauses, breaker openings, window splits). No wall-clock."""
        return self._set.stats

    def summary(self) -> dict:
        out = super().summary()
        if self.cfg.set_gate:
            out["set_stats"] = dict(self._set.stats)
        return out

    def _windows_of(self, rec) -> SourceWindows:
        """The expert's source windows, created on first use (so a pool that never enables the set gate never holds any)."""
        w = rec.meta.get("windows")
        if w is None:
            c = self.cfg
            w = rec.meta["windows"] = SourceWindows(
                max_windows=c.set_gate_windows, max_batches=c.set_gate_window_batches, min_ref=c.set_gate_min_ref, split_alpha=c.set_gate_split_alpha,
                n_perm=c.set_gate_perms, chunk=c.set_gate_chunk, stat=c.set_gate_stat, max_rows=c.set_gate_max_rows)
        return w

    def _update_windows(self, rec, xk: Tensor) -> None:
        """Add the samples a trained batch sent to this expert to its source windows."""
        if self._windows_of(rec).update(xk, generator(self.cfg.seed, 0x57494E, self.step, rec.id), self.step):    # 0x57494E = "WIN"
            self._set.stats["window_splits"] += 1

    def _set_refs(self, eid: int):
        """The expert's references for the set test as (rows, mean) pairs: one per source window with enough rows, else its stored prototypes (a young
        expert's prototypes are its training samples). An expert with neither has no reference and can never be admitted."""
        rec = self.experts[eid]
        w = rec.meta.get("windows")
        pairs = w.ref_pairs() if w is not None else []
        if pairs:
            return pairs
        protos = rec.meta["sig"].protos
        return [(protos, protos.mean(0))] if protos is not None and len(protos) > 0 else []

    def _set_loss(self, eid: int, x: Tensor, y: Tensor) -> float:
        with torch.no_grad():
            return float(F.cross_entropy(self.experts[eid].module(x), y))

    def _set_gate_step(self, x: Tensor, y_t: Optional[Tensor], assigned: Tensor, eids: List[int]):
        """Routing v2 stage 3a. The samples the per-sample gate accepted (and the batch vote assigned) are tested as a SET against the experts' source
        windows. Returns ``(assigned, set_novel_mask or None, verdict or None)``: novel as a set -> the accepted samples are unassigned and the mask tells the
        caller to send them to the probation buffer; known -> one expert trains all of them. Samples the per-sample gate flagged are never touched."""
        gate, cfg = self._set, self.cfg
        accepted = assigned >= 0
        n_acc = int(accepted.sum())
        if n_acc < cfg.batch_vote_min:
            return assigned, None, None
        if gate.paused(self.step):
            gate.stats["paused"] += 1
            return assigned, None, "paused"
        ids, counts = torch.unique(assigned[accepted], return_counts=True)
        vote_owner = int(ids[counts == counts.max()].min())
        dec = gate.decide(x[accepted], y_t[accepted] if y_t is not None else None, eids, self._set_refs, self._set_loss, self.step)
        out = assigned.clone()
        if dec.verdict == "novel":
            out[accepted] = -1
            self.events.add(kind="set_novel", step=self.step, n=n_acc, tried=dec.tried)
        else:
            out[accepted] = dec.owner
            if dec.owner != vote_owner:
                gate.stats["override"] += 1
                self.events.add(kind="set_override", step=self.step, n=n_acc, vote=vote_owner, owner=dec.owner, admitted=dec.admitted)
        if gate.note(dec.verdict, n_acc / assigned.shape[0], self.step):
            self.events.add(kind="breaker_open", step=self.step, until=gate.pause_until)
        return out, (accepted if dec.verdict == "novel" else None), dec.verdict

    # ------------------------------------------------------------------------------------------ routing v2 stage 3b: group answering
    def predict_grouped(self, x: Tensor, groups: Tensor, ctx: Optional[Tensor] = None) -> Prediction:
        """Answer ``x`` with the inputs of every group answered together. ``groups`` is a LongTensor of OPAQUE group ids chosen by the caller (one document, one
        session, one batch; negative = not grouped); they carry no regime or label information. With ``cfg.group_vote`` every group of at least ``cfg.group_min``
        inputs is answered by the expert most of its inputs chose on their own (routing v2 stage 3b); with the flag off, or an empty pool, the answer is
        ``predict(x, ctx)`` itself. ``predict`` is not changed: its signature is pinned by tests/growmoe/test_gm_phylo.py (no side channel into the pool)."""
        pred = self.predict(x, ctx)
        if not self.cfg.group_vote or len(self.experts) == 0:
            return pred
        return self._answer_groups(x, pred, groups)

    def _answer_groups(self, x: Tensor, pred: Prediction, groups: Tensor) -> Prediction:
        """Re-answer the inputs of every large enough group with the group's expert (see ``growmoe.groupvote``). The veto uses EXACT p-values from
        ``_gate_column``, never the cascade's restricted matrix (which holds 0 for experts outside an input's shortlist)."""
        cfg = self.cfg
        groups = torch.as_tensor(groups)
        xn = self._normalize(x.float())

        def accepts(eid: int, idx: Tensor) -> Tensor:
            p, logf = self._gate_column(eid, xn[idx])
            return (logf >= 0.0) if cfg.no_conformal else (p >= cfg.q)

        new_expert, info = vote_groups(pred.expert, groups, cfg.group_min, accepts if cfg.group_veto else None)
        moved = new_expert != pred.expert
        probs = pred.probs
        if moved.any():
            probs = pred.probs.clone()
            with torch.no_grad():
                for eid in torch.unique(new_expert[moved]).tolist():
                    m = moved & (new_expert == eid)
                    probs[m] = self.experts[eid].module.proba(xn[m])
        return Prediction(probs=probs, expert=new_expert, p_nov=pred.p_nov, unknown=pred.unknown, meta={**pred.meta, "group_vote": info})

    _FAST_RULES = ("accept_then_logf", "argmax_logf", "argmax_p", "zscore_logf", "nearest_proto")
'''),
]
