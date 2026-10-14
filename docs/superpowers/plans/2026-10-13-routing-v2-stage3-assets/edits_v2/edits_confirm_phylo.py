"""Amendments A-15 (persistence) and A-16 (group veto level): edits to growmoe/pools/phylo.py. Each entry: (name, old, new). `old` must occur exactly once."""

FILE = "growmoe/pools/phylo.py"

EDITS = [
    ("phylo-confirm-init",
     '''        self._set = SetGate(self.cfg)                                  # routing v2 stage 3a; idle unless cfg.set_gate
''',
     '''        self._set = SetGate(self.cfg)                                  # routing v2 stage 3a; idle unless cfg.set_gate
        self._held: list = []                                          # persistence (A-15): samples of a doubtful batch that wait for the next decision
        self._held_step = -10
'''),

    ("phylo-confirm-probation-divert",
     '''            if len(novel_indices) > 0:
                x_nov = x[novel_indices]
                y_nov = y_t[novel_indices] if y_t is not None else None
                arr_nov = start_arrival + novel_indices
                self.probation.add(x_nov, y_nov, arrival=arr_nov)
''',
     '''            if len(novel_indices) > 0:
                x_nov = x[novel_indices]
                y_nov = y_t[novel_indices] if y_t is not None else None
                arr_nov = start_arrival + novel_indices
                if set_verdict == "held":                              # persistence: the flagged samples of a doubtful batch wait with its accepted samples
                    self._held.append((x_nov, y_nov, arr_nov))
                else:
                    self.probation.add(x_nov, y_nov, arrival=arr_nov)
'''),

    ("phylo-confirm-docstring",
     '''caller to send them to the probation buffer; known -> one expert trains all of them. Samples the per-sample gate flagged are never touched."""''',
     '''caller to send them to the probation buffer; known -> one expert trains all of them. Samples the per-sample gate flagged are never touched.
        Persistence (``set_gate_confirm`` = c > 1, amendment A-15): the first c - 1 doubtful decided batches in a row return "held": their accepted samples are
        unassigned and wait in ``self._held`` (``observe`` adds the flagged samples to the same list instead of the probation buffer); the c-th doubtful batch
        returns "novel" and the held samples join its cluster; a known batch drops them. The circuit breaker always sees the raw verdict."""'''),

    ("phylo-confirm-step-body",
     '''        out = assigned.clone()
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
''',
     '''        verdict = gate.confirm(dec.verdict)                            # persistence (A-15): "known", "held" or "novel" (acted on)
        out = assigned.clone()
        if verdict == "held":
            out[accepted] = -1
            gate.stats["held"] += 1
            self._hold(x, y_t, accepted)
            self.events.add(kind="set_held", step=self.step, n=n_acc, tried=dec.tried)
        elif verdict == "novel":
            self._release_held()
            out[accepted] = -1
            self.events.add(kind="set_novel", step=self.step, n=n_acc, tried=dec.tried)
        else:
            self._drop_held()
            out[accepted] = dec.owner
            if dec.owner != vote_owner:
                gate.stats["override"] += 1
                self.events.add(kind="set_override", step=self.step, n=n_acc, vote=vote_owner, owner=dec.owner, admitted=dec.admitted)
        if gate.note(dec.verdict, n_acc / assigned.shape[0], self.step):
            self.events.add(kind="breaker_open", step=self.step, until=gate.pause_until)
        return out, (accepted if verdict == "novel" else None), verdict
'''),

    ("phylo-confirm-helpers",
     '''    # ------------------------------------------------------------------------------------------ routing v2 stage 3b: group answering
''',
     '''    _HOLD_TTL = 3                                                      # a held batch is only joined by the next decided batch if it comes within this many steps

    def _hold(self, x: Tensor, y_t: Optional[Tensor], accepted: Tensor) -> None:
        """Persistence (A-15): the accepted samples of a doubtful batch wait (the flagged ones follow in ``observe``)."""
        if self.step - self._held_step > self._HOLD_TTL:               # an older, never-resolved hold is stale
            self._held = []
        start = self.samples_seen - x.shape[0]
        self._held.append((x[accepted].clone(), y_t[accepted].clone() if y_t is not None else None, start + torch.where(accepted)[0]))
        self._held_step = self.step

    def _release_held(self) -> None:
        """The next decided batch is doubtful too: the held samples join the probation buffer (and so the cluster of this batch) if they are still fresh."""
        if self._held and self.step - self._held_step <= self._HOLD_TTL:
            for px, py, pa in self._held:
                self.probation.add(px, py, arrival=pa)
            self._set.stats["released"] += 1
        elif self._held:
            self._set.stats["dropped"] += 1
        self._held = []

    def _drop_held(self) -> None:
        """The next decided batch is known: the doubt was a false alarm and the held samples are dropped."""
        if self._held:
            self._set.stats["dropped"] += 1
        self._held = []

    # ------------------------------------------------------------------------------------------ routing v2 stage 3b: group answering
'''),

    ("phylo-veto-level",
     '''            return (logf >= 0.0) if cfg.no_conformal else (p >= cfg.q)
''',
     '''            return (logf >= 0.0) if cfg.no_conformal else (p >= cfg.group_veto_q)
'''),
]
