"""Amendments A-15 (persistence) and A-16 (group veto level): edits to growmoe/config.py. Each entry: (name, old, new). `old` must occur exactly once."""

FILE = "growmoe/config.py"

EDITS = [
    ("config-confirm-field",
     '''    set_gate_breaker_pause: int = 100    # ... for this many steps
''',
     '''    set_gate_breaker_pause: int = 100    # ... for this many steps
    set_gate_confirm: int = 2            # persistence (amendment A-15): a "new as a set" verdict is acted on only when this many decided batches in a row are doubtful;
    #                              the earlier doubtful batches are HELD (their accepted and flagged samples wait, nothing is trained or sent to probation). 1 = act at once.
'''),

    ("config-veto-field",
     '''    group_veto: bool = True      # an input whose exact p-value under the group's expert is below q keeps its own choice
''',
     '''    group_veto: bool = True      # an input whose exact p-value under the group's expert is below group_veto_q keeps its own choice
    group_veto_q: float = 0.2    # amendment A-16: the level of that veto, separate from q (dev seeds: 0.2 balances the three group statements, margins about 3 points each)
'''),

    ("config-confirm-validation",
     '''            raise ValueError("set_gate_breaker_rate must lie in [0, 1)")
''',
     '''            raise ValueError("set_gate_breaker_rate must lie in [0, 1)")
        if not 0.0 < self.group_veto_q <= 1.0:
            raise ValueError(f"group_veto_q must lie in (0, 1], got {self.group_veto_q}")
        if self.set_gate and self.set_gate_confirm > 1 and self.no_probation:
            raise ValueError("set_gate_confirm > 1 holds doubtful samples for the probation buffer, so it cannot be combined with no_probation")
'''),

    ("config-confirm-positive",
     '''"set_gate_breaker_pause", "group_min"):''',
     '''"set_gate_breaker_pause", "group_min", "set_gate_confirm"):'''),
]
