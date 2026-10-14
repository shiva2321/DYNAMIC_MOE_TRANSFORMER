"""Edits to growmoe/config.py (stage 3). Each entry: (name, old, new). `old` must occur exactly once."""

FILE = "growmoe/config.py"

EDITS = [
    ("config-fields",
     '''    gate_shortlist_min_experts: int = 64 # below this many experts the exact scan is used
''',
     '''    gate_shortlist_min_experts: int = 64 # below this many experts the exact scan is used
    set_gate: bool = False       # routing v2 stage 3a: a batch-level test decides whether the accepted samples of a batch are known or new as a SET.
    #                              OFF until its pre-registered run (RV3-5); docs/superpowers/specs/2026-10-13-routing-v2-stage3-design.md.
    set_gate_alpha: float = 0.001        # level of the admission test: an expert admits a batch when the exact permutation p-value exceeds this
    set_gate_perms: int = 999            # permutations per test; the smallest attainable p-value 1/(1+perms) must not exceed the levels
    set_gate_chunk: int = 16             # permutations drawn per early-stop step
    set_gate_candidates: int = 3         # experts tried per batch (those whose nearest reference mean is nearest the batch mean)
    set_gate_stat: str = "meandist"      # "meandist" (distance of means) | "mmd" (squared MMD, RBF kernel, median bandwidth)
    set_gate_ownership: str = "loss"     # who trains an admitted batch: "loss" (lowest label cross-entropy among the admitted) | "nearest"
    set_gate_min_labels: int = 8         # labelled accepted samples needed for loss ownership (fewer: the nearest admitted expert)
    set_gate_min_ref: int = 48           # rows a source window needs before it counts as a reference
    set_gate_windows: int = 4            # source windows kept per expert
    set_gate_window_batches: int = 5     # batches per source window
    set_gate_split_alpha: float = 0.01   # level of the test that opens a new source window
    set_gate_max_rows: int = 512         # rows above which a reference, a stored batch or a tested batch is cut to a random subset (bounds memory and time)
    set_gate_breaker_len: int = 12       # circuit breaker: among the last this-many familiar-looking batches ...
    set_gate_breaker_rate: float = 0.35  # ... more than this share called new as a set pauses the set level ...
    set_gate_breaker_pause: int = 100    # ... for this many steps
    group_vote: bool = False     # routing v2 stage 3b: predict(x, groups=...) answers a group by the majority of its inputs' own choices (OFF until RV3-5)
    group_min: int = 8           # smallest group that is answered together
    group_veto: bool = True      # an input whose exact p-value under the group's expert is below q keeps its own choice
'''),
    ("config-validate",
     '''    route_rule: str = "default"  # test prediction routing rule: "default" | "accept_then_logf" (highest log-density among p >= q; skips supersede filter at test time) | "argmax_logf" | "nearest_proto"
''',
     '''    route_rule: str = "default"  # test prediction routing rule: "default" | "accept_then_logf" (highest log-density among p >= q; skips supersede filter at test time) | "argmax_logf" | "nearest_proto"

    def __post_init__(self) -> None:
        """Refuse stage-3 settings that would silently disable or break the test (a level the permutation count can never reach, an unknown name)."""
        if self.set_gate_stat not in ("meandist", "mmd"):
            raise ValueError(f"set_gate_stat must be 'meandist' or 'mmd', got {self.set_gate_stat!r}")
        if self.set_gate_ownership not in ("loss", "nearest"):
            raise ValueError(f"set_gate_ownership must be 'loss' or 'nearest', got {self.set_gate_ownership!r}")
        for name in ("set_gate_alpha", "set_gate_split_alpha"):
            level = getattr(self, name)
            if not 0.0 < level < 1.0:
                raise ValueError(f"{name} must lie in (0, 1), got {level}")
            if math.floor(level * (1 + self.set_gate_perms) + 1e-9) < 1:                    # the rule of growmoe.settest.max_exceedances
                raise ValueError(f"set_gate_perms={self.set_gate_perms} is too small for {name}={level}: the smallest attainable p-value 1/(1+perms) exceeds the level, so the test could never reject")
        if not 0.0 <= self.set_gate_breaker_rate < 1.0:
            raise ValueError("set_gate_breaker_rate must lie in [0, 1)")
        for name in ("set_gate_perms", "set_gate_chunk", "set_gate_candidates", "set_gate_min_labels", "set_gate_min_ref", "set_gate_windows",
                     "set_gate_window_batches", "set_gate_max_rows", "set_gate_breaker_len", "set_gate_breaker_pause", "group_min"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
'''),
    ("config-import",
     '''from dataclasses import dataclass
''',
     '''import math
from dataclasses import dataclass
'''),
]
