"""Template pools for the test messages (spec section 5.2).

A message is a FRAME (tone, punctuation, chatter slots; it knows nothing about arithmetic) wrapped around a CORE (the request itself, which carries the operator cue).
Frames are shared by every skill, so a frame id alone must not predict the skill (anti-goal A3); the skill lives in the core.

There are two disjoint pools: ``dev`` (design, tuning, bars) and ``confirm`` (the single-use confirmatory run). The confirmatory pool cannot be loaded unless the
environment variable KS_ALLOW_CONFIRM is "1", so dev code cannot touch it by accident.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

# Words that signal an arithmetic request. Chatter and Other content must contain none of them (whole-word match) and no digits.
ARITH_CUES = frozenset({
    "plus", "minus", "times", "divided", "divide", "mod", "modulo", "remainder", "add", "added", "subtract", "subtracted", "multiply", "multiplied",
    "sum", "total", "product", "quotient", "max", "maximum", "larger", "greater", "bigger", "difference", "less", "half", "double", "twice",
})
ARITH_SYMBOLS = frozenset("+*/%=")


@dataclass(frozen=True)
class Core:
    core_id: str
    skill: str
    text: str                  # format string with {a} {b} (digits), {aw} {bw} (number words), {things}
    swap: bool = False         # True: the text shows b before a (commutative skills only)


@dataclass(frozen=True)
class Frame:
    frame_id: str
    text: str                  # format string with {core}, and optionally {pre} and {post} (chatter)
    noise: Optional[str] = None   # "typo" | "case" | "space" | "mix" | None, applied to the core text


@dataclass(frozen=True)
class Pool:
    name: str
    frames: Dict[str, Tuple[Frame, ...]]           # family S1..S8 -> frames
    cores: Dict[str, Tuple[Core, ...]]             # families S1..S5 -> cores
    core_sources: Dict[str, Tuple[str, ...]]       # families S6..S8 -> the families whose cores they wrap
    chatter: Tuple[str, ...]
    things: Tuple[str, ...]
    traps: Tuple[str, ...] = ()                    # sentences that use arithmetic words or numbers for something else; slots {n}


@lru_cache(maxsize=None)
def load_pool(name: str) -> Pool:
    if name == "dev":
        from pool_dev import POOL
        return POOL
    if name == "confirm":
        if os.environ.get("KS_ALLOW_CONFIRM") != "1":
            raise PermissionError("the confirmatory pool is single-use: set KS_ALLOW_CONFIRM=1 only for the pre-registered confirmatory run")
        from pool_confirm import POOL
        return POOL
    raise KeyError(f"unknown pool {name!r}")
