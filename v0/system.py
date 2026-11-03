"""System builder and description for V0 walking skeleton."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Suppress known deprecation/future warnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set offline flags before importing any transformer / hub libraries
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from m1.arms import Arm, build_pool_candidate
from m1.beds import Bed
from m1.queue import load_tuning_map

B3_BEDS = ("B3seq", "B3mix", "B3drift")


class V0Part(dict):
    """Component descriptor that formats cleanly as text."""

    def __str__(self) -> str:
        return f"{self['name']} ({self['source']}): {self['state']}"


def describe_v0() -> List[V0Part]:
    """Return the plain list of parts (encoder, pool, head, flag, what is off) for V0."""
    return [
        V0Part(
            part="encoder",
            name="Frozen encoder",
            source="MiniLM through unimoe/frontends.py",
            state="cached on disk under results/unimoe/emb, offline (d=128 projected)",
        ),
        V0Part(
            part="pool",
            name="Pool with spawn, extend and merge",
            source="growmoe/pools/phylo.py inside unimoe/pools.py UniPool",
            state="tuned operating point of M1 (A1q: inner_steps=50, q=0.05)",
        ),
        V0Part(
            part="head",
            name="Closed-form head in every expert",
            source="ULDA of unimoe/systems2.py",
            state="running LDA head inside every expert",
        ),
        V0Part(
            part="flag",
            name="Unknown flag",
            source="density and conformal flag of the pool, fingerprint shortlist",
            state="weak on topical text, reported with measured AUROC",
        ),
        V0Part(
            part="what_is_off",
            name="What is off",
            source="asking reader, growth after birth, notebook newborns, skip and chance flags",
            state="switched off in V0",
        ),
    ]


def build_v0(
    bed: Bed,
    tuning_dir: Union[str, Path, Dict[str, Any]] = "results/m1/tuning",
    **kwargs: Any,
) -> Arm:
    """Build the V0 arm adapter for the 7-domain real-text beds (B3seq, B3mix, B3drift).

    V0 on real text is exactly the M1 cell H_ULDA: UniPool with ULDA switches built
    at the tuned A1q operating point (inner_steps, q) from the bed's tuning map.

    Args:
        bed: Continual learning bed (must be B3seq, B3mix, or B3drift).
        tuning_dir: Directory or file holding tuning configurations, or tuning dict directly.
        **kwargs: Overrides passed to build_pool_candidate.

    Returns:
        Arm adapter with arm_id="V0".

    Raises:
        ValueError: If bed is not a B3 bed.
    """
    if bed.name not in B3_BEDS and not bed.name.startswith("B3"):
        raise ValueError(
            f"build_v0 is supported on B3 beds only ({', '.join(B3_BEDS)}), got {bed.name!r}. "
            "UniPool candidates are B3 only."
        )

    if isinstance(tuning_dir, (str, Path)):
        tuning_map = load_tuning_map(str(tuning_dir), bed.name)
        if not tuning_map or "A1q" not in tuning_map:
            raise ValueError(
                f"Tuning file for bed {bed.name!r} in {tuning_dir} has no 'A1q' entry; "
                "refusing to fall back to default operating point."
            )
        arm_tuning = tuning_map["A1q"]
    elif isinstance(tuning_dir, dict):
        if "A1q" in tuning_dir:
            arm_tuning = tuning_dir["A1q"]
        elif "inner_steps" in tuning_dir and "q" in tuning_dir:
            arm_tuning = tuning_dir
        else:
            raise ValueError(
                f"Tuning dictionary for bed {bed.name!r} has no 'A1q' entry; "
                "refusing to fall back to default operating point."
            )
    else:
        raise ValueError(
            f"Expected tuning_dir as Path, str or dict, got {type(tuning_dir)}: missing 'A1q' operating point."
        )

    if not isinstance(arm_tuning, dict) or "inner_steps" not in arm_tuning or "q" not in arm_tuning:
        raise ValueError(
            f"Tuning map for bed {bed.name!r} has invalid 'A1q' entry (must contain 'inner_steps' and 'q'): {arm_tuning!r}"
        )

    steps = kwargs.pop("inner_steps", arm_tuning["inner_steps"])
    q = kwargs.pop("q", arm_tuning["q"])

    return build_pool_candidate(
        cand_name="ULDA",
        bed=bed,
        inner_steps=steps,
        q=q,
        arm_id="V0",
        **kwargs,
    )
