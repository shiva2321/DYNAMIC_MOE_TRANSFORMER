"""M1 frozen configuration: seeds, grids, candidate lists, held-default pairs and thresholds.

All frozen constants for the M1 efficiency benchmark are centralized in this module.
No other module should define these constants.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# -----------------------------------------------------------------------------
# Arm Lists & Ordering (Section 4.1, 5.7)
# -----------------------------------------------------------------------------
ALL_ARMS: Tuple[str, ...] = (
    "A1",
    "A1q",
    "A2",
    "A3",
    "A4",
    "A5a",
    "A5b",
    "A6",
    "A7",
    "A7all",
    "A8",
    "A3-2",
    "A4-2",
    "A7-2",
)

# -----------------------------------------------------------------------------
# Seeds (M1 Preregistration Section 5)
# -----------------------------------------------------------------------------
DEV_SEEDS: Tuple[int, ...] = (800, 801, 802, 803, 804)
REPORT_SEEDS: Tuple[int, ...] = tuple(range(810, 820))
RESERVE_SEEDS: Tuple[int, ...] = tuple(range(820, 830))

# -----------------------------------------------------------------------------
# Environment & Pinning (Section 4.4, D-84)
# -----------------------------------------------------------------------------
PINNING_THREADS: int = 2
PINNING_CPUS: Tuple[int, ...] = (0, 2)
BATCH_SIZE: int = 64
POOL_SEED_OFFSET: int = 7

# -----------------------------------------------------------------------------
# Tuning Grids (Section 4.1, 4.3)
# -----------------------------------------------------------------------------
GRID_A1_STEPS: Tuple[int, ...] = (1, 5, 20, 50)

GRID_A1Q_STEPS: Tuple[int, ...] = (1, 5, 20, 50)
GRID_A1Q_Q: Tuple[float, ...] = (0.02, 0.05)

GRID_A2_Q: Tuple[float, ...] = (0.02, 0.05)

GRID_A3_WIDTH: Tuple[int, ...] = (256, 512, 1024, 2048, 4096)
GRID_A3_STEPS: Tuple[int, ...] = (5, 20)

GRID_A4_WIDTH: Tuple[int, ...] = (256, 512, 1024, 2048, 4096)

GRID_A5A_SHRINKAGE: Tuple[float, ...] = (1e-3, 1e-2, 1e-1)
GRID_A5B_SHRINKAGE: Tuple[float, ...] = (0.01, 0.1, 0.5)

GRID_A6_D: Tuple[int, ...] = (1024, 4096)
GRID_A6_RIDGE: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)

GRID_A7_K: Tuple[int, ...] = (1, 5, 15)
GRID_A7ALL_K: Tuple[int, ...] = (1, 5, 15)

GRID_A8_WIDTH: Tuple[int, ...] = (1024, 4096)
GRID_A8_STEPS: Tuple[int, ...] = (5, 20)

# Smoke Grids (Single-value grids for fast testing and verification)
SMOKE_GRID_A1_STEPS: Tuple[int, ...] = (5,)
SMOKE_GRID_A1Q_STEPS: Tuple[int, ...] = (5,)
SMOKE_GRID_A1Q_Q: Tuple[float, ...] = (0.05,)
SMOKE_GRID_A2_Q: Tuple[float, ...] = (0.05,)
SMOKE_GRID_A3_WIDTH: Tuple[int, ...] = (256,)
SMOKE_GRID_A3_STEPS: Tuple[int, ...] = (5,)
SMOKE_GRID_A4_WIDTH: Tuple[int, ...] = (256,)
SMOKE_GRID_A5A_SHRINKAGE: Tuple[float, ...] = (1e-2,)
SMOKE_GRID_A5B_SHRINKAGE: Tuple[float, ...] = (0.1,)
SMOKE_GRID_A6_D: Tuple[int, ...] = (1024,)
SMOKE_GRID_A6_RIDGE: Tuple[float, ...] = (0.1,)
SMOKE_GRID_A7_K: Tuple[int, ...] = (5,)
SMOKE_GRID_A7ALL_K: Tuple[int, ...] = (5,)
SMOKE_GRID_A8_WIDTH: Tuple[int, ...] = (1024,)
SMOKE_GRID_A8_STEPS: Tuple[int, ...] = (5,)

# -----------------------------------------------------------------------------
# Selection Rule Constants (Section 4.3)
# -----------------------------------------------------------------------------
TOLERANCE_TIE_ACCURACY: float = 0.001

EDGE_EXTEND_FACTOR_STEPS: float = 2.5
EDGE_EXTEND_CAP_STEPS: int = 200

EDGE_EXTEND_FACTOR_WIDTH: int = 2
EDGE_EXTEND_CAP_WIDTH: int = 8192

EDGE_EXTEND_FACTOR_D: int = 2
EDGE_EXTEND_CAP_D: int = 8192

# -----------------------------------------------------------------------------
# A2 Candidate Configurations (Section 4.4)
# -----------------------------------------------------------------------------
U_REAL_SWITCHES: Dict[str, Any] = {
    "set_gate": True,
    "set_gate_confirm": 2,
    "sig_dedupe": True,
    "infant_relax": 1.0,
    "growth_rule": "gap",
}

U_DEEP_SWITCHES: Dict[str, Any] = {
    "set_gate": True,
    "set_gate_confirm": 2,
    "sig_dedupe": True,
    "infant_relax": 1.0,
    "growth_rule": "probe",
}

TWIN_FIX: Dict[str, Any] = {"sig_dedupe": True, "infant_relax": 1.0}      # repair of the founding-cluster flaw (F-31, E-037); off by default; removes the chain reaction of E-070

A2_CANDIDATES: Dict[str, Dict[str, Any]] = {
    "shipped": {},
    "shipped+twin": dict(TWIN_FIX),                                     # added by the reviewer after E-070 (spec 4.4 amendment, D-87)
    "shipped+gap": {"growth_rule": "gap"},
    "shipped+skip": {"skip_known": 0.1},
    "shipped+chance": {"transfer_chance": 1.0},
    "U-real": dict(U_REAL_SWITCHES),
    "U-deep": dict(U_DEEP_SWITCHES),
    "U-real+skip": dict(U_REAL_SWITCHES, skip_known=0.1),
    "U-real+chance": dict(U_REAL_SWITCHES, transfer_chance=1.0),
    "U-real+chance-weak": dict(U_REAL_SWITCHES, transfer_chance=1.0, transfer_chance_weak_only=True),
}

SMOKE_A2_CANDIDATES: Dict[str, Dict[str, Any]] = {"shipped": {}}

# On B3 (Stage B-2), UniPool candidates:
B3_UNIPOOL_CANDIDATES: Tuple[str, ...] = ("UG1", "ULDA", "UPLUS")

# -----------------------------------------------------------------------------
# Held-Default Pairs (Section 4.4)
# -----------------------------------------------------------------------------
HELD_DEFAULT_PAIRS: Dict[str, Tuple[str, str]] = {
    "growth": ("shipped", "shipped+gap"),
    "skip": ("shipped", "shipped+skip"),
    "chance": ("shipped", "shipped+chance"),
    "lda_head": ("U-real", "ULDA"),  # B3 only
}

# -----------------------------------------------------------------------------
# Preregistration Thresholds (M1_PREREG_DRAFT.md Sections 6.1 and 6.2)
# -----------------------------------------------------------------------------
# Section 6.1 Decision Rule
THRESHOLD_RULE_6_1_ACC_MARGIN: float = 0.01  # baseline within 1 point of pool accuracy

# Section 6.2 Adoption Rules for Held Defaults
THRESHOLD_LDA_HEAD_PAIRED_GAIN: float = 0.01
THRESHOLD_LDA_HEAD_WALL_CLOCK_MAX_RISE: float = 0.10

THRESHOLD_GAP_GROWTH_MAX_ACC_DROP: float = 0.005
THRESHOLD_GAP_GROWTH_MIN_WALL_CLOCK_DROP: float = 0.15

THRESHOLD_SKIP_KNOWN_MAX_ACC_DROP: float = 0.003
THRESHOLD_SKIP_KNOWN_MIN_OBS_DROP: float = 0.10

THRESHOLD_TRANSFER_CHANCE_MIN_GAIN_ONE_BED: float = 0.01
THRESHOLD_TRANSFER_CHANCE_MAX_COST_B2_B3: float = 0.005

# Section 7 predictions (M1_PREREG_DRAFT.md): the numbers inside the statements M1-1, M1-3, M1-4, M1-6 and M1-8
THRESHOLD_M1_1_LATENCY_RATIO_MIN: float = 1.0       # pool latency per input at least this multiple of the best replay network (no wall-clock gain)
THRESHOLD_M1_1_MACS_RATIO_MAX: float = 0.25         # pool counted multiply-adds per input at most this multiple of the replay network's
THRESHOLD_M1_3_TRAIN_RATIO_MAX: float = 0.5         # pool whole-stream training wall-clock at most this multiple of the best same-memory replay arm
THRESHOLD_M1_4_SHARE_MIN: float = 0.60              # gate, router and bookkeeping share of the pool's observe wall-clock, strictly above
THRESHOLD_M1_6_LDA_GAP_MAX: float = 0.02            # streaming LDA within this distance (either side) of the LDA-head pool
THRESHOLD_M1_8_R_SQUARED_MIN: float = 0.95          # linearity of pool memory against regimes seen
THRESHOLD_M1_8_SLOPE_REL_ERR_MAX: float = 0.25      # slope within this fraction of one expert's bytes
