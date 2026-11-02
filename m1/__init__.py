"""m1: Continual learning benchmark runner for M1 efficiency scoreboard."""

from __future__ import annotations

from m1.beds import Bed, build_bed
from m1.arms import Arm, build_arm, pool_budget_bytes
from m1.cell import (
    cell_filename,
    execute_cell,
    is_valid_cell_record,
    load_cell_record,
    read_pool_budget,
    record_matches_config,
    save_cell_record,
)

__all__ = [
    "Arm",
    "Bed",
    "build_arm",
    "build_bed",
    "cell_filename",
    "execute_cell",
    "is_valid_cell_record",
    "load_cell_record",
    "pool_budget_bytes",
    "read_pool_budget",
    "record_matches_config",
    "save_cell_record",
]

__version__ = "0.2.0"
