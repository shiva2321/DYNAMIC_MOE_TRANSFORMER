# Pre-stage-4 pins (2026-10-21)

Since 2026-10-21 (notebook E-043, decision D-38) `PhyloCfg` has two of the three stage-4 repairs on by default: the router (`router`, with `router_mode="free"`) and the transfer
spawn rule (`spawn_rule="transfer"`). The third, growth from the fit gap (`growth_rule="gap"`, `grow_gap=0.17`), passed its pre-registered statements too but is NOT the default: the
independent review (E-042) found that it replaces the shipped widen-and-deepen policy, and the user decides it. The run is RV2-D (E-041, all nine statements PASS).

Everything recorded before that day was produced by the pool without these repairs. So that those numbers can still be reproduced, every script and test that produced or reproduces a
recorded result of an earlier stage pins the old pool.

## What the pin is
- `from growmoe.config import PRE_STAGE4` is `{"router": False, "spawn_rule": "probe", "growth_rule": "probe"}`; `PhyloCfg(**PRE_STAGE4, ...)` is the pool of Phase 1 to stage 4.1. It lists all
  three switches although `growth_rule` is still "probe" by default, so that nothing has to be touched again if the user decides to make growth from the gap the default.
- The router follows the routing rule: it is on whenever `route_rule` is a fast rule (`accept_then_logf` and the others that every measured pool used) and off with the reference path
  `route_rule="default"`, which applies the supersede filter and is router-blind. `router=True` with `route_rule="default"` is refused. A bare `PhyloCfg()` therefore has the transfer
  rule but not the router; set `route_rule="accept_then_logf"` to get both.

## Where it is written
- Scripts: the three keywords are written out at the `PhyloCfg(...)` call (39 calls, 32 files under `research/`), or once in the shared settings that many scripts import:
  `m0/run_m0.py::A_KW` (also used by `m2a`, `rv2a` and the `m0/diagnose_*` scripts), `rv2b/run_rv2b.py::BASE`, `rv2c/run_rv2c.py::BASE`, `rv2c/golden.py::BASE` (already pinned for
  the earlier switches), `growmoe/bench.py::_phase1_cfg` (so `python -m growmoe.bench --phase 1` still reproduces `phase1/`), `arith/arith_lib.py::Pool._new` (keywords passed by the
  caller still win).
- The RV2-D runner: `rv2d/rv2d_lib.py::arm_flags` starts from `PRE_STAGE4`, so the arm `S3` stays "stage 3 without stage 4" and each other arm lists only what it turns on.
- Tests: the tests that describe the old rules (the interference probe, the shadow-trial growth, the stage-3 fingerprints recorded before stage 4, the router-blind reference loop)
  pass `**PRE_STAGE4` or the single keyword they need; `tests/growmoe/test_gm_adopt_stage4.py` and `test_gm_rv4_config.py` pin the new defaults.

## Not pinned, on purpose
- `scratch/`: throwaway scripts of R11-R12 that no notebook entry cites.
- `demos/` and the library: they follow the defaults (the GrowMoE demo was re-run with them, E-043).
- Text of earlier entries and reports: each result was measured with the pool of its time, and says so.

## The confirmatory run is closed
The 704 records of RV2-D (seeds 85-99) carry the code stamp `ff3438232c5f0dfa`. The adoption changes `growmoe/config.py`, so the stamp of today differs and the queue would read
those cells as missing and run them again. `rv2d/CONFIRM_CLOSED.json` stops that: `jobqueue.py` refuses to start the phase `confirm` and reads its status at the recorded stamp.
`python report_rv2d.py confirm` still regenerates `RESULTS_RV2D.md` and `verdicts_rv2d.json` from the records, unchanged (checked on 2026-10-21).

## Checked at the adoption
`golden.py check` bit-identical (the flag-off pool is unchanged); `python -m pytest tests/growmoe` green; the smoke phase of the RV2-D runner (6 cells: arithmetic, banking77 and
function-regime streams) runs end to end; the GrowMoE demo runs with the new defaults.
