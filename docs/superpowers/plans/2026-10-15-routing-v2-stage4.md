# Routing v2 Stage 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans. Execution method (chosen by the user on 2026-10-15, "approved, write the plan and start building"): **native**, Claude builds every task in the main session, task by task with its tests, then one fresh reviewer checks the whole branch at the end. Steps use checkbox (`- [ ]`) syntax. Tasks 1 to 4 are test-first product code; tasks 5 to 8 are experiments. **One STOP gate:** after the dev review (task 6) and the written pre-registration (task 7) the user's explicit go is needed before the confirmatory seeds 85-99 are spent (single use).

**Goal:** make GrowMoE (PhyloMoE) route by a rule it chooses itself (4a), decide spawn-or-extend by asking the labels (4b) and grow width from the within-batch fit gap (4c), each behind a flag that is OFF until its pre-registered run, with the old pool bit-identical when the flags are off.

**Architecture:** three small pure modules (`growmoe/router.py`, `growmoe/transfer.py`, `growmoe/gapgrow.py`) and thin hooks inside `PhyloMoE` (config fields, `observe`, `predict`, `_handle_cluster`, `_attempt_merges`, `extra_cost`, `accounting.pool_report`). The router acts at answer time only and never influences training. An experiment package `research/growmoe_results/rv2d/` runs every arm on every stream in resumable cells, selects the dev constants by declared rules, freezes the pre-registration and runs the confirmatory seeds.

**Tech stack:** Python 3.12, PyTorch (CPU), NumPy, pytest; existing `growmoe` package (`fairfight.run_fast`, `metrics.paired_diff`, `settest.generator`, `setgate`/`groupvote`); the lab notebook tool `docs/labnotebook/tools/labnb.py`.

**Spec:** `docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md` (approved 2026-10-15; section 13 there records the refinements made while planning, listed below). Research: `research/GROWMOE_STAGE4_RESEARCH.md`. Spike numbers: `research/growmoe_results/stage4_spikes/RESULTS_SPIKES.md` (E-032, claims C-63 to C-66). Root-cause study: E-030.

## Progress log (kept current while building)

| Task | State | Commit | Notes |
|---|---|---|---|
| 1 scaffolding | done 2026-10-15 | 12f570d | 36 tests; suite 728 passed; full golden bit-identical |
| 2 router (4a) | done 2026-10-15 | 906e954 | 12 + 15 tests; the product router selects LDA on a saved arithmetic pool (CV 0.875 / 1.000 / 1.000 / 0.836); old pickled pools load |
| 3 transfer (4b) | done 2026-10-15 | 0724b90 | 7 + 14 tests; **merge audit: 0 of 4 forced duplicates merged** (strict xfail, F-28, Q-19); early stop dropped (R-5) |
| 4 growth (4c) | done 2026-10-15 | a6e1380 | 10 + 15 tests; full suite 801 passed + 2 expected failures; full golden bit-identical; build record E-033 |
| 5 runner and dev runs | dev1 DONE 2026-10-16 (157 cells); selection rule run (grow_gap 0.17, router free-0.01, transfer margin default because the S3-relative expert guard misfires); **dev2 core DONE 2026-10-17 (24 cells; the s3 family is skipped there); the user left the head-start decision to Claude (D-34), so the s3 family was run without the head start by `run_s3_unaliased.py` (core 60 + growth grid 24 + remaining five streams 15, E-038, E-039)** | 5ce5ee6, 850eafa, 6abfcc8 | runner smoke-tested for all families incl. a kill and resume (identical numbers); code stamp per cell (a result counts only for the code that made it); bugs found by the smoke run and by re-reading: softmax under no_grad (85f1ae5), alarm-path scoring (R-7, 850eafa) |
| 6 dev review, verdicts, mutation check | verdict module done (ea72e58); mutation check 24/24 killed; **dev review DONE 2026-10-17** (`RESULTS_DEV_REVIEW.md`, E-039): the declared rules on the unaliased dev view give the same constants; amendments A4-1 (S4-5 and the transfer-margin guard count experts against the true number of functions: **needs the user's OK at the go**), A4-2 (pool seed + 7 for the function-regime streams, in `rv2d_lib.pool_seed`), A4-7 (the twin fix stays out of the confirmatory run) in `docs/superpowers/specs/2026-10-15-routing-v2-stage4-AMENDMENTS.md`; mutation check extended by three verdict mutants | ea72e58 | |
| 6a stage 4.1 (loud cap, share instead of drop, counters, optional twin fix) | built and tested on branch `stage4-1-cap-and-telemetry`: 10 + 9 new tests, golden bit-identical with every new flag at its default, two older tests and the M5 single-network baseline pin `cap_policy="drop"`; **MERGED into `track-g-growmoe` 2026-10-17 (fast-forward, ba0e2e4) after the dev review and before the freeze**; new code stamp ff3438232c5f0dfa | see E-036 | behaviour-neutral until a pool holds 256 experts; `sig_dedupe` (E-037) is OFF until the dev check and a decision |
| 7 pre-registration, then STOP for the go | `freeze_prereg.py --out <scratch>` rehearsed 2026-10-17 on the merged code (dev verdicts S4-1..S4-9 all PASS or NOT MEASURABLE-counted-as-PASS; S4-3, S4-5, S4-7 need the guard streams of the fresh run); the real freeze is NOT done: it waits for the user's OK of A4-1 and the explicit go | | |
| 8 confirmatory run, audit, adoption | | | |

## Global Constraints

- All stage-4 flags (`router`, `spawn_rule="transfer"`, `growth_rule="gap"`) are OFF by default until task 8 decides adoption; with all flags off the pool is **bit-identical** to the recorded one (statement S4-9): `python research/growmoe_results/rv2c/golden.py check --quick` after every task, the full `check` after tasks 1, 4 and 8.
- Test first, always: no product code without a failing test that was run and seen to fail (red), then minimal code (green). A statistical test whose synthetic setting needs tuning at first run gets the SETTING changed and the reason written in its docstring, never a weaker assertion.
- Numbers in any report come from scripts; nothing is typed by hand. Claims are limited to the streams tested.
- Seeds: dev seeds are 100 and above; **85-99 are the stage-4 confirmatory seeds, single use**; 60-79 stay reserved for the stage-3 confirmatory run (awaiting the user's go); 80-84 are used up.
- Memory-lean: all runs are CPU; at most 3 parallel jobs x 3 threads (BUDGET 9 threads) so the machine stays usable; no GPU job is needed. No 8192-square benchmarks.
- Outage-proof (D-30, the PC loses power daily): every run is a cell of at most ten minutes or checkpoints per block; every result is written atomically (temp file + `os.replace`); an unreadable result counts as missing; one idempotent command resumes everything; background launches are hidden and detached (Popen with `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`).
- Never overwrite shared notebook files: new entries by exclusive creation (`open(path, "x")`) with the next free id read right before writing; appends to CLAIMS/DECISIONS/FAILURES read the last id and write atomically. Other tracks write there too (E-027 to E-029 and E-031 are track Z's).
- Stage only my own files by explicit path (`git add <paths>`); never `git add -A`; the geometry track's uncommitted files (`geomlab/phase3_geocore.py`, `research/geomlab_results/phase3/*`, E-031) are not mine.
- Commits end with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- The Bash tool collapses backslash escapes inside heredocs: files are written with the Write/Edit tools, never with `cat <<EOF`.
- A change that passes its pre-registered test becomes the default, the historical entry points are pinned to what they ran, and notes go wherever the old behaviour was documented (standing rule, memory note `adopt-confirmed-fixes-as-default-and-document`).
- Do not send `KICKOFF_RV3_AMENDED_BUILD_DEV_2026-10-14.md` (superseded); do not install `install_autoresume_task.ps1` without the user's decision.

## Refinements to the spec made while planning (recorded in spec section 13, task 1)

1. **R-1, the router is refitted lazily.** The router never influences training, so nothing is fitted inside `observe`. `observe` only counts the call and marks the router stale when an expert was spawned, extended, merged or archived; the first `predict` that finds the router stale (never fitted, expert set changed, marked, or `router_refit_every` observe calls old) refits it. The triggers are the spec's (a), (b), (c); training pays nothing; the fit uses the prototypes of the moment of the query.
2. **R-2, transfer rule details.** The parent is the expert with the lowest mean zero-shot loss on the cluster among those with at least `transfer_min_ref` reference losses (the nearest signature is not used). On the alarm path (`force_parent`) the newest `len(cluster)` reference entries are left out, because the cluster IS those samples. If the test cannot run (fewer than 8 labelled samples, no expert with enough reference losses) the pool takes **today's path** (probe where its conditions hold, otherwise plain spawn).
3. **R-3, accounting.** `PoolReport` gains `extra_scalars` (default 0); `memory_bytes` adds 4 bytes per scalar. `PhyloMoE.extra_cost(d)` returns (parameters, extra predict multiply-adds, extra stored scalars): the selected router rule (LDA and softmax: k*d + k parameters and k*d multiply-adds; 5-NN: no parameters, a second scan of the stored prototypes as multiply-adds and its copy of them as scalars), the reference windows (scalars), and a live growth guard's pre-growth module (parameters).
4. **R-4, misconfiguration is refused.** `router=True` needs a fast `route_rule` (the reference path of `route_rule="default"` is router-blind): `PhyloCfg` raises, and `predict` raises if the rule is changed to `"default"` afterwards.
5. **R-5 (made during the build), the transfer test draws all permutations.** The spec's early stop would have made the reported p a lower bound in the `spawn` and `extend` events for no gain (a call costs about ten milliseconds, once per cluster); `cluster_conflict` reports the full exact p = (1 + c) / (1 + n_perm).
6. **R-7 (made during the build), the alarm path uses the stored honest losses.** When the competence monitor fires, the cluster handed to `_handle_cluster` is the parent's own newest samples, which it may already have trained on; re-scoring them with the current weights understated the conflict. The transfer verdict of the alarm path takes the newest `len(cluster)` entries of the parent's reference window as the cluster's losses (honest: taken before any training on them) and the older entries as the reference.

## Review Focus (spec-implied failure modes; each has a pinned test in the owning task)

1. A pool with ONE expert, experts with too few prototypes, or an EMPTY batch, queried with the router on, must give the shipped answer and never crash (`test_a_pool_with_one_expert_or_a_young_expert_keeps_the_density_rule`, `test_an_empty_batch_is_handled_exactly_like_the_shipped_path`, task 2).
2. Clusters with NO labels, or labels the parent has never seen (class-incremental text), under the transfer rule: unlabelled falls back to today's path; unseen labels are a conflict with finite, JSON-serialisable event fields (`test_an_unlabelled_cluster_is_handled_by_the_path_of_today`, `test_labels_the_parent_has_never_seen_are_a_conflict_with_finite_event_fields`, task 3).
3. Pickle and deepcopy of a pool that holds a fitted router or an active growth guard (the diagnosis pipeline pickles pools; closures would break it) (`test_a_pool_that_holds_a_fitted_router_can_be_pickled_and_copied`, `test_a_pool_with_an_active_guard_can_be_pickled`, tasks 2 and 4).
4. Merging two experts while references or a guard exist: the survivor takes the union of the references and drops a stale guard (`test_a_merge_gives_the_survivor_the_references_of_both`, `test_a_merge_drops_the_guard_and_the_stale_windows`, tasks 3 and 4); and the audit that merge cleans forced duplicates at all (`test_merge_cleans_up_a_forced_duplicate`, task 3).
5. Growth that hits the width cap or the parameter budget must stop quietly, and a router configured with a reference `route_rule` must be refused instead of silently doing nothing (`test_growth_stops_at_the_cap_and_at_the_budget_without_error`, task 4; `test_a_router_with_a_reference_route_rule_is_refused_at_predict_time`, task 2).

## File structure

| File | Action | Responsibility |
|---|---|---|
| `growmoe/config.py` | modify (task 1) | 25 new `PhyloCfg` fields, all defaults OFF, validated in `__post_init__` |
| `growmoe/router.py` | create (task 2) | candidates, cross-validation, selection with hysteresis, `RouteRule` (pure; scorers are classes so a pool pickles) |
| `growmoe/transfer.py` | create (task 3) | exact one-sided permutation test with a margin, `TransferVerdict` |
| `growmoe/gapgrow.py` | create (task 4) | `GapMonitor`, `widen_increment`, `params_of_widen`, `guard_reverts`, `GrowthGuard` |
| `growmoe/pools/phylo.py` | modify (tasks 2-4) | hooks: `__init__`, `observe`, `predict`, `_handle_cluster`, `_attempt_merges`, helpers, `extra_cost` |
| `growmoe/accounting.py` | modify (task 2) | `extra_cost` hook, `extra_scalars` |
| `growmoe/telemetry.py` | modify (task 2) | `EventLog.since(n)` |
| `growmoe/verdicts_rv4.py` | create (task 6) | S4-1 to S4-9 verdict functions and the decision rule (the dev review and the pre-registration use the same code) |
| `tests/growmoe/rv4_helpers.py` | create (task 2), extend (tasks 3, 4) | synthetic data shared by the RV4 tests |
| `tests/growmoe/test_gm_rv4_config.py`, `..._router.py`, `..._router_pool.py`, `..._transfer.py`, `..._transfer_pool.py`, `..._growth.py`, `..._growth_pool.py`, `..._runner.py`, `..._verdicts.py` | create | the tests |
| `research/growmoe_results/rv2c/golden.py` | modify (task 1) | pin the three flags OFF in `BASE` |
| `research/growmoe_results/rv2d/` | create (tasks 5-8) | runner, job queue, selection of dev constants, mutation check, pre-registration, reports |
| `docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md` | modify (task 1) | section 13, the refinements above |
| `docs/labnotebook/*`, memory notes | extend | entries, claims, decisions, failures, manifest after each evidence-bearing task |

---

## Task 1: Scaffolding - config fields, validation, golden pins

**Files:**
- Modify: `growmoe/config.py` (class `PhyloCfg`: new fields after `route_rule`, new checks at the end of `__post_init__`)
- Modify: `research/growmoe_results/rv2c/golden.py` (`BASE`)
- Modify: `docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md` (append section 13)
- Create: `tests/growmoe/test_gm_rv4_config.py`

**Interfaces:**
- Produces: `PhyloCfg.router: bool`, `router_candidates: tuple`, `router_margin: float`, `router_folds: int`, `router_refit_every: int`, `router_min_protos: int`, `router_mode: str`, `router_softmax_max_experts: int`, `spawn_rule: str`, `transfer_alpha`, `transfer_margin`, `transfer_perms`, `transfer_ref`, `transfer_min_ref`, `spawn_pretrain_steps`, `growth_rule: str`, `grow_gap`, `grow_window`, `grow_plateau`, `grow_min_age`, `grow_cooldown`, `grow_factor`, `grow_guard_window`, `grow_guard_eps`, `grow_budget_params`. Every later task reads these names.

- [ ] **Step 1.1: Write the failing test**

Create `tests/growmoe/test_gm_rv4_config.py`:

```python
"""RV4 scaffolding: every stage-4 flag exists, is OFF by default, carries the declared constants and is validated (spec sections 4.3, 5.3, 6.5)."""

import importlib.util
from pathlib import Path

import pytest

from growmoe.config import PhyloCfg

ROOT = Path(__file__).resolve().parents[2]


def _golden():
    spec = importlib.util.spec_from_file_location("golden", ROOT / "research" / "growmoe_results" / "rv2c" / "golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_stage_4_flag_is_off_by_default():
    cfg = PhyloCfg()
    assert cfg.router is False and cfg.spawn_rule == "probe" and cfg.growth_rule == "probe"


def test_the_stage_4_constants_have_the_declared_defaults():
    cfg = PhyloCfg()
    assert (cfg.router_candidates, cfg.router_margin, cfg.router_folds, cfg.router_refit_every, cfg.router_min_protos, cfg.router_mode,
            cfg.router_softmax_max_experts) == (("density", "lda", "softmax", "knn5"), 0.01, 5, 50, 8, "restrict", 64)
    assert (cfg.transfer_alpha, cfg.transfer_margin, cfg.transfer_perms, cfg.transfer_ref, cfg.transfer_min_ref, cfg.spawn_pretrain_steps) == (0.01, 0.25, 999, 512, 64, 60)
    assert (cfg.grow_window, cfg.grow_plateau, cfg.grow_min_age, cfg.grow_cooldown, cfg.grow_factor, cfg.grow_guard_window, cfg.grow_guard_eps,
            cfg.grow_budget_params) == (100, 0.95, 300, 200, 2.0, 100, 0.02, None)
    assert 0.0 < cfg.grow_gap < 1.0


def test_valid_stage_4_settings_are_accepted():
    cfg = PhyloCfg(router=True, route_rule="accept_then_logf", spawn_rule="transfer", growth_rule="gap", router_mode="free", grow_budget_params=10_000)
    assert cfg.router and cfg.spawn_rule == "transfer" and cfg.growth_rule == "gap"


@pytest.mark.parametrize("kw", [
    dict(router_mode="anywhere"),
    dict(router_candidates=("lda",)),                                    # density is the fallback and must be a candidate
    dict(router_candidates=("density", "svm")),
    dict(router_folds=1),
    dict(router_min_protos=3),                                           # fewer than the folds
    dict(router_refit_every=0),
    dict(router_margin=-0.1),
    dict(router_softmax_max_experts=1),
    dict(router=True),                                                   # route_rule defaults to "default": the reference path is router-blind, so refuse instead of silently doing nothing
    dict(router=True, route_rule="default"),
    dict(spawn_rule="oracle"),
    dict(transfer_alpha=0.0), dict(transfer_alpha=1.0),
    dict(transfer_alpha=0.001, transfer_perms=99),                       # 1/(1+99) > 0.001: the test could never reject
    dict(transfer_margin=-0.01),
    dict(transfer_perms=0),
    dict(transfer_ref=32, transfer_min_ref=64),
    dict(transfer_min_ref=4),
    dict(spawn_pretrain_steps=-1),
    dict(growth_rule="always"),
    dict(grow_gap=0.0), dict(grow_gap=1.0),
    dict(grow_window=5),
    dict(grow_plateau=0.0), dict(grow_plateau=1.5),
    dict(grow_min_age=0),
    dict(grow_cooldown=-1),
    dict(grow_factor=1.0),
    dict(grow_guard_window=0),
    dict(grow_guard_eps=-0.1),
    dict(grow_budget_params=0),
])
def test_invalid_stage_4_settings_are_refused(kw):
    with pytest.raises(ValueError):
        PhyloCfg(**kw)


def test_the_golden_file_pins_every_stage_4_flag_off():
    golden = _golden()
    assert golden.BASE["router"] is False and golden.BASE["spawn_rule"] == "probe" and golden.BASE["growth_rule"] == "probe"


def test_with_all_stage_4_flags_off_the_pool_is_bit_identical_to_the_recorded_golden_run():
    golden = _golden()
    assert golden.differences(golden.QUICK) == []
```

- [ ] **Step 1.2: Run it and watch it fail**

Run: `python -m pytest tests/growmoe/test_gm_rv4_config.py -q -p no:cacheprovider`
Expected: the defaults, valid-settings, invalid-settings and golden-pin tests FAIL (`AttributeError: 'PhyloCfg' object has no attribute 'router'` / `DID NOT RAISE` / `KeyError: 'router'`); the last test (bit-identical) passes because nothing reads the flags yet - it is a regression guard, not a red test.

- [ ] **Step 1.3: Add the fields and the validation**

In `growmoe/config.py`, in class `PhyloCfg`, directly after the `route_rule` field line and before `def __post_init__`, insert:

```python
    # ---- routing v2 stage 4 (docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md). Every switch is OFF until its pre-registered run (RV2-D).
    router: bool = False         # 4a: at answer time a router chosen by cross-validation on the stored prototypes picks the expert among those the gate accepts
    router_candidates: tuple = ("density", "lda", "softmax", "knn5")
    router_margin: float = 0.01  # another rule must beat density's cross-validated owner accuracy by MORE than this (and the current rule by more than this to replace it)
    router_folds: int = 5
    router_refit_every: int = 50     # observe calls after which the next predict refits the router (it is also refitted when the expert set changed)
    router_min_protos: int = 8       # below this many stored prototypes in any expert (or with fewer than two experts) the density rule stays
    router_mode: str = "restrict"    # "restrict": best router score among the experts the gate accepts; "free": best score overall
    router_softmax_max_experts: int = 64   # the softmax candidate is skipped above this many experts (bounds the refit time)
    spawn_rule: str = "probe"    # 4b: "probe" = the 60-step interference probe of today; "transfer" = ask the labels (held-out loss of the best expert against its own reference)
    transfer_alpha: float = 0.01
    transfer_margin: float = 0.25    # practical-significance margin in units of ln(classes the parent has seen)
    transfer_perms: int = 999
    transfer_ref: int = 512          # honest per-sample losses kept per expert as its reference
    transfer_min_ref: int = 64       # an expert with fewer reference losses cannot be the parent of a transfer test
    spawn_pretrain_steps: int = 60   # training steps of a child born from a transfer conflict (the probe's child gets the same number)
    growth_rule: str = "probe"   # 4c: "probe" = the shadow-trial policy of today; "gap" = widen when the within-batch fit gap is small and progress has stalled
    grow_gap: float = 0.20       # gap below which a mature expert widens; middle of the declared grid 0.14..0.26, the calibrated value is fixed on dev seeds (RV2-D, task 5)
    grow_window: int = 100       # batches over which the gap and the progress ratio are measured
    grow_plateau: float = 0.95   # mature = the honest loss of the last window is at least this share of the one before it
    grow_min_age: int = 300      # ... and the expert has trained on at least this many batches since birth or its last growth
    grow_cooldown: int = 200     # batches between two growths of one expert (doubles after a reverted growth)
    grow_factor: float = 2.0     # width multiplier of one growth
    grow_guard_window: int = 100     # batches after a growth during which it can be reverted
    grow_guard_eps: float = 0.02     # revert when the honest loss over the guard window is worse than before by more than this share
    grow_budget_params: object = None    # optional cap on the pool's live parameters (None = only GrowthCfg.max_hidden bounds growth)
```

At the end of `__post_init__` (after the last existing `for name in (...)` loop) append:

```python
        # ---- routing v2 stage 4
        if self.router_mode not in ("restrict", "free"):
            raise ValueError(f"router_mode must be 'restrict' or 'free', got {self.router_mode!r}")
        known = ("density", "lda", "softmax", "knn5")
        if "density" not in self.router_candidates or any(c not in known for c in self.router_candidates):
            raise ValueError(f"router_candidates must be drawn from {known} and include 'density' (the fallback), got {self.router_candidates!r}")
        if self.router_folds < 2 or self.router_min_protos < self.router_folds or self.router_refit_every < 1 or self.router_softmax_max_experts < 2:
            raise ValueError("router_folds >= 2, router_min_protos >= router_folds, router_refit_every >= 1 and router_softmax_max_experts >= 2 are required")
        if self.router_margin < 0.0:
            raise ValueError("router_margin must not be negative")
        if self.router and self.route_rule == "default":
            raise ValueError("router=True needs a fast route_rule (for example 'accept_then_logf'): the reference path used by route_rule='default' is router-blind")
        if self.spawn_rule not in ("probe", "transfer"):
            raise ValueError(f"spawn_rule must be 'probe' or 'transfer', got {self.spawn_rule!r}")
        if not 0.0 < self.transfer_alpha < 1.0:
            raise ValueError(f"transfer_alpha must lie in (0, 1), got {self.transfer_alpha}")
        if self.transfer_perms < 1 or math.floor(self.transfer_alpha * (1 + self.transfer_perms) + 1e-9) < 1:     # the rule of growmoe.settest.max_exceedances
            raise ValueError(f"transfer_perms={self.transfer_perms} is too small for transfer_alpha={self.transfer_alpha}: the smallest attainable p-value exceeds the level")
        if self.transfer_margin < 0.0 or self.spawn_pretrain_steps < 0:
            raise ValueError("transfer_margin and spawn_pretrain_steps must not be negative")
        if self.transfer_min_ref < 8 or self.transfer_ref < self.transfer_min_ref:
            raise ValueError("transfer_min_ref must be at least 8 and transfer_ref at least transfer_min_ref")
        if self.growth_rule not in ("probe", "gap"):
            raise ValueError(f"growth_rule must be 'probe' or 'gap', got {self.growth_rule!r}")
        if not 0.0 < self.grow_gap < 1.0 or not 0.0 < self.grow_plateau <= 1.0:
            raise ValueError("grow_gap must lie in (0, 1) and grow_plateau in (0, 1]")
        if self.grow_window < 10 or self.grow_min_age < 1 or self.grow_cooldown < 0 or self.grow_guard_window < 1:
            raise ValueError("grow_window >= 10, grow_min_age >= 1, grow_cooldown >= 0 and grow_guard_window >= 1 are required")
        if self.grow_factor <= 1.0 or self.grow_guard_eps < 0.0:
            raise ValueError("grow_factor must exceed 1 and grow_guard_eps must not be negative")
        if self.grow_budget_params is not None and int(self.grow_budget_params) < 1:
            raise ValueError("grow_budget_params must be None or a positive integer")
```

- [ ] **Step 1.4: Pin the flags in the golden file**

In `research/growmoe_results/rv2c/golden.py`, after the existing

```python
if "set_gate" in PhyloCfg.__dataclass_fields__:                                          # after stage 3 exists: pin both flags OFF so this file keeps testing the flags-off pool
    BASE.update(set_gate=False, group_vote=False)                                       # even after a default is flipped (the recorded file is the same either way)
```

add:

```python
if "router" in PhyloCfg.__dataclass_fields__:                                            # after stage 4 exists: pin its three switches OFF for the same reason
    BASE.update(router=False, spawn_rule="probe", growth_rule="probe")
```

- [ ] **Step 1.5: Run the tests and the golden check**

Run: `python -m pytest tests/growmoe/test_gm_rv4_config.py -q -p no:cacheprovider`
Expected: all pass (about 40 parametrised cases).
Run: `python research/growmoe_results/rv2c/golden.py check`
Expected: `bit-identical` (all 20 fingerprints; about 2 minutes).
Run: `python -m pytest tests/growmoe -q -x -p no:cacheprovider` in the background and read the count when it ends (it was 692 passed before this stage).
Expected: 692 + the new tests, no failures.

- [ ] **Step 1.6: Append section 13 to the spec and commit**

Append to `docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md` a section `## 13. Refinements made while planning (2026-10-15)` containing the four items R-1 to R-4 of this plan's "Refinements" section, word for word. Then:

```bash
git add growmoe/config.py research/growmoe_results/rv2c/golden.py tests/growmoe/test_gm_rv4_config.py docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md docs/superpowers/plans/2026-10-15-routing-v2-stage4.md
git commit -m "feat(growmoe): stage 4 scaffolding - flags (all off), validation, golden pins; plan and spec refinements"
```
(with the attribution line in the message body).

---

## Task 2: Stage 4a - the router chooses its own rule

**Files:**
- Create: `growmoe/router.py`
- Modify: `growmoe/telemetry.py` (`EventLog.since`), `growmoe/pools/phylo.py` (imports, `__init__`, `observe`, `predict`, new methods), `growmoe/accounting.py`
- Create: `tests/growmoe/rv4_helpers.py` (shared synthetic data), `tests/growmoe/test_gm_rv4_router.py` (pure functions), `tests/growmoe/test_gm_rv4_router_pool.py` (pool integration, accounting)

**Interfaces:**
- Consumes: `PhyloCfg.router*` fields (task 1); `VmfSignature(d, max_protos, seed)` with `update`, `fit`, `score`, `_log_c`, `_kappa`, `protos`, `n`; `growmoe.settest.generator(*key)`.
- Produces (task 5 and later rely on these exact names):
  - `growmoe.router.CANDIDATES`, `FITS: Dict[str, Callable]`, scorers `DensityScorer`, `LdaScorer`, `SoftmaxScorer`, `KnnScorer` (each `__call__(Z: Tensor (m, d)) -> Tensor (m, k)`, higher = more likely owner), `fold_assignment(y, k, folds, gen) -> LongTensor`, `cv_owner_accuracy(X, y, k, folds, gen, candidates=CANDIDATES) -> Dict[str, float]`, `select_rule(cv, current, margin) -> str`, `RouteRule` with `name`, `scorer`, `eids`, `cv`, `rows`, `set(name, scorer, eids, cv, rows)`, `cost(d) -> (params, macs, scalars)`.
  - `PhyloMoE._route_state()`, `_route_is_stale()`, `_current_route_rule() -> RouteRule`, `_refit_router()`, `_route_by_router(x, eids, P, LogF, idx)`, `_router_after_observe(mark)`, attributes `_route`, `_route_age`, `_route_dirty`; events `router_refit` {selected, cv, rows, experts} and `router_switch` {from, to}; `PhyloMoE.extra_cost(d) -> (params, macs, scalars)` (tasks 3 and 4 add to it).
  - `EventLog.since(n) -> list[Event]`; `PoolReport.extra_scalars`.

### 2a. The pure functions

- [ ] **Step 2.1: Write the shared synthetic data and the failing tests**

Create `tests/growmoe/rv4_helpers.py` (shared by the RV4 tests; task 3 adds the learnable-regime helpers to it):

```python
"""Synthetic data shared by the RV4 tests."""

import numpy as np
import torch


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def aniso(d=12, k=6, n=16, spacing=0.6, s_info=0.08, s_nuis=1.0, n_nuis=4, seed=0):
    """k classes whose means differ only along coordinate 0, where the noise is small; four other coordinates carry large noise that says nothing about the class.
    A shared-covariance discriminant (LDA) discounts the noisy coordinates; a kernel density over a handful of prototypes is dominated by them.
    (Chosen by a scratch search on 2026-10-15: CV owner accuracy lda 0.97-0.99 against density 0.46-0.60 over seeds 0-3.)"""
    g = np.random.default_rng(seed)
    scale = np.full(d, 0.08)
    scale[0] = s_info
    scale[1:1 + n_nuis] = s_nuis
    X, y = [], []
    for c in range(k):
        m = np.zeros(d)
        m[d - 1] = 3.0                                                 # a shared offset keeps the points away from the origin, so normalising keeps the geometry
        m[0] = (c - (k - 1) / 2) * spacing
        X.append(m + scale * g.standard_normal((n, d)))
        y += [c] * n
    return torch.from_numpy(unit(np.concatenate(X))), torch.tensor(y)


def antipodal(d=16, k=4, n=40, sigma=0.05, seed=0):
    """Expert c holds two OPPOSITE clusters (+v_c and -v_c): its class mean is near zero, so no linear rule can see it, while a density rule can."""
    g = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        v = np.zeros(d)
        v[c] = 1.0
        for s in (1.0, -1.0):
            X.append(s * v + sigma * g.standard_normal((n // 2, d)))
            y += [c] * (n // 2)
    return torch.from_numpy(unit(np.concatenate(X))), torch.tensor(y)
```

Create `tests/growmoe/test_gm_rv4_router.py`:

```python
"""RV4 stage 4a, pure functions: candidates, cross-validation, selection with hysteresis (spec section 4.1)."""

import numpy as np
import pytest
import torch

from growmoe.router import CANDIDATES, FITS, RouteRule, cv_owner_accuracy, fit_softmax, fold_assignment, select_rule
from growmoe.settest import generator
from rv4_helpers import aniso, antipodal, unit


def test_cross_validation_prefers_lda_when_a_shared_covariance_removes_the_noise():
    for seed in range(3):
        X, y = aniso(seed=seed)
        cv = cv_owner_accuracy(X, y, 6, 5, generator(1, seed))
        assert cv["lda"] > cv["density"] + 0.2 and cv["lda"] > 0.9
        assert select_rule(cv, "density", 0.01) == "lda"


def test_cross_validation_keeps_density_for_experts_that_hold_two_opposite_clusters():
    for seed in range(3):
        X, y = antipodal(seed=seed)
        cv = cv_owner_accuracy(X, y, 4, 5, generator(2, seed))
        assert cv["density"] > 0.95 and cv["lda"] < 0.5 and cv["softmax"] < 0.6
        assert select_rule(cv, "density", 0.01) == "density"


def test_every_candidate_fits_well_separated_classes():
    g = np.random.default_rng(0)
    X = np.concatenate([5 * np.eye(12)[c] + g.standard_normal((20, 12)) for c in range(3)])          # three clusters along three axes
    X, y = torch.from_numpy(unit(X)), torch.arange(3).repeat_interleave(20)
    for name, fit in FITS.items():
        scores = fit(X, y, 3, generator(3))(X)
        assert scores.shape == (60, 3), name
        assert (scores.argmax(1) == y).float().mean() > 0.95, name


def test_a_class_without_rows_can_never_be_chosen():
    X, y = aniso(k=3, n=16)
    for name in ("density", "lda", "softmax"):                                    # class 3 exists in the label space but holds nothing
        s = FITS[name](X, y, 4, generator(4))(X)
        assert s.shape == (len(X), 4), name
        assert bool((s[:, 3] == -float("inf")).all()) and bool((s.argmax(1) != 3).all()), name
    s = FITS["knn5"](X, y, 4, generator(4))(X)
    assert bool((s.argmax(1) != 3).all())


def test_softmax_is_class_balanced():
    g = np.random.default_rng(1)                                                  # 90 % of the rows belong to class 0; an unbalanced fit would ignore class 1
    a = g.standard_normal((180, 12)) + 3.0 * np.eye(12)[0]
    b = g.standard_normal((20, 12)) + 3.0 * np.eye(12)[1]
    X, y = torch.from_numpy(unit(np.concatenate([a, b]))), torch.tensor([0] * 180 + [1] * 20)
    s = fit_softmax(X, y, 2, generator(5))(X)
    assert float((s[y == 1].argmax(1) == 1).float().mean()) > 0.8


def test_folds_cover_every_row_once_and_spread_every_class():
    y = torch.tensor([0] * 23 + [1] * 17 + [2] * 11)
    f = fold_assignment(y, 3, 5, generator(6))
    assert f.shape == y.shape and set(f.tolist()) == set(range(5))
    for c in range(3):
        counts = torch.bincount(f[y == c], minlength=5)
        assert int(counts.max() - counts.min()) <= 1


def test_cross_validation_is_a_function_of_its_inputs_and_generator():
    X, y = aniso()
    a = cv_owner_accuracy(X, y, 6, 5, generator(7))
    assert a == cv_owner_accuracy(X, y, 6, 5, generator(7))
    assert a != cv_owner_accuracy(X, y, 6, 5, generator(8))                        # another generator, another fold split
    assert set(a) == set(CANDIDATES)


def test_cross_validation_runs_only_the_requested_candidates():
    X, y = aniso()
    assert set(cv_owner_accuracy(X, y, 6, 5, generator(9), ("density", "lda"))) == {"density", "lda"}


def test_select_rule_keeps_density_unless_another_rule_wins_by_more_than_the_margin():
    base = {"density": 0.50, "lda": 0.75, "softmax": 0.40, "knn5": 0.30}
    assert select_rule(base, "density", 0.25) == "density"                         # exactly the margin is not MORE than the margin
    assert select_rule({**base, "lda": 0.7501}, "density", 0.25) == "lda"
    assert select_rule({**base, "lda": 0.50, "softmax": 0.50}, "density", 0.0) == "density"       # a tie goes to density


def test_select_rule_picks_the_best_of_several_winners():
    assert select_rule({"density": 0.5, "lda": 0.8, "softmax": 0.9, "knn5": 0.7}, "density", 0.01) == "softmax"


def test_hysteresis_keeps_the_current_rule_inside_the_dead_band_and_switches_outside_it():
    dead = {"density": 0.935, "lda": 0.93, "softmax": 0.5, "knn5": 0.5}
    assert select_rule(dead, "lda", 0.01) == "lda"                                 # density leads by 0.005 only: stay
    assert select_rule(dead, "density", 0.01) == "density"
    back = {"density": 0.95, "lda": 0.93, "softmax": 0.5, "knn5": 0.5}
    assert select_rule(back, "lda", 0.01) == "density"                             # density leads by 0.02: go back
    close = {"density": 0.5, "lda": 0.9, "softmax": 0.905, "knn5": 0.5}
    assert select_rule(close, "lda", 0.01) == "lda"                                # softmax leads lda by 0.005 only: stay
    clear = {"density": 0.5, "lda": 0.9, "softmax": 0.915, "knn5": 0.5}
    assert select_rule(clear, "lda", 0.01) == "softmax"                            # softmax leads by 0.015: switch
    assert select_rule(clear, "knn5", 0.01) == "softmax"                           # a current rule that is not even better than density is simply left


def test_a_route_rule_reports_its_cost():
    rule = RouteRule()
    assert rule.name == "density" and rule.cost(12) == (0, 0, 0)
    rule.set("lda", object(), [0, 1, 2], {}, 48)
    assert rule.cost(12) == (3 * 12 + 3, 3 * 12, 0)
    rule.set("softmax", object(), [0, 1, 2], {}, 48)
    assert rule.cost(12) == (3 * 12 + 3, 3 * 12, 0)
    rule.set("knn5", object(), [0, 1, 2], {}, 48)
    assert rule.cost(12) == (0, 48 * 12, 48 * 12)
    rule.set("density", None, [0, 1, 2], {}, 48)
    assert rule.cost(12) == (0, 0, 0)
```

- [ ] **Step 2.2: Run and watch it fail**

Run: `python -m pytest tests/growmoe/test_gm_rv4_router.py -q -p no:cacheprovider`
Expected: collection error `ModuleNotFoundError: No module named 'growmoe.router'`.

- [ ] **Step 2.3: Implement `growmoe/router.py`**

```python
"""Routing v2, stage 4a: which expert ANSWERS (spec docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md, section 4).

The pool's density gate keeps deciding "known or unknown" and keeps its conformal guarantee. The router only decides, among the experts, who answers. It is trained on nothing
but what the experts already store (each expert's signature prototypes, labelled with the expert that holds them), chooses its own rule among a few cheap candidates by
cross-validated owner accuracy, and falls back to the density rule (today's) unless another rule is clearly better.

Everything here is a pure function of its arguments (a generator carries every random draw), so equal inputs give equal outputs. Scorers are small classes, not closures, so a pool
that holds one can still be pickled (the diagnosis pipeline pickles pools).
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .signature import VmfSignature

CANDIDATES = ("density", "lda", "softmax", "knn5")
NEG_INF = -math.inf


class DensityScorer:
    """One kernel density per expert, rebuilt from the rows it was given; score = log kernel density + the vMF normaliser (the rule of the pool's own gate, without its conformal part)."""

    def __init__(self, sigs: List[VmfSignature], present: List[bool]) -> None:
        self.sigs, self.present = sigs, present

    def __call__(self, Z: Tensor) -> Tensor:
        cols = [s.score(Z) + s._log_c[s._kappa] if ok else torch.full((Z.shape[0],), NEG_INF) for s, ok in zip(self.sigs, self.present)]
        return torch.stack(cols, dim=1)


def fit_density(X: Tensor, y: Tensor, k: int, gen: Optional[torch.Generator] = None) -> DensityScorer:
    sigs, present = [], []
    for c in range(k):
        rows = X[y == c]
        ok = rows.shape[0] >= 2
        s = VmfSignature(d=X.shape[1], max_protos=max(256, rows.shape[0]), seed=c)
        if ok:
            s.update(rows)
            s.fit()
        sigs.append(s)
        present.append(ok)
    return DensityScorer(sigs, present)


class LdaScorer:
    """Linear discriminants of class means with one pooled within-class covariance."""

    def __init__(self, w: Tensor, b: Tensor, present: Tensor) -> None:
        self.w, self.b, self.present = w, b, present

    def __call__(self, Z: Tensor) -> Tensor:
        return (Z.double() @ self.w + self.b).masked_fill(~self.present, NEG_INF).float()


def fit_lda(X: Tensor, y: Tensor, k: int, gen: Optional[torch.Generator] = None, reg: float = 1e-3) -> LdaScorer:
    """Class means, the pooled within-class covariance plus ``reg`` times the identity, log priors. Closed form (``gen`` is unused)."""
    Xd = X.double()
    n, d = Xd.shape
    counts = torch.bincount(y, minlength=k)
    present = counts > 0
    mu = torch.zeros(k, d, dtype=torch.float64).index_add_(0, y, Xd) / counts.clamp_min(1).double().unsqueeze(1)
    xc = Xd - mu[y]
    cov = xc.T @ xc / max(n - int(present.sum()), 1) + reg * torch.eye(d, dtype=torch.float64)
    w = torch.linalg.solve(cov, mu.T)                                  # (d, k) = precision @ means^T
    b = -0.5 * (mu * w.T).sum(dim=1) + torch.log(counts.clamp_min(1).double() / n)
    return LdaScorer(w, b, present)


class SoftmaxScorer:
    def __init__(self, w: Tensor, b: Tensor, present: Tensor) -> None:
        self.w, self.b, self.present = w, b, present

    def __call__(self, Z: Tensor) -> Tensor:
        return (Z @ self.w + self.b).masked_fill(~self.present, NEG_INF)


def fit_softmax(X: Tensor, y: Tensor, k: int, gen: torch.Generator, steps: int = 500, lr: float = 0.05, l2: float = 1e-3, batch: int = 2048) -> SoftmaxScorer:
    """Multinomial logistic regression, Adam, L2, **class-balanced loss** (every expert counts the same however many prototypes it holds)."""
    n, d = X.shape
    counts = torch.bincount(y, minlength=k)
    w = torch.zeros(d, k, requires_grad=True)
    b = torch.zeros(k, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    cls_w = (n / (k * counts.clamp_min(1))).float()
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=gen)
        loss = F.cross_entropy(X[idx] @ w + b, y[idx], weight=cls_w) + l2 * (w ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return SoftmaxScorer(w.detach(), b.detach(), counts > 0)


class KnnScorer:
    """Cosine vote of the ``nn`` nearest stored rows (a score per expert: the summed positive similarities of its rows among the neighbours)."""

    def __init__(self, X: Tensor, y: Tensor, k: int, nn: int = 5) -> None:
        self.X, self.y, self.k, self.nn = X, y, k, nn

    def __call__(self, Z: Tensor) -> Tensor:
        top = (Z @ self.X.T).topk(min(self.nn, self.X.shape[0]), dim=1)
        out = torch.zeros(Z.shape[0], self.k)
        out.scatter_add_(1, self.y[top.indices], top.values.clamp_min(0) + 1e-6)
        return out


def fit_knn(X: Tensor, y: Tensor, k: int, gen: Optional[torch.Generator] = None, nn: int = 5) -> KnnScorer:
    return KnnScorer(X.clone(), y.clone(), k, nn)


FITS = {"density": fit_density, "lda": fit_lda, "softmax": fit_softmax, "knn5": fit_knn}


def fold_assignment(y: Tensor, k: int, folds: int, gen: torch.Generator) -> Tensor:
    """Fold of every row: the rows of each expert are shuffled and dealt round-robin, so every fold holds (nearly) the same share of every expert."""
    fold_of = torch.zeros(y.shape[0], dtype=torch.long)
    for c in range(k):
        idx = (y == c).nonzero().squeeze(1)
        if idx.numel():
            fold_of[idx[torch.randperm(idx.numel(), generator=gen)]] = torch.arange(idx.numel()) % folds
    return fold_of


def cv_owner_accuracy(X: Tensor, y: Tensor, k: int, folds: int, gen: torch.Generator, candidates: Sequence[str] = CANDIDATES) -> Dict[str, float]:
    """Mean over folds of how often each candidate, fitted on the other folds, names the expert that holds a held-out row."""
    fold_of = fold_assignment(y, k, folds, gen)
    acc: Dict[str, List[float]] = {name: [] for name in candidates}
    for f in range(folds):
        te = fold_of == f
        if not bool(te.any()):
            continue
        tr = ~te
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen))
        for name in candidates:
            scorer = FITS[name](X[tr], y[tr], k, torch.Generator().manual_seed(seed))
            acc[name].append(float((scorer(X[te]).argmax(1) == y[te]).float().mean()))
    return {name: (sum(v) / len(v) if v else 0.0) for name, v in acc.items()}


def select_rule(cv: Dict[str, float], current: Optional[str], margin: float) -> str:
    """``density`` unless another candidate beats density's cross-validated accuracy by MORE than ``margin`` (then the best such candidate). Hysteresis: when a rule is already in
    use, a different one replaces it only if its accuracy exceeds the current rule's by more than ``margin`` too."""
    others = {n: a for n, a in cv.items() if n != "density"}
    best = max(others, key=others.get) if others else None
    candidate = best if best is not None and cv[best] > cv["density"] + margin else "density"
    if current is not None and current in cv and candidate != current and not cv[candidate] > cv[current] + margin:
        return current
    return candidate


class RouteRule:
    """What the pool holds: the selected rule, its fitted scorer (None for density), the experts (ascending ids) it was fitted for, and the last cross-validation."""

    def __init__(self) -> None:
        self.name: str = "density"
        self.scorer = None
        self.eids: Optional[List[int]] = None
        self.cv: Dict[str, float] = {}
        self.rows: int = 0

    def set(self, name: str, scorer, eids: List[int], cv: Dict[str, float], rows: int) -> None:
        self.name, self.scorer, self.eids, self.cv, self.rows = name, scorer, list(eids), dict(cv), int(rows)

    def cost(self, d: int) -> Tuple[int, int, int]:
        """(parameters, extra multiply-adds per prediction, extra stored scalars) of the selected rule."""
        k = len(self.eids or ())
        if self.name in ("lda", "softmax"):
            return k * d + k, k * d, 0
        if self.name == "knn5":
            return 0, self.rows * d, self.rows * d
        return 0, 0, 0
```

- [ ] **Step 2.4: Run and see it pass**

Run: `python -m pytest tests/growmoe/test_gm_rv4_router.py -q -p no:cacheprovider`
Expected: all pass. If a numeric premise of a test is off (the synthetic generators were chosen with a scratch search, not with this exact code), change the SETTING of the generator (n, spacing, noise) and write the reason in its docstring; do not weaken an assertion.

### 2b. Pool integration

- [ ] **Step 2.5: Write the failing pool tests**

Create `tests/growmoe/test_gm_rv4_router_pool.py`:

```python
"""RV4 stage 4a inside PhyloMoE: lazy refit, restrict/free, accounting, pickling, flag off (spec sections 4.2, 4.4, 4.5)."""

import copy
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from growmoe.accounting import pool_report
from growmoe.config import PhyloCfg
from growmoe.experts import MLPExpert
from growmoe.pools.phylo import PhyloMoE
from growmoe.signature import VmfSignature
from growmoe.streams_m1 import function_regimes

from rv4_helpers import aniso, antipodal, unit

ROOT = Path(__file__).resolve().parents[2]
K = 4                                                                              # label space of the hand-built experts (they are never trained)


def _hand_pool(Xs, **kw):
    """A pool whose experts are built by hand from the given prototype matrices (one per expert). The experts are untrained: these tests look at WHICH expert answers."""
    cfg = dict(q=0.02, set_gate=False, group_vote=False, route_rule="accept_then_logf")
    cfg.update(kw)
    pool = PhyloMoE(PhyloCfg(d=Xs[0].shape[1], hidden=8, n_classes=K, seed=1, **cfg))
    for c, X in enumerate(Xs):
        mod = MLPExpert(d_in=X.shape[1], hidden=8, depth=1, n_classes=K, seed=c)
        mod.mark_seen(torch.arange(K))
        sig = VmfSignature(d=X.shape[1], max_protos=256, seed=c)
        sig.update(X)
        sig.fit()
        pool.add_expert(mod, parent=None, state="active", meta={"sig": sig})
    return pool


def _split(X, y, k):
    return [X[y == c] for c in range(k)]


class _Prefer:
    """A scorer that ranks the highest-numbered expert first for every input (tests how the pool USES a scorer, whatever was fitted)."""

    def __init__(self, k):
        self.k = k

    def __call__(self, Z):
        s = torch.zeros(Z.shape[0], self.k)
        s[:, -1] = 1.0
        return s


def _two_blob_pool(**kw):
    """Two far-apart blobs, 80 stored prototypes each (a conformal p-value cannot fall below 1/(n+1), so fewer than 50 prototypes could never flag anything at q = 0.02),
    and 20 FRESH test inputs from each blob."""
    g = np.random.default_rng(0)
    d = 12
    draw = lambda i, n: torch.from_numpy(unit(np.eye(d)[i] * 8 + g.standard_normal((n, d))))              # noqa: E731
    pool = _hand_pool([draw(0, 80), draw(1, 80)], **kw)
    return pool, draw(0, 20), draw(1, 20)


def _stream_pool(**kw):
    st = function_regimes(100, overlap=0.6, n_regimes=4, n_train=128, n_test=64)
    cfg = PhyloCfg(seed=7, d=st.d, n_classes=st.n_classes, inner_steps=1, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf", set_gate=False, group_vote=False, **kw)
    return st, PhyloMoE(cfg)


# ------------------------------------------------------------------------------------------------ the flag off
def test_with_the_router_off_the_pool_is_bit_identical_to_the_recorded_golden_run():
    spec = importlib.util.spec_from_file_location("golden", ROOT / "research" / "growmoe_results" / "rv2c" / "golden.py")
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    assert golden.differences(golden.QUICK) == []


def test_the_router_never_influences_training():
    st, off = _stream_pool(router=False)
    _, on = _stream_pool(router=True)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)                                                                       # one thread, so equal means bit-equal (as the golden check does)
    try:
        for x, y, ctx in st.algo_batches(64, 7):
            ro, rn = off.observe(x, y, ctx), on.observe(x, y, ctx)
            assert ro.assigned.tolist() == rn.assigned.tolist() and ro.novel.tolist() == rn.novel.tolist()
    finally:
        torch.set_num_threads(threads)
    assert sorted(off.experts) == sorted(on.experts)
    for e in off.experts:
        for a, b in zip(off.experts[e].module.parameters(), on.experts[e].module.parameters()):
            assert torch.equal(a, b)
    assert [(e.kind, e.step) for e in off.events] == [(e.kind, e.step) for e in on.events]          # nothing is fitted inside observe, so not even an event differs


# ------------------------------------------------------------------------------------------------ selection and routing
def test_with_the_router_on_and_density_selected_the_answers_equal_the_shipped_path():
    X, y = antipodal(seed=0)
    Xs = _split(X, y, 4)
    off, on = _hand_pool(Xs), _hand_pool(Xs, router=True)
    Xt, _ = antipodal(seed=1)
    a, b = off.predict(Xt), on.predict(Xt)
    assert on._current_route_rule().name == "density"
    assert torch.allclose(a.probs, b.probs, atol=1e-6) and torch.equal(a.expert, b.expert) and torch.equal(a.unknown, b.unknown) and torch.allclose(a.p_nov, b.p_nov, atol=1e-6)


def test_with_lda_selected_the_router_routes_better_than_density_and_leaves_novelty_alone():
    X, y = aniso(seed=0)
    Xs = _split(X, y, 6)
    off, on = _hand_pool(Xs), _hand_pool(Xs, router=True)
    Xt, yt = aniso(seed=1)
    a, b = off.predict(Xt), on.predict(Xt)
    assert on._current_route_rule().name == "lda"
    assert float((b.expert == yt).float().mean()) > float((a.expert == yt).float().mean()) + 0.2
    assert torch.equal(a.unknown, b.unknown) and torch.allclose(a.p_nov, b.p_nov, atol=1e-6)         # the density gate still decides known or unknown
    assert on.events.of_kind("router_refit")[-1].fields["selected"] == "lda"
    assert on.events.of_kind("router_switch")[-1].fields == {"from": "density", "to": "lda"}


def test_restrict_never_routes_to_an_expert_the_gate_rejects_and_free_does():
    pool, a, _ = _two_blob_pool(router=True)
    pool._current_route_rule().set("lda", _Prefer(2), [0, 1], {}, 80)                              # force a rule that always prefers expert 1
    xa = a[:20]
    pool.cfg.router_mode = "restrict"
    assert pool.predict(xa).expert.tolist() == [0] * 20                                           # blob A is accepted by expert 0 only
    pool.cfg.router_mode = "free"
    assert pool.predict(xa).expert.tolist() == [1] * 20


def test_restrict_falls_back_to_the_best_rule_score_when_the_gate_accepts_nobody():
    pool, _, _ = _two_blob_pool(router=True)
    pool._current_route_rule().set("lda", _Prefer(2), [0, 1], {}, 80)
    far = torch.from_numpy(unit(-np.eye(12)[5][None, :] + 0.01 * np.random.default_rng(1).standard_normal((6, 12))))
    out = pool.predict(far)
    assert bool(out.unknown.all()) and out.expert.tolist() == [1] * 6


def test_a_pool_with_one_expert_or_a_young_expert_keeps_the_density_rule():
    X, y = aniso(seed=0)
    one = _hand_pool(_split(X, y, 6)[:1], router=True)
    young = _hand_pool(_split(X, y, 6), router=True, router_min_protos=1000)
    Xt, _ = aniso(seed=1)
    for pool in (one, young):
        assert pool._current_route_rule().name == "density"
        assert pool.predict(Xt).expert.shape == (len(Xt),)


def test_an_empty_batch_is_handled_exactly_like_the_shipped_path():
    X, y = aniso(seed=0)
    Xs = _split(X, y, 6)
    results = []
    for pool in (_hand_pool(Xs), _hand_pool(Xs, router=True)):
        try:
            results.append(("ok", tuple(pool.predict(torch.empty(0, 12)).expert.shape)))
        except Exception as exc:                                                                  # noqa: BLE001 - the point is that the router adds no NEW way to fail
            results.append(("error", type(exc)))
    assert results[0] == results[1]


def test_a_router_with_a_reference_route_rule_is_refused_at_predict_time():
    pool, a, _ = _two_blob_pool(router=True)
    pool.cfg.route_rule = "default"                                                                # changed after construction: PhyloCfg cannot catch it
    with pytest.raises(ValueError):
        pool.predict(a[:4])


def test_group_answering_follows_the_router():
    pool, a, _ = _two_blob_pool(router=True, router_mode="free", group_vote=True, group_veto=False, group_min=4)
    pool._current_route_rule().set("lda", _Prefer(2), [0, 1], {}, 80)
    out = pool.predict_grouped(a[:10], torch.zeros(10, dtype=torch.long))
    assert out.expert.tolist() == [1] * 10


# ------------------------------------------------------------------------------------------------ lazy refit
def test_staleness_follows_the_expert_set_the_dirty_mark_and_the_age():
    X, y = aniso(seed=0)
    pool = _hand_pool(_split(X, y, 6)[:3], router=True, router_refit_every=5)
    assert pool._route_is_stale()                                                                  # never fitted
    pool._current_route_rule()
    assert not pool._route_is_stale() and pool._route.eids == [0, 1, 2]
    n = len(pool.events.of_kind("router_refit"))
    pool._current_route_rule()
    assert len(pool.events.of_kind("router_refit")) == n                                           # nothing changed: no refit
    pool._route_age = 4
    assert not pool._route_is_stale()
    pool._route_age = 5
    assert pool._route_is_stale()
    pool._current_route_rule()
    assert pool._route_age == 0 and len(pool.events.of_kind("router_refit")) == n + 1
    pool._route_dirty = True
    assert pool._route_is_stale()
    pool._current_route_rule()
    assert not pool._route_dirty
    mod = MLPExpert(d_in=12, hidden=8, depth=1, n_classes=K, seed=9)
    sig = VmfSignature(d=12, max_protos=256, seed=9)
    sig.update(_split(X, y, 6)[3])
    sig.fit()
    pool.add_expert(mod, parent=None, state="active", meta={"sig": sig})
    assert pool._route_is_stale()                                                                  # a new expert
    pool._current_route_rule()
    assert pool._route.eids == [0, 1, 2, 3]


def test_a_spawn_marks_the_router_stale_and_the_next_predict_refits_it():
    st, pool = _stream_pool(router=True)
    x_t = next(iter(st.test_sets().values()))[0]
    seen_spawn = False
    for x, y, ctx in st.algo_batches(64, 7):
        before = len(pool.experts)
        pool.observe(x, y, ctx)
        if len(pool.experts) > before >= 1:
            seen_spawn = True
            assert pool._route_dirty
            pool.predict(x_t)
            assert not pool._route_dirty and pool._route.eids == sorted(pool.experts)
            assert pool.events.of_kind("router_refit")[-1].fields["experts"] == len(pool.experts)
    assert seen_spawn


# ------------------------------------------------------------------------------------------------ accounting, determinism, pickling
def test_accounting_counts_the_parameters_and_multiply_adds_of_the_selected_rule():
    X, y = aniso(seed=0)
    Xs = _split(X, y, 6)
    on, off = _hand_pool(Xs, router=True), _hand_pool(Xs)
    on._current_route_rule()
    assert on._route.name == "lda"
    r_on, r_off = pool_report(on, 12), pool_report(off, 12)
    k, d = 6, 12
    assert r_on.params - r_off.params == k * d + k
    assert r_on.gate_macs - r_off.gate_macs == k * d
    assert r_on.memory_bytes - r_off.memory_bytes == 4 * (k * d + k)
    X2, y2 = antipodal(seed=0)
    dense = _hand_pool(_split(X2, y2, 4), router=True)
    dense._current_route_rule()
    ref = _hand_pool(_split(X2, y2, 4))
    assert pool_report(dense, 16) == pool_report(ref, 16)                                          # density selected: not a byte more


def test_two_identical_pools_select_and_route_identically():
    X, y = aniso(seed=0)
    a, b = _hand_pool(_split(X, y, 6), router=True), _hand_pool(_split(X, y, 6), router=True)
    Xt, _ = aniso(seed=1)
    pa, pb = a.predict(Xt), b.predict(Xt)
    assert a._route.name == b._route.name and a._route.cv == pytest.approx(b._route.cv)
    assert torch.equal(pa.expert, pb.expert) and torch.allclose(pa.probs, pb.probs, atol=1e-6)


def test_a_pool_that_holds_a_fitted_router_can_be_pickled_and_copied():
    X, y = aniso(seed=0)
    pool = _hand_pool(_split(X, y, 6), router=True)
    Xt, _ = aniso(seed=1)
    ref = pool.predict(Xt).expert
    assert pool._route.name == "lda"
    for clone in (pickle.loads(pickle.dumps(pool)), copy.deepcopy(pool)):
        assert torch.equal(clone.predict(Xt).expert, ref)
```

- [ ] **Step 2.6: Run and watch it fail**

Run: `python -m pytest tests/growmoe/test_gm_rv4_router_pool.py -q -p no:cacheprovider`
Expected: FAIL (`AttributeError: ... '_current_route_rule'`, `EventLog` has no `since`, ...). The first test (golden) passes.

- [ ] **Step 2.7: Implement the integration**

(a) `growmoe/telemetry.py`, in `EventLog`, after `of_kind`:

```python
    def since(self, n: int) -> list[Event]:
        """The events added after the first ``n`` (the log only grows)."""
        return self._events[n:]
```

(b) `growmoe/pools/phylo.py`: add to the imports (after `from ..replay import Reservoir`):

```python
from ..router import FITS, RouteRule, cv_owner_accuracy, select_rule
```

In `__init__`, after `self._held_step = -10`, add:

```python
        self._route_state()                                            # routing v2 stage 4a; idle unless cfg.router
```

(c) At the very top of `observe`, directly after the docstring line, add:

```python
        router_mark = len(self.events) if self.cfg.router else 0       # routing v2 stage 4a: where this call's events start
```

and directly before `mean_loss = float(np.mean(losses)) if losses else float("nan")` (after the `# 13h Distillation merge` block) add:

```python
        if self.cfg.router:                                            # routing v2 stage 4a: count the call, mark the router stale on a structural change; nothing is fitted here
            self._router_after_observe(router_mark)
```

(d) In `predict`, replace

```python
        if route_rule not in self._FAST_RULES or len(self.experts) == 0:
            return self._predict_reference(x, ctx)
```

with

```python
        if route_rule not in self._FAST_RULES or len(self.experts) == 0:
            if self.cfg.router and route_rule not in self._FAST_RULES and len(self.experts) > 0:
                raise ValueError(f"cfg.router needs a fast route_rule, got {route_rule!r}: the reference path is router-blind")
            return self._predict_reference(x, ctx)
```

and, directly before `routed_expert = torch.tensor(eids, dtype=torch.long)[idx]`, add:

```python
        if self.cfg.router:                                                             # routing v2 stage 4a: the selected rule replaces the choice among experts, never the known-or-unknown decision
            idx = self._route_by_router(x, eids, P, LogF, idx)
```

(e) New methods, placed right before the `# ---- routing v2 stage 3a: the batch-level gate` comment block (that is, after `observe`):

```python
    # ------------------------------------------------------------------------------------------ routing v2 stage 4a: the router
    _ROUTER_EVENTS = ("spawn", "extend", "merge", "archive")

    def _route_state(self) -> RouteRule:
        """The router's state, created on first use (a pool pickled before stage 4 has none)."""
        if not hasattr(self, "_route"):
            self._route, self._route_age, self._route_dirty = RouteRule(), 0, False
        return self._route

    def _router_after_observe(self, mark: int) -> None:
        """Count the observe call; mark the router stale when an expert was spawned, extended, merged or archived. The router never influences training, so nothing is fitted
        here: the next predict that finds the router stale refits it (lazily)."""
        self._route_state()
        self._route_age += 1
        if any(e.kind in self._ROUTER_EVENTS for e in self.events.since(mark)):
            self._route_dirty = True

    def _route_is_stale(self) -> bool:
        rule = self._route_state()
        return rule.eids is None or rule.eids != sorted(self.experts) or self._route_dirty or self._route_age >= self.cfg.router_refit_every

    def _current_route_rule(self) -> RouteRule:
        if self._route_is_stale():
            self._refit_router()
        return self._route

    def _refit_router(self) -> None:
        """Cross-validate the candidates on the experts' stored prototypes (owner = the expert that holds the prototype), select a rule (density unless another is clearly better),
        fit the selected scorer on all of them. With fewer than two experts, or an expert holding fewer than ``router_min_protos`` prototypes, the density rule stays."""
        cfg, rule = self.cfg, self._route_state()
        eids = sorted(self.experts)
        counts = [self.experts[e].meta["sig"].n for e in eids]
        before = rule.name
        if len(eids) < 2 or min(counts) < cfg.router_min_protos:
            rule.set("density", None, eids, {}, sum(counts))
        else:
            X = torch.cat([self.experts[e].meta["sig"].protos for e in eids])
            y = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(counts)])
            names = tuple(n for n in cfg.router_candidates if not (n == "softmax" and len(eids) > cfg.router_softmax_max_experts))
            cv = cv_owner_accuracy(X, y, len(eids), cfg.router_folds, generator(cfg.seed, 0x52544F, self.step), names)
            chosen = select_rule(cv, before, cfg.router_margin)
            scorer = None if chosen == "density" else FITS[chosen](X, y, len(eids), generator(cfg.seed, 0x52544C, self.step))
            rule.set(chosen, scorer, eids, cv, int(X.shape[0]))
        self._route_age, self._route_dirty = 0, False
        self.events.add(kind="router_refit", step=self.step, selected=rule.name, cv=rule.cv, rows=rule.rows, experts=len(eids))
        if rule.name != before:
            self.events.add(kind="router_switch", step=self.step, **{"from": before, "to": rule.name})

    def _route_by_router(self, x: Tensor, eids: List[int], P: Tensor, LogF: Tensor, idx: Tensor) -> Tensor:
        """``idx`` is the density gate's choice (a column per input). With the rule 'density' selected it is returned untouched (today's path, bit for bit); with another rule the
        scores S = scorer(x) replace it: mode 'restrict' takes the best S among the experts the gate accepts (the best S overall when it accepts none), 'free' the best S overall."""
        if x.shape[0] == 0:
            return idx
        rule = self._current_route_rule()
        if rule.name == "density":
            return idx
        col = {e: j for j, e in enumerate(rule.eids)}
        S = rule.scorer(x)[:, [col[e] for e in eids]]                                   # columns in the order of ``eids``
        if self.cfg.router_mode == "free":
            return S.argmax(dim=1)
        accepted = (LogF >= 0.0) if self.cfg.no_conformal else (P.double() >= float(self.cfg.q))
        best_accepted = S.masked_fill(~accepted, float("-inf")).argmax(dim=1)
        return torch.where(accepted.any(dim=1), best_accepted, S.argmax(dim=1))

    def extra_cost(self, d: int):
        """(parameters, extra multiply-adds per prediction, extra stored scalars) of the stage-4 machinery, read by ``accounting.pool_report``."""
        params = macs = scalars = 0
        if self.cfg.router and hasattr(self, "_route") and self._route.eids is not None:
            p, m, s = self._route.cost(d)
            params, macs, scalars = params + p, macs + m, scalars + s
        return params, macs, scalars
```

(f) `growmoe/accounting.py`: add the field and use the hook. In `PoolReport` add after `naive_params`:

```python
    extra_scalars: int = 0         # stored scalars that are not vectors (reference losses of the transfer test); 4 bytes each
```

Change `_report` to

```python
def _report(n_experts: int, params: int, stored: int, d: int, gate: int, expert: int, naive: int = None, scalars: int = 0) -> PoolReport:
    return PoolReport(n_experts=n_experts, params=params, stored_vectors=stored,
                      memory_bytes=BYTES_PER_FLOAT * params + BYTES_PER_FLOAT * d * stored + BYTES_PER_FLOAT * scalars,
                      gate_macs=gate, expert_macs=expert, predict_macs=gate + expert, naive_params=naive, extra_scalars=scalars)
```

and in `pool_report`'s expert-registry branch replace the final two lines

```python
        expert = int(round(sum(weight_macs(r.module) for r in recs) / len(recs)))
        naive = sum(p.numel() for r in recs for p in r.module.parameters())
        return _report(len(recs), unique_param_count([r.module for r in recs]), stored, d, gate, expert, naive)
```

with

```python
        expert = int(round(sum(weight_macs(r.module) for r in recs) / len(recs)))
        naive = sum(p.numel() for r in recs for p in r.module.parameters())
        x_params = x_macs = x_scalars = 0
        extra = getattr(pool, "extra_cost", None)                                      # routing v2 stage 4: router parameters, reference scalars, a live growth guard
        if extra is not None:
            x_params, x_macs, x_scalars = extra(d)
        return _report(len(recs), unique_param_count([r.module for r in recs]) + x_params, stored, d, gate + x_macs, expert, naive + x_params, x_scalars)
```

- [ ] **Step 2.8: Run the new tests, then the neighbours and the golden check**

Run: `python -m pytest tests/growmoe/test_gm_rv4_router.py tests/growmoe/test_gm_rv4_router_pool.py -q -p no:cacheprovider`
Expected: all pass. A test that depends on the real gate (`test_restrict_*`) may need its blobs moved; fix the SETTING and say why in the docstring.
Run: `python -m pytest tests/growmoe/test_gm_m0_fast_predict.py tests/growmoe/test_gm_m0_accounting.py tests/growmoe/test_gm_rv3_pool.py tests/growmoe/test_gm_phylo.py tests/growmoe/test_gm_telemetry.py -q -p no:cacheprovider`
Expected: unchanged green.
Run: `python research/growmoe_results/rv2c/golden.py check --quick`
Expected: `bit-identical`.

- [ ] **Step 2.9: Commit**

```bash
git add growmoe/router.py growmoe/telemetry.py growmoe/pools/phylo.py growmoe/accounting.py tests/growmoe/rv4_helpers.py tests/growmoe/test_gm_rv4_router.py tests/growmoe/test_gm_rv4_router_pool.py
git commit -m "feat(growmoe): stage 4a - a router that chooses its own rule by cross-validation (flag router, off by default)"
```

---

## Task 3: Stage 4b - spawn or extend by a held-out transfer test

**Files:**
- Create: `growmoe/transfer.py`
- Modify: `growmoe/pools/phylo.py` (`_handle_cluster`, `observe`, `_attempt_merges`, new methods, `extra_cost`)
- Modify: `tests/growmoe/rv4_helpers.py` (learnable regimes)
- Create: `tests/growmoe/test_gm_rv4_transfer.py` (pure), `tests/growmoe/test_gm_rv4_transfer_pool.py` (pool, merge audit)

**Interfaces:**
- Consumes: `PhyloCfg.spawn_rule`, `transfer_alpha`, `transfer_margin`, `transfer_perms`, `transfer_ref`, `transfer_min_ref`, `spawn_pretrain_steps` (task 1); `growmoe.settest.max_exceedances`, `generator`; `PhyloMoE.extra_cost` (task 2).
- Produces:
  - `growmoe.transfer.cluster_conflict(cluster_losses, ref_losses, alpha, n_perm, margin, gen, chunk=64) -> (conflict: bool, p: float, diff: float)`; `TransferVerdict(parent, conflict, p, diff, loss_cluster, loss_ref, margin, n_cluster, n_ref)` with `fields() -> dict` (keys `p_conflict`, `loss_cluster`, `loss_ref`, `margin`, `n_ref`).
  - `PhyloMoE._transfer_reference(rec, exclude) -> list`, `_transfer_verdict(x_taken, y_taken, force_parent) -> Optional[TransferVerdict]`, `_extend_unchanged(...)`, `_pretrain_child(...)`, `_merge_refs(rec_i, rec_j)`; per-expert `rec.meta["ref"]` (a `deque(maxlen=transfer_ref)` of honest per-sample losses, created when first fed); events `extend` and `spawn` carry the verdict's fields when the rule decided.
  - helpers in `tests/growmoe/rv4_helpers.py`: `D`, `C`, `teacher(seed)`, `regime_batch(W, n, seed, perm=0)`, `trained_pool(n_batches=40, n_classes=C, **cfg)`.

### 3a. The pure test

- [ ] **Step 3.1: Write the failing tests**

Create `tests/growmoe/test_gm_rv4_transfer.py`:

```python
"""RV4 stage 4b, the pure test: does the best existing expert fail on the new cluster's labels? (spec section 5.1)"""

import math

import pytest
import torch

from growmoe.settest import generator
from growmoe.transfer import cluster_conflict


def test_false_conflicts_stay_at_or_below_alpha_under_exchangeability():
    false = 0
    for t in range(400):
        g = torch.Generator().manual_seed(1000 + t)
        c, r = torch.randn(32, generator=g).abs(), torch.randn(256, generator=g).abs()                  # one distribution
        conflict, _, _ = cluster_conflict(c, r, 0.05, 199, 0.0, generator(7, t))
        false += conflict
    assert false <= 34                                                                                  # Binomial(400, 0.05): mean 20, its 99.9th percentile is about 34


def test_a_large_shift_is_called_a_conflict():
    g = torch.Generator().manual_seed(1)
    c, r = 2.0 + torch.rand(32, generator=g), torch.rand(256, generator=g)
    conflict, p, diff = cluster_conflict(c, r, 0.01, 999, 0.25, generator(1))
    assert conflict and p <= 0.01 and diff > 1.4


def test_the_margin_blocks_a_small_but_significant_shift():
    g = torch.Generator().manual_seed(2)
    r = torch.randn(512, generator=g).abs()
    c = torch.randn(600, generator=g).abs() + 0.15                                                       # +0.15 nats at n = 600 is significant ...
    open_, p_open, _ = cluster_conflict(c, r, 0.01, 999, 0.0, generator(2))
    shut, p_shut, _ = cluster_conflict(c, r, 0.01, 999, 0.25, generator(2))
    assert open_ and not shut and p_open == p_shut                                                       # ... but not practically significant; only the margin differs


def test_a_cluster_that_is_easier_than_the_reference_is_never_a_conflict():
    g = torch.Generator().manual_seed(3)
    r = 1.0 + torch.rand(256, generator=g)
    conflict, p, diff = cluster_conflict(r[:40] - 0.8, r, 0.01, 999, 0.0, generator(3))
    assert not conflict and p > 0.5 and diff < 0


def test_non_finite_cluster_losses_are_a_conflict():
    for bad in (float("inf"), float("nan")):
        conflict, p, diff = cluster_conflict(torch.tensor([bad, 1.0]), torch.rand(64), 0.01, 999, 0.25, generator(1))
        assert conflict and p == 0.0 and diff == math.inf


def test_the_p_value_is_an_integer_count_over_n_perm_plus_one_and_the_call_is_deterministic():
    g = torch.Generator().manual_seed(4)
    c, r = torch.randn(32, generator=g).abs(), torch.randn(128, generator=g).abs()
    a = cluster_conflict(c, r, 0.5, 99, 0.0, generator(3))
    assert a == cluster_conflict(c, r, 0.5, 99, 0.0, generator(3))
    assert abs(a[1] * 100 - round(a[1] * 100)) < 1e-9


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError):
        cluster_conflict(torch.empty(0), torch.rand(8), 0.01, 999, 0.0, generator(1))
    with pytest.raises(ValueError):
        cluster_conflict(torch.rand(8), torch.empty(0), 0.01, 999, 0.0, generator(1))
    with pytest.raises(ValueError):
        cluster_conflict(torch.rand(8), torch.rand(8), 0.001, 99, 0.0, generator(1))                      # 1/(1+99) > 0.001: could never reject
    with pytest.raises(ValueError):
        cluster_conflict(torch.rand(8), torch.tensor([1.0, float("inf")]), 0.01, 999, 0.0, generator(1))   # a reference must be finite
```

- [ ] **Step 3.2: Run and watch it fail**

Run: `python -m pytest tests/growmoe/test_gm_rv4_transfer.py -q -p no:cacheprovider`
Expected: collection error `ModuleNotFoundError: No module named 'growmoe.transfer'`.

- [ ] **Step 3.3: Implement `growmoe/transfer.py`**

```python
"""Routing v2, stage 4b: the spawn-or-extend question asked of the labels (spec docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md, section 5).

Today a 60-step look-ahead on samples the networks have memorised decides whether a new cluster is a new function or a harder part of an old one; it is close to a coin flip.
The plain question is: does the best existing expert already predict the new cluster's labels? Its zero-shot loss on the cluster is compared with its own honest prequential losses
(its reference). The comparison is an exact one-sided permutation test (the cluster's mean loss is larger than the reference's) with a practical-significance margin, because inside an
expert that holds several operators the harder one is slightly worse than the pooled reference and a bare test over-fires at large n (spike B, E-032).

p = (1 + c) / (1 + n_perm) with c = permutations at least as extreme as the observed split (Phipson and Smyth 2010), valid at every level. All ``n_perm`` permutations are drawn
(a call costs about ten milliseconds and happens once per cluster), so the reported p is the full exact value; the spec's early stop was dropped on purpose (refinement R-5):
it would turn p into a lower bound in the event log.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from .settest import max_exceedances


def cluster_conflict(cluster_losses: Tensor, ref_losses: Tensor, alpha: float, n_perm: int, margin: float, gen: torch.Generator, chunk: int = 64) -> Tuple[bool, float, float]:
    """``(conflict, p, diff)``: ``diff`` = mean cluster loss - mean reference loss; ``conflict`` = (p <= alpha and diff > margin). Cluster losses that are not finite (labels the
    expert has never seen) are a conflict outright (p = 0)."""
    c = torch.as_tensor(cluster_losses, dtype=torch.float64).flatten()
    r = torch.as_tensor(ref_losses, dtype=torch.float64).flatten()
    if c.numel() < 1 or r.numel() < 1:
        raise ValueError("the cluster and the reference must each hold at least one loss")
    if not bool(torch.isfinite(r).all()):
        raise ValueError("reference losses must be finite")
    allowed = max_exceedances(alpha, n_perm)
    if allowed < 0:
        raise ValueError(f"n_perm={n_perm} is too small for alpha={alpha}: the smallest attainable p-value 1/(1+n_perm) exceeds alpha, so the test could never reject")
    if not bool(torch.isfinite(c).all()):
        return True, 0.0, math.inf
    pooled = torch.cat([c, r])
    n_c, n_all = c.numel(), pooled.numel()
    total = pooled.sum()

    def stat(idx: Tensor) -> Tensor:                                    # idx (m, n_c): rows of the pooled sample forming the cluster
        cm = pooled[idx].mean(1)
        return cm - (total - cm * n_c) / (n_all - n_c)

    obs = float(stat(torch.arange(n_c)[None, :])[0])
    exceed = 0
    for lo in range(0, n_perm, chunk):
        m = min(chunk, n_perm - lo)
        exceed += int((stat(torch.rand(m, n_all, generator=gen).argsort(1)[:, :n_c]) >= obs).sum())
    p = (1 + exceed) / (1 + n_perm)
    diff = float(c.mean() - r.mean())
    return bool(p <= alpha and diff > margin), p, diff


@dataclass
class TransferVerdict:
    parent: int          # the expert with the lowest zero-shot loss on the cluster
    conflict: bool
    p: float
    diff: float
    loss_cluster: float
    loss_ref: float
    margin: float
    n_cluster: int
    n_ref: int

    def fields(self) -> dict:
        """What the ``spawn`` and ``extend`` events carry."""
        return {"p_conflict": self.p, "loss_cluster": self.loss_cluster, "loss_ref": self.loss_ref, "margin": self.margin, "n_ref": self.n_ref}
```

- [ ] **Step 3.4: Run and see it pass**

Run: `python -m pytest tests/growmoe/test_gm_rv4_transfer.py -q -p no:cacheprovider`
Expected: all pass (about 10 s).

### 3b. Pool integration

- [ ] **Step 3.5: Extend the shared helpers and write the failing pool tests**

Append to `tests/growmoe/rv4_helpers.py` (add the imports at the top of the file):

```python
import torch.nn.functional as F

from growmoe.config import PhyloCfg
from growmoe.pools.phylo import PhyloMoE

D, C = 16, 4


def teacher(seed):
    """A random linear teacher: label = argmax(x @ W)."""
    return torch.randn(D, C, generator=torch.Generator().manual_seed(seed))


def regime_batch(W, n, seed, perm=0):
    """n inputs uniform on the sphere, labelled by the teacher; ``perm`` shifts every label (the same inputs under a different function).
    A scratch run on 2026-10-15 (hidden 32, 5 inner steps, 40 batches of 64): accuracy 0.81, loss 0.45 on the same function and 4.9 under perm=1 (chance is 1.39)."""
    x = F.normalize(torch.randn(n, D, generator=torch.Generator().manual_seed(seed)), dim=1)
    return x, ((x @ W).argmax(1) + perm) % C


def trained_pool(n_batches=40, n_classes=C, **kw):
    """A pool with ONE expert that learned the teacher; returns (pool, W)."""
    cfg = dict(d=D, hidden=32, n_classes=n_classes, seed=3, q=0.02, alpha_spawn=0.01, m_min=32, inner_steps=5, route_rule="accept_then_logf", set_gate=False, group_vote=False,
               spawn_rule="transfer")
    cfg.update(kw)
    pool, W = PhyloMoE(PhyloCfg(**cfg)), teacher(1)
    for k in range(n_batches):
        pool.observe(*regime_batch(W, 64, 100 + k))
    return pool, W
```

Create `tests/growmoe/test_gm_rv4_transfer_pool.py`:

```python
"""RV4 stage 4b inside PhyloMoE: references, the verdict, extend without touching a weight, spawn with a pretrained child, merge (spec sections 5.2, 5.4, 5.5)."""

import copy
import math
from collections import deque

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from growmoe.accounting import pool_report
from growmoe.config import PhyloCfg
from growmoe.experts import MLPExpert
from growmoe.pools.phylo import PhyloMoE
from growmoe.signature import VmfSignature
from rv4_helpers import D, C, regime_batch, trained_pool, unit


@pytest.fixture(scope="module")
def base():
    pool, W = trained_pool()
    return pool, W


def _fresh(base):
    return copy.deepcopy(base[0]), base[1]


# ------------------------------------------------------------------------------------------------ the references
def test_the_reference_holds_honest_losses_of_a_learned_expert_and_is_capped(base):
    pool, _ = _fresh(base)
    ref = pool.experts[0].meta["ref"]
    assert len(ref) == 512 and ref.maxlen == 512
    assert all(math.isfinite(v) for v in ref) and sum(ref) / len(ref) < math.log(C)                     # below chance: it learned


def test_with_the_probe_rule_no_reference_is_kept_and_no_transfer_field_is_logged():
    pool, W = trained_pool(n_batches=6, spawn_rule="probe")
    assert all("ref" not in r.meta for r in pool.experts.values())
    pool._handle_cluster(*regime_batch(W, 48, 9001, perm=1), evidence=50.0)
    assert not any("p_conflict" in e.fields for e in pool.events)


def test_infants_do_not_feed_the_reference_and_actives_do():
    """Hand-built pool: expert 0 active on blob A, expert 1 an infant on blob B (untrained; only the bookkeeping is under test)."""
    d = 12
    g = np.random.default_rng(0)
    draw = lambda i, n: torch.from_numpy(unit(np.eye(d)[i] * 8 + g.standard_normal((n, d))))              # noqa: E731
    labels = lambda s: torch.randint(0, 4, (64,), generator=torch.Generator().manual_seed(s))
    pool = PhyloMoE(PhyloCfg(d=d, hidden=8, n_classes=4, seed=1, q=0.02, m_min=32, inner_steps=1, route_rule="accept_then_logf", set_gate=False, group_vote=False, spawn_rule="transfer"))
    for i, state in ((0, "active"), (1, "infant")):
        mod = MLPExpert(d_in=d, hidden=8, depth=1, n_classes=4, seed=i)
        sig = VmfSignature(d=d, max_protos=256, seed=i)
        sig.update(draw(i, 80))
        sig.fit()
        pool.add_expert(mod, parent=None if i == 0 else 0, state=state, meta={"sig": sig, "infant_calls_seen": 0 if state == "infant" else pool.cfg.infant_calls})
    for k in range(4):                                                  # infancy lasts infant_calls = 5 observe calls: at the start of the 5th the expert is active
        pool.observe(draw(1, 64), labels(k))
        assert pool.experts[1].state == "infant" and "ref" not in pool.experts[1].meta
    pool.observe(draw(1, 64), labels(9))
    assert pool.experts[1].state == "active" and len(pool.experts[1].meta["ref"]) > 0
    pool.observe(draw(0, 64), labels(10))
    assert len(pool.experts[0].meta["ref"]) > 0


# ------------------------------------------------------------------------------------------------ extend and spawn
def test_a_cluster_the_parent_already_predicts_is_absorbed_without_touching_a_weight(base):
    pool, W = _fresh(base)
    rec = pool.experts[0]
    before = {k: v.clone() for k, v in rec.module.state_dict().items()}
    seen_before, replay_before = rec.meta["sig"].seen, rec.replay.n
    x, y = regime_batch(W, 48, 9000)
    assert pool._handle_cluster(x, y, evidence=50.0) is None and sorted(pool.experts) == [0]
    ev = pool.events.of_kind("extend")[-1]
    assert ev.fields["p_conflict"] > 0.01 and ev.fields["n_ref"] == 512 and "loss_cluster" in ev.fields
    assert all(torch.equal(before[k], v) for k, v in rec.module.state_dict().items())
    assert rec.meta["sig"].seen == seen_before + 48 and rec.replay.n >= replay_before


def test_a_cluster_whose_labels_the_parent_gets_wrong_is_spawned_and_pretrained(base):
    pool, W = _fresh(base)
    x, y = regime_batch(W, 48, 9001, perm=1)
    assert pool._handle_cluster(x, y, evidence=50.0) == 1
    assert sorted(pool.experts) == [0, 1] and pool.experts[1].parent == 0
    f = pool.events.of_kind("spawn")[-1].fields
    assert f["p_conflict"] <= 0.01 and f["loss_cluster"] > f["loss_ref"] + 0.35 and f["margin"] == pytest.approx(0.25 * math.log(C))
    with torch.no_grad():
        child, parent = F.cross_entropy(pool.experts[1].module(x), y), F.cross_entropy(pool.experts[0].module(x), y)
    assert float(child) < float(parent)                                                                 # the pretraining steps taught the child the new labels


def test_the_novel_samples_near_the_parent_are_handed_to_it_on_an_extend(base):
    pool, W = _fresh(base)
    x, y = regime_batch(W, 64, 9002)
    assigned = torch.full((64,), -1, dtype=torch.long)
    pool._handle_cluster(x[:48], y[:48], evidence=50.0, novel_indices=torch.arange(64), x_all=x, assigned=assigned)
    assert pool.events.of_kind("extend") and int((assigned == 0).sum()) > 32


def test_an_unlabelled_cluster_is_handled_by_the_path_of_today(base):
    pool, W = _fresh(base)
    twin = copy.deepcopy(pool)
    twin.cfg.spawn_rule = "probe"
    x, _ = regime_batch(W, 48, 9003)
    y = torch.full((48,), -1)
    assert pool._handle_cluster(x, y, evidence=50.0) == twin._handle_cluster(x, y, evidence=50.0)
    assert [e.kind for e in pool.events] == [e.kind for e in twin.events]


def test_a_parent_with_too_few_reference_losses_falls_back_to_the_probe():
    pool, W = trained_pool(n_batches=1)                                  # the founding batch never feeds the reference
    assert "ref" not in pool.experts[0].meta
    twin = copy.deepcopy(pool)
    twin.cfg.spawn_rule = "probe"
    x, y = regime_batch(W, 48, 9004)
    assert pool._handle_cluster(x, y, evidence=50.0) == twin._handle_cluster(x, y, evidence=50.0)
    assert [e.kind for e in pool.events] == [e.kind for e in twin.events]
    assert not any("p_conflict" in e.fields for e in pool.events)


def test_labels_the_parent_has_never_seen_are_a_conflict_with_finite_event_fields(tmp_path):
    pool, W = trained_pool(n_batches=20, n_classes=6)                    # label space of 6, the teacher uses 0-3
    x, y = regime_batch(W, 48, 9005)
    y = 4 + (y % 2)                                                      # labels 4 and 5: masked in the parent
    assert pool._handle_cluster(x, y, evidence=50.0) == 1
    f = pool.events.of_kind("spawn")[-1].fields
    assert f["p_conflict"] <= 0.01 and math.isfinite(f["loss_cluster"]) and f["margin"] == pytest.approx(0.25 * math.log(4))
    pool.events.to_jsonl(tmp_path / "events.jsonl")                      # strict JSON (allow_nan=False) must accept every field


def test_the_alarm_path_leaves_the_cluster_itself_out_of_the_reference(base):
    pool, W = _fresh(base)
    rec = pool.experts[0]
    assert pool._transfer_reference(rec, 0) == list(rec.meta["ref"])
    assert pool._transfer_reference(rec, 32) == list(rec.meta["ref"])[:-32]
    x, y = regime_batch(W, 32, 9006)
    assert pool._transfer_verdict(x, y, force_parent=0).n_ref == 512 - 32
    assert pool._transfer_verdict(x, y, force_parent=None).n_ref == 512


def test_the_parent_is_the_expert_with_the_lowest_loss_on_the_cluster(base):
    pool, W = _fresh(base)
    pool._handle_cluster(*regime_batch(W, 48, 9007, perm=1), evidence=50.0)         # expert 1 knows the permuted function
    pool.experts[1].meta["ref"] = deque([0.5] * 100, maxlen=512)                     # give it a reference so that it can be a parent
    x, y = regime_batch(W, 48, 9008, perm=1)
    assert pool._transfer_verdict(x, y, None).parent == 1
    x, y = regime_batch(W, 48, 9009)
    assert pool._transfer_verdict(x, y, None).parent == 0


def test_the_accounting_counts_the_reference_scalars(base):
    pool, _ = _fresh(base)
    twin = copy.deepcopy(pool)
    twin.cfg.spawn_rule = "probe"
    a, b = pool_report(pool, D), pool_report(twin, D)
    n = len(pool.experts[0].meta["ref"])
    assert a.extra_scalars == n and b.extra_scalars == 0 and a.memory_bytes - b.memory_bytes == 4 * n and a.params == b.params


def test_the_same_cluster_gives_the_same_verdict_in_two_identical_pools():
    a, W = trained_pool(n_batches=12)
    b, _ = trained_pool(n_batches=12)
    x, y = regime_batch(W, 48, 9010, perm=1)
    va, vb = a._transfer_verdict(x, y, None), b._transfer_verdict(x, y, None)
    assert (va.parent, va.conflict, va.n_ref) == (vb.parent, vb.conflict, vb.n_ref)
    assert va.loss_cluster == pytest.approx(vb.loss_cluster, rel=1e-4) and va.p == pytest.approx(vb.p)           # two runs may differ in the last bit of a threaded matmul


# ------------------------------------------------------------------------------------------------ merge
def test_a_merge_gives_the_survivor_the_references_of_both(base):
    pool, W = _fresh(base)
    pool._handle_cluster(*regime_batch(W, 48, 9011, perm=1), evidence=50.0)
    a, b = pool.experts[0], pool.experts[1]
    a.meta["ref"], b.meta["ref"] = deque([1.0] * 300, maxlen=512), deque([2.0] * 300, maxlen=512)
    pool._merge_refs(a, b)
    ref = list(a.meta["ref"])
    assert len(ref) == 512 and ref[:212] == [1.0] * 212 and ref[212:] == [2.0] * 300                     # the newest 512 of the union, the second expert's entries last


def _forced_duplicate_then_merge(seed):
    """AUDIT (spec section 5.5): merge has never fired in a benchmark. Here a duplicate of one function is forced (the probe can never accept an extend) and, after the
    infant has settled, the pool's own merge routine is asked to clean it up."""
    pool, W = trained_pool(n_batches=30, seed=seed, spawn_rule="probe", delta_forget=-1.0)
    assert pool._handle_cluster(*regime_batch(W, 48, 9100 + seed), evidence=50.0) == 1 and len(pool.experts) == 2
    for k in range(10):
        pool.observe(*regime_batch(W, 64, 9200 + 10 * seed + k))
    n_before = len(pool.experts)
    pool._attempt_merges()
    return pool, n_before


@pytest.mark.parametrize("seed", [0, 1])
def test_merge_cleans_up_a_forced_duplicate(seed):
    """If this fails the finding is the result: do NOT weaken it. Record the failure in the notebook (FAILURES), mark the test xfail(strict=True) with the measured rate in the
    reason, and make transfer_alpha / transfer_margin stricter in the dev grid of task 5 (a false spawn would then never be cleaned up)."""
    pool, n_before = _forced_duplicate_then_merge(seed)
    assert n_before == 2 and len(pool.experts) == 1 and pool.events.of_kind("merge")
```

- [ ] **Step 3.6: Run and watch it fail**

Run: `python -m pytest tests/growmoe/test_gm_rv4_transfer_pool.py -q -p no:cacheprovider`
Expected: FAIL (`AttributeError: ... '_transfer_reference'` and the like; the probe-rule test passes only if the helpers import; `ImportError` for the helpers means Step 3.5 is incomplete).

- [ ] **Step 3.7: Implement the integration in `growmoe/pools/phylo.py`**

(a) Imports (next to the router import):

```python
from ..transfer import TransferVerdict, cluster_conflict
```

(b) `_handle_cluster`. Replace

```python
        if force_parent is not None:
            parent_eid = force_parent
            aff = self.experts[parent_eid].meta["sig"].affinity(x_taken, seed=self.step)
        else:
            affinities = {}
```

with

```python
        verdict = self._transfer_verdict(x_taken, y_taken, force_parent) if self.cfg.spawn_rule == "transfer" else None      # routing v2 stage 4b
        if verdict is not None:
            parent_eid = verdict.parent                                  # the best predictor of the new labels, not the nearest signature
            aff = self.experts[parent_eid].meta["sig"].affinity(x_taken, seed=self.step)
        elif force_parent is not None:
            parent_eid = force_parent
            aff = self.experts[parent_eid].meta["sig"].affinity(x_taken, seed=self.step)
        else:
            affinities = {}
```

Replace

```python
        extended = False
        trained_child = None

        if has_labels and has_affinity and has_replay:
            # 13e Interference Probe
```

with

```python
        extended = False
        trained_child = None

        if verdict is not None:                                          # routing v2 stage 4b: the labels decide, not a probe
            if not verdict.conflict:
                self._extend_unchanged(parent_eid, aff, x_taken, y_taken, verdict, novel_indices, x_all, assigned)
                return None
            trained_child = self._pretrain_child(parent_rec, aff, next_id, x_taken, y_taken)
        elif has_labels and has_affinity and has_replay:
            # 13e Interference Probe
```

In the spawn event of `_handle_cluster` (the one with `expert=child_rec.id,`), add one line after `lam=lam,`:

```python
                lam=lam,
                **({} if verdict is None else verdict.fields()),
```

(c) `observe`: directly before `if alarm_fired and monitor is not None:` insert

```python
                    if self.cfg.spawn_rule == "transfer" and not settling:      # routing v2 stage 4b: the honest per-sample losses are this expert's reference for the transfer test
                        rec.meta.setdefault("ref", deque(maxlen=self.cfg.transfer_ref)).extend(preq_losses.tolist())

```

(d) `_attempt_merges`: after the `if rec_j.meta.get("windows") is not None:` block (the two lines that merge source windows) insert

```python
                        if self.cfg.spawn_rule == "transfer":              # routing v2 stage 4b: the merged expert keeps the references of both
                            self._merge_refs(rec_i, rec_j)
```

(e) New methods, after `extra_cost`'s neighbours (the router block):

```python
    # ------------------------------------------------------------------------------------------ routing v2 stage 4b: spawn or extend by a held-out transfer test
    _TRANSFER_MIN_CLUSTER = 8                                           # fewer labelled samples than this cannot carry the test

    def _transfer_reference(self, rec, exclude: int) -> list:
        """The expert's honest reference losses, newest last. With ``exclude`` > 0 the newest ``exclude`` entries are left out: on the alarm path the cluster IS those samples."""
        ref = list(rec.meta.get("ref", ()))
        return ref[:-exclude] if exclude > 0 else ref

    def _transfer_verdict(self, x_taken: Tensor, y_taken: Tensor, force_parent: Optional[int]) -> Optional[TransferVerdict]:
        """Score the cluster's labelled samples (no training) with every expert that has at least ``transfer_min_ref`` reference losses (only ``force_parent`` on the alarm path);
        the best one is the parent; ``cluster_conflict`` asks whether it fails on the cluster. None when the test cannot run (the caller then takes today's path)."""
        cfg = self.cfg
        valid = y_taken >= 0
        if int(valid.sum()) < self._TRANSFER_MIN_CLUSTER:
            return None
        xv, yv = x_taken[valid], y_taken[valid]
        exclude = int(len(x_taken)) if force_parent is not None else 0
        best = None
        for eid in ([force_parent] if force_parent is not None else list(self.experts)):
            rec = self.experts[eid]
            ref = self._transfer_reference(rec, exclude)
            if len(ref) < cfg.transfer_min_ref:
                continue
            with torch.no_grad():
                losses = F.cross_entropy(rec.module(xv), yv, reduction="none")
            if best is None or float(losses.mean()) < best[0]:
                best = (float(losses.mean()), eid, losses, ref)
        if best is None:
            return None
        _, eid, losses, ref = best
        margin = cfg.transfer_margin * math.log(max(int(self.experts[eid].module.seen.sum()), 2))
        conflict, p, diff = cluster_conflict(losses, torch.tensor(ref), cfg.transfer_alpha, cfg.transfer_perms, margin, generator(cfg.seed, 0x545246, self.step, eid))
        return TransferVerdict(parent=eid, conflict=conflict, p=p, diff=diff, loss_cluster=float(losses.mean()), loss_ref=float(np.mean(ref)), margin=margin,
                               n_cluster=int(losses.numel()), n_ref=len(ref))

    def _extend_unchanged(self, parent_eid: int, aff: float, x_taken: Tensor, y_taken: Tensor, verdict: TransferVerdict, novel_indices, x_all, assigned) -> None:
        """No conflict: the parent already predicts these labels, so the cluster joins its signature and replay and NO weight changes (it keeps training on the stream as usual)."""
        parent_rec = self.experts[parent_eid]
        valid = y_taken >= 0
        parent_rec.meta["sig"].update(x_taken)
        parent_rec.meta["sig"].fit()
        parent_rec.replay.add(x_taken[valid], y_taken[valid])
        self.events.add(kind="extend", step=self.step, expert=parent_eid, affinity=aff, n_cluster=len(x_taken), **verdict.fields())
        if novel_indices is not None and len(novel_indices) > 0 and x_all is not None and assigned is not None:
            near = parent_rec.meta["sig"].pvalue(x_all[novel_indices]) >= self.cfg.q
            if near.any():
                assigned[novel_indices[near]] = parent_eid

    def _pretrain_child(self, parent_rec, aff: float, next_id: int, x_taken: Tensor, y_taken: Tensor):
        """The child of a transfer conflict: initialised exactly as the probe's child is, then ``spawn_pretrain_steps`` steps on the cluster alone (same seeds as the probe's)."""
        cfg = self.cfg
        if cfg.init_mode == "scratch":
            child = MLPExpert(d_in=cfg.d, hidden=cfg.hidden, depth=cfg.depth, n_classes=cfg.n_classes, seed=cfg.seed * 1000 + next_id)
        elif cfg.init_mode == "clone":
            child = parent_rec.module.clone()
        elif cfg.init_mode == "inherit":
            child = parent_rec.module.inherit(rho=cfg.rho_max * (1.0 - aff), sigma=cfg.inherit_sigma, rng=np.random.default_rng([cfg.seed, 0x494E48, next_id]))
        else:
            child = parent_rec.module.shrink_perturb(lam=aff, seed=cfg.seed * 1000 + next_id)
        valid = y_taken >= 0
        xv, yv = x_taken[valid], y_taken[valid]
        child.seen.zero_()
        child.mark_seen(yv)
        opt = torch.optim.Adam(child.parameters(), lr=cfg.lr)
        bs = min(cfg.batch_size, int(yv.numel()))
        for s_idx in range(cfg.spawn_pretrain_steps):
            gen = torch.Generator().manual_seed(cfg.seed + 1000 * self.step + s_idx)
            sel = torch.randint(0, int(yv.numel()), (bs,), generator=gen)
            child.train_step(xv[sel], yv[sel], opt, replay=None)
        return child

    def _merge_refs(self, rec_i, rec_j) -> None:
        """The survivor of a merge keeps the references of both (the newest ``transfer_ref`` of the union, the second expert's entries last)."""
        if "ref" in rec_i.meta or "ref" in rec_j.meta:
            rec_i.meta["ref"] = deque(list(rec_i.meta.get("ref", ())) + list(rec_j.meta.get("ref", ())), maxlen=self.cfg.transfer_ref)
```

(f) `extra_cost`: before `return params, macs, scalars` add

```python
        if self.cfg.spawn_rule == "transfer":
            scalars += sum(len(r.meta["ref"]) for r in self.experts.values() if "ref" in r.meta)
```

- [ ] **Step 3.8: Run the new tests, the neighbours and the golden check**

Run: `python -m pytest tests/growmoe/test_gm_rv4_transfer.py tests/growmoe/test_gm_rv4_transfer_pool.py -q -p no:cacheprovider`
Expected: all pass except possibly `test_merge_cleans_up_a_forced_duplicate` (the audit; see its docstring for what to do if it fails). Settings of synthetic data may be adjusted (reason in the docstring), assertions may not be weakened.
Run: `python -m pytest tests/growmoe/test_gm_phylo.py tests/growmoe/test_gm_rv3_pool.py tests/growmoe/test_gm_rv3_confirm.py tests/growmoe/test_gm_competence.py tests/growmoe/test_gm_m2a_arming.py -q -p no:cacheprovider`
Expected: unchanged green.
Run: `python research/growmoe_results/rv2c/golden.py check --quick`
Expected: `bit-identical`.

- [ ] **Step 3.9: Record the audit and commit**

Write the measured outcome of `test_merge_cleans_up_a_forced_duplicate` (seeds 0 and 1: merged or not) into the scratch file `AUDIT_MERGE.txt` in the scratchpad; it goes into the build notebook entry in task 4. Then:

```bash
git add growmoe/transfer.py growmoe/pools/phylo.py tests/growmoe/rv4_helpers.py tests/growmoe/test_gm_rv4_transfer.py tests/growmoe/test_gm_rv4_transfer_pool.py
git commit -m "feat(growmoe): stage 4b - spawn or extend by a held-out transfer test with a margin (spawn_rule, off by default); merge audit on forced duplicates"
```

---

## Task 4: Stage 4c - growth from the within-batch fit gap, and the build record

**Files:**
- Create: `growmoe/gapgrow.py`
- Modify: `growmoe/pools/phylo.py` (imports, `__init__`, `observe`, `_attempt_merges`, new methods, `extra_cost`)
- Create: `tests/growmoe/test_gm_rv4_growth.py` (pure), `tests/growmoe/test_gm_rv4_growth_pool.py` (pool)

**Interfaces:**
- Consumes: `PhyloCfg.growth_rule`, `grow_gap`, `grow_window`, `grow_plateau`, `grow_min_age`, `grow_cooldown`, `grow_factor`, `grow_guard_window`, `grow_guard_eps`, `grow_budget_params` (task 1); `GrowthCfg.max_hidden`, `min_widen`; `MLPExpert.widen(k, rng)`, `.clone()`, `.hidden`, `.num_params()`, `.blocks`, `.d_in`; `PhyloMoE.extra_cost` (tasks 2, 3).
- Produces:
  - `growmoe.gapgrow.GapMonitor(window)` with `add(pre, post)`, `gap()`, `improvement()`, `mature(min_age, plateau)`, `cooled(cooldown)`, `grew()`, `recent_pre_mean(n)`, attributes `age`, `since_growth`; `widen_increment(hidden, factor, max_hidden, min_widen) -> int`; `params_of_widen(d_in, depth, k) -> int`; `guard_reverts(before, after, eps) -> bool`; `GrowthGuard(module, opt, before, hidden_before, losses)`.
  - `PhyloMoE._gap_growth_step(rec, x, y, pre)`, `_widen_expert(...)`, `_close_guard(...)`, `_forget_growth_state(rec)`, `_live_params()`, `_grow_stats()`; per-expert `rec.meta["gap"]`, `rec.meta["guard"]`, `rec.meta["grow_cooldown"]`; `pool.grow_stats` = {"post_rows", "train_rows", "blocked_cap", "blocked_budget", "reverts"}; events `grow` {expert, kind="widen", increment, gap, improvement, hidden_before, hidden_after, params_added, rule="gap"} and `grow_revert` {expert, hidden_from, hidden_to, before, after}.

### 4a. The pure pieces

- [ ] **Step 4.1: Write the failing tests**

Create `tests/growmoe/test_gm_rv4_growth.py`:

```python
"""RV4 stage 4c, the pure pieces: the fit gap, progress, maturity, cooldown, the growth increment and the guard (spec sections 6.1-6.3)."""

import pytest
import torch

from growmoe.experts import MLPExpert
from growmoe.gapgrow import GapMonitor, guard_reverts, params_of_widen, widen_increment


def _t(v):
    return torch.tensor([float(v)])


def test_gap_is_one_minus_post_over_pre_over_the_window():
    m = GapMonitor(window=3)
    assert m.gap() is None                                                           # the window is not full
    for pre, post in ((2.0, 1.0), (2.0, 1.5), (4.0, 2.0)):
        m.add(_t(pre), _t(post))
    assert m.gap() == pytest.approx(1 - (1.0 + 1.5 + 2.0) / (2.0 + 2.0 + 4.0))
    m.add(_t(1.0), _t(1.0))                                                          # the oldest batch drops out
    assert m.gap() == pytest.approx(1 - (1.5 + 2.0 + 1.0) / (2.0 + 4.0 + 1.0))


def test_gap_sums_all_rows_of_a_batch_and_ignores_empty_batches():
    m = GapMonitor(window=2)
    m.add(torch.tensor([1.0, 3.0]), torch.tensor([0.5, 1.5]))                       # sums 4.0 and 2.0
    m.add(torch.empty(0), torch.empty(0))
    assert m.age == 1 and m.gap() is None
    m.add(torch.tensor([4.0]), torch.tensor([2.0]))
    assert m.gap() == pytest.approx(1 - (2.0 + 2.0) / (4.0 + 4.0))


def test_a_window_with_no_loss_has_no_gap():
    m = GapMonitor(window=2)
    for _ in range(2):
        m.add(_t(0.0), _t(0.0))
    assert m.gap() is None


def test_improvement_compares_the_last_window_with_the_one_before():
    m = GapMonitor(window=2)
    assert m.improvement() is None
    for v in (4.0, 4.0, 2.0, 2.0):
        m.add(_t(v), _t(v / 2))
    assert m.improvement() == pytest.approx(0.5)


def test_maturity_needs_age_and_a_plateau():
    flat = GapMonitor(window=2)
    for _ in range(4):
        flat.add(_t(2.0), _t(2.0))
    assert flat.age == 4 and flat.mature(min_age=4, plateau=0.95) and not flat.mature(min_age=5, plateau=0.95)
    falling = GapMonitor(window=2)
    for v in (4.0, 4.0, 2.0, 2.0):
        falling.add(_t(v), _t(v))
    assert not falling.mature(min_age=1, plateau=0.95)                                # still improving by half


def test_growth_resets_the_windows_and_starts_the_cooldown():
    m = GapMonitor(window=2)
    assert m.cooled(5)                                                                # never grown
    for _ in range(4):
        m.add(_t(2.0), _t(1.0))
    m.grew()
    assert m.age == 0 and m.gap() is None and m.improvement() is None and not m.cooled(2)
    m.add(_t(2.0), _t(1.0))
    assert not m.cooled(2)
    m.add(_t(2.0), _t(1.0))
    assert m.cooled(2)


def test_the_recent_mean_loss_uses_the_last_batches_only():
    m = GapMonitor(window=3)
    assert m.recent_pre_mean(2) != m.recent_pre_mean(2)                               # nan before any batch
    for v in (1.0, 2.0, 3.0, 4.0):
        m.add(_t(v), _t(v))
    assert m.recent_pre_mean(2) == pytest.approx(3.5) and m.recent_pre_mean(100) == pytest.approx(2.5)


def test_widen_increment_doubles_and_respects_the_cap():
    assert widen_increment(16, 2.0, 1024, 4) == 16
    assert widen_increment(700, 2.0, 1024, 4) == 324
    assert widen_increment(1024, 2.0, 1024, 4) == 0
    assert widen_increment(1022, 2.0, 1024, 4) == 0                                   # room 2 is below min_widen
    assert widen_increment(8, 1.5, 1024, 4) == 4


def test_params_of_widen_matches_the_real_expert():
    e = MLPExpert(d_in=16, hidden=8, depth=2, n_classes=4)
    before = e.num_params()
    e.widen(5)
    assert e.num_params() - before == params_of_widen(16, 2, 5)


def test_the_guard_reverts_only_when_the_loss_got_worse_by_more_than_eps():
    assert guard_reverts(1.0, 1.03, 0.02)
    assert not guard_reverts(1.0, 1.02, 0.02)                                         # exactly eps is not more than eps
    assert not guard_reverts(1.0, 0.9, 0.02)
```

- [ ] **Step 4.2: Run and watch it fail**

Run: `python -m pytest tests/growmoe/test_gm_rv4_growth.py -q -p no:cacheprovider`
Expected: collection error `ModuleNotFoundError: No module named 'growmoe.gapgrow'`.

- [ ] **Step 4.3: Implement `growmoe/gapgrow.py`**

```python
"""Routing v2, stage 4c: grow an expert's width from the within-batch fit gap (spec docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md, section 6).

The shadow-trial policy of today never fires on a starved expert (E-030). Spike C (E-032) showed that a short paired trial cannot confirm the need for width, but that one cheap
number tracks it: the share of a fresh batch an expert can fit in its own few training steps, g = 1 - sum(L_post) / sum(L_pre) over the last ``window`` batches, with L_pre the honest
loss before the steps and L_post the loss on the same rows after them. g rises with width (16 / 32 / 64 / 128 / 512 units: 0.08 / 0.12 / 0.17 / 0.23 / 0.39 on the arithmetic stream),
so growth stops by itself near the knee of the accuracy curve. An expert is *mature* when it has trained long enough and its honest loss has stopped improving.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from torch import Tensor


class GapMonitor:
    """Per-expert bookkeeping of the fit gap, of progress, of age and of the cooldown. Pure: it holds numbers only."""

    def __init__(self, window: int) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        self.window = window
        self.pairs: Deque[tuple] = deque(maxlen=window)         # per batch: (sum of honest losses before the steps, sum of losses after them), the same rows
        self.pre: Deque[float] = deque(maxlen=2 * window)       # per batch: mean honest loss before the steps
        self.age = 0                                            # batches seen since birth or the last growth
        self.since_growth: Optional[int] = None                 # batches since the last growth (None = never grown)

    def add(self, pre: Tensor, post: Tensor) -> None:
        """One trained batch: per-row losses before and after the expert's own steps on the same rows."""
        pre, post = pre.double().flatten(), post.double().flatten()
        if pre.numel() == 0:
            return
        self.pairs.append((float(pre.sum()), float(post.sum())))
        self.pre.append(float(pre.mean()))
        self.age += 1
        if self.since_growth is not None:
            self.since_growth += 1

    def gap(self) -> Optional[float]:
        """1 - sum(post) / sum(pre) over the last ``window`` batches; None until the window is full or when there is no loss to share."""
        if len(self.pairs) < self.window:
            return None
        s_pre, s_post = sum(p for p, _ in self.pairs), sum(q for _, q in self.pairs)
        return None if s_pre <= 0.0 else 1.0 - s_post / s_pre

    def improvement(self) -> Optional[float]:
        """Mean honest loss of the last window over that of the window before (below 1 = still improving); None until two windows exist."""
        if len(self.pre) < 2 * self.window:
            return None
        h = list(self.pre)
        prev, last = sum(h[:self.window]) / self.window, sum(h[self.window:]) / self.window
        return None if prev <= 0.0 else last / prev

    def mature(self, min_age: int, plateau: float) -> bool:
        imp = self.improvement()
        return self.age >= min_age and imp is not None and imp >= plateau

    def cooled(self, cooldown: int) -> bool:
        return self.since_growth is None or self.since_growth >= cooldown

    def grew(self) -> None:
        """A growth (or a reverted growth) happened: new windows, age and cooldown start again."""
        self.pairs.clear()
        self.pre.clear()
        self.age = 0
        self.since_growth = 0

    def recent_pre_mean(self, n: int) -> float:
        """Mean honest loss per batch over the last ``n`` batches (all of them when fewer exist; nan when none)."""
        h = list(self.pre)[-n:]
        return sum(h) / len(h) if h else float("nan")


def widen_increment(hidden: int, factor: float, max_hidden: int, min_widen: int) -> int:
    """Hidden units to add to multiply the width by ``factor``, capped at ``max_hidden``; 0 when the cap leaves room for fewer than ``min_widen``."""
    k = min(int(round(hidden * factor)), max_hidden) - hidden
    return k if k >= min_widen else 0


def params_of_widen(d_in: int, depth: int, k: int) -> int:
    """Parameters added by widening every block of a residual MLP expert by k units: w1 (k x d), b1 (k), w2 (d x k)."""
    return depth * (2 * d_in * k + k)


def guard_reverts(before: float, after: float, eps: float) -> bool:
    """True when the honest loss after a growth is worse than before it by MORE than the share ``eps``."""
    return after > (1.0 + eps) * before


@dataclass
class GrowthGuard:
    """What is kept for ``grow_guard_window`` batches after a growth so that it can be undone: the expert and its optimiser as they were."""
    module: object
    opt: object
    before: float                                               # mean honest loss per batch over the window before the growth
    hidden_before: int
    losses: List[float] = field(default_factory=list)           # mean honest loss of each batch since the growth
```

- [ ] **Step 4.4: Run and see it pass**

Run: `python -m pytest tests/growmoe/test_gm_rv4_growth.py -q -p no:cacheprovider`
Expected: all pass.

### 4b. Pool integration

- [ ] **Step 4.5: Write the failing pool tests**

Create `tests/growmoe/test_gm_rv4_growth_pool.py`:

```python
"""RV4 stage 4c inside PhyloMoE: the controller, the guard, caps, budget, accounting (spec sections 6.2-6.6).

The synthetic task is a random two-layer teacher on unit inputs (16 inputs, 8 classes). The growth threshold of the growth tests is NOT a magic number: it is calibrated inside
the module from the measured gap of a starved and an adequate expert (`gamma`), so the tests check the mechanism, not a constant."""

import copy
import pickle

import pytest
import torch
import torch.nn.functional as F

from growmoe.accounting import pool_report
from growmoe.config import GrowthCfg, PhyloCfg
from growmoe.gapgrow import GapMonitor, GrowthGuard
from growmoe.pools.phylo import PhyloMoE
from rv4_helpers import regime_batch, trained_pool

D, C = 16, 8


def _batches(seed=0, scale=3.0, hidden=32, random_labels=False):
    g = torch.Generator().manual_seed(seed)
    w1, w2 = torch.randn(D, hidden, generator=g), torch.randn(hidden, C, generator=g)

    def batch(k):
        gk = torch.Generator().manual_seed(50_000 + 31 * seed + k)
        x = F.normalize(torch.randn(64, D, generator=gk), dim=1)
        if random_labels:
            return x, torch.randint(0, C, (64,), generator=gk)
        return x, (torch.tanh(scale * (x @ w1)) @ w2).argmax(1)
    return batch


def _gap_pool(hidden, max_hidden=1024, **kw):
    base = dict(d=D, hidden=hidden, n_classes=C, seed=1, q=0.02, m_min=32, inner_steps=5, route_rule="accept_then_logf", set_gate=False, group_vote=False,
                growth_rule="gap", grow_gap=0.2, grow_window=20, grow_min_age=40, grow_cooldown=20, grow_guard_window=20)
    base.update(kw)
    return PhyloMoE(PhyloCfg(**base), grow_cfg=GrowthCfg(max_hidden=max_hidden))


def _run(pool, batch, n, start=0):
    for k in range(start, start + n):
        pool.observe(*batch(k))


@pytest.fixture(scope="module")
def gaps():
    batch = _batches()
    out = {}
    for h in (4, 16, 64):
        pool = _gap_pool(h, grow_min_age=10 ** 9)                                       # the monitor runs, nothing may grow
        _run(pool, batch, 150)
        out[h] = pool.experts[0].meta["gap"].gap()
    return out


@pytest.fixture(scope="module")
def gamma(gaps):
    return 0.5 * (gaps[16] + gaps[64])


# ------------------------------------------------------------------------------------------------ the signal
def test_the_fit_gap_rises_with_width(gaps):
    """Premise measured in spike C (E-032) on the arithmetic stream; here on a teacher stream with widths 4, 16, 64."""
    assert 0.0 < gaps[4] < gaps[16] < gaps[64] < 1.0


def test_the_post_loss_costs_one_forward_of_the_trained_rows_and_no_more():
    pool = _gap_pool(8, grow_min_age=10 ** 9)
    _run(pool, _batches(), 30)
    s = pool.grow_stats
    assert s["post_rows"] == s["train_rows"] > 0                                        # at most 64 rows per batch, and a batch has 64


# ------------------------------------------------------------------------------------------------ the controller
def test_a_starved_expert_grows_at_least_twice_and_then_stops(gamma):
    pool = _gap_pool(4, grow_gap=gamma)
    _run(pool, _batches(), 700)
    grows = pool.events.of_kind("grow")
    assert len(grows) >= 2 and all(e.fields["rule"] == "gap" and e.fields["kind"] == "widen" for e in grows)
    assert all(e.fields["gap"] < gamma and e.fields["hidden_after"] == 2 * e.fields["hidden_before"] for e in grows)
    assert pool.experts[0].module.hidden >= 16
    assert grows[-1].step < 700 - 150                                                   # it stopped by itself


def test_an_adequate_expert_does_not_grow(gamma):
    pool = _gap_pool(64, grow_gap=gamma)
    _run(pool, _batches(), 400)
    assert not pool.events.of_kind("grow") and pool.experts[0].module.hidden == 64


def test_maturity_blocks_growth_early_and_for_ever_when_the_age_is_out_of_reach():
    never = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0, grow_min_age=10 ** 6)
    _run(never, _batches(), 200)
    assert not never.events.of_kind("grow")
    early = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0, grow_min_age=60)
    _run(early, _batches(), 300)
    grows = early.events.of_kind("grow")
    assert grows and grows[0].step >= 60


def test_growth_stops_at_the_cap_and_at_the_budget_without_error():
    capped = _gap_pool(4, max_hidden=16, grow_gap=0.9, grow_plateau=1.0)
    _run(capped, _batches(), 400)
    assert capped.experts[0].module.hidden <= 16 and capped.grow_stats["blocked_cap"] > 0
    poor = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0)
    _run(poor, _batches(), 1)
    poor.cfg.grow_budget_params = sum(r.module.num_params() for r in poor.experts.values()) + 10
    _run(poor, _batches(), 300, start=1)
    assert not poor.events.of_kind("grow") and poor.grow_stats["blocked_budget"] > 0


def test_pure_noise_cannot_grow_an_expert_past_its_cap():
    pool = _gap_pool(4, max_hidden=32, grow_gap=0.9, grow_plateau=1.0)
    _run(pool, _batches(random_labels=True), 400)
    assert pool.experts[0].module.hidden <= 32 and len(pool.events.of_kind("grow")) <= 3


def test_the_growth_rule_needs_the_master_switch_and_the_probe_rule_keeps_no_monitor():
    off = _gap_pool(4, grow=False, grow_gap=0.9, grow_plateau=1.0)
    _run(off, _batches(), 120)
    assert "gap" not in off.experts[0].meta and not off.events.of_kind("grow")
    probe = _gap_pool(4, growth_rule="probe")
    _run(probe, _batches(), 60)
    assert "gap" not in probe.experts[0].meta and probe.grow_stats["post_rows"] == 0


# ------------------------------------------------------------------------------------------------ the guard
def _until_first_growth(pool, batch, limit=400):
    for k in range(limit):
        pool.observe(*batch(k))
        if pool.events.of_kind("grow"):
            return k + 1
    raise AssertionError("no growth within the limit: change the setting of the test")


def test_the_guard_reverts_a_growth_followed_by_a_collapse():
    pool, batch = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0), _batches()
    k = _until_first_growth(pool, batch)
    rec = pool.experts[0]
    grown = rec.module.hidden
    assert "guard" in rec.meta and grown > 4
    with torch.no_grad():
        for p in rec.module.parameters():
            p.add_(2.0 * torch.randn_like(p))                                           # sabotage: the loss over the guard window must come out worse
    _run(pool, batch, 22, start=k)
    assert pool.events.of_kind("grow_revert") and rec.module.hidden < grown and "guard" not in rec.meta
    assert pool.grow_stats["reverts"] == 1 and rec.meta["grow_cooldown"] == 40


def test_a_healthy_growth_is_kept_when_the_guard_window_ends():
    """grow_guard_eps is 0.5 here: with the product value 0.02 and a 20-batch window the noise of two window means alone (about 3 %) would revert a healthy growth now and then;
    this test is about the KEEP path. The revert rate at the product settings is a dev-review item (task 6)."""
    pool, batch = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0, grow_cooldown=10_000, grow_guard_eps=0.5), _batches()
    k = _until_first_growth(pool, batch)
    rec = pool.experts[0]
    grown = rec.module.hidden
    _run(pool, batch, 22, start=k)
    assert "guard" not in rec.meta and rec.module.hidden == grown and not pool.events.of_kind("grow_revert")


def test_the_guard_copy_is_counted_while_it_exists():
    pool, batch = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0, grow_guard_window=30, grow_cooldown=10_000), _batches()
    k = _until_first_growth(pool, batch)
    rec = pool.experts[0]
    rep = pool_report(pool, D)
    assert rep.params == rec.module.num_params() + rec.meta["guard"].module.num_params()
    _run(pool, batch, 35, start=k)
    assert "guard" not in rec.meta and pool_report(pool, D).params == rec.module.num_params()


def test_a_pool_with_an_active_guard_can_be_pickled():
    pool, batch = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0, grow_cooldown=10_000), _batches()
    _until_first_growth(pool, batch)
    assert "guard" in pool.experts[0].meta
    for clone in (pickle.loads(pickle.dumps(pool)), copy.deepcopy(pool)):
        x, _ = batch(999)
        assert torch.allclose(clone.predict(x).probs, pool.predict(x).probs, atol=1e-5)


def test_a_merge_drops_the_guard_and_the_stale_windows():
    pool, W = trained_pool(n_batches=3, growth_rule="gap")
    rec = pool.experts[0]
    rec.meta["guard"] = GrowthGuard(module=rec.module.clone(), opt=rec.opt, before=1.0, hidden_before=rec.module.hidden)
    rec.meta["gap"] = GapMonitor(20)
    pool._forget_growth_state(rec)
    assert "guard" not in rec.meta and "gap" not in rec.meta


# ------------------------------------------------------------------------------------------------ determinism and the flag off
def test_two_identical_runs_grow_identically():
    runs = []
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(2):
            pool = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0)
            _run(pool, _batches(), 150)
            runs.append([(e.step, e.fields["increment"], e.fields["hidden_after"]) for e in pool.events.of_kind("grow")])
    finally:
        torch.set_num_threads(threads)
    assert runs[0] == runs[1] and runs[0]


def test_with_the_probe_rule_the_pool_is_bit_identical_to_the_recorded_golden_run():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("golden", root / "research" / "growmoe_results" / "rv2c" / "golden.py")
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    assert golden.differences(golden.QUICK) == []
```

- [ ] **Step 4.6: Run and watch it fail**

Run: `python -m pytest tests/growmoe/test_gm_rv4_growth_pool.py -q -p no:cacheprovider`
Expected: FAIL (`AttributeError: 'PhyloMoE' object has no attribute 'grow_stats'` / `KeyError: 'gap'` / `_forget_growth_state`), except the golden test.

- [ ] **Step 4.7: Implement the integration in `growmoe/pools/phylo.py`**

(a) Imports (next to the others):

```python
from ..gapgrow import GapMonitor, GrowthGuard, guard_reverts, params_of_widen, widen_increment
```

(b) `__init__`, after `self._route_state()`:

```python
        self._grow_stats()                                             # routing v2 stage 4c; idle unless cfg.growth_rule == "gap"
```

(c) `observe`: in the labelled-training block, replace

```python
                        # R6: Task 14 Growth inside Architecture A
                        if (
                            self.cfg.grow
                            and rec.state == "active"
```

with

```python
                        if self.cfg.growth_rule == "gap":              # routing v2 stage 4c: growth from the within-batch fit gap replaces the shadow-trial policy below
                            self._gap_growth_step(rec, xk_v, yk_v, preq_losses)

                        # R6: Task 14 Growth inside Architecture A
                        elif (
                            self.cfg.grow
                            and rec.state == "active"
```

(d) `_attempt_merges`: after the `_merge_refs` hook of task 3 insert

```python
                        if self.cfg.growth_rule == "gap":                  # routing v2 stage 4c: the merged weights are not the guarded ones, and the old windows describe the old module
                            self._forget_growth_state(rec_i)
```

(e) New methods, after the transfer methods:

```python
    # ------------------------------------------------------------------------------------------ routing v2 stage 4c: growth from the within-batch fit gap
    def _grow_stats(self) -> dict:
        """Counters of the growth controller (the steps at which growth was wanted but blocked are counted per step). No wall-clock."""
        if not hasattr(self, "grow_stats"):
            self.grow_stats = {"post_rows": 0, "train_rows": 0, "blocked_cap": 0, "blocked_budget": 0, "reverts": 0}
        return self.grow_stats

    def _live_params(self) -> int:
        return sum(r.module.num_params() for r in self.experts.values())

    @staticmethod
    def _forget_growth_state(rec) -> None:
        rec.meta.pop("guard", None)
        rec.meta.pop("gap", None)

    def _gap_growth_step(self, rec, x: Tensor, y: Tensor, pre: Tensor) -> None:
        """Called after an active expert trained on this batch. ``pre`` = its honest per-sample losses on the batch BEFORE the steps (the competence monitor's input); the losses on
        the same rows AFTER the steps cost one no-grad forward of at most 64 rows. See ``growmoe.gapgrow`` for the signals; the decision: mature, gap below ``grow_gap``, cooled
        down, room under the width cap and the parameter budget -> widen by ``grow_factor`` (function-preserving), keep the old expert for ``grow_guard_window`` batches."""
        cfg, stats = self.cfg, self._grow_stats()
        if not cfg.grow or rec.state != "active":
            return
        m = min(64, int(y.numel()))
        with torch.no_grad():
            post = F.cross_entropy(rec.module(x[:m]), y[:m], reduction="none")
        stats["post_rows"] += m
        stats["train_rows"] += int(y.numel())
        mon = rec.meta.get("gap")
        if mon is None:
            mon = rec.meta["gap"] = GapMonitor(cfg.grow_window)
        mon.add(pre[:m], post)
        guard = rec.meta.get("guard")
        if guard is not None:                                          # a growth is on probation: no new decision until it is confirmed or undone
            guard.losses.append(float(pre.mean()))
            if len(guard.losses) >= cfg.grow_guard_window:
                self._close_guard(rec, guard, mon)
            return
        gap = mon.gap()
        if gap is None or gap >= cfg.grow_gap or not mon.mature(cfg.grow_min_age, cfg.grow_plateau) or not mon.cooled(rec.meta.get("grow_cooldown", cfg.grow_cooldown)):
            return
        k = widen_increment(rec.module.hidden, cfg.grow_factor, self.grow_cfg.max_hidden, self.grow_cfg.min_widen)
        if k == 0:
            stats["blocked_cap"] += 1
            return
        added = params_of_widen(rec.module.d_in, len(rec.module.blocks), k)
        if cfg.grow_budget_params is not None and self._live_params() + added > cfg.grow_budget_params:
            stats["blocked_budget"] += 1
            return
        self._widen_expert(rec, mon, k, added, gap)

    def _widen_expert(self, rec, mon: GapMonitor, k: int, added: int, gap: float) -> None:
        cfg = self.cfg
        old_module, old_opt = rec.module, rec.opt
        new_module = old_module.clone()
        new_module.widen(k, np.random.default_rng([cfg.seed, 0x47524F57, rec.id, self.step]))
        rec.module, rec.opt = new_module, self.make_optimizer(new_module, rec.lr_mult)       # growth replaces parameters: the optimiser is rebuilt, as after any committed growth
        rec.meta["guard"] = GrowthGuard(module=old_module, opt=old_opt, before=mon.recent_pre_mean(cfg.grow_guard_window), hidden_before=old_module.hidden)
        improvement = mon.improvement()
        mon.grew()
        self.events.add("grow", self.step, expert=rec.id, kind="widen", increment=k, gap=gap, improvement=improvement, hidden_before=old_module.hidden,
                        hidden_after=new_module.hidden, params_added=added, rule="gap")

    def _close_guard(self, rec, guard: GrowthGuard, mon: GapMonitor) -> None:
        """The guard window is over: keep the growth, or undo it (the expert and optimiser as they were) when the honest loss got worse by more than ``grow_guard_eps``."""
        cfg = self.cfg
        rec.meta.pop("guard", None)
        after = sum(guard.losses) / len(guard.losses)
        if guard_reverts(guard.before, after, cfg.grow_guard_eps):
            grown = rec.module.hidden
            rec.module, rec.opt = guard.module, guard.opt
            rec.meta["grow_cooldown"] = 2 * rec.meta.get("grow_cooldown", cfg.grow_cooldown)
            mon.grew()
            self._grow_stats()["reverts"] += 1
            self.events.add("grow_revert", self.step, expert=rec.id, hidden_from=grown, hidden_to=rec.module.hidden, before=guard.before, after=after)
```

(f) `extra_cost`: before `return params, macs, scalars` add

```python
        if self.cfg.growth_rule == "gap":                                # a live growth guard keeps the pre-growth module: its parameters are real memory
            params += sum(r.meta["guard"].module.num_params() for r in self.experts.values() if "guard" in r.meta)
```

- [ ] **Step 4.8: Run the new tests**

Run: `python -m pytest tests/growmoe/test_gm_rv4_growth.py tests/growmoe/test_gm_rv4_growth_pool.py -q -p no:cacheprovider`
Expected: all pass. The statistical tests (`gaps`, starved, adequate, maturity, cap, budget, guard) depend on the synthetic task: if one fails at first run, print the measured numbers, change the SETTING (widths, windows, stream length, teacher scale) and record the reason in the test's docstring; never loosen an assertion.

### 4c. Whole-task checks, the build record, the commit

- [ ] **Step 4.9: Full suite and golden check**

Run (background, read the end of the output): `python -m pytest tests/growmoe -q -p no:cacheprovider`
Expected: 692 + all RV4 tests, 0 failures (the merge audit may be an expected xfail).
Run: `python research/growmoe_results/rv2c/golden.py check`
Expected: `bit-identical` (all 20 fingerprints).

- [ ] **Step 4.10: Notebook entry for the build, memory note, commit**

1. Create the build entry (type implementation, track G, status completed, evidence grade A for the tests) with the next free id by exclusive creation (copy the pattern of `make_spike_entry.py` in the scratchpad: read the highest `E-` id right before writing, `open(path, "x")`). Content: what was built (three modules, hooks, flags off), the four refinements R-1 to R-4, test counts (new tests, full suite), golden check result, the merge-audit outcome from step 3.9, files, the reproduction command. No claim rows (no evidence about performance yet); one decision row: D-32 "stage 4 plan approved by the user on 2026-10-15 ('approved, write the plan and start building'); execution native (Claude builds, one fresh reviewer at the end); the confirmatory run waits for a go".
2. `python docs/labnotebook/tools/labnb.py index` then `manifest` then `verify` (expect 0 problems).
3. Update the memory note `stage4-design-2026-10-15.md` (state: tasks 1 to 4 built and committed, flags off, dev runs next) and its line in `MEMORY.md`.
4. Commit:

```bash
git add growmoe/gapgrow.py growmoe/pools/phylo.py tests/growmoe/test_gm_rv4_growth.py tests/growmoe/test_gm_rv4_growth_pool.py docs/labnotebook
git commit -m "feat(growmoe): stage 4c - growth from the within-batch fit gap with a guard (growth_rule, off by default); build record E-0xx"
```

---

## Task 5: The experiment package `rv2d` - cells, queue, selection of the dev constants, dev runs

**Files (all under `research/growmoe_results/rv2d/` unless noted):**
- Create: `rv2d_lib.py` (cells, arms, flag mapping, atomic writes, router variants, family runners), `run_cell.py`, `make_cells.py`, `jobqueue.py`, `rv2d_select.py` (the declared selection rules, pure), `report_rv2d.py`, `RESUME_RV2D.cmd`
- Create: `tests/growmoe/test_gm_rv4_runner.py` (cell ids, flag mapping, selection rules)
- Results (git-tracked except the checkpoints): `cells/*.json`, `cells_dev1.json`, `cells_dev2.json`, `dev_constants.json`, `DEV_SELECTION.md`, `RESULTS_DEV.md`; git-ignored: `cells/*.ckpt`, `cells/*.tmp`, `queue.lock`, `logs/`

**Interfaces:**
- Consumes: everything of tasks 1-4; `growmoe.fairfight.run_fast`, `_Scorer`, `run_record`; `growmoe.metrics.paired_diff`; the stage-3 runner pieces `run_rv2c.build_stream/steps_for/arm_cfg/LoggingPool`; `arith_lib` (`Pool`, `build_train/test/seen`, `op_blocks`, `cell_tables`); `growmoe.streams_real.banking_incremental` with `run_m0.HybridFrontend`.
- Produces:
  - `rv2d_lib.Cell(family, stream, arm, seed, gap, tmargin, phase)` with `.cid` (unique, no dots), `arm_flags(cell) -> dict`, `ARMS`, `ROUTER_SETTINGS`, `GAP_GRID`, `TRANSFER_GRID`, `STREAMS`, `atomic_write_text(path, text)`, `valid_json(path)`, `router_variants(pool, evaluate) -> dict`, `run_cell(cell) -> dict` (the record below).
  - **The cell record** (one JSON per cell): `cid`, `cell`, `acc` (router off), `router` = {"off": {...}, "restrict-0.01": {"acc", "selected", "cv", "rows"}, "restrict-0.03", "free-0.01", "free-0.03"}, `forgetting` (arithmetic sequential only), `n_experts`, `regimes` (stage-3 streams), `params`, `memory_bytes`, `events` (counts by kind), `grow_events` (list of the events' fields), `final_hidden` (list), `grow_stats`, `inner_steps`, `seconds`.
  - `rv2d_select.choose_grow_gap(rows)`, `choose_router_setting(rows)`, `choose_transfer_margin(rows)` -> dicts that carry the chosen value, the whole curve and the reason.

### 5a. Pure pieces first (cell ids, flags, selection rules)

- [ ] **Step 5.1: Write the failing tests**

Create `tests/growmoe/test_gm_rv4_runner.py`:

```python
"""RV4 runner: cell identity, arm flags and the declared selection rules of the dev constants (spec sections 6.7 and 8). The runner files live in research/growmoe_results/rv2d/."""

import sys
from pathlib import Path

import pytest

RV2D = Path(__file__).resolve().parents[2] / "research" / "growmoe_results" / "rv2d"
sys.path.insert(0, str(RV2D))

import rv2d_lib as R        # noqa: E402
import rv2d_select as S     # noqa: E402


def test_cell_ids_are_unique_file_safe_and_dot_free():
    cells = [R.Cell("s3", s, a, seed, gap=g, tmargin=t) for s in ("OV-0.6", "OV-0.85", "F-15") for a, g, t in (("S3", 0, 0), ("S3+G", 0.2, 0), ("S3+G", 0.23, 0), ("S3+T", 0, 0.25), ("S3+T+G", 0.2, 0.25))
             for seed in (100, 101)]
    ids = [c.cid for c in cells]
    assert len(set(ids)) == len(ids) and all("." not in i and "+" not in i and "/" not in i for i in ids)


def test_arm_flags_follow_the_arm_and_refuse_a_missing_constant():
    assert R.arm_flags(R.Cell("s3", "F-15", "S3")) == {}
    assert R.arm_flags(R.Cell("s3", "F-15", "S3+G", gap=0.2)) == {"growth_rule": "gap", "grow_gap": 0.2}
    assert R.arm_flags(R.Cell("s3", "F-15", "S3+T", tmargin=0.25)) == {"spawn_rule": "transfer", "transfer_margin": 0.25}
    assert R.arm_flags(R.Cell("s3", "F-15", "S3+T+G", gap=0.23, tmargin=0.15)) == {"spawn_rule": "transfer", "growth_rule": "gap", "grow_gap": 0.23, "transfer_margin": 0.15}
    with pytest.raises(ValueError):
        R.arm_flags(R.Cell("s3", "F-15", "S3+G"))                      # a growth arm without its threshold
    with pytest.raises(ValueError):
        R.arm_flags(R.Cell("s3", "F-15", "S3+T"))                      # a transfer arm without its margin


def _row(family, stream, arm, seed, acc, params=1000, n_grow=0, gap=0.0, tmargin=0.0, experts=4, router=None):
    return {"family": family, "stream": stream, "arm": arm, "seed": seed, "gap": gap, "tmargin": tmargin, "acc": acc, "params": params, "n_grow": n_grow, "n_experts": experts,
            "router": router or {}}


def test_choose_grow_gap_maximises_the_declared_score_under_the_guards():
    rows = []
    for seed in (100, 101):
        for fam, stream in (("arith-mixed16", "-"), ("arith-mixed128", "-"), ("s3", "F-15"), ("s3", "OV-0.6")):
            rows.append(_row(fam, stream, "S3", seed, 0.30))
            for g, gain, grow, relp in ((0.14, 0.02, 1, 0.1), (0.17, 0.05, 1, 0.2), (0.20, 0.10, 1, 0.2), (0.23, 0.11, 3, 0.9), (0.26, 0.11, 3, 0.9)):
                if fam == "arith-mixed16":
                    gain += 0.10 * (g - 0.14) / 0.12                    # the starved expert profits from the larger thresholds
                rows.append(_row(fam, stream, "S3+G", seed, 0.30 + gain, params=int(1000 * (1 + relp)), n_grow=grow, gap=g))
    out = S.choose_grow_gap(rows)
    assert out["chosen"] == 0.20                                         # 0.23 and 0.26 break the guard (3 growth events, +90 % parameters); 0.20 beats 0.17 on the score
    assert set(out["curve"]) == {0.14, 0.17, 0.20, 0.23, 0.26} and out["curve"][0.23]["guards_ok"] is False


def test_choose_grow_gap_reports_no_choice_when_no_value_satisfies_the_guards():
    rows = []
    for seed in (100, 101):
        for fam, stream in (("arith-mixed16", "-"), ("arith-mixed128", "-"), ("s3", "F-15"), ("s3", "OV-0.6")):
            rows.append(_row(fam, stream, "S3", seed, 0.30))
            for g in (0.14, 0.17, 0.20, 0.23, 0.26):
                rows.append(_row(fam, stream, "S3+G", seed, 0.20, n_grow=5, gap=g))              # every value loses 10 points and grows 5 times
    out = S.choose_grow_gap(rows)
    assert out["chosen"] is None and "per-family" in out["reason"]


def test_choose_router_setting_takes_the_best_mean_gain_among_the_settings_that_hurt_no_guard_stream():
    gains = {                                                              # setting -> gain over no router, per stream (fractions of accuracy)
        "restrict-0.01": {("arith-seq", "-"): 0.12, ("s3", "OV-0.85"): 0.08, ("s3", "S1"): 0.0, ("s3", "F-15"): 0.0},
        "restrict-0.03": {("arith-seq", "-"): 0.12, ("s3", "OV-0.85"): 0.09, ("s3", "S1"): 0.0, ("s3", "F-15"): -0.02},       # loses 2 points on a guard stream
        "free-0.01": {("arith-seq", "-"): 0.13, ("s3", "OV-0.85"): 0.10, ("s3", "S1"): -0.01, ("s3", "F-15"): 0.0},          # loses 1 point on a guard stream
        "free-0.03": {("arith-seq", "-"): 0.05, ("s3", "OV-0.85"): 0.02, ("s3", "S1"): 0.0, ("s3", "F-15"): 0.0},
    }
    rows = []
    for fam, stream in gains["restrict-0.01"]:
        for seed in (100, 101):
            router = {"off": {"acc": 0.40}}
            router.update({s: {"acc": 0.40 + g[(fam, stream)], "selected": "lda"} for s, g in gains.items()})
            rows.append(_row(fam, stream, "S3", seed, 0.40, router=router))
    out = S.choose_router_setting(rows)
    assert out["chosen"] == "restrict-0.01"                                    # restrict-0.03 and free-0.01 break the -0.5 point guard; free-0.03 is feasible but gains less
    assert out["curve"]["free-0.01"]["guards_ok"] is False and out["curve"]["free-0.03"]["guards_ok"] is True


def test_choose_transfer_margin_prefers_the_default_on_a_tie_and_respects_the_expert_guard():
    rows = []
    for seed in (100, 101):
        for stream, fam in (("-", "arith-seq"), ("S1", "s3"), ("S2-5", "s3")):
            rows.append(_row(fam, stream, "S3", seed, 0.30, experts=4))
            for t, gain, exp in ((0.15, 0.10, 4), (0.25, 0.10, 4), (0.35, 0.11, 9)):
                rows.append(_row(fam, stream, "S3+T", seed, 0.30 + gain, tmargin=t, experts=exp))
    out = S.choose_transfer_margin(rows)
    assert out["chosen"] == 0.25                                               # 0.35 breaks the 1.2x expert guard; 0.15 and 0.25 tie, the default wins
```

- [ ] **Step 5.2: Run and watch it fail**

Run: `python -m pytest tests/growmoe/test_gm_rv4_runner.py -q -p no:cacheprovider`
Expected: collection error (`ModuleNotFoundError: No module named 'rv2d_lib'`).

- [ ] **Step 5.3: Implement `rv2d_lib.py` (the pure part) and `rv2d_select.py`**

`research/growmoe_results/rv2d/rv2d_lib.py` - first the part the tests need (the family runners follow in step 5.5):

```python
"""Shared pieces of the RV2-D runner (routing v2 stage 4). Nothing here changes product code. Built for a PC that loses power daily (D-30): every result is written atomically,
an unreadable file counts as missing, long cells checkpoint."""

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (HERE.parent / "rv2c", HERE.parent / "rv2a", HERE.parent / "m2a", HERE.parent / "m0", HERE.parent / "arith", ROOT):
    sys.path.insert(0, str(_p))

CELLS = HERE / "cells"
S3_ARM = "A_set@12-0.5-100"                    # the stage-3 arm of the stage-4 spikes (batch-level gate, breaker len 12, rate 0.5, pause 100)
STREAMS = ["F-30", "S2-30", "F-15", "OV-0.85", "OV-0.6", "OV-0.3", "OV-0.3-MIXED", "OV-0.0", "S1", "S2-5"]      # the ten pre-registered stage-3 streams
ARMS = {"S3": {}, "S3+T": {"spawn_rule": "transfer"}, "S3+G": {"growth_rule": "gap"}, "S3+T+G": {"spawn_rule": "transfer", "growth_rule": "gap"}}
ROUTER_SETTINGS = [("restrict", 0.01), ("restrict", 0.03), ("free", 0.01), ("free", 0.03)]
GAP_GRID = (0.14, 0.17, 0.20, 0.23, 0.26)     # declared in spec section 6.7
TRANSFER_GRID = (0.15, 0.25, 0.35)            # declared in spec section 8


@dataclass(frozen=True)
class Cell:
    family: str                  # "arith-seq" | "arith-mixed16" | "arith-mixed128" | "s3" | "banking"
    stream: str = "-"            # stage-3 stream name for family "s3", else "-"
    arm: str = "S3"              # key of ARMS
    seed: int = 100
    gap: float = 0.0             # grow_gap of the growth arms
    tmargin: float = 0.0         # transfer_margin of the transfer arms
    phase: str = "dev"

    @property
    def cid(self) -> str:
        num = lambda v: str(v).replace(".", "p")                       # noqa: E731 - never a dot in a file name (Path.with_suffix collapses names with dots)
        return f"{self.family}__{self.stream.replace('.', 'p')}__{self.arm.replace('+', '_')}__s{self.seed}__g{num(self.gap)}__t{num(self.tmargin)}"


def arm_flags(cell: Cell) -> dict:
    kw = dict(ARMS[cell.arm])
    if "growth_rule" in kw:
        if cell.gap <= 0.0:
            raise ValueError(f"{cell.cid}: a growth arm needs grow_gap")
        kw["grow_gap"] = cell.gap
    if "spawn_rule" in kw:
        if cell.tmargin <= 0.0:
            raise ValueError(f"{cell.cid}: a transfer arm needs transfer_margin")
        kw["transfer_margin"] = cell.tmargin
    return kw


def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(text.encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def valid_json(path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            json.load(fh)
        return True
    except Exception:
        return False
```

`research/growmoe_results/rv2d/rv2d_select.py`:

```python
"""The declared selection rules of the dev constants (spec sections 6.7 and 8). Pure functions of per-cell rows (dicts with family, stream, arm, seed, gap, tmargin, acc, params,
n_grow, n_experts, router). Nothing here reads a confirmatory seed. Every function returns the chosen value, the whole curve and the reason, so the choice can be audited."""

import numpy as np

GAP_GRID = (0.14, 0.17, 0.20, 0.23, 0.26)
TRANSFER_GRID = (0.15, 0.25, 0.35)
GROUPS = {"arith16": [("arith-mixed16", "-")], "arith128": [("arith-mixed128", "-")], "stage3": [("s3", "F-15"), ("s3", "OV-0.6")]}
GUARD_GROUPS = ("arith128", "stage3")


def _by(rows, family, stream, arm, **where):
    return {r["seed"]: r for r in rows if r["family"] == family and r["stream"] == stream and r["arm"] == arm and all(r.get(k) == v for k, v in where.items())}


def _paired(rows, family, stream, arm, **where):
    base, other = _by(rows, family, stream, "S3"), _by(rows, family, stream, arm, **where)
    seeds = sorted(set(base) & set(other))
    return [(base[s], other[s]) for s in seeds]


def choose_grow_gap(rows) -> dict:
    """Score(gamma) = mean over the three groups of [paired accuracy gain over S3 - 0.1 x relative parameter increase], subject to the guards on the arithmetic-128 and stage-3
    groups: accuracy gain >= -0.005, at most 1 growth event per run on average, relative parameter increase <= 0.25. The largest score among the gammas that satisfy every guard."""
    curve = {}
    for g in GAP_GRID:
        per_group, ok = {}, True
        for name, members in GROUPS.items():
            pairs = [p for fam, st in members for p in _paired(rows, fam, st, "S3+G", gap=g)]
            if not pairs:
                per_group[name] = None
                ok = False
                continue
            gain = float(np.mean([o["acc"] - b["acc"] for b, o in pairs]))
            relp = float(np.mean([o["params"] / b["params"] - 1.0 for b, o in pairs]))
            grows = float(np.mean([o["n_grow"] for _, o in pairs]))
            per_group[name] = {"gain": gain, "rel_params": relp, "grow_events": grows, "score": gain - 0.1 * relp}
            if name in GUARD_GROUPS and not (gain >= -0.005 and grows <= 1.0 and relp <= 0.25):
                ok = False
        score = float(np.mean([v["score"] for v in per_group.values() if v])) if all(per_group.values()) else float("-inf")
        curve[g] = {"groups": per_group, "score": score, "guards_ok": ok}
    feasible = {g: c["score"] for g, c in curve.items() if c["guards_ok"]}
    if not feasible:
        return {"chosen": None, "curve": curve, "reason": "no single value satisfies the guards: declare a per-family value in the pre-registration and add a self-normalising threshold to the open questions"}
    best = max(feasible, key=lambda g: (feasible[g], -g))
    return {"chosen": best, "curve": curve, "reason": f"largest score {feasible[best]:.4f} among the values that satisfy the guards"}


def choose_router_setting(rows) -> dict:
    """Among the four post hoc settings: the best mean gain on the two target streams (arithmetic operator-by-operator, OV-0.85) among the settings whose mean difference on
    every other stream is at least -0.005 (the -0.5 point guard). Ties go to restrict-0.01."""
    settings = ["restrict-0.01", "restrict-0.03", "free-0.01", "free-0.03"]
    targets = {("arith-seq", "-"), ("s3", "OV-0.85")}
    curve = {}
    for s in settings:
        by_stream = {}
        for r in rows:
            if r["arm"] != "S3" or s not in r.get("router", {}) or "off" not in r["router"]:
                continue
            by_stream.setdefault((r["family"], r["stream"]), []).append(r["router"][s]["acc"] - r["router"]["off"]["acc"])
        means = {k: float(np.mean(v)) for k, v in by_stream.items()}
        target = [v for k, v in means.items() if k in targets]
        guards = {k: v for k, v in means.items() if k not in targets}
        curve[s] = {"target_gain": float(np.mean(target)) if target else float("nan"), "worst_guard": min(guards.values()) if guards else 0.0, "per_stream": means,
                    "guards_ok": all(v >= -0.005 for v in guards.values())}
    feasible = {s: c["target_gain"] for s, c in curve.items() if c["guards_ok"] and c["target_gain"] == c["target_gain"]}
    if not feasible:
        return {"chosen": None, "curve": curve, "reason": "every setting breaks a guard stream: the router stays off"}
    best = max(feasible, key=lambda s: (round(feasible[s], 6), -settings.index(s)))
    return {"chosen": best, "curve": curve, "reason": f"best mean gain {feasible[best]:.4f} on the target streams among the settings that hurt no guard stream by more than 0.5 points"}


def choose_transfer_margin(rows) -> dict:
    """Among the declared margins: the best mean paired accuracy gain over S3 on the target (arithmetic operator-by-operator) subject to the guards on the other streams: mean
    difference >= -0.005 and mean expert count <= 1.2 x S3's. Ties go to the default 0.25."""
    curve = {}
    for t in TRANSFER_GRID:
        gains, ok = {}, True
        for fam, st in sorted({(r["family"], r["stream"]) for r in rows if r["arm"] == "S3+T"}):
            pairs = _paired(rows, fam, st, "S3+T", tmargin=t)
            if not pairs:
                continue
            gain = float(np.mean([o["acc"] - b["acc"] for b, o in pairs]))
            ratio = float(np.mean([o["n_experts"] for _, o in pairs]) / max(np.mean([b["n_experts"] for b, _ in pairs]), 1e-9))
            gains[f"{fam}:{st}"] = {"gain": gain, "expert_ratio": ratio}
            if fam != "arith-seq" and not (gain >= -0.005 and ratio <= 1.2):
                ok = False
        target = gains.get("arith-seq:-", {}).get("gain", float("nan"))
        curve[t] = {"target_gain": target, "per_stream": gains, "guards_ok": ok}
    feasible = {t: c["target_gain"] for t, c in curve.items() if c["guards_ok"] and c["target_gain"] == c["target_gain"]}
    if not feasible:
        return {"chosen": None, "curve": curve, "reason": "no margin satisfies the guards: the transfer rule stays off"}
    top = max(round(v, 6) for v in feasible.values())
    tied = [t for t, v in feasible.items() if round(v, 6) == top]
    best = 0.25 if 0.25 in tied else min(tied)
    return {"chosen": best, "curve": curve, "reason": f"best target gain {top:.4f} among the margins that satisfy the guards (ties go to 0.25)"}
```

- [ ] **Step 5.4: Run and see it pass**

Run: `python -m pytest tests/growmoe/test_gm_rv4_runner.py -q -p no:cacheprovider`
Expected: all pass. (`test_choose_router_setting...` builds its rows in a roundabout way; if it is hard to read, simplify the test's construction, never the rule.)

### 5b. The family runners, the queue, the smoke run

- [ ] **Step 5.5: Add the family runners to `rv2d_lib.py`**

Append to `rv2d_lib.py` (imports at call time keep the pure part importable without torch):

```python
# ------------------------------------------------------------------------------------------------ evaluation of one trained pool
def router_variants(pool, evaluate) -> dict:
    """The SAME trained pool answered without the router and under each declared router setting (the router acts at answer time only, so these are paired by construction).
    ``evaluate()`` returns the accuracy of the pool as it is now."""
    from growmoe.router import RouteRule
    saved = (pool.cfg.router, pool.cfg.router_mode, pool.cfg.router_margin)
    out = {}
    try:
        pool.cfg.router = False
        out["off"] = {"acc": float(evaluate())}
        for mode, margin in ROUTER_SETTINGS:
            pool.cfg.router, pool.cfg.router_mode, pool.cfg.router_margin = True, mode, margin
            pool._route, pool._route_age, pool._route_dirty = RouteRule(), 0, False               # a fresh state: one setting's selection must not inherit another's
            acc = float(evaluate())
            out[f"{mode}-{margin}"] = {"acc": acc, "selected": pool._route.name, "cv": pool._route.cv, "rows": pool._route.rows}
    finally:
        pool.cfg.router, pool.cfg.router_mode, pool.cfg.router_margin = saved
        pool._route, pool._route_age, pool._route_dirty = RouteRule(), 0, False
    return out


def _record(cell, pool, evaluate, d, inner_steps, seconds, extra=None) -> dict:
    from collections import Counter
    from growmoe.accounting import pool_report
    variants = router_variants(pool, evaluate)
    rep = pool_report(pool, d)
    rec = {"cid": cell.cid, "cell": asdict(cell), "acc": variants["off"]["acc"], "router": variants, "n_experts": rep.n_experts, "params": rep.params, "memory_bytes": rep.memory_bytes,
           "events": dict(Counter(e.kind for e in pool.events)), "grow_events": [{"step": e.step, **e.fields} for e in pool.events.of_kind("grow")],
           "final_hidden": sorted(int(r.module.hidden) for r in pool.experts.values()), "grow_stats": dict(pool._grow_stats()), "inner_steps": inner_steps, "seconds": seconds}
    rec.update(extra or {})
    return rec


# ------------------------------------------------------------------------------------------------ stage-3 streams and real text
def run_s3(cell: Cell) -> dict:
    import dataclasses
    import time
    import run_rv2c as R3
    from growmoe.fairfight import _Scorer, run_fast
    t0 = time.perf_counter()
    st = R3.build_stream(cell.stream, cell.seed, False)
    steps = R3.steps_for(cell.stream)
    cfg = dataclasses.replace(R3.arm_cfg(S3_ARM, st, cell.seed, steps), **arm_flags(cell))
    holder = []

    def make():
        holder.append(R3.LoggingPool(cfg))
        return holder[0]

    run_fast(make, st, cell.seed, eval_every="final")
    pool = holder[0]
    return _record(cell, pool, lambda: _Scorer(pool, st).row().mean(), st.d, steps, time.perf_counter() - t0, {"regimes": len(st.regimes)})


def run_banking(cell: Cell) -> dict:
    import dataclasses
    import time
    import run_m0 as M0
    from growmoe.config import PhyloCfg
    from growmoe.fairfight import _Scorer, run_fast
    from growmoe.pools.phylo import PhyloMoE
    from growmoe.streams_real import banking_incremental
    t0 = time.perf_counter()
    st = banking_incremental(seed=cell.seed, frontend=M0.HybridFrontend(d=128, seed=0))
    cfg = PhyloCfg(d=st.d, n_classes=st.n_classes, seed=cell.seed, inner_steps=20, route_rule="accept_then_logf", q=0.02, alpha_spawn=0.01, m_min=32, set_gate=True,
                   set_gate_breaker_len=12, set_gate_breaker_rate=0.5, set_gate_breaker_pause=100, set_gate_confirm=2, group_veto_q=0.2)
    cfg = dataclasses.replace(cfg, **arm_flags(cell))
    holder = []

    def make():
        holder.append(PhyloMoE(cfg))
        return holder[0]

    run_fast(make, st, cell.seed, eval_every="final")
    pool = holder[0]
    return _record(cell, pool, lambda: _Scorer(pool, st).row().mean(), st.d, 20, time.perf_counter() - t0, {"regimes": len(st.regimes)})


# ------------------------------------------------------------------------------------------------ arithmetic (checkpointed: a power cut costs one block / one chunk)
def _ckpt_load(path):
    import pickle
    try:
        with open(path, "rb") as fh:                                   # written by this module (atomic) in this folder; never loaded from anywhere else
            return pickle.load(fh)
    except Exception:
        return None


def _ckpt_save(path, obj) -> None:
    import pickle
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(pickle.dumps(obj, protocol=4))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _exact_mean(learner, test, per_cell):
    import arith_lib as L
    ex, _ = L.cell_tables(test, learner.predict(test)["pred"], per_cell)
    return float(ex.mean())


def run_arith_seq(cell: Cell) -> dict:
    """Operator-by-operator arithmetic (the stage-3 pool, hidden 512, replay 20,000, 5 inner steps): exact match, forgetting, experts. One checkpoint per block."""
    import time
    import numpy as np
    import torch
    import arith_lib as L
    from growmoe import arith as A
    t0 = time.perf_counter()
    train, test, small = L.build_train("FULL", cell.seed, 8000), L.build_test(200), L.build_test(40)
    learner = L.Pool(5, True, name=cell.arm, hidden=512, replay_size=20000, **arm_flags(cell))
    ckpt = CELLS / f"{cell.cid}.ckpt"
    state = _ckpt_load(ckpt) if ckpt.exists() else None
    snaps, done = [], -1
    torch.set_num_threads(3)
    if state is not None:
        learner.pool, snaps, done = state["pool"], state["snaps"], state["done"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["np_rng"])
    else:
        learner._new(cell.seed)
    for k, (oi, idx) in enumerate(L.op_blocks(train, cell.seed)):
        if k <= done:
            continue
        learner._feed(train, idx)
        out = learner.predict(small)
        snaps.append([float(v) for v in small.exact(out["pred"]).reshape(A.N_OPS, -1).mean(-1)])
        _ckpt_save(ckpt, {"pool": learner.pool, "snaps": snaps, "done": k, "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state()})
    arr = np.array(snaps)
    forgetting = float(np.mean(arr.max(0) - arr[-1]))
    rec = _record(cell, learner.pool, lambda: _exact_mean(learner, test, 200), A.D_MODEL, 5, time.perf_counter() - t0, {"forgetting": forgetting, "snapshots": snaps})
    if ckpt.exists():
        ckpt.unlink()
    return rec


def run_arith_mixed(cell: Cell) -> dict:
    """Mixed-operator arithmetic in random order from a small (16) or ordinary (128) expert. One checkpoint per 150 batches."""
    import time
    import numpy as np
    import torch
    import arith_lib as L
    from growmoe import arith as A
    t0 = time.perf_counter()
    hidden = 16 if cell.family == "arith-mixed16" else 128
    train, test = L.build_train("FULL", cell.seed, 8000), L.build_test(200)
    learner = L.Pool(5, False, name=cell.arm, hidden=hidden, replay_size=20000, **arm_flags(cell))
    order = np.random.default_rng([cell.seed, 0x5F1]).permutation(train.n_samples)
    chunks = [order[i:i + 9600] for i in range(0, len(order), 9600)]          # 150 batches of 64
    ckpt = CELLS / f"{cell.cid}.ckpt"
    state = _ckpt_load(ckpt) if ckpt.exists() else None
    done = -1
    torch.set_num_threads(3)
    if state is not None:
        learner.pool, done = state["pool"], state["done"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["np_rng"])
    else:
        learner._new(cell.seed)
    for k, idx in enumerate(chunks):
        if k <= done:
            continue
        learner._feed(train, idx)
        _ckpt_save(ckpt, {"pool": learner.pool, "done": k, "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state()})
    rec = _record(cell, learner.pool, lambda: _exact_mean(learner, test, 200), A.D_MODEL, 5, time.perf_counter() - t0)
    if ckpt.exists():
        ckpt.unlink()
    return rec


RUNNERS = {"arith-seq": run_arith_seq, "arith-mixed16": run_arith_mixed, "arith-mixed128": run_arith_mixed, "s3": run_s3, "banking": run_banking}


def run_cell(cell: Cell) -> dict:
    return RUNNERS[cell.family](cell)
```

`run_cell.py`:

```python
"""Run ONE cell and write its result atomically. Safe to run again: a finished cell is skipped.   python run_cell.py --phase dev1 --cid <cid>"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rv2d_lib as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--cid", required=True)
    a = ap.parse_args()
    out = R.CELLS / f"{a.cid}.json"
    if R.valid_json(out):
        return
    cells = {R.Cell(**c).cid: R.Cell(**c) for c in json.loads((HERE / f"cells_{a.phase}.json").read_text(encoding="utf-8"))}
    R.CELLS.mkdir(exist_ok=True)
    R.atomic_write_text(out, json.dumps(R.run_cell(cells[a.cid])))


if __name__ == "__main__":
    main()
```

`make_cells.py` (writes the three cell lists; `dev2` needs `dev_constants.json`):

```python
"""Write the cell lists. dev1: baselines + the calibration sweeps. dev2: every arm at the chosen constants. confirm: seeds 85-99 (single use), constants frozen in bars_rv2d.json.

    python make_cells.py dev1 | dev2 | confirm
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rv2d_lib import ARMS, GAP_GRID, STREAMS, TRANSFER_GRID, Cell  # noqa: E402

DEV_ARITH, DEV_MIXED, DEV_S3, DEV_BANK = (100, 101), (100, 101, 102), (100, 101, 102), (100, 101)
CALIBRATION_STREAMS = ("F-15", "OV-0.6")                                   # the two stage-3 families of the gamma calibration (spec 6.7)
TRANSFER_STREAMS = ("F-15", "OV-0.6", "OV-0.85", "S1", "S2-5")             # guard streams for the transfer margin


def dev1():
    cells = [Cell("arith-seq", "-", "S3", s) for s in DEV_ARITH] + [Cell(f, "-", "S3", s) for f in ("arith-mixed16", "arith-mixed128") for s in DEV_MIXED]
    cells += [Cell("s3", st, "S3", s) for st in STREAMS for s in DEV_S3] + [Cell("banking", "-", "S3", s) for s in DEV_BANK]
    cells += [Cell("arith-seq", "-", "S3+T", s, tmargin=t) for t in TRANSFER_GRID for s in DEV_ARITH]
    cells += [Cell("s3", st, "S3+T", s, tmargin=t) for st in TRANSFER_STREAMS for t in TRANSFER_GRID for s in DEV_S3]
    cells += [Cell("banking", "-", "S3+T", s, tmargin=t) for t in TRANSFER_GRID for s in DEV_BANK]
    cells += [Cell(f, "-", "S3+G", s, gap=g) for f in ("arith-mixed16", "arith-mixed128") for g in GAP_GRID for s in DEV_MIXED]
    cells += [Cell("s3", st, "S3+G", s, gap=g) for st in CALIBRATION_STREAMS for g in GAP_GRID for s in DEV_S3]
    return cells


def dev2():
    c = json.loads((HERE / "dev_constants.json").read_text(encoding="utf-8"))
    g, t = c["grow_gap"], c["transfer_margin"]
    cells = []
    for arm in ("S3+T", "S3+G", "S3+T+G"):
        kw = dict(gap=g if "G" in arm else 0.0, tmargin=t if "T" in arm else 0.0)
        cells += [Cell("arith-seq", "-", arm, s, **kw) for s in DEV_ARITH]
        cells += [Cell("s3", st, arm, s, **kw) for st in STREAMS for s in DEV_S3]
        cells += [Cell("banking", "-", arm, s, **kw) for s in DEV_BANK]
        if "T" not in arm:
            cells += [Cell(f, "-", arm, s, **kw) for f in ("arith-mixed16", "arith-mixed128") for s in DEV_MIXED]
    return cells


def confirm():
    c = json.loads((HERE / "bars_rv2d.json").read_text(encoding="utf-8"))["constants"]                # frozen before any confirmatory seed
    g, t = c["grow_gap"], c["transfer_margin"]
    all15, six, five = tuple(range(85, 100)), tuple(range(85, 91)), tuple(range(85, 90))
    cells = []
    for arm in ARMS:
        kw = dict(gap=g if "G" in arm else 0.0, tmargin=t if "T" in arm else 0.0, phase="confirm")
        cells += [Cell("arith-seq", "-", arm, s, **kw) for s in six]
        cells += [Cell("s3", st, arm, s, **kw) for st in STREAMS for s in all15]
        cells += [Cell("banking", "-", arm, s, **kw) for s in five]
        if "T" not in arm:
            cells += [Cell(f, "-", arm, s, **kw) for f in ("arith-mixed16", "arith-mixed128") for s in all15]
    return cells


if __name__ == "__main__":
    which = sys.argv[1]
    cells = {"dev1": dev1, "dev2": dev2, "confirm": confirm}[which]()
    seen, uniq = set(), []
    for c in cells:
        if c.cid not in seen:
            seen.add(c.cid)
            uniq.append(c)
    (HERE / f"cells_{which}.json").write_text(json.dumps([asdict(c) for c in uniq], indent=0), encoding="utf-8")
    print(f"{which}: {len(uniq)} cells")
```

`jobqueue.py`: copy `research/growmoe_results/diagnosis/jobqueue.py` and replace `build_jobs()` by:

```python
def build_jobs(phase):
    cells = json.loads((HERE / f"cells_{phase}.json").read_text(encoding="utf-8"))
    order = {"arith-seq": 0, "arith-mixed128": 1, "arith-mixed16": 1, "banking": 2, "s3": 3}               # the long checkpointed cells first
    J = []
    for c in sorted(cells, key=lambda c: (order[c["family"]], c["seed"])):
        cid = R.Cell(**c).cid
        J.append(Job(cid, ["run_cell.py", "--phase", phase, "--cid", cid], [f"cells/{cid}.json"], 3))
    return J
```
with `BUDGET = 9`, `import rv2d_lib as R`, `phase` taken from `--phase` (default `dev1`), `is_running` matching on `--cid <cid>`, `cmd_detach` passing `--phase`, the final report call replaced by `report_rv2d.py <phase>`. Also create `RESUME_RV2D.cmd`:

```bat
@echo off
rem Resume the RV2-D runs after a power cut: finished cells are skipped, checkpointed cells resume.  Usage: RESUME_RV2D.cmd dev1|dev2|confirm
cd /d "%~dp0"
python jobqueue.py run --detach --phase %1
```
and add to `.gitignore`: `research/growmoe_results/rv2d/cells/*.ckpt`, `*.tmp`, `queue.lock`, `logs/`.

- [ ] **Step 5.6: Smoke-run one cell of every family at tiny size**

For each family run one real cell with `--phase smoke` using a shrunken configuration (an environment variable `RV2D_SMOKE=1` makes the runners use 3 blocks of 400 examples per operator, `per_cell=20`, the first two stage-3 batches... implement as a `SMOKE = bool(os.environ.get("RV2D_SMOKE"))` branch in each runner that cuts the stream; the smoke result goes to `cells_smoke/` and is deleted afterwards). Expected: every family writes a record with all keys listed in **Interfaces**; the router block has five entries; `acc` equals `router["off"]["acc"]`. Then kill a cell in the middle (`Stop-Process`) and rerun to see it resume from its checkpoint.

- [ ] **Step 5.7: Generate and run dev1**

```bash
python research/growmoe_results/rv2d/make_cells.py dev1
python research/growmoe_results/rv2d/jobqueue.py run --detach --phase dev1
python research/growmoe_results/rv2d/jobqueue.py status --phase dev1
```
Expected: about 157 cells, 3 running at a time. Estimated 4-5 hours of wall-clock; it is resumable (`RESUME_RV2D.cmd dev1`). While it runs: continue with task 6's pure parts (the verdict module) - they do not need results.

- [ ] **Step 5.8: Select the dev constants (declared rules, no judgement)**

Write `rv2d_select_cli.py` (30 lines: load all `cells/*.json` of dev1 into rows with the fields of `rv2d_select`, call the three `choose_*` functions, write `dev_constants.json` = {"grow_gap", "router_mode", "router_margin", "transfer_margin", "curves"} and `DEV_SELECTION.md` with the curves and reasons, all numbers from the functions). If a chooser returns `None`, the corresponding component is recorded as "stays off / per-family value", and the reason goes to the dev review.

- [ ] **Step 5.9: Generate and run dev2, then the dev report**

```bash
python research/growmoe_results/rv2d/make_cells.py dev2
python research/growmoe_results/rv2d/jobqueue.py run --detach --phase dev2
python research/growmoe_results/rv2d/report_rv2d.py dev          # RESULTS_DEV.md
```
`report_rv2d.py` builds, from the cell JSONs only (no hand-typed number): (1) router table per stream: accuracy off / on at the chosen setting, paired difference with the 95 % interval from `paired_diff`, the selected rule and its cross-validated accuracy; (2) transfer table: S3+T minus S3 per stream (accuracy, forgetting for arithmetic, experts, spawns, extends); (3) growth table: the gamma curve (score, gain, parameter increase, growth events per group), width reached, reverts, extra forward cost `post_rows / (train_rows * inner_steps * 6)`; (4) the full stack; (5) wall-clock per arm.

- [ ] **Step 5.10: Commit the runner and the dev results**

```bash
git add research/growmoe_results/rv2d tests/growmoe/test_gm_rv4_runner.py .gitignore
git commit -m "research(growmoe): RV2-D runner (resumable cells), declared selection rules, dev runs of every arm"
```
(`cells/` JSONs are small and are evidence; `.ckpt`, `.tmp`, `logs/` are ignored.)

---

## Task 6: Dev review, amendments, verdict functions, mutation spot-check

**Files:**
- Create: `growmoe/verdicts_rv4.py`, `tests/growmoe/test_gm_rv4_verdicts.py`
- Create: `research/growmoe_results/rv2d/mutation_check.py`, `docs/superpowers/specs/2026-10-15-routing-v2-stage4-AMENDMENTS.md`
- Modify (only if the review demands it, each with a recorded reason): product files and tests

**Interfaces:**
- Produces: `growmoe.verdicts_rv4.rv2d_verdicts(runs, bars, golden) -> {id: {"rule", "verdict", "detail", "expected", "as_expected"}}` for S4-1 ... S4-9 and `rv2d_decision(result) -> {"4a": ..., "4b": ..., "4c": ..., "text": ...}`. `runs` = the cell records (rows with `family`, `stream`, `arm`, `seed` added by `report_rv2d.load()`); `bars` = `bars_rv2d.json["bars"]`.

- [ ] **Step 6.1: Verdict functions, test first**

Create `tests/growmoe/test_gm_rv4_verdicts.py`:

```python
"""RV2-D verdicts: the statements S4-1 to S4-9 and the decision rule (spec section 8). Records are synthetic; the point is that every bar is read from the bars and that one broken
effect breaks exactly its own statement."""

import copy

import pytest

from growmoe.verdicts_rv4 import IDS, rv2d_decision, rv2d_verdicts

SEL = "restrict-0.01"
BARS = {"router_setting": SEL}
S3_STREAMS = ("F-30", "S2-30", "F-15", "OV-0.85", "OV-0.6", "OV-0.3", "OV-0.3-MIXED", "OV-0.0", "S1", "S2-5")


def rec(family, stream, arm, seed, acc, on=None, experts=4, params=1000, grow=0, forgetting=None, hidden=128, post=0, train=1):
    return {"family": family, "stream": stream, "arm": arm, "seed": seed, "acc": acc, "router": {"off": {"acc": acc}, SEL: {"acc": acc if on is None else on}}, "n_experts": experts,
            "params": params, "grow_events": [{}] * grow, "final_hidden": [hidden], "forgetting": forgetting, "grow_stats": {"post_rows": post, "train_rows": train}, "inner_steps": 5}


def good_runs():
    """A run set in which every statement holds with room to spare: 15 seeds for the stage-3 streams and the mixed arithmetic, 6 for operator-by-operator arithmetic, 5 for banking77."""
    runs = []
    for s in range(85, 100):
        runs += [rec("arith-mixed128", "-", "S3", s, 0.34), rec("arith-mixed128", "-", "S3+G", s, 0.34, params=1100, grow=1),
                 rec("arith-mixed16", "-", "S3+G", s, 0.32, hidden=128, post=3, train=100)]
        for st in S3_STREAMS:
            runs += [rec("s3", st, "S3", s, 0.40, on=0.50 if st == "OV-0.85" else 0.40), rec("s3", st, "S3+T", s, 0.40), rec("s3", st, "S3+G", s, 0.40, params=1050, grow=1),
                     rec("s3", st, "S3+T+G", s, 0.40)]
    for s in range(85, 91):
        runs += [rec("arith-seq", "-", "S3", s, 0.27, on=0.40, forgetting=0.15, experts=3), rec("arith-seq", "-", "S3+T", s, 0.36, forgetting=0.07, experts=6),
                 rec("arith-seq", "-", "S3+T+G", s, 0.36, on=0.46, forgetting=0.07, experts=6)]
    for s in range(85, 90):
        runs += [rec("banking", "-", "S3", s, 0.70, experts=6), rec("banking", "-", "S3+T", s, 0.70, experts=7), rec("banking", "-", "S3+G", s, 0.70)]
    return runs


def _edit(runs, where, **change):
    out = copy.deepcopy(runs)
    for r in out:
        if where(r):
            for k, v in change.items():
                if k == "on":
                    r["router"][SEL]["acc"] = v
                elif k == "off":
                    r["router"]["off"]["acc"] = v
                elif k == "experts":
                    r["n_experts"] = v
                elif k == "grow":
                    r["grow_events"] = [{}] * v
                else:
                    r[k] = v
    return out


BREAKS = {
    "S4-1": lambda runs: _edit(runs, lambda r: r["family"] == "arith-seq" and r["arm"] == "S3", on=0.29),
    "S4-2": lambda runs: _edit(runs, lambda r: r["stream"] == "OV-0.85" and r["arm"] == "S3", on=0.40),
    "S4-3": lambda runs: _edit(runs, lambda r: r["stream"] == "S1" and r["arm"] == "S3", on=0.30),
    "S4-4": lambda runs: _edit(runs, lambda r: r["family"] == "arith-seq" and r["arm"] == "S3+T", acc=0.30),
    "S4-5": lambda runs: _edit(runs, lambda r: r["family"] == "banking" and r["arm"] == "S3+T", experts=12),
    "S4-6": lambda runs: _edit(runs, lambda r: r["family"] == "arith-mixed16", acc=0.20),
    "S4-7": lambda runs: _edit(runs, lambda r: r["family"] == "arith-mixed128" and r["arm"] == "S3+G", grow=3),
    "S4-8": lambda runs: _edit(runs, lambda r: r["family"] == "arith-seq" and r["arm"] == "S3+T+G", on=0.40),
}


def test_every_statement_passes_when_every_effect_is_large():
    res = rv2d_verdicts(good_runs(), BARS, {"differences": []})
    assert {k: v["verdict"] for k, v in res.items()} == {k: "PASS" for k in IDS}
    assert all(v["rule"] for v in res.values())
    d = rv2d_decision(res)
    assert d["4a"] and d["4b"] and d["4c"]


@pytest.mark.parametrize("sid", sorted(BREAKS))
def test_breaking_one_effect_breaks_exactly_its_statement(sid):
    res = rv2d_verdicts(BREAKS[sid](good_runs()), BARS, {"differences": []})
    assert {k: v["verdict"] != "PASS" for k, v in res.items()} == {k: k == sid for k in IDS}


def test_the_decision_rule_adopts_each_component_on_its_own_statements():
    d = rv2d_decision(rv2d_verdicts(BREAKS["S4-3"](good_runs()), BARS, {"differences": []}))
    assert (d["4a"], d["4b"], d["4c"]) == (False, True, True)                      # the router guard failed: 4a stays off, the others proceed
    d = rv2d_decision(rv2d_verdicts(BREAKS["S4-1"](good_runs()), BARS, {"differences": []}))
    assert (d["4a"], d["4b"], d["4c"]) == (True, True, True)                       # S4-2 alone carries 4a with S4-3 ("S4-1 or S4-2")
    d = rv2d_decision(rv2d_verdicts(BREAKS["S4-7"](good_runs()), BARS, {"differences": []}))
    assert (d["4a"], d["4b"], d["4c"]) == (True, True, False)


def test_a_failing_golden_check_stops_everything():
    res = rv2d_verdicts(good_runs(), BARS, {"differences": ["default|S1-smoke"]})
    d = rv2d_decision(res)
    assert res["S4-9"]["verdict"] == "FAIL" and not (d["4a"] or d["4b"] or d["4c"]) and "bug, not a result" in d["text"]


def test_missing_data_is_not_measurable_and_blocks_the_decision():
    res = rv2d_verdicts([r for r in good_runs() if r["family"] != "banking"], BARS, {"differences": []})
    assert res["S4-3"]["verdict"] != "PASS" and res["S4-5"]["verdict"] != "PASS"
    d = rv2d_decision(rv2d_verdicts([r for r in good_runs() if r["arm"] != "S3+T+G"], BARS, {"differences": []}))
    assert "cannot" in d["text"] and not (d["4a"] or d["4b"] or d["4c"])
```

Run red (`ModuleNotFoundError: growmoe.verdicts_rv4`). Then create `growmoe/verdicts_rv4.py`:

```python
"""RV2-D (routing v2 stage 4): the statements S4-1 to S4-9 on per-cell run records, and the decision rule (spec docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md, section 8).

A record carries ``family``, ``stream``, ``arm``, ``seed``, ``acc`` (router off), ``router`` ({"off": {"acc"}, "<mode>-<margin>": {"acc", ...}}), ``n_experts``, ``params``,
``grow_events``, ``final_hidden``, ``forgetting`` (operator-by-operator arithmetic), ``grow_stats``, ``inner_steps``. Differences are in points (x100), paired by seed.
A multi-stream statement is PASS when it holds on all, FAIL on none, PARTIAL otherwise, NOT MEASURABLE when data are missing (the rule of the earlier stages).
"""

import math
from typing import Optional, Sequence

import numpy as np

from .fairfight import _combine
from .metrics import paired_diff

_POINTS = 100.0
IDS = tuple(f"S4-{i}" for i in range(1, 10))
EXPECTED_RV2D = {k: "PASS" for k in IDS}
S3_STREAMS = ("F-30", "S2-30", "F-15", "OV-0.85", "OV-0.6", "OV-0.3", "OV-0.3-MIXED", "OV-0.0", "S1", "S2-5")
GUARD_STREAMS = tuple(s for s in S3_STREAMS if s != "OV-0.85")
BARS_RV2D = {"S4_1_lo": 5.0, "S4_2_lo": 4.0, "S4_3": -0.5, "S4_4_gain": 5.0, "S4_4_forgetting": 9.0, "S4_4_min_experts": 5, "S4_4_seeds_below_allowed": 1,
             "S4_5_gain": -0.5, "S4_5_ratio": 1.2, "S4_6_frac": 0.9, "S4_6_seed_share": 13 / 15, "S4_6_width": (64, 256), "S4_6_cost": 0.05,
             "S4_7_gain": -0.5, "S4_7_grow": 1.0, "S4_7_params": 0.25, "S4_8_acc": 44.0, "S4_8_forgetting": 8.0}


def _fin(x: float) -> Optional[float]:
    return float(x) if math.isfinite(x) else None


def _index(runs: Sequence[dict]) -> dict:
    ix: dict = {}
    for r in runs:
        ix.setdefault((r["family"], r["stream"], r["arm"]), {})[int(r["seed"])] = r
    return ix


def _diff(a: Optional[dict], b: Optional[dict], fa, fb) -> Optional[dict]:
    """Paired difference fa(record of a) - fb(record of b) in points over the seeds both have; None when there is no common seed."""
    if not a or not b:
        return None
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    d = paired_diff([fa(a[s]) * _POINTS for s in seeds], [fb(b[s]) * _POINTS for s in seeds])
    return {"n": d["n"], "mean_diff_points": d["mean_diff"], "ci_lo": _fin(d["ci_lo"]), "ci_hi": _fin(d["ci_hi"])}


def _holds(d: Optional[dict], condition) -> Optional[dict]:
    if d is not None:
        d["holds"] = bool(condition(d))
    return d


def _mean(recs: dict, f) -> float:
    return float(np.mean([f(r) for r in recs.values()]))


def rv2d_verdicts(runs: Sequence[dict], bars: dict, golden: Optional[dict] = None, expected: Optional[dict] = None) -> dict:
    bars = {**BARS_RV2D, **bars}
    sel = bars["router_setting"]
    on, off, acc = (lambda r: r["router"][sel]["acc"]), (lambda r: r["router"]["off"]["acc"]), (lambda r: r["acc"])
    ix = _index(runs)
    g = lambda family, stream, arm: ix.get((family, stream, arm))                                                   # noqa: E731
    detail = {}

    def router_gain(family, stream):
        recs = g(family, stream, "S3")
        return _diff(recs, recs, on, off)

    detail["S4-1"] = _holds(router_gain("arith-seq", "-"), lambda d: d["ci_lo"] is not None and d["ci_lo"] >= bars["S4_1_lo"])
    detail["S4-2"] = _holds(router_gain("s3", "OV-0.85"), lambda d: d["ci_lo"] is not None and d["ci_lo"] >= bars["S4_2_lo"])
    detail["S4-3"] = {f"{f}:{s}": _holds(router_gain(f, s), lambda d: d["mean_diff_points"] >= bars["S4_3"]) for f, s in [("s3", s) for s in GUARD_STREAMS] + [("banking", "-")]}

    s3, t = g("arith-seq", "-", "S3"), g("arith-seq", "-", "S3+T")
    d4 = _diff(t, s3, acc, acc)
    if d4 is not None:
        forget = _POINTS * _mean(t, lambda r: r["forgetting"])
        below = sum(r["n_experts"] < bars["S4_4_min_experts"] for r in t.values())
        d4.update(forgetting_points=forget, seeds_below_min_experts=below, n_seeds=len(t))
        d4["holds"] = bool(d4["mean_diff_points"] >= bars["S4_4_gain"] and forget <= bars["S4_4_forgetting"] and below <= bars["S4_4_seeds_below_allowed"])
    detail["S4-4"] = d4

    def transfer_guard(family, stream):
        a, b = g(family, stream, "S3+T"), g(family, stream, "S3")
        d = _diff(a, b, acc, acc)
        if d is not None:
            ratio = _mean(a, lambda r: r["n_experts"]) / max(_mean(b, lambda r: r["n_experts"]), 1e-9)
            d.update(expert_ratio=ratio)
            d["holds"] = bool(d["mean_diff_points"] >= bars["S4_5_gain"] and ratio <= bars["S4_5_ratio"])
        return d
    detail["S4-5"] = {f"{f}:{s}": transfer_guard(f, s) for f, s in [("s3", s) for s in GUARD_STREAMS] + [("banking", "-")]}

    small, big = g("arith-mixed16", "-", "S3+G"), g("arith-mixed128", "-", "S3")
    d6 = None
    if small and big and set(small) & set(big):
        seeds = sorted(set(small) & set(big))
        reach = sum(small[s]["acc"] >= bars["S4_6_frac"] * big[s]["acc"] for s in seeds)
        need = math.ceil(bars["S4_6_seed_share"] * len(seeds) - 1e-9)
        width = float(np.mean([np.mean(small[s]["final_hidden"]) for s in seeds]))
        cost = float(np.mean([small[s]["grow_stats"]["post_rows"] / max(small[s]["grow_stats"]["train_rows"] * small[s]["inner_steps"] * 6, 1) for s in seeds]))
        lo, hi = bars["S4_6_width"]
        d6 = {"n": len(seeds), "seeds_reaching": reach, "needed": need, "mean_final_width": width, "extra_forward_cost": cost,
              "holds": bool(reach >= need and lo <= width <= hi and cost <= bars["S4_6_cost"])}
    detail["S4-6"] = d6

    def growth_guard(family, stream):
        a, b = g(family, stream, "S3+G"), g(family, stream, "S3")
        d = _diff(a, b, acc, acc)
        if d is not None:
            seeds = sorted(set(a) & set(b))
            grows = float(np.mean([len(a[s]["grow_events"]) for s in seeds]))
            relp = float(np.mean([a[s]["params"] / b[s]["params"] - 1.0 for s in seeds]))
            d.update(growth_events_per_run=grows, rel_params=relp)
            d["holds"] = bool(d["mean_diff_points"] >= bars["S4_7_gain"] and grows <= bars["S4_7_grow"] and relp <= bars["S4_7_params"])
        return d
    detail["S4-7"] = {f"{f}:{s}": growth_guard(f, s) for f, s in [("arith-mixed128", "-")] + [("s3", s) for s in S3_STREAMS]}

    full = g("arith-seq", "-", "S3+T+G")
    d8 = None
    if full:
        a8, f8 = _POINTS * _mean(full, on), _POINTS * _mean(full, lambda r: r["forgetting"])
        d8 = {"n": len(full), "exact_match_points": a8, "forgetting_points": f8, "holds": bool(a8 >= bars["S4_8_acc"] and f8 <= bars["S4_8_forgetting"])}
    detail["S4-8"] = d8
    detail["S4-9"] = None if golden is None else {"differences": list(golden["differences"]), "holds": not golden["differences"]}

    exp = {**EXPECTED_RV2D, **(expected or {})}
    rules = {
        "S4-1": f"arithmetic operator-by-operator, S3 pools: exact match with the router minus without, interval lower bound >= +{bars['S4_1_lo']:.1f} points",
        "S4-2": f"OV-0.85, S3 pools: router minus no router, interval lower bound >= +{bars['S4_2_lo']:.1f} points",
        "S4-3": f"each other stage-3 stream and banking77, S3 pools: router minus no router, mean >= {bars['S4_3']:+.1f} points",
        "S4-4": f"arithmetic operator-by-operator: S3+T minus S3 mean >= +{bars['S4_4_gain']:.1f} points, mean forgetting <= {bars['S4_4_forgetting']:.1f} points, at most "
                f"{bars['S4_4_seeds_below_allowed']} seed with fewer than {bars['S4_4_min_experts']} experts",
        "S4-5": f"each other stage-3 stream and banking77: S3+T minus S3 mean >= {bars['S4_5_gain']:+.1f} points and mean experts <= {bars['S4_5_ratio']} x S3's",
        "S4-6": f"arithmetic mixed from 16 units, S3+G: accuracy >= {bars['S4_6_frac']} x the 128-unit S3 of the same seed in at least {bars['S4_6_seed_share']:.2f} of the seeds, mean final width in "
                f"{bars['S4_6_width']}, extra forward cost <= {100 * bars['S4_6_cost']:.0f} % of training",
        "S4-7": f"arithmetic mixed from 128 units and every stage-3 stream: S3+G minus S3 mean >= {bars['S4_7_gain']:+.1f} points, growth events per run <= {bars['S4_7_grow']}, "
                f"relative parameter increase <= {bars['S4_7_params']}",
        "S4-8": f"arithmetic operator-by-operator, S3+T+G with the router: exact match >= {bars['S4_8_acc']:.0f} points and forgetting <= {bars['S4_8_forgetting']:.0f} points",
        "S4-9": "all stage-4 flags off: the pool is bit-identical to the recorded one (golden fingerprints of rv2c/golden.py)",
    }
    verdict = {k: _combine(v) for k, v in detail.items()}
    return {k: {"rule": rules[k], "verdict": verdict[k], "detail": detail[k], "expected": exp[k], "as_expected": None if verdict[k] == "NOT MEASURABLE" else verdict[k] == exp[k]} for k in IDS}


def rv2d_decision(result: dict) -> dict:
    """What the pre-registration says to do: ``{"4a": bool, "4b": bool, "4c": bool, "text": str}``. 4a needs (S4-1 or S4-2) and S4-3, 4b needs S4-4 and S4-5, 4c needs S4-6 and S4-7;
    S4-9 failing is a bug, not a result; S4-8 (the full stack) flags the adoption when it misses."""
    none = {"4a": False, "4b": False, "4c": False}
    v = {k: result[k]["verdict"] for k in IDS}
    missing = [k for k in IDS if v[k] == "NOT MEASURABLE"]
    if missing:
        return {**none, "text": "The RV2-D decision cannot be applied: " + ", ".join(missing) + " not measurable."}
    if v["S4-9"] != "PASS":
        return {**none, "text": "S4-9 FAIL: with all stage-4 flags off the pool is not bit-identical to the recorded one. Stop: this is a bug, not a result."}
    a = (v["S4-1"] == "PASS" or v["S4-2"] == "PASS") and v["S4-3"] == "PASS"
    b = v["S4-4"] == "PASS" and v["S4-5"] == "PASS"
    c = v["S4-6"] == "PASS" and v["S4-7"] == "PASS"
    parts = [f"4a (router) {'adopt: router=True becomes the default' if a else 'NOT adopted: stays off'}",
             f"4b (transfer rule) {'adopt: spawn_rule=transfer becomes the default' if b else 'NOT adopted: stays off'}",
             f"4c (growth from the fit gap) {'adopt: growth_rule=gap becomes the default' if c else 'NOT adopted: stays off'}"]
    if (a or b or c) and v["S4-8"] != "PASS":
        parts.append(f"the full stack missed S4-8 ({v['S4-8']}): adoption is flagged, say so everywhere")
    return {"4a": a, "4b": b, "4c": c, "text": "; ".join(parts) + "."}
```

Run green: `python -m pytest tests/growmoe/test_gm_rv4_verdicts.py -q -p no:cacheprovider`. (`_combine` is the private helper of `growmoe/fairfight.py`; if importing it is awkward, copy its six lines.)

- [ ] **Step 6.2: The dev review (read, do not tune)**

Read `RESULTS_DEV.md` and `DEV_SELECTION.md` against the predictions of the spec (section 8) and write the answers into `docs/superpowers/specs/2026-10-15-routing-v2-stage4-AMENDMENTS.md`, one line each with the number: (a) router: selected rule per stream, gain on arithmetic and OV-0.85, any stream below -0.5; (b) transfer: gain on arithmetic, forgetting, expert counts everywhere (banking77 will spawn at every regime: is the count within 1.2 x?), false spawns on iid streams; (c) growth: width reached from 16, the whole gamma curve, **revert rate of the guard at the product settings** (if more than 20 % of growths revert, the 2 % threshold is noise: raise `grow_guard_eps` or `grow_guard_window` here, as a recorded amendment, and rerun the affected dev cells), growth events on 128-unit starts; (d) the stage-3 statements S3-1 to S3-9 still hold with every component on (rerun the stage-3 tests and `golden.py check`; the S3 streams are the S3 arm itself); (e) anything surprising. Amendments may change a constant or a mechanism before the pre-registration; each states what it changes and why. Rerun only the affected cells.

- [ ] **Step 6.3: Mutation spot-check of the decision functions**

Create `research/growmoe_results/rv2d/mutation_check.py`. Mutants are applied to a COPY of the code in a temporary directory, so a power cut can never leave one in the repository; the baseline (unmutated) tests must pass in the copy first.

```python
"""Mutation spot-check of the stage-4 decision functions (spec section 9: at least 90 % of the mutants must be killed).

    python mutation_check.py [--only router|transfer|gapgrow|phylo]

For each mutant: one exact-text replacement in a temporary copy of growmoe/, tests/growmoe/ and the golden files, then the owning tests with `pytest -x`. Killed = the tests fail.
A survivor needs either a new test (red, green) or a written reason that the mutant is equivalent.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RT = ["tests/growmoe/test_gm_rv4_router.py"]
RP = ["tests/growmoe/test_gm_rv4_router_pool.py"]
TT = ["tests/growmoe/test_gm_rv4_transfer.py"]
TP = ["tests/growmoe/test_gm_rv4_transfer_pool.py"]
GT = ["tests/growmoe/test_gm_rv4_growth.py"]
GP = ["tests/growmoe/test_gm_rv4_growth_pool.py"]

MUTANTS = [   # (group, name, file, exact old text, new text, owning tests)
    ("router", "margin: > becomes >=", "growmoe/router.py", 'cv[best] > cv["density"] + margin', 'cv[best] >= cv["density"] + margin', RT),
    ("router", "margin dropped", "growmoe/router.py", 'cv[best] > cv["density"] + margin', 'cv[best] > cv["density"]', RT),
    ("router", "hysteresis: > becomes >=", "growmoe/router.py", "not cv[candidate] > cv[current] + margin", "not cv[candidate] >= cv[current] + margin", RT),
    ("router", "folds: one extra fold id", "growmoe/router.py", "torch.arange(idx.numel()) % folds", "torch.arange(idx.numel()) % (folds + 1)", RT),
    ("router", "folds: train on the held-out fold", "growmoe/router.py", "tr = ~te", "tr = te", RT),
    ("router", "softmax: class weights dropped", "growmoe/router.py", "cls_w = (n / (k * counts.clamp_min(1))).float()", "cls_w = torch.ones(k)", RT),
    ("router", "lda: empty class not masked", "growmoe/router.py", "(Z.double() @ self.w + self.b).masked_fill(~self.present, NEG_INF).float()", "(Z.double() @ self.w + self.b).float()", RT),
    ("transfer", "conflict: > margin becomes >=", "growmoe/transfer.py", "bool(p <= alpha and diff > margin)", "bool(p <= alpha and diff >= margin)", TT),
    ("transfer", "conflict: and becomes or", "growmoe/transfer.py", "bool(p <= alpha and diff > margin)", "bool(p <= alpha or diff > margin)", TT),
    ("transfer", "ties not counted", "growmoe/transfer.py", ">= obs).sum())", "> obs).sum())", TT),
    ("gapgrow", "gap formula", "growmoe/gapgrow.py", "else 1.0 - s_post / s_pre", "else s_post / s_pre", GT),
    ("gapgrow", "improvement inverted", "growmoe/gapgrow.py", "else last / prev", "else prev / last", GT),
    ("gapgrow", "plateau: >= becomes >", "growmoe/gapgrow.py", "imp is not None and imp >= plateau", "imp is not None and imp > plateau", GT),
    ("gapgrow", "age: >= becomes >", "growmoe/gapgrow.py", "self.age >= min_age and", "self.age > min_age and", GT),
    ("gapgrow", "guard: > becomes >=", "growmoe/gapgrow.py", "return after > (1.0 + eps) * before", "return after >= (1.0 + eps) * before", GT),
    ("gapgrow", "increment: >= becomes >", "growmoe/gapgrow.py", "return k if k >= min_widen else 0", "return k if k > min_widen else 0", GT),
    ("phylo", "router: restrict accepts p == q no more", "growmoe/pools/phylo.py", "(P.double() >= float(self.cfg.q))\n        best_accepted", "(P.double() > float(self.cfg.q))\n        best_accepted", RP),
    ("phylo", "router: young expert threshold", "growmoe/pools/phylo.py", "min(counts) < cfg.router_min_protos", "min(counts) <= cfg.router_min_protos", RP),
    ("phylo", "growth: gap >= becomes >", "growmoe/pools/phylo.py", "gap >= cfg.grow_gap", "gap > cfg.grow_gap", GP),
    ("phylo", "growth: budget > becomes >=", "growmoe/pools/phylo.py", "self._live_params() + added > cfg.grow_budget_params", "self._live_params() + added >= cfg.grow_budget_params", GP),
    ("phylo", "transfer: min reference < becomes <=", "growmoe/pools/phylo.py", "if len(ref) < cfg.transfer_min_ref:", "if len(ref) <= cfg.transfer_min_ref:", TP),
    ("phylo", "transfer: alarm path keeps the cluster in the reference", "growmoe/pools/phylo.py", "return ref[:-exclude] if exclude > 0 else ref", "return ref", TP),
    ("phylo", "transfer: margin scale", "growmoe/pools/phylo.py", "math.log(max(int(self.experts[eid].module.seen.sum()), 2))", "math.log(max(int(self.experts[eid].module.seen.sum()), 1))", TP),
]


def prepare(tmp: Path) -> None:
    for sub in ("growmoe", "tests/growmoe"):
        shutil.copytree(ROOT / sub, tmp / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    d = tmp / "research/growmoe_results/rv2c"
    d.mkdir(parents=True)
    for f in ("golden.py", "golden_flags_off.json"):
        shutil.copy(ROOT / "research/growmoe_results/rv2c" / f, d / f)


def run_tests(tmp: Path, tests) -> bool:
    """True when the tests PASS."""
    p = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests], cwd=tmp, env={**os.environ, "PYTHONPATH": str(tmp)},
                       capture_output=True, text=True, timeout=1800)
    return p.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="rv2d_mut_") as t:
        tmp = Path(t)
        prepare(tmp)
        owners = sorted({tuple(m[5]) for m in MUTANTS if a.only in (None, m[0])})
        for tests in owners:
            assert run_tests(tmp, list(tests)), f"the unmutated tests {tests} do not pass in the copy: fix that first"
        results = []
        for group, name, rel, old, new, tests in MUTANTS:
            if a.only not in (None, group):
                continue
            path = tmp / rel
            src = path.read_text(encoding="utf-8")
            assert src.count(old) == 1, f"{name}: the old text occurs {src.count(old)} times in {rel}"
            path.write_text(src.replace(old, new), encoding="utf-8")
            try:
                survived = run_tests(tmp, tests)
            finally:
                path.write_text(src, encoding="utf-8")
            results.append({"group": group, "name": name, "killed": not survived})
            print(f"{'KILLED  ' if not survived else 'SURVIVED'} {group}: {name}", flush=True)
    killed = sum(r["killed"] for r in results)
    print(f"\n{killed}/{len(results)} mutants killed ({100 * killed / max(len(results), 1):.0f} %)")
    (HERE / "mutation_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
```

Run it (about 45-90 minutes; it is not resumable, but it only touches a temporary copy). Expected: at least 90 % killed. For every survivor: add the test that kills it (red on the mutant, green on the code) or write why it is equivalent into the amendments file.

- [ ] **Step 6.4: Notebook, memory, commit**

Notebook: an experiment entry for the dev runs (question, prediction from the spec, method, the tables copied from `RESULTS_DEV.md`, interpretation in plain words first, what went wrong, consequence), claims for what dev shows (grade B, exploratory, not pre-registered), failures if any; `labnb.py index/manifest/verify`; memory note updated. Commit `docs`, `growmoe/verdicts_rv4.py`, tests, `mutation_check.py`, results.

---

## Task 7: Pre-registration (then STOP for the user's go)

**Files:** Create `research/growmoe_results/rv2d/freeze_prereg.py`, then its outputs `PREREG.md`, `bars_rv2d.json`, `bars_rv2d.sha256`.

- [ ] **Step 7.1: Write and run `freeze_prereg.py`** (pattern: `cmd_prereg` of `run_rv2c.py`). It refuses to run if `bars_rv2d.sha256` exists. It writes `bars_rv2d.json` = {"bars": the numbers of task 6.1 (copied from the spec's table, not from dev), "constants": {"grow_gap", "router_mode", "router_margin", "transfer_margin", "grow_guard_eps", "grow_guard_window"} from `dev_constants.json` and the amendments, "arms", "streams", "seeds": confirmatory 85-99 (arithmetic operator-by-operator 85-90, banking77 85-89), "expected": the dev verdicts (`NOT MEASURABLE` counted as PASS as in stage 3), "golden": the current differences list and the recorded file's sha256, "written": timestamp}, freezes it with `run_m0.freeze_check` (sha256 file), and writes `PREREG.md`: the statements with their rules, the arms, the constants and how each was chosen (the declared rules and the dev curves), the decision rules verbatim from the spec (section 8), the kill rules, what is reported and not gated, provenance (sha256 of `phylo.py`, `router.py`, `transfer.py`, `gapgrow.py`, `config.py`, `accounting.py`, `verdicts_rv4.py`, `rv2d_lib.py`, `golden_flags_off.json`, `bars_rv2d.json`).
- [ ] **Step 7.2: Checks.** `python research/growmoe_results/rv2c/golden.py check` = bit-identical; the full suite green; `labnb.py verify` = 0 problems.
- [ ] **Step 7.3: Commit** (`PREREG.md`, `bars_rv2d.json`, `bars_rv2d.sha256`) and **STOP**: present to the user the dev summary (what each component did on dev in plain words, the chosen constants, the expected verdicts), the wall-clock estimate of the confirmatory run from the measured cell times, and ask for the go. **Do not start any cell with a seed from 85 to 99 before the go.**

---

## Task 8: Confirmatory run, report, independent audit, adoption

**Files:** `research/growmoe_results/rv2d/RESULTS_RV2D.md` (generated); adoption edits listed in step 8.5.

- [ ] **Step 8.1: Run (after the go).** `python make_cells.py confirm` (about 700 cells: ten stage-3 streams x 15 seeds x 4 arms, arithmetic operator-by-operator 6 seeds x 4, banking77 5 seeds x 4, arithmetic mixed 15 seeds x 2 widths x 2 arms), `python jobqueue.py run --detach --phase confirm`; after any outage `RESUME_RV2D.cmd confirm`. Nothing is rerun, dropped or re-seeded for being inconvenient; a cell that crashes is reported as crashed.
- [ ] **Step 8.2: Report.** `python report_rv2d.py confirm` writes `RESULTS_RV2D.md` (every statement with its rule, verdict, paired difference, interval, n; the decision per component from `rv2d_decision`; reported-not-gated tables).
- [ ] **Step 8.3: Independent audit (D-14).** One fresh general-purpose subagent gets only `PREREG.md`, `bars_rv2d.json`, the `cells/` folder and the instruction to recompute S4-1 to S4-9 with its own script (without importing `verdicts_rv4`), run `golden.py check`, compare with `RESULTS_RV2D.md` and list every discrepancy, every crashed or missing cell and every place where the report says more than the data show. Its findings are fixed or recorded; the audit goes into the notebook entry.
- [ ] **Step 8.4: Notebook.** The experiment entry (plain words first, tables copied from the report, caveats), claims with grade A for what the pre-registered run decided, the decision rows, failures. `labnb.py index/manifest/verify`.
- [ ] **Step 8.5: Adoption, only per `rv2d_decision`, one component at a time.** For each adopted component: flip its default in `PhyloCfg` (test first: change the default-value tests of `test_gm_rv4_config.py` to the new default, watch them fail, flip, watch them pass), pin the old behaviour (`router=False` / `spawn_rule="probe"` / `growth_rule="probe"`) in the historical entry points exactly where stage 3 pinned `set_gate=False`: `growmoe/bench._phase1_cfg`, `research/growmoe_results/m0/run_m0.py` (`A_KW`), `rv2a/run_rv2a.py`, `rv2b/run_rv2b.py` BASE, `rv2c/run_rv2c.py` BASE, the M5 scripts, the diagnosis and spike runners that must reproduce old numbers (find them with `grep -rn "set_gate=False" research growmoe`), extend the pin tests (`test_gm_rv2a_pins.py`, `test_gm_rv2b_pins.py`) to cover the new flags, update the comments in `config.py` (the ON-since line with the evidence), the GrowMoE docs that describe the defaults, the memory notes. A component whose statement failed stays OFF and every document says so. Full suite and full golden check after each flip (the golden file pins the flags OFF, so it must stay identical).
- [ ] **Step 8.6: One fresh reviewer for the whole branch.** Dispatch one reviewer (the code-review skill or a general-purpose subagent on the most capable model) with: the spec, this plan, the diff of `growmoe/` and `tests/growmoe/` since the commit before task 1, the PREREG and RESULTS. Ask for correctness bugs, silent no-ops, test gaps, pickling and determinism problems, and places where the code differs from the spec. Fix what is real, record the rest, commit.
- [ ] **Step 8.7: Final commit** with the notebook, memory and the manifest refreshed; report to the user in plain words: what each component did, what is now default, what stayed off, what remains (Phase B, open questions Q1 to Q7).

---

## Phase B: research spikes with kill rules (only after task 8's first results; not part of the confirmatory claim)

- **P1, splitting a stationary mixture (4d).** Script `research/growmoe_results/stage4_spikes/spike_split.py`: train the iid mixed-operator pool (one expert, hidden 512) on dev seeds 100-101, take the expert's last 2,000 assigned inputs, cluster them (k-means in the input space, k = 2 to 6), compute the normalised mutual information with the operator labels and the permutation test (below 0.01) of per-cluster prequential losses. **Kill rule:** NMI below 0.6 for every k -> record "not feasible by this route" in the notebook and stop. Otherwise write a design amendment for an online split (propose by clustering, test by paired prequential comparison, accept; duplicates are the merge's job) and ask the user.
- **P2, the streaming recipe (4e).** Script `spike_recipe.py`: per-expert learning-rate decay with restarts at birth, alarm and growth (a linear decay to 10 % over 2,000 batches), and Schedule-Free AdamW if its implementation is under 60 lines, on arithmetic iid and operator-by-operator at dev seeds. **Kill rule:** less than +2 points on arithmetic (iid and operator-by-operator) or any guard stream below -0.5 -> drop and record.

---

## Self-review (run before handing over; results recorded here when the plan is committed)

1. **Spec coverage.** Section 2 (findings) is background. Section 3 scope: 4a task 2, 4b task 3, 4c task 4, config/counters/events/accounting tasks 1-4, one pre-registered run tasks 5-8, adoption task 8.5, Phase B at the end. Section 4.1 pure functions -> 2a; 4.2 pool changes -> 2b (R-1 lazy refit, R-4 refusal); 4.3 config -> task 1; 4.4 accounting -> 2b step 2.7(f) and R-3; 4.5 tests -> 2a/2b; 4.6 research arms (stacked candidate, MLP, soft mixing) -> measured on dev only through an amendment (not built). Section 5.1 -> 3a; 5.2 -> 3b (R-2); 5.3 -> task 1; 5.4 -> 3a/3b; 5.5 merge audit -> 3b. Section 6.1-6.3 -> 4a/4b; 6.4 not built (depth); 6.5 -> task 1; 6.6 -> 4b; 6.7 calibration protocol -> `choose_grow_gap` (task 5); 6.8 risks -> dev review 6.2. Section 7 -> Phase B. Section 8 evaluation -> tasks 5-8 (arms, streams, seeds, statements S4-1 to S4-9, decision and kill rules in `verdicts_rv4`). Section 9 tests/outage/honesty -> every task; mutation check -> 6.3. Section 10 task order -> this plan. Section 11 open questions stay open. Section 12 decisions -> recorded in D-31 and D-32.
2. **Placeholder scan.** The only constants left open on purpose are the ones the spec declares as calibrated on dev (`grow_gap`, router mode and margin, transfer margin), chosen by functions with unit tests. The report formatting code of `report_rv2d.py` and `rv2d_select_cli.py` is specified by its tables and written at build time; it contains no decision logic.
3. **Type consistency.** `extra_cost(d) -> (params, macs, scalars)` is defined in task 2 and extended in tasks 3 and 4 with the same return shape; `RouteRule.set(name, scorer, eids, cv, rows)` is used by the tests and by `_refit_router` and `router_variants`; `TransferVerdict.fields()` keys match the event assertions; the growth event keys match the report's reads (`hidden_after`, `increment`, `gap`); `Cell.cid` is the file stem everywhere.
4. **Review focus.** Each of the five items has its pinned test (see the list at the top).
