# M2-A pre-registration: repair the newborn false alarm, then test on fresh seeds

Date 2026-10-09. Written by Claude before any dev tuning of the repaired pool and before any seed 10-19 run. Read-only once a confirmatory result exists; changes go into `AMENDMENT.md`. Rules: `docs/superpowers/handoff/SCOPE_CHARTER.md`. Background: `research/ROUTING_RESEARCH_2026-10.md` (sections 4-6) and the M0 record `research/growmoe_results/m0/` (verdict B1 FAIL for the frozen pool; that verdict stands).

## 0. Provenance, stated honestly

The repair was found AFTER the M0 verdict, by exploratory work on dev seeds 100-102 (and two fresh diagnostic seeds 200-201 that feed no statement): a routing/assignment audit (D1, D3a), an oracle-gating comparison (D3b), five mechanism ablations (D2, D4, D5, D6 and the guard sensitivity check), and a probe of the monitor's baseline. Roughly twenty configurations were tried on dev seeds before this one was chosen. That is a garden of forking paths, so the dev result (F-15: 39.2 % to 46.8 % at 5 steps) is a hypothesis. This file fixes the ONE repair and the ONE test of it before any new seed is touched. Seeds 10-19 have never been used for anything.

## 1. The question

Does arming a newborn expert's competence monitor only after infancy (`PhyloCfg.arm_monitor_after_infancy`), with the step budget re-tuned on dev seeds, make the pool beat the best single network by the B1 margin on fresh seeds, without hurting streams that need the alarm?

## 2. What changes (and what does not)

* Code: one config field, default **False** (so every M0 result stays reproducible), and a few lines in `growmoe/pools/phylo.py`: while an expert is an infant its monitor is not fed or tested, and its monitor history is cleared when infancy ends. Seven tests in `tests/growmoe/test_gm_m2a_arming.py` (two controls that pin the fault, five behaviour tests; breaking the fix fails exactly the two behaviour tests). Full GrowMoE suite: 419 passed.
* Nothing else in the pool changes: `q = 0.02, alpha_spawn = 0.01, m_min = 32, route_rule = accept_then_logf`, `infant_calls = 5`, replay, probation, extend probe, growth, merge are as in Phase 1 / M0.
* Only the training budget `inner_steps` is re-tuned (section 4).

## 3. Arms and streams (fresh seeds 10-19, ten seeds)

| Arm | Definition |
|---|---|
| `A_old` | The M0 pool exactly: flag off, `inner_steps` per family from `research/growmoe_results/m0/tuning.json` (perm 20, types 5, func 50) |
| `A_new` | Flag on, `inner_steps` per family from the dev tuning in section 4, frozen by hash before the first seed 10 run |
| `B2star` | M0's best single network, chosen on M0 dev seeds, NOT re-tuned: width 4096, buffer 8192, 20 steps (`m0/tuning.json`) |
| `B4`, `kNN_all` | Context rows (oracle-routed experts at M0's 5 steps; lookup storing everything), no statement reads them |

| Stream | Configuration | Role |
|---|---|---|
| F-15, F-30 | `function_regimes`, 800 train and 400 test per regime, two revisits (as M0) | primary (P1, P2) |
| S1 | `s1_types(n_train=400, n_test=150)`, hashed features (as M0) | guard G1 |
| S2-5 | 5 permuted-digit regimes, schedule `[0,1,2,3,4,1,3]`, 400/100 (as M0) | guard G2 |
| S2-30 | 30 permuted-digit regimes, spec schedule, 400/100 (as M0) | guard G3 |
| S4 | `s4_outliers(n_regimes=5, n_train=400, n_test=100, outlier_rate=0.01, label_noise=0.1)` | guard G4 (outliers and noisy labels provoke alarms) |
| S7 | `s7_mapping(with_ctx=False)`: 4 regimes, same inputs, label permutations | guard G5 (the alarm must still catch a remap) |

S3 (gradual drift) is NOT used: on dev it was identical with alarms on, off or delayed, so it cannot detect a regression. `A_old`/`A_new` on S2-30, S4, S7 use the `perm` family budget; F-30 uses `func`.

## 4. Tuning protocol (dev seeds 100-102 only)

`A_new` only, flag on, one family at a time on its dev stream: `func` on F-15, `perm` on S2-5, `types` on S1. `inner_steps` in {1, 5, 20, 50}; if the best value is the largest, extend by x2.5 up to 200; ties go to the cheaper value; objective = mean final accuracy over the three dev seeds (same procedure as M0, `growmoe.fairfight.tune_steps`). The result is written to `tuning_m2a.json` and its sha256 to `tuning_m2a.sha256`; the run script refuses to start if the file changes afterwards.

## 5. Statements and decision rule

| ID | Statement | Rule | Expected |
|---|---|---|---|
| P1 | The repaired pool beats the best single network | On F-15 **and** F-30: mean paired `A_new - B2star` >= +5.0 points and the 95 % t interval excludes 0. PASS both, PARTIAL one, FAIL neither | PASS (dev margin about +10) |
| P2 | The repair itself moves the pool | On F-15 and F-30: mean paired `A_new - A_old` >= +3.0 points | PASS (dev +7) |
| G1 | S1 not hurt | `A_new - A_old` >= -3.0 points and the 95 % interval is not entirely below 0 | PASS, with risk: dev S1 74.1 vs 76.0 and noisy |
| G2 | S2-5 not hurt | same rule | PASS (dev 91.7 vs 91.9) |
| G3 | S2-30 not hurt | same rule | PASS |
| G4 | S4 not hurt | same rule | PASS (dev +1.9, very noisy) |
| G5 | The alarm still catches label remaps | S7: mean alarms per run of `A_new` >= 3.0 | PASS (dev 3.7 of 4 regimes) |

Statistics: mean and 95 % t interval over the ten seeds, paired differences by `growmoe.metrics.paired_diff` (as M0). Verdicts are computed by `growmoe.fairfight.m2a_verdicts` from the per-seed JSON (11 tests), never typed. A guard that fails by less than one point is still a guard FAIL; the report states the size.

**Decision (`growmoe.fairfight.m2a_decision`).** P1 PASS and all guards PASS: propose the flag as the PhyloMoE default (a separate, reviewed change). P1 PASS but a guard did not pass: keep it as an option and fold the cause into the routing-v2 design. P1 PARTIAL: the track continues only to resolve it (more regimes, a memory-matched network). **P1 FAIL: second kill. The MoE track stops claiming an accuracy advantage and re-scopes to a compute saver and abstention layer over an episodic memory.** The M0 verdict (B1 FAIL for the frozen pool) is not revised by any outcome here; this is reported as a separate follow-up.

## 6. Integrity and resources

* `A_old` and `B2star` are re-run on seeds 10-19 with the M0 configurations; nothing from M0's test seeds 0-9 is reused or re-read.
* Per-seed records and `RESULTS_M2A.md` are written by `run_m2a.py`; no hand edits; deviations go to `AMENDMENT.md`.
* Two workers x three threads, peak memory about 1.6 GB per worker (the M0 run measured 1.6 GB); other jobs on this PC share CPU and RAM.
* Expected cost: dev tuning about 15 minutes, confirmatory run about 40 minutes of wall-clock.

## 7. Out of scope

Routing v2 (ambiguity deferral, two-signal arbitration, soft top-2, cheap gate), the extend-vs-spawn probe on held-out replay, any other change to the pool, any further seed. Those get their own design and pre-registration.
