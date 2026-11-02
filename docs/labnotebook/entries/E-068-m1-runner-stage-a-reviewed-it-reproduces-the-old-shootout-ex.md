# E-068 - M1 runner stage A reviewed: it reproduces the old shootout exactly at 3 threads, the replay arm depends on the thread count, and the shipped pool fragments at q 0.05

- **ID:** E-068
- **Date:** 2026-11-02
- **Track:** W
- **Type:** audit
- **Status:** completed
- **Evidence grade:** B for the reproduction and for the fragmentation (script-generated, five development seeds, deterministic; not pre-registered); C for the thread-count finding (one seed); the mechanism of the fragmentation is PARTIAL
- **Code / commit:** `m1/` (arms, beds, cell, queue, run, harness_check, diag_switches), `tests/m1/test_m1.py`, `results/m1/`; the code is the commit that adds `m1/` to the repository (the records carry the commit that was HEAD when they were made, e8e76d3, because the runner was not yet committed)
- **People:** Gemini built stage A (second job, D-78); Claude reviewed, repaired, reran and diagnosed it; nobody else has checked the repairs

## Question
Is the stage-A runner of the M1 efficiency scoreboard a faithful instrument, that is, does it reproduce the old unaliased shootout, and what does Baseline v1 (arm A1) actually do on bed B1? A faithful instrument is the precondition for every later M1 number; a surprising A1 would change what M1 has to measure first.

## Why we asked it
The runner was delivered by Gemini under the advance go D-78 with a report that listed no deviations, as the first Gemini job did (D-80). The spec (M1_RUNNER_A_SPEC.md) required a harness check against the old shootout, and arm A1 must be Baseline v1 (BASELINE_V1.md, tag `baseline-v1`). Follows E-067 (road plan) and D-77 (M1 preregistration approved).

## Prediction (written BEFORE the numbers, if there was one)
Harness check, from the spec: A1 and A3 within 2 points of the old means and memory bytes equal. Nothing was predicted for Baseline v1 on B1; the first cell was a surprise, so the diagnosis below is exploratory.

## Method (how)
- Read the delivered files (kept in `D:\shm_m1runner_a` until the user deletes that folder) against the spec and BASELINE_V1.md; listed the defects; repaired them in the main folder (D-82, no clones); wrote 11 more tests (18 in all). Details: `docs/superpowers/handoff/M1_RUNNER_A_RESULT.md` section 7.
- Harness check at the old shootout's setting (3 threads on logical processors 0, 2, 4): `python -m m1.harness_check --seeds 100 101 102 103 104 --thread-probe`; and at the M1 pinning (2 threads on 0, 2): `python -m m1.harness_check --seeds 100 --threads 2 --cpus 0,2`.
- The four seed-800 development cells: `python -m m1.queue --split dev --beds B1 --ns 10 --arms A1,A3,A7,A8 --seeds 800 --out results/m1/dev --cpus 0,2 --threads 2`.
- Diagnosis of the first A1 cell (67 experts for 10 regimes): `python -m m1.diag_switches --seeds 800 801 802 803 804 --extras-seed 800` (development seeds only), then a read of `research/growmoe_results/rv2d/rv2d_lib.py` for the configuration of the stage-4 confirmatory pools.
- Seeds: harness 100-104 (old development seeds, allowed for the harness only); cells and diagnosis 800-804 (M1 development seeds). No report seed was touched.

## Results (what came out)
**Harness check at 3 threads** (`results/m1/harness/harness_check_N10_seeds100_101_102_103_104.json`): the legacy pool (A1L) and the replay network (A3) reproduce the old shootout in accuracy, bytes and buffer on all five seeds.

<!-- generated from results/m1/harness/harness_check_N10_seeds100_101_102_103_104.json ; pinning {'threads': 3, 'cpus': [0, 2, 4], 'blas_limited': True} -->

| Seed | Old pool acc (%) | New pool acc (%) | Diff (points) | Old bytes | New bytes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 100 | 43.325 | 43.325 | +0.000 | 4,004,160 | 4,004,160 |
| 101 | 42.600 | 42.600 | +0.000 | 4,004,160 | 4,004,160 |
| 102 | 41.450 | 41.450 | +0.000 | 4,371,808 | 4,371,808 |
| 103 | 44.375 | 44.375 | +0.000 | 4,004,160 | 4,004,160 |
| 104 | 43.225 | 43.225 | +0.000 | 4,004,160 | 4,004,160 |
| **Mean** | **42.995** | **42.995** | **+0.000** | | |

| Seed | Old replay acc (%) | New replay acc (%) | Diff (points) | Old bytes | New bytes | Old buffer | New buffer | Budget (old pool bytes) | Unaliased pool bytes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 100 | 36.550 | 36.550 | +0.000 | 4,003,872 | 4,003,872 | 3,695 | 3,695 | 4,004,160 | 4,004,160 |
| 101 | 36.600 | 36.600 | +0.000 | 4,003,872 | 4,003,872 | 3,695 | 3,695 | 4,004,160 | 4,004,160 |
| 102 | 37.425 | 37.425 | +0.000 | 4,003,872 | 4,003,872 | 3,695 | 3,695 | 4,004,160 | 4,371,808 |
| 103 | 36.650 | 36.650 | +0.000 | 4,003,872 | 4,003,872 | 3,695 | 3,695 | 4,004,160 | 4,004,160 |
| 104 | 36.775 | 36.775 | +0.000 | 4,003,872 | 4,003,872 | 3,695 | 3,695 | 4,004,160 | 4,004,160 |
| **Mean** | **36.800** | **36.800** | **+0.000** | | | | | | |

| Threads | Logical processors | BLAS limited | Replay acc, seed 100 (%) | Old (%) | Same as old |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 3 | [0, 2, 4] | True | 36.550 | 36.550 | True |
| 2 | [0, 2] | True | 37.225 | 36.550 | False |
| 2 | [0, 2] | True | 37.225 | 36.550 | False |
| 1 | [0] | True | 36.175 | 36.550 | False |

all_ok (accuracy within 2 points and bytes equal, both arms, all seeds): True

**Seed-800 development cells** (`results/m1/dev/B1_n10_A*__seed800.json`, 2 threads on 0 and 2; development hints, not M1 results):

<!-- generated from results/m1/dev/B1_n10_A*__seed800.json -->

| Arm | Accuracy (%) | Forgetting (points) | Memory (bytes) | Update wall-clock (s) | Update MACs | Answer b=1: median ms / MACs | Answer b=32: median ms / MACs | Schema | Commit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A1 | 30.68 | -0.10 | 18,779,104 | 12.93 | 61,699,858,432 | 1.392 / 91,520 | 12.269 / 4,313,088 | 1.1.0 | e8e76d3b0f |
| A3 | 37.72 | 4.48 | 7,027,232 | 10.14 | 603,142,225,920 | 0.137 / 525,312 | 0.323 / 16,809,984 | 1.1.0 | e8e76d3b0f |
| A7 | 22.07 | -0.45 | 4,915,200 | 0.06 | 0 | 0.197 / 1,228,800 | 0.850 / 39,321,600 | 1.1.0 | e8e76d3b0f |
| A8 | 36.27 | 2.53 | 9,132,576 | 12.95 | 628,752,384,000 | 0.168 / 1,049,600 | 0.514 / 33,587,200 | 1.1.0 | e8e76d3b0f |

A1 experts at the end: 67

all four records share the budget: True

**Diagnosis** (`results/m1/dev/diag_switches_B1_n10.json`):

<!-- generated from results/m1/dev/diag_switches_B1_n10.json ; pinning {'threads': 2, 'cpus': [0, 2], 'blas_limited': True} -->

Core configurations, every development seed:

| Configuration | Seeds | Accuracy per seed (%) | Mean accuracy (%) | Experts per seed | Mean memory (MB) | Spawn / extend events per seed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A1L pre-stage-4 pool (probe spawn, q 0.02, router off) | 800, 801, 802, 803, 804 | 43.75, 43.88, 40.83, 43.17, 41.47 | 42.62 | 10, 10, 11, 10, 10 | 4.08 | 10/0, 10/0, 11/0, 10/0, 10/0 |
| A1 Baseline v1 (transfer spawn, q 0.05, router on), 5 steps | 800, 801, 802, 803, 804 | 30.68, 31.15, 30.42, 31.67, 31.07 | 31.00 | 67, 67, 61, 67, 67 | 18.49 | 67/0, 67/0, 61/1, 67/0, 67/0 |
| A1 with q 0.02 | 800, 801, 802, 803, 804 | 44.00, 44.42, 42.68, 42.97, 41.10 | 43.03 | 10, 10, 10, 10, 10 | 4.02 | 10/0, 10/0, 10/1, 10/0, 10/0 |
| A1 with the probe spawn rule | 800, 801, 802, 803, 804 | 42.42, 40.25, 38.05, 40.35, 41.58 | 40.53 | 15, 29, 25, 23, 26 | 8.01 | 15/19, 29/19, 25/18, 23/22, 26/21 |

One switch at a time, seed 800 only:

| Configuration | Accuracy (%) | Experts | Memory (MB) | Events |
| :--- | :--- | :--- | :--- | :--- |
| A1L with the router | 43.75 | 10 | 4.00 | {'spawn': 10, 'router_refit': 1} |
| A1L with the transfer spawn rule | 44.00 | 10 | 4.02 | {'spawn': 10} |
| A1L with q 0.05 | 42.42 | 15 | 5.48 | {'spawn': 15, 'extend': 19} |
| A1L with the shortlist from 64 experts | 43.75 | 10 | 4.00 | {'spawn': 10} |
| A1 with 1 step per batch (the class default) | 18.32 | 7 | 2.02 | {'spawn': 7, 'extend': 12, 'router_refit': 1} |
| A1 with the router off | 30.68 | 67 | 18.78 | {'spawn': 67, 'alarm': 1} |

**What the stage-4 confirmatory pools used.** `research/growmoe_results/rv2d/rv2d_lib.py` line 172 builds them with `inner_steps=20, route_rule="accept_then_logf", q=0.02, alpha_spawn=0.01, m_min=32, set_gate=True`, with the spawn rule, router and growth rule switched per arm. The class defaults, which the `shipped` system of Baseline v1 uses, are q 0.05 and the set gate off.

## Interpretation (what we understood, and on what basis)
Plain words. (1) The instrument is faithful: set to the old thread count it gives the old numbers to the last digit, for both arms and all five seeds, so the stream walk, the seed offset, the memory accounting and the adapters are right. (2) The replay network is a little different every time the number of threads changes (by about one point on one seed, 36.175 % to 37.225 %) because large matrix products add floating-point numbers in a different order and training amplifies the difference; at a fixed thread count it repeats exactly. The small-expert pool did not move. So every arm has to be run, and compared, at the same pinning. (3) Baseline v1 as the code builds it does something the shootout pool never did: on ten regimes it makes 61 to 67 experts and loses about 12 points (31.0 % against 42.6 % on average over the five development seeds). The cause is the pair "q 0.05 and the transfer spawn rule": with q 0.02 the same pool makes 10 experts and scores like the old one; with q 0.05 and the older probe rule it makes 15 to 29. (4) The router and the transfer rule were adopted as defaults on the strength of a confirmatory run that used q 0.02 and the set gate; the default pool differs from that run in exactly those two respects.

Technical statement. Measured: C-193 to C-195. Read from code: C-196. Inferred: the thread-count mechanism, and the idea that the transfer verdict seldom runs for young experts (it needs `transfer_min_ref` = 64 reference losses per expert) so that most flagged clusters become experts; the event counts show spawns with almost no extend events, which fits, but the verdict was not instrumented.

## Caveats and what went wrong
- One bed (function regimes), N = 10, 5 steps per batch, five development seeds. N = 30 and 100 are not run; the fragmentation could be worse or better there.
- The cells are at the as-built operating point, not tuned. M1 requires every arm to be tuned on development seeds (preregistration section 3); this is stage B.
- The mechanism is not probed; an instrumented run of the transfer verdict would settle it.
- The thread-count finding is one seed and one replay width.
- The staleness guard compares the arm's configuration, not the stream generator or the pool code; a change to `growmoe/` needs a deliberate rerun.
- Gemini's report said "none" and was wrong in nine places (F-108); BASELINE_V1.md did not state the per-bed overrides of the pools it describes (F-109, my own gap).
- The first version of my harness check used the old replay's own bytes as the budget and 2 threads; it was replaced by the version here.

## Consequence
- D-83 (stage A accepted after repairs; A1's operating point is tuned in stage B), D-84 (thread counts for reproduction and for M1).
- New: C-193 to C-196, F-108, F-109, Q-80.
- Stage B spec (next, needs the user's go for the Gemini job): a tuning driver for A1 (q and steps per bed) and A3 (width and steps), the A2 candidates including the stage-4 configuration, beds B2 and B3, arms A4 to A6, the report and verdict scripts, the router-refit time added to the pool's update cost.
- For BP1: the question of Q-80, whether the shipped defaults should be the configuration in which the router and transfer rule were confirmed.

## Evidence
- `results/m1/harness/harness_check_N10_seeds100_101_102_103_104.json` - harness check at 3 threads with the thread probe
- `results/m1/harness/harness_check_N10_seeds100.json` - the same for seed 100 at the M1 pinning (2 threads)
- `results/m1/dev/B1_n10_A1__seed800.json`, `_A3_`, `_A7_`, `_A8_` - the four development cells
- `results/m1/dev/diag_switches_B1_n10.json` - the one-switch diagnosis
- `docs/superpowers/handoff/M1_RUNNER_A_RESULT.md` - Gemini's report as delivered, with the review in section 7
- `research/growmoe_results/rv2d/rv2d_lib.py` - the configuration of the stage-4 confirmatory pools (line 172)

## Reproduce
```
python -m pytest tests/m1 -q
python -m m1.harness_check --seeds 100 101 102 103 104 --thread-probe
python -m m1.queue --split dev --beds B1 --ns 10 --arms A1,A3,A7,A8 --seeds 800 --out results/m1/dev --cpus 0,2 --threads 2
python -m m1.diag_switches --seeds 800 801 802 803 804 --extras-seed 800
```
