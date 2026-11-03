# V0 Fix Round (Job V-2): Implementation & Verification Report

**Date:** 2026-11-04  
**Author:** Agent G (Implementer, Gemini)  
**Reviewer:** Claude (Design & Verification)  
**Task Spec:** `docs/superpowers/handoff/V0_FIX_SPEC.md`  
**Execution Context:** Pinned to logical processors 8 and 10 with 2 threads (`costlab.env.fixed_threads(n=2, cpus=[8, 10])`), concurrently running with scored benchmark on cores 0 and 2.

---

## 1. Summary of Changes

In response to the E-075 review findings, the following files were updated:
1. `v0/system.py`:
   - Updated `build_v0`: strictly requires `A1q` in tuning map; raises `ValueError` with clear message if missing, refusing silent fallback to default operating point (5 inner steps, q=0.05).
2. `v0/demo.py`:
   - Expanded comparator set to include: `A3`, `A7`, `A5a`, `A5b`, `A6`, and `A7all`.
   - Added `--with-reference` CLI flag to optionally evaluate `A8` (Offline Joint Network), explicitly labeled `"OFFLINE REFERENCE, NOT A METHOD"`.
   - Updated memory tracking for `A3` and `A7`: recorded `memory_budget_bytes` (V0 final memory), `final_memory_bytes`, and `capped_by_stream_size` (`final_memory_bytes < 0.95 * memory_budget_bytes`).
   - Updated summary generation:
     - Reports forgetting, update seconds, and answer latency (batch 1 and 32) for every arm.
     - Compares V0 final memory with raw training stream size (`bed.total_samples * bed.d * 4` bytes).
     - Prints capped disclosure: `"{arm_id} could not use the budget: it already holds the whole training stream, {mem_mb:.2f} MB"`.
     - Dynamically names every comparator ahead of V0 with its accuracy difference AND its memory next to V0's memory; or states `"no comparator printed here is ahead on this stream"`.
     - Removed the phrase `"all three honest comparators"` everywhere.
     - Adds silent flag disclosure when false-new rate == 0 and out-of-domain rate == 0: `"The unknown flag never fired at its configured level (on known or on out-of-domain examples); the AUROC measures the novelty score, not the flag."`
   - Updated `figure.png` and `table.md` to plot and tabulate all comparators using colors from `m1.report.ARM_COLORS`.
3. `v0/report.py`:
   - Updated merged report generator to format and plot all comparators (`A3`, `A7`, `A5a`, `A5b`, `A6`, `A7all`, and `A8` if present) using `m1.report.ARM_COLORS`.
4. `tests/v0/test_v0.py`:
   - Monkeypatched `m1.cell.count_macs` in test (c) `test_assembly_v0_equals_h_ulda` using a stub returning `MacCount(macs=0, flops=0, by_op={})`, preventing the second multiply-adds profiling pass (`_mac_pass`).
   - Kept exact equality assertions on accuracy and per-regime accuracy between V0 and `execute_cell(..., profile="det")`.
   - Added assertion verifying V0 accuracy against stored M1 dev cell `results/m1/dev_b/B3seq_n22_H_ULDA__seed800.json`.
   - Updated test (d) `test_comparators_batch_order_and_replay_memory` to verify batch hashes for all comparators and check `memory_budget_bytes` and `capped_by_stream_size`.
   - Added `test_build_v0_missing_a1q_raises_error`.
   - Added `test_summary_ahead_comparators_and_no_all_three`.
   - Added `test_summary_capped_arms_and_raw_stream_comparison`.
   - Added `test_summary_silent_flag_disclosure`.

---

## 2. Test Execution & Output

All 11 tests in `tests/v0` were run under pinned environment (`cpus=[8, 10]`, 2 threads). The complete output was redirected to `docs/superpowers/handoff/V0_FIX_TEST_OUTPUT.txt`:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\Asta\AppData\Local\Programs\Python\Python312\python.exe
cachedir: .pytest_cache
benchmark: 5.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: D:\spherical_hyperspace_memory
configfile: pytest.ini
plugins: anyio-4.13.0, libtmux-0.62.0, benchmark-5.2.3
collecting ... collected 11 items

tests/v0/test_v0.py::test_build_v0_config_and_bed_rejection PASSED       [  9%]
tests/v0/test_v0.py::test_build_v0_missing_a1q_raises_error PASSED       [ 18%]
tests/v0/test_v0.py::test_demo_short_run_and_files PASSED                [ 27%]
tests/v0/test_v0.py::test_assembly_v0_equals_h_ulda PASSED               [ 36%]
tests/v0/test_v0.py::test_comparators_batch_order_and_replay_memory PASSED [ 45%]
tests/v0/test_v0.py::test_auroc_handmade_scores PASSED                   [ 54%]
tests/v0/test_v0.py::test_report_on_synthetic_fixtures PASSED            [ 63%]
tests/v0/test_v0.py::test_rerun_finished_demo_skips PASSED               [ 72%]
tests/v0/test_v0.py::test_summary_ahead_comparators_and_no_all_three PASSED [ 81%]
tests/v0/test_v0.py::test_summary_capped_arms_and_raw_stream_comparison PASSED [ 90%]
tests/v0/test_v0.py::test_summary_silent_flag_disclosure PASSED          [100%]

============================== warnings summary ===============================
C:\Users\Asta\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\cuda\__init__.py:61
  C:\Users\Asta\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\cuda\__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
    import pynvml  # type: ignore[import]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================== 11 passed, 1 warning in 301.24s (0:05:01) ==================
```

**Measured test suite duration:** 301.24 seconds (5 minutes 1 second).  
*Runtime breakdown:* `test_assembly_v0_equals_h_ulda` took 223.00 seconds (`execute_cell` took 103 seconds, `run_demo` took 120 seconds). See Section 5 for deviation disclosure.

---

## 3. Measured Smoke Runs (Dev Seed 800)

The three schedules on dev seed 800 were run into `results/v0/smoke2_seq_800`, `results/v0/smoke2_mix_800`, and `results/v0/smoke2_drift_800`, followed by `v0.report` into `results/v0/report2`:

### 3.1 Measured Runtimes
- `smoke2_seq_800`: 274 seconds (4m 34s)
- `smoke2_mix_800`: 75 seconds (1m 15s)
- `smoke2_drift_800`: 98 seconds (1m 38s)
- `report2`: 4 seconds

### 3.2 Summary Table (`results/v0/report2/table.md`)
```markdown
# DEV (development seeds only) V0 Walking Skeleton Merged Report

| Schedule | Seed | V0 Acc | A3 Acc | A7 Acc | A5a Acc | A5b Acc | A6 Acc | A7all Acc | V0 Experts | V0 Mem (MB) | Mean AUROC | V0 Latency b1 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seq | 800 | 65.02% | 60.07% | 64.54% | 61.34% | 63.01% | 70.14% | 64.54% | 20 | 14.01 | 0.6803 | 4.04 |
| mix | 800 | 65.21% | 58.79% | 53.92% | 61.50% | 64.08% | 70.02% | 64.40% | 1 | 1.06 | 0.8502 | 0.53 |
| drift | 800 | 63.19% | 57.21% | 54.74% | 61.26% | 66.12% | 69.55% | 64.67% | 1 | 1.18 | 0.8755 | 0.52 |
```

### 3.3 Detailed Summaries

#### Smoke Run 1: Sequential (`results/v0/smoke2_seq_800/summary.txt`)
```text
V0 WALKING SKELETON DEMONSTRATION (Schedule: seq, Seed: 800)
LABEL: DEMONSTRATION, NOT A RESULT
--------------------------------------------------------------------------------
1. BEHAVIOUR AND ACCURACY:
V0 final mean accuracy across all seen regimes: 65.02%.
V0 forgetting up to stream end: 5.08%.
Comparator A3 final accuracy: 60.07%, forgetting: 10.41%.
Comparator A7 final accuracy: 64.54%, forgetting: 2.77%.
Comparator A5a final accuracy: 61.34%, forgetting: 6.64%.
Comparator A5b final accuracy: 63.01%, forgetting: 3.55%.
Comparator A6 final accuracy: 70.14%, forgetting: 3.55%.
Comparator A7all final accuracy: 64.54%, forgetting: 2.77%.
On this stream, comparator(s) ahead of V0 on final mean accuracy: A6 (diff +5.12%, memory 259.91 MB vs V0 14.01 MB).

2. COST NEXT TO BEHAVIOUR:
V0 spawned 20 experts with a final memory footprint of 14,687,420 bytes (14.01 MB).
V0 cumulative update time: 40.38 seconds.
V0 answer latency: 4.04 ms (batch 1), 5.58 ms (batch 32).
Raw training stream size: 6,504,448 bytes (6.20 MB); V0 final memory is 14.01 MB (2.26x raw stream size).
A3 could not use the budget: it already holds the whole training stream, 6.77 MB
A7 could not use the budget: it already holds the whole training stream, 6.20 MB
A3 memory: 7,097,844 bytes (6.77 MB), update: 9.27s, latency: 0.14 ms (b1), 0.18 ms (b32).
A7 memory: 6,504,448 bytes (6.20 MB), update: 0.10s, latency: 0.34 ms (b1), 1.61 ms (b32).
A5a memory: 129,536 bytes (0.12 MB), update: 0.70s, latency: 0.06 ms (b1), 0.12 ms (b32).
A5b memory: 8,256,000 bytes (7.87 MB), update: 0.34s, latency: 4.59 ms (b1), 6.41 ms (b32).
A6 memory: 272,531,456 bytes (259.91 MB), update: 38.91s, latency: 0.27 ms (b1), 1.76 ms (b32).
A7all memory: 6,504,448 bytes (6.20 MB), update: 0.01s, latency: 0.32 ms (b1), 2.10 ms (b32).

3. UNKNOWN FLAG & NOVELTY DETECTION:
V0 false new rate on known test examples: 0.00%.
V0 out-of-domain detection rate: 0.00%.
V0 rank AUROC vs out-of-domain sets (1 - p_nov): 0.6803.
  - vs stories: 0.6336
  - vs wiki: 0.6855
  - vs science_questions: 0.7218
The unknown flag never fired at its configured level (on known or on out-of-domain examples); the AUROC measures the novelty score, not the flag.
A novelty AUROC value near 0.5 means the unknown flag cannot distinguish out-of-domain text from known topics and does not work.
```

#### Smoke Run 2: Interleaved Mix (`results/v0/smoke2_mix_800/summary.txt`)
```text
V0 WALKING SKELETON DEMONSTRATION (Schedule: mix, Seed: 800)
LABEL: DEMONSTRATION, NOT A RESULT
--------------------------------------------------------------------------------
1. BEHAVIOUR AND ACCURACY:
V0 final mean accuracy across all seen regimes: 65.21%.
V0 forgetting up to stream end: 0.34%.
Comparator A3 final accuracy: 58.79%, forgetting: 1.88%.
Comparator A7 final accuracy: 53.92%, forgetting: 1.33%.
Comparator A5a final accuracy: 61.50%, forgetting: 0.04%.
Comparator A5b final accuracy: 64.08%, forgetting: 0.79%.
Comparator A6 final accuracy: 70.02%, forgetting: -0.77%.
Comparator A7all final accuracy: 64.40%, forgetting: -0.93%.
On this stream, comparator(s) ahead of V0 on final mean accuracy: A6 (diff +4.81%, memory 259.91 MB vs V0 1.06 MB).

2. COST NEXT TO BEHAVIOUR:
V0 spawned 1 experts with a final memory footprint of 1,109,484 bytes (1.06 MB).
V0 cumulative update time: 13.57 seconds.
V0 answer latency: 0.53 ms (batch 1), 0.75 ms (batch 32).
Raw training stream size: 5,224,448 bytes (4.98 MB); V0 final memory is 1.06 MB (0.21x raw stream size).
A3 memory: 1,108,980 bytes (1.06 MB), update: 5.53s, latency: 0.11 ms (b1), 0.18 ms (b32).
A7 memory: 1,108,992 bytes (1.06 MB), update: 0.06s, latency: 0.15 ms (b1), 0.33 ms (b32).
A5a memory: 129,536 bytes (0.12 MB), update: 0.55s, latency: 0.06 ms (b1), 0.12 ms (b32).
A5b memory: 8,256,000 bytes (7.87 MB), update: 0.28s, latency: 4.25 ms (b1), 6.95 ms (b32).
A6 memory: 272,531,456 bytes (259.91 MB), update: 28.27s, latency: 0.36 ms (b1), 2.13 ms (b32).
A7all memory: 5,224,448 bytes (4.98 MB), update: 0.01s, latency: 0.23 ms (b1), 1.30 ms (b32).

3. UNKNOWN FLAG & NOVELTY DETECTION:
V0 false new rate on known test examples: 0.00%.
V0 out-of-domain detection rate: 0.00%.
V0 rank AUROC vs out-of-domain sets (1 - p_nov): 0.8502.
  - vs stories: 0.7796
  - vs wiki: 0.8784
  - vs science_questions: 0.8926
The unknown flag never fired at its configured level (on known or on out-of-domain examples); the AUROC measures the novelty score, not the flag.
A novelty AUROC value near 0.5 means the unknown flag cannot distinguish out-of-domain text from known topics and does not work.
```

#### Smoke Run 3: Drift (`results/v0/smoke2_drift_800/summary.txt`)
```text
V0 WALKING SKELETON DEMONSTRATION (Schedule: drift, Seed: 800)
LABEL: DEMONSTRATION, NOT A RESULT
--------------------------------------------------------------------------------
1. BEHAVIOUR AND ACCURACY:
V0 final mean accuracy across all seen regimes: 63.19%.
V0 forgetting up to stream end: 6.78%.
Comparator A3 final accuracy: 57.21%, forgetting: 8.85%.
Comparator A7 final accuracy: 54.74%, forgetting: 4.70%.
Comparator A5a final accuracy: 61.26%, forgetting: 3.90%.
Comparator A5b final accuracy: 66.12%, forgetting: 1.67%.
Comparator A6 final accuracy: 69.55%, forgetting: 1.71%.
Comparator A7all final accuracy: 64.67%, forgetting: 1.00%.
On this stream, comparator(s) ahead of V0 on final mean accuracy: A5b (diff +2.94%, memory 7.87 MB vs V0 1.18 MB); A6 (diff +6.36%, memory 259.91 MB vs V0 1.18 MB); A7all (diff +1.48%, memory 4.98 MB vs V0 1.18 MB).

2. COST NEXT TO BEHAVIOUR:
V0 spawned 1 experts with a final memory footprint of 1,240,044 bytes (1.18 MB).
V0 cumulative update time: 13.41 seconds.
V0 answer latency: 0.52 ms (batch 1), 0.73 ms (batch 32).
Raw training stream size: 5,222,400 bytes (4.98 MB); V0 final memory is 1.18 MB (0.24x raw stream size).
A3 memory: 1,239,540 bytes (1.18 MB), update: 5.49s, latency: 0.11 ms (b1), 0.18 ms (b32).
A7 memory: 1,239,552 bytes (1.18 MB), update: 0.06s, latency: 0.11 ms (b1), 0.31 ms (b32).
A5a memory: 129,536 bytes (0.12 MB), update: 0.50s, latency: 0.06 ms (b1), 0.09 ms (b32).
A5b memory: 8,256,000 bytes (7.87 MB), update: 0.29s, latency: 4.66 ms (b1), 6.57 ms (b32).
A6 memory: 272,531,456 bytes (259.91 MB), update: 29.87s, latency: 0.45 ms (b1), 1.86 ms (b32).
A7all memory: 5,222,400 bytes (4.98 MB), update: 0.01s, latency: 0.30 ms (b1), 1.68 ms (b32).

3. UNKNOWN FLAG & NOVELTY DETECTION:
V0 false new rate on known test examples: 0.00%.
V0 out-of-domain detection rate: 0.00%.
V0 rank AUROC vs out-of-domain sets (1 - p_nov): 0.8755.
  - vs stories: 0.8358
  - vs wiki: 0.8875
  - vs science_questions: 0.9033
The unknown flag never fired at its configured level (on known or on out-of-domain examples); the AUROC measures the novelty score, not the flag.
A novelty AUROC value near 0.5 means the unknown flag cannot distinguish out-of-domain text from known topics and does not work.
```

---

## 4. Disclosures & Truthgate Integrity Items

1. **Job V-1 Commands Log Disclosure:**
   In Job V-1, `docs/superpowers/handoff/V0_BUILD_COMMANDS.txt` recorded only 5 commands and was incomplete/hand-edited. In this fix round (Job V-2), every command executed was directly appended via shell invocation before execution to `docs/superpowers/handoff/V0_FIX_COMMANDS.txt` without manual editing.
2. **What Was Stubbed / Mocked:**
   In test `tests/v0/test_v0.py::test_assembly_v0_equals_h_ulda`, `m1.cell.count_macs` was monkeypatched with `lambda fn, **kw: MacCount(macs=0, flops=0, by_op={})` as specified in Item 1.4, strictly to bypass the heavy `_mac_pass` FLOP counting loop in `execute_cell`. No accuracy metrics or model behaviors were mocked or approximated.
3. **What Was NOT Run:**
   - Frozen report seeds 810–829 were not run (strictly forbidden during development).
   - Arm `A8` was not run in the smoke runs (it requires `--with-reference` and costs ~2 additional minutes).
   - Test suites outside `tests/v0` (`tests/m1`, etc.) were not run, respecting machine pinning rules while the scored benchmark was active.

---

## 5. Compliance Matrix

| # | Spec Item (`V0_FIX_SPEC.md`) | Status | Notes / Deviations |
|---|---|---|---|
| 1.1 | **Comparators Set**: Default set: A3, A7, A5a, A5b, A6, A7all built with `m1.arms.build_arm` and tuned entry from `B3seq.json`. Flag `--with-reference` adds A8 labeled "OFFLINE REFERENCE, NOT A METHOD". Reports final mean accuracy, forgetting, memory bytes, update seconds, answer latency b1/b32. Summary sentence names all comparators ahead with diff and memory next to V0 memory; "no comparator printed here is ahead on this stream" if none. Phrase "all three honest comparators" removed. Figures and report script use `ARM_COLORS`. | **COMPLIANT** | Fully implemented and verified. All 6 default comparators evaluated in smoke runs. Figure and report scripts updated with color-blind safe `ARM_COLORS`. All occurrences of "all three" removed. |
| 1.2 | **Memory Honesty**: For A3 and A7, record `memory_budget_bytes`, `final_memory_bytes`, `capped_by_stream_size` (true if final memory < 0.95 * budget). Summary prints sentence per capped arm and compares V0 memory with raw training stream size (`bed.total_samples * d * 4` bytes). Test (d) checks final memory and capped flag. | **COMPLIANT** | On `seq`, A3 (6.77 MB) and A7 (6.20 MB) capped by stream size (budget 14.01 MB). Generated sentences: `"A3 could not use the budget: it already holds the whole training stream, 6.77 MB"`, `"A7 could not use the budget: it already holds the whole training stream, 6.20 MB"`, and raw stream size comparison printed. Verified by tests. |
| 1.3 | **Silent Flag Disclosure**: When false-new rate == 0 and out-of-domain rate == 0 at stream end, summary adds: "The unknown flag never fired at its configured level (on known or on out-of-domain examples); the AUROC measures the novelty score, not the flag." | **COMPLIANT** | Fired in all three smoke runs (rates were 0.00% across all schedules). Disclosure appended verbatim. Verified by unit test that it does not appear when rates > 0. |
| 1.4 | **Test Speed**: In test (c), monkeypatch `m1.cell.count_macs` by a stub returning `MacCount(0, 0, {})` so `execute_cell(..., profile="det")` does not run its second pass. Exact equality assertion on accuracy and per-regime accuracy preserved. Added assertion against stored dev cell `results/m1/dev_b/B3seq_n22_H_ULDA__seed800.json`. Whole `tests/v0` suite target under 2 minutes. | **DEVIATION DISCLOSED** | Code changes fully implemented and exact equalities pass. However, measured runtime of `tests/v0` was 301.24s (5m01s), exceeding the 2-minute goal. On 2 pinned CPU threads (logical processors 8, 10), `execute_cell` took 103s (evaluating 22 regimes at each of 22 boundaries), and `run_demo` took 120s (evaluating 28 phase boundaries, AUROC, latency). Two full 28-phase stream evaluations on CPU with 2 threads physically require ~223s. |
| 1.5 | **`build_v0` Tuning Fallback**: Raises `ValueError` with a clear message if tuning map has no `A1q` entry for the bed (no default fallback). Add test. | **COMPLIANT** | Implemented in `v0/system.py`. Verified by `test_build_v0_missing_a1q_raises_error`. |
| 1.6 | **Commands Log**: Create `docs/superpowers/handoff/V0_FIX_COMMANDS.txt`. Append every command line before execution via shell invocation. Do not edit by hand. Disclose that V-1 commands file was incomplete. | **COMPLIANT** | Log created and updated strictly via same-command shell invocation. Disclosed in Section 4. |
| 1.7 | **New Tests**: Tests for items 1-3 on small fixtures (names every comparator ahead and its memory; capped flag and sentence; silent flag sentence appears only when both rates are 0; no sentence contains "all three"). | **COMPLIANT** | Implemented `test_summary_ahead_comparators_and_no_all_three`, `test_summary_capped_arms_and_raw_stream_comparison`, `test_summary_silent_flag_disclosure`. All pass. |
| 1.8 | **Smoke Runs**: Development seed 800 pinned to logical processors 8 and 10 with 2 threads into `results/v0/smoke2_seq_800`, `smoke2_mix_800`, `smoke2_drift_800`, then `v0.report` into `results/v0/report2`. Pytest output redirected to `docs/superpowers/handoff/V0_FIX_TEST_OUTPUT.txt`. | **COMPLIANT** | All three smoke runs and merged report executed and verified. Output written to `V0_FIX_TEST_OUTPUT.txt`. |
