# Independent review: the RV2-D confirmatory run (routing v2, stage 4), 2026-10-21

Reviewer: a fresh Claude session with no access to the author's conversation. Brief: `docs/superpowers/handoff/REVIEW_BRIEF_RV2D_CONFIRMATORY_2026-10-21.md`, steps 1 to 5 (step 6, adoption, skipped as instructed).

State reviewed: branch `track-g-growmoe`, HEAD `6d61eff`. I changed nothing under `growmoe/`, `research/` or `docs/` except this file, which is the only file I created in the repository. Everything I ran wrote to a scratch folder outside the repository (bytecode writing and the pytest cache were switched off). I did not run `report_rv2d.py`, `jobqueue.py`, `freeze_prereg.py`, or any git command that writes. `OMP_NUM_THREADS=2` for everything, except where section 1.7 says otherwise. Environment: Windows 11, Python 3.12.10, torch 2.6.0+cu124, numpy 2.2.6, scipy 1.17.1.

Note for the author: a second worktree `D:/shm_adopt` (branch `adopt-stage4-defaults`, also at `6d61eff`) was running test suites while I worked. It is outside this review and I did not look at it.

## 0. Result in short

No blocker. The record set is intact, the eight recomputable verdicts reproduce from the raw JSON with code that imports neither verdict module, S4-9 re-runs bit-identical, the freeze (bars hash, code stamp, 15 provenance hashes) is intact, and eight confirmatory cells re-run from scratch reproduce to the last bit (four of them only at the run's own thread configuration), including the one record that was produced by a resumed checkpoint.

Four should-fix items concern what the notebook and the verdict code say, not what the verdicts are: (SF-1) in the "growth" arms the gap rule replaces the shipped widen-and-deepen policy, and the full-stack arm (S3+T+G) had no growth of any kind, so the notebook's reading of the 1.2 to 1.9 point loss is wrong and the default flip of 4c has a consequence the notebook does not state; (SF-2) the expert-count statements are false for OV-0.3-MIXED; (SF-3) the confirmatory streams cannot test the "extend" half of the transfer rule, and the one measurable cost of the rule (F-30) is systematic; (SF-4) S4-5 as coded, and as re-coded by the audit, covers 10 of the 11 units that "each other stream" names because both leave out OV-0.85; including it changes nothing (-0.278, 1.00 x the true count), but the audit is not independent on this point. Amendment A4-1 is justified and honestly disclosed (section 2, SF-3 iv).

## 1. Commands run and what they printed

All paths are relative to `D:/spherical_hyperspace_memory`. Scratch scripts live in `C:/Users/Asta/AppData/Local/Temp/claude/D--spherical-hyperspace-memory/8ff4dd42-bd7d-4b54-ba1a-934ace205802/scratchpad/` (called SCR below); the two that matter are reproduced in the appendices.

### 1.1 Freeze and code integrity (brief step 1)

```
git log --oneline -n 3
  6d61eff docs(labnotebook): manifest after E-041
  d6bde2e results(growmoe): RV2-D confirmatory run (seeds 85-99, 704 cells) - all nine pre-registered statements PASS ...
  1821943 tools(growmoe): independent audit script for the RV2-D verdicts ...
git status --short
  M geomlab/phase3_geocore.py ; ?? E-031 entry, research/geomlab_results/phase3/*      (another track; not part of this work)

python -I -c "import hashlib; print(hashlib.sha256(open('research/growmoe_results/rv2d/bars_rv2d.json','rb').read()).hexdigest())"
  f445f86018dc10349375a1f424b69dd3176d551014b67b09973db676783d3d0d     (= bars_rv2d.sha256)

git diff 44edd74 --stat -- growmoe research/growmoe_results/rv2d/rv2d_lib.py               -> empty
git diff 44edd74 HEAD --stat -- growmoe research/growmoe_results/rv2d/rv2d_lib.py research/growmoe_results/rv2d/rv2d_select.py   -> empty
git diff 44edd74 HEAD --stat (everything except cells/ and docs/labnotebook/): the brief, the plan's progress log, AUDIT_RV2D.md,
  RESULTS_RV2D.md, audit_rv2d.py, jobqueue.py (queue driver, unstamped), verdicts_rv2d.json
git log --format=%h -- <each of research/growmoe_results/rv2d/PREREG.md, bars_rv2d.json, bars_rv2d.sha256, cells_confirm.json>   -> only 44edd74 for each
```

The code stamp is not a trap: I recomputed it from git blobs (`SCR/stamp_at.py`, same algorithm as `rv2d_lib.code_stamp`): `0fd76e3` gives `e468a8911472d079` (the dev stamp), and `ba0e2e4`, `bf4bb77`, `44edd74`, `646bd40`, HEAD all give `ff3438232c5f0dfa`. The 15 provenance hashes of `PREREG.md` all match the current files (LF-normalised sha256, first 12 hex digits; 0 mismatches). The mutation check's stored stamp `539b8e4a8d025519` equals the stamp it computes now, 27 of 27 mutants killed.

The stamp covers `growmoe/**/*.py` and `rv2d_lib.py` only. The runners also import `rv2c/run_rv2c.py`, `arith/arith_lib.py`, `m0/run_m0.py`, `rv2a/run_rv2a.py`; `git diff 44edd74 HEAD --stat` over `research/growmoe_results/{rv2c,rv2a,m2a,m0,arith}` is empty and their last commits (14 to 15 October) predate the dev runs.

### 1.2 Record set (brief steps 1 and 2)

`python -I SCR/my_integrity.py` (reads `cells_confirm.json` and every `cells/*.json`, compares as sets):

```
cells_confirm.json entries: 704, duplicates 0, phases {'confirm': 704}
non-confirm records in cells/: {('dev', seeds>=100, stamp e468a8911472d079): 171}
records: 704  missing records: 0  extra records: 0   file name != cid: 0
stamps: {'ff3438232c5f0dfa': 704}   non-finite accuracies (all five router blocks): 0
seeds per unit: ten stage-3 streams 85-99 x 4 arms; arith-seq 85-90 x 4; banking 85-89 x 4; arith-mixed16 and -mixed128 85-99 x {S3, S3+G}
gap 0.17 on every G arm, transfer margin 0.25 on every T arm, 0.0 elsewhere
regimes recorded: F-15 15, F-30 30, OV-0.0/0.3/0.3-MIXED/0.6/0.85 10, S1 7, S2-5 5, S2-30 30, banking 6, arithmetic none
```

`cells/` also holds 171 dev records (seeds >= 100, other stamp). They are filtered out by phase, seed range and stamp in both the report and the audit; none is mislabelled. Five `.ckpt` files lie beside finished records (one dev, four confirm; see N-7).

Seeds 85-99 appear in no other script under `research/` or `growmoe/` (grep for seed lists), and no confirm log or record is older than the start time D-37 gives (00:44 EDT): first log 10:14:27 IST = 00:44:27 EDT, first record 10:24:52 IST, last 13:48:01 IST, nothing earlier; the frozen files are timestamped 17 October.

### 1.3 The author's audit (brief step 2)

```
python research/growmoe_results/rv2d/audit_rv2d.py          (no --write; exit status 0)
  Records read: 704 of phase confirm with the code stamp ff3438232c5f0dfa; integrity problems: 0.
  S4-1 +12.95 [+10.53, +15.38] n=6 | S4-2 +10.11 [+9.61, +10.60] n=15 | S4-3 worst guard mean +0.00 over 10/10 units
  S4-4 gain +6.14 (n=6), forgetting 7.39, seeds with < 5 experts 0 | S4-5 worst accuracy mean -0.43, largest expert ratio 1.07, 10/10 units
  S4-6 15/15 seeds reach 0.9 (need 13), width 124, extra forward 3.33 % | S4-7 worst mean +0.00, most growth events 0.00, 11/11 units
  S4-8 exact match 46.59 (n=6), forgetting 7.08          every verdict PASS and equal to verdicts_rv2d.json; S4-9 not recomputed
```

### 1.4 My own recomputation (brief step 2 and the task's item b)

`python -I SCR/my_recompute.py` (Appendix A; reads only `cells/*.json`, imports no `growmoe`, no `audit_rv2d`, no `report_rv2d`):

```
S4-1  router gain arith-seq   +12.954 [+10.533, +15.376] n=6    per seed 12.55 13.46 13.60 16.41 9.27 12.43
S4-2  router gain OV-0.85     +10.105 [+9.605, +10.605] n=15     per seed 8.1 to 11.43
S4-3  worst of 10 guard units +0.000 (OV-0.6 +1.717 [+1.399, +2.034], OV-0.3 +0.433, F-15 +0.111, OV-0.0 +0.150, OV-0.3-MIXED +0.148, rest exactly 0)
S4-4  T minus S3 arith-seq    +6.139 [+4.282, +7.997]; forgetting 7.392; experts S3 [5,4,4,3,2,4], S3+T [6,6,6,6,6,6]; per seed +2.96 +6.13 +7.86 +7.02 +5.52 +7.35
S4-5  worst mean -0.431 (F-30 [-0.663, -0.199]); largest ratio against the true count 1.067 (S1); against S3's count: S2-30 1.500, S1 1.556, S2-5 1.562, banking 1.429
S4-6  15/15 seeds >= 0.9 (min ratio 0.918, mean 0.952); mean final width 124.3; extra forward 3.333 %
S4-7  11 units, all +0.000 [+0.000, +0.000], 0 growth events, +0.0 % parameters; S3+G is bit-equal to S3 in every seed
S4-8  S3+T+G with router 46.586 (per seed 46.6 46.6 46.3 47.0 46.0 47.0; t-interval [46.18, 46.99]); forgetting 7.081 ([6.58, 7.58])
```

Every number equals the author's report and audit to the printed precision. Variants: all four router settings recorded in the cells give the same verdicts (`SCR/my_settings.py`: restrict-0.01/0.03 S4-1 +12.76, S4-2 +9.86, free-0.01/0.03 +12.95 and +10.11, worst guard 0.000 in all four), so the pre-registered choice `free-0.01` did not decide anything.

Reading `growmoe/verdicts_rv4.py` against the spec table of section 8 line by line: bars (`BARS_RV2D` equals `bars_rv2d.json` equals the spec table), directions, seeds and the decision rule (`rv2d_decision`) say what the spec says. S4-4 and S4-5 use the router-off accuracy ("shipped gate"), S4-1 to S4-3 and S4-8 the router-on accuracy, as the spec requires. Streams: S4-1, S4-2, S4-4, S4-6, S4-7 (mixed-128 plus all ten stage-3 streams) and S4-8 are as specified; S4-3 uses the ten streams minus OV-0.85 plus banking77 (right: OV-0.85 is S4-2's target); **S4-5 uses the same list (line 96 of `verdicts_rv4.py`, line 94 of `audit_rv2d.py`), which leaves out OV-0.85, although "each other stream" there means every stream other than the arithmetic target of S4-4 (SF-4)**. The other deviation is A4-1 in `transfer_guard`, which is intended.

### 1.5 S4-9, golden (brief step 3)

```
python research/growmoe_results/rv2c/golden.py check          (52 s on this machine, 20 fingerprints, single thread inside)
  bit-identical        exit 0
```

### 1.6 The code that was run (brief step 4)

`git diff 0fd76e3 ba0e2e4 -- growmoe research/growmoe_results/rv2d/rv2d_lib.py research/growmoe_results/rv2d/rv2d_select.py` is 6 files, 180 insertions, 24 deletions; I read all of it. With every flag at its default and fewer than 256 experts:

* `phylo.py`: the three `sig_cls(...)` calls became `_new_sig(seed)` which passes `dedupe` only when `sig_dedupe` is on (same arguments otherwise); the cap checks `len(experts) < max_experts` became `not full or cap_policy == "share"` with `full = len(experts) >= max_experts` (identical when not full; `full` is computed at one point and no expert is added between it and the later check); `_note_cap_reached` adds an event only at the cap, `_count_infant_suppression` only counts; the latter calls `sig.pvalue(x)` on infants, which may run `signature.fit()` a little earlier than before, and `fit()` is deterministic and draws from no generator. `config.py`: two new fields with defaults `"share"` and `False` plus a validation. `signature.py`: `dedupe` default False. `rv2d_lib.py`: `pool_seed`. `verdicts_rv4.py` and `rv2d_select.py`: A4-1 only.
* Tests (`python -m pytest tests/growmoe/test_gm_s41_cap.py tests/growmoe/test_gm_s41_dedupe.py tests/growmoe/test_gm_rv4_verdicts.py tests/growmoe/test_gm_rv4_runner.py tests/growmoe/test_gm_rv4_config.py -q -p no:cacheprovider`): 80 passed in 18 s, including `test_a_pool_that_never_reaches_its_cap_does_not_depend_on_the_policy`. I did not run the whole suite (the other worktree was).
* Direct check on the stage-3-ON path, which the golden does not cover (N-5): `python -I SCR/my_rerun.py cells_s3u <5 cids>` re-ran five dev cells with the frozen code (`s3__S2-5__S3`, `s3__S2-5__S3_T_G`, `s3__S1__S3`, `s3__S1__S3_T`, `s3__OV-0p85__S3`, seed 100, 2 threads like the dev driver) and compared every field with the stored dev records made at `0fd76e3` (stamp `e468a89...`, pool seed + 7 by the patched driver): bit-identical in accuracy, the whole router block (selected rule, cross-validation numbers, rows), expert counts, events, parameters; the only difference is the metadata field `pool_seed_offset`, which the library does not write. This also shows that `rv2d_lib.pool_seed` (A4-2) applies the same offset as the dev driver.
* `cap_policy="share"` acts only at 256 experts; the largest confirmatory pool holds 31.

I found no behaviour change on the default path.

### 1.7 Re-running confirmatory cells (the task's item a/c, and integrity)

`python -I SCR/my_rerun.py cells <cid> ...` runs a stored confirm cell with the current code (it calls `rv2d_lib.run_cell`, writing checkpoints to the scratch folder) and compares all fields except timings.

| cell | thread setting | result |
|---|---|---|
| s3 S2-5 S3 s85, s3 OV-0.85 S3 s87, s3 F-30 S3+T s88 (the run used OMP 1 for these) | OMP 2 | bit-identical |
| banking S3+T s85 (the run used OMP 2 for banking) | OMP 2 | bit-identical (original setting) |
| s3 S1 S3+T s86 | OMP 2 | accuracy 0.7667 vs stored 0.7619 (+0.48 points) |
| s3 F-30 S3 s88 | OMP 2 | 0.42600 vs 0.42583 |
| arith-mixed16 S3+G s85 | 2 torch threads (cap) | 0.3223 vs 0.3195; growth-event gaps differ in the 7th digit |
| arith-mixed128 S3 s90 (the resumed record) | 2 torch threads (cap) | 0.3363 vs 0.3348 |
| the same four, at the run's own setting (OMP 1 for the function-regime cells; 3 torch threads, which the arithmetic cells set themselves, cap lifted for this run only) | as the run | **all four bit-identical, 0 differences** |

So all eight stored records are reproduced exactly (four of them only when the run's own thread setting is used), and `arith-mixed128 S3 s90`, whose first process died on `PermissionError: [WinError 5]` in `os.replace` of its checkpoint and was finished by a second process from the checkpoint, equals an uninterrupted run. Changing the thread count changes the numbers in four of the seven cells run at a different count (N-1).

### 1.8 Run history

`research/growmoe_results/rv2d/logs/queue_scheduler.log` and the 704 `q_*.err` files (script in the transcript, not kept): 704 distinct confirm cells launched, 705 launches (one repeat: `arith-mixed128__-__S3__s90`, 02:21:43 and 02:24:47 EDT); 703 `.err` files hold exactly one start and no traceback; `s90` holds two starts and the `PermissionError` above. No power cut, no other restart. The queue driver was reconfigured in the first quarter hour (E-041: 12 threads instead of 9; the code diff of `jobqueue.py`: per-cell `OMP_NUM_THREADS`, below-normal priority and `OMP_WAIT_POLICY=PASSIVE` above a budget of 9; the log shows 8 launches in one second at 01:02:22 EDT); the four `arith-seq` cells of seed 85 started under the old configuration.

### 1.9 Extra measurements for the findings

* `SCR/my_growth.py`, `my_growth2.py`, `my_extra.py` (growth events, widths, per-seed comparisons), `my_cost.py` (CPU seconds), `my_forget.py` (the stored `forgetting` equals the value recomputed from the stored `snapshots` in all 24 arith-seq records, max difference 0.0).
* `SCR/my_timing.py 6 3` (S3 against S3+G on `arith-mixed128`, first 6 chunks = 900 batches, identical trajectories, alternating order, 3 repetitions, 2 threads, other jobs on the machine): CPU seconds S3 35.8 / 39.7 / 35.9, S3+G 25.3 / 20.0 / 16.6; wall 19.1 / 20.1 / 18.1 against 13.5 / 10.1 / 8.3.
* `SCR/my_probe_count.py`: during the same S3 run `GrowthPolicy.step` (the shipped shadow trial) was called 8 times and took 9.2 s of 20.0 s wall (46 %), committing nothing.

## 2. Findings, ranked

Blockers: none.

### Should fix

**SF-1. The "growth" arms replace the shipped growth policy; in the full-stack arm no growth of any kind happened. E-041, C-86, C-87 and the commit headline misread this, and the default flip of 4c has a consequence they do not state.**

Evidence.
* Code, `growmoe/pools/phylo.py` lines 838 to 846: `if self.cfg.growth_rule == "gap": self._gap_growth_step(...)` and `elif self.cfg.grow and ... GrowthPolicy ...`. With `"gap"` the shadow-trial policy of today, which can widen and deepen, never runs. The spec says "probe = today's policy" (6.2) and leaves depth out of 4c (6.4); neither E-041 nor the claims say that the flip removes deepening.
* Records, operator-by-operator arithmetic, events by arm: S3 has 7 shipped events (5 deepen, 2 widen) in seeds 85, 87, 88, 89; S3+T has 8 (7 deepen, 1 widen) in seeds 87, 88, 89; S3+G has 4 gap-rule widenings 512 to 1024 (gaps 0.1695, 0.1681, 0.1695, 0.1699, just under the 0.17 threshold) in seeds 85, 88, 89, 90; S3+T+G has **0 events** (all six pools at 512 units, 802,632 parameters). `final_hidden` does not show depth, `params` does.
* S3+T against S3+T+G, per seed:

| seed | T off | T+G off | diff | T on | T+G on | diff | shipped events in T |
|---|---|---|---|---|---|---|---|
| 85 | 33.98 | 33.98 | 0.00 | 46.65 | 46.65 | 0.00 | none |
| 86 | 34.47 | 34.47 | 0.00 | 46.62 | 46.62 | 0.00 | none |
| 87 | 34.92 | 34.77 | -0.15 | 46.53 | 46.26 | -0.27 | 2 deepen |
| 88 | 37.46 | 33.97 | -3.49 | 52.38 | 46.99 | -5.39 | 2 deepen |
| 89 | 37.96 | 34.31 | -3.65 | 51.67 | 46.02 | -5.66 | 1 widen, 3 deepen |
| 90 | 35.22 | 35.22 | 0.00 | 46.98 | 46.98 | 0.00 | none |

T+G minus T = -1.21 [-3.13, +0.70] without the router, -1.89 [-4.85, +1.08] with it. All of it is the loss of the shipped deepening in three seeds; the other three seeds agree to the last digit.
* Where it is stated: E-041 L44 ("S3+T+G minus S3+T is -1.2 ... -1.9"), L50 ("it stays silent on every stream except the one built to starve an expert"), L57 ("the one place where an adopted component lowers a measured number"), C-86 ("stays silent elsewhere", "lowers the full stack by 1.2 to 1.9 points"), C-87 ("48.5 % with the router and the transfer rule but without growth"; "the arm without growth"), the commit headline ("growth silent except where an expert is starved"). S3+T is not "without growth" (8 shipped events), the gap rule is not silent on arith-seq (4 firings), and in the full stack the gap rule did nothing.
* The dev review had little to see: in its two seeds the shipped policy fired once in seed 100 of S3 (a widen) and of S3+T (a widen and a deepen), and swapping it for the gap rule was neutral to slightly positive there (S3+G minus S3 +0.33, T+G minus T +0.20 points; `RESULTS_DEV_REVIEW.md` L43 reports S3+G +0.2 [-2.0, +2.3]). The three confirmatory seeds with shipped events (87, 88, 89) are the first data in which the swap shows a loss.

Consequences.
* S4-8's "full stack" (46.6 / 7.1) is transfer rule plus router with growth switched off. It passes by a wide margin (interval [46.18, 46.99] against 44), so the verdict stands, but the sentence "the three repairs together" is not what that arm shows.
* S4-7 cannot see the swap: at 128 units neither policy fires on the ten stage-3 streams, mixed-128 or banking77 (S3+G and S3 are identical in accuracy, router block, expert count and parameters in 170 of 170 pairs), and arith-seq is not in S4-7. Adopting 4c flips `growth_rule` to a policy that cannot deepen; the only data where the shipped policy fires at large width say -2.3 [-4.8, +0.2] (S3), and -1.2 / -1.9 (T), none significant with six seeds.
* In fairness, measured here and absent from the notebook: (i) the gap controller is cheaper than what it replaces. On identical trajectories S3 spent 46 % of its wall time in shadow trials that committed nothing; S3+G needed 0.46 to 0.71 of S3's CPU (section 1.9). The pre-registered "extra forward cost <= 5 %" (3.33 %) is therefore a one-sided number; net, 4c saves compute at 128 units. (ii) At 16 units the swap fixes a failure mode: the shipped policy lands the 16-unit expert at 256 / 64 / 24 / [16, 64] units depending on the seed (6 / 7 / 1 / 1 seeds), S3 reaches 0.9 x the 128-unit pool in 6 of 15 seeds and S3+G in 15 of 15 (9 seeds only S3+G passes, 0 the other way; exact sign test p = 0.004). The mean difference +2.1 [-1.1, +5.3] is carried by one seed (94: S3 stuck at 24 units, +19.4; without it +0.85).
* The declared score of the calibration rule (accuracy gain minus 0.1 x relative parameter increase, spec 6.7) is negative for the chosen gamma (dev: -0.0094 averaged over the three families, -0.027 for mixed-16; confirmatory mixed-16: +2.1 points for +48 % parameters = -0.027): 4c is supported by S4-6's reliability criterion and by the saving in compute, not by that score.

Fix: correct the five places above; in the 4c adoption note say that `growth_rule="gap"` replaces the widen-and-deepen probe policy, give the arith-seq numbers and the two benefits, and let the user decide 4c explicitly (the pre-registered rule says adopt, so this is a disclosure and a decision, not a failed statement). Alternatives for the user: adopt 4a and 4b now and 4c with the note; or keep 4c off until a variant that keeps deepening has its own test.

**SF-2. The expert-count statements contradict the records for OV-0.3-MIXED.**

Records: `regimes` = 10, S3 holds 5.80 experts and S3+T 5.73 (RESULTS_RV2D L51 shows 5.8 -> 5.7). `mix_pairs` (`rv2a/run_rv2a.py` L50) interleaves consecutive segments, so every batch mixes two regimes and the stream has 5 sources; the pool finds about 5.7, which is 1.15 times the sources. E-041 L42 ("On the other streams both give the true count (... OV-* 10.0-10.3 / 10.0-10.1)") and C-88 ("it founds the true number on F-15, F-30 and the OV streams"; "five of the pre-registered streams") are false for this stream (neither arm founds 10; six streams under-count). Side effect: the A4-1 guard reads 0.57 on this unit (denominator 10) and is vacuous there; against 5 sources it is 1.15, still under 1.2. The other numbers in E-041 (every table value, interval and ratio) match `RESULTS_RV2D.md` and my recomputation.

Fix: reword E-041 L42 and C-88; note that OV-0.3-MIXED is a stationary mixture the pool cannot split (spec 2.1, Phase B 4d).

**SF-3. What S4-4 and S4-5 can and cannot say about the transfer rule (scope of 4b), and A4-1.**

i. In all twelve units the first visit of every regime is a new function or a new label block (function regimes, S2 permutations, S1 and banking77 with disjoint labels, one operator per block), so "spawn at every onset" is always the right answer. S3+T spawned 2,000 times and extended 49 times over the 150 ten-stream cells (133.3 spawns per seed against 137 regimes). The amended guard compares with the number of regimes, not of distinct functions, so it can fail only when the rule spawns more than once per regime; and no confirmatory stream has a regime that should be extended (same labelling function, new inputs). That branch rests on unit tests and spike B, where the bare test over-fired on known "add" at 18.5 % for n = 128 (spec 2.3); a wrongly founded expert is permanent (F-28).
ii. The rule has a small, systematic cost where the shipped pool already spawns correctly: F-30 -0.431 [-0.663, -0.199], negative in 13 of 15 seeds (sign test p = 0.007), F-15 -0.339 [-0.740, +0.062]. It passes the bar -0.5 by 0.069 on the mean; the interval reaches -0.66.
iii. S4-4: +6.14 [+4.28, +8.00], bar +5 on the mean (disclosed in E-041 L56).
iv. A4-1. Read literally, S4-5 fails on four of ten units (S2-30 1.50, S1 1.56, S2-5 1.56, banking77 1.43 times S3's count; my numbers equal E-041 L46), so the amendment decided the adoption of 4b. It is justified: S3 itself under-spawns on the five units E-041 names (arith-seq 3.7 of 6, S1 4.8 of 7, S2-30 20.0 of 30, S2-5 3.2 of 5, banking77 4.2 of 6; six with OV-0.3-MIXED, SF-2), the guard exists to stop cascades of wrongly founded experts and the true count is what it stands for, the change was made on dev evidence before the freeze, the user accepted it on 17 October, and it is disclosed in the amendments table, the PREREG row and E-041 L46 with the literal ratios. I find it honest. What I would add is (i) and (ii), and that the commit headline ("all nine pre-registered statements PASS") does not carry the qualifier that S4-5 passes only under A4-1.

Fix: put i and ii into the 4b adoption note and the caveat column of C-85. Optional: a small pre-registered follow-up on a stream whose regimes share a labelling function.

**SF-4. S4-5 as coded covers 10 of the 11 units that "each other stream" names, and the audit inherited the same list.**

The spec row reads "Transfer guard: each other stream and banking77, minus S3" and sits next to S4-4, whose target is arithmetic; "other" therefore means every stream except that target, i.e. all ten stage-3 streams plus banking77. `verdicts_rv4.py` line 96 uses `GUARD_STREAMS` (the ten minus OV-0.85, right for S4-3 because OV-0.85 is S4-2's target) and `audit_rv2d.py` line 94 uses one `units` list for S4-3 and S4-5. So OV-0.85 is missing from S4-5 in the verdict and in the "independent" recomputation, and `verdicts_rv2d.json` has 10 S4-5 units. It matters a little more than a clerical slip because OV-0.85 was the stream that sat exactly on the guard edge in dev (`DEV_SELECTION.md`, margin 0.25: -0.5; per seed -0.65, -0.30, -0.55, mean -0.500 against the bar -0.5). On the confirmatory seeds it passes: T minus S3 = -0.278 [-0.624, +0.067], 10.00 experts against 10 regimes (1.00 x; 0.97 x S3's count), so the verdict does not change.

Fix: state in E-041 that S4-5 as coded has 10 units and give the 11th (above); add OV-0.85 to the audit's S4-5 list (the audit is not stamped, `growmoe/verdicts_rv4.py` is, so do not edit that one). This is where the audit inherits the verdict code's reading instead of the spec's (N-4).

### Notes

**N-1. Reproducibility depends on the thread count; E-041 says it was not tested.** Section 1.7: at the run's own configuration every re-run cell is bit-identical; run at 2 threads instead of 1 (function-regime cells) or 3 (arithmetic cells), four of those seven cells differ, by up to 0.48 points on a single cell (expert counts unchanged). This does not threaten a verdict: the arms of a unit ran under one configuration (the 170 identical S3/S3+G pairs show it), and the tightest margin (S4-5, F-30, 0.069 on a mean of 15 cells) is not thread-sensitive in the two F-30 cells I tried (shifts of 0.002 and 0). But any later comparison against these records must pin threads, and the records do not store them. Suggest recording `torch.get_num_threads()` and the OMP variables per record and replacing "I did not test whether any single cell differs" (E-041 L58) with this result.

**N-2. The PREREG arm description does not match the mixed-arithmetic cells.** PREREG says S3 is the stage-3 pool "(batch-level gate on, ...)". `run_arith_mixed` builds `L.Pool(5, False, ...)`, i.e. `set_gate=False`; the mixed records have no `set_*` events (arith-seq, built with `True`, has them). No verdict depends on it (S4-6 and S4-7 compare arms of the same configuration), but the PREREG should say so.

**N-3. S4-8 combines a router-on accuracy with a router-off forgetting.** `forgetting` comes from snapshots taken during training, router off (recomputed from `snapshots`, exact). The sentence "46.6 % exact match and 7.1 points of forgetting" (C-87) mixes the two.

**N-4. Independence of the audit.** No shared arithmetic bug found (my recomputation agrees with both to the printed precision); one shared reading of "which streams" (SF-4). The audit and the report share these inputs: the same bars file, the t-interval, the `regimes` field, and the S4-6 cost formula `post_rows / (train_rows x inner_steps x 6)`, which the audit copied (the 6 is fwd+bwd x (new + replay rows, `replay_frac` 0.5) per `train_expert`; I checked that it follows from the code, and SF-1 gives the measured comparison). The audit skips records of another phase or seed silently and does not compare the record set with `cells_confirm.json`; I did (section 1.2). A count check against 704 and the cell list would close that.

**N-5. S4-9 covers the stage-1/2 pool only.** `golden.py` BASE pins `set_gate` and `group_vote` off, so the S3 arm (stage 3 on) has no fingerprint. My five dev re-runs (section 1.6) close it for those cells; a stage-3-on fingerprint in the golden would close it permanently before the defaults flip.

**N-6. Process wording.** (a) "Most likely to fail, in my order: S4-5, S4-8, S4-6, S4-7" (E-041, "written BEFORE the numbers") first appears in the results commit (I searched E-039, E-040, DECISIONS, the plan, the handoff folder, and `git log -S` over the entries); it may well have been thought before the numbers, but no document shows it. (b) D-37 reads the user's "continue then" as the explicit go for the single-use seeds; the pre-registration says the run starts only on an explicit go. It is disclosed in D-37, and it is irreversible. (c) The "go" is recorded only in the author's words.

**N-7. Orphan checkpoints and a Windows file error.** Five `.ckpt` files remain beside finished records (dev `arith-mixed128 S3_T s100`; confirm `arith-mixed128 S3 s85, s87`, `arith-mixed16 S3 s86, s87`). Their `.err` logs show one start each, so the records are from uninterrupted runs. The one `PermissionError` (WinError 5, `os.replace` of a checkpoint) points to scanner or indexer interference on the 11 MB pickles. Harmless here; delete the orphans.

**N-8. S4-7 is vacuous at 128 units.** Neither policy fires on any of the 11 units, so the zero difference is by construction; it shows that gamma = 0.17 lies below the 128-unit gap (0.227 in the spikes), not that growth is harmless. The author says as much.

**N-9. CPU-seconds table.** Four `arith-seq` cells (S3 s85, S3 s86, S3+T s85, S3+T+G s85) took 5,201 to 7,087 CPU-s against 970 to 2,485 for the other twenty, apparently from the oversubscribed phase and the old scheduler configuration; compare medians, not means (`my_cost.py`). The S3 versus S3+G direction is real (SF-1), the magnitudes in `RESULTS_RV2D.md` are noisy.

**N-10. Small things.** The confirmatory records do not store the pool seed (the dev records of the s3u family carry `pool_seed_offset`); PREREG's "at least 0.87 of the seeds" is 13/15 = 0.867; D-37 says "--detach (9 threads)" while E-041 says 12 threads.

**N-11. Heads-up for the adoption step (step 6, not reviewed).** `PhyloCfg.__post_init__` raises when `router=True` and `route_rule == "default"` (`growmoe/config.py` L176-177), and `route_rule` itself defaults to `"default"`. Every confirmatory cell set `route_rule="accept_then_logf"` explicitly (the stage-3 arm configuration, the banking configuration, the arithmetic `Pool`). Flipping `router=True` as the default therefore either makes every bare `PhyloCfg()` raise or needs a change of the `route_rule` default, which is a behaviour change outside both what the run tested and S4-9's flags-off claim (the golden configuration `default_rule` uses the shipped `route_rule`). E-041's adoption list (`router=True`, `router_mode="free"`, `spawn_rule="transfer"`, `growth_rule="gap"`, `grow_gap=0.17`) does not mention it.

**N-12. Operating point.** All four arms of a unit use the inner-step budget tuned for the S3 arm (`inner_steps` is identical across arms and seeds in every unit: 5, 20 for banking77/S2, 200 for S1). The constants of the new components were tuned on dev seeds by declared rules, but the training budget was not re-tuned per arm; the user's standing rule is to tune every arm and report the operating point. Since the comparison is controlled and the new arms are, if anything, at a disadvantage, this does not weaken a PASS, but "+6.1 points" and the like are statements at S3's operating point.

**N-13. Claims that are fine.** C-84 (router): every number matches; "recovers the routing loss" is relative to the oracle gap of E-030, no oracle was run on these pools (OV-0.85: +10.1 here against oracle gains of 11 to 14 points in the spikes, spec 2.3). C-85 apart from SF-2/SF-3. C-86 apart from SF-1. Evidence grades and the "provisional until the reviewer" caveat are appropriate.

## 3. What I checked and found sound

* Record set: 704 of 704, exact set equality with `cells_confirm.json`, one stamp, no duplicates, no extra, all accuracies finite, filenames equal cids.
* Freeze: bars hash, code stamp, provenance hashes, git history of the frozen files, no later code change (`git diff 44edd74` empty for the stamped files); the helper directories outside the stamp unchanged since 14 to 15 October.
* Verdicts: S4-1 to S4-8 recomputed independently, equal to the report; S4-9 re-run bit-identical; the audit and the report agree.
* The verdict rules equal the spec table (bars, direction, streams, seeds), apart from A4-1 (intended) and the stream list of S4-5 (SF-4).
* Stage 4.1 and A4-2: no behaviour change on the default path (reading, 80 tests, golden, five dev cells re-run bit-identical).
* Reproduction: eight confirmatory cells reproduce bit-for-bit (four of them only at the run's own thread configuration), including the resumed record.
* The router verdicts do not depend on the router setting; the S3+G cells with no growth are identical to S3 in 170 of 170 pairs.
* No earlier use of seeds 85-99; no record or log older than the stated start of the run.

## 4. Not reviewed

The whole test suite (the other worktree ran it); the stage-3 statements S3-1 to S3-9 with the adopted components; the adoption branch; the dev review E-039 beyond the parts cited; the claim that `golden_flags_off.json` was recorded before any stage-3 edit (git shows only its first commit, 14 October, which contains stage 3); the mutation script's mutants beyond their stored result (they cover S4-5's verdict rule only, not S4-1 to S4-4 and S4-6 to S4-8, which the independent recomputation covers).

## Appendix A. `my_recompute.py` (the independent recomputation; reads only the records)

```python
import json, glob, math, collections
import numpy as np
from scipy import stats
R = collections.defaultdict(dict)
for f in glob.glob(r"D:\spherical_hyperspace_memory\research\growmoe_results\rv2d\cells\*.json"):
    r = json.load(open(f, encoding="utf-8")); c = r["cell"]
    if c["phase"] == "confirm": R[(c["family"], c["stream"], c["arm"])][c["seed"]] = r
SET = "free-0.01"
on  = lambda r: 100*r["router"][SET]["acc"]; off = lambda r: 100*r["router"]["off"]["acc"]; acc = lambda r: 100*r["acc"]
def ci(d):
    d = np.asarray(d, float); m = d.mean(); k = len(d)
    h = stats.t.ppf(0.975, k-1)*d.std(ddof=1)/math.sqrt(k); return m, m-h, m+h
def pair(a, b, fa, fb):
    s = sorted(set(a) & set(b)); return ci([fa(a[x]) - fb(b[x]) for x in s])
ALL = ["F-30","S2-30","F-15","OV-0.85","OV-0.6","OV-0.3","OV-0.3-MIXED","OV-0.0","S1","S2-5"]; GUARD = [s for s in ALL if s != "OV-0.85"]
S3 = R[("arith-seq","-","S3")]
print("S4-1", pair(S3, S3, on, off), "bar lower >= 5")
S3o = R[("s3","OV-0.85","S3")]
print("S4-2", pair(S3o, S3o, on, off), "bar lower >= 4")
units = [("s3", s) for s in GUARD] + [("banking", "-")]
print("S4-3 worst", min(pair(R[(f,s,"S3")], R[(f,s,"S3")], on, off)[0] for f, s in units), "bar >= -0.5")
T = R[("arith-seq","-","S3+T")]
print("S4-4", pair(T, S3, acc, acc), "forgetting", 100*np.mean([T[s]["forgetting"] for s in T]), "seeds<5 experts", sum(T[s]["n_experts"] < 5 for s in T))
rows = []
for f, s in units:                                  # A4-1: experts against the recorded number of regimes
    a, b = R[(f,s,"S3+T")], R[(f,s,"S3")]
    rows.append((pair(a, b, acc, acc)[0], np.mean([a[x]["n_experts"] for x in a]) / np.mean([a[x]["regimes"] for x in a])))
print("S4-5 worst mean", min(r[0] for r in rows), "max ratio", max(r[1] for r in rows), "bars -0.5 / 1.2")
sm, bg = R[("arith-mixed16","-","S3+G")], R[("arith-mixed128","-","S3")]
sd = sorted(set(sm) & set(bg))
print("S4-6", sum(sm[x]["acc"] >= 0.9*bg[x]["acc"] for x in sd), "of", len(sd), "width", np.mean([np.mean(sm[x]["final_hidden"]) for x in sd]),
      "cost", np.mean([sm[x]["grow_stats"]["post_rows"]/(sm[x]["grow_stats"]["train_rows"]*sm[x]["inner_steps"]*6) for x in sd]))
for f, s in [("arith-mixed128","-")] + [("s3", x) for x in ALL]:
    a, b = R[(f,s,"S3+G")], R[(f,s,"S3")]
    print("S4-7", f, s, pair(a, b, acc, acc)[0], np.mean([len(a[x]["grow_events"]) for x in a]), np.mean([a[x]["params"]/b[x]["params"]-1 for x in a]))
F = R[("arith-seq","-","S3+T+G")]
print("S4-8", np.mean([on(F[s]) for s in F]), 100*np.mean([F[s]["forgetting"] for s in F]), "bars 44 / 8")
```

## Appendix B. `my_rerun.py` (core; re-runs stored cells with the current code and diffs every field except timings)

```python
import sys, json, os
from pathlib import Path
RV2D = Path(r"D:\spherical_hyperspace_memory\research\growmoe_results\rv2d")
sys.dont_write_bytecode = True; sys.path.insert(0, str(RV2D))
import torch, rv2d_lib as R
R.CELLS = Path(r"<scratch folder>")                       # checkpoints of the arithmetic cells go here, not into cells/
orig = torch.set_num_threads
torch.set_num_threads = lambda n: orig(min(n, int(os.environ.get("REVIEW_MAX_THREADS", "2"))))
def cmp(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) if k not in ("seconds", "cpu_seconds", "code_stamp")
                for d in (cmp(a[k], b[k], f"{path}/{k}") if k in a and k in b else [f"{path}/{k}: missing"])]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in cmp(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: new {a!r} vs old {b!r}"]
for cid in sys.argv[2:]:                                   # argv[1] = "cells" (confirm) or "cells_s3u" (dev, no head start)
    old = json.loads((RV2D / sys.argv[1] / f"{cid}.json").read_text(encoding="utf-8"))
    new = json.loads(json.dumps(R.run_cell(R.Cell(**old["cell"]))))
    print(cid, "differences:", cmp(new, old))
```

DO NOT ADOPT UNTIL: the notebook is corrected where SF-1 to SF-4 show it wrong or incomplete (growth_rule=gap replaces the shipped widen-and-deepen policy and the S3+T+G arm had no growth of any kind; OV-0.3-MIXED holds 5.7 experts, not 10; S4-5 as coded has 10 units, the 11th passes; the confirmatory streams cannot test the extend branch of the transfer rule) and the user has decided 4c with its measured cost of -1.2 to -2.3 points on operator-by-operator arithmetic (against about half the growth compute) in view; no verdict changes, and 4a and 4b can be adopted as soon as the text is fixed.
