# E-035 - Seed aliasing: the function-regime streams give the pool a head start

- **ID:** E-035
- **Date:** 2026-10-16
- **Track:** G
- **Type:** audit
- **Status:** completed (exploratory; the size on earlier claims is NOT yet measured)
- **Evidence grade:** B for the mechanism and for the size on the dev runner (2 stream seeds per cell, 5 streams); C for the reach on earlier entries (read from code and entry lists, not re-run)
- **Code / commit:** the commit that adds this entry; `research/growmoe_results/merge_study/alias_check.py`, `alias_check_s3.py`, `idshift_check.py`
- **People:** Claude found it while building the merge study (E-034); not yet reviewed by anyone else

## Question
Why did a clean pool in my healing harness score 73 % while the same pool with only its expert ids shifted scored 68 %? Is the cause in the harness or in the project's streams, and how large is it on the project's own runs?

## Why we asked it
The end-to-end study of E-034 needed a control for the expert ids; the control (the ids advance as if a spawn had happened, nothing is founded) lost 5-6 points against the clean pool, which no false spawn could explain.

## Prediction (written BEFORE the numbers, if there was one)
None. The first guess was an effect of the injection call; the id-shift control without any injection refuted it.

## Method (how)
- `alias_check.py`: builds the teacher of regime r with `_balanced_teacher(seed=s)` and an expert with `MLPExpert(seed=s)` and compares the first-layer weights (`torch.equal`); runs a clean pool with pool seed = stream seed and = stream seed + 7 on 3 stream seeds at two regime lengths.
- `idshift_check.py`: arms clean, id_shift_only, inject_then_remove, inject_keep at pool seed offset 0 and 7 (3 seeds, regimes of 20 batches).
- `alias_check_s3.py`: the dev runner's own configuration (`rv2d_lib.run_s3`: stage-3 pool, arms S3 and S3+T) on the streams F-15, OV-0.3, OV-0.6, S1, S2-5, stream seeds 100 and 101, with `cfg.seed` = stream seed (as the dev runs do) and = stream seed + 7.
- Reach: `grep PhyloCfg` over `research/growmoe_results/*` for the seed used by each runner.

## Results (what came out)
- The first 8 hidden units of an expert with seed s are bit-identical to the hidden units of the teacher with seed s (True for s = 0, 1, 5; False for another seed).
- Harness (hidden 32): clean pool 74.5 / 73.5 / 72.2 % with the head start against 68.2 / 69.5 / 65.4 % without (regimes of 10 batches); 76.8 / 76.5 / 75.5 against 71.4 / 72.3 / 68.9 (20 batches). With the head start the id-shift control loses about 5.4 points against the clean pool; without it the two are equal (see `idshift_check_offset0.txt` and `idshift_check_offset7.txt`).
- Dev runner, S3 arm, head start in points: F-15 +8.2 and +7.7, OV-0.3 +5.3 and +4.2, OV-0.6 +0.5 and +1.5, S1 +0.9 and +0.9, S2-5 +2.2 and +5.8. Transfer rule minus S3: F-15 +1.1 / +1.7 with the head start, +0.3 / -0.1 without; OV-0.3 +1.2 / +1.5 and -0.4 / -0.2; S1 +1.4 / +1.4 and +2.2 / -0.3; S2-5 -0.8 / +4.2 and -0.6 / +8.6 (tables in the note, section 6).
- Reach: the runners of M0, M2A, RV2-A, RV2-B, RV2-C, M5 and, through `run_rv2c`, the stage-4 dev runner set the pool seed to the stream seed.

## Interpretation (what we understood, and on what basis)
Plain words. The labels of a function-regime stream come from a small random network (the teacher). The pool builds a new expert from a random network seeded with a number that, for expert k, equals the seed of the teacher of regime k, and both come from the same random generator. So the new expert starts with some of the teacher's own hidden units. That helps only pools that found exactly one expert per regime in order; a pool with a false spawn or a missed spawn loses the alignment. The size depends on the stream and on the width of the experts: large on the plain function regimes (F-15 about 8 points), small on the strongly overlapping ones. Technical statement: seed aliasing between `MLPExpert(seed = s*1000 + r)` teachers and `MLPExpert(seed = cfg.seed*1000 + id)` experts when cfg.seed equals the stream seed; measured on the S3 and S3+T arms only.

## Caveats and what went wrong
- Two stream seeds per cell; only the S3 and S3+T arms; F-30 and S2-30 were not run; arithmetic and banking streams have no teacher and are unaffected.
- The earlier entries that mention function regimes (E-013, E-014, E-015, E-017, E-019, E-021, E-023, E-025, E-032) were not re-run; whether and how much each claim moves is open (Q-23).
- My own mistake that made this visible: `alias_check.py` kept its meaning only until I added the pool seed offset to the shared factory; the script now sets the offset itself.

## Consequence
F-30 and Q-23 recorded; D-33 (dev2 not started). Options for the user are in the note (section 6, decision 5): re-run the function-regime dev cells with the offset, keep dev and run a control of the arm differences, or fix the generator and re-check the earlier headline numbers. No earlier claim was changed in the register; the user decides.

## Evidence
- `research/growmoe_results/merge_study/alias_check_output.txt`, `idshift_check_offset0.txt`, `idshift_check_offset7.txt`, `alias_check_s3_*.json`
- `research/GROWMOE_MERGE_RESEARCH.md` section 6 (tables)

## Reproduce
```
cd research/growmoe_results/merge_study
python alias_check.py
python idshift_check.py 1280 12 0,1,2 0
python alias_check_s3.py F-15 --seeds 100 101 --arms S3,S3+T
```
