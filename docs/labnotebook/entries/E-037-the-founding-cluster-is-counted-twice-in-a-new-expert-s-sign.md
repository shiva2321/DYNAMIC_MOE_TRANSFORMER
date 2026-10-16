# E-037 - The founding cluster is counted twice in a new expert's signature: why the infant window exists

- **ID:** E-037
- **Date:** 2026-10-16
- **Track:** G
- **Type:** experiment (a defect found while looking for a fix to the infant window of E-034)
- **Status:** mechanism and unit-level fix completed; the system-level check on the dev streams is RUNNING (cells in `research/growmoe_results/rv2d/cells_s3d_x1/`, 30 cells)
- **Evidence grade:** A for the mechanism (a call trace, a measured table on 3 seeds, unit tests); B for the study of regime lengths (3 seeds); the effect on the dev streams is not yet known
- **Code / commit:** the commit that adds this entry; flag `sig_dedupe` and `VmfSignature(dedupe=...)` on branch `stage4-1-cap-and-telemetry` (OFF by default); scripts in `research/growmoe_results/merge_study/`
- **People:** Claude found, measured and fixed it (behind a flag); not reviewed by anyone else yet

## Question
E-034 left a silent failure: a new expert accepts every input for 5 batches (the x4 infant relaxation), which blinds the pool to a regime change in that window, and without the relaxation the pool founds one expert per batch. The user asked: "if a concrete reliable fix is possible then fix it, otherwise document it". Is there a fix, and why does the pool need the relaxation at all?

## Why we asked it
E-034 table of the infant window (C-74) and the user's decision of 2026-10-16 (E-036). Open question Q-22 (can the relaxation be split into "attract training samples" and "clear novelty"?).

## Prediction (written BEFORE the numbers, if there was one)
None for the mechanism: I looked at the events of the x1 run (relaxation off) first and saw that the batch after a spawn was flagged 100 % novel by the expert that had just been founded from that very region. My guess was a non-exchangeable calibration set.

## Method (how)
1. `infant_diag` (scratch): per-batch events of a pool with the relaxation off, seed 0, regimes of 10 batches.
2. `update_trace.py`: log every `VmfSignature.update` call in the one observe call that founds expert 1 (independent of any distance computation).
3. `twin_signature_diag.py`: for the newest expert, before each later batch of its regime: prototypes, prototypes with an exact twin, share of calibration scores at the maximum, mean p-value of the batch, share at the floor 1/(n+1); with and without the twin fix; seeds 0, 1, 2.
4. `infant_dedupe_study.py`: the pool of `infant_study.py` (stage-2 gate, transfer test, nothing injected, pool seed = stream seed + 7), 12 function regimes of 10 / 5 / 3 / 2 batches, arms: shipped (x4), x1, x4 + twin fix, x1 + twin fix, seeds 0, 1, 2.
5. Unit tests of the fix (tests/growmoe/test_gm_s41_dedupe.py (on the branch), 9 tests, on the branch).
6. Dev check: `research/growmoe_results/rv2d/run_s3_dedupe.py` runs the stage-4 dev cells (streams F-15, OV-0.6, OV-0.85, S1, S2-5; arms S3 and S3+T+G; seeds 100-102) with the twin fix and x1, paired with `cells_s3u/` (same pool seed offset 7).

## Results (what came out)
- **The call trace** (`update_trace_output.txt`): in the single observe call that founds expert 1, its signature is updated twice with the same 64 rows; the second call reports 64 of 64 rows already held. The first update is the spawn (`child_sig.update(x_taken)`), the second is the training loop (`rec.meta["sig"].update(xk)`) after the cluster's samples were assigned to the new expert.
- **What it does to the calibration** (`twin_signature_diag_output.txt`, the same pattern on 3 seeds): right after the founding the signature holds 128 prototypes, all with an exact twin, 100 % of its calibration scores equal the maximum (the twin value, kappa 256 minus log 127), the next batch of the expert's own region has mean p-value 0.008 (the floor 1/129) and 100 % of its samples sit at the floor. The mean p-value of that region rises 0.19 / 0.16 / 0.15 (2 batches later), 0.32 / 0.32 / 0.40 (4), 0.42 / 0.37 / 0.41 (6) and reaches about 0.4-0.5 (the honest value 0.5) only after 7-8 batches; 22-30 prototypes still have a twin at 8 batches. With the twin fix the mean p-value is 0.39-0.58 from the first batch on and no prototype has a twin.
- **The relaxation hides it**: floor 0.0078 x 4 = 0.031 is above q = 0.02, so for the batch right after a founding the infant accepts its own region; without the relaxation (x1) the same pool founds an expert per batch (121 / 121 / 69 experts for 12 regimes of 10 batches, accuracy 50.4 against 67.7).
- **Regime-length study** (`RESULTS_INFANT_DEDUPE.md`; experts at the end for seeds 0, 1, 2 and mean accuracy):

| batches per regime | shipped (x4, twins) | x1, twins | x4 + twin fix | x1 + twin fix |
|---|---|---|---|---|
| 10 | 12, 12, 12 (67.7) | 121, 121, 69 (50.4) | 12, 12, 12 (67.7) | 12, 12, 12 (67.7) |
| 5 | 10, 12, 5 (55.0) | 26, 57, 15 (44.5) | 10, 12, 5 (55.6) | 10, 12, 5 (55.6) |
| 3 | 1, 11, 8 (45.4) | 1, 27, 12 (38.8) | 1, 7, 4 (40.0) | 1, 11, 8 (45.9) |
| 2 | 1, 3, 4 (35.1) | 1, 6, 9 (35.2) | 1, 3, 4 (34.3) | 1, 4, 6 (36.6) |

  With the twin fix the relaxation changes nothing at 10 and 5 batches (identical to the digit), and at 3 and 2 batches x1 is better than x4 (the regime that arrives inside the window is no longer taken by the infant). x1 plus the twin fix finds as many regimes as the shipped pool, or more, in all 12 settings (seed by regime length), and its accuracy differs from the shipped pool's by -1.0 to +4.0 points per setting (mean +0.65; the two negative settings are -0.04 and -1.04, the largest gain is +3.96 at 2 batches).
- **The 5-batch misses are not the infant window.** With the twin fix and x1, seed 2 still founds only 5 experts for 12 regimes of 5 batches; the log shows the transfer test saying "no conflict" because the reference expert has not learned yet (its honest reference loss 1.45 is near chance ln 4 = 1.39, the new regime's zero-shot loss 1.5-1.9, margin 0.35). That is a separate limit of the stage-4b rule (new open question).
- **Unit tests**: the founding cluster enters the signature once with the flag and twice without (64 against 128); with `infant_relax` 1 and no twin fix a regime of six batches founds more than two experts, with the fix exactly two; with the fix and x1 a new region arriving two batches after a spawn is seen as new, with x4 it is not.

## Interpretation (what we understood, and on what basis)
Plain words. When a new expert is born, the pool writes its first samples into the expert's "memory of where my data lives" (the signature), and a moment later writes the same samples again because they are also its first training data. A memory that holds every point twice judges a fresh point of the same region as a stranger, because each stored point has a perfect copy of itself next to it and nothing fresh can look that familiar. The x4 relaxation papers over that for a few batches; the pool is then blind to the next regime during exactly those batches. Measured: the mechanism (trace, table), the cascade without the relaxation, the neutrality of the fix at long regimes, its small gain at short ones. Not measured: the effect on the other dev streams (running), on real text, and on pools whose clusters arrive gradually (the doubling is then partial).

The shipped pool at 3 batches per regime found more experts than the fixed pool with the relaxation kept (11 against 7 in seed 1): the broken calibration also makes experts over-sensitive for several batches, which sometimes helps and sometimes causes a cascade. Part of what the earlier dev results measured is therefore this behaviour, and it is not known which parts (Q-24).

## Caveats and what went wrong
- Three seeds and one stream family for the regime-length study; the transfer rule is on in that pool, and at 5 or fewer batches it limits the result more than the window does.
- The fix in the study is a monkeypatch of `VmfSignature.update`; the product flag does the same (tested), but the dev check runs through the patch.
- The twin count uses a cosine tolerance of 1e-5 (float32 rounding); points that are merely very close are not dropped.
- I first read the 5-batch misses as the blind window; the per-batch log showed the transfer test, not the infant, decided them.

## Consequence
Candidate default: `sig_dedupe=True` and `infant_relax=1.0` together. Adoption waits for (a) the dev check on the guard streams, (b) a decision on whether to fold it into the stage-4 confirmatory run (it changes every arm equally but the dev constants were chosen under the doubled signature) or to run it as its own stage. Q-22 answered by this entry; Q-24 and Q-25 opened; F-31 recorded.

## Evidence
- `research/growmoe_results/merge_study/update_trace.py`, `update_trace_output.txt`
- `research/growmoe_results/merge_study/twin_signature_diag.py`, `twin_signature_diag_output.txt`
- `research/growmoe_results/merge_study/infant_dedupe_study.py`, `infant_dedupe_study.json`, `RESULTS_INFANT_DEDUPE.md`
- `research/growmoe_results/rv2d/run_s3_dedupe.py`, `cells_s3d_x1/` (the dev check)
- tests/growmoe/test_gm_s41_dedupe.py (on the branch) (on branch `stage4-1-cap-and-telemetry`)

## Reproduce
```
cd research/growmoe_results/merge_study
python update_trace.py
python twin_signature_diag.py
python infant_dedupe_study.py
cd ../rv2d && python run_s3_dedupe.py run --variant x1 --workers 2
```
