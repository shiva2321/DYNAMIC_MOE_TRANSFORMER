# growmoe flag: skip training on batches an expert already knows (E-063)

Date: 2026-11-01. Asked for by the user ("we can make this a flag which can be turned on/off", 2026-11-01), after the answer to "do we have a skip?" was no: the real pool trains every batch it routes to a known expert.

## The rule

`PhyloCfg.skip_known` (default `None` = off). A batch routed to an expert is **skipped** when all of these hold:

1. the expert is `active` (not an infant) and has trained on at least `skip_min_assigned` (256) samples;
2. the competence alarm did not fire on this batch (the alarm path is untouched);
3. the mean loss the expert had on the batch **before any step** (the prequential loss the pool already computes for the competence monitor and the transfer reference) is at most `skip_known` x ln(classes the expert has seen), i.e. a fraction of the chance loss.

A skipped batch takes **no optimiser steps and no growth decision**, but still enters the expert's reservoir, its `n_assigned` / `last_active` counters, its signature (updated earlier in `observe`) and, in a `UniPool`, its closed-form LDA head (one cheap pass, never forgets). `NbPool` notebooks never skip (their training is the closed-form absorb itself). Counters: `pool.skip_stats` = batches and rows trained / skipped, also in `summary()["skip"]`.

Why the prequential loss: it is measured on the batch before the expert has seen it, so it is the expert's honest "do I already know this?" and not a memorised fit. A batch the expert has just trained on would always look known.

## What it can and cannot save (measured on dev, seed 700, machine shared with a test run, so timings are rough)

Wall time inside `train_expert` as a share of wall time inside `observe`: function regimes (f10) 1.0 s of 16.6 s (6 %), text teaching bed 1.2 of 10.6 s (11 %), banking77 1.3 of 6.5 s (20 %). The gate and routing take the rest (the earlier measurement of 2026-10-09 said about 97 % of compute; the share here is 80-94 % of wall time). So even a rule that skipped every batch could save at most 6-20 % of observe time on these beds. My earlier statement that the saving "would be large" was too strong.

## Dev findings (seed 700; thresholds chosen here, nothing tuned on the report seeds)

| bed | arm | accuracy | rows skipped | train s of observe s |
|---|---|---|---|---|
| f10 (function regimes, 128 units) | off / 0.1 / 0.25 / 0.5 | 0.4940 (all) | 0 % (all) | 1.0 of 16.6 |
| t (text teaching bed) | off | 0.9316 | 0 % | 1.2 of 10.6 |
| | 0.1 | 0.9311 | 55.7 % | 0.7 of 8.1 |
| | 0.25 | 0.9284 | 80.0 % | 0.3 of 7.9 |
| | 0.5 | 0.9242 | 88.2 % | 0.2 of 8.2 |
| bank (banking77, 128 units) | off | 0.8522 | 0 % | 1.3 of 6.5 |
| | 0.1 | 0.8533 | 2.8 % | 1.0 of 4.5 |
| | 0.25 | 0.8589 | 17.1 % | 0.8 of 4.3 |
| | 0.5 | 0.8578 | 30.6 % | 0.7 of 4.3 |

UniMoE real suite, dev seeds 520 and 521 (cells recomputed after a first attempt overlapped a mutation check and a timing bug; every number here is from the clean rerun):

| schedule | arm | accuracy (520 / 521) | rows skipped (520 / 521) | experts (520 / 521) | train s of observe s (520 / 521) |
|---|---|---|---|---|---|
| mix | off / 0.1 | 0.6632 / 0.6594 (both) | 0 % | 1 / 1 | 0.5 of 4.5 / 0.5 of 1.7 |
| mix | 0.25 | 0.6655 / 0.6603 | 5.6 % / 5.1 % | 1 / 1 | |
| drift | off / 0.1 | 0.6610 / 0.6621 (both) | 0 % | 1 / 1 | 0.4 of 4.1 / 0.5 of 1.9 |
| drift | 0.25 | 0.6617 / 0.6627 | 1.4 % / 5.9 % | 1 / 1 | |
| seq | off | 0.6443 / 0.6255 | 0 % | 12 / 14 | 6.5 of 27.1 / 5.2 of 17.1 |
| seq | 0.1 | 0.6432 / 0.6264 | 9.9 % / 9.0 % | 12 / 14 | 5.6 of 23.0 / 4.5 of 16.2 |
| seq | 0.25 | 0.6425 / 0.6221 | 26.6 % / 18.6 % | 12 / 16 | 4.7 of 22.0 / 3.9 of 15.0 |

Reading: on hard streams (function regimes, where accuracy is 0.49) the expert is never under half of chance loss, so the rule never fires and nothing changes. On the easy text bed it fires on more than half of the rows at 0.1 for a loss of 0.0005; the cost grows with the threshold (0.0032 at 0.25, 0.0074 at 0.5). On banking77 it fires on 3 % to 31 % of rows with no accuracy cost. Two thresholds go forward: **0.1** (the conservative one) and **0.25**; 0.5 is dropped.

## Beds and arms for the checks (report seeds; one process at a time on a quiet machine)

- UniMoE real suite (seven text domains, schedules seq / mix / drift), ULDA pool, seeds 500-504 (used by E-054 and E-062, so a paired comparison only); `fsmoe/skip_check.py --bed useq|umix|udrift`.
- Function regimes f10 and f30, shipped pool with 128 units, seeds 710-714.
- Text teaching bed `t`, seeds 710-714.
- banking77 class-incremental, default pool, seeds 710-712.
- Arms: `off`, `k0.1`, `k0.25`. Dev seeds for the UniMoE suite: 520, 521.

## Predictions (written BEFORE the report-seed runs; S5 after the UniMoE dev cells)

- **S1 equality.** The full suite passes with the flag off (1,101 tests: 1,090 earlier + 11 new, 2 expected failures as before; observed before this section was written) and the `off` UniMoE cells equal the stored flag-off cells of E-062 (`results/unimoe/p4_chance`, ULDA-off) to 1e-9 in all 15 cells.
- **S2 hard streams.** On f10 and f30 the rule skips under 2 % of batches at both thresholds and accuracy is within 0.002 of `off`.
- **S3 text bed.** At 0.1 it skips at least 40 % of rows with a mean accuracy within 0.003 of `off`; at 0.25 it skips at least 70 % with accuracy within 0.006.
- **S4 banking77.** At 0.1 it skips under 10 % of rows and accuracy is within 0.005 of `off`; at 0.25 it skips 10-30 % with accuracy within 0.01.
- **S5 UniMoE suite** (set from the dev cells on seeds 520 and 521, below). Share of rows skipped, and change in the mean final accuracy against `off`:
  seq at 0.1: 5-15 % of rows, within 0.004; seq at 0.25: 12-35 %, within 0.006; mix and drift at 0.1: at most 2 %, within 0.002; mix and drift at 0.25: at most 10 %, within 0.004.
- **S6 saving is small** (revised after the UniMoE dev cells showed `train_expert` at 24-30 % of `observe` in the sequential schedule). On every bed the wall time inside `train_expert` is at most 35 % of the time inside `observe` for `off`, and no arm saves more than 25 % of `observe` time.
- **S7 structure.** At 0.1 the number of experts is within 1 of `off` in every cell, and at 0.25 within 2.

Judged as a count of predictions held; nothing is tuned after the run.

## Limits

Timings are wall clock on one PC and vary with load; the rows-skipped counts and accuracy are deterministic. The rule skips on the expert's own loss only, so a confidently wrong expert (a wrong, high-confidence mapping that has become stale) would also skip: the competence alarm still fires on a loss change, but a stable wrong belief is not seen by this rule. The UniMoE seeds were used before. No arm tunes the threshold per bed.
