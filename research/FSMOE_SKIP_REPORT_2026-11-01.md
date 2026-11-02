# The skip flag: stop training on batches an expert already knows (2026-11-01)

You said we can make the skip a flag. It is in growmoe, off by default, and I ran the standard checks. The short version: **it does what it should and costs almost nothing in accuracy, but it saves little total time, because training is only a small part of the run.** Predictions first (7, of which 6 held). Figure: [FIGURE_SKIP.png](../results/fsmoe/K/FIGURE_SKIP.png). Every number is copied from a generated table (links at the end).

## The short answer

1. **The flag.** `PhyloCfg.skip_known` (default `None` = off). When set to, say, 0.1: a batch routed to a settled expert is skipped if the loss the expert had on it **before any training step** is at most 0.1 times the chance loss. A skipped batch takes no optimiser steps, but it still goes into the expert's memory, its fingerprint and (in the pool with a closed-form head) the head. The competence alarm is untouched, and notebook experts never skip.
2. **Off means unchanged.** The whole suite passes (1,102 tests: 1,090 earlier plus 12 new, the same 2 expected failures), and the 15 flag-off UniMoE cells equal the stored earlier cells exactly (largest gap 0). 24 deliberate breakages of the new code were each caught by a test.
3. **It fires where the data is easy and almost never where it is hard.** Share of rows skipped at 0.1 / 0.25: function regimes 0 % / 0 % (the expert never gets under half of chance loss there); text teaching bed 54 % / 78 %; banking77 9 % / 21 %; the real seven-domain suite: sequential 7 % / 14 %, mixed 0 % / 6 %, drifting 0 % / 5 %.
4. **Accuracy cost is within noise.** Mean change against flag off, at 0.1 / 0.25: text bed -0.0008 / -0.0009; banking77 -0.0036 / +0.0008; sequential suite 0.0000 / -0.0006; mixed and drifting 0.0000 / +0.0005 and +0.0001. The number of experts changes by at most 1 in any cell.
5. **Total time saved is small.** Training is 7 to 28 % of the pool's time; the rest is the gate and routing. Even where 78 % of the rows are skipped (the text bed) the whole run gets only about 7 % faster, because training time falls by 77 % but is 12 % of the total. **This confirms the correction I made last turn:** I had said the saving would be large, and it is not.
6. **One of my predictions missed, and the reason is my timing design.** I predicted no arm would save more than 25 % of run time; the mixed and drifting schedules showed 26 % and 21 %. At 0.1 those arms skipped nothing and computed exactly what the flag-off arm computed, so the "saving" is a cold start: the first cell of each process (the flag-off arm of the first seed) ran up to about 3 times slower than the later ones (4.1 s against 1.5 s on the mixed schedule). Leaving that first cell out (an after-the-fact look, marked exploratory) the whole-run saving is between -1 % and +7.5 % everywhere.

## What I would do with it

Leave it off by default. It is a safe opt-in for easy, repetitive streams, where it cuts training work by half or more at no visible cost. If total speed is the goal, the thing to attack is the gate, which takes most of the time; skipping training cannot touch that.

## What I got wrong, and what to know

- **The "large saving" claim** (corrected above).
- **A process mistake:** my first attempt at the UniMoE dev cells overlapped a mutation check that temporarily edits the source, so a run could have loaded a deliberately broken file. I stopped it, deleted those cells and ran them again cleanly; the numbers here are from the clean run. A timing wrapper also copied its counters instead of sharing them (timings read zero in the first dev attempt); fixed.
- **A tidy-up:** I first wrote the check results into another lab's folder (`results/fsmoe/S`), then moved them to their own (`results/fsmoe/K`); nothing of the other lab was overwritten.
- **A rule that stays blunt:** it judges an expert by its own loss, so an expert that is stably and confidently wrong would also skip; the competence alarm only catches a change.
- The UniMoE seeds 500-504 were used in earlier rounds, so that comparison is paired, not fresh; the threshold was chosen on dev seeds (700 and 520-521), not on the report seeds.

## Evidence

[SKIP_CHECK.md](../results/fsmoe/K/SKIP_CHECK.md) (the scorecard and the per-bed table), [FIGURE_SKIP.png](../results/fsmoe/K/FIGURE_SKIP.png), design and predictions [2026-11-01-fsmoe-skip-design.md](../docs/superpowers/specs/2026-11-01-fsmoe-skip-design.md), notebook entry E-063; code `growmoe/config.py`, `growmoe/pools/phylo.py` (the rule), `unimoe/pools.py` (closed-form head still fed), `fsmoe/nbpool.py` (notebooks never skip), `fsmoe/skip_check.py`, `fsmoe/skip_table.py`, `fsmoe/skip_figure.py`; tests `tests/growmoe/test_gm_skip_known.py` (11) and one in `tests/fsmoe/test_fs_nbpool.py`.
