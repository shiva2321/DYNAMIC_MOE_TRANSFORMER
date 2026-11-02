# E-063 - growmoe flag to skip training on batches an expert already knows: it fires on easy streams at no visible accuracy cost and saves little total time

- **ID:** E-063
- **Date:** 2026-11-01
- **Track:** G
- **Type:** experiment
- **Status:** completed; follows E-062; the second change to growmoe/ made in the FS-MoE work (asked for by the user)
- **Evidence grade:** B (script-generated, predictions declared before the report-seed runs, flag-off equality tested; UniMoE seeds 500-504 were used by E-054 and E-062 and serve as a paired comparison only; NOT a pre-registration; no independent reviewer)
- **Code / commit:** not committed (standing rule); changed `growmoe/config.py` (`skip_known`, `skip_min_assigned`, validation), `growmoe/pools/phylo.py` (`_known_batch`, `_absorb_only`, the branch in `observe`, `skip_stats`, `summary()["skip"]`), `unimoe/pools.py` (`_lda_absorb`, `_absorb_only`: a skipped batch still enters the closed-form head), `fsmoe/nbpool.py` (notebooks never skip); new `fsmoe/skip_check.py`, `fsmoe/skip_table.py`, `fsmoe/skip_figure.py`; tests `tests/growmoe/test_gm_skip_known.py` (11) and one in `tests/fsmoe/test_fs_nbpool.py` (1,102 pass over tests/growmoe, tests/unimoe and tests/fsmoe, 2 expected failures as before; 24 of 24 mutants of the new code killed); design `docs/superpowers/specs/2026-11-01-fsmoe-skip-design.md`
- **People:** Claude designed, built and ran it on the user's words "we can make this a flag which can be turned on/off" and "Yes, start" (2026-11-01), then alone while the user was away; no reviewer yet

## Question
Can the real pool stop training on a batch its expert already knows, behind a flag with the default unchanged, and what does that do to accuracy, the number of experts and the time of a run? (Asked after the answer to "do we have a skip?" was no.)

## Why we asked it
The pool trains every batch it routes to a known expert. The user wanted a switchable skip. I had said the compute saving "would be large"; the earlier measurement had the gate at about 97 % of compute, which says otherwise.

## Prediction (written BEFORE the report-seed runs; S5 and S6 fixed after the UniMoE dev cells)
`docs/superpowers/specs/2026-11-01-fsmoe-skip-design.md`, S1-S7. Scored by `fsmoe/skip_table.py`: **6 of 7 held**; S6 missed.

## Method (how)
`skip_known = k` (default None = off): a batch routed to an expert that is active, has trained on at least `skip_min_assigned` (256) samples and on which the competence alarm did not fire is skipped when its mean prequential loss (before any step) is at most k x ln(classes the expert has seen). A skipped batch takes no optimiser steps and no growth decision; it still enters the reservoir, the counters, the signature and (UniPool) the closed-form head. Checks (`fsmoe/skip_check.py`, one JSON per cell in `results/fsmoe/K`): UniMoE real suite (seq / mix / drift, ULDA pool, seeds 500-504), function regimes f10 and f30 (shipped pool, 128 units, seeds 710-714), text teaching bed (710-714), banking77 (710-712); arms off, 0.1, 0.25; wall time inside `train_expert` and `observe` recorded per cell; dev seeds 700 and 520-521.

## Results (copied from `results/fsmoe/K/SKIP_CHECK.md`)
- Equality: 1,102 tests pass; the 15 flag-off UniMoE cells equal the stored E-062 cells (largest gap 0).
- Rows skipped at 0.1 / 0.25 and accuracy change against off: f10 and f30 0 % / 0 %, +0.0000; text bed 54.3 % / 77.5 %, -0.0008 / -0.0009; banking77 9.0 % / 20.7 %, -0.0036 / +0.0008; UniMoE seq 7.1 % / 14.3 %, +0.0000 / -0.0006; mix 0.0 % / 5.8 %, +0.0000 / +0.0005; drift 0.0 % / 5.2 %, +0.0000 / +0.0001. Experts within 1 of off in every cell.
- Time (S6 missed): train share of observe time with the flag off 7-25 % (f10 8.1, f30 7.1, text 12.0, banking 22.1, seq 23.6, mix 24.8, drift 19.8); the declared check "no arm saves more than 25 % of observe time" failed on mix (25.8 / 27.5 %) and drift (21.3 / 21.7 %), where 0.1 skipped nothing: the first cell of each process (the flag-off arm of the first seed) carries a cold start (mix 4.1 s against 1.5 s). Exploratory, first seed of each bed left out: whole-run time saved at 0.1 / 0.25 -0.9 / -1.0 % (f10), 0.0 / +0.3 % (f30), +5.2 / +7.5 % (text), +1.8 / +3.7 % (banking), +2.1 / +4.6 % (seq), -0.7 / +1.7 % (mix), +0.5 / +0.9 % (drift); training time saved on the text bed 51 / 77 %, banking 16 / 26 %, seq 7 / 14 %.

## Interpretation (what we understood, and on what basis)
Plain words: the rule skips only what the expert already predicts, so it fires on easy repetitive data and hardly at all on hard data, and it costs nothing visible in accuracy; but training is a small part of the pool's time (the gate and routing are the rest), so even skipping most of the training saves only a few percent of a run. Technical: measured: the counts, accuracies and times above; not isolated: why banking at 0.1 reads -0.0036 against +0.0008 at 0.25 (three seeds, per-seed gaps of both signs), read as noise.

## Caveats and what went wrong
- I said the saving "would be large" (F-88). It is not.
- My first UniMoE dev attempt overlapped a mutation check that edits the source temporarily, so a run could have loaded a broken file: stopped, cells deleted, rerun cleanly (F-89). A timing wrapper copied its counters, so dev timings read zero: fixed, and a cell cached without timings now raises (F-90).
- I first wrote the cells into another lab's results folder (results/fsmoe/S) and moved them to results/fsmoe/K; nothing was overwritten (F-91).
- S6 missed because my timing design had no warm-up (F-92).
- UniMoE seeds 500-504 were used before (paired comparison only); thresholds were chosen on dev seeds 700 and 520-521; the rule judges an expert by its own loss, so a stably wrong expert would also skip.

## Consequence
Report `research/FSMOE_SKIP_REPORT_2026-11-01.md`, figure `results/fsmoe/K/FIGURE_SKIP.png`. The flag stays OFF by default (D-60): an opt-in for easy, repetitive streams. If run time is the goal, the gate is the thing to attack (Q-68). Claims C-176 to C-178; failures F-88 to F-92; decisions D-60, D-61.

## Evidence
- `research/FSMOE_SKIP_REPORT_2026-11-01.md` - the plain-English report
- `results/fsmoe/K/SKIP_CHECK.md` - the scorecard and the per-bed table
- `results/fsmoe/K/FIGURE_SKIP.png` - summary figure
- `docs/superpowers/specs/2026-11-01-fsmoe-skip-design.md` - design, dev findings, predictions

## Reproduce
```
python -m pytest tests/growmoe tests/unimoe tests/fsmoe
python -m fsmoe.skip_check run --bed f10 --seeds 710 711 712 713 714 --arms off k0.1 k0.25     (also f30, t; bank with seeds 710-712; umix, udrift, useq with seeds 500-504)
python -m fsmoe.skip_table
python -m fsmoe.skip_figure
```
