# E-059 - FS-MoE hybrid: new-class items exempt from the domain throttle sit on the same arithmetic-code trade-off

- **ID:** E-059
- **Date:** 2026-10-31
- **Track:** G
- **Type:** experiment
- **Status:** completed on report seeds 710-714 (5 seeds); follows E-058
- **Evidence grade:** B (script-generated, 5 seeds, predictions declared before any hybrid cell and scored by script; the "dial" reading of the grid is post hoc and exploratory, grade C; NOT a pre-registration; no independent reviewer)
- **Code / commit:** not committed (standing rule); `fsmoe/balance.py` (mode `hybrid`, `new_cap`, `used_new` counters), `fsmoe/e2e.py` (`run_hybrid`, `HYBRID_ARMS`), `fsmoe/report_h.py`, `fsmoe/scorecard_h.py`, `fsmoe/figures_h.py`; tests `tests/fsmoe/test_fs_balance.py` (89 pass in tests/fsmoe; 14 new mutants of the hybrid code all killed, and the 14 of E-058 still killed after a refactor of `_order`; 50 mutants killed over E-055 to E-059); design `docs/superpowers/specs/2026-10-31-fsmoe-hybrid-design.md`; growmoe/ and unimoe/ untouched
- **People:** Claude designed, built and ran it on the user's answer "Hybrid new-code-first" (2026-10-31) to the follow-up question after E-058 (Q-59); no reviewer yet

## Question
Can the arithmetic-code trade-off of E-058 (D1: arithmetic 0.67, code 0.80; arrival order: 0.20, 1.00) be broken by keeping equal domain shares but training EVERY item of a never-taught class (Q-59)?

## Why we asked it
User choice after E-058. The E-058 report proposed "new-behaviour items first" as the hybrid; reading the code showed D1 already orders never-taught classes first inside each domain (the tool's new classes carry ids that are not taught ids), so the quantity, not the order, limits D1 (85 code items trained of 358 new-class ones produced). The design therefore exempts never-taught classes from the throttle.

## Prediction (written BEFORE the numbers)
`docs/superpowers/specs/2026-10-31-fsmoe-hybrid-design.md`: H1-H8. Scored by `fsmoe/scorecard_h.py`: **4 of 8 held** (H3, H4, H5, H8); missed H1 (some hybrid arm with arithmetic >= 0.50 AND code >= 0.90: H8 0.467 / 1.00, H12 0.533 / 0.80), H2 (grid monotone: arithmetic goes up with m, not down: wrong direction), H6 (code known held-out 0.90-0.94: H4 0.943), H7 (known held-out >= 0.86 in every hybrid arm: H12 0.850).

## Method (how)
- Same stream, teaching, pool, global head, tool, domain head and human labels as E-057 A3 / E-058 (228 human labels in seed 710; the count varies a little by seed).
- Mode `hybrid`: items of classes the pool was never taught go to one queue in order of arrival (any domain); a chunk is formed the moment the queue holds m items or every domain holds its share of the known part (16 - m slots, equal shares, `sub` rule); chunk = the queue + known items of each domain up to its share, oldest first; no replay; known surplus of the abundant domain unused; remaining items flushed at the end under the same rule.
- Arms: D1r (`sub`, re-run with the new counters, reproduced E-058's D1 in 5 of 5 seeds), H4, H8, H12 (m = 4, 8, 12, a grid reported as a grid). A3b and D2 are the E-058 results of the same seeds.
- Counters (all modes): fresh items trained per domain, split into never-taught-class and known items.
- Seeds: dev 700 (smoke test); report 710-714.

## Results (copied from the generated files)
`results/fsmoe/B/REPORT_H.md`, `results/fsmoe/SCORECARD_H.md` (mean over 5 seeds):
- Learned, all 28 / arithmetic (3) / code (5) / language (20): A3b 0.32 / 0.20 / 1.00 / 0.17; D2 0.35 / 0.40 / 0.92 / 0.20; D1r 0.35 / 0.67 / 0.80 / 0.19; H4 0.34 / 0.33 / 1.00 / 0.18; H8 0.36 / 0.47 / 1.00 / 0.18; H12 0.34 / 0.53 / 0.80 / 0.19.
- Known skills in held-out wordings, all / arithmetic / code / language (before the stream 0.885 / 0.935 / 0.960 / 0.795): A3b 0.883 / 0.914 / 0.959 / 0.798; D2 0.868 / 0.925 / 0.957 / 0.761; D1r 0.849 / 0.906 / 0.906 / 0.772; H4 0.872 / 0.912 / 0.943 / 0.788; H8 0.864 / 0.918 / 0.928 / 0.784; H12 0.850 / 0.893 / 0.910 / 0.777.
- Fresh items trained (arithmetic / code / language): D1r 100 / 85 / 85, all 85 code items of never-taught classes (of 358 produced); H4 100 / 786 / 128; H8 100 / 523 / 125; H12 100 / 404 / 76. Chunks: 17, 107, 55, 46. New-class code items trained: 358 in every hybrid arm; known code items trained: 0 (D1), 428 (H4), 165 (H8), 47 (H12).
- Per seed (710-714), code learned: H12 0.80 / 0.20 / 1.00 / 1.00 / 1.00; D1r 0.60 / 1.00 / 1.00 / 0.60 / 0.80; arithmetic learned moves in steps of 0.33 per seed (3 skills). H12 opened 5-6 classes per seed with purity 1.0; H8 and D1r opened the same number of classes.
- Post hoc ordering by known code items trained (D1 0, H12 47, H8 165, A5 of E-057 190, H4 428, A3b about 1,980): arithmetic learned 0.67 / 0.53 / 0.47 / 0.40 / 0.33 / 0.20; known held-out all 0.849 / 0.850 / 0.864 / 0.869 / 0.872 / 0.883; code known 0.906 / 0.910 / 0.928 / 0.933 / 0.943 / 0.959; language known 0.772 / 0.777 / 0.784 / 0.792 / 0.788 / 0.798 (one inversion). D2 (about 1,980 known code items, replayed arithmetic) is the exception: arithmetic 0.40.

## Interpretation (what we understood, and on what basis)
Plain words: training every new-class item while sharing the rest of each chunk equally gives a better compromise (H8: all code skills, arithmetic 0.47) than arrival order or the replay variant, but it does not break the trade-off: arithmetic above 0.6 still needs code throttled, and the share of all 28 skills learned stays about 0.35 with language at 0.18. The arms differ chiefly in how many known code items they train, and that number orders arithmetic learned and known-skill accuracy in opposite directions. Technical: measured: the table above, the counters, the reproduction of D1; not measured: the cause (the dial reading comes from arms that differ in chunk composition and count; D2 does not fit; arithmetic has 3 skills, one skill in one seed is 0.07 of the mean); H12's code drops in seeds 710 and 711 are unexplained.

## Caveats and what went wrong
- My E-058 proposal was already implemented in D1 (F-77); I also named replay a candidate cause of the language drop although D1 uses none (corrected in E-058, C-152, Q-60).
- H2's direction was wrong and H1 missed by 0.03; 4 of 8 predictions held (F-78).
- The first draft took new-class items round-robin over classes; the mutation check showed it could never matter (the queue never exceeds m at a build); removed before any run (F-79).
- Three arithmetic skills, five seeds, grid of three values of one parameter; the dial ordering is exploratory.
- One code domain of generated templates; known held-out accuracy intervals about +-0.02.

## Consequence
Report: `research/FSMOE_HYBRID_REPORT_2026-10-31.md`; figure `results/fsmoe/FIGURE_HYBRID.png`. Q-59 answered (partly): the hybrid is a better compromise, not a way out; new Q-61 (the dial test). Decisions D-52, D-53. Claims C-153..C-156. Nothing becomes a default: A3 stays the end-to-end reference, H8 is a candidate that awaits the user's choice.

## Evidence
- `research/FSMOE_HYBRID_REPORT_2026-10-31.md` - the plain-English report
- `results/fsmoe/B/REPORT_H.md` - generated tables
- `results/fsmoe/SCORECARD_H.md` - the scorecard
- `results/fsmoe/FIGURE_HYBRID.png` - the figure
- `docs/superpowers/specs/2026-10-31-fsmoe-hybrid-design.md` - design and predictions
- Also changed during this step (the manifest was re-generated for them): `research/FSMOE_BALANCE_REPORT_2026-10-31.md`, E-058, C-152, Q-60 (the replay sentence)

## Reproduce
```
python -m fsmoe.e2e hybrid --seeds 710 711 712 713 714
python -m fsmoe.report_h --seeds 710 711 712 713 714
python -m fsmoe.scorecard_h --seeds 710 711 712 713 714
python -m fsmoe.figures_h --seeds 710 711 712 713 714
python -m pytest tests/fsmoe
```
