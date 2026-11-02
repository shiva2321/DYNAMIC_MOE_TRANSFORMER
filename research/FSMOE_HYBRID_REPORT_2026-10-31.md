# FS-MoE hybrid: new classes exempt from the domain throttle (2026-10-31)

You chose the hybrid. Same rules: predictions first (8, of which 4 held), dev seed 700 for the smoke test only, report seeds 710-714, every number copied from a generated table (links at the end). Figure: [FIGURE_HYBRID.png](../results/fsmoe/FIGURE_HYBRID.png).

## The short answer

1. **First, a correction to my own proposal.** I recommended "equal domain shares, but the tool's new-behaviour items first". D1 already does exactly that (inside every domain, never-taught classes go first). What limited D1's code was the **quantity**: it trained 85 code items, all of never-taught classes, while the tool produced 358 of them. So the real hybrid exempts never-taught classes from the throttle: every one of them is trained, 4, 8 or 12 per chunk (H4, H8, H12), and the chunk is filled with known items in equal domain shares.
2. **The hybrid is a better compromise than the earlier arms, not a way out of the trade-off.** The best one, H8, learns **all 5 code skills (1.00) and 0.47 of the arithmetic skills**, with known skills in new wordings at 0.864 (0.885 before the stream). Arrival order gave arithmetic 0.20 at the same code 1.00, and the replay variant D2 gave 0.40 and 0.92. But H8 misses the bar I set in advance (arithmetic 0.50 or more with code 0.90 or more) by 0.03, and arithmetic has only 3 skills: the gap to D2 is about one skill over the five seeds.
3. **Arithmetic 0.67 is still only reached by throttling code** (D1: code 0.80). H12 gets arithmetic 0.53 but its code learned drops in 2 of 5 seeds (to 0.20 in seed 711 and 0.80 in seed 710), a cause I did not investigate.
4. **The headline stays where it was:** never-taught skills learned of all 28 are 0.34 (H4), 0.36 (H8), 0.34 (H12), against 0.32 for arrival order and 0.35 for D1 and D2. Language is 0.18-0.19 in every arm.
5. **What the grid shows after the fact, and is worth testing:** with the new-class items fixed, the arms differ mostly in how many *known* code items they train (0, 47, 165, 428, about 1,980). That number orders the results in one direction: the more known code items, the lower arithmetic learned (0.67, 0.53, 0.47, 0.33, 0.20) and the higher the accuracy on known skills in new wordings (0.849, 0.850, 0.864, 0.872, 0.883). The earlier arm A5 (190 known code items) lands where this predicts. So the free known code items look like a **dial** between "learn the new arithmetic" and "keep known skills accurate", not a pure waste or a pure gain. This is an ordering of means over arms that differ in other ways too, so it is a hypothesis for the next test, not a finding.

## What was done

Same stream, pool, global head, tool and human channels as the end-to-end test (arm A3; 228 human labels). Only how verified items enter the pool changes:
- **Never-taught classes** (a human-labelled new skill of any domain, or a class the tool opened) wait in one queue, in order of arrival; a chunk is formed the moment the queue holds m of them. None waits for another domain.
- **Known items** fill the rest of the chunk (16 - m slots) in equal shares of the domains seen, oldest first; known items are never replayed, and the surplus of code's known items is not used.
- **Grid:** m = 4, 8, 12 (H4, H8, H12), reported as a grid, not a pick. **D1r** re-runs D1 with counters for new-class items and reproduced it in 5 of 5 seeds.

| arm | learned all 28 | arithmetic (3) | code (5) | language (20) | known code items trained | known, new wordings: all | arithmetic | code | language |
|---|---|---|---|---|---|---|---|---|---|
| before the stream | | | | | | 0.885 | 0.935 | 0.960 | 0.795 |
| A3b arrival order (E-058) | 0.32 | 0.20 | 1.00 | 0.17 | about 1,980 | 0.883 | 0.914 | 0.959 | 0.798 |
| D2 equal shares, scarce replayed (E-058) | 0.35 | 0.40 | 0.92 | 0.20 | about 1,980 | 0.868 | 0.925 | 0.957 | 0.761 |
| D1 equal shares, code throttled | 0.35 | 0.67 | 0.80 | 0.19 | 0 | 0.849 | 0.906 | 0.906 | 0.772 |
| H4 hybrid, 4 new per chunk | 0.34 | 0.33 | 1.00 | 0.18 | 428 | 0.872 | 0.912 | 0.943 | 0.788 |
| **H8 hybrid, 8 new per chunk** | **0.36** | **0.47** | **1.00** | **0.18** | 165 | **0.864** | 0.918 | 0.928 | 0.784 |
| H12 hybrid, 12 new per chunk | 0.34 | 0.53 | 0.80 | 0.19 | 47 | 0.850 | 0.893 | 0.910 | 0.777 |

Items trained: D1 100 / 85 / 85 (arithmetic / code / language), H4 100 / 786 / 128, H8 100 / 523 / 125, H12 100 / 404 / 76; chunks 17, 107, 55, 46. Arithmetic learned moves in steps of 0.33 per seed (3 skills), so a difference of 0.07 in the mean is one skill in one seed.

## What I take from it

- **H8 is the setting I would pick if you want one**: all 5 new code skills, arithmetic at 0.47, known skills 0.864, no extra labels. It is a modest gain over D2 and arrival order, within the noise on arithmetic. If arithmetic matters most, D1 is still the only setting above 0.6, at the price of one or two of the 5 code skills in 3 of 5 seeds.
- **The trade-off is structural in this design**, and the dial in point 5 suggests where it sits: the known code items. A cleaner test than another arm: hold the chunking fixed and vary only the number of known code items (0, 50, 150, 400, 2,000), and separately freeze the head's covariance on the teaching rows (the candidate mechanism from the last report).
- **Language did not move and will not under any chunking**: 0.18-0.19 in every arm, 20 of the 28 skills; its limit is which items get labelled.

## What I would do next (your call)

1. **Language selection** (cluster the unusual items, label one per cluster): the main lever on the headline, which has stayed at about 0.35 in every round since E-057.
2. **The dial test above** (known code count at fixed chunking, plus the frozen-covariance variant): it turns the post hoc ordering into a measured cause, and tells whether the trade-off can be removed rather than chosen.

## What went wrong or changed on the way

- Hit rate of the 8 predictions: 4 held (D1r reproduces D1; headline between 0.33 and 0.38; language learned at most 0.30; language known accuracy at most 0.79), 4 missed. H1 missed by 0.03 (H8 arithmetic 0.467), H6 by 0.003 (H4 code known accuracy 0.943 against a 0.90-0.94 band), H7 missed in H12 (0.850), and **H2 was wrong in direction**: I expected arithmetic to fall as m grows; it rises, because m does not change how many new-class items are trained (all 358 in every arm) but how many known ones accompany them.
- My first hybrid draft took the new-class items round-robin over their classes. The mutation check showed that rule could never matter (the queue never holds more than m items when a chunk is formed), so it was removed before any run and the design text says so.
- Last report I called replay a candidate cause of the language drop; D1 uses no replay and drops too, so I corrected that sentence there.
- Two figure layouts needed fixing (overlapping scatter labels, a wrapped title).
- Tests: 89 pass in `tests/fsmoe`; 14 mutants of the hybrid code are all caught, and the 14 of the last round still are after the refactor.

## Evidence

[REPORT_H.md](../results/fsmoe/B/REPORT_H.md), [SCORECARD_H.md](../results/fsmoe/SCORECARD_H.md), [FIGURE_HYBRID.png](../results/fsmoe/FIGURE_HYBRID.png), design and predictions [2026-10-31-fsmoe-hybrid-design.md](../docs/superpowers/specs/2026-10-31-fsmoe-hybrid-design.md), notebook entry E-059; code `fsmoe/balance.py` (mode `hybrid`), `fsmoe/e2e.py` (`run_hybrid`).
