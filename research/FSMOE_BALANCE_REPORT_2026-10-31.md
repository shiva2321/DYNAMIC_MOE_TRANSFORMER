# FS-MoE domain-balanced training: what it fixes and what it costs (2026-10-31)

You asked for domain balance first. Same rules as before: predictions written first (12 of them, 5 held), dev seed 700 for smoke tests only, report seeds 710-714, every number copied from a generated table (links at the end). Figure: [FIGURE_BALANCE.png](../results/fsmoe/FIGURE_BALANCE.png).

## The short answer

1. **Balancing trades code for arithmetic; it does not win on both.** With equal domain shares in every training chunk and the surplus of code unused (D1), arithmetic learned goes from **0.20 to 0.67** (the level of the system without the tool), but code learned falls from **1.00 to 0.80** (4 of the 5 new code skills) and known code skills in new wordings from 0.959 to 0.906. The middle variant (D2, scarce domains replayed) lands between: arithmetic 0.40, code 0.92, code in new wordings 0.957.
2. **The headline does not move.** Never-taught skills learned, out of 28: A3 0.32, D1 0.35, D2 0.35 (intervals overlap completely). Twenty of the 28 are language skills and language stays at 0.17-0.20 in every arm.
3. **My second guess was wrong.** I thought the shared closed-form head favoured the code classes simply because it had seen thousands of them (class counts act as a prior). Removing that prior (uniform class priors) changed nothing: arithmetic 0.33 against 0.20 (within noise), and with the balanced chunks it is the same 0.40 to the digit. So the crowding does not come from the head's class counts; where it does come from is untested (see below).
4. **A cost I had not predicted:** known language skills in new wordings drop under both balanced arms (D1 0.772, D2 0.761, against 0.795 before the stream and 0.798 for A3). I do not know why. It is not replay: D1 uses no replay at all and drops too (0.772).

## What was done

Same stream, same pool and global head, same tool and human channels as the end-to-end test (arm A3). Only how verified items enter the pool changes: instead of arrival order, every chunk of 16 is built from the domains seen so far in equal shares (6/5/5 with three domains). Inside a domain, items of classes the pool was never taught come first, spread over those classes, then known items, oldest first.
- **D1 (`sub`)**: a chunk is formed only when every domain holds its share; the pace is the scarcest domain's, and the surplus of code is never used (17 chunks, 100 / 85 / 85 items for arithmetic / code / language).
- **D2 (`over`)**: a chunk is formed as soon as any domain holds its share; a domain that is short fills its share by replay from its own reservoir (500 items), so every code item is used once and the scarce domains repeat (468 chunks; 100 / 2,338 / 128 fresh items).
- **A domain head** guesses the domain of human-labelled items (a closed-form head fitted on the teaching rows); it places **0.985** of them correctly (arithmetic against language; code items come from the tool, so this is the easy half).
- **A3u / D2u** repeat A3 and D2 with the global head's class priors removed.

| arm | learned all 28 | arithmetic (3) | code (5) | language (20) | known, new wordings: all | arithmetic | code | language |
|---|---|---|---|---|---|---|---|---|
| before the stream | | | | | 0.885 | 0.935 | 0.960 | 0.795 |
| A3b arrival order (= E-057 A3) | 0.32 | 0.20 | 1.00 | 0.17 | 0.883 | 0.914 | 0.959 | 0.798 |
| D1 equal shares, surplus unused | 0.35 | 0.67 | 0.80 | 0.19 | 0.849 | 0.906 | 0.906 | 0.772 |
| D2 equal shares, scarce replayed | 0.35 | 0.40 | 0.92 | 0.20 | 0.868 | 0.925 | 0.957 | 0.761 |
| A3u arrival order, uniform priors | 0.34 | 0.33 | 1.00 | 0.18 | 0.881 | 0.908 | 0.958 | 0.797 |
| D2u equal shares + uniform priors | 0.35 | 0.40 | 0.92 | 0.20 | 0.867 | 0.927 | 0.955 | 0.760 |

## What I take from it

- **No arm dominates.** D1 is the arithmetic-friendly setting, A3 (and A3u) the code-friendly one, D2 a compromise. Choosing among them is a choice about which domain you value more, not a free improvement. D1 also leaves the tool's thousands of free labels almost unused, which is a waste.
- **The tool still pays for itself in labels** (the human budget stays at 0.59 of the all-human setting), and balanced chunks bring the headline back to the level of the no-tool system (0.35 against A1's 0.36), so the tool's code self-labelling costs nothing in the headline once the mixture is balanced. It does not add to it either.
- **Language is the binding limit.** 20 of 28 skills, 0.17-0.20 in every arm: balancing cannot give language more labelled items than the 10 % human budget provides, and choosing which items to label is the weak step (Q-54).
- **Caution on arithmetic:** only 3 skills, 5 seeds; the interval for D1 (0.37 to 0.96) and A3's (-0.03 to 0.43) barely separate. Read it as "balancing clearly helps arithmetic", not as the exact size.

## What I would do next (your call)

1. **Hybrid:** under equal domain shares, give priority to the tool's new-behaviour items (which open classes) over its known items, so code reaches 1.00 at D1's low pace while arithmetic keeps 0.67. This is the obvious combination of the two columns above. (Correction, written when I built it: D1 already puts never-taught classes first inside every domain; what limits its code is the quantity, 85 items at the scarcest domain's pace, not the order. The real hybrid exempts new-class items from the throttle, see the hybrid report.)
2. **Find where the crowding enters:** the shared covariance of the closed-form head (estimated from all items, so thousands of code items shape it) and the pool's experts are the candidates; test by freezing the covariance on teaching rows, or by running the head alone and the pool alone. The class priors are ruled out.
3. **Better selection for language** (clustering the unusual items and labelling one per cluster), the main lever on the headline.

## What went wrong or changed on the way

- Hit rate of the 12 predictions: 5 held, 7 missed. Missed: D1's code accuracy, D2's three-part claim, D2's overall and language accuracy, and all three predictions that depended on the uniform-prior idea (it was the wrong explanation).
- The first tests of the balance code let 4 of 14 mutants survive (the share split, ordering, the "any domain ready" rule and the reservoir sampling); I strengthened the tests and all 14 are now caught, 83 tests pass in `tests/fsmoe`.
- A reproduction check: the new A3b arm reproduced E-057's A3 in 5 of 5 seeds.
- The figure's first layout had no room for its legends; I fixed it before sharing.

## Evidence

[REPORT_B.md](../results/fsmoe/B/REPORT_B.md), [SCORECARD_B.md](../results/fsmoe/SCORECARD_B.md), [FIGURE_BALANCE.png](../results/fsmoe/FIGURE_BALANCE.png), design and predictions [2026-10-31-fsmoe-balance-design.md](../docs/superpowers/specs/2026-10-31-fsmoe-balance-design.md), notebook entry E-058; code `fsmoe/balance.py`, `fsmoe/e2e.py` (`run_balance`, `run_priors`), `fsmoe/learn_lab.py` (`uniform_priors`).
