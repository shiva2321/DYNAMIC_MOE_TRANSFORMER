# E-058 - FS-MoE domain-balanced training: equal domain shares trade code for arithmetic, the head's class priors are not the cause

- **ID:** E-058
- **Date:** 2026-10-31
- **Track:** G
- **Type:** experiment
- **Status:** completed on report seeds 710-714 (5 seeds); follows E-057
- **Evidence grade:** B (script-generated, 5 seeds, predictions declared before the cells that test them and scored by script; the uniform-prior arms and their predictions M9-M12 were added after the first balance results and are exploratory; NOT a pre-registration; no independent reviewer)
- **Code / commit:** not committed (standing rule); `fsmoe/balance.py` (`BalancedLearner`, `DomainHead`), `fsmoe/e2e.py` (`deploy_e2e` with `balance` / `domain_head`, `run_balance`, `run_priors`), `fsmoe/learn_lab.py` (`GlobalFast(uniform_priors=...)`), `fsmoe/report_b.py`, `fsmoe/scorecard_b.py`, `fsmoe/figures_b.py`; tests `tests/fsmoe/test_fs_balance.py`, `tests/fsmoe/test_fs_learn_lab.py` (83 pass in tests/fsmoe; 14 mutants of the new code, 4 survived the first tests, all 14 killed after strengthening them; 36 mutants killed over E-055 to E-058); design `docs/superpowers/specs/2026-10-31-fsmoe-balance-design.md`; growmoe/ and unimoe/ untouched
- **People:** Claude designed, built and ran it on the user's answer "Balance domains first" (2026-10-31) to the follow-up question after E-057 (Q-56); no reviewer yet

## Question
In the end-to-end stream the tool labels thousands of known code items for free and arrival-order chunks are mostly code; arithmetic learned fell from 0.67 (A1) to 0.20 (A3). Does building every chunk from equal domain shares (Q-56) remove that crowding without losing code? And is the cause the shared head's class counts?

## Why we asked it
User choice after E-057 ("Balance domains first"); E-057 C-146 (partial crowding by free known code items).

## Prediction (written BEFORE the numbers)
`docs/superpowers/specs/2026-10-31-fsmoe-balance-design.md`: M1-M8, and an addendum M9-M12 (uniform class priors in the global head) written after the first balance results and before the prior arms. Scored by `fsmoe/scorecard_b.py`: **5 of 12 held** (M1, M4, M5, M8, M11); missed M2 (D1 code known held-out in 0.93-0.95), M3 (D2 arithmetic >= 0.50 and code >= 0.95 and code known >= 0.95), M6 (D2 known held-out >= 0.88), M7 (D2 language known held-out >= 0.78), M9, M10, M12 (all three rested on the class-prior explanation).

## Method (how)
- Same stream, pool, global head, tool and human channels as E-057 arm A3 (human labels 228 in every arm of seed 710, tool on all code inputs); only how verified items enter the pool changes.
- Arms: **A3b** arrival order (reproduced E-057's A3 in 5 of 5 seeds); **D1** `sub`: chunks of 16 in equal shares of the domains seen (6/5/5), formed only when every domain holds its share, code's surplus unused; **D2** `over`: formed as soon as any domain holds its share, short domains filled by replay from a reservoir of 500 verified items per domain; inside a domain, never-taught classes first (round-robin over those classes), then known items oldest first; **A3u / D2u**: A3 and D2 with the global head's class priors removed (`uniform_priors`: the log class counts are subtracted from the LDA logits before the softmax).
- A closed-form 3-class domain head fitted on the teaching rows gives the domain of human-labelled items (code items get the code domain from the tool).
- Evaluation as E-057 (learned = accuracy >= 0.5 on rows the stream never contained, per domain; known skills in held-out wordings before and after).
- Seeds: dev 700 (smoke test); report 710-714.

## Results (copied from the generated files)
`results/fsmoe/B/REPORT_B.md`, `results/fsmoe/SCORECARD_B.md` (mean over 5 seeds; intervals in the report):
- Learned, all 28 / arithmetic (3) / code (5) / language (20): A3b 0.32 / 0.20 / 1.00 / 0.17; D1 0.35 / 0.67 / 0.80 / 0.19; D2 0.35 / 0.40 / 0.92 / 0.20; A3u 0.34 / 0.33 / 1.00 / 0.18; D2u 0.35 / 0.40 / 0.92 / 0.20.
- Known skills in held-out wordings, all / arithmetic / code / language (before the stream 0.885 / 0.935 / 0.960 / 0.795): A3b 0.883 / 0.914 / 0.959 / 0.798; D1 0.849 / 0.906 / 0.906 / 0.772; D2 0.868 / 0.925 / 0.957 / 0.761; A3u 0.881 / 0.908 / 0.958 / 0.797; D2u 0.867 / 0.927 / 0.955 / 0.760. Taught wordings 0.965-0.969 in every arm.
- Chunks and fresh items trained (arithmetic / code / language): D1 17 chunks, 100 / 85 / 85; D2 468 chunks, 100 / 2,338 / 128.
- Domain head: 0.985 of the human-labelled items in the right domain (arithmetic against language). The head with uniform priors reads taught skills 0.959 and held-out wordings 0.895 after teaching, the same as with its priors.
- Intervals are wide for arithmetic (D1 0.67 [0.37, 0.96], A3b 0.20 [-0.03, 0.43]) and overlap for the headline (A3b 0.32 [0.28, 0.37], D1 0.35 [0.21, 0.49], D2 0.35 [0.29, 0.41]).

## Interpretation (what we understood, and on what basis)
Plain words: forcing every domain to have an equal say in each training chunk gives arithmetic its learning back, but only by taking it away from code; no setting wins on both, and the share of all 28 skills learned does not change (language, 20 of the 28, stays at 0.17-0.20). The head's class counts are not what crowded arithmetic out. Technical: measured: the trade in learned fractions by arm, the null effect of uniform priors (A3u against A3b, D2u against D2), the domain head's accuracy. Not measured: why arithmetic is crowded (candidates: the shared covariance of the closed-form head and the pool's experts, both shaped by thousands of code items; the covariance test was not run) and why language's held-out accuracy falls under both balanced arms (not replay: D1 uses none and drops as well).

## Caveats and what went wrong
- Three arithmetic skills and five seeds: very wide intervals; the direction (balancing helps arithmetic) is supported, the size is not.
- M9-M12 and the prior arms came after seeing the first balance results (exploratory); 7 of 12 predictions missed, including all those that rested on the prior explanation (F-75).
- The first tests of `balance.py` were too weak: 4 of 14 mutants survived (quota remainder, order of new against known items, the "any domain ready" rule of `over`, reservoir sampling); fixed (F-76). A first uniform-priors test saturated at probability 1.0 and tested nothing; replaced by overlapping classes.
- One code domain of generated templates; the domain head's task is the easy half (code comes from the tool).

## Consequence
Report: `research/FSMOE_BALANCE_REPORT_2026-10-31.md`; figure `results/fsmoe/FIGURE_BALANCE.png`. Q-56 answered (balance trades, does not dominate); new Q-58..Q-60. Decisions D-50, D-51. Claims C-148..C-152. Nothing becomes a default: A3 stays the end-to-end reference.

## Evidence
- `research/FSMOE_BALANCE_REPORT_2026-10-31.md` - the plain-English report
- `results/fsmoe/B/REPORT_B.md` - generated tables
- `results/fsmoe/SCORECARD_B.md` - the scorecard
- `results/fsmoe/FIGURE_BALANCE.png` - the figure
- `docs/superpowers/specs/2026-10-31-fsmoe-balance-design.md` - design and predictions

## Reproduce
```
python -m fsmoe.e2e balance --seeds 710 711 712 713 714
python -m fsmoe.e2e priors --seeds 710 711 712 713 714
python -m fsmoe.report_b --seeds 710 711 712 713 714
python -m fsmoe.scorecard_b --seeds 710 711 712 713 714
python -m fsmoe.figures_b --seeds 710 711 712 713 714
python -m pytest tests/fsmoe
```
