# E-060 - FS-MoE growth lab: notebook-first right-sizing works, later growth is idle, the tier hurts

- **ID:** E-060
- **Date:** 2026-11-01
- **Track:** G
- **Type:** experiment
- **Status:** completed on report seeds 710-714 (5 seeds, three scenarios); follows E-059; new compact lab, NOT the production pool
- **Evidence grade:** B (script-generated, 5 seeds, predictions declared before any report-seed run and scored by script, an oracle answer key computed per seed; governor knobs tuned on dev seeds 700-702; NOT a pre-registration; no independent reviewer)
- **Code / commit:** not committed (standing rule); `fsmoe/grow/` (`world.py`, `system.py`, `governor.py`, `lab.py`, `report_g.py`, `scorecard_g.py`, `figures_g.py`); tests `tests/fsmoe/test_fs_grow.py` (31 tests; 120 pass in tests/fsmoe; 25 mutants of the new code killed, one only after a test was added; 75 mutants killed over E-055 to E-060); design `docs/superpowers/specs/2026-11-01-fsmoe-growth-design.md`; growmoe/ and unimoe/ imported (`MLPExpert`, `GrowthPolicy`, `guard_reverts`, `LdaHead`), never edited
- **People:** Claude designed, built and ran it on the user's request (2026-10-31, 2026-11-01) "experts can grow as needed and the system can be grown in both directions, governed algorithmically" and the answer "Yes, build it" to the plan; no reviewer yet

## Question
Can experts and the pool be grown as needed, in width, depth, number of experts and a taller system, by one governor that grows only on evidence, checks on unseen data, keeps the cheapest option that works and undoes what did not help? Does that beat fixed-size experts at equal memory (the earlier record: growth almost never fired, C-122, Q-07, Q-41)?

## Why we asked it
User requirement after E-059; notes: wrong spawns are permanent (merge research), growth never fired in real tests, the taller system (`stack.py`) was never evaluated (R12b paused, D-10), experts live too short to be judged (Q-41).

## Prediction (written BEFORE the report-seed runs)
`docs/superpowers/specs/2026-11-01-fsmoe-growth-design.md`: P1-P14, fixed after dev runs on seeds 700-702. Scored by `fsmoe/grow/scorecard_g.py`: **8 of 14 held** (P1, P2, P3, P4, P7, P8, P10, P14); missed P5 (depth-need regimes with 2+ blocks: 0.11 of 9), P6 (none-need regimes with growth: 0.32 of 31), P9 (G at least GnoNB minus 0.005: 0.730 against 0.672), P11 (big: +0.014 over F8, not +0.02), P12 (tier refused in 4 of 5 seeds: built in 5 of 5), P13 (big purity 0.924).

## Method (how)
- World: inputs are directions on a sphere (d = 32), K = 6 classes; regimes with centres at least 0.3 apart in cosine (big: 8 groups, siblings at most 0.45); families `lin`, `wide` (max of 24 linear pieces), `deep` (products of four tanh features). Oracle answer key per seed: small (8 units), wide (64) and deep (8 x 3) nets trained offline on 12,000 samples; need = width / depth / none by a 5-point margin. Phases of 4 regimes, 400 batches of 128, half old regimes. Scenarios: `mixed` (4 lin, 8 wide, 4 deep), `easy` (8 lin), `big` (36 lin, 6 wide, 6 deep; flat gate 1,536 multiply-adds per sample against a budget of 640).
- System (`system.py`): expert = closed-form notebook (`LdaHead`) + optional `MLPExpert`, answer = 50/50 mixture; routing by cosine to centroids with a per-expert acceptance cosine (0.5 % quantile of its own, floor 0.38); room of unaccepted samples, dense groups found by pairwise cosine; a group whose median cosine to an existing centroid is at least 0.25 is absorbed; 15 % of every expert's samples held out and never trained; optional two-step routing (tier) with a full-scan fallback; multiply-add accounting.
- Governor (`governor.py`): notebook -> network by a held-out test of candidate sizes (8x1, 8x3, 32x1, 64x1; 200 steps) against the notebook, cheapest within 1.5 points of the best, margin 2 points, judged again when data doubles; growth trial = `GrowthPolicy` (widen x2/x4, deepen; 240 steps) only for experts with 6+ batches, 600+ training and 100+ held-out samples and held-out error above 0.04, budget `trial_ratio` 1.0, guard of 2 looks (undo if held-out loss worse by 5 %); tier when the flat gate exceeds 640: k-means groups over centroids, kept only if routing agrees on at least 98 % of held-out samples and costs at most 80 %.
- Arms: F8, F16, F32, F64, F8d3 (fixed experts), BF (brute force), G, Gng (no width/depth growth), GnoNB (network at birth), Gnt (no tier). Seeds: dev 700-702 (tuning, three repairs), report 710-714.
- Dev tuning (mixed, 700-702): promotion steps and trial ratio; mean accuracy 0.733 (300 steps, ratio 2), 0.725 (150), 0.717 (3 sizes), 0.733 (200 steps, ratio 1.0 = the default G, the cheapest that matched the best).

## Results (copied from the generated files)
`results/fsmoe/G/REPORT_G.md`, `results/fsmoe/SCORECARD_G.md` (mean over 5 seeds):
- Mixed, accuracy / worst regime / total parameters / k multiply-adds per answer: F8 0.669 / 0.506 / 34k / 1.6; F16 0.690 / 0.537 / 42k / 2.1; F32 0.711 / 0.572 / 59k / 3.1; F64 0.726 / 0.595 / 92k / 5.2; F8d3 0.673 / 0.523 / 54k / 2.8; BF 0.732 / 0.623 / 563k / 34.6; Gng 0.730 / 0.580 / 56k / 3.0; GnoNB 0.672 / 0.506 / 38k / 1.8; **G 0.730 / 0.580 / 56k / 3.0**.
- Training and governing multiply-adds (G-MACs): G 6.5 + 29.0 (4.5 times); GnoNB 4.3 + 4.5; BF 125.7 + 0; F64 20.8.
- Against the oracle (G): width-need 40 regimes: 0.85 with 16+ units; depth-need 9: 0.67 wider, 0.11 deeper, accuracy 0.630 (F8 0.558, BF 0.670); none-need 31: 0.42 notebook only, 0.23 wide, 0.10 deeper, accuracy 0.801. Events per seed (G): spawn 16.0, absorb 12.8, promote 13.2, keep notebook 16.0, growth trial rejected 14.2, grow 0.6, keep 0.4, revert 0.2.
- Easy: G 0.902 (13.6k parameters, 0.89 k-MACs, 0 growth events), F8 0.861 (17.1k), F32 0.859, BF 0.842 (281k, 56 growth events per seed).
- Big (48 regimes): F8 0.742 / worst 0.502; governor without tier 0.774 / 0.564 / purity 0.954 / 2.64 k-MACs; G with tier 0.756 / 0.471 / purity 0.924 / 2.33 k-MACs, tier built 3.6 times per seed (5 of 5 seeds).
- Pool against regimes (G): 1.00 experts per regime in all scenarios; purity 0.986 (mixed), 0.991 (easy), 0.924 (big).
- Speed note (measured while the runs were going): one training step of these networks is 1.0-1.8 ms on 2 CPU threads and 1.4-2.3 ms on the RTX 3060 (batch 128), so the runs used 6 parallel CPU processes.

## Interpretation (what we understood, and on what basis)
Plain words: giving each new expert the right size at birth, by testing candidate sizes on examples it never trained on with the notebook as the control, is what made the system as accurate as big fixed experts at a bit over half the memory and answer cost; growing experts later by a trial did nothing in these tests; the pool grew one expert per regime without duplicates; a coarse layer of expert groups cut the cost a little but lost accuracy and the check that allowed it was too weak. Technical: measured: all of the above in this lab; the gain over fixed experts comes entirely from the birth test (G - Gng = +0.000; GnoNB = F8); the tier was accepted on routing agreement (at least 0.98 on 60 held-out samples per expert) and cost, which does not protect the answer accuracy or purity (training also uses the tier routing, so misrouted samples train the wrong expert; the pool also grows between tier rebuilds); depth is not tested properly (9 regimes, width substitutes); the extra 23 % wide networks for none-need regimes partly reflect the governor's 2-point bar against the oracle's 5.

## Caveats and what went wrong
- A lab: separated regimes, labels with every sample, centroid routing; the production pool (conformal gate, vMF signatures, probation) is untouched, so transfer is not shown.
- The tier check was too weak (F-81); 8 of 14 predictions held (F-80).
- Dev runs found three design faults before any report seed (global room tightness never fired; duplicates founded from outlier tails; close regimes merged); a unit test found a real bug (a keyword clash in the refused-budget log call); the mutation check found a missing test (agreement condition of the tier) (F-82).
- The first launch of the 80 report runs failed for ten minutes on carriage returns in the job file; nothing was lost.
- I said before building that the governor would settle the held decision on the width "gap rule"; it does not (the lab uses the shadow-trial policy).

## Consequence
Report: `research/FSMOE_GROWTH_REPORT_2026-11-01.md`; figure `results/fsmoe/FIGURE_GROWTH.png`. New Q-62..Q-64. Decisions D-54, D-55. Claims C-157..C-163. Nothing becomes a default: the lab is not wired into the pool; the notebook-first newborn with a size test is the candidate to port; the tier must not be adopted as built.

## Evidence
- `research/FSMOE_GROWTH_REPORT_2026-11-01.md` - the plain-English report
- `results/fsmoe/G/REPORT_G.md` - generated tables
- `results/fsmoe/SCORECARD_G.md` - the scorecard
- `results/fsmoe/FIGURE_GROWTH.png` - the figure
- `docs/superpowers/specs/2026-11-01-fsmoe-growth-design.md` - design, dev findings and predictions
- `results/fsmoe/G/oracle_mixed_s710.json` - the oracle answer key (one file per scenario and seed)

## Reproduce
```
python -m fsmoe.grow.lab oracle --scenario mixed --seeds 710 711 712 713 714     (also easy, big)
python -m fsmoe.grow.lab run --scenario mixed --seeds 710 --arms G F8 F16 F32 F64 F8d3 BF Gng GnoNB   (also easy: G F8 F32 BF; big: G Gnt F8; one process per seed and arm, OMP_NUM_THREADS=2)
python -m fsmoe.grow.report_g --seeds 710 711 712 713 714
python -m fsmoe.grow.scorecard_g --seeds 710 711 712 713 714
python -m fsmoe.grow.figures_g --seeds 710 711 712 713 714
python -m pytest tests/fsmoe
```
