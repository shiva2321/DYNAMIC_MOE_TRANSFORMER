# E-055 - FS-MoE: a recognizer lab, a three-domain system and traces; look-based novelty hits one wall, outcomes pass it

- **ID:** E-055
- **Date:** 2026-10-31
- **Track:** G
- **Type:** experiment
- **Status:** completed on report seeds 710-714 (5 seeds); a second batch of report seeds (715-719) was embedded in the background and is NOT used in any claim
- **Evidence grade:** B (script-generated, 5 seeds, every prediction declared before the cells that test it and scored by script; NOT a pre-registration: no hash-frozen bars; no independent reviewer; the trace results are on a synthetic world)
- **Code / commit:** not committed (standing rule); new package `fsmoe/` (domains, reader, frontend, detectors, fastslow, invariance, recog_lab, system_lab, team, ledger, traces, report and scorecard scripts), tests `tests/fsmoe/` (46 pass), design `docs/superpowers/specs/2026-10-31-fsmoe-design.md`; growmoe/ and unimoe/ untouched
- **People:** Claude designed, built and ran it on the user's request of 2026-10-31 ("build the full system, test and analyze, give me reports"); no reviewer yet

## Question
After a long discussion of routing, novelty and nuance, the user said they were stuck on a reliable, efficient recognizer of novelty and nuance for their architecture. Can the two be told apart from the input alone (a known skill in a new wording versus a genuinely new near-twin skill), in language, arithmetic and code? What does the whole system (fast-slow experts, teams, growth) do on such a stream, and what carries the distinction if the look cannot?

## Why we asked it
User request. Follows E-045 (look-based detectors tie at one wall), E-049..E-051 (calibration on verified outcomes) and E-054 (the LDA head in every expert). New here: a code domain with execution as the outside anchor, one shared pair of readers for three domains, fast-slow disagreement as a signal, and a synthetic world with real next states to test result prediction and depth growth.

## Prediction (written BEFORE the numbers)
`docs/superpowers/specs/2026-10-31-fsmoe-design.md` sections 8, 8b, 8c: 15 numeric predictions (8 for the recognizer, 4 for the system, 3 for the traces), written before the cells that test them (the recognizer ones before any cell, the system and trace ones after the dev-seed recognizer readout and before any system or trace cell). Scored by `fsmoe/scorecard.py`: **4 held, 11 missed (27 %)** (Stage 1 scored 49 %, P4 45 %).

## Method (how)
- Test bed: arithmetic (8 operations, 8 wording families), code (16 function families, 9 styles; every snippet executed on fixed probe inputs and checked against the family's reference behaviour), language (77 banking intents, 5 families). Withheld (never taught): 3 / 5 / 20 skills, near-twins and far. Far texts from other registers; for code, real standard-library functions.
- Readers: Qwen2.5-1.5B last-token state (layer 26, fixed suffix) plus MiniLM, concatenated; frontend = PCA fitted on unlabelled teaching rows only (128 numbers per reader). Chosen on dev seed 700 (`results/fsmoe/pilot_700.txt`): Qwen alone read language at 0.24 on held-out wordings, MiniLM alone read code at 0.78.
- Layer R (`fsmoe/recog_lab.py`): signals knn, class-conditional Mahalanobis, vMF density, fast-slow disagreement (JS divergence), confidence, in the raw space and a learned invariance map (contrastive, teaching rows only); fusions; group test; calibration (conformal, rank ceil((n+1)(1-q))) on the held-out teaching rows or on n = 40 known inputs of the stream's own wordings (30 draws); evaluation on rows never used for fitting or calibration.
- Layer S (`fsmoe/system_lab.py`): the validated pool `ULDA` taught arithmetic, code, language; equal-memory kNN and replay, streaming LDA, joint network, each with its own recipe tuned on dev seed 700 (`results/fsmoe/S/tuning.json`); teams; a deployment stream (known skills in all wordings, 800 withheld-skill items, 600 far texts) with a verification budget spent by policies none, random, novelty, uncertainty, oracle (and, added after the readout, oracle budget curve and a diversity-aware novelty policy).
- Layer T (`fsmoe/traces.py`): synthetic state-transition world; recognition by look vs next-state error; a stuck-then-grow rule for width and depth.
- Seeds: dev 700 (all choices); report 710-714 (R, S, T). Seeds 400-429 and 600-604 untouched.

## Results (what came out; all copied from the generated files)
- Recognizer (`results/fsmoe/R/REPORT_R.md`): calibrated on the teaching wording only, 16-96 % of known skills in held-out wordings are called new. Calibrated on the stream's own wordings the false flag is 0.04-0.08 but near-twin skills are found 0.126 (arithmetic), 0.217 (code), 0.136 (language) at best; AUROC 0.41-0.72 for nearly all signals, 0.78 at best (code, learned map); no signal or fusion meets the declared bars in any domain (R8 held). Far texts are found 0.95-1.00; far code skills 0.93 (kNN); far intents within language 0.31. Disagreement is no better than distance (R4 missed); the learned map adds 0.05-0.07 AUROC in arithmetic and code and costs far-text detection (R2 held, R3/R5/R6 missed). Kill rule fired (no AUROC 0.95 on near-twins anywhere).
- Traces (`results/fsmoe/T/REPORT_T.md`): with 6 teaching surfaces the check of every expert's next-state error has AUROC 0.920 / 0.934 / 0.805 (noise 0.05 / 0.3 / 1.0) for known-on-new-surface vs withheld regimes; the look 0.127; the look-shortlists-one-expert cascade 0.467 / 0.467 / 0.480 (chance); cost 26,112 vs 1,216 multiply-adds per input. With 2 surfaces 0.879 / 0.882 / 0.740. Depth: the declared rule (grow when the fit is worse than 20 % of the variance) never fired; a 5 % variant (added after seeing that) widens, beats larger fixed nets on the first composed map, creeps on a hard one, deepens 0.2-0.4 times per run; at equal size two blocks beat one on composed maps.
- System (`results/fsmoe/S/REPORT_S.md`): pool 0.965 taught / 0.895 new wording (3 experts, 12.9 MB, forgetting 0.000); replay 0.952 / 0.832, kNN 0.934 / 0.815, LDA 0.945 / 0.866, joint 0.961 / 0.821. Team 0.878 vs top-1 0.882, mean size 1.02 (S3 held; S1 missed only on kNN at 0.031). Deployment: novelty ranking precision 0.264 vs random 0.132 (below the 0.28 predicted) but 4.0 of 28 skills met; learned fraction 0.00 / 0.08 / 0.10 / 0.06 / 0.17 (none / random / novelty / uncertainty / oracle at 5 %); oracle at 10 / 20 / 40 % 0.31 / 0.21 / 0.23; diverse at 5 / 10 % 0.16 / 0.28 (meets 11.2 / 23.6 skills); known-skill new-wording accuracy 0.885 before, 0.63-0.83 after any policy that teaches new skills (S2, S4 missed).

## Interpretation (what we understood, and on what basis)
Plain words: from the input alone, a known skill in a new wording and a new skill that reads almost the same cannot be told apart by any of the cheap signals tried; the more sensitive the signal, the more it also flags the new wording. What separates them is whether the experts' PREDICTIONS match what happens next, so novelty is best treated as act, compare with the outcome, then learn, with the look used for far novelty and for ranking what to check. Technical: calibrating on verified outcomes of the stream's own wording is what keeps false flags near q (measured); the power ceiling of look-based scores is about 0.13-0.22 at q = 0.05 (measured, 5 seeds); the result signal needs experts taught on varied surfaces and costs one expert pass each (measured on a synthetic world only). The mechanism behind the loss of known-skill new-wording accuracy after learning new skills is a guess (the near-twin stealing hypothesis was not measured).

## Caveats and what went wrong
- The trace world is synthetic and its surfaces are additive nuisance features; real code traces are untested.
- Seeds differ a lot in the deployment cells (learned fraction intervals wide, e.g. oracle at 5 % 0.17 [-0.11, 0.45]); the diverse policy and the oracle curve were added after the main readout (exploratory); the 5 % depth variant and the two interaction cases were added after the first depth run.
- The recognizer lab ran on 5 of the 10 planned report seeds; the others were still embedding.
- I ran two Qwen jobs at once and filled the 12 GB GPU; my request to stop the duplicate was denied, both finished (F-68).
- My first draft of the plain-English report had four wrong or unmeasured statements; each was caught by checking it against the generated tables before delivery (F-69).
- Not built: the trust ledger inside the pool (unit-tested only), right-sizing at birth, a shared trunk, system-level vertical growth, a token-level stage.

## Consequence
Report to the user: `research/FSMOE_REPORT_2026-10-31.md`. Open: Q-45..Q-49. Decision D-44..D-46.

## Evidence
- `research/FSMOE_REPORT_2026-10-31.md` - the plain-English report
- `results/fsmoe/R/REPORT_R.md`, `results/fsmoe/S/REPORT_S.md`, `results/fsmoe/T/REPORT_T.md` - generated tables
- `results/fsmoe/SCORECARD.md` - the prediction scorecard
- `results/fsmoe/pilot_700.txt` - the frontend pilot (dev seed)
- `results/fsmoe/S/tuning.json` - per-arm recipes (dev seed)
- `docs/superpowers/specs/2026-10-31-fsmoe-design.md` - design and predictions

## Reproduce
```
python -m fsmoe.embed_bed --seeds 700 710 711 712 713 714
python -m fsmoe.pilot --seed 700
python -m fsmoe.system_lab tune
python -m fsmoe.recog_lab run --seeds 710 711 712 713 714
python -m fsmoe.system_lab run --seeds 710 711 712 713 714
python -m fsmoe.system_lab extra --seeds 710 711 712 713 714
python -m fsmoe.traces recognise --seeds 710 711 712 713 714
python -m fsmoe.traces depth --seeds 710 711 712 713 714
python -m fsmoe.report_r --seeds 710 711 712 713 714
python -m fsmoe.report_s --seeds 710 711 712 713 714
python -m fsmoe.report_t --seeds 710 711 712 713 714
python -m fsmoe.scorecard --r-seeds 710 711 712 713 714
python -m pytest tests/fsmoe
```
