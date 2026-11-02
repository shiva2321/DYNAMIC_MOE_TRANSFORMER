# E-064 - Nuance detection: the nuance lab had already settled most of the ideas; a character-aware front end does not help (stories are the failure)

- **ID:** E-064
- **Date:** 2026-11-01
- **Track:** G
- **Type:** experiment (plus a reconciliation of the log)
- **Status:** completed on the tuning seeds; nothing run on the report seeds or the single-use seeds
- **Evidence grade:** B (script-generated on the tuning seeds 150-154, predictions written before the run; exploratory, NOT a pre-registration, no independent reviewer)
- **Code / commit:** not committed (standing rule); new `research/knowledge_status/char_front_check.py` (reuses `run_e2.build_world`, `baselines.py`, `evaluate.py`, `reps.py`; edits nothing of the lab); results `research/knowledge_status/results/char_front/`; design and predictions `docs/superpowers/specs/2026-11-01-nuance-reconciliation-and-char-front-design.md`; report `research/NUANCE_RECONCILIATION_REPORT_2026-11-01.md`
- **People:** Claude built and ran it on the user's go of 2026-11-01 ("Yes, start", then "do what you can ... you will have freedom to work freely"); no reviewer yet

## Question
The user proposed five ideas for a better nuance detector in chat (chunk vectors, a typo-aware front end, a stability test, asking once per cluster, predictive coding; earlier also wave or phase vectors) and asked whether such a mechanism can be built now. What does the nuance lab's own record already say about each, and does the one untested cheap idea (a character-aware front end) help?

## Why we asked it
The standing rule is to check the log before calling an idea untested. I proposed these ideas in chat before doing that; the user then asked me to start.

## Prediction (written BEFORE the run, in the design note)
N1 to N5 (spelling-only reader: false unseen at most 0.20 and found at most 0.55; MiniLM: false unseen 0.20-0.50 and found at least 0.75; the 0.5-weight mix lowers noisy-text false unseen by 0.10 for two of three detectors; nothing reaches 0.05 with 0.85 found; neither mix meets the lab's go-on rule). **2 of 5 held** (N4, N5).

## Method (how)
The lab's own worlds for tuning seeds 150-154 (pool `dev`: two Solid, one Thin, three never-taught skills per seed; teaching in wordings S1-S3, held-out calibration in S1-S3, tests in S4-S8), its three look-based detectors (vMF conformal, 1-NN, Mahalanobis, `BASELINE_PARAMS`), calibration at q = 0.05; four readers: R1_minilm384, R0_hash, and [R1, w x R0] with w = 0.5, 1.0 (each part centred on the taught examples and unit length first). The log reconciliation is a reading of `PROGRESS.md`, the E2 tables and the G1-G4c result files.

## Results (copied from `research/knowledge_status/results/char_front/TABLE_150_154.md`)
False unseen on S4-S8 (vMF / 1-NN / Mahalanobis), stories S4, noisy text S7, unseen skills found:
- R1: 0.342 / 0.329 / 0.223; S4 0.978 / 0.993 / 0.801; S7 0.515 / 0.562 / 0.240; found 0.555 / 0.260 / 0.127.
- R0: 0.121 / 0.221 / 0.221; S4 0.099 / 0.585 / 0.599; S7 0.263 / 0.453 / 0.398; found 0.359 / 0.313 / 0.257.
- R1 + 0.5 R0: 0.297 / 0.327 / 0.269; S4 0.830 / 0.993 / 0.890; S7 0.441 / 0.548 / 0.347; found 0.563 / 0.282 / 0.150.
- R1 + 1.0 R0: 0.332 / 0.317 / 0.274; S4 0.906 / 0.983 / 0.890; S7 0.485 / 0.477 / 0.367; found 0.682 / 0.332 / 0.153.
Reconciliation: wording diversity in teaching, projecting out shift directions, margins, clean-up, prompts, leave-one-wording-out, the function check on single answers, group evidence and a rewrite-based trust diagnostic are all on record as failed or partial; the asking reader closes the arithmetic wording shift (0.025 false-new, 0.974 found on dev seeds 100-109) but loses on topical novelty; no fixed fusion works in every domain.

## Interpretation (what we understood, and on what basis)
Plain words: a spelling view does not help because the dominant failure is stories, not spelling; the meaning model flags nearly every known story (0.80 to 0.99), and the lab's story is the whole word problem with the operation implicit in it, so there is nothing to chunk away. Technical: measured: the table above; read from the log: the record of the other ideas. Not tested: predictive-coding surprise as such (the autoencoder descriptor of B3b is the nearest proxy and ties the others), wave or phase vectors, chunking on a bed with filler around the request.

## Caveats and what went wrong
- I proposed two ideas (wording diversity, group asking) that were already on record as failures, and said stories "bury the request in long text", which does not fit this benchmark (F-93, F-94).
- The first run of the script failed on a missing path (`growmoe` not importable from the lab folder): fixed by running with the repo on `PYTHONPATH`; no result was affected.
- Tuning seeds only, arithmetic requests only, three look-based detectors only, one calibration level.

## Consequence
Report `research/NUANCE_RECONCILIATION_REPORT_2026-11-01.md`. The nuance track's open items are the ones the lab itself named: wiring the two-reader gate into the pool behind a flag, distilling the two language-model questions into one head, and a vocabulary-free reader. Claims C-179, C-180; failures F-93 to F-95; decision D-62; question Q-69.

## Evidence
- `research/NUANCE_RECONCILIATION_REPORT_2026-11-01.md`
- `research/knowledge_status/results/char_front/TABLE_150_154.md`
- `docs/superpowers/specs/2026-11-01-nuance-reconciliation-and-char-front-design.md`
- `research/knowledge_status/PROGRESS.md` (the log read), `research/knowledge_status/results/E2_TABLE.md`

## Reproduce
```
cd research/knowledge_status
set PYTHONPATH=D:/spherical_hyperspace_memory;D:/spherical_hyperspace_memory/research/knowledge_status
python char_front_check.py --seeds 150-154
```
