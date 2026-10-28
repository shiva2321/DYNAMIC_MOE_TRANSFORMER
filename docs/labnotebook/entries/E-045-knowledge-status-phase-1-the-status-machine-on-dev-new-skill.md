# E-045 - Knowledge status Phase 1 - the status machine on dev: new skills are found, known skills in new wordings are not trusted

- **ID:** E-045
- **Date:** 2026-10-27
- **Track:** G
- **Type:** experiment
- **Status:** completed
- **Evidence grade:** B (script-generated, 30 dev seeds, three readers, predictions and the rule for the bars written before the run; dev seeds, so exploratory, not a frozen pre-registration)
- **Code / commit:** branch `knowledge-status-phase1`; a2e2124 (status machine and baselines), 11a2282 (segmentation, evaluation, runner, rules and prediction), 0e2c045 (baseline settings from the tuning seeds), 9ea7a94 (results and diagnosis); worktree D:/shm_ks
- **People:** Claude designed, built and reviewed (no independent reviewer yet); the user decided at Stop Point 2 (2026-10-28) to look for the cause and try fixes

## Question
With the learned-map reader of E-044, can a status machine (solid, thin, unseen, other) whose alarm is calibrated on held-out data of the TAUGHT wordings only (S1-S3) find skills that were never taught without raising false alarms on known skills in wordings it never saw (S4-S8)? How does it compare with the seven standard detectors held to the same calibration?

## Why we asked it
The pool's two costly errors are novelty-alarm errors (E-034, E-030), and E-044 showed that a reader exists whose near-twin ordering is right. The calibration claim (spec 6.2): at q = 0.05 the false-unseen rate on S4-S8 must be at most 0.05.

## Prediction (written BEFORE the numbers)
Copied from `research/knowledge_status/PROGRESS.md` by script (rules for the bars, the baselines' tuning, and the prediction):

## 2026-10-27, Task 11 (before any number): what is built, the rules I fix now, and my prediction

**What we built (all tested; no real number yet).** `segment.py` cuts a message into parts (the answer key's own parts; sentences and clauses; sliding windows of 8 words). `evaluate.py` scores: the status table (macro F1 over solid / thin / unseen, the confusion counts, false "unseen" on new wordings, Other content wrongly called unseen), the same numbers for plain flag-or-not detectors, which skill a part is nearest to, the two-part messages (B4a, B4b, B6), the wording claim and the labelled cost estimate. `run_e2.py` runs it all: per seed one teaching assignment (two skills taught 2,000 examples, one taught 20, three never taught), a measured competence table, teaching, calibration, tests, near-twins, traps, everyday sentences, never-taught content and 120 two-part messages; for every reader the status machine at q = 0.01, 0.02, 0.05, 0.10 in the STANDARD condition (calibration from held-out S1-S3 parts only) and the ADAPTED one (a few S4-S8 parts in the calibration; reported next to it, never gated), the seven standard detectors at the same calibration, and group tests of 1, 4, 8 and 16 parts. Readers: the learned map on Qwen layer 26 (the one you chose), plain Qwen layer 26 (no map) and the hashed word-matcher (the control that must do badly).

**Rules fixed now, before looking (so they cannot be bent by the results).**
1. **Bars.** The draft bars of the spec stay the reading of PASS / fail. For the freeze, each bar is re-set from the dev seeds by one rule: a bar to stay ABOVE (macro F1, AUROC, the B4 rates) becomes max(draft bar, dev mean - 2 sd over seeds); a bar to stay BELOW (false-unseen rates) becomes min(draft bar, dev mean + 2 sd). So a dev result can tighten a draft bar and can never make it easier. The dev values come from the learned-map reader, status machine, q = 0.05, standard condition.
2. **Solid / Thin / grey thresholds (`THRESHOLDS`) stay at the draft values** (Solid = at least 500 examples and accuracy 0.90; Thin = at most 100 examples, or accuracy between chance + 0.10 and 0.60). They are defined on the measured competence table of Task 7, not on the status results, so there is nothing to tune them against. If the competence table shows that a Solid skill (2,000 examples) lands in the grey zone in more than a quarter of the seeds, I report it at the stop and propose a change; I do not change it silently.
3. **The headline level is q = 0.05.** The curve over 0.01, 0.02, 0.05, 0.10 is shown; no level is picked afterwards.
4. **The standard detectors are tuned on other seeds (150-154), one setting per detector,** by the highest mean AUROC (taught against untaught parts), before the report seeds 100-129 are run; the chosen settings are written next to the results. The status machine has no tuned setting (q is fixed; the kernel width is chosen by its own leave-one-out likelihood).
5. **Calibration is held-out data of the taught groups in S1 to S3 only** (120 parts per group, none of them a teaching text); a Thin skill still gets 120 calibration parts, which is more than its 20 teaching examples: a declared simplification, because a conformal test needs at least 19 (q = 0.05) or 99 (q = 0.01) calibration parts.
6. **Seeds.** The map was tuned on seeds 105-107 (Task 9) and these lie inside the report seeds 100-129; the map's settings were insensitive (12 variants within 0.01), but I note the overlap here.

**Prediction (written before the dev run; learned-map reader, status machine, q = 0.05, standard condition, mean over seeds).**
- False "unseen" on the new wordings S4-S8: **0.08 to 0.25** (bar 0.05, so I expect it to fail); on the taught wordings S1-S3: 0.03 to 0.08. If this is right, the wording-invariance claim becomes "needs adaptation to new wording" or "fails".
- Untaught skills found as unseen: **0.35 to 0.65** (the taught-against-untaught AUROC of the reader was 0.89, and division and remainder look like each other).
- Macro F1 over solid / thin / unseen: **0.60 to 0.80** (bar 0.85, I expect it to fail).
- Thin separated from Solid (AUROC): **0.95 or more** (the number of taught examples differs by a factor of 100; this is almost by construction).
- Other content called unseen: **0.04 to 0.12**.
- Adapted condition: false unseen falls toward **0.05 to 0.10**.
- Plain Qwen layer 26 (no map): clearly worse than the learned map on false unseen (0.20 to 0.45). The hashed word-matcher: worst, false unseen above 0.30.
- Standard detectors: none beats the status machine by more than 0.05 on both false unseen and unseen found at the same calibration; kNN and Mahalanobis are the strongest. The pool's own gate (vMF) is the weakest on the Thin skill.
- Two-part messages (gold parts): B4a leftover is the unseen part **0.45 to 0.75**, known part not flagged **0.80 to 0.93**; B4b nothing flagged **0.75 to 0.90**, Other labelled **0.85 to 0.95**. Automatic cutting: within 0.10 of the gold number for sentence/clause pieces, not for sliding windows.
- Group tests: unseen skills found rises with the batch size (0.85 or more at 16); false unseen for batches of taught skills stays at or below 0.15.

## Method (how)
- One SEED is one teaching assignment: two skills taught about 2,000 examples (Solid), one taught 20 (Thin), three never taught; a measured competence table (a small solver per taught skill, scored on problems it never saw) turns the count and accuracy into the status truth (solid / thin / grey, by `assignments.status_truth`); grey cases are reported and never scored.
- The unit is a PART (the request span). Teaching, calibration (120 held-out S1-S3 parts per group, none a teaching text; 80 + 40 for Other content), adaptation (a few S4-S8 parts, a separately reported condition that never gates) and tests are all parts. Tests: held-out wordings S4-S8 and fresh taught-wording parts of all six skills, near-twins, traps with test-only templates, everyday sentences, never-taught content (science questions), 120 two-part messages (known + unseen, known + other, unseen + other).
- Machine: one von Mises-Fisher signature per group (the expert pool's own `VmfSignature`), conformal p-values against the group's HELD-OUT calibration scores; a part is unseen only if EVERY group flags it (p <= q); otherwise it goes to the group with the nearest prototype; Solid or Thin by the status-truth rule on count and competence. q = 0.01, 0.02, 0.05 (headline), 0.10.
- Readers: R5_R4_qwen_L26 (the learned map on Qwen layer 26, the headline, user decision 2026-10-27), R4_qwen_L26 (plain, centred) and R0_hash (the word-matcher control). Vectors are centred on the teaching mean and made unit length.
- Baselines (same data, same shared calibration, threshold from the same held-out parts): the pool's vMF gate, kNN distance, Mahalanobis, Relative Mahalanobis, classifier confidence, energy, autoencoder descriptor; their one setting each was chosen on the tuning seeds 150-154 by the AUROC taught-vs-untaught (`research/knowledge_status/results/BASELINE_TUNING.md`). Group tests of 1, 4, 8 and 16 parts use the permutation test of `growmoe.settest`.
- Seeds 100-129 (30 seeds, 90 cells); 150-154 for tuning only. Bars: the draft bars of the spec; a bar can be tightened from the dev seeds by `fix_bar` (max(draft, mean - 2 sd) to stay above, min(draft, mean + 2 sd) to stay below) and never loosened.
- Command: `PYTHONPATH=<repo root> python run_e2.py --pool dev --seeds 100-129 --reps R5_R4_qwen_L26,R4_qwen_L26,R0_hash` (run from `research/knowledge_status`; about 85 s per cell).

## Results (what came out)
All tables are copied by script from `research/knowledge_status/results/E2_TABLE.md` (generated by `run_e2.py`).

## B3a, the status machine at q = 0.05 (calibration on S1 to S3 only)

| reader | seeds | macro F1 over solid/thin/unseen (bar 0.85) | false unseen on S4-S8 (bar 0.05) | thin vs solid AUROC (bar 0.85) | Other content called unseen (bar 0.05) | unseen skills found |
|---|---|---|---|---|---|---|
| R5_R4_qwen_L26 | 30 | 0.85 [0.52 - 0.92] PASS | 0.29 [0.17 - 0.43] fail | 0.84 [0.59 - 0.98] fail | 0.25 [0.11 - 0.36] fail | 0.89 [0.18 - 1.00] |
| R4_qwen_L26 | 30 | 0.82 [0.70 - 0.88] fail | 0.39 [0.29 - 0.48] fail | 0.89 [0.76 - 0.97] PASS | 0.21 [0.09 - 0.33] fail | 0.90 [0.60 - 0.98] |
| R0_hash | 30 | 0.61 [0.39 - 0.76] fail | 0.12 [0.04 - 0.24] fail | 0.88 [0.73 - 0.98] PASS | 0.16 [0.00 - 0.27] fail | 0.36 [0.10 - 0.60] |

### The adapted condition at q = 0.05 (reported, not gated) and the wording claim

| reader | false unseen on S4-S8, standard | false unseen on S4-S8, adapted | false unseen on taught wordings S1-S3 | claim from the means (bar 0.05) |
|---|---|---|---|---|
| R5_R4_qwen_L26 | 0.29 [0.17 - 0.43] | 0.17 [0.09 - 0.29] | 0.05 [0.01 - 0.12] | fails |
| R4_qwen_L26 | 0.39 [0.29 - 0.48] | 0.17 [0.07 - 0.30] | 0.05 [0.00 - 0.12] | fails |
| R0_hash | 0.12 [0.04 - 0.24] | 0.02 [0.00 - 0.12] | 0.03 [0.00 - 0.13] | needs adaptation to new wording |

### The curve over q (standard condition)

| reader | q | false unseen on S4-S8 | unseen skills found | Other content called unseen | near-twins of taught skills called unseen | near-twins of untaught skills found |
|---|---|---|---|---|---|---|
| R5_R4_qwen_L26 | 0.01 | 0.12 [0.00 - 0.30] | 0.56 [0.07 - 1.00] | 0.11 [0.00 - 0.29] | 0.01 [0.00 - 0.09] | 0.62 [0.00 - 1.00] |
| R5_R4_qwen_L26 | 0.02 | 0.19 [0.04 - 0.37] | 0.75 [0.10 - 1.00] | 0.19 [0.01 - 0.31] | 0.02 [0.00 - 0.13] | 0.80 [0.09 - 1.00] |
| R5_R4_qwen_L26 | 0.05 | 0.29 [0.17 - 0.43] | 0.89 [0.18 - 1.00] | 0.25 [0.11 - 0.36] | 0.05 [0.00 - 0.16] | 0.93 [0.22 - 1.00] |
| R5_R4_qwen_L26 | 0.1 | 0.36 [0.20 - 0.54] | 0.95 [0.35 - 1.00] | 0.29 [0.17 - 0.43] | 0.09 [0.00 - 0.25] | 0.97 [0.34 - 1.00] |
| R4_qwen_L26 | 0.01 | 0.27 [0.11 - 0.41] | 0.72 [0.07 - 0.97] | 0.05 [0.00 - 0.16] | 0.01 [0.00 - 0.12] | 0.64 [0.00 - 0.90] |
| R4_qwen_L26 | 0.02 | 0.31 [0.24 - 0.42] | 0.81 [0.45 - 0.97] | 0.09 [0.00 - 0.22] | 0.02 [0.00 - 0.12] | 0.75 [0.28 - 0.95] |
| R4_qwen_L26 | 0.05 | 0.39 [0.29 - 0.48] | 0.90 [0.60 - 0.98] | 0.21 [0.09 - 0.33] | 0.04 [0.00 - 0.12] | 0.86 [0.66 - 1.00] |
| R4_qwen_L26 | 0.1 | 0.44 [0.31 - 0.57] | 0.94 [0.81 - 1.00] | 0.33 [0.17 - 0.48] | 0.09 [0.00 - 0.21] | 0.92 [0.77 - 1.00] |
| R0_hash | 0.01 | 0.02 [0.00 - 0.07] | 0.11 [0.00 - 0.31] | 0.01 [0.00 - 0.07] | 0.01 [0.00 - 0.09] | 0.11 [0.00 - 0.37] |
| R0_hash | 0.02 | 0.04 [0.00 - 0.14] | 0.19 [0.01 - 0.40] | 0.04 [0.00 - 0.17] | 0.02 [0.00 - 0.19] | 0.19 [0.00 - 0.49] |
| R0_hash | 0.05 | 0.12 [0.04 - 0.24] | 0.36 [0.10 - 0.60] | 0.16 [0.00 - 0.27] | 0.06 [0.00 - 0.21] | 0.39 [0.00 - 0.78] |
| R0_hash | 0.1 | 0.22 [0.09 - 0.39] | 0.57 [0.20 - 0.81] | 0.32 [0.06 - 0.48] | 0.13 [0.00 - 0.38] | 0.62 [0.15 - 0.92] |

## B3b, the baselines at the same calibration (q = 0.05, standard condition)

Each baseline flags a part as unseen when its novelty score is above the threshold taken from the same held-out calibration parts. AUROC = parts of untaught skills against parts of taught skills, no threshold.

| reader | method | false unseen on S4-S8 | unseen skills found | AUROC taught vs untaught | Other content called unseen | false unseen, adapted |
|---|---|---|---|---|---|---|
| R5_R4_qwen_L26 | **status machine** | 0.29 [0.17 - 0.43] | 0.89 [0.18 - 1.00] | n/a | 0.25 [0.11 - 0.36] | 0.17 [0.09 - 0.29] |
| R5_R4_qwen_L26 | vmf_conformal | 0.31 [0.11 - 0.47] | 0.95 [0.75 - 1.00] | 0.95 [0.88 - 0.99] | 0.23 [0.10 - 0.36] | 0.19 [0.07 - 0.38] |
| R5_R4_qwen_L26 | knn_distance | 0.33 [0.14 - 0.50] | 0.96 [0.73 - 1.00] | 0.94 [0.87 - 0.99] | 0.22 [0.10 - 0.37] | 0.20 [0.07 - 0.35] |
| R5_R4_qwen_L26 | mahalanobis | 0.33 [0.15 - 0.50] | 0.97 [0.79 - 1.00] | 0.95 [0.89 - 0.99] | 0.27 [0.14 - 0.38] | 0.22 [0.11 - 0.39] |
| R5_R4_qwen_L26 | relative_mahalanobis | 0.01 [0.00 - 0.09] | 0.06 [0.00 - 0.46] | 0.44 [0.10 - 0.95] | 0.37 [0.14 - 0.55] | 0.01 [0.00 - 0.09] |
| R5_R4_qwen_L26 | classifier_confidence | 0.19 [0.08 - 0.31] | 0.51 [0.22 - 0.74] | 0.77 [0.57 - 0.88] | 0.06 [0.00 - 0.21] | 0.15 [0.06 - 0.24] |
| R5_R4_qwen_L26 | energy | 0.13 [0.00 - 0.28] | 0.41 [0.00 - 0.85] | 0.74 [0.41 - 0.93] | 0.10 [0.00 - 0.33] | 0.11 [0.00 - 0.23] |
| R5_R4_qwen_L26 | autoencoder_descriptor | 0.28 [0.06 - 0.47] | 0.88 [0.64 - 1.00] | 0.92 [0.79 - 0.98] | 0.17 [0.04 - 0.26] | 0.19 [0.03 - 0.38] |
| R4_qwen_L26 | **status machine** | 0.39 [0.29 - 0.48] | 0.90 [0.60 - 0.98] | n/a | 0.21 [0.09 - 0.33] | 0.17 [0.07 - 0.30] |
| R4_qwen_L26 | vmf_conformal | 0.31 [0.06 - 0.44] | 0.81 [0.41 - 0.95] | 0.89 [0.83 - 0.94] | 0.16 [0.00 - 0.30] | 0.21 [0.06 - 0.38] |
| R4_qwen_L26 | knn_distance | 0.41 [0.30 - 0.49] | 0.93 [0.86 - 0.99] | 0.90 [0.85 - 0.94] | 0.16 [0.00 - 0.28] | 0.25 [0.15 - 0.38] |
| R4_qwen_L26 | mahalanobis | 0.39 [0.29 - 0.47] | 0.92 [0.82 - 0.97] | 0.88 [0.82 - 0.93] | 0.25 [0.06 - 0.37] | 0.27 [0.16 - 0.35] |
| R4_qwen_L26 | relative_mahalanobis | 0.00 [0.00 - 0.01] | 0.00 [0.00 - 0.00] | 0.72 [0.45 - 0.90] | 0.28 [0.13 - 0.39] | 0.00 [0.00 - 0.01] |
| R4_qwen_L26 | classifier_confidence | 0.30 [0.17 - 0.45] | 0.91 [0.73 - 1.00] | 0.91 [0.84 - 0.97] | 0.21 [0.05 - 0.37] | 0.19 [0.10 - 0.31] |
| R4_qwen_L26 | energy | 0.24 [0.06 - 0.41] | 0.87 [0.60 - 1.00] | 0.92 [0.85 - 0.98] | 0.22 [0.08 - 0.38] | 0.17 [0.04 - 0.28] |
| R4_qwen_L26 | autoencoder_descriptor | 0.27 [0.15 - 0.46] | 0.85 [0.65 - 0.92] | 0.89 [0.85 - 0.93] | 0.06 [0.00 - 0.15] | 0.18 [0.09 - 0.34] |
| R0_hash | **status machine** | 0.12 [0.04 - 0.24] | 0.36 [0.10 - 0.60] | n/a | 0.16 [0.00 - 0.27] | 0.02 [0.00 - 0.12] |
| R0_hash | vmf_conformal | 0.12 [0.03 - 0.30] | 0.35 [0.15 - 0.56] | 0.82 [0.76 - 0.88] | 0.22 [0.05 - 0.47] | 0.09 [0.01 - 0.17] |
| R0_hash | knn_distance | 0.20 [0.05 - 0.32] | 0.35 [0.13 - 0.45] | 0.81 [0.75 - 0.87] | 0.46 [0.25 - 0.61] | 0.13 [0.04 - 0.22] |
| R0_hash | mahalanobis | 0.16 [0.07 - 0.30] | 0.24 [0.10 - 0.42] | 0.77 [0.69 - 0.85] | 0.38 [0.21 - 0.52] | 0.11 [0.03 - 0.17] |
| R0_hash | relative_mahalanobis | 0.10 [0.03 - 0.19] | 0.11 [0.05 - 0.20] | 0.77 [0.69 - 0.84] | 0.36 [0.16 - 0.53] | 0.08 [0.03 - 0.15] |
| R0_hash | classifier_confidence | 0.10 [0.03 - 0.31] | 0.25 [0.13 - 0.47] | 0.71 [0.61 - 0.81] | 0.04 [0.00 - 0.14] | 0.09 [0.03 - 0.20] |
| R0_hash | energy | 0.08 [0.01 - 0.29] | 0.31 [0.16 - 0.51] | 0.72 [0.62 - 0.83] | 0.07 [0.01 - 0.12] | 0.07 [0.01 - 0.25] |
| R0_hash | autoencoder_descriptor | 0.09 [0.03 - 0.15] | 0.19 [0.04 - 0.42] | 0.71 [0.60 - 0.86] | 0.00 [0.00 - 0.03] | 0.08 [0.03 - 0.15] |

## E3, tangled messages (status machine, q = 0.05)

B4a: a known part with an unseen part; B4b: a known part with Other content; B4c: an unseen part with Other content. B6: B4a under automatic segmentation must stay within 0.10 of the gold-span number.

| reader | segmentation | B4a leftover is the unseen part (bar 0.85) | B4a known part not flagged (bar 0.95) | B4b nothing flagged (bar 0.95) | B4b Other labelled (bar 0.95) | B4c unseen found | B6 within 0.10 |
|---|---|---|---|---|---|---|---|
| R5_R4_qwen_L26 | gold | 0.66 [0.15 - 0.85] fail | 0.67 [0.50 - 0.85] fail | 0.64 [0.47 - 0.78] fail | 0.97 [0.90 - 1.00] PASS | 0.86 [0.12 - 1.00] | n/a |
| R5_R4_qwen_L26 | auto_sentence | 0.49 [0.12 - 0.68] | 0.58 [0.35 - 0.88] | 0.53 [0.25 - 0.72] | 0.93 [0.85 - 1.00] | 0.84 [0.15 - 1.00] | 0.13 of seeds |
| R5_R4_qwen_L26 | auto_window | 0.37 [0.15 - 0.55] | 0.53 [0.25 - 0.82] | 0.23 [0.10 - 0.60] | 0.57 [0.28 - 0.80] | 0.85 [0.23 - 1.00] | 0.07 of seeds |
| R4_qwen_L26 | gold | 0.67 [0.47 - 0.80] fail | 0.51 [0.28 - 0.72] fail | 0.45 [0.20 - 0.60] fail | 0.91 [0.68 - 1.00] fail | 0.96 [0.68 - 1.00] | n/a |
| R4_qwen_L26 | auto_sentence | 0.47 [0.33 - 0.68] | 0.45 [0.28 - 0.65] | 0.32 [0.10 - 0.55] | 0.88 [0.60 - 0.97] | 0.94 [0.78 - 1.00] | 0.03 of seeds |
| R4_qwen_L26 | auto_window | 0.36 [0.23 - 0.50] | 0.42 [0.15 - 0.62] | 0.08 [0.00 - 0.25] | 0.51 [0.23 - 0.70] | 0.94 [0.78 - 1.00] | 0.03 of seeds |
| R0_hash | gold | 0.35 [0.07 - 0.55] fail | 0.92 [0.80 - 1.00] fail | 0.82 [0.62 - 0.97] fail | 0.92 [0.78 - 1.00] fail | 0.37 [0.12 - 0.78] | n/a |
| R0_hash | auto_sentence | 0.27 [0.10 - 0.50] | 0.83 [0.60 - 1.00] | 0.69 [0.38 - 0.93] | 0.92 [0.75 - 1.00] | 0.39 [0.10 - 0.70] | 0.50 of seeds |
| R0_hash | auto_window | 0.24 [0.03 - 0.45] | 0.75 [0.55 - 0.95] | 0.36 [0.12 - 0.75] | 0.57 [0.30 - 0.88] | 0.47 [0.15 - 0.82] | 0.10 of seeds |

Baselines on the tangled messages (gold parts; they cannot label Other content, so only the flagging numbers are shown):

| reader | method | B4a leftover is the unseen part | B4a known part not flagged | B4b nothing flagged |
|---|---|---|---|---|
| R5_R4_qwen_L26 | vmf_conformal | 0.71 [0.53 - 0.88] | 0.66 [0.38 - 0.90] | 0.63 [0.40 - 0.93] |
| R5_R4_qwen_L26 | knn_distance | 0.73 [0.55 - 0.90] | 0.66 [0.40 - 0.85] | 0.62 [0.42 - 0.88] |
| R5_R4_qwen_L26 | mahalanobis | 0.74 [0.55 - 0.90] | 0.66 [0.40 - 0.85] | 0.61 [0.38 - 0.85] |
| R5_R4_qwen_L26 | relative_mahalanobis | 0.07 [0.00 - 0.80] | 1.00 [1.00 - 1.00] | 0.70 [0.47 - 0.85] |
| R5_R4_qwen_L26 | classifier_confidence | 0.47 [0.23 - 0.78] | 0.83 [0.57 - 1.00] | 0.78 [0.53 - 0.97] |
| R5_R4_qwen_L26 | energy | 0.40 [0.00 - 0.82] | 0.88 [0.62 - 1.00] | 0.80 [0.57 - 0.97] |
| R5_R4_qwen_L26 | autoencoder_descriptor | 0.69 [0.40 - 0.95] | 0.70 [0.45 - 1.00] | 0.68 [0.38 - 0.97] |
| R4_qwen_L26 | vmf_conformal | 0.67 [0.47 - 0.80] | 0.62 [0.38 - 1.00] | 0.56 [0.33 - 0.80] |
| R4_qwen_L26 | knn_distance | 0.68 [0.57 - 0.88] | 0.50 [0.30 - 0.75] | 0.45 [0.25 - 0.68] |
| R4_qwen_L26 | mahalanobis | 0.70 [0.55 - 0.88] | 0.55 [0.38 - 0.78] | 0.42 [0.12 - 0.65] |
| R4_qwen_L26 | relative_mahalanobis | 0.00 [0.00 - 0.00] | 1.00 [1.00 - 1.00] | 0.68 [0.45 - 0.88] |
| R4_qwen_L26 | classifier_confidence | 0.74 [0.55 - 0.88] | 0.73 [0.53 - 0.90] | 0.70 [0.47 - 0.93] |
| R4_qwen_L26 | energy | 0.77 [0.47 - 0.93] | 0.81 [0.53 - 0.95] | 0.77 [0.50 - 0.97] |
| R4_qwen_L26 | autoencoder_descriptor | 0.70 [0.45 - 0.90] | 0.69 [0.53 - 0.90] | 0.69 [0.50 - 0.82] |
| R0_hash | vmf_conformal | 0.33 [0.12 - 0.55] | 0.90 [0.65 - 1.00] | 0.81 [0.55 - 0.95] |
| R0_hash | knn_distance | 0.39 [0.17 - 0.53] | 0.81 [0.65 - 0.95] | 0.59 [0.28 - 0.90] |
| R0_hash | mahalanobis | 0.26 [0.07 - 0.50] | 0.86 [0.60 - 1.00] | 0.63 [0.38 - 0.85] |
| R0_hash | relative_mahalanobis | 0.13 [0.00 - 0.28] | 0.91 [0.72 - 1.00] | 0.62 [0.30 - 0.80] |
| R0_hash | classifier_confidence | 0.19 [0.03 - 0.38] | 0.91 [0.70 - 0.97] | 0.89 [0.75 - 0.95] |
| R0_hash | energy | 0.20 [0.05 - 0.38] | 0.91 [0.72 - 1.00] | 0.92 [0.85 - 1.00] |
| R0_hash | autoencoder_descriptor | 0.19 [0.05 - 0.53] | 0.92 [0.72 - 1.00] | 0.92 [0.80 - 1.00] |

## Group tests (batches of parts of one skill judged together, q = 0.05)

| reader | method | batch size | false unseen (taught skills) | unseen skills found |
|---|---|---|---|---|
| R5_R4_qwen_L26 | status machine (every group must flag) | 1 | 0.14 [0.00 - 0.50] | 0.45 [0.00 - 1.00] |
| R5_R4_qwen_L26 | status machine (every group must flag) | 4 | 0.54 [0.30 - 0.85] | 0.98 [0.70 - 1.00] |
| R5_R4_qwen_L26 | status machine (every group must flag) | 8 | 0.70 [0.40 - 0.95] | 0.98 [0.80 - 1.00] |
| R5_R4_qwen_L26 | status machine (every group must flag) | 16 | 0.78 [0.40 - 1.00] | 0.99 [0.85 - 1.00] |
| R5_R4_qwen_L26 | group test, nearest group only | 1 | 0.29 [0.05 - 0.65] | 0.85 [0.30 - 1.00] |
| R5_R4_qwen_L26 | group test, nearest group only | 4 | 0.56 [0.30 - 0.85] | 0.97 [0.70 - 1.00] |
| R5_R4_qwen_L26 | group test, nearest group only | 8 | 0.70 [0.50 - 0.90] | 0.99 [0.85 - 1.00] |
| R5_R4_qwen_L26 | group test, nearest group only | 16 | 0.79 [0.40 - 1.00] | 0.99 [0.85 - 1.00] |
| R4_qwen_L26 | status machine (every group must flag) | 1 | 0.02 [0.00 - 0.10] | 0.10 [0.00 - 0.60] |
| R4_qwen_L26 | status machine (every group must flag) | 4 | 0.25 [0.10 - 0.50] | 0.98 [0.70 - 1.00] |
| R4_qwen_L26 | status machine (every group must flag) | 8 | 0.38 [0.20 - 0.65] | 1.00 [1.00 - 1.00] |
| R4_qwen_L26 | status machine (every group must flag) | 16 | 0.54 [0.20 - 0.95] | 1.00 [1.00 - 1.00] |
| R4_qwen_L26 | group test, nearest group only | 1 | 0.09 [0.00 - 0.30] | 0.45 [0.05 - 0.70] |
| R4_qwen_L26 | group test, nearest group only | 4 | 0.27 [0.05 - 0.45] | 0.98 [0.70 - 1.00] |
| R4_qwen_L26 | group test, nearest group only | 8 | 0.41 [0.25 - 0.75] | 1.00 [1.00 - 1.00] |
| R4_qwen_L26 | group test, nearest group only | 16 | 0.55 [0.20 - 0.95] | 1.00 [1.00 - 1.00] |
| R0_hash | status machine (every group must flag) | 1 | 0.00 [0.00 - 0.05] | 0.00 [0.00 - 0.05] |
| R0_hash | status machine (every group must flag) | 4 | 0.15 [0.00 - 0.55] | 0.52 [0.20 - 0.85] |
| R0_hash | status machine (every group must flag) | 8 | 0.32 [0.10 - 0.70] | 0.93 [0.75 - 1.00] |
| R0_hash | status machine (every group must flag) | 16 | 0.54 [0.10 - 1.00] | 1.00 [1.00 - 1.00] |
| R0_hash | group test, nearest group only | 1 | 0.07 [0.00 - 0.25] | 0.16 [0.00 - 0.35] |
| R0_hash | group test, nearest group only | 4 | 0.20 [0.05 - 0.55] | 0.69 [0.35 - 1.00] |
| R0_hash | group test, nearest group only | 8 | 0.32 [0.05 - 0.70] | 0.97 [0.80 - 1.00] |
| R0_hash | group test, nearest group only | 16 | 0.57 [0.10 - 1.00] | 1.00 [1.00 - 1.00] |

## Offline consequence (an ESTIMATE, not gated)

- `R5_R4_qwen_L26`: per seed 28.2 false unseen and 22.3 missed unseen parts; OFFLINE ESTIMATE of the cost of the status errors: 269 pool points = 28.2 false_unseen x 4 + 22.2667 missed_unseen x 7. An estimate from recorded costs (E-034, E-030), not measured online and not gated; whether statuses drive learning well belongs to Phase 2.
- `R4_qwen_L26`: per seed 38.0 false unseen and 21.8 missed unseen parts; OFFLINE ESTIMATE of the cost of the status errors: 305 pool points = 38 false_unseen x 4 + 21.8 missed_unseen x 7. An estimate from recorded costs (E-034, E-030), not measured online and not gated; whether statuses drive learning well belongs to Phase 2.
- `R0_hash`: per seed 11.6 false unseen and 135.3 missed unseen parts; OFFLINE ESTIMATE of the cost of the status errors: 994 pool points = 11.6333 false_unseen x 4 + 135.3 missed_unseen x 7. An estimate from recorded costs (E-034, E-030), not measured online and not gated; whether statuses drive learning well belongs to Phase 2.

## Bars: draft and dev-fixed

Rule written before the dev run: a bar to stay above becomes max(draft, mean - 2 sd over the dev seeds), a bar to stay below becomes min(draft, mean + 2 sd); a dev result can tighten a draft bar and never loosen it. The dev-fixed bars are the ones the freeze records. Source of the dev values: the first reader above, status machine, q = 0.05, standard condition.

| measure | draft bar | dev mean | dev-fixed bar |
|---|---|---|---|
| macro_f1 (at least) | 0.85 | 0.855 | 0.850 |
| false_unseen_heldout (at most) | 0.05 | 0.289 | 0.050 |
| thin_vs_solid_auroc (at least) | 0.85 | 0.844 | 0.850 |
| other_as_unseen (at most) | 0.05 | 0.249 | 0.050 |
| b4a_residual_hit (at least) | 0.85 | 0.660 | 0.850 |
| b4a_known_not_flagged (at least) | 0.95 | 0.672 | 0.950 |
| b4b_nothing_flagged (at least) | 0.95 | 0.643 | 0.950 |
| b4b_other_labelled (at least) | 0.95 | 0.968 | 0.950 |

Where the false alarms and the misses come from (copied from `research/knowledge_status/results/E2_DIAG.txt`, produced by `diag_e2.py`):

Diagnosis of E2, R5_R4_qwen_L26, status machine q = 0.05, standard condition, seeds 100-129

1. False unseen of taught skills (solid or thin truth), by wording family
   S1  n=  737    8.7%   (taught wording)
   S2  n=  736    4.6%   (taught wording)
   S3  n=  717    2.0%   (taught wording)
   S4  n=  604   66.6%   (held-out wording)
   S5  n=  353   19.3%   (held-out wording)
   S6  n=  634    4.1%   (held-out wording)
   S7  n=  646   46.7%   (held-out wording)
   S8  n=  683    7.0%   (held-out wording)
   by skill and taught state, held-out wordings only:
   add  solid n=  320   34.4%
   add  thin  n=  200   26.5%
   div  thin  n=  240   21.7%
   max  solid n=  440   27.7%
   max  thin  n=  240   23.3%
   mod  thin  n=  440   23.2%
   mul  thin  n=  600   35.8%
   sub  solid n=  280   37.1%
   sub  thin  n=  160   20.0%

2. Untaught skills: found as unseen, and (when missed) which group took the part
   add  n= 1120 found  90.0%   missed parts taken by: mod 59%, sub 37%, div 4%, other 1%
   sub  n= 1050 found  90.3%   missed parts taken by: mod 92%, other 3%, div 2%, add 2%, max 1%
   mul  n= 1050 found  94.6%   missed parts taken by: mod 86%, div 14%
   div  n=  980 found  80.3%   missed parts taken by: mod 99%, sub 1%
   mod  n= 1190 found  86.9%   missed parts taken by: div 97%, max 2%, other 1%
   max  n=  910 found  94.7%   missed parts taken by: mod 88%, div 4%, add 4%, other 2%, sub 2%

3. Other content called unseen
   other           S6      n= 2414   51.8%
   other           other   n= 3000    3.3%
   trap            trap    n= 1800   47.2%
   unseen_content  unseen  n= 1800   62.2%

4. Distance of the wordings from the taught prototypes (taught skills): mean best cosine to the own group's prototypes, and the median conformal p-value (0.5 is what in-distribution parts give)
   S1  mean best cosine 0.992   median p 0.488
   S2  mean best cosine 0.994   median p 0.446
   S3  mean best cosine 0.997   median p 0.545
   S4  mean best cosine 0.950   median p 0.025
   S5  mean best cosine 0.994   median p 0.463
   S6  mean best cosine 0.991   median p 0.504
   S7  mean best cosine 0.894   median p 0.070
   S8  mean best cosine 0.992   median p 0.488

5. Local evidence of parts a skill covers (taught count: Solid 2000, Thin 20)
   Solid, S1  n=  244 median    33.5   10th percentile     1.0
   Solid, S2  n=  251 median    13.0   10th percentile     2.0
   Solid, S3  n=  246 median    76.0   10th percentile    12.0
   Solid, S4  n=   41 median     0.0   10th percentile     0.0
   Solid, S5  n=  113 median    10.0   10th percentile     0.0
   Solid, S6  n=  224 median    34.5   10th percentile     3.0
   Solid, S7  n=   97 median     1.0   10th percentile     0.0
   Solid, S8  n=  229 median    47.0   10th percentile     2.0
   Thin  all wordings n= 2703 median    20.0   90th percentile   117.0

How the predictions did (copied from `research/knowledge_status/PROGRESS.md`):

## 2026-10-27, Task 11 (the numbers): the machine finds new skills well; the false alarms on new wordings come almost entirely from two wordings

**What we did.** Ran E2 (single parts) and E3 (two-part messages) on 30 dev seeds with three readers: 90 pieces, all finished. Results: `research/knowledge_status/results/E2_TABLE.md` (full tables), `research/knowledge_status/results/E2_DIAG.txt` (where the misses come from). Suite: 206 tests pass.

**Headline: learned-map reader, status machine, q = 0.05, calibration on S1 to S3 only (mean over 30 seeds, [lowest - highest]).**

| measure | bar | result | what I predicted |
|---|---|---|---|
| Overall status score (macro F1 over solid / thin / unseen) | at least 0.85 | **0.855** [0.52 - 0.92] PASS, by 0.005 | 0.60 - 0.80 (better than predicted) |
| Known skills wrongly called "unseen" on new wordings S4-S8 | at most 0.05 | **0.289** [0.17 - 0.43] **fail** | 0.08 - 0.25 (worse) |
| the same on the taught wordings S1-S3 | - | 0.05 | 0.03 - 0.08 (right) |
| Thin separated from Solid (AUROC) | at least 0.85 | **0.844** fail, by 0.006 | 0.95 or more (wrong) |
| Everyday content called unseen | at most 0.05 | **0.249** fail (but see point 4) | 0.04 - 0.12 |
| Never-taught skills found as unseen | - | 0.89 | 0.35 - 0.65 (better) |
| Adapted condition, known skills called unseen | - | 0.17 | 0.05 - 0.10 (worse) |
| Two-part: leftover is the unseen part (B4a) | at least 0.85 | 0.66 fail | 0.45 - 0.75 (right) |
| Two-part: known part not flagged (B4a) | at least 0.95 | 0.67 fail | 0.80 - 0.93 (worse) |
| Two-part: nothing flagged, request plus everyday text (B4b) | at least 0.95 | 0.64 fail | 0.75 - 0.90 (worse) |
| Two-part: everyday part labelled "other" (B4b) | at least 0.95 | 0.97 PASS | 0.85 - 0.95 (right) |
| Automatic cutting within 0.10 of the gold number (B6) | - | 13 % of seeds (sentences), 7 % (sliding windows) | "sentences yes, windows no" (sentences also no) |

The wording claim from the calibration rule is **"fails"**: even with a few new-wording examples in the calibration (adapted) the rate is 0.17, not below 0.05. Plain Qwen without the learned map: 0.39 (the map helps by 10 points). The word-matcher control: 0.12 but it finds only 0.36 of the new skills, which is the failure it is supposed to show. All dev-fixed bars equal the draft bars: no dev result beat a draft bar by enough to tighten it.

**What this means, in plain words.**
1. **The false alarms are concentrated in two wordings.** Known skills are wrongly called new in **67 % of short stories (S4)** and **47 % of typo / case / spacing noise (S7)**; swapped operand order (S5) 19 %; chatter (S6) 4 % and register (S8) 7 % are about as good as the taught wordings (2 to 9 %). Stories and noisy text sit systematically farther from the taught examples (best cosine 0.95 and 0.89 against 0.99; median p-value 0.025 and 0.07 against about 0.5). The group test shows the same: a batch of 16 known-skill parts in new wordings is called new in 78 % of cases, a whole-group shift, not noise. Caveat: with gold spans a part is the request alone, so S6 and S8 (which differ from the taught wordings mostly in the sentence around the request) are not really tested by this unit.
2. **New skills are found well:** 80 to 95 % per skill (division 80 %, remainder 87 %, the rest 90 to 95 %). The misses are taken by the **remainder or division group**: they absorb each other (97 to 99 %), and also most of the rest.
3. **Against the standard methods the machine is level, not better.** At the same alarm level, kNN, Mahalanobis and the pool's own gate give false alarms 0.31 to 0.33 and find 0.95 to 0.97; the machine gives 0.29 and 0.89: the same trade-off. Classifier confidence and energy raise fewer false alarms but miss half the new skills (found 0.41 to 0.51); Relative Mahalanobis fails (AUROC 0.44). The value of the machine is the statuses (thin, other, calibrated per group), not detection power.
4. **The "everyday content" number is misleading.** Ordinary everyday sentences are called new only 3.3 % of the time (fine). The 25 % is dragged up by the hand-written chatter sentences of the S6 template (52 %), which are not the same kind of text as the taught everyday content, and by traps with test-only templates (47 %); never-taught content (science questions) is found only 62 % of the time. This is a **defect of my test bed** (the spec says the chatter should be drawn from the same corpora; the dev pool uses a hand-written list). I found it after seeing the numbers, so I record it as found afterwards.
5. **The Thin-vs-Solid failure is the wording problem again:** the local evidence of a Solid part in a story or noisy wording is about zero (median 0 and 1, against 33 to 76 in taught wordings).
6. **My own declared rule fired:** 17 of 60 "Solid" skills (28 %, over the quarter I set) are not solid by measured accuracy. With 2,000 examples the small solver learns addition (0.94), subtraction (0.91) and maximum (0.95) but not multiplication (0.33), division (0.72) or remainder (0.58). I did not change any threshold; the choice is yours (see below).
7. **Offline cost estimate** (not gated, an estimate): 269 pool points per seed for the learned-map reader, 305 for plain Qwen, 994 for the word-matcher.

**What I propose, not yet done (your call at Stop Point 2).**
- **A. One bounded round on the wording problem before Task 12**, with its prediction written first: (i) calibrate on *leave-one-wording-out* data (each taught wording is tested against prototypes of the other taught wordings, which uses S1-S3 only and teaches the alarm what a wording shift looks like); (ii) reading the text after generic clean-up (case and spacing). Training with S7-style noise would break the rule that S4-S8 are never used for teaching, so I would not do that without your say.
- **B. Fix the test bed** (teach the everyday content from the same source as the chatter) before the freeze, and say so.
- **C. The small solver:** improve it for multiplication, division and remainder, or accept that only addition, subtraction and maximum can be Solid.
- Then Task 12 (worked traces), and the from-scratch reader (Task 11b) as you decided.

## Interpretation (what we understood, and on what basis)
In plain words: the machine is good at spotting skills it was never taught (it finds 89 % of their parts) and at saying other, thin and solid, but it does not trust known skills in new wordings: 29 % of them are called "new" where the bar is 5 %, and almost all of that is two wordings, short stories (67 %) and typo / case / spacing noise (47 %); the other wordings are as good as the taught ones (2 to 9 %). The shift is systematic, not noise: a batch of 16 known-skill parts in new wordings is called new in 78 % of cases. At the same alarm level the standard detectors that look at the vectors (kNN, Mahalanobis, the pool's gate) make the same trade (false alarms 0.31 to 0.33, found 0.95 to 0.97 against the machine's 0.29 and 0.89); classifier confidence and energy raise fewer false alarms but miss half the new skills; Relative Mahalanobis fails (AUROC 0.44). So the machine's value is the graded statuses and the per-group calibration, not extra detection power. Measured: the tables. Assumed: that the same holds on the confirmatory pool.

## Caveats and what went wrong
- A test-bed defect found AFTER the numbers: the dev pool's S6 chatter sentences are a hand-written list, not drawn from the corpora as spec 5.2 says, so they are a different kind of text from the taught Other content; they are called unseen 52 % of the time and drag "Other content called unseen" to 0.25 (ordinary everyday sentences: 3.3 %). To be fixed before the freeze and recorded as found afterwards.
- The declared THRESHOLDS rule fired: 17 of 60 Solid-state skills (28 %) are grey or thin by measured accuracy (mul 0.33, div 0.72, mod 0.58 at 2,000 examples; add 0.94, sub 0.91, max 0.95). Nothing was changed.
- With gold spans a part is the request alone, so S6 (chatter) and S8 (register) differ from the taught wordings mostly in the sentence around the request and are hardly tested by this unit.
- The baselines use one pooled threshold (the standard way); the machine calibrates each group on its own held-out scores. A detector whose scores differ by class is penalised by pooling.
- The Thin skill is calibrated on 120 held-out parts although it was taught 20 examples (a conformal test needs at least 19 at q = 0.05 and 99 at q = 0.01): a declared simplification.
- The Thin-vs-Solid AUROC (0.844, bar 0.85) fails for the wording reason: the local evidence of a Solid part in a story or noisy wording is about zero. My prediction (0.95 or more) was wrong.
- Commit-hook (truthgate) warnings, disclosed: the claim label "needs adaptation to new wording" is computed from two rates and the bar, not hard-coded; the "best of N seeds" warning in `tune_baselines.py` is a maximum over tuning SETTINGS of a mean over ALL tuning seeds.
- A planted-bug check (a deliberate bug per mutant) was run on `status.py`, `baselines.py`, `evaluate.py`, `segment.py`, `run_e2.py`: every mutant but one equivalent mutant is caught after nine missing tests were added. One interruption left a planted bug in `status.py` for a moment; it was found by search, restored and compared with the backup before the commit.
- No independent reviewer has looked at the code yet (Task 14 gate).

## Consequence
Stop Point 2 was reached and the user chose to find the cause and try fixes: E-046.

## Evidence
- `research/knowledge_status/results/E2_TABLE.md` - the E2 and E3 tables (generated)
- `research/knowledge_status/results/e2` - one result file per reader and seed
- `research/knowledge_status/results/E2_DIAG.txt` - the breakdown by wording and skill
- `research/knowledge_status/results/BASELINE_TUNING.md` - the baselines' settings (tuning seeds 150-154)
- `research/knowledge_status/results/COMPETENCE_TABLE.md` - the competence sweep behind the status truth
- `research/knowledge_status/PROGRESS.md` - the predictions, the rules and the plain-English account
- `research/knowledge_status/status.py`, `research/knowledge_status/baselines.py`, `research/knowledge_status/evaluate.py`, `research/knowledge_status/segment.py`, `research/knowledge_status/run_e2.py` - the code

## Reproduce
```
cd research/knowledge_status
PYTHONPATH=<repo root> python run_e2.py --pool dev --seeds 100-129 --reps R5_R4_qwen_L26,R4_qwen_L26,R0_hash
PYTHONPATH=<repo root> python diag_e2.py > results/E2_DIAG.txt
```
