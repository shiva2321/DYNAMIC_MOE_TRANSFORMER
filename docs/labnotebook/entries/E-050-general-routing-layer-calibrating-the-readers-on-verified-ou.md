# E-050 - General routing layer: calibrating the readers on verified outcomes of the stream makes the mix honest in four domains, with modest power and a fragile cut (labs G4 and G4b)

- **ID:** E-050
- **Date:** 2026-10-30
- **Track:** G
- **Type:** experiment
- **Status:** completed for the tuning seeds (G4 and G4b); the robust calibration (G4c) and the growmoe experts as container are not built
- **Evidence grade:** B (script-generated on the tuning seeds 150-152; every rule and prediction written in PROGRESS.md before the numbers; exploratory, NOT a frozen pre-registration, NOT the single-use seeds; no independent reviewer yet)
- **Code / commit:** branch `knowledge-status-phase1`, commits 2eaf66b to 1120e7b; code in `research/knowledge_status/routing_general/` (`outcome.py`, `lab_outcome.py`), tests `tests/knowledge_status/test_rg_outcome.py`, `test_rg_lab_outcome.py`; worktree D:/shm_ks
- **People:** Claude built and ran it on the user's go of 2026-10-30 ("yes run G4 first")

## Question
E-049 showed that rewrites do not say which reader to trust. A pool does receive OUTCOMES: a label, or a stated answer that a taught expert reproduces or not. If every reader's cut is calibrated on verified known messages of the stream's own wording, and the readers (the asking reader and the pool's density gate) are fused by "either", does the mix stay honest (false flags near the nominal 5 %) and keep the better reader's power in every domain? And what does a noisy answer check do to it?

## Why we asked it
E-048 (no single reader, no fixed fusion) and E-049 (no roster-only trust diagnostic) left the layer without a way to mix its readers. The layer's verifier contract (design note section 2) already names the missing ingredient: the expert's own answer against a stated answer. The October adapted-calibration result for the look-based status machine (0.17 of known skills still called new) and the finding that only about 10 % of skills have competent experts were read first and carried into the predictions.

## Prediction (written BEFORE the numbers, if there was one)
In `research/knowledge_status/PROGRESS.md`, blocks of 2026-10-30. C8 (at 30 verified messages, in every domain the either-fusion flags at most 0.10 of held-out known and finds at least the best single reader's rate - 0.05): predicted to hold in all four domains with probability 25 %; it held. C9 (arithmetic false flags from at least 0.30 to at most 0.10): a sanity check, predicted 90 %; held. C10 (with a contaminated stream the fusion keeps two thirds of its power): predicted to pass with 20 %, then 40 % for the paired rerun; failed by 0.003 unpaired, passed by 0.006 paired.

## Method (how)
- **Outcomes** (`lab_outcome.py`): the stream's known messages are split per class into a pool and an evaluation half. A pool message is VERIFIED when its ownership is confirmed by an outcome: text domains, its label is a taught class (exact); arithmetic, the numbers can be read and SOME taught expert gives the stated answer (noisy). A random sample of n verified messages calibrates every reader's cut (arm verified), or is added to the held-out teaching-wording messages (arm union); arm fixed is E-048's calibration.
- **Systems**: each of the four readers alone, either- and both-fusion of the pair (asking reader, vMF), either-fusion of all four; conformal p-values recalibrated by leave-one-out (`measures.py`). Headline n = 30, n in {10, 20, 30, 40}, 5 random draws per cell, judged on the evaluation half of the known messages and on the never-taught and foreign messages.
- **G4b** (`contamination_arms`): the stream also holds never-taught messages (a quarter of it); the answer check wrongly vouches for some of them. The first run drew the clean and contaminated samples from different random streams (5 draws); the paired run uses one random order (the clean sample is the same order without the never-taught messages) and 20 draws.
- Seeds 150-152; the worlds of E-049, so every asked answer was cached; no download; `growmoe/` untouched; no rule changed after a result (the G4b pairing changed the measurement, not C10).

## Results (what came out)
Numbers copied by script from `research/knowledge_status/results/ROUTING_GENERAL_G4_150_152.md` (3 seeds; the G4b rows are the paired run; the unpaired first run is kept in `research/knowledge_status/results/routing_general/ROUTING_GENERAL_G4b_unpaired_150_152.md`).

### arithmetic (3 seeds; 120 known, 120 never-taught, 236 foreign per seed)

Verification: 0.558 of the known messages are verified (Solid-taught owners 0.725, Thin 0.225); never-taught messages wrongly vouched for 0.158, other unowned 0.115. Verified messages available in the pool at n=30: 29.6 on average.

|---|---|---|---|---|---|---|---|---|
| ask_exemplars | 0.109 | 0.669 | 0.913 | 0.039 | 0.253 | 0.911 | 0.106 | 0.667 |
| vmf_conformal | 0.366 | 0.756 | 0.705 | 0.018 | 0.047 | 0.704 | 0.337 | 0.641 |
| mahalanobis | 0.407 | 0.925 | 0.774 | 0.029 | 0.030 | 0.710 | 0.369 | 0.882 |
| knn_distance | 0.350 | 0.361 | 0.635 | 0.020 | 0.036 | 0.638 | 0.296 | 0.323 |
| either_pair | 0.350 | 0.800 | 0.740 | 0.054 | 0.280 | 0.877 | 0.286 | 0.613 |
| both_pair | 0.302 | 0.919 | 0.885 | 0.018 | 0.068 | 0.812 | 0.256 | 0.894 |
| either_all | 0.387 | 0.917 | 0.777 | 0.074 | 0.284 | 0.853 | 0.291 | 0.591 |

Verified messages needed (either-fusion of the pair, arm verified):

|---|---|---|---|---|
| 10 | 10.0 | 0.000 | 0.000 | 0.000 |
| 20 | 20.0 | 0.079 | 0.380 | 1.000 |
| 30 | 29.6 | 0.054 | 0.280 | 0.999 |
| 40 | 33.4 | 0.047 | 0.212 | 0.999 |

Contamination (G4b): the stream also holds never-taught messages (share 0.25), the answer check vouches for some of them; on average 0.091 of the verified sample at n=30 is never-taught. Power is judged on never-taught messages that never entered a calibration; the clean arm is the same split without the contamination.

|---|---|---|---|---|
| ask_exemplars | 0.030 | 0.221 | 0.015 | 0.147 |
| vmf_conformal | 0.025 | 0.045 | 0.024 | 0.034 |
| either_pair | 0.051 | 0.248 | 0.036 | 0.171 |
| both_pair | 0.019 | 0.074 | 0.015 | 0.074 |
| either_all | 0.069 | 0.253 | 0.051 | 0.178 |

### banking (3 seeds; 180 known, 300 never-taught, 150 foreign per seed)

Verification: 1.000 of the known messages are verified (Solid-taught owners 1.000, Thin nan); never-taught messages wrongly vouched for 0.000. Verified messages available in the pool at n=30: 30.0 on average.

|---|---|---|---|---|---|---|---|---|
| ask_exemplars | 0.053 | 0.341 | 0.799 | 0.021 | 0.186 | 0.796 | 0.045 | 0.324 |
| vmf_conformal | 0.056 | 0.830 | 0.965 | 0.041 | 0.642 | 0.953 | 0.057 | 0.824 |
| mahalanobis | 0.045 | 0.747 | 0.958 | 0.041 | 0.634 | 0.948 | 0.046 | 0.744 |
| knn_distance | 0.050 | 0.726 | 0.952 | 0.029 | 0.596 | 0.944 | 0.049 | 0.724 |
| either_pair | 0.051 | 0.721 | 0.952 | 0.059 | 0.676 | 0.935 | 0.045 | 0.712 |
| both_pair | 0.061 | 0.612 | 0.903 | 0.019 | 0.378 | 0.893 | 0.056 | 0.601 |
| either_all | 0.036 | 0.724 | 0.953 | 0.082 | 0.792 | 0.931 | 0.036 | 0.724 |

Verified messages needed (either-fusion of the pair, arm verified):

|---|---|---|---|---|
| 10 | 10.0 | 0.000 | 0.000 | 0.000 |
| 20 | 20.0 | 0.096 | 0.810 | 1.000 |
| 30 | 30.0 | 0.059 | 0.676 | 1.000 |
| 40 | 40.0 | 0.067 | 0.687 | 1.000 |

Contamination (G4b): the stream also holds never-taught messages (share 0.25), the answer check vouches for some of them; on average 0.000 of the verified sample at n=30 is never-taught. Power is judged on never-taught messages that never entered a calibration; the clean arm is the same split without the contamination.

|---|---|---|---|---|
| ask_exemplars | 0.032 | 0.216 | 0.032 | 0.216 |
| vmf_conformal | 0.039 | 0.654 | 0.039 | 0.654 |
| either_pair | 0.068 | 0.700 | 0.068 | 0.700 |
| both_pair | 0.033 | 0.432 | 0.033 | 0.432 |
| either_all | 0.096 | 0.788 | 0.096 | 0.788 |

### agnews (3 seeds; 180 known, 300 never-taught, 150 foreign per seed)

Verification: 1.000 of the known messages are verified (Solid-taught owners 1.000, Thin nan); never-taught messages wrongly vouched for 0.000. Verified messages available in the pool at n=30: 30.0 on average.

|---|---|---|---|---|---|---|---|---|
| ask_exemplars | 0.061 | 0.489 | 0.855 | 0.037 | 0.253 | 0.849 | 0.056 | 0.488 |
| vmf_conformal | 0.064 | 0.528 | 0.880 | 0.013 | 0.215 | 0.879 | 0.048 | 0.456 |
| mahalanobis | 0.056 | 0.501 | 0.889 | 0.020 | 0.220 | 0.884 | 0.056 | 0.486 |
| knn_distance | 0.047 | 0.540 | 0.895 | 0.027 | 0.313 | 0.889 | 0.045 | 0.522 |
| either_pair | 0.060 | 0.546 | 0.886 | 0.047 | 0.390 | 0.883 | 0.056 | 0.527 |
| both_pair | 0.061 | 0.621 | 0.904 | 0.018 | 0.270 | 0.900 | 0.052 | 0.580 |
| either_all | 0.070 | 0.621 | 0.889 | 0.066 | 0.512 | 0.884 | 0.061 | 0.561 |

Verified messages needed (either-fusion of the pair, arm verified):

|---|---|---|---|---|
| 10 | 10.0 | 0.000 | 0.000 | 0.000 |
| 20 | 20.0 | 0.061 | 0.538 | 0.681 |
| 30 | 30.0 | 0.047 | 0.390 | 0.571 |
| 40 | 40.0 | 0.050 | 0.466 | 0.620 |

Contamination (G4b): the stream also holds never-taught messages (share 0.25), the answer check vouches for some of them; on average 0.000 of the verified sample at n=30 is never-taught. Power is judged on never-taught messages that never entered a calibration; the clean arm is the same split without the contamination.

|---|---|---|---|---|
| ask_exemplars | 0.031 | 0.305 | 0.031 | 0.305 |
| vmf_conformal | 0.029 | 0.288 | 0.029 | 0.288 |
| either_pair | 0.055 | 0.458 | 0.055 | 0.458 |
| both_pair | 0.026 | 0.341 | 0.026 | 0.341 |
| either_all | 0.069 | 0.540 | 0.069 | 0.540 |

### newsgroups (3 seeds; 180 known, 300 never-taught, 150 foreign per seed)

Verification: 1.000 of the known messages are verified (Solid-taught owners 1.000, Thin nan); never-taught messages wrongly vouched for 0.000. Verified messages available in the pool at n=30: 30.0 on average.

|---|---|---|---|---|---|---|---|---|
| ask_exemplars | 0.053 | 0.148 | 0.774 | 0.022 | 0.069 | 0.773 | 0.050 | 0.143 |
| vmf_conformal | 0.060 | 0.254 | 0.736 | 0.026 | 0.134 | 0.732 | 0.059 | 0.250 |
| mahalanobis | 0.049 | 0.226 | 0.773 | 0.036 | 0.149 | 0.770 | 0.047 | 0.226 |
| knn_distance | 0.053 | 0.251 | 0.771 | 0.039 | 0.160 | 0.767 | 0.050 | 0.242 |
| either_pair | 0.041 | 0.202 | 0.767 | 0.048 | 0.193 | 0.763 | 0.044 | 0.212 |
| both_pair | 0.056 | 0.257 | 0.804 | 0.034 | 0.151 | 0.798 | 0.053 | 0.235 |
| either_all | 0.036 | 0.223 | 0.780 | 0.093 | 0.311 | 0.772 | 0.033 | 0.218 |

Verified messages needed (either-fusion of the pair, arm verified):

|---|---|---|---|---|
| 10 | 10.0 | 0.000 | 0.000 | 0.000 |
| 20 | 20.0 | 0.067 | 0.260 | 0.400 |
| 30 | 30.0 | 0.048 | 0.193 | 0.352 |
| 40 | 40.0 | 0.040 | 0.182 | 0.336 |

Contamination (G4b): the stream also holds never-taught messages (share 0.25), the answer check vouches for some of them; on average 0.000 of the verified sample at n=30 is never-taught. Power is judged on never-taught messages that never entered a calibration; the clean arm is the same split without the contamination.

|---|---|---|---|---|
| ask_exemplars | 0.034 | 0.101 | 0.034 | 0.101 |
| vmf_conformal | 0.035 | 0.179 | 0.035 | 0.179 |
| either_pair | 0.067 | 0.262 | 0.067 | 0.262 |
| both_pair | 0.026 | 0.132 | 0.026 | 0.132 |
| either_all | 0.097 | 0.359 | 0.097 | 0.359 |

## Verdicts (rules fixed before the run)

- C8, arithmetic: either-fusion calibrated on the verified sample flags 0.054 of held-out known (<= 0.1; the fixed calibration flags 0.350) and finds 0.280 (best reader alone 0.253, need >= that - 0.05). **PASS**
- C8, banking: either-fusion calibrated on the verified sample flags 0.059 of held-out known (<= 0.1; the fixed calibration flags 0.051) and finds 0.676 (best reader alone 0.642, need >= that - 0.05). **PASS**
- C8, agnews: either-fusion calibrated on the verified sample flags 0.047 of held-out known (<= 0.1; the fixed calibration flags 0.060) and finds 0.390 (best reader alone 0.313, need >= that - 0.05). **PASS**
- C8, newsgroups: either-fusion calibrated on the verified sample flags 0.048 of held-out known (<= 0.1; the fixed calibration flags 0.041) and finds 0.193 (best reader alone 0.160, need >= that - 0.05). **PASS**
- C8 overall (all four domains): **PASS**
- C9 (sanity, true by construction): arithmetic fixed 0.350 (>= 0.30), verified 0.054 (<= 0.10). **PASS**
- C10 (arithmetic, contaminated stream): the either-fusion flags 0.036 of held-out known (<= 0.1) and finds 0.171 against 0.248 without the contamination (need >= two thirds of it, 0.165); 0.091 of the sample was never-taught. **PASS**


## Interpretation (what we understood, and on what basis)
Plain words first. **Calibrating on verified messages of the stream's own wording makes every reader honest**: all calibrated systems flag 0.02-0.08 of held-out known messages in all four domains, where the fixed calibration flagged 0.350 in arithmetic. The look-based readers then show what they are worth under a wording shift: almost nothing (vMF found 0.047, Mahalanobis 0.030, kNN 0.036 of the never-taught messages in arithmetic; AUROC 0.64-0.71). They only looked powerful when they over-flagged. In the topic domains, where the old calibration was already fair, the numbers hardly change except for the loss of power that a small sample causes. **The either-mix of the asking reader and the density gate loses nothing at this operating point**: it finds as much as or more than the best single reader in every domain (0.280 against 0.253, 0.676 against 0.642, 0.390 against 0.313, 0.193 against 0.160); cut-free its AUROC is slightly below the best single reader's (0.877 against 0.911, 0.935 against 0.953, 0.883 against 0.889, 0.763 against 0.773).

**What this is not.** Power at an honest false-flag rate is modest (0.28, 0.68, 0.39, 0.19). With 30 verified messages the 5 % cut is the largest of 30 scores (an effective rate of about 3 %), from 40 it is the second largest; power therefore falls from n = 20 to n = 40 in arithmetic (0.380, 0.280, 0.212) while the false-flag rate falls from 0.079 to 0.047: the n = 20 row over-flags. Below 19 verified messages the systems are silent. **The cut is fragile to a noisy outcome**: the answer check vouches for 15.8 % of never-taught arithmetic messages by coincidence (small operands), about 9 % of a sample drawn from a stream with a quarter never-taught messages, and each such message sets the cut. That costs about a third of the power (31 % paired, 34 % unpaired; the C10 bar sits at 33 %, so the two runs fall on opposite sides of it and neither verdict is a finding).

Technical statement: a conformal cut on a small verified sample is valid in false-flag rate for exchangeable data (C-104 shows it holds across domains), its power is the power of the reader's ROC at the sample's quantile, and a few contaminants move a max-order-statistic cut by the contaminants' scores (C-105). Measured: the tables above. Assumed: that the verified sample is exchangeable with the evaluation half (it is a random split of one test set), that the stream's wording does not drift, and that a label or stated answer is available for a few dozen messages.

## Caveats and what went wrong
- **The unpaired first G4b run was noise-limited by my own design (F-51)**: different random streams for the clean and the contaminated sample made the text domains differ by up to 0.10 where there was nothing to differ; its FAIL by 0.003 and the paired run's PASS by 0.006 are the same effect on opposite sides of a bar.
- **Predictions missed (F-52)**: arithmetic power at the verified cut (asking reader 0.253 against 0.45-0.62, pair 0.280 against 0.45-0.62, vMF 0.047 against 0.15-0.40); false verification of never-taught messages 0.158 (0.03-0.12); text-domain fusion power (banking 0.676 against 0.76-0.84, agnews 0.390 against 0.50-0.62); the union arm in arithmetic (0.286 against 0.12-0.25); I expected the Thin skill's absence from the sample to hurt the look-based readers and it did not (vMF known flagged 0.018).
- The verified sample is drawn from a random half of the same test set the evaluation half comes from, so exchangeability holds by construction; a stream that drifts, or a check that verifies only the easy messages (arithmetic: Solid-taught owners 0.725, the Thin owner 0.225), would break it. The arithmetic evaluation is the 120 known messages per seed; three seeds, three taught skills.
- The answer check uses SOME taught expert reproducing the stated answer; a competent-only check would be cleaner and cover less (October: about 10 % of skills).
- Not done: a robust calibration; the asking reader's closed list still hands never-taught skills to the nearest card (E-049); the growmoe experts as the container.

## Consequence
The layer now has a working general mechanism for mixing its readers: calibrate each reader's cut on verified outcomes of the stream's own wording (at least 40 verified messages; none below 19), fuse by "either" (answer-more) or "both" (refuse-more), keep the choice a stated policy. It is honest in false flags in all four domains and modest in power, and it is fragile to a noisy outcome check. Next, in this order: G4c, a robust calibration against wrongly verified messages (leave out the largest scores, or require a second independent outcome), predictions first; then the growmoe experts as the container (needs the vector design for language arithmetic); then self-discovery of cards. Nothing is frozen and the confirmatory seeds 400-429 are unspent.

## Evidence
- `research/knowledge_status/results/ROUTING_GENERAL_G4_150_152.md` (generated; paired G4b), `research/knowledge_status/results/routing_general/ROUTING_GENERAL_G4b_unpaired_150_152.md` (the first, unpaired run), `research/knowledge_status/results/routing_general/g4/` (cells), run logs
- `research/knowledge_status/PROGRESS.md` (predictions, results, the measurement refinement, in order)

## Reproduce
```
cd D:/shm_ks
python research/knowledge_status/routing_general/lab_outcome.py 150 152
python -m pytest tests/knowledge_status/test_rg_*.py
```

