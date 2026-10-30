# E-049 - General routing layer: rewrites do not say which reader to trust, and the same experts behind an asking gate and a vMF gate tie on net score (labs G3 and G2, and a correction to E-048)

- **ID:** E-049
- **Date:** 2026-10-30
- **Track:** G
- **Type:** experiment
- **Status:** completed (corrected run); G4, trust estimated from outcomes, is not built
- **Evidence grade:** B (script-generated on the tuning seeds 150-152; every rule and prediction written in PROGRESS.md before the numbers; exploratory, NOT a frozen pre-registration, NOT the single-use seeds; no independent reviewer yet)
- **Code / commit:** branch `knowledge-status-phase1`, commits 5cad2fd to f11be04; code in `research/knowledge_status/routing_general/` (`shift.py`, `lab_shift.py`, `gates.py`, `pool.py`, `lab_pool.py`), tests `tests/knowledge_status/test_rg_*.py`; worktree D:/shm_ks
- **People:** Claude built and ran it on the user's go of 2026-10-30 ("yes go ahead with step 1 then step 2"); the flaw in the known set was found by Claude from an implausible oracle row

## Question
Step 1 (G3): can a domain-free, label-free diagnostic computed from the roster's own held-out messages (the flag rate of each reader on LM rewrites of them) say which reader to trust under a wording shift, so the layer can mix the asking reader and the density gate by itself? Step 2 (G2): with the SAME trained experts behind the pool's own vMF gate, the general layer's asking gate, a mix and an oracle, which gate gives better answers, and is the asking gate worth at least 0.15 more net score?

## Why we asked it
E-048 found that no single reader works in every domain and that no fixed fusion does either (C-98, C-100); the design note (section 6, ruling R3) asked for trust to come from the roster's own data, and the user asked for the pool wiring. The pool (`growmoe/pools/phylo.py`) is vector-in, has its own gate, ignores `ctx` and may not be edited, so G2 uses a pool-shaped container of our own (`GatedPool`) at inference time with the lab's arithmetic solvers as experts.

## Prediction (written BEFORE the numbers, if there was one)
In `research/knowledge_status/PROGRESS.md`, blocks of 2026-10-30. G3: C6a (Spearman >= 0.6 between shift flag and true known-flagged, bad readers untrusted, 80 % of good readers trusted) predicted to FAIL (55 %); C6b (the trusted-either system within 0.05 of the best usable reader in all four domains) predicted to FAIL in at least one domain (60 %). Both failed. G2: C7 (ask NET at least 0.15 above vMF) predicted to HOLD (80 %); after the corrected rerun I put 30 % on it; it failed (-0.030).

## Method (how)
- **G3** (`shift.py`, `lab_shift.py`): per domain and seed the held-out known messages are split per class into a half that sets each reader's 5 % cut and a half that Qwen2.5-1.5B rewrites with the four generic instructions of `rewrites.py` (at most 30 sources per class; a generic filter keeps one-line, non-chatter rewrites that keep every digit string; arithmetic keeps the stricter number check). SHIFT FLAG = share of rewrites flagged at the reader's cut; trusted when at most 0.10. Systems: the fixed pair fusions, "trusted_either" (either-fusion of the trusted readers), the same with the rewrites added to the calibration ("calV"), and "trusted_all_either". Text domains use 60 held-out messages per class.
- **G2** (`gates.py`, `pool.py`, `lab_pool.py`): arithmetic in everyday language. Experts: the solver of each taught skill (`lab_function.train_solvers`), operands read by a fixed parser (a declared simplification). Gates: oracle (knows the owner), ask (card reader, claim rule tau 0.5), vmf (the pool's density gate on MiniLM, one signature per expert, cut at 5 % of held-out known messages), mix (abstain when either abstains, route with ask). Known messages: every taught skill in the new wordings S4-S8 (120 per seed); unowned messages: never-taught skills in new wordings (120) and in teaching wordings (90), everyday sentences (236), traps (60). NET = correct - wrong on known - mean share of unowned messages answered, one point each (declared before).
- Seeds 150-152; no download; `growmoe/` untouched; the first G2 run and the arithmetic rows of G1 and G3 used a known set that left out `grey` skills (F-49); all three were rerun, no bar changed.

## Results (what came out)
Numbers copied by script from `research/knowledge_status/results/ROUTING_GENERAL_G3_150_152.md` and `ROUTING_GENERAL_G2_150_152.md` (corrected runs, 3 seeds, mean).

## Lab G3

### arithmetic (3 seeds; per seed 169 cut messages, 190 rewrites, 120 known, 120 never-taught, 236 foreign)

|---|---|---|---|---|---|---|
| ask_exemplars | 0.099 | 0.100 | 0.694 | 0.988 | 0.921 | yes |
| vmf_conformal | 0.314 | 0.381 | 0.731 | 1.000 | 0.689 | no |
| mahalanobis | 0.511 | 0.417 | 0.933 | 1.000 | 0.698 | no |
| knn_distance | 0.040 | 0.336 | 0.331 | 1.000 | 0.628 | yes |

|---|---|---|---|---|
| either_fixed | 0.364 | 0.825 | 1.000 | 0.738 |
| both_fixed | 0.300 | 0.914 | 1.000 | 0.882 |
| trusted_either | 0.100 | 0.694 | 0.988 | 0.919 |
| trusted_either_calV | 0.067 | 0.528 | 0.973 | 0.921 |
| trusted_all_either | 0.314 | 0.372 | 1.000 | 0.666 |

Trusted pair readers per seed: {'ask_exemplars': 3}. Who is it? top-1 on known / on rewrites: ask 0.933 / 0.946; logistic regression on MiniLM 0.700 / 0.653.

### banking (3 seeds; per seed 180 cut messages, 635 rewrites, 180 known, 300 never-taught, 150 foreign)

|---|---|---|---|---|---|---|
| ask_exemplars | 0.087 | 0.044 | 0.321 | 0.904 | 0.799 | yes |
| vmf_conformal | 0.270 | 0.057 | 0.836 | 1.000 | 0.967 | no |
| mahalanobis | 0.276 | 0.046 | 0.740 | 1.000 | 0.961 | no |
| knn_distance | 0.165 | 0.046 | 0.719 | 1.000 | 0.953 | no |

|---|---|---|---|---|
| either_fixed | 0.076 | 0.797 | 1.000 | 0.953 |
| both_fixed | 0.054 | 0.580 | 0.960 | 0.909 |
| trusted_either | 0.044 | 0.321 | 0.904 | 0.798 |
| trusted_either_calV | 0.030 | 0.224 | 0.867 | 0.799 |
| trusted_all_either | 0.044 | 0.321 | 0.904 | 0.798 |

Trusted pair readers per seed: {'ask_exemplars': 3}. Who is it? top-1 on known / on rewrites: ask 0.911 / 0.878; logistic regression on MiniLM 0.989 / 0.978.

### agnews (3 seeds; per seed 60 cut messages, 141 rewrites, 180 known, 300 never-taught, 150 foreign)

|---|---|---|---|---|---|---|
| ask_exemplars | 0.088 | 0.085 | 0.553 | 0.527 | 0.854 | yes |
| vmf_conformal | 0.082 | 0.074 | 0.518 | 0.722 | 0.874 | yes |
| mahalanobis | 0.086 | 0.061 | 0.498 | 0.818 | 0.883 | yes |
| knn_distance | 0.073 | 0.063 | 0.563 | 0.773 | 0.892 | yes |

|---|---|---|---|---|
| either_fixed | 0.069 | 0.581 | 0.751 | 0.879 |
| both_fixed | 0.070 | 0.617 | 0.664 | 0.897 |
| trusted_either | 0.070 | 0.514 | 0.731 | 0.869 |
| trusted_either_calV | 0.035 | 0.326 | 0.533 | 0.875 |
| trusted_all_either | 0.087 | 0.623 | 0.856 | 0.878 |

Trusted pair readers per seed: {'ask_exemplars + vmf_conformal': 2, 'vmf_conformal': 1}. Who is it? top-1 on known / on rewrites: ask 0.867 / 0.812; logistic regression on MiniLM 0.913 / 0.891.

### newsgroups (3 seeds; per seed 180 cut messages, 404 rewrites, 180 known, 300 never-taught, 150 foreign)

|---|---|---|---|---|---|---|
| ask_exemplars | 0.106 | 0.052 | 0.131 | 0.062 | 0.770 | no |
| vmf_conformal | 0.077 | 0.056 | 0.257 | 0.556 | 0.740 | yes |
| mahalanobis | 0.080 | 0.048 | 0.233 | 0.644 | 0.779 | yes |
| knn_distance | 0.073 | 0.059 | 0.250 | 0.624 | 0.775 | yes |

|---|---|---|---|---|
| either_fixed | 0.037 | 0.196 | 0.404 | 0.768 |
| both_fixed | 0.054 | 0.216 | 0.216 | 0.802 |
| trusted_either | 0.048 | 0.212 | 0.509 | 0.747 |
| trusted_either_calV | 0.033 | 0.192 | 0.458 | 0.748 |
| trusted_all_either | 0.044 | 0.210 | 0.567 | 0.780 |

Trusted pair readers per seed: {'vmf_conformal': 2, 'ask_exemplars + vmf_conformal': 1}. Who is it? top-1 on known / on rewrites: ask 0.752 / 0.691; logistic regression on MiniLM 0.820 / 0.784.

## Verdicts (rules fixed before the run)

- C6a (the diagnostic): Spearman 0.044 (>= 0.6); readers with a true known-flagged rate above 0.15: 3, all untrusted: False; readers at or under 0.1: 13, share trusted 0.69 (>= 0.80). **FAIL**
- C6b, arithmetic: trusted-either flags 0.100 of known (<= 0.15) and finds 0.694 (best usable reader 0.694, need >= that - 0.05). **PASS**
- C6b, banking: trusted-either flags 0.044 of known (<= 0.15) and finds 0.321 (best usable reader 0.836, need >= that - 0.05). **FAIL**
- C6b, agnews: trusted-either flags 0.070 of known (<= 0.15) and finds 0.514 (best usable reader 0.563, need >= that - 0.05). **PASS**
- C6b, newsgroups: trusted-either flags 0.048 of known (<= 0.15) and finds 0.212 (best usable reader 0.257, need >= that - 0.05). **PASS**
- C6b overall (all four domains): **FAIL**

## Lab G2

|---|---|---|---|---|---|---|---|---|
| oracle | 1.000 | 0.497 | 0.503 | 0.497 | 1.000 | 0.696 | 0.100 | -0.006 |
| ask | 0.917 | 0.447 | 0.469 | 0.488 | 0.979 | 0.633 | 0.075 | -0.253 |
| vmf | 0.619 | 0.286 | 0.333 | 0.459 | 0.936 | 0.396 | 0.067 | -0.223 |
| mix | 0.603 | 0.300 | 0.303 | 0.493 | 0.977 | 0.421 | 0.058 | -0.094 |

Share of messages nobody owns that got an answer:

| gate | unknown | teaching_wording | foreign | trap |
|---|---|---|---|---|
| oracle | 0.000 | 0.000 | 0.000 | 0.000 |
| ask | 0.419 | 0.504 | 0.000 | 0.000 |
| vmf | 0.244 | 0.459 | 0.000 | 0.000 |
| mix | 0.114 | 0.252 | 0.000 | 0.000 |

Expert accuracy on problems it never saw: Solid experts 0.753, Thin experts 0.127.

**C7** (asking gate NET at least 0.15 above the vMF gate's): ask -0.253, vmf -0.223, difference -0.030. **FAIL**

## Interpretation (what we understood, and on what basis)
Plain words first. **Rewrites by the language model do not tell the layer which reader to trust.** They move a topic reader far more than natural variation does (banking: the look-based readers flag 17-28 % of the rewrites and 5 % of real known messages, so they are wrongly distrusted) and a function reader far less than a real change of wording (arithmetic: kNN flags 4 % of the rewrites and 34 % of the real new-wording messages). Only the asking reader behaves the same on rewrites and on real messages in every domain. So the diagnostic has no rank information (Spearman 0.04) and the rule 'trust at most 0.10' throws away the best reader in banking.

**With identical experts the asking gate and the pool's gate are two operating points, not a winner.** On known skills in new wordings the asking gate answers 92 % of them and keeps almost the oracle's precision (0.488 against 0.497), so it gets 0.447 right where the pool's gate gets 0.286; but a closed list cannot refuse a skill nobody taught (it hands it to the nearest card: 42-50 % of those messages are answered), where the vMF gate refuses more and the mix refuses most. Under equal weights the net score is a tie because the experts are right only about half the time even behind the oracle (net -0.006): abstaining is worth as much as answering. In this test bed the lever is expert competence, not routing (the same finding as the 2026-10-09 diagnosis).

Technical statement: a reader's flag rate on generic rewrites is not a calibrated estimate of its flag rate under the stream's own shift, in either direction (C-102); the asking gate dominates the density gate in coverage and correct answers on shifted known messages and is dominated on refusing unowned skills (C-103). Measured: the tables above (3 seeds, dev). Assumed: that the pattern holds beyond arithmetic for the gate comparison (only the identification and novelty tables of E-048 cover the topic domains).

## Caveats and what went wrong
- **F-49**: my arithmetic known set left out the taught skills the status scoring calls `grey`; the first G2 oracle row (0.263 correct) exposed it; G1, G3 and G2 were rerun and E-048 carries an erratum. The corrected numbers are the ones reported here.
- **Misses (F-50)**: I predicted the pool's own gate would route known messages at 0.33-0.55; among the messages it does not refuse it routes 0.936 right (it refuses the ones that look different). I predicted the NET order oracle > ask > mix > vmf; it is oracle > mix > vmf > ask. I predicted Solid-taught experts at 0.85-0.97; the grey ones are mediocre (0.56-0.87), so Solid experts average 0.753.
- The container is mine, not the growmoe pool: PhyloMoE's own experts are not used (that needs a vector design for language arithmetic and would add a second confound).
- Three seeds, three taught skills per seed, a fixed parser that reads operands in order of appearance (wrong for 'subtract 5 from 17'), arithmetic only for G2; the equal weighting of a wrong answer to a known and to an unowned message is my choice (the components are reported).
- Not done: the trust of a reader estimated from OUTCOMES (a stated answer or a label says whether a taught expert owns a message), which the layer's verifier contract already names.

## Consequence
The roster-only trust diagnostic is dropped (C-102). The layer keeps the asking reader and the density gate as two readers whose mix is a policy with a price, not a derived fact: refuse-more (mix) when a wrong answer to a message nobody owns is costly, answer-more (ask) when coverage is. Next, in this order: G4, estimate each reader's trust from verified outcomes in the stream's own wording (stated answers checked by competent experts, labels), test it on all four domains with predictions first; then the growmoe experts as the container (needs the vector design for language arithmetic); then self-discovery of cards. Nothing is frozen and the confirmatory seeds 400-429 are unspent.

## Evidence
- `research/knowledge_status/results/ROUTING_GENERAL_G3_150_152.md`, `ROUTING_GENERAL_G2_150_152.md`, `ROUTING_GENERAL_G1_150_152.md` (generated; corrected runs)
- `research/knowledge_status/results/routing_general/superseded_grey_filter/` (the first runs, kept)
- `research/knowledge_status/results/routing_general/` (g1, g2, g3 cells, cached model answers, rewrites, run logs)
- `research/knowledge_status/PROGRESS.md` (predictions, first results, the flaw, corrected results, in order)

## Reproduce
```
cd D:/shm_ks
python research/knowledge_status/routing_general/lab_general.py 150 152
python research/knowledge_status/routing_general/lab_shift.py 150 152
python research/knowledge_status/routing_general/lab_pool.py 150 152
python -m pytest tests/knowledge_status/test_rg_*.py
```

