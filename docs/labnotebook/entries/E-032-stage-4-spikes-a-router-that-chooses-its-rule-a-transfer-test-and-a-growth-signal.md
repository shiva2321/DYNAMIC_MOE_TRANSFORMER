# E-032 - Stage-4 spikes: a router that chooses its own rule, a held-out transfer test, and a growth signal

- **ID:** E-032
- **Date:** 2026-10-15
- **Track:** G
- **Type:** experiment (spikes that decide a design; throwaway scripts)
- **Status:** completed (exploratory; feeds the stage-4 spec)
- **Evidence grade:** B (script-generated and reproducible, 2 to 3 seeds per cell, NOT pre-registered)
- **Code / commit:** `research/growmoe_results/stage4_spikes/` (no product file touched)
- **People:** Claude designed, ran and wrote it at the user's request

## Question
The root-cause study (E-030) named four faults. Before designing repairs: (A) what must a learned router store and does one routing rule win everywhere, (B) can "the best existing expert already predicts the new labels" separate a new function from a known one, (C) can the need for width be seen on the live stream, and which signal tracks it?

## Prediction (expectations before the runs; nothing was pre-registered)
(A) a router trained on the stored prototypes recovers the routing loss on arithmetic and, with enough care, on every stream; (B) the transfer test has power; (C) a short paired trial on the live stream would confirm that widening helps, and a natural-expansion-style score would separate starved from adequate experts.
Outcome: (A) recovered on arithmetic, NOT on every stream (LDA alone loses 23 to 32 points on S1); (B) confirmed, with a needed margin; (C) both expectations REFUTED, a different cheap signal (the within-batch fit gap) tracks width.

## Method (how)
A: the saved final pools of the operator-by-operator runs of E-030 (4 pools) and fresh pools on the stage-3 streams and real text (dev seeds 100-102, the stage-3 arm A_set); routers fitted on nothing but each expert's signature prototypes: softmax regression, LDA (shared covariance), nearest class mean, 5-nearest vote, against the pool's density rule; selection by 5-fold cross-validation of the owner prediction (density unless another rule is better by more than 1 point). B: expert 0 of the saved pools (holds max, add, sub); clusters of 32, 64 and 128 held-out example rows per operator, exact permutation test at 0.01 against 512 reference rows. C: single experts of 16 to 512 units with the pool's recipe on the mixed-operator stream, 3 seeds: paired shadow trials (clone widened x2, random-in / zero-out and GradMax-style initialisation, same next batches, honest prequential loss), a simplified natural-expansion score, and three gap definitions (replay, recent window, within-batch) with reservoirs of 256 and 20,000.

## Results (tables copied from `research/growmoe_results/stage4_spikes/RESULTS_SPIKES.md`)
### A. What a router needs (routing accuracy / exact match %, 2 seeds)
| variant | stored prototypes per expert | softmax | LDA (shared covariance) | nearest class mean |
|---|---|---|---|---|
| default | 16 | 97.8 / 38.0 | 100.0 / 40.5 | 93.3 / 34.7 |
| default | 32 | 99.9 / 40.3 | 100.0 / 40.5 | 98.7 / 38.6 |
| default | 64 | 100.0 / 40.5 | 100.0 / 40.5 | 100.0 / 40.4 |
| default | 256 | 100.0 / 40.5 | 100.0 / 40.5 | 100.0 / 40.5 |
| default | reservoir 20000 | 78.6 / 26.1 | 100.0 / 40.5 | 100.0 / 40.5 |
| forced_spawn | 16 | 100.0 / 47.6 | 100.0 / 47.6 | 99.2 / 46.8 |
| forced_spawn | 32 | 100.0 / 47.6 | 100.0 / 47.6 | 100.0 / 47.6 |
| forced_spawn | 64 | 100.0 / 47.6 | 100.0 / 47.6 | 100.0 / 47.6 |
| forced_spawn | 256 | 100.0 / 47.6 | 100.0 / 47.6 | 100.0 / 47.6 |
| forced_spawn | reservoir 20000 | 100.0 / 47.6 | 100.0 / 47.6 | 100.0 / 47.6 |

### A3. Selection by cross-validation (19 pools and streams)
| unit | experts | CV owner accuracy: density / LDA / softmax / 5-NN | selected | exact %: shipped gate | LDA alone | selected | routing oracle |
|---|---|---|---|---|---|---|---|
| F-15-s100 | 15 | 100 / 100 / 100 / 100 | density | 51.2 | 51.3 | 51.2 | 51.3 |
| F-15-s101 | 15 | 100 / 100 / 100 / 100 | density | 49.7 | 49.8 | 49.7 | 49.8 |
| OV-0.6-s100 | 10 | 97 / 99 / 99 / 99 | lda | 43.0 | 45.0 | 45.0 | 45.1 |
| OV-0.6-s101 | 10 | 96 / 100 / 100 / 99 | softmax | 42.0 | 44.0 | 44.0 | 44.0 |
| OV-0.85-s100 | 10 | 58 / 89 / 91 / 64 | softmax | 27.7 | 38.0 | 38.4 | 41.5 |
| OV-0.85-s101 | 10 | 60 / 89 / 90 / 63 | softmax | 29.7 | 38.1 | 37.8 | 41.1 |
| S1-s100 | 5 | 92 / 74 / 72 / 89 | density | 79.0 | 56.3 | 79.0 | 84.8 |
| S1-s101 | 5 | 96 / 61 / 61 / 89 | density | 79.2 | 48.6 | 79.2 | 84.6 |
| S1-s102 | 4 | 91 / 60 / 60 / 85 | density | 78.5 | 46.2 | 78.5 | 80.4 |
| S2-5-s100 | 5 | 100 / 100 / 100 / 100 | density | 93.2 | 93.2 | 93.2 | 93.2 |
| S2-5-s101 | 2 | 100 / 100 / 100 / 100 | density | 90.4 | 90.4 | 90.4 | 90.4 |
| S2-5-s102 | 3 | 100 / 100 / 100 / 100 | density | 90.6 | 90.6 | 90.6 | 90.6 |
| arith-default-s0 | 4 | 88 / 100 / 100 / 85 | lda | 26.5 | 40.6 | 40.6 | 40.6 |
| arith-default-s1 | 4 | 87 / 100 / 100 / 84 | lda | 26.7 | 40.4 | 40.4 | 40.4 |
| arith-forced_spawn-s0 | 6 | 88 / 100 / 100 / 79 | lda | 34.5 | 46.9 | 46.9 | 46.9 |
| arith-forced_spawn-s1 | 6 | 86 / 100 / 100 / 78 | lda | 35.1 | 48.4 | 48.4 | 48.4 |
| banking-s100 | 5 | 95 / 93 / 90 / 93 | density | 81.1 | 78.1 | 81.1 | 85.6 |
| banking-s101 | 5 | 93 / 89 / 87 / 93 | density | 87.2 | 79.7 | 87.2 | 90.8 |
| banking-s102 | 4 | 95 / 92 / 90 / 94 | density | 82.9 | 77.9 | 82.9 | 86.3 |

### B. Transfer test (share of 100 clusters called a new function)
| cluster of | known to the parent | n = 32 | n = 64 | n = 128 |
|---|---|---|---|---|
| add | yes | 4.5 % | 12.5 % | 18.5 % |
| sub | yes | 0.0 % | 0.0 % | 0.0 % |
| mul | no | 100.0 % | 100.0 % | 100.0 % |
| mod | no | 100.0 % | 100.0 % | 100.0 % |
| div | no | 100.0 % | 100.0 % | 100.0 % |
| max | yes | 0.0 % | 0.0 % | 0.0 % |

### C. Growth signal and shadow trial (3 seeds)
| hidden units | gap at batch 900 | gap at batch 2300 | trial gain after 20 / 80 batches, random in / zero out (mean, 3 seeds, both checkpoints) | same, GradMax-style |
|---|---|---|---|---|
| 16 | 0.03 (0.01-0.05) | 0.03 (0.02-0.04) | -0.07 % / +0.15 % | -0.72 % / +0.06 % |
| 32 | 0.06 (0.05-0.07) | 0.07 (0.06-0.09) | +0.05 % / +0.03 % | -0.33 % / -0.21 % |
| 64 | 0.12 (0.11-0.12) | 0.14 (0.12-0.17) | -0.45 % / -0.35 % | -1.14 % / -0.85 % |
| 128 | 0.16 (0.16-0.17) | 0.21 (0.20-0.22) | -0.82 % / -0.54 % | -0.79 % / -0.65 % |
| 512 | 0.39 (0.37-0.40) | 0.46 (0.44-0.48) | -1.47 % / -0.93 % | -1.35 % / -0.76 % |

### C2. Which gap is robust
| hidden units | reservoir | (a) replay gap | (c) recent-window gap | (b) within-batch gap | prequential loss |
|---|---|---|---|---|---|
| 16 | 256 | 0.784 (0.781-0.789) | 0.182 (0.175-0.193) | 0.113 (0.103-0.120) | 1.574 |
| 32 | 256 | 0.890 (0.885-0.894) | 0.252 (0.250-0.254) | 0.150 (0.143-0.155) | 1.434 |
| 64 | 256 | 0.935 (0.930-0.942) | 0.328 (0.318-0.334) | 0.193 (0.188-0.200) | 1.309 |
| 128 | 256 | 0.958 (0.955-0.960) | 0.387 (0.384-0.390) | 0.239 (0.231-0.245) | 1.208 |
| 512 | 256 | 0.975 (0.970-0.979) | 0.503 (0.497-0.507) | 0.328 (0.324-0.332) | 1.128 |
| 16 | 20000 | 0.028 (0.012-0.050) | 0.101 (0.093-0.114) | 0.077 (0.073-0.083) | 1.205 |
| 32 | 20000 | 0.061 (0.049-0.071) | 0.165 (0.163-0.166) | 0.116 (0.111-0.120) | 1.125 |
| 64 | 20000 | 0.118 (0.110-0.123) | 0.247 (0.235-0.257) | 0.172 (0.158-0.185) | 1.037 |
| 128 | 20000 | 0.163 (0.161-0.165) | 0.327 (0.319-0.334) | 0.227 (0.220-0.231) | 0.971 |
| 512 | 20000 | 0.386 (0.368-0.404) | 0.570 (0.564-0.575) | 0.389 (0.383-0.398) | 0.994 |

## Interpretation (what we understood, and on what basis)
Plain words. A router needs very little: 16 stored inputs per expert are enough for a closed-form discriminant (LDA) to send every arithmetic input to the right expert, which raises exact match from 26.6 to 40.5 and from 34.8 to 47.6, equal to the routing oracle. But no rule wins everywhere: LDA alone is far worse than the pool's own density rule on S1 and on real text, where the experts hold several separate clusters. A router that cross-validates its candidates on the stored prototypes and keeps density unless another rule is clearly better was never worse than the shipped gate in 19 trials (worst +0.00, mean +4.0), gained 12 to 14 points on arithmetic and 8 to 11 on the heavily overlapping stream OV-0.85, and left 2 to 6 points on the table on S1 and banking77. The transfer test works as a question ("does the best expert already predict these labels?"): new operators were called new in all 300 clusters; but inside an expert that holds several operators, the known but harder operator add is called new in 4.5 to 18.5 % of clusters, growing with cluster size, so the test needs a practical-significance margin. Capacity need cannot be confirmed by a short live trial: after 80 batches widening a starved expert gains 0.03 to 0.15 % of the loss against a noise level of the same size; the trial can only veto. What does track width is the share of a fresh batch the expert can fit with its own training steps (0.08 to 0.11 at 16 units, 0.33 to 0.39 at 512), tight across seeds and, unlike the replay gap, independent of the reservoir.
Technical statement: measured owner accuracy of routing rules on stored prototypes, end-to-end exact match, exact-test conflict rates, paired prequential loss differences, relative fit gaps. Assumed: that prototypes are a fair sample of what an expert holds (they are a uniform reservoir of its assigned inputs) and that the stage-3 streams at dev seeds are representative.

## Caveats and what went wrong
- Exploratory, throwaway scripts; the growth signal is measured on the arithmetic mixed stream only (its calibration on other families is task 5 of the plan); the cross-validated selection is optimistic by construction (prototypes are training samples).
- The router is evaluated at answer time on the final pool; online refitting during training is not simulated (the router does not influence training in the design).
- The same file-name bug as in E-030 occurred again (`Path.with_suffix` on a stream name with a dot, "OV-0.6" and "OV-0.85" wrote one file); found from the report, fixed by using names without dots, the two overlapping streams were rerun.

## Consequence
Written into `docs/superpowers/specs/2026-10-15-routing-v2-stage4-design.md`: the router chooses its rule by cross-validation (4a), the spawn-or-extend decision becomes a transfer test with a margin (4b), growth is triggered by the within-batch fit gap with a guard (4c). Not adopted: the shadow trial as confirmation, GradMax initialisation, the simplified expansion score.

## Evidence
- `research/growmoe_results/stage4_spikes/RESULTS_SPIKES.md`
- `research/growmoe_results/stage4_spikes/spike_router.py`, `spike_router_streams.py`, `spike_router_select.py`, `spike_transfer.py`, `spike_growth_signal.py`, `spike_gap_definition.py`, `report_spikes.py`
- results: `research/growmoe_results/stage4_spikes/router_select/`, `growth_signal/`, `gap_definition/`, `spike_router.json`, `spike_transfer.json`

## Reproduce
```
python research/growmoe_results/stage4_spikes/spike_router.py          # needs the pools saved by the E-030 sequential runs (diagnosis/seqprobe3, git-ignored)
python research/growmoe_results/stage4_spikes/spike_router_select.py --streams arith,F-15,OV-0.6,OV-0.85,S1,S2-5,banking --seeds 100,101
python research/growmoe_results/stage4_spikes/spike_transfer.py
python research/growmoe_results/stage4_spikes/spike_growth_signal.py --seed 0
python research/growmoe_results/stage4_spikes/spike_gap_definition.py --seed 0
python research/growmoe_results/stage4_spikes/report_spikes.py
```
About 40 minutes of compute in total on a 12-thread PC.
