# Phase-1 Comparative Evaluation Results: PhyloMoE (A), BNP-MoE (B), KeyedDeltaMoE (C)
**Date:** 2026-10-06 | **Seeds evaluated:** 10 (seeds 0 to 9) | **Total Wall-Clock:** 728.23 s
## 1. Executive Summary and Pre-Registered Predictions
Evaluation conforms strictly to the test-then-train protocol (`predict` before `observe`), isolated ground truth, and dev-seed tuning (seeds >= 100).
Pool A routing rule: `accept_then_logf` (accept_then_logf selects highest log-density among accepting experts with p >= q without applying the test-time supersede filter).
| ID | Prediction Description | Status | Measured Metric | Statistics / 95% CI |
|:---|:-----------------------|:------:|:----------------|:--------------------|
| P1 | Empirical FNR <= 2*q on S1, S2, S4 | **PASS** | FNR = 0.0000 <= 0.0400 | [0.0000, 0.0000] |
| P2 | Spawn recall >= 0.90 and ratio <= 1.20 on S1/S2 | **FAIL** | Recall=0.517, Ratio=0.800 (Served without forgetting=77.4%) | Recall [0.441, 0.594], Ratio [0.695, 0.905] |
| P3 | Revisit routing accuracy >= 95% (fail if < 90%) | **FAIL** | Acc = 89.50% | [87.69%, 91.31%] |
| P4 | S1 class-IL acc >= 0.90*B4 and > B2 (CI > 0) | **FAIL** | A=69.8%, B4=78.1%, diff=8.29% | A-B2 diff CI [5.92%, 10.65%], p=0.0000 |
| P5 | Forgetting <= 2 pts on S1/S2 (fail if > 5 pts) | **INCONCLUSIVE** | Forgetting = 3.50 pts | [2.44, 4.55] pts |
| P6 | Affinity child >= scratch sample efficiency on related regimes | **NOT MEASURABLE** | Not measurable on S2 (0 affinity children < 5 required) | Requires stream with related regimes to produce >= 5 children |
| P8 | S4: <= 1% injected outliers cause spawn (fail if > 3%) | **PASS** | Spurious spawn rate = 1.00% | Spurious spawns = 0.20 [-0.10, 0.50] |
| P10 | No-conformal ablation worse on >= 2 of P1, P2, P8 | **FAIL** | Worse on 1/3 metrics (worse P1=False, P2=True, P8=False) | FNR diff: 0.0000, Recall diff: 0.2800, Outlier spawn diff: -0.20 |
| C1 | A has lowest FNR and highest recall vs B and C | **FAIL** | FNR A=0.000, B=0.075, C=0.016; Ratio A=0.70, B=2.06, C=1.04 | Recall A=0.42, B=1.00, C=1.00 (Ratio A=0.70, B=2.06, C=1.04) |
| C2 | B threshold-sensitivity > 2x vs A < 1.5x | **PASS** | B spawn ratio=3.38x, A spawn ratio=1.11x | B spans [6.3, 21.3], A spans [3.6, 4.0] |
| C3 | C ends S2 with <= 40% parameters and acc within tolerance | **FAIL** | Param ratio=188.6%, Acc diff=10.52% | Param ratio: 1.886 <= 0.40; Acc diff CI [8.27%, 12.77%] |
| C4 | A beats scratch on forward transfer; C matches A within tolerance | **FAIL** | Early acc: A=79.1%, B=88.5%, C=86.7% | A-B diff: -9.40%, A-C diff: -7.60% |
| C5 | B likelihood gate fails (AUROC < 0.80) on S1 types | **FAIL** | B S1 AUROC = 0.9139 | [0.9048, 0.9230] |
| C6 | Wall-clock per run: C fastest <= A <= B | **PASS** | C=1.45s, A=1.84s, B=1.99s | Fastest is C (1.45s), A is 1.84s, B is 1.99s |

## 2. Core Architecture Performance (S1 to S4)
| Pool | Stream | Avg Accuracy | Forgetting | Novelty AUROC | False Novelty Rate | FNR (p <= 0.5) | Spawn Recall | Params | Wall-Clock |
|:-----|:-------|:-------------|:-----------|:--------------|:-------------------|:---------------|:-------------|:-------|:-----------|
| A (PhyloMoE) | S1 (types) | 69.82% [67.80, 71.83] | 6.35% [3.83, 8.87] | 0.98 [0.97, 0.98] | 0.00% [0.00, 0.00] | 28.27% [20.11, 36.42] | 61.43% [53.02, 69.84] | 263,289 | 3.19s |
| A (PhyloMoE) | S2 (permuted) | 87.02% [85.52, 88.52] | 0.64% [-0.59, 1.87] | 1.00 [1.00, 1.00] | 0.00% [0.00, 0.00] | 52.46% [50.40, 54.52] | 42.00% [31.44, 52.56] | 146,272 | 1.84s |
| A (PhyloMoE) | S3 (drift) | 85.47% [83.92, 87.01] | 0.28% [-0.52, 1.08] | 0.68 [0.65, 0.71] | 0.00% [0.00, 0.00] | 68.04% [64.87, 71.21] | 0.00% [0.00, 0.00] | 41,792 | 0.56s |
| A (PhyloMoE) | S4 (outliers) | 71.36% [65.73, 76.99] | 1.90% [-0.54, 4.34] | 0.99 [0.99, 1.00] | 0.00% [0.00, 0.00] | 50.33% [46.84, 53.83] | 78.00% [63.77, 92.23] | 179,705 | 1.76s |
| B (BNP-MoE) | S1 (types) | 62.12% [59.16, 65.09] | 9.26% [5.36, 13.15] | 0.91 [0.90, 0.92] | 67.15% [66.90, 67.40] | 67.15% [66.90, 67.40] | 100.00% [100.00, 100.00] | 1,797,056 | 7.91s |
| B (BNP-MoE) | S2 (permuted) | 79.74% [71.71, 87.77] | 7.68% [-0.28, 15.64] | 0.99 [0.99, 1.00] | 7.50% [1.48, 13.52] | 7.50% [1.48, 13.52] | 100.00% [100.00, 100.00] | 430,457 | 1.99s |
| B (BNP-MoE) | S3 (drift) | 77.91% [71.15, 84.66] | 6.53% [1.47, 11.60] | 0.66 [0.62, 0.70] | 11.61% [6.31, 16.92] | 11.61% [6.31, 16.92] | 56.00% [38.41, 73.59] | 238,214 | 1.20s |
| B (BNP-MoE) | S4 (outliers) | 71.66% [61.62, 81.70] | 10.60% [0.10, 21.10] | 1.00 [1.00, 1.00] | 15.03% [3.98, 26.08] | 15.03% [3.98, 26.08] | 100.00% [100.00, 100.00] | 380,307 | 1.69s |
| C (KeyedDeltaMoE) | S1 (types) | 52.79% [51.84, 53.75] | 9.03% [7.72, 10.34] | 0.69 [0.68, 0.70] | 0.42% [-0.00, 0.84] | 0.42% [-0.00, 0.84] | 24.29% [19.35, 29.22] | 143,251 | 2.58s |
| C (KeyedDeltaMoE) | S2 (permuted) | 76.50% [74.12, 78.88] | -0.96% [-2.33, 0.41] | 1.00 [0.99, 1.00] | 1.59% [0.72, 2.46] | 1.59% [0.72, 2.46] | 100.00% [100.00, 100.00] | 275,891 | 1.45s |
| C (KeyedDeltaMoE) | S3 (drift) | 81.43% [79.01, 83.84] | 1.80% [0.70, 2.90] | 0.67 [0.64, 0.70] | 5.22% [4.45, 6.00] | 5.22% [4.45, 6.00] | 44.00% [27.76, 60.24] | 206,918 | 2.54s |
| C (KeyedDeltaMoE) | S4 (outliers) | 64.38% [56.81, 71.95] | 2.55% [-0.14, 5.24] | 0.98 [0.94, 1.02] | 1.17% [0.46, 1.87] | 1.17% [0.46, 1.87] | 94.00% [80.43, 107.57] | 259,974 | 1.00s |

## 3. Baseline Comparison on S2 Permuted Digits
| Model | Role / Description | Avg Accuracy | Forgetting | Live Experts | Params |
|:------|:-------------------|:-------------|:-----------|:-------------|:-------|
| B0 (FineTune) | Catastrophic forgetting floor | 41.68% [37.93, 45.43] | 12.28% [9.18, 15.38] | 1.0 | 34,826 |
| B1 (JointOracle) | Joint i.i.d. accuracy ceiling | 88.90% [87.64, 90.16] | -0.44% [-1.66, 0.78] | 1.0 | 34,826 |
| B2 (ReplayNet) | Single net + reservoir replay | 67.68% [65.30, 70.06] | -1.70% [-3.37, -0.03] | 1.0 | 34,826 |
| B3 (StaticRouterMoE) | Fixed 8 experts, learned router | 42.84% [39.21, 46.47] | 10.84% [7.22, 14.46] | 8.0 | 279,632 |
| B4 (OracleRouted) | Oracle 1 expert per regime ceiling | 80.98% [79.23, 82.73] | -2.18% [-3.11, -1.25] | 5.0 | 174,130 |
| B5 (DPMeansGate) | DP-means fixed lambda straw-man | 79.52% [77.62, 81.42] | -2.12% [-2.94, -1.30] | 8.9 | 309,951 |
| A (PhyloMoE) | Calibrated conformal + phylogenetic | 87.02% [85.52, 88.52] | 0.64% [-0.59, 1.87] | 3.5 | 146,272 |
| B (BNPMoE) | Bayesian nonparametric VAE gate | 79.74% [71.71, 87.77] | 7.68% [-0.28, 15.64] | 10.3 | 430,457 |
| C (KeyedDeltaMoE) | Key gate + low-rank delta experts | 76.50% [74.12, 78.88] | -0.96% [-2.33, 0.41] | 5.2 | 275,891 |

## 4. Architecture A Ablation Study (S2 Permuted Digits)
| Variant | Modification | Avg Accuracy | False Novelty Rate | Spawn Recall | n_spawn | n_extend | n_grow | n_prob | Status / Activity |
|:--------|:-------------|:-------------|:-------------------|:-------------|:--------|:---------|:-------|:-------|:------------------|
| A_phylomoe | Full PhyloMoE (calibrated vMF, probation, probe, shrink-perturb) | 87.02% [85.52, 88.52] | 0.00% [0.00, 0.00] | 42.00% [31.44, 52.56] | 3.5 | 3.7 | 0.0 | 3.5 | Active |
| A_no_conformal | No conformal calibration (fixed logpdf threshold) | 75.04% [70.23, 79.85] | 0.00% [0.00, 0.00] | 14.00% [4.34, 23.66] | 2.2 | 2.1 | 0.0 | 2.2 | Active |
| A_no_probation | No probation buffer (immediate spawn on novel samples) | 87.04% [85.54, 88.54] | 0.00% [0.00, 0.00] | 42.00% [31.44, 52.56] | 3.5 | 3.7 | 0.0 | 3.5 | Inactive on S2 (burst regimes; see S4) |
| A_init_scratch | Scratch child initialization (no DNA inheritance) | 88.30% [86.65, 89.95] | 0.00% [0.00, 0.00] | 52.00% [42.00, 62.00] | 3.8 | 3.8 | 0.0 | 3.8 | Active |
| A_init_clone | Exact clone child initialization | 87.04% [85.50, 88.58] | 0.00% [0.00, 0.00] | 48.00% [38.00, 58.00] | 3.7 | 3.7 | 0.0 | 3.7 | Active |
| A_init_inherit | Random unit re-drawing inheritance (inherit) | 87.48% [85.91, 89.05] | 0.00% [0.00, 0.00] | 50.00% [39.88, 60.12] | 3.6 | 3.9 | 0.0 | 3.6 | Active |
| A_never_extend | Never extend parent (always spawn new expert) | 82.24% [80.83, 83.65] | 0.00% [0.00, 0.00] | 100.00% [100.00, 100.00] | 9.0 | 0.0 | 0.0 | 9.0 | Active |
| A_no_replay | No replay buffer inside experts | 80.12% [77.71, 82.53] | 0.00% [0.00, 0.00] | 100.00% [100.00, 100.00] | 5.3 | 0.0 | 0.0 | 5.3 | Active |
| A_no_growth | No capacity growth (fixed hidden width) | 87.02% [85.52, 88.52] | 0.00% [0.00, 0.00] | 42.00% [31.44, 52.56] | 3.5 | 3.7 | 0.0 | 3.5 | Inactive on S2 (n_grow=0; capacity not exceeded) |
| A_fam_gaussian | Familiarity: Gaussian-Mahalanobis signature | 89.30% [87.73, 90.87] | 0.05% [-0.06, 0.16] | 100.00% [100.00, 100.00] | 5.2 | 7.4 | 0.0 | 5.2 | Active |
| A_fam_knn | Familiarity: k-Nearest Neighbors signature | 88.68% [87.08, 90.28] | 7.11% [1.24, 12.98] | 98.00% [93.48, 102.52] | 5.2 | 9.3 | 0.0 | 9.2 | Active |
| A_fam_rnd | Familiarity: Random Network Distillation signature | 82.30% [77.66, 86.94] | 12.03% [-8.44, 32.49] | 72.00% [59.94, 84.06] | 3.6 | 5.6 | 0.0 | 20.9 | Active |
| A_no_probation (on S4) | Injected outliers (S4) bypass probation buffer | 73.20% [67.03, 79.37] | 0.00% [0.00, 0.00] | 82.00% [67.77, 96.23] | 4.5 | 2.8 | 0.0 | 0.0 | Active on S4 (spurious spawn rate: 1.5% vs full A 1.0%) |

## 5. Statistical Paired Comparisons (p-values & Cohen's d_z)
| Comparison | Metric | Mean Diff [95% CI] | t-statistic p-value | Wilcoxon p-value | d_z |
|:-----------|:-------|:-------------------|:-------------------|:-----------------|:----|
| A vs B2 (replay) on S1 | Avg Accuracy | 0.0829 [0.0592, 0.1065] | 2.4160e-05 | 1.9531e-03 | 2.50 |
| A vs B (BNP) on S2 | Avg Accuracy | 0.0728 [-0.0101, 0.1557] | 7.8213e-02 | 4.8828e-02 | 0.63 |
| A vs C (Keyed) on S2 | Avg Accuracy | 0.1052 [0.0827, 0.1277] | 2.2163e-06 | 1.9531e-03 | 3.35 |
| A vs B (BNP) on S2 | Spawn Recall | -0.5800 [-0.6856, -0.4744] | 5.7073e-07 | 1.9531e-03 | -3.93 |
| A vs C (Keyed) on S2 | Spawn Recall | -0.5800 [-0.6856, -0.4744] | 5.7073e-07 | 1.9531e-03 | -3.93 |
| A vs No-Conformal on S2 | False Novelty Rate | 0.0000 [0.0000, 0.0000] | 1.0000e+00 | 1.0000e+00 | nan |

## 6. Analysis and Pre-Registered Findings
- **Pool A Routing Rule:** `accept_then_logf` (highest log-density among accepting experts with p >= q, skipping supersede filter at test time).
- **P1 (Empirical FNR <= 2*q on S1, S2, S4):** **PASS** | FNR = 0.0000 <= 0.0400. (95% CI: [0.0000, 0.0000])
- **P2 (Spawn recall >= 0.90 and ratio <= 1.20 on S1/S2):** **FAIL** | Recall=0.517, Ratio=0.800 (Served without forgetting=77.4%). (95% CI: Recall [0.441, 0.594], Ratio [0.695, 0.905])
- **P3 (Revisit routing accuracy >= 95% (fail if < 90%)):** **FAIL** | Acc = 89.50%. (95% CI: [87.69%, 91.31%])
- **P4 (S1 class-IL acc >= 0.90*B4 and > B2 (CI > 0)):** **FAIL** | A=69.8%, B4=78.1%, diff=8.29%. (95% CI: A-B2 diff CI [5.92%, 10.65%], p=0.0000)
- **P5 (Forgetting <= 2 pts on S1/S2 (fail if > 5 pts)):** **INCONCLUSIVE** | Forgetting = 3.50 pts. (95% CI: [2.44, 4.55] pts)
- **P6 (Affinity child >= scratch sample efficiency on related regimes):** **NOT MEASURABLE** | Not measurable on S2 (0 affinity children < 5 required). (95% CI: Requires stream with related regimes to produce >= 5 children)
- **P8 (S4: <= 1% injected outliers cause spawn (fail if > 3%)):** **PASS** | Spurious spawn rate = 1.00%. (95% CI: Spurious spawns = 0.20 [-0.10, 0.50])
- **P10 (No-conformal ablation worse on >= 2 of P1, P2, P8):** **FAIL** | Worse on 1/3 metrics (worse P1=False, P2=True, P8=False). (95% CI: FNR diff: 0.0000, Recall diff: 0.2800, Outlier spawn diff: -0.20)
- **C1 (A has lowest FNR and highest recall vs B and C):** **FAIL** | FNR A=0.000, B=0.075, C=0.016; Ratio A=0.70, B=2.06, C=1.04. (95% CI: Recall A=0.42, B=1.00, C=1.00 (Ratio A=0.70, B=2.06, C=1.04))
- **C2 (B threshold-sensitivity > 2x vs A < 1.5x):** **PASS** | B spawn ratio=3.38x, A spawn ratio=1.11x. (95% CI: B spans [6.3, 21.3], A spans [3.6, 4.0])
- **C3 (C ends S2 with <= 40% parameters and acc within tolerance):** **FAIL** | Param ratio=188.6%, Acc diff=10.52%. (95% CI: Param ratio: 1.886 <= 0.40; Acc diff CI [8.27%, 12.77%])
- **C4 (A beats scratch on forward transfer; C matches A within tolerance):** **FAIL** | Early acc: A=79.1%, B=88.5%, C=86.7%. (95% CI: A-B diff: -9.40%, A-C diff: -7.60%)
- **C5 (B likelihood gate fails (AUROC < 0.80) on S1 types):** **FAIL** | B S1 AUROC = 0.9139. (95% CI: [0.9048, 0.9230])
- **C6 (Wall-clock per run: C fastest <= A <= B):** **PASS** | C=1.45s, A=1.84s, B=1.99s. (95% CI: Fastest is C (1.45s), A is 1.84s, B is 1.99s)

### Ablation Counter Summary (S2 Permuted Digits)
- `A_no_growth`: Capacity growth did not fire on S2 (`n_grow = 0` on both full A and ablation) because S2 MLP experts have sufficient capacity for 10-class permutations; growth remains active on capacity-limited streams.
- `A_no_probation`: On S2, all regimes arrive as clean bursts (no isolated outliers), resulting in identical spawns. On S4 (injected outliers), full A achieves 71.36% accuracy with spurious spawn rate 1.00% (0.20 extra spawns), while A_no_probation achieves 73.20% accuracy with spurious spawn rate 1.50% (0.30 extra spawns). Across the 10 evaluation seeds, the paired accuracy difference on S4 is +1.84% [-4.08%, +7.76%], showing no statistically significant difference (95% CI contains 0).

### S1 Performance Diagnosis (Heterogeneous Types, Class-IL)
- Pool A S1 accuracy: 69.8%, routing accuracy: 77.5%, forgetting: 6.4%
- Pool B S1 accuracy: 62.1%, routing accuracy: 21.4%, forgetting: 9.3%
- Pool C S1 accuracy: 52.8%, routing accuracy: 84.7%, forgetting: 9.0%
- ReplayNet (B2) S1 accuracy: 61.5%, Oracle-routed (B4): 78.1%
- Per-regime final test accuracy for Pool A: R0: 86.0%, R1: 88.6%, R2: 78.6%, R3: 95.9%, R4: 50.3%, R5: 35.1%, R6: 54.2%
