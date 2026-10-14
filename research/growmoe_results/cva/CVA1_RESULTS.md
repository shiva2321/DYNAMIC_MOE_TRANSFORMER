# CVA-1 Results: Gate and Evidence Verification

VERIFICATION (criteria fixed in advance by Claude, 2026-10-14)

Generated: 2026-10-14 02:11:36
Dimension: d = 128 everywhere | Simulation seeds: 0-19 (20 seeds)

---

## Experiment 1.1: Gate False Novelty

**Criterion**: For n_fit >= 128, mean share of inputs flagged <= 1.5*q and mean share of batches flagged <= 1.5*q. (For n_fit = 32 smallest attainable p-value 1/(n+1) > q=0.02, so 'never flags' is expected note).

| n_fit | q | Mean Inputs Flagged | Std Inputs | Mean Batches Flagged | Std Batches | Threshold (1.5*q) | Criterion | Verdict |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 32 | 0.02 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.030 | mean <= 0.030 | NOTE |
| 32 | 0.05 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.075 | mean <= 0.075 | NOTE |
| 64 | 0.02 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.030 | mean <= 0.030 | NOTE |
| 64 | 0.05 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.075 | mean <= 0.075 | NOTE |
| 128 | 0.02 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.030 | mean <= 0.030 | PASS |
| 128 | 0.05 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.075 | mean <= 0.075 | PASS |
| 256 | 0.02 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.030 | mean <= 0.030 | PASS |
| 256 | 0.05 | 0.00020 | 0.00068 | 0.00253 | 0.00859 | 0.075 | mean <= 0.075 | PASS |
| 512 | 0.02 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.030 | mean <= 0.030 | PASS |
| 512 | 0.05 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.075 | mean <= 0.075 | PASS |
| 1024 | 0.02 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.030 | mean <= 0.030 | PASS |
| 1024 | 0.05 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.075 | mean <= 0.075 | PASS |

---

## Experiment 1.2: Gate Power Against Overlap

**Criterion**: No pass/fail; report share of inputs flagged and AUROC of p_nov (A vs B).

| rho | Mean Flagged (B) | Std Flagged | Mean AUROC | Std AUROC | Criterion | Verdict |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 1.0000 | 0.0000 | 0.9973 | 0.0034 | Report | REPORT |
| 0.30 | 0.9993 | 0.0007 | 0.9972 | 0.0034 | Report | REPORT |
| 0.60 | 0.8742 | 0.0571 | 0.9880 | 0.0028 | Report | REPORT |
| 0.85 | 0.0369 | 0.0830 | 0.8237 | 0.0115 | Report | REPORT |
| 0.95 | 0.0000 | 0.0001 | 0.6235 | 0.0105 | Report | REPORT |

---

## Experiment 1.3: Evidence-Test False Spawns (Negative Control)

**Criterion**: Share of clusters with evidence > log(1/0.01) must be <= 0.03 in every cell. A cell above 0.03 is registered as DEFECT-EVIDENCE-UNCALIBRATED with its measured share.

| n_fit | Variant | Cluster Size m | Share Above Threshold | Mean Evidence | Std Evidence | Criterion | Verdict |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 64 | direct | 32 | 0.085 | -17.41 | 16.56 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 64 | direct | 64 | 0.290 | -7.51 | 23.65 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 64 | direct | 128 | 0.790 | 35.08 | 38.47 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 64 | direct | 256 | 1.000 | 140.81 | 57.38 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 64 | incremental | 32 | 0.085 | -17.41 | 16.56 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 64 | incremental | 64 | 0.290 | -7.51 | 23.65 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 64 | incremental | 128 | 0.790 | 35.08 | 38.47 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 64 | incremental | 256 | 1.000 | 140.81 | 57.38 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 256 | direct | 32 | 0.005 | -38.77 | 15.58 | share <= 0.03 | PASS |
| 256 | direct | 64 | 0.015 | -45.83 | 22.28 | share <= 0.03 | PASS |
| 256 | direct | 128 | 0.055 | -41.42 | 29.18 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 256 | direct | 256 | 0.335 | -11.21 | 40.85 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 256 | incremental | 32 | 0.005 | -38.77 | 15.58 | share <= 0.03 | PASS |
| 256 | incremental | 64 | 0.015 | -45.83 | 22.28 | share <= 0.03 | PASS |
| 256 | incremental | 128 | 0.055 | -41.42 | 29.18 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 256 | incremental | 256 | 0.335 | -11.21 | 40.85 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 1024 | direct | 32 | 0.005 | -35.12 | 16.06 | share <= 0.03 | PASS |
| 1024 | direct | 64 | 0.025 | -39.96 | 22.93 | share <= 0.03 | PASS |
| 1024 | direct | 128 | 0.125 | -34.70 | 31.32 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 1024 | direct | 256 | 0.465 | 3.98 | 46.50 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 1024 | incremental | 32 | 0.005 | -35.12 | 16.06 | share <= 0.03 | PASS |
| 1024 | incremental | 64 | 0.025 | -39.96 | 22.93 | share <= 0.03 | PASS |
| 1024 | incremental | 128 | 0.125 | -34.70 | 31.32 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 1024 | incremental | 256 | 0.465 | 3.98 | 46.50 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |

---

## Experiment 1.3: Evidence-Test True Spawns (Positive Control)

**Criterion**: Share above threshold >= 0.99 at rho 0.0 and 0.3 for m >= 64. If this control fails, formula is wrong: STOP and report.

| rho | n_fit | Variant | Cluster Size m | Share Above Threshold | Mean Evidence | Std Evidence | Criterion | Verdict |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 256 | direct | 32 | 1.000 | 598.46 | 29.25 | Report | REPORT |
| 0.00 | 256 | direct | 64 | 1.000 | 1226.19 | 39.54 | >= 0.99 | PASS |
| 0.00 | 256 | direct | 128 | 1.000 | 2505.09 | 64.45 | >= 0.99 | PASS |
| 0.00 | 256 | direct | 256 | 1.000 | 5092.25 | 97.07 | >= 0.99 | PASS |
| 0.30 | 256 | direct | 32 | 1.000 | 401.61 | 25.92 | Report | REPORT |
| 0.30 | 256 | direct | 64 | 1.000 | 832.47 | 35.93 | >= 0.99 | PASS |
| 0.30 | 256 | direct | 128 | 1.000 | 1718.08 | 58.77 | >= 0.99 | PASS |
| 0.30 | 256 | direct | 256 | 1.000 | 3515.87 | 89.10 | >= 0.99 | PASS |
| 0.60 | 256 | direct | 32 | 1.000 | 209.28 | 22.16 | Report | REPORT |
| 0.60 | 256 | direct | 64 | 1.000 | 448.40 | 31.14 | Report | REPORT |
| 0.60 | 256 | direct | 128 | 1.000 | 949.86 | 49.36 | Report | REPORT |
| 0.60 | 256 | direct | 256 | 1.000 | 1976.47 | 76.01 | Report | REPORT |
| 0.85 | 256 | direct | 32 | 0.990 | 52.74 | 18.70 | Report | REPORT |
| 0.85 | 256 | direct | 64 | 1.000 | 136.27 | 26.00 | Report | REPORT |
| 0.85 | 256 | direct | 128 | 1.000 | 324.88 | 37.63 | Report | REPORT |
| 0.85 | 256 | direct | 256 | 1.000 | 723.63 | 57.10 | Report | REPORT |

---

## Experiment 1.4: Evidence-Test False Spawns with Second Live Expert (Regime C at rho = 0)

**Criterion**: Share above threshold <= 0.03 in every cell.

| n_fit | Variant | Cluster Size m | Share Above Threshold | Mean Evidence | Std Evidence | Criterion | Verdict |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 64 | direct | 32 | 0.080 | -18.52 | 16.51 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 64 | direct | 64 | 0.290 | -8.32 | 23.78 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 64 | direct | 128 | 0.770 | 34.21 | 36.71 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 64 | direct | 256 | 0.995 | 141.98 | 56.52 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 64 | incremental | 32 | 0.080 | -18.52 | 16.51 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 64 | incremental | 64 | 0.290 | -8.32 | 23.78 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 64 | incremental | 128 | 0.770 | 34.21 | 36.71 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 64 | incremental | 256 | 0.995 | 141.98 | 56.52 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 256 | direct | 32 | 0.000 | -38.98 | 15.59 | share <= 0.03 | PASS |
| 256 | direct | 64 | 0.010 | -45.49 | 22.14 | share <= 0.03 | PASS |
| 256 | direct | 128 | 0.040 | -43.51 | 29.96 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 256 | direct | 256 | 0.320 | -11.79 | 40.86 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 256 | incremental | 32 | 0.000 | -38.25 | 16.36 | share <= 0.03 | PASS |
| 256 | incremental | 64 | 0.010 | -44.55 | 22.63 | share <= 0.03 | PASS |
| 256 | incremental | 128 | 0.055 | -41.28 | 30.26 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 256 | incremental | 256 | 0.405 | -8.31 | 46.93 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 1024 | direct | 32 | 0.010 | -36.64 | 15.42 | share <= 0.03 | PASS |
| 1024 | direct | 64 | 0.025 | -41.87 | 21.94 | share <= 0.03 | PASS |
| 1024 | direct | 128 | 0.105 | -36.38 | 31.73 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 1024 | direct | 256 | 0.475 | 3.50 | 44.03 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 1024 | incremental | 32 | 0.020 | -36.89 | 15.94 | share <= 0.03 | PASS |
| 1024 | incremental | 64 | 0.030 | -42.48 | 23.15 | share <= 0.03 | PASS |
| 1024 | incremental | 128 | 0.095 | -38.01 | 31.35 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
| 1024 | incremental | 256 | 0.500 | 0.60 | 45.80 | share <= 0.03 | DEFECT-EVIDENCE-UNCALIBRATED |
