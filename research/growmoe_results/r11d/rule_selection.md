# GrowMoE R11d: Routing Rule Selection and Filter Characterization Report
**Date:** 2026-10-06 | **Benchmark Configuration:** q = 0.02 everywhere | **Dev Seeds:** 100-109 | **Test Seeds:** 0-9
## 1. Candidate Rule Comparison Across Streams
| Stream | Split | Rule Variant | Accuracy Mean [95% CI] | Diff vs Default [95% CI] | p-value (t-test) |
|:-------|:------|:-------------|:-----------------------|:-------------------------|:-----------------|
| S1 | Dev (100-109) | `default` | 58.07% [52.69%, 63.45%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S1 | Dev (100-109) | `default_nofilter` | 70.77% [68.88%, 72.67%] | +12.70% [+6.48%, +18.93%] | 1.2632e-03 |
| S1 | Dev (100-109) | `accept_then_logf` | 70.81% [68.92%, 72.69%] | +12.74% [+6.54%, +18.95%] | 1.2094e-03 |
| S1 | Dev (100-109) | `atl_filter` | 58.10% [52.70%, 63.51%] | +0.04% [-0.03%, +0.10%] | 2.2287e-01 |
| S1 | Dev (100-109) | `argmax_logf` | 71.51% [69.43%, 73.60%] | +13.45% [+7.11%, +19.78%] | 9.7267e-04 |
| S1 | Dev (100-109) | `nearest_proto` | 70.97% [69.39%, 72.55%] | +12.90% [+6.67%, +19.14%] | 1.1484e-03 |
| S2 | Dev (100-109) | `default` | 87.30% [85.42%, 89.18%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S2 | Dev (100-109) | `default_nofilter` | 86.56% [84.31%, 88.81%] | -0.74% [-1.62%, +0.14%] | 9.1007e-02 |
| S2 | Dev (100-109) | `accept_then_logf` | 86.58% [84.35%, 88.81%] | -0.72% [-1.59%, +0.15%] | 9.4656e-02 |
| S2 | Dev (100-109) | `atl_filter` | 87.32% [85.47%, 89.17%] | +0.02% [-0.03%, +0.07%] | 3.4344e-01 |
| S2 | Dev (100-109) | `argmax_logf` | 86.58% [84.35%, 88.81%] | -0.72% [-1.59%, +0.15%] | 9.4656e-02 |
| S2 | Dev (100-109) | `nearest_proto` | 86.58% [84.35%, 88.81%] | -0.72% [-1.61%, +0.17%] | 9.8862e-02 |
| S3 | Dev (100-109) | `default` | 86.00% [84.06%, 87.94%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S3 | Dev (100-109) | `default_nofilter` | 86.00% [84.06%, 87.94%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S3 | Dev (100-109) | `accept_then_logf` | 86.00% [84.06%, 87.94%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S3 | Dev (100-109) | `atl_filter` | 86.00% [84.06%, 87.94%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S3 | Dev (100-109) | `argmax_logf` | 86.00% [84.06%, 87.94%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S3 | Dev (100-109) | `nearest_proto` | 86.00% [84.06%, 87.94%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S4 | Dev (100-109) | `default` | 74.68% [68.22%, 81.14%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S4 | Dev (100-109) | `default_nofilter` | 73.86% [67.56%, 80.16%] | -0.82% [-1.89%, +0.25%] | 1.1750e-01 |
| S4 | Dev (100-109) | `accept_then_logf` | 74.14% [67.72%, 80.56%] | -0.54% [-1.66%, +0.58%] | 3.0233e-01 |
| S4 | Dev (100-109) | `atl_filter` | 74.96% [68.39%, 81.53%] | +0.28% [+0.11%, +0.45%] | 4.4048e-03 |
| S4 | Dev (100-109) | `argmax_logf` | 74.36% [68.00%, 80.72%] | -0.32% [-1.45%, +0.81%] | 5.3879e-01 |
| S4 | Dev (100-109) | `nearest_proto` | 74.36% [67.99%, 80.73%] | -0.32% [-1.42%, +0.78%] | 5.2861e-01 |
| S1 | Test (0-9) | `default` | 55.96% [48.23%, 63.70%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S1 | Test (0-9) | `default_nofilter` | 69.75% [67.74%, 71.77%] | +13.79% [+5.89%, +21.69%] | 3.3650e-03 |
| S1 | Test (0-9) | `accept_then_logf` | 69.82% [67.80%, 71.83%] | +13.86% [+6.02%, +21.69%] | 3.1106e-03 |
| S1 | Test (0-9) | `atl_filter` | 56.03% [48.23%, 63.83%] | +0.07% [-0.08%, +0.22%] | 3.4344e-01 |
| S1 | Test (0-9) | `argmax_logf` | 69.98% [67.84%, 72.13%] | +14.02% [+6.01%, +22.03%] | 3.3020e-03 |
| S1 | Test (0-9) | `nearest_proto` | 69.14% [66.71%, 71.57%] | +13.18% [+5.44%, +20.93%] | 3.9048e-03 |
| S2 | Test (0-9) | `default` | 87.30% [85.87%, 88.73%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S2 | Test (0-9) | `default_nofilter` | 87.02% [85.52%, 88.52%] | -0.28% [-0.77%, +0.21%] | 2.3001e-01 |
| S2 | Test (0-9) | `accept_then_logf` | 87.02% [85.52%, 88.52%] | -0.28% [-0.77%, +0.21%] | 2.3001e-01 |
| S2 | Test (0-9) | `atl_filter` | 87.34% [85.95%, 88.73%] | +0.04% [-0.05%, +0.13%] | 3.4344e-01 |
| S2 | Test (0-9) | `argmax_logf` | 87.08% [85.61%, 88.55%] | -0.22% [-0.67%, +0.23%] | 2.9709e-01 |
| S2 | Test (0-9) | `nearest_proto` | 87.06% [85.58%, 88.54%] | -0.24% [-0.73%, +0.25%] | 3.0093e-01 |
| S3 | Test (0-9) | `default` | 85.47% [83.92%, 87.01%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S3 | Test (0-9) | `default_nofilter` | 85.47% [83.92%, 87.01%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S3 | Test (0-9) | `accept_then_logf` | 85.47% [83.92%, 87.01%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S3 | Test (0-9) | `atl_filter` | 85.47% [83.92%, 87.01%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S3 | Test (0-9) | `argmax_logf` | 85.47% [83.92%, 87.01%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S3 | Test (0-9) | `nearest_proto` | 85.47% [83.92%, 87.01%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S4 | Test (0-9) | `default` | 71.30% [65.11%, 77.49%] | +0.00% [+0.00%, +0.00%] | 1.0000e+00 |
| S4 | Test (0-9) | `default_nofilter` | 71.28% [65.69%, 76.87%] | -0.02% [-1.05%, +1.01%] | 9.6587e-01 |
| S4 | Test (0-9) | `accept_then_logf` | 71.36% [65.73%, 76.99%] | +0.06% [-1.00%, +1.12%] | 9.0133e-01 |
| S4 | Test (0-9) | `atl_filter` | 71.38% [65.16%, 77.60%] | +0.08% [-0.16%, +0.32%] | 4.6203e-01 |
| S4 | Test (0-9) | `argmax_logf` | 71.46% [65.85%, 77.07%] | +0.16% [-0.89%, +1.21%] | 7.3845e-01 |
| S4 | Test (0-9) | `nearest_proto` | 71.22% [65.51%, 76.93%] | -0.08% [-1.10%, +0.94%] | 8.6325e-01 |

## 2. Supersede Filter Isolation (Test Seeds, Stream S1)
| Rule Variant | S1 Test Accuracy | Filter Active at Test | Diff vs Default |
|:-------------|:-----------------|:----------------------|:----------------|
| `default` | 55.96% [48.23%, 63.70%] | Yes | +0.00% [+0.00%, +0.00%] |
| `default_nofilter` | 69.75% [67.74%, 71.77%] | No | +13.79% [+5.89%, +21.69%] |
| `accept_then_logf` | 69.82% [67.80%, 71.83%] | No | +13.86% [+6.02%, +21.69%] |
| `atl_filter` | 56.03% [48.23%, 63.83%] | Yes | +0.07% [-0.08%, +0.22%] |

## 3. Unknown-Flag Identity Check
Verification that `Prediction.unknown` is identical across all routing rules (since unknown is determined by conformal FDR threshold q prior to routing):
- **Unknown flags identical across all rules, streams, and seeds:** `True`

## 4. Concept Remap Characterization (L3 Limitation)
Evaluation on 5 seeds x 200 inputs where identical inputs receive remapped labels (genuine competence alarm created successor):
- `default` routing: **100.0%** routed to successor (100% adherence to supersede chain)
- `accept_then_logf` routing: **49.9%** routed to successor (per seed: 54.0%, 52.5%, 48.5%, 47.0%, 47.5%; stale parent retains remainder; exact ties go to older expert)
- `argmax_logf` routing: **48.3%** routed to successor (per seed: 51.5%, 51.5%, 48.0%, 47.0%, 43.5%)

## 5. Training-Time Assignment Decision (Dev Seeds 100-109)
| Stream | Predict-Only (Filter in Training) | Filter Disabled in Training & Test | Paired Difference [95% CI] | p-value |
|:-------|:---------------------------------|:-----------------------------------|:---------------------------|:--------|
| S1 | 70.81% [68.92%, 72.69%] | 68.24% [66.30%, 70.18%] | -2.57% [-5.44%, +0.30%] | 0.0732 |
| S2 | 86.58% [84.35%, 88.81%] | 87.50% [85.39%, 89.61%] | +0.92% [-0.40%, +2.24%] | 0.1485 |
| S4 | 74.14% [67.72%, 80.56%] | 72.30% [65.91%, 78.69%] | -1.84% [-3.98%, +0.30%] | 0.0832 |

**Decision:** Keep the supersede filter active during training assignment (no statistical gain from removing it in training; S1 and S4 show slight negative trends).
