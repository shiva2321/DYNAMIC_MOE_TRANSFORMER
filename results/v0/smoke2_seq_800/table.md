# V0 Walking Skeleton Demonstration: Schedule `seq`, Seed `800`
**Status:** DEMONSTRATION, NOT A RESULT

## Final Performance Summary

| Arm | Final Accuracy | Forgetting | Memory (Bytes) | Memory (MB) | Update Time (s) | Latency Batch 1 (ms) | Latency Batch 32 (ms) | Notes |
|---|---|---|---|---|---|---|---|---|
| V0 (UniPool) | 65.02% | 5.08% | 14,687,420 | 14.01 MB | 40.38 | 4.04 ms | 5.58 ms | - |
| A3 (ReplayNet) | 60.07% | 10.41% | 7,097,844 | 6.77 MB | 9.27 | 0.14 ms | 0.18 ms | Capped by stream size |
| A7 (kNN) | 64.54% | 2.77% | 6,504,448 | 6.20 MB | 0.10 | 0.34 ms | 1.61 ms | Capped by stream size |
| A5a (Streaming LDA) | 61.34% | 6.64% | 129,536 | 0.12 MB | 0.70 | 0.06 ms | 0.12 ms | - |
| A5b (Mahalanobis) | 63.01% | 3.55% | 8,256,000 | 7.87 MB | 0.34 | 4.59 ms | 6.41 ms | - |
| A6 (Random-Feature Ridge) | 70.14% | 3.55% | 272,531,456 | 259.91 MB | 38.91 | 0.27 ms | 1.76 ms | - |
| A7all (kNN All) | 64.54% | 2.77% | 6,504,448 | 6.20 MB | 0.01 | 0.32 ms | 2.10 ms | - |

## Novelty / Unknown Flag Diagnostics (V0)

- **Final False New Rate (Known Test Marked Unknown):** 0.00%
- **Final OOD Detection Rate (Out-of-Domain Marked Unknown):** 0.00%
- **Rank AUROC (Novelty Score 1 - p_nov):**
  - `stories`: 0.6336
  - `wiki`: 0.6855
  - `science_questions`: 0.7218
  - **Mean AUROC:** 0.6803

## Progression Tables (Per Phase Boundary)

### Phase 0 Boundary (Regimes Seen: 1)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 60.00% | 60.67% | 56.00% | 56.00% | 61.67% | 68.00% | 56.00% |
| Memory Footprint | 0.70 MB | 0.76 MB | 0.20 MB | 0.12 MB | 0.25 MB | 259.91 MB | 0.20 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |

### Phase 1 Boundary (Regimes Seen: 2)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 60.95% | 60.56% | 56.57% | 51.91% | 61.89% | 64.45% | 56.57% |
| Memory Footprint | 1.40 MB | 1.09 MB | 0.52 MB | 0.12 MB | 0.94 MB | 259.91 MB | 0.52 MB |
| Experts | 2 | 1 (net) | - | - | - | - | - |

### Phase 2 Boundary (Regimes Seen: 3)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 59.63% | 62.14% | 57.83% | 49.83% | 61.85% | 64.83% | 57.83% |
| Memory Footprint | 2.10 MB | 1.41 MB | 0.85 MB | 0.12 MB | 1.01 MB | 259.91 MB | 0.85 MB |
| Experts | 3 | 1 (net) | - | - | - | - | - |

### Phase 3 Boundary (Regimes Seen: 4)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 60.47% | 62.49% | 60.30% | 52.35% | 58.09% | 65.74% | 60.30% |
| Memory Footprint | 2.24 MB | 1.74 MB | 1.17 MB | 0.12 MB | 1.01 MB | 259.91 MB | 1.17 MB |
| Experts | 3 | 1 (net) | - | - | - | - | - |

### Phase 4 Boundary (Regimes Seen: 5)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 65.04% | 65.18% | 63.52% | 52.29% | 57.31% | 68.16% | 63.52% |
| Memory Footprint | 2.38 MB | 2.06 MB | 1.50 MB | 0.12 MB | 1.01 MB | 259.91 MB | 1.50 MB |
| Experts | 3 | 1 (net) | - | - | - | - | - |

### Phase 5 Boundary (Regimes Seen: 6)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 67.40% | 68.91% | 65.30% | 54.59% | 56.69% | 69.41% | 65.30% |
| Memory Footprint | 3.08 MB | 2.39 MB | 1.82 MB | 0.12 MB | 1.01 MB | 259.91 MB | 1.82 MB |
| Experts | 4 | 1 (net) | - | - | - | - | - |

### Phase 6 Boundary (Regimes Seen: 7)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 69.37% | 69.54% | 65.46% | 54.85% | 57.59% | 71.11% | 65.46% |
| Memory Footprint | 3.79 MB | 2.71 MB | 2.15 MB | 0.12 MB | 1.01 MB | 259.91 MB | 2.15 MB |
| Experts | 5 | 1 (net) | - | - | - | - | - |

### Phase 7 Boundary (Regimes Seen: 8)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 72.16% | 70.15% | 68.40% | 58.80% | 62.26% | 74.05% | 68.40% |
| Memory Footprint | 4.48 MB | 2.91 MB | 2.34 MB | 0.12 MB | 2.20 MB | 259.91 MB | 2.34 MB |
| Experts | 6 | 1 (net) | - | - | - | - | - |

### Phase 8 Boundary (Regimes Seen: 9)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 73.63% | 72.51% | 70.37% | 61.82% | 66.15% | 75.96% | 70.37% |
| Memory Footprint | 5.15 MB | 3.10 MB | 2.54 MB | 0.12 MB | 2.96 MB | 259.91 MB | 2.54 MB |
| Experts | 7 | 1 (net) | - | - | - | - | - |

### Phase 9 Boundary (Regimes Seen: 10)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 74.28% | 70.39% | 71.44% | 62.86% | 68.17% | 77.03% | 71.44% |
| Memory Footprint | 5.85 MB | 3.30 MB | 2.73 MB | 0.12 MB | 3.46 MB | 259.91 MB | 2.73 MB |
| Experts | 8 | 1 (net) | - | - | - | - | - |

### Phase 10 Boundary (Regimes Seen: 11)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 73.89% | 71.38% | 71.84% | 63.92% | 69.88% | 77.23% | 71.84% |
| Memory Footprint | 6.55 MB | 3.49 MB | 2.93 MB | 0.12 MB | 4.91 MB | 259.91 MB | 2.93 MB |
| Experts | 9 | 1 (net) | - | - | - | - | - |

### Phase 11 Boundary (Regimes Seen: 12)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 75.10% | 72.62% | 73.13% | 65.04% | 71.83% | 78.11% | 73.13% |
| Memory Footprint | 7.24 MB | 3.69 MB | 3.12 MB | 0.12 MB | 5.61 MB | 259.91 MB | 3.12 MB |
| Experts | 10 | 1 (net) | - | - | - | - | - |

### Phase 12 Boundary (Regimes Seen: 13)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 76.86% | 76.62% | 74.92% | 67.62% | 73.97% | 79.80% | 74.92% |
| Memory Footprint | 7.85 MB | 3.79 MB | 3.22 MB | 0.12 MB | 5.86 MB | 259.91 MB | 3.22 MB |
| Experts | 11 | 1 (net) | - | - | - | - | - |

### Phase 13 Boundary (Regimes Seen: 14)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 76.28% | 75.00% | 74.11% | 67.68% | 72.25% | 79.38% | 74.11% |
| Memory Footprint | 8.54 MB | 3.99 MB | 3.42 MB | 0.12 MB | 5.98 MB | 259.91 MB | 3.42 MB |
| Experts | 12 | 1 (net) | - | - | - | - | - |

### Phase 14 Boundary (Regimes Seen: 15)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 77.04% | 77.08% | 74.95% | 68.77% | 73.63% | 80.13% | 74.95% |
| Memory Footprint | 9.24 MB | 4.18 MB | 3.62 MB | 0.12 MB | 6.11 MB | 259.91 MB | 3.62 MB |
| Experts | 13 | 1 (net) | - | - | - | - | - |

### Phase 15 Boundary (Regimes Seen: 16)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 75.69% | 74.39% | 73.75% | 68.34% | 73.16% | 79.07% | 73.75% |
| Memory Footprint | 9.94 MB | 4.38 MB | 3.81 MB | 0.12 MB | 6.24 MB | 259.91 MB | 3.81 MB |
| Experts | 14 | 1 (net) | - | - | - | - | - |

### Phase 16 Boundary (Regimes Seen: 17)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 73.45% | 71.02% | 71.64% | 67.02% | 70.99% | 77.38% | 71.64% |
| Memory Footprint | 10.63 MB | 4.57 MB | 4.01 MB | 0.12 MB | 6.55 MB | 259.91 MB | 4.01 MB |
| Experts | 15 | 1 (net) | - | - | - | - | - |

### Phase 17 Boundary (Regimes Seen: 18)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 71.36% | 68.03% | 69.66% | 65.42% | 68.61% | 75.68% | 69.66% |
| Memory Footprint | 11.33 MB | 4.77 MB | 4.20 MB | 0.12 MB | 6.99 MB | 259.91 MB | 4.20 MB |
| Experts | 16 | 1 (net) | - | - | - | - | - |

### Phase 18 Boundary (Regimes Seen: 19)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 68.97% | 67.73% | 67.65% | 64.17% | 66.10% | 73.59% | 67.65% |
| Memory Footprint | 12.03 MB | 4.96 MB | 4.40 MB | 0.12 MB | 7.37 MB | 259.91 MB | 4.40 MB |
| Experts | 17 | 1 (net) | - | - | - | - | - |

### Phase 19 Boundary (Regimes Seen: 20)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 68.31% | 66.20% | 67.26% | 64.09% | 65.81% | 73.01% | 67.26% |
| Memory Footprint | 12.72 MB | 5.16 MB | 4.59 MB | 0.12 MB | 7.50 MB | 259.91 MB | 4.59 MB |
| Experts | 18 | 1 (net) | - | - | - | - | - |

### Phase 20 Boundary (Regimes Seen: 21)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 66.88% | 64.46% | 66.25% | 63.30% | 65.42% | 72.11% | 66.25% |
| Memory Footprint | 13.42 MB | 5.35 MB | 4.79 MB | 0.12 MB | 7.68 MB | 259.91 MB | 4.79 MB |
| Experts | 19 | 1 (net) | - | - | - | - | - |

### Phase 21 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 65.04% | 62.38% | 64.25% | 61.52% | 64.08% | 70.02% | 64.25% |
| Memory Footprint | 14.12 MB | 5.55 MB | 4.98 MB | 0.12 MB | 7.87 MB | 259.91 MB | 4.98 MB |
| Experts | 20 | 1 (net) | - | - | - | - | - |

### Phase 22 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 65.13% | 65.66% | 64.19% | 61.32% | 62.26% | 70.02% | 64.19% |
| Memory Footprint | 14.12 MB | 6.28 MB | 5.71 MB | 0.12 MB | 7.87 MB | 259.91 MB | 5.71 MB |
| Experts | 20 | 1 (net) | - | - | - | - | - |

### Phase 23 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 65.17% | 64.33% | 64.19% | 61.33% | 62.42% | 70.14% | 64.19% |
| Memory Footprint | 14.11 MB | 6.38 MB | 5.81 MB | 0.12 MB | 7.87 MB | 259.91 MB | 5.81 MB |
| Experts | 20 | 1 (net) | - | - | - | - | - |

### Phase 24 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 65.26% | 63.67% | 64.32% | 61.32% | 62.47% | 70.15% | 64.32% |
| Memory Footprint | 14.09 MB | 6.48 MB | 5.91 MB | 0.12 MB | 7.87 MB | 259.91 MB | 5.91 MB |
| Experts | 20 | 1 (net) | - | - | - | - | - |

### Phase 25 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 65.17% | 61.67% | 64.46% | 61.34% | 62.07% | 70.06% | 64.46% |
| Memory Footprint | 14.06 MB | 6.57 MB | 6.01 MB | 0.12 MB | 7.87 MB | 259.91 MB | 6.01 MB |
| Experts | 20 | 1 (net) | - | - | - | - | - |

### Phase 26 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 65.02% | 61.04% | 64.39% | 61.26% | 62.18% | 70.02% | 64.39% |
| Memory Footprint | 14.03 MB | 6.67 MB | 6.11 MB | 0.12 MB | 7.87 MB | 259.91 MB | 6.11 MB |
| Experts | 20 | 1 (net) | - | - | - | - | - |

### Phase 27 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 65.02% | 60.07% | 64.54% | 61.34% | 63.01% | 70.14% | 64.54% |
| Memory Footprint | 14.01 MB | 6.77 MB | 6.20 MB | 0.12 MB | 7.87 MB | 259.91 MB | 6.20 MB |
| Experts | 20 | 1 (net) | - | - | - | - | - |
