# V0 Walking Skeleton Demonstration: Schedule `drift`, Seed `800`
**Status:** DEMONSTRATION, NOT A RESULT

## Final Performance Summary

| Arm | Final Accuracy | Forgetting | Memory (Bytes) | Memory (MB) | Update Time (s) | Latency Batch 1 (ms) | Latency Batch 32 (ms) | Notes |
|---|---|---|---|---|---|---|---|---|
| V0 (UniPool) | 63.19% | 6.78% | 1,240,044 | 1.18 MB | 13.41 | 0.52 ms | 0.73 ms | - |
| A3 (ReplayNet) | 57.21% | 8.85% | 1,239,540 | 1.18 MB | 5.49 | 0.11 ms | 0.18 ms | - |
| A7 (kNN) | 54.74% | 4.70% | 1,239,552 | 1.18 MB | 0.06 | 0.11 ms | 0.31 ms | - |
| A5a (Streaming LDA) | 61.26% | 3.90% | 129,536 | 0.12 MB | 0.50 | 0.06 ms | 0.09 ms | - |
| A5b (Mahalanobis) | 66.12% | 1.67% | 8,256,000 | 7.87 MB | 0.29 | 4.66 ms | 6.57 ms | - |
| A6 (Random-Feature Ridge) | 69.55% | 1.71% | 272,531,456 | 259.91 MB | 29.87 | 0.45 ms | 1.86 ms | - |
| A7all (kNN All) | 64.67% | 1.00% | 5,222,400 | 4.98 MB | 0.01 | 0.30 ms | 1.68 ms | - |

## Novelty / Unknown Flag Diagnostics (V0)

- **Final False New Rate (Known Test Marked Unknown):** 0.00%
- **Final OOD Detection Rate (Out-of-Domain Marked Unknown):** 0.00%
- **Rank AUROC (Novelty Score 1 - p_nov):**
  - `stories`: 0.8358
  - `wiki`: 0.8875
  - `science_questions`: 0.9033
  - **Mean AUROC:** 0.8755

## Progression Tables (Per Phase Boundary)

### Phase 0 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 37.27% | 32.75% | 32.70% | 36.47% | 33.50% | 37.37% | 32.70% |
| Memory Footprint | 0.72 MB | 1.06 MB | 0.50 MB | 0.12 MB | 4.98 MB | 259.91 MB | 0.50 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |

### Phase 1 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 45.01% | 40.54% | 40.21% | 42.17% | 39.33% | 44.52% | 40.21% |
| Memory Footprint | 0.87 MB | 1.18 MB | 1.00 MB | 0.12 MB | 5.86 MB | 259.91 MB | 1.00 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |

### Phase 2 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 49.36% | 44.91% | 44.11% | 44.26% | 43.42% | 49.38% | 44.89% |
| Memory Footprint | 0.87 MB | 1.18 MB | 1.18 MB | 0.12 MB | 6.30 MB | 259.91 MB | 1.49 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |

### Phase 3 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 53.21% | 48.40% | 46.70% | 48.18% | 49.25% | 53.32% | 48.85% |
| Memory Footprint | 1.03 MB | 1.18 MB | 1.18 MB | 0.12 MB | 6.55 MB | 259.91 MB | 1.99 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |

### Phase 4 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 55.06% | 48.66% | 48.30% | 51.03% | 51.22% | 56.04% | 51.24% |
| Memory Footprint | 1.03 MB | 1.18 MB | 1.18 MB | 0.12 MB | 6.74 MB | 259.91 MB | 2.49 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |

### Phase 5 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 57.14% | 51.91% | 48.67% | 53.32% | 56.11% | 59.72% | 54.18% |
| Memory Footprint | 1.18 MB | 1.18 MB | 1.18 MB | 0.12 MB | 6.99 MB | 259.91 MB | 2.99 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |

### Phase 6 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 58.40% | 53.01% | 49.86% | 54.95% | 58.82% | 61.80% | 56.09% |
| Memory Footprint | 1.18 MB | 1.18 MB | 1.18 MB | 0.12 MB | 7.24 MB | 259.91 MB | 3.49 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |

### Phase 7 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 61.11% | 53.64% | 50.59% | 58.51% | 61.13% | 64.71% | 59.18% |
| Memory Footprint | 1.18 MB | 1.18 MB | 1.18 MB | 0.12 MB | 7.68 MB | 259.91 MB | 3.98 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |

### Phase 8 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 63.18% | 56.52% | 53.23% | 60.32% | 64.31% | 68.76% | 63.02% |
| Memory Footprint | 1.18 MB | 1.18 MB | 1.18 MB | 0.12 MB | 7.87 MB | 259.91 MB | 4.48 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |

### Phase 9 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 63.19% | 57.21% | 54.74% | 61.26% | 66.12% | 69.55% | 64.67% |
| Memory Footprint | 1.18 MB | 1.18 MB | 1.18 MB | 0.12 MB | 7.87 MB | 259.91 MB | 4.98 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |
