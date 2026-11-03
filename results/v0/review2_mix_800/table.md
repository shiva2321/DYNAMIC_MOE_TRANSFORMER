# V0 Walking Skeleton Demonstration: Schedule `mix`, Seed `800`
**Status:** DEMONSTRATION, NOT A RESULT

## Final Performance Summary

| Arm | Final Accuracy | Forgetting | Memory (Bytes) | Memory (MB) | Update Time (s) | Latency Batch 1 (ms) | Latency Batch 32 (ms) | Notes |
|---|---|---|---|---|---|---|---|---|
| V0 (UniPool) | 65.21% | 0.34% | 1,109,484 | 1.06 MB | 12.57 | 0.51 ms | 0.70 ms | - |
| A3 (ReplayNet) | 58.79% | 1.88% | 1,108,980 | 1.06 MB | 4.89 | 0.11 ms | 0.18 ms | - |
| A7 (kNN) | 53.92% | 1.33% | 1,108,992 | 1.06 MB | 0.05 | 0.10 ms | 0.26 ms | - |
| A5a (Streaming LDA) | 61.50% | 0.04% | 129,536 | 0.12 MB | 0.47 | 0.06 ms | 0.09 ms | - |
| A5b (Mahalanobis) | 64.08% | 0.79% | 8,256,000 | 7.87 MB | 0.25 | 3.31 ms | 5.47 ms | - |
| A6 (Random-Feature Ridge) | 70.02% | -0.77% | 272,531,456 | 259.91 MB | 26.97 | 0.23 ms | 1.56 ms | - |
| A7all (kNN All) | 64.40% | -0.93% | 5,224,448 | 4.98 MB | 0.01 | 0.21 ms | 1.05 ms | - |

## Novelty / Unknown Flag Diagnostics (V0)

- **Final False New Rate (Known Test Marked Unknown):** 0.00%
- **Final OOD Detection Rate (Out-of-Domain Marked Unknown):** 0.00%
- **Rank AUROC (Novelty Score 1 - p_nov):**
  - `stories`: 0.7796
  - `wiki`: 0.8784
  - `science_questions`: 0.8926
  - **Mean AUROC:** 0.8502

## Progression Tables (Per Phase Boundary)

### Phase 0 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 59.69% | 49.14% | 52.22% | 56.43% | 56.06% | 59.30% | 52.22% |
| Memory Footprint | 0.91 MB | 1.06 MB | 1.00 MB | 0.12 MB | 7.87 MB | 259.91 MB | 1.00 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |

### Phase 1 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 62.28% | 54.13% | 53.36% | 58.53% | 59.66% | 64.95% | 57.84% |
| Memory Footprint | 1.06 MB | 1.06 MB | 1.06 MB | 0.12 MB | 7.87 MB | 259.91 MB | 1.99 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |

### Phase 2 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 63.76% | 55.75% | 53.24% | 59.98% | 61.72% | 67.57% | 61.41% |
| Memory Footprint | 1.06 MB | 1.06 MB | 1.06 MB | 0.12 MB | 7.87 MB | 259.91 MB | 2.99 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |

### Phase 3 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 64.25% | 57.83% | 53.75% | 61.20% | 63.72% | 68.95% | 63.36% |
| Memory Footprint | 1.06 MB | 1.06 MB | 1.06 MB | 0.12 MB | 7.87 MB | 259.91 MB | 3.99 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |

### Phase 4 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (Streaming LDA) | A5b (Mahalanobis) | A6 (Random-Feature Ridge) | A7all (kNN All) |
|---|---|---|---|---|---|---|---|
| Mean Accuracy | 65.21% | 58.79% | 53.92% | 61.50% | 64.08% | 70.02% | 64.40% |
| Memory Footprint | 1.06 MB | 1.06 MB | 1.06 MB | 0.12 MB | 7.87 MB | 259.91 MB | 4.98 MB |
| Experts | 1 | 1 (net) | - | - | - | - | - |
