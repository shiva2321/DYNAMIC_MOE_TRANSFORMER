# V0 Walking Skeleton Demonstration: Schedule `mix`, Seed `800`
**Status:** DEMONSTRATION, NOT A RESULT

## Final Performance Summary

| Arm | Final Accuracy | Memory (Bytes) | Memory (MB) | Latency Batch 1 (ms) | Latency Batch 32 (ms) |
|---|---|---|---|---|---|
| V0 (UniPool) | 65.21% | 1,109,484 | 1.06 MB | 0.54 ms | 0.73 ms |
| A3 (ReplayNet) | 58.79% | 1,108,980 | 1.06 MB | 0.11 ms | 0.19 ms |
| A7 (kNN) | 53.92% | 1,108,992 | 1.06 MB | 0.10 ms | 0.29 ms |
| A5a (Streaming LDA) | 61.50% | 129,536 | 0.12 MB | 0.06 ms | 0.09 ms |

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
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |
|---|---|---|---|---|
| Mean Accuracy (Seen Regimes) | 59.69% | 49.14% | 52.22% | 56.43% |
| Memory Footprint | 0.91 MB | 1.06 MB | 1.00 MB | 0.12 MB |
| Experts | 1 | 1 | - | - |
| False New Rate (Known) | 0.00% | - | - | - |
| OOD Detection Rate | 8.33% | - | - | - |

### Phase 1 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |
|---|---|---|---|---|
| Mean Accuracy (Seen Regimes) | 62.28% | 54.13% | 53.36% | 58.53% |
| Memory Footprint | 1.06 MB | 1.06 MB | 1.06 MB | 0.12 MB |
| Experts | 1 | 1 | - | - |
| False New Rate (Known) | 0.00% | - | - | - |
| OOD Detection Rate | 8.00% | - | - | - |

### Phase 2 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |
|---|---|---|---|---|
| Mean Accuracy (Seen Regimes) | 63.76% | 55.75% | 53.24% | 59.98% |
| Memory Footprint | 1.06 MB | 1.06 MB | 1.06 MB | 0.12 MB |
| Experts | 1 | 1 | - | - |
| False New Rate (Known) | 0.00% | - | - | - |
| OOD Detection Rate | 13.00% | - | - | - |

### Phase 3 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |
|---|---|---|---|---|
| Mean Accuracy (Seen Regimes) | 64.25% | 57.83% | 53.75% | 61.20% |
| Memory Footprint | 1.06 MB | 1.06 MB | 1.06 MB | 0.12 MB |
| Experts | 1 | 1 | - | - |
| False New Rate (Known) | 0.00% | - | - | - |
| OOD Detection Rate | 0.00% | - | - | - |

### Phase 4 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |
|---|---|---|---|---|
| Mean Accuracy (Seen Regimes) | 65.21% | 58.79% | 53.92% | 61.50% |
| Memory Footprint | 1.06 MB | 1.06 MB | 1.06 MB | 0.12 MB |
| Experts | 1 | 1 | - | - |
| False New Rate (Known) | 0.00% | - | - | - |
| OOD Detection Rate | 0.00% | - | - | - |
