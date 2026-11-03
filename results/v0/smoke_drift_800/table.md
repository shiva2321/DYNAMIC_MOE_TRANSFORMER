# V0 Walking Skeleton Demonstration: Schedule `drift`, Seed `800`
**Status:** DEMONSTRATION, NOT A RESULT

## Final Performance Summary

| Arm | Final Accuracy | Memory (Bytes) | Memory (MB) | Latency Batch 1 (ms) | Latency Batch 32 (ms) |
|---|---|---|---|---|---|
| V0 (UniPool) | 63.19% | 1,240,044 | 1.18 MB | 0.55 ms | 0.95 ms |
| A3 (ReplayNet) | 57.21% | 1,239,540 | 1.18 MB | 0.13 ms | 0.18 ms |
| A7 (kNN) | 54.74% | 1,239,552 | 1.18 MB | 0.17 ms | 0.44 ms |
| A5a (Streaming LDA) | 61.26% | 129,536 | 0.12 MB | 0.06 ms | 0.09 ms |

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
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |
|---|---|---|---|---|
| Mean Accuracy (Seen Regimes) | 37.27% | 32.75% | 32.70% | 36.47% |
| Memory Footprint | 0.72 MB | 1.06 MB | 0.50 MB | 0.12 MB |
| Experts | 1 | 1 | - | - |
| False New Rate (Known) | 0.00% | - | - | - |
| OOD Detection Rate | 22.67% | - | - | - |

### Phase 1 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |
|---|---|---|---|---|
| Mean Accuracy (Seen Regimes) | 45.01% | 40.54% | 40.21% | 42.17% |
| Memory Footprint | 0.87 MB | 1.18 MB | 1.00 MB | 0.12 MB |
| Experts | 1 | 1 | - | - |
| False New Rate (Known) | 0.00% | - | - | - |
| OOD Detection Rate | 6.78% | - | - | - |

### Phase 2 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |
|---|---|---|---|---|
| Mean Accuracy (Seen Regimes) | 49.36% | 44.91% | 44.11% | 44.26% |
| Memory Footprint | 0.87 MB | 1.18 MB | 1.18 MB | 0.12 MB |
| Experts | 1 | 1 | - | - |
| False New Rate (Known) | 0.00% | - | - | - |
| OOD Detection Rate | 11.22% | - | - | - |

### Phase 3 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |
|---|---|---|---|---|
| Mean Accuracy (Seen Regimes) | 53.21% | 48.40% | 46.70% | 48.18% |
| Memory Footprint | 1.03 MB | 1.18 MB | 1.18 MB | 0.12 MB |
| Experts | 1 | 1 | - | - |
| False New Rate (Known) | 0.00% | - | - | - |
| OOD Detection Rate | 14.11% | - | - | - |

### Phase 4 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |
|---|---|---|---|---|
| Mean Accuracy (Seen Regimes) | 55.06% | 48.66% | 48.30% | 51.03% |
| Memory Footprint | 1.03 MB | 1.18 MB | 1.18 MB | 0.12 MB |
| Experts | 1 | 1 | - | - |
| False New Rate (Known) | 0.00% | - | - | - |
| OOD Detection Rate | 0.00% | - | - | - |

### Phase 5 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |
|---|---|---|---|---|
| Mean Accuracy (Seen Regimes) | 57.14% | 51.91% | 48.67% | 53.32% |
| Memory Footprint | 1.18 MB | 1.18 MB | 1.18 MB | 0.12 MB |
| Experts | 1 | 1 | - | - |
| False New Rate (Known) | 0.00% | - | - | - |
| OOD Detection Rate | 16.22% | - | - | - |

### Phase 6 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |
|---|---|---|---|---|
| Mean Accuracy (Seen Regimes) | 58.40% | 53.01% | 49.86% | 54.95% |
| Memory Footprint | 1.18 MB | 1.18 MB | 1.18 MB | 0.12 MB |
| Experts | 1 | 1 | - | - |
| False New Rate (Known) | 0.00% | - | - | - |
| OOD Detection Rate | 18.78% | - | - | - |

### Phase 7 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |
|---|---|---|---|---|
| Mean Accuracy (Seen Regimes) | 61.11% | 53.64% | 50.59% | 58.51% |
| Memory Footprint | 1.18 MB | 1.18 MB | 1.18 MB | 0.12 MB |
| Experts | 1 | 1 | - | - |
| False New Rate (Known) | 0.00% | - | - | - |
| OOD Detection Rate | 19.00% | - | - | - |

### Phase 8 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |
|---|---|---|---|---|
| Mean Accuracy (Seen Regimes) | 63.18% | 56.52% | 53.23% | 60.32% |
| Memory Footprint | 1.18 MB | 1.18 MB | 1.18 MB | 0.12 MB |
| Experts | 1 | 1 | - | - |
| False New Rate (Known) | 0.00% | - | - | - |
| OOD Detection Rate | 18.44% | - | - | - |

### Phase 9 Boundary (Regimes Seen: 22)
| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |
|---|---|---|---|---|
| Mean Accuracy (Seen Regimes) | 63.19% | 57.21% | 54.74% | 61.26% |
| Memory Footprint | 1.18 MB | 1.18 MB | 1.18 MB | 0.12 MB |
| Experts | 1 | 1 | - | - |
| False New Rate (Known) | 0.00% | - | - | - |
| OOD Detection Rate | 0.00% | - | - | - |
