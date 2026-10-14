# CVA-2B Results: Long-stream Lifecycle Verification

VERIFICATION (criteria fixed in advance by Claude, 2026-10-14)

Generated: 2026-10-14 03:47:27
Stream: s2_permuted (30 regimes, 150 segments of 200 samples = 30,000 samples, ~470 batches of 64)
Seeds: [10, 11, 12, 13, 14]

---

## Summary of Predeclared Criteria

| Criterion | Description | Value / Measurement | Threshold | Verdict |
|:---:|:---|:---:|:---:|:---:|
| **(a)** | Lifecycle events occur (dormant, archive, revive) | Dormant: True, Archive: True, Revive: True | >= 1 each | **PASS** |
| **(b)** | Mean paired accuracy (LIFE_ON - LIFE_OFF) | -0.41 percentage points | >= -1.0 pt | **PASS** |
| **(c)** | End-of-run invariants (archived opt/replay=None, sig kept, revived keeps eid) | All invariants hold on all seeds | Strictly hold | **PASS** |
| **(d)** | Memory saved (share of LIFE_OFF) | Stored vectors: 95.9%, Memory bytes: 97.3% | Report only | **REPORT** |
| **Det** | Determinism repeat on Seed 10 | Acc match: True, Events match: True | Identical | **PASS** |

---

## Paired Performance Comparison

| Seed | Acc (LIFE_ON) | Acc (LIFE_OFF) | Paired Diff (pts) | Experts (ON / OFF) | Stored Vecs (ON / OFF) | Memory MB (ON / OFF) | Gate MACs (ON / OFF) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 92.10% | 93.03% | -0.93 pt | 20 / 20 | 9792 / 10240 | 7.44 MB / 7.66 MB | 655360 / 655360 |
| 11 | 92.13% | 92.67% | -0.53 pt | 21 / 21 | 10136 / 10752 | 7.74 MB / 8.04 MB | 688128 / 688128 |
| 12 | 88.27% | 88.80% | -0.53 pt | 16 / 16 | 7968 / 8192 | 6.02 MB / 6.13 MB | 524288 / 524288 |
| 13 | 92.47% | 92.70% | -0.23 pt | 20 / 20 | 9904 / 10240 | 7.49 MB / 7.66 MB | 655360 / 655360 |
| 14 | 90.63% | 90.43% | +0.20 pt | 17 / 17 | 8312 / 8704 | 6.32 MB / 6.51 MB | 557056 / 557056 |

---

## Lifecycle Event Counts (LIFE_ON)

| Seed | Spawns | Extends | Alarms | Dormant Transitions | Archives | Revives | Merges | Max Non-Active |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 20 | 14 | 5 | 77 | 35 | 69 | 0 | 12 |
| 11 | 21 | 10 | 1 | 75 | 35 | 65 | 0 | 12 |
| 12 | 16 | 17 | 4 | 59 | 26 | 54 | 0 | 10 |
| 13 | 20 | 12 | 2 | 77 | 27 | 66 | 0 | 12 |
| 14 | 17 | 15 | 2 | 58 | 24 | 51 | 0 | 10 |

---

## Timeline Snapshots (Every 25 observe calls, Seed 10 LIFE_ON)

| Call | Experts | Infant | Active | Dormant | Archived | Stored Vectors |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 25 | 3 | 1 | 2 | 0 | 0 | 1280 |
| 50 | 9 | 1 | 8 | 0 | 0 | 4480 |
| 75 | 12 | 1 | 11 | 0 | 0 | 6144 |
| 100 | 17 | 1 | 12 | 4 | 0 | 8704 |
| 125 | 20 | 0 | 14 | 6 | 0 | 10240 |
| 150 | 20 | 0 | 11 | 7 | 2 | 9728 |
| 175 | 20 | 0 | 10 | 6 | 4 | 9216 |
| 200 | 20 | 0 | 13 | 4 | 3 | 9304 |
| 225 | 20 | 0 | 13 | 6 | 1 | 9624 |
| 250 | 20 | 0 | 12 | 8 | 0 | 9960 |
| 275 | 20 | 0 | 11 | 7 | 2 | 9504 |
| 300 | 20 | 0 | 13 | 4 | 3 | 9192 |
| 325 | 20 | 0 | 15 | 5 | 0 | 9712 |
| 350 | 20 | 0 | 12 | 8 | 0 | 9720 |
| 375 | 20 | 0 | 12 | 8 | 0 | 9848 |
| 400 | 20 | 0 | 11 | 5 | 4 | 8992 |
| 425 | 20 | 0 | 13 | 4 | 3 | 9248 |
| 450 | 20 | 0 | 14 | 5 | 1 | 9592 |
| 475 | 20 | 0 | 13 | 7 | 0 | 9784 |
| 500 | 20 | 0 | 9 | 8 | 3 | 9136 |
| 525 | 20 | 0 | 11 | 5 | 4 | 8744 |
| 550 | 20 | 0 | 13 | 5 | 2 | 9336 |
| 575 | 20 | 0 | 12 | 5 | 3 | 9136 |
| 600 | 20 | 0 | 12 | 8 | 0 | 9792 |
