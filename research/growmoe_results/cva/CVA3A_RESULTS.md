# CVA-3A Results: Competence Monitor False Alarm Audit

VERIFICATION (criteria fixed in advance by Claude, 2026-10-14)

Generated: 2026-10-14 03:24:34
Stream: function_regimes (stationary segment 0, 20,000 samples fed in 313 batches of 64)
Evaluation seeds: 10-19 | Latent dimension: d = 128 | arm_monitor_after_infancy = True

---

## Summary

- **Alarms per 10,000 labelled samples (competence=True)**: **0.05** (range: **0.00** to **0.50**; mean total alarms: 0.1 per 20k stream)
- **Did any seed end with more than one expert (competence=True)?**: **NO** (all seeds ended with exactly 1 expert)
- **Control (competence=False)**: All seeds produced exactly 0 alarms: **CONFIRMED** (ended with exactly 1 expert)

---

## Runs with Competence Monitor Active (`competence=True`)

| Seed | Total Samples | Alarms | Alarms / 10k | Alarm Call Indices | Spawns | Extends | Final Experts | Runtime (s) |
|:---:|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| 10 | 20000 | 1 | 0.50 | 92 | 1 | 1 | 1 | 10.7 |
| 11 | 20000 | 0 | 0.00 | None | 1 | 0 | 1 | 12.9 |
| 12 | 20000 | 0 | 0.00 | None | 1 | 0 | 1 | 10.4 |
| 13 | 20000 | 0 | 0.00 | None | 1 | 0 | 1 | 7.3 |
| 14 | 20000 | 0 | 0.00 | None | 1 | 0 | 1 | 8.1 |
| 15 | 20000 | 0 | 0.00 | None | 1 | 0 | 1 | 10.0 |
| 16 | 20000 | 0 | 0.00 | None | 1 | 0 | 1 | 9.3 |
| 17 | 20000 | 0 | 0.00 | None | 1 | 0 | 1 | 8.8 |
| 18 | 20000 | 0 | 0.00 | None | 1 | 0 | 1 | 8.8 |
| 19 | 20000 | 0 | 0.00 | None | 1 | 0 | 1 | 8.9 |

---

## Negative Control Runs (`competence=False`)

| Seed | Total Samples | Alarms | Spawns | Extends | Final Experts | Runtime (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 20000 | 0 | 1 | 0 | 1 | 8.7 |
| 11 | 20000 | 0 | 1 | 0 | 1 | 8.8 |
| 12 | 20000 | 0 | 1 | 0 | 1 | 8.7 |
| 13 | 20000 | 0 | 1 | 0 | 1 | 5.8 |
| 14 | 20000 | 0 | 1 | 0 | 1 | 6.2 |
| 15 | 20000 | 0 | 1 | 0 | 1 | 9.1 |
| 16 | 20000 | 0 | 1 | 0 | 1 | 9.2 |
| 17 | 20000 | 0 | 1 | 0 | 1 | 9.1 |
| 18 | 20000 | 0 | 1 | 0 | 1 | 9.1 |
| 19 | 20000 | 0 | 1 | 0 | 1 | 9.3 |
