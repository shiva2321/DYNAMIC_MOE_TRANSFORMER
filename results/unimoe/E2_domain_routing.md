# E2: domain-level routing on the six text domains (sequential schedule, seeds 500-502; generated)

| router | accuracy | banking | agnews | newsgroups | emotion | sst2 | svamp |
|---|---|---|---|---|---|---|---|
| pool (U-real routing) | 0.833 (seeds 0.862, 0.868, 0.769) | 0.98 | 0.75 | 0.67 | 0.84 | 0.78 | 0.95 |
| nearest centroid | 0.781 (seeds 0.786, 0.778, 0.779) | 0.98 | 0.64 | 0.46 | 0.86 | 0.88 | 1.00 |
| logistic-regression domain classifier | 0.893 (seeds 0.891, 0.897, 0.891) | 0.99 | 0.72 | 0.86 | 0.91 | 0.80 | 0.99 |
| asking reader (six cards + none) | 0.647 (seeds 0.682, 0.682, 0.576) | 0.88 | 0.84 | 0.10 | 0.76 | 0.48 | 1.00 |

Asking reader says "none of these" to 0.329 of known text messages (false refusals) and to out-of-distribution text: stories 0.863, wiki 0.847, science_questions 0.957.
