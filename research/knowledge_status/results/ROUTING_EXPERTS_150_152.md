# Lab C, experts that share a number encoder (tuning seeds 150-152, 1-2 digit operands, held-out problems)

Accuracy on problems never seen: the skill alone (separate) against the skill trained together with helper skills at 2000 examples each (shared encoder, own head). Chance for div is about 0.46 and for mod 0.24 (their answers are mostly 0 and small).

| skill | taught | separate | shared | helpers (shared net) |
|---|---|---|---|---|
| sub | 20 | 0.02 | 0.02 | add 0.84, max 0.91 |
| sub | 100 | 0.04 | 0.15 | add 0.82, max 0.92 |
| sub | 500 | 0.17 | 0.41 | add 0.78, max 0.92 |
| mul | 20 | 0.05 | 0.10 | add 0.78, sub 0.62, max 0.93 |
| mul | 100 | 0.10 | 0.15 | add 0.76, sub 0.54, max 0.92 |
| mul | 500 | 0.18 | 0.31 | add 0.73, sub 0.50, max 0.91 |
| div | 20 | 0.41 | 0.47 | add 0.78, sub 0.65, max 0.93 |
| div | 100 | 0.46 | 0.52 | add 0.76, sub 0.59, max 0.93 |
| div | 500 | 0.53 | 0.59 | add 0.74, sub 0.63, max 0.94 |
| mod | 20 | 0.18 | 0.17 | add 0.78, sub 0.61, max 0.93 |
| mod | 100 | 0.35 | 0.31 | add 0.76, sub 0.54, max 0.92 |
| mod | 500 | 0.52 | 0.52 | add 0.76, sub 0.54, max 0.94 |
| max | 20 | 0.15 | 0.16 | add 0.88, sub 0.75 |
| max | 100 | 0.44 | 0.55 | add 0.79, sub 0.61 |
| max | 500 | 0.79 | 0.78 | add 0.80, sub 0.65 |
