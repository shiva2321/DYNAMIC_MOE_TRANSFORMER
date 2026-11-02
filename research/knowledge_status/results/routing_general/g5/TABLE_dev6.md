# Lab G5, the pool fed by different keys (seeds 150-154; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-skill | acc taught skills (teach / new wording) | acc new skill (teach / new wording) | acc all four groups | unassigned share | language-model passes per seed | cache hits (wrong share of hits) | pool s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| emb | 5 | 2.0 / 2.6 | 0.00 | 0.60 | 0.08 | 0.720 (0.753 / 0.687) | 0.410 (0.440 / 0.380) | 0.565 | 0.16 | 0 | - | 4 |
| emb:c | 5 | 3.0 / 4.0 | 0.20 | 0.80 | 0.09 | 0.736 (0.785 / 0.687) | 0.345 (0.395 / 0.295) | 0.540 | 0.18 | 0 | - | 6 |
| both:c | 5 | 3.4 / 4.4 | 0.20 | 0.80 | 0.31 | 0.797 (0.835 / 0.758) | 0.395 (0.375 / 0.415) | 0.596 | 0.17 | 2060 | - | 6 |
| both:c,e0.5 | 5 | 3.4 / 5.0 | 0.40 | 1.20 | 0.32 | 0.832 (0.850 / 0.813) | 0.407 (0.405 / 0.410) | 0.620 | 0.13 | 2060 | - | 7 |
| both:c,e0.5,k0.95 | 5 | 3.4 / 4.4 | 0.40 | 0.60 | 0.31 | 0.854 (0.857 / 0.852) | 0.385 (0.415 / 0.355) | 0.620 | 0.13 | 1927 | 37% (5.7%) | 6 |
| oracle:c | 5 | 3.0 / 5.2 | 1.40 | 0.80 | 0.56 | 0.882 (0.883 / 0.882) | 0.330 (0.290 / 0.370) | 0.606 | 0.14 | 0 | - | 5 |

## Paired against `emb` (same seeds; mean difference, and the number of seeds in which the arm is better)

| arm | acc all | acc taught in new wordings | acc new skill | NMI expert-skill |
|---|---|---|---|---|
| emb:c | -0.025 (1/5) | +0.000 (2/5) | -0.065 (1/5) | +0.016 (2/5) |
| both:c | +0.031 (3/5) | +0.072 (5/5) | -0.015 (3/5) | +0.236 (4/5) |
| both:c,e0.5 | +0.055 (4/5) | +0.127 (5/5) | -0.002 (2/5) | +0.239 (4/5) |
| both:c,e0.5,k0.95 | +0.055 (4/5) | +0.165 (5/5) | -0.025 (2/5) | +0.228 (4/5) |
| oracle:c | +0.041 (3/5) | +0.195 (5/5) | -0.080 (2/5) | +0.479 (5/5) |
