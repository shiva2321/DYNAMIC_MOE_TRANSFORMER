# Lab G5, the pool fed by different keys (seeds 150-154; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-skill | acc taught skills (teach / new wording) | acc new skill (teach / new wording) | acc all four groups | unassigned share | language-model passes per seed | cache hits (wrong share of hits) | pool s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| emb | 5 | 2.0 / 2.6 | 0.00 | 0.60 | 0.08 | 0.720 (0.753 / 0.687) | 0.410 (0.440 / 0.380) | 0.565 | 0.16 | 0 | - | 4 |
| both:c | 5 | 3.4 / 4.4 | 0.20 | 0.80 | 0.31 | 0.797 (0.835 / 0.758) | 0.395 (0.375 / 0.415) | 0.596 | 0.17 | 2060 | - | 6 |
| both:c,w3 | 5 | 3.2 / 4.6 | 0.20 | 1.20 | 0.35 | 0.787 (0.817 / 0.757) | 0.362 (0.375 / 0.350) | 0.575 | 0.15 | 2060 | - | 5 |
| both:c,e0.5 | 5 | 3.4 / 5.0 | 0.40 | 1.20 | 0.32 | 0.832 (0.850 / 0.813) | 0.407 (0.405 / 0.410) | 0.620 | 0.13 | 2060 | - | 7 |
| both:c,e2 | 5 | 3.0 / 5.2 | 1.40 | 0.80 | 0.14 | 0.707 (0.747 / 0.667) | 0.390 (0.445 / 0.335) | 0.548 | 0.18 | 2060 | - | 8 |
| both:c,e0.5,w2 | 5 | 3.2 / 4.0 | 0.00 | 0.80 | 0.33 | 0.823 (0.845 / 0.800) | 0.372 (0.380 / 0.365) | 0.597 | 0.15 | 2060 | - | 5 |

## Paired against `emb` (same seeds; mean difference, and the number of seeds in which the arm is better)

| arm | acc all | acc taught in new wordings | acc new skill | NMI expert-skill |
|---|---|---|---|---|
| both:c | +0.031 (3/5) | +0.072 (5/5) | -0.015 (3/5) | +0.236 (4/5) |
| both:c,w3 | +0.010 (3/5) | +0.070 (4/5) | -0.047 (2/5) | +0.272 (4/5) |
| both:c,e0.5 | +0.055 (4/5) | +0.127 (5/5) | -0.002 (2/5) | +0.239 (4/5) |
| both:c,e2 | -0.017 (2/5) | -0.020 (3/5) | -0.020 (2/5) | +0.066 (4/5) |
| both:c,e0.5,w2 | +0.032 (3/5) | +0.113 (4/5) | -0.037 (3/5) | +0.252 (4/5) |
