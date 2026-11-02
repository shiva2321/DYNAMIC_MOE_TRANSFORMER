# Lab G5, the pool fed by different keys (seeds 150-154; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-skill | acc taught skills (teach / new wording) | acc new skill (teach / new wording) | acc all four groups | unassigned share | language-model passes per seed | cache hits (wrong share of hits) | pool s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| emb | 5 | 2.0 / 2.6 | 0.00 | 0.60 | 0.08 | 0.720 (0.753 / 0.687) | 0.410 (0.440 / 0.380) | 0.565 | 0.16 | 0 | - | 4 |
| both:c,e0.5 | 5 | 3.4 / 5.0 | 0.40 | 1.20 | 0.32 | 0.832 (0.850 / 0.813) | 0.407 (0.405 / 0.410) | 0.620 | 0.13 | 2060 | - | 7 |
| both:c,e0.5,t0.9 | 5 | 3.4 / 4.2 | 0.20 | 0.60 | 0.31 | 0.848 (0.863 / 0.833) | 0.447 (0.485 / 0.410) | 0.648 | 0.14 | 1317 | 57% (5.5%) | 6 |
| both:c,e0.5,t0.95 | 5 | 3.4 / 5.0 | 0.80 | 0.80 | 0.29 | 0.830 (0.845 / 0.815) | 0.420 (0.435 / 0.405) | 0.625 | 0.14 | 1685 | 45% (3.8%) | 7 |
| both:c,e0.5,t0.9,m0.8 | 5 | 3.4 / 4.6 | 0.40 | 0.80 | 0.30 | 0.846 (0.862 / 0.830) | 0.443 (0.460 / 0.425) | 0.644 | 0.13 | 1438 | 53% (2.6%) | 6 |
| both:c,e0.5,t0.8 | 5 | 3.4 / 4.4 | 0.40 | 0.60 | 0.27 | 0.830 (0.860 / 0.800) | 0.482 (0.515 / 0.450) | 0.656 | 0.17 | 934 | 69% (8.7%) | 6 |

## Paired against `emb` (same seeds; mean difference, and the number of seeds in which the arm is better)

| arm | acc all | acc taught in new wordings | acc new skill | NMI expert-skill |
|---|---|---|---|---|
| both:c,e0.5 | +0.055 (4/5) | +0.127 (5/5) | -0.002 (2/5) | +0.239 (4/5) |
| both:c,e0.5,t0.9 | +0.083 (4/5) | +0.147 (5/5) | +0.037 (3/5) | +0.233 (4/5) |
| both:c,e0.5,t0.95 | +0.060 (4/5) | +0.128 (5/5) | +0.010 (3/5) | +0.214 (4/5) |
| both:c,e0.5,t0.9,m0.8 | +0.079 (4/5) | +0.143 (5/5) | +0.033 (3/5) | +0.219 (4/5) |
| both:c,e0.5,t0.8 | +0.091 (4/5) | +0.113 (5/5) | +0.072 (2/5) | +0.193 (4/5) |
