# Lab G5, the pool fed by different keys (seeds 150-154; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-skill | acc taught skills (teach / new wording) | acc new skill (teach / new wording) | acc all four groups | unassigned share | reader s per seed | pool s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| emb | 5 | 2.0 / 2.6 | 0.00 | 0.60 | 0.08 | 0.720 (0.753 / 0.687) | 0.410 (0.440 / 0.380) | 0.565 | 0.16 | 0 | 4 |
| key | 5 | 2.0 / 2.4 | 0.00 | 0.40 | 0.15 | 0.658 (0.642 / 0.675) | 0.275 (0.285 / 0.265) | 0.467 | 0.14 | 82 | 3 |
| both | 5 | 2.6 / 3.2 | 0.00 | 0.60 | 0.19 | 0.769 (0.787 / 0.752) | 0.402 (0.410 / 0.395) | 0.586 | 0.14 | 0 | 5 |
| oracle | 5 | 2.0 / 3.8 | 0.60 | 1.20 | 0.36 | 0.812 (0.825 / 0.800) | 0.593 (0.610 / 0.575) | 0.702 | 0.12 | 0 | 4 |

## Paired against `emb` (same seeds; mean difference, and the number of seeds in which the arm is better)

| arm | acc all | acc taught in new wordings | acc new skill | NMI expert-skill |
|---|---|---|---|---|
| key | -0.098 (1/5) | -0.012 (2/5) | -0.135 (1/5) | +0.078 (3/5) |
| both | +0.021 (3/5) | +0.065 (4/5) | -0.007 (3/5) | +0.112 (3/5) |
| oracle | +0.137 (4/5) | +0.113 (5/5) | +0.183 (3/5) | +0.279 (5/5) |
