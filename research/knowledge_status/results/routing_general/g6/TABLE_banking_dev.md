# Lab G6, banking: the pool fed by different keys (seeds 150-154; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-class | taught classes: plain / shifted | new class: plain / shifted | all four groups | LM passes per seed | cheap-reader answers (disagree) |
|---|---|---|---|---|---|---|---|---|---|---|
| emb | 5 | 1.2 / 2.4 | 0.40 | 0.80 | 0.14 | 0.887 / 0.870 | 0.940 / 0.940 | 0.909 | 0 | - |
| emb:c | 5 | 1.2 / 2.4 | 0.40 | 0.80 | 0.14 | 0.887 / 0.870 | 0.940 / 0.940 | 0.909 | 0 | - |
| key:c | 5 | 1.0 / 1.0 | 0.00 | 0.00 | nan | 0.460 / 0.457 | 0.200 / 0.190 | 0.327 | 682 | - |
| both:c | 5 | 1.4 / 3.0 | 0.80 | 0.80 | 0.18 | 0.970 / 0.927 | 0.800 / 0.960 | 0.914 | 682 | - |
| both:c,e0.5 | 4 | 2.0 / 2.5 | 0.25 | 0.25 | 0.05 | 0.987 / 0.946 | 0.975 / 0.938 | 0.961 | 682 | - |
| both:c,e0.5,s | 4 | 1.0 / 2.0 | 0.25 | 0.75 | 0.05 | 0.975 / 0.925 | 0.900 / 0.888 | 0.922 | 682 | - |

## Paired against `emb` (same seeds; mean difference with its standard error, and the number of seeds in which the arm is better)

| arm | all four groups | taught classes, shifted | new class (both wordings) | NMI expert-class |
|---|---|---|---|---|
| emb:c | +0.000 (0.000), 0/5 | +0.000 (0.000), 0/5 | +0.000 (0.000), 0/5 | +0.000 (0.000), 0/5 |
| key:c | -0.582 (0.084), 0/5 | -0.413 (0.078), 0/5 | -0.745 (0.206), 1/5 | +nan (nan), 1/5 |
| both:c | +0.005 (0.048), 4/5 | +0.057 (0.019), 4/5 | -0.060 (0.086), 2/5 | +0.036 (0.127), 2/5 |
| both:c,e0.5 | +0.031 (0.007), 4/4 | +0.037 (0.008), 4/4 | +0.031 (0.012), 3/4 | -0.136 (0.052), 0/4 |
| both:c,e0.5,s | -0.008 (0.018), 1/4 | +0.017 (0.022), 2/4 | -0.031 (0.026), 1/4 | -0.132 (0.019), 0/4 |
