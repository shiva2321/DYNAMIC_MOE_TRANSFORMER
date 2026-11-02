# Lab G6, newsgroups: the pool fed by different keys (seeds 150-154; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-class | taught classes: plain / shifted | new class: plain / shifted | all four groups | LM passes per seed | cheap-reader answers (disagree) |
|---|---|---|---|---|---|---|---|---|---|---|
| emb | 3 | 2.7 / 2.7 | 0.00 | 0.00 | nan | 0.797 / 0.764 | 0.000 / 0.000 | 0.390 | 0 | - |
| emb:p16 | 3 | 2.7 / 3.0 | 0.00 | 0.33 | nan | 0.767 / 0.722 | 0.267 / 0.250 | 0.501 | 0 | - |
| key:c,p16 | 3 | 1.0 / 1.0 | 0.00 | 0.00 | nan | 0.417 / 0.444 | 0.000 / 0.000 | 0.215 | 1086 | - |
| both:c,p16 | 3 | 3.0 / 3.0 | 0.00 | 0.00 | nan | 0.806 / 0.767 | 0.042 / 0.050 | 0.416 | 1086 | - |
| both:c,e0.5,p16 | 3 | 3.0 / 3.3 | 0.00 | 0.33 | nan | 0.811 / 0.758 | 0.267 / 0.342 | 0.544 | 1086 | - |
| both:c,e0.5,s,p16 | 3 | 2.0 / 2.0 | 0.00 | 0.00 | 0.00 | 0.822 / 0.764 | 0.717 / 0.758 | 0.765 | 1086 | - |
| both:c,e0.5,t0.9,m0.8,p16 | 3 | 3.0 / 3.3 | 0.00 | 0.33 | nan | 0.800 / 0.744 | 0.267 / 0.342 | 0.538 | 1725 | 0% (0.0%) |

## Paired against `emb` (same seeds; mean difference with its standard error, and the number of seeds in which the arm is better)

| arm | all four groups | taught classes, shifted | new class (both wordings) | NMI expert-class |
|---|---|---|---|---|
| emb:p16 | +0.111 (0.111), 1/3 | -0.042 (0.042), 0/3 | +0.258 (0.258), 1/3 | +nan (nan), 0/3 |
| key:c,p16 | -0.175 (0.076), 0/3 | -0.319 (0.160), 0/3 | +0.000 (0.000), 0/3 | +nan (nan), 0/3 |
| both:c,p16 | +0.026 (0.019), 2/3 | +0.003 (0.015), 1/3 | +0.046 (0.046), 1/3 | +nan (nan), 0/3 |
| both:c,e0.5,p16 | +0.154 (0.104), 3/3 | -0.006 (0.039), 2/3 | +0.304 (0.233), 2/3 | +nan (nan), 0/3 |
| both:c,e0.5,s,p16 | +0.375 (0.013), 3/3 | +0.000 (0.005), 1/3 | +0.737 (0.019), 3/3 | +nan (nan), 0/3 |
| both:c,e0.5,t0.9,m0.8,p16 | +0.148 (0.108), 3/3 | -0.019 (0.033), 1/3 | +0.304 (0.233), 2/3 | +nan (nan), 0/3 |
