# Lab G6, newsgroups: the pool fed by different keys (seeds 150-154; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-class | taught classes: plain / shifted | new class: plain / shifted | all four groups | LM passes per seed | cheap-reader answers (disagree) |
|---|---|---|---|---|---|---|---|---|---|---|
| emb | 3 | 2.7 / 2.7 | 0.00 | 0.00 | nan | 0.797 / 0.764 | 0.000 / 0.000 | 0.390 | 0 | - |
| emb:a0.2 | 3 | 2.7 / 2.7 | 0.00 | 0.00 | nan | 0.797 / 0.764 | 0.000 / 0.000 | 0.390 | 0 | - |
| emb:a0.5 | 3 | 2.7 / 2.7 | 0.00 | 0.00 | nan | 0.797 / 0.764 | 0.000 / 0.000 | 0.390 | 0 | - |
| emb:a0.5,p16 | 3 | 2.7 / 3.0 | 0.00 | 0.33 | nan | 0.767 / 0.722 | 0.267 / 0.250 | 0.501 | 0 | - |
| both:c,e0.5 | 3 | 3.0 / 3.0 | 0.00 | 0.00 | nan | 0.822 / 0.794 | 0.042 / 0.058 | 0.429 | 1086 | - |
| both:c,e0.5,a0.2 | 3 | 3.0 / 3.0 | 0.00 | 0.00 | nan | 0.822 / 0.794 | 0.042 / 0.058 | 0.429 | 1086 | - |
| both:c,e0.5,a0.5 | 3 | 3.0 / 3.0 | 0.00 | 0.00 | nan | 0.822 / 0.794 | 0.042 / 0.058 | 0.429 | 1086 | - |
| both:c,e0.5,a0.5,p16 | 3 | 3.0 / 3.3 | 0.00 | 0.33 | nan | 0.811 / 0.758 | 0.267 / 0.342 | 0.544 | 1086 | - |

## Paired against `emb` (same seeds; mean difference with its standard error, and the number of seeds in which the arm is better)

| arm | all four groups | taught classes, shifted | new class (both wordings) | NMI expert-class |
|---|---|---|---|---|
| emb:a0.2 | +0.000 (0.000), 0/3 | +0.000 (0.000), 0/3 | +0.000 (0.000), 0/3 | +nan (nan), 0/3 |
| emb:a0.5 | +0.000 (0.000), 0/3 | +0.000 (0.000), 0/3 | +0.000 (0.000), 0/3 | +nan (nan), 0/3 |
| emb:a0.5,p16 | +0.111 (0.111), 1/3 | -0.042 (0.042), 0/3 | +0.258 (0.258), 1/3 | +nan (nan), 0/3 |
| both:c,e0.5 | +0.039 (0.020), 3/3 | +0.031 (0.006), 3/3 | +0.050 (0.050), 1/3 | +nan (nan), 0/3 |
| both:c,e0.5,a0.2 | +0.039 (0.020), 3/3 | +0.031 (0.006), 3/3 | +0.050 (0.050), 1/3 | +nan (nan), 0/3 |
| both:c,e0.5,a0.5 | +0.039 (0.020), 3/3 | +0.031 (0.006), 3/3 | +0.050 (0.050), 1/3 | +nan (nan), 0/3 |
| both:c,e0.5,a0.5,p16 | +0.154 (0.104), 3/3 | -0.006 (0.039), 2/3 | +0.304 (0.233), 2/3 | +nan (nan), 0/3 |
