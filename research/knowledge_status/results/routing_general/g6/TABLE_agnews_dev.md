# Lab G6, agnews: the pool fed by different keys (seeds 150-152; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-class | taught classes: plain / shifted | new class: plain / shifted | all four groups | LM passes per seed | cheap-reader answers (disagree) |
|---|---|---|---|---|---|---|---|---|---|---|
| emb | 3 | 3.0 / 3.3 | 0.33 | 0.00 | nan | 0.772 / 0.825 | 0.033 / 0.133 | 0.441 | 0 | - |
| emb:c | 3 | 3.0 / 3.3 | 0.33 | 0.00 | nan | 0.772 / 0.825 | 0.033 / 0.133 | 0.441 | 0 | - |
| key:c | 3 | 1.0 / 1.0 | 0.00 | 0.00 | nan | 0.414 / 0.403 | 0.000 / 0.000 | 0.204 | 1088 | - |
| both:c | 3 | 2.7 / 2.7 | 0.00 | 0.00 | nan | 0.853 / 0.828 | 0.000 / 0.000 | 0.420 | 1088 | - |
| both:c,e0.5 | 3 | 2.7 / 3.0 | 0.33 | 0.00 | nan | 0.858 / 0.844 | 0.042 / 0.150 | 0.474 | 1088 | - |
| both:c,e0.5,s | 3 | 2.0 / 2.0 | 0.00 | 0.00 | 0.00 | 0.811 / 0.886 | 0.700 / 0.758 | 0.789 | 1088 | - |

## Paired against `emb` (same seeds; mean difference with its standard error, and the number of seeds in which the arm is better)

| arm | all four groups | taught classes, shifted | new class (both wordings) | NMI expert-class |
|---|---|---|---|---|
| emb:c | +0.000 (0.000), 0/3 | +0.000 (0.000), 0/3 | +0.000 (0.000), 0/3 | +nan (nan), 0/3 |
| key:c | -0.237 (0.092), 0/3 | -0.422 (0.124), 0/3 | -0.083 (0.083), 0/3 | +nan (nan), 0/3 |
| both:c | -0.021 (0.040), 1/3 | +0.003 (0.041), 1/3 | -0.083 (0.083), 0/3 | +nan (nan), 0/3 |
| both:c,e0.5 | +0.033 (0.017), 2/3 | +0.019 (0.042), 1/3 | +0.012 (0.012), 1/3 | +nan (nan), 0/3 |
| both:c,e0.5,s | +0.348 (0.101), 3/3 | +0.061 (0.064), 2/3 | +0.646 (0.187), 3/3 | +nan (nan), 0/3 |
