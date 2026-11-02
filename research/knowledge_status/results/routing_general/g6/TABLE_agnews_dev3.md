# Lab G6, agnews: the pool fed by different keys (seeds 150-154; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-class | taught classes: plain / shifted | new class: plain / shifted | all four groups | LM passes per seed | cheap-reader answers (disagree) |
|---|---|---|---|---|---|---|---|---|---|---|
| emb | 3 | 3.0 / 3.3 | 0.33 | 0.00 | nan | 0.772 / 0.825 | 0.033 / 0.133 | 0.441 | 0 | - |
| emb:a0.2 | 3 | 3.0 / 3.3 | 0.33 | 0.00 | nan | 0.772 / 0.825 | 0.033 / 0.133 | 0.441 | 0 | - |
| emb:a0.5 | 3 | 3.0 / 3.3 | 0.33 | 0.00 | nan | 0.772 / 0.825 | 0.033 / 0.133 | 0.441 | 0 | - |
| emb:a0.5,p16 | 3 | 3.0 / 4.0 | 0.33 | 0.67 | 0.04 | 0.728 / 0.781 | 0.517 / 0.675 | 0.675 | 0 | - |
| both:c,e0.5 | 3 | 2.7 / 3.0 | 0.33 | 0.00 | nan | 0.858 / 0.844 | 0.042 / 0.150 | 0.474 | 1088 | - |
| both:c,e0.5,a0.2 | 3 | 2.7 / 3.0 | 0.33 | 0.00 | nan | 0.858 / 0.850 | 0.067 / 0.158 | 0.483 | 1088 | - |
| both:c,e0.5,a0.5 | 3 | 2.7 / 3.0 | 0.33 | 0.00 | nan | 0.858 / 0.850 | 0.067 / 0.158 | 0.483 | 1088 | - |
| both:c,e0.5,a0.5,p16 | 3 | 2.7 / 3.3 | 0.33 | 0.33 | 0.05 | 0.817 / 0.772 | 0.575 / 0.758 | 0.731 | 1088 | - |

## Paired against `emb` (same seeds; mean difference with its standard error, and the number of seeds in which the arm is better)

| arm | all four groups | taught classes, shifted | new class (both wordings) | NMI expert-class |
|---|---|---|---|---|
| emb:a0.2 | +0.000 (0.000), 0/3 | +0.000 (0.000), 0/3 | +0.000 (0.000), 0/3 | +nan (nan), 0/3 |
| emb:a0.5 | +0.000 (0.000), 0/3 | +0.000 (0.000), 0/3 | +0.000 (0.000), 0/3 | +nan (nan), 0/3 |
| emb:a0.5,p16 | +0.234 (0.141), 2/3 | -0.044 (0.061), 2/3 | +0.513 (0.307), 2/3 | +nan (nan), 1/3 |
| both:c,e0.5 | +0.033 (0.017), 2/3 | +0.019 (0.042), 1/3 | +0.012 (0.012), 1/3 | +nan (nan), 0/3 |
| both:c,e0.5,a0.2 | +0.042 (0.024), 2/3 | +0.025 (0.041), 2/3 | +0.029 (0.029), 1/3 | +nan (nan), 0/3 |
| both:c,e0.5,a0.5 | +0.042 (0.024), 2/3 | +0.025 (0.041), 2/3 | +0.029 (0.029), 1/3 | +nan (nan), 0/3 |
| both:c,e0.5,a0.5,p16 | +0.290 (0.106), 3/3 | -0.053 (0.061), 1/3 | +0.583 (0.248), 3/3 | +nan (nan), 1/3 |
