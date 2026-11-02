# Lab G6, banking: the pool fed by different keys (seeds 150-154; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-class | taught classes: plain / shifted | new class: plain / shifted | all four groups | LM passes per seed | cheap-reader answers (disagree) |
|---|---|---|---|---|---|---|---|---|---|---|
| emb | 5 | 1.2 / 2.2 | 0.40 | 0.60 | 0.13 | 0.930 / 0.900 | 0.960 / 0.900 | 0.922 | 0 | - |
| both:c,e0.5 | 5 | 2.0 / 2.8 | 0.40 | 0.40 | 0.08 | 0.977 / 0.930 | 0.970 / 0.980 | 0.964 | 682 | - |
| both:c,e0.5,w3 | 5 | 2.2 / 2.6 | 0.20 | 0.20 | 0.05 | 0.980 / 0.917 | 0.960 / 0.950 | 0.952 | 682 | - |
| both:c,e0.5,w6 | 5 | 2.2 / 2.8 | 0.40 | 0.20 | 0.05 | 0.970 / 0.903 | 0.850 / 0.940 | 0.916 | 682 | - |
| both:c,e0.5,s | 5 | 1.0 / 1.4 | 0.20 | 0.20 | 0.03 | 0.907 / 0.877 | 0.770 / 0.850 | 0.851 | 682 | - |

## Paired against `emb` (same seeds; mean difference with its standard error, and the number of seeds in which the arm is better)

| arm | all four groups | taught classes, shifted | new class (both wordings) | NMI expert-class |
|---|---|---|---|---|
| both:c,e0.5 | +0.042 (0.010), 5/5 | +0.030 (0.017), 3/5 | +0.045 (0.031), 3/5 | -0.053 (0.081), 1/5 |
| both:c,e0.5,w3 | +0.029 (0.019), 4/5 | +0.017 (0.014), 3/5 | +0.025 (0.040), 2/5 | -0.085 (0.090), 1/5 |
| both:c,e0.5,w6 | -0.007 (0.044), 4/5 | +0.003 (0.019), 2/5 | -0.035 (0.083), 2/5 | -0.085 (0.090), 1/5 |
| both:c,e0.5,s | -0.072 (0.089), 3/5 | -0.023 (0.029), 2/5 | -0.120 (0.138), 1/5 | -0.105 (0.044), 0/5 |
