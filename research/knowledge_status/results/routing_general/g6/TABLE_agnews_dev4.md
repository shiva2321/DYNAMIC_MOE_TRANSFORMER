# Lab G6, agnews: the pool fed by different keys (seeds 150-154; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-class | taught classes: plain / shifted | new class: plain / shifted | all four groups | LM passes per seed | cheap-reader answers (disagree) |
|---|---|---|---|---|---|---|---|---|---|---|
| emb | 3 | 3.0 / 3.3 | 0.33 | 0.00 | nan | 0.772 / 0.825 | 0.033 / 0.133 | 0.441 | 0 | - |
| emb:p16 | 3 | 3.0 / 4.0 | 0.33 | 0.67 | 0.04 | 0.728 / 0.781 | 0.517 / 0.675 | 0.675 | 0 | - |
| key:c,p16 | 3 | 1.0 / 1.0 | 0.00 | 0.00 | nan | 0.414 / 0.403 | 0.000 / 0.000 | 0.204 | 1088 | - |
| both:c,p16 | 3 | 2.7 / 3.3 | 0.00 | 0.67 | nan | 0.861 / 0.789 | 0.475 / 0.600 | 0.681 | 1088 | - |
| both:c,e0.5,p16 | 3 | 2.7 / 3.3 | 0.33 | 0.33 | 0.05 | 0.817 / 0.761 | 0.567 / 0.758 | 0.726 | 1088 | - |
| both:c,e0.5,s,p16 | 3 | 2.0 / 2.0 | 0.00 | 0.00 | 0.00 | 0.806 / 0.892 | 0.725 / 0.808 | 0.808 | 1088 | - |
| both:c,e0.5,t0.9,m0.8,p16 | 3 | 2.7 / 3.3 | 0.33 | 0.33 | 0.05 | 0.817 / 0.761 | 0.567 / 0.750 | 0.724 | 1725 | 0% (0.0%) |

## Paired against `emb` (same seeds; mean difference with its standard error, and the number of seeds in which the arm is better)

| arm | all four groups | taught classes, shifted | new class (both wordings) | NMI expert-class |
|---|---|---|---|---|
| emb:p16 | +0.234 (0.141), 2/3 | -0.044 (0.061), 2/3 | +0.513 (0.307), 2/3 | +nan (nan), 1/3 |
| key:c,p16 | -0.237 (0.092), 0/3 | -0.422 (0.124), 0/3 | -0.083 (0.083), 0/3 | +nan (nan), 0/3 |
| both:c,p16 | +0.240 (0.169), 2/3 | -0.036 (0.034), 1/3 | +0.454 (0.352), 2/3 | +nan (nan), 1/3 |
| both:c,e0.5,p16 | +0.285 (0.111), 3/3 | -0.064 (0.056), 0/3 | +0.579 (0.252), 3/3 | +nan (nan), 1/3 |
| both:c,e0.5,s,p16 | +0.367 (0.085), 3/3 | +0.067 (0.068), 2/3 | +0.683 (0.151), 3/3 | +nan (nan), 0/3 |
| both:c,e0.5,t0.9,m0.8,p16 | +0.283 (0.110), 3/3 | -0.064 (0.056), 0/3 | +0.575 (0.250), 3/3 | +nan (nan), 1/3 |
