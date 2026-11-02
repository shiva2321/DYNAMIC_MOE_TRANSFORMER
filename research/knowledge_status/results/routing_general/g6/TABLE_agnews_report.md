# Lab G6, agnews: the pool fed by different keys (seeds 100-104; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-class | taught classes: plain / shifted | new class: plain / shifted | all four groups | LM passes per seed | cheap-reader answers (disagree) |
|---|---|---|---|---|---|---|---|---|---|---|
| emb | 5 | 2.8 / 3.2 | 0.20 | 0.20 | nan | 0.827 / 0.808 | 0.240 / 0.290 | 0.541 | 0 | - |
| both:c,e0.5 | 5 | 2.4 / 2.6 | 0.20 | 0.00 | nan | 0.880 / 0.782 | 0.130 / 0.130 | 0.480 | 1088 | - |
| emb:p16 | 5 | 2.8 / 3.6 | 0.40 | 0.40 | nan | 0.792 / 0.768 | 0.430 / 0.555 | 0.636 | 0 | - |
| key:c,p16 | 5 | 1.0 / 1.0 | 0.00 | 0.00 | nan | 0.643 / 0.580 | 0.220 / 0.215 | 0.415 | 1088 | - |
| both:c,e0.5,p16 | 5 | 2.4 / 3.2 | 0.20 | 0.60 | 0.00 | 0.853 / 0.738 | 0.550 / 0.615 | 0.689 | 1088 | - |
| both:c,e0.5,s,p16 | 5 | 2.0 / 2.0 | 0.00 | 0.00 | 0.00 | 0.852 / 0.825 | 0.685 / 0.700 | 0.765 | 1088 | - |
| both:c,e0.5,t0.9,m0.8,p16 | 5 | 2.4 / 3.2 | 0.20 | 0.60 | 0.00 | 0.853 / 0.738 | 0.550 / 0.615 | 0.689 | 1086 | 0% (0.0%) |

## Paired against `emb` (same seeds; mean difference with its standard error, and the number of seeds in which the arm is better)

| arm | all four groups | taught classes, shifted | new class (both wordings) | NMI expert-class |
|---|---|---|---|---|
| both:c,e0.5 | -0.061 (0.071), 2/5 | -0.027 (0.008), 0/5 | -0.135 (0.123), 0/5 | +nan (nan), 0/5 |
| emb:p16 | +0.095 (0.074), 2/5 | -0.040 (0.044), 2/5 | +0.227 (0.174), 2/5 | +nan (nan), 1/5 |
| key:c,p16 | -0.127 (0.120), 1/5 | -0.228 (0.121), 1/5 | -0.047 (0.218), 1/5 | +nan (nan), 0/5 |
| both:c,e0.5,p16 | +0.148 (0.144), 3/5 | -0.070 (0.019), 0/5 | +0.318 (0.285), 3/5 | +nan (nan), 0/5 |
| both:c,e0.5,s,p16 | +0.224 (0.079), 4/5 | +0.017 (0.009), 4/5 | +0.427 (0.157), 4/5 | +nan (nan), 0/5 |
| both:c,e0.5,t0.9,m0.8,p16 | +0.148 (0.144), 3/5 | -0.070 (0.019), 0/5 | +0.318 (0.285), 3/5 | +nan (nan), 0/5 |
