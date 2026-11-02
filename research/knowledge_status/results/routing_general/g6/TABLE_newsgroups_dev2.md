# Lab G6, newsgroups: the pool fed by different keys (seeds 150-154; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-class | taught classes: plain / shifted | new class: plain / shifted | all four groups | LM passes per seed | cheap-reader answers (disagree) |
|---|---|---|---|---|---|---|---|---|---|---|
| emb | 3 | 2.7 / 2.7 | 0.00 | 0.00 | nan | 0.797 / 0.764 | 0.000 / 0.000 | 0.390 | 0 | - |
| both:c,e0.5 | 3 | 3.0 / 3.0 | 0.00 | 0.00 | nan | 0.822 / 0.794 | 0.042 / 0.058 | 0.429 | 1086 | - |
| both:c,e0.5,w3 | 3 | 2.7 / 2.7 | 0.00 | 0.00 | nan | 0.808 / 0.781 | 0.017 / 0.033 | 0.410 | 1086 | - |
| both:c,e0.5,w6 | 3 | 2.7 / 2.7 | 0.00 | 0.00 | nan | 0.797 / 0.783 | 0.000 / 0.017 | 0.399 | 1086 | - |
| both:c,e0.5,s | 3 | 2.0 / 2.0 | 0.00 | 0.00 | 0.00 | 0.822 / 0.769 | 0.700 / 0.733 | 0.756 | 1086 | - |

## Paired against `emb` (same seeds; mean difference with its standard error, and the number of seeds in which the arm is better)

| arm | all four groups | taught classes, shifted | new class (both wordings) | NMI expert-class |
|---|---|---|---|---|
| both:c,e0.5 | +0.039 (0.020), 3/3 | +0.031 (0.006), 3/3 | +0.050 (0.050), 1/3 | +nan (nan), 0/3 |
| both:c,e0.5,w3 | +0.019 (0.003), 3/3 | +0.017 (0.017), 2/3 | +0.025 (0.025), 1/3 | +nan (nan), 0/3 |
| both:c,e0.5,w6 | +0.009 (0.009), 2/3 | +0.019 (0.020), 2/3 | +0.008 (0.008), 1/3 | +nan (nan), 0/3 |
| both:c,e0.5,s | +0.366 (0.017), 3/3 | +0.006 (0.003), 2/3 | +0.717 (0.029), 3/3 | +nan (nan), 0/3 |
