# Lab G6: predictions T1-T8 on the report seeds

8 declared predictions: **3 held, 5 missed** (banking seeds 100-109, news seeds 100-104).

| id | prediction | outcome | measured |
|---|---|---|---|
| T1 | default recipe, news domains: the new-class accuracy of `emb` and of `both:c,e0.5` is at most 0.25 in both domains | MISSED | agnews: emb 0.265, fused 0.130; newsgroups: emb 0.000, fused 0.445 |
| T2 | `emb:p16` raises the new-class accuracy over default `emb` by at least 0.20 in AG News and 0.10 in 20 Newsgroups | HELD | agnews: +0.227; newsgroups: +0.642 |
| T3 | news domains: `both:c,e0.5,p16` within 0.08 of `emb:p16` on all four groups | HELD | agnews: +0.053 (4/5 seeds better); newsgroups: -0.017 (3/5 seeds better) |
| T4 | banking: `both:c,e0.5` beats `emb` on all four groups by at least 0.02 on average and in at least 7 of 10 seeds | MISSED | -0.015, better in 3/10 seeds |
| T5 | `key:c,p16` is at least 0.25 below `emb:p16` on all four groups in all three domains | MISSED | banking: -0.623; agnews: -0.222; newsgroups: -0.503 |
| T6 | banking: the shuffled control is at least 0.03 below `both:c,e0.5,p16`; news: it is not more than 0.10 above `emb:p16` | MISSED | banking -0.011; agnews +0.129; newsgroups +0.121 |
| T7 | banking head arm: answers at least 25 % of the stream, at most 5 % of its answers differ from the full reader, at least 20 % fewer distinct texts asked, four-group accuracy within 0.03 of `both:c,e0.5,p16` | MISSED | 22% answered, 2.1% differ, 24% fewer texts, accuracy +0.017 |
| T8 | news domains: the head answers at most 10 % of the stream's messages | HELD | agnews: 0%; newsgroups: 0% |

## banking: the pool fed by different keys (seeds 100-109; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-class | taught classes: plain / shifted | new class: plain / shifted | all four groups | LM passes per seed | cheap-reader answers (disagree) |
|---|---|---|---|---|---|---|---|---|---|---|
| emb | 10 | 1.3 / 3.0 | 0.80 | 0.90 | 0.20 | 0.922 / 0.877 | 0.990 / 0.980 | 0.942 | 0 | - |
| both:c,e0.5 | 10 | 1.7 / 3.0 | 0.70 | 0.60 | 0.08 | 0.938 / 0.920 | 0.895 / 0.955 | 0.927 | 682 | - |
| emb:p16 | 10 | 1.3 / 3.0 | 0.80 | 0.90 | 0.20 | 0.922 / 0.875 | 0.990 / 0.980 | 0.942 | 0 | - |
| key:c,p16 | 10 | 1.0 / 1.0 | 0.00 | 0.00 | nan | 0.548 / 0.528 | 0.100 / 0.100 | 0.319 | 682 | - |
| both:c,e0.5,p16 | 10 | 1.7 / 3.0 | 0.70 | 0.60 | 0.08 | 0.937 / 0.920 | 0.895 / 0.955 | 0.927 | 682 | - |
| both:c,e0.5,s,p16 | 10 | 1.1 / 1.6 | 0.10 | 0.40 | 0.04 | 0.972 / 0.910 | 0.930 / 0.850 | 0.915 | 682 | - |
| both:c,e0.5,t0.9,m0.8,p16 | 10 | 1.7 / 3.1 | 0.70 | 0.70 | 0.11 | 0.930 / 0.908 | 0.965 / 0.970 | 0.943 | 516 | 22% (2.1%) |

## Paired against `emb` (same seeds; mean difference with its standard error, and the number of seeds in which the arm is better)

| arm | all four groups | taught classes, shifted | new class (both wordings) | NMI expert-class |
|---|---|---|---|---|
| both:c,e0.5 | -0.015 (0.029), 3/10 | +0.043 (0.029), 6/10 | -0.060 (0.050), 2/10 | -0.112 (0.037), 1/10 |
| emb:p16 | -0.000 (0.000), 0/10 | -0.002 (0.002), 0/10 | +0.000 (0.000), 0/10 | +0.000 (0.000), 0/10 |
| key:c,p16 | -0.623 (0.051), 0/10 | -0.348 (0.066), 0/10 | -0.885 (0.098), 0/10 | +nan (nan), 2/10 |
| both:c,e0.5,p16 | -0.015 (0.030), 3/10 | +0.043 (0.029), 6/10 | -0.060 (0.050), 2/10 | -0.112 (0.037), 1/10 |
| both:c,e0.5,s,p16 | -0.027 (0.025), 2/10 | +0.033 (0.015), 6/10 | -0.095 (0.036), 0/10 | -0.159 (0.022), 0/10 |
| both:c,e0.5,t0.9,m0.8,p16 | +0.001 (0.019), 3/10 | +0.032 (0.028), 6/10 | -0.017 (0.013), 2/10 | -0.090 (0.038), 2/10 |


## agnews: the pool fed by different keys (seeds 100-104; mean over seeds)

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


## newsgroups: the pool fed by different keys (seeds 100-104; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-class | taught classes: plain / shifted | new class: plain / shifted | all four groups | LM passes per seed | cheap-reader answers (disagree) |
|---|---|---|---|---|---|---|---|---|---|---|
| emb | 5 | 2.8 / 2.8 | 0.00 | 0.00 | nan | 0.837 / 0.833 | 0.000 / 0.000 | 0.417 | 0 | - |
| both:c,e0.5 | 5 | 2.4 / 3.4 | 0.80 | 0.20 | nan | 0.882 / 0.815 | 0.410 / 0.480 | 0.647 | 1087 | - |
| emb:p16 | 5 | 2.8 / 3.4 | 0.00 | 0.60 | nan | 0.830 / 0.788 | 0.630 / 0.655 | 0.726 | 0 | - |
| key:c,p16 | 5 | 1.0 / 1.0 | 0.00 | 0.00 | nan | 0.453 / 0.440 | 0.000 / 0.000 | 0.223 | 1087 | - |
| both:c,e0.5,p16 | 5 | 2.4 / 3.6 | 0.80 | 0.40 | 0.07 | 0.880 / 0.802 | 0.520 / 0.635 | 0.709 | 1087 | - |
| both:c,e0.5,s,p16 | 5 | 2.0 / 2.0 | 0.00 | 0.00 | 0.00 | 0.893 / 0.868 | 0.830 / 0.795 | 0.847 | 1087 | - |
| both:c,e0.5,t0.9,m0.8,p16 | 5 | 2.4 / 3.6 | 0.80 | 0.40 | 0.07 | 0.880 / 0.802 | 0.515 / 0.625 | 0.705 | 1084 | 0% (0.0%) |

## Paired against `emb` (same seeds; mean difference with its standard error, and the number of seeds in which the arm is better)

| arm | all four groups | taught classes, shifted | new class (both wordings) | NMI expert-class |
|---|---|---|---|---|
| both:c,e0.5 | +0.229 (0.094), 5/5 | -0.018 (0.026), 3/5 | +0.445 (0.183), 4/5 | +nan (nan), 0/5 |
| emb:p16 | +0.308 (0.083), 4/5 | -0.045 (0.023), 1/5 | +0.642 (0.169), 4/5 | +nan (nan), 0/5 |
| key:c,p16 | -0.194 (0.040), 0/5 | -0.393 (0.075), 0/5 | +0.000 (0.000), 0/5 | +nan (nan), 0/5 |
| both:c,e0.5,p16 | +0.292 (0.086), 4/5 | -0.032 (0.025), 2/5 | +0.578 (0.158), 4/5 | +nan (nan), 0/5 |
| both:c,e0.5,s,p16 | +0.429 (0.021), 5/5 | +0.035 (0.013), 5/5 | +0.813 (0.036), 5/5 | +nan (nan), 0/5 |
| both:c,e0.5,t0.9,m0.8,p16 | +0.288 (0.085), 4/5 | -0.032 (0.025), 2/5 | +0.570 (0.157), 4/5 | +nan (nan), 0/5 |

