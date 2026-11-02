# Layer L report: the new-skill learning path (seeds [710, 711, 712, 713, 714])

The pool was re-taught with the same seed; the recorded teaching numbers reproduced exactly in 5 of 5 seeds. Same stream and policies as E-055; the arms differ only in how verified items reach the pool (fsmoe/learn_lab.py). learned = answers a never-taught skill with accuracy of at least 0.5 on rows the stream never contained. `flipped` = known held-out-wording items that were right before the stream and wrong after; the three right-hand columns split them by the class they are now assigned to.

| policy | budget | mode | verified | skills met / withheld | learned fraction | novel accuracy (end) | known ho before | known ho after | known tw after | flipped (share of known ho) | flipped to a new skill | to its twin | to another known skill | consolidations | replay memory MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.10 | interleaved | 533 [528, 537] | 28.0 [28.0, 28.0] / 28 | 0.34 [-0.03, 0.71] | 0.366 [0.049, 0.682] | 0.885 [0.874, 0.895] | 0.728 [0.494, 0.961] | 0.955 [0.915, 0.995] | 0.158 [-0.071, 0.388] | 0.97 [0.95, 1.00] | 0.01 [-0.01, 0.03] | 0.01 [0.00, 0.03] | 0.0 [0.0, 0.0] | 0.00 [0.00, 0.00] |
| oracle | 0.10 | focused | 533 [528, 537] | 28.0 [28.0, 28.0] / 28 | 0.71 [0.63, 0.80] | 0.638 [0.568, 0.707] | 0.885 [0.874, 0.895] | 0.665 [0.569, 0.761] | 0.946 [0.926, 0.966] | 0.228 [0.142, 0.314] | 0.95 [0.93, 0.98] | 0.01 [0.01, 0.02] | 0.03 [0.01, 0.05] | 59.6 [57.3, 61.9] | 0.00 [0.00, 0.00] |
| oracle | 0.10 | focused_replay | 533 [528, 537] | 28.0 [28.0, 28.0] / 28 | 0.77 [0.69, 0.85] | 0.677 [0.621, 0.733] | 0.885 [0.874, 0.895] | 0.709 [0.633, 0.785] | 0.930 [0.904, 0.957] | 0.189 [0.118, 0.261] | 0.69 [0.66, 0.73] | 0.07 [0.05, 0.10] | 0.23 [0.20, 0.26] | 59.6 [57.3, 61.9] | 2.05 [2.05, 2.05] |
| diverse | 0.10 | interleaved | 552 [552, 552] | 23.6 [21.5, 25.7] / 28 | 0.21 [-0.03, 0.44] | 0.238 [0.045, 0.431] | 0.885 [0.874, 0.895] | 0.713 [0.602, 0.824] | 0.951 [0.935, 0.968] | 0.193 [0.083, 0.303] | 0.37 [0.15, 0.59] | 0.11 [0.05, 0.18] | 0.52 [0.29, 0.75] | 0.0 [0.0, 0.0] | 0.00 [0.00, 0.00] |
| diverse | 0.10 | focused | 552 [552, 552] | 23.6 [21.5, 25.7] / 28 | 0.27 [0.17, 0.37] | 0.253 [0.178, 0.328] | 0.885 [0.874, 0.895] | 0.680 [0.633, 0.728] | 0.954 [0.947, 0.961] | 0.235 [0.186, 0.284] | 0.35 [0.21, 0.50] | 0.13 [0.09, 0.16] | 0.52 [0.36, 0.68] | 19.8 [16.0, 23.6] | 0.00 [0.00, 0.00] |
| diverse | 0.10 | focused_replay | 552 [552, 552] | 23.6 [21.5, 25.7] / 28 | 0.25 [0.16, 0.34] | 0.284 [0.222, 0.345] | 0.885 [0.874, 0.895] | 0.788 [0.763, 0.812] | 0.954 [0.949, 0.958] | 0.126 [0.097, 0.156] | 0.20 [0.13, 0.26] | 0.14 [0.12, 0.16] | 0.67 [0.60, 0.73] | 19.8 [16.0, 23.6] | 2.05 [2.05, 2.05] |

## Wording-balance arms (added after the first cells; the old skills see the stream's wordings through verified known items)

(5 seeds)

| arm | policy | budget | mode | verified | skills met / withheld | learned fraction | novel accuracy (end) | known ho before | known ho after | known tw after | flipped (share of known ho) | flipped to a new skill | to its twin | to another known skill |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pool | balanced | 0.10 | interleaved | 552 [552, 552] | 11.2 [9.6, 12.8] / 28 | 0.03 [-0.03, 0.09] | 0.030 [-0.027, 0.086] | 0.885 [0.874, 0.895] | 0.867 [0.838, 0.896] | 0.972 [0.965, 0.978] | 0.024 [-0.017, 0.065] | 0.01 [-0.02, 0.05] | 0.15 [-0.12, 0.42] | 0.64 [0.13, 1.15] |
| pool | balanced | 0.10 | focused_replay | 552 [552, 552] | 11.2 [9.6, 12.8] / 28 | 0.09 [0.02, 0.17] | 0.099 [0.050, 0.148] | 0.885 [0.874, 0.895] | 0.818 [0.781, 0.855] | 0.964 [0.960, 0.968] | 0.085 [0.045, 0.124] | 0.34 [0.01, 0.67] | 0.13 [0.05, 0.21] | 0.53 [0.28, 0.78] |

## Global closed-form head next to the pool (addendum 2 of the design)

(5 seeds; after teaching only, the pool with the global head reads taught skills 0.959 [0.952, 0.966] and held-out wordings 0.895 [0.890, 0.900], the pool alone 0.965 [0.960, 0.970] and 0.895 [0.889, 0.900])

| arm | policy | budget | mode | verified | skills met / withheld | learned fraction | novel accuracy (end) | known ho before | known ho after | known tw after | flipped (share of known ho) | flipped to a new skill | to its twin | to another known skill |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pool + global head | oracle | 0.10 | interleaved | 533 [528, 537] | 28.0 [28.0, 28.0] / 28 | 0.79 [0.69, 0.88] | 0.707 [0.625, 0.790] | 0.885 [0.878, 0.893] | 0.816 [0.760, 0.873] | 0.968 [0.963, 0.973] | 0.073 [0.020, 0.126] | 0.95 [0.91, 0.98] | 0.01 [0.00, 0.02] | 0.04 [0.00, 0.08] |
| pool + global head | diverse | 0.10 | interleaved | 552 [552, 552] | 23.6 [21.5, 25.7] / 28 | 0.36 [0.23, 0.50] | 0.380 [0.307, 0.452] | 0.885 [0.878, 0.893] | 0.858 [0.844, 0.872] | 0.966 [0.961, 0.971] | 0.045 [0.033, 0.056] | 0.56 [0.33, 0.79] | 0.08 [0.04, 0.12] | 0.36 [0.12, 0.60] |
| pool + global head | balanced | 0.10 | interleaved | 552 [552, 552] | 11.2 [9.6, 12.8] / 28 | 0.10 [0.05, 0.15] | 0.140 [0.116, 0.164] | 0.885 [0.878, 0.893] | 0.882 [0.869, 0.894] | 0.969 [0.963, 0.974] | 0.015 [0.005, 0.026] | 0.61 [0.44, 0.79] | 0.13 [0.07, 0.19] | 0.25 [0.06, 0.44] |

## The wording-confound test (closed-form classifier, no pool)

A classifier fitted on the taught skills' teaching rows plus m examples of each of the 28 withheld skills; read on taught skills in held-out wordings (known ho) and on the withheld skills (novel); `errors to new skills` = share of the known-ho errors that land on a withheld skill. Mean [95 % interval over seeds].

## m = 10 examples per new skill (5 seeds)

| new skills' examples | known ho | novel | known-ho errors that land on a new skill |
|---|---|---|---|
| none | 0.878 [0.871, 0.885] | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] |
| tw | 0.858 [0.848, 0.868] | 0.778 [0.760, 0.796] | 0.21 [0.18, 0.25] |
| ho | 0.789 [0.780, 0.799] | 0.813 [0.793, 0.832] | 0.60 [0.58, 0.63] |
| mixed | 0.813 [0.803, 0.824] | 0.833 [0.816, 0.851] | 0.53 [0.48, 0.58] |
| balanced | 0.849 [0.842, 0.855] | 0.828 [0.810, 0.846] | 0.44 [0.41, 0.47] |

## m = 20 examples per new skill (5 seeds)

| new skills' examples | known ho | novel | known-ho errors that land on a new skill |
|---|---|---|---|
| none | 0.878 [0.871, 0.885] | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] |
| tw | 0.855 [0.846, 0.864] | 0.807 [0.793, 0.821] | 0.25 [0.23, 0.28] |
| ho | 0.773 [0.761, 0.785] | 0.859 [0.834, 0.884] | 0.67 [0.64, 0.69] |
| mixed | 0.800 [0.787, 0.812] | 0.873 [0.855, 0.892] | 0.59 [0.56, 0.63] |
| balanced | 0.854 [0.843, 0.864] | 0.860 [0.839, 0.881] | 0.47 [0.46, 0.49] |

