# Layer S report: the system on the three-domain stream (seeds [710, 711, 712, 713, 714])

Taught in the order arithmetic, code, language; read on test rows the pool never trained on. tw = taught skills in a teaching wording, ho = taught skills in a HELD-OUT wording. Mean [95 % interval over seeds]. FS = the validated pool with the closed-form head in every expert (`ULDA`). Baselines get the pool's memory (kNN, replay) and their own tuned recipe (dev seed).

## S1 teaching

| system | tw (mean of 3 domains) | ho (mean of 3 domains) | tw arith | tw code | tw lang | ho arith | ho code | ho lang | forgetting arith | forgetting code | experts | memory MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FS | 0.965 [0.960, 0.970] | 0.895 [0.889, 0.900] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.895 [0.880, 0.910] | 0.940 [0.925, 0.955] | 0.939 [0.937, 0.942] | 0.805 [0.800, 0.809] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 3.0 [3.0, 3.0] | 12.9 [12.5, 13.2] |
| replay_eqmem | 0.952 [0.944, 0.961] | 0.832 [0.815, 0.849] | 0.999 [0.997, 1.001] | 1.000 [1.000, 1.000] | 0.858 [0.832, 0.884] | 0.814 [0.774, 0.855] | 0.928 [0.908, 0.949] | 0.754 [0.734, 0.773] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.0 [1.0, 1.0] | 8.8 [8.8, 8.8] |
| knn_eqmem | 0.934 [0.930, 0.938] | 0.815 [0.810, 0.820] | 0.993 [0.987, 0.998] | 1.000 [1.000, 1.000] | 0.811 [0.797, 0.824] | 0.804 [0.790, 0.819] | 0.960 [0.952, 0.967] | 0.680 [0.670, 0.690] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.0 [1.0, 1.0] | 8.4 [8.4, 8.4] |
| lda | 0.945 [0.935, 0.956] | 0.866 [0.859, 0.873] | 0.995 [0.990, 0.999] | 1.000 [1.000, 1.000] | 0.841 [0.813, 0.869] | 0.887 [0.863, 0.911] | 0.964 [0.958, 0.969] | 0.748 [0.740, 0.755] | 0.005 [0.001, 0.010] | 0.000 [0.000, 0.000] | 1.0 [1.0, 1.0] | 0.4 [0.4, 0.4] |
| joint | 0.961 [0.951, 0.971] | 0.821 [0.808, 0.835] | 0.999 [0.997, 1.001] | 1.000 [1.000, 1.000] | 0.884 [0.856, 0.913] | 0.819 [0.776, 0.863] | 0.879 [0.854, 0.905] | 0.766 [0.752, 0.780] | 0.001 [-0.001, 0.003] | 0.000 [0.000, 0.000] | 1.0 [1.0, 1.0] | 8.8 [8.8, 8.8] |

FS events over the whole stream (summed over seeds): alarm 4, extend 4, router_refit 15, set_held 1, spawn 15.

## S3 teams (held-out wordings, taught skills)

| answer | accuracy | mean team size |
|---|---|---|
| top-1 (the pool's own routing) | 0.882 [0.878, 0.885] | 1 |
| adaptive team (mass 0.99, cap 3, tau 0.1) | 0.878 [0.874, 0.883] | 1.02 [1.01, 1.02] |

## S2 deployment: a budget of verified outcomes, spent by a policy

A withheld skill counts as learned when the pool answers it with accuracy >= 0.5 on rows the stream never contained. `precision` = share of verified items that were novel (the stream is about 14 % novel). ho before / after = accuracy on held-out wordings of taught skills before and after the stream.

| policy | budget | verified | precision | skills met / withheld | learned fraction | novel accuracy (end) | novel accuracy (online, 2nd half) | known ho before | known ho after | known tw after | experts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.00 | 0 [0, 0] | n/a | 0.0 [0.0, 0.0] / 28 | 0.00 [0.00, 0.00] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.885 [0.874, 0.895] | 0.885 [0.874, 0.895] | 0.972 [0.966, 0.978] | 3.0 [3.0, 3.0] |
| random | 0.05 | 276 [276, 276] | 0.13 [0.11, 0.15] | 15.8 [14.2, 17.4] / 28 | 0.08 [-0.06, 0.21] | 0.088 [-0.053, 0.229] | 0.117 [-0.083, 0.317] | 0.885 [0.874, 0.895] | 0.787 [0.667, 0.906] | 0.959 [0.940, 0.977] | 4.0 [4.0, 4.0] |
| novelty | 0.05 | 276 [276, 276] | 0.26 [0.25, 0.28] | 4.0 [3.1, 4.9] / 28 | 0.10 [0.08, 0.12] | 0.094 [0.077, 0.112] | 0.191 [0.131, 0.250] | 0.885 [0.874, 0.895] | 0.760 [0.610, 0.910] | 0.964 [0.941, 0.987] | 4.0 [4.0, 4.0] |
| uncertainty | 0.05 | 276 [276, 276] | 0.10 [0.08, 0.11] | 15.0 [13.0, 17.0] / 28 | 0.06 [0.02, 0.11] | 0.089 [0.054, 0.124] | 0.024 [0.016, 0.032] | 0.885 [0.874, 0.895] | 0.873 [0.856, 0.890] | 0.971 [0.965, 0.977] | 3.8 [3.2, 4.4] |
| oracle | 0.05 | 275 [275, 275] | 1.00 [1.00, 1.00] | 28.0 [28.0, 28.0] / 28 | 0.17 [-0.11, 0.45] | 0.214 [0.006, 0.421] | 0.264 [0.073, 0.455] | 0.885 [0.874, 0.895] | 0.828 [0.732, 0.924] | 0.970 [0.961, 0.978] | 5.4 [4.0, 6.8] |
| random | 0.10 | 552 [552, 552] | 0.13 [0.12, 0.13] | 21.4 [19.0, 23.8] / 28 | 0.05 [-0.09, 0.19] | 0.061 [-0.098, 0.220] | 0.080 [-0.136, 0.296] | 0.885 [0.874, 0.895] | 0.862 [0.806, 0.918] | 0.968 [0.958, 0.977] | 4.2 [3.6, 4.8] |
| novelty | 0.10 | 552 [552, 552] | 0.22 [0.21, 0.22] | 8.2 [6.0, 10.4] / 28 | 0.14 [0.12, 0.16] | 0.153 [0.127, 0.180] | 0.337 [0.260, 0.413] | 0.885 [0.874, 0.895] | 0.751 [0.678, 0.824] | 0.962 [0.957, 0.967] | 4.0 [4.0, 4.0] |

## S2 extra (added after the main readout): the oracle's budget curve and a diversity-aware novelty policy

The pool was re-taught with the same seed; the recorded teaching numbers reproduced exactly in 5 of 5 seeds. `diverse` = novelty ranking restricted to inputs far from each other (fsmoe.system_lab.diverse_pick).

| policy | budget | verified | precision | skills met / withheld | learned fraction | novel accuracy (end) | known ho before | known ho after | experts |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.10 | 533 [528, 537] | 1.00 [1.00, 1.00] | 28.0 [28.0, 28.0] / 28 | 0.31 [-0.11, 0.73] | 0.336 [-0.034, 0.706] | 0.885 [0.874, 0.895] | 0.738 [0.493, 0.982] | 5.0 [4.1, 5.9] |
| oracle | 0.20 | 790 [783, 797] | 1.00 [1.00, 1.00] | 28.0 [28.0, 28.0] / 28 | 0.21 [-0.24, 0.67] | 0.236 [-0.145, 0.617] | 0.885 [0.874, 0.895] | 0.783 [0.547, 1.020] | 4.6 [3.9, 5.3] |
| oracle | 0.40 | 800 [800, 800] | 1.00 [1.00, 1.00] | 28.0 [28.0, 28.0] / 28 | 0.23 [-0.24, 0.70] | 0.234 [-0.142, 0.611] | 0.885 [0.874, 0.895] | 0.782 [0.541, 1.023] | 4.4 [3.7, 5.1] |
| diverse | 0.05 | 276 [276, 276] | 0.26 [0.22, 0.29] | 11.2 [9.6, 12.8] / 28 | 0.16 [0.10, 0.22] | 0.166 [0.150, 0.183] | 0.885 [0.874, 0.895] | 0.629 [0.574, 0.685] | 4.2 [3.6, 4.8] |
| diverse | 0.10 | 552 [552, 552] | 0.26 [0.25, 0.28] | 23.6 [21.5, 25.7] / 28 | 0.28 [-0.04, 0.59] | 0.269 [0.034, 0.504] | 0.885 [0.874, 0.895] | 0.740 [0.644, 0.837] | 5.0 [4.1, 5.9] |
