# Notebook-first pool report (seeds [710, 711, 712, 713, 714])

Mean [95 % interval over seeds]. Network parameters = the pool's parameter count (notebook experts carry one inert scalar each). Memory = the pool's bytes (networks, closed-form heads, replay, signatures, held-out samples). Judging = multiply-adds spent training candidate networks, in billions.

## Function regimes, 10 regimes (growmoe viability stream)

| arm | accuracy | worst regime | experts | experts per regime | network parameters | memory MB | answer k-MACs | judging G-MACs | experts with a network | wall seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| shipped pool, 8-unit networks | 0.355 [0.327, 0.383] | 0.188 [0.159, 0.216] | 6.4 [5.7, 7.1] | 0.64 [0.57, 0.71] | 23859 [21324, 26395] | 3.98 [3.54, 4.42] | 266.0 [197.3, 334.8] | 0.0 [0.0, 0.0] | 6.4 [5.7, 7.1] | 36 [27, 44] |
| shipped pool, 32-unit networks | 0.488 [0.477, 0.498] | 0.436 [0.416, 0.455] | 10.0 [10.0, 10.0] | 1.00 [1.00, 1.00] | 98960 [98960, 98960] | 5.33 [5.20, 5.45] | 377.9 [377.9, 377.9] | 0.0 [0.0, 0.0] | 10.0 [10.0, 10.0] | 28 [24, 31] |
| shipped pool (128-unit networks, the default) | 0.493 [0.489, 0.498] | 0.429 [0.411, 0.446] | 10.0 [10.0, 10.0] | 1.00 [1.00, 1.00] | 345680 [345680, 345680] | 6.31 [6.19, 6.44] | 402.4 [402.4, 402.4] | 0.0 [0.0, 0.0] | 10.0 [10.0, 10.0] | 28 [23, 32] |
| notebook-first | 0.497 [0.488, 0.506] | 0.419 [0.387, 0.451] | 10.0 [10.0, 10.0] | 1.00 [1.00, 1.00] | 3634 [-6427, 13695] | 4.94 [4.80, 5.08] | 369.0 [368.0, 370.0] | 31.0 [31.0, 31.0] | 0.2 [-0.4, 0.8] | 41 [35, 46] |
| notebook-first, candidate sizes 8 and 32 | 0.493 [0.482, 0.503] | 0.406 [0.367, 0.444] | 10.0 [10.0, 10.0] | 1.00 [1.00, 1.00] | 5947 [-782, 12676] | 4.95 [4.84, 5.07] | 369.2 [368.6, 369.8] | 10.5 [10.5, 10.5] | 0.6 [-0.1, 1.3] | 31 [28, 35] |

## Function regimes, 30 regimes

| arm | accuracy | worst regime | experts | experts per regime | network parameters | memory MB | answer k-MACs | judging G-MACs | experts with a network | wall seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| shipped pool, 8-unit networks | 0.293 [0.250, 0.336] | 0.157 [0.133, 0.181] | 13.6 [9.4, 17.8] | 0.45 [0.31, 0.59] | 50701 [35140, 66261] | 9.04 [7.69, 10.39] | 544.6 [458.7, 630.5] | 0.0 [0.0, 0.0] | 13.6 [9.4, 17.8] | 120 [104, 135] |
| shipped pool, 32-unit networks | 0.479 [0.474, 0.484] | 0.418 [0.391, 0.445] | 30.0 [30.0, 30.0] | 1.00 [1.00, 1.00] | 296880 [296880, 296880] | 15.94 [15.81, 16.06] | 1115.1 [1115.1, 1115.1] | 0.0 [0.0, 0.0] | 30.0 [30.0, 30.0] | 81 [62, 100] |
| shipped pool (128-unit networks, the default) | 0.485 [0.481, 0.488] | 0.412 [0.394, 0.430] | 30.0 [30.0, 30.0] | 1.00 [1.00, 1.00] | 1037040 [1037040, 1037040] | 18.90 [18.77, 19.02] | 1139.7 [1139.7, 1139.7] | 0.0 [0.0, 0.0] | 30.0 [30.0, 30.0] | 82 [64, 100] |
| notebook-first | 0.482 [0.479, 0.486] | 0.422 [0.409, 0.436] | 30.0 [30.0, 30.0] | 1.00 [1.00, 1.00] | 11493 [-89, 23075] | 14.80 [14.65, 14.94] | 1106.3 [1105.9, 1106.6] | 93.0 [93.0, 93.0] | 1.2 [0.2, 2.2] | 147 [134, 160] |
| notebook-first, candidate sizes 8 and 32 | 0.479 [0.471, 0.486] | 0.396 [0.354, 0.438] | 30.0 [30.0, 30.0] | 1.00 [1.00, 1.00] | 20077 [7524, 32630] | 14.83 [14.67, 14.99] | 1106.5 [1106.2, 1106.9] | 31.4 [31.4, 31.4] | 2.4 [0.7, 4.1] | 96 [88, 103] |

## Growth-lab world through the real pool (16 regimes, 8 need a wide network)

| arm | accuracy | worst regime | experts | experts per regime | network parameters | memory MB | answer k-MACs | judging G-MACs | experts with a network | wall seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| shipped pool, 8-unit networks | 0.568 [0.484, 0.652] | 0.366 [0.322, 0.410] | 13.2 [10.1, 16.3] | 0.82 [0.63, 1.02] | 12961 [12022, 13901] | 3.08 [2.20, 3.96] | 112.7 [86.8, 138.7] | 0.0 [0.0, 0.0] | 13.2 [10.1, 16.3] | 478 [403, 553] |
| shipped pool, 32-unit networks | 0.600 [0.546, 0.654] | 0.379 [0.298, 0.460] | 14.6 [12.0, 17.2] | 0.91 [0.75, 1.07] | 36077 [29714, 42439] | 3.55 [2.78, 4.33] | 125.4 [104.7, 146.2] | 0.0 [0.0, 0.0] | 14.6 [12.0, 17.2] | 484 [422, 547] |
| shipped pool (128-unit networks, the default) | 0.607 [0.562, 0.653] | 0.408 [0.346, 0.470] | 14.2 [13.2, 15.2] | 0.89 [0.82, 0.95] | 133895 [117734, 150056] | 3.83 [3.33, 4.33] | 129.0 [120.3, 137.7] | 0.0 [0.0, 0.0] | 14.2 [13.2, 15.2] | 495 [430, 560] |
| notebook-first | 0.627 [0.597, 0.657] | 0.337 [0.295, 0.378] | 14.8 [13.8, 15.8] | 0.93 [0.86, 0.99] | 11983 [7201, 16765] | 3.81 [3.35, 4.27] | 149.1 [90.1, 208.0] | 11.3 [10.5, 12.1] | 5.0 [2.7, 7.3] | 521 [470, 572] |
| notebook-first, candidate sizes 8 and 32 | 0.634 [0.614, 0.653] | 0.359 [0.313, 0.405] | 15.0 [14.1, 15.9] | 0.94 [0.88, 0.99] | 5545 [1302, 9787] | 3.88 [3.34, 4.41] | 150.4 [92.6, 208.1] | 3.8 [3.6, 4.0] | 3.0 [0.5, 5.5] | 513 [467, 560] |

## FS-MoE teaching bed (3 domains, real text embeddings)

| arm | known skills, teaching wording | known skills, new wording | experts | network parameters | memory MB | judging G-MACs | experts with a network | wall seconds |
|---|---|---|---|---|---|---|---|---|
| shipped pool, 8-unit networks | 0.963 [0.957, 0.969] | 0.891 [0.884, 0.898] | 3.0 [3.0, 3.0] | 94023 [94023, 94023] | 12.13 [11.79, 12.48] | 0.0 [0.0, 0.0] | 3.0 [3.0, 3.0] | 23 [18, 27] |
| shipped pool, 32-unit networks | 0.963 [0.958, 0.967] | 0.896 [0.890, 0.902] | 3.0 [3.0, 3.0] | 130959 [130959, 130959] | 12.29 [11.94, 12.64] | 0.0 [0.0, 0.0] | 3.0 [3.0, 3.0] | 23 [19, 27] |
| shipped pool (128-unit networks, the default) | 0.965 [0.960, 0.970] | 0.895 [0.889, 0.900] | 3.0 [3.0, 3.0] | 278703 [278703, 278703] | 12.85 [12.48, 13.22] | 0.0 [0.0, 0.0] | 3.0 [3.0, 3.0] | 23 [19, 27] |
| notebook-first | 0.961 [0.956, 0.966] | 0.890 [0.885, 0.895] | 3.0 [3.0, 3.0] | 6271 [-11132, 23674] | 11.01 [10.58, 11.44] | 86.6 [77.4, 95.8] | 0.2 [-0.4, 0.8] | 31 [29, 34] |
| notebook-first, candidate sizes 8 and 32 | 0.962 [0.957, 0.967] | 0.890 [0.885, 0.895] | 3.0 [3.0, 3.0] | 15001 [-11066, 41068] | 11.06 [10.58, 11.53] | 36.9 [32.9, 40.8] | 0.4 [-0.3, 1.1] | 25 [24, 27] |

## Exploratory ablation (added after the first results, not pre-declared): which part of the port does the work?

The worse-than-chance rule alone on the shipped pool (`x` arms), and notebook experts only with no size test (`NBn`). Accuracy, experts per regime, network parameters and memory; the growth-world bed has the 8-unit arm only.

| bed | arm | accuracy | experts per regime | network parameters | memory MB |
|---|---|---|---|---|---|
| f10 | shipped pool, 8-unit networks | 0.355 [0.327, 0.383] | 0.64 [0.57, 0.71] | 23859 [21324, 26395] | 3.98 [3.54, 4.42] |
| f10 | ablation: shipped pool, 8-unit networks + the worse-than-chance rule | 0.480 [0.474, 0.486] | 1.02 [0.96, 1.08] | 38026 [35955, 40096] | 5.35 [4.49, 6.22] |
| f10 | shipped pool (128-unit networks, the default) | 0.493 [0.489, 0.498] | 1.00 [1.00, 1.00] | 345680 [345680, 345680] | 6.31 [6.19, 6.44] |
| f10 | ablation: shipped pool, 128-unit networks + the worse-than-chance rule | 0.491 [0.487, 0.496] | 1.02 [0.96, 1.08] | 352877 [332894, 372861] | 6.39 [6.12, 6.66] |
| f10 | notebook-first | 0.497 [0.488, 0.506] | 1.00 [1.00, 1.00] | 3634 [-6427, 13695] | 4.94 [4.80, 5.08] |
| f10 | ablation: notebook experts only (never judged) | 0.493 [0.486, 0.500] | 1.00 [1.00, 1.00] | 10 [10, 10] | 5.57 [5.45, 5.69] |
| f30 | shipped pool, 8-unit networks | 0.293 [0.250, 0.336] | 0.45 [0.31, 0.59] | 50701 [35140, 66261] | 9.04 [7.69, 10.39] |
| f30 | ablation: shipped pool, 8-unit networks + the worse-than-chance rule | 0.471 [0.466, 0.477] | 1.00 [1.00, 1.00] | 111840 [111840, 111840] | 15.20 [15.07, 15.32] |
| f30 | shipped pool (128-unit networks, the default) | 0.485 [0.481, 0.488] | 1.00 [1.00, 1.00] | 1037040 [1037040, 1037040] | 18.90 [18.77, 19.02] |
| f30 | ablation: shipped pool, 128-unit networks + the worse-than-chance rule | 0.478 [0.467, 0.488] | 1.02 [1.00, 1.04] | 1057781 [1034272, 1081290] | 18.97 [18.81, 19.13] |
| f30 | notebook-first | 0.482 [0.479, 0.486] | 1.00 [1.00, 1.00] | 11493 [-89, 23075] | 14.80 [14.65, 14.94] |
| f30 | ablation: notebook experts only (never judged) | 0.481 [0.474, 0.487] | 1.00 [1.00, 1.00] | 30 [30, 30] | 16.61 [16.49, 16.72] |
| g | shipped pool, 8-unit networks | 0.568 [0.484, 0.652] | 0.82 [0.63, 1.02] | 12961 [12022, 13901] | 3.08 [2.20, 3.96] |
| g | ablation: shipped pool, 8-unit networks + the worse-than-chance rule | 0.596 [0.567, 0.625] | 0.94 [0.79, 1.08] | 13665 [11549, 15781] | 3.63 [2.94, 4.31] |
| g | shipped pool (128-unit networks, the default) | 0.607 [0.562, 0.653] | 0.89 [0.82, 0.95] | 133895 [117734, 150056] | 3.83 [3.33, 4.33] |
| g | notebook-first | 0.627 [0.597, 0.657] | 0.93 [0.86, 0.99] | 11983 [7201, 16765] | 3.81 [3.35, 4.27] |
| g | ablation: notebook experts only (never judged) | 0.620 [0.593, 0.647] | 0.95 [0.78, 1.12] | 424 [127, 722] | 4.14 [3.46, 4.82] |
| t | shipped pool, 8-unit networks | 0.927 [0.921, 0.933] | 1.00 [1.00, 1.00] | 94023 [94023, 94023] | 12.13 [11.79, 12.48] |
| t | ablation: shipped pool, 8-unit networks + the worse-than-chance rule | 0.927 [0.921, 0.933] | 1.00 [1.00, 1.00] | 94023 [94023, 94023] | 12.13 [11.79, 12.48] |
| t | shipped pool (128-unit networks, the default) | 0.930 [0.925, 0.935] | 1.00 [1.00, 1.00] | 278703 [278703, 278703] | 12.85 [12.48, 13.22] |
| t | ablation: shipped pool, 128-unit networks + the worse-than-chance rule | 0.930 [0.925, 0.935] | 1.00 [1.00, 1.00] | 278703 [278703, 278703] | 12.85 [12.48, 13.22] |
| t | notebook-first | 0.925 [0.920, 0.930] | 1.00 [1.00, 1.00] | 6271 [-11132, 23674] | 11.01 [10.58, 11.44] |
| t | ablation: notebook experts only (never judged) | 0.926 [0.920, 0.931] | 1.00 [1.00, 1.00] | 3 [3, 3] | 10.99 [10.62, 11.36] |

## Networks the size test gave (notebook-first, all beds and seeds)

| units x blocks | experts |
|---|---|
| 8 x 1 | 12 |
| 8 x 3 | 3 |
| 32 x 1 | 10 |
| 64 x 1 | 8 |
