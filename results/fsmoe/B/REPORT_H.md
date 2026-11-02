# Hybrid report (seeds [710, 711, 712, 713, 714])

The E-057 A3 setting (pool + global head, tool on all code inputs, 228 human labels); only how verified items enter the pool changes. Arms A3b and D2 are the E-058 results of the same seeds; D1r re-runs D1 with counters for new-class items and reproduced it in 5 of 5 seeds. Never-taught skills: 3 arithmetic, 5 code, 20 language.

## Never-taught skills learned and what was trained

| arm | learned all 28 | arithmetic (3) | code (5) | language (20) | chunks | fresh items trained: arithmetic | code | language | of which new-class: arithmetic | code | language |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A3b arrival order (E-058) | 0.32 [0.28, 0.37] | 0.20 [-0.03, 0.43] | 1.00 [1.00, 1.00] | 0.17 [0.09, 0.25] | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| D2 over: scarce domains replayed (E-058) | 0.35 [0.29, 0.41] | 0.40 [0.05, 0.75] | 0.92 [0.78, 1.06] | 0.20 [0.12, 0.28] | 468 [466, 469] | 100 [92, 108] | 2338 [2330, 2345] | 128 [118, 138] | n/a | n/a | n/a |
| D1 sub, re-run with the new counters | 0.35 [0.21, 0.49] | 0.67 [0.37, 0.96] | 0.80 [0.55, 1.05] | 0.19 [0.09, 0.29] | 17 [15, 19] | 100 [92, 108] | 85 [77, 93] | 85 [77, 93] | 39 [33, 45] | 85 [77, 93] | 30 [24, 36] |
| H4 hybrid, 4 new-class items per chunk | 0.34 [0.27, 0.42] | 0.33 [0.04, 0.63] | 1.00 [1.00, 1.00] | 0.18 [0.09, 0.27] | 107 [105, 109] | 100 [92, 108] | 786 [772, 799] | 128 [118, 138] | 39 [33, 45] | 358 [350, 365] | 30 [24, 36] |
| H8 hybrid, 8 new-class items per chunk | 0.36 [0.29, 0.43] | 0.47 [0.24, 0.69] | 1.00 [1.00, 1.00] | 0.18 [0.09, 0.27] | 55 [54, 56] | 100 [92, 108] | 523 [512, 533] | 125 [116, 135] | 39 [33, 45] | 358 [350, 365] | 30 [24, 36] |
| H12 hybrid, 12 new-class items per chunk | 0.34 [0.21, 0.46] | 0.53 [0.31, 0.76] | 0.80 [0.37, 1.23] | 0.19 [0.11, 0.27] | 46 [43, 49] | 100 [92, 108] | 404 [395, 414] | 76 [72, 81] | 39 [33, 45] | 358 [350, 365] | 30 [24, 36] |

## Known skills after the stream (taught skills in held-out wordings; before the stream in the last row)

| arm | all | arithmetic | code | language | taught wordings, all |
|---|---|---|---|---|---|
| A3b arrival order (E-058) | 0.883 [0.873, 0.892] | 0.914 [0.890, 0.938] | 0.959 [0.950, 0.968] | 0.798 [0.781, 0.814] | 0.969 [0.962, 0.976] |
| D2 over: scarce domains replayed (E-058) | 0.868 [0.857, 0.878] | 0.925 [0.894, 0.956] | 0.957 [0.954, 0.960] | 0.761 [0.741, 0.780] | 0.965 [0.958, 0.971] |
| D1 sub, re-run with the new counters | 0.849 [0.816, 0.881] | 0.906 [0.849, 0.963] | 0.906 [0.867, 0.946] | 0.772 [0.742, 0.801] | 0.968 [0.960, 0.976] |
| H4 hybrid, 4 new-class items per chunk | 0.872 [0.860, 0.883] | 0.912 [0.884, 0.939] | 0.943 [0.936, 0.950] | 0.788 [0.769, 0.807] | 0.968 [0.961, 0.974] |
| H8 hybrid, 8 new-class items per chunk | 0.864 [0.847, 0.882] | 0.918 [0.871, 0.966] | 0.928 [0.918, 0.938] | 0.784 [0.752, 0.815] | 0.968 [0.962, 0.974] |
| H12 hybrid, 12 new-class items per chunk | 0.850 [0.822, 0.878] | 0.893 [0.828, 0.958] | 0.910 [0.895, 0.925] | 0.777 [0.746, 0.808] | 0.967 [0.960, 0.973] |
| before the stream | 0.885 [0.878, 0.893] | 0.935 [0.923, 0.947] | 0.960 [0.956, 0.965] | 0.795 [0.777, 0.812] | 0.969 [0.962, 0.975] |

Human labels in every arm: [224, 228, 229, 230, 231] (one value per seed in the first row of each seed).
