# End-to-end report (seeds [710, 711, 712, 713, 714])

One stream over arithmetic, code and language (the stream of E-055: known skills in all wordings, never-taught skills, far texts). A0 pool alone, human labels on 10 % of all inputs; A1 pool + global closed-form head, same labels; A2 as A1 with the sandbox tool on 10 % of the code inputs and human labels on 10 % of the others; A3 tool on ALL code inputs; A4 as A3 with human labels on 20 % of the others. Never-taught skills: 3 arithmetic, 5 code, 20 language. A skill is learned when accuracy on rows the stream never contained is at least 0.5; for the code skills the tool opened, an id counts as that skill's class by the majority truth of its items (evaluation only).

Stream: 5890 items, 2395 of them code-looking (49 far code snippets from the standard library). After teaching only (class slots 121): pool alone 0.965 [0.960, 0.970] taught / 0.895 [0.889, 0.900] held-out wordings; pool + global head 0.959 [0.952, 0.966] / 0.895 [0.890, 0.900].

## Never-taught skills learned, and the cost

| arm | human labels used | tool calls | classes the tool opened (5 true new) | purity of those classes | learned (all 28) | arithmetic (3) | code (5) | language (20) | novel accuracy (all) |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 386 [380, 393] | 0 [0, 0] | 0.0 [0.0, 0.0] | n/a | 0.21 [-0.03, 0.44] | 0.40 [-0.28, 1.08] | 0.52 [0.04, 1.00] | 0.10 [-0.05, 0.25] | 0.238 [0.045, 0.431] |
| A1 | 386 [380, 393] | 0 [0, 0] | 0.0 [0.0, 0.0] | n/a | 0.36 [0.23, 0.50] | 0.67 [0.25, 1.08] | 0.76 [0.44, 1.08] | 0.22 [0.08, 0.36] | 0.380 [0.307, 0.452] |
| A2 | 228 [225, 232] | 237 [232, 242] | 5.0 [5.0, 5.0] | 1.00 [1.00, 1.00] | 0.34 [0.22, 0.47] | 0.53 [0.31, 0.76] | 0.84 [0.63, 1.05] | 0.19 [0.09, 0.29] | 0.339 [0.268, 0.410] |
| A3 | 228 [225, 232] | 2382 [2371, 2394] | 5.4 [4.7, 6.1] | 1.00 [1.00, 1.00] | 0.32 [0.28, 0.37] | 0.20 [-0.03, 0.43] | 1.00 [1.00, 1.00] | 0.17 [0.09, 0.25] | 0.340 [0.289, 0.390] |
| A4 | 521 [512, 531] | 2382 [2371, 2394] | 5.4 [4.7, 6.1] | 1.00 [1.00, 1.00] | 0.33 [0.29, 0.37] | 0.27 [0.08, 0.45] | 1.00 [1.00, 1.00] | 0.17 [0.14, 0.20] | 0.404 [0.365, 0.442] |

## Known skills after the stream (taught skills in held-out wordings; before the stream in the last row)

| arm | all | arithmetic | code | language | taught wordings, all |
|---|---|---|---|---|---|
| A0 | 0.713 [0.602, 0.824] | 0.865 [0.765, 0.965] | 0.831 [0.677, 0.986] | 0.543 [0.334, 0.752] | 0.951 [0.935, 0.968] |
| A1 | 0.858 [0.844, 0.872] | 0.913 [0.877, 0.949] | 0.933 [0.912, 0.953] | 0.766 [0.730, 0.803] | 0.966 [0.961, 0.971] |
| A2 | 0.869 [0.850, 0.887] | 0.899 [0.851, 0.947] | 0.944 [0.934, 0.953] | 0.785 [0.761, 0.809] | 0.968 [0.962, 0.974] |
| A3 | 0.883 [0.873, 0.892] | 0.914 [0.890, 0.938] | 0.959 [0.950, 0.968] | 0.798 [0.781, 0.814] | 0.969 [0.962, 0.976] |
| A4 | 0.885 [0.873, 0.897] | 0.916 [0.883, 0.949] | 0.961 [0.958, 0.964] | 0.800 [0.780, 0.821] | 0.968 [0.962, 0.974] |
| before the stream | 0.885 [0.874, 0.895] | 0.950 [0.935, 0.965] | 0.936 [0.932, 0.939] | 0.811 [0.786, 0.837] | 0.972 [0.966, 0.978] |

## The tool

| arm | known-family labels | new-behaviour labels | no outcome | slot overflow | non-code texts run | far code snippets run in the sandbox (of the stream's) | classes opened beyond the 5 true |
|---|---|---|---|---|---|---|---|
| A2 | 108 [103, 113] | 96 [89, 103] | 33 [29, 37] | 0.0 [0.0, 0.0] | 0 [0, 0] | 13.2 [9.5, 16.9] (45 [39, 52]) | 0.0 [0.0, 0.0] |
| A3 | 1980 [1980, 1980] | 358 [350, 365] | 45 [39, 51] | 0.0 [0.0, 0.0] | 0 [0, 0] | 17.8 [13.0, 22.6] (45 [39, 52]) | 0.4 [-0.3, 1.1] |
| A4 | 1980 [1980, 1980] | 358 [350, 365] | 45 [39, 51] | 0.0 [0.0, 0.0] | 0 [0, 0] | 17.8 [13.0, 22.6] (45 [39, 52]) | 0.4 [-0.3, 1.1] |

## Diagnostic arms (added after the first five arms; see the addendum of the design)

A1r and A3r re-run A1 and A3 with the missing counters; they reproduced the first runs in 10 of 10 cells. A5 = A3 with the tool's KNOWN items trained on at a rate of 10 % (the new-behaviour items always).

| arm | human labels (arith / code / lang) | tool-labelled known items trained | learned all | arithmetic | code | language | known ho all | arithmetic | code | language |
|---|---|---|---|---|---|---|---|---|---|---|
| A1r | 125 [122, 129] / 107 [101, 113] / 154 [150, 158] | 0 [0, 0] | 0.36 [0.23, 0.50] | 0.67 [0.25, 1.08] | 0.76 [0.44, 1.08] | 0.22 [0.08, 0.36] | 0.858 [0.844, 0.872] | 0.913 [0.877, 0.949] | 0.933 [0.912, 0.953] | 0.766 [0.730, 0.803] |
| A3r | 104 [95, 113] / 0 [0, 0] / 125 [115, 135] | 1980 [1980, 1980] | 0.32 [0.28, 0.37] | 0.20 [-0.03, 0.43] | 1.00 [1.00, 1.00] | 0.17 [0.09, 0.25] | 0.883 [0.873, 0.892] | 0.914 [0.890, 0.938] | 0.959 [0.950, 0.968] | 0.798 [0.781, 0.814] |
| A5 | 104 [95, 113] / 0 [0, 0] / 125 [115, 135] | 190 [173, 207] | 0.31 [0.22, 0.41] | 0.40 [0.05, 0.75] | 0.84 [0.63, 1.05] | 0.17 [0.09, 0.25] | 0.869 [0.850, 0.888] | 0.910 [0.866, 0.955] | 0.933 [0.924, 0.941] | 0.792 [0.766, 0.818] |
