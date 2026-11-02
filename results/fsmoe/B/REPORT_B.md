# Domain-balanced training report (seeds [710, 711, 712, 713, 714])

The E-057 A3 setting (pool + global head, tool on ALL code inputs, human labels on 10 % of the other inputs); only how the verified items enter the pool changes. A3b reproduced E-057's A3 in 5 of 5 seeds. Never-taught skills: 3 arithmetic, 5 code, 20 language.

## Never-taught skills learned

| arm | learned all 28 | arithmetic (3) | code (5) | language (20) | novel accuracy (all) | chunks | items trained: arithmetic | code | language |
|---|---|---|---|---|---|---|---|---|---|
| A3b arrival order (the E-057 A3) | 0.32 [0.28, 0.37] | 0.20 [-0.03, 0.43] | 1.00 [1.00, 1.00] | 0.17 [0.09, 0.25] | 0.340 [0.289, 0.390] | n/a | n/a (all arrivals) | n/a (all arrivals) | n/a (all arrivals) |
| D1 sub: equal shares, surplus unused | 0.35 [0.21, 0.49] | 0.67 [0.37, 0.96] | 0.80 [0.55, 1.05] | 0.19 [0.09, 0.29] | 0.350 [0.257, 0.444] | 17 [15, 19] | 100 [92, 108] | 85 [77, 93] | 85 [77, 93] |
| D2 over: equal shares, scarce domains replayed | 0.35 [0.29, 0.41] | 0.40 [0.05, 0.75] | 0.92 [0.78, 1.06] | 0.20 [0.12, 0.28] | 0.336 [0.284, 0.389] | 468 [466, 469] | 100 [92, 108] | 2338 [2330, 2345] | 128 [118, 138] |

## Known skills after the stream (taught skills in held-out wordings; before the stream in the last row)

| arm | all | arithmetic | code | language | taught wordings, all |
|---|---|---|---|---|---|
| A3b arrival order (the E-057 A3) | 0.883 [0.873, 0.892] | 0.914 [0.890, 0.938] | 0.959 [0.950, 0.968] | 0.798 [0.781, 0.814] | 0.969 [0.962, 0.976] |
| D1 sub: equal shares, surplus unused | 0.849 [0.816, 0.881] | 0.906 [0.849, 0.963] | 0.906 [0.867, 0.946] | 0.772 [0.742, 0.801] | 0.968 [0.960, 0.976] |
| D2 over: equal shares, scarce domains replayed | 0.868 [0.857, 0.878] | 0.925 [0.894, 0.956] | 0.957 [0.954, 0.960] | 0.761 [0.741, 0.780] | 0.965 [0.958, 0.971] |
| before the stream | 0.885 [0.878, 0.893] | 0.935 [0.923, 0.947] | 0.960 [0.956, 0.965] | 0.795 [0.777, 0.812] | 0.969 [0.962, 0.975] |

Domain head: 0.985 [0.973, 0.998] of the human-labelled items placed in the right domain (arithmetic against language; code items come from the tool).

## The class priors of the global head (addendum of the design)

The global closed-form head answers with its class counts removed (uniform priors); `A3u` keeps the arrival-order chunks of A3, `D2u` uses the `over` chunks of D2. After teaching only, the head with uniform priors reads taught skills 0.959 [0.952, 0.966] and held-out wordings 0.895 [0.890, 0.899] (with its priors: 0.959 [0.952, 0.966] and 0.895 [0.890, 0.900]).

| arm | learned all 28 | arithmetic (3) | code (5) | language (20) | known ho: all | arithmetic | code | language | taught wordings, all |
|---|---|---|---|---|---|---|---|---|---|
| A3u arrival order, uniform priors | 0.34 [0.27, 0.42] | 0.33 [0.04, 0.63] | 1.00 [1.00, 1.00] | 0.18 [0.09, 0.27] | 0.881 [0.871, 0.891] | 0.908 [0.883, 0.933] | 0.958 [0.950, 0.967] | 0.797 [0.781, 0.813] | 0.969 [0.962, 0.975] |
| D2u over, uniform priors | 0.35 [0.29, 0.41] | 0.40 [0.05, 0.75] | 0.92 [0.78, 1.06] | 0.20 [0.12, 0.28] | 0.867 [0.856, 0.878] | 0.927 [0.896, 0.958] | 0.955 [0.952, 0.958] | 0.760 [0.740, 0.781] | 0.965 [0.959, 0.970] |
