# Hybrid scorecard (seeds [710, 711, 712, 713, 714])

8 declared predictions: **4 held, 4 missed**.

| id | prediction | outcome | measured |
|---|---|---|---|
| H1 | at least one hybrid arm has arithmetic learned >= 0.50 AND code learned >= 0.90 | MISSED | H4: arithmetic 0.333, code 1.000; H8: arithmetic 0.467, code 1.000; H12: arithmetic 0.533, code 0.800 |
| H2 | the grid is monotone: code learned H12 >= H8 >= H4 and arithmetic learned H12 <= H4 | MISSED | code 1.000 / 1.000 / 0.800; arithmetic 0.333 / 0.467 / 0.533 |
| H3 | D1r reproduces D1's learned fraction and known held-out accuracy to 1e-9 in 5 of 5 seeds | HELD | 5 of 5 seeds |
| H4 | learned fraction of all 28 between 0.33 and 0.38 in every hybrid arm | HELD | H4 0.343, H8 0.357, H12 0.336 |
| H5 | language learned <= 0.30 in every hybrid arm | HELD | H4 0.180, H8 0.180, H12 0.190 |
| H6 | code known skills in held-out wordings between 0.90 and 0.94 in every hybrid arm | MISSED | H4 0.943, H8 0.928, H12 0.910 |
| H7 | known skills in held-out wordings, all domains, >= 0.86 in every hybrid arm | MISSED | H4 0.872, H8 0.864, H12 0.850 |
| H8 | language known skills in held-out wordings <= 0.79 in every hybrid arm | HELD | H4 0.788, H8 0.784, H12 0.777 |
