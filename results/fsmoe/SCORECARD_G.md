# Growth scorecard (seeds [710, 711, 712, 713, 714])

14 declared predictions: **8 held, 6 missed**.

| id | prediction | outcome | measured |
|---|---|---|---|
| P1 | G beats F8 and F16 by at least 0.02 each and is at most 0.01 below the best fixed arm | HELD | G 0.730, F8 0.669, F16 0.690, best fixed 0.726 |
| P2 | G is at least as accurate as the best fixed arm of equal memory (at most 1.2 x G's parameters) | HELD | G 0.730 with 56425 parameters; equal-memory arms ['F8', 'F16', 'F32', 'F8d3'] best 0.711 |
| P3 | BF has at least 5 times G's parameters and an accuracy at most 0.01 above G's | HELD | BF 562896 parameters (10.0 x), accuracy 0.732 against G 0.730 |
| P4 | at least 80 % of width-need regimes are served by an expert with at least 16 units in G | HELD | 0.85 of 40 |
| P5 | at least 40 % of depth-need regimes are served by an expert with at least 2 blocks in G | MISSED | 0.11 of 9 |
| P6 | at most 20 % of none-need regimes are served by an expert with at least 16 units or 2 blocks in G | MISSED | 0.32 of 31 |
| P7 | easy: G has at most 1 growth event per seed on average and an accuracy at least 0.01 above F8's with fewer parameters | HELD | 0.0 growth events per seed; G 0.902 (13596 parameters), F8 0.861 (17064) |
| P8 | growth adds at most 0.02 to the notebook-first arm without it (G - Gng <= 0.02) | HELD | G 0.730, Gng 0.730, difference +0.000 |
| P9 | G is at least as accurate as GnoNB minus 0.005 with no more parameters | MISSED | G 0.730 (56425), GnoNB 0.672 (38099) |
| P10 | growth trials and promotion tests in G cost at most 12 times the training multiply-adds | HELD | 4.5 times |
| P11 | big: G is at least 0.02 above F8 with no more multiply-adds per answer | MISSED | G 0.756 (2325 MACs), F8 0.742 (2606) |
| P12 | big: the tier is refused (never built) in at least 4 of 5 seeds | MISSED | never built in 0 of 5 seeds; built events per seed [4, 3, 3, 4, 4] |
| P13 | G ends with 0.95-1.10 experts per regime and purity at least 0.95 (mixed and big) | MISSED | mixed: 1.00 experts per regime, purity 0.986; big: 1.00 experts per regime, purity 0.924 |
| P14 | G's worst-regime accuracy is at least F8's (mixed) | HELD | G 0.580, F8 0.506 |
