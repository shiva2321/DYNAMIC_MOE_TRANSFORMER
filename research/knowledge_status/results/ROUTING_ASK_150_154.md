# Lab A, the asking reader (pool dev, seeds 150-154, q = 0.05)

The language model is asked which operation a message wants; known and unseen are decided on its answer probabilities (calibration on S1-S3 held-out parts only). The look-based machine on the same seeds (tuning seeds 150-154, results/FUNCTION_LAB.md): known skills in new wordings called unseen 0.256, never-taught skills found 0.903.

| measure | conformal (calibrated on S1-S3) | plain naming rule |
|---|---|---|
| known skills in new wordings S4-S8 called unseen (bar 0.05) | 0.145 | 0.052 |
| never-taught skills found | 0.826 | 0.989 |
| known skills in the taught wordings S1-S3 called unseen | 0.047 | |
| macro F1 (Solid / Thin / Unseen) | 0.891 | |
| Thin against Solid, AUROC | 1.000 | |
| everyday sentences called unseen | 0.003 | |
| look-alike traps called unseen | 0.603 | |

Operation named correctly (top option = the true operation), by wording: S1 0.960, S2 1.000, S3 1.000, S4 0.918, S5 1.000, S6 0.986, S7 0.875, S8 0.983
By skill: add 0.997, sub 0.989, mul 0.986, div 0.994, mod 0.871, max 0.949
Known skills called unseen, by new wording: S4 0.287, S5 0.122, S6 0.048, S7 0.260, S8 0.048

Covering set, known skills, taught wordings S1-S3: true skill in the set 0.940, empty 0.047, one expert 0.749, two or more 0.204, mean size when not empty 1.21
Covering set, known skills, new wordings S4-S8: true skill in the set 0.785, empty 0.145, one expert 0.642, two or more 0.213, mean size when not empty 1.24
Covering set, never-taught skills: true skill in the set nan, empty 0.826, one expert 0.174, two or more 0.000, mean size when not empty 1.00

Real problems (SVAMP, 300 one-step problems, add / sub / mul / div; reported once, never used to choose anything):
- operation named correctly 0.770
- problems of a taught skill called unseen 0.515 (plain rule 0.167); the look-based machine called 0.98 of them unseen
- problems of a never-taught skill called unseen 0.629 (plain rule 0.907)
