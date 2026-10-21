# Independent audit of the RV2-D verdicts (confirmatory records)

Records read: 704 of phase confirm with the code stamp ff3438232c5f0dfa; integrity problems: 0.


| ID | recomputed value | bar | recomputed verdict |
|---|---|---|---|
| S4-1 | +12.95 [+10.53, +15.38] n=6 | lower bound >= +5.0 | PASS |
| S4-2 | +10.11 [+9.61, +10.60] n=15 | lower bound >= +4.0 | PASS |
| S4-3 | worst guard mean +0.00 over 10/10 units | every mean >= -0.5 | PASS |
| S4-4 | gain +6.14 (n=6), forgetting 7.39, seeds with < 5 experts 0 | gain >= +5, forgetting <= 9, at most 1 seed with < 5 experts | PASS |
| S4-5 | worst accuracy mean -0.43, largest expert ratio 1.07, 11/11 units | every mean >= -0.5 and experts <= 1.2 x true functions (A4-1) | PASS |
| S4-6 | 15/15 seeds reach 0.9 (need 13), width 124, extra forward 3.33 % | >= 13 of 15 seeds reach 0.9, width in [64, 256], extra forward <= 5 % | PASS |
| S4-7 | worst mean +0.00, most growth events per run 0.00, largest parameter increase +0.0 %, 11/11 units | every mean >= -0.5, growth events per run <= 1, parameters <= +25 % | PASS |
| S4-8 | exact match 46.59 (n=6), forgetting 7.08 | exact match >= 44, forgetting <= 8 | PASS |

## Against the reported verdicts (`verdicts_rv2d.json`)

| ID | audit | report | agree |
|---|---|---|---|
| S4-1 | PASS | PASS | yes |
| S4-2 | PASS | PASS | yes |
| S4-3 | PASS | PASS | yes |
| S4-4 | PASS | PASS | yes |
| S4-5 | PASS | PASS | yes |
| S4-6 | PASS | PASS | yes |
| S4-7 | PASS | PASS | yes |
| S4-8 | PASS | PASS | yes |
| S4-9 | (not recomputed) | PASS | |
