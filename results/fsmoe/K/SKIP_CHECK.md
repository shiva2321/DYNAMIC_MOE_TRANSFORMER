# The skip flag: standard checks and predictions S1-S7

7 declared predictions: **6 held, 1 missed**.

| id | prediction | outcome | measured |
|---|---|---|---|
| S1 | the full suite passes (at least 1,101 tests) and the 15 flag-off UniMoE cells equal the stored E-062 cells to 1e-9 | HELD | suite: 1102 passed, 2 xfailed, 6 warnings in 434.13s (0:07:14); 15 cells compared, largest gap 0.00e+00 |
| S2 | f10 and f30: at both thresholds under 2 % of batches skipped and accuracy within 0.002 of off | HELD | f10 k0.1: 0.0% of batches, accuracy +0.0000; f10 k0.25: 0.0% of batches, accuracy +0.0000; f30 k0.1: 0.0% of batches, accuracy +0.0000; f30 k0.25: 0.0% of batches, accuracy +0.0000 |
| S3 | text bed: 0.1 skips at least 40 % of rows within 0.003 of off; 0.25 skips at least 70 % within 0.006 | HELD | 0.1: 54.3% of rows, accuracy -0.0008; 0.25: 77.5%, -0.0009 |
| S4 | banking77: 0.1 skips under 10 % of rows within 0.005 of off; 0.25 skips 10-30 % within 0.01 | HELD | 0.1: 9.0%, -0.0036; 0.25: 20.7%, +0.0008 |
| S5 | UniMoE suite: share of rows skipped inside the declared range and accuracy within the declared tolerance, six cells | HELD | seq k0.1: 7.1% of rows (declared 5%-15%), accuracy +0.0000 (within 0.004); seq k0.25: 14.3% of rows (declared 12%-35%), accuracy -0.0006 (within 0.006); mix k0.1: 0.0% of rows (declared 0%-2%), accuracy +0.0000 (within 0.002); mix k0.25: 5.8% of rows (declared 0%-10%), accuracy +0.0005 (within 0.004); drift k0.1: 0.0% of rows (declared 0%-2%), accuracy +0.0000 (within 0.002); drift k0.25: 5.2% of rows (declared 0%-10%), accuracy +0.0001 (within 0.004) |
| S6 | train_expert takes at most 35 % of observe time with the flag off, and no arm saves more than 25 % of observe time | MISSED | f10: train share 8.1%, saved k0.1 +4.3%, k0.25 +4.4%; f30: train share 7.1%, saved k0.1 +2.4%, k0.25 +3.0%; t: train share 12.0%, saved k0.1 +10.5%, k0.25 +12.4%; bank: train share 22.1%, saved k0.1 +9.1%, k0.25 +10.4%; useq: train share 23.6%, saved k0.1 +4.8%, k0.25 +7.1%; umix: train share 24.8%, saved k0.1 +25.8%, k0.25 +27.5%; udrift: train share 19.8%, saved k0.1 +21.3%, k0.25 +21.7% |
| S7 | the number of experts is within 1 of off at 0.1 and within 2 at 0.25 in every cell | HELD | largest difference: 0.1 1, 0.25 1 |

## Per bed (mean over seeds; accuracy, experts, batches and rows skipped, wall seconds inside train_expert / observe)

| bed | arm | seeds | accuracy | experts | batches skipped | rows skipped | train s | observe s |
|---|---|---|---|---|---|---|---|---|
| f10 | off | 5 | 0.4931 | 10.0 | 0.0% | 0.0% | 0.9 | 11.7 |
| f10 | k0.1 | 5 | 0.4931 | 10.0 | 0.0% | 0.0% | 1.0 | 11.2 |
| f10 | k0.25 | 5 | 0.4931 | 10.0 | 0.0% | 0.0% | 0.9 | 11.2 |
| f30 | off | 5 | 0.4845 | 30.0 | 0.0% | 0.0% | 2.4 | 33.5 |
| f30 | k0.1 | 5 | 0.4845 | 30.0 | 0.0% | 0.0% | 2.4 | 32.7 |
| f30 | k0.25 | 5 | 0.4845 | 30.0 | 0.0% | 0.0% | 2.4 | 32.5 |
| t | off | 5 | 0.9298 | 3.0 | 0.0% | 0.0% | 1.1 | 9.7 |
| t | k0.1 | 5 | 0.9290 | 3.0 | 54.8% | 54.3% | 0.5 | 8.7 |
| t | k0.25 | 5 | 0.9289 | 3.0 | 77.6% | 77.5% | 0.3 | 8.5 |
| bank | off | 3 | 0.8464 | 6.0 | 0.0% | 0.0% | 1.0 | 4.7 |
| bank | k0.1 | 3 | 0.8429 | 6.0 | 16.0% | 9.0% | 0.9 | 4.3 |
| bank | k0.25 | 3 | 0.8473 | 6.0 | 29.2% | 20.7% | 0.8 | 4.2 |
| useq | off | 5 | 0.6454 | 19.6 | 0.0% | 0.0% | 5.0 | 22.2 |
| useq | k0.1 | 5 | 0.6454 | 19.8 | 8.3% | 7.1% | 4.6 | 21.1 |
| useq | k0.25 | 5 | 0.6447 | 19.8 | 14.9% | 14.3% | 4.3 | 20.6 |
| umix | off | 5 | 0.6631 | 1.0 | 0.0% | 0.0% | 0.4 | 2.0 |
| umix | k0.1 | 5 | 0.6631 | 1.0 | 0.0% | 0.0% | 0.4 | 1.5 |
| umix | k0.25 | 5 | 0.6636 | 1.0 | 5.9% | 5.8% | 0.4 | 1.4 |
| udrift | off | 5 | 0.6423 | 1.2 | 0.0% | 0.0% | 0.4 | 2.4 |
| udrift | k0.1 | 5 | 0.6423 | 1.2 | 0.0% | 0.0% | 0.4 | 1.9 |
| udrift | k0.25 | 5 | 0.6424 | 1.2 | 5.9% | 5.2% | 0.4 | 1.9 |
