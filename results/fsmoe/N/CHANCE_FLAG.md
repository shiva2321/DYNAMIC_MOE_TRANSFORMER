# The worse-than-chance flag: standard checks and predictions W1-W6

6 declared predictions: **4 held, 2 missed**.

| id | prediction | outcome | measured |
|---|---|---|---|
| W1 | UniMoE suite, ULDA: weak-only mean on seq and drift within 0.003 of flag off, and at least 8 of 10 cells within 0.002 | MISSED | seq 0.6454 vs off 0.6454; drift 0.6236 vs 0.6423; 9 of 10 cells within 0.002 (unconditional rule: seq 0.6343, drift 0.6236) |
| W2 | UniMoE suite, 16-unit experts: weak-only mean on seq and drift within 0.004 of flag off | HELD | seq 0.5898 vs off 0.5898; drift 0.3784 vs 0.3823 |
| W3 | f30: the shipped pool with 8-unit networks and the weak-only rule keeps at least 0.9 experts per regime and scores at least 0.44 | HELD | 1.00 experts per regime, accuracy 0.471 (off 0.293, unconditional 0.471) |
| W4 | f10: the shipped pool with 8-unit networks and the weak-only rule keeps at least 0.9 experts per regime and scores at least 0.44 | HELD | 1.02 experts per regime, accuracy 0.480 (off 0.355, unconditional 0.480) |
| W5 | f10 and f30: the shipped pool with 128-unit networks and the weak-only rule is within 0.005 of flag off | MISSED | f10: 0.4913 vs 0.4931 (unconditional 0.4913), f30: 0.4777 vs 0.4845 (unconditional 0.4777) |
| W6 | banking77: weak-only with 128-unit experts within 0.01 of flag off, with 16-unit experts at least off minus 0.01 | HELD | 128 units 0.8464 vs 0.8464 (unconditional 0.8464); 16 units 0.8193 vs 0.8193 (unconditional 0.8193) |

## UniMoE real suite (seven text domains; seeds 500-504; mean final accuracy, experts, spawns, extends)

| schedule | arm | accuracy | experts | spawns | extends | max gap to the stored E-054 cell (flag off) |
|---|---|---|---|---|---|---|
| seq | ULDA-off | 0.6454 | 19.60 | 19.60 | 2.60 | 9.09e-04 |
| seq | ULDA-on | 0.6343 | 20.80 | 20.80 | 1.60 | - |
| seq | ULDA-onw | 0.6454 | 19.60 | 19.60 | 2.60 | - |
| seq | h16-off | 0.5898 | 20.00 | 20.00 | 1.80 | - |
| seq | h16-on | 0.5931 | 20.40 | 20.40 | 1.80 | - |
| seq | h16-onw | 0.5898 | 20.00 | 20.00 | 1.80 | - |
| mix | ULDA-off | 0.6631 | 1.00 | 1.00 | 0.00 | 0.00e+00 |
| mix | ULDA-on | 0.6631 | 1.00 | 1.00 | 0.00 | - |
| mix | ULDA-onw | 0.6631 | 1.00 | 1.00 | 0.00 | - |
| mix | h16-off | 0.4420 | 1.00 | 1.00 | 0.00 | - |
| mix | h16-on | 0.4420 | 1.00 | 1.00 | 0.00 | - |
| mix | h16-onw | 0.4420 | 1.00 | 1.00 | 0.00 | - |
| drift | ULDA-off | 0.6423 | 1.20 | 1.20 | 0.60 | 0.00e+00 |
| drift | ULDA-on | 0.6236 | 1.40 | 1.40 | 0.40 | - |
| drift | ULDA-onw | 0.6236 | 1.40 | 1.40 | 0.40 | - |
| drift | h16-off | 0.3823 | 1.20 | 1.20 | 0.60 | - |
| drift | h16-on | 0.3784 | 1.40 | 1.40 | 0.40 | - |
| drift | h16-onw | 0.3784 | 1.40 | 1.40 | 0.40 | - |

## Function regimes (seeds 710-714; accuracy, experts per regime)

| bed | arm | accuracy | experts per regime |
|---|---|---|---|
| f10 | U8 | 0.3551 | 0.64 |
| f10 | U8x | 0.4801 | 1.02 |
| f10 | U8w | 0.4801 | 1.02 |
| f10 | U128 | 0.4931 | 1.00 |
| f10 | U128x | 0.4913 | 1.02 |
| f10 | U128w | 0.4913 | 1.02 |
| f30 | U8 | 0.2930 | 0.45 |
| f30 | U8x | 0.4714 | 1.00 |
| f30 | U8w | 0.4714 | 1.00 |
| f30 | U128 | 0.4845 | 1.00 |
| f30 | U128x | 0.4777 | 1.02 |
| f30 | U128w | 0.4777 | 1.02 |

## banking77 class-incremental (seeds 710-712)

| arm | accuracy | experts | spawns | extends |
|---|---|---|---|---|
| off128 | 0.8464 | 6.0 | 6.0 | 0.0 |
| on128 | 0.8464 | 6.0 | 6.0 | 0.0 |
| off16 | 0.8193 | 6.0 | 6.0 | 0.0 |
| on16 | 0.8193 | 6.0 | 6.0 | 0.0 |
| onw128 | 0.8464 | 6.0 | 6.0 | 0.0 |
| onw16 | 0.8193 | 6.0 | 6.0 | 0.0 |
