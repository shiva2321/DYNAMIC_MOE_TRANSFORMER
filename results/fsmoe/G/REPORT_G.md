# Growth report (seeds [710, 711, 712, 713, 714])

Mean [95 % interval over seeds]. Parameters = network + notebook + centroid scalars. Multiply-adds per answer = gate + expert + notebook, in thousands; training and governing multiply-adds in billions.

## Mixed needs (16 regimes: 4 linear, 8 wide, 4 deep)

| arm | mean accuracy | worst regime | total parameters | of which networks | answer k-MACs | training G-MACs | governing G-MACs | experts |
|---|---|---|---|---|---|---|---|---|
| fixed, 8 units | 0.669 [0.653, 0.684] | 0.506 [0.466, 0.545] | 34128 [34128, 34128] | 14048 [14048, 14048] | 1.58 [1.58, 1.58] | 4.0 [3.8, 4.3] | 0.0 [0.0, 0.0] | 16.0 [16.0, 16.0] |
| fixed, 16 units | 0.690 [0.676, 0.703] | 0.537 [0.498, 0.576] | 42448 [42448, 42448] | 22368 [22368, 22368] | 2.10 [2.10, 2.10] | 6.4 [6.1, 6.8] | 0.0 [0.0, 0.0] | 16.0 [16.0, 16.0] |
| fixed, 32 units | 0.711 [0.702, 0.721] | 0.572 [0.557, 0.587] | 59088 [59088, 59088] | 39008 [39008, 39008] | 3.14 [3.14, 3.14] | 11.2 [10.6, 11.8] | 0.0 [0.0, 0.0] | 16.0 [16.0, 16.0] |
| fixed, 64 units | 0.726 [0.715, 0.737] | 0.595 [0.570, 0.620] | 92368 [92368, 92368] | 72288 [72288, 72288] | 5.22 [5.22, 5.22] | 20.8 [19.6, 21.9] | 0.0 [0.0, 0.0] | 16.0 [16.0, 16.0] |
| fixed, 8 units x 3 blocks | 0.673 [0.660, 0.686] | 0.523 [0.485, 0.561] | 53840 [53840, 53840] | 33760 [33760, 33760] | 2.81 [2.81, 2.81] | 9.7 [9.2, 10.2] | 0.0 [0.0, 0.0] | 16.0 [16.0, 16.0] |
| brute force (every expert grows at every look) | 0.732 [0.719, 0.745] | 0.623 [0.607, 0.639] | 562896 [562896, 562896] | 542816 [542816, 542816] | 34.63 [34.63, 34.63] | 125.7 [117.7, 133.6] | 0.0 [0.0, 0.0] | 16.0 [16.0, 16.0] |
| notebook-first, no width/depth growth | 0.730 [0.717, 0.744] | 0.580 [0.567, 0.594] | 55990 [53098, 58881] | 35910 [33018, 38801] | 2.95 [2.77, 3.13] | 6.5 [6.1, 6.8] | 22.3 [19.8, 24.8] | 16.0 [16.0, 16.0] |
| network at birth + governor | 0.672 [0.658, 0.687] | 0.506 [0.466, 0.545] | 38099 [37130, 39068] | 18019 [17050, 18988] | 1.83 [1.77, 1.89] | 4.3 [4.0, 4.5] | 4.5 [4.2, 4.8] | 16.0 [16.0, 16.0] |
| governed (all four directions) | 0.730 [0.717, 0.744] | 0.580 [0.567, 0.594] | 56425 [54098, 58751] | 36345 [34018, 38671] | 2.98 [2.84, 3.12] | 6.5 [6.1, 6.9] | 29.0 [26.1, 31.9] | 16.0 [16.0, 16.0] |

### Growth against the oracle (every regime of every seed counted once; `width` = a network of at least 16 units, `depth` = at least 2 blocks, `notebook` = no network)

| arm | need (regimes) | with width | with depth | notebook only | mean accuracy |
|---|---|---|---|---|---|
| governed (all four directions) | width (40) | 0.85 | 0.05 | 0.03 | 0.698 |
| governed (all four directions) | depth (9) | 0.67 | 0.11 | 0.00 | 0.630 |
| governed (all four directions) | none (31) | 0.23 | 0.10 | 0.42 | 0.801 |
| notebook-first, no width/depth growth | width (40) | 0.82 | 0.03 | 0.03 | 0.698 |
| notebook-first, no width/depth growth | depth (9) | 0.67 | 0.11 | 0.00 | 0.630 |
| notebook-first, no width/depth growth | none (31) | 0.23 | 0.10 | 0.42 | 0.801 |
| network at birth + governor | width (40) | 0.23 | 0.00 | 0.00 | 0.632 |
| network at birth + governor | depth (9) | 0.00 | 0.00 | 0.00 | 0.558 |
| network at birth + governor | none (31) | 0.00 | 0.03 | 0.00 | 0.757 |
| brute force (every expert grows at every look) | width (40) | 1.00 | 1.00 | 0.00 | 0.710 |
| brute force (every expert grows at every look) | depth (9) | 1.00 | 1.00 | 0.00 | 0.670 |
| brute force (every expert grows at every look) | none (31) | 1.00 | 1.00 | 0.00 | 0.779 |
| fixed, 8 units | width (40) | 0.00 | 0.00 | 0.00 | 0.625 |
| fixed, 8 units | depth (9) | 0.00 | 0.00 | 0.00 | 0.558 |
| fixed, 8 units | none (31) | 0.00 | 0.00 | 0.00 | 0.757 |

### What the governor did (events per seed, mean)

| arm | spawn | absorb | promote | keep notebook | growth trial rejected | grow | keep | revert |
|---|---|---|---|---|---|---|---|---|
| governed (all four directions) | 16.0 | 12.8 | 13.2 | 16.0 | 14.2 | 0.6 | 0.4 | 0.2 |
| notebook-first, no width/depth growth | 16.0 | 12.8 | 13.2 | 16.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| network at birth + governor | 16.0 | 12.8 | 0.0 | 0.0 | 18.0 | 4.4 | 3.4 | 0.8 |
| brute force (every expert grows at every look) | 16.0 | 12.8 | 0.0 | 0.0 | 0.0 | 112.0 | 0.0 | 0.0 |

## Easy (8 linear regimes: nothing needs growth)

| arm | mean accuracy | worst regime | total parameters | answer k-MACs | growth events per seed | experts |
|---|---|---|---|---|---|---|
| fixed, 8 units | 0.861 [0.850, 0.871] | 0.822 [0.803, 0.840] | 17064 [17064, 17064] | 1.33 [1.33, 1.33] | 0.0 | 8.0 [8.0, 8.0] |
| fixed, 32 units | 0.859 [0.846, 0.873] | 0.820 [0.795, 0.846] | 29544 [29544, 29544] | 2.89 [2.89, 2.89] | 0.0 | 8.0 [8.0, 8.0] |
| brute force (every expert grows at every look) | 0.842 [0.836, 0.847] | 0.804 [0.792, 0.815] | 281448 [281448, 281448] | 34.37 [34.37, 34.37] | 56.0 | 8.0 [8.0, 8.0] |
| governed (all four directions) | 0.902 [0.892, 0.912] | 0.846 [0.808, 0.884] | 13596 [10650, 16543] | 0.89 [0.52, 1.25] | 0.0 | 8.0 [8.0, 8.0] |

## Big (48 regimes in 8 groups: the flat gate costs 1,536 multiply-adds per sample, the budget is 640)

| arm | mean accuracy | worst regime | total parameters | answer k-MACs | experts | purity | tier built / refused (events per seed) |
|---|---|---|---|---|---|---|---|
| fixed, 8 units | 0.742 [0.736, 0.748] | 0.502 [0.468, 0.536] | 102384 [102384, 102384] | 2.61 [2.61, 2.61] | 48.0 [48.0, 48.0] | 0.954 [0.952, 0.956] | 0.0 / 0.0 |
| governor without the tier | 0.774 [0.768, 0.780] | 0.564 [0.540, 0.588] | 103749 [98369, 109129] | 2.64 [2.52, 2.76] | 48.0 [48.0, 48.0] | 0.954 [0.952, 0.956] | 0.0 / 0.0 |
| governed (all four directions) | 0.756 [0.750, 0.762] | 0.471 [0.416, 0.526] | 108790 [101536, 116045] | 2.33 [2.18, 2.47] | 48.0 [48.0, 48.0] | 0.924 [0.918, 0.930] | 3.6 / 0.0 |

## Pool against the regimes (G)

| scenario | experts | regimes | purity | regimes with an expert of their own |
|---|---|---|---|---|
| mixed | 16.0 [16.0, 16.0] | 16 | 0.986 [0.984, 0.989] | 1.000 [1.000, 1.000] |
| easy | 8.0 [8.0, 8.0] | 8 | 0.991 [0.990, 0.992] | 1.000 [1.000, 1.000] |
| big | 48.0 [48.0, 48.0] | 48 | 0.924 [0.918, 0.930] | 1.000 [1.000, 1.000] |
