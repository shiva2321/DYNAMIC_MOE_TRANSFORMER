# From-scratch reader arms, operating points (tuning seeds 150 to 154, E1 ceiling)

Rule fixed before the run: the setting with the highest mean of B1a, B1b and B1c; ties to the smaller network. Dev and confirmatory seeds were not touched.

## R5_FS_strict

| setting | mean(B1a, B1b, B1c) |
|---|---|
| {"epochs": 300, "hidden": 1024, "lr": 0.001} | 0.6398 (winner) |
| {"epochs": 800, "hidden": 1024, "lr": 0.001} | 0.6395 |
| {"epochs": 300, "hidden": 256, "lr": 0.001} | 0.6381 |
| {"epochs": 800, "hidden": 256, "lr": 0.001} | 0.6377 |
| {"epochs": 800, "hidden": 256, "lr": 0.003} | 0.6271 |
| {"epochs": 300, "hidden": 256, "lr": 0.003} | 0.6257 |
| {"epochs": 800, "hidden": 1024, "lr": 0.003} | 0.6208 |
| {"epochs": 300, "hidden": 1024, "lr": 0.003} | 0.6196 |

## R5_FS_pre

| setting | mean(B1a, B1b, B1c) |
|---|---|
| {"epochs": 300, "hidden": 1024, "lr": 0.001} | 0.7415 (winner) |
| {"epochs": 800, "hidden": 1024, "lr": 0.001} | 0.7402 |
| {"epochs": 300, "hidden": 256, "lr": 0.001} | 0.7392 |
| {"epochs": 800, "hidden": 256, "lr": 0.001} | 0.7360 |
| {"epochs": 800, "hidden": 1024, "lr": 0.003} | 0.7279 |
| {"epochs": 300, "hidden": 256, "lr": 0.003} | 0.7256 |
| {"epochs": 300, "hidden": 1024, "lr": 0.003} | 0.7251 |
| {"epochs": 800, "hidden": 256, "lr": 0.003} | 0.7233 |

