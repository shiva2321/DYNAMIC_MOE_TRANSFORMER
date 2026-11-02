# Lab G5: predictions G1-G9 on the report seeds

9 declared predictions: **6 held, 3 missed** (seeds 100-109, 10 seeds).

| id | prediction | outcome | measured |
|---|---|---|---|
| G1 | `both:c,e0.5` beats `emb` on known skills in new wordings by at least +0.05 on average and in at least 7 of 10 seeds | HELD | +0.143, better in 10/10 seeds |
| G2 | its NMI expert-skill is at least 0.15 above `emb` on average and above it in at least 7 seeds | HELD | +0.311, better in 9/10 seeds |
| G3 | its all-groups accuracy is at least 0.02 above `emb` on average | HELD | +0.120, better in 10/10 seeds |
| G4 | `key:c` is at least 0.10 below `emb` on the new skill on average | MISSED | -0.072, better in 3/10 seeds |
| G5 | the cache arm cuts language-model passes by 25-50 % against `both:c,e0.5`, at most 10 % of its hits are wrong, all-groups accuracy within 0.03 and NMI within 0.05 | MISSED | passes cut 6%, wrong hits 5.5%, accuracy -0.014, NMI -0.035 |
| G6 | `oracle:c` has an NMI at least 0.10 above the fused arm | HELD | +0.180, above in 9/10 seeds |
| G7 | `emb` spawns at most 0.3 experts per seed in phase 2 on average | HELD | 0.10 per seed |
| G8 | the E-062 flag gives `emb:c` at least 0.8 more experts after phase 3 than `emb` without raising all-groups accuracy by more than 0.02 | MISSED | experts +0.70, accuracy -0.012 |
| G9 | the reader costs between 25 and 60 ms per message (the seeds whose language-model passes were not already cached) | HELD | 50 ms (range 23-89, 10 seeds) |

## The table of the report seeds (seeds 100-109; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-skill | acc taught skills (teach / new wording) | acc new skill (teach / new wording) | acc all four groups | unassigned share | language-model passes per seed | cache hits (wrong share of hits) | pool s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| emb | 10 | 2.5 / 2.7 | 0.10 | 0.10 | 0.07 | 0.729 (0.777 / 0.681) | 0.555 (0.570 / 0.540) | 0.642 | 0.16 | 0 | - | 4 |
| emb:c | 10 | 3.0 / 3.4 | 0.40 | 0.00 | 0.10 | 0.722 (0.781 / 0.662) | 0.539 (0.557 / 0.520) | 0.630 | 0.17 | 0 | - | 4 |
| key:c | 10 | 3.1 / 4.2 | 0.80 | 0.30 | 0.38 | 0.765 (0.786 / 0.745) | 0.483 (0.487 / 0.477) | 0.624 | 0.18 | 2063 | - | 4 |
| both:c | 10 | 3.0 / 4.1 | 0.70 | 0.40 | 0.37 | 0.828 (0.838 / 0.817) | 0.692 (0.718 / 0.667) | 0.760 | 0.16 | 2063 | - | 6 |
| both:c,e0.5 | 10 | 3.0 / 4.1 | 0.60 | 0.50 | 0.38 | 0.840 (0.855 / 0.824) | 0.684 (0.708 / 0.660) | 0.762 | 0.16 | 2063 | - | 6 |
| both:c,e0.5,k0.95 | 10 | 3.0 / 3.6 | 0.40 | 0.20 | 0.35 | 0.832 (0.848 / 0.816) | 0.662 (0.677 / 0.647) | 0.747 | 0.16 | 1943 | 37% (5.5%) | 6 |
| oracle:c | 10 | 3.0 / 5.2 | 1.30 | 0.90 | 0.56 | 0.889 (0.880 / 0.897) | 0.728 (0.745 / 0.710) | 0.808 | 0.15 | 0 | - | 5 |

## Paired against `emb` (same seeds; mean difference, and the number of seeds in which the arm is better)

| arm | acc all | acc taught in new wordings | acc new skill | NMI expert-skill |
|---|---|---|---|---|
| emb:c | -0.012 (3/10) | -0.018 (2/10) | -0.016 (3/10) | +0.026 (5/10) |
| key:c | -0.018 (4/10) | +0.064 (8/10) | -0.072 (3/10) | +0.311 (10/10) |
| both:c | +0.118 (10/10) | +0.137 (10/10) | +0.138 (9/10) | +0.301 (10/10) |
| both:c,e0.5 | +0.120 (10/10) | +0.143 (10/10) | +0.129 (10/10) | +0.311 (9/10) |
| both:c,e0.5,k0.95 | +0.105 (9/10) | +0.135 (10/10) | +0.108 (7/10) | +0.276 (9/10) |
| oracle:c | +0.166 (9/10) | +0.217 (10/10) | +0.173 (8/10) | +0.491 (10/10) |
