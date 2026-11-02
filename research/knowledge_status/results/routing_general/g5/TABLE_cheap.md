# Lab G5, the pool fed by different keys (seeds 100-109; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-skill | acc taught skills (teach / new wording) | acc new skill (teach / new wording) | acc all four groups | unassigned share | language-model passes per seed | cache hits (wrong share of hits) | pool s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| emb | 10 | 2.5 / 2.7 | 0.10 | 0.10 | 0.07 | 0.729 (0.777 / 0.681) | 0.555 (0.570 / 0.540) | 0.642 | 0.16 | 0 | - | 4 |
| both:c,e0.5 | 10 | 3.0 / 4.1 | 0.60 | 0.50 | 0.38 | 0.840 (0.855 / 0.824) | 0.684 (0.708 / 0.660) | 0.762 | 0.16 | 2063 | - | 6 |
| both:c,e0.5,k0.95 | 10 | 3.0 / 3.6 | 0.40 | 0.20 | 0.35 | 0.832 (0.848 / 0.816) | 0.662 (0.677 / 0.647) | 0.747 | 0.16 | 1562 | 35% (5.2%) | 6 |
| both:c,e0.5,k0.9 | 10 | 3.0 / 4.0 | 0.60 | 0.40 | 0.35 | 0.825 (0.843 / 0.807) | 0.692 (0.693 / 0.693) | 0.759 | 0.16 | 1091 | 56% (7.1%) | 6 |
| both:c,e0.5,t0.9 | 10 | 3.4 / 3.9 | 0.40 | 0.10 | 0.31 | 0.797 (0.806 / 0.787) | 0.575 (0.593 / 0.557) | 0.686 | 0.19 | 919 | 60% (5.8%) | 5 |
| both:c,e0.5,t0.9,m0.8 | 10 | 3.1 / 3.7 | 0.50 | 0.10 | 0.34 | 0.842 (0.857 / 0.827) | 0.641 (0.652 / 0.630) | 0.742 | 0.17 | 1005 | 56% (2.1%) | 6 |

## Paired against `emb` (same seeds; mean difference, and the number of seeds in which the arm is better)

| arm | acc all | acc taught in new wordings | acc new skill | NMI expert-skill |
|---|---|---|---|---|
| both:c,e0.5 | +0.120 (10/10) | +0.143 (10/10) | +0.129 (10/10) | +0.311 (9/10) |
| both:c,e0.5,k0.95 | +0.105 (9/10) | +0.135 (10/10) | +0.108 (7/10) | +0.276 (9/10) |
| both:c,e0.5,k0.9 | +0.117 (10/10) | +0.126 (10/10) | +0.138 (9/10) | +0.275 (9/10) |
| both:c,e0.5,t0.9 | +0.044 (8/10) | +0.107 (9/10) | +0.020 (6/10) | +0.237 (9/10) |
| both:c,e0.5,t0.9,m0.8 | +0.100 (9/10) | +0.147 (10/10) | +0.086 (8/10) | +0.268 (9/10) |
