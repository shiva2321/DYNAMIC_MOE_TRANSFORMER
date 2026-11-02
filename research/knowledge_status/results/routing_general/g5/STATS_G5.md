# Lab G5: paired differences against `emb` (seeds 100-109)

Mean difference per seed, its standard error, the range over seeds, and the exact sign test (two-sided) on the seeds where the arm is better or worse.

## accuracy, all four groups

| arm | mean diff | s.e. | min / max | better in | sign test p |
|---|---|---|---|---|---|
| emb:c | -0.012 | 0.025 | -0.131 / +0.140 | 3/10 | 1.000 |
| key:c | -0.018 | 0.032 | -0.135 / +0.169 | 4/10 | 0.754 |
| both:c | +0.118 | 0.020 | +0.056 / +0.235 | 10/10 | 0.002 |
| both:c,e0.5 | +0.120 | 0.015 | +0.062 / +0.229 | 10/10 | 0.002 |
| both:c,e0.5,k0.95 | +0.105 | 0.028 | -0.085 / +0.256 | 9/10 | 0.021 |
| oracle:c | +0.166 | 0.044 | -0.175 / +0.310 | 9/10 | 0.021 |

## known skills, new wordings

| arm | mean diff | s.e. | min / max | better in | sign test p |
|---|---|---|---|---|---|
| emb:c | -0.018 | 0.015 | -0.100 / +0.050 | 2/10 | 0.453 |
| key:c | +0.064 | 0.028 | -0.083 / +0.167 | 8/10 | 0.109 |
| both:c | +0.137 | 0.015 | +0.083 / +0.233 | 10/10 | 0.002 |
| both:c,e0.5 | +0.143 | 0.014 | +0.058 / +0.225 | 10/10 | 0.002 |
| both:c,e0.5,k0.95 | +0.135 | 0.017 | +0.042 / +0.217 | 10/10 | 0.002 |
| oracle:c | +0.217 | 0.016 | +0.142 / +0.292 | 10/10 | 0.002 |

## known skills, both wordings

| arm | mean diff | s.e. | min / max | better in | sign test p |
|---|---|---|---|---|---|
| emb:c | -0.007 | 0.010 | -0.062 / +0.050 | 3/10 | 1.000 |
| key:c | +0.037 | 0.024 | -0.075 / +0.113 | 7/10 | 0.344 |
| both:c | +0.099 | 0.015 | +0.033 / +0.171 | 10/10 | 0.002 |
| both:c,e0.5 | +0.111 | 0.015 | +0.038 / +0.179 | 10/10 | 0.002 |
| both:c,e0.5,k0.95 | +0.103 | 0.017 | +0.025 / +0.171 | 10/10 | 0.002 |
| oracle:c | +0.160 | 0.013 | +0.083 / +0.212 | 10/10 | 0.002 |

## the new skill

| arm | mean diff | s.e. | min / max | better in | sign test p |
|---|---|---|---|---|---|
| emb:c | -0.016 | 0.053 | -0.238 / +0.325 | 3/10 | 1.000 |
| key:c | -0.072 | 0.060 | -0.338 / +0.287 | 3/10 | 0.344 |
| both:c | +0.138 | 0.040 | -0.012 / +0.400 | 9/10 | 0.021 |
| both:c,e0.5 | +0.129 | 0.031 | +0.025 / +0.362 | 10/10 | 0.002 |
| both:c,e0.5,k0.95 | +0.108 | 0.065 | -0.325 / +0.463 | 7/10 | 0.344 |
| oracle:c | +0.173 | 0.091 | -0.525 / +0.537 | 8/10 | 0.109 |

## NMI expert-skill

| arm | mean diff | s.e. | min / max | better in | sign test p |
|---|---|---|---|---|---|
| emb:c | +0.026 | 0.024 | -0.125 / +0.127 | 5/10 | 0.453 |
| key:c | +0.311 | 0.025 | +0.198 / +0.423 | 10/10 | 0.002 |
| both:c | +0.301 | 0.031 | +0.149 / +0.430 | 10/10 | 0.002 |
| both:c,e0.5 | +0.311 | 0.046 | -0.033 / +0.473 | 9/10 | 0.021 |
| both:c,e0.5,k0.95 | +0.276 | 0.045 | -0.035 / +0.427 | 9/10 | 0.021 |
| oracle:c | +0.491 | 0.038 | +0.334 / +0.725 | 10/10 | 0.002 |

## Accuracy per test group (mean over seeds)

| arm | known skills, teaching wordings | known skills, new wordings | new skill, teaching wordings | new skill, new wordings |
|---|---|---|---|---|
| emb | 0.777 | 0.681 | 0.570 | 0.540 |
| emb:c | 0.781 | 0.662 | 0.557 | 0.520 |
| key:c | 0.786 | 0.745 | 0.487 | 0.477 |
| both:c | 0.838 | 0.817 | 0.718 | 0.667 |
| both:c,e0.5 | 0.855 | 0.824 | 0.708 | 0.660 |
| both:c,e0.5,k0.95 | 0.848 | 0.816 | 0.677 | 0.647 |
| oracle:c | 0.880 | 0.897 | 0.745 | 0.710 |
