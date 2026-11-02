# Lab G5: paired differences against `emb` (seeds 100-109)

Mean difference per seed, its standard error, the range over seeds, and the exact sign test (two-sided) on the seeds where the arm is better or worse.

## accuracy, all four groups

| arm | mean diff | s.e. | min / max | better in | sign test p |
|---|---|---|---|---|---|
| both:c,e0.5 | +0.120 | 0.015 | +0.062 / +0.229 | 10/10 | 0.002 |
| both:c,e0.5,k0.95 | +0.105 | 0.028 | -0.085 / +0.256 | 9/10 | 0.021 |
| both:c,e0.5,k0.9 | +0.117 | 0.019 | +0.035 / +0.254 | 10/10 | 0.002 |
| both:c,e0.5,t0.9 | +0.044 | 0.027 | -0.165 / +0.158 | 8/10 | 0.109 |
| both:c,e0.5,t0.9,m0.8 | +0.100 | 0.019 | -0.002 / +0.231 | 9/10 | 0.021 |

## known skills, new wordings

| arm | mean diff | s.e. | min / max | better in | sign test p |
|---|---|---|---|---|---|
| both:c,e0.5 | +0.143 | 0.014 | +0.058 / +0.225 | 10/10 | 0.002 |
| both:c,e0.5,k0.95 | +0.135 | 0.017 | +0.042 / +0.217 | 10/10 | 0.002 |
| both:c,e0.5,k0.9 | +0.126 | 0.020 | +0.025 / +0.225 | 10/10 | 0.002 |
| both:c,e0.5,t0.9 | +0.107 | 0.030 | -0.075 / +0.208 | 9/10 | 0.021 |
| both:c,e0.5,t0.9,m0.8 | +0.147 | 0.023 | +0.025 / +0.233 | 10/10 | 0.002 |

## known skills, both wordings

| arm | mean diff | s.e. | min / max | better in | sign test p |
|---|---|---|---|---|---|
| both:c,e0.5 | +0.111 | 0.015 | +0.038 / +0.179 | 10/10 | 0.002 |
| both:c,e0.5,k0.95 | +0.103 | 0.017 | +0.025 / +0.171 | 10/10 | 0.002 |
| both:c,e0.5,k0.9 | +0.096 | 0.017 | +0.038 / +0.179 | 10/10 | 0.002 |
| both:c,e0.5,t0.9 | +0.068 | 0.028 | -0.121 / +0.158 | 8/10 | 0.109 |
| both:c,e0.5,t0.9,m0.8 | +0.113 | 0.020 | +0.004 / +0.188 | 10/10 | 0.002 |

## the new skill

| arm | mean diff | s.e. | min / max | better in | sign test p |
|---|---|---|---|---|---|
| both:c,e0.5 | +0.129 | 0.031 | +0.025 / +0.362 | 10/10 | 0.002 |
| both:c,e0.5,k0.95 | +0.108 | 0.065 | -0.325 / +0.463 | 7/10 | 0.344 |
| both:c,e0.5,k0.9 | +0.138 | 0.039 | -0.038 / +0.450 | 9/10 | 0.021 |
| both:c,e0.5,t0.9 | +0.020 | 0.063 | -0.387 / +0.338 | 6/10 | 0.754 |
| both:c,e0.5,t0.9,m0.8 | +0.086 | 0.047 | -0.138 / +0.412 | 8/10 | 0.109 |

## NMI expert-skill

| arm | mean diff | s.e. | min / max | better in | sign test p |
|---|---|---|---|---|---|
| both:c,e0.5 | +0.311 | 0.046 | -0.033 / +0.473 | 9/10 | 0.021 |
| both:c,e0.5,k0.95 | +0.276 | 0.045 | -0.035 / +0.427 | 9/10 | 0.021 |
| both:c,e0.5,k0.9 | +0.275 | 0.047 | -0.071 / +0.415 | 9/10 | 0.021 |
| both:c,e0.5,t0.9 | +0.237 | 0.043 | -0.074 / +0.410 | 9/10 | 0.021 |
| both:c,e0.5,t0.9,m0.8 | +0.268 | 0.044 | -0.072 / +0.416 | 9/10 | 0.021 |

## Accuracy per test group (mean over seeds)

| arm | known skills, teaching wordings | known skills, new wordings | new skill, teaching wordings | new skill, new wordings |
|---|---|---|---|---|
| emb | 0.777 | 0.681 | 0.570 | 0.540 |
| both:c,e0.5 | 0.855 | 0.824 | 0.708 | 0.660 |
| both:c,e0.5,k0.95 | 0.848 | 0.816 | 0.677 | 0.647 |
| both:c,e0.5,k0.9 | 0.843 | 0.807 | 0.693 | 0.693 |
| both:c,e0.5,t0.9 | 0.806 | 0.787 | 0.593 | 0.557 |
| both:c,e0.5,t0.9,m0.8 | 0.857 | 0.827 | 0.652 | 0.630 |
