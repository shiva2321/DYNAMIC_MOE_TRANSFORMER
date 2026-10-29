# Lab E, the boundary question (seeds 150-154)

Stage one: the model is asked whether the message asks for a calculation at all; the threshold is calibrated on held-out arithmetic requests (taught skills, wordings S1-S3) so that about alpha of them are called Other. Stage two: the rule tau0.5 of lab A.

Separating arithmetic requests from look-alike traps by the Yes-probability: AUROC 0.986; from everyday sentences: 0.980.

| | tau0.5 alone | two-stage, alpha 0.01 | two-stage, alpha 0.05 |
|---|---|---|---|
| known skills in new wordings called unseen | 0.053 | 0.053 | 0.048 |
| never-taught skills found | 0.990 | 0.984 | 0.965 |
| look-alike traps called unseen | 0.600 | 0.347 | 0.100 |
| everyday sentences called unseen | 0.460 | 0.207 | 0.090 |
| everyday sentences labelled Other | 0.001 | 0.616 | 0.813 |
| traps labelled Other | 0.000 | 0.420 | 0.807 |
| real arithmetic in new wordings wrongly called Other | | 0.005 | 0.023 |
| never-taught arithmetic wrongly called Other | | 0.006 | 0.026 |
| traps called Other | | 0.420 | 0.807 |
| everyday sentences called Other | | 0.616 | 0.813 |
