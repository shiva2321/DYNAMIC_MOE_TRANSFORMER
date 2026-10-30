# Lab G2, the same experts behind four gates (arithmetic in everyday language, seeds 150-152)

Per seed about 120 known messages in new wordings (a right answer exists) and messages nobody owns (120 never-taught, 90 never-taught in teaching wordings, 236 everyday, 60 look-alike traps), where any answer is wrong. Experts: one solver per taught skill, numbers read by a fixed parser (declared simplification). Net = correct - wrong on the known messages - the mean share of unowned messages that got an answer.

| gate | answered | correct | wrong | precision (correct / answered) | routed right | correct, Solid owner | correct, Thin owner | NET |
|---|---|---|---|---|---|---|---|---|
| oracle | 1.000 | 0.497 | 0.503 | 0.497 | 1.000 | 0.696 | 0.100 | -0.006 |
| ask | 0.917 | 0.447 | 0.469 | 0.488 | 0.979 | 0.633 | 0.075 | -0.253 |
| vmf | 0.619 | 0.286 | 0.333 | 0.459 | 0.936 | 0.396 | 0.067 | -0.223 |
| mix | 0.603 | 0.300 | 0.303 | 0.493 | 0.977 | 0.421 | 0.058 | -0.094 |

Share of messages nobody owns that got an answer (lower is better):

| gate | unknown | teaching_wording | foreign | trap |
|---|---|---|---|---|
| oracle | 0.000 | 0.000 | 0.000 | 0.000 |
| ask | 0.419 | 0.504 | 0.000 | 0.000 |
| vmf | 0.244 | 0.459 | 0.000 | 0.000 |
| mix | 0.114 | 0.252 | 0.000 | 0.000 |

Expert accuracy on problems it never saw: Solid experts 0.753, Thin experts 0.127.

**C7** (asking gate NET at least 0.15 above the vMF gate's): ask -0.253, vmf -0.223, difference -0.030. **FAIL**
