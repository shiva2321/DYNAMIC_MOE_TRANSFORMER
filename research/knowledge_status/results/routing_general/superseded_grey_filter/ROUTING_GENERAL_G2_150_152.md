# Lab G2, the same experts behind four gates (arithmetic in everyday language, seeds 150-152)

Per seed about 80 known messages in new wordings (a right answer exists) and messages nobody owns (120 never-taught, 90 never-taught in teaching wordings, 236 everyday, 60 look-alike traps), where any answer is wrong. Experts: one solver per taught skill, numbers read by a fixed parser (declared simplification). Net = correct - wrong on the known messages - the mean share of unowned messages that got an answer.

| gate | answered | correct | wrong | precision (correct / answered) | routed right | correct, Solid owner | correct, Thin owner | NET |
|---|---|---|---|---|---|---|---|---|
| oracle | 1.000 | 0.263 | 0.737 | 0.263 | 1.000 | 0.975 | 0.100 | -0.475 |
| ask | 0.896 | 0.229 | 0.667 | 0.251 | 0.972 | 0.925 | 0.075 | -0.668 |
| vmf | 0.617 | 0.167 | 0.450 | 0.259 | 0.940 | 0.600 | 0.067 | -0.459 |
| mix | 0.600 | 0.200 | 0.400 | 0.308 | 0.985 | 0.850 | 0.058 | -0.291 |

Share of messages nobody owns that got an answer (lower is better):

| gate | unknown | teaching_wording | foreign | trap |
|---|---|---|---|---|
| oracle | 0.000 | 0.000 | 0.000 | 0.000 |
| ask | 0.419 | 0.504 | 0.000 | 0.000 |
| vmf | 0.244 | 0.459 | 0.000 | 0.000 |
| mix | 0.114 | 0.252 | 0.000 | 0.000 |

Expert accuracy on problems it never saw: Solid experts 0.753, Thin experts 0.127.

**C7** (asking gate NET at least 0.15 above the vMF gate's): ask -0.668, vmf -0.459, difference -0.209. **FAIL**
