# Wording-confound test (closed-form classifier, no pool)

A classifier fitted on the taught skills' teaching rows plus m examples of each of the 28 withheld skills; read on taught skills in held-out wordings (known ho) and on the withheld skills (novel); `errors to new skills` = share of the known-ho errors that land on a withheld skill. Mean [95 % interval over seeds].

## m = 10 examples per new skill (5 seeds)

| new skills' examples | known ho | novel | known-ho errors that land on a new skill |
|---|---|---|---|
| none | 0.878 [0.871, 0.885] | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] |
| tw | 0.858 [0.848, 0.868] | 0.778 [0.760, 0.796] | 0.21 [0.18, 0.25] |
| ho | 0.789 [0.780, 0.799] | 0.813 [0.793, 0.832] | 0.60 [0.58, 0.63] |
| mixed | 0.813 [0.803, 0.824] | 0.833 [0.816, 0.851] | 0.53 [0.48, 0.58] |
| balanced | 0.849 [0.842, 0.855] | 0.828 [0.810, 0.846] | 0.44 [0.41, 0.47] |

## m = 20 examples per new skill (5 seeds)

| new skills' examples | known ho | novel | known-ho errors that land on a new skill |
|---|---|---|---|
| none | 0.878 [0.871, 0.885] | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] |
| tw | 0.855 [0.846, 0.864] | 0.807 [0.793, 0.821] | 0.25 [0.23, 0.28] |
| ho | 0.773 [0.761, 0.785] | 0.859 [0.834, 0.884] | 0.67 [0.64, 0.69] |
| mixed | 0.800 [0.787, 0.812] | 0.873 [0.855, 0.892] | 0.59 [0.56, 0.63] |
| balanced | 0.854 [0.843, 0.864] | 0.860 [0.839, 0.881] | 0.47 [0.46, 0.49] |
