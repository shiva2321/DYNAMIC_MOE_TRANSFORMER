# Layer E report: execution outcomes on real code (seeds [710, 711, 712, 713, 714])

Known = five taught list functions in held-out styles (template styles and real library idioms); near = withheld product and count_odd; far = withheld real library functions (length, median, range, second largest, mean, last). AUROC = known in held-out styles vs withheld. Flags at the conformal cut (q = 0.05, n = 40 known snippets). M = lists each snippet is executed on. `fingerprint` = outputs on fixed probes against the taught families' reference outputs (nothing learned: a ceiling).

Snippets per seed (mean): known in teaching styles 60 [60, 60], known in held-out styles 180 [180, 180] (60 [60, 60] library idioms), near 87 [87, 87], far 36 [36, 36]. Transitions per snippet at M = 1, 3, 10: 6, 15, 48 (first seed).

## M = 1 executions per snippet

| signal | AUROC vs near-twins | AUROC vs real library functions | flag known, template style | flag known, library idiom | flag known, teaching style | near found (template) | near found (library) | far found |
|---|---|---|---|---|---|---|---|---|
| look | 0.588 [0.566, 0.610] | 0.904 [0.895, 0.913] | 0.000 [-0.000, 0.001] | 0.167 [0.139, 0.195] | 0.000 [0.000, 0.000] | 0.004 [-0.000, 0.008] | 0.099 [0.032, 0.166] | 0.344 [0.259, 0.430] |
| cascade | 0.930 [0.923, 0.937] | 0.928 [0.909, 0.948] | 0.022 [0.012, 0.033] | 0.067 [0.044, 0.091] | 0.000 [0.000, 0.000] | 0.470 [0.244, 0.695] | 0.505 [0.367, 0.644] | 0.491 [0.413, 0.569] |
| result_all | 1.000 [1.000, 1.000] | 0.963 [0.873, 1.054] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.900 [0.715, 1.085] |
| fingerprint | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

## M = 3 executions per snippet

| signal | AUROC vs near-twins | AUROC vs real library functions | flag known, template style | flag known, library idiom | flag known, teaching style | near found (template) | near found (library) | far found |
|---|---|---|---|---|---|---|---|---|
| look | 0.588 [0.566, 0.610] | 0.904 [0.895, 0.913] | 0.000 [-0.000, 0.001] | 0.167 [0.139, 0.195] | 0.000 [0.000, 0.000] | 0.004 [-0.000, 0.008] | 0.099 [0.032, 0.166] | 0.344 [0.259, 0.430] |
| cascade | 0.933 [0.925, 0.940] | 0.927 [0.905, 0.948] | 0.022 [0.008, 0.035] | 0.073 [0.055, 0.091] | 0.000 [0.000, 0.000] | 0.550 [0.529, 0.570] | 0.561 [0.426, 0.696] | 0.435 [0.253, 0.618] |
| result_all | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| fingerprint | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

## M = 10 executions per snippet

| signal | AUROC vs near-twins | AUROC vs real library functions | flag known, template style | flag known, library idiom | flag known, teaching style | near found (template) | near found (library) | far found |
|---|---|---|---|---|---|---|---|---|
| look | 0.588 [0.566, 0.610] | 0.904 [0.895, 0.913] | 0.000 [-0.000, 0.001] | 0.167 [0.139, 0.195] | 0.000 [0.000, 0.000] | 0.004 [-0.000, 0.008] | 0.099 [0.032, 0.166] | 0.344 [0.259, 0.430] |
| cascade | 0.938 [0.927, 0.948] | 0.927 [0.908, 0.946] | 0.022 [0.010, 0.034] | 0.080 [0.076, 0.084] | 0.000 [0.000, 0.000] | 0.555 [0.527, 0.583] | 0.607 [0.520, 0.694] | 0.421 [0.292, 0.550] |
| result_all | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| fingerprint | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
