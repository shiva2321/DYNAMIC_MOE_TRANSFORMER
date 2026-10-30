# Lab G3, rewrites of known messages as a diagnostic of trust (seeds 150-152)

Shift flag = share of rewrites of held-out known messages that a reader flags at its own cut (5 % of the other half of the held-out messages). True known-flagged = the same reader on the real known test (arithmetic: new wordings; text domains: no shift). A reader is trusted when its shift flag is at most 0.1.

## arithmetic (3 seeds; per seed 169 cut messages, 190 rewrites, 120 known, 120 never-taught, 236 foreign)

| reader | shift flag | TRUE known flagged | never-taught found | foreign found | AUROC known vs never-taught | trusted |
|---|---|---|---|---|---|---|
| ask_exemplars | 0.099 | 0.100 | 0.694 | 0.988 | 0.921 | yes |
| vmf_conformal | 0.314 | 0.381 | 0.731 | 1.000 | 0.689 | no |
| mahalanobis | 0.511 | 0.417 | 0.933 | 1.000 | 0.698 | no |
| knn_distance | 0.040 | 0.336 | 0.331 | 1.000 | 0.628 | yes |

| system | known flagged | never-taught found | foreign found | AUROC known vs never-taught |
|---|---|---|---|---|
| either_fixed | 0.364 | 0.825 | 1.000 | 0.738 |
| both_fixed | 0.300 | 0.914 | 1.000 | 0.882 |
| trusted_either | 0.100 | 0.694 | 0.988 | 0.919 |
| trusted_either_calV | 0.067 | 0.528 | 0.973 | 0.921 |
| trusted_all_either | 0.314 | 0.372 | 1.000 | 0.666 |

Trusted pair readers per seed: {'ask_exemplars': 3}. Who is it? top-1 on known / on rewrites: ask 0.933 / 0.946; logistic regression on MiniLM 0.700 / 0.653.

## banking (3 seeds; per seed 180 cut messages, 635 rewrites, 180 known, 300 never-taught, 150 foreign)

| reader | shift flag | TRUE known flagged | never-taught found | foreign found | AUROC known vs never-taught | trusted |
|---|---|---|---|---|---|---|
| ask_exemplars | 0.087 | 0.044 | 0.321 | 0.904 | 0.799 | yes |
| vmf_conformal | 0.270 | 0.057 | 0.836 | 1.000 | 0.967 | no |
| mahalanobis | 0.276 | 0.046 | 0.740 | 1.000 | 0.961 | no |
| knn_distance | 0.165 | 0.046 | 0.719 | 1.000 | 0.953 | no |

| system | known flagged | never-taught found | foreign found | AUROC known vs never-taught |
|---|---|---|---|---|
| either_fixed | 0.076 | 0.797 | 1.000 | 0.953 |
| both_fixed | 0.054 | 0.580 | 0.960 | 0.909 |
| trusted_either | 0.044 | 0.321 | 0.904 | 0.798 |
| trusted_either_calV | 0.030 | 0.224 | 0.867 | 0.799 |
| trusted_all_either | 0.044 | 0.321 | 0.904 | 0.798 |

Trusted pair readers per seed: {'ask_exemplars': 3}. Who is it? top-1 on known / on rewrites: ask 0.911 / 0.878; logistic regression on MiniLM 0.989 / 0.978.

## agnews (3 seeds; per seed 60 cut messages, 141 rewrites, 180 known, 300 never-taught, 150 foreign)

| reader | shift flag | TRUE known flagged | never-taught found | foreign found | AUROC known vs never-taught | trusted |
|---|---|---|---|---|---|---|
| ask_exemplars | 0.088 | 0.085 | 0.553 | 0.527 | 0.854 | yes |
| vmf_conformal | 0.082 | 0.074 | 0.518 | 0.722 | 0.874 | yes |
| mahalanobis | 0.086 | 0.061 | 0.498 | 0.818 | 0.883 | yes |
| knn_distance | 0.073 | 0.063 | 0.563 | 0.773 | 0.892 | yes |

| system | known flagged | never-taught found | foreign found | AUROC known vs never-taught |
|---|---|---|---|---|
| either_fixed | 0.069 | 0.581 | 0.751 | 0.879 |
| both_fixed | 0.070 | 0.617 | 0.664 | 0.897 |
| trusted_either | 0.070 | 0.514 | 0.731 | 0.869 |
| trusted_either_calV | 0.035 | 0.326 | 0.533 | 0.875 |
| trusted_all_either | 0.087 | 0.623 | 0.856 | 0.878 |

Trusted pair readers per seed: {'ask_exemplars + vmf_conformal': 2, 'vmf_conformal': 1}. Who is it? top-1 on known / on rewrites: ask 0.867 / 0.812; logistic regression on MiniLM 0.913 / 0.891.

## newsgroups (3 seeds; per seed 180 cut messages, 404 rewrites, 180 known, 300 never-taught, 150 foreign)

| reader | shift flag | TRUE known flagged | never-taught found | foreign found | AUROC known vs never-taught | trusted |
|---|---|---|---|---|---|---|
| ask_exemplars | 0.106 | 0.052 | 0.131 | 0.062 | 0.770 | no |
| vmf_conformal | 0.077 | 0.056 | 0.257 | 0.556 | 0.740 | yes |
| mahalanobis | 0.080 | 0.048 | 0.233 | 0.644 | 0.779 | yes |
| knn_distance | 0.073 | 0.059 | 0.250 | 0.624 | 0.775 | yes |

| system | known flagged | never-taught found | foreign found | AUROC known vs never-taught |
|---|---|---|---|---|
| either_fixed | 0.037 | 0.196 | 0.404 | 0.768 |
| both_fixed | 0.054 | 0.216 | 0.216 | 0.802 |
| trusted_either | 0.048 | 0.212 | 0.509 | 0.747 |
| trusted_either_calV | 0.033 | 0.192 | 0.458 | 0.748 |
| trusted_all_either | 0.044 | 0.210 | 0.567 | 0.780 |

Trusted pair readers per seed: {'vmf_conformal': 2, 'ask_exemplars + vmf_conformal': 1}. Who is it? top-1 on known / on rewrites: ask 0.752 / 0.691; logistic regression on MiniLM 0.820 / 0.784.

## Verdicts (rules fixed before the run)

- C6a (the diagnostic): Spearman 0.044 (>= 0.6); readers with a true known-flagged rate above 0.15: 3, all untrusted: False; readers at or under 0.1: 13, share trusted 0.69 (>= 0.80). **FAIL**
- C6b, arithmetic: trusted-either flags 0.100 of known (<= 0.15) and finds 0.694 (best usable reader 0.694, need >= that - 0.05). **PASS**
- C6b, banking: trusted-either flags 0.044 of known (<= 0.15) and finds 0.321 (best usable reader 0.836, need >= that - 0.05). **FAIL**
- C6b, agnews: trusted-either flags 0.070 of known (<= 0.15) and finds 0.514 (best usable reader 0.563, need >= that - 0.05). **PASS**
- C6b, newsgroups: trusted-either flags 0.048 of known (<= 0.15) and finds 0.212 (best usable reader 0.257, need >= that - 0.05). **PASS**
- C6b overall (all four domains): **FAIL**
