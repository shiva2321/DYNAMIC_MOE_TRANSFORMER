# P4 / I3: the refusal layer on the live pool (generated)

Sequential schedule, seeds 503, 504. Calibration: 60 verified known messages (mean over 5 draws), level q = 0.05, evaluation: the rest of the known text messages (<= 150 per regime) and 300 messages of each foreign set. Arithmetic regimes are digit codes and are not part of this test.

## UG0: primary

| system | false refusal | coverage | accuracy (all) | accuracy (answered) | science_questions refused | stories refused | wiki refused |
|---|---|---|---|---|---|---|---|
| pool flag (built-in, uncalibrated) | 0.000 | 1.000 | 0.599 | 0.599 | 0.000 | 0.000 | 0.000 |
| pool calibrated | 0.064 | 0.936 | 0.599 | 0.610 | 0.535 | 0.316 | 0.416 |
| reader claim (uncalibrated) | 0.382 | 0.618 | 0.599 | 0.676 | 0.948 | 0.848 | 0.855 |
| reader calibrated | 0.055 | 0.945 | 0.599 | 0.614 | 0.118 | 0.240 | 0.093 |
| either calibrated (pool + reader) | 0.073 | 0.927 | 0.599 | 0.617 | 0.463 | 0.317 | 0.339 |
| density calibrated (Mahalanobis) | 0.052 | 0.948 | 0.599 | 0.613 | 0.612 | 0.163 | 0.500 |
| either calibrated (pool + reader + density) | 0.061 | 0.939 | 0.599 | 0.615 | 0.565 | 0.231 | 0.401 |

## UG0: more verified messages (200)

| system | false refusal | coverage | accuracy (all) | accuracy (answered) | science_questions refused | stories refused | wiki refused |
|---|---|---|---|---|---|---|---|
| pool flag (built-in, uncalibrated) | 0.000 | 1.000 | 0.598 | 0.598 | 0.000 | 0.000 | 0.000 |
| pool calibrated | 0.045 | 0.955 | 0.598 | 0.608 | 0.498 | 0.257 | 0.365 |
| reader claim (uncalibrated) | 0.381 | 0.619 | 0.598 | 0.674 | 0.948 | 0.848 | 0.855 |
| reader calibrated | 0.052 | 0.948 | 0.598 | 0.613 | 0.095 | 0.227 | 0.082 |
| either calibrated (pool + reader) | 0.048 | 0.952 | 0.598 | 0.610 | 0.422 | 0.217 | 0.293 |
| density calibrated (Mahalanobis) | 0.041 | 0.959 | 0.598 | 0.610 | 0.606 | 0.117 | 0.484 |
| either calibrated (pool + reader + density) | 0.053 | 0.947 | 0.598 | 0.614 | 0.562 | 0.206 | 0.386 |

## UG0: looser level (q = 0.10)

| system | false refusal | coverage | accuracy (all) | accuracy (answered) | science_questions refused | stories refused | wiki refused |
|---|---|---|---|---|---|---|---|
| pool flag (built-in, uncalibrated) | 0.000 | 1.000 | 0.599 | 0.599 | 0.000 | 0.000 | 0.000 |
| pool calibrated | 0.111 | 0.889 | 0.599 | 0.619 | 0.643 | 0.482 | 0.550 |
| reader claim (uncalibrated) | 0.382 | 0.618 | 0.599 | 0.676 | 0.948 | 0.848 | 0.855 |
| reader calibrated | 0.097 | 0.903 | 0.599 | 0.624 | 0.256 | 0.415 | 0.214 |
| either calibrated (pool + reader) | 0.117 | 0.883 | 0.599 | 0.626 | 0.598 | 0.499 | 0.479 |
| density calibrated (Mahalanobis) | 0.103 | 0.897 | 0.599 | 0.624 | 0.813 | 0.360 | 0.682 |
| either calibrated (pool + reader + density) | 0.126 | 0.874 | 0.599 | 0.631 | 0.743 | 0.460 | 0.591 |

## UG0: AUROC of each reader's novelty score (known text against each foreign set; 0.5 = chance)

| reader | science_questions | stories | wiki |
|---|---|---|---|
| pool | 0.867 | 0.828 | 0.838 |
| asking reader | 0.812 | 0.821 | 0.768 |
| density | 0.942 | 0.825 | 0.900 |

## UPLUS: primary

| system | false refusal | coverage | accuracy (all) | accuracy (answered) | science_questions refused | stories refused | wiki refused |
|---|---|---|---|---|---|---|---|
| pool flag (built-in, uncalibrated) | 0.000 | 1.000 | 0.616 | 0.616 | 0.000 | 0.000 | 0.000 |
| pool calibrated | 0.064 | 0.936 | 0.616 | 0.626 | 0.535 | 0.316 | 0.416 |
| reader claim (uncalibrated) | 0.382 | 0.618 | 0.616 | 0.690 | 0.948 | 0.848 | 0.855 |
| reader calibrated | 0.055 | 0.945 | 0.616 | 0.630 | 0.118 | 0.240 | 0.093 |
| either calibrated (pool + reader) | 0.073 | 0.927 | 0.616 | 0.632 | 0.463 | 0.317 | 0.339 |
| density calibrated (Mahalanobis) | 0.052 | 0.948 | 0.616 | 0.629 | 0.612 | 0.163 | 0.500 |
| either calibrated (pool + reader + density) | 0.061 | 0.939 | 0.616 | 0.631 | 0.565 | 0.231 | 0.401 |

## UPLUS: more verified messages (200)

| system | false refusal | coverage | accuracy (all) | accuracy (answered) | science_questions refused | stories refused | wiki refused |
|---|---|---|---|---|---|---|---|
| pool flag (built-in, uncalibrated) | 0.000 | 1.000 | 0.617 | 0.617 | 0.000 | 0.000 | 0.000 |
| pool calibrated | 0.045 | 0.955 | 0.617 | 0.626 | 0.498 | 0.257 | 0.365 |
| reader claim (uncalibrated) | 0.381 | 0.619 | 0.617 | 0.691 | 0.948 | 0.848 | 0.855 |
| reader calibrated | 0.052 | 0.948 | 0.617 | 0.630 | 0.095 | 0.227 | 0.082 |
| either calibrated (pool + reader) | 0.048 | 0.952 | 0.617 | 0.628 | 0.422 | 0.217 | 0.293 |
| density calibrated (Mahalanobis) | 0.041 | 0.959 | 0.617 | 0.628 | 0.606 | 0.117 | 0.484 |
| either calibrated (pool + reader + density) | 0.053 | 0.947 | 0.617 | 0.632 | 0.562 | 0.206 | 0.386 |

## UPLUS: looser level (q = 0.10)

| system | false refusal | coverage | accuracy (all) | accuracy (answered) | science_questions refused | stories refused | wiki refused |
|---|---|---|---|---|---|---|---|
| pool flag (built-in, uncalibrated) | 0.000 | 1.000 | 0.616 | 0.616 | 0.000 | 0.000 | 0.000 |
| pool calibrated | 0.111 | 0.889 | 0.616 | 0.634 | 0.643 | 0.482 | 0.550 |
| reader claim (uncalibrated) | 0.382 | 0.618 | 0.616 | 0.690 | 0.948 | 0.848 | 0.855 |
| reader calibrated | 0.097 | 0.903 | 0.616 | 0.639 | 0.256 | 0.415 | 0.214 |
| either calibrated (pool + reader) | 0.117 | 0.883 | 0.616 | 0.642 | 0.598 | 0.499 | 0.479 |
| density calibrated (Mahalanobis) | 0.103 | 0.897 | 0.616 | 0.641 | 0.813 | 0.360 | 0.682 |
| either calibrated (pool + reader + density) | 0.126 | 0.874 | 0.616 | 0.647 | 0.743 | 0.460 | 0.591 |

## UPLUS: AUROC of each reader's novelty score (known text against each foreign set; 0.5 = chance)

| reader | science_questions | stories | wiki |
|---|---|---|---|
| pool | 0.867 | 0.828 | 0.838 |
| asking reader | 0.812 | 0.821 | 0.768 |
| density | 0.942 | 0.825 | 0.900 |
