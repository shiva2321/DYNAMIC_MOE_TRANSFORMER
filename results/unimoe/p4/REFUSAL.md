# P4 / I3: the refusal layer on the live pool (generated)

Sequential schedule, seeds 500, 501, 502. Calibration: 60 verified known messages (mean over 5 draws), level q = 0.05, evaluation: the rest of the known text messages (<= 150 per regime) and 300 messages of each foreign set. Arithmetic regimes are digit codes and are not part of this test.

## UG0: primary

| system | false refusal | coverage | accuracy (all) | accuracy (answered) | science_questions refused | stories refused | wiki refused |
|---|---|---|---|---|---|---|---|
| pool flag (built-in, uncalibrated) | 0.000 | 1.000 | 0.589 | 0.589 | 0.000 | 0.000 | 0.000 |
| pool calibrated | 0.056 | 0.944 | 0.589 | 0.606 | 0.458 | 0.205 | 0.368 |
| reader claim (uncalibrated) | 0.328 | 0.672 | 0.589 | 0.657 | 0.941 | 0.862 | 0.846 |
| reader calibrated | 0.053 | 0.947 | 0.589 | 0.600 | 0.174 | 0.294 | 0.130 |
| either calibrated (pool + reader) | 0.073 | 0.927 | 0.589 | 0.609 | 0.445 | 0.330 | 0.344 |
| density calibrated (Mahalanobis) | 0.036 | 0.964 | 0.589 | 0.600 | 0.502 | 0.073 | 0.389 |
| either calibrated (pool + reader + density) | 0.052 | 0.948 | 0.589 | 0.604 | 0.452 | 0.221 | 0.323 |

## UG0: more verified messages (200)

| system | false refusal | coverage | accuracy (all) | accuracy (answered) | science_questions refused | stories refused | wiki refused |
|---|---|---|---|---|---|---|---|
| pool flag (built-in, uncalibrated) | 0.000 | 1.000 | 0.591 | 0.591 | 0.000 | 0.000 | 0.000 |
| pool calibrated | 0.054 | 0.946 | 0.591 | 0.608 | 0.477 | 0.220 | 0.390 |
| reader claim (uncalibrated) | 0.327 | 0.673 | 0.591 | 0.660 | 0.941 | 0.862 | 0.846 |
| reader calibrated | 0.051 | 0.949 | 0.591 | 0.602 | 0.172 | 0.293 | 0.122 |
| either calibrated (pool + reader) | 0.052 | 0.948 | 0.591 | 0.606 | 0.398 | 0.254 | 0.290 |
| density calibrated (Mahalanobis) | 0.041 | 0.959 | 0.591 | 0.603 | 0.551 | 0.085 | 0.429 |
| either calibrated (pool + reader + density) | 0.053 | 0.947 | 0.591 | 0.606 | 0.525 | 0.214 | 0.365 |

## UG0: looser level (q = 0.10)

| system | false refusal | coverage | accuracy (all) | accuracy (answered) | science_questions refused | stories refused | wiki refused |
|---|---|---|---|---|---|---|---|
| pool flag (built-in, uncalibrated) | 0.000 | 1.000 | 0.589 | 0.589 | 0.000 | 0.000 | 0.000 |
| pool calibrated | 0.113 | 0.887 | 0.589 | 0.616 | 0.614 | 0.389 | 0.534 |
| reader claim (uncalibrated) | 0.328 | 0.672 | 0.589 | 0.657 | 0.941 | 0.862 | 0.846 |
| reader calibrated | 0.113 | 0.887 | 0.589 | 0.614 | 0.370 | 0.505 | 0.308 |
| either calibrated (pool + reader) | 0.115 | 0.885 | 0.589 | 0.620 | 0.578 | 0.488 | 0.456 |
| density calibrated (Mahalanobis) | 0.089 | 0.911 | 0.589 | 0.616 | 0.737 | 0.236 | 0.609 |
| either calibrated (pool + reader + density) | 0.107 | 0.893 | 0.589 | 0.617 | 0.706 | 0.416 | 0.534 |

## UG0: AUROC of each reader's novelty score (known text against each foreign set; 0.5 = chance)

| reader | science_questions | stories | wiki |
|---|---|---|---|
| pool | 0.850 | 0.793 | 0.834 |
| asking reader | 0.831 | 0.840 | 0.794 |
| density | 0.931 | 0.808 | 0.892 |

## UPLUS: primary

| system | false refusal | coverage | accuracy (all) | accuracy (answered) | science_questions refused | stories refused | wiki refused |
|---|---|---|---|---|---|---|---|
| pool flag (built-in, uncalibrated) | 0.000 | 1.000 | 0.627 | 0.627 | 0.000 | 0.000 | 0.000 |
| pool calibrated | 0.056 | 0.944 | 0.627 | 0.644 | 0.458 | 0.205 | 0.368 |
| reader claim (uncalibrated) | 0.328 | 0.672 | 0.627 | 0.691 | 0.941 | 0.862 | 0.846 |
| reader calibrated | 0.053 | 0.947 | 0.627 | 0.638 | 0.174 | 0.294 | 0.130 |
| either calibrated (pool + reader) | 0.073 | 0.927 | 0.627 | 0.648 | 0.445 | 0.330 | 0.344 |
| density calibrated (Mahalanobis) | 0.036 | 0.964 | 0.627 | 0.638 | 0.502 | 0.073 | 0.389 |
| either calibrated (pool + reader + density) | 0.052 | 0.948 | 0.627 | 0.642 | 0.452 | 0.221 | 0.323 |

## UPLUS: more verified messages (200)

| system | false refusal | coverage | accuracy (all) | accuracy (answered) | science_questions refused | stories refused | wiki refused |
|---|---|---|---|---|---|---|---|
| pool flag (built-in, uncalibrated) | 0.000 | 1.000 | 0.628 | 0.628 | 0.000 | 0.000 | 0.000 |
| pool calibrated | 0.054 | 0.946 | 0.628 | 0.645 | 0.477 | 0.220 | 0.390 |
| reader claim (uncalibrated) | 0.327 | 0.673 | 0.628 | 0.692 | 0.941 | 0.862 | 0.846 |
| reader calibrated | 0.051 | 0.949 | 0.628 | 0.639 | 0.172 | 0.293 | 0.122 |
| either calibrated (pool + reader) | 0.052 | 0.948 | 0.628 | 0.643 | 0.398 | 0.254 | 0.290 |
| density calibrated (Mahalanobis) | 0.041 | 0.959 | 0.628 | 0.640 | 0.551 | 0.085 | 0.429 |
| either calibrated (pool + reader + density) | 0.053 | 0.947 | 0.628 | 0.643 | 0.525 | 0.214 | 0.365 |

## UPLUS: looser level (q = 0.10)

| system | false refusal | coverage | accuracy (all) | accuracy (answered) | science_questions refused | stories refused | wiki refused |
|---|---|---|---|---|---|---|---|
| pool flag (built-in, uncalibrated) | 0.000 | 1.000 | 0.627 | 0.627 | 0.000 | 0.000 | 0.000 |
| pool calibrated | 0.113 | 0.887 | 0.627 | 0.654 | 0.614 | 0.389 | 0.534 |
| reader claim (uncalibrated) | 0.328 | 0.672 | 0.627 | 0.691 | 0.941 | 0.862 | 0.846 |
| reader calibrated | 0.113 | 0.887 | 0.627 | 0.650 | 0.370 | 0.505 | 0.308 |
| either calibrated (pool + reader) | 0.115 | 0.885 | 0.627 | 0.657 | 0.578 | 0.488 | 0.456 |
| density calibrated (Mahalanobis) | 0.089 | 0.911 | 0.627 | 0.653 | 0.737 | 0.236 | 0.609 |
| either calibrated (pool + reader + density) | 0.107 | 0.893 | 0.627 | 0.656 | 0.706 | 0.416 | 0.534 |

## UPLUS: AUROC of each reader's novelty score (known text against each foreign set; 0.5 = chance)

| reader | science_questions | stories | wiki |
|---|---|---|---|
| pool | 0.850 | 0.793 | 0.834 |
| asking reader | 0.831 | 0.840 | 0.794 |
| density | 0.931 | 0.808 | 0.892 |
