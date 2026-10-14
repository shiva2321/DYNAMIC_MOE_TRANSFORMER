# RV3 dev sensitivity (EXPLORATORY, dev seeds 100-109, not pre-registered)

## baseline (level 0.001, 999 permutations)

| Stream | A_set - A_vote (points) | experts set / vote | known batches called new | doubtful (held) | breaker opens |
|---|---|---|---|---|---|
| F-30 |  -0.48 [-2.62, +1.66] | 30.4 / 30.1 | 4 of 3260 | 0 | 0 |
| S2-30 |  +0.19 [-0.29, +0.67] | 20.3 / 20.6 | 3 of 1690 | 0 | 0 |
| F-15 |  -0.05 [-0.13, +0.04] | 15.3 / 15.2 | 1 of 1760 | 0 | 0 |
| OV-0.85 |  +6.42 [+4.89, +7.95] | 10.3 / 7.1 | 3 of 1260 | 0 | 0 |
| OV-0.6 | +18.12 [+16.60, +19.64] | 10.3 / 5.7 | 1 of 1260 | 0 | 0 |
| OV-0.3 |  -0.41 [-1.15, +0.32] | 10.1 / 10.0 | 1 of 1260 | 0 | 0 |
| OV-0.3-MIXED |  -0.04 [-0.93, +0.85] | 5.9 / 5.5 | 47 of 1350 | 0 | 6 |
| OV-0.0 |  -1.02 [-2.56, +0.52] | 10.3 / 10.1 | 1 of 1260 | 0 | 0 |
| S1 |  -0.60 [-1.66, +0.46] | 4.5 / 4.3 | 0 of 400 | 0 | 0 |
| S2-5 |  +0.00 [+0.00, +0.00] | 3.8 / 3.8 | 0 of 340 | 0 | 0 |
| F-15-SORTED |  -0.53 [-1.52, +0.45] | 26.6 / 20.9 | 129 of 1760 | 0 | 20 |

Statements on these dev runs: S3-1 PASS, S3-2 PARTIAL, S3-3 PASS, S3-4 PASS, S3-5 PASS, S3-6 PASS, S3-7 PARTIAL, S3-8 PASS, S3-9 PASS

3a: 3a is NOT adopted (S3-2 did not pass: kill rule); set_gate stays False. 3b: 3b is NOT adopted (S3-7 PARTIAL); group_vote stays False.

## persist2

| Stream | A_set - A_vote (points) | experts set / vote | known batches called new | doubtful (held) | breaker opens |
|---|---|---|---|---|---|
| F-30 |  -0.17 [-0.54, +0.20] | 30.1 / 30.1 | 0 of 3260 | 33 | 0 |
| S2-30 |  +0.14 [-0.16, +0.44] | 21.1 / 20.6 | 2 of 1690 | 36 | 0 |
| F-15 |  -0.01 [-0.03, +0.02] | 15.1 / 15.2 | 0 of 1760 | 15 | 0 |
| OV-0.85 |  +5.93 [+4.00, +7.86] | 10.2 / 7.1 | 1 of 1260 | 91 | 0 |
| OV-0.6 | +17.17 [+15.74, +18.59] | 10.9 / 5.7 | 0 of 1260 | 84 | 0 |
| OV-0.3 |  -0.14 [-0.73, +0.45] | 10.0 / 10.0 | 0 of 1260 | 3 | 0 |
| OV-0.3-MIXED |  -0.17 [-1.10, +0.76] | 5.9 / 5.5 | 36 of 1350 | 18 | 6 |
| OV-0.0 |  +0.03 [-0.13, +0.18] | 10.1 / 10.1 | 0 of 1260 | 2 | 0 |
| S1 |  -0.04 [-0.74, +0.66] | 4.8 / 4.3 | 0 of 400 | 14 | 1 |
| S2-5 |  +0.00 [+0.00, +0.00] | 3.8 / 3.8 | 0 of 340 | 0 | 0 |
| F-15-SORTED |  -0.22 [-1.23, +0.78] | 22.6 / 20.9 | 98 of 1760 | 51 | 20 |

Statements on these dev runs: S3-1 PASS, S3-2 PASS, S3-3 PASS, S3-4 PARTIAL, S3-5 PASS, S3-6 PASS, S3-7 PARTIAL, S3-8 PASS, S3-9 PASS

3a: 3a is NOT adopted (S3-4 did not pass: kill rule); set_gate stays False. 3b: 3b is NOT adopted (S3-7 PARTIAL); group_vote stays False.

## persist3

| Stream | A_set - A_vote (points) | experts set / vote | known batches called new | doubtful (held) | breaker opens |
|---|---|---|---|---|---|
| F-30 |  -0.17 [-0.54, +0.20] | 30.1 / 30.1 | 0 of 3260 | 33 | 0 |
| S2-30 |  +0.11 [-0.21, +0.43] | 21.1 / 20.6 | 2 of 1690 | 36 | 0 |
| F-15 |  -0.01 [-0.03, +0.02] | 15.1 / 15.2 | 0 of 1760 | 15 | 0 |
| OV-0.85 |  +5.91 [+3.98, +7.84] | 10.2 / 7.1 | 1 of 1260 | 91 | 0 |
| OV-0.6 | +17.26 [+15.58, +18.95] | 10.1 / 5.7 | 0 of 1260 | 84 | 0 |
| OV-0.3 |  -0.14 [-0.73, +0.45] | 10.0 / 10.0 | 0 of 1260 | 3 | 0 |
| OV-0.3-MIXED |  +0.00 [-0.82, +0.82] | 5.9 / 5.5 | 36 of 1350 | 18 | 6 |
| OV-0.0 |  +0.03 [-0.13, +0.18] | 10.1 / 10.1 | 0 of 1260 | 2 | 0 |
| S1 |  -0.05 [-0.74, +0.64] | 4.8 / 4.3 | 0 of 400 | 14 | 1 |
| S2-5 |  +0.00 [+0.00, +0.00] | 3.8 / 3.8 | 0 of 340 | 0 | 0 |
| F-15-SORTED |  -0.22 [-1.23, +0.78] | 22.6 / 20.9 | 98 of 1760 | 51 | 20 |

Statements on these dev runs: S3-1 PASS, S3-2 PASS, S3-3 PASS, S3-4 PASS, S3-5 PASS, S3-6 PASS, S3-7 PARTIAL, S3-8 PASS, S3-9 PASS

3a: adopt 3a: set_gate=True becomes the PhyloMoE default, test-first, with the historical entry points pinned (set_gate=False). 3b: 3b is NOT adopted (S3-7 PARTIAL); group_vote stays False.

## Group veto level, persist2 pools (post-hoc; group answer minus per-input answer, points; mean over seeds)

| veto level | OV-0.85 pure64 | OV-0.6 pure64 | OV-0.85 pure16 | worst mixed 50/50 (stream) | worst mixed 90/10 (stream) | S1 mixed 50/50 |
|---|---|---|---|---|---|---|
| 0.02 | +12.66 | +1.49 | +11.52 | -5.84 (S1) | +0.00 (S2-5) | -5.84 |
| 0.05 | +12.15 | +1.35 | +11.11 | -4.92 (S1) | +0.00 (S2-5) | -4.92 |
| 0.1 | +11.12 | +1.17 | +10.15 | -3.63 (S1) | +0.00 (S2-5) | -3.63 |
| 0.2 | +9.08 | +0.82 | +8.33 | -1.58 (S1) | +0.00 (S2-5) | -1.58 |
| 0.4 | +5.49 | +0.26 | +5.05 | -0.23 (S1) | -0.01 (S2-30) | -0.23 |

## Group veto level, persist3 pools (post-hoc; group answer minus per-input answer, points; mean over seeds)

| veto level | OV-0.85 pure64 | OV-0.6 pure64 | OV-0.85 pure16 | worst mixed 50/50 (stream) | worst mixed 90/10 (stream) | S1 mixed 50/50 |
|---|---|---|---|---|---|---|
| 0.02 | +12.66 | +1.70 | +11.53 | -5.83 (S1) | +0.00 (S2-5) | -5.83 |
| 0.05 | +12.16 | +1.55 | +11.13 | -4.92 (S1) | +0.00 (S2-5) | -4.92 |
| 0.1 | +11.12 | +1.31 | +10.16 | -3.65 (S1) | +0.00 (S2-5) | -3.65 |
| 0.2 | +9.08 | +0.90 | +8.33 | -1.58 (S1) | +0.00 (S2-5) | -1.58 |
| 0.4 | +5.49 | +0.26 | +5.05 | -0.23 (S1) | -0.01 (S2-30) | -0.23 |

