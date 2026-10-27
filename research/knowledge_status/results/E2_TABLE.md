# E2 and E3, status of parts and of tangled messages (pool `dev`, seeds 100-129)

Mean over seeds, with [min - max]; PASS / fail is the mean against the DRAFT bar of spec section 7 (see the last table for the dev-fixed bars). A seed is one teaching assignment: two skills Solid (about 2,000 examples), one Thin (about 20), three never taught. Calibration for the STANDARD condition uses held-out S1 to S3 parts only; the ADAPTED condition (a few S4 to S8 parts in the calibration) is reported next to it and is not gated: if only the adapted condition reaches the bar the claim is 'needs adaptation to new wording'. `R5_<id>` is the learned map on top of `<id>`; `R0_hash` is the word-matcher control.

## B3a, the status machine at q = 0.05 (calibration on S1 to S3 only)

| reader | seeds | macro F1 over solid/thin/unseen (bar 0.85) | false unseen on S4-S8 (bar 0.05) | thin vs solid AUROC (bar 0.85) | Other content called unseen (bar 0.05) | unseen skills found |
|---|---|---|---|---|---|---|
| R5_R4_qwen_L26 | 30 | 0.85 [0.52 - 0.92] PASS | 0.29 [0.17 - 0.43] fail | 0.84 [0.59 - 0.98] fail | 0.25 [0.11 - 0.36] fail | 0.89 [0.18 - 1.00] |
| R4_qwen_L26 | 30 | 0.82 [0.70 - 0.88] fail | 0.39 [0.29 - 0.48] fail | 0.89 [0.76 - 0.97] PASS | 0.21 [0.09 - 0.33] fail | 0.90 [0.60 - 0.98] |
| R0_hash | 30 | 0.61 [0.39 - 0.76] fail | 0.12 [0.04 - 0.24] fail | 0.88 [0.73 - 0.98] PASS | 0.16 [0.00 - 0.27] fail | 0.36 [0.10 - 0.60] |

### The adapted condition at q = 0.05 (reported, not gated) and the wording claim

| reader | false unseen on S4-S8, standard | false unseen on S4-S8, adapted | false unseen on taught wordings S1-S3 | claim from the means (bar 0.05) |
|---|---|---|---|---|
| R5_R4_qwen_L26 | 0.29 [0.17 - 0.43] | 0.17 [0.09 - 0.29] | 0.05 [0.01 - 0.12] | fails |
| R4_qwen_L26 | 0.39 [0.29 - 0.48] | 0.17 [0.07 - 0.30] | 0.05 [0.00 - 0.12] | fails |
| R0_hash | 0.12 [0.04 - 0.24] | 0.02 [0.00 - 0.12] | 0.03 [0.00 - 0.13] | needs adaptation to new wording |

### The curve over q (standard condition)

| reader | q | false unseen on S4-S8 | unseen skills found | Other content called unseen | near-twins of taught skills called unseen | near-twins of untaught skills found |
|---|---|---|---|---|---|---|
| R5_R4_qwen_L26 | 0.01 | 0.12 [0.00 - 0.30] | 0.56 [0.07 - 1.00] | 0.11 [0.00 - 0.29] | 0.01 [0.00 - 0.09] | 0.62 [0.00 - 1.00] |
| R5_R4_qwen_L26 | 0.02 | 0.19 [0.04 - 0.37] | 0.75 [0.10 - 1.00] | 0.19 [0.01 - 0.31] | 0.02 [0.00 - 0.13] | 0.80 [0.09 - 1.00] |
| R5_R4_qwen_L26 | 0.05 | 0.29 [0.17 - 0.43] | 0.89 [0.18 - 1.00] | 0.25 [0.11 - 0.36] | 0.05 [0.00 - 0.16] | 0.93 [0.22 - 1.00] |
| R5_R4_qwen_L26 | 0.1 | 0.36 [0.20 - 0.54] | 0.95 [0.35 - 1.00] | 0.29 [0.17 - 0.43] | 0.09 [0.00 - 0.25] | 0.97 [0.34 - 1.00] |
| R4_qwen_L26 | 0.01 | 0.27 [0.11 - 0.41] | 0.72 [0.07 - 0.97] | 0.05 [0.00 - 0.16] | 0.01 [0.00 - 0.12] | 0.64 [0.00 - 0.90] |
| R4_qwen_L26 | 0.02 | 0.31 [0.24 - 0.42] | 0.81 [0.45 - 0.97] | 0.09 [0.00 - 0.22] | 0.02 [0.00 - 0.12] | 0.75 [0.28 - 0.95] |
| R4_qwen_L26 | 0.05 | 0.39 [0.29 - 0.48] | 0.90 [0.60 - 0.98] | 0.21 [0.09 - 0.33] | 0.04 [0.00 - 0.12] | 0.86 [0.66 - 1.00] |
| R4_qwen_L26 | 0.1 | 0.44 [0.31 - 0.57] | 0.94 [0.81 - 1.00] | 0.33 [0.17 - 0.48] | 0.09 [0.00 - 0.21] | 0.92 [0.77 - 1.00] |
| R0_hash | 0.01 | 0.02 [0.00 - 0.07] | 0.11 [0.00 - 0.31] | 0.01 [0.00 - 0.07] | 0.01 [0.00 - 0.09] | 0.11 [0.00 - 0.37] |
| R0_hash | 0.02 | 0.04 [0.00 - 0.14] | 0.19 [0.01 - 0.40] | 0.04 [0.00 - 0.17] | 0.02 [0.00 - 0.19] | 0.19 [0.00 - 0.49] |
| R0_hash | 0.05 | 0.12 [0.04 - 0.24] | 0.36 [0.10 - 0.60] | 0.16 [0.00 - 0.27] | 0.06 [0.00 - 0.21] | 0.39 [0.00 - 0.78] |
| R0_hash | 0.1 | 0.22 [0.09 - 0.39] | 0.57 [0.20 - 0.81] | 0.32 [0.06 - 0.48] | 0.13 [0.00 - 0.38] | 0.62 [0.15 - 0.92] |

## B3b, the baselines at the same calibration (q = 0.05, standard condition)

Each baseline flags a part as unseen when its novelty score is above the threshold taken from the same held-out calibration parts. AUROC = parts of untaught skills against parts of taught skills, no threshold.

| reader | method | false unseen on S4-S8 | unseen skills found | AUROC taught vs untaught | Other content called unseen | false unseen, adapted |
|---|---|---|---|---|---|---|
| R5_R4_qwen_L26 | **status machine** | 0.29 [0.17 - 0.43] | 0.89 [0.18 - 1.00] | n/a | 0.25 [0.11 - 0.36] | 0.17 [0.09 - 0.29] |
| R5_R4_qwen_L26 | vmf_conformal | 0.31 [0.11 - 0.47] | 0.95 [0.75 - 1.00] | 0.95 [0.88 - 0.99] | 0.23 [0.10 - 0.36] | 0.19 [0.07 - 0.38] |
| R5_R4_qwen_L26 | knn_distance | 0.33 [0.14 - 0.50] | 0.96 [0.73 - 1.00] | 0.94 [0.87 - 0.99] | 0.22 [0.10 - 0.37] | 0.20 [0.07 - 0.35] |
| R5_R4_qwen_L26 | mahalanobis | 0.33 [0.15 - 0.50] | 0.97 [0.79 - 1.00] | 0.95 [0.89 - 0.99] | 0.27 [0.14 - 0.38] | 0.22 [0.11 - 0.39] |
| R5_R4_qwen_L26 | relative_mahalanobis | 0.01 [0.00 - 0.09] | 0.06 [0.00 - 0.46] | 0.44 [0.10 - 0.95] | 0.37 [0.14 - 0.55] | 0.01 [0.00 - 0.09] |
| R5_R4_qwen_L26 | classifier_confidence | 0.19 [0.08 - 0.31] | 0.51 [0.22 - 0.74] | 0.77 [0.57 - 0.88] | 0.06 [0.00 - 0.21] | 0.15 [0.06 - 0.24] |
| R5_R4_qwen_L26 | energy | 0.13 [0.00 - 0.28] | 0.41 [0.00 - 0.85] | 0.74 [0.41 - 0.93] | 0.10 [0.00 - 0.33] | 0.11 [0.00 - 0.23] |
| R5_R4_qwen_L26 | autoencoder_descriptor | 0.28 [0.06 - 0.47] | 0.88 [0.64 - 1.00] | 0.92 [0.79 - 0.98] | 0.17 [0.04 - 0.26] | 0.19 [0.03 - 0.38] |
| R4_qwen_L26 | **status machine** | 0.39 [0.29 - 0.48] | 0.90 [0.60 - 0.98] | n/a | 0.21 [0.09 - 0.33] | 0.17 [0.07 - 0.30] |
| R4_qwen_L26 | vmf_conformal | 0.31 [0.06 - 0.44] | 0.81 [0.41 - 0.95] | 0.89 [0.83 - 0.94] | 0.16 [0.00 - 0.30] | 0.21 [0.06 - 0.38] |
| R4_qwen_L26 | knn_distance | 0.41 [0.30 - 0.49] | 0.93 [0.86 - 0.99] | 0.90 [0.85 - 0.94] | 0.16 [0.00 - 0.28] | 0.25 [0.15 - 0.38] |
| R4_qwen_L26 | mahalanobis | 0.39 [0.29 - 0.47] | 0.92 [0.82 - 0.97] | 0.88 [0.82 - 0.93] | 0.25 [0.06 - 0.37] | 0.27 [0.16 - 0.35] |
| R4_qwen_L26 | relative_mahalanobis | 0.00 [0.00 - 0.01] | 0.00 [0.00 - 0.00] | 0.72 [0.45 - 0.90] | 0.28 [0.13 - 0.39] | 0.00 [0.00 - 0.01] |
| R4_qwen_L26 | classifier_confidence | 0.30 [0.17 - 0.45] | 0.91 [0.73 - 1.00] | 0.91 [0.84 - 0.97] | 0.21 [0.05 - 0.37] | 0.19 [0.10 - 0.31] |
| R4_qwen_L26 | energy | 0.24 [0.06 - 0.41] | 0.87 [0.60 - 1.00] | 0.92 [0.85 - 0.98] | 0.22 [0.08 - 0.38] | 0.17 [0.04 - 0.28] |
| R4_qwen_L26 | autoencoder_descriptor | 0.27 [0.15 - 0.46] | 0.85 [0.65 - 0.92] | 0.89 [0.85 - 0.93] | 0.06 [0.00 - 0.15] | 0.18 [0.09 - 0.34] |
| R0_hash | **status machine** | 0.12 [0.04 - 0.24] | 0.36 [0.10 - 0.60] | n/a | 0.16 [0.00 - 0.27] | 0.02 [0.00 - 0.12] |
| R0_hash | vmf_conformal | 0.12 [0.03 - 0.30] | 0.35 [0.15 - 0.56] | 0.82 [0.76 - 0.88] | 0.22 [0.05 - 0.47] | 0.09 [0.01 - 0.17] |
| R0_hash | knn_distance | 0.20 [0.05 - 0.32] | 0.35 [0.13 - 0.45] | 0.81 [0.75 - 0.87] | 0.46 [0.25 - 0.61] | 0.13 [0.04 - 0.22] |
| R0_hash | mahalanobis | 0.16 [0.07 - 0.30] | 0.24 [0.10 - 0.42] | 0.77 [0.69 - 0.85] | 0.38 [0.21 - 0.52] | 0.11 [0.03 - 0.17] |
| R0_hash | relative_mahalanobis | 0.10 [0.03 - 0.19] | 0.11 [0.05 - 0.20] | 0.77 [0.69 - 0.84] | 0.36 [0.16 - 0.53] | 0.08 [0.03 - 0.15] |
| R0_hash | classifier_confidence | 0.10 [0.03 - 0.31] | 0.25 [0.13 - 0.47] | 0.71 [0.61 - 0.81] | 0.04 [0.00 - 0.14] | 0.09 [0.03 - 0.20] |
| R0_hash | energy | 0.08 [0.01 - 0.29] | 0.31 [0.16 - 0.51] | 0.72 [0.62 - 0.83] | 0.07 [0.01 - 0.12] | 0.07 [0.01 - 0.25] |
| R0_hash | autoencoder_descriptor | 0.09 [0.03 - 0.15] | 0.19 [0.04 - 0.42] | 0.71 [0.60 - 0.86] | 0.00 [0.00 - 0.03] | 0.08 [0.03 - 0.15] |

## E3, tangled messages (status machine, q = 0.05)

B4a: a known part with an unseen part; B4b: a known part with Other content; B4c: an unseen part with Other content. B6: B4a under automatic segmentation must stay within 0.10 of the gold-span number.

| reader | segmentation | B4a leftover is the unseen part (bar 0.85) | B4a known part not flagged (bar 0.95) | B4b nothing flagged (bar 0.95) | B4b Other labelled (bar 0.95) | B4c unseen found | B6 within 0.10 |
|---|---|---|---|---|---|---|---|
| R5_R4_qwen_L26 | gold | 0.66 [0.15 - 0.85] fail | 0.67 [0.50 - 0.85] fail | 0.64 [0.47 - 0.78] fail | 0.97 [0.90 - 1.00] PASS | 0.86 [0.12 - 1.00] | n/a |
| R5_R4_qwen_L26 | auto_sentence | 0.49 [0.12 - 0.68] | 0.58 [0.35 - 0.88] | 0.53 [0.25 - 0.72] | 0.93 [0.85 - 1.00] | 0.84 [0.15 - 1.00] | 0.13 of seeds |
| R5_R4_qwen_L26 | auto_window | 0.37 [0.15 - 0.55] | 0.53 [0.25 - 0.82] | 0.23 [0.10 - 0.60] | 0.57 [0.28 - 0.80] | 0.85 [0.23 - 1.00] | 0.07 of seeds |
| R4_qwen_L26 | gold | 0.67 [0.47 - 0.80] fail | 0.51 [0.28 - 0.72] fail | 0.45 [0.20 - 0.60] fail | 0.91 [0.68 - 1.00] fail | 0.96 [0.68 - 1.00] | n/a |
| R4_qwen_L26 | auto_sentence | 0.47 [0.33 - 0.68] | 0.45 [0.28 - 0.65] | 0.32 [0.10 - 0.55] | 0.88 [0.60 - 0.97] | 0.94 [0.78 - 1.00] | 0.03 of seeds |
| R4_qwen_L26 | auto_window | 0.36 [0.23 - 0.50] | 0.42 [0.15 - 0.62] | 0.08 [0.00 - 0.25] | 0.51 [0.23 - 0.70] | 0.94 [0.78 - 1.00] | 0.03 of seeds |
| R0_hash | gold | 0.35 [0.07 - 0.55] fail | 0.92 [0.80 - 1.00] fail | 0.82 [0.62 - 0.97] fail | 0.92 [0.78 - 1.00] fail | 0.37 [0.12 - 0.78] | n/a |
| R0_hash | auto_sentence | 0.27 [0.10 - 0.50] | 0.83 [0.60 - 1.00] | 0.69 [0.38 - 0.93] | 0.92 [0.75 - 1.00] | 0.39 [0.10 - 0.70] | 0.50 of seeds |
| R0_hash | auto_window | 0.24 [0.03 - 0.45] | 0.75 [0.55 - 0.95] | 0.36 [0.12 - 0.75] | 0.57 [0.30 - 0.88] | 0.47 [0.15 - 0.82] | 0.10 of seeds |

Baselines on the tangled messages (gold parts; they cannot label Other content, so only the flagging numbers are shown):

| reader | method | B4a leftover is the unseen part | B4a known part not flagged | B4b nothing flagged |
|---|---|---|---|---|
| R5_R4_qwen_L26 | vmf_conformal | 0.71 [0.53 - 0.88] | 0.66 [0.38 - 0.90] | 0.63 [0.40 - 0.93] |
| R5_R4_qwen_L26 | knn_distance | 0.73 [0.55 - 0.90] | 0.66 [0.40 - 0.85] | 0.62 [0.42 - 0.88] |
| R5_R4_qwen_L26 | mahalanobis | 0.74 [0.55 - 0.90] | 0.66 [0.40 - 0.85] | 0.61 [0.38 - 0.85] |
| R5_R4_qwen_L26 | relative_mahalanobis | 0.07 [0.00 - 0.80] | 1.00 [1.00 - 1.00] | 0.70 [0.47 - 0.85] |
| R5_R4_qwen_L26 | classifier_confidence | 0.47 [0.23 - 0.78] | 0.83 [0.57 - 1.00] | 0.78 [0.53 - 0.97] |
| R5_R4_qwen_L26 | energy | 0.40 [0.00 - 0.82] | 0.88 [0.62 - 1.00] | 0.80 [0.57 - 0.97] |
| R5_R4_qwen_L26 | autoencoder_descriptor | 0.69 [0.40 - 0.95] | 0.70 [0.45 - 1.00] | 0.68 [0.38 - 0.97] |
| R4_qwen_L26 | vmf_conformal | 0.67 [0.47 - 0.80] | 0.62 [0.38 - 1.00] | 0.56 [0.33 - 0.80] |
| R4_qwen_L26 | knn_distance | 0.68 [0.57 - 0.88] | 0.50 [0.30 - 0.75] | 0.45 [0.25 - 0.68] |
| R4_qwen_L26 | mahalanobis | 0.70 [0.55 - 0.88] | 0.55 [0.38 - 0.78] | 0.42 [0.12 - 0.65] |
| R4_qwen_L26 | relative_mahalanobis | 0.00 [0.00 - 0.00] | 1.00 [1.00 - 1.00] | 0.68 [0.45 - 0.88] |
| R4_qwen_L26 | classifier_confidence | 0.74 [0.55 - 0.88] | 0.73 [0.53 - 0.90] | 0.70 [0.47 - 0.93] |
| R4_qwen_L26 | energy | 0.77 [0.47 - 0.93] | 0.81 [0.53 - 0.95] | 0.77 [0.50 - 0.97] |
| R4_qwen_L26 | autoencoder_descriptor | 0.70 [0.45 - 0.90] | 0.69 [0.53 - 0.90] | 0.69 [0.50 - 0.82] |
| R0_hash | vmf_conformal | 0.33 [0.12 - 0.55] | 0.90 [0.65 - 1.00] | 0.81 [0.55 - 0.95] |
| R0_hash | knn_distance | 0.39 [0.17 - 0.53] | 0.81 [0.65 - 0.95] | 0.59 [0.28 - 0.90] |
| R0_hash | mahalanobis | 0.26 [0.07 - 0.50] | 0.86 [0.60 - 1.00] | 0.63 [0.38 - 0.85] |
| R0_hash | relative_mahalanobis | 0.13 [0.00 - 0.28] | 0.91 [0.72 - 1.00] | 0.62 [0.30 - 0.80] |
| R0_hash | classifier_confidence | 0.19 [0.03 - 0.38] | 0.91 [0.70 - 0.97] | 0.89 [0.75 - 0.95] |
| R0_hash | energy | 0.20 [0.05 - 0.38] | 0.91 [0.72 - 1.00] | 0.92 [0.85 - 1.00] |
| R0_hash | autoencoder_descriptor | 0.19 [0.05 - 0.53] | 0.92 [0.72 - 1.00] | 0.92 [0.80 - 1.00] |

## Group tests (batches of parts of one skill judged together, q = 0.05)

| reader | method | batch size | false unseen (taught skills) | unseen skills found |
|---|---|---|---|---|
| R5_R4_qwen_L26 | status machine (every group must flag) | 1 | 0.14 [0.00 - 0.50] | 0.45 [0.00 - 1.00] |
| R5_R4_qwen_L26 | status machine (every group must flag) | 4 | 0.54 [0.30 - 0.85] | 0.98 [0.70 - 1.00] |
| R5_R4_qwen_L26 | status machine (every group must flag) | 8 | 0.70 [0.40 - 0.95] | 0.98 [0.80 - 1.00] |
| R5_R4_qwen_L26 | status machine (every group must flag) | 16 | 0.78 [0.40 - 1.00] | 0.99 [0.85 - 1.00] |
| R5_R4_qwen_L26 | group test, nearest group only | 1 | 0.29 [0.05 - 0.65] | 0.85 [0.30 - 1.00] |
| R5_R4_qwen_L26 | group test, nearest group only | 4 | 0.56 [0.30 - 0.85] | 0.97 [0.70 - 1.00] |
| R5_R4_qwen_L26 | group test, nearest group only | 8 | 0.70 [0.50 - 0.90] | 0.99 [0.85 - 1.00] |
| R5_R4_qwen_L26 | group test, nearest group only | 16 | 0.79 [0.40 - 1.00] | 0.99 [0.85 - 1.00] |
| R4_qwen_L26 | status machine (every group must flag) | 1 | 0.02 [0.00 - 0.10] | 0.10 [0.00 - 0.60] |
| R4_qwen_L26 | status machine (every group must flag) | 4 | 0.25 [0.10 - 0.50] | 0.98 [0.70 - 1.00] |
| R4_qwen_L26 | status machine (every group must flag) | 8 | 0.38 [0.20 - 0.65] | 1.00 [1.00 - 1.00] |
| R4_qwen_L26 | status machine (every group must flag) | 16 | 0.54 [0.20 - 0.95] | 1.00 [1.00 - 1.00] |
| R4_qwen_L26 | group test, nearest group only | 1 | 0.09 [0.00 - 0.30] | 0.45 [0.05 - 0.70] |
| R4_qwen_L26 | group test, nearest group only | 4 | 0.27 [0.05 - 0.45] | 0.98 [0.70 - 1.00] |
| R4_qwen_L26 | group test, nearest group only | 8 | 0.41 [0.25 - 0.75] | 1.00 [1.00 - 1.00] |
| R4_qwen_L26 | group test, nearest group only | 16 | 0.55 [0.20 - 0.95] | 1.00 [1.00 - 1.00] |
| R0_hash | status machine (every group must flag) | 1 | 0.00 [0.00 - 0.05] | 0.00 [0.00 - 0.05] |
| R0_hash | status machine (every group must flag) | 4 | 0.15 [0.00 - 0.55] | 0.52 [0.20 - 0.85] |
| R0_hash | status machine (every group must flag) | 8 | 0.32 [0.10 - 0.70] | 0.93 [0.75 - 1.00] |
| R0_hash | status machine (every group must flag) | 16 | 0.54 [0.10 - 1.00] | 1.00 [1.00 - 1.00] |
| R0_hash | group test, nearest group only | 1 | 0.07 [0.00 - 0.25] | 0.16 [0.00 - 0.35] |
| R0_hash | group test, nearest group only | 4 | 0.20 [0.05 - 0.55] | 0.69 [0.35 - 1.00] |
| R0_hash | group test, nearest group only | 8 | 0.32 [0.05 - 0.70] | 0.97 [0.80 - 1.00] |
| R0_hash | group test, nearest group only | 16 | 0.57 [0.10 - 1.00] | 1.00 [1.00 - 1.00] |

## Misassignment between skills (which taught group a part's nearest prototype belongs to; rows = true skill, summed over seeds)

`R5_R4_qwen_L26`

| true skill | add | sub | mul | div | mod | max | other |
|---|---|---|---|---|---|---|---|
| add | 1118 | 426 | 454 | 73 | 68 | 289 | 8 |
| sub | 388 | 1173 | 322 | 176 | 217 | 100 | 66 |
| mul | 462 | 164 | 1198 | 290 | 263 | 62 | 24 |
| div | 47 | 361 | 114 | 1241 | 588 | 27 | 95 |
| mod | 90 | 210 | 173 | 733 | 1019 | 62 | 199 |
| max | 247 | 23 | 33 | 299 | 266 | 1176 | 56 |

`R4_qwen_L26`

| true skill | add | sub | mul | div | mod | max | other |
|---|---|---|---|---|---|---|---|
| add | 1103 | 463 | 575 | 38 | 60 | 172 | 25 |
| sub | 449 | 1084 | 357 | 256 | 57 | 114 | 125 |
| mul | 481 | 227 | 1167 | 278 | 160 | 109 | 41 |
| div | 32 | 277 | 144 | 1209 | 605 | 41 | 165 |
| mod | 178 | 153 | 201 | 596 | 1014 | 81 | 263 |
| max | 131 | 33 | 107 | 202 | 279 | 1180 | 168 |

`R0_hash`

| true skill | add | sub | mul | div | mod | max | other |
|---|---|---|---|---|---|---|---|
| add | 875 | 248 | 481 | 148 | 209 | 122 | 353 |
| sub | 272 | 949 | 364 | 125 | 195 | 81 | 456 |
| mul | 413 | 278 | 993 | 195 | 192 | 50 | 342 |
| div | 255 | 238 | 186 | 1043 | 284 | 34 | 433 |
| mod | 307 | 245 | 355 | 300 | 759 | 41 | 479 |
| max | 159 | 92 | 191 | 103 | 111 | 973 | 471 |

## Offline consequence (an ESTIMATE, not gated)

- `R5_R4_qwen_L26`: per seed 28.2 false unseen and 22.3 missed unseen parts; OFFLINE ESTIMATE of the cost of the status errors: 269 pool points = 28.2 false_unseen x 4 + 22.2667 missed_unseen x 7. An estimate from recorded costs (E-034, E-030), not measured online and not gated; whether statuses drive learning well belongs to Phase 2.
- `R4_qwen_L26`: per seed 38.0 false unseen and 21.8 missed unseen parts; OFFLINE ESTIMATE of the cost of the status errors: 305 pool points = 38 false_unseen x 4 + 21.8 missed_unseen x 7. An estimate from recorded costs (E-034, E-030), not measured online and not gated; whether statuses drive learning well belongs to Phase 2.
- `R0_hash`: per seed 11.6 false unseen and 135.3 missed unseen parts; OFFLINE ESTIMATE of the cost of the status errors: 994 pool points = 11.6333 false_unseen x 4 + 135.3 missed_unseen x 7. An estimate from recorded costs (E-034, E-030), not measured online and not gated; whether statuses drive learning well belongs to Phase 2.

## Bars: draft and dev-fixed

Rule written before the dev run: a bar to stay above becomes max(draft, mean - 2 sd over the dev seeds), a bar to stay below becomes min(draft, mean + 2 sd); a dev result can tighten a draft bar and never loosen it. The dev-fixed bars are the ones the freeze records. Source of the dev values: the first reader above, status machine, q = 0.05, standard condition.

| measure | draft bar | dev mean | dev-fixed bar |
|---|---|---|---|
| macro_f1 (at least) | 0.85 | 0.855 | 0.850 |
| false_unseen_heldout (at most) | 0.05 | 0.289 | 0.050 |
| thin_vs_solid_auroc (at least) | 0.85 | 0.844 | 0.850 |
| other_as_unseen (at most) | 0.05 | 0.249 | 0.050 |
| b4a_residual_hit (at least) | 0.85 | 0.660 | 0.850 |
| b4a_known_not_flagged (at least) | 0.95 | 0.672 | 0.950 |
| b4b_nothing_flagged (at least) | 0.95 | 0.643 | 0.950 |
| b4b_other_labelled (at least) | 0.95 | 0.968 | 0.950 |
