# E1, the information-ceiling table (pool `dev`, seeds 100-104)

Mean over seeds, with [min - max]. Draft bars: B1a >= 0.95 (chance 0.14), B1b >= 0.95 (chance 0.50), B1c >= 0.90 (chance 0.50). Centred = the mean of the teaching embeddings is subtracted first (the primary value); raw = cosine on the vectors as they are.

| representation | seeds | B1a probe (whole message) | B1a probe (gold span) | B1b near-twin AUROC (centred) | B1b raw | B1c taught-vs-untaught AUROC (centred) | B1c raw |
|---|---|---|---|---|---|---|---|
| R0_hash | 5 | 0.54 [0.53 - 0.56] fail | 0.72 [0.70 - 0.74] | 0.00 [0.00 - 0.01] fail | 0.01 [0.01 - 0.01] | 0.68 [0.67 - 0.68] fail | 0.66 [0.65 - 0.67] |
| R1_minilm384 | 5 | 0.81 [0.79 - 0.82] fail | 0.81 [0.80 - 0.83] | 0.03 [0.02 - 0.04] fail | 0.04 [0.03 - 0.05] | 0.74 [0.73 - 0.74] fail | 0.69 [0.68 - 0.70] |
| R1_minilm128 | 5 | 0.73 [0.70 - 0.75] fail | 0.78 [0.76 - 0.81] | 0.04 [0.03 - 0.05] fail | 0.05 [0.04 - 0.06] | 0.69 [0.68 - 0.70] fail | 0.67 [0.66 - 0.68] |
| R2_mpnet | 5 | 0.80 [0.79 - 0.81] fail | 0.81 [0.79 - 0.84] | 0.10 [0.08 - 0.12] fail | 0.12 [0.10 - 0.13] | 0.74 [0.73 - 0.75] fail | 0.71 [0.69 - 0.72] |
| R3_bert | 5 | 0.62 [0.60 - 0.64] fail | 0.74 [0.73 - 0.75] | 0.00 [0.00 - 0.00] fail | 0.02 [0.01 - 0.03] | 0.62 [0.60 - 0.63] fail | 0.56 [0.55 - 0.57] |
| R4_gpt2_L4 | 5 | 0.52 [0.50 - 0.54] fail | 0.70 [0.68 - 0.72] | 0.00 [0.00 - 0.01] fail | 0.00 [0.00 - 0.00] | 0.53 [0.52 - 0.55] fail | 0.51 [0.51 - 0.52] |
| R4_gpt2_L6 | 5 | 0.53 [0.51 - 0.54] fail | 0.69 [0.67 - 0.71] | 0.01 [0.00 - 0.01] fail | 0.00 [0.00 - 0.00] | 0.53 [0.51 - 0.55] fail | 0.51 [0.50 - 0.51] |
| R4_gpt2_L8 | 5 | 0.57 [0.55 - 0.59] fail | 0.70 [0.67 - 0.72] | 0.01 [0.01 - 0.02] fail | 0.01 [0.00 - 0.01] | 0.53 [0.51 - 0.55] fail | 0.51 [0.51 - 0.52] |
| R4_qwen_L9 | 5 | 0.67 [0.65 - 0.70] fail | 0.76 [0.72 - 0.78] | 0.01 [0.01 - 0.01] fail | 0.01 [0.01 - 0.02] | 0.55 [0.53 - 0.56] fail | 0.52 [0.50 - 0.53] |
| R4_qwen_L14 | 5 | 0.74 [0.72 - 0.74] fail | 0.83 [0.83 - 0.85] | 0.01 [0.01 - 0.01] fail | 0.02 [0.02 - 0.03] | 0.58 [0.57 - 0.60] fail | 0.54 [0.53 - 0.55] |
| R4_qwen_L18 | 5 | 0.85 [0.82 - 0.87] fail | 0.85 [0.80 - 0.88] | 0.04 [0.03 - 0.04] fail | 0.06 [0.05 - 0.06] | 0.61 [0.60 - 0.62] fail | 0.57 [0.56 - 0.58] |

**B2 validity of the test bed:** R0 is below the bar by +0.41 on B1a and +0.95 on B1b (a gap of 0.15 or more on either is required): test bed VALID
