# E-026 - Learning arithmetic from examples: GrowMoE against dense, transformer, lookup and told-the-operator experts

- **ID:** E-026
- **Date:** 2026-10-14
- **Track:** G
- **Type:** experiment
- **Status:** completed (72 runs; exploratory)
- **Evidence grade:** B (script-generated, reproducible, 2 seeds for the main setting and 1-2 for the length regimes; NOT pre-registered; the pilot informed the setup)
- **Code / commit:** `growmoe/arith.py`, `tests/growmoe/test_gm_arith.py`, `research/growmoe_results/arith/` (uncommitted, track-g-growmoe working tree; product code = stage 3 with amendments A-15 and A-16)
- **People:** Claude designed, built, ran and wrote it (the user asked for the experiment; Gemini was idle)

## Question
Can GrowMoE and its competitors LEARN +, -, *, mod, integer division and max on numbers of 1 to 5 digits from examples alone, with no arithmetic algorithm, table or helper given to anyone? Do they generalise to problems they never saw, to lengths they never saw, are the routing decisions of GrowMoE to specialised experts consistent, and what else happens?

## Why we asked it
The user wants to know what the system really does on a task where lookup cannot work and old hand-coded arithmetic shortcuts (earlier ISHMN work) were removed on purpose. It is also a test of the claim "experts for different regimes" on regimes that are operators.

## Prediction (written BEFORE the numbers)
None pre-registered (exploratory). Stated expectations after the pilot and before the main runs: dense learners generalise addition, subtraction and max but not multiplication or real division; nothing extrapolates to longer numbers; a mixed (i.i.d.) stream gives GrowMoE no reason to grow; the batch-level gate (stage 3) should see the operator switches when the operators arrive one by one.

## Method (how)
Data format (`growmoe/arith.py`, 12 unit tests, answers checked against plain Python): two operands of 1-5 digits as 5 right-aligned one-hot digit slots (PAD for unused leading slots), operator one-hot, output slot one-hot (127 numbers padded to 128); label = the symbol in that output slot (0-9, PAD, NEG); a problem is right only if every output slot is right (exact match). The number of answer slots an operator can need (add 6, sub 5 + sign, mul 10, mod/div/max 5) is part of the FORMAT given to everyone. Held-out problems are a hash split (10 % of all problems of every length can never be trained on). Training: 8,000 problems per operator (48,000 problems, 298,000 examples). Test: 200 held-out problems per (operator, length of A, length of B), identical for every learner, regime and seed, plus a sample of each learner's own training problems ("seen").
Learners: dense MLP small (0.1 M parameters) and large (2.2 M), a 3-layer transformer (0.6 M, decoder only, greedy decoding, most significant digit first, no reversal trick, no scratchpad), kNN, experts told the operator (per operator, per operator and slot; references), and GrowMoE (`PhyloMoE`) as shipped and tuned over expert width, replay size and steps per batch (`h512, s5, r20000`; 20 steps was worse than 5), with and without the stage-3 gate; continual baselines (plain network, replay network, MLP with a 20,000-example replay buffer). Protocols: iid (everything mixed) and seq (operators arrive one block after another: max, add, sub, mod, div, mul). Training-length regimes: FULL (1-5 digits), HOLE (no operand of 3 digits), SHORT (1-3 digits only). Extra: the output slots of one problem read as a group (stage-3b vote), a data-size sweep, per-slot accuracy, carry chains.

## Results (what came out; tables copied from `research/growmoe_results/arith/RESULTS_ARITH.md`)
**Unseen problems, training lengths 1-5 (exact match %, mean over seeds):**

| learner | seeds | seen | add | sub | mul | mod | div | max | mod hard | div hard | mean | digit acc | params | stored vecs | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GrowMoE | 2 | 28.7 | 13.1 | 7.9 | 3.5 | 41.2 | 49.5 | 47.1 | 12.5 | 5.6 | 27.1 | 71.0 | 93,036 | 512 | 196 |
| GrowMoE-h512-s5-r20000 | 2 | 40.6 | 29.8 | 20.0 | 5.7 | 46.8 | 51.5 | 66.3 | 14.9 | 6.8 | 36.7 | 76.4 | 133,772 | 20,256 | 155 |
| MLP-L | 2 | 100.0 | 83.5 | 68.3 | 17.5 | 57.9 | 60.2 | 95.1 | 18.5 | 19.0 | 63.8 | 87.3 | 2,243,596 | 0 | 111 |
| MLP-S | 2 | 77.0 | 60.3 | 43.5 | 11.6 | 53.0 | 57.6 | 83.5 | 17.2 | 16.0 | 51.6 | 83.1 | 101,900 | 0 | 35 |
| Oracle-op | 2 | 100.0 | 75.4 | 51.7 | 11.4 | 54.5 | 59.7 | 86.7 | 16.1 | 20.8 | 56.6 | 84.7 | 2,009,160 | 0 | 99 |
| Oracle-oppos | 2 | 100.0 | 85.0 | 62.7 | 12.0 | 55.4 | 58.2 | 88.0 | 16.8 | 19.4 | 60.2 | 85.8 | 12,389,820 | 0 | 102 |
| Transformer | 2 | 74.3 | 78.9 | 88.9 | 21.0 | 28.4 | 40.8 | 91.8 | 23.6 | 29.9 | 58.3 | 85.5 | 602,124 | 0 | 520 |
| kNN | 2 | 100.0 | 0.2 | 0.8 | 1.5 | 14.7 | 25.1 | 7.2 | 9.0 | 4.6 | 8.2 | 54.3 | 0 | 296,000 | 0 |

**mod and div split by case** (a < b: the answer is a or 0; only a >= b with b >= 2 needs real division):

| learner | mod: a<b | mod: b=1 | mod: a>=b, b>=2 | div: a<b | div: b=1 | div: a>=b, b>=2 |
|---|---|---|---|---|---|---|
| GrowMoE | 68.5 | 67.3 | 12.5 | 94.1 | 37.3 | 5.6 |
| GrowMoE-h512-s5-r20000 | 78.0 | 60.6 | 14.9 | 95.5 | 57.2 | 6.8 |
| MLP-L | 94.9 | 100.0 | 18.5 | 98.6 | 97.9 | 19.0 |
| MLP-S | 86.3 | 98.4 | 17.2 | 97.8 | 75.6 | 16.0 |
| Oracle-op | 90.5 | 99.6 | 16.1 | 98.2 | 62.3 | 20.8 |
| Oracle-oppos | 91.6 | 100.0 | 16.8 | 98.1 | 38.3 | 19.4 |
| Transformer | 29.5 | 100.0 | 23.6 | 48.0 | 93.7 | 29.9 |
| kNN | 19.8 | 23.6 | 9.0 | 47.0 | 2.1 | 4.6 |

**Length change** (HOLE = no 3-digit operand in training, SHORT = trained on 1-3 digits only; "new lengths" = problems whose lengths were not in training):

| learner | protocol | regime | in-training lengths | new lengths | new: add | sub | mul | mod | div | max | digit acc on new lengths |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GrowMoE-h512-s5-r20000 | iid | FULL (2 seeds) | 36.7 | - | - | - | - | - | - | - | - |
| GrowMoE-h512-s5-r20000 | iid | HOLE (1 seeds) | 39.8 | 26.0 | 18.3 | 10.8 | 2.6 | 31.9 | 44.5 | 47.8 | 72.4 |
| GrowMoE-h512-s5-r20000 | iid | SHORT (1 seeds) | 52.5 | 12.9 | 0.0 | 0.0 | 2.2 | 32.7 | 42.4 | 0.0 | 54.8 |
| GrowMoE-h512-s5-r20000 | seq | FULL (2 seeds) | 32.8 | - | - | - | - | - | - | - | - |
| GrowMoE-h512-s5-r20000+s3 | seq | FULL (2 seeds) | 26.9 | - | - | - | - | - | - | - | - |
| GrowMoE-h512-s5-r20000+s3 | seq | HOLE (1 seeds) | 29.3 | 19.5 | 10.3 | 9.3 | 2.4 | 29.7 | 42.3 | 23.1 | 67.1 |
| GrowMoE-h512-s5-r20000+s3 | seq | SHORT (1 seeds) | 48.6 | 8.6 | 0.2 | 0.1 | 2.4 | 21.5 | 27.7 | 0.0 | 51.3 |
| MLP-L | iid | FULL (2 seeds) | 63.8 | - | - | - | - | - | - | - | - |
| MLP-L | iid | HOLE (2 seeds) | 66.0 | 43.9 | 49.1 | 34.3 | 6.8 | 43.4 | 49.1 | 80.6 | 80.1 |
| MLP-L | iid | SHORT (2 seeds) | 78.2 | 13.8 | 0.0 | 0.2 | 2.7 | 39.2 | 40.7 | 0.0 | 62.2 |
| Oracle-op | iid | FULL (2 seeds) | 56.6 | - | - | - | - | - | - | - | - |
| Oracle-op | iid | HOLE (2 seeds) | 59.0 | 37.6 | 37.7 | 24.3 | 3.4 | 38.6 | 50.8 | 70.9 | 78.5 |
| Oracle-op | iid | SHORT (2 seeds) | 75.4 | 13.0 | 0.0 | 0.1 | 2.7 | 38.0 | 37.3 | 0.0 | 61.8 |
| Oracle-op | seq | FULL (2 seeds) | 56.4 | - | - | - | - | - | - | - | - |
| Oracle-op | seq | HOLE (2 seeds) | 59.3 | 38.0 | 41.1 | 24.8 | 2.9 | 38.3 | 50.3 | 70.8 | 78.7 |
| Oracle-op | seq | SHORT (2 seeds) | 74.9 | 13.0 | 0.0 | 0.1 | 2.7 | 38.0 | 36.9 | 0.0 | 61.7 |
| Transformer | iid | FULL (2 seeds) | 58.3 | - | - | - | - | - | - | - | - |
| Transformer | iid | HOLE (2 seeds) | 63.8 | 52.2 | 67.9 | 67.7 | 8.7 | 44.0 | 47.5 | 77.5 | 81.7 |
| Transformer | iid | SHORT (2 seeds) | 77.6 | 12.6 | 0.0 | 0.3 | 1.6 | 35.4 | 38.2 | 0.0 | 60.1 |
| kNN | iid | FULL (2 seeds) | 8.2 | - | - | - | - | - | - | - | - |
| kNN | iid | HOLE (2 seeds) | 7.7 | 6.6 | 0.0 | 0.1 | 1.3 | 12.4 | 23.9 | 2.1 | 55.2 |
| kNN | iid | SHORT (2 seeds) | 9.9 | 7.2 | 0.0 | 0.0 | 1.8 | 17.5 | 23.8 | 0.0 | 45.9 |

**What the missing lengths cost (like for like).** The two columns above are not comparable by themselves (a 4-5 digit sum is harder than a 3-digit one). The fair comparison is the same learner on the same test cells, trained with those lengths (the FULL run) and without them (HOLE / SHORT run; the same 8,000 problems per operator, drawn only from the allowed lengths):

| learner | protocol | regime (seeds without / with) | six operators: with | without | gap | add: with | without | sub: with | without | max: with | without |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GrowMoE-h512-s5-r20000 | iid | HOLE (1 / 2) | 33.4 | 26.0 | 7.4 | 22.9 | 18.3 | 14.8 | 10.8 | 63.4 | 47.8 |
| GrowMoE-h512-s5-r20000 | iid | SHORT (1 / 2) | 31.7 | 12.9 | 18.8 | 19.5 | 0.0 | 13.5 | 0.0 | 62.1 | 0.0 |
| GrowMoE-h512-s5-r20000+s3 | seq | HOLE (1 / 2) | 25.1 | 19.5 | 5.6 | 15.7 | 10.3 | 12.2 | 9.3 | 29.4 | 23.1 |
| GrowMoE-h512-s5-r20000+s3 | seq | SHORT (1 / 2) | 21.0 | 8.6 | 12.3 | 10.0 | 0.2 | 9.6 | 0.1 | 22.2 | 0.0 |
| MLP-L | iid | HOLE (2 / 2) | 62.4 | 43.9 | 18.6 | 85.8 | 49.1 | 64.8 | 34.3 | 97.6 | 80.6 |
| MLP-L | iid | SHORT (2 / 2) | 57.6 | 13.8 | 43.8 | 76.2 | 0.0 | 57.6 | 0.2 | 94.1 | 0.0 |
| Oracle-oppos | iid | HOLE (2 / 2) | 60.3 | 38.0 | 22.3 | 87.8 | 37.8 | 61.5 | 29.5 | 93.0 | 72.6 |
| Oracle-oppos | iid | SHORT (2 / 2) | 54.9 | 13.5 | 41.4 | 79.3 | 0.0 | 54.1 | 0.1 | 85.8 | 0.0 |
| Transformer | iid | HOLE (2 / 2) | 57.0 | 52.2 | 4.8 | 81.1 | 67.9 | 92.9 | 67.7 | 91.8 | 77.5 |
| Transformer | iid | SHORT (2 / 2) | 52.0 | 12.6 | 39.4 | 69.6 | 0.0 | 83.4 | 0.3 | 89.4 | 0.0 |

**GrowMoE routing** (NMI = what the expert says about the operator / slot / length; purity of the operator):

| learner | protocol | regime | seed | experts | used on test | NMI operator | NMI slot | NMI length | purity (operator) | abstain (unknown) % |
|---|---|---|---|---|---|---|---|---|---|---|
| GrowMoE | iid | FULL | 0 | 1 | 1 | 0.00 | 0.00 | 0.00 | 100.0 | 0.0 |
| GrowMoE | iid | FULL | 1 | 1 | 1 | 0.00 | 0.00 | 0.00 | 100.0 | 0.2 |
| GrowMoE | iid | HOLE | 0 | 1 | 1 | 0.00 | 0.00 | 0.00 | 100.0 | 0.0 |
| GrowMoE | iid | SHORT | 0 | 1 | 1 | 0.00 | 0.00 | 0.00 | 100.0 | 45.8 |
| GrowMoE+s3 | seq | FULL | 0 | 6 | 6 | 0.68 | 0.12 | 0.00 | 87.8 | 0.0 |
| GrowMoE-h512-s5-r20000 | iid | FULL | 0 | 1 | 1 | 0.00 | 0.00 | 0.00 | 100.0 | 0.0 |
| GrowMoE-h512-s5-r20000 | iid | FULL | 1 | 1 | 1 | 0.00 | 0.00 | 0.00 | 100.0 | 0.2 |
| GrowMoE-h512-s5-r20000 | iid | HOLE | 0 | 1 | 1 | 0.00 | 0.00 | 0.00 | 100.0 | 0.0 |
| GrowMoE-h512-s5-r20000 | iid | SHORT | 0 | 1 | 1 | 0.00 | 0.00 | 0.00 | 100.0 | 45.8 |
| GrowMoE-h512-s5-r20000 | seq | FULL | 0 | 1 | 1 | 0.00 | 0.00 | 0.00 | 100.0 | 0.3 |
| GrowMoE-h512-s5-r20000 | seq | FULL | 1 | 1 | 1 | 0.00 | 0.00 | 0.00 | 100.0 | 0.2 |
| GrowMoE-h512-s5-r20000+s3 | seq | FULL | 0 | 4 | 4 | 0.50 | 0.11 | 0.00 | 84.2 | 0.2 |
| GrowMoE-h512-s5-r20000+s3 | seq | FULL | 1 | 4 | 4 | 0.50 | 0.11 | 0.00 | 84.8 | 0.0 |
| GrowMoE-h512-s5-r20000+s3 | seq | HOLE | 0 | 4 | 4 | 0.51 | 0.11 | 0.00 | 84.9 | 0.0 |
| GrowMoE-h512-s5-r20000+s3 | seq | SHORT | 0 | 2 | 2 | 0.37 | 0.13 | 0.00 | 93.4 | 46.7 |

**Same trained pool read out in groups (the slots of one problem answered together):**

| learner | protocol | regime | seed | exact: individual | exact: grouped | NMI operator (grouped) | NMI slot (grouped) | purity (grouped) | experts used (grouped) |
|---|---|---|---|---|---|---|---|---|---|
| GrowMoE | iid | FULL | 0 | 26.1 | 26.1 | 0.00 | 0.00 | 100.0 | 1 |
| GrowMoE | iid | FULL | 1 | 28.0 | 28.0 | 0.00 | 0.00 | 100.0 | 1 |
| GrowMoE | iid | HOLE | 0 | 24.2 | 24.2 | 0.00 | 0.00 | 100.0 | 1 |
| GrowMoE | iid | SHORT | 0 | 22.0 | 22.0 | 0.00 | 0.00 | 100.0 | 1 |
| GrowMoE+s3 | seq | FULL | 0 | 26.4 | 31.4 | 0.79 | 0.11 | 92.7 | 6 |
| GrowMoE-h512-s5-r20000 | iid | FULL | 0 | 37.3 | 37.3 | 0.00 | 0.00 | 100.0 | 1 |
| GrowMoE-h512-s5-r20000 | iid | FULL | 1 | 36.1 | 36.1 | 0.00 | 0.00 | 100.0 | 1 |
| GrowMoE-h512-s5-r20000 | iid | HOLE | 0 | 34.9 | 34.9 | 0.00 | 0.00 | 100.0 | 1 |
| GrowMoE-h512-s5-r20000 | iid | SHORT | 0 | 27.1 | 27.1 | 0.00 | 0.00 | 100.0 | 1 |
| GrowMoE-h512-s5-r20000 | seq | FULL | 0 | 34.0 | 34.0 | 0.00 | 0.00 | 100.0 | 1 |
| GrowMoE-h512-s5-r20000 | seq | FULL | 1 | 31.7 | 31.7 | 0.00 | 0.00 | 100.0 | 1 |
| GrowMoE-h512-s5-r20000+s3 | seq | FULL | 0 | 26.6 | 35.0 | 0.60 | 0.10 | 90.1 | 4 |
| GrowMoE-h512-s5-r20000+s3 | seq | FULL | 1 | 27.1 | 35.0 | 0.60 | 0.10 | 90.6 | 4 |
| GrowMoE-h512-s5-r20000+s3 | seq | HOLE | 0 | 25.8 | 33.9 | 0.61 | 0.10 | 90.8 | 4 |
| GrowMoE-h512-s5-r20000+s3 | seq | SHORT | 0 | 23.0 | 24.7 | 0.39 | 0.13 | 94.4 | 2 |

**Consistency of the routing** (adjusted Rand index on the same test examples):

| learner | protocol | comparison | ARI |
|---|---|---|---|
| GrowMoE | iid | FULL: seed 0 vs seed 1 | 0.00 |
| GrowMoE | iid | seed 0: FULL vs HOLE | 0.00 |
| GrowMoE | iid | seed 0: FULL vs SHORT | 0.00 |
| GrowMoE-h512-s5-r20000 | iid | FULL: seed 0 vs seed 1 | 0.00 |
| GrowMoE-h512-s5-r20000 | iid | seed 0: FULL vs HOLE | 0.00 |
| GrowMoE-h512-s5-r20000 | iid | seed 0: FULL vs SHORT | 0.00 |
| GrowMoE-h512-s5-r20000 | seq | FULL: seed 0 vs seed 1 | 0.00 |
| GrowMoE-h512-s5-r20000+s3 | seq | FULL: seed 0 vs seed 1 | 0.42 |
| GrowMoE-h512-s5-r20000+s3 | seq | seed 0: FULL vs HOLE | 0.42 |
| GrowMoE-h512-s5-r20000+s3 | seq | seed 0: FULL vs SHORT | 0.25 |
| GrowMoE (grouped read-out) | iid | FULL: seed 0 vs seed 1 | 0.00 |
| GrowMoE-h512-s5-r20000 (grouped read-out) | iid | FULL: seed 0 vs seed 1 | 0.00 |
| GrowMoE-h512-s5-r20000 (grouped read-out) | seq | FULL: seed 0 vs seed 1 | 0.00 |
| GrowMoE-h512-s5-r20000+s3 (grouped read-out) | seq | FULL: seed 0 vs seed 1 | 0.59 |

**New lengths: reuse of the specialist, abstention, novelty AUROC:**

| learner | protocol | regime | seed | reuse of the in-length specialist | abstain on in-length % | abstain on new length % | AUROC new vs in length |
|---|---|---|---|---|---|---|---|
| GrowMoE | iid | HOLE | 0 | 100.0 | 0.0 | 0.0 | 0.53 |
| GrowMoE | iid | SHORT | 0 | 100.0 | 0.3 | 71.4 | 0.98 |
| GrowMoE-h512-s5-r20000 | iid | HOLE | 0 | 100.0 | 0.0 | 0.0 | 0.53 |
| GrowMoE-h512-s5-r20000 | iid | SHORT | 0 | 100.0 | 0.3 | 71.4 | 0.98 |
| GrowMoE-h512-s5-r20000+s3 | seq | HOLE | 0 | 86.4 | 0.0 | 0.0 | 0.54 |
| GrowMoE-h512-s5-r20000+s3 | seq | SHORT | 0 | 93.0 | 0.7 | 72.6 | 0.98 |

**Operators one after the other (continual):**

| learner | seeds | add | sub | mul | mod | div | max | mean | forgetting | experts | params | stored vecs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FineTuneNet | 2 | 0.0 | 0.0 | 5.8 | 1.8 | 4.0 | 1.0 | 2.1 | 32.6 | nan | 35,084 | 0 |
| GrowMoE+s3 | 1 | 18.1 | 10.4 | 3.4 | 38.4 | 36.1 | 51.8 | 26.4 | 4.0 | 6.0 | 210,504 | 8,832 |
| GrowMoE-h512-s5-r20000 | 2 | 26.1 | 17.4 | 7.7 | 39.4 | 48.7 | 57.7 | 32.8 | 11.8 | 1.0 | 166,668 | 20,256 |
| GrowMoE-h512-s5-r20000+s3 | 2 | 19.3 | 16.2 | 6.1 | 46.7 | 44.7 | 28.3 | 26.9 | 15.6 | 4.0 | 666,672 | 25,632 |
| MLP-L-replay | 2 | 11.6 | 15.6 | 12.2 | 42.9 | 54.6 | 35.3 | 28.7 | 28.9 | nan | 2,243,596 | 20,000 |
| MLP-L-seq | 2 | 0.0 | 0.0 | 10.6 | 1.7 | 5.1 | 3.3 | 3.5 | 55.1 | nan | 2,243,596 | 0 |
| Oracle-op | 2 | 73.0 | 51.8 | 10.6 | 54.7 | 60.2 | 87.8 | 56.4 | 0.0 | 6.0 | 2,009,160 | 0 |
| ReplayNet | 2 | 1.5 | 1.7 | 5.8 | 17.0 | 37.2 | 5.5 | 11.5 | 25.0 | nan | 35,084 | 2,048 |

**Data size (FULL lengths, seed 0):**

| learner | problems per operator | add | sub | mul | mod | div | max | digit acc | seen |
|---|---|---|---|---|---|---|---|---|---|
| MLP-L | 500 | 10.2 | 5.5 | 3.6 | 32.1 | 48.8 | 30.1 | 67.7 | 100.0 |
| MLP-L | 2,000 | 43.3 | 30.3 | 8.7 | 51.9 | 54.7 | 73.1 | 79.6 | 100.0 |
| MLP-L | 4,000 | 65.9 | 47.1 | 13.0 | 56.6 | 57.9 | 86.3 | 83.7 | 100.0 |
| MLP-L | 8,000 | 83.7 | 69.0 | 17.3 | 57.1 | 60.5 | 95.0 | 87.3 | 100.0 |
| MLP-L | 16,000 | 92.8 | 85.3 | 22.8 | 61.6 | 66.5 | 98.5 | 89.9 | 97.8 |
| Transformer | 500 | 16.2 | 6.7 | 3.5 | 46.0 | 48.0 | 75.8 | 71.2 | 85.0 |
| Transformer | 2,000 | 71.1 | 46.5 | 11.5 | 54.4 | 57.5 | 94.6 | 84.3 | 92.7 |
| Transformer | 4,000 | 84.9 | 78.4 | 16.0 | 52.8 | 62.3 | 96.6 | 87.7 | 92.4 |
| Transformer | 8,000 | 85.3 | 85.1 | 22.9 | 17.6 | 18.5 | 89.5 | 84.7 | 69.5 |
| Transformer | 16,000 | 86.3 | 89.1 | 33.3 | 44.5 | 73.0 | 90.8 | 89.2 | 86.9 |

## Interpretation (what we understood, and on what basis)
Plain words. Learning from examples works for the operations whose digits depend on few other digits: with about 10,000 examples per operator a dense network gets addition right on unseen problems 83 % of the time, subtraction 68 %, max 95 %, and a small transformer reaches subtraction 89 %. It does not learn multiplication (only the units digit and, partly, the leading digit are learned; the middle digits stay near guessing) and not real division or remainders (5 to 30 % on the cases that need the operation; the 40 to 60 % overall scores come from shortcuts: when a < b the remainder is a and the quotient 0). Nothing extrapolates: trained on 1 to 3 digits, every learner scores 0.0 to 0.3 % on 4 and 5 digit additions, subtractions and maxima. A missing in-between length is partly bridged, and the like-for-like comparison (same test cells, trained with and without 3-digit numbers) shows how much: the transformer goes from 81 to 68 % on additions, from 93 to 68 % on subtractions and from 92 to 78 % on maxima (it keeps most of its skill), the large MLP from 86 to 49 %, 65 to 34 % and 98 to 81 %, GrowMoE (one seed without the length) from 23 to 18 %, 15 to 11 % and 63 to 48 % (a low level, but it keeps roughly the same share of its skill as the transformer). An earlier version of this entry compared 68 % with "about 99 %" on other, easier lengths; that was not like for like and is withdrawn.
GrowMoE in this task. In a mixed stream it stays ONE small expert (nothing looks new), and with its few-step online training it learns arithmetic much worse than a dense network (27 to 37 % mean against 64 %). Its gate does one thing well: it says "unknown" for numbers longer than anything it saw (novelty AUROC 0.98, 72 % of such problems flagged against 0.3 % of ordinary ones) but not for a length that is only missing in between (AUROC 0.53). When the operators arrive one by one, a pool without the stage-3 gate stays ONE expert and forgets little (11.8 points, final mean 32.8 %), because the 20,000 stored examples inside the pool work as a replay buffer. With the stage-3 gate it detects all five operator switches and ends with four experts that line up with the operators (one for add, sub and max together, one each for mod, div and mul, purity 90 to 96 %), identically in both seeds and when 3-digit numbers are removed; reading the slots of a problem as a group raises exact match from 26.6 to 35.0 %. That gives a readable structure but NOT better numbers: with the gate the final mean is 26.9 % and the forgetting 15.6 points, i.e. the one-expert pool is ahead on both, mainly because max falls to 28.3 % (57.7 % without the gate) when the shared add/sub/max expert is overwritten. Both pools forget far less than a plain network (55.1 points) and a dense network with a 20,000-example replay buffer (28.9 points, final mean 28.7 %), but they are well below experts that are told the operator (0 forgetting, 56.4 % mean). The spawn / extend / merge decisions are judged by accuracy, which is uninformative when accuracy is near chance, and that is the likely reason the gate's split is not exploited.
Technical statement: measured exact match, digit accuracy, NMI, adjusted Rand index, novelty AUROC and forgetting (best minus final per operator) as defined in `report_arith.py`; assumed: that the slot-wise format (one example per answer digit) does not favour any learner unfairly (the transformer decodes autoregressively, the others answer slots independently).

## Caveats and what went wrong
- Exploratory, not pre-registered; two seeds for the main setting, one or two for HOLE and SHORT (GrowMoE HOLE/SHORT: one seed).
- GrowMoE was tuned only over expert width, replay size and steps per batch; the shipped defaults (one 128-wide expert, replay 256) learn almost nothing here (27 % mean). More tuning of its training budget (many epochs are impossible in its online design) would help but cannot change the structural findings (no growth in a mixed stream, forgetting, extrapolation).
- The transformer is small (0.6 M parameters, 100 epochs) and trained once per setting; its mod/div scores jump between data sizes (17.6 % at 8,000 problems per operator against 52.8 % at 4,000 and 44.5 % at 16,000), so they are unstable, not a measurement of capacity.
- The output-slot layout per operator is part of the format, and every learner is asked about one slot at a time; a model that wrote the answer in reversed order (least significant digit first) would likely do better, which we did not test because it is a format trick.
- mod and div overall scores are dominated by shortcuts; always read the split by case.
- Bugs found on the way: a wrong expectation in my own unit test (a remainder case), a first plan that evaluated only at the end of a run (the checks of the data format were added before any model ran), and two queues that started the same job (killed the duplicate; the results are deterministic).
- The earlier GrowMoE pilots showed that more steps per batch (20) are worse than 5, because each batch is overfitted before the next arrives.

## Consequence
GrowMoE is not a good way to learn arithmetic itself, and no learner here extrapolates, but its gate and its operator-by-operator routing show real, consistent structure. Follow-ups worth doing: force one expert per operator to separate routing from the lifecycle; a reversed-output transformer as an explicit format-trick ablation; an unseen-operator test; more seeds for the length regimes.

## Evidence
- `research/growmoe_results/arith/RESULTS_ARITH.md` and the figures `research/growmoe_results/arith/results/fig_routing.png`, `fig_by_length.png`, `fig_forgetting.png`, `fig_unseen_by_operator.png`, `fig_data_curve.png`
- `research/growmoe_results/arith/results/` (72 json files with per-seed numbers and npz files with every test prediction), `research/growmoe_results/arith/results_curve/`
- `growmoe/arith.py`, `tests/growmoe/test_gm_arith.py`, `research/growmoe_results/arith/arith_lib.py`, `run_arith.py`, `report_arith.py`, `launch_main.sh`, `launch_main2.sh`, `launch_curve.sh`

## Reproduce
```
bash research/growmoe_results/arith/launch_main.sh gpu_iid      # also gpu_seq, cpu_iid, cpu_seq (and launch_main2.sh, launch_curve.sh)
python research/growmoe_results/arith/report_arith.py
python -m pytest tests/growmoe/test_gm_arith.py -q
```
About 76 minutes on a 12-thread PC with an RTX 3060 (two GPU queues, two CPU queues).
