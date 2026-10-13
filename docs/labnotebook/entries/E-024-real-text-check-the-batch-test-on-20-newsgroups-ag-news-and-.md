# E-024 - Real-text check: the batch test on 20 Newsgroups, AG News and banking77 embeddings

- **ID:** E-024
- **Date:** 2026-10-13
- **Track:** G
- **Type:** experiment
- **Status:** completed (clean setting only; the live pool on a real stream is still open, Q-04)
- **Evidence grade:** B (reproducible, 3 seeds, exploratory, NOT pre-registered)
- **Code / commit:** `research/growmoe_results/rv3/spike_real_text.py`, run from the detached worktree at commit 090ece8 (product code untouched); hashes in `evidence/MANIFEST.md`
- **People:** Claude (design, run, analysis); user (asked for this check before any more building)

## Question
Every stream behind E-023 is synthetic, and its regimes are shifted means, which is exactly what a distance-of-means test looks for. On real text, does the batch-level test still see what the per-sample gate cannot, with its error rate intact?

## Why we asked it
The user asked whether any effort was being wasted (this conversation, 2026-10-13). The biggest risk named was building stage 3 on synthetic data only (Q-04: one small real-data check exists). A cheap real-data check should come before more building.

## Prediction
None written before the run (exploratory). The stated decision rule, given to the user beforehand: if the clean test still has power on real embeddings, proceed with stage 3; if not, switch to MMD or rethink.

## Method (how)
Real text from the local Hugging Face cache (no download): 20 Newsgroups (SetFit copy), AG News, banking77 (its 20 most frequent intents). Frozen all-MiniLM-L6-v2 -> the fixed random projection to 128 -> unit length (the pool's own `HybridFrontend` recipe). Per category one `VmfSignature` (at most 256 stored prototypes) from reference documents (256 for 20 Newsgroups and AG News, 64 for banking77) and held-out test documents (256, 512, 64). Test batches of b documents (b = 8, 16, 64 for the first two datasets, 8, 16, 32 for banking77) from the same category (must not be rejected) or from another category (should be): for each category its 4 nearest categories by centre cosine plus 4 random ones. Per-sample test: conformal p-values, Benjamini-Hochberg at q = 0.02 (the pool's gate), summarised as the share of a batch's inputs flagged. Batch test: permutation test on the distance of means against the stored prototypes, 999 permutations. Seeds 100-102 choose the documents. 20 Newsgroups: 17 of 20 categories had enough documents.

## Results (what came out)
From `spike_real_text_summary.txt` (generated). Seeds pooled.

| dataset (centre cosine of tested foreign pairs: median, max) | batch | per-sample test: share of foreign inputs flagged, AUROC | batch test: false alarm @0.05 / @0.001 | detection @0.05 / @0.001 | AUROC |
|---|---|---|---|---|---|
| 20news, 17 categories (0.32, 0.86) | 8 | 0.193, 0.663 | 0.055 / 0.001 | 0.975 / 0.820 | 0.994 |
| | 16 | 0.171, 0.645 | 0.049 / 0.000 | 0.997 / 0.968 | 0.999 |
| | 64 | 0.167, 0.654 | 0.064 / 0.010 | 1.000 / 1.000 | 0.995 |
| agnews, 4 categories (0.30, 0.72) | 8 | 0.191, 0.662 | 0.055 / 0.001 | 0.967 / 0.822 | 0.993 |
| | 16 | 0.178, 0.650 | 0.049 / 0.003 | 0.998 / 0.962 | 0.998 |
| | 64 | 0.171, 0.648 | 0.104 / 0.000 | 1.000 / 1.000 | 1.000 |
| banking77, 20 intents (0.68, 0.91) | 8 | 0.339, 0.676 | 0.052 / 0.000 | 0.999 / 0.974 | 1.000 |
| | 16 | 0.342, 0.680 | 0.037 / 0.000 | 1.000 / 0.998 | 1.000 |
| | 32 | 0.338, 0.679 | 0.067 / 0.000 | 1.000 / 1.000 | 1.000 |

The per-sample test flags 0.000 of the inputs of a batch from the category it knows. Among the foreign batches the per-sample test is nearly blind to (it flags fewer than 30 % of their inputs: 75 % of the 20 Newsgroups batches, 73-74 % of AG News, 64-65 % of banking77) the batch test detects 96.7 / 99.6 / 100 % at level 0.05 and 76.6 / 95.7 / 100 % at level 0.001 for batches of 8 / 16 / 64 (20 Newsgroups). On the 20 % most similar foreign pairs (20 Newsgroups centre cosine at least 0.52; banking77 at least 0.79) the per-sample test flags 0.2-2.6 % of the inputs and the batch test detects, at level 0.001: 44 % (b = 8), 85 % (b = 16), 100 % (b = 64) on 20 Newsgroups, and 88 % (b = 8), 99 % (b = 16), 100 % (b = 32) on banking77.

## Interpretation (what we understood, and on what basis)
Plain words. The premise survives real text. The surprise is the other direction: on real text the per-sample gate is much blinder than on the synthetic streams. There it was blind only when regime centres were 0.6 or closer; here it already misses most pairs at a centre cosine of 0.3 to 0.4, because real documents of one category are spread widely around their centre. So real text regimes would be merged by today's gate far more than the synthetic overlap streams suggest, and the batch-level level matters more, not less. The price: tiny batches. At 8 documents the hardest pairs are caught only 44 % of the time (20 Newsgroups); from 16 documents on, detection is 85 % and above at the strictest level, and from 32 it is complete. For the design this means group and batch sizes of 16 to 32 or more should be the stated operating range on real text, not the 8 that sufficed on the synthetic streams (E-023 G).

Technical: the false-alarm rate is at its level in the clean setting (0.037-0.067 at 0.05) because the held-out documents are exchangeable with the reference documents by construction. Two small cells sit above nominal and are worth watching in a pre-registered run (AG News b = 64: 10 of 96 own batches at 0.05; 20 Newsgroups b = 64: 2 of 204 at 0.001), within what chance allows.

## Caveats and what went wrong
- Clean setting only: one signature per category against held-out documents. Nothing here tests the live pool (window references, ownership, contamination, spawn, label conflicts) on real data; Q-04 stays open for that.
- The categories are topics. Real regime changes can be subtler (style, time, source); topical shifts are the easy case for a mean-based test, and MMD was not run on real text.
- The pool's 128-d random projection of MiniLM embeddings is used on purpose (it is the pool's input), but it throws information away; a richer representation would only help.
- 3 of 20 Newsgroups categories were dropped for lack of documents; AG News has 4 categories, so its foreign pairs are 3 per category.
- Per-sample "flagged share" is a coarse summary of the gate; the gate's own decisions are per input.
- First run was fast and went through without a rerun (embedding took 172 s for 20 Newsgroups on CPU).

## Consequence
Stage 3 stays worth pursuing and its reason is stronger on real text; the design gets one change: the stated operating range for batches and groups is 16 to 32 inputs or more (pre-registration statements keep a small-group arm at 16 and add 8 as a reported, not gated, size). Next real-data step (needs your decision): a live-pool real-text stream (M5) with a label design chosen so that merging experts can actually hurt; until then every real-data claim stops at "the test sees the shift".

## Addendum (2026-10-13): does a better-shaped space fix the single-input gate? (`spike_real_geometry.py`)
Asked by the user ("geometry is the part for meaning and understanding"). Same clean test, one seed (100), batches of 8 documents, 2 nearest and 2 random foreign categories, on the cached 128-d embeddings, after re-shaping the space: centred, or ZCA-whitened (shrinkage 0.05), each with statistics from all categories (an upper bound) or from three random categories only (what a stream would know early). Per-sample AUROC (own against foreign, share of inputs flagged), share of foreign batches nearly invisible to the per-sample test, batch-test detection at level 0.001 (all pairs, and the 20 % most similar):

| space | 20news AUROC | invisible | batch detection (all / hardest 20 %) | banking77 AUROC | invisible | batch detection (all / hardest 20 %) |
|---|---|---|---|---|---|---|
| raw (the pool's input) | 0.648 | 76.7 % | 75.0 / 27.1 % | 0.684 | 63.1 % | 96.4 / 82.0 % |
| centred, all data | 0.653 | 76.4 % | 74.0 / 23.4 % | 0.762 | 47.5 % | 96.7 / 85.2 % |
| whitened, all data | 0.685 | 71.9 % | 56.3 / 23.0 % | 0.732 | 53.6 % | 87.5 / 64.8 % |
| centred, early statistics | 0.631 | 80.1 % | 73.8 / 23.0 % | 0.719 | 56.2 % | 96.2 / 85.9 % |
| whitened, early statistics | 0.599 | 84.7 % | 69.6 / 30.8 % | 0.679 | 64.2 % | 90.5 / 68.0 % |

Reading: re-shaping the space with centring or whitening moves the single-input gate by -0.05 to +0.08 AUROC (best case: banking77 centred with statistics of all categories, 0.684 to 0.762) and never gets it near what the batch test does; with statistics a stream could really have early, the effect is small or negative. Whitening costs the batch test power (it amplifies low-variance directions). The weakness of the single-input gate on real text is therefore mostly a limit of what one document can say about its category (the within-category spread), not a defect of the shape of the space; collecting inputs into sets is what overcomes it. Not tested: a learned metric (contrastive or supervised projection, a fine-tuned encoder), which uses information that centring and whitening do not; the original 384-d space; more than one seed.

## Evidence
- `research/growmoe_results/rv3/spike_real_text.py`, `spike_real_text.json` (36,420 batch records), `spike_real_text_summary.txt`, `spike_real_text_log.txt`, `spike_real_geometry.py`, `spike_real_geometry_log.txt`

## Reproduce
```
python research/growmoe_results/rv3/spike_real_text.py
```
(needs the cached datasets (the SetFit copy of 20 Newsgroups, ag_news, banking77) and the all-MiniLM-L6-v2 model in the local Hugging Face cache; about 10 minutes on 3 threads)
