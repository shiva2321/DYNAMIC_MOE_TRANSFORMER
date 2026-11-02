# Lab G5, the trained head: predictions H1-H7 on the report seeds

7 declared predictions: **4 held, 3 missed** (seeds 100-109).

| id | prediction | outcome | measured |
|---|---|---|---|
| H1 | the main head arm uses at least 30 % fewer language-model passes than the unique-text count of the uncached fused arm (at most 1,444 per seed) | HELD | 919 per seed (55% fewer than 2063) |
| H2 | at most 7 % of the head's answers disagree with the full reader | HELD | 5.8% |
| H3 | its all-groups accuracy is within 0.03 and its NMI within 0.05 of the uncached fused arm on average | MISSED | accuracy -0.076, NMI -0.074 |
| H4 | the head disagrees less often than the cache at 0.9 did on the report seeds while answering at least 50 % of the messages | HELD | head 5.8% at 60% answered; cache 7.1% |
| H5 | the safer arm (m0.8) has at most 4 % disagreements and cuts at least 20 % of the passes | HELD | 2.1% disagreements, 51% fewer passes |
| H6 | in no seed is the main head arm more than 0.10 below the uncached fused arm in all-groups accuracy | MISSED | worst seed -0.267 |
| H7 | the main head arm beats the pool today on all-groups accuracy in at least 9 of 10 seeds | MISSED | +0.044, better in 8/10 seeds |

## The table of the report seeds (seeds 100-109; mean over seeds)

| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-skill | acc taught skills (teach / new wording) | acc new skill (teach / new wording) | acc all four groups | unassigned share | language-model passes per seed | cache hits (wrong share of hits) | pool s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| emb | 10 | 2.5 / 2.7 | 0.10 | 0.10 | 0.07 | 0.729 (0.777 / 0.681) | 0.555 (0.570 / 0.540) | 0.642 | 0.16 | 0 | - | 4 |
| both:c,e0.5 | 10 | 3.0 / 4.1 | 0.60 | 0.50 | 0.38 | 0.840 (0.855 / 0.824) | 0.684 (0.708 / 0.660) | 0.762 | 0.16 | 2063 | - | 6 |
| both:c,e0.5,k0.9 | 10 | 3.0 / 4.0 | 0.60 | 0.40 | 0.35 | 0.825 (0.843 / 0.807) | 0.692 (0.693 / 0.693) | 0.759 | 0.16 | 1091 | 56% (7.1%) | 6 |
| both:c,e0.5,t0.9 | 10 | 3.4 / 3.9 | 0.40 | 0.10 | 0.31 | 0.797 (0.806 / 0.787) | 0.575 (0.593 / 0.557) | 0.686 | 0.19 | 919 | 60% (5.8%) | 5 |
| both:c,e0.5,t0.9,m0.8 | 10 | 3.1 / 3.7 | 0.50 | 0.10 | 0.34 | 0.842 (0.857 / 0.827) | 0.641 (0.652 / 0.630) | 0.742 | 0.17 | 1005 | 56% (2.1%) | 6 |

## Paired against `emb` (same seeds; mean difference, and the number of seeds in which the arm is better)

| arm | acc all | acc taught in new wordings | acc new skill | NMI expert-skill |
|---|---|---|---|---|
| both:c,e0.5 | +0.120 (10/10) | +0.143 (10/10) | +0.129 (10/10) | +0.311 (9/10) |
| both:c,e0.5,k0.9 | +0.117 (10/10) | +0.126 (10/10) | +0.138 (9/10) | +0.275 (9/10) |
| both:c,e0.5,t0.9 | +0.044 (8/10) | +0.107 (9/10) | +0.020 (6/10) | +0.237 (9/10) |
| both:c,e0.5,t0.9,m0.8 | +0.100 (9/10) | +0.147 (10/10) | +0.086 (8/10) | +0.268 (9/10) |
