# The real pool fed by a reader: lab G5 (E-065)

Date: 2026-11-01. The user asked (2026-11-01, after the nuance report) to build, test, improve and report the next step of that report: wire the asking reader into the pool (Q-69) and make it cheaper. This note fixes the design, what the dev seeds showed, and the predictions before the report-seed runs.

## Design

**The wiring is a featurizer, not an edit of growmoe.** The shipped pool (`UniPool`, ULDA, 128-unit experts) takes feature vectors. The asking reader (`routing_general/reader.py`: a multiple-choice question over the taught skills and "none of these", Qwen2.5-1.5B, one forward pass) gives each request a *key*: its probabilities over the cards. The pool's gate (the vMF signature) and its experts then see the key instead of, or next to, the sentence vector. growmoe is not touched (the E-062 flag is used as a recipe switch where stated).

**Stream** (`routing_general/lab_poolfeed.py`, per seed; the knowledge-status generator, single-digit operands, answers as the label, numbers read by the generator: a declared simplification, the lab is about the gate):
1. three taught skills, one after another, in the teaching wordings S1-S3 (200 per skill);
2. the same three in NEW wordings S4-S8 (300 per skill, mixed): nothing new to learn, so a spawn here would be a false one;
3. a fourth skill nobody taught appears in all wordings (300), mixed with the three (100 each).
Tests are never in the stream: taught skills in teaching wordings, taught skills in new wordings, the new skill in teaching wordings, the new skill in new wordings (40 per skill and group). The key arms differ ONLY in the key: `emb` (MiniLM 128-d: the pool as it is), `key` (reader probabilities), `both` (the two side by side), `oracle` (true skill one-hot). Options after a colon: `c` = the worse-than-chance flag (E-062), `e0.5` = embedding weight 0.5, `w` = key weight, `h` = hard key, `d` = digit weight, `y` = a new card for every expert born after teaching, `k0.95` = a verdict cache: a request whose sentence vector is within cosine 0.95 of one the language model already judged gets that verdict, anything else goes to the language model.

**Measures:** accuracy per test group, experts after each phase, spawns in phase 2 (false) and 3, NMI between the expert a training message went to and its true skill (phases 2-3), language-model passes, wall time.

## What the dev seeds showed (150-154, five seeds, means; `results/routing_general/g5/TABLE_dev*.md`)

| arm | experts after phase 3 | NMI expert-skill | acc known skills, new wordings | acc new skill | acc all four groups | LM passes per seed |
|---|---|---|---|---|---|---|
| emb (the pool today) | 2.6 | 0.08 | 0.687 | 0.410 | 0.565 | 0 |
| emb:c | 4.0 | 0.09 | 0.687 | 0.345 | 0.540 | 0 |
| key:c | 4.4 | 0.30 | 0.777 | 0.208 | 0.493 | 2060 |
| both:c | 4.4 | 0.31 | 0.758 | 0.395 | 0.596 | 2060 |
| both:c, e0.5 | 5.0 | 0.32 | 0.813 | 0.407 | 0.620 | 2060 |
| both:c, e0.5, k0.95 (cache) | 4.4 | 0.31 | 0.852 | 0.385 | 0.620 | 1927 (37 % hits, 5.7 % of hits wrong) |
| oracle:c | 5.2 | 0.56 | 0.882 | 0.330 | 0.606 | 0 |

Findings that shaped the arms: (1) the look-based pool does NOT false-spawn on the wording shift (0.0 spawns in phase 2): the pool's gate is lenient, so the premise of the lab's "false alarm" does not carry over to the pool as such; (2) even a perfect key leaves the pool with 2.0 experts for the 3 taught skills after phase 1 (3.0 with the E-062 flag): the pool's own lifecycle is a limit; (3) the reader key lines the experts up with the skills (NMI 0.30 against 0.08) but the closed list cannot name the new skill, so `key` alone loses on it, and the sentence vector next to the key repairs that; (4) tried and dropped on dev: a hard (one-hot) key, key weights 2-3, digit weight 0.3, embedding weight 2 (worse), and **a new card for every newborn expert** (`y`: new-skill accuracy 0.128 / 0.255 against 0.208 / 0.395 without it: the card made of two routed messages gives an unstable key). The embedding weight 0.5 was chosen on dev (best all-groups accuracy among the fused variants); the cache threshold 0.95 against 0.9 (0.9 hits 56 % with 8.6 % wrong). These are dev choices on five seeds; the report seeds are not touched.

## Predictions (written BEFORE the report-seed runs; report seeds 100-109, ten seeds, paired against `emb`)

- **G1.** `both:c,e0.5` beats `emb` on known skills in new wordings by at least +0.05 on average and in at least 7 of 10 seeds.
- **G2.** `both:c,e0.5` has an NMI expert-skill at least 0.15 above `emb` on average and above it in at least 7 of 10 seeds.
- **G3.** `both:c,e0.5` has an all-groups accuracy at least 0.02 above `emb` on average.
- **G4.** `key:c` is at least 0.10 below `emb` on the new skill on average (the closed list cannot name it).
- **G5.** The cache arm `both:c,e0.5,k0.95` uses 25-50 % fewer language-model passes than `both:c,e0.5`, at most 10 % of its cache hits are wrong, and its all-groups accuracy is within 0.03 and its NMI within 0.05 of `both:c,e0.5` on average.
- **G6.** `oracle:c` has an NMI at least 0.10 above `both:c,e0.5`: a perfect key still beats the reader.
- **G7.** `emb` spawns at most 0.3 experts per seed in phase 2 on average (no false spawns).
- **G8.** The E-062 flag gives `emb:c` at least 0.8 more experts after phase 3 than `emb` without raising all-groups accuracy by more than 0.02.
- **G9.** The reader costs between 25 and 60 ms per message on this PC (GPU, RTX 3060, batched).

Scored as a count; nothing is tuned after the run. Nothing touches the single-use seeds 400-429 or 600-604.

## Limits

Arithmetic requests only, one reader family (the same Qwen2.5-1.5B), single-digit operands and a perfect number parser, one pool recipe (128-unit experts, 5 inner steps), a stream of four skills, ten report seeds. The pool's own lifecycle (it merges skills even with a perfect key) limits what any key can show; the new-skill accuracy is noisy for that reason. A deployed system would also need names for the taught skills (a teacher) and cards for skills the reader cannot name.

## Results (report seeds 100-109; `results/routing_general/g5/SCORECARD_G5.md`, `STATS_G5.md`, `TABLE_explore.md`, figure `FIGURE_G5.png`)

**6 of 9 predictions held; G4, G5 and G8 missed.** The fused arm (`both:c,e0.5`) against the pool as it is today (`emb`), paired over ten seeds:

| measure | emb | fused | difference (s.e.) | better in |
|---|---|---|---|---|
| accuracy, all four groups | 0.642 | 0.762 | +0.120 (0.015) | 10/10 |
| known skills in new wordings | 0.681 | 0.824 | +0.143 (0.014) | 10/10 |
| the new skill | 0.555 | 0.684 | +0.129 (0.031) | 10/10 |
| NMI expert-skill | 0.07 | 0.38 | +0.311 (0.046) | 9/10 |

Ceiling `oracle:c`: accuracy 0.808, NMI 0.56. Reader only (`key:c`): NMI +0.311 but accuracy -0.018 (new skill -0.072).

Misses: **G4** `key:c` is 0.072 below `emb` on the new skill (bar 0.10: direction right, size smaller). **G5** the cache arm cut 6 % of the language-model passes (bar 25-50 %), wrong hits 5.5 %, accuracy -0.014, NMI -0.035: my baseline, the table's pass count of the uncached arms (2,063), is the number of UNIQUE message texts per seed (an exact-duplicate cache is already in the reader), against 3,060 messages served; a semantic cache at 0.95 adds only 6 % on top of that. **G8** the E-062 flag gave `emb:c` +0.70 experts after phase 3 (bar 0.8) and -0.012 accuracy (the accuracy half held). G7 held (0.10 false spawns per seed for `emb`: the pool does not false-spawn on this wording shift) and G9 held (50 ms per message).

**Exploratory controls, declared after the report numbers were seen (so they are labelled exploratory):**
- Shuffled verdicts (the same reader outputs given to other messages, `both:c,e0.5,s`): known skills in new wordings -0.022 (1/10 seeds), NMI -0.072 (0/10) against `emb`; so the gain on known skills and the alignment of experts come from what the reader said about each message.
- Re-weighting alone (`emb:e0.5`, no reader): the new skill +0.100, accuracy +0.047, known skills in new wordings -0.007, NMI -0.020: **much of the fused arm's gain on the new skill is the weight of the digits against the sentence vector, not the reader.**
- No flag (`both:e0.5`): accuracy +0.094, known skills in new wordings +0.112, NMI +0.106: the E-062 flag adds about +0.03 accuracy and doubles the alignment (NMI +0.311).
- Reader only without the flag (`key`): accuracy -0.032, NMI +0.141.
- Verdict cache at 0.9 (`both:c,e0.5,k0.9`), FIRST COUNT (superseded by the corrected counters in the addendum below: 1,091 distinct texts, 47 % fewer, 7.1 % of answers differ): 1,329 passes per seed (36 % fewer than the unique-text count), 6.9 % of hits wrong, accuracy +0.117 and NMI +0.275 (against +0.120 and +0.311 uncached).
- Dropped on the dev seeds: a hard key, key weights 2-3, digit weight 0.3, embedding weight 2, a new card for every newborn expert (new-skill accuracy fell from 0.395 to 0.255).

## Addendum: a trained head instead of the cache (the user's suggestion, 2026-11-01: "a small trained head on the reader's verdicts in place of the cache ... could work and can be reliable and helpful")

**Design** (`VerdictHead` in `lab_poolfeed.py`, options `t<confidence>` and `m<min cosine>`): logistic regression on the sentence vector, trained online on the reader's confident verdicts (probability at least 0.9) for the messages it had to ask about, refit after each batch. It answers a message alone only when its own probability is at least the confidence AND the message is at least `m` (cosine, default 0.6) close to a message it was trained on; every other message goes to the language model and is learned from (test messages are answered but never learned from). The reader's full verdicts for the messages the head answered are known (cached) and used only to count disagreements. 17 tests with stand-ins for the model, all passing.

**Dev seeds 150-154 (five seeds, means):**

| arm | LM passes per seed | answered by the head | disagree with the full reader | all four groups | NMI |
|---|---|---|---|---|---|
| fused, no head (`both:c,e0.5`) | 2,060 | - | - | 0.620 | 0.32 |
| cache cos 0.9 (`both:c,k0.9`, without the vector weight 0.5) | 1,336 | 56 % | 8.6 % | 0.588 | 0.31 |
| head t0.95 | 1,685 | 45 % | 3.8 % | 0.625 | 0.29 |
| head t0.9 | 1,317 | 57 % | 5.5 % | 0.648 | 0.31 |
| head t0.9, m0.8 | 1,438 | 53 % | 2.6 % | 0.644 | 0.30 |
| head t0.8 | 934 | 69 % | 8.7 % | 0.656 | 0.27 |

(The cache row is from the earlier dev table, run without the vector weight 0.5, so its accuracy is not comparable to the other rows; the head rows are in `TABLE_dev7.md`.) The head answers about as many messages as the cache for fewer disagreements. Nothing is tuned on the report seeds.

**Predictions for the report seeds 100-109 (written BEFORE the head arms are run there; main arm `both:c,e0.5,t0.9`, safer arm `both:c,e0.5,t0.9,m0.8`; the cache arm `both:c,e0.5,k0.9` and the uncached fused arm are already run):**
- **H1.** The main head arm uses at least 30 % fewer language-model passes than the unique-text count of the uncached fused arm (2,063), at most 1,444 per seed on average.
- **H2.** At most 7 % of the head's answers disagree with the full reader.
- **H3.** The main head arm's all-groups accuracy is within 0.03 of the uncached fused arm and its NMI within 0.05, on average.
- **H4.** The head disagrees less often than the cache at 0.9 did on the report seeds (6.9 %) while answering at least 50 % of the messages.
- **H5.** The safer arm (m0.8) has at most 4 % disagreements and cuts at least 20 % of the passes.
- **H6.** In no seed is the main head arm's all-groups accuracy more than 0.10 below the uncached fused arm's.
- **H7.** The main head arm beats the pool today (`emb`) on all-groups accuracy in at least 9 of 10 seeds.

### Head results on the report seeds (`results/routing_general/g5/SCORECARD_G5_head.md`; counters corrected, see below)

**4 of 7 held (H1, H2, H4, H5); H3, H6 and H7 missed.** Distinct texts asked per seed and stream messages answered, against the uncached fused arm (2,063 distinct texts), seeds 100-109:

| arm | texts asked per seed | stream answered by the cheap reader | answers that differ from the full reader | all four groups | NMI | against `emb` |
|---|---|---|---|---|---|---|
| fused, no cheap reader | 2,063 | - | - | 0.762 | 0.38 | +0.120 (10/10) |
| cache cos 0.95 | 1,562 (-24 %) | 35 % | 5.2 % | 0.747 | 0.35 | +0.105 (9/10) |
| cache cos 0.9 | 1,091 (-47 %) | 56 % | 7.1 % | 0.759 | 0.35 | +0.117 (10/10) |
| head t0.9 | 919 (-55 %) | 60 % | 5.8 % | 0.686 | 0.31 | +0.044 (8/10) |
| head t0.9, m0.8 | 1,005 (-51 %) | 56 % | 2.1 % | 0.742 | 0.34 | +0.100 (9/10) |

The main head arm (t0.9, nearest-neighbour guard 0.6) missed H3 (accuracy -0.076 against the fused arm) and H6 (worst seed -0.267: in seed 100 it disagreed with the reader on 11.9 % of its answers (the safer head 2.4 %) and the pool split into five experts (three with the safer head)). The safer head (m0.8) is the most faithful to the reader (2.1 %) at -0.020 accuracy and half the passes; the cache at 0.9 gives slightly better pool accuracy (0.759) because it returns the language model's own verdict vectors. On the ten dev seeds 150-159 the main head was level with the fused arm (0.629 against 0.624), so the head's effect on the pool is seed-set sensitive. A warm-up of 60 verdicts (`n60`) and sharpened verdicts (`u`) did not help on dev (0.613 and 0.601 against 0.607).

**A counting error found and fixed:** the cache and head arms first counted every call to the reader, and the test messages are evaluated after each of the three phases, so repeated test messages were counted again and the savings looked smaller (cache at 0.95: 6 % first, 24 % corrected; head: 44 % first, 55 % corrected). The counters now count DISTINCT texts sent to the language model and, for the hit shares, only the stream's messages. The cache and head arms were rerun on the report seeds with the corrected counters. The scorecards of the first count are kept as `SCORECARD_G5_first_count.md`, `SCORECARD_G5_head_first_count.md` and `TABLE_explore_first_count.md`. Under the corrected counters G5 (cache cuts 25-50 % of the passes) is still MISSED, at 24 %.
