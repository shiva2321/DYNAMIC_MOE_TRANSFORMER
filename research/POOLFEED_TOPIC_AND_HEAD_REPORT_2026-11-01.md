# The reader-fed pool on banking and news, and a trained head instead of the cache (final report, 2026-11-01)

You said "yes, test it" (banking intents and news topics) and that a trained head on the reader's verdicts "could work and can be reliable and helpful". I did both. Figure: [FIGURE_G6.png](knowledge_status/results/routing_general/g6/FIGURE_G6.png). Every number is copied from a generated file (links at the end).

## The short answer

1. **In topical domains the reader does not reliably help.** The earlier gain was in arithmetic, where the sentence vector confuses wording with meaning. On banking intents the fused key (sentence vector plus reader) scores **0.927 against 0.942** for the pool as it is, better in 3 of 10 seeds. In AG News it is +0.05 (4 of 5 seeds) and in 20 Newsgroups -0.02 once the pool is tuned, inside the noise. It costs 680 to 1,090 language-model passes per seed. So: a win only where wording and meaning come apart.
2. **The bigger lever is the pool itself.** With its default recipe the pool does not learn a new news class at all (accuracy 0.27 in AG News, 0.00 in 20 Newsgroups), because a new cluster needs 32 messages before it can become an expert. Lowering that to 16 raises the new-class accuracy by **+0.23 (AG News) and +0.64 (20 Newsgroups)**.
3. **A multi-expert pool is a liability in a topical domain with few classes.** When I gave the pool shuffled (meaningless) verdicts, it collapsed into two experts and scored best in both news domains: 0.77 and 0.85, against 0.64 and 0.73 for the plain pool, because one expert learns all four classes.
4. **The trained head works, and is the most faithful cheap reader.** On the arithmetic stream, the safer head (it answers only when it is confident **and** the message is close to ones it has seen) answers 56% of the stream alone, **disagrees with the full reader on only 2.1%** of those, and asks the language model 51% fewer distinct texts (1,005 against 2,063), at -0.02 accuracy against the full reader. The simpler cache at similarity 0.9 saves 47% (1,091) with 7.1% disagreement and keeps the pool's accuracy slightly better (0.759 against 0.742), because it returns the reader's own verdict vectors.
5. **The head is not free of risk.** Its main setting (no similarity guard beyond 0.6) lost 0.076 accuracy on the report seeds, with one seed down 0.267 after early confident mistakes. The safer setting avoids that. In the news domains the head never became confident enough to answer (0%).
6. **Predictions written first:** the head 4 of 7, the topical test 3 of 8. Many misses were small, and the topical misses include a result from five dev seeds that did not replicate on ten report seeds.

## What was built

- **Trained head** (`VerdictHead` in `lab_poolfeed.py`): logistic regression on the sentence vector, trained online on the reader's confident verdicts, answering only above a confidence and a nearest-neighbour similarity, learning only from stream messages. Options for a warm-up and for sharpened verdicts were tried and dropped.
- **Topical lab** (`lab_poolfeed_topic.py`): banking77, AG News and 20 Newsgroups, three taught classes one after another, the same classes with half of the messages in a shifted wording (typos, case flips, doubled spaces, a polite sentence before and after), then one class nobody taught. Same pool and arms as before.
- **Fixed on the way:** the cost counters double-counted test messages (the first savings looked smaller; corrected and rerun), a stream so small that the news reader runs took ten minutes per seed (shrunk), and a stream design where `m_min` blocked learning (found by looking at spawns).
- **Checks:** 28 tests across the two labs; 16 of 16 deliberate breakages of the new code caught (six survived the first round; tests were added until all were caught).

## Results

| arm (mean over seeds) | banking77 (10 seeds) | AG News (5) | 20 Newsgroups (5) |
|---|---|---|---|
| pool as it is | 0.942 | 0.541 | 0.417 |
| pool, smallest cluster 16 | 0.942 | 0.636 | 0.726 |
| reader + vector | 0.927 | 0.480 | 0.647 |
| reader + vector, smallest cluster 16 | 0.927 | 0.689 | 0.709 |
| shuffled verdicts, smallest cluster 16 (control) | 0.915 | 0.765 | 0.847 |
| reader only | 0.319 | 0.415 | 0.223 |

Accuracy over four groups (known classes in plain and shifted wording, the new class in both). Banking's head arm: 0.943, 22% of the stream answered by the head, 2.1% disagreement, 24% fewer texts asked. The new-class accuracy and the per-arm detail are in `SCORECARD_G6.md`.

## What it means

- **Do not carry the reader to topical domains by default.** Plain sentence vectors already separate topics; the reader adds cost and little else, and alone it cannot classify (0.22 to 0.42).
- **The one place it helped is a surprise:** in 20 Newsgroups with the default recipe, the reader's "none of these" let the pool spawn an expert (new class 0.445 against 0.000), because the reader fixed what a smaller cluster size also fixes.
- **For cost, prefer the safer head or the cache at 0.9.** The head has the lowest disagreement (2.1%) and is the one I would trust to answer alone; the cache gives the better pool accuracy per pass. Both save about half of the reader's passes on arithmetic.

## What I got wrong, and what to know

- **The five-seed dev gain in banking (+0.042) did not replicate** on ten report seeds (-0.015); the topical predictions leaned on it. Five dev seeds are too few to set bars on sizes.
- **The first cost numbers were wrong** (double-counted tests); I corrected them and kept the first-count scorecards in the results folder. One earlier prediction flipped meaning under the correction (the cache saving: 6% first, 24% corrected, still under its 25% bar).
- **Five topical predictions missed**; two by a hair (22% against 25%, AG News -0.222 against -0.25) and one in an interesting way (shuffled verdicts won).
- **My first topical figure title was again written before the numbers** and said "a small gain on known classes"; I corrected it before sending.
- **Limits:** synthetic, label-preserving wording shifts, not natural paraphrase; three domains; one reader model; five seeds in the news domains because their reader passes are slow; the oracle arm is not used here (the label is the key); the `m_min` 16 recipe was chosen on dev after the failure was seen.

## What I would do next (your call)

1. **Try a pool that does not split a topical stream into experts it does not need** (a cap, or merging experts that predict the same classes), since the shuffled control says one expert is better there.
2. **Test the reader where wording and meaning come apart in real language**, not only in arithmetic: natural paraphrases or style changes of the same intent.
3. **Train the head on more than the sentence vector** (for example on a small reader's own hidden state) and check whether it can answer in the news domains, where it stayed silent.

## Evidence

[SCORECARD_G6.md](knowledge_status/results/routing_general/g6/SCORECARD_G6.md) (eight topical predictions and tables), [SCORECARD_G5_head.md](knowledge_status/results/routing_general/g5/SCORECARD_G5_head.md) (seven head predictions), [STATS_G5_cheap.md](knowledge_status/results/routing_general/g5/STATS_G5_cheap.md), [TABLE_cheap.md](knowledge_status/results/routing_general/g5/TABLE_cheap.md), design notes [2026-11-01-poolfeed-topic-design.md](../docs/superpowers/specs/2026-11-01-poolfeed-topic-design.md) and [2026-11-01-poolfeed-design.md](../docs/superpowers/specs/2026-11-01-poolfeed-design.md), notebook entry E-066; code `research/knowledge_status/routing_general/lab_poolfeed.py`, `lab_poolfeed_topic.py` and their table, score and figure scripts; tests `tests/knowledge_status/test_rg_poolfeed.py`, `test_rg_poolfeed_topic.py`.
