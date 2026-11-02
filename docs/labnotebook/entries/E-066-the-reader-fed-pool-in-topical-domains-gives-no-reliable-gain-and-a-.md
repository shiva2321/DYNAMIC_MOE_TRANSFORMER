# E-066 - The reader-fed pool in topical domains gives no reliable gain (the pool's smallest cluster matters more), and a trained head answers half of the reader's questions with 2.1 % disagreement

- **ID:** E-066
- **Date:** 2026-11-01
- **Track:** G
- **Type:** experiment
- **Status:** completed on the lab's report seeds (banking 100-109, news 100-104; arithmetic head 100-109); NOT a pre-registration; the single-use seeds 400-429 untouched
- **Evidence grade:** B (script-generated; predictions written before the report-seed runs; dev seeds for every choice; no independent reviewer; one reader model; synthetic label-preserving wording shifts)
- **Code / commit:** not committed (standing rule); new `research/knowledge_status/routing_general/lab_poolfeed_topic.py`, `table_topic.py`, `score_topic.py`, `figure_topic.py`; extended `lab_poolfeed.py` (the trained head `VerdictHead`, options t, m, n, u, a, p, corrected counters), `table_poolfeed.py`, `score_poolfeed.py`; tests `tests/knowledge_status/test_rg_poolfeed.py` (23) and `test_rg_poolfeed_topic.py` (7), 28 in all passing, 16 of 16 mutants of the new code killed (six survived the first round and tests were added); the whole `tests/knowledge_status` suite passed before the head was added (659); growmoe and unimoe are NOT edited; designs `docs/superpowers/specs/2026-11-01-poolfeed-topic-design.md` and the addendum of `2026-11-01-poolfeed-design.md`
- **People:** Claude designed, built and ran it on the user's "Yes, test it" and "a trained head instead of the cache ... could work and can be reliable and helpful" (2026-11-01); no reviewer yet

## Question
Does the fused key (sentence vector plus the asking reader's verdict, with the E-062 flag) still help the real pool, or only cost, in topical domains where the nuance lab found plain sentence vectors stronger than the asking reader (Q-70)? Can a small trained head on the reader's verdicts replace the cache (Q-71)?

## Why we asked it
E-065 showed +0.120 accuracy in arithmetic, where the sentence vector follows wording; the nuance lab's double dissociation said the reader should not help where the new thing differs in topic. The user suggested a trained head as a more reliable cheap reader.

## Prediction (written BEFORE the report-seed runs)
Head H1-H7 (addendum of the pool-feed design note): **4 of 7 held**; topical T1-T8 (topical design note): **3 of 8 held** (T2, T3, T8).

## Method (how)
Topical lab (`lab_poolfeed_topic.py`): banking77, AG News, 20 Newsgroups from the local cache; per seed four classes (three taught one after another, natural wording; the same three with half of the messages shifted: typos, case flips, doubled spaces, one polite sentence before and one after; then one class nobody taught, half shifted); label = class; cards = the taught classes (label name, two taught messages) plus "none of these"; same pool and arms as E-065, options c, e0.5, s, t, m, p16 (`m_min` 16). Head (`VerdictHead`): logistic regression on the 128-d sentence vector trained online on the reader's confident verdicts (probability at least 0.9), answering only when its own probability is at least t and the message is at least m (cosine) from a training message, learning only from stream messages. Cost counted as distinct texts sent to the language model and stream-only answer shares.

## Results (copied from `research/knowledge_status/results/routing_general/g6/SCORECARD_G6.md`, `research/knowledge_status/results/routing_general/g5/SCORECARD_G5_head.md`, `research/knowledge_status/results/routing_general/g5/TABLE_cheap.md`)
- All four groups, pool as it is / fused / fused with `m_min` 16 / shuffled verdicts with `m_min` 16 / reader only: banking 0.942 / 0.927 / 0.927 / 0.915 / 0.319; AG News 0.541 / 0.480 / 0.689 / 0.765 / 0.415; 20 Newsgroups 0.417 / 0.647 / 0.709 / 0.847 / 0.223. Banking fused against `emb`: -0.015 (s.e. 0.029), 3 of 10 seeds; AG fused p16 against `emb:p16` +0.053 (4/5), NG -0.017 (3/5).
- New class (both wordings), default recipe: AG 0.265, NG 0.000; `m_min` 16: +0.227 and +0.642.
- Head in banking: 22 % of the stream answered, 2.1 % of answers differ from the full reader, 516 against 682 distinct texts, accuracy 0.943 (fused 0.927); in the news domains 0 % answered.
- Arithmetic, distinct texts asked per seed (uncached 2,063): cache 0.95 1,562 (-24 %, 5.2 % differ, all groups 0.747), cache 0.9 1,091 (-47 %, 7.1 %, 0.759), head t0.9 919 (-55 %, 5.8 %, 0.686), head t0.9 m0.8 1,005 (-51 %, 2.1 %, 0.742); uncached fused 0.762, `emb` 0.642.

## Interpretation (what we understood, and on what basis)
Plain words: where the sentence vector already separates classes, the reader adds a cost and no reliable gain; the pool's own settings decide whether a new class is learned at all, and a pool that splits a few-class topical stream into several experts does worse than one that cannot, which is why meaningless verdicts scored best in the news domains. The head is the most faithful cheap reader when guarded by both a confidence and a nearest-neighbour check. Technical: measured: the numbers above; the collapse to two experts under shuffled verdicts is read from the expert counts and events, not probed; the reason the pool's accuracy is a little better with the cache's exact verdicts than with the head's is not isolated (the head hands over its own soft probabilities; sharpening them did not help on dev).

## Caveats and what went wrong
- The banking dev gain (+0.042, five seeds) did not replicate on ten report seeds (F-101); five topical predictions and three head predictions missed (F-101, F-102).
- The cost counters of the cache and head arms double-counted test messages; corrected, the arithmetic cache and head arms rerun, E-065 and its report corrected (F-103).
- The first news streams were too large for the reader (about ten minutes per seed): stopped, shrunk, trimmed to 40 words, banking dev recomputed (F-104). The first topical figure title was written before the numbers and was wrong (F-105). Six mutants survived the first mutation round (F-106).
- Limits: synthetic wording shifts, three domains, one reader, five seeds in the news domains, the oracle arm not used (the label is the key); the `m_min` 16 recipe was chosen on dev after the failure was seen.

## Consequence
Report `research/POOLFEED_TOPIC_AND_HEAD_REPORT_2026-11-01.md`, figure `research/knowledge_status/results/routing_general/g6/FIGURE_G6.png`. The reader-fed pool is not recommended for topical domains by default; the trained head with both guards (t0.9, m0.8) is the reliable cheap reader, the cache at 0.9 the better accuracy per pass. Claims C-187 to C-192; failures F-101 to F-106; decisions D-65, D-66; questions Q-72, Q-73 (Q-70 answered, Q-71 partly).

## Evidence
- `research/POOLFEED_TOPIC_AND_HEAD_REPORT_2026-11-01.md` - the plain-English report
- `research/knowledge_status/results/routing_general/g6/SCORECARD_G6.md` - the eight topical predictions and tables
- `research/knowledge_status/results/routing_general/g5/SCORECARD_G5_head.md` - the seven head predictions
- `research/knowledge_status/results/routing_general/g5/TABLE_cheap.md` - the corrected cheap-reader table
- `research/knowledge_status/results/routing_general/g6/FIGURE_G6.png` - summary figure
- `docs/superpowers/specs/2026-11-01-poolfeed-topic-design.md` - topical design, dev findings, predictions, results

## Reproduce
```
set PYTHONPATH=D:/spherical_hyperspace_memory
python research/knowledge_status/routing_general/lab_poolfeed_topic.py run --domain banking --seeds 100 101 102 103 104 105 106 107 108 109 --arms key:c,p16 emb both:c,e0.5 emb:p16 both:c,e0.5,p16 both:c,e0.5,s,p16 both:c,e0.5,t0.9,m0.8,p16
python research/knowledge_status/routing_general/lab_poolfeed_topic.py run --domain agnews --seeds 100 101 102 103 104 --arms (the same)      (also --domain newsgroups)
python research/knowledge_status/routing_general/lab_poolfeed.py run --seeds 100 101 102 103 104 105 106 107 108 109 --arms both:c,e0.5,k0.95 both:c,e0.5,k0.9 both:c,e0.5,t0.9 both:c,e0.5,t0.9,m0.8
python research/knowledge_status/routing_general/score_topic.py --banking 100-109 --news 100-104
python research/knowledge_status/routing_general/score_poolfeed.py --seeds 100-109 --head
python -m pytest tests/knowledge_status/test_rg_poolfeed.py tests/knowledge_status/test_rg_poolfeed_topic.py
```
