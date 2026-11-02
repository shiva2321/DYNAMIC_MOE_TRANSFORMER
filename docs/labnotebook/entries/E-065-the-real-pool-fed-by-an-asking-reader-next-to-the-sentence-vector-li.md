# E-065 - The real pool fed by an asking reader next to the sentence vector: +0.12 accuracy in 10 of 10 seeds; the reader alone does not help, and a cheap cache saves 36 % of its passes at cosine 0.9

- **ID:** E-065
- **Date:** 2026-11-01
- **Track:** G
- **Type:** experiment
- **Status:** completed on the lab's report seeds 100-109 (dev seeds 150-154 for every choice); NOT a pre-registration, the single-use seeds 400-429 untouched
- **Evidence grade:** B (script-generated; nine predictions written before the report-seed runs; exploratory controls declared after the numbers and labelled; no independent reviewer; one domain, one reader model)
- **Code / commit:** not committed (standing rule); new `research/knowledge_status/routing_general/lab_poolfeed.py` (stream, keys, arms, verdict cache, dynamic cards, shuffle control), `table_poolfeed.py`, `score_poolfeed.py`, `stats_poolfeed.py`, `figure_poolfeed.py`; tests `tests/knowledge_status/test_rg_poolfeed.py` (12 tests, 16 of 16 mutants killed); growmoe, unimoe and the nuance lab's existing modules are NOT edited (the E-062 flag is used as a recipe switch); design `docs/superpowers/specs/2026-11-01-poolfeed-design.md`
- **People:** Claude designed, built and ran it on the user's message "BUILD AND TEST AND REPORT AND IMPROVE AS NEEDED AND ANALYZE AND GIVE ME FINAL REPORT" about the nuance report (2026-11-01); no reviewer yet

## Question
If the asking reader's verdict is given to the real pool as part of its input, does the pool route and learn better on a stream where known skills change wording and a new skill appears, and what does it cost? Can the reader be made cheaper? (Q-69; the open items of E-064.)

## Why we asked it
The nuance lab found a double dissociation (embeddings for topical novelty, an asking reader for wording and function shifts) but never connected the reader to the pool, and flagged two costs: two language-model passes per message and a reader that cannot name a skill nobody taught.

## Prediction (written BEFORE the report-seed runs)
G1-G9 in the design note. Scored by `score_poolfeed.py`: **6 of 9 held**; missed G4 (`key:c` is 0.072 below `emb` on the new skill, bar 0.10), G5 (the cache saved 6 % of passes against the unique-text count, bar 25-50 %; my baseline counted exact repeats already), G8 (the E-062 flag added 0.70 experts, bar 0.8).

## Method (how)
Per seed (`lab_poolfeed.py`): three taught skills one after another in the teaching wordings S1-S3 (200 each), the same three in the new wordings S4-S8 (300 each, mixed), then a fourth skill nobody taught in all wordings (300) mixed with the three (100 each); single-digit operands, the label is the answer, numbers read by the generator; tests never in the stream, four groups (known skills in teaching / new wordings, new skill in teaching / new wordings, 40 per skill and group). The pool is `UniPool` ULDA with 128-unit experts, 5 inner steps, seed + 7. Arms differ only in the key the pool sees: `emb` (MiniLM 128-d), `key` (reader probabilities over the closed list of cards: the taught skills with teacher names and two taught messages, plus "none of these"; Qwen2.5-1.5B), `both`, `oracle` (true skill). Options: `c` the E-062 worse-than-chance flag, `e0.5` vector weight, `k` verdict cache, `y` dynamic cards, `s` shuffled verdicts. Measures: accuracy per group, experts, spawns, NMI between the expert a training message went to and its skill, language-model passes and time.

## Results (copied from `research/knowledge_status/results/routing_general/g5/SCORECARD_G5.md`, `STATS_G5.md`, `TABLE_explore.md`; seeds 100-109, means)
- Fused `both:c,e0.5` against `emb`: all four groups 0.762 against 0.642 (+0.120, s.e. 0.015, better in 10/10, exact sign test p 0.002); known skills in new wordings 0.824 against 0.681 (+0.143, 10/10); new skill 0.684 against 0.555 (+0.129, 10/10); NMI 0.38 against 0.07 (+0.311, 9/10). `oracle:c`: 0.808, NMI 0.56.
- `key:c` (reader only): NMI +0.311 (10/10), all groups -0.018, new skill -0.072. `emb:c`: all groups -0.012. `emb`: 0.10 false spawns per seed in phase 2.
- Controls (exploratory): shuffled verdicts `both:c,e0.5,s`: known skills in new wordings -0.022 (1/10), NMI -0.072 (0/10); `emb:e0.5` (no reader): new skill +0.100, all groups +0.047, NMI -0.020; no flag `both:e0.5`: all groups +0.094, NMI +0.106; `key` (no flag): all groups -0.032.
- Cost: 50 ms per message (range 23-89 over ten seeds), 2,063 unique messages per seed out of 3,060 served; cache at 0.95: 37 % of messages answered from the cache, 5.5 % of hits wrong, 1,943 passes (6 % fewer), all groups +0.105; cache at 0.9: 1,329 passes (36 % fewer than 2,063, 57 % fewer than served), 6.9 % wrong, all groups +0.117, NMI +0.275.
- Dev only: dynamic cards for newborn experts lowered new-skill accuracy (0.395 to 0.255); a hard key, key weights 2-3, digit weight 0.3 and vector weight 2 were worse; with all skills mixed in one stream the pool makes one expert, so phase 1 teaches skills one after another.

## Interpretation (what we understood, and on what basis)
Plain words: the reader's verdict tells the pool what a request means, and the sentence vector tells it what the request looks like; together the experts line up with the skills and known skills in new wordings are answered better. The reader alone cannot name a skill nobody taught, so it does not help there, and some of the apparent gain on the new skill is just a change of weights between digits and vector. Technical: measured: all of the above; the mechanism for the known-skill gain (information in the verdicts) is supported by the shuffled control; the claim that the new-skill gain is mostly re-weighting rests on `emb:e0.5` (+0.100, 6/10 seeds, p 0.5) against the fused +0.129 and is a reading, not isolated; not tested: any domain other than arithmetic, any reader other than Qwen2.5-1.5B, a learned head instead of a cache.

## Caveats and what went wrong
- **Correction (later the same day, E-066):** the pass counts of the cache arms above double-counted the test messages (evaluated after each of three phases); counted per distinct text the cache at 0.9 asks 47 % fewer texts (1,091, 7.1 % of answers differ) and at 0.95 24 % fewer (1,562); see claim C-191 and F-103. The cache figures in the Results section are the first count.
- Three predictions missed (F-96); the G5 baseline was my own counting error (F-97).
- The first figure title said the new skill does not improve (written from dev numbers); fixed before it was sent (F-98).
- The first stream mixed all skills and the pool made one expert (a stream design error, found on the first smoke test); dynamic cards, which I expected to help the new skill, hurt it (F-99).
- The controls were added after the report numbers were seen, so they are exploratory (F-100). Vector weight 0.5 and the cache thresholds were chosen on five dev seeds.
- A declared simplification: the generator reads the numbers; the lab is about the gate, not parsing. Even a perfect key leaves the pool at 0.808 and merging skills in phase 1 without the flag.

## Consequence
Final report `research/POOLFEED_FINAL_REPORT_2026-11-01.md`, figure `research/knowledge_status/results/routing_general/g5/FIGURE_G5.png`. Working design: sentence vector plus reader verdict plus the worse-than-chance flag plus a cache; the missing parts are skill names or cards for what the reader cannot name, a test outside arithmetic, and a cheaper reader. Claims C-181 to C-186; failures F-96 to F-100; decisions D-63, D-64; questions Q-70, Q-71.

## Evidence
- `research/POOLFEED_FINAL_REPORT_2026-11-01.md` - the plain-English final report
- `research/knowledge_status/results/routing_general/g5/SCORECARD_G5.md` - nine predictions and the report-seed table
- `research/knowledge_status/results/routing_general/g5/STATS_G5.md` - paired statistics
- `research/knowledge_status/results/routing_general/g5/TABLE_explore.md` - the exploratory controls
- `research/knowledge_status/results/routing_general/g5/FIGURE_G5.png` - summary figure
- `docs/superpowers/specs/2026-11-01-poolfeed-design.md` - design, dev findings, predictions, results

## Reproduce
```
set PYTHONPATH=D:/spherical_hyperspace_memory
python research/knowledge_status/routing_general/lab_poolfeed.py run --seeds 100 101 102 103 104 105 106 107 108 109 --arms key:c emb emb:c both:c both:c,e0.5 both:c,e0.5,k0.95 oracle:c
python research/knowledge_status/routing_general/score_poolfeed.py --seeds 100-109
python research/knowledge_status/routing_general/stats_poolfeed.py --seeds 100-109
python research/knowledge_status/routing_general/figure_poolfeed.py --seeds 100-109
python -m pytest tests/knowledge_status/test_rg_poolfeed.py
```
