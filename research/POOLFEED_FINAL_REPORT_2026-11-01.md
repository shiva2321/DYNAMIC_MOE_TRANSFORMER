# A reader in front of the pool: build, test, improve, analysis (final report, 2026-11-01)

**Correction (same day, later):** the pass counts of the cache arms in this report double-counted the test messages (they are evaluated after each of the three phases). Corrected, counting each distinct text once: the cache at similarity 0.9 asks 47% fewer texts (1,091 per seed, 7.1% of its answers differ from the reader), at 0.95 24% fewer (1,562). The corrected numbers and the trained head are in `POOLFEED_TOPIC_AND_HEAD_REPORT_2026-11-01.md`; the figures below marked *first count* are superseded.

You asked me to take the nuance report's next steps (wire the asking reader into the pool, make it cheaper) and to build, test, improve, analyse and report. This is that report. Figure: [FIGURE_G5.png](knowledge_status/results/routing_general/g5/FIGURE_G5.png). Every number is copied from a generated file (links at the end).

## The short answer

1. **Yes, it can be wired in, and it helps a lot, but only in one form.** Giving the pool the reader's verdict **next to** the sentence vector (and turning on the earlier worse-than-chance flag) lifts accuracy over the four test groups from 0.642 to 0.762 (+0.120), in **10 of 10 seeds**. Known skills in new wordings go from 0.681 to 0.824, and the experts line up with the skills (NMI 0.07 to 0.38). A perfect key would reach 0.808.
2. **The reader alone does not help accuracy.** It lines the experts up with the skills (NMI +0.31) but loses 0.07 on the new skill: its list of categories has no name for something nobody taught.
3. **The gain on known skills is real information, the gain on the new skill is mostly something else.** When I handed the reader's verdicts to the wrong messages (a control), the gain on known skills vanished (-0.022) and the alignment went negative. But weighting the digits more against the sentence vector, with no reader at all, gave +0.100 on the new skill. So the reader's contribution is on known skills and on routing, not on the new skill.
4. **The look-based pool does not false-spawn on a wording shift here** (0.10 spawns per seed). The false alarms of the earlier labs were about a detector, not about this pool's gate, which is lenient. What the reader fixes is the opposite problem: the pool merges different skills into one expert.
5. **Cost.** About 50 ms per message on this GPU, 2,063 unique messages per seed (about 100 s). A cheap cache that answers from the nearest earlier verdict, at cosine 0.9, cuts the distinct texts asked by 47% (1,091; *first count 36%*) with 7.1% of its answers differing from the reader and almost no loss (+0.117 accuracy). At the safer 0.95 it saves 24% (*first count 6%*).
6. **Nine predictions written first: 6 held, 3 missed** (sizes of two effects and a baseline I counted wrongly; details below).

## What was built

- The lab `routing_general/lab_poolfeed.py`: a stream of arithmetic requests in everyday wording (three taught skills one after another; the same skills in new wordings; a fourth skill nobody taught), the real pool (`UniPool`, 128-unit experts), and four ways to feed it: sentence vector, reader verdicts, both, or the true skill (ceiling). The pool code is not edited; the wiring is a featurizer.
- A **verdict cache** (cheap reader), **dynamic cards** (a new card for each newborn expert), a shuffled-verdict control, tables, paired statistics, a scorer and the figure. 12 new tests; 16 of 16 deliberate breakages of the code were caught.

## Results on the report seeds (100-109, means)

| what the pool sees | experts line up (NMI) | known skills, new wordings | the new skill | all four groups | LM passes per seed |
|---|---|---|---|---|---|
| sentence vector only (the pool today) | 0.07 | 0.681 | 0.555 | 0.642 | 0 |
| + worse-than-chance flag | 0.10 | 0.662 | 0.539 | 0.630 | 0 |
| reader only (+ flag) | 0.38 | 0.745 | 0.483 | 0.624 | 2,063 |
| reader + vector (+ flag) | 0.37 | 0.817 | 0.692 | 0.760 | 2,063 |
| reader + vector, vector weight 0.5 (+ flag) | 0.38 | 0.824 | 0.684 | 0.762 | 2,063 |
| the same + verdict cache at 0.95 | 0.35 | 0.816 | 0.662 | 0.747 | 1,562 (first count 1,943) |
| perfect key, true skill (+ flag) | 0.56 | 0.897 | 0.728 | 0.808 | 0 |

(The "new skill" column is the mean of its two wordings; the full per-group table is in `STATS_G5.md`.) The paired difference of the tuned reader-plus-vector arm against the pool today: accuracy +0.120 (s.e. 0.015, exact sign test p = 0.002), known skills in new wordings +0.143, new skill +0.129, NMI +0.311.

## How it improved, step by step (dev seeds 150-154, chosen before the report seeds)

| step | outcome |
|---|---|
| first stream with all skills mixed | one expert for everything: the stream had to teach skills one after another |
| reader verdict as the pool's key | experts line up (NMI 0.08 to 0.15) but accuracy falls on the new skill |
| + worse-than-chance flag | alignment doubles (NMI 0.30) |
| keep the sentence vector next to the verdict | the new skill recovers (0.21 to 0.40) |
| vector weight 0.5 | best dev accuracy of the fused variants (0.620); 2.0 is worse |
| hard (one-hot) verdict, verdict weight 2-3, digit weight 0.3 | worse; dropped |
| **a new card for every newborn expert** | **worse** (new skill 0.395 to 0.255): a card made of two routed messages gives an unstable key; dropped |
| verdict cache | 0.95: 37% of messages hit, 5.7% of hits wrong, but only 6% fewer passes than counting unique texts; 0.9: 36% fewer |

## What the controls say (declared after seeing the report numbers, so exploratory)

- **Shuffled verdicts:** accuracy on known skills in new wordings -0.022 (better in 1 of 10 seeds) and alignment -0.072 (0 of 10). The reader's information is what helps there.
- **No reader, vector re-weighted:** +0.100 on the new skill, +0.047 overall. This is why I credit the reader only with known skills and routing.
- **No flag:** accuracy +0.094 but alignment only +0.106; the flag roughly doubles the alignment and adds +0.03.

## What I got wrong, and what to know

- **Three predictions missed:** the reader alone lost 0.072 on the new skill (I said 0.10; right direction), the E-062 flag added 0.70 experts (I said 0.8), and the cache saved 24% where I said 25-50% (first counted as 6%: the counter also double-counted the repeated test messages, found and fixed later; the table's 2,063 is the number of distinct texts, which already counts exact repeats once).
- **My first figure title was wrong** (it said the new skill does not improve, written from dev numbers); fixed before sending.
- **The tuning is small:** vector weight 0.5 and the cache thresholds were picked on five dev seeds; the report seeds are the lab's own report seeds, not a frozen pre-registration, and the single-use seeds 400-429 are untouched.
- **A declared simplification:** the numbers in a request are read by the generator, not parsed, so the lab is about the gate.
- **The pool is a limit:** even a perfect key leaves accuracy at 0.808 and merges skills in phase 1 (3.0 experts only with the flag).
- Arithmetic only, one reader model, four skills. In topical domains (banking intents, news topics) the lab's earlier finding was that plain sentence vectors already work better than the asking reader, so this result should not be carried there without a test.

## What this means for a general nuance algorithm

The working design is now concrete: **keep the sentence vector for what a message looks like, add the reader's verdict for what it means, use the worse-than-chance flag so weak experts cannot swallow new skills, and put a cheap cache in front of the reader.** It beats the pool as it is, measured, in this domain. What it still needs: **names or cards for skills the reader cannot name** (my attempt to make them from newborn experts failed), **a test outside arithmetic**, and a cheaper reader than a language-model pass per new message.

## What I would do next (your call)

1. **Run the same wiring on banking intents and news topics**, where the lab found plain vectors strong, to see whether the fused key still helps or just costs.
2. **Make the new-skill card from outcomes** (messages whose stated answers one expert reproduces) instead of from routing, since routing made it unstable.
3. **Train a small head on the reader's verdicts** (not just cache them), checking it abstains on anything it has not seen.
4. **Wave or phase vectors** stay untested on purpose: nothing in these failures was about order or roles.

## Evidence

[SCORECARD_G5.md](knowledge_status/results/routing_general/g5/SCORECARD_G5.md) (nine predictions and the table), [STATS_G5.md](knowledge_status/results/routing_general/g5/STATS_G5.md) (paired statistics), [TABLE_explore.md](knowledge_status/results/routing_general/g5/TABLE_explore.md) (controls), design and predictions [2026-11-01-poolfeed-design.md](../docs/superpowers/specs/2026-11-01-poolfeed-design.md), notebook entry E-065; code `research/knowledge_status/routing_general/lab_poolfeed.py` and its table, score, stats and figure scripts; tests `tests/knowledge_status/test_rg_poolfeed.py`.
