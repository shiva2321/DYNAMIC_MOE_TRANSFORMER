# The reader-fed pool in topical domains: lab G6 (E-066)

Date: 2026-11-01. The user said "Yes, test it" to the next step of the pool-feed report: does the fused key (sentence vector plus the asking reader's verdict, with the E-062 flag) still help, or only cost, where the nuance lab found plain sentence vectors stronger than the asking reader (banking intents, news topics)? Q-70.

## Design (`routing_general/lab_poolfeed_topic.py`, tests `tests/knowledge_status/test_rg_poolfeed_topic.py`, 5 tests)

The same pool, arms and options as lab G5 (`lab_poolfeed.py`), on three labelled corpora from the local cache: banking77 (77 intents), AG News (4 topics) and 20 Newsgroups (20 groups). Per seed, four classes are drawn among those with enough teaching messages; three are taught, one is new. The label is the class.
- phase 1: the three taught classes one after another, natural wording (banking 64 messages each, news 96);
- phase 2: the same classes, new messages, half in a SHIFTED wording (banking 48, news 96 each);
- phase 3: the new class (96 messages, half shifted) mixed with the three (30 / 32 each).
The shifted wording is label-preserving and domain-neutral: the nuance lab's noisy text (typos, case flips, double spaces) plus one polite chatter sentence before and one after (the test bed's chatter list). Messages are cut to their first 40 words. Tests are new messages: taught classes plain / shifted, new class plain / shifted (banking 20 per class and group, news 40). The reader's cards are the taught classes (label name made readable, two taught messages each) plus "none of these". Arms: `emb` (sentence vector), `key` (reader verdict), `both`, options `c` (E-062 flag), `e0.5` (vector weight), `s` (shuffled verdicts, the control), `t0.9,m0.8` (the trained head of the G5 addendum), `p16` (the pool's smallest cluster `m_min` 16 instead of 32). The oracle arm is not used here: the label IS the key.

## What the dev seeds showed (banking 150-154, news 150-152; `results/routing_general/g6/TABLE_*_dev*.md`)

1. **Banking, default recipe:** `emb` scores 0.922 over the four groups; the fused arm `both:c,e0.5` 0.964 (+0.042, better in 5 of 5 seeds; taught classes shifted +0.030, new class +0.045); the shuffled control is below `emb` (-0.097 with p16), so the gain comes from what the reader said. The reader alone (`key`) is useless here (0.32): four numbers cannot carry a topical label.
2. **The pool does not learn the new class in the news domains with the default recipe, whatever the key:** `emb` new-class accuracy 0.03 / 0.13 (AG News) and 0.00 / 0.00 (20 Newsgroups), no spawn in phase 3. A higher spawn level (`alpha_spawn` 0.2, 0.5) changes nothing; a **smaller minimum cluster (`m_min` 16)** lets the pool spawn and learn it: `emb:p16` new class 0.52 / 0.68 (AG) and 0.27 / 0.25 (NG), all groups 0.675 and 0.501 against 0.441 and 0.390. The shuffled-verdict control looks best in the news domains (0.79 / 0.76) because noise collapses the pool into two experts that learn every class: a multi-expert pool is worse than one expert in a topical domain, which is a finding about the pool, not about the reader.
3. **With `m_min` 16, the fused arm against `emb:p16`:** AG News 0.726 against 0.675 (+0.051, 3/3 seeds; taught classes shifted -0.064), 20 Newsgroups 0.544 against 0.501 (+0.043). Key weights 3 and 6 did not help; the vector weight 0.5 was kept from G5.
4. **The head** answered 32 % of the stream's messages in banking (2.8 % of its answers differ from the full reader) and 0 % in the news domains (never confident and similar enough). The cost counters of the cheap-reader arms in the first dev runs double-counted the test messages (evaluated after each of the three phases); they were corrected (distinct texts asked, stream messages only) and the arithmetic cheap-reader arms were rerun before this note was written, so no head or cache saving in this note comes from the old counters.

## Predictions (written BEFORE the report-seed runs: banking seeds 100-109, AG News and 20 Newsgroups seeds 100-104)

Arms: `emb`, `both:c,e0.5` (default recipe), `emb:p16`, `key:c,p16`, `both:c,e0.5,p16`, `both:c,e0.5,s,p16`, `both:c,e0.5,t0.9,m0.8,p16`.
- **T1.** With the default recipe the pool does not learn the new news class: the new-class accuracy (mean of plain and shifted) of `emb` and of `both:c,e0.5` is at most 0.25 in both AG News and 20 Newsgroups.
- **T2.** `emb:p16` raises the new-class accuracy over default `emb` by at least 0.20 in AG News and at least 0.10 in 20 Newsgroups.
- **T3.** In the news domains, `both:c,e0.5,p16` is within 0.08 of `emb:p16` on all four groups (the reader neither clearly helps nor hurts there).
- **T4.** In banking, `both:c,e0.5` beats `emb` on all four groups by at least 0.02 on average and in at least 7 of 10 seeds.
- **T5.** `key:c,p16` is at least 0.25 below `emb:p16` on all four groups in all three domains.
- **T6.** In banking the shuffled control `both:c,e0.5,s,p16` is at least 0.03 below `both:c,e0.5,p16`; in the news domains it is not more than 0.10 above `emb:p16`.
- **T7.** In banking the head arm answers at least 25 % of the stream's messages, at most 5 % of its answers differ from the full reader, it asks at least 20 % fewer distinct texts than the uncached fused arm, and its four-group accuracy is within 0.03 of `both:c,e0.5,p16`.
- **T8.** In the news domains the head answers at most 10 % of the stream's messages.

Scored as a count; nothing is tuned after the run; seeds 400-429 and 600-604 untouched.

## Limits

Three domains, one reader model, one pool recipe, label-preserving synthetic wording shifts (not natural paraphrase), three taught classes and one new class per seed, five seeds in the news domains because their language-model passes are slow. The pool's own lifecycle dominates the news results (a smaller `m_min` matters more than any key). The `m_min` 16 recipe was chosen on dev seeds after seeing the failure and is applied to the p16 arms only; the default-recipe arms are reported beside them.

## Results (report seeds: banking 100-109, AG News and 20 Newsgroups 100-104; `results/routing_general/g6/SCORECARD_G6.md`, figure `FIGURE_G6.png`)

**3 of 8 predictions held (T2, T3, T8); T1, T4, T5, T6 and T7 missed.** Mean over seeds, all four groups (new class, both wordings):

| arm | banking77 | AG News | 20 Newsgroups |
|---|---|---|---|
| `emb` (the pool today) | 0.942 (0.985) | 0.541 (0.265) | 0.417 (0.000) |
| `both:c,e0.5` (default recipe) | 0.927 (0.925) | 0.480 (0.130) | 0.647 (0.445) |
| `emb:p16` | 0.942 (0.985) | 0.636 (0.493) | 0.726 (0.643) |
| `both:c,e0.5,p16` | 0.927 (0.925) | 0.689 (0.583) | 0.709 (0.578) |
| `both:c,e0.5,s,p16` (shuffled verdicts) | 0.915 (0.890) | 0.765 (0.693) | 0.847 (0.813) |
| `key:c,p16` (reader only) | 0.319 (0.100) | 0.415 (0.218) | 0.223 (0.000) |
| head `both:c,e0.5,t0.9,m0.8,p16` | 0.943 (0.968) | 0.689 (0.583) | 0.705 (0.570) |

- **Banking:** the fused arm is not better than the pool today: -0.015 (s.e. 0.029), better in 3 of 10 seeds; taught classes in shifted wording +0.043 (6/10), new class -0.060, NMI -0.112. The +0.042 of the five dev seeds did not replicate (T4 missed). The head answered 22 % of the stream with 2.1 % disagreement and asked 24 % fewer distinct texts (516 against 682) at +0.017 against the fused arm (T7 missed on the 25 % share only).
- **News domains:** with the default recipe the pool learns the new class badly with sentence vectors (AG 0.265, NG 0.000); the fused arm made it worse in AG (0.130) and much better in NG (0.445, 5/5 seeds). A smaller minimum cluster (`m_min` 16) raised `emb` by +0.227 (AG) and +0.642 (NG) on the new class (T2 held). With `m_min` 16 the fused arm against `emb:p16`: AG +0.053 (4/5 seeds), NG -0.017 (3/5): within noise (T3 held).
- **The noisy key beats every informative key in the news domains** (+0.129 AG, +0.121 NG over `emb:p16`; T6 missed): shuffled verdicts collapse the pool to two experts, which then learn all four classes. In a topical domain with few classes a multi-expert pool is worse than one expert.
- **The reader alone cannot classify topics** (0.319 / 0.415 / 0.223; T5 missed narrowly in AG News at -0.222 against the -0.25 bar). The head answered 0 % of the stream in the news domains (T8 held).
- **Misses:** T1 AG News `emb` 0.265 against the 0.25 bar and NG fused 0.445 (the reader's "none of these" lets the pool spawn in NG with the default recipe); T4 the banking dev gain did not replicate; T5 AG -0.222; T6 banking shuffled -0.011 against -0.03 and the news collapse above; T7 22 % against 25 %.
