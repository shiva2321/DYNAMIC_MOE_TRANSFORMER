# Nuance detection: what the lab already settled, and one new test (E-064)

Date: 2026-11-01. The user asked (2026-11-01) to "perfect" nuance detection in language and tried five ideas in chat: chunk vectors, a typo-aware front end, a stability test, asking once per cluster, and predictive coding (and, earlier, wave or phase vectors). Per the standing rule (check the log before calling an idea untested) I read the knowledge-status lab's log (`research/knowledge_status/PROGRESS.md`, its results folder) first. This note says what each idea's record is and which one new test follows.

## The record (each line: idea, what was found, where)

| idea | status in the lab | evidence |
|---|---|---|
| Teach diverse wordings so the reader learns to ignore rewording | **Tested, failed.** Generic LLM rewrites in teaching: false alarm 0.262 / 0.232 against 0.256, and 0.14-0.16 of never-taught skills lost; a contaminated ceiling (writer told the held-out wordings) halves it to 0.144, still 3x the bar | PROGRESS 2026-10-28, `results/REWRITE_LAB.md` |
| Project out the directions the new wordings move into | **Tested, failed.** Removing the top shift directions of stories: 62 % to 54 % false alarm; the displacement is scattered (each story moves its own way) | `results/E2_WHY.txt` |
| Margin (nearest minus second nearest) instead of closeness | **Tested.** Reaches the bar only by finding 0.25 of the new skills | `results/WORDING_LAB_STAGE1/2.md` |
| Clean-up of case and spacing; closing prompts; leave-one-wording-out calibration | **Tested.** 0.250, worse, 0.252 against 0.256 | same |
| Stability of answers (the "function check") | **Tested.** Works where it can act, and it can act on very little; a single-example check is a bad trade; evidence over a batch of 16+ is the version that holds | PROGRESS 2026-10-28 |
| Ask once per cluster / group evidence | **Tested** (windows made from the asked operation): with real solvers 0.228-0.241 false alarm (no gain), with perfect solvers of the taught skills 0.064-0.153 | `results/ROUTING_GROUP_ASKED_150_154.md` |
| A reader that asks (multiple choice over the taught skills plus "none") | **Works for wording shifts in arithmetic** (0.025 false-new on dev seeds 100-109, bar 0.05; found 0.974), **loses on topical novelty** (banking: look-based finds 0.86 of never-taught intents, the asking reader 0.50); no fixed fusion works in every domain; a rewrite-based trust diagnostic fails (Spearman 0.003) | `ROUTING_ALGORITHM_2026-10-29.md`, G1-G4c |
| Chunking / segmentation | **Used where a message holds several parts** (sentence and window pieces in the tangled-message tests: look-based 0.667 / 0.702 found / left alone against the asking reader's 0.975 / 0.975). **Not applicable to the story wording itself**: in this benchmark the unit is the request (a "story" S4 part is the whole word problem, the operation is implicit in it), so there is no filler to cut away | E2/E3 tables, `generator.py` |
| Predictive-coding style surprise | **Not tested as such.** The nearest proxy, the autoencoder descriptor (latent reconstruction error), ties the others: 0.28 false unseen / 0.88 found (learned-map reader) against vMF 0.31 / 0.95 | `results/E2_TABLE.md` B3b |
| Character-aware front end | **Not tested as a fusion.** The word-and-letter-count reader alone has a low false alarm (0.12) and finds little (0.36) | `results/E2_TABLE.md` |
| Wave / phase / VSA binding | Not part of this lab. Own measurements: capacity ~ D / (7.5 ln N), so binding needs thousands of dimensions; no text benchmark | `research/VSA_GEOMETRY_TO_INTELLIGENCE.md` |

In short the lab's finding is a double dissociation: where the new thing differs in topic, the sentence embedding already separates it; where only the function or wording changes, the embedding follows the wording and only a reader that understands holds. My earlier chat statement that "stories bury the request in long text" does not fit this benchmark (the story is the request), and two of the five ideas (wording diversity, group asking) are already on record as failures. I told the user so in the report.

## The one new test: a character-aware front end (E-064)

`research/knowledge_status/char_front_check.py` runs the lab's own worlds (tuning seeds 150-154, pool `dev`) and detectors (vMF conformal, 1-NN, Mahalanobis, `BASELINE_PARAMS`, q = 0.05, calibration on held-out S1-S3 parts) on four readers: R1 (MiniLM sentence model), R0 (hashed word and letter groups), and the concatenations [R1, w x R0] with w = 0.5 and 1.0 (each part centred on the taught examples and unit length first). No wording is added to the teaching, no weight is tuned on test parts (the two w values are fixed here). Results: `results/char_front/`.

Rule (the lab's own go-on rule, fixed before): a concatenation **helps** if, for a detector, the false-unseen rate on S4-S8 falls by at least 0.05 against R1 and the share of never-taught skills found falls by at most 0.05.

### Predictions (written before the run)

- **N1.** R0 alone: false unseen on S4-S8 at most 0.20 for every detector, and unseen found at most 0.55.
- **N2.** R1 alone: false unseen on S4-S8 between 0.20 and 0.50 for every detector, found at least 0.75.
- **N3.** The noisy-text wording S7: [R1, 0.5 x R0] lowers its false-unseen rate against R1 by at least 0.10 for at least two of the three detectors, while stories S4 stay within 0.05 of R1.
- **N4.** No reader and detector pair reaches false unseen at most 0.05 with found at least 0.85.
- **N5.** The helps rule is met by neither concatenation for at least two of the three detectors (a character view fixes the spelling noise but not the stories, and the found rate pays for it).

If N3 holds and N5 fails for one concatenation, it is a real, if partial, gain and goes to the report seeds 100-109 once. Nothing here touches the single-use seeds 400-429.

## Limits

Arithmetic requests in eight wordings only; one language-model-free reader family; tuning seeds only. A positive result would still leave the stories unsolved.

## Results and scoring of N1-N5 (tuning seeds 150-154; table `research/knowledge_status/results/char_front/TABLE_150_154.md`)

**2 of 5 held (N4, N5); N1, N2 and N3 missed.**

- N1 MISSED: the spelling-only reader's false unseen on S4-S8 is 0.121 (vMF), 0.221 (1-NN), 0.221 (Mahalanobis): two of three detectors above 0.20. Found 0.359 / 0.313 / 0.257 (inside the limit of 0.55).
- N2 MISSED: MiniLM false unseen 0.342 / 0.329 / 0.223 (inside 0.20-0.50) but found 0.555 / 0.260 / 0.127, far under the 0.75 I guessed. At a 5 % calibration cut the sentence-meaning detectors find few new arithmetic skills.
- N3 MISSED: for [R1, 0.5 R0] the noisy-text false unseen changes by -0.074 (vMF), -0.014 (1-NN) and +0.107 (Mahalanobis) against R1; no detector falls by 0.10, and stories move by -0.15 (vMF), 0.00 and +0.09.
- N4 HELD: no pair reaches false unseen at most 0.05 with found at least 0.85.
- N5 HELD: neither concatenation meets the go-on rule for any detector (largest fall in false unseen: 0.045 for [R1, 0.5 R0] with vMF, and its found rate does not fall, but 0.045 is under 0.05).

Verdict: a character-aware view does not help; the dominant false alarm is on stories (0.80-0.99 of known stories flagged by the meaning model), which a spelling view cannot reach.
