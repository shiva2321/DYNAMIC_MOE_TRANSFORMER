# FS-MoE domain-balanced training (design, 2026-10-31)

The user chose "Balance domains first" among the next steps of E-057 (Q-56). Same rules: dev seed 700 for smoke tests only, report seeds 710-714, predictions first, nothing committed, growmoe/ and unimoe/ untouched.

## The problem (E-057)
With the tool on all code inputs (A3) the learner receives 1,980 free known code items and 358 new-behaviour ones against 228 human labels; chunks of 16 in order of arrival are mostly code. Code learns everything (1.00) and its wordings recover, but arithmetic falls from 0.67 (A1) to 0.20, language does not move, and a 10 % rate on the known code items (A5) trades the arithmetic gain (0.40) against code (1.00 to 0.84; code wordings 0.959 to 0.933).

## What is built
A learner that decides how verified items enter the pool by DOMAIN, not by arrival order (`fsmoe/balance.py`):
* the domain of an item is the channel for tool items (code) and a closed-form domain head (a 3-class LDA fitted on the teaching rows, whose domain is known) for human-labelled items; the system never sees truth;
* every chunk of 16 is split equally between the domains seen so far (6 / 5 / 5);
* inside a domain, items of classes the pool has never been taught come first (balanced over their classes), then known items, oldest first;
* `sub` (undersample the abundant): a chunk is formed only when every domain seen so far holds its quota, so the pace is the scarcest domain's and the surplus of code items is never used;
* `over` (oversample the scarce): a chunk is formed when ANY domain holds its quota of fresh items; a domain that is short fills its quota by replay from its own reservoir of verified items (500 per domain, uniform), so every code item is used once and the scarce domains' items are replayed.
At the end of the stream the remaining items are flushed at the same quotas (`sub`: until the scarcest domain is empty; `over`: until nothing fresh is left).

## Arms (the E-057 A3 setting: pool + global head, tool on ALL code inputs, human labels on 10 % of the other inputs; same stream, teaching and seeds)
* A3b: unchanged arrival-order chunks (must reproduce E-057's A3).
* D1: `sub`.
* D2: `over`.
Measures as in E-057: never-taught skills learned by domain, known skills in held-out and taught wordings by domain, items trained per domain, chunks, accuracy of the domain head on the human-labelled items.

## Predictions (written before any balanced cell)
* M1: D1 raises arithmetic learned from A3's 0.20 to at least 0.50 and keeps code learned at 0.80 or more.
* M2: D1's code known skills in held-out wordings end between 0.93 and 0.95 (fewer known code items than A3's 0.959; like A5's 0.933).
* M3: D2 raises arithmetic learned to at least 0.50 AND keeps code learned at 0.95 or more AND code known skills in held-out wordings at 0.95 or more (every code item is still used).
* M4: language learned stays at 0.30 or less under D1 and D2 (language is a selection problem, not a balance problem).
* M5: D2's learned fraction of all 28 is at least 0.34 (A1: 0.36).
* M6: D2's known skills in held-out wordings, all domains, end at 0.88 or more (before the stream 0.885).
* M7: D2's language known skills in held-out wordings stay at 0.78 or more (replaying the few human items does not over-fit language).
* M8: the domain head puts at least 0.98 of the human-labelled items in the right domain.

### Addendum (written after the five report seeds of A3b, D1, D2, before any `u` arm): the class priors of the global head
What was seen: D1 restores arithmetic (0.20 to 0.67) at code's expense (code 0.80, code wordings 0.906); D2 is in between (arithmetic 0.40, code 0.92, language wordings 0.761); neither beats the headline 0.36 of the all-human setting. Hypothesis (not a finding): the global closed-form head weights every class by its count (LDA priors n_c / n), so thousands of free labelled code items skew its answers against the small human-labelled classes of the other domains, whatever the chunking of the pool; D1 helped because it also starved the head of code items. Test: the same head with UNIFORM class priors (the log count of each class is subtracted from its logit at answer time; nothing else changes), in two arms: `A3u` (arrival-order chunks as A3, so no throttling) and `D2u` (the `over` chunks of D2). Exploratory; written to `results/fsmoe/B/B_priors_s<seed>.json`.
* M9: A3u learns at least 0.50 of the arithmetic skills and at least 0.95 of the code skills, with code known skills in held-out wordings at 0.95 or more.
* M10: A3u learns at least 0.36 of all 28 never-taught skills (the all-human setting A1: 0.36).
* M11: A3u's known skills in held-out wordings, all domains, end at 0.88 or more.
* M12: D2u's arithmetic learned is at least 0.10 above D2's.
