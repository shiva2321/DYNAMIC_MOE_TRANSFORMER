# FS-MoE: fast-slow experts, a novelty and nuance recognizer, and smart growth (design, 2026-10-31)

Status: written by Claude after a long discussion with the user on 2026-10-31. The user opted out of further questions ("build the full system, test and analyze, give me reports; I will examine them once it is built and working"), so there is no approval gate; every call I made in the user's place is listed in section 9.

## 1. Intent (the user's words, shortened)

"We are stuck at building a reliable, working and efficient routing and recognizer of novelty and nuance for our architecture." Experts should be of different kinds for different jobs, each with a fast part (closed-form statistics, like LDA) and a slow part (a small network), more than one head, able to share, and able to be used as a team when one specialist is not enough. The system must grow horizontally (more experts) and vertically (depth) dynamically, by cheap statistics the experts already hold and not by trial training. It is tested on natural language, arithmetic and code, so that we can see what it understands, where, and how it evolves. Both mistakes (calling a familiar thing new, missing a new thing) hurt equally.

## 2. What already exists and is reused unchanged (never edited)

`growmoe/` (PhyloMoE pool, set test, gap growth, MLPExpert with function-preserving widen/deepen, baselines), `unimoe/` (UniPool, LdaHead, DensityHead, equal-memory baselines, the pool protocol observe/predict), `research/knowledge_status/` (arithmetic nuance test bed with eight wording families, learned map R5, readers). Findings that shape this design: look-based detectors tie at one wall (E-045: known skills in new wordings called unseen 0.27-0.41 at q = 0.05); calibrating each reader on VERIFIED outcomes of the stream's own wording keeps false flags honest but power is modest (E-049..E-051); the LDA head inside every expert was the biggest expert gain (E-054); growth needs experts that live long enough (Q-41).

## 3. What is new

1. A code domain with execution-verified ground truth, and a language domain with the same structure as the arithmetic one (skills x wording families x near-twins x withheld skills): `fsmoe/domains/`.
2. One shared reader for all three domains (Qwen2.5-1.5B-Instruct, last-token hidden state at a late layer after a fixed generic suffix; local cache only), cached by text hash with atomic writes: `fsmoe/reader.py`.
3. A recognizer made of several different kinds of signal and a cascade, compared at matched false-flag rate (look in raw space, look in a learned invariance space, fast-slow disagreement, expert confidence, group test, calibration on verified outcomes): `fsmoe/recognizer.py`, `fsmoe/invariance.py`.
4. A fast-slow multi-head expert view (label head, density head, disagreement) over the UniPool experts, adaptive teams at answer time, trust ledger with forgiveness: `fsmoe/team.py`, `fsmoe/ledger.py`.
5. Growth additions: right-sizing at birth and a depth trigger that fires only when widening has stopped helping (`fsmoe/growth.py`), evaluated against the shipped rules.
6. A trace experiment for the result-predicting head and for depth growth on sequences with real next states (`fsmoe/traces.py`).

## 4. Test bed

Every domain gives items with: text, skill, wording family, a near-twin skill (one operator or one word away), and an outside anchor.

| Domain | Skills | Teaching families | Held-out families | Withheld skills | Anchor |
|---|---|---|---|---|---|
| arithmetic | 8 operations (add, sub, mul, mod, floordiv, max, min, absdiff) | symbols, number words, verbal | stories, operands reordered, chatter, typos, register | 3 operations never taught | the answer is computed by the generator (a checker, never a feature) |
| code | 16 function families on lists, strings, integers | loop, builtin/comprehension, recursion/reduce | renamed identifiers, comments and docstrings, formatting and type hints, boilerplate, short names | 5 families never taught | every variant is EXECUTED on fixed probe inputs; its output signature must equal the family's reference and differ from every other family |
| language | 77 banking intents | the original utterance | typos (with casing and spacing noise), chatter, register, framing | 20 intents never taught (confusable twins included) and AG News / 20 Newsgroups / emotion sentences as far novelty | the intent label |

Near-twin pairs: sum/product, max/min, count_even/count_odd, upper/lower, add/sub, mod/floordiv; banking twins are found by class-mean cosine. Seeds: dev (all choices) 700-704, report 710-719; seeds 400-429 and 600-604 stay unspent; a pool's seed is the stream's seed + 7 (C-75).

## 5. Layer R: the recognizer (the part you are stuck on)

For each domain, the detectors and their spaces are compared on the same items with the same shared calibration rule (conformal cut at rank ceil((n+1)(1-q)), derive the effective false-flag rate from the code, report it):
* spaces: raw reader (standardised, projected), learned invariance map (linear, 64 numbers, fitted contrastively on teaching wordings of taught skills only, same-skill other-surface positives, near-twin hard negatives; the code refuses a held-out family or a withheld skill);
* signals: kNN distance, class-conditional Mahalanobis (the LDA head's statistics), vMF density, fast-slow disagreement (LDA head vs MLP head on the same input), expert confidence (posterior margin), and the group test (g = 8, 16, 32);
* calibration: on teaching wordings only (naive), on verified outcomes of the stream's own wordings (n = 40, answers checked by the anchor), per expert versus global;
* fusion: either, both, rank mean;
* measures: false flag on known skills in held-out wordings, found withheld skills split near-twin / far, AUROC, cost (multiply-adds per input), all with 95 % intervals over seeds.
Bars (declared now): false flag on known-in-new-wording <= 0.05 AND found withheld skills >= 0.90 AND twin found >= 0.70, in all three domains. Kill rule: if no combination reaches AUROC 0.95 on the near-twin category in any domain, report the wall and stop tuning detectors; the system layer then uses the best calibrated cascade and honest refusal.

## 6. Layer S: the system

UniPool (U-real settings plus lda_blend 0.6, the D-43 default) wrapped by the recognizer, teams, ledger and growth additions, streamed over the three domains as one continual stream of skills with later held-out wordings and withheld skills. Arms: U-real (shipped), +fast-slow blend, +team, +recognizer refusal, +growth additions; baselines: equal-memory kNN, equal-memory replay, streaming LDA, joint offline ceiling. Measures: accuracy on taught skills in teaching and held-out wordings, refusal of withheld skills, false refusals, forgetting, experts and parameters over time, growth events with their triggers, team size, compute per input. Every arm gets its learning rate tuned on dev seeds (F-57).

## 7. Layer T: traces (result head, depth)

A synthetic sequence stream with real next states (program-like state vectors, instruction regimes, several surface encodings of the same instruction). Experts as transition models (latent to next latent) with the same fast-slow pairing; the router judged by next-state prediction error against the fingerprint alone; depth growth judged on a composition stream. Small, self-contained, one report.

## 8. Predictions (written before any Layer R cell was run; Layer S and T predictions are appended, dated, before their runs)

* R1 raw-reader look-based detectors, naive calibration: false flag on known-in-new-wording >= 0.20 in arithmetic and code.
* R2 the learned map alone does not bring that below 0.15 in any domain.
* R3 calibration on verified stream outcomes brings the false flag to within 0.04-0.08 of q = 0.05 in all domains, with power for withheld skills 0.3-0.7.
* R4 fast-slow disagreement alone: AUROC 0.80 or higher in every domain, never more than 0.03 above the best look-based score.
* R5 near-twin withheld skills are found <= 0.50 by every single signal at q = 0.05 (code and language); far skills >= 0.90.
* R6 the group test at g = 16 detects far withheld skills >= 0.90 at false alarm <= 0.06 and near-twins <= 0.60.
* R7 verified-outcome calibration controls the false flag better than teaching-wording calibration in arithmetic and code (both within 0.02 of q only for the former).
* R8 the declared bars fail in at least two of three domains for every single signal and for the best fusion.

### 8b. Layer S predictions (written 2026-10-31 after the dev-seed Layer R readout and before any Layer S cell was run)

What the dev readout changed: the kill rule fired (no signal reaches AUROC 0.95 on near-twins; best 0.83), so the recognizer is not used as a hard gate. Layer S therefore tests it for what it can do: rank inputs by novelty so that a limited budget of VERIFIED outcomes (a label, an executed answer) is spent where a new skill is likely. Verified items are training events; a withheld skill is learned when its outcomes arrive and the pool's transfer rule spawns or extends.
* S1 (continual learning over the three domains, teaching phases in the order arithmetic, code, language): the LDA-blended pool ends within 0.03 of the offline joint network on taught skills in the teaching wording and within 0.05 on held-out wordings; forgetting <= 0.03; it ties equal-memory replay and kNN within +-0.03 (the UniMoE result); 3 to 12 experts; at most 2 growth events.
* S2 (deployment stream, 5 % of each batch verified): novelty-ranked selection verifies at least twice as many novel items as random (precision >= 0.28 against the 0.14 novel share); by the end the pool answers >= 70 % of the withheld skills with accuracy >= 0.5 under novelty-ranked selection and <= 40 % under random selection; no verification learns none.
* S3 (teams): the adaptive team (mass 0.99, cap 3) is within +-0.02 of top-1 on held-out wordings and has mean size <= 1.3; it does not pay on this bed.
* S4: held-out-wording accuracy of known skills RISES during deployment under every policy that verifies known items (the stream teaches the wording), more under novelty-ranked than under random selection.

### 8c. Layer T predictions (written 2026-10-31 before any Layer T cell was run)

* T1 (recognition on traces, noise 0.05 / 0.3 / 1.0 on the next state): the result signal (prediction error of the nearest taught regime's expert in units of its teaching error) finds withheld regimes >= 0.90 / >= 0.70 / <= 0.50 at the conformal cut (q = 0.05, n = 40) and its AUROC against known skills on held-out surfaces is >= 0.98 at noise 0.05; the look signal finds <= 0.15 of withheld regimes at every noise level (same state distribution) while its false flag on held-out surfaces is <= 0.10 only after the stream calibration.
* T2 (cost): the cascade (look shortlists one expert, the result judges it) costs <= 0.4 of the check against every expert (result_all) and loses <= 0.05 of its power.
* T3 (depth): starting from 8 hidden units, the stuck-then-grow rule widens on every case, deepens at least once on the composed cases and never on the simple ones, and ends within 1.5x of the best fixed net's error with <= 1.5x its parameters on the composed cases (composition is where depth should matter).

Not built in this round (stated now, so the report cannot hide it): integration of the trust ledger into the pool, right-sizing at birth and the depth trigger as pool rules, a shared trunk. They exist as tested modules or are deferred to Layer T.

## 9. Calls I made in the user's place

Python and PyTorch inside this repository, a new package `fsmoe/` and tests `tests/fsmoe/`, no edits to stamped files; the Qwen2.5-1.5B reader for all three domains (the best of the readers measured in E-044); code domain synthetic and executed (real MBPP code only as an optional far-novelty probe); arithmetic and language domains regenerated here with compact templates (same family structure as the knowledge-status bed, not its cached messages); results are skill recognition, not computing answers (the result head lives in Layer T); nothing committed (standing rule); everything small, checkpointed, relaunchable by one idempotent command (power cuts, RAM).
