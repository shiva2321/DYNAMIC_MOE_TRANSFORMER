# FS-MoE follow-up: execution outcomes on real code, and the new-skill learning path (design, 2026-10-31)

The user answered "Both, in order" to the two gaps named in E-055: (1) does the every-expert prediction-error check work when the outcome is the EXECUTION of real code (Q-45), (2) can the pool learn a verified new skill from about 20 examples without losing known skills' new-wording accuracy (Q-46, Q-47). Same rules as the main design: dev seed 700 for choices, report seeds 710-714, predictions first, nothing committed, growmoe/ and unimoe/ untouched.

## Part 1: execution outcomes (Layer E)

A function on a list of integers has a PREFIX OUTCOME: o_k = f(x[:k]). The transition (o_{k-1}, x_k) -> o_k is what the code does to its state when one more element arrives, and it is observed by running the code, in any style (loop, builtin, recursion, renamed, formatted, a one-liner from the standard library). Experts are small networks that predict o_k from (o_{k-1}, x_k), one per taught family, trained on executions of the TEACHING styles only. A snippet is judged by the prediction errors of its executions, in units of each expert's own teaching error.
* Families: taught sum, max, min, count_even, count_positive; withheld near-twins product, count_odd (the bed's own near-twins); withheld far, real library functions: length, median, range (max minus min), second largest, mean, last element.
* Snippets: template snippets of every style of `fsmoe/domains/code.py` (3 teaching, 6 held-out) and REAL library idioms of the known and withheld functions (`sum`, `math.fsum`, `functools.reduce(operator.add, ...)`, `sorted(x)[-1]`, `heapq.nlargest`, `statistics.median`, `statistics.fmean`, ...). Every snippet is run; only snippets that execute are used.
* Signals per snippet: look (the Mahalanobis distance of the snippet text to the taught code skills, the Layer R signal), result_all (the smallest normalized error over the five experts), cascade (the look names one expert and only its error is used), and the behaviour fingerprint (outputs on the fixed probe inputs against the taught families' reference outputs; a ceiling, since nothing is learned).
* Number of executions per snippet M = 1, 3, 10 lists (about 4, 13 and 45 transitions).
* Measures: AUROC of known skills in held-out styles (template and library) against withheld near-twins and against withheld real library functions; flags at the conformal cut (n = 40 known snippets, q = 0.05, 30 draws).

### Predictions (written before any Layer E cell)
* E1: at M = 10, result_all separates known-in-held-out-styles from near-twin withheld skills at AUROC >= 0.95, while the look stays <= 0.85.
* E2: real library idioms of KNOWN functions are flagged no more often than template styles: false flag <= 0.10 at the stream cut.
* E3: result_all finds >= 0.90 of the withheld real library functions at that cut.
* E4: the cascade is within 0.05 AUROC of result_all on near-twins (the look knows the family well in code), at about a fifth of the cost.
* E5: at M = 1 (about four transitions) result_all still reaches AUROC >= 0.85 on near-twins.
* E6: the behaviour fingerprint reaches AUROC 1.00.

## Part 2: the new-skill learning path (Layer L)

First a diagnosis (Q-46): after the pool learns new skills, which known held-out-wording items turned wrong and into what? Then three ways of learning verified items, all after the same teaching and with the same stream and policies:
* `interleaved` (E-055): verified items go to the pool in chunks of 16, whatever their skills.
* `focused`: verified items wait in a buffer per skill; a skill with at least 10 verified items is taught as its own batch (its items repeated up to 32, 5 passes), the way the pool learned its first skills.
* `focused_replay`: as `focused`, but every batch is half new-skill items, half replay of known items (a bounded memory of 2,000 items, counted in the memory budget: the teaching rows sampled at random plus every known item verified so far).
Policies: oracle and diverse-novelty at 10 %. Measures as in E-055 (learned fraction, accuracy of known skills in taught and held-out wordings after).

### Predictions (written before any Layer L cell)
* L0: of the known held-out-wording items that turn from right to wrong, at least half are assigned to a newly learned skill (the twin stealing them).
* L1: `focused` lifts the learned fraction under diverse 10 % from 0.28 to at least 0.50.
* L2: `focused_replay` keeps known held-out-wording accuracy at 0.85 or more (loss <= 0.04 from 0.885) while learning at least 0.40.
* L3: under the oracle at 10 %, `focused_replay` learns at least 0.70.
* L4: known skills in taught wordings stay at 0.95 or more in every arm.

### Addendum (written after the dev seed and the first report-seed arms, before any `balanced` cell): a wording-balance hypothesis
What the first cells showed: under `interleaved`, 95-99 % of the known held-out-wording items that turn wrong go to a newly learned skill (L0 will hold); on the dev seed `focused` learns 0.71 of the new skills (against 0.11) but known held-out accuracy falls from 0.885 to 0.581. My reading (a hypothesis, not a finding): a new skill is learned from stream items in all wordings, an old skill only from its teaching wording, so an old skill's item in a stream wording looks more like the new skill that has seen that wording than like its own. If that is right, giving the old skills the same wordings should repair it. Test: policy `balanced` = half of the verification budget on diversity-ranked novelty, half on random KNOWN items (verified, so they enter the replay memory in the stream's wordings), 10 % in total, with `interleaved` and `focused_replay`. Exploratory, because it was designed after seeing the first cells; written to `results/fsmoe/L/L_extra_s<seed>.json`.
### Addendum 2 (written after the main learning arms and the closed-form confound test, before any `global_fast` cell)
What was seen: `fsmoe/confound.py` (closed-form classifier, no pool) learns the 28 new skills at 0.83 from 10 examples each, where the pool learns 0.25-0.77, and the loss of known held-out-wording accuracy depends on the wording of the new skills' examples (-0.02 with teaching wordings, -0.09 with held-out wordings; balanced with known held-out examples -0.03). Hypothesis: the pool's bottleneck is ROUTING new-skill items to an expert that holds the class, not learning; a global closed-form head, which needs no routing, learns a new skill from a few examples at once. Test: `global_fast` = the pool plus one global LDA head over all classes, updated by every observed item (teaching rows and verified items) and mixed into the answer as p = 0.5 p_pool + 0.5 p_lda (a linear mixture, so a class the pool has never seen can still win); verified items on the interleaved path; policies oracle, diverse and balanced at 10 %. Exploratory (designed after the closed-form test).
* L7: `global_fast` + `balanced`: learned fraction >= 0.60 and known held-out-wording accuracy >= 0.85.
* L8: `global_fast` + `diverse` (no known items balancing the wordings): learned fraction >= 0.60 and known held-out accuracy <= 0.82 (the wording imbalance remains).
* L9: after teaching only, `global_fast` reads taught skills within 0.01 of the pool alone (taught and held-out wordings).

* L5: under `balanced` + `focused_replay`, known held-out-wording accuracy stays at 0.85 or more while the learned fraction is at least 0.40.
* L6: under `balanced` + `interleaved`, the learned fraction stays at or below 0.30 (the focused batches, not the known items, are what teach new skills).
