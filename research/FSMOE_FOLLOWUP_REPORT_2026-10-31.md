# FS-MoE follow-up: real code settles novelty, and a fast memory fixes most of the new-skill learning (2026-10-31)

You asked for both gaps, in order. Same rules as before: predictions written first, dev seed 700 for smoke tests, report seeds 710-714, every number below copied from a generated table (links at the end). Figure: [FIGURE_FOLLOWUP.png](../results/fsmoe/FIGURE_FOLLOWUP.png).

## The short answer

1. **Gap 1, real code: yes, execution settles it.** When the code is actually run, a small expert per taught function predicts what one more list element does to the code's state. Checking all five experts separates known functions in new styles from the withheld near-twins (product, count_odd) at AUROC **1.000**, and from real standard-library functions at 0.963 with ONE list (6 steps) and 1.000 with three or more. The text-only look gets 0.59 against the near-twins, and it wrongly calls **17 %** of real library one-liners of known functions "new" (the execution check: 0 %). So for code (and, by the same logic, arithmetic) the outside anchor is the execution itself.
2. **Gap 2, new-skill learning: the problem was not what I guessed.** The loss of known skills' new-wording accuracy comes from a wording asymmetry: a new skill is learned from examples in the stream's wordings, the old skills only from their teaching wording, so the old skills' items in those wordings go to the new skills (97 % of the flips in the oracle arm). A closed-form test with no pool shows it cleanly: new skills taught from teaching-wording examples cost the old skills 0.02; from held-out-wording examples 0.09; when the old skills also get held-out-wording examples, 0.03.
3. **The fix that worked is a fast memory at the system level.** One global closed-form head next to the pool (no routing, one pass) learns the new skills from the same budget of checked answers: **0.79** of the 28 never-taught skills under the oracle (the pool alone 0.34; teaching each new skill as its own batch with replay 0.77), and known skills in new wordings end at 0.816 instead of 0.728. With diversity-ranked selection (no oracle) it reaches 0.36 learned and 0.858 (the pool alone 0.21 and 0.713). It costs 0.006 on taught skills after teaching only.
4. **What is still open:** known skills in new wordings do not fully recover (0.816 and 0.858, against 0.885 before), and selecting what to check by look alone is still weak (0.36 learned against the oracle's 0.79). Predictions: 9 of 16 held (execution 5 of 6, learning 4 of 10).

## Gap 1: execution outcomes on real code

**What I did.** A function on a list has a prefix outcome: `f(x[:k])`. The step from `f(x[:k-1])` and the new element `x[k]` to `f(x[:k])` is what the code does to its state, and you see it by running the code in any style. One small network per taught family (sum, max, min, count_even, count_positive) was taught from executions of the TEACHING styles only. A snippet is judged by how well the experts predict its steps, in units of each expert's own teaching error. The snippets are template code in all 9 styles of the test bed plus **real library idioms** (`sum`, `math.fsum`, `functools.reduce(operator.add, ...)`, `sorted(x)[-1]`, `heapq.nlargest`, `statistics.median`, ...). Withheld: product and count_odd (the near-twins) and six real library functions (length, median, range, second largest, mean, last element).

| signal (one list, 6 steps) | AUROC vs near-twins | AUROC vs real library functions | known function, real library idiom, called new |
|---|---|---|---|
| look (text only) | 0.588 | 0.904 | 0.167 |
| the look names one expert, only it judges | 0.930 | 0.928 | 0.067 |
| every expert judges | 1.000 | 0.963 | 0.000 |
| behaviour fingerprint (no learning; a ceiling) | 1.000 | 1.000 | 0.000 |

With three or more lists, judging by every expert is 1.000 against both. The cheap cascade keeps AUROC about 0.93 at one fifth of the expert checks. **Limits:** these are list functions whose state is one number (a running total or count); a taught function with no such recurrence (say, median) would need a different expert; and running the code on inputs you choose is possible for code and arithmetic, not for language.

## Gap 2: the new-skill learning path

**Diagnosis (Q-46).** Of the known held-out-wording items that turn from right to wrong in the old interleaved arms, 97 % under the oracle and 37 % under diversity-ranked selection (52 % go to other known skills there) end up assigned to a newly learned skill. The cause is the wording asymmetry above (closed-form test, [REPORT_L.md](../results/fsmoe/L/REPORT_L.md), last section).

**Learning arms** (oracle = only new items checked; diverse = ranked by novelty and spread out; 10 % of inputs checked; 5 seeds, wide intervals):

| how verified items reach the system | learned (oracle) | known skills in new wordings after (oracle) | learned (diverse) | after (diverse) |
|---|---|---|---|---|
| pool, interleaved chunks (as before; 0.885 before the stream) | 0.34 | 0.728 | 0.21 | 0.713 |
| pool, each new skill taught as its own batch | 0.71 | 0.665 | 0.27 | 0.680 |
| pool, own batch + replay of known items | 0.77 | 0.709 | 0.25 | 0.788 |
| **pool + one global closed-form head** | **0.79** | **0.816** | **0.36** | **0.858** |

Teaching a new skill as its own batch fixes the learning under the oracle but not the old skills' wordings, and replay of known items does not protect them much either (0.709 against 0.665; my reading is that the replayed items are in the teaching wording, which I did not test). Balancing the budget with known items protects the wordings (0.867 with the pool, 0.882 with the global head) but halves the budget for new skills, which then learn almost nothing (0.03 and 0.10). The global head wins on both measures because it needs no routing: the pool's experts have to be found for a new class, the closed-form head just updates its class statistics.

## What I would do next (your call)

1. Keep the global closed-form head beside the pool as the system's fast memory (it is an option wrapper, `GlobalFast`; nothing in the pool or the defaults changed).
2. In code and arithmetic let execution be the verifier: checking an input there costs a run, not a human label, so a much larger budget than 10 % becomes affordable (not tested here).
3. For the wording gap: verify known items in the stream's wordings too, but cheaply (execution gives that for free in code and arithmetic); language still needs a labelled feedback loop.
4. Put the pieces together in one stream (execution verifier for code and arithmetic, global head, diversity ranking) and test it end to end.

## What went wrong or changed on the way

- The `balanced` arms and the global-head arms were designed after seeing the first learning cells; they are exploratory, and their predictions (written before their runs) mostly missed (L5, L7, L8).
- The "interleaved" arm here is not bit-identical to the old one (chunk boundaries differ: 0.34 learned against 0.31 earlier).
- The wording-balance hypothesis was tested with a closed-form classifier before the pool; the pool-level `balanced` arm alone did not show it (it removed the novelty budget).
- The oracle is an upper bound, not a method (it is told which inputs are new).

## Evidence

[REPORT_E.md](../results/fsmoe/E/REPORT_E.md), [REPORT_L.md](../results/fsmoe/L/REPORT_L.md), [SCORECARD_FOLLOWUP.md](../results/fsmoe/SCORECARD_FOLLOWUP.md), design and predictions: [2026-10-31-fsmoe-followup-design.md](../docs/superpowers/specs/2026-10-31-fsmoe-followup-design.md), notebook entry E-056; code `fsmoe/exec_traces.py`, `fsmoe/learn_lab.py`, `fsmoe/confound.py`; 59 tests in tests/fsmoe, 15 mutants killed.
