# E-056 - FS-MoE follow-up: execution outcomes on real code settle novelty; a global fast memory fixes most of the new-skill learning

- **ID:** E-056
- **Date:** 2026-10-31
- **Track:** G
- **Type:** experiment
- **Status:** completed on report seeds 710-714 (5 seeds); follows E-055
- **Evidence grade:** B (script-generated, 5 seeds, predictions declared before the cells that test them and scored by script; the `balanced` and global-head arms and the closed-form confound test were designed after seeing earlier cells and are exploratory; NOT a pre-registration; no independent reviewer)
- **Code / commit:** not committed (standing rule); `fsmoe/exec_traces.py`, `fsmoe/learn_lab.py`, `fsmoe/confound.py`, `fsmoe/report_e.py`, `fsmoe/report_l.py`, `fsmoe/scorecard_followup.py`, `fsmoe/figures_followup.py`; tests `tests/fsmoe/test_fs_exec_traces.py`, `tests/fsmoe/test_fs_learn_lab.py` (59 pass in tests/fsmoe, 15 mutants killed); design `docs/superpowers/specs/2026-10-31-fsmoe-followup-design.md`; growmoe/ and unimoe/ untouched
- **People:** Claude designed, built and ran it on the user's answer "Both, in order" (2026-10-31) to the two gaps named in E-055; no reviewer yet

## Question
Gap 1 (Q-45): does checking every expert's prediction error separate new functions from new wordings when the outcome is the execution of real code? Gap 2 (Q-46, Q-47): why does the pool lose known skills' new-wording accuracy when it learns new skills, and can a verified new skill be learned from about 20 examples without that loss?

## Why we asked it
User request. E-055 found that no look-based signal separates a known skill in a new wording from a new near-twin skill (13-22 % found at 5 % false flags), that next-state error does in a synthetic world, and that learning new skills in deployment is slow and costs known skills' new-wording accuracy (0.885 to 0.63-0.83).

## Prediction (written BEFORE the numbers)
`docs/superpowers/specs/2026-10-31-fsmoe-followup-design.md`: E1-E6 (execution), L0-L4 (learning arms), and two addenda written after the dev seed and the first report-seed arms and before the cells they concern: L5, L6 (`balanced` arms) and L7-L9 (global-head arms). Scored by `fsmoe/scorecard_followup.py`: **9 of 16 held (execution 5 of 6, learning 4 of 10)**.

## Method (how)
- Layer E (`fsmoe/exec_traces.py`): list functions; prefix outcome o_k = f(x[:k]); step (o_{k-1}, x_k) -> o_k observed by running the code. One small network per taught family (sum, max, min, count_even, count_positive), trained on executions of the teaching styles only (loop, builtin, recursion). Snippets: template code in all 9 styles of `fsmoe/domains/code.py` and real library idioms (sum, math.fsum, reduce, sorted, heapq, statistics, ...); withheld near-twins product and count_odd and six real library functions (length, median, range, second largest, mean, last). Signals: look (Mahalanobis distance of the snippet text, same readers and frontend as E-055), cascade (the look names one expert), every expert judges, behaviour fingerprint (a ceiling). M = 1, 3, 10 lists. Conformal cut with n = 40 known snippets, q = 0.05.
- Layer L (`fsmoe/learn_lab.py`): the E-055 pool re-taught with the same seed (teaching numbers reproduced in 5 of 5 seeds), the same stream and policies; arms `interleaved`, `focused` (a new skill taught as its own batch every 10 verified items, 5 passes), `focused_replay` (half replay of a 2,000-item memory); extra arms `balanced` (half the budget on known items) and `GlobalFast` (one global closed-form head, mixture p = 0.5 p_pool + 0.5 p_lda). Policies oracle and diverse novelty at 10 %.
- Confound test (`fsmoe/confound.py`): a closed-form classifier fitted on teaching rows plus m = 10 or 20 examples of each withheld skill in teaching, held-out, mixed or balanced wordings; read on taught skills in held-out wordings.
- Seeds: dev 700 (smoke tests; nothing tuned); report 710-714. Commands in Reproduce.

## Results (copied from the generated files)
- Execution (`results/fsmoe/E/REPORT_E.md`): every expert judging gives AUROC 1.000 against near-twins at M = 1, 3 and 10, and 0.963 / 1.000 / 1.000 against real library functions; look 0.588 and 0.904; cascade 0.930-0.938 and 0.927-0.928; fingerprint 1.000. Known functions in real library idioms called new at the stream cut: look 0.167, cascade 0.067-0.080, every expert 0.000. One list gives 6 steps.
- Diagnosis (`results/fsmoe/L/REPORT_L.md`): of the known held-out-wording items that turn wrong in the interleaved arms, 0.97 (oracle) and 0.37 (diverse; 0.52 go to other known skills) go to a newly learned skill. Closed-form test, m = 10: known held-out accuracy 0.878 with no new skill, 0.858 with teaching-wording examples, 0.789 with held-out-wording examples, 0.813 mixed, 0.849 balanced; novel skills 0.778 / 0.813 / 0.833 / 0.828; known-ho errors landing on a new skill 0.21 / 0.60 / 0.53 / 0.44.
- Learning arms (oracle 10 % | diverse 10 %): interleaved learned 0.34 | 0.21, known ho after 0.728 | 0.713; focused 0.71 | 0.27, 0.665 | 0.680; focused_replay 0.77 | 0.25, 0.709 | 0.788; known tw after 0.930-0.955. `balanced` + interleaved learned 0.03, known ho 0.867; + focused_replay 0.09, 0.818. Global head: oracle 0.79 learned, known ho 0.816; diverse 0.36, 0.858; balanced 0.10, 0.882; after teaching only 0.959 taught / 0.895 held-out against 0.965 / 0.895 for the pool alone (L9 held).

## Interpretation (what we understood, and on what basis)
Plain words: when code can be run, running it is the novelty detector: a known function in a new style behaves the same, a new function behaves differently, and a small learned expert per function says so after one list. The pool's loss of old skills' new-wording accuracy comes from a wording asymmetry (new skills learned from stream wordings, old ones from the teaching wording): measured with a closed-form classifier (-0.02 vs -0.09 vs -0.03), and 97 % of the pool's flips go to new skills in the oracle arm. A global closed-form head next to the pool learns the same verified items far better (0.79 against 0.34) because nothing has to be routed. Technical: measured for the execution result (small deterministic list functions; one-number prefix state), the confound test and the arm comparison; the claim that missing routing is the pool's bottleneck is an inference from the global head's gain, not a measurement of the routing itself.

## Caveats and what went wrong
- The execution check covers list functions whose prefix state is one number; a taught function without such a recurrence would need another expert; execution on chosen inputs does not exist for language.
- The `balanced` and global-head arms and the confound test were designed after seeing earlier cells (exploratory); their predictions mostly missed (L5, L7, L8).
- The interleaved arm here is not bit-identical to E-055's (chunk boundaries differ: oracle 10 % learned 0.34 against 0.31).
- Intervals are wide (seeds differ a lot, e.g. oracle interleaved learned 0.34 [-0.03, 0.71]).
- The oracle policy is an upper bound, not a method; the global head still loses 0.07 (oracle) and 0.03 (diverse) of known held-out accuracy.
- Not done: the pieces in one end-to-end stream; language.

## Consequence
Report: `research/FSMOE_FOLLOWUP_REPORT_2026-10-31.md`. Q-45 and Q-46 answered, Q-47 partly; new Q-50..Q-53. Decision D-47 (global head kept as an option, not a default). Claims C-137..C-142.

## Evidence
- `research/FSMOE_FOLLOWUP_REPORT_2026-10-31.md` - the plain-English report
- `results/fsmoe/E/REPORT_E.md`, `results/fsmoe/L/REPORT_L.md` - generated tables
- `results/fsmoe/L/CONFOUND.md` - the closed-form confound test
- `results/fsmoe/SCORECARD_FOLLOWUP.md` - the scorecard
- `docs/superpowers/specs/2026-10-31-fsmoe-followup-design.md` - design and predictions

## Reproduce
```
python -m fsmoe.exec_traces run --seeds 710 711 712 713 714
python -m fsmoe.learn_lab run --seeds 710 711 712 713 714
python -m fsmoe.learn_lab extra --seeds 710 711 712 713 714
python -m fsmoe.learn_lab fast --seeds 710 711 712 713 714
python -m fsmoe.confound --seeds 710 711 712 713 714
python -m fsmoe.report_e --seeds 710 711 712 713 714
python -m fsmoe.report_l --seeds 710 711 712 713 714
python -m fsmoe.scorecard_followup --seeds 710 711 712 713 714
python -m fsmoe.figures_followup --seeds 710 711 712 713 714
python -m pytest tests/fsmoe
```
