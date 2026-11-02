# FS-MoE end to end: the tool, the human channel and the fast memory in one stream (2026-10-31)

You said yes to the end-to-end test. Same rules: predictions first, dev seed 700 for smoke tests only, report seeds 710-714, every number copied from a generated table (links at the end). Figure: [FIGURE_E2E.png](../results/fsmoe/FIGURE_E2E.png).

## The short answer

1. **Code now learns by itself.** In one stream over arithmetic, code and language, code-looking inputs go to a sandboxed tool that runs the function and reads its behaviour. A taught function's behaviour gives that function's label; a new behaviour opens a class of the system's own. With the tool on all code inputs (A3) the system learns **all 5 never-taught code skills with no human label on code**, opens 5.4 classes for the 5 (purity 1.00), and uses **0.59 of the human labels** of the all-human setting (228 against 386). The tool run on only 10 % of the code inputs already learns 0.84 of them.
2. **Known skills keep their new wordings.** After the stream, known skills in held-out wordings stand at **0.883** against 0.885 before (code 0.959, up from 0.936; arithmetic 0.914, from 0.950; language 0.798, from 0.811); taught wordings 0.969 against 0.972. With human labels only they ended at 0.858.
3. **But the whole system does not learn more of the 28 never-taught skills:** 0.32 for A3 against 0.36 for A1 (pool + global head, human labels only on 10 % of everything). Twenty of the 28 are language skills and language stays at 0.17; **doubling the human labels does not help it (0.17 again)**. That looks like a selection problem, not a label problem: choosing what to label by look is weak (AUROC about 0.55 in language).
4. **A side effect I had not planned for:** arithmetic learning dropped from 0.67 (A1) to 0.20 (A3). The cause is partly that the tool labels 1,980 known code items for free and every one is trained on, which crowds the others out: training them at 10 % raises arithmetic to 0.40, but costs code (1.00 down to 0.84) and code's new-wording accuracy (0.959 to 0.933). Only 3 arithmetic skills, so the intervals are wide. Predictions: 7 of 11 held.

## What was put together

Teaching as before, then one deployment stream (the stream of E-055: known skills in all wordings, never-taught skills, far texts). One system: the pool with the closed-form head in every expert plus the global closed-form head (E-056), verified items entering as interleaved chunks, diversity-ranked choice of what to verify (E-055). Verification has channels:
- **Tool channel, code only.** A text that parses as Python with a function is run on fixed probe inputs; its behaviour is what kind of input it accepts and its outputs on the probes. It equals a taught family's behaviour (known: that label) or it is new (a class opens, with an id of the system's own). Arithmetic cannot use this channel, because the request is natural language and running it would require already understanding it; language neither.
- **Human channel, everything else.** A budget of labelled feedback, a label being the true skill (a far text wastes the check).
- **Safety.** Only code that passes a name allowlist (imports of math and sys.stderr only, no dunder access, every name it reads bound by itself or a safe builtin) is run, with restricted builtins and a step limit. **No text from outside code was ever run, in any arm or seed.** 17.8 of the stream's 45 far standard-library snippets (39 %) ran inside the sandbox, which has no file, network or process access, and they opened 0.4 spurious classes per seed. This is not an OS-level container; it is not claimed to be one.

| arm | what changes | human labels | tool runs | code learned | arithmetic | language | all 28 | known, new wordings |
|---|---|---|---|---|---|---|---|---|
| A0 | pool alone, human labels on 10 % of all inputs | 386 | 0 | 0.52 | 0.40 | 0.10 | 0.21 | 0.713 |
| A1 | + the global head | 386 | 0 | 0.76 | 0.67 | 0.22 | 0.36 | 0.858 |
| A2 | + tool on 10 % of code, human 10 % of the rest | 228 | 237 | 0.84 | 0.53 | 0.19 | 0.34 | 0.869 |
| A3 | tool on all code | 228 | 2,382 | 1.00 | 0.20 | 0.17 | 0.32 | 0.883 |
| A4 | as A3, human labels doubled | 521 | 2,382 | 1.00 | 0.27 | 0.17 | 0.33 | 0.885 |

## What I take from it

- The tool channel is the one place where the world supplies the label: code needs no human, and its free known items even repair the wording gap in code. That part of your idea works.
- What is left is the language problem (20 of 28 skills), which looks selection-limited, and a balance problem: free labelled data from one domain must not crowd out the others.
- A fair caution: A3 is not better than A1 on the headline fraction, so "end to end" here means the code domain became self-sufficient and the human budget fell by 41 % at no cost in known skills, not that every new skill is now learned.

## What I would do next (your call)

1. **Domain-balanced training:** cap each domain's share of every chunk so that free labels from the tool cannot crowd out arithmetic and language.
2. **Better selection for language:** choose what to label by the global head's uncertainty and by clustering the unusual items first and labelling one representative per cluster (the look alone has AUROC about 0.55 there).
3. **An outcome channel for arithmetic:** a stated answer or a calculator gives a prediction-error check like the traces of E-055, once the request is read.
4. **A real container for the sandbox** before running any code that is not generated by us.

## What went wrong or changed on the way

- My first sandbox draft was a blacklist plus restricted builtins while the design said allowlist; the safety test showed 58 % of the real library snippets ran, and I replaced it with a true name allowlist before any end-to-end result (27 % then ran in the sandbox, 0 outside code).
- Prediction P9 mixed up "executed in the sandbox" with "produced a behaviour"; it is scored as written and missed, and the harm measure that matters (spurious classes: 0.4 per seed) is reported next to it.
- The five arms were fixed in advance; A5 and the re-runs with counters were added after seeing the first results (exploratory; the re-runs reproduced the first A1 and A3 in 10 of 10 cells).
- The first figure had a broken layout (a long legend squeezed the panels); fixed and the two earlier figures re-rendered with the same helper.

## Evidence

[REPORT_X.md](../results/fsmoe/X/REPORT_X.md), [SCORECARD_E2E.md](../results/fsmoe/SCORECARD_E2E.md), design and predictions [2026-10-31-fsmoe-e2e-design.md](../docs/superpowers/specs/2026-10-31-fsmoe-e2e-design.md), notebook entry E-057; code `fsmoe/sandbox.py`, `fsmoe/e2e.py`; 70 tests in tests/fsmoe, 22 mutants killed in total.
