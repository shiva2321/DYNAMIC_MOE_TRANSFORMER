# FS-MoE end to end: one stream, the pieces together (design, 2026-10-31)

The user answered "Yes, end to end" to the next step proposed after E-056 (Q-53). Same rules: dev seed 700 for smoke tests only, report seeds 710-714, predictions first, nothing committed, growmoe/ and unimoe/ untouched.

## What is put together

One deployment stream over the three domains (the stream of E-055: known skills in all wordings, 800 never-taught skills' items, 600 far texts, same seeds), one system: the pool with the closed-form head in every expert plus the global closed-form head (`GlobalFast`, E-056), verified items entering as interleaved chunks, diversity-ranked selection of what to verify (E-055). What is new is that verification now has CHANNELS and the system finds its own labels where the world allows:
* **Tool channel (code only).** A code-looking input (it parses as a Python function) is run in a restricted sandbox on fixed probe inputs; its behaviour fingerprint (kind of input it accepts and the outputs on the probes) either equals a taught family's fingerprint (a known skill: the label comes from the taught family, no human) or is new (a NEW behaviour: the system opens a class for that fingerprint with an id of its own, again with no human). Arithmetic cannot use this channel: the request is natural language, and running it would require already understanding it. Language neither.
* **Human channel (arithmetic and language).** A labelled feedback budget (a share of the inputs of those domains), spent by the diversity-ranked policy; a label is the true skill (an item with no skill, a far text, wastes the check).
* Safety: only code that passes a static whitelist (imports of math, sys.stderr only, no dunder access, no file, network or process names) and runs with a restricted set of builtins and a step limit is executed; every other text is never run. Far texts from outside code are never executed.

## Arms (same teaching, same stream, same seeds; every pool has 121 class slots, 101 for the skills of the bed and 20 for the ids the tool assigns)
* A0 pool alone, human labels on 10 % of ALL inputs, diversity-ranked (the E-055 setting: no global head, no tool).
* A1 pool + global head, human labels on 10 % of all inputs (the E-056 best).
* A2 pool + global head, tool on 10 % of the code inputs, human labels on 10 % of the arithmetic and language inputs.
* A3 as A2 with the tool on ALL code inputs.
* A4 as A3 with human labels on 20 % of the arithmetic and language inputs.

Measures: never-taught skills learned (accuracy >= 0.5 on rows the stream never contained), for the 5 code skills a prediction counts as right when it is the class the tool opened for that skill (the class's skill is the majority truth of its items; the system never sees truth); known skills in taught and in held-out wordings after the stream, per domain; human labels and tool calls used; classes the tool opened against the 5 true new code skills (and the purity of those classes); executions of far texts.

## Predictions (written before any end-to-end cell)
* P1: A3 learns at least 0.80 of the 5 never-taught code skills with no human label on code.
* P2: A3 opens at most 8 classes for them (the 5 true ones plus at most 3 spurious or split).
* P3: A2 (tool on 10 % of code inputs) learns at least 0.40 of the code skills.
* P4: A3 learns at least 0.10 more of ALL 28 never-taught skills than A1.
* P5: A3 uses at most 0.65 of A1's human labels.
* P6: A3's known skills in held-out wordings in code end at 0.95 or more (the tool labels known code items in every style).
* P7: language stays the weak domain under A3: language skills learned <= 0.35 and language known skills in held-out wordings <= 0.85.
* P8: A4 (human labels doubled) learns at least 0.05 more language skills than A3.
* P9 (safety): no text from outside code is ever executed, and at most 5 % of the far code snippets (real standard-library functions) are executed.

### Addendum (written after the five report seeds of A0-A4, before any A5 cell): the arithmetic dip
What was seen: A3 and A4 learn all code skills with no human label on code and keep the known skills' new wordings at 0.883 (before: 0.885), but arithmetic learning falls from 0.67 (A1) to 0.20 (A3), and language does not improve with doubled labels. Hypothesis (not a finding): the tool labels 1,980 KNOWN code items for free and every one of them is trained on, so the learner's chunks are mostly code and the arithmetic and language classes are crowded out. Test: `A5` = A3 with the tool-labelled KNOWN items trained on at a rate of 10 % (the new-behaviour items always), plus the counters that were missing (human labels by domain, known items trained), and a re-run of A3 with those counters (it must reproduce the first A3). Exploratory, written after seeing the A0-A4 results; results in `results/fsmoe/X/X_extra_s<seed>.json`.
* Q1: A5 learns at least 0.50 of the arithmetic skills (A3: 0.20) and at least 0.80 of the code skills, with code known skills in held-out wordings at 0.93 or more.
* Q2: the human labels by domain show arithmetic and language receiving the same share of the labels in A3 as in A1 (so the dip is not a label-allocation effect): the arithmetic share in A3 within 0.05 of A1's share of arithmetic labels among non-code labels.
