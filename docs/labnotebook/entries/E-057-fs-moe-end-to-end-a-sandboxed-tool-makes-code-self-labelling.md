# E-057 - FS-MoE end to end: a sandboxed tool makes code self-labelling; language stays selection-limited

- **ID:** E-057
- **Date:** 2026-10-31
- **Track:** G
- **Type:** experiment
- **Status:** completed on report seeds 710-714 (5 seeds); follows E-055 and E-056
- **Evidence grade:** B (script-generated, 5 seeds, predictions declared before the cells that test them and scored by script; arm A5 and the counter re-runs were added after the first results and are exploratory; NOT a pre-registration; no independent reviewer)
- **Code / commit:** not committed (standing rule); `fsmoe/sandbox.py`, `fsmoe/e2e.py`, `fsmoe/report_x.py`, `fsmoe/scorecard_e2e.py`, `fsmoe/figures_e2e.py`; tests `tests/fsmoe/test_fs_sandbox.py`, `tests/fsmoe/test_fs_e2e.py` (70 pass in tests/fsmoe, 22 mutants killed over E-055 to E-057); design `docs/superpowers/specs/2026-10-31-fsmoe-e2e-design.md`; growmoe/ and unimoe/ untouched
- **People:** Claude designed, built and ran it on the user's answer "Yes, end to end" (2026-10-31) to Q-53; no reviewer yet

## Question
Put the pieces together in one stream (Q-53): a restricted execution tool as the verifier for code, labelled feedback for arithmetic and language, the global closed-form head, diversity-ranked selection. What does the whole system learn, what does it cost in human labels, and what happens to known skills and to the other domains?

## Why we asked it
User request after E-056 (execution outcomes settle novelty on real code; a global closed-form head learns new skills far better than the pool alone). Arithmetic has no executable form (a natural-language request would have to be understood first), so the design gives it, like language, only the human channel.

## Prediction (written BEFORE the numbers)
`docs/superpowers/specs/2026-10-31-fsmoe-e2e-design.md`: P1-P9, and an addendum (Q1, Q2) written after the first five arms and before arm A5. Scored by `fsmoe/scorecard_e2e.py`: **7 of 11 held**; missed: P4 (A3 learns 0.10 more of all 28 than A1), P8 (doubled labels help language), P9 (at most 5 % of far code snippets executed), Q1 (A5 recovers arithmetic to 0.50).

## Method (how)
- Stream: the stream of E-055 (same seeds, same order; asserted equal to `system_lab.build_stream`), 5,890 items, 2,395 code-looking. Pools with 121 class slots (101 bed skills plus 20 ids for the tool). Every pool re-taught per seed; arms are deep copies.
- Tool (`fsmoe/sandbox.py`): a text is run only if it parses with a public top-level function and passes a name allowlist (imports of math and sys.stderr only, no dunder access, no class, every name read bound by the text or a safe builtin), then runs with restricted builtins and a step limit. Behaviour = (first of strings, integers, lists whose five probes all run, their outputs). Verified: one behaviour per function across 9 styles, 16 distinct behaviours, attack texts never run, runaway loop stopped, non-code never run.
- Arms: A0 pool alone with human labels on 10 % of all inputs; A1 pool + global head, same labels; A2 tool on 10 % of code inputs, human labels on 10 % of the rest; A3 tool on all code inputs; A4 as A3 with human labels doubled; A5 (after the first results) as A3 with the tool's known items trained at 10 %.
- Evaluation: never-taught skills learned (accuracy >= 0.5 on rows the stream never contained); for tool-opened classes an id counts as the skill that is the majority truth of its items (evaluation only); known skills in taught and held-out wordings, per domain; counts of human labels and tool runs.
- Seeds: dev 700 (smoke test); report 710-714.

## Results (copied from the generated files)
- `results/fsmoe/X/REPORT_X.md`: A3 learns 1.00 of the 5 code skills (A2 0.84) with 0 human labels on code, opens 5.4 classes (A2 5.0), purity 1.00, human labels 228 against A1's 386 (0.59), tool runs 2,382 (1,980 known-family labels, 358 new-behaviour labels, 45 no outcome). Known skills in held-out wordings after the stream: A1 0.858, A3 0.883, A4 0.885 (before 0.885; code 0.959 from 0.936, arithmetic 0.914 from 0.950, language 0.798 from 0.811); taught wordings 0.969 (before 0.972). Never-taught skills learned (all 28): A0 0.21, A1 0.36, A2 0.34, A3 0.32, A4 0.33; arithmetic 0.40 / 0.67 / 0.53 / 0.20 / 0.27; language 0.10 / 0.22 / 0.19 / 0.17 / 0.17.
- Diagnostic arms (reproduced A1 and A3 in 10 of 10 cells): human labels by domain A1 125 / 107 / 154 (arithmetic / code / language), A3 104 / 0 / 125, arithmetic share of the non-code labels 0.449 against 0.454 (Q2 held); A5 (190 known items trained instead of 1,980) arithmetic 0.40, code 0.84, language 0.17, known held-out 0.869 (code 0.933).
- Safety: non-code texts run 0 in every arm and seed; far standard-library snippets run in the sandbox 17.8 of 45 (39 %), classes opened beyond the 5 true: 0.4 per seed.

## Interpretation (what we understood, and on what basis)
Plain words: where the world can tell the system what an input is (code, by running it), the system labels it itself, learns the new functions with no human, and keeps its old skills' new wordings; where it cannot (arithmetic, language), nothing improved by taking labels from the tool, and language stays hard because choosing what to label is weak, not because labels are few (doubling them changed nothing). Technical: measured for code (self-labelling, purity, label savings, known-skill recovery); the language result is consistent with E-055's AUROC about 0.55 for look-based selection there, but the selection mechanism was not isolated here; the crowding-out of arithmetic is partly supported (A5 0.40 against 0.20) and partly unexplained; with 3 arithmetic skills the intervals are wide.

## Caveats and what went wrong
- The sandbox is a language-level allowlist with restricted builtins and a step limit, not an OS container; it was tested against attack texts and real library code but is not claimed to be secure against hostile code.
- My first draft was a blacklist; the safety test (58 % of real library snippets ran) caught it and it was replaced before any result (F-73). P9 mixed "executed" with "produced a behaviour" (F-74).
- A5 and the counter re-runs were designed after the first results (exploratory).
- Behaviour fingerprints are exact (a ceiling in E-056); real code with side effects, several arguments or non-deterministic output needs more.
- The stream is the one of E-055; one code domain of generated templates.

## Consequence
Report: `research/FSMOE_E2E_REPORT_2026-10-31.md`. Q-53 partly answered; new Q-54..Q-57. Decision D-48, D-49. Claims C-143..C-147.

## Evidence
- `research/FSMOE_E2E_REPORT_2026-10-31.md` - the plain-English report
- `results/fsmoe/X/REPORT_X.md` - generated tables
- `results/fsmoe/SCORECARD_E2E.md` - the scorecard
- `docs/superpowers/specs/2026-10-31-fsmoe-e2e-design.md` - design and predictions

## Reproduce
```
python -m fsmoe.e2e run --seeds 710 711 712 713 714
python -m fsmoe.e2e extra --seeds 710 711 712 713 714
python -m fsmoe.report_x --seeds 710 711 712 713 714
python -m fsmoe.scorecard_e2e --seeds 710 711 712 713 714
python -m fsmoe.figures_e2e --seeds 710 711 712 713 714
python -m pytest tests/fsmoe
```
