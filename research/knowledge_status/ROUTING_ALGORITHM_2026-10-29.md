# The routing and novelty algorithm, version 1 (2026-10-29)

Written for the user, in plain words first. Numbers are copied from the generated files named in E-047 (`docs/labnotebook/entries/E-047-*.md`). Everything below is a DEV result: nothing is pre-registered and the single-use seeds 400-429 are unspent.

## What the algorithm does

For every part of an incoming message it returns three things: a **status** (Solid, Thin, Unseen, or Other content), a **team** of experts (none, one, or a few), and **what to do** (answer, answer with caution and collect more, learn from scratch, pass on). It is built from parts, each tested on its own, then put together.

## The steps

0. **Read the numbers once** and hand them to every expert that is asked (sharing what matters: the numbers are read one time, and the experts that train together share one number encoder).
1. **Boundary question.** Ask the language model: "Does this message ask for a calculation at all?" (Yes / No). If the Yes-probability is below a threshold, the part is **Other content** (not for the arithmetic experts). The threshold is set on held-out arithmetic requests of the taught skills only, so that about 5 % of known arithmetic would fall below it (alpha 0.05).
2. **Identification question.** Ask: "Which operation is this?" (add, subtract, multiply, divide, remainder, larger of two, none). One forward pass; the answer is a probability for each option.
3. **Status.** If the most probable operation is one nobody taught, or its probability is below 0.5, the part is **Unseen** (a new skill). Otherwise it is the named taught skill, **Solid** if that skill's expert was measured competent (at least 500 teaching examples and accuracy 0.90 on problems it never saw), **Thin** if not.
4. **Team.** Candidates are the usable taught experts, most probable first. Add the next one only while the team holds less than 99 % of the probability the model puts on the taught skills, at most three. On average this is 1.06 experts per part: one when the model is sure, two or three when it is torn, none when the part is new.
5. **Check.** If the part comes with its answer (a worked example, a teacher, a stream's feedback), the team members that are competent try it, and the one that reproduces the answer decides. For a group of parts that the model gave the same operation, a permutation test (the stated answers shuffled over the group) says whether an expert really owns the group and not by coincidence. With no answer, the most probable expert answers and extra experts would only cost.
6. **Act.** Solid: answer. Thin: answer with caution and route more teaching examples to that expert (a skill taught few examples cannot be trusted to verify anything; examples are what make it competent). Unseen: collect, spawn an expert, or hold. Other: pass on.

## What was shown, step by step

| step | measured | result | seeds |
|---|---|---|---|
| 2, 3 identification | known skills in new wordings called new (bar 0.05) | **0.025**, look-based machine 0.279 | report 100-109 |
| 2, 3 | never-taught skills found | 0.974, look-based 0.866 | report |
| 2, 3 | real human-written problems (SVAMP) of Solid skills called new | **0.138**, look-based 0.997 | report |
| 2, 3 | tangled message: the new part found / the known part left alone (bars 0.85 / 0.95) | 0.975 / 0.975, look-based 0.667 / 0.702 | report |
| 2 | operation named correctly | 0.96 overall, 0.92 in stories, 0.88 with typos, 0.87 for remainder | tuning 150-154 |
| 1 boundary | everyday sentences / look-alike traps called new (alpha 0.05) | 0.092 / 0.060 (without the question: 0.709 / 0.602) | report |
| 1 boundary | tangled: nothing flagged next to Other content (bar 0.95) | 0.872 (alpha 0.05), 0.733 (alpha 0.01) | report |
| 1 boundary | tangled: the new part found, with the question | 0.825 (alpha 0.05), 0.915 (alpha 0.01) | report |
| 4 team | settles on the right skill, with the stated answer | one expert 0.970; adaptive team **0.986 at 1.06 experts**; fixed pair 0.997 at 2.0 | report |
| 5 check | coincidences taken for a known skill (never-taught parts taken back) | at most 0.014 up to windows of 8 | tuning |
| 5 check | false "new" calls rescued, perfect experts, windows of one skill / mixed windows | 0.020 / 0.178 (from 0.256) | tuning |
| 5 check | the same, windows made from the asked operation (96 % pure): windows of 8 / 16 | 0.093 / 0.064; coincidences 0.004 / 0.050 | tuning |
| 5 check | the same with the real experts | 0.231 (from 0.256) | tuning |
| 0, 6 sharing | weak skill accuracy alone / shared (500 examples) | sub 0.17 / 0.41, mul 0.18 / 0.31; the strong skill add falls 0.93 / about 0.78 | tuning 150-152 |

## What is not proven, and the limits that matter

- **Not confirmatory.** The design was made on seeds 150-154 and reported on dev seeds 100-109. A pre-registered run of THIS design needs a new question pool, a new freeze and your go; the old freeze (look-based reader) is unaffected and parked.
- **The reader leans on the model's vocabulary.** It can name a skill nobody taught here because the model knows what "multiplication" means. It cannot recognise a skill outside that vocabulary. This is the price of wording robustness, and the reason the function check (step 5) stays in the design: a check by answers does not depend on words.
- **The boundary has a price.** No single threshold for step 1 gives a clean Other boundary and the best detection of the new part of a tangled message. alpha 0.05 favours the boundary, 0.01 the new part. Neither meets the 0.95 bar of the spec for "nothing flagged next to Other".
- **Competence comes from examples.** The reader says which expert; whether that expert is any good is its own measured accuracy. Shared encoders help weak skills a little and hurt strong ones; a skill taught 20 examples stays unusable either way.
- **Two language-model passes per part** (the boundary question and the identification question), against one for the look-based reader. A small head distilled from the answers would bring it back to one pass; not done.
- **Not wired into the expert pool.** Nothing here shows the pool routes or grows better with this status. `growmoe/` was not touched.
- **The remainder skill is the weakest** (named correctly 0.87). The real-problem set has four operations only.
- **A calibrated version of the asking reader lost** to the plain rule (the model looks near-certain on the easy taught wordings, which pushes the threshold to about 0.99). The plain rule has no statistical guarantee.

## What to do next (in this order)

1. Make the group check's null keep the operand structure (shuffle answers among parts with the same operand sizes): with windows made from the asked operation it already reaches 0.064, but small-operand coincidences reach 0.050 at windows of 16.
2. Distil the two questions into one small head on the model's hidden state, so the reader costs one pass.
3. Wire status and team into the pool's gate on a small stream and measure the pool (accuracy, experts, compute).
4. With your go: write a new question pool, freeze a pre-registration of this design, and run it once on fresh seeds.
5. A wider or second shared trunk for the experts, and a way to send teaching examples to Thin experts first.
