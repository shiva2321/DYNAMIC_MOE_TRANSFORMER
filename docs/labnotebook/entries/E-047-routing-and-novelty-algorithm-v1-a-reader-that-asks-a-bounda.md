# E-047 - Routing and novelty algorithm v1: a reader that asks, a boundary question, teams of experts, group evidence and shared experts (labs A to E)

- **ID:** E-047
- **Date:** 2026-10-29
- **Track:** G
- **Type:** experiment
- **Status:** completed
- **Evidence grade:** B (script-generated; designed on the tuning seeds 150-154, reported on the dev seeds 100-109 that no choice used; every rule and prediction written in PROGRESS.md before the numbers; exploratory, NOT a frozen pre-registration and NOT the single-use seeds; no independent reviewer yet)
- **Code / commit:** branch `knowledge-status-phase1`, commits c76d689 to e4d0fc0; code in `research/knowledge_status/routing_lab/`, tests `tests/knowledge_status/test_rl_*.py`; worktree D:/shm_ks
- **People:** Claude designed, built and ran it overnight on the user's instruction ("competence first, wording next, several experts but only as many as reasonable and useful, sharing what matters, design and prove the routing algorithm"); the user parked the confirmatory run (seeds 400-429 unspent)

## Question
E-045 and E-046 found that the look-based status machine cannot keep a known skill known when its wording changes (0.256 of them called new on the tuning seeds, 0.98 of real problems), and that the two repairs tried (the function check, wording diversity in teaching) did not reach the bar. Can a reader that ASKS the language model which operation a message wants (instead of comparing embeddings) do it, what does it cost at the Other-content boundary, how many experts should answer a part, and does evidence over a group or shared experts help?

## Why we asked it
The user's goal is a routing and novelty algorithm for the expert pool: know what is new, which expert knows what and how well, and use one or several experts as far as it is reasonable. E-046 left the wording claim failed for the look-based reader; the user chose the competence route first and the wording route second, then added that routing may use more than one expert, only as many as are usable and helpful, with sharing.

## Prediction (written BEFORE the numbers)
In `research/knowledge_status/PROGRESS.md`, blocks of 2026-10-29 (plan; Lab A; the Lab A addendum with the several-experts design and one fix; Lab B; Lab C; Lab D and its mass rule; Lab E; and the report-seed predictions), each with a rule for going on. Results against the predictions are in the same file after each run. Where a prediction was wrong it says so.

## Method (how)
- **Lab A, the asking reader** (`research/knowledge_status/routing_lab/ask_reader.py`, `lab_ask.py`, `ask_rules.py`, `lab_ask_report.py`). Qwen2.5-1.5B-Instruct, chat template, options A-G = add, sub, mul, div, remainder, larger of two, none; one forward pass, the probabilities of the seven letters. The prompt is fixed before any run. Status decided in three ways: conformal (class-conditional, calibration on held-out S1-S3 parts only), the plain rule (unseen iff the top option is a skill nobody taught), tau0.5 (also unseen when the top probability is below 0.5; chosen offline on the tuning seeds by the recorded costs, 4 per false alarm and 7 per miss), and a hybrid (the calibrated none class first). Real problems (300 SVAMP, report only), the tangled messages (gold parts) and the look-based machine's own cells for the same seeds are in the same table.
- **Lab B, group evidence** (`group_check.py`, `lab_group.py`). Parts called unseen by look, taken in windows; a competent taught skill is present in a window if it explains more stated answers than answer-shuffled copies (2000 shuffles, Holm); mixed windows and same-skill windows; real solvers and perfect-solver ceilings.
- **Lab F, windows made from the asked operation** (`lab_group_asked.py`). The flagged parts are grouped by the operation the model named and cut into windows of g; same parts, machine, solvers and seeds as Lab B.
- **Lab C, shared experts** (`experts.py`, `lab_experts.py`). One shared two-layer number encoder, a head per skill, balanced sampling; sub, mul, div, mod, max taught 20, 100, 500, helpers 2000 each; held-out problems.
- **Lab D, teams** (`team.py`, `lab_team.py`). Team policies (top1, conformal set, top2, top3, all, adaptive by probability mass), with and without the stated answer, offline from the cached answers.
- **Lab E, the boundary question** (`lab_gate.py`, `lab_gate_report.py`). A second Yes / No question ("does this ask for a calculation?"), threshold calibrated on held-out taught arithmetic at level alpha, then tau0.5.
- No downloads (SVAMP was approved earlier), no bar changed, no edit of any stamped file or of `growmoe/`; the planted-bug sweeps found and closed three real test gaps (zero divisor, nearest-skill tie-break, `assess_by_rule`); all numbers below are copied from the generated files by script.

## Results (what came out)

### Lab A on the report seeds 100-109 (`research/knowledge_status/results/ROUTING_ASK_REPORT_100_109.md`)
| measure | look-based machine | tau0.5 (asks) | hyb0.5 (none class first) | conf0.05 (asks, calibrated) |
|---|---|---|---|---|
| known skills in new wordings called unseen | 0.279 | 0.025 | 0.008 | 0.206 |
| never-taught skills found | 0.866 | 0.974 | 0.703 | 0.712 |
| macro F1 (Thin / Solid labels come from the assignment here: read it as identification only) | 0.842 | 0.975 | 0.816 | 0.844 |
| everyday sentences called unseen | 0.247 | 0.709 | 0.002 | 0.006 |
| look-alike traps called unseen | 0.418 | 0.602 | 0.300 | 0.590 |
| near-twins labelled right | 0.942 | 0.982 | 0.889 | 0.914 |
| tangled: new part found (B4a residual hit) | 0.667 | 0.975 | 0.700 | 0.637 |
| tangled: known part left alone (B4a) | 0.702 | 0.975 | 0.990 | 0.830 |
| tangled: nothing flagged when known + Other (B4b) | 0.660 | 0.270 | 0.982 | 0.823 |
| tangled: Other part labelled Other (B4b) | 0.978 | 0.003 | 0.995 | 0.995 |
| real problems: Solid skills called unseen (B7) | 0.997 | 0.138 | 0.069 | 0.651 |
| real problems: never-taught skills found | 0.825 | 0.855 | 0.509 | 0.578 |

### Lab E on the report seeds 100-109 (`research/knowledge_status/results/ROUTING_GATE_REPORT_100_109.md`)
| measure | look-based machine | tau0.5 alone | two-stage, alpha 0.05 | two-stage, alpha 0.01 |
|---|---|---|---|---|
| known skills in new wordings called unseen | 0.279 | 0.025 | 0.023 | 0.024 |
| never-taught skills found | 0.866 | 0.974 | 0.907 | 0.956 |
| everyday sentences called unseen | 0.247 | 0.709 | 0.092 | 0.208 |
| look-alike traps called unseen | 0.418 | 0.602 | 0.060 | 0.360 |
| everyday sentences labelled Other | 0.746 | 0.002 | 0.863 | 0.701 |
| look-alike traps labelled Other | 0.582 | 0.000 | 0.908 | 0.500 |
| near-twins labelled right | 0.942 | 0.982 | 0.898 | 0.966 |
| tangled: new part found (B4a residual hit) | 0.667 | 0.975 | 0.825 | 0.915 |
| tangled: known part left alone (B4a) | 0.702 | 0.975 | 0.982 | 0.975 |
| tangled: nothing flagged when known + Other (B4b) | 0.660 | 0.270 | 0.872 | 0.733 |
| tangled: Other part labelled Other (B4b) | 0.978 | 0.003 | 0.863 | 0.653 |
| tangled: new part found next to Other (B4c) | 0.790 | 0.992 | 0.835 | 0.948 |
| real problems: Solid skills called unseen (B7) | 0.997 | 0.138 | 0.138 | 0.138 |
| real problems: never-taught skills found | 0.825 | 0.855 | 0.851 | 0.855 |

### Lab E on the tuning seeds (`research/knowledge_status/results/ROUTING_GATE_150_154.md`)
| | tau0.5 alone | two-stage, alpha 0.01 | two-stage, alpha 0.05 |
|---|---|---|---|
| known skills in new wordings called unseen | 0.053 | 0.053 | 0.048 |
| never-taught skills found | 0.990 | 0.984 | 0.965 |
| look-alike traps called unseen | 0.600 | 0.347 | 0.100 |
| everyday sentences called unseen | 0.460 | 0.207 | 0.090 |
| everyday sentences labelled Other | 0.001 | 0.616 | 0.813 |
| traps labelled Other | 0.000 | 0.420 | 0.807 |
| real arithmetic in new wordings wrongly called Other | | 0.005 | 0.023 |
| never-taught arithmetic wrongly called Other | | 0.006 | 0.026 |
| traps called Other | | 0.420 | 0.807 |
| everyday sentences called Other | | 0.616 | 0.813 |

### Lab D, teams (`research/knowledge_status/results/ROUTING_TEAM_100_109.md`; tuning seeds in `ROUTING_TEAM_150_154.md`)
| experts allowed (team and verifiers) | policy | recall | size | right, no answer | right, with the stated answer |
|---|---|---|---|---|---|
| every taught skill | top1 | 0.970 | 1.00 | 0.970 | 0.970 |
| every taught skill | set | 0.753 | 0.75 | 0.753 | 0.753 |
| every taught skill | top2 | 0.998 | 2.00 | 0.970 | 0.997 |
| every taught skill | top3 | 1.000 | 3.00 | 0.970 | 0.997 |
| every taught skill | all | 1.000 | 3.00 | 0.970 | 0.997 |
| every taught skill | mass0.9 | 0.980 | 1.02 | 0.970 | 0.979 |
| every taught skill | mass0.99 | 0.988 | 1.06 | 0.970 | 0.986 |
| only the Solid skills | top1 | 0.981 | 1.00 | 0.981 | 0.981 |
| only the Solid skills | set | 0.729 | 0.73 | 0.729 | 0.729 |
| only the Solid skills | top2 | 1.000 | 1.33 | 0.981 | 1.000 |
| only the Solid skills | top3 | 1.000 | 1.33 | 0.981 | 1.000 |
| only the Solid skills | all | 1.000 | 1.33 | 0.981 | 1.000 |
| only the Solid skills | mass0.9 | 0.983 | 1.00 | 0.981 | 0.983 |
| only the Solid skills | mass0.99 | 0.990 | 1.01 | 0.981 | 0.990 |

### Lab B, group evidence (`research/knowledge_status/results/ROUTING_GROUP_150_154.md`)
| solvers | windows | g, alpha | false-new | found | macro F1 | coincidences (untaught taken back) | right skill | flagged known parts rescued |
|---|---|---|---|---|---|---|---|---|
| real | mixed | 4, 0.05 | 0.256 | 0.903 | 0.866 | 0.000 | nan | 0.000 |
| real | mixed | 6, 0.01 | 0.256 | 0.903 | 0.866 | 0.000 | nan | 0.000 |
| real | mixed | 8, 0.01 | 0.246 | 0.901 | 0.867 | 0.002 | 1.000 | 0.029 |
| real | mixed | 16, 0.01 | 0.231 | 0.893 | 0.865 | 0.010 | 0.900 | 0.071 |
| real | same_skill | 4, 0.05 | 0.236 | 0.903 | 0.871 | 0.000 | 1.000 | 0.057 |
| real | same_skill | 6, 0.01 | 0.231 | 0.902 | 0.872 | 0.001 | 1.000 | 0.071 |
| real | same_skill | 8, 0.01 | 0.236 | 0.903 | 0.871 | 0.000 | 1.000 | 0.057 |
| real | same_skill | 16, 0.01 | 0.226 | 0.897 | 0.870 | 0.006 | 1.000 | 0.086 |
| perfect_solid | mixed | 4, 0.05 | 0.256 | 0.903 | 0.866 | 0.000 | nan | 0.000 |
| perfect_solid | mixed | 6, 0.01 | 0.256 | 0.900 | 0.864 | 0.003 | nan | 0.000 |
| perfect_solid | mixed | 8, 0.01 | 0.243 | 0.896 | 0.744 | 0.007 | 1.000 | 0.039 |
| perfect_solid | mixed | 16, 0.01 | 0.191 | 0.865 | 0.689 | 0.038 | 0.864 | 0.204 |
| perfect_solid | same_skill | 4, 0.05 | 0.256 | 0.903 | 0.866 | 0.000 | nan | 0.000 |
| perfect_solid | same_skill | 6, 0.01 | 0.190 | 0.899 | 0.826 | 0.004 | 1.000 | 0.203 |
| perfect_solid | same_skill | 8, 0.01 | 0.153 | 0.896 | 0.706 | 0.007 | 0.892 | 0.318 |
| perfect_solid | same_skill | 16, 0.01 | 0.151 | 0.854 | 0.695 | 0.049 | 0.750 | 0.330 |
| perfect_taught | mixed | 4, 0.05 | 0.256 | 0.903 | 0.866 | 0.000 | nan | 0.000 |
| perfect_taught | mixed | 6, 0.01 | 0.256 | 0.902 | 0.803 | 0.001 | nan | 0.000 |
| perfect_taught | mixed | 8, 0.01 | 0.223 | 0.889 | 0.690 | 0.014 | 0.729 | 0.123 |
| perfect_taught | mixed | 16, 0.01 | 0.178 | 0.851 | 0.677 | 0.051 | 0.964 | 0.261 |
| perfect_taught | same_skill | 4, 0.05 | 0.256 | 0.903 | 0.866 | 0.000 | nan | 0.000 |
| perfect_taught | same_skill | 6, 0.01 | 0.133 | 0.903 | 0.764 | 0.000 | 1.000 | 0.429 |
| perfect_taught | same_skill | 8, 0.01 | 0.073 | 0.892 | 0.700 | 0.010 | 1.000 | 0.606 |
| perfect_taught | same_skill | 16, 0.01 | 0.020 | 0.865 | 0.698 | 0.038 | 0.993 | 0.853 |

### Lab F, group evidence with windows made from the asked operation (`research/knowledge_status/results/ROUTING_GROUP_ASKED_150_154.md`)
| solvers | g, alpha | false-new | found | coincidences | right skill |
|---|---|---|---|---|---|
| real | 6, 0.01 | 0.241 | 0.903 | 0.000 | 1.000 |
| real | 8, 0.01 | 0.236 | 0.900 | 0.003 | 1.000 |
| real | 16, 0.01 | 0.228 | 0.896 | 0.007 | 1.000 |
| perfect_solid | 6, 0.01 | 0.179 | 0.903 | 0.000 | 1.000 |
| perfect_solid | 8, 0.01 | 0.170 | 0.895 | 0.008 | 1.000 |
| perfect_solid | 16, 0.01 | 0.157 | 0.863 | 0.040 | 1.000 |
| perfect_taught | 6, 0.01 | 0.153 | 0.903 | 0.000 | 1.000 |
| perfect_taught | 8, 0.01 | 0.093 | 0.899 | 0.004 | 1.000 |
| perfect_taught | 16, 0.01 | 0.064 | 0.853 | 0.050 | 1.000 |

### Lab C, shared experts (`research/knowledge_status/results/ROUTING_EXPERTS_150_152.md`)
| skill | taught | separate | shared | helpers (shared net) |
|---|---|---|---|---|
| sub | 20 | 0.02 | 0.02 | add 0.84, max 0.91 |
| sub | 100 | 0.04 | 0.15 | add 0.82, max 0.92 |
| sub | 500 | 0.17 | 0.41 | add 0.78, max 0.92 |
| mul | 20 | 0.05 | 0.10 | add 0.78, sub 0.62, max 0.93 |
| mul | 100 | 0.10 | 0.15 | add 0.76, sub 0.54, max 0.92 |
| mul | 500 | 0.18 | 0.31 | add 0.73, sub 0.50, max 0.91 |
| div | 20 | 0.41 | 0.47 | add 0.78, sub 0.65, max 0.93 |
| div | 100 | 0.46 | 0.52 | add 0.76, sub 0.59, max 0.93 |
| div | 500 | 0.53 | 0.59 | add 0.74, sub 0.63, max 0.94 |
| mod | 20 | 0.18 | 0.17 | add 0.78, sub 0.61, max 0.93 |
| mod | 100 | 0.35 | 0.31 | add 0.76, sub 0.54, max 0.92 |
| mod | 500 | 0.52 | 0.52 | add 0.76, sub 0.54, max 0.94 |
| max | 20 | 0.15 | 0.16 | add 0.88, sub 0.75 |
| max | 100 | 0.44 | 0.55 | add 0.79, sub 0.61 |
| max | 500 | 0.79 | 0.78 | add 0.80, sub 0.65 |

### Lab A on the tuning seeds, the first design (`research/knowledge_status/results/ROUTING_ASK_150_154.md`) and the rule comparison (`research/knowledge_status/results/ROUTING_ASK_RULES_150_154.md`)
| measure | conformal (calibrated on S1-S3) | plain naming rule |
|---|---|---|
| known skills in new wordings S4-S8 called unseen (bar 0.05) | 0.145 | 0.052 |
| never-taught skills found | 0.826 | 0.989 |
| known skills in the taught wordings S1-S3 called unseen | 0.047 | |
| macro F1 (Solid / Thin / Unseen) | 0.891 | |
| Thin against Solid, AUROC | 1.000 | |
| everyday sentences called unseen | 0.003 | |
| look-alike traps called unseen | 0.603 | |

| rule | false-new S4-S8 | never-taught found | cost | traps unseen | everyday unseen | real: taught called unseen | real: never-taught found |
|---|---|---|---|---|---|---|---|
| plain | 0.052 | 0.989 | 0.287 | 0.530 | 0.415 | 0.167 | 0.907 |
| tau0.3 | 0.052 | 0.989 | 0.287 | 0.530 | 0.417 | 0.167 | 0.907 |
| tau0.5 | 0.052 | 0.990 | 0.280 | 0.600 | 0.460 | 0.177 | 0.908 |
| tau0.7 | 0.072 | 0.991 | 0.347 | 0.657 | 0.635 | 0.234 | 0.920 |
| tau0.9 | 0.090 | 0.993 | 0.407 | 0.657 | 0.868 | 0.373 | 0.944 |
| conf0.05 | 0.145 | 0.826 | 1.800 | 0.603 | 0.003 | 0.515 | 0.629 |
| conf0.02 | 0.090 | 0.821 | 1.613 | 0.500 | 0.002 | 0.385 | 0.559 |
| conf0.01 | 0.063 | 0.770 | 1.860 | 0.427 | 0.001 | 0.311 | 0.422 |
| hyb0.0 | 0.020 | 0.822 | 1.327 | 0.397 | 0.002 | 0.075 | 0.590 |
| hyb0.5 | 0.020 | 0.822 | 1.327 | 0.397 | 0.002 | 0.075 | 0.590 |
| hyb0.7 | 0.023 | 0.822 | 1.340 | 0.397 | 0.002 | 0.084 | 0.590 |

## Interpretation (what we understood, and on what basis)
**Plain words.** Asking beats looking. When the language model is asked "which operation is this?" it gets the operation right 96 % of the time even in stories and with typos, so a known skill written in a new way stays known (called new 2.3-2.5 % on fresh seeds, against 28 % for the look-based machine), a skill nobody taught is found 91-97 % of the time (87 %), real human-written problems of a known skill are called new 14 % of the time (99.7 %), and a message with a known and an unknown part is read part by part: the known part is left alone 98 % of the time (70 %), the unknown part is found 83-98 % of the time (67 %).
**What asking cannot do alone.** It names an operation even for sentences that are not arithmetic (everyday sentences called new 71 %, look-alike traps 60 % with the plain rule). A second question, asked on its own, "does this ask for a calculation at all?", with its threshold set on arithmetic requests only, repairs most of it (everyday 9 %, traps 6 %, alpha 0.05) at a price: no single threshold gives both a clean boundary and the best detection of the new part of a tangled message (alpha 0.05: boundary better, new part 0.825; alpha 0.01: new part 0.915, boundary worse).
**Why the calibrated version lost.** Calibrating on the easy taught wordings makes the model look certain (probability near 0.99), so the 5 % threshold sits near 0.99 and every slightly less certain answer in a new wording is flagged. The plain rule, which uses only what the model names, does not have that problem; it has no statistical guarantee, which is a cost to remember.
**How many experts.** One expert is right 97 % of the time. A team sized by the probability the model holds (add the next most probable expert until 99 % of the probability on the taught skills is held, at most 3) reaches 98.6 % with 1.06 experts per part; a fixed team of two reaches 99.7 % with 2.0 experts, a team of three adds nothing. Extra experts help ONLY when something checks them (the stated answer, i.e. a worked example, a teacher, a stream's feedback): without it the team answers with its most probable member and extra experts are cost. The conformal covering set is the wrong way to form a team (right 0.75: it is empty for a quarter of the known parts). This is the reasonable-and-helpful rule the user asked for: more than one expert when the part is ambiguous or there is a check, one otherwise, none when the part is new.
**Evidence over a group.** The permutation test removes random coincidences but the gain depends on the windows being made of one skill: with perfect experts it takes the false alarm to 0.020 in same-skill windows of 16 and only to 0.178 in the mixed windows a stream gives. Windows made from the operation the model names (Lab F) are 96 % pure and give 0.093 (g = 8) and 0.064 (g = 16), most of the way to the ceiling; with the real experts it is still about nothing (0.228). Left over, and not predicted: coincidences are systematic for small operands (mul(a, 1) = max(a, 1)), up to 0.050 at g = 16, which the shuffled-answer null cannot see.
**Sharing.** Experts that share a number encoder help the weak ones (sub 0.17 to 0.41 and mul 0.18 to 0.31 at 500 examples, max 0.44 to 0.55 at 100) and cost the strong ones (add 0.93 alone, 0.73-0.88 shared with one trunk of width 128). A skill taught 20 examples stays unusable either way. Competence comes from examples; for a Thin expert the routing's job is to send it more teaching examples, not to trust it earlier.
**Technical statement, measured against assumed.** Measured: everything in the tables, on dev seeds. Assumed: that the dev test bed (templates S1-S8 and 300 SVAMP problems) represents what the system will read; that the model's prior knowledge of the operation vocabulary is an acceptable stand-in for reading (it is what makes a never-taught skill detectable and it cannot recognise a skill outside the vocabulary); that reading the two numbers is a separate part (the verification and team rows take the operands as given).

## Caveats and what went wrong
- The declared rule for the first design (conformal at q = 0.05) FAILED on the tuning seeds (false alarm 0.145 against 0.12, real problems 0.515 against 0.35); the plain rule that works was seen afterwards (post hoc) and was therefore re-tested, with predictions written first, on fresh report seeds, where it held.
- The report seeds 100-109 are dev seeds: their look-based cells existed for earlier work, and the design was made on 150-154 only, but they are not the single-use confirmatory seeds and the new design has not been pre-registered. A confirmatory test of THIS design needs a new pool, a new freeze and the user's go.
- Real-problem numbers were visible while the rules were compared, though the choice used only the synthetic cost. SVAMP covers four operations.
- My predictions were wrong several times: naming accuracy too pessimistic; everyday sentences and the boundary far worse than predicted for the plain rule; the tangled new part lower than predicted with the gate; Lab B too optimistic for mixed windows; Lab C's sub and the helpers' loss. `all` equals `top3` while only three skills are taught, so "too many experts hurt" could not be tested.
- Thin-against-Solid is not measured by the asking reader (the status's Thin / Solid comes from the expert's measured accuracy and count); the AUROC 1.000 and the macro F1 in the tuning tables of Lab A are artefacts of reading Thin / Solid off the assignment: not quotable.
- Mod is the weakest named skill (0.87). The asking reader reads two forward passes of a 1.5 B model per part (the look-based reader one); a small head distilled from the asked answers is the obvious way to get one pass back (not done).
- Not integrated with the pool: nothing here shows the pool routes better with this status; `growmoe/` was not touched (spec NG1).
- A mutation sweep harness started in the background was mistaken for finished once; I restored one file from git while it was still running, which cost nothing but could have, and is recorded here.

## Consequence
The routing / novelty algorithm is specified in `research/knowledge_status/ROUTING_ALGORITHM_2026-10-29.md`. Next, in order: (1) a null that keeps the operand structure for the group check (systematic coincidences, Lab F); (2) distil the asked answers into a small head so the reader costs one pass; (3) wire the status and team into the pool's gate on a small stream and measure the pool; (4) pre-register a confirmatory run of the new design with a new pool and new seeds, if the user gives the go; (5) a wider trunk or a second trunk for the shared experts. Registers: F-40 to F-44 in FAILURES.md, C-91 to C-97 in CLAIMS.md.

## Evidence
- `research/knowledge_status/results/ROUTING_ASK_REPORT_100_109.md`, `ROUTING_GATE_REPORT_100_109.md`, `ROUTING_TEAM_100_109.md`, `ROUTING_GROUP_150_154.md`, `ROUTING_GROUP_ASKED_150_154.md`, `ROUTING_EXPERTS_150_152.md`, `ROUTING_ASK_150_154.md`, `ROUTING_ASK_RULES_150_154.md`, `ROUTING_GATE_150_154.md`, `ROUTING_TEAM_150_154.md`
- `research/knowledge_status/results/routing_lab/` (per-seed cells, cached model answers, logs)
- `research/knowledge_status/PROGRESS.md` (plan, predictions, results, in order)

## Reproduce
```
PYTHONPATH=<repo> python research/knowledge_status/routing_lab/lab_ask.py 150 154
PYTHONPATH=<repo> python research/knowledge_status/routing_lab/ask_rules.py 150 154
PYTHONPATH=<repo> python research/knowledge_status/routing_lab/lab_ask_report.py 100 109
PYTHONPATH=<repo> python research/knowledge_status/routing_lab/lab_gate.py 150 154
PYTHONPATH=<repo> python research/knowledge_status/routing_lab/lab_gate_report.py 100 109
PYTHONPATH=<repo> python research/knowledge_status/routing_lab/lab_team.py 100 109
PYTHONPATH=<repo> python research/knowledge_status/routing_lab/lab_group.py 150 154
PYTHONPATH=<repo> python research/knowledge_status/routing_lab/lab_group_asked.py 150 154
PYTHONPATH=<repo> python research/knowledge_status/routing_lab/lab_experts.py 150 152
```
