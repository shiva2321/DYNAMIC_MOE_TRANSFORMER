# E-046 - Knowledge status Phase 1 - why known skills in new wordings are called new, the repairs that did not reach the bar, and the relation-from-traces dev result

- **ID:** E-046
- **Date:** 2026-10-28
- **Track:** G
- **Type:** experiment
- **Status:** completed
- **Evidence grade:** B (script-generated, tuning seeds 150-154 for the wording work and dev seeds for E4, every rule and prediction written in PROGRESS.md before the numbers; exploratory, not a frozen pre-registration; no independent reviewer yet)
- **Code / commit:** branch `knowledge-status-phase1`; 3f92b25 / dc7b040 / de869b3 (investigation and mitigations lab), fb25e31 / 7491817 / 3339169 (function check), 98a82c6 / fadef29 (rewrites and the E4 runner), 7f152bf (traces); worktree D:/shm_ks
- **People:** Claude designed, built and ran; the user chose to investigate (2026-10-28, "investigate why and how, then see what solutions and mitigations we can do"), chose the order (function check first, then wording diversity in teaching) and then accepted the result as the Phase 1 finding ("yes agree, carry on with task 14"); no independent reviewer yet

## Question
E-045 found that the status machine finds skills that were never taught but calls known skills "new" when they are written in a wording that was not taught (stories and noisy text above all). Is the cause the reader or the detector? Can a repair bring the false-new rate on S4 to S8 to the bar (0.05) without losing the found rate? And does the relation check from worked traces (B5) work on the same machine?

## Why we asked it
The calibration claim (spec 6.2) fails on dev (E-045), and a status machine that cannot trust a known skill in a new wording is of little use as the self-knowledge part of the later routing system. The user asked for the reason and for mitigations before more was built.

## Prediction (written BEFORE the numbers)
Written in `research/knowledge_status/PROGRESS.md` before each step (see the dated blocks of 2026-10-28), with a pass mark: a repair is kept only if the false-new rate falls by at least 0.05 while the found rate falls by at most 0.05 (the rule for the clean rewrite arms), and for E4 the prediction was 0.85 to 0.97 of genuine traces with the parent exactly right and at most 0.02 false relations on decoys. Where my predictions missed, PROGRESS.md says so and this entry repeats it below.

## Method (how)
- Why (`diag_shift.py`, tuning seeds 150-154): stage by stage (raw Qwen layer 26, centred, learned map) how far each wording sits below the taught wordings in standard deviations of the best cosine to its own skill's examples; whether the detector or the reader is responsible.
- Mitigations lab (`lab_wording.py`, stage 1 reader variants, stage 2 machine variants): clean-up of case and spacing, two closing prompts, leave-one-wording-out calibration, a margin gate.
- Function check (`function_check.py`, `lab_function.py`, `lab_function_ceiling.py`): read the two numbers of a part, ask the taught solvers for the answer under both operand orders, and decide from agreement; single example and batches; the ceiling with perfect solvers.
- Wording diversity in teaching (`rewrites.py`, `lab_rewrites.py`): Qwen writes rewrites of teaching sentences from four generic instructions, kept only if the numbers are the same and Qwen's own answer equals the true answer; arms: map only, full (also prototypes and calibration), and a CONTAMINATED ceiling where the writer is told to produce stories, typos and registers (diagnostic only).
- E4 (`traces.py`, `run_e3.py`): worked traces of multiplication (repeated addition) and division (repeated subtraction) in four wordings, three decoy kinds; a step counts only if the machine assigns it to the skill as Solid AND the taught solver reproduces its result; the chain must reach the stated answer. Seeds where addition and subtraction are really Solid, 40 traces per kind per seed.
- No downloads, no bar changes; all numbers below are copied from the generated files by script.

## Results (what came out)

### Why the alarm fires (raw file, copied)
```
Why do new wordings set off the alarm?  tuning seeds 150-154, status machine q = 0.05, standard condition

Q1. Stage by stage: how far each wording sits below the taught wordings, in standard deviations of the best cosine to the own skill's examples (0 = as close as a taught wording; negative = farther)
    wording          raw Qwen   plain (centred)   learned-map reader
    S1                 +0.07        +0.09            -0.19
    S2                 -0.14        -0.17            -0.03
    S3                 +0.07        +0.08            +0.19
    S4                 -5.61        -5.50            -5.91
    S5                 -0.90        -1.45            -0.62
    S6                 -0.17        -0.17            +0.04
    S7                 -3.26        -3.60           -33.59
    S8                 +0.09        +0.04            -1.29
    noise:case         -2.96        -3.56           -30.31
    noise:space        -0.49        -0.54            -1.08
    noise:typo         -1.82        -2.33           -18.63

    known skills called unseen, and nearest group right:   plain reader | learned-map reader
    S1              n=  85   unseen   1.2% |   5.9%    nearest right  98.8% | 100.0%
    S2              n=  89   unseen   6.7% |   3.4%    nearest right  98.9% | 100.0%
    S3              n=  96   unseen   3.1% |   1.0%    nearest right  99.0% |  97.9%
    S4              n=  73   unseen  97.3% |  67.1%    nearest right  82.2% |  89.0%
    S5              n=  45   unseen  31.1% |  22.2%    nearest right 100.0% | 100.0%
    S6              n=  83   unseen   6.0% |   7.2%    nearest right  98.8% |  97.6%
    S7              n=  78   unseen  59.0% |  44.9%    nearest right  70.5% |  73.1%
    S8              n=  81   unseen   2.5% |   4.9%    nearest right  95.1% |  96.3%
    noise:case      n= 240   unseen  65.4% |  61.7%    nearest right  66.7% |  66.7%
    noise:space     n= 255   unseen   9.8% |   9.8%    nearest right  96.1% |  96.9%
    noise:typo      n= 202   unseen  39.6% |  30.7%    nearest right  87.1% |  81.2%

Q3. ORACLE (looks at the test wording, a diagnostic and not a fix): false alarm after removing the top k shift directions of that wording, learned-map reader
    S4:  k=0: top  62.4%   k=1: top  53.6% / random  69.1%   k=2: top  53.6% / random  74.6%   k=4: top  53.6% / random  75.4%   k=8: top  53.6% / random  80.0%
    S7:  k=0: top  45.0%   k=1: top  60.1% / random  49.6%   k=2: top  66.7% / random  70.3%   k=4: top  82.0% / random  66.7%   k=8: top  82.0% / random  96.0%
    S5:  k=0: top  18.3%   k=1: top  13.3% / random  50.0%   k=2: top  20.0% / random  33.3%   k=4: top  20.0% / random  53.7%   k=8: top  20.0% / random  85.0%

Q4. AUROC of 'known skill in this wording' (0) against 'never-taught skill in the same wording' (1); 0.5 is chance
    wording  closeness (best cosine)              kernel density                       margin (nearest minus second neare
    S1                                    0.971                                0.982                                0.783
    S2                                    0.983                                0.990                                0.828
    S3                                    0.986                                0.988                                0.850
    S4                                    0.793                                0.815                                0.830
    S5                                    0.991                                0.998                                0.917
    S6                                    0.978                                0.979                                0.797
    S7                                    0.738                                0.770                                0.689
    S8                                    0.946                                0.971                                0.786

Q5. Length
    S1              words median   3.0   false alarm   5.9%
    S2              words median   5.0   false alarm   3.4%
    S3              words median   6.0   false alarm   1.0%
    S4              words median  14.0   false alarm  67.1%
    S5              words median   5.0   false alarm  22.2%
    S6              words median   4.0   false alarm   7.2%
    S7              words median   6.0   false alarm  44.9%
    S8              words median   4.0   false alarm   4.9%
    noise:case      words median   5.0   false alarm  61.7%
    noise:space     words median   5.0   false alarm   9.8%
    noise:typo      words median   5.0   false alarm  30.7%
    taught wordings S1-S3: Spearman(words, best cosine) +0.11;  false alarm by length third (short/middle/long):   5.0% /   2.5% /   0.0%
    held-out wordings S4-S8: Spearman(words, best cosine) -0.45;  false alarm by length third (short/middle/long):  11.7% /  14.2% /  63.7%
```

### Mitigations lab, stage 1: reader variants
# Mitigations lab, stage 1 (reader variants, tuning seeds 150-154)

### Reader variants, machine as it is

| reader | false alarm S4-S8 | stories S4 | noisy text S7 | swapped order S5 | taught wordings | never-taught skills found | macro F1 | Other called unseen | passed by the margin gate |
|---|---|---|---|---|---|---|---|---|---|
| plain | 0.256 | 0.624 | 0.450 | 0.183 | 0.037 | 0.903 | 0.866 | 0.275 | 0 |
| clean | 0.250 | 0.639 | 0.376 | 0.217 | 0.031 | 0.890 | 0.861 | 0.232 | 0 |
| p1 | 0.333 | 0.933 | 0.476 | 0.133 | 0.062 | 0.869 | 0.821 | 0.348 | 0 |
| p2 | 0.309 | 0.772 | 0.517 | 0.125 | 0.052 | 0.877 | 0.828 | 0.368 | 0 |
| clean+p1 | 0.251 | 0.730 | 0.322 | 0.133 | 0.053 | 0.863 | 0.843 | 0.314 | 0 |
| clean+p2 | 0.261 | 0.741 | 0.297 | 0.133 | 0.050 | 0.872 | 0.838 | 0.331 | 0 |

**Winner by the declared rule: `plain`**

### Mitigations lab, stage 2: machine variants
# Mitigations lab, stage 2 (machine variants, tuning seeds 150-154)

### Machine variants on the `plain` reader (the stage-1 winner; the choice is made here)

| machine | false alarm S4-S8 | stories S4 | noisy text S7 | swapped order S5 | taught wordings | never-taught skills found | macro F1 | Other called unseen | passed by the margin gate |
|---|---|---|---|---|---|---|---|---|---|
| as is | 0.256 | 0.624 | 0.450 | 0.183 | 0.037 | 0.903 | 0.866 | 0.275 | 0 |
| lofo | 0.252 | 0.624 | 0.433 | 0.183 | 0.033 | 0.891 | 0.858 | 0.275 | 0 |
| margin 0.10 | 0.063 | 0.113 | 0.185 | 0.000 | 0.013 | 0.250 | 0.511 | 0.195 | 210 |
| margin 0.25 | 0.068 | 0.130 | 0.185 | 0.000 | 0.013 | 0.292 | 0.542 | 0.204 | 198 |
| margin 0.50 | 0.114 | 0.195 | 0.286 | 0.000 | 0.018 | 0.722 | 0.785 | 0.266 | 66 |
| lofo + margin 0.25 | 0.068 | 0.130 | 0.185 | 0.000 | 0.013 | 0.284 | 0.536 | 0.204 | 196 |

**Winner by the declared rule: `as is`**

### Function check
# Function-check lab (tuning seeds 150-154, learned-map reader, status machine q = 0.05)

A part called unseen by its look is overturned when a COMPETENT taught skill (Solid state, solver at 0.90 on unseen problems) reproduces the stated answer from the numbers in the message.

| | before | after |
|---|---|---|
| known skills in new wordings called unseen (bar 0.05) | 0.256 | 0.223 |
| ... of competent skills (add, sub, max when Solid) | 0.300 | 0.000 |
| ... of the other taught skills (thin, weak solver) | 0.266 | 0.261 |
| never-taught skills found | 0.903 | 0.889 |
| macro F1 | 0.866 | 0.864 |

- wording S4: called unseen 0.624 -> 0.563
- wording S5: called unseen 0.183 -> 0.142
- wording S6: called unseen 0.039 -> 0.039
- wording S7: called unseen 0.450 -> 0.393
- wording S8: called unseen 0.048 -> 0.048

- share of the known test parts that belong to competent skills (the most the check can rescue): 0.100
- when a known part is overturned, the explaining skill is the right one: 0.923
- never-taught parts wrongly explained by a taught skill (coincidence): 0.014
- two numbers found in the message: S1 1.000, S2 1.000, S3 1.000, S4 1.000, S5 1.000, S6 1.000, S7 1.000, S8 1.000
- competent skills per seed: 150: max; 151: none; 152: none; 153: none; 154: none

### Function check, ceiling with perfect solvers
# Ceiling of the function check with perfect solvers (tuning seeds 150-154)

| | known skills called unseen (bar 0.05) | stories S4 | noisy text S7 | never-taught skills found | macro F1 | right skill when overturned | coincidences |
|---|---|---|---|---|---|---|---|
| no function check | 0.256 | n/a | n/a | 0.903 | 0.866 | n/a | n/a |
| i: perfect for Solid-state skills | 0.104 | 0.269 | 0.188 | 0.749 | 0.656 | 0.610 | 0.154 |
| ii: perfect for every taught skill | 0.000 | 0.000 | 0.000 | 0.739 | 0.655 | 0.957 | 0.164 |
| i, batches of 4 (3 of 4 must agree) | 0.213 | 0.477 | 0.387 | 0.885 | 0.696 | 1.000 | 0.018 |
| ii, batches of 4 (3 of 4 must agree) | 0.174 | 0.397 | 0.295 | 0.882 | 0.696 | 0.907 | 0.021 |

### Wording diversity in teaching (rewrites)
# Rewrite lab (tuning seeds 150-154, learned-map reader on Qwen layer 26, status machine q = 0.05)

| arm | known skills in new wordings called unseen (bar 0.05) | stories S4 | noisy text S7 | register S8 | never-taught skills found | macro F1 | rewrites kept (teaching) |
|---|---|---|---|---|---|---|---|
| nothing (the machine as it is) | 0.256 | 0.624 | 0.450 | 0.048 | 0.903 | 0.866 | 0 |
| map only | 0.262 | 0.656 | 0.462 | 0.017 | 0.762 | 0.783 | 348 per seed |
| full | 0.232 | 0.559 | 0.435 | 0.017 | 0.740 | 0.776 | 348 per seed |
| ceiling (CONTAMINATED DIAGNOSTIC) | 0.144 | 0.307 | 0.310 | 0.017 | 0.860 | 0.871 | 324 per seed |

The ceiling arm tells the writer to produce stories, typos and registers, so its numbers on those wordings are not a wording-invariance claim.

### E4, relation from worked traces (dev seeds)
# E4, relation from worked traces (learned-map reader; 10 of 15 candidate seeds had addition and subtraction really Solid)

B5: the parent set of a multiplication trace is exactly {add}, of a division trace exactly {sub}, in at least 0.90 of the traces; a decoy gets a false relation in at most 0.10.

| trace | seeds | traces per seed | parent exactly right | any parent named | chain valid | steps verified | steps the machine calls Solid | bar |
|---|---|---|---|---|---|---|---|---|
| mul (genuine) | 10 | 40 | 0.447 | 0.447 | 1.000 | 0.211 | 0.223 | at least 0.90: fail |
| div (genuine) | 10 | 40 | 0.508 | 0.508 | 1.000 | 0.444 | 0.457 | at least 0.90: fail |
| mul, decoy wrong_steps | 10 | 40 | n/a | 0.000 | 0.000 | 0.194 | 0.217 | at most 0.10: PASS |
| div, decoy wrong_steps | 10 | 40 | n/a | 0.000 | 0.000 | 0.292 | 0.450 | at most 0.10: PASS |
| max, decoy irrelevant_valid_steps | 10 | 40 | n/a | 0.000 | 0.000 | 0.106 | 0.120 | at most 0.10: PASS |
| mod, decoy unrelated_skill | 10 | 40 | n/a | 0.000 | 0.000 | 0.044 | 0.044 | at most 0.10: PASS |

Genuine traces by wording (0 symbols, 1 verbal, 2 number words, 3 story), share with the parent exactly right:
- mul: 0: 0.060, 1: 0.780, 2: 0.890, 3: 0.060
- div: 0: 0.020, 1: 1.000, 2: 0.960, 3: 0.050

## Interpretation (what we understood, and on what basis)
Plain words: the reader (Qwen layer 26 plus the learned map) puts a known skill written as a short story or with typos far from where the taught wordings sit, so a detector that asks "is this near what I was taught?" has to say no. The detector is doing its job; the distance it is given is wrong for those wordings. Repairs that clean the text, change the prompt, change the calibration, or ask the solvers whether the numbers give the claimed answer either do not move the false alarm enough or only work where the solver is competent, which is rare in the small-solver setting. Wording diversity in teaching helps only when the writer is TOLD the held-out wordings (the contaminated ceiling, not a clean result), and generic rewrites cost found skills. So the calibration claim fails for this reader and the honest Phase 1 finding is: it finds new skills, it reports Solid / Thin / Other, and it does not yet trust a known skill in a wording it was not taught.

For E4 the check "does the trace really compute this?" is sound (no decoy was ever given a parent; every genuine chain was valid), but "is this step a known skill?" fails for the symbol steps ("4 + 4 = 8") and for story steps, so no parent is named there; verbal and number-word steps work. Measured: the split by wording above. Not tested: that the cause is the answer being written in the step and stories being a distrusted wording; the cheap test is the same traces with the step written as a bare request.

Technical statement: the shift is a property of the reader's geometry (stage-by-stage shifts in the why-file), not of the conformal calibration; none of the repairs on the tuning seeds met the pre-declared rule; one contaminated arm halves the false-new rate but stays at three times the bar; B5 on dev: genuine rows below the 0.90 bar, decoys within the 0.10 bar.

## Caveats and what went wrong
- Tuning and dev seeds only (150-154 for the wording work, 100-129 for E2, the E4 seeds listed in its table): exploratory, not confirmatory; the freeze (E-047) and the single-use run decide the claims.
- My predictions missed in three places, and PROGRESS.md records each: closing prompts hurt instead of helping; generic rewrites cost a visible share of the found rate (not predicted; see the rewrite table); the E4 genuine rate is half of what I predicted, and its weak wordings were symbols and stories, not number words.
- The planted-bug checks found test gaps repeatedly (status/baselines, evaluate/segment, run_e2, function check, wording, rewrites, traces); each was closed with a targeted test, only equivalent mutants remain.
- The dev chatter list was hand-written and not drawn from the corpora that supply the taught "Other content" (spec 5.2): a defect of the test bed found after the numbers were seen; it is fixed in the confirmatory pool, not in dev.
- Realism (B7) could not be built from the local data (see PROGRESS.md and E-047).

## Consequence
The finding was accepted by the user as the Phase 1 result (2026-10-28). Tasks 14 to 16 follow: confirmatory pool and frozen pre-registration (E-047), the single-use run, the report and an independent review. Phase 2 inherits two concrete problems: the reader's distrust of stories and noisy text, and traces whose steps carry the answer. A from-scratch reader arm (Task 11b) is still owed by the user's earlier decision.

## Evidence
- `research/knowledge_status/PROGRESS.md` - dated blocks with every prediction, rule and verdict
- `research/knowledge_status/results/E2_WHY.txt`, `WORDING_LAB_STAGE1.md`, `WORDING_LAB_STAGE2.md`, `FUNCTION_LAB.md`, `FUNCTION_CEILING.md`, `REWRITE_LAB.md`, `E4_TABLE.md`
- `research/knowledge_status/results/rewrite_lab/`, `function_lab/`, `wording_lab/`, `e4/` - per-seed cells

## Reproduce
```
cd research/knowledge_status
PYTHONPATH=<repo root> python diag_shift.py
PYTHONPATH=<repo root> python lab_wording.py
PYTHONPATH=<repo root> python lab_function.py
PYTHONPATH=<repo root> python lab_rewrites.py
PYTHONPATH=<repo root> python run_e3.py --candidates 100-159 --count 10 --per-kind 40
```
