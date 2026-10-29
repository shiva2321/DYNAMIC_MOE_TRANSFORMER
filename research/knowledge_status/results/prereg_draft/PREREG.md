# Pre-registration of the Phase 1 confirmatory run (frozen, single use)

Everything in this file and in `bars.json` is fixed BEFORE any confirmatory seed runs. `bars.sha256` is the sha256 of the canonical JSON of `bars.json`, and `PREREG.md` must be exactly what `bars.json` renders to (the runners check both); changing any bar, seed, pool file, code file or constant changes a hash. The confirmatory run is single use: it is run once, through `run_confirm.py` and nothing else, whatever it shows, and its result is reported as it comes.

## What is frozen

- Seeds: confirmatory 400 to 429 (30 seeds, never used before). Dev seeds 100 to 129 set the bars.
- Readers (representations): R5_R4_qwen_L26, R4_qwen_L26, R0_hash. Headline reader `R5_R4_qwen_L26` (the frozen Qwen2.5-1.5B layer-26 state with a learned linear map trained on S1 to S3 teaching data only); `R4_qwen_L26` (the same state without the map) and `R0_hash` (hashed word pieces, the word-matcher control) are controls. The from-scratch arms of Task 11b take part only if the rule of D-40 keeps them on dev (see below). Verdicts are made on the headline reader; the other readers are reported.
- Calibration: held-out S1 to S3 parts only; headline level q = 0.05; the curve over q in [0.01, 0.02, 0.05, 0.1] is reported; group-test sizes [1, 4, 8, 16].
- Taught-state rule (`THRESHOLDS`): {"solid_min_acc": 0.9, "solid_min_count": 500, "thin_acc_margin": 0.1, "thin_max_acc": 0.6, "thin_max_count": 100}.
- Test messages: the confirmatory template pool `pool_confirm.py`, sha256 `257e56f31fb04f5286f55605037fd202acb72679a293a7818a43742a9e8dc6a4`.
- Realism set (B7): `results/realism_set.jsonl` (SVAMP one-step problems), sha256 `6f1582f212e12372ecd0b529e54cdd9f5758adb99f6a1cce61cdbbe21bdcf812`.
- Code: stamp `ff3b592e1c3af520` over 52 files (all of `research/knowledge_status/*.py` and the `growmoe` files they import); the runners must refuse to run if it differs.
- Sizes and baseline settings: see `bars.json` (`sizes`, `baseline_params`). A measure needs at least 10 confirmatory seeds that define it, otherwise it is not judged.
- How many dev seeds stand behind each bar is in the table (E1 ran on dev seeds 100 to 104 only; E4 on the seeds where addition and subtraction are really Solid; the AUROC for Thin against Solid only where a seed has both kinds of part; B7 only where a seed has a Solid and an Unseen skill among add, sub, mul, div); a bar rests on fewer seeds where that column is small.

The confirmatory pool shares no template with the dev pool (identical normalised templates: 0); the share of its word 5-grams found in the dev pool is 0.028 overall and at most 0.098 in any one family (limit 0.2).

Message length as a cue (anti-goal A1, no model; `check_length_leak.py`): accuracy of guessing the skill from the length of the request / of the whole message alone, inside one wording family, dev pool against confirmatory pool (chance 0.167, or 0.333 in S5). Some leak is natural; the confirmatory pool is somewhat higher in a few families (S8, S1, S4, S7 on the request length) and the check is within one family only. The real test of surface cues is B2.

| family | dev: request, whole | confirmatory: request, whole |
|---|---|---|
| S1 | 0.303, 0.210 | 0.367, 0.283 |
| S2 | 0.268, 0.174 | 0.252, 0.177 |
| S3 | 0.500, 0.317 | 0.530, 0.407 |
| S4 | 0.487, 0.334 | 0.548, 0.263 |
| S5 | 0.453, 0.412 | 0.473, 0.399 |
| S6 | 0.270, 0.180 | 0.247, 0.185 |
| S7 | 0.353, 0.298 | 0.396, 0.268 |
| S8 | 0.280, 0.227 | 0.403, 0.273 |

## The statements, their bars and what we expect

Bars come from the rule written before the dev run: a bar to stay above is max(draft, dev mean - 2 sd); a bar to stay below is min(draft, dev mean + 2 sd); a dev result can tighten a draft bar and never loosen it. The rule was written for the status measures (E2 and E3); it was applied to the ceiling (E1), the traces (E4) and the realism check (B7) after their dev numbers were known, which is disclosed below. The expected verdict is what the dev means give against the frozen bars; it is a prediction, not a result.

| measure | statement | bar | draft | dev mean (sd) [min - max], seeds | expected | rule that gave the bar |
|---|---|---|---|---|---|---|
| b1a (at least) | B1a | 0.950 | 0.950 | 0.922 (0.007) [0.914 - 0.932], 5 | FAIL | max(draft, mean - 2 sd) over the dev seeds |
| b1b (at least) | B1b | 0.991 | 0.950 | 0.997 (0.003) [0.992 - 1.000], 5 | PASS | max(draft, mean - 2 sd) over the dev seeds |
| b1c (at least) | B1c | 0.900 | 0.900 | 0.890 (0.003) [0.884 - 0.893], 5 | FAIL | max(draft, mean - 2 sd) over the dev seeds |
| b2_gap (at least) | B2 | 0.150 | 0.150 | no dev value | PASS | draft stands (no dev value exists for this measure) |
| macro_f1 (at least) | B3a | 0.850 | 0.850 | 0.855 (0.079) [0.518 - 0.924], 30 | PASS | max(draft, mean - 2 sd) over the dev seeds |
| false_unseen_heldout (at most) | B3a | 0.050 | 0.050 | 0.289 (0.061) [0.175 - 0.433], 30 | FAIL | min(draft, mean + 2 sd) over the dev seeds |
| thin_vs_solid_auroc (at least) | B3a | 0.850 | 0.850 | 0.844 (0.100) [0.595 - 0.981], 20 | FAIL | max(draft, mean - 2 sd) over the dev seeds |
| other_as_unseen (at most) | B3a | 0.050 | 0.050 | 0.249 (0.055) [0.111 - 0.356], 30 | FAIL | min(draft, mean + 2 sd) over the dev seeds |
| trap_as_unseen (at most) | B3a | 0.050 | 0.050 | 0.472 (0.131) [0.200 - 0.733], 30 | FAIL | min(draft, mean + 2 sd) over the dev seeds |
| b4a_residual_hit (at least) | B4a | 0.850 | 0.850 | 0.660 (0.144) [0.150 - 0.850], 30 | FAIL | max(draft, mean - 2 sd) over the dev seeds |
| b4a_known_not_flagged (at least) | B4a | 0.950 | 0.950 | 0.672 (0.087) [0.500 - 0.850], 30 | FAIL | max(draft, mean - 2 sd) over the dev seeds |
| b4b_nothing_flagged (at least) | B4b | 0.950 | 0.950 | 0.643 (0.080) [0.475 - 0.775], 30 | FAIL | max(draft, mean - 2 sd) over the dev seeds |
| b4b_other_labelled (at least) | B4b | 0.950 | 0.950 | 0.968 (0.030) [0.900 - 1.000], 30 | PASS | max(draft, mean - 2 sd) over the dev seeds |
| b5_mul_parent (at least) | B5 | 0.900 | 0.900 | 0.447 (0.116) [0.200 - 0.625], 10 | FAIL | max(draft, mean - 2 sd) over the dev seeds |
| b5_div_parent (at least) | B5 | 0.900 | 0.900 | 0.508 (0.054) [0.400 - 0.625], 10 | FAIL | max(draft, mean - 2 sd) over the dev seeds |
| b5_decoy_worst (at most) | B5 | 0.000 | 0.100 | 0.000 (0.000) [0.000 - 0.000], 10 | PASS | min(draft, mean + 2 sd) over the dev seeds |
| b6_gap (at most) | B6 | 0.100 | 0.100 | 0.178 (0.071) [0.025 - 0.300], 30 | FAIL | min(draft, mean + 2 sd) over the dev seeds |
| b7_gap (at most) | B7 | 0.100 | 0.100 | 0.325 (0.044) [0.248 - 0.428], 13 | FAIL | min(draft, mean + 2 sd) over the dev seeds |

### Statements (spec section 7), decision rule and expected verdict

**B1a.** Wording invariance: a linear probe for skill identity (six skills plus none), trained on S1 to S3, is right on the held-out wordings S4 to S8 (chance 1/7).
- Decision rule: pass when the mean over the 30 confirmatory seeds is at least the frozen bar; the headline reader is R5_R4_qwen_L26.
- Measures: b1a.
- Expected verdict: **FAIL**.

**B1b.** Near-twin separation: by distance alone, parts of the same skill are closer than a part and its near-twin (same sentence, other operator); AUROC (chance 0.5).
- Decision rule: pass when the mean over the 30 seeds is at least the frozen bar.
- Measures: b1b.
- Expected verdict: **PASS**.

**B1c.** Taught against untaught: the distance to the nearest taught prototype separates parts of taught skills from parts of untaught skills; AUROC (chance 0.5).
- Decision rule: pass when the mean over the 30 seeds is at least the frozen bar.
- Measures: b1c.
- Expected verdict: **FAIL**.

**B2.** Validity of the test bed: the word-matcher control R0 falls short of the B1a bar or the B1b bar by at least 0.15; otherwise the test bed is broken and no result counts (anti-goal A1).
- Decision rule: pass when R0 (mean over the 30 seeds) is below the spec's B1a bar OR the spec's B1b bar (the draft bar, or the frozen bar if that is lower) by at least the frozen margin; a fail makes the whole run invalid.
- Measures: r0_b1a, r0_b1b, b2_gap.
- Expected verdict: **PASS**.

**B3a.** Statuses of arithmetic parts: macro F1 over Solid / Thin / Unseen; false 'Unseen' on the held-out wordings S4 to S8 of Solid skills at q = 0.05 with calibration on S1 to S3 only (the wording claim; the number measured covers the TAUGHT skills, Solid and Thin, not only Solid as the spec bar reads); Thin separated from Solid by AUROC; Other content called new arithmetic, everyday sentences and look-alike traps each against the bar.
- Decision rule: pass only when ALL FIVE measures pass (macro F1 and AUROC at or above their bars, the three false-new rates at or below theirs); gated on the STANDARD calibration; the adapted condition is reported beside it and cannot pass the claim.
- Measures: macro_f1, false_unseen_heldout, thin_vs_solid_auroc, other_as_unseen, trap_as_unseen.
- Expected verdict: **FAIL**. EXPECTED TO FAIL on the calibration claim: known skills in wordings that were not taught (above all short stories and noisy text) are called new far more often than the bar allows. Dev: the reader, not the detector, causes the shift (E-046; results/WORDING_LAB_STAGE1/2.md, FUNCTION_LAB.md, REWRITE_LAB.md); none of the repairs tried reached the bar. Per measure on dev (table above): the false-new rate and the traps miss their bars widely, Thin against Solid and macro F1 sit on the bar with a wide spread (a coin flip), the everyday-sentences number is expected to move (see Predictions). What is expected to hold is the finding of NEW skills (see Predictions), not the reporting of Solid / Thin / Other.

**B3b.** The baselines of spec 6.2 (vMF conformal, k-nearest neighbour, Mahalanobis, Relative Mahalanobis, classifier confidence, energy, autoencoder descriptor, group test) are reported on the same cases with the same calibration, so every claim has its control. Reported, not gated.
- Decision rule: reported only; no gate.
- Measures: none (reported).
- Expected verdict: **REPORTED**.

**B4a.** A known part plus a new arithmetic part (gold spans): the left-over part is the truly unseen one, and the known part is not flagged.
- Decision rule: pass only when both numbers are at or above their bars (gold spans).
- Measures: b4a_residual_hit, b4a_known_not_flagged.
- Expected verdict: **FAIL**. Expected to fail for the same reason: a known part written in a new wording is flagged, so the 'known part not flagged' number is low.

**B4b.** An arithmetic part plus Other content (gold spans): nothing is flagged as new arithmetic, and the Other part is labelled as Other.
- Decision rule: pass only when both numbers are at or above their bars (gold spans).
- Measures: b4b_nothing_flagged, b4b_other_labelled.
- Expected verdict: **FAIL**. Expected to fail: the 'nothing flagged' number is low on dev (table above) for the same wording reason, while 'Other part labelled Other' is near its bar.

**B5.** Relation from worked traces: the parent set of a multiplication trace is exactly {add} and of a division trace exactly {sub}; decoy traces get a false relation rarely.
- Decision rule: pass only when the multiplication rate and the division rate are at or above their bars AND the worst decoy rate is at or below its bar.
- Measures: b5_mul_parent, b5_div_parent, b5_decoy_worst.
- Expected verdict: **FAIL**. Expected to FAIL on the genuine traces (the dev means are in the table above; my prediction of 0.85 to 0.97, written before the dev run, missed) and to PASS on the decoys. By wording on dev (results/E4_TABLE.md): verbal and number-word traces work, symbol traces ('4 + 4 = 8') and story traces almost never. Likely cause (not tested): the machine does not call those step texts Solid, a new-wording problem again. The confirmatory pool writes its S1 requests differently, so the symbol rows may move; this is stated, not predicted. The decoy bar is 0.000 because every dev decoy rate was exactly 0 (the rule min(draft, mean + 2 sd) leaves no room when there is no spread); a single false relation in the confirmatory run would fail that measure, and the genuine rows are expected to fail B5 regardless.

**B6.** With automatic segmentation (the sentence / clause cutter, chosen on dev) the two B4a numbers stay within 0.10 of the gold-span numbers (the larger of the two gaps).
- Decision rule: pass when the larger gap between the automatic and the gold-span B4a numbers is at or below the bar.
- Measures: b6_gap.
- Expected verdict: **FAIL**. Expected to fail: the dev mean of the larger gap (table above) is well over the bar.

**B7.** Realism: the Solid-against-Unseen result on a real-text set agrees with the template result within 0.10.
- Decision rule: pass when the gap is at or below the bar; not_run if the realism set could not be scored (see the Realism section).
- Measures: b7_gap.
- Expected verdict: **FAIL**. Realism: scored on the one-step problems of the realism set (Task 13); see the Realism section of this file for the set and the rule.

## Realism check (B7)

One-step arithmetic word problems written by people (SVAMP, MIT licence), downloaded once with the user's permission, restricted to problems whose reference equation is one operation on two numbers of the problem; 300 rows (add 75, div 75, mul 74, sub 76; SVAMP has no remainder or maximum problems and only a few dozen usable multiplication problems), drawn with seed 100 and fixed in `results/realism_set.jsonl`, whose hash the runner checks before it starts. No code other than the realism runner loads SVAMP, and the runner refuses any realism problem found among teaching, calibration or adaptation texts (`realism.assert_not_realism`). Per seed the same machine judges (a) the template parts of the held-out wordings S4 to S8 and (b) the realism problems; on each the balanced accuracy of the call 'Unseen or not' is taken over parts whose skill is Solid (should not be called Unseen) and parts whose skill is Unseen (should be); the template side holds the same four operations. B7 is the absolute value of the mean difference (realism minus templates) over the seeds that have both a Solid and an Unseen skill among add, sub, mul, div. A 50-row spot check of the labels (`results/realism_spotcheck.md`) was read by an independent reviewer: 48 of 50 clearly right, two doubtful, none wrong.
Two confounds are named, not removed: a realism problem is a whole story with two to five numbers, a template part is one request, and about a quarter of the problems have an operand above 99 (the template bed uses one or two digits). A row restricted to problems whose operands are both at most 99 is reported next to B7 and not gated. Only about 4 in 10 dev seeds can be scored (13 of the 30 dev seeds), so there is a real chance (about 1 in 10 by a binomial estimate) that fewer than the minimum number of confirmatory seeds can be scored, in which case B7 is not judged.

## From-scratch reader arms (Task 11b, user decision D-40)

No pretrained model. `R5_FS_strict`: a hashed bag of word pieces, a small network trained contrastively on the teaching data only. `R5_FS_pre`: the same network first pre-trained without labels on everyday sentences from the local corpora, then trained the same way. Operating points were chosen on tuning seeds 150 to 154 (results/FS_TUNING.md). The rule of D-40, applied to the dev means: within 0.05 of the headline reader on B1b, B1c, macro F1 and false-new = kept; within 0.15 = parked (not deleted) until the user decides the whole system is from scratch; further = not worth it now. Only a kept arm is run in the confirmatory run (measured and reported, never gated); an arm that is not kept is not run there, and its dev result below is the finding.

| arm | status on dev | worst gap | B1b | B1c | macro F1 | false-new | never-taught skills found |
|---|---|---|---|---|---|---|---|
| R5_FS_pre | not worth it now | 0.425 | 0.859 | 0.686 | 0.429 | 0.008 | 0.008 |
| R5_FS_strict | not worth it now | 0.478 | 0.798 | 0.586 | 0.376 | 0.015 | 0.027 |

The gap counts only being worse than the headline reader (a better value is no gap). A low false-new rate here is not a fix of the wording problem: these arms call almost nothing new, as the last column shows (the headline reader finds 0.894).

## Predictions that are not bars

- Finding new skills stays high: the share of never-taught arithmetic parts called unseen is expected near its dev mean (0.894); no bar is registered for it.
- 'Other content called new arithmetic' (everyday sentences) is expected to move DOWN from its dev value: on dev the chatter parts were cut together with the frame words around them and the chatter list was hand-written; in the confirmatory pool the S6 frames put only whitespace around the chatter sentence, so the parts scored as Other are exactly sentences from the corpora that supply the taught Other content. It may or may not reach its bar; the traps number is a separate measure and is expected to stay far above its bar.
- The hashed word-matcher R0 is expected to fail B1a and B1b by a wide margin (dev means 0.543 and 0.003 against the draft bars 0.95 and 0.95), which is what B2 needs. B2 can show that the test bed does not let a word-matcher pass B1; it cannot show that no cue exists in S4 to S8.
- A confirmatory mean outside [dev min, dev max] widened by one dev standard deviation is reported as a surprise next to its verdict.

## Decision rules

1. Each statement is judged from the mean over the confirmatory seeds that have the measure (all 30, except where a seed cannot define it: Thin against Solid needs both kinds of part; the E4 traces need a seed where addition and subtraction are really Solid; B7 needs a Solid and an Unseen skill among add, sub, mul, div; fewer than the minimum number of seeds means the measure is not judged) against its frozen bar, by `verdict.py` (flip-tested), nothing else.
2. The wording claim is gated on the STANDARD calibration (S1 to S3 only). The adapted condition (a few S4 to S8 parts in the calibration) is reported, never gated: if only it reaches the bar, the claim reads 'needs adaptation to new wording'.
3. B3b (baselines) is reported with every claim; nothing is gated on it. The offline consequence estimate, the group-test numbers, the per-wording B5 rows, the small-operand B7 row and the information-ceiling readers that are run for E1 only are reported and not gated.
4. No bar, pool, seed, constant or code file is changed after the run; the run is not repeated; a failure is reported with its number. `run_confirm.py` is the only entry point; the development runners refuse the confirmatory pool.

## Kill rules and paper decision (computed by `verdict.consequences`, printed at the top of the result)

- B2 fails: the test bed is invalid for this run (anti-goal A1) and none of the other results counts.
- B1b is below the spec's draft bar (or the frozen bar if that is lower) for the headline reader: that reader cannot see an operator change; no status claim built on it counts and Phase 1 ends with that negative result (spec section 7).
- B1a or B1c fails for the headline reader: the spec says a representation that fails B1 is not used for status. The user decided (D-40) to carry the reader on regardless; therefore every status claim (B3a to B7) is reported as CONDITIONAL on a reader that missed B1, and a pass of such a claim is not a spec pass. B1a and B1c are expected to fail (dev), so the result is expected to carry the CONDITIONAL banner.
- A statement whose bar is missed is a reported negative result, not a reason to re-run. Single use means exactly once.
- Paper rule (spec section 10): write a paper if the test bed and the information-ceiling analysis (C-a and C-b) are done on fresh seeds and the reading finds nothing that already does them; add a method claim (C-c status machine, C-d relation from traces) only for statements whose bars passed.

## Not run, stated in advance

- The Conflict condition of spec 5.6 (a rule change on a taught skill) was not built; only `flag_conflict` is unit-tested. Nothing about Conflict is reported.
- The information-ceiling ladder beyond the readers run here (the E1-only readers listed above) is dev-only evidence; C-b on fresh seeds covers exactly those readers.
- The spec 5.6 count sweep (one skill taught 5, 20, 100, 500, 2000 examples) is not part of this run.
- A5 (right on templates and wrong on real phrasing) is checked only by B7, on one real-text source and four operations.
- The open THRESHOLDS question: on dev about a quarter of the skills taught as 'Solid' (2000 examples) are not solid by measured accuracy (multiplication, division and remainder are hard for the small solver); the thresholds stay at the draft values and the status truth is taken from the measured accuracy.

## Found after the dev numbers were seen (disclosed)

- The dev chatter defect: the dev S6 chatter was a short hand-written list, not drawn from the corpora of the taught Other content, and the dev scorer cut the Other parts together with the frame words. The confirmatory pool fixes both (corpus sentences, whitespace-only frames). Dev and confirmatory numbers on chatter and on Other content are not comparable.
- The wording finding (none of the repairs tried reached the bar, E-046) and the E4 prediction miss; both are why B3a and B5 are expected to fail. The bars were NOT changed because of them.
- The confirmatory S1 forms were written after the E4 diagnosis; the S1 form '{a} + {b} = ?' is close to the symbol trace step '4 + 4 = 8', and the confirmatory S2 has no 'equals' form that matches the number-word step. So the per-wording B5 rows may reshuffle (symbol rows up, number-word rows down) for template reasons, not for relation reasons. The trace wordings and the separators of tangled messages are the dev ones, shared by both pools.
- The bar rule was extended to E1, E4 and B7 after their dev numbers were known; the decoy bar is 0.000 because every dev decoy rate was exactly 0.
- Measure definitions differ from the spec's words in two places: the false-new rate on held-out wordings is measured over TAUGHT skills (Solid and Thin), and 'Other content' is two measures, everyday sentences and look-alike traps, each against the bar.
- At the part level S6 and S8 parts are verbatim S1 to S3 request templates with held-out numbers (their dev false-new rates sit near the taught-wording rate), and some S5 test parts repeat a taught problem with its numbers swapped; the 'held-out wordings' S4 to S8 are therefore not all equally held out (see E2_DIAG.txt).
