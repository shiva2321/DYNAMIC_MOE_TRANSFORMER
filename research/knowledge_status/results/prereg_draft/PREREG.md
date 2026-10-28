# Pre-registration of the Phase 1 confirmatory run (frozen, single use)

Everything in this file and in `bars.json` is fixed BEFORE any confirmatory seed runs. `bars.sha256` is the sha256 of the canonical JSON of `bars.json`; changing any bar, seed, pool file, code file or constant changes a hash. The confirmatory run is single use: it is run once, whatever it shows, and its result is reported as it comes.

## What is frozen

- Seeds: confirmatory 400 to 429 (30 seeds, never used before). Dev seeds 100 to 129 set the bars.
- Readers (representations): R5_R4_qwen_L26, R4_qwen_L26, R0_hash; headline reader `R5_R4_qwen_L26` (the frozen Qwen2.5-1.5B layer-26 state with a learned linear map trained on S1 to S3 teaching data only); the other two are controls.
- Calibration: held-out S1 to S3 parts only; headline level q = 0.05; the curve over q in [0.01, 0.02, 0.05, 0.1] is reported; group-test sizes [1, 4, 8, 16].
- Taught-state rule (`THRESHOLDS`): {"solid_min_acc": 0.9, "solid_min_count": 500, "thin_acc_margin": 0.1, "thin_max_acc": 0.6, "thin_max_count": 100}.
- Test messages: the confirmatory template pool `pool_confirm.py`, sha256 `85951b2b6ade867e534afd03b590b4c138748fa9f40919bceede43aa9a0a0012`.
- Code: stamp `dc015a0facb9766e` over 47 files (all of `research/knowledge_status/*.py` and the `growmoe` files they import); the runners must refuse to run if it differs.
- Sizes and baseline settings: see `bars.json` (`sizes`, `baseline_params`).
- How many dev seeds stand behind each bar is in the table (E1 ran on dev seeds 100 to 104 only, E4 on the seeds where addition and subtraction are really Solid, the AUROC for Thin against Solid only where a seed has both); a bar rests on fewer seeds where that column is small.

The confirmatory pool shares no template with the dev pool (identical normalised templates: 0); the share of its word 5-grams found in the dev pool is 0.028 overall and at most 0.098 in any one family (limit 0.2).

## The statements, their bars and what we expect

Bars come from the rule written before the dev run: a bar to stay above is max(draft, dev mean - 2 sd); a bar to stay below is min(draft, dev mean + 2 sd); a dev result can tighten a draft bar and never loosen it. The expected verdict is what the dev means give against the frozen bars; it is a prediction, not a result.

| measure | statement | bar | draft | dev mean (sd) [min - max], seeds | rule that gave the bar |
|---|---|---|---|---|---|
| b1a (at least) | B1a | 0.950 | 0.950 | 0.922 (0.007) [0.914 - 0.932], 5 | max(draft, mean - 2 sd) over the dev seeds |
| b1b (at least) | B1b | 0.991 | 0.950 | 0.997 (0.003) [0.992 - 1.000], 5 | max(draft, mean - 2 sd) over the dev seeds |
| b1c (at least) | B1c | 0.900 | 0.900 | 0.890 (0.003) [0.884 - 0.893], 5 | max(draft, mean - 2 sd) over the dev seeds |
| macro_f1 (at least) | B3a | 0.850 | 0.850 | 0.855 (0.079) [0.518 - 0.924], 30 | max(draft, mean - 2 sd) over the dev seeds |
| false_unseen_heldout (at most) | B3a | 0.050 | 0.050 | 0.289 (0.061) [0.175 - 0.433], 30 | min(draft, mean + 2 sd) over the dev seeds |
| thin_vs_solid_auroc (at least) | B3a | 0.850 | 0.850 | 0.844 (0.100) [0.595 - 0.981], 20 | max(draft, mean - 2 sd) over the dev seeds |
| other_as_unseen (at most) | B3a | 0.050 | 0.050 | 0.249 (0.055) [0.111 - 0.356], 30 | min(draft, mean + 2 sd) over the dev seeds |
| b4a_residual_hit (at least) | B4a | 0.850 | 0.850 | 0.660 (0.144) [0.150 - 0.850], 30 | max(draft, mean - 2 sd) over the dev seeds |
| b4a_known_not_flagged (at least) | B4a | 0.950 | 0.950 | 0.672 (0.087) [0.500 - 0.850], 30 | max(draft, mean - 2 sd) over the dev seeds |
| b4b_nothing_flagged (at least) | B4b | 0.950 | 0.950 | 0.643 (0.080) [0.475 - 0.775], 30 | max(draft, mean - 2 sd) over the dev seeds |
| b4b_other_labelled (at least) | B4b | 0.950 | 0.950 | 0.968 (0.030) [0.900 - 1.000], 30 | max(draft, mean - 2 sd) over the dev seeds |
| b5_mul_parent (at least) | B5 | 0.900 | 0.900 | 0.447 (0.116) [0.200 - 0.625], 10 | max(draft, mean - 2 sd) over the dev seeds |
| b5_div_parent (at least) | B5 | 0.900 | 0.900 | 0.508 (0.054) [0.400 - 0.625], 10 | max(draft, mean - 2 sd) over the dev seeds |
| b5_decoy_worst (at most) | B5 | 0.000 | 0.100 | 0.000 (0.000) [0.000 - 0.000], 10 | min(draft, mean + 2 sd) over the dev seeds |
| b6_gap (at most) | B6 | 0.100 | 0.100 | 0.178 (0.071) [0.025 - 0.300], 30 | min(draft, mean + 2 sd) over the dev seeds |
| b7_gap (at most) | B7 | 0.100 | 0.100 | no dev value | draft stands (no dev value exists for this measure) |
| b2_gap (at least) | B2 | 0.150 | 0.150 | no dev value | draft stands (no dev value exists for this measure) |

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
- Decision rule: pass when R0 (mean over the 30 seeds) is below the frozen B1a bar OR the frozen B1b bar by at least the frozen margin; a fail makes the whole run invalid.
- Measures: r0_b1a, r0_b1b, b2_gap.
- Expected verdict: **PASS**.

**B3a.** Statuses of arithmetic parts: macro F1 over Solid / Thin / Unseen; false 'Unseen' on the held-out wordings S4 to S8 of Solid skills at q = 0.05 with calibration on S1 to S3 only (the wording claim); Thin separated from Solid by AUROC; Other content called new arithmetic.
- Decision rule: pass only when ALL FOUR measures pass (macro F1 and AUROC at or above their bars, the two false-new rates at or below theirs); gated on the STANDARD calibration; the adapted condition is reported beside it and cannot pass the claim.
- Measures: macro_f1, false_unseen_heldout, thin_vs_solid_auroc, other_as_unseen.
- Expected verdict: **FAIL**. EXPECTED TO FAIL on the calibration claim: known skills in wordings that were not taught (above all short stories and noisy text) are called new far more often than the bar allows. Dev: the reader, not the detector, causes the shift (E-046; results/WORDING_LAB_STAGE1/2.md, FUNCTION_LAB.md, REWRITE_LAB.md); none of the repairs tried reached the bar. What is expected to hold is the part that finds NEW skills (see the unseen-found prediction below) and the Solid / Thin / Other reporting.

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
- Decision rule: pass when the gap is at or below the bar; NOT RUN in Phase 1 if no realism set exists (see Not run).
- Measures: b7_gap.
- Expected verdict: **NOT RUN**. NOT RUN: the realism set of Task 13 could not be built from the local data (GSM8K gives 0 to 11 one-step problems; no other labelled set is cached; nothing is downloaded, NG5). Stated here so that its absence is part of the registration.

## Predictions that are not bars

- Finding new skills stays high: the share of never-taught arithmetic parts called unseen is expected near its dev mean (0.894); no bar is registered for it.
- The 'Other content called new arithmetic' number is expected to be LOWER than on dev: in the confirmatory pool the S6 chatter sentences come from the same corpora as the taught Other content (the dev pool used a short hand-written list, a defect of the dev test bed found after the dev numbers were seen). The traps stay hand-written. It is still expected to be above its bar.
- The hashed word-matcher R0 fails B1a and B1b by a wide margin (dev: 0.54 and 0.01), which is what B2 needs.
- A confirmatory mean outside [dev min, dev max] widened by one dev standard deviation is reported as a surprise next to its verdict.

## Decision rules

1. Each statement is judged from the mean over the confirmatory seeds that have the measure (all 30, except where a seed cannot define it: Thin against Solid needs both kinds of part; the E4 traces need a seed where addition and subtraction are really Solid) against its frozen bar, by `verdict.py` (flip-tested), nothing else.
2. The wording claim is gated on the STANDARD calibration (S1 to S3 only). The adapted condition (a few S4 to S8 parts in the calibration) is reported, never gated: if only it reaches the bar, the claim reads 'needs adaptation to new wording'.
3. B3b (baselines) is reported with every claim; nothing is gated on it. Conflict, the offline consequence estimate and the group-test numbers are reported and not gated.
4. No bar, pool, seed, constant or code file is changed after the run; the run is not repeated; a failure is reported with its number.

## Kill rule and paper decision

- If B2 fails on the confirmatory seeds, the test bed is invalid for this run (anti-goal A1) and none of the other results counts.
- If B1b fails for the headline reader, that reader cannot see an operator change; no status claim built on it counts, and Phase 1 ends with that negative result (spec section 7).
- A statement whose bar is missed is a reported negative result, not a reason to re-run. Single use means exactly once.
- Paper rule (spec section 10): write a paper if the test bed and the information-ceiling analysis (C-a and C-b) are done on fresh seeds and the reading finds nothing that already does them; add a method claim (C-c status machine, C-d relation from traces) only for statements whose bars passed.

## Not run, stated in advance

- B7 (realism check): not run; see the note under B7. It is a limit of Phase 1 and will be listed as one.
- Conflict status: reported if it fires on its few cases, not gated (spec 5.6).

## Found after the dev numbers were seen (disclosed)

- The dev chatter defect described above; fixed in the confirmatory pool before the freeze, so dev and confirmatory numbers on chatter are not comparable.
- The wording finding (none of the repairs tried reached the bar, E-046) and the E4 prediction miss; both are why B3a and B5 are expected to fail. The bars were NOT changed because of them.
