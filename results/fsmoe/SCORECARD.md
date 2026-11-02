# FS-MoE prediction scorecard (report seeds R [710, 711, 712, 713, 714], S [710, 711, 712, 713, 714], T [710, 711, 712, 713, 714])

15 declared predictions: **4 held, 11 missed**. Each is a predicate on seed means, written in the design before the cells that test it (docs/superpowers/specs/2026-10-31-fsmoe-design.md, sections 8-8c).

| id | prediction | outcome | measured |
|---|---|---|---|
| R1 | raw look-based detectors, naive calibration: false flag on known skills in held-out wordings >= 0.20 in arithmetic and code | HELD | min 0.319 |
| R2 | the learned map alone does not bring that below 0.15 in any domain | HELD | min 0.420 |
| R3 | stream calibration: false flag within 0.04-0.08 in all domains AND best power for withheld (near) skills 0.3-0.7 | MISSED | false flag 0.057-0.083; best near power per domain 0.126, 0.217, 0.136 |
| R4 | fast-slow disagreement alone: AUROC >= 0.80 in every domain and never more than 0.03 above the best look-based score | MISSED | arith 0.650 vs 0.597; code 0.688 vs 0.716; lang 0.553 vs 0.621 |
| R5 | near-twin withheld skills found <= 0.50 by every single signal (code, language) AND far skills >= 0.90 by the best signal | MISSED | best near 0.217; best far code 0.930, language 0.306 |
| R6 | group test g = 16: far withheld >= 0.90 at false alarm <= 0.06 and near-twins <= 0.60 (code, language) | MISSED | far 1.000, 0.588; lowest false alarm 0.070, 0.059; near 0.365, 0.187 |
| R7 | calibration on verified outcomes controls the false flag better than teaching-wording calibration in arithmetic and code, and lands within 0.02 of q | MISSED | stream 0.083, 0.067 vs teach 0.929, 0.958 |
| R8 | the declared bars fail in at least two of three domains for every single signal and the best fusion | HELD | signals/fusions passing the bars per domain: {'arith': 0, 'code': 0, 'lang': 0} |
| S1 | the LDA-blended pool ends within 0.03 of the joint network on taught skills (teaching wording) and 0.05 on held-out wordings, forgetting <= 0.03, within +-0.03 of equal-memory replay and kNN, 3-12 experts | MISSED | joint-FS gap tw -0.004, ho -0.073; worst forgetting 0.000; FS minus replay/kNN tw 0.013, 0.031; experts 3.0 |
| S2 | 5 % budget: novelty-ranked precision >= 0.28 (twice the novel share), learned fraction >= 0.70 under novelty and <= 0.40 under random; none learns zero | MISSED | precision novelty 0.264 vs random 0.132; learned fraction novelty 0.100, random 0.079, oracle 0.171 |
| S3 | adaptive team within +-0.02 of top-1 on held-out wordings, mean size <= 1.3 | HELD | team minus top-1 -0.0032; mean size 1.02 |
| S4 | held-out-wording accuracy of known skills RISES during deployment under every policy that verifies known items, more under novelty-ranked than random | MISSED | rise random -0.098, novelty -0.125, oracle -0.056 (oracle verifies no known item) |
| T1 | result signal finds withheld regimes >= 0.90 / >= 0.70 / <= 0.50 at noise 0.05 / 0.3 / 1.0 (6 teaching surfaces), AUROC >= 0.98 at 0.05, look finds <= 0.15 | MISSED | found 0.875 / 0.893 / 0.558; AUROC at 0.05 0.920; look at most 0.085 |
| T2 | the cascade (look shortlists one expert, result judges it) costs <= 0.4 of result_all and loses <= 0.05 of its power | MISSED | cost ratio 0.21; power cascade 0.057 vs result_all 0.893 (noise 0.3) |
| T3 | the stuck-then-grow rule widens on every case, deepens at least once on the composed cases and never on the simple ones, ends within 1.5x of the best fixed net's error (composed cases) | MISSED | widen on all four: False; deepen on both composed: False; deepen on a simple one: False; error ratio to best fixed 1.499, 1.505 (declared threshold 0.2) |
