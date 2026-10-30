# E-051 - General routing layer: no calibrator protects the cut from wrongly verified messages for free; contamination mostly moves the operating point, the plain cut fails safe (lab G4c)

- **ID:** E-051
- **Date:** 2026-10-30
- **Track:** G
- **Type:** experiment
- **Status:** completed for the tuning seeds (G4c and its post-hoc analysis); the growmoe experts as container and self-discovery of cards are not built
- **Evidence grade:** B (script-generated on the tuning seeds 150-152; rules C11-C13 and every prediction written in PROGRESS.md before the numbers; the matched-false-flag analysis is POST-HOC with its own prediction written before it was computed; exploratory, NOT a frozen pre-registration, NOT the single-use seeds; no independent reviewer yet)
- **Code / commit:** branch `knowledge-status-phase1`, commits ffe77a9 (plan and predictions), d78dcdc (code and tests), fa488d9 and 77e6f4b (results, post-hoc); code in `research/knowledge_status/routing_general/` (`robust.py`, `lab_robust.py`, `posthoc_g4c.py`), tests `tests/knowledge_status/test_rg_robust.py` (20, mutation-checked: 7 of 8 mutants caught, the survivor is an equivalent change to the random-stream tag of the injected modes) and `test_rg_posthoc_g4c.py` (13); worktree D:/shm_ks
- **People:** Claude built and ran it on the user's go of 2026-10-30 ("yes run G4c")

## Question
E-050 left one weakness open: a few never-taught messages that the outcome check wrongly vouches for sit in the verified calibration sample and, because the cut of a 30-message sample is its largest score, one of them can set it (about a third of the power lost, C-105). Can the cut be protected, by dropping the largest calibration scores ("trim k"), by requiring a second opinion ("agree": the asking reader's own top option must be an expert the outcome names), or by a larger sample, and what does the protection cost in false flags?

## Why we asked it
The user chose G4c after E-050. Reading the G4 tables first showed a property of my own fusion that I had not stated (F-53): the either-fusion of two readers flags about the SUM of their rates, so a one-rank trim costs the pair about twice what it costs one reader.

## Prediction (written BEFORE the numbers, if there was one)
In `research/knowledge_status/PROGRESS.md`, block "Lab G4c ... plan and predictions written BEFORE the run" (commit ffe77a9). C11 (arithmetic, real answer check: some calibrator among trim1, trim2, trim3, agree has both known-flagged rates at most 0.10 and keeps at least 0.85 of its own clean power under contamination): predicted to pass with 40 %; failed. C12 (single readers in the text domains flag within 0.03 of (k+1)/(n+1)): predicted 85 %; failed by 0.003 in two cells, because my formula is only right while n - k <= 38 (F-54). C13 (one calibrator passes the C11 test in all four domains under 10 % injected contamination): predicted 5 %; failed. The post-hoc block "Does contamination move the cut along one curve?" was written before that analysis: 1 of 4 predictions held.

## Method (how)
- **Calibrators** (`lab_robust.py`, `robust.py`): all applied per reader to the same verified sample, then the either-fusion of the pair (asking reader, vMF): plain (E-050), trim k for k = 1, 2, 3, 6 (drop the k largest scores), agree. Sample sizes 30, 40, 60 (arithmetic has about 33 verified messages, so its larger rows are not larger samples).
- **Contamination**: real (arithmetic's answer check on a stream with a quarter never-taught messages, as G4b); injected 10 % and 20 % (never-taught messages added as if verified so that they make that share of the sample, in every domain; content-independent, which the real coincidences are not). Paired: the clean arm is the same random order without the wrong messages; 20 draws; the random streams of G4b are reused, so plain in the real mode reproduces it (it does, to three decimals).
- **Post-hoc** (`posthoc_g4c.py`, reads the cached cells): the clean curve of found against known flagged is drawn through the clean points of every calibrator, mode and sample size (corners added, linear in between); a contaminated point is read against it ("lost at matched false flags"). For one reader every calibrator is judged on the same messages, so its points belong to one curve; for the pair the curve is approximate.
- Seeds 150-152, cached asked answers, no download, `growmoe/` untouched, no bar changed after a number.

## Results (what came out)
Numbers copied by script from `research/knowledge_status/results/ROUTING_GENERAL_G4c_150_152.md` and `ROUTING_GENERAL_G4c_posthoc_150_152.md` (3 seeds, 20 paired draws; tables at n = 30, either-fusion of the pair; retention = contaminated found / clean found of the same calibrator).

### arithmetic, mode real, n = 30

| calibrator | n in sample | never-taught share | clean: known flagged | found | contaminated: known flagged | found | retention |
|---|---|---|---|---|---|---|---|
| plain | 29.8 | 0.091 | 0.051 | 0.248 | 0.036 | 0.171 | 0.689 |
| trim1 | 29.8 | 0.091 | 0.101 | 0.432 | 0.077 | 0.279 | 0.645 |
| trim2 | 29.8 | 0.091 | 0.145 | 0.596 | 0.114 | 0.416 | 0.697 |
| trim3 | 29.8 | 0.091 | 0.175 | 0.714 | 0.148 | 0.551 | 0.772 |
| trim6 | 29.8 | 0.091 | 0.276 | 0.922 | 0.242 | 0.838 | 0.909 |
| agree | 26.8 | 0.011 | 0.149 | 0.756 | 0.144 | 0.739 | 0.978 |

### arithmetic, mode inj10, n = 30

| calibrator | n in sample | never-taught share | clean: known flagged | found | contaminated: known flagged | found | retention |
|---|---|---|---|---|---|---|---|
| plain | 29.9 | 0.104 | 0.050 | 0.230 | 0.037 | 0.141 | 0.612 |
| trim1 | 29.9 | 0.104 | 0.101 | 0.427 | 0.080 | 0.262 | 0.614 |
| trim2 | 29.9 | 0.104 | 0.146 | 0.602 | 0.114 | 0.397 | 0.659 |
| trim3 | 29.9 | 0.104 | 0.183 | 0.726 | 0.152 | 0.525 | 0.723 |
| trim6 | 29.9 | 0.104 | 0.286 | 0.925 | 0.253 | 0.836 | 0.904 |
| agree | 27.0 | 0.021 | 0.149 | 0.756 | 0.139 | 0.722 | 0.955 |

### banking, mode inj10, n = 30

| calibrator | n in sample | never-taught share | clean: known flagged | found | contaminated: known flagged | found | retention |
|---|---|---|---|---|---|---|---|
| plain | 30.0 | 0.101 | 0.060 | 0.676 | 0.022 | 0.362 | 0.535 |
| trim1 | 30.0 | 0.101 | 0.126 | 0.858 | 0.059 | 0.558 | 0.651 |
| trim2 | 30.0 | 0.101 | 0.175 | 0.916 | 0.100 | 0.697 | 0.761 |
| trim3 | 30.0 | 0.101 | 0.225 | 0.947 | 0.145 | 0.825 | 0.871 |
| trim6 | 30.0 | 0.101 | 0.374 | 0.980 | 0.280 | 0.950 | 0.969 |
| agree | 30.0 | 0.015 | 0.114 | 0.762 | 0.105 | 0.700 | 0.918 |

### agnews, mode inj10, n = 30

| calibrator | n in sample | never-taught share | clean: known flagged | found | contaminated: known flagged | found | retention |
|---|---|---|---|---|---|---|---|
| plain | 30.0 | 0.101 | 0.044 | 0.416 | 0.026 | 0.267 | 0.643 |
| trim1 | 30.0 | 0.101 | 0.088 | 0.634 | 0.052 | 0.468 | 0.738 |
| trim2 | 30.0 | 0.101 | 0.138 | 0.751 | 0.086 | 0.626 | 0.834 |
| trim3 | 30.0 | 0.101 | 0.191 | 0.821 | 0.121 | 0.727 | 0.885 |
| trim6 | 30.0 | 0.101 | 0.342 | 0.917 | 0.264 | 0.881 | 0.961 |
| agree | 30.0 | 0.026 | 0.123 | 0.707 | 0.122 | 0.697 | 0.985 |

### newsgroups, mode inj10, n = 30

| calibrator | n in sample | never-taught share | clean: known flagged | found | contaminated: known flagged | found | retention |
|---|---|---|---|---|---|---|---|
| plain | 30.0 | 0.101 | 0.060 | 0.240 | 0.047 | 0.196 | 0.816 |
| trim1 | 30.0 | 0.101 | 0.121 | 0.377 | 0.092 | 0.315 | 0.835 |
| trim2 | 30.0 | 0.101 | 0.174 | 0.481 | 0.142 | 0.413 | 0.858 |
| trim3 | 30.0 | 0.101 | 0.221 | 0.564 | 0.185 | 0.498 | 0.882 |
| trim6 | 30.0 | 0.101 | 0.373 | 0.768 | 0.320 | 0.713 | 0.928 |
| agree | 30.0 | 0.009 | 0.357 | 0.772 | 0.355 | 0.769 | 0.997 |

### banking, the price of trimming on a clean sample (nothing wrongly verified)

| n | calibrator | theory (k+1)/(n+1) | asking reader | vMF | pair: known flagged | pair: found |
|---|---|---|---|---|---|---|
| 30 | plain | 0.032 | 0.032 | 0.039 | 0.068 | 0.700 |
| 30 | trim1 | 0.065 | 0.061 | 0.071 | 0.121 | 0.860 |
| 30 | trim2 | 0.097 | 0.095 | 0.104 | 0.176 | 0.912 |
| 30 | trim3 | 0.129 | 0.128 | 0.137 | 0.232 | 0.945 |
| 30 | trim6 | 0.226 | 0.221 | 0.220 | 0.371 | 0.978 |
| 40 | plain | 0.024 | 0.053 | 0.059 | 0.058 | 0.653 |
| 40 | trim1 | 0.049 | 0.081 | 0.089 | 0.154 | 0.896 |
| 40 | trim2 | 0.073 | 0.081 | 0.089 | 0.154 | 0.896 |
| 40 | trim3 | 0.098 | 0.106 | 0.107 | 0.190 | 0.928 |
| 40 | trim6 | 0.171 | 0.179 | 0.174 | 0.299 | 0.969 |
| 60 | plain | 0.016 | 0.057 | 0.051 | 0.066 | 0.717 |
| 60 | trim1 | 0.033 | 0.072 | 0.068 | 0.128 | 0.875 |
| 60 | trim2 | 0.049 | 0.072 | 0.068 | 0.128 | 0.875 |
| 60 | trim3 | 0.066 | 0.088 | 0.085 | 0.155 | 0.911 |
| 60 | trim6 | 0.115 | 0.139 | 0.128 | 0.232 | 0.956 |

### banking, retention under 10 % injected contamination as the sample grows

| n | plain | trim1 | trim3 | trim6 | agree |
|---|---|---|---|---|---|
| 30 | 0.535 | 0.651 | 0.871 | 0.969 | 0.918 |
| 40 | 0.482 | 0.678 | 0.775 | 0.947 | 0.895 |
| 60 | 0.480 | 0.620 | 0.685 | 0.881 | 0.893 |

## Verdicts (rules fixed before the run)

- C11, trim1: clean known flagged 0.101, contaminated 0.077 (<= 0.1); found 0.279 against 0.432 clean, retention 0.645 (need >= 0.85). **FAIL**
- C11, trim2: clean known flagged 0.145, contaminated 0.114 (<= 0.1); found 0.416 against 0.596 clean, retention 0.697 (need >= 0.85). **FAIL**
- C11, trim3: clean known flagged 0.175, contaminated 0.148 (<= 0.1); found 0.551 against 0.714 clean, retention 0.772 (need >= 0.85). **FAIL**
- C11, agree: clean known flagged 0.149, contaminated 0.144 (<= 0.1); found 0.739 against 0.756 clean, retention 0.978 (need >= 0.85). **FAIL**
- C11 overall (plain keeps 0.689): **FAIL**
- C12 (single readers, clean, text domains): n=30 k=0: 0.033 vs 0.032; n=30 k=1: 0.063 vs 0.065; n=30 k=2: 0.098 vs 0.097; n=30 k=3: 0.127 vs 0.129; n=60 k=0: 0.049 vs 0.016; n=60 k=1: 0.066 vs 0.033; n=60 k=2: 0.066 vs 0.049; n=60 k=3: 0.081 vs 0.066; n=60 k=6: 0.127 vs 0.115. **FAIL** (tolerance 0.03)
- C13 overall (one calibrator in every domain under inj10): **FAIL**

### Post-hoc: one curve of false flags against found

### arithmetic, mode real, n = 30: contaminated points against the clean curve

| calibrator | pair: known flagged | found | lost at matched | asking reader: lost | vMF: lost | retention as defined in G4c | retention at matched false flags (pair) |
|---|---|---|---|---|---|---|---|
| plain | 0.036 | 0.171 | -0.007 | -0.035 | 0.009 | 0.689 | 1.043 |
| trim1 | 0.077 | 0.279 | 0.056 | -0.005 | 0.002 | 0.645 | 0.832 |
| trim2 | 0.114 | 0.416 | 0.062 | -0.010 | 0.001 | 0.697 | 0.870 |
| trim3 | 0.148 | 0.551 | 0.053 | -0.020 | 0.007 | 0.772 | 0.912 |
| trim6 | 0.242 | 0.838 | 0.044 | -0.015 | -0.003 | 0.909 | 0.950 |
| agree | 0.144 | 0.739 | -0.145 | -0.033 | 0.000 | 0.978 | n/a |

agree against the pair's trim curve (arithmetic, clean): known flagged 0.149, found 0.756; the curve finds 0.605 at that rate, so agree is ABOVE it by 0.150. The asking reader agrees with the named experts on 0.804 of the verified known messages (1 minus that: 0.196); under agree the pair flags 0.149 and the asking reader alone 0.128.

agree against the pair's trim curve (banking, clean): known flagged 0.114, found 0.762; the curve finds 0.841 at that rate, so agree is BELOW it by 0.079. The asking reader agrees with the named experts on 0.894 of the verified known messages (1 minus that: 0.106); under agree the pair flags 0.114 and the asking reader alone 0.078.

#### The corrected flag-rate formula against the clean text-domain single readers

| n | k | observed (mean of the two readers over the three text domains) | (k+1)/(n+1), declared | corrected formula |
|---|---|---|---|---|
| 30 | 0 | 0.033 | 0.032 | 0.032 |
| 30 | 1 | 0.063 | 0.065 | 0.065 |
| 30 | 2 | 0.098 | 0.097 | 0.097 |
| 30 | 3 | 0.127 | 0.129 | 0.129 |
| 40 | 0 | 0.050 | 0.024 | 0.049 |
| 40 | 1 | 0.076 | 0.049 | 0.073 |
| 40 | 2 | 0.076 | 0.073 | 0.073 |
| 40 | 3 | 0.099 | 0.098 | 0.098 |
| 40 | 6 | 0.170 | 0.171 | 0.171 |
| 60 | 0 | 0.049 | 0.016 | 0.049 |
| 60 | 1 | 0.066 | 0.033 | 0.066 |
| 60 | 2 | 0.066 | 0.049 | 0.066 |
| 60 | 3 | 0.081 | 0.066 | 0.082 |
| 60 | 6 | 0.127 | 0.115 | 0.131 |

Largest difference between observed and the corrected formula: 0.004.

#### Largest distance of a contaminated single-reader point from the clean curve (prediction 1: at most 0.05)

- arithmetic/real/ask_exemplars: 0.035
- arithmetic/real/vmf_conformal: 0.009
- arithmetic/inj10/ask_exemplars: 0.033
- arithmetic/inj10/vmf_conformal: 0.014
- arithmetic/inj20/ask_exemplars: 0.033
- arithmetic/inj20/vmf_conformal: 0.010
- banking/inj10/ask_exemplars: 0.019
- banking/inj10/vmf_conformal: 0.174  (over 0.05)
- banking/inj20/ask_exemplars: 0.030
- banking/inj20/vmf_conformal: 0.162  (over 0.05)
- agnews/inj10/ask_exemplars: 0.017
- agnews/inj10/vmf_conformal: 0.033
- agnews/inj20/ask_exemplars: 0.020
- agnews/inj20/vmf_conformal: 0.041
- newsgroups/inj10/ask_exemplars: 0.009
- newsgroups/inj10/vmf_conformal: 0.028
- newsgroups/inj20/ask_exemplars: 0.008
- newsgroups/inj20/vmf_conformal: 0.018

## Interpretation (what we understood, and on what basis)
Plain words first. **Nothing in the grid protects the cut for free, and one of my yardsticks was wrong.** (1) Trimming lowers the cut, so the system flags more of everything, known and never-taught messages alike; each dropped score costs about 1/(n+1) false flags per reader and the pair pays about twice. Even trimming one score takes the pair to 0.095-0.127 in the text domains and 0.101 in arithmetic, against a bar of 0.10 (C-107, C-108). (2) The "keep 85 % of your own clean power" bar of C11 could not be met by trimming, because trimming also raises the clean power steeply (arithmetic: 0.248, 0.432, 0.596, 0.714 at known flagged 0.051, 0.101, 0.145, 0.175); it compares a calibrator with itself at another operating point (F-55). (3) Read at the same false-flag rate, contamination hardly hurts one reader: the contaminated points lie on the clean curve (within 0.035 in arithmetic, 0.041 in the other domains, except the pool's gate in banking, see below). **The plain cut fails safe**: with wrong messages in the sample it flags LESS (arithmetic 0.051 to 0.036, banking 0.060 to 0.022 at 10 % and 0.078 to 0.017 at 20 %), never more than nominal, and at matched false flags the pair loses about nothing in arithmetic, agnews and newsgroups. So most of E-050's "a third of the power" was a cut sitting at a more conservative point, not a reader that got worse (C-109 qualifies C-105). (4) For the pair, trimmed contaminated points sit 0.02-0.21 below the pair's own clean curve, because contamination moves the two readers' cuts unequally. (5) **The second opinion is the reader's own refusal rate again**: agree keeps almost all its power (retention 0.95-1.00; the never-taught share of its sample falls to 0.009-0.067) because it removes everything the reader finds novel, and it gives up the 5 % control to do so (known flagged 0.114 banking, 0.123 agnews, 0.149 arithmetic, 0.357 newsgroups where the reader agrees on 0.596 of the verified messages). It is above the pair's trim curve in arithmetic by 0.150 (it spends the budget on the stronger reader there) and below it in banking by 0.079 (where the pool's gate is the stronger one) (C-110).

Technical statement: the cut of a verified sample is an order statistic, so it moves by the contaminants' scores (E-050) and a trim moves it by whole ranks; for a single reader the false-flag rate after trimming k of n scores is (k + m - r + 1)/(n + 1), m = n - k, r = ceil((m+1)(1-q)) (checked by simulation and against the text domains within 0.004), which is (k+1)/(n+1) only while n - k <= 38; for the either-fusion of two readers it is about twice that, because after leave-one-out recalibration the fused cut is "above the best retained score of some reader". Measured: the tables above. Assumed: exchangeability of the sample with the evaluation half, content-independent contamination in the injected modes, and a piecewise-linear curve through sparse points.

## Caveats and what went wrong
- **C11-C13 FAIL as declared; C12 fails because of my formula (F-54)**: (k+1)/(n+1) is wrong above n - k = 38; the corrected formula fits within 0.004. The verdict stays FAIL, the bar is not changed.
- **The retention yardstick (F-55)** cannot show a gain from trimming; I saw that only in the tables. The post-hoc analysis that fixes it is not a declared rule, and 3 of its 4 predictions missed as written (single readers within 0.05: missed in banking where the pool's gate lies ABOVE the chord by 0.17, probably because the clean curve has no point below 0.039 known flagged; plain and trim1 both under 0.04 lost: trim1 0.056; agree about 1 minus the agreement share within 0.06: agnews 0.079).
- The matched-false-flag reading is only as good as a straight line through a few clean points; it understates a concave curve, so the true loss is somewhat larger than measured where the point lies between two clean points. Banking cannot be read this way at low false-flag rates. An exact per-draw curve from the evaluation scores is not computed.
- The injected contamination is content-independent; the real arithmetic coincidences are milder (retention 0.689 real against 0.612 injected 10 %, with the same share of the sample). Three seeds; arithmetic and the named experts: the outcome names the true owner for only 0.889 of verified known messages (coincident matches), the reader agrees with a named expert on 0.804.
- Misses of predictions (F-56): the trim retention (0.80-0.97 predicted, 0.645-0.772 observed in arithmetic), trim1 clean found 0.432 (0.33-0.42), agree found 0.756 (0.35-0.55), agnews plain retention 0.643 (0.65-0.85), newsgroups trim1 gain 0.019 (0.05-0.15).

## Consequence
The default stays **plain**: it is the only calibrator that never flags more than the nominal rate under any contamination tested, and what it pays for a noisy check is unused false-flag budget (arithmetic: 0.036 of 0.05 used), not a wrong decision. Report the effective false-flag rate with every result. Trimming remains an explicit policy knob with a published price (one reader (k+1)/(n+1) while n - k <= 38, the pair about twice that); agree is not a calibration. The lever is the purity of the outcome check (15.8 % false verification in arithmetic); a second INDEPENDENT outcome does not exist in these test beds, and the reader's own opinion is circular. New open question from (4) and (5): how a fixed false-flag budget should be split between the readers (the stronger reader should get more of it; which one is stronger is not known without never-taught messages). Next, for the user's decision: the growmoe experts as container, or a budget split between readers (G4d), or self-discovery of cards. Nothing is frozen and the confirmatory seeds 400-429 are unspent.

## Evidence
- `research/knowledge_status/results/ROUTING_GENERAL_G4c_150_152.md`, `ROUTING_GENERAL_G4c_posthoc_150_152.md` (generated), `results/routing_general/g4c/` (cells and run log)
- `research/knowledge_status/PROGRESS.md` (plan, rules, predictions, results, post-hoc, in order)

## Reproduce
```
cd D:/shm_ks
python research/knowledge_status/routing_general/lab_robust.py 150 152
python research/knowledge_status/routing_general/posthoc_g4c.py
python -m pytest tests/knowledge_status/test_rg_robust.py tests/knowledge_status/test_rg_posthoc_g4c.py
```

