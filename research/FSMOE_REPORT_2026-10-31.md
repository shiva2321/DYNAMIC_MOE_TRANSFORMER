# FS-MoE: what I built, what it can do and what it cannot (2026-10-31)

Plain words first, the exact numbers next to them. Every number below was copied from a generated table (links at the end); seeds 710-714 are the report seeds, nothing was tuned on them.
One figure sums it up: [FIGURE_SUMMARY.png](../results/fsmoe/FIGURE_SUMMARY.png).

## The short answer

1. **The problem you are stuck on is not solved by looking.** Telling "the same skill in a new wording" from "a genuinely new skill that looks almost the same" (add vs subtract, sum vs product, two banking intents) cannot be done by any signal I built from the input alone. At a false-alarm rate of about 4-8 %, the best signal finds only 13 % (arithmetic), 22 % (code) and 14 % (language) of the new near-twin skills. The cut-free score (AUROC; 0.5 is a coin flip, 1.0 is perfect) is 0.41-0.72 for nearly every signal and 0.78 at best, against the 0.95 I declared as the bar. By my own kill rule I stopped tuning detectors.
2. **What does separate them is the outcome (in a synthetic world; not yet shown on real code).** When the next state arrives, asking every expert "what did you predict?" separates a new function from a new wording at AUROC 0.92 (low noise), 0.93 (medium) and 0.81 (high noise), where the look-based score gets 0.13, because it reads a new wording as novelty and is blind to a new function. The cheap shortcut (let the look pick one expert, then judge it) fails completely (0.47, chance). So novelty is best handled as **act, then compare with the outcome**, not as "recognize first".
3. **The system around it works as a learner.** The validated pool with a closed-form head in every expert (3 experts, 12.9 MB) reads taught skills at 0.965 and the same skills in new wordings at 0.895, with no forgetting. Equal-memory replay gets 0.952 / 0.832, kNN 0.934 / 0.815, streaming LDA 0.945 / 0.866, the offline joint network 0.961 / 0.821. Teams give nothing on this bed (1.02 experts per input, accuracy 0.878 vs 0.882).
4. **Learning a new skill while deployed is the weak link.** With 5 % of inputs checked, the pool ends up answering only 6-17 % of the 28 never-taught skills; even the oracle (only new items checked, 19-29 examples per skill) learns just 21-31 %. Spreading the checks over many different skills (a policy I added after seeing this) is the one fix that worked: at 10 % it meets 23.6 of the 28 skills and learns 28 %, level with the oracle's 31 %. But learning new skills costs known skills their new-wording accuracy (0.885 before, 0.63-0.83 after); I suspect the new near-twins steal them but did not measure that.
5. **My predictions mostly missed:** of 15 declared numeric predictions, 4 held and 11 missed (27 %; earlier stages scored 45-49 %). The misses are the story: I expected a cheap signal to carry more.

## What I built

| Part | What it is | Where |
|---|---|---|
| Test bed | arithmetic (8 operations, 8 wording families), code (16 function families, 9 styles; every snippet is EXECUTED and must match its family's behaviour), language (77 banking intents, 5 wording families); 3 + 5 + 20 skills withheld, near-twins and far | [fsmoe/domains](../fsmoe/domains) |
| Readers | Qwen2.5-1.5B (code, arithmetic) plus MiniLM (language), concatenated; chosen on a dev pilot because Qwen alone read language at 0.24 on new wordings | [fsmoe/reader.py](../fsmoe/reader.py), [pilot_700.txt](../results/fsmoe/pilot_700.txt) |
| Recognizer lab | 5 signals (distance, vMF density, kNN, fast-slow disagreement, confidence) in two spaces (raw and a learned invariance map), fusions, group test, two calibrations | [fsmoe/recog_lab.py](../fsmoe/recog_lab.py) |
| Fast-slow expert | closed-form LDA part + small network, heads: label, density, confidence, disagreement | [fsmoe/fastslow.py](../fsmoe/fastslow.py) |
| System layer | the validated pool (`ULDA`) taught across the three domains, equal-memory baselines, teams, verification policies | [fsmoe/system_lab.py](../fsmoe/system_lab.py) |
| Teams, trust ledger | adaptive team (smallest set holding 99 % of the routing weight); a forgiving trust ledger (Axelrod-style: nice, retaliating, forgiving, generous once) | [team.py](../fsmoe/team.py), [ledger.py](../fsmoe/ledger.py) |
| Traces lab | a world with real next states: result-predicting experts, recognition by next-state error, a stuck-then-grow rule for width and depth | [fsmoe/traces.py](../fsmoe/traces.py) |

46 tests (tests/fsmoe), and a mutation check: 9 deliberate breaks of key rules, all caught, files restored byte for byte. Nothing in growmoe/ or unimoe/ was edited. Nothing is committed.

## What the tests showed

**The recognizer (3 domains, 5 seeds).** The naive cut (calibrated on the teaching wording) calls 16-96 % of known skills in new wordings "new": unusable. Calibrating on 40 known inputs from the stream's own wordings (what verified outcomes provide) brings that to 6-8 %, close to the 5 % aimed at. But power stays low:

| domain | best signal: new near-twins found | new far skills found | far texts found | best AUROC (near-twin) |
|---|---|---|---|---|
| arithmetic | 0.126 (disagreement) | no far operation exists | 0.99-1.00 (distance signals) | 0.65 raw, 0.68 with the map |
| code | 0.217 (kNN) | 0.93 (kNN) | 1.00 (kNN, distance) | 0.72 raw, 0.78 with the map |
| language | 0.136 (vMF) | 0.31 (intents far from taught ones) | 0.95-0.99 (distance signals) | 0.62 raw, 0.62 with the map |

The learned invariance map helped the near-twin score a little in arithmetic and code and hurt far-text detection (arithmetic kNN 0.999 to 0.83). Fast-slow disagreement is no better than distance (0.65 / 0.69 / 0.55 against 0.60 / 0.72 / 0.57). Fusing signals ("either") just adds their false flags. A group test (8-16 inputs from one source) finds near-twins 0.15-0.37 of the time at 6-11 % false alarms, and far code skills perfectly; for language it finds far intents only about half the time.

**Traces (5 seeds).** Checking every expert's next-state error is the only signal that works, and it costs 21 times the look (26,112 vs 1,216 multiply-adds per input). It needs experts taught on varied surfaces: with 2 surfaces AUROC is 0.88, with 6 it is 0.92-0.93. A new style moves the look score so much that, as a novelty signal, it ranks known inputs in a new style as MORE novel than new functions (AUROC 0.13). Depth growth: my first rule (grow when stuck and the fit is worse than 20 % of the variance) never fired, because even 8 hidden units explained over 90 % of the variance. A stricter variant (5 %, added after seeing this) widens when it should: on the first composed map its error (0.046, 3.6 k parameters) beats every fixed net I tried, which have 1.3-2.5 times its parameters, and on the second it beats all but the 3-block net (0.064 against 0.060). But it creeps too slowly on a hard map (stops at 38 units, error 0.187 against 0.076 for a fixed 128) and deepens only 0.2-0.4 times per run. Depth itself does pay where composition is: at equal size (about 4.9 k parameters) two blocks of 32 beat one block of 64 on the composed maps (0.055 against 0.061, 0.070 against 0.079) and lose on the pairwise-interaction map (0.115 against 0.089); the rule just rarely chooses it.

**The system (5 seeds).** See the short answer; per domain the pool reads arithmetic 1.00 / 0.94, code 1.00 / 0.94, language 0.895 / 0.805 (taught / new wording). Verified-budget table (learned = answers a never-taught skill with accuracy of at least 0.5; wide intervals, seeds differ a lot):

| policy | budget | share of verified items that were new | skills met (of 28) | learned | known skills in new wordings, after |
|---|---|---|---|---|---|
| none | 0 | n/a | 0 | 0.00 | 0.885 (unchanged) |
| random | 5 % | 0.13 | 15.8 | 0.08 | 0.787 |
| novelty-ranked | 5 % | 0.26 | 4.0 | 0.10 | 0.760 |
| uncertainty-ranked | 5 % | 0.10 | 15.0 | 0.06 | 0.873 |
| oracle (only new items) | 5 % / 10 % / 20 % / 40 % | 1.00 | 28 | 0.17 / 0.31 / 0.21 / 0.23 | 0.828 / 0.738 / 0.783 / 0.782 |
| diverse novelty (added later) | 5 % / 10 % | 0.26 | 11.2 / 23.6 | 0.16 / 0.28 | 0.629 / 0.740 |

Novelty ranking doubles the share of new items checked but spends them on the same few skills (4 of 28). Diversity fixes the breadth. The pool's own learning of a new skill from a few interleaved examples is slow (even 28 examples per skill give about a quarter), and every policy that teaches it new skills lowers the new-wording accuracy of the known ones (the cause is a guess, not measured).

## What went wrong or changed on the way

- Qwen alone read language badly (0.24 on new wordings), so I added MiniLM as a second reader; decided on the dev seed only.
- My first trace world gave experts only two surfaces and made "look" and "result" look equally bad; I added a diversity control (2 vs 6 surfaces) and kept both results.
- The depth rule never triggered as declared; the stricter variant and two harder cases were added after seeing that, and are marked exploratory.
- The diverse policy and the oracle's budget curve were added after the main readout (marked with an asterisk in the figure).
- I ran two Qwen embedding jobs at once and filled the GPU (12 GB). I tried to stop the duplicate, the request was denied, so I left both running; they finished without error.
- The recognizer lab ran on 5 of the 10 planned report seeds (710-714) because the others were still embedding; the rest finish in the background and can be added by re-running the report.

## What I would do next (your call)

1. Make "act, then compare with the outcome" the novelty mechanism: run the experts' prediction check on every input whose outcome arrives (the 21 times cost is paid only there) and keep the look-based score for far novelty and for ranking what to verify.
2. In code and arithmetic, use execution itself as the outcome. A snippet's behaviour on fixed inputs is a perfect "function fingerprint"; that is the real outside anchor, and the bed already computes it.
3. Fix the new-skill learning path: class-focused batches, more steps for a new expert, and protection of the twin's new-wording accuracy; then re-test the diverse policy.
4. Language needs a labelled feedback loop; nothing here separates new intents from new wordings by look.

Not built: the trust ledger inside the pool, right-sizing a new expert at birth, a shared trunk, system-level vertical growth (adding a layer of experts), the token-level stage. The ledger and teams exist as tested modules only.

## Evidence

[REPORT_R.md](../results/fsmoe/R/REPORT_R.md), [REPORT_S.md](../results/fsmoe/S/REPORT_S.md), [REPORT_T.md](../results/fsmoe/T/REPORT_T.md), [SCORECARD.md](../results/fsmoe/SCORECARD.md), the design with every prediction written before the run: [2026-10-31-fsmoe-design.md](../docs/superpowers/specs/2026-10-31-fsmoe-design.md), notebook entry E-055.
