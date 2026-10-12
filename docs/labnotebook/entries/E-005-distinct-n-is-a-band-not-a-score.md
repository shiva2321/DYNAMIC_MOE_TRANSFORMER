# E-005 - Distinct-n is a band, not a score: a metric that scored the sampler

- **ID:** E-005
- **Date:** 2026-09-21
- **Track:** I
- **Type:** audit
- **Status:** completed
- **Evidence grade:** B
- **Code / commit:** `ishmn/metric_validation.py`
- **People:** Claude

## Question
Does the 500M-parameter harness's "Distinct-2 = 1.000, zero repetition collapse" mean the model writes fluent text?

## Why we asked it
The same checkpoint, decoded greedily, emitted `" the the the the the the the the the the the the the world."`.

## Prediction
None.

## Method (how)
Calibrate the metric on known references: natural prose, a worked math solution, uniform random tokens, and a degenerate repeat, at 60 and 25 words.

## Results (what came out)
| Text | D-2 at 60 words | D-2 at 25 words |
|---|---|---|
| natural English prose | 0.917 | 1.000 |
| worked math solution | 0.958 | 1.000 |
| uniform random tokens | 1.000 | 1.000 |
| "the the the ..." | 0.042 | 0.042 |

## Interpretation
1.000 is what noise scores, not fluency. The harness sampled at temperature 0.7 from a near-uniform 50k distribution, so it scored the sampler. Below about 45 words the metric is dead (fluent references also hit 1.000); the harness' 25-token limit lies inside that dead zone. Length-robust substitutes (`top_token_share`, `longest_run`) caught the collapse.

## Caveats and what went wrong
We had trusted a "success" number from a harness that was never calibrated against a known-bad and a known-good text.

## Consequence
Never quote a Distinct-n without its sample length and a length-matched reference; report greedy output next to sampled output. Third instance of "a number quoted without the control that gives it meaning" (E-001, E-004).

## Evidence
- `ishmn/metric_validation.py` - the calibration

## Reproduce
Run the calibration in `ishmn/metric_validation.py` on the reference texts above.
