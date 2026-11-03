# M1 Wide Baselines GPU Repeat Report (DEV)

## Executive Summary

This report measures the repeated performance of the wide baselines (`A3`, `A3-2`, `A4`, `A4-2`, `A6`, `A8`) on NVIDIA GPU relative to host CPU execution, while continual pools (`A1`, `A2`) remain on CPU.

### Key Findings:
- **B1_n10**: For bed B1_n10, the Rule 6.1 (R1_primary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **B1_n100**: For bed B1_n100, the Rule 6.1 (R1_primary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **B1_n30**: For bed B1_n30, the Rule 6.1 (R1_primary) efficiency verdict remains UNCHANGED: baselines A4 trigger under both CPU and GPU hardware readings.
- **B2**: For bed B2, the Rule 6.1 (R1_primary) efficiency verdict remains UNCHANGED: baselines A3, A4, A5a, A6, A7, A7all trigger under both CPU and GPU hardware readings.
- **B3drift**: For bed B3drift, the Rule 6.1 (R1_primary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **B3mix**: For bed B3mix, the Rule 6.1 (R1_primary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **B3seq**: For bed B3seq, the Rule 6.1 (R1_primary) efficiency verdict remains UNCHANGED: baselines A3, A4, A5a, A7, A7all trigger under both CPU and GPU hardware readings.

## Performance Comparison Tables

### Bed B1_n10

| Arm | Seeds | CPU Up (s) | GPU Up (s) | Up Speedup | CPU Ans1 (ms) | GPU Ans1 (ms) | Ans1 Speedup | CPU Ans32 (ms) | GPU Ans32 (ms) | Ans32 Speedup | Peak VRAM (MB) | CPU Acc | GPU Acc | Paired Diff (95% CI) | MACs Equal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A3 | 1 | n/a (det) | 11.55 | n/a (det) | n/a (det) | 0.45 | n/a (det) | n/a (det) | 0.43 | n/a (det) | 191.2 | 39.50% | 37.60% | -1.90% | MISMATCH |
| A8 | 1 | n/a (det) | 29.51 | n/a (det) | n/a (det) | 0.45 | n/a (det) | n/a (det) | 0.46 | n/a (det) | 135.8 | 35.70% | 36.65% | +0.95% | EQUAL |

### Bed B2

| Arm | Seeds | CPU Up (s) | GPU Up (s) | Up Speedup | CPU Ans1 (ms) | GPU Ans1 (ms) | Ans1 Speedup | CPU Ans32 (ms) | GPU Ans32 (ms) | Ans32 Speedup | Peak VRAM (MB) | CPU Acc | GPU Acc | Paired Diff (95% CI) | MACs Equal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A3 | 1 | 2.01 | 2.45 | **0.82x** | 0.12 | 0.43 | 0.28x | 0.22 | 0.44 | 0.50x | 74.5 | 84.78% | 84.78% | 0.00% | EQUAL |
| A4 | 1 | 1.75 | 3.69 | **0.47x** | 0.10 | 0.42 | 0.24x | 0.14 | 0.45 | 0.32x | 71.3 | 84.67% | 84.78% | +0.11% | EQUAL |
| A6 | 1 | 1.67 | 0.67 | **2.52x** | 0.13 | 0.40 | 0.32x | 0.56 | 0.64 | 0.87x | 616.2 | 91.89% | 91.89% | 0.00% | EQUAL |
| A8 | 1 | 6.36 | 7.85 | **0.81x** | 0.12 | 0.43 | 0.28x | 0.22 | 0.44 | 0.51x | 75.8 | 88.00% | 88.78% | +0.78% | EQUAL |

## Restated Rule 6.1 Efficiency Verdicts

| Bed | Reading | CPU Verdict | GPU Verdict | Status | Verdict Impact |
|---|---|---|---|---|---|
| B1_n10 | R1 Primary (Wall-Clock) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B1_n10 | R1 Secondary (MACs) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B1_n10 | R2 Primary (Wall-Clock) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B1_n10 | R2 Secondary (MACs) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B1_n100 | R1 Primary (Wall-Clock) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B1_n100 | R1 Secondary (MACs) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B1_n100 | R2 Primary (Wall-Clock) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B1_n100 | R2 Secondary (MACs) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B1_n30 | R1 Primary (Wall-Clock) | PASS (trig: A4) | PASS (trig: A4) | PASS | UNCHANGED |
| B1_n30 | R1 Secondary (MACs) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B1_n30 | R2 Primary (Wall-Clock) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B1_n30 | R2 Secondary (MACs) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B2 | R1 Primary (Wall-Clock) | PASS (trig: A3, A4, A5a, A6, A7, A7all) | PASS (trig: A3, A4, A5a, A6, A7, A7all) | PASS | UNCHANGED |
| B2 | R1 Secondary (MACs) | PASS (trig: A4, A5a) | PASS (trig: A4, A5a) | PASS | UNCHANGED |
| B2 | R2 Primary (Wall-Clock) | PASS (trig: A3-2, A5a, A6, A7-2, A7all) | PASS (trig: A3-2, A5a, A6, A7-2, A7all) | PASS | UNCHANGED |
| B2 | R2 Secondary (MACs) | PASS (trig: A5a) | PASS (trig: A5a) | PASS | UNCHANGED |
| B3drift | R1 Primary (Wall-Clock) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B3drift | R1 Secondary (MACs) | PASS (trig: A5a) | PASS (trig: A5a) | PASS | UNCHANGED |
| B3drift | R2 Primary (Wall-Clock) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B3drift | R2 Secondary (MACs) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B3mix | R1 Primary (Wall-Clock) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B3mix | R1 Secondary (MACs) | PASS (trig: A5a) | PASS (trig: A5a) | PASS | UNCHANGED |
| B3mix | R2 Primary (Wall-Clock) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B3mix | R2 Secondary (MACs) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |
| B3seq | R1 Primary (Wall-Clock) | PASS (trig: A3, A4, A5a, A7, A7all) | PASS (trig: A3, A4, A5a, A7, A7all) | PASS | UNCHANGED |
| B3seq | R1 Secondary (MACs) | PASS (trig: A4, A5a) | PASS (trig: A4, A5a) | PASS | UNCHANGED |
| B3seq | R2 Primary (Wall-Clock) | PASS (trig: A5b, A7-2, A7all) | PASS (trig: A5b, A7-2, A7all) | PASS | UNCHANGED |
| B3seq | R2 Secondary (MACs) | FAIL (trig: none) | FAIL (trig: none) | FAIL | UNCHANGED |

## Verdict Decisions and Statements

### B1_n10
- **R1_primary**: For bed B1_n10, the Rule 6.1 (R1_primary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **R1_secondary**: For bed B1_n10, the Rule 6.1 (R1_secondary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **R2_primary**: For bed B1_n10, the Rule 6.1 (R2_primary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **R2_secondary**: For bed B1_n10, the Rule 6.1 (R2_secondary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.

### B1_n100
- **R1_primary**: For bed B1_n100, the Rule 6.1 (R1_primary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **R1_secondary**: For bed B1_n100, the Rule 6.1 (R1_secondary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **R2_primary**: For bed B1_n100, the Rule 6.1 (R2_primary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **R2_secondary**: For bed B1_n100, the Rule 6.1 (R2_secondary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.

### B1_n30
- **R1_primary**: For bed B1_n30, the Rule 6.1 (R1_primary) efficiency verdict remains UNCHANGED: baselines A4 trigger under both CPU and GPU hardware readings.
- **R1_secondary**: For bed B1_n30, the Rule 6.1 (R1_secondary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **R2_primary**: For bed B1_n30, the Rule 6.1 (R2_primary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **R2_secondary**: For bed B1_n30, the Rule 6.1 (R2_secondary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.

### B2
- **R1_primary**: For bed B2, the Rule 6.1 (R1_primary) efficiency verdict remains UNCHANGED: baselines A3, A4, A5a, A6, A7, A7all trigger under both CPU and GPU hardware readings.
- **R1_secondary**: For bed B2, the Rule 6.1 (R1_secondary) efficiency verdict remains UNCHANGED: baselines A4, A5a trigger under both CPU and GPU hardware readings.
- **R2_primary**: For bed B2, the Rule 6.1 (R2_primary) efficiency verdict remains UNCHANGED: baselines A3-2, A5a, A6, A7-2, A7all trigger under both CPU and GPU hardware readings.
- **R2_secondary**: For bed B2, the Rule 6.1 (R2_secondary) efficiency verdict remains UNCHANGED: baselines A5a trigger under both CPU and GPU hardware readings.

### B3drift
- **R1_primary**: For bed B3drift, the Rule 6.1 (R1_primary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **R1_secondary**: For bed B3drift, the Rule 6.1 (R1_secondary) efficiency verdict remains UNCHANGED: baselines A5a trigger under both CPU and GPU hardware readings.
- **R2_primary**: For bed B3drift, the Rule 6.1 (R2_primary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **R2_secondary**: For bed B3drift, the Rule 6.1 (R2_secondary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.

### B3mix
- **R1_primary**: For bed B3mix, the Rule 6.1 (R1_primary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **R1_secondary**: For bed B3mix, the Rule 6.1 (R1_secondary) efficiency verdict remains UNCHANGED: baselines A5a trigger under both CPU and GPU hardware readings.
- **R2_primary**: For bed B3mix, the Rule 6.1 (R2_primary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.
- **R2_secondary**: For bed B3mix, the Rule 6.1 (R2_secondary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.

### B3seq
- **R1_primary**: For bed B3seq, the Rule 6.1 (R1_primary) efficiency verdict remains UNCHANGED: baselines A3, A4, A5a, A7, A7all trigger under both CPU and GPU hardware readings.
- **R1_secondary**: For bed B3seq, the Rule 6.1 (R1_secondary) efficiency verdict remains UNCHANGED: baselines A4, A5a trigger under both CPU and GPU hardware readings.
- **R2_primary**: For bed B3seq, the Rule 6.1 (R2_primary) efficiency verdict remains UNCHANGED: baselines A5b, A7-2, A7all trigger under both CPU and GPU hardware readings.
- **R2_secondary**: For bed B3seq, the Rule 6.1 (R2_secondary) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings.

