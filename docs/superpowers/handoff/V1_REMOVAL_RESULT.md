# V1 / Probe P2: Exact Downdating, Pool Removal, and Removal Audit Result

**Job:** V-3 (Probe P2 / V1 Walking Skeleton)  
**Date:** 2026-11-04  
**Implementer:** Agent G (Gemini)  
**Reviewer:** Claude  
**Directory:** `D:\spherical_hyperspace_memory`  
**Execution Environment:** Pinned to logical processors 8 and 10 with 2 threads (`costlab.env.fixed_threads(n=2, cpus=[8, 10])`), Windows cp1252.

---

## 1. Executive Summary

This report delivers the implementation, unit verification, and empirical audit for **Probe P2** ("Is removal exact?") and **Version V1** of the walking skeleton, under the strict rules of `docs/superpowers/handoff/V1_REMOVAL_SPEC.md`.

### Core Findings
1. **Closed-form heads (Part A):**
   - `ClassMean`, `KnnMemory` (unbounded), `MahalanobisProto` (A5b), and a new class `ExactLda` achieve **100% bit-for-bit exact downdating** ($\text{max difference} \le 10^{-13}$, $100.0000\%$ argmax test agreement) relative to a model that never saw the removed classes across all 12 experiments (R1-R4, seeds 800-802).
   - `RandomFeatureRidge` (A6) achieves exact downdating ($\text{max difference} \le 7.45 \times 10^{-8}$, $100\%$ argmax agreement) when provided with the removed training samples $(x_{\text{rem}}, y_{\text{rem}})$ to downdate the pooled Gram matrix $G \leftarrow G - \phi^T \phi$.
   - `StreamingLDA` (A5a) is **not exact**: its recursive online covariance update $\Sigma_t = \frac{t-1}{t} \Sigma_{t-1} + \frac{t-1}{t^2} (x_t - \mu_{c, t-1})(x_t - \mu_{c, t-1})^T$ entangles class arrival order non-linearly across classes. Measured difference to a never-trained model reaches up to $5.04 \times 10^{-4}$ in covariance, $1.04$ in weights, and $8.17\%$ in predicted probabilities (argmax agreement $98.84\% - 99.98\%$). Its distance to `ExactLda` has a mean covariance Frobenius norm difference of $0.004605$ and argmax agreement of $99.49\%$.
2. **V0 Pool Removal with Certificate (Part B):**
   - In `v0.system.build_v0` (UniPool with ULDA switches), exposed expert modules cannot be downdated analytically and must be dropped.
   - Dropping exposed experts incurs **collateral loss of retained classes** that were learned only by those experts:
     - **R1 (1 class):** 1 expert dropped, 15-18 collateral classes lost, retained accuracy drops from $0.6468$ to $0.6091$.
     - **R2 (5 classes):** 2-4 experts dropped, 26-58 collateral classes lost, retained accuracy drops from $0.6455$ to $0.5320$.
     - **R3 (entire small domain):** 1-2 experts dropped, **0 collateral classes lost**. Retained accuracy matches or exceeds never-trained pool: $0.6806$ ($P_{\text{rem}}$) vs $0.6779$ ($P_{\text{nev}}$).
     - **R4 (1 regime):** Seeds 800 and 801 dropped 2 experts with 0-2 collateral classes lost. However, in seed 802, a **phylogenetic lineage cascade** occurred: 1 directly exposed early parent expert cascaded through 18 descendants, dropping all 19 experts ($100\%$ of pool) and causing 123 collateral classes to be lost (retained accuracy $0.0029$).
   - For all retained experts whose founding batches matched the never-trained pool, **expert weight difference was 0.00e+00** (exact identity).
   - In all 12 pool removal experiments across thousands of test examples, $P_{\text{removed}}$ predicted a removed class label **0 times** ($0.0\%$ accuracy on removed classes).
   - An independent trace audit (`scan_traces`) confirmed 0 residual traces in 11 of 12 runs. In R3 seed 801, 44 prototype traces with cosine $> 0.999$ were detected in retained expert signatures, and the certificate honestly reported `certified_clean: False`.

---

## 2. Deliverables and Files Created

All work adheres to the fence rules: no existing files modified; work confined strictly to `v1/`, `tests/v1/`, `results/v1/`, and handoff documents.

1. **`v1/downdate.py`**:
   - `ExactLda`: Closed-form batch LDA classifier accumulating per-class sufficient statistics (`_cnt`, `_sum`, `_outer`), computing pooled within-class covariance $\Sigma_W = \frac{1}{N} \sum_c (\text{outer}_c - \mu_c \text{sum}_c^T) + \alpha I$, and solving discriminant weights.
   - `remove_classes(model, labels, data=None)`: Exact downdating functions for `ClassMean`, `KnnMemory`, `MahalanobisProto`, `RandomFeatureRidge`, `ExactLda`, and approximate downdating for `StreamingLDA`.
2. **`v1/pool_removal.py`**:
   - `ExposureTracker`: Instance-level non-invasive wrapper tracking direct exposure (training/replay batches, reservoirs, signatures, source windows, calibration data) and computing transitive closure over phylogenetic lineage (`pool._parents`).
   - `scan_traces(pool, labels, removed_samples=None)`: Independent 9-component forensic audit inspecting `seen` masks, reservoirs, meta LDA stats, VMF signatures, domain LDA scatter, probation window, held evaluation buffer, bootstrap buffers, and router rules.
   - `remove_labels(pool, labels, data=None, tracker=None)`: Drops all directly/transitively exposed experts, purges retained expert reservoirs/stats, cleans pool buffers, refits router via `pool._refit_router()`, and issues a signed JSON certificate.
   - `compare_pool_removal(p_full, p_removed, p_never, bed, labels, seed)`: Rigorous evaluation of retained and removed test accuracy, routing agreement, prediction agreement, and founding-batch expert weight matching.
3. **`tests/v1/test_v1.py`**:
   - 6 test cases verifying downdating exactness, `ExactLda` formula equivalence, missing label error handling, planted trace detection & clearance, JSON certificate validity, and experiment run idempotence.
4. **`v1/run_heads_experiments.py`**:
   - Automated runner for Part A closed-form heads across removal sets R1-R4 and seeds 800-802.
5. **`v1/run_pool_experiments.py`**:
   - Automated runner for Part B pool removal experiments with CPU pinning, single-pass batched test evaluation, and JSON serialization.
6. **`v1/summarize_results.py` & `v1/generate_audit_data.py`**:
   - Data verification and extraction scripts reading raw JSON records.
7. **`results/v1/`**:
   - 12 heads JSON records: `heads_r{1,2,3,4}_s{800,801,802}.json`
   - 12 pool JSON records: `pool_r{1,2,3,4}_s{800,801,802}.json`
8. **Handoff Documentation**:
   - `docs/superpowers/handoff/P2_REMOVAL_AUDIT.md`: Complete component audit table with exact file/line references, classifications, measured numbers, plain-English summary, and 3 best cost-benefit recommendations.
   - `docs/superpowers/handoff/V1_REMOVAL_RESULT.md`: This report.
   - `docs/superpowers/handoff/V1_REMOVAL_COMMANDS.txt`: Shell invocation appended command log.
   - `docs/superpowers/handoff/V1_REMOVAL_TEST_OUTPUT.txt`: Pytest execution output.

---

## 3. Measured Results

### Part A: Closed-Form Heads Downdating

Tested on `B3seq` 22 regimes, 125 classes across R1 (1 class), R2 (5 classes), R3 (6 classes of `svamp`), R4 (2 classes of `trec6` regime 0) on dev seeds 800, 801, 802.

| Model | Classification | Max Stat Diff | Max Weight Diff | Max Prob Diff | Min Argmax Agreement | Test Accuracy ($P_{\text{rem}}$ vs $P_{\text{nev}}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`ClassMean`** | **Exact** | `0.00e+00` | `0.00e+00` | `0.00e+00` | `1.000000` ($100.0\%$) | $52.52\%$ vs $52.52\%$ |
| **`KnnMemory`** | **Exact** | `0.00e+00` | `0.00e+00` | `0.00e+00` | `1.000000` ($100.0\%$) | $59.27\%$ vs $59.27\%$ |
| **`MahalanobisProto`** | **Exact** | `0.00e+00` | `0.00e+00` | `0.00e+00` | `1.000000` ($100.0\%$) | $57.73\%$ vs $57.73\%$ |
| **`RandomFeatureRidge`** | **Exact** *(w/ data)* | $\le 7.45\times 10^{-8}$ | $\le 1.08\times 10^{-8}$ | $\le 7.45\times 10^{-9}$ | `1.000000` ($100.0\%$) | $57.85\%$ vs $57.85\%$ |
| **`ExactLda`** | **Exact** | `0.00e+00` | `0.00e+00` | `0.00e+00` | `1.000000` ($100.0\%$) | $56.28\%$ vs $56.28\%$ |
| **`StreamingLDA`** | **Approximate** | $5.04\times 10^{-4}$ | $1.04\times 10^{0}$ | $8.17\times 10^{-2}$ | `0.988424` ($98.84\%$) | $56.28\%$ vs $56.28\%$ |

#### Distance of `StreamingLDA` (A5a) to `ExactLda`:
- Covariance Frobenius norm difference: mean $0.004605$ (min $0.004461$, max $0.004862$).
- Accuracy difference on retained test sets: mean $-0.000059$ (min $-0.000878$, max $+0.000919$).
- Argmax prediction agreement: mean $0.994943$ (min $0.992828$, max $0.996457$).

---

### Part B: V0 Pool Removal (`build_v0`)

Tested on `B3seq` 22 regimes with UniPool/ULDA switches on dev seeds 800, 801, 802.

| Removal Set & Seed | Dropped Experts | Retained Experts | Collateral Classes Lost | Retained Acc $P_{\text{full}}$ | Retained Acc $P_{\text{rem}}$ | Retained Acc $P_{\text{nev}}$ | Router Agreement | Argmax Agreement | Matched Experts (diff=0) | Removed Pred Count | Residual Traces Found |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **R1 s800** | 1 | 19 | 18 | $0.6502$ | $0.6124$ | $0.6499$ | $0.9493$ | $0.9486$ | 19 / 19 | 0 / 15 | 0 |
| **R1 s801** | 1 | 18 | 15 | $0.6469$ | $0.6088$ | $0.6470$ | $0.9514$ | $0.9514$ | 18 / 18 | 0 / 19 | 0 |
| **R1 s802** | 1 | 18 | 18 | $0.6434$ | $0.6060$ | $0.6434$ | $0.9542$ | $0.9542$ | 18 / 18 | 0 / 18 | 0 |
| **R2 s800** | 2 | 18 | 26 | $0.6489$ | $0.5727$ | $0.6497$ | $0.9190$ | $0.9162$ | 17 / 18 | 0 / 85 | 0 |
| **R2 s801** | 4 | 15 | 58 | $0.6457$ | $0.4970$ | $0.6597$ | $0.6130$ | $0.6461$ | 15 / 15 | 0 / 90 | 0 |
| **R2 s802** | 3 | 16 | 33 | $0.6420$ | $0.5263$ | $0.6410$ | $0.9000$ | $0.9000$ | 16 / 16 | 0 / 90 | 0 |
| **R3 s800** | 2 | 18 | **0** | $0.6785$ | $0.6822$ | $0.6822$ | $1.0000$ | $1.0000$ | 18 / 18 | 0 / 600 | 0 |
| **R3 s801** | 1 | 18 | **0** | $0.6762$ | $0.6823$ | $0.6825$ | $0.9948$ | $0.9948$ | 18 / 18 | 0 / 600 | 44 *(cert flagged)* |
| **R3 s802** | 2 | 17 | **0** | $0.6701$ | $0.6772$ | $0.6691$ | $0.0891$ | $0.7020$ | 12 / 17 | 0 / 600 | 0 |
| **R4 s800** | 2 | 18 | 0 | $0.6494$ | $0.6324$ | $0.6509$ | $0.5551$ | $0.7212$ | 16 / 18 | 0 / 300 | 0 |
| **R4 s801** | 2 | 17 | 2 | $0.6434$ | $0.6265$ | $0.6408$ | $0.8274$ | $0.7867$ | 15 / 17 | 0 / 396 | 0 |
| **R4 s802** | 19 | 0 | **123** | $0.6434$ | $0.0029$ | $0.6543$ | $0.0000$ | $0.0026$ | 0 / 0 | 0 / 300 | 0 |

---

## 4. Verification and Measured Runtimes

All commands executed with CPU affinity pinned to cores 8 and 10 with 2 threads (`costlab.env.fixed_threads(n=2, cpus=[8, 10])`). Every individual command completed well under the 2-minute (120s) ceiling.

### Test Suite (`tests/v1/test_v1.py`):
```
tests/v1/test_v1.py::test_exact_downdating_equality PASSED               [ 16%]
tests/v1/test_v1.py::test_exact_lda_matches_brute_force PASSED           [ 33%]
tests/v1/test_v1.py::test_remove_classes_absent_class_raises PASSED      [ 50%]
tests/v1/test_v1.py::test_scan_traces_finds_planted_trace_and_cleans_after_removal PASSED [ 66%]
tests/v1/test_v1.py::test_certificate_is_valid_json PASSED               [ 83%]
tests/v1/test_v1.py::test_repeat_finished_experiment_skipped PASSED      [100%]
======================== 6 passed, 1 warning in 2.93s =========================
```
- Total test time: **2.93 seconds** (well under the 120-second limit).

### Experiment Execution Runtimes:
- **Heads Experiments (12 runs):** 15s to 20s per removal set / seed.
- **Pool Experiments (12 runs):**
  - Pool R1 s800: `53.24s` (removal: 3.19s, never_train: 40.24s, comparison: 9.80s)
  - Pool R1 s801: `41.33s` (removal: 3.05s, never_train: 29.51s, comparison: 8.76s)
  - Pool R1 s802: `52.69s` (removal: 3.18s, never_train: 39.24s, comparison: 10.26s)
  - Pool R2 s800: `50.79s` (removal: 7.51s, never_train: 34.24s, comparison: 9.04s)
  - Pool R2 s801: `43.57s` (removal: 3.15s, never_train: 29.95s, comparison: 10.47s)
  - Pool R2 s802: `42.26s` (removal: 2.97s, never_train: 30.24s, comparison: 9.05s)
  - Pool R3 s800: `42.52s` (removal: 2.86s, never_train: 30.15s, comparison: 9.51s)
  - Pool R3 s801: `44.05s` (removal: 3.83s, never_train: 30.50s, comparison: 9.71s)
  - Pool R3 s802: `39.25s` (removal: 3.18s, never_train: 26.84s, comparison: 9.23s)
  - Pool R4 s800: `42.73s` (removal: 3.01s, never_train: 30.29s, comparison: 9.42s)
  - Pool R4 s801: `48.84s` (removal: 3.39s, never_train: 34.99s, comparison: 10.45s)
  - Pool R4 s802: `31.90s` (removal: 2.85s, never_train: 26.04s, comparison: 3.00s)

---

## 5. Compliance Matrix

| Spec Section | Requirement | Status | Deviations & Details |
| :--- | :--- | :--- | :--- |
| **0. Goal & Rules** | Define exactness as float64, diff $\le 10^{-6}$, identical predictions; pinned CPU (cores 8 & 10, 2 threads); no existing file edits; commands < 2 min. | **COMPLIANT** | Zero existing files modified. All work pinned to cores 8 & 10, 2 threads. Long CPU run on cores 0 & 2 undisturbed. Maximum command runtime was 53.24s (< 120s limit). |
| **1. Data** | `B3seq` 22 regimes, seeds 800-802; removal sets R1 (1 class), R2 (5 classes), R3 (small domain `svamp`), R4 (regime 0 of `trec6`); never-trained stream. | **COMPLIANT** | Built via `m1.beds.build_bed("B3seq", 22, seed)` with offline transformers. Exact same seeds and order used for full, removed, and never-trained streams. |
| **2. Part A: Closed-Form Heads** | Implement `remove_classes` for `ClassMean`, `KnnMemory`, `MahalanobisProto`, `RandomFeatureRidge`, and new `ExactLda`. Measure stat, weight, prob diffs and argmax agreement. Compare `StreamingLDA` to `ExactLda`. | **COMPLIANT** | Implemented in `v1/downdate.py`. `ClassMean`, `KnnMemory`, `MahalanobisProto`, `ExactLda` are 100% exact. `RandomFeatureRidge` exact with data ($\le 7.45\times 10^{-8}$). `StreamingLDA` measured as approximate ($5.04\times 10^{-4}$ stat diff, $1.04$ weight diff). All 12 JSON files written to `results/v1/`. |
| **3. Part B: Pool Removal** | Enumerate all structures holding data in V0 pool (`build_v0`); track exposure with lineage closure; implement `remove_labels` and `scan_traces`; return JSON certificate; compare $P_{\text{full}}$, $P_{\text{rem}}$, $P_{\text{nev}}$; test removed class accuracy. | **COMPLIANT** | Implemented in `v1/pool_removal.py`. Audited 9 pool and expert structures. Tracked exposure and phylogenetic cascade. Removed class accuracy is strictly $0.0\%$ (0 predictions). Weight difference on matched experts is 0.00e+00. All 12 JSON files written to `results/v1/`. |
| **4. Audit Table** | `docs/superpowers/handoff/P2_REMOVAL_AUDIT.md` with one row per component, location, stores, classification, measured diff, exact code change; summary; 3 best cost-benefit changes; no typed numbers. | **COMPLIANT** | Written to `docs/superpowers/handoff/P2_REMOVAL_AUDIT.md`. Covers 17 distinct component rows with file:line citations. All numbers verified against `results/v1/*.json`. |
| **5. Tests & Handoff** | `tests/v1/test_v1.py` under 2 min; small fixtures; equality, formula match, error handling, planted trace, valid certificate, idempotence skip; commands logged; pytest output redirected; result file. | **COMPLIANT** | 6 unit tests passed in 2.93s. Pytest output in `V1_REMOVAL_TEST_OUTPUT.txt`. Commands logged to `V1_REMOVAL_COMMANDS.txt`. All cited symbols exist in codebase. |

---

## 6. Disclosures & Negative Results

1. **Phylogenetic Lineage Cascade (R4 Seed 802):**
   - Transitive closure over child descendants ($p \in \text{exposed} \implies \text{child} \in \text{exposed}$) can wipe out the entire pool if an early ancestor expert was exposed. In R4 seed 802, Expert 0 was directly exposed, causing all 18 subsequent experts to be dropped. The pool had 0 retained experts and lost 123 collateral classes, collapsing retained accuracy to $0.0029$.
   - **Recommendation:** Implement lineage re-anchoring to prune the parent but retrain children from their local replay reservoirs.
2. **Collateral Class Loss from Monolithic Experts:**
   - In continual MoE pools, individual experts learn multiple classes. Because expert module weights cannot be analytically downdated, dropping an expert destroys knowledge of all other classes learned exclusively by that expert. Dropping 1 expert in R1 lost 15-18 classes; dropping 4 experts in R2 lost 58 classes.
3. **Trace Leakage in Expert Prototypes (R3 Seed 801):**
   - In 1 out of 12 runs (R3 seed 801), `scan_traces` detected 44 prototype centroids in a retained expert that had cosine similarity $> 0.999$ to removed samples. The certificate honestly reported `certified_clean: False`. Prototypes must be recomputed from filtered reservoirs upon deletion.
4. **Data Dependency for Quadratic Form Downdating:**
   - `RandomFeatureRidge` and `UniPool.domain_lda` require the original training samples $(x_{\text{rem}}, y_{\text{rem}})$ to downdate their shared quadratic matrices ($G \leftarrow G - \phi^T \phi$, $S \leftarrow S - x^T x$). Without the removed data, exact unlearning of these components is impossible without storing per-class second-moment accumulators.
5. **Streaming LDA is Intrinsically Non-Removable:**
   - Online recursive covariance updates cannot be downdated without stream replay. `StreamingLDA` should be replaced with `ExactLda`.
