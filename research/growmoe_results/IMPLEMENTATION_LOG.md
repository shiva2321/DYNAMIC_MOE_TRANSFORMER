# GrowMoE implementation log (running; feeds `research/GROWMOE_REPORT.md`)

Findings and plan deviations recorded while building, in order. Numbers are measured, not predicted.
Reproduce with `python -m pytest tests/growmoe -q`; scratch measurement scripts were run from the session scratchpad.

## T4b — `growmoe/signature.py` (vMF kernel-density signature)

* Leave-one-out conformal p-values are valid. 40 independent reservoirs of 256 prototypes, 1000 fresh same-distribution
  points each: P(p ≤ 0.05) = 0.046–0.049 (theory 12/257 = 0.0467), P(p ≤ 0.10) ≈ 0.097–0.100, mean p ≈ 0.50.
  Reusing the prototypes as their own calibration set (LOO) costs no detectable validity.
* κ selection by LOO normalised likelihood tracks the true concentration within one grid step
  (true κ 30 / 60 / 100 / 250 / 600 → chosen 32 / 64 / 64 / 128 / 256) and is identical across 30 stream orders.
* **Limitation (measured):** the one-class score is weak on *broad* clusters. AUROC against a different vMF cluster,
  d = 128, 200 prototypes: κ = 10 → 0.64, κ = 30 → 0.88, κ = 100 → 0.9997, κ = 250 → 1.0. Oracle two-class AUROC at κ = 30
  would be ≈ 0.995, so the gap is one-class estimation noise (curse of dimensionality), not an information limit.
  → Phase-0 spike 1 must report the AUROC-vs-concentration curve on the real latents, and the alternative scores.
* Test hygiene: a single calibration set's coverage is Beta-distributed (±0.013 at n = 256), so the coverage test averages
  10 independent reservoirs; the reservoir-uniformity test pools 5 seeds and checks quartile occupancy.

## T5 — `growmoe/experts.py` (residual MLP expert)

* Growth is exactly function-preserving: float64 max error ≤ 1e-12 for `widen` and `deepen`; float32 ≤ 1e-5.
* Expert construction does not touch the global torch RNG (head built under `fork_rng`, weights from a per-expert generator).
* A-GEM: the plan's "conflicting replay gradient" is not automatic. With 8 seen classes at initialisation, or many unrelated
  inputs, the CE gradients of two label assignments usually *agree* (both push down the same dominant wrong class; cross-sample
  terms dominate). Only for **two** seen classes is one input under opposite labels an exact anti-parallel conflict
  (CE gradient = p1·δ for label 0, −p0·δ for label 1). The test builds that scenario and asserts its own precondition;
  the projection arithmetic is tested separately as a pure function (`agem_project`).
  → In the pools, expect A-GEM to fire rarely at initialisation and more once the heads have specialised; log how often.

## T6 — `growmoe/competence.py` (loss-change monitor)  — deviates from the plan

* **Deviation:** thresholds are in units of the warm-up standard deviation (scale-free), not an absolute `delta = 0.005`
  and a hand-set `lam`. A fixed absolute threshold is exactly the failure that left `spawn_threshold = 0.40` inert in `ishmn/`.
* **Finding:** the first version (block-bootstrap of the warm-up window only) measured **3.5 % false alarms vs 1 % nominal**
  (14/400). Cause, measured: the bootstrap threshold varies a lot between warm-ups (sd 4.65 at block 4, 2.0 at block 1,
  around the exact 13.4) and the false-alarm probability is convex in it; block 4 on 50 points is also biased low.
  Ignoring the baseline's estimation error would give 8.4 instead of 13.4.
* **Fix:** λ = max(λ_Gaussian-null, λ_bootstrap), both re-running the whole procedure (baseline estimated from `warmup`
  samples) on every replicate. Marginal false-alarm rate by stream (nominal 0.010; warm-up 50, horizon 300, 1500 streams):

  | stream | bootstrap b=1 | bootstrap b=4 | Gaussian only | **max(G, b=1)** (shipped) | max(G, b=4) |
  |---|---|---|---|---|---|
  | Gaussian i.i.d. | 0.015 | 0.037 | 0.016 | **0.011** | 0.011 |
  | exponential i.i.d. | 0.021 | 0.053 | 0.100 | **0.021** | 0.040 |
  | lognormal(0.5) i.i.d. | 0.029 | 0.061 | 0.092 | **0.029** | 0.045 |
  | AR(1) r = 0.3 | 0.111 | 0.067 | 0.115 | **0.096** | 0.055 |
  | AR(1) r = 0.5 | 0.277 | 0.082 | 0.287 | **0.247** | 0.079 |

* **Limitations to carry forward:** (i) autocorrelated streams are NOT controlled at `block = 1` and cannot be estimated from
  50 samples — feed *per-sample prequential losses* (independent given the model), arm only after the loss plateaus, and
  use `block > 1` where dependence is known; re-check on the real streams in Task 17. (ii) Detection delay is the price of
  the conservative threshold: a 3σ step is flagged in median 6 steps (p90 8), a 2σ step in 10 (p90 14), a 1σ step in 27
  (only 61 % within 30). (iii) The warm-up is 50 *monitor updates*; S1/S2/S6 have only ~8–16 batches per regime, so the pool
  must feed one loss per **sample**, not per batch.

## T7 — `growmoe/growth.py` (propose → validate → commit) — probe length set by measurement

* **The first test scenario was wrong, and the policy was right.** A 96-wide random teacher with 2000 samples is
  *data-limited at every student width* (val accuracy 0.62–0.65 for widths 4…128, and wide students overfit: val loss
  1.02 → 2.18 over 1500 steps), so declining to grow was correct. A capacity-limited task needs a low-complexity teacher:
  with a width-4 teacher, students of width ≤ 4 sit at val loss ≈ 0.75 and width 16 reaches 0.53.
* **Function-preserving growth engages slowly.** Widen (random in, zero out) vs an equally trained control, relative held-out
  gain: −1…+1 % after 30–120 probe steps, +4…+11 % after 240, +9…+19 % after 480. A 60-step probe would reject useful growth
  forever (and the back-off would then freeze the expert at its plateau).
* **The probe gain is noisy** (training noise, not validation noise, so the paired bootstrap cannot see it): over 16 seeds
  widen+4 gains +3.3 ± 3.0 % at 240 steps, +11.3 ± 5.9 % at 480, +18.3 ± 9.0 % at 960. Averaging the last checkpoints does not
  help (sd 5.9 → 5.2 while the mean shrinks): the variance is trajectory-level (new-unit init, batch order).
* **Operating characteristic of the gated policy** (12 rng seeds each, `min_gain` 2 %, Bonferroni α = 0.05 over 2 candidates):

  | scenario | accept @240 steps | accept @480 steps |
  |---|---|---|
  | capacity-limited (should grow) | 50 % | **100 %**, best-candidate gain +12.4 % |
  | sufficient capacity (should not) | 8 % | **0 %** (growth hurts: −4.9 %) |
  | data-limited, 60 samples (should not) | 0 % | **0 %** |

  → `GrowthCfg.train_steps = 480`. Cost per decision ≈ 3 × 480 optimiser steps (≈ 1.5 s at d = 32); the pool must account for
  probe compute separately from stream steps when comparing against baselines.
* Without the equally trained **control**, an under-trained expert would "benefit from growth" merely from the extra probe
  steps; the control is what makes the sufficient-capacity row 0 %.
* Not done (out of scope, noted for the report): GradMax-style initialisation of the new units' incoming weights would make
  new capacity engage faster than random-in/zero-out and allow a shorter probe.

## T8 — `growmoe/streams.py` (S1–S7) — what the benchmark actually looks like (measured)

* **S1 types — the frontend does not separate every type.** Test sample → regime of its most similar training sample
  (7 regimes, own-regime accuracy): digits 1.000, wine 1.000, breast-cancer 0.995, sinusoid 1.000, **AR(2) 0.770,
  pydoc text 0.755, stdlib code 0.895**. Text and code windows confuse each other (49 and 21 misses); AR(2) windows are
  noise-like (own-regime NN cosine 0.575 vs 0.492 for the best other regime). These are the broad, overlapping regimes where
  the one-class vMF signature is weakest (see T4b). Kept deliberately: Phase-0 spike 1 must report novelty AUROC per type, and
  the go/no-go "AUROC ≥ 0.95 on S1" may fail on exactly these three.
* **S2 permuted digits — regimes are not orthogonal.** Pixels are non-negative and the hash signs belong to output positions,
  so they cancel in cross-regime inner products: every regime shares a large common direction (each regime mean has cos
  0.71–0.80 with the global mean; 38 % of squared norm lies along it; class-mean |cos| across regimes 0.47, not ~0.09).
  They are still cleanly separable: own-regime NN cosine 0.959 vs 0.715 for the best other regime, 100 % of 30 regimes' test
  samples closer to their own regime. (Quasi-orthogonality of unrelated regimes holds only for S6, which is generated
  directly on the sphere: class-mean |cos| ≈ 0.07.)
* **S3 drift:** class-mean cosine 0.835 between consecutive thirds of the stream and 0.706 between first and last; stage test-set
  mean directions φ=0 vs φ=1 have cos 0.61 — gradual, moderate total drift.
* **S4:** realised outlier rate 0.0100 (asked 0.01), label noise among inliers 0.0982 (asked 0.10), n = 10,000.
* **S5 — deviation from the spec.** With the specified 2,000 samples at d = 128 every teacher is data-limited: best student test
  accuracy 0.36–0.57 whatever the student width (8/32/128), even for the width-8 teacher. At 20,000 samples the ladder becomes
  informative: teacher w8 d1 → students h8/h32/h128 = 0.78/0.82/0.82 (saturates); w32 d2 → 0.53/0.56/0.66 (climbs with width);
  w128 d3 → 0.47/0.45/0.49 (still hard). Default is now `n_train = 20000`, `n_test = 1000`, with an `epochs` option.
  Teacher head biases are balanced (class frequencies 0.11–0.15; an unbalanced random teacher reached 0.8 for one class).
* **S6:** 18/60 regimes related, class-mean cosine to parent 0.893–0.897 (target 0.9, by bisection on the rotation scale),
  6 revisit segments.

## T9 — `growmoe/metrics.py`
* Hand-verifiable throughout: AUROC with ties as ½, spawn precision/recall/ratio/time on a constructed event list (first spawn
  per regime at/after its onset is the true positive; duplicates, owner-less experts and pre-onset spawns are false
  positives), exact forgetting / backward / forward transfer on a hand-built accuracy matrix (with `learned_at` for revisit rows),
  and `mean_ci` / `paired_diff` (t interval, paired-t and Wilcoxon p) cross-checked against the t distribution (p = 0.03268 for
  the t = 3.207, df = 4 example).

## T10 — `growmoe/baselines.py` (B0–B5) — measured behaviour (S2 prefix, 3 regimes, 640 train/regime, 5 seeds)

| baseline | mean final accuracy | note |
|---|---|---|
| B0 fine-tune | 0.47–0.60 (regime 0: **0.18–0.39**) | forgets heavily |
| B1 joint | 0.90–0.94 | ceiling (extra compute and memory) |
| B2 replay (global reservoir 2048) | 0.88–0.91 | |
| B3 static MoE (8 experts, top-2) | 0.47–0.58 | a fixed expert count does not cure forgetting |
| B4 oracle-routed | 0.87–0.89 | **zero forgetting exactly** (tested by equality); accuracy capped by only 60 steps/regime |
| B5 DP-means λ = 0.8 | 0.84–0.89, 5–6 experts for 3 regimes | |

* **Plan guess corrected:** "B4 > 0.95" is false (0.886) because per-regime training is short; B4's defining property is zero
  forgetting, which is what the test now pins.
* **S6 is a weak forgetting test at small scale:** with 3 near-orthogonal regimes B0 keeps regime 0 at 0.73 (little interference
  between orthogonal subspaces). S2 (shared DC component) is the stream on which forgetting bites.
* **B5's fixed λ is brittle (the motivation for conformal gating):** λ = 0.3 / 0.8 / 1.2 / 1.8 → 127 / 5 / 2 / 1 experts and mean
  accuracy 0.88 / 0.89 / 0.72 / 0.48 (too large a λ merges regimes; too small fragments them).
* **B3 routing collapse is real:** without the load-balancing bias one expert takes 97 % of selections; with `bias_rate = 0.05`
  the maximum share falls to 0.56 (coefficient of variation 1.44 → 0.73); `bias_rate = 0.2` oscillates (0.76).

## T11 — Phase-0 spikes (scripts in `research/spikes/`, raw results in `research/growmoe_results/phase0/`)

### Spike 1 — novelty scores (S1/S2/S6; first-visit regime vs seen regimes; 5 seeds; mean [95 % CI])
| detector | S1 | S2 | S6 |
|---|---|---|---|
| Gaussian-Mahalanobis (uncalibrated min-distance) | 0.991 | 1.000 | 0.996 |
| Gaussian-Mahalanobis conformal (split) | 0.977 | 0.998 | 0.993 |
| kNN (k = 5) | 0.963 | 1.000 | 0.998 |
| **vMF-KDE conformal (256 prototypes)** | **0.973 [0.967, 0.979]** | **0.999** | **0.993** |
| vMF-KDE raw | 0.963 | 1.000 | 0.995 |
| vMF-KDE conformal, 64 / 1024 prototypes | 0.969 / 0.975 | 0.993 / 1.000 | 0.984 / 0.995 |
| vMF-KDE conformal, κ grid to 2048 | 0.973 | 0.999 | 0.993 (identical: the κ cap is not limiting) |
| KDE ∪ Mahalanobis (Bonferroni) | 0.972 | 0.998 | 0.995 (no gain) |
| VAE-ELBO | 0.968 | 0.912 | **0.779** |
| key-activation | **0.744** | 0.999 | 0.902 |
| RND | 0.986 | 0.984 | 0.925 |

* **Go/no-go (pre-registered): GO.** vMF-KDE conformal 0.973 ≥ 0.95 on S1 and 0.999 ≥ 0.85 on S2.
* The uncalibrated Mahalanobis lead on S1 (0.991) is not usable and mostly disappears once it is calibrated like the KDE
  (0.977, overlapping CI): no evidence that a Gaussian signature beats the KDE under the same conformal gate.
* The whole S1 gap is **one arrival: stdlib code after pydoc text** (KDE-conformal 0.859, kNN 0.783, Mahalanobis 0.947,
  RND 0.943, VAE 0.936). Every other S1 arrival is ≥ 0.99 for the KDE.
* **Related-but-new regimes (S6, class-mean cos 0.9 to a parent):** KDE 0.980, kNN 0.993, Mahalanobis 0.986 but
  **VAE 0.590, key-activation 0.657, RND 0.759** → approaches B and C will tend to reuse the parent instead of spawning.
* **S1 ordered type pairs (fit on ONE type, test each other; 42 pairs, 5 seeds):** mean AUROC KDE 0.935, Mahalanobis 0.947,
  kNN 0.945, RND 0.942, **VAE 0.893, key-activation 0.723**; pairs below 0.8: KDE 6, Mahalanobis 6, kNN 5, RND 6, **VAE 11,
  key-activation 26**. **C5 is supported in letter** (VAE fails on 11/42 pairs, min 0.47, e.g. AR(2)→wine 0.47). But the KDE
  also fails on 6 pairs (min 0.36, ar2→digits): a signature fitted on a DIFFUSE regime (AR(2)) is a weak one-class reference for
  every density method — the failure A's competence monitor (known-input / wrong-output) is meant to catch.
* **Defaults unchanged:** 256 prototypes, κ grid ≤ 256, conformal per-expert calibration (it also beats the raw max-score
  rule across regimes, 0.973 vs 0.963 on S1, because per-expert calibration makes scores comparable).

### Spike 2 — child initialisation (S6 parent → child; 10 seeds; two noise levels)
* **First attempt discarded:** at the default S6 noise every arm reached 100 % accuracy within 10 steps (the task could not
  discriminate initialisations). Redesigned to the low-sample regime a newborn expert faces (batches of 8, accuracy after
  every step from step 0), noise σ = 1.0 and 2.0.
* Area under the learning curve vs scratch (σ = 1.0 / 2.0):

  | arm | unrelated | related, cos 0.9 |
  |---|---|---|
  | clone | −0.035 / **−0.122** | +0.042 / +0.134 |
  | clone + fresh head | −0.004 / −0.017 | +0.012 / +0.086 |
  | inherit(ρ = 0.75 × (1 − affinity)) — the spec's rule | −0.008 / −0.038 | +0.042 / +0.128 |
  | **shrink-and-perturb, λ = affinity** | **+0.000 / +0.002** | **+0.042 / +0.132** |

* **Pre-registered go/no-go for the ρ-redraw rule: NO-GO** (worse than scratch on unrelated, p < 0.001 at σ = 2).
  **Design change, post-hoc and recorded:** A's child initialisation is `MLPExpert.shrink_perturb(λ = affinity)`
  (λ = 0 is exactly scratch, λ = 1 a clone); `inherit(ρ, σ)` stays as an ablation arm. To be re-validated end to end in Phase 1.
* The trained **head** carries both the harm (unrelated) and the benefit (related): only an affinity-weighted blend is right.
* **Affinity** (`VmfSignature.affinity`) tracks relatedness almost linearly: unrelated 0.01, cos 0.5 → 0.29–0.39,
  0.7 → 0.54–0.64, 0.9 → 0.84–0.88, 0.97 → 0.94–0.96.
* Zero-shot accuracy of an inherited child on a related regime (cos ≥ 0.9) is 0.97–1.0: such a regime needs **no new expert**
  at all; the extend-vs-spawn probe must find that (spawning there only costs parameters).

### Spike 3 / growth policy (T7 revised) — what the S5 ladder forced
* **Greedy increments are myopic.** Original policy (+50 % candidates) on S5 (3 seeds): w8 teacher grows to width ≈ 10.7 and
  *beats a static width-128 student* (0.803 vs 0.779); data-limited n = 2000: no growth anywhere (correct); but for the w32
  d2 teacher it never grew (acc 0.570 vs 0.598 for a static wide student). Measured gain of widening a width-4 student,
  relative held-out loss vs the equally trained control: +4 units **+0.0 %**, +12 → +1.7 %, +28 → +2.7 %, +60 → **+5.3 %**.
  → candidates are now `widen` to {1.5, 4, 16} × the current width plus `deepen`; the **cheapest sufficient** candidate is
  committed (largest-gain would always pick the biggest jump for +0.1 %).
* **A loss-only gate falsely accepts for memorised experts.** 60 samples, width-64 student: held-out loss 2.81 before the probe,
  **4.6 for the equally trained control** (Adam keeps inflating logits at train loss ≈ 0) and 3.9–4.2 for the widest shadow —
  "+11 % over the control" while 49 % worse than where the expert started; held-out ERROR got worse (−2.5 %). → new gate:
  a candidate must also be strictly better than the **status quo** (pre-probe held-out loss). Free, threshold-free.
* A held-out-error gate was measured instead and rejected: it is correct on all five scenarios but has far less power at
  realistic validation sizes (A: +4 units p = 0.08–0.23).
* **Power warning for Phase 1:** a pool expert's validation split is a fraction of its ≤ 256-sample replay buffer. Significance
  tests will rarely pass at that size; growth will be conservative unless experts that may grow get larger buffers.

### Spike 3b (final growth policy, 3 seeds) — test accuracy grown / static width-4 / static width-128
| n_train | teacher | commits | final width | params | grown | w4 | w128 |
|---|---|---|---|---|---|---|---|
| 20000 | w8 d1 | 2.3 | 14.0 | 5,270 | **0.816** | 0.721 | 0.779 |
| 20000 | w32 d2 | 1.0 | 48.0 | 14,008 | **0.594** | 0.567 | 0.598 |
| 20000 | w128 d3 | 0.7 | 44.0 | 12,980 | 0.497 | 0.510 | 0.482 |
| 2000 | w8 d1 / w32 d2 / w128 d3 | **0 / 0 / 0** (all 9 runs) | 4 | 2,700 | 0.552 / 0.409 / 0.369 | 0.562 / 0.414 / 0.373 | 0.527 / 0.438 / 0.401 |

Pre-registered expectation met: grows when the teacher is larger and data suffice (w8: +0.095 over w4 and +0.037 over w128 with 1/7 of
its parameters; w32: matches w128 within 0.004 at 40 % of its parameters), never grows when data-limited. The hard w128 d3 teacher is
data/optimisation-limited: growth is neutral (-0.013 vs w4, inside noise).

## STATUS at hand-over to the Gemini-implements / Claude-reviews workflow (see docs/superpowers/handoff/GROWMOE_HANDOFF.md)
* 240 tests pass (`python -m pytest tests/growmoe -q`, 79 s). Tasks 0-12 and the Phase-0 spikes (Task 11) are complete; Tasks 13-22 remain.
* Nothing has been committed (everything under growmoe/, tests/growmoe/, research/, docs/ is untracked); `ishmn/` is untouched.
