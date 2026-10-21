# Component Cards

This document provides a single-page reference card for each component in the Scope Charter Component Register (`docs/superpowers/handoff/SCOPE_CHARTER.md`).
Status assignments are authoritative and governed strictly by empirical verification under controls.

---

## 1. Lorentz Model, Fused Attention, and Norm-Gated Neuron (Maths)
- **Location:** `geometry/lorentz.py`, `geometry/attention.py`, `geometry/neuron.py`
- **Status:** **VERIFIED (identities, tests)**
- **What it is:** Exact mathematical operations on the hyperboloid $\mathbb{H}^D$ (Lorentz model). Implements Minkowski inner products $\langle u, v \rangle_L = -u_0 v_0 + \sum_{i=1}^D u_i v_i$, squared Lorentzian distance $d_L^2(u, v) = -2 - 2 \langle u, v \rangle_L$, tangent-space projection, exponential/logarithmic maps, centroid aggregation, and norm-gated nonlinearities.
- **Evidence:** Verified by 131 tests in `tests/test_lorentz_attention.py`, `tests/test_geometry.py`, and `tests/test_geometry_consistency.py`. Pass analytical identity checks against Poincare conversions, float64 precision checks, and antipodal accuracy limits.
- **Limits:** Pure mathematical primitives. In self-attention sequence models, no model-quality benefit over Euclidean models with temperature was demonstrated.

---

## 2. Hyperbolic Tree Embeddings (Small Dimension)
- **Location:** `geometry/poincare.py`, `geometry/lorentz.py`, `research/yggdrasil_poc/`
- **Status:** **VERIFIED (static tree)**
- **What it is:** Continuous embedding of hierarchical directed trees (e.g. 1,093-node trees) into hyperbolic space $\mathbb{H}^d$ using distance-preserving tree embedding objectives.
- **Evidence:** `research/yggdrasil_poc/poc_a4_tree_embedding_tuned.py` results. In low dimensions ($d = 4..16$), hyperbolic embeddings achieve significantly lower distortion and higher reconstruction accuracy than Euclidean embeddings of the same dimension.
- **Limits:** Demonstrated for static hierarchical graphs and trees; not proven beneficial inside dynamic sequence representations or autoregressive language models.

---

## 3. Snapped Rotation State Machine (`SnappedRotationState`)
- **Location:** `geometry/snapped_state.py`, `research/yggdrasil_poc/poc_b_state_tracking.py`
- **Status:** **VERIFIED on $Z_6$, $S_3$; NO DEMONSTRATED ADVANTAGE over GRU (both fail on $S_5$)**
- **What it is:** Recurrent discrete state machine for algebraic sequence tracking. Input symbols apply orthogonal transitions $R_g \in SO(k)$ generated via matrix exponentials of anti-symmetric generators. At each step, the continuous state vector is snapped to the nearest entry of a learned spherical codebook on $S^{k-1}$ using straight-through vector quantization.
- **Evidence:** `research/yggdrasil_results/poc_b_table.txt`, `tests/test_snapped_state.py`, and `research/yggdrasil_results/s5_comparison_table.txt`. Achieves $100.0\% \pm 0.0\%$ exact sequence product accuracy across all lengths $T \in \{16, 32, 64, 128, 256\}$ across all 10/10 test seeds on both abelian $Z_6$ and non-abelian $S_3$ (zero label leakage).
- **Limits:** Requires codebook size scaling with group order ($K = 4|G|$); fails on larger groups like $S_5$ ($|G|=120$) where random initialization remains at chance across codebook sizes $K \in \{480, 960, 1920\}$ ($0.7\% - 1.0\% \pm 0.5\% - 0.7\%$, chance $0.83\%$). A standard GRU baseline (5,094 parameters) is equally exact on $Z_6$ ($99.8\% \pm 0.2\%$) and $S_3$ ($100.0\% \pm 0.0\%$) at $T=256$, and likewise fails at chance on $S_5$ ($0.8\% \pm 0.6\%$). Across all tested groups, no empirical niche or advantage over a standard GRU is demonstrated. Not integrated directly into transformer attention blocks.



---

## 4. Hyperbolic Attention Heads for Language Models
- **Location:** `yggformer/` (`attention.py`, `block.py`, `model.py`)
- **Status:** **NOT SUPPORTED (not proven at this scale)**
- **What it is:** Multi-head transformer self-attention where queries and keys are projected onto the hyperboloid, with attention logits computed via negative scaled Lorentzian distance $-d_L^2(q, k) / \tau$ and optional torus phase modulation.
- **Evidence:** `research/yggformer_results/e6_lm_results.json`. After properly controlling for temperature scaling, Euclidean attention with learnable temperature (`euclid_temp`, PPL $9.47 \pm 0.21$) matches hyperbolic attention (`hyp`, PPL $9.63 \pm 0.30$; `hyp_phase_bias_null`, PPL $9.41 \pm 0.22$). All Bonferroni-corrected $p$-values vs Euclidean are $> 0.2$. In `hyp_phase_bias`, curvature collapsed to flat Euclidean ($c = 0.03$). Top-16 keys capture $96.5\% - 98.9\%$ of mass vs $98.1\%$ for Euclidean, disproving natural sparsity.
- **Limits:** Not proven beneficial at this scale. Do not deploy for general sequence modeling or LM self-attention.

---

## 5. Branch-and-Bound Tree Retrieval
- **Location:** `yggformer/tree_search.py`, `research/yggformer_results/bench_y5_sparse_retrieval.py`
- **Status:** **NOT USEFUL at $N \le 4096$**
- **What it is:** Hierarchical spatial clustering of Lorentz key vectors into bounding cones, searching top-$B$ attention keys via branch-and-bound pruning without exhaustive pairwise distance computation.
- **Evidence:** `research/yggformer_results/e5_results.json`. While branch-and-bound prunes keys algorithmically (evaluating only $2.2\% - 4.5\%$ of keys at $N=4096$ while capturing $100\%$ mass ratio), in-memory Python tree traversal requires $4.56 - 12.39$ ms, which is $36\text{x} - 60\text{x}$ slower than a fair vectorized dense GPU baseline ($0.085 - 0.216$ ms).
- **Limits:** Impractical for in-VRAM sequence attention at $N \le 4096$. Useful only if keys exceed GPU memory capacity ($N \gg 4096$) or reside on external hierarchical storage.

---

## 6. Fast Pairwise-Distance CUDA Kernel
- **Location:** `rust_yggdrasil/src/gpu_kernels.cu`, `geometry/gpu_kernel.py`
- **Status:** **PARTIAL (2.6x vs engine kernel; 4x target unmet)**
- **What it is:** Custom tiled CUDA kernel computing pairwise hybrid distance matrices across Lorentz, torus, and Euclidean coordinates.
- **Evidence:** `research/yggformer_results/e7_results.json`. Achieves $1.04$ ms execution time at $N=M=4096$ (a $67.5\text{x}$ speedup over naive PyTorch eager reference at $70.2$ ms). Compared against the existing naive engine kernel ($2.72$ ms), speedup is $2.6\text{x}$.
- **Limits:** The pre-registered acceptance criterion of $\ge 4\text{x}$ against the engine kernel was unmet ($2.6\text{x}$ achieved). Attention-preparation fusion was measured at $0.83 - 0.97$ ms and was not implemented.

---

## 7. GrowMoE Phase 1 (Pools A, B, C, Bench)
- **Location:** `growmoe/` (`pools/`, `bench.py`)
- **Status:** **VERIFIED (R10b approved)**

- **What it is:** Continual lifelong mixture-of-experts architecture using conformal novelty detection, Bayesian non-parametric growth, and phylogenic expert branching.
- **Evidence:** `research/growmoe_results/phase1/RESULTS.md`. On S2 (interleaved regimes), Pool A (PhyloMoE) achieves $87.3\% \pm 1.4\%$ accuracy with $0.5\%$ forgetting using $3.5$ experts (vs ReplayNet $67.7\%$, oracle $81.0\%$). Calibrated gate achieves empirical FNR $= 0.00\%$ and novelty AUROC $0.98 - 1.00$.
- **Limits:** Lower accuracy on heterogeneous mixed-type stream S1 ($56.0\%$ with $21.5\%$ forgetting). Phase 2 (vertical scaling, `stack.py`, and routing diagnosis D1) is currently in progress.
- **Update 2026-10-10 (M2-A):** the Phase 1 and M0 numbers above/below were produced with the competence monitor armed at an expert's birth (`arm_monitor_after_infancy=False`). That setting made every newborn raise a false drift alarm; the repaired default is ON (card 13). `python -m growmoe.bench --phase 1` pins the old behaviour so the recorded Phase 1 results still reproduce.

---

## 8. ISHMN / Universal Cognitive Engine
- **Location:** `ishmn/`
- **Status:** **UNVERIFIED (headline claims failed audit)**
- **What it is:** Legacy monolithic hybrid neuro-symbolic cognitive engine.
- **Evidence:** Audit findings and memory notes dated 2026-09-21.
- **Limits:** **FROZEN: Do not touch, do not modify, do not integrate.**

---

## 9. Geometry Core Protocol Library & Health Diagnostics
- **Location:** `geomlab/protocol.py`, `geomlab/diagnostics.py`, `tests/geomlab/test_z0_infrastructure.py`
- **Status:** **VERIFIED (Infrastructure & Telemetry)**
- **What it is:** Automated experimental protocol enforcement tools (`check_prereg`, `tune_lr` with boundary expansion, `run_seeds`, `paired_stats`, `bonferroni`, `write_results`, `render_poc_b_table`, `validate_report`) and representation geometry health telemetry (`effective_rank`, `isotropy`, `uniformity_loss`, `intrinsic_dim_twonn`, `hubness_skew`, `norm_stats`, `operating_point_adapter`).
- **Evidence:** `research/geomlab_results/z0/tables.txt`, `research/geomlab_results/z0/AMENDMENT.md`, `tests/geomlab/test_z0_infrastructure.py`. Verifies effective rank ($31.97 \pm 0.00$ on 32-D Gaussian; $1.000$ on rank-1 line), spherical isotropy/uniformity, TwoNN intrinsic dimension ($0.979 \pm 0.022$ on 1-D circle, $1.003 \pm 0.013$ on 1-D segment, $1.985 \pm 0.045$ on 2-D manifold, $2.957 \pm 0.044$ on 3-D manifold in $\mathbb{R}^{20}$), exact byte-for-byte dry-run table reproduction (2003 bytes CRLF), and 100% sensitivity report diff validation catching single-digit alterations.
- **Limits:** TwoNN requires random Poisson-like local sampling (not deterministic lattice points) and float64 exact distance computation (`compute_mode="donot_use_mm_for_euclid_dist"`) to avoid catastrophic floating point cancellation on closely spaced pairs. TwoNN is reliable for intrinsic dimensions up to about $5-10$ at $N \sim 3000$; under-estimates true dimension by $\approx 25\%$ at $d=32$ ($24.6 \pm 0.3$ vs $32$) due to finite-sample density decay in high-dimensional space; never use TwoNN as an uncalibrated pass/fail health gate in high dimensions without this caveat. Pairwise distance matrices subsample to $N \le 4000$ to maintain lean memory consumption (<4 GB).

---

## 10. GrowMoE Routing Rule (`accept_then_logf`)
- **Location:** `growmoe/pools/phylo.py`, `growmoe/config.py`
- **Status:** **PROPOSED (Pool A routing rule; review R11d approved)**
- **What it is:** Test-time routing arbitration rule for PhyloMoE. Evaluates conformal inlier p-values for all experts ($p_k(x)$); determines eligibility by $p_k(x) \ge q$; selects the expert with the highest signature log-density $\log f_k(x)$ among eligible candidates (falling back to argmax log-density if no expert accepts). Skips the test-time supersede filter (`_filter_superseded`), preventing mature parents from being prematurely eliminated by newly spawned children in overlapping feature space.
- **Evidence:** `research/growmoe_results/r11d/rule_selection.json` and `research/growmoe_results/r11d/rule_selection.md`. On heterogeneous types stream S1 at benchmark operating point ($q = 0.02$ everywhere), test accuracy improves from $55.96\% \pm 7.74\%$ (`default`) to $69.82\% \pm 2.01\%$ (`accept_then_logf`), a $+13.86$ point gain ($95\%$ CI $[+6.02, +21.69]$, $p = 3.11\text{e-}3$). Supersede filter isolation confirms the entire gain originates from skipping the filter ($55.96\%$ with filter vs $69.75\%$ without filter). On streams S2, S3, S4, performance is within noise at benchmark configuration (S2: $-0.28\%$ $[-0.77\%, +0.21\%]$; S3: $0.00\%$; S4: $+0.06\%$ $[-1.00\%, +1.12\%]$). Conformal `unknown` flags are identical across all routing rules.
- **Limits:** **Concept remap limitation (L3):** When identical input coordinates receive remapped class labels (concept drift after a competence alarm), `default` routes $100\%$ of samples to the successor, whereas `accept_then_logf` routes $\approx 50\%$ (measured $47\text{--}54\%$ per seed; mean $49.9\%$) to the successor because the supersede filter is skipped at test time and both experts accept the input region; exact log-density ties go to the older expert by list order. A true label-evidence arbiter is required for remapped label drift (deferred to Phase 2). The supersede filter is retained during training-time assignment.

---

## 11. Spherical Residual Stream (`SphericalStream` / nGPT Family)
- **Location:** `geomlab/spherical_stream.py`, `tests/geomlab/test_z1_spherical_stream.py`, `research/geomlab_results/z1/`
- **Status:** **PARTIAL - UNDER REVIEW (was recorded as VERIFIED; see the review note below)**
- **Review note (2026-10-12, Claude; lab notebook E-008, F-08, Q-09):** the final-loss and token-speedup comparisons are made against the baseline's *final* validation loss, and the baseline over-fits: its mean validation loss is lowest at 1.903 at step 400 (mean of per-seed minima 1.898) and rises to 2.3265 at step 1000, which is **lower than nGPT's final 1.9525**. `compute_speedup` defines the speedup as the tokens an arm needs to reach the baseline's final loss, which rewards an over-trained baseline (an exact copy of the baseline scores 4.67x). nGPT's own validation curves were not stored, so whether it beats an early-stopped baseline cannot be read from `results.json`. The numbers below are reported as originally recorded; the claim needs a re-run with a fair baseline target and stored curves for every arm before it is VERIFIED.
- **Original status line:** VERIFIED (Token Speedup >= 2.0x on $S^{d-1}$, Isolating Control Outperformed; Eager Wall-Clock Parity without Fused Kernels)
- **What it is:** A hyperspherical normalized residual stream architecture for sequence models. All hidden representation vectors $h_t$ are strictly constrained to the unit sphere $S^{d-1}$ via $\ell_2$ normalization in float32. Matrix weights are unit-norm normalized along output coordinates. Residual stream updates follow geodesic interpolation / eigenstep scaling:
  $$h_{t+1} = \text{Norm}(h_t + \alpha_A \cdot (A(h_t) - h_t))$$
  with learnable per-dimension metric scaling vectors ($S_A, S_M, S_{qk}, S_{uv}$).
- **Evidence:** `research/geomlab_results/z1/results.json`, `tables.txt`. Benchmarked across 10 independent test seeds (0..9) on language modeling (8.19M tokens/seed, 1,000 steps, context 256, RTX 3060 bfloat16).
  1. Final validation loss reduced from $2.3265 \pm 0.0206$ (`base`) to $1.9525 \pm 0.0111$ (`ngpt`), a paired difference of $-0.3740$ ($95\%$ CI $[-0.3962, -0.3517]$, $p = 2.97\text{e-}11$, Cohen's $d_z = -12.03$).
  2. Token speedup of $2.10\times \pm 0.15$ ($1.95\times - 2.45\times$ across seeds, exceeding the $\ge 2.0\times$ pre-registered criterion).
  3. Decisively outperforms the isolating control `base_scaled` ($2.0767 \pm 0.0161$), with paired difference $-0.1242$ ($p = 9.05\text{e-}8$, Cohen's $d_z = -4.87$), proving that the speedup is an intrinsic consequence of hypersphere geometry ($S^{d-1}$) and geodesic updates rather than additional learnable scale parameters.
  4. Representation geometry health: Eliminates unconstrained norm explosion (base norm $63.97 \pm 14.33$ vs ngpt $1.000 \pm 0.000$), maintains high effective rank ($177.08 \pm 0.81$ vs collapsed $62.04 \pm 1.89$ for `base_scaled`), maintains high spherical dispersion (uniformity loss $-3.026 \pm 0.024$), and cuts representation hubness skewness by more than half ($1.11 \pm 0.09$ vs $2.31 \pm 0.33$).
- **Limits:** Wall-clock overhead in unfused eager PyTorch: Per-step execution time is $133.8 \pm 2.1$ ms (`ngpt`) vs $64.2 \pm 0.5$ ms (`base`), yielding net wall-clock speedup of $1.01\times \pm 0.07$. Production deployment requires custom fused Triton/CUDA normalization kernels to turn the $2.1\times$ token speedup into real wall-clock training throughput.

---

## 12. Associative Memory Geometry: Recall per Byte (Self-Dual Cones & Normalized Delta)
- **Location:** `geomlab/memory_geometry.py`, `tests/geomlab/test_z2_memory_geometry.py`, `research/geomlab_results/z2/`
- **Status:** **VERIFIED (Self-Dual Cones / KATA-PSD Achieves Highest Continuous Capacity per Byte; Legacy VMF Competitive in Discrete Slots)**
- **What it is:** Geometric state representations for recurrent associative key-value memory. Evaluates self-dual positive semi-definite cone feature maps (`kata_psd`, expanding outer product capacity into rank-1 PSD cones), $\ell_2$-normalized key projections (`delta_l2`, transforming erasure into exact orthogonal projections on $S^{d-1}$), geometric Power-Spectrum of Time decay (`post_decay`), triadic tensor products (`triadic`), and discrete spherical von Mises-Fisher attention (`legacy_vmf` from `ishmn/memory.py`).
- **Evidence:** `research/geomlab_results/z2/results.json`, `tables.txt`. Benchmarked across 10 independent test seeds (0..9) on Multi-Query Associative Recall (MQAR) at context length $T=128$, pair sweep $P \in \{4, 8, 16, 32\}$, and length extrapolation to $T=512$ (RTX 3060 bfloat16/float32 state).
  1. `kata_psd` achieves the highest recall accuracy at dense capacity: $69.4\% \pm 2.8\%$ at $P=16$ pairs and $45.8\% \pm 2.2\%$ at $P=32$ pairs (vs $10.2\% \pm 3.4\%$ and $5.3\% \pm 1.8\%$ for standard `gated_delta`), a paired difference of $+59.16\%$ ($95\%$ CI $[+55.75\%, +62.57\%]$, $p = 2.25\text{e-}11$, Cohen's $d_z = +12.41$, Bonferroni $p = 1.58\text{e-}10$).
  2. Projecting keys to the unit sphere in delta rules (`delta_l2`) more than doubles recall over unnormalized `gated_delta` ($24.0\% \pm 3.1\%$ vs $10.2\% \pm 3.4\%$ at $P=16$, paired diff $+13.80\%$, $p = 3.35\text{e-}5$, Cohen's $d_z = +2.40$), confirming the geometric hypothesis that orthogonal key projection prevents associative erasure cross-talk.
  3. `legacy_vmf` (ISHMN continuous spherical von Mises-Fisher memory with discrete slots) achieves $63.9\% \pm 1.6\%$ at $P=16$ and $44.3\% \pm 1.6\%$ at $P=32$ using only $9,344$ state bytes ($3.5\times$ smaller state footprint than matrix memories), but incurs $2.1\times$ higher compute runtime ($760.5 \pm 28.8$ ms/step vs $360.6 \pm 12.7$ ms/step for `kata_psd`).
  4. Length extrapolation ($T=128 \to 512$ at $P=8$): All geometric recurrent arms preserve recall without length collapse (`kata_psd` $-0.2\%$, `legacy_vmf` $-0.4\%$, `delta_l2` $-0.5\%$).
- **Limits:** In eager PyTorch, `kata_psd` requires quadratic cone outer products per step ($360.6$ ms/step); parallel scan formulations require specialized chunked associative scan kernels. `legacy_vmf` is sequential and compute-bound per slot.

---

## 13. Competence-Monitor Arming After Infancy (`arm_monitor_after_infancy`, ON by default since 2026-10-10)
- **Location:** `growmoe/config.py` (`PhyloCfg.arm_monitor_after_infancy`), `growmoe/pools/phylo.py` (`observe`, `_forget_competence_history`), `growmoe/competence.py` (docstring), `tests/growmoe/test_gm_m2a_arming.py`, `research/growmoe_results/m2a/`.
- **Status:** **VERIFIED (pre-registered confirmatory test on fresh seeds 10-19; all 7 statements PASS)**
- **What it is:** A newborn expert's `LossChangeMonitor` is neither fed nor tested while the expert is an infant (`infant_calls` batches), and its history is cleared when infancy ends, so its baseline is measured on the settled expert. Before, the baseline came from the untrained network (mean loss 2.47); after the first batch of training the newborn's held-out loss rose to 4.3, the one-sided CUSUM read that as concept drift, fired on the newborn itself at its second batch, and the alarm path replaced its weights with a 60-step fine-tune on 32 samples (about one false alarm per regime).
- **Evidence:** `research/growmoe_results/m2a/RESULTS_M2A.md`. With the repair and the step budget re-tuned on dev seeds (5 per batch on function streams), the pool beats the best single network (M0's width-4096 / buffer-8192 replay net) by $+10.0$ points [$+8.0$, $+12.0$] on F-15 and $+11.6$ [$+9.5$, $+13.8$] on F-30 (bar $+5$); the repair alone is worth $+7.6$ / $+7.2$ over the M0 frozen pool. Guards: S1 $+3.9$, S2-5 $+1.0$, S2-30 $+2.3$, S4 $-0.4$ (intervals in the results file), and the alarm still fires on label remaps (S7: $4.1$ alarms per 4-regime run). 7 unit tests; breaking the fix fails exactly the 2 behaviour tests.
- **Limits:** One synthetic stream family carries the accuracy claim; wide intervals on three guards; the S1 budget (125 steps) was picked on three noisy dev seeds; the repair was found after the M0 verdict (M0's B1 FAIL for the frozen pool stands). Efficiency is NOT improved: at 30 regimes the pool uses 1.4x the single network's memory and about the same multiply-adds, and the gate scan is 97 % of its compute. Pass `arm_monitor_after_infancy=False` to reproduce Phase 1 / M0 (`bench.run_phase1` and `research/growmoe_results/m0/run_m0.py` already do).

---

## 14. Batch-Consistent Assignment (`batch_vote`, ON by default since 2026-10-12)
- **Location:** `growmoe/config.py` (`PhyloCfg.batch_vote`, `batch_vote_share`, `batch_vote_min`), `growmoe/pools/phylo.py` (`_batch_consistent`, `observe`), `growmoe/streams_m1.py` (`function_regimes(overlap, mixed_tail)`), `tests/growmoe/test_gm_rv2a_*.py`, `research/growmoe_results/rv2a/`.
- **Status:** **VERIFIED (pre-registered confirmatory test on fresh seeds 20-39; all 11 statements PASS, every one as expected)**
- **What it is:** When one expert holds at least half of a batch's accepted samples (and at least 8 were accepted), all accepted samples of that batch train that expert. Before, each sample picked its expert alone, so when regimes overlapped some samples trained the wrong expert (training purity 78 % at centre cosine 0.3 in the dev-seed spikes). Novel samples are untouched; prediction is unchanged.
- **Evidence:** `research/growmoe_results/rv2a/RESULTS_RV2A.md` (20 paired seeds, A_vote - A_new). Overlapping regimes at centre cosine 0.3: +11.4 points [+10.0, +12.9] (33.6 % -> 45.1 %); at 0.6: +5.8 [+4.5, +7.1]. Function streams: F-30 +6.4 [+5.4, +7.5] (43.3 % -> 49.8 %), F-15 +3.0 [+1.7, +4.3]. Guards: S1 +1.4 [-0.2, +3.1], S2-5 +0.3, S2-30 +2.3, S4 +9.7, no overlap +1.7, 50/50 mixed batches +5.5, and the label-remap alarm still fires (4.1 per run). With the vote the pool builds exactly one expert per regime on F-30 (30.0 experts, 0 alarms, 0 extends).
- **Limits:** Known regimes that later arrive mixed 70/30 at zero overlap lose about 1.3 points on the minority regimes (-1.3 [-2.8, +0.2]; guard bar -3.0). It does NOT stop regimes merging at centre cosine >= 0.6 (25.3 % at 0.6 against about 48 % for oracle-routed experts; stage 3 of the routing-v2 spec is open). The V1 bar was lowered from +8.0 to +6.5 by the pre-registration rule (dev effect +10.5 on the product stream); the result clears both. Synthetic function-regime streams carry most of the accuracy claim. Efficiency is not improved (stage 2, the fingerprint shortlist, is not built).


## 15. Stage 4: Learned Router and Transfer-Based Spawn Rule (`router`, `spawn_rule="transfer"`, ON by default since 2026-10-21; growth from the fit gap is available, not the default)
- **Location:** `growmoe/config.py` (`PhyloCfg.router`, `router_mode`, `spawn_rule`, `growth_rule`, `PRE_STAGE4`), `growmoe/router.py`, `growmoe/transfer.py`, `growmoe/gapgrow.py`, `growmoe/pools/phylo.py` (`_refit_router`, `_route_by_router`, `_transfer_verdict`, `_gap_growth_step`), `tests/growmoe/test_gm_rv4_*.py`, `tests/growmoe/test_gm_adopt_stage4.py`, `research/growmoe_results/rv2d/`.
- **Status:** **VERIFIED for the router and the transfer rule (pre-registered confirmatory run on fresh seeds 85-99, 704 cells, all nine statements PASS, amendment A4-1 accepted before the run; the independent review E-042 reproduced every verdict, no blocker)**. Growth from the fit gap passed its statements too, but its default is **held for the user's decision** (E-042, D-38).
- **What it is:** The router: at answer time a small classifier, chosen by cross-validation on the experts' stored prototypes (density, LDA, softmax or 5-NN), picks the expert, so overlapping tasks no longer go to the wrong expert; it needs a fast route rule (`accept_then_logf`) and stays off with the reference path `route_rule="default"`. The transfer rule: when a new labelled cluster appears, the best existing expert's held-out loss on it is tested against that expert's own reference (exact permutation test with a margin): a conflict founds a new expert, no conflict extends the old one; it replaces the 60-step interference probe, which under-spawns. Growth from the fit gap (`growth_rule="gap"`) widens an expert when its fit gap is small and progress has stalled; it REPLACES the shipped shadow-trial growth, which can also deepen.
- **Evidence:** `research/growmoe_results/rv2d/RESULTS_RV2D.md`, `REVIEW_FOLLOWUP_CHECKS.md`; notebook E-041, E-042, E-043. Router: operator-by-operator arithmetic +13.0 [+10.5, +15.4] points, OV-0.85 +10.1 [+9.6, +10.6], nothing negative elsewhere. Transfer rule: arithmetic +6.1 [+4.3, +8.0] with forgetting 15.2 -> 7.4 and the true number of experts (6) where the shipped pool founds 3.7; S2-30 +2.9, banking77 +2.6, S2-5 +1.8; small costs where the shipped pool is already right (F-30 -0.43, negative in 13 of 15 seeds).
- **Limits:** the router costs about 2 s per refit at 12 experts, 90 % of it in the softmax candidate; the transfer rule's extend branch is not tested at scale (no confirmatory regime should be extended); every statement is at the training budget tuned for the shipped pool; a bare `PhyloCfg()` has the transfer rule but no router (it needs `route_rule="accept_then_logf"`). `from growmoe.config import PRE_STAGE4` reproduces the pool of the earlier stages; every older script and test is pinned to it (`research/growmoe_results/PRE_STAGE4_PINS.md`).
