# Yggdrasil architecture hand-over (track Y): Gemini implements, Claude reviews, the user mediates

This is a SECOND track, independent of the GrowMoE rounds. **Do not start it until the user says "start track Y".** Do not interleave rounds of the two tracks.
Read this whole file, then `research/YGGDRASIL_ARCHITECTURE_RESEARCH.md` (the why, the measurements, the evidence tags). Same working rules as `GROWMOE_HANDOFF.md` sections 0 and 8; they are repeated
where they matter. Everything below is written for an implementer with no memory of earlier sessions.

---------------------------------------------------------------------------------------------------------------------------
## 0. How we work (unchanged)

* **Gemini** implements, test-first, **one round at a time**, runs everything and reports facts. **Claude** reads the report and targeted files, re-runs the suite, replies `APPROVE`, `CHANGES: 1) ... 2) ...`
  or `ANSWER:`. Claude has a small token budget: reports must be compact and complete. **The user** mediates and decides scope and commits.
* Loop: the user gives you a round -> you implement only that round -> full suite -> fill the report template (section 8) -> STOP -> wait for the verdict.
* You must NOT: commit or push (unless the user says so); modify `ishmn/`; touch `growmoe/` or its tests; install packages; download data; use Python `hash()`; tune on test seeds; weaken or delete an assertion to
  pass a test without saying so; invent a numeric threshold without writing its unit and how it was measured. Ambiguity -> "Questions for the mediator", never a guess.
* About the user: knowledgeable but not an expert, learning alongside the models, and their own ideas can be wrong. Explain simply WITHOUT losing the technical meaning (define a term once, do not drop it) and say kindly,
  with a measurement, when an idea (including Gemini's earlier geometry claims) does not hold. **Settle disagreements by measuring, not by arguing.**

---------------------------------------------------------------------------------------------------------------------------
## 1. Goal (the user's words, condensed)

A new neuron, attention and transformer-like architecture compatible with the Yggdrasil geometry (hyperbolic hierarchy + torus phase + radial salience) that overcomes real shortcomings of mainstream
architectures, plus the GPU kernel work it needs. Claude's research found the geometry is strong for **hierarchy at small dimension**, is **not** a fix for order-dependent state, and is **numerically
demanding**. The plan is to test the cheapest design first ("Ygg-heads": geometry inside the attention heads of an ordinary transformer) against a parameter-matched Euclidean transformer, with kill criteria.

---------------------------------------------------------------------------------------------------------------------------
## 2. Non-negotiable constraints

* New code lives in `yggformer/` (new standalone package, create it) and `tests/yggformer/` (**no `tests/yggformer/__init__.py`**, same reason as growmoe). Results in `research/yggformer_results/`.
  Existing `geometry/` may be *extended* only where a round says so; never break its 68 existing geometry tests (see section 3).
* Local data only (scikit-learn bundled sets, Python stdlib text/code, synthetic). No downloads.
* **Numeric grading only. No magic thresholds.** Every threshold is an error rate or a tolerance with units. Anything tuned is tuned on **dev seeds >= 100 only**, then frozen; test seeds are 0-9. Claims need
  >= 10 seeds, mean +- 95 % CI (t-interval), paired tests, **parameter-matched baselines**.
* **Tune every geometry separately.** The research's first tree experiment used one learning rate for all geometries and printed a wrong conclusion ("hyperbolic stops winning at high d"); a per-geometry learning-rate
  grid reversed it (`research/yggdrasil_poc/poc_a3_optimisation_sensitivity.py`, `poc_a4_tree_embedding_tuned.py`). Any comparison without a per-arm tuning grid (and the grid written in the report) will be rejected.
* **Operating-point report is mandatory in every YggFormer result** (section 5). A "win" for the hyperbolic heads counts only if they actually operate in the hyperbolic regime and the effect is stable across seeds.
* A prediction that does not hold is a result, not a bug to hide. Pre-registered predictions are in research doc section 7 (E1-E11); write the prediction in the report *before* the numbers.
* **Memory-lean.** The user already complained about memory usage. Rules: never allocate an (N, M, D) pairwise tensor with N*M*D > 5e7 floats (use `geometry.attention.yggdrasil_attention`, never the pair-by-pair
  `reference_attention`, above T = 256); keep each job under 4 GB VRAM; **one GPU job at a time** and check `nvidia-smi` first (the machine has one RTX 3060, 12 GB, also used by the IDE); `del` big tensors and
  `torch.cuda.empty_cache()` between runs; CPU jobs <= 4 threads and <= 2 in parallel; report the peak memory of every job. No 8192^2 reference benchmarks.

---------------------------------------------------------------------------------------------------------------------------
## 3. Environment (Windows 11) and traps

* Repo root `D:\spherical_hyperspace_memory`. Python 3.12, **torch 2.6.0+cu124, RTX 3060 (compute 8.6) available**, numpy 2.x, scipy, scikit-learn, pytest 9.
* Geometry regression: `python -m pytest tests/test_geometry.py tests/test_geometry_hybrid.py tests/test_geometry_layers.py tests/test_geometry_consistency.py tests/test_gpu_kernel_bridge.py tests/test_lorentz_attention.py -q`
  -> **68 passed (~20 s)**. Run it at the start of every round and paste the last line.
* **Console encoding is cp1252: scripts must print ASCII only** (no Greek letters, arrows). Filter the torch `pynvml` FutureWarning (`warnings.filterwarnings("ignore")` before importing torch).
* Do not rely on backslash escapes inside shell heredocs; use the editor/patch tool or write a `.py` file. Jobs longer than ~2 minutes: background + log file + poll. Scripts outside pytest need `PYTHONPATH=.`.
* GPU engine: always construct `YggdrasilKernel(0, require_gpu=True)` in tests; the silent CPU fallback once hid a broken PTX (CUDA error 218, missing NUL terminator).
* `torch.compile`/Triton are not reliable on native Windows here: do not depend on them.

---------------------------------------------------------------------------------------------------------------------------
## 4. What already exists (read the code and the tests; they are the specification)

**`geometry/lorentz.py` `LorentzModel(dim, c=1.0)`**: points are charts `x_s in R^D` (time coordinate derived `x0 = sqrt(1/c + |x_s|^2)`; no boundary). Methods: `time, lift, project, origin, inner, inner_chart,
sq_distance, distance, exp_map, log_map, exp_map_zero, log_map_zero, centroid, tangent_mean, from_poincare, to_poincare`.

**`geometry/attention.py`**: `yggdrasil_attention(q_s, k_s, v_s, *, c, tau, theta_q, theta_k, kappa, key_bias, is_causal, attn_mask, aggregate='lorentz'|'tangent'|'euclid', compute_dtype=None) -> (B,H,T,Dv)`
(one SDPA call on augmented vectors; shapes `(B,H,T,*)`; `c, tau, kappa` floats or tensors broadcastable to `(B,H,1,1)`; default compute dtype float32), `reference_attention` (pair-by-pair oracle for tests only),
`augment`, `safe_radius(dtype)` (bf16 2, fp16 6, fp32 50), `clip_radius(x_s, r_max)`. The logit is `(1/tau)<q,k>_L + kappa*sum cos(theta_q - theta_k) + key_bias`.

**`geometry/neuron.py` `NormGatedLinear(in, out, curvature=1.0)`**: `a_j = <w_j,x> + b_j - c_j*sqrt(1/c+|x|^2)`; `causal_character()` -> sign tells prototype (<0) / horosphere (0) / half-space (>0) neuron.

**Measured facts that override assumptions** (details and tags in the research doc):

| Fact | Consequence for you |
|---|---|
| Hyperbolic >> Euclid at small d on a 1093-node tree (d=4: MAP 1.00 vs 0.36 at 2000 steps); equal at d=32; product H x T is **no better** than pure H at equal total d (same distortion, lower retrieval MAP; with only 2 hyperbolic dims it fails) | keep the hyperbolic block >= 8 dims and the torus block small relative to it (suggest M <= D/4; a design choice, not a measurement) |
| Hyperbolic QK in half precision loses accuracy like R^2 (bf16 unsafe beyond radius ~2, fp16/TF32 ~8) | QK in fp32 by default; clip radius; **never QK-norm** the hyperbolic q/k |
| Fused augmented SDPA = 1.0-1.4x plain core, O(T) memory | no custom attention kernel |
| Torus (abelian) cannot do S3; unsnapped torus drifts beyond the training horizon even on Z6 (100 % -> 9 %); lattice-snapped torus does not drift (flat ~94 % to length 256) | state tracking is not a geometry job; see Y4 |
| GRU: exact to length 256 on S3 and Z6; Householder/SO(k): perfect at 16, 44-60 % at 256 | baseline for Y4 |
| Existing pairwise kernel 2.0 ms vs GEMM floor 0.34 ms at 4096^2 | Y7 |

---------------------------------------------------------------------------------------------------------------------------
## 5. The operating-point report (`yggformer/report.py`, round Y1)

`operating_point(model, batch, qk_layers=None) -> dict`, JSON-serialisable, ASCII keys. Per layer and head: learned curvature `c`; `u = sqrt(c) * asinh(sqrt(c) * |x_s|)` quantiles (median, p95, max) for q and k;
`frac_heads_euclidean = mean(median_u < 0.37)`; top-1 attention mass (mean over queries); attention entropy; `var_share` = share of the variance of the logit contributed by the hyperbolic / phase / bias terms;
`radius_order_corr` = Spearman correlation of the per-token key radius between two seeds (computed by the experiment runner, not by the function). Rule of thumb from the literature audit (arXiv 2607.05268): hyperbolic
effects are negligible below u ~ 0.37 and meaningful above ~ 0.84. State the thresholds as "starting values from the literature" in reports; do not tune them on test seeds.

---------------------------------------------------------------------------------------------------------------------------
## 6. Rounds

### Y0 - environment check (no code)
Run the geometry regression (68 passed), `python research/yggdrasil_poc/poc_c_attention_and_neuron.py` (peak < 1.5 GB) and compare its printed identities/precision table with `research/yggdrasil_results/poc_c_output.txt`;
run `python research/yggdrasil_poc/poc_b_state_tracking.py report`. Report any difference > 20 % from the stored numbers. Takes minutes.

### Y1 - `yggformer/` skeleton + operating-point report
Files: `yggformer/config.py` (`YggCfg` dataclass: `vocab, d_model, n_layers, n_heads, head_dim, ffn_mult, max_len, head_geometry in {"euclid","hyp","hyp_phase","hyp_phase_bias"}, n_phase (default head_dim//4),
c_init=1.0, learn_c=True, tau_init=1.0, qk_dtype="float32", r_max=None (None -> safe_radius(qk_dtype)), aggregate="lorentz", null_key=False, seed`), `yggformer/attention.py` (`YggMultiHeadAttention`: q/k/v/o
projections, optional phase and bias projections, per-head `c = softplus(gamma)+1e-3` and `tau = softplus(...)`, a learnable per-head scale on q_s/k_s **initialised so the median u at step 0 is ~0.3-0.5**
(measured: default random projections start at median u ~ 1.9, see `research/yggdrasil_poc/poc_d_training_smoke.py`, a working minimal reference for this module), radius clip before the product, calls `yggdrasil_attention`; `head_geometry="euclid"` calls plain
`F.scaled_dot_product_attention` and must have the **same parameter count** as the others via a documented rule), `yggformer/block.py` (pre-norm block, RMSNorm on sublayer input only, residual stream untouched),
`yggformer/model.py` (`YggFormer(cfg)`: embedding, blocks, tied output head; `forward(tokens) -> logits`), `yggformer/report.py` (section 5), `yggformer/params.py` (`match_params(cfg_a, cfg_b)` -> adjust `ffn_mult` so
`|N_a - N_b| / N_a <= 1 %`). Tests (`tests/yggformer/test_yf_*.py`): euclid mode equals a hand-written SDPA reference to 1e-6; with `c=1e-6, aggregate="euclid"` equals Gaussian (L2) attention to 2e-4;
causality property test (perturb token t+1, logits at <= t unchanged, exactly); `operating_point` keys finite and present; 50 training steps in bf16 autocast with fp32 QK give no NaN/Inf; `match_params` within 1 %;
the model is deterministic for a fixed seed. Peak memory: T = 128, d_model = 128, 4 layers must run on CPU in < 1 GB.

### Y2 - hierarchy benchmark design memo, then generators (**two steps; wait for APPROVE between them**)
Step 1 (memo, <= 1 page, no sweep yet): propose the benchmark that tests whether *hyperbolic heads* help when the head dimension is small. Claude's two candidate tasks (you may propose better):
**T-A in-context hierarchical few-shot**: a latent tree (depth 3-7, branching 2-4) with leaf vectors generated by a branching random walk in D_data = 64 dimensions; the context lists labelled leaves (label = ancestor cluster at
depth k), the query is a new leaf from the same tree, the answer is its label; attention must compare in `head_dim` in {8, 16, 32, 64}. **T-B transitive closure**: is-a facts "x is-a y" in context, query "is x a descendant of z"
at hop distance 1-5. For each task state: what the Euclidean and hyperbolic heads each need to do, the pre-registered prediction (research doc E3: Ygg >= Euclid + 10 pp at head_dim <= 16 and depth >= 5, no difference at
head_dim >= 64), the metric, the chance level, and why the task cannot be solved by a trivial shortcut (assert it with a bag-of-features baseline that must stay near chance).
Step 2 (after APPROVE): `yggformer/tasks/hierarchy.py` generators, deterministic from a seed, with tests (precondition assertions: tree depth and branching are what was asked; labels recoverable from the generator;
shortcut baselines near chance; no train/test leakage of trees).

### Y3 - E3 sweep and E4 numerics
Arms: `euclid`, `hyp`, `hyp_phase`, `hyp_phase_bias`, each parameter-matched, per-arm learning-rate grid on dev seeds (write the grid), head_dim in {8,16,32,64}, tree depth in {3,5,7}, 10 test seeds, >= 2 layers.
Report accuracy (mean +- CI), paired differences vs `euclid`, the operating-point report per arm, wall-clock, peak memory. E4: repeat the best `hyp` configuration with QK in fp32 / tf32 / fp16 / bf16, with and without
`clip_radius`; report NaN rate, radius quantiles and accuracy. Apply the kill rule honestly: no gain at head_dim <= 16 -> say so plainly in the report; Claude and the user decide.

### Y4 - E2b: snapped non-abelian state (S3 and S5 word problems)
Reuse `research/yggdrasil_poc/poc_b_state_tracking.py` (models `torus, so3, gru, torus_lat, so8, hh2`; results `research/yggdrasil_results/poc_b_table.txt`). Add S5 (`itertools.permutations(range(5))`, order 120) as a third task and
two new arms: (1) **VQ-snapped rotation state**: the state vector after every token is replaced by its nearest entry of a learned codebook (`K = 4*|G|`, straight-through gradient, commitment loss, dead-code revival);
(2) **finite-group lookup state**: soft permutation/Sinkhorn representation of the group elements with an exact multiplication-table update. Train on lengths <= 16 (S3) / <= 24 (S5), test to 256; learning rate per arm on dev seeds
over **all** tasks; 10 seeds; report seeds >= 95 % at length 256. Pre-registered: >= 95 % at 256 on S3 for at least one snapped arm; if all < 80 %, say that state tracking must rely on gated recurrences.

### Y5 - E5: sparse retrieval from a trained model, then a tree-routed prototype
Part 1 (no new training): in the Y3 models and an `euclid` control, measure the attention mass captured by the top-B keys, B in {4,8,16,32}, at T in {1k, 4k, 16k} (chunked; keep memory < 4 GB), for random-context batches.
Part 2 (only if part 1 supports it): a CPU-first branch-and-bound top-B retrieval over a tree of keys: nodes store the Lorentz centroid and the maximal geodesic radius; a node's logit upper bound uses the triangle inequality on
the geodesic distance then the monotone map `-(cosh(sqrt(c) d) - 1)/(c tau)`; compare recall, captured mass, node visits and wall-clock against dense attention and against an equivalent Euclidean ball tree.

### Y6 - E8/E9: small language model and attention sinks
Char-level LM on local text (Python stdlib source and docs; hold out whole files), <= 10 M parameters, arms `euclid`, `hyp`, `hyp_phase`, `hyp_phase_bias`, `hyp_phase_bias+null_key`, plus a Euclidean head with the
head-wise sigmoid output gate of Qiu et al. (arXiv 2505.06708) as the sink-fix control. Per-arm learning-rate grid on dev seeds; 10 seeds; perplexity +- CI, paired tests; first-token attention mass and max activation norm (E9); operating-point report.
Expectation (research doc): within +-2 % of Euclid; anything above is a pleasant surprise, anything below is a result.

### Y7 - kernels (`rust_yggdrasil/`, `geometry/gpu_kernel.py`)
(a) **GEMM + epilogue pairwise distance**: Poincare/Lorentz part from norms and one matmul then elementwise acosh; torus part via cos/sin dot products (smooth surrogate) or the existing wrapped formula in the epilogue; radial part
elementwise. First as a PyTorch reference (`torch.matmul`), then a CUDA kernel built with `rust_yggdrasil/build_ptx.py` if the PyTorch version leaves > 2x on the table. Acceptance: agrees with the current engine metric to 2e-4,
>= 4x faster at 4096^2 on the RTX 3060 (current: 2.0 ms; GEMM floor 0.34 ms), peak memory O(N*M). (b) one fused op for the attention preparation (concatenation, cos/sin, time coordinate, centroid renormalisation) if it
removes > 0.3 ms of eager overhead (measured: ~0.4 ms prep + ~0.13 ms renorm). (c) stretch, only with Claude's go-ahead: flash-style kernel with split-precision QK on the manifold columns.
Always `require_gpu=True` in tests; add multi-arch build options to `build_ptx.py`; print and assert the NUL terminator is kept.

### Y8 - tree-routed experts and GrowMoE integration (**memo only**)
A <= 2 page design memo: how the hyperbolic position of a token becomes its tree address (direction at radius r selects the child, radius = depth), how GrowMoE's novelty/growth events add a subtree, what the router costs,
what is measured vs assumed, and the experiments (E10). No code until Claude approves the memo and the GrowMoE track is complete.

---------------------------------------------------------------------------------------------------------------------------
## 7. Writing tests here (each rule is a mistake already made once)

1. Assert the scenario's precondition (tree depth, group order, radius range) before the property under test.
2. Statistical tests pool seeds; measure the margin first and pin the threshold with a stated reason (`# measured 0.12-0.19 over 5 seeds`).
3. Test the property, not the implementation: exact identities (`yggdrasil_attention` == `reference_attention` to 1e-10 in float64), invariants (causality, flat limit), hand-computable cases.
4. A test that cannot fail is worthless; a benchmark in which every arm scores 100 % is a failed design - raise the difficulty and record that you did.
5. Red first: show the failing output for the stated reason, then green, then the full suite. Deterministic, fast, cached builders; keep new test files under ~1 minute each.
6. A GPU test must prove it ran on the GPU (`require_gpu=True`, or assert `tensor.is_cuda`).

---------------------------------------------------------------------------------------------------------------------------
## 8. Report template (Gemini -> Claude, via the mediator)

```
ROUND: Y<n> <title>                       STATUS: done | blocked
FILES: created/modified (paths)           SUITE: "<N> passed in <T>s" (last line of the new tests) + geometry regression last line
RED EVIDENCE: failing-test summary lines from the first run of the new tests (<= 10 lines)
PLAIN SUMMARY (for the user): 3-5 lines, simple words, technical terms kept and defined once; say what each number means and how sure we are (measured vs assumed)
PREDICTION (written BEFORE the numbers): ...
WHAT I BUILT: <= 10 lines, only what is not obvious from the code
MEASUREMENTS: tables (numbers, seeds, CI, paired tests) + the learning-rate grid per arm + peak memory per job
OPERATING POINT: the section-5 report for each YggFormer arm (or "n/a")
DEVIATIONS FROM THIS HANDOFF: each with the measurement that justifies it (or "none")
FAILURES / SURPRISES: things that did not hold (or "none")
QUESTIONS FOR THE MEDIATOR: numbered, each with options and my recommendation (or "none")
```
Claude's checklist: tests read before code; red evidence plausible; no vacuous or seed-lucky assertions; per-arm tuning grid present; parameter matching verified; operating point reported; thresholds backed by measured margins;
no `ishmn/`/`growmoe/` change; memory caps respected; suite re-run by Claude; prediction written before numbers.

---------------------------------------------------------------------------------------------------------------------------
## 9. Known risks to watch (report if you see them)

* Hyperbolic heads collapsing to near-Euclidean (curvature drifting to ~0): watch `frac_heads_euclidean`; consider a floor on `c` as an ablation, not as the default.
* Radius blow-up and NaNs in the Lorentz centroid at large radius: try `aggregate="tangent"`; report which arm needed it.
* Benchmarks where a shortcut (identity matching, bag of features) solves the task regardless of geometry.
* Learning-rate sensitivity dominating the comparison (the first tree experiment): always grid per arm, report the grid, and check the chosen value is not on the grid's edge.
* Parameter matching hidden by embedding/FFN differences: report counts per component.
