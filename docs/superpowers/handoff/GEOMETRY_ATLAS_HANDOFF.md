# Track Z hand-over: verify the geometry components of the general-intelligence system (Gemini implements, Claude reviews, the user mediates)

**Start only when the user says "start track Z".** Best given to the agent that built Track Y (it knows `geometry/`) or a fresh agent; never interleave with Track G rounds. Read this whole file, then `research/GEOMETRY_ATLAS_2026-10.md` (why, evidence tags, references), `docs/superpowers/handoff/SCOPE_CHARTER.md` (scope, drift rules) and `docs/COMPONENT_CARDS.md` (what is already verified).

---------------------------------------------------------------------------------------------------------------------------
## 0. How we work (unchanged)
Gemini implements, test-first, **one round at a time**; Claude reviews the report, re-runs the code, replies `APPROVE`, `CHANGES: 1) ...` or `ANSWER:`; the user mediates and decides. Claude has a small token budget: reports must be compact and complete.
You must NOT: commit/push; edit anything outside the Track-Z fences (section 2); install packages; download data; use Python `hash()`; tune on test seeds; weaken an assertion silently; invent a numeric threshold without a unit and a measurement; **edit a results file by hand**; retype a table. Ambiguity (a paper's detail, a baseline's definition) goes under "Questions for the mediator": do not guess.
About the user: knowledgeable but not an expert, learning with us; explain simply, keep technical terms (define once), say kindly when an idea (including the atlas's) does not hold. Settle disagreements by measuring.

## 1. Goal
For each geometric idea in the atlas, decide with a controlled, honest experiment whether it earns a place in the **Geometry Core** (the verified component library for the general-intelligence system). Nothing is integrated into `ishmn/`, `growmoe/` or any model: Track Z only produces verified (or rejected) components, each with a component card.

## 2. Scope fences
* **Create and edit only:** `geomlab/` (new package), `tests/geomlab/` (**no `__init__.py`**), `research/geomlab_results/<round>/`, `docs/COMPONENT_CARDS.md` (append cards).
* **Read-only imports allowed** (to reuse or use as baselines, never edit): `geometry/`, `yggformer/`, `research/yggdrasil_poc/`, `ishmn/` (e.g. `ishmn/memory.py` as the legacy-memory baseline in Z2), `growmoe/`.
* Local data only: Python standard-library source and docs (`sysconfig.get_paths()["stdlib"]`), scikit-learn bundled sets, synthetic generators. No downloads.
* **Memory-lean (the user complained once):** models <= ~10M parameters, sequences <= 2048, no (N, M, D) pairwise tensors above 5e7 floats, **one GPU job at a time**, check `nvidia-smi` first, report peak memory per job. The machine is one RTX 3060 (12 GB) shared with the IDE.

## 3. Environment (Windows 11) and traps
Repo root `D:\spherical_hyperspace_memory`; Python 3.12, torch 2.6+cu124; run from the root; `pytest.ini` sets `pythonpath=.`. Console is cp1252: **ASCII-only output**. Filter the `pynvml` warning. Do not rely on backslash escapes in heredocs. Jobs > 2 minutes: background + log + poll. Always `torch.manual_seed` and use explicit generators; same seed -> same numbers. Geometry regression to run at the start of each round: `python -m pytest tests/test_geometry.py tests/test_geometry_hybrid.py tests/test_geometry_layers.py tests/test_geometry_consistency.py tests/test_gpu_kernel_bridge.py tests/test_lorentz_attention.py tests/test_snapped_state.py tests/yggformer -q` (currently 140 passed).

## 4. The experimental protocol (applies to every round; Z0 builds the tools)
1. **Pre-registration before any run:** write `research/geomlab_results/<round>/PREREG.md` containing arms, tasks, metrics, the prediction (direction, and the decision rule), the kill rule, the learning-rate grid, seeds. Claude checks that its file time precedes every results file.
2. **Per-arm tuning:** >= 4 learning-rate values (log-spaced), tuned on dev seeds >= 100; if the best value is at the grid edge, extend the grid and rerun; report the grid and the chosen value for every arm. Test seeds 0-9. Mean +- 95 % CI (t), paired tests (t and Wilcoxon), Cohen's d_z, Bonferroni across arms.
3. **A control that isolates the geometry** in every comparison (the hyperbolic lesson: the Euclidean arm must get the same extra parameters/temperature/gate).
4. **Geometry-health report** for every arm (effective rank, isotropy, intrinsic dimension, norm drift, operating point where relevant).
5. **Results pipeline:** a script writes `results.json` (per-seed metrics included); a second step renders the tables *from the JSON only*; `validate_report()` re-renders and diffs against the delivered text and fails on any difference. Never edit a results file; if the code changes, re-run. Claude re-runs and diffs.
6. **A pre-registered check may not be replaced or removed;** if it fails, report it as failed, fix the cause, and write the history to `AMENDMENT.md` (PREREG is read-only once results exist).
6b. **Prose may not exceed the tables**; failures are reported as failures; "not measurable" is not "fail"; a conclusion is never written before the data.
7. **Fidelity:** implement baselines from their papers' equations; list every deviation; a baseline that is clearly weaker than the paper's reported behaviour must be debugged before the comparison counts.

## 5. Rounds (one at a time; wait for APPROVE)

### Z0 - protocol library and geometry-health diagnostics (infrastructure)
Create `geomlab/protocol.py` (tools of section 4: `tune_lr` with the edge check, `run_seeds`, `paired_stats`, `bonferroni`, `write_results`, `render_tables`, `validate_report`, `check_prereg`) and `geomlab/diagnostics.py` (`effective_rank` (entropy of normalised singular values), `isotropy` (mean pairwise cosine and the uniformity loss log E exp(-2|x-y|^2) on the sphere), `intrinsic_dim_twonn`, `hubness_skew(k)`, `norm_stats`, and an adapter to `yggformer.report.operating_point`). Tests: effective rank ~ d for isotropic Gaussians and small for low-rank data; TwoNN near 3 for a 3-D manifold embedded in 20-D; the validator fails when one digit of a table is changed; `check_prereg` fails without a PREREG file or with a PREREG newer than the results; a dry run reproduces `research/yggdrasil_results/poc_b_table.txt` from stored JSON byte for byte. No scientific claims in Z0.

### Z1 - spherical stream (nGPT family) [atlas A1, A3]
Read nGPT (arXiv 2410.01131) and "Training nGPT" (2608.01284). Arms on a small decoder-only LM (<= 8M parameters, context 256, char-level on stdlib source, hold out whole files): `base` (RoPE + RMSNorm + AdamW), `base_qknorm`, `ngpt` (unit-norm embeddings/hidden/weight vectors along the embedding dimension, learned per-dimension step sizes on the attention and MLP residual updates, learned scaling of q/k/u/v/logits, weights renormalised after every step, no weight decay), `ngpt_stream_only` (normalised stream, free weights), `base_muon`, `ngpt_muon`. Optional second wave: the angular-update-control and log-LR-decay items of the 2026 recipe. Also run MQAR (small) as a recall check. Metrics: tokens to reach the `base` final validation loss (speedup), final loss, wall-clock/step, width of the LR range within 5 % of the best loss, geometry health, norm drift. Required tests: after each optimiser step every weight vector has unit norm (1e-5); stream vectors unit norm at every layer; the normalised model at init is deterministic; speedup computation tested on synthetic curves. **Decision rule:** adopt `SphericalStream` only if the net wall-clock speedup is >= 1.15x with a paired CI above 1. Design note to carry forward: a normalised stream has no norm channel, so also test a model with an explicit scalar salience channel (atlas P-1).

### Z2 - memory geometry: recall per byte [atlas C2, B3]
MQAR generator (keys, values, queries; vary pairs, vocabulary, d_model, and the number of state bytes). Same tiny model around each mixer: embedding + 2 x [mixer + MLP]. Mixers (sequential naive PyTorch is fine for T <= 512): `softmax` (reference ceiling), `linear` (elu+1 features), `gated_delta` (S_t = S_{t-1}(alpha_t (I - beta_t k k^T)) + beta_t v k^T), `delta_l2` (same with L2-normalised keys), `kata_psd` (read arXiv 2607.17419: rank-one PSD / self-dual-cone features), `triadic` (read 2609.36529: tensor-product state, capacity d to d^2), PoST log-spaced decay vs random decay on the gated variants (2604.07658), **`legacy_vmf` = the ISHMN memory imported read-only from `ishmn/memory.py` behind a thin adapter**. Metrics: accuracy vs number of stored pairs at equal state bytes; extrapolation from training length 128 to 2048; speed. Tests: with orthogonal keys and pairs <= d every delta variant recalls 100 %; softmax recalls 100 % on the smallest setting; the adapter does not modify `ishmn/`. **Decision rule:** pick the memory with the best accuracy at equal bytes; report honestly where the legacy memory stands. Do not tune the legacy memory more than the others.

### Z3 - transition geometry: exact state tracking, in parallel [atlas C1]
Tasks (group word problems, prefix product after every symbol, train lengths <= 16-24, test up to 16x): S3, S4 (24 elements), A5 (60), S5 (120, not in SO(3)), Dyck-2 (nested depth 4-8, predict validity/closing symbol), parity. Arms: `gru`, `vq_rot` (`geometry.snapped_state.SnappedRotationState`), `deltaproduct` (n_h in {1, 2, 4}), `ckda` (delta rule with gate in [-1, 1] and beta in [0, 2], arXiv 2609.24797), `mamba3_rot` (data-dependent rotary/complex state, 2603.15569 and 2609.28273), `path_like` (accumulated data-dependent Householder products, 2505.16381), and the synthesis **`scor` (N1)**: inside chunks of C tokens compute the orthogonal transitions in parallel (scan), at chunk boundaries snap the state to a learned codebook of size >= 4x the number of reachable states. Metrics: exactness vs length, seeds >= 95 % at the longest length, parallelisability (yes/no, measured throughput at T = 512), parameters. Tests: group tables (identity, associativity, inverses; S3/S4/A5 realised as 3-D rotation matrices); unsnapped rotations drift beyond the training horizon (reproduces the earlier finding); snapping makes a noise-perturbed state exact again; `scor` equals sequential application when C = 1. **Decision rule:** if `scor` is not better than plain `ckda`/`deltaproduct` at 16x length, keep `vq_rot` as the sequential exact component only. Include the GRU row on S5 (it was never measured).

### Z4 - position geometry [atlas B1, B2]
Arms: RoPE, NoPE, PoPE (arXiv 2509.10534), LieRE-1D (2406.10322), FoPE (2412.17739), learned torus phases, and a PaTH-like arm for T <= 512 (2505.16381). Same small LM, train context 256, evaluate without fine-tuning at 256/512/1024/2048, plus a synthetic retrieval-at-offset task. Metrics: perplexity ratio at 4x and 8x length; retrieval accuracy vs offset. Tests: for fixed content the logit depends only on the offset for RoPE/PoPE/LieRE; PoPE magnitude is non-negative. **Decision rule:** adopt only if better than RoPE with a paired CI excluding zero at 4x length.

### Z5 - hierarchy: is hyperbolic needed? [atlas K1, R3]
Build a **real hierarchy** from the Python standard library with `ast` (package > module > class > function/method; deterministic; 10^4-2x10^4 nodes; skip `test` packages). Arms: hyperbolic (Poincare and Lorentz, per-arm LR grid), **spectral/orthogonal Euclidean** (eigenvectors of the ancestry/co-occurrence matrix, truncated; also Laplacian eigenmaps), tuned gradient Euclidean, product H x T. d in {4, 8, 16, 32, 64}. Metrics: MAP (parent/children), distortion on sampled pairs, hold-out-edge link prediction (10 % of parent-child edges), memory-lean chunked evaluation. Tests: tree builder correctness (ancestors, depths), metrics on a hand-made tree. **Decision rule:** if spectral Euclidean ties hyperbolic from d = 8 up, retire hyperbolic from the roadmap and keep only the card.

### Z6 - step control for latent loops [atlas P1] (design memo first)
Write a <= 1-page design memo and wait for APPROVE before running. Setting: a small looped (shared-weight) model on compositional tasks (S3 chains, modular-arithmetic chains) with K loops at training and 2x/4x loops at test. Arms: full step, learned scalar step, fixed quarter step (arXiv 2609.16665), **geodesic step on the unit sphere with a learned maximum angle (N3)**, each with and without halting. Metrics: accuracy at 2x/4x loop depth, share of full-step failures recovered. Tests: the geodesic step stays on the sphere (norm 1 within 1e-6) and its angle never exceeds the cap.

### Z7 - asymmetric distances for planning [atlas P2] (design memo first)
Design memo, wait for APPROVE. Setting: grid mazes with and without one-way edges (generator in `geomlab/`), offline random-walk trajectories, train on small mazes, test on larger unseen ones. Arms: symmetric distance head, IQE, MRN, a TMD-lite objective (arXiv 2509.20478). Metrics: shortest-path distance error, greedy planning success. Tests: d(x, x) = 0, the triangle inequality holds on random triples, asymmetry is present on directed mazes.

(Z8 is Track G's hypothesis list; Z9 is an optional notebook; neither starts without Claude's go-ahead.)

## 6. Report template
```
ROUND: Z<n> <title>                  STATUS: done | blocked
FILES: created/modified              SUITE: last line of the new tests + geometry regression last line
PREREG: path and file time (must precede results)
RED EVIDENCE: failing-test lines from the first run (<= 10 lines)
PLAIN SUMMARY (for the user): 3-5 lines, simple words, technical terms kept and defined once; measured vs assumed
WHAT I BUILT: <= 10 lines
RESULTS: the rendered tables pasted VERBATIM from the script output + LR grids and chosen values + peak memory per job
GEOMETRY HEALTH: per arm
DECISION: the rule from the PREREG, applied, with the result (ADOPT / KEEP AS IS / REJECT)
DEVIATIONS FROM THIS HANDOFF / FROM THE PAPERS: each with justification (or "none")
FAILURES / SURPRISES: (or "none")
QUESTIONS FOR THE MEDIATOR: numbered, options + recommendation (or "none")
DRAFT COMPONENT CARD: what, evidence file, limits
```
Claude's checklist: PREREG precedes results; per-arm grid with no edge optimum; the isolating control present; tables identical to a clean regeneration (Claude re-runs); prose does not exceed tables; baseline fidelity; geometry-health present; scope fences respected; memory caps respected; suite green.

## 7. Writing tests here
Assert preconditions before properties; test invariants (norms, relative-offset dependence, group axioms, triangle inequality) not implementations; a benchmark where every arm is at ceiling or at chance is a failed design (raise or lower the difficulty and say so); red first, then green, then the full suite; deterministic and fast (< 1 minute per new test file where possible).

## 8. Known risks (report if seen)
Gains at 30B not appearing at 1-10M (report as "not reproduced at small scale", not as a refutation); learning-rate sensitivity dominating (nGPT especially); baselines implemented too weakly; learned curvature/scale parameters collapsing; seeds where a recurrent arm fails to converge (report counts, not only means); a benchmark that a shortcut solves.
