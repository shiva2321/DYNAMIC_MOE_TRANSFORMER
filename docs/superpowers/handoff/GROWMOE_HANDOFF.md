# GrowMoE hand-over: Gemini implements, Claude reviews, the user mediates

Read this whole file before touching code. It is written for an implementer with no memory of the earlier sessions.
Companion files: `docs/superpowers/specs/2026-10-04-growmoe-design.md` (the approved design), `docs/superpowers/plans/2026-10-04-growmoe.md`
(the 22-task plan), `research/DYNAMIC_MOE_RESEARCH_AND_DESIGN.md` (literature + rationale), `research/spikes/RESULTS.md` (**measured
decisions that override the spec**), `research/growmoe_results/IMPLEMENTATION_LOG.md` (every finding so far, with numbers).

---------------------------------------------------------------------------------------------------------------------------
## 0. How we work

| Who | Role |
|---|---|
| **Gemini** (you, if you are reading this as the implementer) | Implements, test-first, one round at a time. Runs everything. Reports facts. |
| **Claude** | Designer/reviewer. Reads your report and targeted files, runs the suite, replies `APPROVE`, `CHANGES` (itemised) or `ANSWER`. Does not implement. Has a small token budget, so **reports must be compact and complete**. |
| **The user** | Mediator and decision-maker. Copies messages between you and Claude. Final say on scope, commits, and disputes. |

**The loop.** (1) The user gives you a *round* from section 7. (2) You implement only that round, test-first (write tests, run them,
paste the failing output, implement, run again). (3) You run the **full suite** and fill the report template in section 9. (4) You STOP and
wait. (5) The user relays Claude's verdict. Do not start the next round before `APPROVE`.

**You must NOT:** commit or push (unless the user says so); modify anything under `ishmn/`; install packages; download data; add
`tests/growmoe/__init__.py`; use Python `hash()`; read ground truth (`gt`) inside any pool; tune anything on test seeds; weaken, delete or
loosen an assertion to make a test pass without saying so in the report; invent a numeric threshold without writing down its unit and how it was
measured. **When something in this file is ambiguous or seems wrong, do not guess: put it under "Questions for the mediator" in your report.**

**About the user.** Knowledgeable but not an expert, learning alongside the models, and has said their own ideas can be wrong. Explain simply WITHOUT losing the
technical meaning (define terms rather than dropping them), and if one of their ideas looks wrong or untested, say so kindly with the reason and a measurement.

**Settling disagreements** (Claude vs you): do not argue, measure. Write a small script that settles it and put the numbers in the report.

---------------------------------------------------------------------------------------------------------------------------
## 1. Goal (the user's words, condensed)

A dynamic Mixture-of-Experts that (1) **recognises data novelty automatically and is not bound to a data type**, (2) **knows whether it has
already seen something**, (3) **spawns a new expert from a parent's "DNA"** or reuses/trains an existing expert, and (4) can **scale experts
vertically (layers/width) and horizontally (more experts/columns/system)** as needed - a working, useful, *testable* algorithm. Three
architectures are built and compared (the user asked to "see the difference"): **A PhyloMoE** (calibrated conformal gate + affinity-scaled
inheritance), **B BNP-MoE** (CN-DPM-style likelihood gate, scratch experts), **C Keyed Delta-MoE** (shared base + low-rank deltas, key gate).
Scope: Phases 0-3 (single pool -> vertical growth -> columns/system). **Phase 4 (integrating with the user's `ishmn/` code) must NOT be
started without the user's explicit go-ahead.**

---------------------------------------------------------------------------------------------------------------------------
## 2. Non-negotiable constraints

* Nothing under `ishmn/` is modified. New code: `growmoe/`, tests: `tests/growmoe/`, results: `research/growmoe_results/`.
* Local data only (scikit-learn bundled sets, Python stdlib text/code, synthetic). No downloads.
* **Numeric/exact grading only. No fixed "magic" thresholds**: every threshold is an error rate (q, alpha) or a tolerance with units
  (nats, accuracy points, samples, calls). A number tuned on data is tuned on **development streams only** (seeds >= 100), then frozen;
  test seeds are 0-9. Claims need >= 10 seeds, mean +- 95 % CI (`metrics.mean_ci`), paired tests (`metrics.paired_diff`).
* Algorithms never see ground truth (regime ids, first-visit flags, outlier flags, clean labels). `Stream.batches` yields `(x, y, gt)` for the
  **benchmark runner only**; pools get `Stream.algo_batches` -> `(x, y, ctx)`.
* Deterministic: all randomness from explicit seeded `numpy.random.Generator` / `torch.Generator`; same seed -> identical event log.
* Report failures as failures. A prediction that does not hold is a result, not a bug to hide. Pre-registered predictions: spec section 8
  (C1-C6) and research doc section 7.5 (P1-P10).
* Test-first. Each task: red (fails for the right reason) -> green -> full suite green.

---------------------------------------------------------------------------------------------------------------------------
## 3. Environment (Windows 11) and traps that already cost time

* Repo root `D:\spherical_hyperspace_memory`. Python 3.12.10, torch 2.6 (CPU), numpy 2.x, scipy, scikit-learn 1.5.2, pytest 9.
* Run from the repo root: `python -m pytest tests/growmoe -q` (currently **240 passed, ~80 s**). `pytest.ini` sets `pythonpath=.`, `testpaths=tests`.
  Scripts run outside pytest need `PYTHONPATH=.` (or `sys.path.insert(0, ".")` as the spike scripts do).
* **Test files are `tests/growmoe/test_gm_<module>.py`; there is NO `tests/growmoe/__init__.py`** (it would shadow the real `growmoe` package).
* **The console encoding is cp1252: scripts must print ASCII only** (no Greek letters, arrows, emoji) or they crash at the end of a long run.
* A torch `pynvml` FutureWarning is filtered in `tests/growmoe/conftest.py`; scripts should `warnings.filterwarnings("ignore")` before importing torch.
* When patching files from the shell, do not rely on backslash escapes inside heredocs (`\n` became a real newline and broke a script).
  Use the editor/patch tool, or write a `.py` patch file.
* Jobs longer than ~2 minutes: run in the background with output to a log file and poll the log. Spike-style scripts take `--seeds`.
* Building streams and training pools is the slow part of tests: cache builders with `functools.lru_cache` and use small `n_train`/`d`.
* Constructing a module must not consume the **global** torch RNG (`MLPExpert` builds `nn.Linear` under `torch.random.fork_rng(devices=[])`; do the same).

---------------------------------------------------------------------------------------------------------------------------
## 4. What already exists (all green, 240 tests)

`growmoe/` (read the docstrings; signatures below are exact):

* `config.py`: `CommonCfg(d=128, hidden=128, depth=1, n_classes=64, lr=3e-3, batch_size=64, inner_steps=1, replay_size=256, replay_frac=0.5, seed=0)`,
  `GrowthCfg(widen_factors=(1.5,4.0,16.0), min_widen=4, train_steps=480, batch_size=64, lr=3e-3, min_gain=0.02, alpha=0.05, n_boot=1000, backoff0=1,
  max_backoff=64, max_hidden=1024, max_depth=8)`, `PhyloCfg(CommonCfg)(q=0.05, alpha_spawn=0.01, delta_forget=0.05)`. `BNPCfg`, `KeyedCfg` do not exist yet.
* `types.py`: `StepRecord(novel, p_nov, assigned, loss, spawned, n_experts)`, `Prediction(probs, expert, p_nov, unknown)`,
  `ContinualPool` Protocol (`name`, `events`, `observe(x, y=None, ctx=None)`, `predict(x, ctx=None)`, `summary()`).
* `telemetry.py`: `Event(kind, step, fields)`, `EventLog.add(kind, step, **fields)`, `.of_kind`, `.to_jsonl`, `.from_jsonl`, `len`, iteration.
* `frontend.py`: `HashFrontend(d=128, seed=0, ...)`; call it on numpy arrays / list of str|bytes -> unit vectors. Type-agnostic, training-free.
* `conformal.py`: `conformal_pvalues(cal, test)` = `(1 + #{cal <= test}) / (n + 1)` (LOW score = novel); `benjamini_hochberg(p, q) -> bool mask`.
* `vmf.py`: `log_vmf_norm(d, kappa)`, `mean_resultant_length`, `sample_vmf(mu, kappa, n, rng)`.
* `signature.py`: `VmfSignature(d, max_protos=256, kappa_grid=(2..256), seed=0)`: `update(x)` (reservoir), `fit()`, `score(x)` (= lse_j kappa<p_j,x> - log n),
  `logpdf(x)` (adds the exact vMF normaliser), `pvalue(x)` (conformal vs leave-one-out scores, small = novel), `nn_similarity()`, `mean_direction()`,
  `affinity(x)` in [0,1], `merge_from(other)`; properties `n`, `protos`, `kappa`, `cal_scores`, `seen`; refits lazily when the reservoir changes.
* `experts.py`: `MLPExpert(d_in, hidden, depth, n_classes, seed)` (residual blocks `h + W2 gelu(W1 LN(h))`, linear head, **seen-class mask**: unseen classes get
  logit -1e9): `forward`, `proba`, `mark_seen(y)`, `loss`, `num_params`, `clone`, `widen(k)`, `deepen()` (both exactly function-preserving),
  `inherit(rho, sigma)`, **`shrink_perturb(lam, seed)`**, `train_step(u, y, opt, replay=(x,y)|None, project=False)` (A-GEM when `project`), attributes
  `hidden`, `blocks`, `seen`, `head`; `agem_project(g, g_ref)`. **Growth replaces parameters: rebuild the optimiser after `widen`/`deepen`.**
* `competence.py`: `LossChangeMonitor(alpha=0.01, warmup=50, horizon=1000, delta=0.5, ...)`: `update(loss) -> alarm`, `reset()`, `armed`, `lam`, `statistic`.
  Scale-free one-sided CUSUM; feed it **per-sample prequential losses** (one per sample, test-then-train), arm it only after the expert's loss has plateaued.
* `growth.py`: `plateau(history, window=40, tol=0.01)`, `GrowthPolicy(cfg).step(expert, x_tr, y_tr, x_va, y_va, rng) -> GrowthDecision(kind, gain,
  params_added, p_value, reason, increment, tried)`; owns its back-off state, so keep ONE policy object per expert. Mutates `expert` only on commit.
* `streams.py`: `Stream(name, d, n_classes, regimes, segments, meta)` with `.batches(bs, seed)` (-> x, y, gt) and `.algo_batches(bs, seed)` (-> x, y, ctx),
  `.test_sets()`, `.test_ctx()`; builders `s1_types`, `s2_permuted`, `s3_drift`, `s4_outliers`, `s5_ladder`/`s5_grid`, `s6_long`, `s7_mapping`.
  `gt = {"regime", "segment", "first_visit", "is_outlier", "y_clean"}`. `Regime.parent` is ground-truth lineage (analysis only).
* `metrics.py`: `auroc`, `false_novelty_rate(novel, known_mask)`, `novel_detection_rate`, `regime_owner(assigned, regimes)`, `routing_purity`,
  `spawn_stats(events, owner_of, onsets)` (expects `spawn` events with a `fields["expert"]` id), `accuracy_stats(R, learned_at, chance)`, `mean_ci`, `paired_diff`.
* `baselines.py`: `FineTune`(B0), `JointOracle`(B1), `ReplayNet`(B2), `StaticRouterMoE`(B3), `OracleRouted(regime_of, cfg)`(B4; `set_eval_regime(r)` before `predict`),
  `DPMeansGate(cfg, lam)`(B5). `replay.py`: `Reservoir(capacity, d, rng)` with `add`, `sample`, `.data`.
* `pools/base.py`: `ExpertRecord(id, module, parent, birth_step, state, n_assigned, last_active, lr_mult, replay, opt, meta)` and `BasePool(cfg)`:
  `tick()`, `add_expert(module, parent=None, state="active", meta=None)`, `remove_expert`, `set_state`, `set_lr_mult`, `ancestors`, `set_parent` (cycle-safe),
  `train_expert(eid, x, y, replay=True, project=False)`, `summary()`; ids unique/monotone/never reused; lineage survives removal; one optimiser per expert.

`research/spikes/` holds the three Phase-0 scripts. Baseline sanity numbers (S2 prefix, 3 regimes, 640 train each, 5 seeds): B0 final accuracy 0.47-0.60
(regime 0: 0.18-0.39), B1 0.90-0.94, B2 0.88-0.91, B3 0.47-0.58, B4 0.87-0.89 (zero forgetting, exactly), B5(lam 0.8) 0.84-0.89 with 5-6 experts.

---------------------------------------------------------------------------------------------------------------------------
## 5. Decisions already made by measurement (full table: `research/spikes/RESULTS.md`)

1. Signature = `VmfSignature` (256 prototypes), calibrated per expert; novelty uses the conformal p-value, **routing among known experts uses the raw
   `logpdf`** (p-values of a known sample are uniform for every overlapping expert and say nothing about which one is closest).
2. **Child initialisation = `parent.module.shrink_perturb(lam=affinity)`** (affinity from the parent's signature vs the new cluster). `inherit(rho, sigma)` is an ablation.
3. Growth = multi-scale candidates + cheapest sufficient + status-quo gate (already implemented in `growth.py`; do not change it without measurements).
4. Novelty is hard for diffuse regimes and for "code after text" in S1; related-but-new regimes (cos 0.9) are flagged by KDE/kNN/Mahalanobis but missed by VAE/RND/key
   gates. A related regime is solved zero-shot by an inherited child, so the extend-vs-spawn probe must be able to choose **extend**.
5. S5 default is 20,000 training samples (2,000 is data-limited for every teacher). S2 regimes share a strong common direction (not orthogonal) but are separable.
6. Power warning: growth tests need validation data; pool experts have <= 256 replay samples. Measure and report (Task 14).

---------------------------------------------------------------------------------------------------------------------------
## 6. Conventions every pool follows

* `observe(x, y=None, ctx=None) -> StepRecord`; `x` is `(B, d)` unit vectors, `y` is `(B,)` long in the global label space or `None` (unlabeled: regime
  discovery only, **no training call may happen**). `ctx` is optional and ignored unless a task says otherwise. Signature of `observe`/`predict` must contain no
  ground-truth parameter (a test asserts it).
* Event `step` = `pool.step` at call entry (0-based index of the `observe` call); `tick()` once at the end of the call. Event kinds and required fields:
  `spawn{expert, parent, affinity, evidence, n_cluster, init, lam}`, `extend{expert, affinity, d_old, gap, n_cluster}`,
  `grow{expert, kind, increment, gain, p, params_added, vs_status}`, `merge{kept, removed, change_kept, change_removed}`, `archive{expert}`, `revive{expert}`,
  `alarm{expert, statistic}`, `abstain{reason, n_cluster}`. Every structural change must have one, carrying the statistic that triggered it.
* `StepRecord.assigned` = the expert that owns the sample after this call (it trained on it if labels were given); `-1` only for samples still in probation,
  in the bootstrap buffer, or dropped at the expert cap. `novel` = flagged by the pool's own rule; `p_nov` in [0,1], low = novel; `loss` = mean prequential/training
  loss over trained samples (`nan` if none); `spawned` = ids born in this call.
* `Prediction.expert` = routed expert id (-1 if none exist), `unknown` = abstain flag at level q, `probs` = best guess even when unknown. `predict` never learns
  and never updates a signature.
* An expert's seen-class mask: a newborn expert's mask is **reset to the labels of its own cluster** (an expert only claims classes it has trained on).
* Regime ground truth lives only in `bench.py` (Task 17). Onset of a regime = index of the first batch of its first-visit segment.

---------------------------------------------------------------------------------------------------------------------------
## 7. Rounds (do them in order; one round per report)

### R1 - `growmoe/probation.py` + PhyloCfg fields
```python
class Probation:
    def __init__(self, cap: int, d: int)
    def add(self, x, y, arrival: int)        # arrival = running count of samples the pool has observed; evicts the OLDEST when full; y may be None (stored as -1)
    def expire(self, now: int, ttl: int)     # drop entries with now - arrival > ttl
    def __len__(self)
    @staticmethod
    def tau(d: int, n: int, edge_null: float = 0.5) -> float
        # cosine threshold at which two RANDOM unit vectors are linked with probability edge_null / n, so the random graph is sub-critical:
        # tau = scipy.stats.norm.ppf(1 - edge_null / n) / sqrt(d)   (cos of random pairs ~ N(0, 1/d); ~0.27 for d=128, n=512)
    def clusters(self, min_neighbors: int = 4, m_min: int = 32, edge_null: float = 0.5) -> list   # index arrays, largest first
        # DBSCAN-style: A = (cos >= tau), no self loops; core = degree >= min_neighbors; connected components of the graph restricted to core points;
        # a non-core point joins a component if adjacent to one of its cores; keep components with >= m_min points
    def take(self, idx) -> (x, y)            # remove and return those entries
```
Why this and not "tau = quantile of the known data's nearest-neighbour similarity" (the original spec): with tight known regimes (NN cosine 0.96) a diffuse
new regime (AR(2), NN cosine 0.58) would never connect and never spawn; the random-null threshold is data-type-agnostic. Core filtering keeps isolated outliers out of
real clusters. Add to `PhyloCfg` (units in comments; all are STARTING values to be tuned on dev streams in R10): `m_min=32` (samples), `min_share=0.05` (fraction of the
recent stream a regime must make up; probation TTL = ceil(m_min / min_share) samples), `min_neighbors=4`, `edge_null=0.5`, `pool_cap=512`, `probe_steps=60`,
`probe_min_affinity=0.05`, `init_mode="shrink_perturb"` (`"scratch"|"clone"|"inherit"`), `rho_max=0.75`, `inherit_sigma=0.05`, `infant_calls=5`,
`infant_lr_mult=2.0`, `infant_relax=4.0`, `dormant_after=400`, `archive_after=800` (calls), `merge_every=50`, `delta_merge=0.02` (accuracy), `merge_overlap=0.2`,
`max_experts=256` (**stage 4.1, 2026-10-16**: a full pool follows `cap_policy`, default `"share"` = the cluster joins the nearest expert and the pool says so once; `"drop"` is the old silent behaviour, pinned in the historical baselines such as the M5 single network; `sig_dedupe=False` is an OFF switch for the twice-counted founding cluster, notebook E-037), `agem=False`, `competence=True`, `monitor_alpha=0.01`, `monitor_warmup=50`, `arm_monitor_after_infancy=True` (**added in M2-A, ON by default since 2026-10-10**: see 13f/13g below), `batch_vote=True`, `batch_vote_share=0.5`, `batch_vote_min=8` (**added in RV2-A, ON by default since 2026-10-12**: see 13k below), `grow=True`, and the stage-4 repairs `router=None` (resolved in `__post_init__`: **on with a fast `route_rule`, off with the router-blind reference path `route_rule="default"`**), `router_mode="free"` and `spawn_rule="transfer"` (**added in stage 4, ON by default since 2026-10-21**: pre-registered run RV2-D, notebook E-041, independent review E-042, adoption E-043), and `growth_rule="probe"`, `grow_gap=0.17` (growth from the fit gap, `"gap"`, passed its pre-registered statements but is NOT the default: it replaces the widen-and-deepen policy of R6, see E-042); `from growmoe.config import PRE_STAGE4` and `PhyloCfg(**PRE_STAGE4)` run the pool as it was before: no router, the 13e interference probe, the shadow-trial growth of R6.
Tests: 200 random unit vectors (d=128) -> no cluster; one vMF burst (kappa 60, 40 points) among 200 random outliers -> exactly one cluster holding >= 90 % of the burst and
<= 2 outliers; two bursts around orthogonal centres -> two clusters; a burst of 20 (< m_min) -> none; TTL expiry and capacity eviction; `tau` ~ 0.27 +- 0.01 for (128, 512),
decreasing in d, and Monte-Carlo mean degree of random vectors at `tau` within [0.3, 0.8].

### R2 - PhyloMoE stages 13a-13b (`growmoe/pools/phylo.py`, `tests/growmoe/test_gm_phylo.py`)
`PhyloMoE(BasePool)`, `name = "A_phylomoe"`, `__init__(cfg: PhyloCfg = None, grow_cfg: GrowthCfg = None)`. Per expert `meta`: `"sig"` (VmfSignature), `"monitor"`, `"recent"`.
Build small toy streams from `streams.s6_long(seed, n_regimes=k, d=64, n_train=..., related_frac=0, revisit_frac=0)` (clean latent regimes, 8 shared labels) and an S2 prefix.

* **13a Bootstrap.** While no expert exists, buffer `(x, y)`; once >= `m_min` samples are buffered, found expert 0 from scratch (`MLPExpert(cfg.d, cfg.hidden, cfg.depth,
  cfg.n_classes, seed=cfg.seed*1000+id)`), signature updated with all buffered samples, trained on them if labelled; log `spawn{parent=None, init="scratch"}`. Before founding: `novel=False`,
  `assigned=-1`. Test: a single-regime stream -> exactly 1 expert, no spawn before `m_min` samples.
* **13b Gate and known/novel split.** For every live expert k (including dormant/archived): `p_k = sig_k.pvalue(x)`, `logf_k = sig_k.logpdf(x)`; for infants `p_k = min(1, p_k * infant_relax)`.
  `p_nov = max_k p_k`; `novel = benjamini_hochberg(p_nov, q)`. A known sample is assigned to `argmax_k logf_k` among experts with `p_k >= min(q, p_nov)`.
  `predict` uses the same gate (no learning, no signature update): `expert = argmax logf`, `probs` from that expert's `proba`, `unknown = BH(p_nov, q)`; empty pool -> uniform probs,
  expert -1, p_nov 0, unknown True. Tests: samples of a seen regime -> empirical false-novelty <= 2q over 5 seeds (pool trained on 1 regime, scored on fresh samples of it);
  an unseen regime is flagged with power > 0.9 once the first expert is calibrated; routing among 3 trained regimes >= 0.95.

### R3 - 13c probation + 13d spawn/DNA
* **13c.** Novel samples go to `Probation` (stored with `y` or -1); `expire(now, ttl = ceil(m_min / min_share))` each call; clusters from `Probation.clusters`. Tests: 50 isolated outliers spread
  across 50 batches -> 0 spawns; one coherent burst of 3*m_min -> exactly 1 spawn; the pool never exceeds `pool_cap`; old entries are dropped.
* **Evidence test** for a cluster C: `evidence = sum_{u in C} [ logf_{C,LOO}(u) - max_k logf_k(u) ]`, where `logf_{C,LOO}` = `cal_scores + log_vmf_norm(d, kappa)` of a temporary
  `VmfSignature` fitted on C (use <= 256 members), and `max_k logf_k(u)` over live experts. Proceed only if `evidence > log(1 / alpha_spawn)`, else the samples stay in probation.
* **13d Spawn.** parent = `argmax_k sig_k.affinity(Cx)`, `aff` = that value; child module = per `init_mode` (default `parent.module.shrink_perturb(aff, seed)`; `"scratch"` fresh; `"clone"`;
  `"inherit"` = `parent.module.inherit(rho_max*(1-aff), inherit_sigma, rng)`); child `seen` reset to cluster labels; signature <- cluster; replay <- cluster; state `"infant"`;
  `lr_mult = infant_lr_mult`; log `spawn` with `evidence`, `affinity`, `init`, `lam`. Samples of the cluster get `assigned = child`. Tests: parent = max-affinity expert; `lam` grows with
  affinity (constructed affinities); with `init_mode="clone"` the child's outputs equal the parent's at birth; lineage event recorded.

### R4 - 13e interference probe + 13f infant protection
* **13e.** If labels exist, `aff >= probe_min_affinity` and the parent has replay: split the parent's replay into halves (R_fit, R_probe; need >= 16 in R_probe else spawn). Build
  `child = init(parent, aff)` and `shadow = parent.module.clone()`; give both fresh Adam optimisers; for `probe_steps` steps feed the SAME cluster batches to both (shadow batches mixed
  50/50 with R_fit). Then `d_old = L(R_probe; shadow) - L(R_probe; parent)` and `gap = L(cluster; shadow) - L(cluster; child)` (mean cross-entropy, nats).
  **Extend iff `d_old <= delta_forget` and `gap <= delta_forget`**; else spawn and KEEP the already-trained `child`. Extend = copy the shadow's weights into the parent, add the cluster to the
  parent's signature and replay, log `extend{d_old, gap}`. Unlabeled -> no probe, spawn. **Stage-4 addendum (2026-10-21):** the probe is the spawn rule only with `spawn_rule="probe"` (`PRE_STAGE4`); the default is now `spawn_rule="transfer"`
  (a held-out loss test of the best expert against its own reference, `growmoe/transfer.py`), which founds the true number of experts where the probe under-spawns (E-041). Tests: an aligned task (new cluster the parent already solves) -> extend; a conflicting task (same inputs region,
  permuted labels) -> spawn; the extended parent's loss on its held-out replay rises by <= `delta_forget` + noise tolerance; changing `delta_forget` flips the decision on a constructed borderline case.
* **13f.** Infants (`infant_calls` observe calls after birth) use `infant_relax` in the gate and `infant_lr_mult`; then become `active` with `lr_mult = 1`. Test: a regime that arrives as 5
  consecutive class-clusters (high mutual affinity) does **not** spawn 5 experts (the probe extends the infant).
  **M2-A addendum (2026-10-10):** while an expert is an infant its competence monitor is neither fed nor tested (`arm_monitor_after_infancy`, default ON) and its history is cleared when infancy ends.
  The original text above (monitor armed at birth) made every newborn raise a false drift alarm at its second batch; see `research/growmoe_results/m2a/RESULTS_M2A.md`.
  Set the flag to `False` to reproduce Phase 1 / M0.

### R5 - 13g competence, 13h merge/archive/revive, 13i unlabeled, 13j determinism, mini-integration
* **13g.** Per expert, feed `monitor.update` one prequential loss per labelled sample (loss computed before training on it). On alarm: log `alarm`; build a function-level cluster from the expert's recent
  labelled samples (`meta["recent"]`, deque of `m_min`) and call the cluster step with `skip_evidence=True, force_parent=k`; the probe should find the conflict and spawn; mark the child
  `meta["supersedes"] = k`; reset the monitor. Routing tie-break without `ctx`: among experts with `p >= q`, prefer one that supersedes another (recency) - this is the known limitation of input-only gating
  (design doc 6.5) and must be reported separately. Tests: same inputs with remapped labels (`s7_mapping(with_ctx=False)`) -> alarm and a spawn; stationary stream -> no alarm.
  **M2-A addendum (2026-10-10):** feed the monitor only for non-infant experts (default `arm_monitor_after_infancy=True`); a real remap on a settled expert still alarms (S7: 4.1 alarms per 4-regime run).
* **13h.** Dormant after `dormant_after` calls without an assigned sample (still gates, no training); archived after `archive_after` (drop replay and optimiser, keep module + signature, still gates);
  assignment revives it (recreate optimiser, log `revive`). Every `merge_every` calls, for non-infant pairs with mutual overlap `min(mean p_j(protos_i), mean p_i(protos_j)) >= merge_overlap`: distil the younger into
  the older (train a clone of the older for `probe_steps` steps on the union of both replays), accept iff accuracy on each side's held-out half drops by <= `delta_merge`; then replace weights,
  `sig.merge_from`, union replay, remove the younger, reparent its children, log `merge`. Tests per plan: two duplicate experts merge within `delta_merge`; an archived expert still gates and revives on its regime's return (same id).
* **13i.** `y=None`: gate + probation + evidence, spawn untrained signature-only experts, **never** call `train_step` (a test monkeypatches it to raise). 3-regime stream -> 3 experts.
* **13j.** Same seed -> identical `events.to_jsonl` text; `inspect.signature` of `observe`/`predict` has only `x, y, ctx` / `x, ctx`.
* **13k (RV2-A, ON by default since 2026-10-12).** After the per-sample assignment and before the probation buffer, `PhyloMoE._batch_consistent(assigned)` sends every accepted sample of the batch to the expert most of them were routed to, when that expert holds at least `batch_vote_share` (0.5) of them and at least `batch_vote_min` (8) were accepted; novel samples (-1) are untouched, ties go to the smallest id, the input tensor is not modified. Tests: `tests/growmoe/test_gm_rv2a_vote.py`. Pins: `bench._phase1_cfg` and `run_m0.A_KW` set `batch_vote=False` (`tests/growmoe/test_gm_rv2a_pins.py`). Evidence: `research/growmoe_results/rv2a/RESULTS_RV2A.md`; component card 14.
* **Mini-integration:** `s6_long(d=64, n_regimes=3, n_train=640, related_frac=0, revisit_frac=0)` plus a revisit: exactly 3 experts, final accuracy >= 0.9, the revisit does not spawn.

### R6 - Task 14 growth inside A
Per expert keep a loss history and ONE `GrowthPolicy`. When `plateau(loss_hist)` holds, the expert is active (not infant) and has >= 128 replay samples: split its replay 80/20 (seeded), call
`policy.step`; on commit rebuild the optimiser (keep `lr_mult`), log `grow{..., vs_status}`; clear the history after every check. Tests: a single-regime S5-style stream grows only while held-out loss improves;
final parameters <= 2x the smallest sufficient fixed expert; `grow` events carry the gain statistics; never grows on a data-limited regime (60-sample replay). **Stage-4 note (2026-10-21):** `growth_rule="gap"` (widen when the within-batch fit gap is below `grow_gap` = 0.17 and progress has stalled, with a guard window and a revert, `growmoe/gapgrow.py`) is available and passed its pre-registered statements (E-041), but it is not the default: it REPLACES this policy, so it never deepens, and on operator-by-operator arithmetic the shipped deepening is worth 1.2 to 1.9 points (not significant with six seeds; E-042). It rescues a 16-unit expert in 15 of 15 seeds where this policy does in 6 of 15, and costs about half the growth compute. The user decides its default. **Measure and report how often the policy fires with
`replay_size=256` versus a larger buffer** (power warning in section 5) and propose a justified per-expert buffer size for growth-eligible experts.

### R7 - Task 15 architecture B (`growmoe/pools/bnp.py`, `BNPCfg(CommonCfg)` with `alpha_dp, tau_b, n_stm, vae_steps, vae_lr`)
`VAEDensity(d=128, h=64, z=16)`: tanh MLP encoder/decoder, Gaussian decoder with learned diagonal log-variance clamped to [-8, 4]; `elbo(u) -> (B,)` (single-sample in training, posterior-mean z when scoring), `fit_step(u, opt)`.
`BNPMoE(BasePool)`: every expert = `MLPExpert` + `VAEDensity`. Responsibilities `log pi_k + log p_k(u)` with `pi_k ~ n_k` (CRP). A sample is novel if `max_k(log pi_k + log p_k(u)) < log alpha_dp + mu_ll - tau_b * sigma_ll`,
where `mu_ll, sigma_ll` are running (Welford) mean/std of the best-component log-likelihood of accepted samples. Rejected samples go to a short-term memory; when `|STM| >= n_stm` spawn exactly one **scratch** expert
(fresh classifier + fresh VAE) trained on the STM. Accepted samples train their top-1 component (hard responsibility). No growth, merge or archive (as published). `p_nov` = a monotone surrogate in [0,1].
Tests per plan (ELBO improves with training and is higher in-cluster than out; STM below `n_stm` never spawns; >= `n_stm` coherent rejected samples spawn exactly one; larger `tau_b` => fewer spawns on the same stream;
protocol conformance; deterministic).

### R8-R9 - Task 16 architecture C (`growmoe/delta.py`, `growmoe/pools/keyed.py`, `KeyedCfg(CommonCfg)` with `c, n_stm, rank, key_ema`)
`DeltaExpert(base, rank=16, lineage=[])`: effective weight `W = W_base + sum_{a in lineage} A_a B_a^T + A B^T` over a SHARED FROZEN base MLP; `fork()` (child lineage = parent lineage + frozen snapshot of the
parent's own delta; own delta zero-initialised = copy-on-write, function-preserving), `rank_up(k)`, `add_block()` (both function-preserving), `flatten(r)` (truncated SVD), `merge_siblings(other)` (valid because base and ancestors
are shared). It must expose the same expert API as `MLPExpert` (`train_step`, `proba`, `mark_seen`, `num_params`, `clone`) so `BasePool` helpers work. Design decision for you to make and justify with a measurement: how the shared base is obtained
(suggested: train it on the bootstrap regime, then freeze). `KeyedDeltaMoE`: per-expert key = EMA of the normalised mean input; gate = z-score of `cos(key_e, u)` on that expert's own samples; novel if `max_e z_e < -c`; same STM spawn rule as B;
spawn = fork from the nearest key; sibling delta merge; archive/revive (key keeps gating). Tests per plan (fork function-preserving, copy-on-write isolation, rank_up/add_block preserving, flatten error bound, merge within tolerance,
memory per expert <= 40 % of an equivalent `MLPExpert`, key gate AUROC > 0.9 on two regimes, protocol, deterministic).

### R10 - Task 17 benchmark runner and the Phase-1 comparison (`growmoe/bench.py`, CLI `python -m growmoe.bench --phase 1 --seeds 10 --out research/growmoe_results/phase1`)
`run_stream(make_pool, stream, seed, batch_size=64) -> RunResult`: **test-then-train** per batch (`predict` first, then `observe`), ground-truth bookkeeping outside the pool, accuracy matrix R after each segment
(`accuracy_stats` with `learned_at` = row after each regime's first-visit segment, `chance = 1/n_classes_in_regime`), per-sample novelty flags vs `gt["first_visit"]` / `is_outlier`, spawn stats (`regime_owner` over
assigned vs true regimes; `onsets`), parameters, wall-clock. `tune(make_pool, grid, dev_streams, budget=12) -> best_cfg` uses **dev seeds >= 100 only** (assert it); freeze, then run test seeds 0-9 for A, B, C, B0-B5 and the ablations
(no conformal / no probation / scratch|clone|inherit|shrink_perturb init / always-spawn / never-extend / no replay / no growth / familiarity in {Gaussian-Mahalanobis, kNN, vMF-KDE, RND} - the latter needs signature classes with the
`VmfSignature` interface: `update, fit, score, logpdf, pvalue, affinity, nn_similarity, merge_from, n, protos`). Write `research/growmoe_results/phase1/RESULTS.md` with P1-P6, P8, P10 and C1-C6 as pass/fail tables, CIs and paired tests.
Sanity guardrails (not targets): on S2 A should spawn ~1 expert per regime (recall >= 0.9, spawned/true <= 1.2) with false-novelty <= 2q; on S4 <= 1 % of injected outliers cause a spawn; the baselines must reproduce the ranges in section 4.

### Later - Phase 2 (Tasks 18-19: `stack.py`, vertical scaling, S5/S6) and Phase 3 (Tasks 20-21: `system.py`, columns, hybrids) and Task 22 (final report). Details are in the plan; Claude will refine them when R10 is approved.

---------------------------------------------------------------------------------------------------------------------------
## 8. How to write tests in this repo (each rule is a mistake already made once)

1. **Assert the scenario's precondition.** If a test needs "a real gradient conflict", assert the conflict first; never go hunting for a seed that passes.
2. **Statistical tests average over independent seeds.** One conformal calibration set's coverage is Beta-distributed (+-0.013 at n=256); one reservoir's mean rank has sd 178. Pool 5-10 seeds, set the window from the sampling distribution.
3. **Measure the margin before pinning a threshold**, then pin it with a stated reason (`# measured 0.18-0.39 over 5 seeds`). If a threshold was guessed in the plan and measurement disagrees, change the test and say why.
4. **Test the property, not the implementation.** Prefer hand-computable arrays (AUROC 0.75 on four points), exact identities (lam=0 equals a fresh expert), and invariants (function-preserving growth to 1e-12 in float64).
5. A test that cannot fail is worthless: a spike or test that saturates (every arm 100 %) is a failed design - raise the difficulty and record that you did.
6. Red first: run the new tests and confirm they fail **for the stated reason** (missing module/attribute or the behaviour under test), and paste that output.
7. Deterministic, fast, cached builders; the whole suite stays under ~3 minutes.

---------------------------------------------------------------------------------------------------------------------------
## 9. Report template (Gemini -> Claude, via the mediator) and what Claude checks

```
ROUND: R<n> <title>                       STATUS: done | blocked
FILES: created/modified (paths)           SUITE: "<N> passed in <T>s" (paste the last line of the full run)
RED EVIDENCE: the failing-test summary lines from the first run of the new tests (<= 10 lines)
PLAIN SUMMARY (for the user): 3-5 lines in simple words. Keep the technical terms but define each once, say what each number means
                and how sure we are (measured vs assumed). The user is knowledgeable but still learning; do not water the meaning down.
WHAT I BUILT: <= 10 lines, only what is not obvious from the code
MEASUREMENTS: tables (numbers, seeds, CI) for every threshold I pinned and every claim I make
DEVIATIONS FROM THIS HANDOFF: each with the measurement that justifies it (or "none")
FAILURES / SURPRISES: things that did not hold (or "none")
QUESTIONS FOR THE MEDIATOR: numbered, each with the options and my recommendation (or "none")
```

Claude's review checklist: tests read before code; red evidence plausible; no vacuous or seed-lucky assertions; thresholds backed by measured margins; no `ishmn/` change, no `hash()`, no ground truth in pool
signatures, no tuning on test seeds; events carry the triggering statistics; deviations justified by numbers; the suite re-run by Claude; results compared with the section 4/7 guardrails. Verdicts: `APPROVE`, `CHANGES: 1) ... 2) ...`
(each item testable), or `ANSWER:` to your questions.

---------------------------------------------------------------------------------------------------------------------------
## 10. Known risks to watch (report if you see them)

* Probation clustering vs shared structure: S2 regimes share a common direction (cross-regime cosine ~0.45-0.7), so two simultaneously novel regimes could merge into one component. Report any under-spawn.
* A diffuse regime (AR(2)) makes a weak signature that can wrongly claim foreign data; competence and the probe are the net. Report S1 per-type recall.
* Signature reservoirs are uniform over history, so under drift (S3) they lag; the probe's *extend* path is what should absorb drift. Report spawn counts on S3.
* Probe power and cost: each cluster event trains two models for `probe_steps`; report wall-clock per event.
* Everything in R1's PhyloCfg is a starting value. Nothing may be tuned on test seeds.
