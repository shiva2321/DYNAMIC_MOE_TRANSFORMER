# GrowMoE design spec — three self-growing, novelty-aware MoE architectures, compared head to head

Date 2026-10-04 · Source research and idea bank: `research/DYNAMIC_MOE_RESEARCH_AND_DESIGN.md` (read it for the literature, the ISHMN lessons and the cross-domain ideas; this spec records only what was decided and the exact definitions).

## 1. Decisions (from the user, 2026-10-04)

| Decision | Value |
|---|---|
| Architectures | Build **all three** — A (calibrated phylogenetic), B (Bayesian-nonparametric / CN-DPM-style), C (key-addressed delta "DNA" experts) — and compare them on identical streams |
| Scope | **Full arc, Phases 0–3** (spikes → core pools + expert growth → vertical stack → system-level columns and clade routing) |
| Data | **Local only** — scikit-learn bundled datasets, Python stdlib text/code on disk, synthetic generators. No downloads |
| Location | Standalone package `growmoe/` + `tests/growmoe/`; **nothing in `ishmn/` is modified** |
| Process | Test-first (red → green) for every production module; no commits unless the user asks |

## 2. Requirements being tested (restated from the research doc)

R1 calibrated novelty with a verified false-novelty rate · R2 data-type agnostic (one config, many types) · R3 known-by-which-expert or *unknown* (abstain) · R4 spawn-from-parent "DNA" · R5 use / train / extend / spawn decision · R6 vertical and horizontal scaling by expert, layer, system · R7 testable with ground-truth streams, baselines, ablations, pre-registered criteria.

## 3. Common interfaces (all architectures implement these)

```python
# growmoe/types.py
@dataclass
class StepRecord:            # returned by pool.observe(...)
    novel: Tensor            # (B,) bool   — flagged novel by the pool's own rule
    p_nov: Tensor            # (B,) float  — novelty p-value or a [0,1] surrogate (low = novel)
    assigned: Tensor         # (B,) long   — expert id trained on this sample, -1 if none (pooled/novel)
    loss: float              # mean training loss over trained samples (nan if none)
    spawned: list[int]       # ids of experts born during this call
    n_experts: int

@dataclass
class Prediction:            # returned by pool.predict(...) — never learns
    probs: Tensor            # (B, C) class probabilities (mixture/top-1 of experts; best guess even if unknown)
    expert: Tensor           # (B,) long   — routed expert id, -1 if none exists
    p_nov: Tensor            # (B,) float
    unknown: Tensor          # (B,) bool   — abstain flag at the pool's configured level

class ContinualPool(Protocol):
    name: str
    events: EventLog
    def observe(self, x, y=None, ctx=None) -> StepRecord: ...
    def predict(self, x, ctx=None) -> Prediction: ...
    def summary(self) -> dict: ...   # n_experts, n_params, active_params_per_sample, lineage (parent ids), states
```

* `x` is always a `(B, d)` tensor of **unit vectors** produced by a frontend; `y` is `(B,)` long labels in a **global label space** of size `n_classes` (or `None` = unsupervised: regime discovery only, no expert training); `ctx` is an optional `(B, c)` context.
* **Algorithms never see ground truth** (regime ids, first-visit flags, outlier flags). The benchmark keeps them separately; a test asserts the pool APIs have no such parameter.
* Telemetry: every structural change appends `Event(kind, step, fields)` with kinds `spawn | extend | grow | merge | prune | archive | revive | alarm`.

## 4. Shared components

| Module | Contract |
|---|---|
| `frontend.HashFrontend(d=128, seed=0)` | Training-free, deterministic (no Python `hash()`), maps numeric arrays of any shape (feature hashing with signs) or bytes/str (hashed 1–3-gram counts) to unit vectors. `salt_by_shape=False` by default so data type is **not** leaked through the map |
| `conformal.conformal_pvalues(cal, test)` | `p = (1 + #{cal ≤ test}) / (n + 1)` for scores where *low = novel*; `benjamini_hochberg(p, q) -> bool mask` |
| `signature.VmfSignature` | ≤ `max_protos` prototypes (reservoir sampling), vMF-KDE score `lse_j(κ⟨p_j,u⟩) − log n`, κ chosen by leave-one-out **normalised** likelihood over a grid (normaliser `log_vmf_norm(d, κ)` via scaled Bessel), LOO calibration scores, `pvalue(u)`, `logpdf(u)` |
| `experts.MLPExpert` | Residual MLP blocks `h ← h + W2 σ(W1 LN(h))` + linear head over the global label space with a **seen-class mask**; `widen(k)`, `deepen()`, `inherit(rho, sigma)` (re-draw a fraction `rho` of hidden units, add noise `sigma`), `clone()`. Growth and `inherit(0,0)` are **function-preserving** (atol 1e-6) |
| `competence.LossChangeMonitor` | Page–Hinkley-style change test on prequential loss; stationary false-alarm rate ≤ nominal |
| `growth.GrowthPolicy` | Plateau + under-fit detection; **propose → validate → commit** on shadow copies (widen vs deepen), significance by paired bootstrap, exponential back-off on rejection |

## 5. The three architectures (exact differences)

| | **A — PhyloMoE** | **B — BNP-MoE (CN-DPM-style)** | **C — Keyed Delta-MoE** |
|---|---|---|---|
| Expert | full `MLPExpert` | full `MLPExpert` + per-expert `VAEDensity` (d→64→z16→64→d, Gaussian decoder with learned diagonal variance) | **shared frozen base** MLP + per-expert **low-rank delta** (`DeltaExpert`, rank r) + masked head; weights = base + Σ frozen ancestor deltas + own delta |
| Gate / novelty | per-expert `VmfSignature`; `p_nov = max_k p_k`; Benjamini–Hochberg at level `q` | DP-mixture responsibilities `log π_k + log p_k(u)` (ELBO) vs `log α_dp + log p_new`; `log p_new = μ_ll − τ_B·σ_ll` (running statistics of accepted samples) | key activation `cos(K_e, u)` z-scored per expert (`μ_e, σ_e`); novel if `max_e z_e < −c` |
| Evidence before birth | probation pool → similarity graph (edge cos ≥ τ_in, τ_in = low quantile of known data's NN similarity) → cluster size ≥ `m_min` and Σ normalised-density log-LR > log(1/α_spawn) | short-term memory (STM) of rejected samples; spawn when `|STM| ≥ n_stm` | same STM rule as B (so C vs B isolates parameterisation + key gate) |
| Adapt vs spawn | **interference probe**: shadow-parent trained K steps on the cluster; `Δ_old = L_old(shadow) − L_old(parent)`; extend iff `Δ_old ≤ δ_forget` and shadow ≈ child on the cluster; else spawn | none (always spawn from scratch, as published) | none (always fork) |
| Child init (DNA) | `inherit(parent, ρ = ρ_max·(1 − affinity), σ)`; lineage + novelty stats recorded | scratch (as published); `clone` variant available for ablation | **fork**: child's frozen lineage = parent's lineage + snapshot of parent's delta; own delta zero-init (copy-on-write, function-preserving) |
| Competence monitor | yes (known-input / wrong-output) | no | no |
| Growth | widen / deepen via `GrowthPolicy` | none (as published) | rank-up (widen analogue) and extra residual delta block (deepen analogue) via `GrowthPolicy` |
| Merge | distil younger into older on union of reservoirs, verify on probe | none | **sibling delta average** (valid: shared base and ancestor snapshot) + verify on probe |
| Archive / revive | yes (signature keeps gating) | no | yes (key keeps gating) |
| Main tunables | `q`, `δ_forget`, `α_spawn`, `m_min`, `ρ_max` | `α_dp`, `τ_B`, `n_stm`, VAE steps | `c`, `n_stm`, rank `r`, key EMA rate |

**Hybrids after the three work:** A-gate × delta experts, B-gate × clone init (ablation matrix in the comparison task).

## 6. Scaling beyond one pool (Phases 2–3)

* **Phase 2 — `StackMoE` (vertical).** L−1 *representation pools* (experts `R^d → R^d`, residual, identity-initialised so a new expert/layer is function-preserving) followed by a classifier pool. Layer-l gate uses a `VmfSignature` of that layer's input; **mature representation experts are frozen** (only infants and the classifier pool train) so stored prototypes stay valid. `add_layer()` inserts an identity layer on every path; commit by propose → validate → commit against the widen/deepen alternatives.
* **Phase 3 — `SystemMoE` (columns + clade routing).** A top-level conformal gate over *columns* (each a pool or stack) using the union signature of the column's experts; **column birth** when a novel cluster is far from every column (new modality), **expert birth inside a column** when it is moderately novel. Two-level routing (column → leaf) must match flat routing accuracy while evaluating ≥ 3× fewer signatures at ≥ 100 regimes.

## 7. Streams (all local; deterministic by seed; ground truth kept outside the algorithms)

| Stream | Regimes | Labels | Size |
|---|---|---|---|
| **S1 types** | digits (image 8×8), wine, breast-cancer, sinusoid windows (4 freq. classes), AR(2) windows (3 classes), English bytes (4 pydoc topics), Python-code bytes (4 stdlib modules) | disjoint label blocks (class-IL), 30 labels | 600 train / 200 test per regime (small sets bootstrapped with jitter) |
| **S2 permuted digits** | 30 fixed pixel permutations of digits + revisits `[0..14, 3, 7, 0, 12, 15..29, 5, 20, 1]` | shared 10 labels (domain-IL) | 1,000 train / 500 test per segment |
| **S3 drift** | permutation interpolation: swap-fraction φ(t) 0 → 1 | shared | 4,000 samples |
| **S4 outliers + noise** | S2 prefix with injected isolated outliers (Gaussian / foreign-type vectors) at 1 % and label noise 5–20 % | shared | as S2 |
| **S5 capacity ladder** | teacher MLPs width {8, 32, 128} × depth {1, 2, 3} on `d=128` inputs | 8 classes | 2,000 train / 500 test |
| **S6 long horizon** | 60 and 100 regimes: one random orthogonal rotation per regime of the same 8-class Gaussian-mixture family; 30 % of regimes are *related* (small rotation of a parent) | shared 8 | 500 train / 200 test |
| **S7 same inputs, new mapping** | identical inputs, class-permuted labels, with and without a context vector | shared | 600 / 200 |

## 8. Baselines and ablations

B0 sequential fine-tune · B1 joint i.i.d. (ceiling) · B2 single network + reservoir replay · B3 static learned-router MoE (fixed E, load-balance bias) · B4 **oracle-routed** one expert per regime (routing ceiling) · B5 DP-means gating with fixed λ. Ablations: no conformal (fixed threshold), no probation, scratch / clone / affinity-scaled init, always-spawn / never-extend, no replay, no growth, familiarity ∈ {Gaussian-Mahalanobis, kNN, vMF-KDE, RND}.

## 9. Protocol, metrics and statistics

* **Test-then-train:** for each batch, `predict` is called and logged *before* `observe`. After each segment evaluate every regime's held-out set (accuracy matrix, no regime id given).
* **Novelty:** sample-level AUROC (`1 − p_nov` for first-visit vs seen-regime samples), false-novelty rate per 1,000 known samples, empirical FPR vs nominal `q`.
* **Regime recovery:** an expert *owns* a regime if it handled the plurality of that regime's training samples; spawn precision/recall, spawned/true-regimes ratio, time-to-spawn (samples), revisit routing accuracy.
* **Learning:** average accuracy, forgetting, forward transfer (samples to 90 % on a related regime, child vs scratch), calibration of abstention (coverage vs accuracy).
* **Efficiency:** experts, parameters, active parameters/sample, wall-clock, buffer memory.
* **Rules:** ≥ 10 seeds, mean ± 95 % CI, paired tests; **hyper-parameters tuned only on development streams** (different seeds and regime draws, ≤ 12 configurations per method) then frozen; exact numeric grading; the same tuning budget for A, B, C.

## 10. Pre-registered predictions (written before any code) [H]

Single-pool predictions P1–P10 are listed in the research document §7.5 and apply to A. **Comparative predictions:**

| # | Prediction | Failure if |
|---|---|---|
| C1 | A has the lowest false-novelty rate and the highest regime recall of the three on S1 and S2 | B or C beats A on either, CI excluding 0 |
| C2 | B is the most threshold-sensitive: its spawn count varies > 2× over ±1 grid step of `τ_B`, vs < 1.5× for A over ±1 step of `q` | A is no less sensitive than B |
| C3 | C ends S2 with ≤ 40 % of A's parameters and accuracy within 3 points of A | > 40 % or > 3 points worse |
| C4 | A's affinity-scaled child beats scratch (B) on forward transfer for related regimes, and C's fork matches A within 10 % at lower memory | A no better than scratch; or C > 10 % worse |
| C5 | B's likelihood gate fails (AUROC < 0.8) on at least one S1 type pair | B ≥ 0.8 on all pairs |
| C6 | Wall-clock per run: B slowest (VAE training), C fastest | order differs |

## 11. Gates

Each phase ends with: full `tests/growmoe` suite green; a results table for that phase's predictions written to `research/growmoe_results/`; honest failures reported, not tuned away. Phase 4 (ISHMN integration) is **not** started without the user's go-ahead.
