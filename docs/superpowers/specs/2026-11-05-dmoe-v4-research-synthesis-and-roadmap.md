# dMoE v4 — Research Synthesis & Roadmap

Repository: `D:\Dynamic_MOE_Transformer` · Date: 2026-11-05 · Status: **research only. Nothing built, no code changed.**
Method: read all of `dmoe/`, the Phase 1–3 docs, and the 20-seed result files; ~35 web searches; 7 papers read in depth (HTML/abstract; PDFs were unreadable to the fetch tool, so some numbers below come from model-extracted summaries and are flagged).

**Evidence tiers used below**
- **[A] Established** — replicated across groups or standard practice.
- **[B] Credible** — peer-reviewed (NeurIPS/ICLR/ICML 2025–26), single group, authors' own numbers.
- **[C] Preprint** — arXiv only, no independent replication found. Treat as a hypothesis generator, not as proof.
- **[H] Hypothesis** — my synthesis. Not in the literature as stated. Each has a cheap falsifier in §7.

---

## 1. Where the architecture stands (verified from your own data)

20 seeds, no replay, 3 toy byte-domains (`results/dmoe_v2_noreplay_routing/`):

| Condition | avg final acc | Notes |
|---|---|---|
| Dense transformer, no replay | 0.349 | forgets A (0.003) and B (0.094), keeps C (0.951) |
| **dMoE, no replay** | **0.349** | paired diff vs dense = **−0.0005 ± 0.0023** (n=20) → zero |
| dMoE, mutation disabled | 0.348 | growth/freeze machinery contributes nothing |
| dMoE + global LDA fast memory | 0.508 | same as dense + LDA (0.508) — gain is from LDA, not from dMoE |
| dMoE with replay | 0.952 | equals joint-training ceiling (0.952) |
| `dMoE_v3_ZeroReplay` (SetGate + LocalLDA + **backbone frozen after domain A**) | **0.953**, retention 1.000, bwt −0.005 | 20 seeds, passes F1–F5 (`results/dmoe_v3_zero_replay/results.json`, merged 13:09). **Correction:** my first draft quoted an earlier single-seed result (0.363, bwt 50 nats) produced by the code *before* the 12:44–12:53 edits. See §1.3 for what the pass does and does not show. |

Routing diagnostics (`route_domain_id_acc_mean_layers`, chance = 0.33): dMoE 0.36 ± 0.03, **untrained network 0.48 ± 0.08**; JSD gap between domains ≈ 0.000. **The experts exist but nothing sends the right inputs to them.**

### 1.1 Why — three coupled failures, each traceable to code

**D1. The address space drifts.** Encoder, attention, norms, decoder and manifold stay trainable across domains (`continual.py:32`, `freeze_backbone_on_domain=False` by default; `domain_transition` at `:210` only freezes experts). Frozen experts therefore see shifting inputs and write into a shifting readout. Stored statistics (exemplars in `experts.py:229`, LDA sufficient statistics in `fast_memory.py`) are computed in a feature space that keeps moving. This is the "semantic drift" failure documented for prototype methods (LDC, ECCV 2024) [B].

**D2. The router is unprotected.** One `nn.Linear` + softmax over all experts (`experts.py:320`). A spawned expert's router row is a copy of the most-used expert's row plus N(0, 0.01) noise (`experts.py:408-410`, `:581-583`), so the new expert starts as a near-duplicate that competes for the *old* domain's tokens. Nothing penalises capturing old inputs. Recent MoE-CL papers name exactly this *routing drift*: "expert expansion can shift old-class routing by normalizing over an expanded expert set, even when historical expert parameters are frozen" (StaR-MoE, 2605.17571) [C].
The v3 path avoids the learned router with a batch-level permutation test on `chunk0` magnitudes (`model.py:171-204`, `routing.py`). That (a) is a task-ID oracle in disguise (one expert for the *whole batch* at *every layer*, so it cannot handle mixed streams), (b) runs on drifting latents, and (c) does 60–100 CPU permutations with `.cpu()` per forward (`routing.py:112`).

**D3. The readout is overconfident when wrong.** Unseen classes get bias `-1e9` (`fast_memory.py:85`); a mis-routed expert emits near-infinite-loss logits. This is the likely source of `loss_A_final = 55.7` in v3. LDA lifts accuracy (+0.16) but *worsens* backward transfer (7.6–8.4 nats) because its logits are confident on stale features.

### 1.2 Other findings that matter for the goals you listed
- **"Thinking" is not trained to help.** `_run_simulation_loop` (`model.py:107-161`) re-applies the last layer with damping; training uses `random.randint(1, max_steps)` (`:115-119`) and no loss rewards extra steps or checks that they reduce error. Convergence is judged by the manifold's own prediction error at eval only. The norm-explosion fix (`_complex_unit_norm`) stabilises it but does not make it useful.
- **It is not a next-token model, yet.** Attention defaults to `causal=False` (`attention.py:68`) and the layer never passes `causal` (`layer.py:44`). Every result so far is a bidirectional seq→seq mapping on a 3-domain toy (L=16, d_model=32, domain = one header byte). Good unit test; weak evidence about generalisation or language.
- **Efficiency.** With `top_k=None` the MoE runs *every* expert on *every* token (`experts.py:501-508`) — dense compute with extra parameters. The sparse path does a device sync per layer (`.tolist()`, `:365`). Complex arithmetic costs 2× multiplications at equal parameter count and 4× at equal width; no ablation in the repo shows complex beats real at equal FLOPs. The 2.12M→105K parameter drop in Phase 3 came from removing the vocabulary matrices, not from complex ops.
- **JEPA** uses an EMA teacher plus a variance hinge (`jepa.py`). That is the 2022-era recipe; LeJEPA (below) replaces it.

### 1.3 Update (2026-11-05, after the full v3 run): what the v3 pass shows and does not show

`dMoE_v3_ZeroReplay` is a legitimate, pre-registered pass: accuracy 0.9533 ± 0.0043, retention 1.000, bwt −0.005 nats, +0.603 over dense no-replay, nulls ≤ 0.209 (shuffled targets) / 0.076 (blind) / 0.003 (untrained), max condition 0.9613 < 0.97 ceiling guard. It is also direct empirical confirmation of the D1/D2 diagnosis: **freeze the shared representation, isolate per-domain experts with their own closed-form heads, and route exactly — forgetting is zero.**

But the configuration is the opposite of what the project now requires:

| Property of the v3 pass | Evidence | Why it matters |
|---|---|---|
| Backbone frozen after domain A | `freeze_backbone_on_domain=True`; `_freeze_backbone()` freezes encoder, manifold, attn, norms, decoder (`continual.py:219-234`) | Not "always plastic". The "foundation" is 240 steps of domain A. |
| Domain boundaries are supplied by the harness | `learner.domain_transition(...)` called by the script; mutation is *disabled* whenever SetGate is on (`continual.py:197`) | Not task-free. Growth is not driven by novelty. |
| Routing = reading a header byte, per batch | `SetGate` on chunk-0 latents (`model.py:171-189`); in the toy, slot 0 holds a domain-specific prefix byte (10–11 / 12–13 / 14–15) and every batch is one domain | A batch-level task-ID oracle; cannot handle mixed streams. |
| Task is a per-position lookup | `generate_easy_domain_batch`: `y_t = f_domain(x_t, t mod 4)` (+5% noise) — no context needed | A closed-form head over byte features can memorise it; it cannot separate a learner from a lookup table. |
| Eval reads out only the routed expert's closed-form LDA | `predict_logits` returns `expert.predict_logits(...)` (`continual.py:281-292`) | The neural path is not what is being scored. |
| Accuracy matches but **calibration does not** | v3 NLL (nats) A/B/C = 2.229 / 2.231 / 2.426 vs dense-replay 0.407 / 0.416 / 0.419; Bayes-optimal for this noise ≈ 0.316 | Fine for argmax accuracy, unusable as a language/probability model (probably overconfident LDA logits — not yet verified). |

So v3 is the **L0 reference ceiling with a frozen backbone**, not evidence for the target regime. The open scientific question is exactly the gap between it and an always-plastic, task-ID-free learner (see the Phase A design doc).

---

## 2. What the field says (organised by your goals)

### 2.1 Continual learning without replay

| Idea | Evidence | What it buys you | Caveat | Tier |
|---|---|---|---|---|
| **Frozen/slow backbone + closed-form head** (RanPAC, ACIL/DS-AL/GACL, Analytic Subspace Routing) | RLS head on frozen features equals joint training for the head, no exemplars; RanPAC −20…62% error vs prior on 7 benchmarks | Exact, replay-free, no forgetting *in the head* | Needs a good frozen feature space; linear head under-fits; ASR notes difficulty preserving general knowledge in LLMs | A/B |
| **Drift compensation** (LDC ECCV'24, ADC CVPR'24) | Oracle drift correction recovers much of the loss; LDC learns a projector old→new features from current data + stored prototypes | Lets the backbone keep learning while old class stats stay valid | Approximate; prototype methods degrade with small first task | B |
| **Sparse, specific updates** (Sparse Memory Finetuning, Meta/Berkeley 2510.15103) | 1.3B model, memory layer: 1M slots, k=32 lookups; update only top-t slots by TF-IDF vs a background of generic usage. NQ F1 drop **11%** vs 71% LoRA / 89% full FT at equal new-fact accuracy | Forgetting from *shared* parameters is avoided by touching only input-specific ones | Only factual QA, one 1.3B model. **SGD, not AdamW** — "adaptive step sizes, weight decay, and momentum can interact with sparsity in unexpected ways" | B |
| **Null-space / orthogonal updates** (AlphaEdit ICLR'25, O-LoRA, InfLoRA; GPM family) | Project the update away from the subspace spanned by old inputs → old outputs unchanged (AlphaEdit claims a guarantee for preserved knowledge) | Replay-free guarantee, costs one d×d covariance per layer | Capacity shrinks as tasks accumulate; guarantee is for linear layers on the preserved distribution | A/B |
| **Router protection** (StaR-MoE 2605.17571; MoRAM ICML'26; LLaVA-DyMoE; LEGO ICML'26) | StaR-MoE stores per-class mean/diag-covariance of *router inputs* (no exemplars) and aligns the router's distribution with a KL loss; proves a drift bound √(2Nξ); beats SEMA/InfLoRA/SD-LoRA on IN-R, IN-A, VTAB (frozen ViT-B/16). MoRAM drops the router: each rank-1 expert scores inputs by its own key | Directly addresses D2 | CV benchmarks, frozen pretrained ViT | B/C |
| **Forgetting tracks KL on new data** (RL's Razor, ICLR'26) | Forgetting is predicted by KL(fine-tuned‖base) measured on the *new* task, no old data needed; on-policy / self-generated targets move the model less | A free forgetting meter and a replay-free regulariser (LwF-style) | Shown on LLM/robot fine-tuning; a predictor, not a guarantee | B |
| **Multi-timescale memory** (Nested Learning / Hope, Google NeurIPS'25) | Chain of MLP blocks updated at different frequencies; Hope beats Titans/Transformer++ at 760M/30B tokens (Wiki ppl 18.68 vs 24.18) | A principled fast→slow organisation | Authors: forgetting "is not 'solved' in general" (the blog says "mitigate or even completely avoid"). CL numbers appear only in figures. Transformers still win in-context recall. | B |
| **Sleep** (Google 2606.03979, blog Oct 2026) | Step-scheduled consolidation into a new low-rank expert + "dreaming" on self-generated data; reported near-zero drop in continual translation; Qwen3-8B AIME-24 79.2 vs GRPO 76.4 | Replay-free *storage*; capacity grows by schedule | Abstract says "with replay" while §3.3 says "should not simply replay raw data"; dreams are context-conditioned; no CL numbers in text; ~4× slower than SFT at matched steps; own baselines only | C |
| **Fly-style expansion + k-WTA** (Shen/Dasgupta/Navlakha 2023; Fly-CL 2510.16877) | Random sparse expansion + winner-take-all + simple readout significantly boosts CL, extended to pretrained backbones | Decorrelates classes → less interference, nearly free | Gains shown on small/vision benchmarks | B/C |
| **Plasticity matters too** (Dohare et al., Nature 2024) | Standard nets lose plasticity over long task streams; reinitialising low-utility units restores it | Freezing everything eventually costs learning ability | — | A |

**Key conclusion of the CL literature:** protect the *address/route* and the *shared representation* first; the expert content is the easy part. Your v2/v3 work protected content and left both of the hard things moving.

### 2.2 Latent simulation, thinking, "understanding"

| Idea | Evidence | Relevance | Tier |
|---|---|---|---|
| **TRM** (Samsung, 2510.04871) | 2-layer, 7M params, recursion over (answer y, latent z) with **deep supervision**: 45% ARC-AGI-1, 8% ARC-AGI-2 (HRM 27M: 40.3/5.0). ARC Prize's analysis credits deep supervision, not hierarchy. ~1000× augmentation caveat | The recipe your loop lacks | B |
| **Attractor Models** (USC, 2605.12466) | Backbone proposes ŷ₀; a small weight-tied module is iterated to a fixed point with ŷ₀ re-injected; trained by implicit differentiation with a **one-step** gradient (≈constant memory). 770M comparable to a 1.3B Transformer; Sudoku-Extreme 91.4%, Maze-Hard 93.1% at 27M (TRM 7M: 74.7 / 85.3). Authors admit the contraction assumption is local; v1 only; at 7M it loses to TRM | A principled replacement for the damped re-application loop | C |
| **Energy-Based Transformers** (ICLR'26) | Predict by minimising a learned energy over candidates; claims up to 35% faster pretraining scaling, +29% inference-time gain; larger gains OOD. Needs replay buffer + Langevin for stability | Energy = verifier = uncertainty = compute knob | B |
| **Mixture-of-Recursions** (NeurIPS'25) | Shared block, per-token router chooses recursion depth; ≈2× throughput (MoR-4) | Per-token adaptive depth with shared weights | B |
| **Depth-recurrent LMs** (Huginn et al.) | Follow-up analysis: more recurrent steps help GSM8K only marginally, far below explicit CoT | Calibrates expectations: latent recurrence is not magic at LM scale | B |
| **Continuous Thought Machines** (Sakana, NeurIPS'25) | Per-neuron temporal models; synchronisation between neurons is the representation; stops early on easy inputs; costlier to train | Radical but expensive; a research platform | B |
| **World models** (Dreamer 4 2509.24527; V-JEPA 2-AC) | Dreamer 4 reached diamonds in Minecraft *purely offline*; learns action conditioning from ~100 h action-labelled out of 2500 h video (85% of full). V-JEPA 2: 1.2B, 62 h robot data → zero-shot MPC planning in latent space | Passive observation builds the dynamics; a little interaction grounds it | B |
| **LeJEPA / SIGReg** (LeCun & Balestriero, 2511.08544) | Pushes embeddings to an **isotropic Gaussian** with one regulariser; removes teacher/stop-grad/predictor heuristics; linear cost | Fixes collapse by construction; Gaussian latents make downstream closed-form stats well-posed | B/C |
| **LLM-JEPA** (NeurIPS'25 / ICLR'26) | JEPA term added to the token loss improves fine-tuning and pretraining; needs paired "views" (text↔code) | Latent prediction is useful *beside* CE for language | B |
| **Large Concept Models** (Meta 2412.08821) | Next-*sentence* prediction in a frozen SONAR space (1.6B, 1.3T tokens) | Concept-level prediction is feasible; no clear win over token LLMs | B |
| **ARC Prize 2025 report** (2601.10904) | Winners: test-time training and refinement loops; compositional reasoning and interactive learning unsolved; 24% on ARC-AGI-2 needed hundreds of thousands of synthetic examples → "reasoning remains knowledge-bound" | Sets honest expectations and a test bed | A/B |

### 2.3 Efficiency and "any input"

| Idea | Evidence | Tier |
|---|---|---|
| **H-Net / dynamic chunking** (ICLR'26) | Learned byte boundaries, 2-stage hierarchy overtakes a BPE Transformer after ~30B bytes at 760M–1.3B; largest gains on code, Chinese, DNA (≈4× data efficiency on DNA) | B |
| **Hashed n-gram memory** (DeepSeek *Engram*, Jan 2026; also in BLT) | O(1) lookup as a second sparsity axis beside MoE; U-shaped allocation law; Engram-27B > MoE-27B at equal FLOPs; a 100B table on CPU costs ~3% throughput. (Secondary sources only; I did not read the paper.) | B/C |
| **Product-key / million-expert layers** (PEER, DeepMind 2407.04153) | Single-neuron experts retrieved with product keys; best perplexity at fixed FLOPs vs dense and coarse MoE | B |
| **Aux-loss-free balancing** (DeepSeek-V3) | Bias-term load balancing avoids routing collapse without gradient interference | A |
| **Delta-rule hybrids** (Gated DeltaNet / Kimi Linear 2510.26692; MesaNet ICLR'26) | Kimi Linear: 3:1 linear:full attention, −75% KV cache, up to 6× decode at 1M context (authors). MesaNet solves the in-context ridge regression *exactly per step* by conjugate gradient, trading test-time compute for quality | B |
| **Complex-valued transformers** (Phase-Coherent Transformer 2605.10123, later revision 2609.22415) | Gains require redesigned phase-aware attention; the revision admits some comparisons were equal-width not parameter-matched, and by multiplication count the complex model costs ~2× | C |
| **Attention as VSA binding** (2512.14709) | Reads attention + residual stream as an approximate vector-symbolic architecture with bundling/binding | C |

---

## 3. Design principles distilled

1. **Separate address from content.** Addressing (keys, routing, novelty) must live in a space that changes slowly or not at all; content (expert values, memories) can change fast.
2. **Statistics, not samples.** Everything you need to protect old knowledge can be a fixed-size statistic: per-layer input covariance (null-space), per-slot usage counts (TF-IDF specificity), per-domain Gaussians over router inputs (StaR-MoE), class means (LDA). Cost is O(L·d²) + O(N), *independent of how much data you've seen*. That is the answer to "replay is too expensive at scale".
3. **Content-addressed, sparse, self-routing memory** rather than a shared learned softmax router (no shared router weights to drift; rank-stable top-k).
4. **Calibrated abstention.** A system that can say "no expert owns this" can grow, think longer, or defer — instead of emitting confident garbage.
5. **Thinking = energy descent with deep supervision and a halting rule**, with the same energy reused as the novelty/uncertainty signal.
6. **Multi-timescale learning**: fast exact solves → sparse medium-speed slots → rare, constrained slow updates, with consolidation by self-generated latents under a fixed compute budget.
7. **Measure honestly**: header-free domains, mixed streams, real byte corpora, bits-per-byte, and a per-component forgetting attribution.

---

## 4. Proposed v4 architecture (sketch)

```
bytes ──► dynamic chunker (H-Net style, learned boundaries) ─┐
                                                              ▼
        ┌─ ADDRESS PATH (slow / frozen after foundation phase) ──────────────┐
        │  encoder_A  (= JEPA EMA teacher, SIGReg-regularised, isotropic)    │
        │  → addresses a;  stats: Σ_d (mean, cov) per domain, layer cov C_l  │
        └──────────────────────────────────┬─────────────────────────────────┘
                                           │  keys·a  (content-addressed top-k)
   ┌───────────────────────────────────────▼──────────────────────────────┐
   │ CONTENT PATH: sparse memory-experts (PEER/MoRAM-style rank-1 atoms)   │
   │  • old atoms frozen; new atoms: keys + values in null-space of C_l    │
   │  • updates only to TF-IDF-specific slots, SGD, projected to null-space │
   │  • hashed byte n-gram table (Engram) for static knowledge lane        │
   └───────────────────────────────────────┬──────────────────────────────┘
                                           ▼
   THINK LOOP (per-token halting): state (y,z); ŷ0 re-injected; energy E(x,y,z)
     steps: z ← z − η∇E / TRM-style update;  deep supervision; 1-step implicit grad
     E also = novelty/uncertainty → {more steps | spawn atoms | abstain}
                                           ▼
   READOUT: neural head  ⊕  exact ridge/LDA fast weights (per-context, MesaNet-like)
            calibrated: temperature + abstain; no −1e9 logits
   WORLD MODEL (later): z_{t+1} = f(z_t, action) trained with JEPA/SIGReg;
            plan by MPC/CEM in latent space scored by E.
   SLEEP (budgeted): sample latents from stored Gaussians / world-model rollouts,
            on-policy self-distil fast→slow; never touches raw data.
```

---

## 5. Out-of-the-box ideas (my synthesis — all [H])

**H1. One statistic, two guarantees (null-space keys *and* values).** Store per-layer input covariance C_l. Initialise a new atom's key in null(C_l) so old inputs score ≈0 on it (it cannot steal them — fixes D2 by construction), and project its value update off range(C_l) (cannot change old outputs). MoRAM's top-k over *unnormalised* key scores is already rank-stable for old atoms; H1 additionally stops new atoms from entering the top-k for old inputs. *Implementation note:* project the **update after the optimiser**, not the gradient before it — Adam's per-parameter scaling leaves the null-space, which is consistent with the SMF paper's AdamW warning. *Risk:* null-space capacity; at d=32 it will be exhausted trivially — test at d≥128.

**H2. Reuse the JEPA EMA teacher as the frozen address encoder.** You already pay for a slow copy of the encoder (`jepa.py`, decay 0.996). Freeze it after a foundation phase and use *it* — not the live encoder — for routing keys, SetGate exemplars and LDA features. Drift in D1 then cannot invalidate stored statistics.

**H3. SIGReg latents + closed-form novelty.** If latents are isotropic Gaussian (LeJEPA), (a) shared-covariance LDA is nearly the Bayes rule, (b) a Hotelling-T² / Mahalanobis test gives an analytic p-value, replacing SetGate's 60–100 CPU permutations with one GPU matmul. Same question (is this batch exchangeable with the expert's reference?), O(d²), no host sync, and usable per-token rather than per-batch.

**H4. Energy as the single control signal.** One learned scalar E(x,z) drives (i) per-token recursion depth (halt when ΔE small), (ii) spawn/grow (E high for all experts), (iii) abstention. Today these are three unrelated mechanisms (manifold error, entropy thresholds, permutation test).

**H5. Latent dreaming instead of data replay.** Sleep samples from stored per-domain Gaussians over *latents* (d numbers, not 16-byte sequences, not decoder rollouts) and distils fast→slow. Compute is a fixed budget per consolidation, not proportional to history. Pseudo-rehearsal from class-conditional Gaussians is known in prototype CL; using it to anchor routers is StaR-MoE; combining it with null-space updates and an energy-gated schedule is the new part.

**H6. Fly front-end on the readout.** Sparse random expansion + k-WTA before the LDA head decorrelates class means (smaller off-diagonal scatter), improving conditioning and cutting interference at near-zero cost.

**H7. Give complex numbers a job: binding.** Unit-modulus complex vectors are the FHRR model of vector-symbolic architectures: element-wise multiply *binds* role⊗filler, conjugate *unbinds*. If you keep complex arithmetic, make phase carry compositional structure and test it on compositional-generalisation tasks; otherwise a real-valued ablation at equal FLOPs may win and save 2–4× compute. Either outcome is informative.

**H8. Hashed byte n-grams as the "memory lane" for a tokenizer-free model.** BLT-style rolling-hash n-gram embeddings (and Engram) give O(1) static-knowledge lookup; each entry is touched only by inputs containing that n-gram, so updates are sparse and low-interference by construction — a continual-learning store that needs no router.

---

## 6. Efficiency plan (independent of the research bets)
1. Make `top_k` the default; replace the Python expert loop with grouped GEMM; remove the per-layer `.tolist()` sync.
2. Replace SetGate permutation tests with H3.
3. Swap the fixed `chunk_size=4` for learned boundaries (H-Net); gains are largest exactly on heterogeneous input (code, non-Latin scripts, DNA).
4. Real-vs-complex ablation at equal FLOPs *before* anything else is built on complex ops.
5. Long streams: delta-rule hybrid mixer (Gated DeltaNet / Kimi-style 3:1) with exact fast-weight solves.
6. Per-token recursion depth (MoR) once the think loop is trained with deep supervision.

---

## 7. Roadmap: pre-registered, falsifiable, cheapest first

Benchmark ladder (each rung must be passed before claiming the next):
**B0** existing toy → **B1** header-free domains (routing must use content) → **B2** *mixed-domain batches* (no task-ID) → **B3** real byte corpora in sequence (e.g. English prose, source code, non-Latin text, DNA/binary), metric bits-per-byte per domain over time → **B4** algorithmic reasoning (Sudoku/maze/ARC-lite, OOD depth) → **B5** synthetic action-conditioned worlds.

| # | Experiment | Decides | Falsifier / pass criterion | Cost |
|---|---|---|---|---|
| E0a | **Forgetting attribution** on B0: freeze everything except new expert; then unfreeze encoder / attention / decoder / router one at a time | Is D1 or D2 the dominant cause? | If retention ≈ 1.0 with only experts trainable, D1/D2 are the whole story | hours |
| E0b | **Oracle routing**: force the true domain expert vs learned routing | Routing vs representation | If oracle routing still forgets → D1/D3 dominate | hours |
| E1 | Frozen EMA address encoder + content-addressed atoms (no softmax router) + calibrated abstention, **zero replay, no task-ID** | H2 + MoRAM-style routing | Avg acc ≥ 0.70 and A-retention ≥ 0.75 on B2 over 20 seeds (your own F1/F2), paired vs dense no-replay | days |
| E2 | Hotelling-T² vs permutation SetGate | H3 | Same or better known/novel AUROC, ≥10× faster, per-token capable | day |
| E3 | Null-space keys + values (d ≥ 128) | H1 | Retention ≥ 0.9 at ≤ 3 domains; capacity curve vs #domains reported honestly | days |
| E4 | LwF-style KL on current data (RL's-Razor proxy) as a cheap add-on | whether free forgetting meter/regulariser helps | Reduces BWT ≥ 25% vs E1 without hurting new-domain acc | day |
| E5 | **Think loop**: current simulation loop vs TRM-style deep-supervised recursion vs Attractor-style fixed point, on B4 | Is thinking learnable at your scale? | Accuracy increases monotonically with steps and generalises to deeper OOD instances; old loop fails this | week |
| E6 | Energy head as halting/novelty signal | H4 | Energy-halted compute < fixed-depth compute at equal accuracy | week |
| E7 | Real-vs-complex at equal FLOPs, incl. binding task | H7 | Complex wins on compositional tasks at equal FLOPs, or is dropped | days |
| E8 | B3 byte-corpora stream, 3–5 domains, BPB with and without each component | Whether any of this survives contact with real data | Retention in BPB within X% of joint training; zero replay | weeks |
| E9 | Latent dreaming budget sweep (H5) | Whether sleep adds retention beyond E1–E3 | Retention gain > cost of dreams at 2 budgets | week |

---

## 8. What could make this wrong
- **Scale.** Everything you have is d_model = 32–64 on 3 toy domains. Null-space capacity, the value of frozen foundations, and sleep all behave differently at LLM scale in either direction.
- **The strongest replay-free results above are frozen-pretrained-backbone results** (RanPAC, ACIL, StaR-MoE on a ViT-B/16). Your model must first *earn* a good frozen feature space; E8 tests whether a small byte model can.
- **Most 2026 items are single-group preprints** (Attractor, StaR-MoE, Sleep, PCT). I found no independent replications. Treat their numbers as upper bounds.
- **Two source/blog discrepancies worth remembering:** Nested Learning blog ("completely avoid" forgetting) vs paper ("not solved in general"); Sleep blog/body ("not simply replay raw data") vs abstract ("with replay").
- **My reading depth.** PDFs were unreadable to the fetch tool; large parts of several papers came from HTML or model-extracted summaries (flagged in the reports). Figures-only numbers (class-incremental, continual translation) were not recoverable. Engram was read only via secondary sources.
- **Plasticity.** Freezing the address space trades away adaptation. Mitigation: a foundation phase that is broad, a slow-update path, and utility-based reinitialisation (continual backprop).

---

## 9. Decisions (answered by the user, 2026-11-05)
**Answers:** (1) **Always plastic** — no permanently frozen parameters. (2) **Both** task-ID-aware and task-ID-free regimes, so capabilities are known in each. (3) **Use complex first, then ablate against real** and decide from a significance test. (4) **Continual learning and thinking first, then efficiency overall.** Consequences are worked out in `2026-11-05-dmoe-v4-phase-a-plastic-taskfree-cl-and-thinking-design.md`. Original questions kept below for the record.
1. **Frozen foundation vs always-plastic.** Is a broad pretraining phase followed by a frozen/slow address encoder acceptable, or must every parameter stay trainable forever? Almost every replay-free result in the literature depends on this.
2. **Task-ID-free?** Should success mean mixed streams with no domain labels (B2), or is batch-level domain identity acceptable?
3. **Complex-valued core:** keep (and justify via binding, H7) or ablate against real?
4. **First target:** continual learning (E0–E4), thinking (E5–E6), or efficiency (§6)?

---

## 10. Sources
Continual learning: [Nested Learning](https://arxiv.org/abs/2512.24695) · [Google blog](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) · [Sleep paper](https://arxiv.org/abs/2606.03979) · [Sleep blog](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/) · [Sparse Memory Finetuning](https://arxiv.org/abs/2510.15103) · [StaR-MoE](https://arxiv.org/abs/2605.17571) · [MoRAM](https://arxiv.org/abs/2506.21035) · [LEGO](https://icml.cc/virtual/2026/67405) · [RL's Razor](https://arxiv.org/abs/2509.04259) · [AlphaEdit](https://arxiv.org/abs/2410.02355) · [LDC](https://arxiv.org/abs/2407.08536) · [RanPAC](https://arxiv.org/abs/2307.02251) · [DS-AL](https://arxiv.org/abs/2403.17503) · [GACL](https://arxiv.org/abs/2403.15706) · [Analytic Subspace Routing](https://arxiv.org/abs/2503.13575) · [Fly-CL](https://arxiv.org/abs/2510.16877) · [Loss of plasticity (Nature 2024)](https://doi.org/10.1038/s41586-024-07711-7) · [Titans](https://arxiv.org/abs/2501.00663) · [ATLAS](https://arxiv.org/abs/2505.23735)
Latent reasoning / world models: [TRM](https://arxiv.org/abs/2510.04871) · [Attractor Models](https://arxiv.org/abs/2605.12466) · [Energy-Based Transformers](https://arxiv.org/abs/2507.02092) · [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) · [Depth-recurrent analysis](https://arxiv.org/abs/2507.02199) · [Continuous Thought Machines](https://arxiv.org/abs/2505.05522) · [Dreamer 4](https://arxiv.org/abs/2509.24527) · [V-JEPA 2](https://ai.meta.com/vjepa/) · [LeJEPA](https://arxiv.org/abs/2511.08544) · [LLM-JEPA](https://arxiv.org/abs/2509.14252) · [Large Concept Models](https://arxiv.org/abs/2412.08821) · [ARC Prize 2025 report](https://arxiv.org/abs/2601.10904)
Efficiency / architecture: [H-Net](https://arxiv.org/abs/2507.07955) · [PEER](https://arxiv.org/abs/2407.04153) · [DeepSeek-V3](https://arxiv.org/abs/2412.19437) · [DeepSeek Engram (repo)](https://github.com/deepseek-ai/Engram) · [MesaNet](https://arxiv.org/abs/2506.05233) · [Kimi Linear](https://arxiv.org/abs/2510.26692) · [Phase-Coherent Transformer](https://arxiv.org/abs/2605.10123) · [PCT revision](https://arxiv.org/abs/2609.22415) · [Attention as Binding](https://arxiv.org/abs/2512.14709)
