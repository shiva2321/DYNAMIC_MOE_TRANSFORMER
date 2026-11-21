# Training dMoE for meaning and generalization — research synthesis and idea map

Repository: `D:\Dynamic_MOE_Transformer` · Branch `v5-arch-efficiency` · Date: 2026-11-20
Status: **research and brainstorming only. Nothing in `dmoe/` or `experiments/` was changed.** One read-only diagnostic script was run from the session scratchpad (§1.1); it is not in the repo.

**Evidence tiers** (same convention as `2026-11-05-dmoe-v4-research-synthesis-and-roadmap.md`)
- **[V]** Verified by me on this repo today (code read, or a run).
- **[A]** Established / replicated. **[B]** Peer-reviewed, single group. **[C]** Preprint, no replication found.
- **[H]** My hypothesis. Each has a falsifier in §5.
- Literature items below were found with web search and read at **abstract / snippet level only**. I did not read the full papers. Treat numbers as the authors' own.

---

## 0. The short version

1. **The training signal, not the architecture, is the weak link for "understanding".** The v7 core already drops next-byte loss, which is the right instinct. But what replaces it is narrow: a small hand-written paraphrase/flip rule list, one 77-intent banking dataset, and a world-model target that is a shallow byte-embedding. [V]
2. **Part of the stack is currently untrained under the v7 loss.** In a one-step gradient audit, the Thinker (563K params), the decoder, the abstain head and the imagination projection received **zero gradient**. Any "Thinker vs Think0" comparison in F4/F5 would compare a random-init module to nothing. [V]
3. **Stability was bought with plasticity.** In `real_pilot_v1` (3 seeds), Cog_SPGC cut forgetting from 52% to 7.9% but finished with *worse* bits-per-byte (2.98) than the plain no-replay dense model (2.84). [V]
4. **The strongest ideas share one shape: give every learned component a verifiable job.** Concretely: (a) latent prediction with deep targets, (b) a large, verified family of meaning-preserving and meaning-changing edits, (c) interchange-intervention training so an expert *is* a variable, (d) episodic meta-learning so rules are induced in context, (e) route by *function*, not by domain.
5. **One decision is yours** (§6): may a pretrained model act as a *teacher during training* (never in the final stack)? At ~5M params and ~4 MB of training bytes, real-text meaning will not emerge from the stream alone.

---

## 1. What your code and data say today (diagnosis)

### 1.1 Under the v7 loss, whole modules get no gradient [V]
`compute_v7_non_token_loss` (`dmoe/real_model.py:1345`) sums: world-model loss, SIGReg on `z_expr_pool`, equivalence on `z_expr_pool`, contradiction on `z_ctx_pool` (+ a `need` BCE), router balance. All of those are computed **before** the Thinker runs (`:1235` world-model loss, `:1265-1266` pools, `:1253-1260` Thinker). The decoder is detached (`detach_decoder: true`), so the Thinker output only feeds a detached decoder.

Audit (scratch script, `smoke_v7` config: d_model 192, 6 layers, 1 expert, random ASCII bytes, Track-2 loss weights, one backward):

| Module | Params | Tensors with no gradient |
|---|---|---|
| `thinker` | 563,282 | 35 of 35 |
| `decoder` | 217,472 | 6 of 6 |
| `wm_imagine_proj` / `wm_imagine_gate` | 37,056 / 1 | all |
| `abstain_head`, `concept_quantizer` | 193 / 9,216 | all |
| `ms_chunker`, `layers`, `world_model`, `context_router`, `memory`, heads | — | receive gradient |

Caveats: init state, one step, one expert (so router balance loss is exactly 0). Layers 3 and 5 each have 8 gradient-free tensors that I did not investigate (likely idle single-expert/adapter pieces). The audit does not cover Track 1, whose loss adds a classification term.
**Why it matters:** pre-registered falsifiers F4 and F5 (`results/real_pilot_v4/PREREG.md`) claim Thinker-driven gains.

### 1.2 The "meaning" signal is narrow and templated [V code-read; shortcut risk is H]
Per 16-sequence batch (`experiments/run_v7_cognitive_pilot.py:341`): 3 paraphrase pairs, 2 "nuance flip" pairs, 1 same-intent + 2 different-intent Banking77 pairs.
- **Paraphrase** = first matching entry of a 12-rule list (`" is "→"'s "`, `" + "→" plus "`, …) applied once, in the second half of the window; fallbacks swap `", "→"; "` or `" the "→" a "`.
- **Flip** = first match of a ~14-rule list (`" is "→" is not "`, `" >= "→" < "`, …) or `+7` on a number.
- Because the first match wins, in English prose `" is "` likely dominates *both* lists. A model can then separate "same" from "flipped" by spotting `" is not "` versus `"'s "`. That is a lookup, not meaning. (A rule-hit histogram would confirm or kill this in minutes.)
- Some paraphrase rules are not meaning-preserving in every domain (`" == "→" is "` is false in Python).
- The only real semantic variety is Banking77: **one domain, 77 fixed intents**, and it is also the headline metric (F1). The test split is disjoint, but it measures intent clustering inside one distribution.
- Credit: the overlap-trap metric is well designed. The bag-of-bytes null scores AUC 0.25 on it (smoke run), so it is adversarial against lexical shortcuts by construction.

### 1.3 The world-model target is surface-level [V]
`LatentWorldModel` targets are an EMA copy of the **input encoder only** (`byte_embed`, `chunk_proj`, `pos_embed`, chunker; `dmoe/world_model.py:6-8, 121-133`). It predicts the next chunk's *embedding of raw bytes*. That is next-chunk prediction in continuous space, not prediction of abstract content. I-JEPA, V-JEPA and LLM-JEPA predict **deep encoder outputs** [B].

### 1.4 Routing has no meaning-aligned signal [V code-read]
Routing is key-prototype similarity (`routing.py:801-846`); the only training terms on it are `0.05*load_bal + 0.25*rate_reg` (`:909`), weighted 0.5. Nothing asks "experts should differ by *what they compute*". Earlier pilots report routing NMI ≈ 0.47-0.54 against **domain** labels, which is the easiest kind of "meaning". 2026 MoE work reports that balance losses homogenize routers (§3C).

### 1.5 Stability–plasticity and cost [V, `real_pilot_v1`, 3 seeds, 400 steps/domain]

| Arm | Final BPB ↓ | Forgetting | Train s |
|---|---|---|---|
| Dense_Joint (ceiling) | 2.165 | — | 48 |
| **Cog_Mixed** | **2.071** | — | 190 |
| Dense_Replay | 2.369 | 14.3% | 50 |
| Cog_Full | 2.596 | 31.2% | 256 |
| Dense_NoReplay | 2.840 | 52.2% | 50 |
| **Cog_SPGC** | **2.977** | **7.9%** | 275 |

SPGC buys retention by learning less (intransigence). Cog_Mixed beating Dense_Joint is the one place the architecture shows a plain learning advantage; it deserves its own look. The Cog stack is ~5× slower than dense at equal parameters.

### 1.6 Scale [V]
~5.3M parameters, 4,096,000 training bytes total, BPB 2-3. This is a pre-semantic regime. Whatever looks like meaning here is mostly *supplied by labels or templates*.

### 1.7 What is solid
`SPGC` passes its six pre-registered falsifiers on 20 test seeds on the synthetic compositional belief task (`results/belief_training_v3/results_v1.json`, V1-V6 all true). That task has a real generalization axis (64 held-out concept pairs of 256). It is the best current testbed for "understanding" and should anchor the next experiments.

---

## 2. What "understanding" should mean operationally

Define it by five testable properties, then ask which training signal buys each:

| # | Property | Test |
|---|---|---|
| U1 | **Invariance** to meaning-preserving surface change | same-meaning AUC on edit families *not seen in training* |
| U2 | **Sensitivity** to minimal meaning-changing edits | flip AUC with lexical-overlap-matched negatives |
| U3 | **Systematic composition** | accuracy on unseen combinations of known parts, in a new rendering |
| U4 | **Causal use** | swapping an internal variable changes the output as the abstract model predicts |
| U5 | **Fast adaptation** | few-shot learning curve on a new domain/binding, with and without weight updates |

Next-token loss rewards these only indirectly. Pure non-token losses reward them only as far as the *data and transformations* encode them, which is why §1.2 matters.

---

## 3. Literature map (new since the 2026-11-05 roadmap; abstract-level)

### A. Objectives beyond next-token prediction
| Idea | What it says | Fit | Tier |
|---|---|---|---|
| **Concept-level objectives** ([2601.11791](https://arxiv.org/abs/2601.11791), EACL'26) | Group surface forms into one concept; reports lower perplexity, better domain-shift robustness | Direct match to equivalence learning, with a concept inventory | B |
| **Future Summary Prediction** ([2510.14751](https://arxiv.org/abs/2510.14751), ICLR'26) | Auxiliary head predicts a *summary* of the long-range future (bag-of-words or reverse-LM embedding); beats NTP and multi-token prediction at 3B/8B | Cheap, label-free way to push latents toward long-horizon content | B |
| **LLM-JEPA** ([2509.14252](https://arxiv.org/abs/2509.14252)) | JEPA term *beside* token loss helps fine-tuning; needs paired views | Supports JEPA + a grounding signal, not JEPA alone | B |
| **RLP** ([2510.01265](https://arxiv.org/abs/2510.01265), ICLR'26) | Reward "thinking" by the log-likelihood gain it gives over a no-think EMA baseline; dense, verifier-free | A principled way to train the Thinker's steps/halting | B |
| **Othello-GPT** ([2210.13382](https://arxiv.org/abs/2210.13382)) | Sequence models trained on a synthetic task develop board-state representations that are causally used | Template for U4 probes and interventions; synthetic only | A/B |

### B. Training that induces systematic generalization
| Idea | What it says | Tier |
|---|---|---|
| **Interchange-intervention training** ([2112.00826](https://arxiv.org/abs/2112.00826), [2106.02997](https://arxiv.org/abs/2106.02997)) | Train so that swapping internal activations between inputs changes outputs as a chosen high-level causal model predicts; zero loss ⇒ the causal model is an abstraction of the net | B (2021-22, small benchmarks) |
| **Meta-learning for compositionality** ([Nature 2023](https://doi.org/10.1038/s41586-023-06668-3)) | Train on *episodes* with re-randomized symbol meanings; matched human-like systematic generalization. Contested: [2506.01820](https://arxiv.org/abs/2506.01820) | B, disputed |
| **Formal-language / procedural pre-pretraining** ([2502.19249](https://arxiv.org/abs/2502.19249), [2505.22308](https://arxiv.org/abs/2505.22308), [2608.03930](https://arxiv.org/abs/2608.03930)) | Hierarchical formal languages before text; reports ~33% fewer tokens for equal loss at 1B and reuse of attention heads | B/C |
| **Data factors** ([2311.04420](https://arxiv.org/abs/2311.04420), [2507.07102](https://arxiv.org/abs/2507.07102)) | Diversity of primitives and combinations beats raw volume | B |
| **Complexity control** ([2501.08537](https://arxiv.org/abs/2501.08537)) | Small init + large weight decay helps learn compositional primitives (note: your SPGC already uses wd 0.2-1.5) | B/C |
| **ACL'26 Findings** (snippet only) | Atomic + compositional supervision can establish a generalization circuit *without* a long grokking phase | C |

### C. MoE training (specialization, routing, dynamic k)
| Idea | What it says | Tier |
|---|---|---|
| **Load-balance loss harms specialization** ([2604.09780](https://arxiv.org/abs/2604.09780), [2605.12476](https://arxiv.org/abs/2605.12476)) | Balancing suppresses shared directions / makes router vectors ~3× more similar | C |
| **Specialization & coupling losses** ([2505.22323](https://arxiv.org/abs/2505.22323), [2512.23447](https://arxiv.org/abs/2512.23447)) | Orthogonality + variance loss; expert-router coupling (ERC) loss; coupling appears more important than orthogonality | C |
| **Loss-free balancing** (DeepSeek-V3, in the 11-05 doc) | Bias term in dispatch instead of an aux loss | A |
| **Depth scaling + Muon → balanced experts** ([ICML'26](https://icml.cc/virtual/2026/72451)) | Init/optimizer choices fix routing collapse early | B |
| **Dynamic k** ([AdaMoE 2406.13233](https://arxiv.org/abs/2406.13233), [Top-p 2403.07652](https://arxiv.org/abs/2403.07652), [Expert Threshold 2603.11535](https://arxiv.org/abs/2603.11535)) | Null experts and thresholds give per-token compute; training signals are mostly balance/rate losses | B/C |

### D. Optimizer reliability
- **Muon speeds grokking** ([2504.16041](https://arxiv.org/abs/2504.16041)) [B], but a 2026 preprint reports Muon-trained models *lose* generalization after grokking at the representation-readout interface, and that freezing embedding/readout prevented it ([2608.07436](https://arxiv.org/abs/2608.07436)) [C]. SPGC uses exactly that split (trunk on Muon, embeddings/heads on AdamW). Treat as a hazard to monitor, not a fact.

### E. Adaptive / continual training
- Meta-learned update rules ([1806.06928](https://arxiv.org/abs/1806.06928)) and Hebbian/gradient fast-weight plasticity in Transformers ([2510.21908](https://arxiv.org/abs/2510.21908)) [B/C]. Relevant to "learn how much to change", not hand-set `lr`/`prec`.

### F. Teacher signals for byte models
- Byte-level distillation ([2604.07466](https://arxiv.org/abs/2604.07466)) and approximate-likelihood matching ([2503.20083](https://arxiv.org/abs/2503.20083)) show cross-tokenizer transfer into byte students. I found **nothing** on distilling a *sentence encoder's similarity structure* into a byte model; the idea in §4-I9 would be new for you, not a replication.

---

## 4. Ideas (brainstorm), grouped by what they fix

### Group 1 — Make the signal real (cheap, certain)
**I1. Gradient-coverage gate [H→cheap].** Turn the §1.1 audit into a unit test: every enabled module must get non-zero gradient under the headline loss, or be declared inference-only. Catches the Thinker bug class permanently. Cost: hours.

**I2. Give the Thinker a latent job [H].** Replace byte-CE deep supervision with: (a) per-step loss on the **latent** equivalence/contradiction objective, (b) monotone-progress loss on that objective, (c) halting trained RLP-style: reward = improvement of a future-latent/summary prediction over a no-think EMA baseline. Falsifier: accuracy rises monotonically with steps and generalizes to deeper OOD instances; Think0 is beaten.

**I3. Deep-target JEPA + Future Summary head [H].** Predict the EMA/SIGReg *trunk* `z_expr` of the next chunks (multi-horizon), plus a pooled summary of the next 64 bytes (hashed byte-n-gram bag as the byte-level analogue of FSP's bag-of-words). Removes the surface-level target of §1.3. Label-free.

### Group 2 — Make "meaning" supervision large and honest
**I4. Verified edit-family library [H].** Replace the two rule lists with many families, each *verified* by execution or algebra:
- preserving: whitespace/format, number spelling, commutative reorder, alpha-renaming of variables (code), algebraic rewrites checked by a CAS, unit conversions;
- changing: constant edits, comparison/negation flips, off-by-one — confirmed by running the code or recomputing the answer.
Train on families A; test on held-out families B. Verification makes labels trustworthy and effectively unlimited, and the held-out family is the anti-shortcut control. Metric: leave-one-family-out U1/U2.

**I5. Causal Abstraction → interchange-intervention training on experts [H, strongest "understanding" idea].** On `BeliefMeaningTask` the high-level model is already known: `m = val(op,a,b)` → `attr = g(m, ctx)` → `prop = h(m)`. Assign experts (or expert subspaces) to variables (`m`, `ctx`) and train: swap an expert's output between two inputs and require the answer the causal model predicts. At zero loss the abstraction is guaranteed [B]. This makes *an expert = a variable*, which is the opposite of today's domain-specialist experts. Metric: interchange-intervention accuracy plus held-out-pair accuracy.

**I6. Episodic meta-learning for rule induction [H on MLC].** Re-randomize bindings (`phi`, `g`, `h`) **per episode**; give a few examples in context; ask for held-out queries. The only winning strategy is to induce the rule. Your LTM slots become the in-episode binding store; SPGC handles durable consolidation. Metric: accuracy on *new* bindings with zero weight updates, plus few-shot curve (U5). Risk: MLC claims are contested.

**I7. Procedural pre-pretraining of the trunk [H].** A small fraction of the budget on Dyck/hierarchical languages, copy, sort, counting, modular arithmetic before the text stream; pairs naturally with Muon's fast grokking. Metric: length-generalization on Dyck depth; BPB on the code/math streams at equal budget.

### Group 3 — Make the MoE adaptive in the right way
**I8. Route by function, not by domain [H].**
1. Swap the softmax load-balance term for **loss-free bias balancing**; add an **expert-router coupling** term (your keys are already prototypes, so key ≈ the expert's input signature is a natural fit).
2. Measure routing NMI against **operation/rule labels**, not domain labels; add expert-ablation effects per function family.
3. **Learned compute need:** train `need` with a *reducible-error* target (how much extra compute lowered the latent loss on this chunk) instead of only a target-rate regularizer. Falsifier: `k` correlates with measured difficulty; ablating expert j hurts only function j.

**I9. Teacher-geometry distillation (needs your decision, §6) [H].** Match the *pairwise similarity structure* (Gram matrix) of a pretrained sentence encoder on the same text windows, instead of copying its coordinates. Student stays always-plastic; teacher never in the inference stack. This is the only route I see to real-text semantics at your scale. Label results "teacher-informed".

### Group 4 — Make training reliable and adaptive
**I10. Consolidate understanding, not memorization [H].** Gate SPGC's Fisher/BONG precision by *meaning-consistency* (the example passes equivalence/intervention checks) rather than surprise alone, and make forward transfer (speed of learning a new domain) co-primary with forgetting. Falsifier: forgetting ≤ 15% **and** final BPB ≤ Dense_Replay (today's SPGC fails the second).

**I11. Muon guard rails [C-motivated].** Track a held-out probe *after* train-fit, keep a slow/decayed readout, and select checkpoints on held-out quality. Hours of work; covers the §3D hazard.

**I12. Meta-learned plasticity [H, later].** Learn per-group `lr`/precision from a held-out forward-transfer + retention objective (meta-continual learning), replacing hand-tuned `spgc_lr_*`/`prec_scale` constants.

---

## 5. Proposed shape and order

```
P0  procedural pre-pretrain (I7)                      — structural heads, ~cheap
P1  stream: deep-target JEPA + FSP (I3)               — world structure, label-free
      + verified edit families, held-out families (I4) — invariance / sensitivity
P2  episodic binding tasks (I6) with interchange       — rule induction + causal variables
      interventions on experts (I5)
Always: Thinker on latent progress/RLP reward (I2) · route-by-function (I8)
        SPGC consolidation gated by meaning-consistency (I10) · Muon guard rails (I11)
Gate:  gradient-coverage test (I1) before every run
```

**Order (cheapest decisive first, one new mechanism per experiment, matched controls: Dense+same losses, dictionary, bag-of-bytes):**
1. **Days:** I1, I11, rule-hit histogram for §1.2, leave-one-family-out evaluation on the *current* data (tells you how much of F1 is shortcut).
2. **~1-2 weeks:** I3, I4, I8(1-2), I2.
3. **~2-4 weeks:** I6 + I5 on `BeliefMeaningTask` (the best generalization testbed you have), then I7.
4. **Later / decision-dependent:** I9, I10, I12.

**Falsifier summary**
| Idea | Killed if… |
|---|---|
| I1 | audit finds all modules covered (then §1.1 was a config artifact) |
| I3 | linear-probe/retrieval no better than the shallow-target baseline |
| I4 | leave-one-family-out AUC collapses to the bag-of-bytes null |
| I5 | interchange accuracy ≤ matched multitask training, or held-out pairs not better |
| I6 | new-binding accuracy with zero updates ≤ chance-plus-dictionary |
| I8 | routing NMI vs function ≤ vs domain; expert ablations not function-specific |
| I10 | cannot reach forgetting ≤ 15% with BPB ≤ Dense_Replay |

---

## 6. Open decision

**May a pretrained model be a teacher during training (I9), with the final model standing alone and always plastic?**
- **From scratch only:** clean claim; semantics stay limited to synthetic compositional worlds (I5/I6 are fully compatible).
- **Teacher-informed:** the only route to real-text meaning at ~4 MB of training bytes; results must be labeled teacher-informed; it relaxes the "no frozen pretrained encoder" purity in `NEXT_PATH.md` §0 at the training level only.
- My pick: **both tracks** — from-scratch on the synthetic compositional track, teacher-informed on the real-text track, reported separately.

## 7. What could make this wrong
- **Scale.** Everything above is 5M params and ~4 MB. Several cited gains were at 1B-8B.
- **My audit is narrow:** one step, at init, one expert, Track-2 loss. The Thinker may receive gradient in other arms or later in training.
- **Single-group, abstract-level evidence.** Most 2026 items are preprints and I did not read the papers. MLC is contested; IIT is 2021-22 on small benchmarks; the Muon-collapse result is five seeds on modular addition.
- **The shortcut claim in §1.2 is a hypothesis** until the rule-hit histogram and leave-one-family-out run exist.
- **No new training was run.** All proposed gains are untested on this repo.
