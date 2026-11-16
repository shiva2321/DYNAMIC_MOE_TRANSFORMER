# Research consolidation — design (DRAFT for approval)

Date: 2026-11-11 · Status: **APPROVED by the user (E: scope = "AI-research core"; archive = `D:\Research_Archive`) and EXECUTED the same day.** The body below is the design as approved (written before any copying); outcome: `D:\Research_Archive\VERIFICATION.md`, curated output: `research_base/`.
Deviations from the draft: (1) scope narrowed by the user's choice to E:\AI_Research, universal_substrait, unified, Neuron_v2, CONVOLUTION_RESEARCH (prime_research, Hive not included); (2) `E:\AI_Research\docs\research\findings.md` (379 KB, 88 entries) was found during the build and added to the evidence ledger (X-41..X-48) and cards (M22); (3) `.env.example` treated as a template (copied), `.env` withheld.
Author: Claude (designer). Decisions marked **[my call]** are mine and are listed in §9 for you to check.

---

## 1. Intent and why

You have run several research threads over ~8 months (NSCK / Node_network → NSCK_demo → AI_Research/Concerto → universal_substrait → spherical_hyperspace_memory (SHM, GrowMoE) → this repo, dMoE v2–v4 + Track K). The knowledge is spread over many folders, partly duplicated, with claims of very different reliability (a graded SHM claims register next to AI-written "papers" whose own state file says Phase 2 results are mock).

**Goal:** (a) one place to start the next research phase from — only what is *proven working and useful*, with its evidence and its landmines — inside `D:\Dynamic_MOE_Transformer`; (b) everything else preserved, indexed and findable, in one archive; (c) an honest answer to "what is worth publishing"; (d) **proof that nothing was lost**.

## 2. Goals, non-goals, anti-goals

**Goals**
- G1. `research_base/` in this repo: a short, trustworthy "state of knowledge" + mechanism cards + evidence ledger + negative-results list + methodology kit + proven reference code. A new session can start from it without opening any other folder.
- G2. `D:\Research_Archive\`: a complete, indexed copy of every in-scope source's *knowledge* (code, docs, small results, papers, git history), with a master INDEX.
- G3. A manifest-based loss check: every file of every source is accounted for as copied / deliberately-not-copied (with reason) / duplicate-of (with the twin's path).
- G4. A publication assessment with reasons and evidence.

**Non-goals**
- Not rewriting or "fixing" any source code. Not merging git histories. Not wiring archived code into the `dmoe/` package (that is design work for the next phase). Not re-running multi-hour experiments.
- Not deleting, moving or modifying anything in the source folders — including the folders themselves.

**Anti-goals (would make this a failure even if it looks tidy)**
- A tidy tree that silently dropped a result, a negative result, a prereg, a script that produces a result, or uncommitted work.
- Promoting a claim to "proven" because a README says so. (Already caught in these repos: SHM's "+10 GSM8K" was string grading; NSCK Phase-2 scores are synthetic; the Track K "win" was a dictionary.)
- Copying secrets or personal data (`.env`, Hive's 68 MB conversation database) into a second location.

## 3. Sources and scope

| Source | Real content | On disk | Git | Uncommitted | In scope |
|---|---|---|---|---|---|
| `D:\Dynamic_MOE_Transformer` (hub) | 46 commits; `dmoe/` 19 modules, 20 test files, 10 eval scripts, 20-seed results + PREREGs, 13 specs/plans | 4.3 GB (4.2 GB `.venv`) | `v4`; remote DYNAMIC_MOE_TRANSFORMER | 1 file | **hub — stays and grows** |
| `D:\spherical_hyperspace_memory` (SHM) | ~170k lines Py (883 files), 7 Rust; 77-entry lab notebook, 207 graded claims, 115 failures; 1,927 tests | 4.0 GB (3.8 GB regenerable `.npz/.npy`) | `main`, `track-g-growmoe`; same GitHub remote as hub + `Try_2` | 24 files | yes |
| `D:\NSCK_demo` | 617 Py, 72 Rs, 61 md, 32 PDFs/papers; Rust crates (vsa, snn, societal, bridge) | 18.5 GB (14 GB LFS model weights in `.git`; 1.9 GB Rust `target/`; 1.1 GB knowledgepacks) | `main`; remote TANZLAB/NSCK_Demo | 171 files | yes |
| `D:\Node_network` | 648 Py, 46 Rs, 107 md; 10 draft papers + experiments + results JSON; 481 commits, ~75 branches (NSCK_V1…V4, NGCN, NDGN, copilot/*) | 6.5 GB (2.3 GB `.gguf`, 2.1 GB venvs, 1.5 GB Rust builds) | `NSCK_V4`; remote shiva2321/Node_network | 136 files | yes |
| `E:\AI_Research` (+ `E:\unified`, `E:\universal_substrait`, `E:\Neuron_v2`, `E:\CONVOLUTION_RESEARCH`) | Concerto, ground, sophia, PhasorUnitaryNet, Resonance_X, ResonantSpark, WaveLearn, VSA_GROUND_REBUILD, swarm, REXC… ; `universal_substrait` (paper drafts, pre-registered neurogenesis study); `unified/` package; `Neuron_v2` (research-integrity-kit) | AI_Research 66 GB but only ~25 MB code+docs; universal_substrait 18 GB (checkpoints) | various | not checked | **PROPOSED — needs your yes (Q1)** |
| `E:\prime_research`, `E:\universal-ai-memory` (Hive) | primes engine; TypeScript MCP memory graph product | 0.26 GB / 0.19 GB | — | — | borderline (Q1) |
| `E:\Price_signal_analysis`, `Financial_Data_Analysis_System`, `News_Analysis`, `TradingData`, `Omnipdf`, `assembly_toolkit`, `vhdl_toolkit` | trading / PDF / EDA / assembly products — not this research | — | — | — | **out** (indexed as "out of scope, left in place") |

Duplication found: `D:\NSCK_demo\python\core` is **byte-identical (273/273 files)** to `E:\AI_Research\NSCK_demo`; vs `D:\Node_network\nsck` 147 of 171 shared files identical (NSCK_demo has 102 more, Node_network 15 more). They are one lineage. The archive keeps one copy and records the twins.

## 4. What the analysis found (evidence snapshot — seeds the INDEX and the publication verdicts)

**Hub (dMoE).** v2 growth = dense (paired Δ −0.0005 ± 0.0023, n=20). v3 zero-replay passes its 5 falsifiers (0.953, retention 1.000) *with a frozen backbone, supplied boundaries, header-byte routing and a per-position lookup task*; NLL 2.2 vs 0.41 for replay (calibration failure). v4 mixed-stream passes 9/9 falsifiers (0.938, routing 0.984) but domains still carry header bytes in slot 0. **Track K fails 4 of 9 falsifiers** (K2 burst 0.605 vs 0.85; K1 retention 0.71; NLL; control isolation): dMoE = Dense+Upkeeper (Δ −0.016, t=−0.92); a plain dictionary with counters gets 0.95 vs ~0.55 for every neural arm; held-out-context accuracy ≈ chance for all (0.06 = 1/16). The project's own §14.2 audit says this. Tests: **125 pass, 1 fail** (the known JEPA probe test).

**SHM.** The best-governed evidence in the whole estate: 207 claims, 14 grade A (pre-registered, fresh seeds, reproduced), 173 B, 13 REFUTED (kept in place), pre-reg with dev/test seed separation, independent reviewer re-measuring. Solid: repaired pool beats best single net ≥5 pts on function streams (C-20 A); batch-vote training (C-23 A); fingerprint shortlist 6–11× gate saving at no accuracy loss (C-25 A); learned closed-form router +13.0 [10.5, 15.4] (C-84 A); transfer rule founds the true number of experts (C-85 A); gap-growth rescues undersized experts, 15/15 vs 6/15 (C-86 A); forgetting 55.1 → 15.2 → 7.1 (C-87 A); closed-form LDA head per expert is the largest single gain (C-123 B). Refuted/weak: pool vs replay "+19" (C-13), growth rarely fires (3/70 runs), merge removes ~nothing (C-68), OOD flag silent on real text (C-120), **pool ties kNN/replay on real text (C-112)**, pool-level exact removal leaks (E-076, open). Everything positive sits on a **frozen pretrained encoder**; its own "trained representation" milestone M5 is unstarted. ISHMN (the original system) is retracted by its own README. SHM's own verdict on itself: "modest technical report, not a method paper."

**NSCK_demo / Node_network.** Real engineering (Rust VSA/SNN/societal crates; 1,918 tests reported) with weak science: `AGENT_STATE.md` admits Phase-2 scores are 100% synthetic, external validation scores are hardcoded, ROME extraction is a stub. Transplant paper: Spearman ρ 0.98–0.99 is the standard random-projection/SimHash (Johnson–Lindenstrauss) property, not a finding; **its bit-flip robustness experiment reports recall@10 = 0.04 at every flip rate 0–30 %, i.e. it measures nothing**; the downstream synonym task is 10 facts / 40 queries, 65 % vs 25 % random; "analogy 78 %" is MiniLM's geometry. Papers 6–8 are self-designed pass/fail checks ("10/10 vetoed", "Sally-Anne passed"); Paper 10 (SoCL = EWC with a graph-centrality-scaled Fisher) was shown on digit tasks only. `FactGate`/`BeliefScorer` = the same prior-count rule the hub's Upkeeper has, with hand-set weights and no measured precision/recall.

**E: set (from the 2026-11-06 scouting report + my triage; not yet independently re-verified by me).** `universal_substrait`: its pre-registered neurogenesis H1 **failed** (p=0.39; replication p=0.99), V2 redistributes forgetting, 28.6 % slower, had a shared replay buffer in both arms. AI_Research/Concerto: honest kill-lists; best items are **surprise-priced escalation to an exact store (EXP-3b: 1.000 at 17.5 % escalation, +0.758 over random)** and **prefix-surprise calibration (EXP-4: 96 % vs 66.6 %)**, both synthetic.

**Cross-repo conclusion.** Nobody has shown continual learning on a *plastic, self-trained* representation. That is the open part and the natural "fresh path".

## 5. Design

Two destinations, one rule: **copy only, source folders are read-only to me.**

```
D:\Dynamic_MOE_Transformer\                  (existing repo; dmoe/ tests/ scripts/ results/ docs/ untouched)
  research_base\                             NEW — curated, vetted, small (<~50 MB), start here
    README.md  00_START_HERE.md              1-page state of knowledge: proven / refuted / open / next
    01_evidence_ledger\                      claim -> evidence -> grade -> where reproduced -> status  (SHM claims merged with hub results + E: items)
    02_mechanism_cards\                      one card per mechanism: idea, evidence, code pointer, verdict TAKE/HOLD/SKIP, how to test it next
    03_negative_results\                     refuted claims, failures, landmines — "do not redo"
    04_methodology_kit\                      prereg template, dev/test seed rules, mutation testing, audit recipe, labnb.py, evaluation traps list
    05_benchmarks_and_baselines\             streams/beds (T1-T3, Track K, SHM function/arith streams), baseline set + equalised-budget rules
    06_open_questions_and_next_path\         fresh-path agenda (X1–X8 + SHM Q-list + the open "plastic trained representation" gap)
    07_publication_assessment.md
    code_reference\<source>\...              proven reference code, UNMODIFIED copies with provenance header + the tests that cover it
  docs\superpowers\specs\2026-11-11-research-consolidation-design.md   (this file)

D:\Research_Archive\                         NEW — outside the repo so the GitHub remote stays lean
  INDEX.md  INDEX.json                       master index: every source -> what it is, dates, state, key claims, verdict, archive path, ORIGINAL path
  README.md                                  how to use; how to verify; what is NOT here and why
  MANIFEST\<source>.csv                      EVERY source file: path, size, mtime, sha256 (non-bulk), status (copied | bulk-not-copied | secret-not-copied | duplicate-of:<path>), reason
  MANIFEST\BULK_NOT_COPIED.md                bulk items with size, how to regenerate (venvs, Rust target/, checkpoints, weights, knowledgepacks, caches)
  VERIFICATION.md                            loss-check result + test re-runs from the archived copies
  01_spherical_hyperspace_memory\            working-tree copy (minus bulk) + repo.bundle (git bundle --all) + SOURCE_CARD.md + PROVENANCE.md
  02_NSCK_demo\   03_Node_network\   04_...  (same shape; E: sources if approved)
  99_out_of_scope\INDEX.md                   pointers to the unrelated E: products, left in place
  tools\build_archive.py  verify_archive.py  the scripts that did it (re-runnable, auditable)
```

**What is copied vs not (rule, applied by script, every decision logged in the manifest):**
- Copied: `.py .rs .toml .md .tex .json .jsonl .csv .txt .yml .sh .png .pdf` and everything else ≤ 25 MB that is not in a bulk class. Uncommitted and untracked work is included (the working tree is what is copied).
- Bulk-not-copied (listed with size and regeneration note; **originals stay where they are**): `.venv*`, `target/` (Rust builds), `__pycache__`, `.pytest_cache`, `.idea`, checkpoints (`.pt .ckpt`), model weights (`.gguf .bin .safetensors`), cached arrays (`.npz .npy` > 5 MB, SHM `_cache/`), knowledgepacks (`.kp`), `hf_cache`, duplicate worktree copies (`.claude/worktrees`).
- Never copied (listed as such): `.env` files, `*.sqlite*` memory-graph databases, anything that looks like a token.
- Git history: `git bundle create --all` per repo (every branch, including Node_network's ~75). Git-LFS objects (NSCK_demo's 14 GB of model weights) are not in a bundle; recorded in BULK.
- Duplicates: stored once; the manifest lists the twin's path and hash (e.g. NSCK_demo ↔ AI_Research\NSCK_demo).

**"Proven" bar for `research_base/` (my definition — applied to every item, with the reason written on its card):**
1. claim is grade A or B in its own ledger *with a control/baseline next to it*, or is a measured, falsifier-checked result in this repo; and
2. it is not refuted/superseded; and
3. code exists and its tests pass **when re-run from the archived copy** (else the card says UNVERIFIED, never "working").
Below the bar ≠ thrown away: it goes to HOLD (promising/unproven) or SKIP (refuted/landmine) with the evidence, and the files stay in the archive.

**Initial TAKE list** (each becomes a card; full verification happens in the build): closed-form LDA head per expert (C-123/C-131; hub `fast_memory.py`); batch-level conformal novelty gate + vMF signatures with its single-regime-batch breaker (C-12/15/32–39, C-38); fingerprint shortlist (C-25); learned closed-form router on ~16 prototypes/expert (C-57/64/84); transfer/spawn-or-extend rule incl. weak-parent rule (C-85, C-166/172); function-preserving widen/deepen (C-58) and gap-growth (C-86); exact removal for closed-form heads (E-076; pool-level = HOLD); evidence-accumulating key-value store with counters / Upkeeper (Track K); surprise-priced escalation and prefix-surprise calibration (Concerto; synthetic → HOLD until X1/X2); `DynamicWarmupAdamW` (X3 pending); `costlab` cost meter; the whole **methodology kit** (prereg, dev/test seeds, claims register, mutation testing, independent reviewer, `labnb.py`).
**Initial SKIP/landmines:** ISHMN/Riemannian analogy engine (retracted), hyperbolic heads (C-07), branch-and-bound retrieval (C-08), barycentric merge (C-68), neurogenesis-as-shield (H1 failed), Kuramoto/oscillator routing, TDA monitors, mental-rehearsal veto without a world model, "pool beats replay +19" (C-13), NSCK mock-scored phases, NSCK bit-flip experiment.

## 6. Verification plan (how "nothing lost" is proven)
1. `build_archive.py` walks each source and writes the manifest *before* copying; `verify_archive.py` re-walks the source afterwards and asserts: every file is `copied` (sha256 equal to the archive copy), `bulk-not-copied`, `secret-not-copied` or `duplicate-of`; **zero unaccounted files**; no file in the archive that is not in a source. Counts and bytes reported per source.
2. Source folders are hashed before and after (non-bulk files) → proves they were not modified.
3. `git bundle verify` + `git bundle list-heads` for each bundle; head list compared with `git for-each-ref` of the source.
4. Tests re-run **from the archived copies**: hub (125/126 today), SHM (`tests/` — environment = global Python 3.12.10 / torch 2.6.0+cu124, identical to the lab lock), NSCK Python tests where the env allows. Results in `VERIFICATION.md`; anything not run is labelled NOT RUN.
5. Link check on every INDEX/ledger/card path. Every claim ID cited exists in its ledger.

## 7. Publication assessment (preliminary; finalised at the end of the build)
- **Worth writing: a methodology / negative-results paper** ("how continual-learning benchmark conclusions failed under pre-registration — a case study across five rounds of self-audit"). Evidence exists now: SHM's 115 failures (training-budget artifact, metric true by construction, operating-point mismatch, under-trained baselines, double-counted parameters), the hub's Track K audit (dictionary beats neural; dMoE = Dense+Upkeeper; header oracles in v3/v4; calibration failure hidden by accuracy), universal_substrait's failed pre-registered H1. It is the only claim where the evidence is complete and the contribution does not depend on a win.
- **Maybe, after M1/M2-style work: a modest technical report on the SHM pool** — grade-A claims only (calibrated gate, shortlist, router, transfer rule, growth rescue). Not a method paper today: frozen encoder, own streams, ties kNN/replay on real text, counted-not-clocked efficiency, no standard benchmark, no independent replication. SHM's own assessment agrees.
- **Not publishable as is:** dMoE v3/v4 (toy 3-domain, d_model 32, header-routed); Track K as a "win"; NSCK papers 1–10 (random-projection transplant is textbook SimHash/JL; mock/self-designed evaluations; broken robustness experiment); ISHMN (retracted).
- Open door for a real contribution: the unsolved "continual learning on a plastic, self-trained representation, task-ID-free" regime — nobody in these repos has shown it; it is the fresh path.

## 8. Risks
- **Disk**: D: has 65.8 GB free; archive expected ≲ 2 GB (bulk excluded). E: scope (66 GB AI_Research) is also bulk-excluded → ≲ 1 GB code/docs.
- **Google Drive sync** is active on parts of E: (stages uploads in `E:\.tmp.driveupload`, mirrors deletions to trash). I only read from E:, never write there, so no mirroring risk.
- **Long paths / Windows**: copy script uses `\\?\` paths; failures are logged, not skipped silently.
- **My reading depth**: I read the ledgers/READMEs/results of the sources, not every file. The index will state, per source, what was read vs only inventoried.

## 9. Calls I made — please check
1. **Copy-only.** Nothing moved or deleted; you can delete originals yourself later using the manifest as the checklist (your rule: never delete code).
2. **Archive location** `D:\Research_Archive\` (sibling), not inside the repo.
3. **Git bundles** for history instead of copying `.git` dirs (compact, verifiable); LFS weights not bundled.
4. **Secrets/personal data not copied** (`.env`, Hive sqlite).
5. **No commits, no pushes, no branch switch**; new hub files stay uncommitted on `v4` until you say.
6. **"Proven" bar** as defined in §5, stricter than any single repo's own README.
7. **research_base code is unmodified copies**, not wired into `dmoe/`.

## 10. Open for you (Q1)
Which E:\ folders are in scope — see §3.
