# Scope charter: what each agent may do (read this before every round)

**Purpose.** Build individually *verified* components. The user will integrate only components whose status is VERIFIED. Nothing is integrated, renamed or "improved" on the side.
A component is VERIFIED when it has (1) code and tests, (2) a prediction written before the numbers, (3) >= 10 seeds, mean +- 95 % CI, a tuned and controlled baseline, (4) numbers pasted from a script-generated file, (5) its limits written down.

## 1. Component register (Claude updates it; status changes only after a review)
| Component | Where | Status | Evidence | Limits |
|---|---|---|---|---|
| Lorentz model, fused attention, norm-gated neuron (maths) | `geometry/lorentz.py attention.py neuron.py` | VERIFIED (identities, tests) | `tests/test_lorentz_attention.py` | no model-quality benefit shown |
| Hyperbolic tree embeddings, small d | `geometry/`, `research/yggdrasil_poc/poc_a4*` | VERIFIED (static tree) | poc_a4/a5 results | not shown inside a language model |
| Snapped rotation state (`vq_rot`, packaged as `geometry/snapped_state.py`) | `geometry/snapped_state.py`, `research/yggdrasil_poc/poc_b_state_tracking.py` | VERIFIED on Z6, S3 (exact to length 256); NOT on S5; **no advantage over a plain GRU shown** (the GRU is equally exact on Z6/S3; S5 GRU not measured) | `research/yggdrasil_results/poc_b_table.txt`, `tests/test_snapped_state.py` | needs K = 4*|G|; S5 at chance |
| Hyperbolic attention heads for LM | `yggformer/` | NOT SUPPORTED | `e6_lm_results.json` | no gain after controls |
| Tree (branch-and-bound) retrieval | `yggformer/tree_search.py` | NOT USEFUL at N <= 4096 | `e5_results.json` | 36-60x slower than dense |
| Fast pairwise-distance kernel | `rust_yggdrasil/`, `geometry/gpu_kernel.py` | PARTIAL (2.6x vs engine kernel; 4x target unmet) | `e7_results.json` | |
| GrowMoE Phase 1 (pools A, B, C, bench) | `growmoe/` | VERIFIED (R10b approved); **claims since restated: M0 showed 'A beats replay' was a training-budget artifact, see the next two rows** | `research/growmoe_results/phase1/RESULTS.md` | weak on mixed types (S1); Phase 2 in progress |
| GrowMoE PhyloMoE: repaired lifecycle (`arm_monitor_after_infancy`) | `growmoe/pools/phylo.py` | VERIFIED (M2-A, fresh seeds 10-19, 7 of 7) | `research/growmoe_results/m2a/RESULTS_M2A.md` | one synthetic stream family carries the accuracy claim; no efficiency gain (gate = 97 % of compute) |
| GrowMoE PhyloMoE: batch vote (`batch_vote`) | `growmoe/pools/phylo.py` | VERIFIED (RV2-A, fresh seeds 20-39, 11 of 11) | `research/growmoe_results/rv2a/RESULTS_RV2A.md`, card 14 | centre cosine >= 0.6 still merges; about 1.3 points lost on known regimes that later arrive mixed |
| GrowMoE PhyloMoE: fingerprint shortlist gate (routing v2 stage 2) | `growmoe/` (to be extended) | CANDIDATE (dev-seed spikes only) | `research/growmoe_results/rv2/RESULTS_RV2_SPIKES.md`, spec `docs/superpowers/specs/2026-10-10-routing-v2-design.md` | flag OFF until a pre-registered test passes |
| *Candidates from the 2026 geometry atlas (Track Z; all UNTESTED)* | `geomlab/` (to be created) | CANDIDATE | `research/GEOMETRY_ATLAS_2026-10.md` | see the atlas |
| SphericalStream (nGPT family) | Z1 | CANDIDATE | - | decides: unit-sphere residual stream |
| ConeMemory (recall per byte) incl. verification of the legacy ISHMN memory | Z2 | CANDIDATE | - | decides: memory geometry |
| OrthoState / SCOR (parallel orthogonal state + snapping) | Z3 | CANDIDATE | - | decides: exact state tracking in parallel |
| PhasePosition (PoPE/LieRE) | Z4 | CANDIDATE | - | decides: position geometry |
| HierIndex (spectral Euclid vs hyperbolic on a real hierarchy) | Z5 | CANDIDATE | - | decides whether hyperbolic stays |
| StepControl, QuasiMetricHead | Z6, Z7 | CANDIDATE | - | design memos first |
| ISHMN / Universal Cognitive Engine | `ishmn/` | UNVERIFIED (headline claims failed audit) | memory notes of 2026-09-21 | **frozen: do not touch, do not integrate** |

## 2. Scope fences
* **Agent G (GrowMoE)** may edit only `growmoe/`, `tests/growmoe/`, `research/growmoe_results/`, `scratch/diagnose_*`. **Agent Y (Yggdrasil)** may edit only `geometry/`, `rust_yggdrasil/`, `yggformer/`, `tests/yggformer/`, the geometry tests, `research/yggdrasil_*`, `research/yggformer_results/`, `docs/COMPONENT_CARDS.md`.
* **Agent Z (Track Z, geometry atlas)** may create/edit only `geomlab/`, `tests/geomlab/` (no `__init__.py`), `research/geomlab_results/`, and append cards to `docs/COMPONENT_CARDS.md`; it may import `geometry/`, `yggformer/`, `growmoe/`, `ishmn/` read-only (e.g. the legacy memory as a Z2 baseline). Track Z starts only when the user says "start track Z".
* **Nobody** edits `ishmn/`, `crates/`, `quarantine/`, `artifacts/`, `experiments/`. Shared files (`pyproject.toml`, `pytest.ini`, `GEMINI.md`, this charter) only with a note in the report. Changes to shared contracts (`growmoe/types.py`, `pools/base.py`, `config.py`, `geometry/__init__.py`) must be listed with a reason.

## 3. Drift rules (each one is a stop-and-ask trigger)
1. One round at a time; implement only what the round says. Anything else goes under "Questions for the mediator".
2. Do not change code after producing results unless you re-run, or prove the change is a no-op (a regression test) and say so.
3. Numbers in a report are **pasted from a script-generated file, never retyped, and result files are never edited by hand (Claude re-runs the benchmark and diffs; a hand-edited result is rejected)**; prose may not claim more than the table shows; a failed prediction is reported as failed; "not measurable" is not "fail", and a conclusion is never written before the data.
4. No new top-level folders or packages, no moving or deleting files, no deleting results, no commits, no installs, no downloads.
5. A closed track stays closed. A hypothesis is labelled a hypothesis. Memory-lean: one GPU job at a time, report peak memory.
6. **A pre-registered check may not be replaced or removed.** If it fails, report it as failed, find and fix the cause, and document the history in `AMENDMENT.md`; PREREG is read-only once results exist.
7. **Integration is forbidden** until the register says VERIFIED and the user says "integrate".
8. **A comparison that justifies a change runs at the benchmark's own configuration** (the same config object, preferably the same code path); a script that re-implements the evaluation with other parameters is not evidence (R11c: S2-S4 were compared at q = 0.05 while the benchmark uses 0.02). State the configuration in the table caption.
9. **Reproducibility:** every random source is seeded from the experiment seed (no unseeded global RNG, no `hash()`); a result row that two clean runs do not reproduce is a bug (R11c: `RndSignature` created a generator and never used it), and new components get a determinism test.
10. **A failing existing test is information, not an obstacle.** Report what it protects and what the new behaviour does to it; changing a default or an assertion to make it pass is not a fix (R11c: the supersede-remap test).

## 4. The only allowed next work
* **Agent G:** R11d = APPROVED (section M); **R12 = CHANGES (section N)**: the learning loop is verified, growth triggering, the pre-registration and several report numbers are not. Next: **R12b only** = the six numbered items at the end of section N (rewrite `phase2/PREDICTIONS.md` against the real `s5_grid`/`s6_long` builders with an AMENDMENT.md; growth-trigger validation on dev seeds with trigger rates; wire growth + telemetry; fix tests (d)/(e) and the promotion rule; report numbers only from generated files). No Task 19, no S5/S6 runs, no deletions without asking. Reference: `scratch/claude_ref_r12_checks.py`.
* **Agent G, routing v2 (the user's decision of 2026-10-10; R12b / vertical growth stays paused by the user's decision):** rounds of `docs/superpowers/handoff/ROUTING_V2_HANDOFF.md`, one at a time, nothing else. **Status 2026-10-12: RV2-1 to RV2-3 done by the implementer and verified, RV2-4 (adoption) done by Claude; next is RV2-5.** Fresh seeds 20-39 (stage 1) and 40-59 (stage 2) may be run only in the round that authorises them. Documentation of an adopted change is written by Claude, not by the implementer.
* **Agent Y (Track Y is closed):** (1) notes N1-N3 of section F as text fixes; (2) package the snapped state as `geometry/snapped_state.py` (`SnappedRotationState`, clean API, deterministic, limits in the docstring) with a fast regression test (Z6 and S3 exact to T = 256 on 3 seeds, no label leakage); (3) write `docs/COMPONENT_CARDS.md`, one short card per register row (what, evidence file, limits). Then STOP. No hyperbolic-attention work, no new benchmarks.
* **Agent Z (after "start track Z"):** round Z0 first (protocol library + geometry-health diagnostics), then Z1-Z7 one at a time per `docs/superpowers/handoff/GEOMETRY_ATLAS_HANDOFF.md`; every round starts with a PREREG file; nothing is integrated.
* **Agent Z status:** Z0 and Z0b are APPROVED (sections J and K of the review file). Next: **Z1 only** (spherical stream, nGPT family) per `GEOMETRY_ATLAS_HANDOFF.md` plus the Z1 start guidance in section K of the review file; PREREG first; then report and wait for APPROVE before Z2.
