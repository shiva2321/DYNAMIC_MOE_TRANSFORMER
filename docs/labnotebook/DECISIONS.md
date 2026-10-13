# Decision log

Each decision: what, when, who, why, alternatives, and what it rests on. A decision is never rewritten; a reversal is a new row.

| ID | Date | Decision | Who | Why / basis | Alternatives considered | Entry |
|---|---|---|---|---|---|---|
| D-01 | 2026-10-04 | Three-party workflow: Gemini implements, Claude designs and reviews, the user mediates | user | conserve Claude usage; reviewing is cheaper than implementing | Claude implements everything | E-022 |
| D-02 | 2026-09-21 | Numeric / exact grading only; arms differ in exactly one bit; n >= 400 for small effects; never cite the +10 figure | Claude | E-001 | string grading, small n | E-001 |
| D-03 | 2026-09-21 | Measure on the real distribution; a random baseline beside every geometry number | Claude | E-004 | Gaussian probes | E-004 |
| D-04 | 2026-10-04 | Pre-registration first (statements, bars, expected outcomes, kill rule) and verdicts computed by script | Claude | E-011: reports contradicted their own tables | prose verdicts | E-009, E-013 |
| D-05 | 2026-10-04 | Dev seeds >= 100 for tuning; test seeds are single-use. **Used:** 0-9 (Phase 1, M0), 10-19 (M2-A), 20-39 (RV2-A). **Reserved:** 40-59 (stage 2). | Claude | a seed that chose a setting cannot test it | reuse | E-013, E-015, E-019 |
| D-06 | 2026-10-05 | Tune every arm including its training budget; kNN and a tuned replay network in every continual-learning table | Claude | F-04, F-10 | one setting for all | E-007, E-012 |
| D-07 | 2026-10-04 | Signature = von Mises-Fisher kernel density, 256 prototypes, conformal p-values with Benjamini-Hochberg | Claude | Phase-0 spike 1 (AUROC 0.973 / 0.999); wider kappa grid, 64 / 1024 prototypes, Mahalanobis variants gave no significant gain | VAE likelihood (B), keys (C), Gaussian | E-010 |
| D-08 | 2026-10-05 | Child expert init = shrink-perturb scaled by signature affinity; growth policy = cheapest candidate passing every gate | Claude | Phase-0 spikes 2 and 3 | the spec's rho-redraw rule; greedy growth | E-010 |
| D-09 | 2026-10-10 | A pre-registered pass makes a repair the default; the historical entry points are pinned to the old value; the change is documented everywhere it matters | user | "if it really helps then keep it on by default and make note of it and mention wherever it matters" | opt-in flags | E-016, E-020 |
| D-10 | 2026-10-09 | R12b / vertical growth paused | user | wants to know first whether dynamic MoE helps at all for continual learning | continue growth work | E-012 |
| D-11 | 2026-10-09 | A frozen pretrained text encoder only for realistic text streams (M5), not now | Claude, user agreed | not needed for the function-regime tests | encoder everywhere | E-012 |
| D-12 | 2026-10-09 | An episodic memory tier is an optional arm to test after M1, not assumed | user (undecided), Claude plan | pros and cons not yet measured | build it in | E-012 |
| D-13 | 2026-10-10 | Routing v2 in three stages: batch vote (plain primary, restricted vote a pre-declared fallback built only if a guard fails), fingerprint shortlist with an exact second look, calibrated batch novelty behind a spike | user approved "A" | E-017 | stage 1 only; labels as evidence (refuted) | E-018 |
| D-14 | 2026-10-12 | Audit recipe for an implementer run: compare mtimes, recompute recorded hashes, regenerate the report and compare byte for byte, check that only planned test files are new, re-run the suite | Claude | F-16 | trust | E-019 |
| D-15 | 2026-10-12 | Commit only when the user asks; GrowMoE on branch `track-g-growmoe`; other tracks stay uncommitted until asked | user, Claude | standing rule | commit everything | E-020 |
| D-16 | 2026-10-06 | Hyperbolic only for static hierarchies; no hyperbolic attention heads | kill rule | E-007 | continue Track Y | E-007 |
| D-17 | 2026-10-06 | Integration is forbidden until the register says VERIFIED and the user says "integrate" | user | one component at a time | integrate early | E-022 |
| D-18 | 2026-10-09 | Claude does light rounds (baselines, accounting, streams, runs); Gemini does deep changes (gate, signature, expert code) | user | token economy | all to Gemini | E-012 |
| D-19 | 2026-10-12 | Fingerprint = k-means sub-centroids (k chosen on dev, expected 4) rather than random landmarks | Claude | E-017: landmarks worse at equal cost | centroid only; wave features | E-018 |
| D-20 | 2026-10-12 | A lab notebook is kept for the whole project; claims, failures, decisions and evidence hashes live there | user | a future paper needs records and proofs | scattered reports | this notebook |
| D-21 | 2026-10-13 | Do not build smarter per-sample arbitration (confidence, entropy, soft mixtures) | Claude | E-023 E: no headroom, confidence pick loses up to 6.9 points | arbitration upgrades from FLEX / TOOD | E-023 |
| D-22 | 2026-10-13 | The batch-level novelty test uses per-expert source windows as references, candidates = nearest window mean, ownership of an admitted batch by label loss | Claude | E-023 D: stored prototypes and one window give 14-30 % false novelty on merged experts; vote ownership starves new experts; nearest-mean ownership costs S1 2.4 points | stored prototypes; one recent window; vote or nearest-mean ownership | E-023 |
| D-23 | 2026-10-13 | At inference a group is answered by the vote of its inputs own choices with a per-sample veto, not by a permutation test | Claude | E-023 F: the vote carries the signal (+12.1 at OV-0.85); the plain vote loses 15-38 points on mixed groups, the veto removes that | set-test routing; plain vote | E-023 |
| D-24 | 2026-10-13 | Stage 3 (set gate, group answering) stays a design until the user approves it; then flags OFF until a pre-registered run on seeds 60-79 | Claude | brainstorming hard gate; stage 2 owns seeds 40-59 | build now | E-023 |
