# Paper notes

What could be written, what it would rest on, and what is still missing. Nothing here is written yet; this keeps the evidence organised.

## Paper A (method): a calibrated, cheaper gatekeeper for a self-growing mixture of experts

**Working claim.** A growing expert pool whose gate (i) controls false novelty with conformal p-values, (ii) trains each expert on batch-consistent assignments, and (iii) routes through a cheap fingerprint shortlist with an exact second look, beats a tuned single network on streams where the answer must be computed, handles overlapping regimes, and costs a fraction of the exact gate - with **novelty decisions provably identical** to the exact gate.

**Rests on (when verified):** C-20 (repaired lifecycle, E-015), C-23 / C-24 (batch vote, E-019), C-25 / C-26 (shortlist, E-021 - pending), C-22 (contamination mechanism, E-017), honest C-21 (not cheaper today) and C-17 (the frozen pool failed). Negative results to include: F-17, F-18.

**Evidence that exists:** two pre-registered fresh-seed tests with 20 paired seeds each (grade A); spikes for the mechanisms (grade B); a hash-frozen pre-registration trail; reproducible code on a branch with 475 tests.

**Missing before submission**
1. Stage 2 result (E-021) and, if it passes, a measured scaling curve to hundreds of experts.
2. A real-data benchmark (text / code with a frozen encoder; Q-04) and at least one public continual-learning benchmark, so the claim is not only about our synthetic function-regime streams.
3. Published baselines re-implemented under the same protocol (Expert Gate, a prompt-pool method, SEMA / MaRS-style growth, a router-free GMM gate) and a tuned replay network - in every table (D-06).
4. Component ablations: each of the three stages on and off, plus sensitivity to q, batch size, k, shortlist size.
5. Interleaved and long-horizon streams (Q-08, Q-11); heavy overlap (Q-02).
6. A statistical analysis plan fixed in advance for the final tables, with a multiplicity argument (many dev variants were tried; test seeds were used once).
7. Independent replication of at least one result by a party other than the implementer (the reviewer regenerated reports byte for byte, but the same people designed the tests).

**Threats to validity to state:** synthetic streams; one expert family (small MLPs, hidden width 128, d = 128); small scale; author-designed benchmarks; the implementer and the reviewer are AI agents working with one human; dev-seed exploration before each pre-registration (forking paths, mitigated by fresh seeds and frozen bars); default flips change the code after earlier results (mitigated by pins).

## Paper B (methodology / negative results): how a continual-learning benchmark lied to us five times

**Working claim.** Common evaluation choices in continual-learning MoE work can produce confident wrong conclusions; we document five in our own pipeline, each caught by a control, and give an audit recipe.

**The cases (all with evidence in the notebook):** a training-budget artifact (replay under-trained at 1 step, F-10, E-012); a metric true by construction (F-05); an operating-point mismatch (q = 0.05 vs 0.02, F-07); an over-trained baseline making a speedup (F-08, E-008); a parameter count that double-counted a shared module (F-11); an unpinned default changing old results (E-016, E-020); plus the earlier string-grading and Gaussian-probe cases (F-01, F-12).

**Evidence that exists:** almost everything, as generated files plus the reviews. **Missing:** a clean write-up with each artifact reproduced by a single script from the recorded commit, the audit recipe (D-14) evaluated on held-out cases, and a literature check of how often the same errors appear elsewhere.

## Reproducibility package (for either paper)

Commit hash (branch `track-g-growmoe`); `evidence/MANIFEST.json` with sha256 of every result file; the environment in README section 5; seeds (D-05); the exact commands (each entry's "Reproduce"); the pre-registrations and amendments; the unit tests; no downloaded data (all streams are generated deterministically from the seed, except the MiniLM text stream which uses a local cached encoder). Run `python docs/labnotebook/tools/labnb.py verify` before every release.
