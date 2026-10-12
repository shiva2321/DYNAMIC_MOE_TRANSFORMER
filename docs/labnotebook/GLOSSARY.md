# Glossary (plain words first, then the technical meaning)

| Term | Plain meaning | Technical meaning |
|---|---|---|
| Expert | a small specialist network for one kind of task | an MLP (default hidden width 128, one residual block) with its own optimiser, a 256-sample replay reservoir and a signature |
| Gate / gatekeeper | the doorman who decides which expert handles an input and when a new expert is needed | the routing and novelty machinery of `PhyloMoE` (signatures, conformal p-values, arbitration) |
| Regime | one kind of task or data source in a stream | a distribution of inputs with its own labelling function; in the function-regime streams each regime has its own hidden teacher network |
| Signature | what an expert remembers about the inputs it has seen | a von Mises-Fisher kernel-density estimate over up to 256 stored unit vectors (prototypes), with a concentration kappa fitted by leave-one-out likelihood |
| Conformal p-value | "how unusual is this input for this expert", with a guaranteed error rate | the rank of the input's score among the expert's leave-one-out calibration scores, (1 + count) / (n + 1); valid if the input is exchangeable with the calibration set |
| Benjamini-Hochberg (BH), q | a rule that keeps the share of false "new!" alarms at or below q | controls the false discovery rate at level q over the p-values of a batch; q = 0.02 in our benchmarks |
| Novel / known | "no expert recognises this" / "some expert does" | novel = BH rejects the largest p-value over experts at level q |
| Probation buffer | a waiting room for inputs nobody recognised | novel samples are kept, clustered and tested; a cluster must pass an evidence test before an expert is created |
| Spawn / extend | create a new expert / reuse and retrain the closest one | decided by the interference probe: train a child and a shadow copy of the parent on the new data and compare forgetting and gain |
| Infant | a newborn expert under protection | for `infant_calls` batches: a relaxed gate (p x 4) and a doubled learning rate; its competence monitor is silent (since 2026-10-10) |
| Competence monitor / CUSUM | an alarm that rings when an expert starts doing worse than it used to | a one-sided CUSUM on the standardised prequential loss with a Monte-Carlo-calibrated threshold |
| Prequential loss | the loss on a sample before the model learns from it | test-then-train |
| Oracle routing (B4) | the best case: someone tells each input its true regime | experts trained and used with the regime id; an upper-bound arm |
| kNN-all | the "just remember everything" baseline | nearest-neighbour classifier storing every training sample |
| Replay net (B2, B2*) | one ordinary network that rehearses old samples from a buffer | tuned over steps, width and buffer; B2* is the best single network found |
| Function-regime stream | a stream where lookup cannot win | each regime's label is a hidden random network applied to the within-regime noise, so close inputs do not share labels |
| Centre cosine (overlap) | how alike two regimes' input regions are | cosine between regime centres on the unit sphere, 0 = unrelated, 0.85 = nearly the same region |
| Batch vote | let the whole batch go to the expert most of it chose | if one expert holds >= 50 % of the accepted samples (at least 8), all accepted samples train it |
| Training purity | how clean each expert's training data is | share of what each expert trained on that came from its dominant regime |
| Fingerprint | a tiny summary of an expert used for a quick first look | k unit sub-centroids of its stored prototypes; stage-1 score = best cosine |
| Shortlist / cascade | look at all experts cheaply, examine only the best few carefully | top-m experts by fingerprint get the exact score |
| Second look | check again, exactly, anything that looks new | every input whose restricted p-value is <= q is rescored by the exact scan over all experts |
| MACs | multiply-adds, a measure of computing work | the gate costs (stored vectors) x d per input |
| Pre-registration (PREREG) | write down the pass marks before the test | statements, bars, expected outcomes, seeds, budgets, decision rule; hash-frozen before the first test seed |
| Kill rule | what we will do if the key statement fails | written in the pre-registration (e.g. stop claiming an accuracy advantage) |
| Dev / test seeds | seeds used to choose settings / seeds used once to judge | dev >= 100; test seeds are single-use (D-05) |
| Paired interval | the difference between two methods run on the same seeds, with its uncertainty | mean paired difference with a 95 % t-interval (`metrics.paired_diff`) |
| Evidence grade | how much to trust a result | A / B / C / L / R / H (README section 3) |
| AMENDMENT | a recorded change to a pre-registered protocol | never edits the PREREG; explains what changed and why |
| Hash-freeze | a fingerprint of a file taken before it is used | sha256 stored in `*.sha256`; the run refuses to start if the file changed |
