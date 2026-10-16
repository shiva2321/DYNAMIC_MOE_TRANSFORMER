# Stage 4.1 — a full pool says so and shares instead of dropping; read-only counters for redundancy and for the infant window; an optional fix for the twice-counted founding cluster

Date 2026-10-16. Status: **built and tested on branch `stage4-1-cap-and-telemetry`** (worktree `D:/shm_stage41`); not merged into `track-g-growmoe` until the stage-4 dev review is finished (see section 6). Path: bounded (a new setting and some counters on code that already exists; the user chose the policy in the decisions below).

## In plain words

GrowMoE is allowed at most `max_experts` experts (default 256). Until now, when the pool was full and a region it had never seen arrived, the pool did **nothing, and said nothing**: the samples were not learned, the expert that answered them did not know them, and no event, counter or warning existed. The merge study (E-034) measured what that costs once the pool is full: the regimes met after it fills are answered at chance (about 25 %), and forcing the cluster into the nearest expert instead is 5-7 points better in all three settings measured (56.5 / 55.6 / 59.0 % against 50.7 / 48.3 / 53.4 %).

This stage changes two things, and neither can change a result before the pool is full:

1. **A full pool is loud.** One `cap_reached` event when the pool first holds `max_experts` experts, one `RuntimeWarning` the first time the cap actually turns a region away or shares it, and counters in `summary()["cap"]`.
2. **A full pool shares instead of dropping** (`cap_policy="share"`, the new default; `"drop"` is the shipped behaviour and stays available). The cluster is added to the signature and the replay of the nearest expert and nothing else is changed; from then on the gate sends that region to the expert, which trains on it like on its own samples. Nothing is deleted and no expert is created. When the labels of the new region conflict with the expert's, the expert becomes a compromise: that is the price of the cap, and it is reported (`cap_share` events, `shared` counters) so that nobody finds out later.

Two read-only tools come with it, so that the next decisions rest on measurements and not on guesses (and one optional, OFF-by-default fix, item 5):

3. `redundancy_report()` lists pairs of experts that cover the same inputs (the shipped merge rule's overlap) with the share of prototypes on which they predict the same class.
4. `summary()["infant"]` counts how often the young-expert relaxation (`infant_relax`, x4) let a sample through that the batch would otherwise have called new.
5. `PhyloCfg.sig_dedupe` (default False): a signature never holds the same point twice. It exists because looking for a fix of the infant window showed the real cause: the founding cluster of every new expert enters its signature twice (notebook E-037). Nothing changes while the flag is off.

## 1. Decisions (from the user, 2026-10-16)

1. *Full pool*: "no dropping unless necessary because sometimes some knowledge and things are important but they are used/activated very rarely too" -> `cap_policy="share"` by default, and **no expert is ever evicted** by this stage. Eviction of a redundant expert (the best policy in the merge study when the shortage is caused by redundancy: 66.1 / 64.6 against 56.5 / 55.6 for sharing) is withheld until redundancy can be certified with honest evidence, because signature overlap alone cannot tell a twin from a rarely used expert that holds a different function on the same inputs (the report says so in `agreement`). That belongs to a later, separately pre-registered stage (rival tracking on live traffic).
2. *Repair evidence*, *sequence* and *head start*: left to me. Chosen: repair by live-traffic rival tracking in a later stage; this stage is behaviour-neutral until the cap is reached and goes in before the freeze of the confirmatory run, in a separate branch so the running dev queue (which fingerprints `growmoe/**`) is not disturbed.
3. *Answer-time averaging of overlapping experts*: yes, once validated on a real stream (a later stage; it changes nothing on a pool without twins, C-72).
4. *Infant window*: fix it if a concrete, reliable fix exists, otherwise document it. This stage counts it (section 4) and carries the candidate fix behind `sig_dedupe` (section 4a); making it the default needs the dev check of E-037 and a decision.

## 2. What is built

| Part | Where | Behaviour |
|---|---|---|
| `PhyloCfg.cap_policy` | `growmoe/config.py` | `"share"` (default) or `"drop"`; anything else raises `ValueError` |
| `cap_reached` event | `PhyloMoE._note_cap_reached` | once, the first time `len(experts) >= max_experts` (at the end of `observe`, and after the bootstrap); fields `experts`, `cap`, `policy`; no warning (reaching the cap changes nothing by itself) |
| `RuntimeWarning` | `PhyloMoE._warn_cap_once` | once per pool, the first time the cap acts; says whether the region is shared or lost and to raise `max_experts` |
| `cap_share` event | `PhyloMoE._cap_cluster` | policy `share`: fields `expert` (the nearest expert), `samples`, `experts`, `cap`, `affinity`; the cluster joins that expert's signature and replay, novel samples of the current batch that its signature now accepts are assigned to it |
| `cap_blocked` event | `PhyloMoE._note_cap_blocked` | policy `drop`: counted every time, an event for the 1st, 2nd, 4th, 8th ... time; fields `samples`, `blocked`, `experts`, `cap` |
| `summary()["cap"]` | `PhyloMoE.summary` | `policy`, `cap`, `reached_step`, `blocked`, `blocked_samples`, `shared`, `shared_samples` |
| `summary()["infant"]` | `PhyloMoE.summary` | `batches_with_infant`, `novel_to_mature`, `suppressed` (definitions in section 4) |
| `redundancy_report(limit=64)` | `PhyloMoE` | `{"experts", "compared", "pairs": [{"a", "b", "overlap", "agreement"}], "truncated"}` |
| `VmfSignature(dedupe=False)`, `PhyloCfg.sig_dedupe=False` | `growmoe/signature.py`, `growmoe/config.py`, `PhyloMoE._new_sig` | with the flag on, `update` skips a point whose cosine to a held prototype exceeds 1 - 1e-5 and does not count it as seen; the pool builds every signature (bootstrap, spawn, evidence test) through `_new_sig`, which passes `dedupe=True` only when the flag is on (a custom `sig_cls` without the argument keeps working); a signature pickled before the flag existed still updates |

Guards in `observe` (both the probation branch and the `no_probation` branch): a full pool under `"drop"` skips clustering exactly as before and counts it; under `"share"` it clusters and `_handle_cluster` reaches `_cap_cluster` where it used to `return None`. The competence-alarm path also goes through `_cap_cluster` (under `"drop"` it is the old `return None`, now counted).

Counters are created on first use, so a pool pickled before this stage works (`_cap_state`, `_infant_state`, the same pattern as the router's state).

## 3. Statements the tests pin (`tests/growmoe/test_gm_s41_cap.py`)

| ID | Statement |
|---|---|
| S41-1 | With the cap not reached the pool is bit-identical to before: the 20 flag-off fingerprints of `research/growmoe_results/rv2c/golden.py` match (checked on the branch: "bit-identical"), and two pools that differ only in `cap_policy` produce the same events and the same answers when the cap is never reached |
| S41-2 | `cap_policy` is validated and defaults to `"share"` |
| S41-3 | Under `"drop"` a full pool founds nothing, writes `cap_blocked`, counts, and warns once; under `"share"` it founds nothing, writes `cap_share` (never `cap_blocked`), and the cluster is in the signature (`seen` grew by at least `m_min`, the fresh inputs of that region get a higher p-value) and in the replay |
| S41-4 | `cap_reached` and the warning each come once per pool |
| S41-5 | A pool pickled before this stage (no counters) still works |
| S41-6 | `redundancy_report` flags a forced twin and not experts of different regimes, and is read-only (no event, same answers, same weights) |
| S41-7 | Infant suppression is positive with `infant_relax` 4, zero with 1, and never exceeds `novel_to_mature` |
| S41-8 | `summary()` always carries the `cap` and `infant` blocks |
| S41-9 | `VmfSignature` without `dedupe` keeps a repeated point (128 prototypes after the same 64 twice); with it the repeat is dropped, only exact repeats are dropped (close points stay), and the calibration of a doubled cluster is ruined (mean p-value of fresh points below 0.02) while the deduplicated one stays honest (above 0.3) |
| S41-10 | With `sig_dedupe` the founding cluster enters the new expert's signature once (64 prototypes against 128); with the relaxation off a regime of six batches founds exactly two experts with the flag and more than two without; with the flag and no relaxation a new region that arrives two batches after a spawn is still seen as new, with the shipped x4 it is not |

## 4. The counters of the infant window

`novel_to_mature` adds, in every observe call that has an infant, the samples the mature experts alone flag as new (Benjamini-Hochberg at level `q` on their best p-value). `suppressed` adds the samples that the batch **would** have flagged had the infants' p-values not been multiplied by `infant_relax`, but that the relaxed gate let through (one more Benjamini-Hochberg call on the infants' unrelaxed p-values). Because Benjamini-Hochberg is monotone in the p-values, `suppressed <= novel_to_mature` always; `infant_relax` 1 gives 0. The counters are exact for the exact gate; with `gate_shortlist > 0` the mature columns outside the shortlist are 0, so both are upper bounds there. Nothing in this code feeds back into the pool.

## 4a. The optional fix of the founding cluster (flag `sig_dedupe`, OFF)

In the one observe call that founds an expert, `_handle_cluster` puts the 64 samples of the cluster into the new signature (`child_sig.update(x_taken)`); the cluster's samples are then assigned to the new expert and the training loop adds the same rows again (`rec.meta["sig"].update(xk)`): 64 of 64 rows were already held (call trace, notebook E-037). A signature that holds every point twice has the twin value as every calibration score, so fresh points of the expert's own region get the smallest possible p-value for the next batch and recover to about 0.5 only after 7-8 batches. The shipped x4 `infant_relax` lifts the floor above the acceptance level for one batch (and blinds the pool to a new regime for four); without it the pool founds an expert per batch.

Measured so far (E-037): the mechanism on three seeds; with the fix and `infant_relax` 1 the pool is never worse than the shipped one in 12 settings (regimes of 10, 5, 3 and 2 batches, three seeds each) and finds more regimes at 2 batches. The dev check on the guard streams, arithmetic and banking77 is running (`research/growmoe_results/rv2d/run_s3_dedupe.py`, results in `RESULTS_S3_DEDUPE.md`). The flag stays OFF, and `infant_relax` is not touched, until that check and a decision say otherwise; if it becomes the default the historical entry points are pinned to the old behaviour (standing rule).

## 5. What this stage does not do

- It never evicts, merges, moves, or retires an expert (decision 1).
- It does not repair wrongly created experts, and it does not average answers of twins (later stages, own pre-registration).
- It does not change `infant_relax`, and with every flag at its default it does not change how the gate is calibrated (section 4a).
- It does not make the cap a good idea: a pool that shares is a compromise, and `summary()["cap"]` is how to see that it happened.

## 6. Order of work

The dev queue of stage 4 stamps its cells with a hash of `growmoe/**` and `rv2d_lib.py`; a cell counts as done only for the current stamp. This stage therefore lives on its own branch and is merged into `track-g-growmoe` after the dev review and **before** the freeze of the confirmatory run (the freeze takes the stamp of the merged code). With every flag off and the cap not reached it changes no result (S41-1), so it cannot disturb the pre-registered comparisons; the only moment it acts is a pool that holds 256 experts, and no stored result of this project reached that (the largest pool, N = 100 regimes, ends with 100).
