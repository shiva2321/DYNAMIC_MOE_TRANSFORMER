# Phase 0 results — go/no-go decisions and the design changes they forced

Scripts: `spike_novelty_scores.py`, `spike_init_modes.py`, `spike_growth_trigger.py` (all throwaway, labelled, run from the
repo root). Raw numbers: `research/growmoe_results/phase0/*.json`. Narrative and every table:
`research/growmoe_results/IMPLEMENTATION_LOG.md` (sections T4b-T11). Seeds: 5 (spike 1), 10 (spike 2), 3 (spike 3).

## Verdicts

| Spike | Pre-registered rule | Result | Verdict |
|---|---|---|---|
| 1 novelty scores | vMF-KDE AUROC >= 0.95 on S1 and >= 0.85 on S2 | conformal KDE: S1 **0.973** [0.967, 0.979], S2 **0.999**, S6 0.993 | **GO** |
| 2 child init | affinity-scaled child not worse than scratch on unrelated, better on related | spec's rho-redraw rule: worse than scratch on unrelated (AULC -0.038, p < 0.001 at sigma 2) | **NO-GO as specified** -> adjusted rule below |
| 3 growth trigger | grows when the teacher is larger, not when data-limited | original greedy policy: never grew for the w32 teacher | **FAIL -> fixed**; fixed policy: **GO** |

## Design changes the spikes justify (these OVERRIDE the original spec; all post-hoc, to be re-validated in Phase 1)

1. **Child initialisation = `MLPExpert.shrink_perturb(lam = signature.affinity(cluster))`**, not the rho-redraw `inherit`.
   Area under the low-sample learning curve vs scratch (sigma 1 / 2): unrelated +0.000 / +0.002, related (class-mean cos 0.9)
   +0.042 / +0.132, equal to a clone on related and to scratch on unrelated, whereas a clone is -0.035 / -0.122 on unrelated.
   `inherit(rho, sigma)` stays as an ablation arm. A related regime with cos >= 0.9 is solved zero-shot by the inherited child
   (accuracy 0.97-1.0), so the extend-vs-spawn probe must be able to choose "extend" there.
2. **Growth policy** (`growth.py`): candidates are `widen` to {1.5, 4, 16} x the current width plus `deepen`; the
   **cheapest candidate passing every gate** is committed; gates = gain over an equally trained control >= 2 % relative,
   paired-bootstrap p < alpha / n_candidates, and **strictly better held-out loss than the expert before the probe**
   (status-quo gate: a memorised expert's control degrades 2.81 -> 4.6 under Adam, so "beats the control" alone falsely accepts).
   Probe = 480 steps. Operating characteristic (12 rng seeds): capacity-limited 100 % accept, data-limited 0/12,
   sufficient-capacity 0/12, S5 w32 d2 62 % (always the +60 jump).
3. **Signature stays vMF-KDE conformal** (256 prototypes, kappa grid <= 256): a wider kappa grid, 64/1024 prototypes, a
   Gaussian-Mahalanobis signature (once calibrated) and a KDE-union-Mahalanobis gate gave no significant gain.

## Facts to carry into Phase 1 (measured)

* **Novelty AUROC is not uniform.** S1: every arrival >= 0.99 for the KDE except **stdlib code after pydoc text (0.859)**.
  S1 ordered type pairs (fit on one type): KDE 0.935 (6/42 pairs < 0.8, min 0.36 = ar2 -> digits), VAE-ELBO 0.893 (11/42, min 0.47),
  key-activation 0.723 (26/42). A signature fitted on a DIFFUSE regime (AR(2)) is a weak one-class reference for every density
  method; the competence monitor is the designed safety net.
* **Related-but-new regimes (S6, class-mean cos 0.9):** KDE 0.980, kNN 0.993, Mahalanobis 0.986, but VAE 0.590, key-activation
  0.657, RND 0.759 -> B (VAE gate) and C (key gate) are expected to reuse the parent instead of spawning.
* **Pre-registered C5 supported in letter** (the VAE likelihood gate fails, AUROC < 0.8, on 11/42 S1 type pairs).
* **Affinity** (`VmfSignature.affinity`) ~ class-mean cosine: unrelated 0.01, 0.7 -> 0.54-0.64, 0.9 -> 0.84-0.88, 0.97 -> 0.94-0.96.
* **Growth on S5 (n = 20000, 3 seeds, final test accuracy grown / static w4 / static w128):** w8 d1 0.816 / 0.721 / 0.779 (2.3
  commits, width 14); w32 d2 0.594 / 0.567 / 0.598 (1.0 commit, width 48, 40 % of the wide student's parameters); w128 d3
  0.497 / 0.510 / 0.482 (hard teacher: data/optimisation-limited, growth neither helps nor hurts); n = 2000: **0 commits in
  all 9 runs** (data-limited).
* **Power warning:** growth significance tests need validation data. A pool expert's replay buffer is <= 256 samples
  (validation split ~50), so growth will rarely fire inside a pool unless growth-eligible experts get larger buffers.
  Task 14 must measure this and report.
