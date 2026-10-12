# E-014 - Routing investigation: where is the loss? (gate fine, lifecycle bug found)

- **ID:** E-014
- **Date:** 2026-10-09 (evening)
- **Track:** G
- **Type:** experiment
- **Status:** completed
- **Evidence grade:** B (exploratory, dev seeds 100-102, not pre-registered)
- **Code / commit:** diagnostics `research/growmoe_results/m0/diagnose_*.py` (D1-D6) and their outputs; later fix in `growmoe/pools/phylo.py` (E-015)
- **People:** Claude; user: "so basically we need better routing algorithm... let's address that first... some parts are in universal substrait on the E drive and some in this project and also check the internet"

## Question
M0 (E-013) said the pool realises only a fraction of what oracle-routed experts achieve. Is the loss in routing (the gate) or somewhere else?

## Why we asked it
A pre-registered FAIL (B1) with a large oracle gap; the user's reading was "routing". We decomposed routing into four jobs: detect novelty, assign samples for learning, select the expert at test time, and manage the lifecycle (spawn / extend / merge).

## Prediction
None; diagnostics. The user's hypothesis "the routing is the problem" was tested as stated.

## Method (how)
Six diagnostics on dev seeds (D1 oracle-gated analyses; D3a / D3b routing causes; D4 / D5 machinery ablations; D6 arming), an inventory of `E:\universal_substrait`, `E:\unified` and `E:\universal-ai-memory`, and a literature pass (abstract level).

## Results (what came out)
- **The gate is fine.** Test-time routing is 92-99 % accurate; training-time assignment is 99 %+ pure after the first two batches of a regime (those two batches leak 6-26 % of samples to older experts - this reappears in E-017).
- **Oracle gating with A's own recipe** (replay mixing) reaches 52.7 % at 5 steps on F-15 (dev); M0's B4 (48.5 %) was understated.
- **Root cause = a lifecycle bug.** The competence monitor (a one-sided CUSUM on prequential loss) is armed at the expert's birth from the loss of the *untrained* network (mean 2.47). After the first heavy training batch the newborn's held-out loss rises to about 4.3 as it over-fits a tiny batch; the CUSUM reads that as concept drift and fires on the newborn itself; the alarm path then replaces its weights with a 60-step fine-tune on 32 samples (`_handle_cluster(force_parent=newborn)`). It fires about once per regime (alarm 13 per run, extend 13 per run). `growmoe/competence.py` itself says the monitor should be armed only after the loss has plateaued.
- A scratch fix (reset infants' monitors) lifted F-15 at 5 steps to 46.8 %; alarms switched off entirely gave 46.3 / 47.6 but S1 fell 76.0 -> 70.0, so alarms are useful on overlapping regimes. More steps hurt once the fault is gone, so the 50-step budget M0 tuned for A was an artifact.
- Gate cost: 83-96 % of A's multiply-adds; calibration count is welded to prototype count (conformal floor 1/(n+1)).
- `E:\universal_substrait`: a phasor-resonance gate with one learned key per expert, Gram-Schmidt keys, a newborn bonus switched off for replay, token-sorted dispatch (18.5x); its own pre-registered H1 failed but per-domain retention improved on 8 / 10 domains at 28.6 % throughput cost; lessons: cold-start monopoly, newborn disadvantage. `E:\unified`: `VSARouter` (few sub-prototypes plus margin), `FamiliarityGate` (split-conformal nearest-neighbour novelty), `EpistemicRouter` (cost cascade). `E:\universal-ai-memory`: an episodic graph, not routing.
- Literature (`L`): router drift (SAME, LLaVA-DyMoE, Spectral MoE), gradient-free routers (Latent-LoRA GMM, Expert Gate, Arrow, FlyPrompt), FLEX (long-horizon routing recovers only 28-50 % of the oracle gap; calibrated soft mixtures beat hard), loss-based gating, confidence-based task prediction, sublinear routing (PEER, FFF).

## Interpretation
The premise "routing is the problem" was only half right: routing was fine; a lifecycle repair, not a better gate, was the first fix. The efficiency half of the routing problem (the cost of the scan) is real and separate.

## Caveats and what went wrong
Dev seeds, many variants tried (a garden of forking paths): the 46.8 % is a hypothesis until confirmed on fresh seeds - which is what E-015 does. The `E:` drive inventory is by reading code, not by running it.

## Consequence
Option A (repair + one pre-registered fresh-seed test) approved by the user ("sure"); routing v2 (cheaper gate, overlap handling) queued behind it.

## Evidence
- `research/ROUTING_RESEARCH_2026-10.md` - inventory, literature, measurements, options
- `research/growmoe_results/m0/diagnostics_D1.json`, `research/growmoe_results/m0/diagnostics_D3.json`, `research/growmoe_results/m0/diagnostics_D4.json`, `research/growmoe_results/m0/diagnostics_D5.json`, `research/growmoe_results/m0/diagnostics_D6.json`, `research/growmoe_results/m0/diagnostics_guards.json`
- `research/growmoe_results/m0/diagnose_routing_causes.py`, `research/growmoe_results/m0/diagnose_machinery.py`, `research/growmoe_results/m0/diagnose_machinery2.py`, `research/growmoe_results/m0/diagnose_arming.py`, `research/growmoe_results/m0/diagnose_guards.py`

## Reproduce
Each `diagnose_*.py` is run from the repository root and writes the matching `diagnostics_*.json` / `*_log.txt` beside it.
