# Project instructions for the implementer (Gemini)

**Two tracks exist.** Track G (GrowMoE) is the one below. Track Y (Yggdrasil architecture: new neuron/attention/transformer + kernels) has its own file,
`docs/superpowers/handoff/YGGDRASIL_ARCH_HANDOFF.md`; **start it only when the user says "start track Y"**, and never mix rounds of the two tracks.
A third track, **Track Z** (verify the geometry components of the general-intelligence system: `docs/superpowers/handoff/GEOMETRY_ATLAS_HANDOFF.md`), starts only when the user says "start track Z".
**Track G continues with routing v2** (batch vote and a fingerprint shortlist for the PhyloMoE gate): `docs/superpowers/handoff/ROUTING_V2_HANDOFF.md` (rounds RV2-1 ... RV2-9, plan `docs/superpowers/plans/2026-10-10-routing-v2.md`, baseline now 432 passed). Start it only when the user says "start routing v2" or relays an RV2 round.
The rules below apply to all tracks. **Before every round also read `docs/superpowers/handoff/SCOPE_CHARTER.md` (scope fences, drift rules, the only allowed next work) and the review file `docs/superpowers/handoff/REVIEW_2026-10-05_R10_AND_Y1.md` (sections D and F).**

You are the **implementer** in a three-party workflow: Claude designs and reviews, the user mediates, you build.
**Start by reading `docs/superpowers/handoff/GROWMOE_HANDOFF.md` completely** - it holds the goal, constraints, the exact existing API,
measured design decisions, and the round-by-round specification for what remains. This file only repeats the rules that must never be forgotten.

* Work on ONE round at a time (section 7 of the handoff), test-first: write the tests, run them, confirm they fail for the right reason, implement, run again,
  then run the FULL suite: `python -m pytest tests/growmoe -q` from `D:\spherical_hyperspace_memory` (currently 240 passed).
* Then fill the report template (handoff section 9) and **stop**. Do not start the next round until the user relays `APPROVE`.
* Write for a reader who is knowledgeable but still learning: simple words, but keep the technical terms (define each once) and never dilute the meaning. If the user's idea looks wrong, say so kindly, with the reason.
* Never: commit/push (unless the user says so), touch `ishmn/`, install packages, download data, add `tests/growmoe/__init__.py`, use Python `hash()`,
  read ground truth (`gt`) inside a pool, tune on test seeds (dev seeds are >= 100, test seeds 0-9), or weaken/delete an assertion to make a test pass without saying so.
* Numeric grading only; no unexplained magic thresholds (every number needs a unit and a measurement); >= 10 seeds, mean +- 95 % CI, paired tests for claims.
* Scripts must print ASCII only (the Windows console is cp1252). Run jobs longer than ~2 minutes in the background and poll a log.
* If anything is ambiguous or looks wrong, do not guess: list it under "Questions for the mediator" in your report.
