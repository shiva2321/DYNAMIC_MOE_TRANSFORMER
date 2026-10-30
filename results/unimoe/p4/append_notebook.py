"""One-off: append the P4 rows (F-62..F-65, C-121..C-125, Q-41..Q-44, D-43) to the lab notebook trackers. Run once from D:/shm_um; refuses to run twice."""
from pathlib import Path

NB = Path(__file__).resolve().parents[3] / "docs" / "labnotebook"


def append(fn, rows, first_id):
    p = NB / fn
    s = p.read_text(encoding="utf-8")
    if f"| {first_id} |" in s:
        raise SystemExit(f"{fn} already has {first_id}")
    if not s.endswith("\n"):
        s += "\n"
    p.write_text(s + "\n".join(rows) + "\n", encoding="utf-8", newline="\n")


append("FAILURES.md", [
    "| F-62 | A one-line mutation-test leftover sat in `unimoe/runner.py` through all of Stage 1 and P4: the per-phase known-message flag rate was hard-coded to 0.0 (the runner test that catches it was not re-run after the mutation check) | Claude / diffing the stamped files against git while checking why every cached cell looked stale | none on the published numbers: the end-of-stream value recomputed from the stored test arrays is 0.0 for every system (`results/unimoe/known_flag_rate_repaired.json`); only the per-phase maximum is unverified (nothing cites it); all earlier cells carry the stamp of the mutated file | after any mutation check run the whole suite and `git diff --stat` before the next commit; keep new code out of stamped files (the density reader was moved to its own module for this reason) | E-054 |",
    "| F-63 | My first small-start growth grid (theta 0.005-0.1) had the sign backwards: growth fires when the one-step gap is BELOW theta, so a smaller theta fires less; the sign-based predictions are void | Claude / reading the grid's identical results and then the rule | one tuning round (about 20 minutes) | write the direction of a threshold in the plan, from the code, before choosing a grid | E-054 |",
    "| F-64 | The small-start growth test (16 units, real suite) repeated Stage 1's E6 (16 units, fast maturity, mixed: 0.442 -> 0.478 with one event, the same numbers) which I had not looked up; five predictions missed | Claude / reading PROGRESS.md afterwards | a predictions block and four cells | grep the progress log for the configuration before predicting it | E-054 |",
    "| F-65 | Hit rate of the 38 numeric predictions of the four changes and their follow-ups: 17 held, 21 missed (45 %); misses: the LDA head beat my range twice, the domain head missed 3 of 5, the refusal bar of 0.6 was out of reach at AUROC 0.85, and a threshold transferred from the arithmetic stream to text where growth is structurally blocked | Claude / the scoreboards | nothing beyond the record | read what limits a mechanism on the target stream (experience per expert, score quality) before predicting its effect | E-054 |",
], "F-62")

append("CLAIMS.md", [
    "| C-121 | A growth statistic taken from ONE gradient step on a copy of the expert, with maturity scaled to the stream, grows a 16-unit expert on a stream that needs capacity (six arithmetic operators, 10,000 examples each): sequential 0.725 -> 0.782 (7.7 events, 475 k parameters; a fixed 512-unit expert 0.829 with 803 k), mixed 0.678 -> 0.718, and also at 5 inner steps (0.736 -> 0.779); the Stage 1 rule with the same maturity fires nothing | SUPPORTED (dev, 3 seeds; threshold tuned on one seed) | B | E-054 | tested on one stream family; the threshold 0.2 does not transfer to text |",
    "| C-122 | Automatic vertical growth cannot fire on the real sequential suite from a small start at any threshold (0.2-0.9): experts get a median of 5 batches, the rule needs age >= 30 and windows >= 10; with one expert (mixed, drifting) it fires once, mixed +0.036 over fixed 16, closing 23 % of the gap to 128 units, drifting no gain; with the tuned 128 units it fires nothing and equals U-real | SUPPORTED (dev, 3 seeds) | B | E-054 | the per-visit right-sizing that would fix it is not built (Q-41) |",
    "| C-123 | A running LDA head inside every expert, blended in log-probability, is the largest single improvement of the pool: paired gain over U-real +0.022 [+0.010, +0.033] sequential, +0.069 [+0.057, +0.082] mixed, +0.126 [+0.100, +0.153] drifting (5 seeds, all 15 differences positive); under oracle routing 0.669 -> 0.750, 0.663 -> 0.752, 0.604 -> 0.751; the pool reaches the offline joint ceiling in mixed (-0.006) and drifting (-0.028), 0.021 below it sequentially, beats equal-memory replay in mixed by 0.036 and kNN by 0.16 / 0.11 | SUPPORTED (dev, 5 seeds incl. 2 confirmation seeds) | B | E-054 | frozen sentence vectors only; a trained representation (Stage 2) may behave differently (Q-43); development seeds, not pre-registered |",
    "| C-124 | A supervised domain head (class-level LDA summed over label blocks) adds little: -0.001 / +0.010 / +0.032 over U-real (sequential / mixed / drifting), wrong-domain 10.8 -> 9.9 %, 9.6 -> 7.0 %, 14.2 -> 7.1 %, nothing once the LDA head is in (UPLUS = ULDA +- 0.001); the soft power makes no difference; what remains is the task-ambiguity floor | SUPPORTED (dev, 5 seeds) | B | E-054 | the posterior is saturated, so a different fusion would not change this |",
    "| C-125 | The refusal layer wired to the live pool turns a silent flag (0.000 of foreign text) into a controlled one: with 60 verified messages at q = 0.05 the pool score refuses 0.21-0.46 of three foreign text sets at 5.6 % false refusal, a density (Mahalanobis) reader 0.07-0.50 at 3.6 %, all three fused 0.22-0.45 at 5.2 % (confirmation 0.23-0.57 at 6.1 %); AUROC 0.79-0.93; answering only the accepted messages gains +0.011 to +0.021 accuracy; the bar of 0.6 on every set was NOT reached because the scores' AUROC is about 0.85 and a fused cut is set to the same total level | SUPPORTED for the calibration, REFUTED for the bar (dev, 3 + 2 seeds) | B | E-054 | six text domains, sequential schedule, frozen 128-d vectors; the verified messages come from the evaluation test sets (disjoint) |",
], "C-121")

append("OPEN_QUESTIONS.md", [
    "| Q-41 | Can an expert's width be decided from the few batches a short task gives (median 5 in the sequential real suite)? | decide once per visit of a regime from how well the expert fits the regime it just finished (a probe over the visit, not a moving window); test at 16 units on the real suite | open (E-054) |",
    "| Q-42 | Can a better representation (the full 384-d embedding, or one trained for novelty) raise the novelty AUROC from about 0.85 to about 0.95 so that refusal reaches 0.6 of foreign text at 5 % false refusal? | the same refusal experiment on the unprojected embedding | open (E-054) |",
    "| Q-43 | Does the LDA head's dominance over trained MLP experts survive a TRAINED representation (token-level Stage 2), or is it a property of frozen sentence vectors? | Stage 2 with and without the LDA head under the same blend | open (E-054) |",
    "| Q-44 | How much of the remaining gap between the pool (0.646-0.663) and the told-the-regime LDA experts (0.745-0.752) is the task-ambiguity floor and how much is routing? | measure the share of messages whose label is ambiguous across domains against the pool's wrong-domain rate | open (E-054) |",
], "Q-41")

append("DECISIONS.md", [
    "| D-43 | 2026-10-30 | The LDA head becomes the default of the next UniMoE system: recommended configuration ULDA (lambda 0.5-0.75), domain head optional, growth from the one-step probe only where experts live long enough; U-real stays pinned as the Stage 1 reference and its cells are not recomputed. The stamped files were not edited for this (it would invalidate hundreds of cached cells); the default changes in the next package version | Claude | C-123 (5 seeds, all differences positive, effect far above noise); standing rule of the user: confirmed repairs go in as defaults, old entry points stay pinned | keep U-real as the default; make every variant a flag | E-054 |",
], "D-43")
print("done")
