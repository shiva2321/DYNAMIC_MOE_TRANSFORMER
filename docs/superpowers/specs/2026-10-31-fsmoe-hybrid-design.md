# FS-MoE hybrid: new-class items exempt from the domain throttle (design, 2026-10-31)

The user chose "Hybrid new-code-first" among the next steps of E-058. Same rules: dev seed 700 for smoke tests only, report seeds 710-714, predictions first, nothing committed, growmoe/ and unimoe/ untouched.

## A correction to my own proposal
The E-058 report proposed the hybrid as "equal domain shares, but the tool's new-behaviour items first". That is what D1 already does: inside every domain `BalancedLearner._order` puts items of classes the pool was never taught first (the tool's new-behaviour items carry the ids the system opened itself, which are not among the taught ids). What limits D1's code is not the ORDER but the QUANTITY: D1's pace is the scarcest domain's, about 17 chunks with a code quota of 5, so it trains 85 code items although the tool produced about 358 new-behaviour ones in the stream. The real hybrid therefore has to exempt the new-class items from the throttle.

## What is built (`fsmoe/balance.py`, mode `hybrid`, parameter `new_cap` = m)
* An item whose class the pool was never taught (human-labelled never-taught skill of any domain, or a tool-opened class) goes to a NEW queue; every other item goes to its domain's known queue, as before.
* A chunk is formed as soon as EITHER the new queue holds m items OR every domain seen (by its known items) holds its share of the known part (the `sub` rule on a known part of 16 - m slots, split equally over the domains seen).
* A chunk = the new queue (at most m items, in order of arrival: a chunk is formed the moment the queue holds m items, so it never holds more; my first draft said "round-robin over classes", which the mutation check showed to be vacuous for that reason, and it was removed before any run) + from every domain seen up to its share of known items (oldest first), whatever is available. No replay.
* At the end of the stream the remaining items are flushed under the same rule (new queue until empty, known items until the scarcest domain is empty); known surplus of the abundant domain is never used, as in D1.
* Nothing else changes: same stream, teaching, pool, global head, tool, domain head, human labels (228 in every arm).

## Arms (the E-057 A3 setting; the E-058 arms A3b, D1, D2 are NOT re-run except D1, as a reproduction check)
* D1r: `sub` again, with the new counters (must reproduce D1 exactly).
* H4, H8, H12: `hybrid` with m = 4, 8, 12 new-class items per chunk (a grid over the one free parameter, reported as a grid, not a pick).
New counters (all modes): fresh items trained per domain split into new-class and known items, so "D1 already trains new-class code items first" is measured, not argued.

## Measures
As in E-058: never-taught skills learned by domain, known skills in held-out and taught wordings by domain, chunks, items trained per domain (new-class and known).

## Predictions (written before any hybrid cell)
Reference (E-058, means of 5 seeds), arithmetic learned / code learned: A3b 0.20 / 1.00, D2 0.40 / 0.92, D1 0.67 / 0.80.
* H1 (the main one): at least one of H4, H8, H12 has arithmetic learned >= 0.50 AND code learned >= 0.90 (it breaks the trade-off of D1 and D2). I give this about 30 %.
* H2: the grid is monotone: code learned (H12) >= (H8) >= (H4) and arithmetic learned (H12) <= (H4), in the means over the 5 seeds.
* H3: D1r reproduces D1's learned fraction and known held-out accuracy (all domains) to 1e-9 in 5 of 5 seeds.
* H4: the headline stays: never-taught skills learned of all 28 in every hybrid arm between 0.33 and 0.38.
* H5: language learned stays at 0.30 or less in every hybrid arm.
* H6: code known skills in held-out wordings end between 0.90 and 0.94 in every hybrid arm (the hybrid trains a few more known code items than D1's 85, fewer than A5's 190, and A3's 0.959 needs thousands).
* H7: known skills in held-out wordings, all domains, end at 0.86 or more in every hybrid arm (D1 0.849, D2 0.868, A3 0.883).
* H8: language known skills in held-out wordings stay at 0.79 or less in every hybrid arm (no replay is used, so a drop that persists is not a replay artifact; D1, with no replay, already read 0.772).
