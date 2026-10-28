"""Side check (not part of the suite): WHY do stories (S4) and noisy text (S7) set off the "unseen" alarm for skills the system knows? Tuning seeds 150 to 154 only (not report seeds).

    PYTHONPATH=<repo root> python diag_shift.py [first_seed last_seed] > results/E2_WHY.txt

Q1  map or reader?   The shift of every wording measured at three stages: the raw Qwen layer-26 vector, the plain reader (centred), and the learned-map reader (the headline).
Q2  which noise?     Case flips, typos and extra spaces applied one at a time to fresh taught-wording parts (generator.apply_noise, used here only to MEASURE, never to teach).
Q3  how many directions?  An ORACLE diagnostic (it looks at the test wording, so it is not a fix): remove the top k shift directions from the parts of one wording and see how much of the
    false alarm goes away; random directions as the control.
Q4  which score?     AUROC of "a known skill in this wording" against "a skill never taught, in the same wording" for: closeness to the taught examples, kernel density, and the margin
    between the nearest and the second-nearest group.
Q5  length?          False alarms and closeness against the number of words.
"""

import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

import assignments
import baselines as BL
import evaluate as EV
import generator as G
import reps
import run_e2 as R
import skills
import status as ST

NOISE = ("case", "typo", "space")
HELD = ("S4", "S5", "S6", "S7", "S8")
TAUGHT_W = tuple(assignments.TEACH_FAMILIES)


def extended_world(seed: int) -> R.World:
    """The ordinary world plus noisy copies of the fresh taught-wording parts of the taught skills (one noise kind at a time)."""
    world = R.build_world(seed, "dev", R.SIZES)
    rng = np.random.default_rng([seed, 0xD1A6])
    base = [(t, c) for t, c in zip(world.test_texts, world.test_cases) if c.kind == "arith" and c.family in TAUGHT_W and c.status in ("solid", "thin")]
    for kind in NOISE:
        for t, c in base:
            nt = G.apply_noise(t, kind, rng)
            if nt != t:
                world.test_texts.append(nt)
                world.test_cases.append(EV.Case(c.status, f"noise:{kind}", c.skill, "arith"))
    return world


def unit(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def main(seeds) -> None:
    flag = defaultdict(lambda: defaultdict(list))          # reader -> family -> flagged?
    own = defaultdict(lambda: defaultdict(list))
    shift = defaultdict(lambda: defaultdict(list))         # stage -> family -> z-shift per seed
    oracle = defaultdict(lambda: defaultdict(list))        # (family, kind) -> k -> flagged?
    auroc = defaultdict(lambda: defaultdict(list))         # family -> statistic -> auroc per seed
    length = defaultdict(list)
    for seed in seeds:
        world = extended_world(seed)
        texts = world.all_texts()
        index = {t: i for i, t in enumerate(texts)}
        rep = reps.get_rep("R4_qwen_L26")
        X = reps.cached_encode(rep, texts, R.CACHE)
        finals = {"plain": R._reader(world, X, index, seed, False, None), "map": R._reader(world, X, index, seed, True, None)}
        stages = {"raw": unit(X), "plain": finals["plain"], "map": finals["map"]}
        cases, ttexts = world.test_cases, world.test_texts
        rows = np.array([index[t] for t in ttexts])
        fam = np.array([c.family for c in cases])
        skill = np.array([c.skill if c.skill else "" for c in cases])
        known = np.array([c.kind == "arith" and c.status in ("solid", "thin") for c in cases])
        untaught = np.array([c.kind == "arith" and c.status == "unseen" for c in cases])
        teach_rows = {s: np.array([index[e.span] for e in world.teach[s]]) for s in world.taught}

        # Q1: shift per stage (z of the best cosine to the own skill's prototypes against the fresh taught-wording parts)
        for name, Z in stages.items():
            best = np.full(len(cases), np.nan)
            for s in world.taught:
                sel = np.flatnonzero(known & (skill == s))
                best[sel] = (Z[rows[sel]] @ Z[teach_rows[s]].T).max(axis=1)
            ref = known & np.isin(fam, TAUGHT_W)
            for f in sorted(set(fam[known])):
                m = known & (fam == f)
                shift[name][f].append(float((np.nanmean(best[m]) - np.nanmean(best[ref])) / (np.nanstd(best[ref]) + 1e-9)))

        # the machine of each reader, and its calls
        machines, assessed, V = {}, {}, {}
        for name in ("plain", "map"):
            P = finals[name]
            V[name] = (lambda P: (lambda ts: P[[index[t] for t in ts]]))(P)
            groups = {s: ST.SkillGroup(s, V[name]([e.span for e in world.teach[s]]), len(world.teach[s]), None, world.comp[s]["accuracy"]) for s in world.taught}
            other = ST.SkillGroup("other", V[name]([i.text for i in world.teach_none]), len(world.teach_none), None, 1.0)
            cal_vec, cal_grp, _, _ = R.calibration_arrays(world, V[name])
            machines[name] = (ST.StatusMachine(groups, other, cal_vec, cal_grp, q=R.HEADLINE_Q, seed=seed), groups, other)
            assessed[name] = machines[name][0].assess_many(P[rows])
            for a, c in zip(assessed[name], cases):
                if c.kind == "arith" and c.status in ("solid", "thin"):
                    flag[name][c.family].append(a.status == "unseen")
                    own[name][c.family].append(a.nearest == c.skill)

        # Q3: oracle removal of the top shift directions (map reader, one wording at a time)
        m, groups, other = machines["map"]
        P = finals["map"]
        ref_mean = {s: P[rows[known & (skill == s) & np.isin(fam, TAUGHT_W)]].mean(axis=0) for s in world.taught}
        rng = np.random.default_rng([seed, 0x5E])
        for f in ("S4", "S7", "S5"):
            D = np.stack([P[rows[known & (skill == s) & (fam == f)]].mean(axis=0) - ref_mean[s] for s in world.taught if (known & (skill == s) & (fam == f)).any()])
            _, _, Vt = np.linalg.svd(D, full_matrices=False)
            sel = np.flatnonzero(known & (fam == f))
            for k in (0, 1, 2, 4, 8):
                for kind in ("top", "random"):
                    if k == 0 and kind == "random":
                        continue
                    B = Vt[:k] if kind == "top" else np.linalg.qr(rng.standard_normal((P.shape[1], max(k, 1))))[0].T[:k]
                    Y = P[rows[sel]]
                    Y = unit(Y - (Y @ B.T) @ B) if k else Y
                    out = m.assess_many(Y)
                    oracle[(f, kind)][k].append(float(np.mean([a.status == "unseen" for a in out])))

        # Q4: which score separates a known skill in a wording from a never-taught skill in the same wording
        Zf = finals["map"]
        names = list(groups) + ["other"]
        protos = {n: Zf[teach_rows[n]] for n in world.taught}
        protos["other"] = Zf[np.array([index[i.text] for i in world.teach_none])]
        allrows = rows
        cos = np.stack([(Zf[allrows] @ protos[n].T).max(axis=1) for n in names], axis=1)
        srt = np.sort(cos, axis=1)
        best_cos, margin = srt[:, -1], srt[:, -1] - srt[:, -2]
        kde = np.stack([m._sig[j].score(Zf[allrows].astype(np.float32)).double().numpy() for j in range(len(names))], axis=1).max(axis=1)
        stats = {"closeness (best cosine)": -best_cos, "kernel density": -kde, "margin (nearest minus second nearest group)": -margin}
        for f in ("S4", "S7", "S5", "S8", "S6"):
            a_idx, b_idx = np.flatnonzero(known & (fam == f)), np.flatnonzero(untaught & (fam == f))
            if len(a_idx) and len(b_idx):
                y = np.r_[np.zeros(len(a_idx)), np.ones(len(b_idx))]
                for sname, s in stats.items():
                    auroc[f][sname].append(float(roc_auc_score(y, np.r_[s[a_idx], s[b_idx]])))
        for f in TAUGHT_W:
            a_idx, b_idx = np.flatnonzero(known & (fam == f)), np.flatnonzero(untaught & (fam == f))
            if len(a_idx) and len(b_idx):
                y = np.r_[np.zeros(len(a_idx)), np.ones(len(b_idx))]
                for sname, s in stats.items():
                    auroc[f][sname].append(float(roc_auc_score(y, np.r_[s[a_idx], s[b_idx]])))

        # Q5: length
        words = np.array([len(t.split()) for t in ttexts])
        flagged_map = np.array([a.status == "unseen" for a in assessed["map"]])
        for i in np.flatnonzero(known):
            length["rows"].append((cases[i].family, int(words[i]), float(best_cos[i]), bool(flagged_map[i])))
        print(f"seed {seed} done", file=sys.stderr, flush=True)

    pct = lambda v: f"{100 * np.mean(v):5.1f}%"
    families = sorted(set(flag["map"]) | set(flag["plain"]))
    print(f"Why do new wordings set off the alarm?  tuning seeds {min(seeds)}-{max(seeds)}, status machine q = {R.HEADLINE_Q}, standard condition\n")
    print("Q1. Stage by stage: how far each wording sits below the taught wordings, in standard deviations of the best cosine to the own skill's examples (0 = as close as a taught wording; negative = farther)")
    print("    wording          raw Qwen   plain (centred)   learned-map reader")
    for f in sorted(shift["raw"]):
        print(f"    {f:15s} {np.mean(shift['raw'][f]):+8.2f} {np.mean(shift['plain'][f]):+12.2f} {np.mean(shift['map'][f]):+16.2f}")
    print("\n    known skills called unseen, and nearest group right:   plain reader | learned-map reader")
    for f in families:
        print(f"    {f:15s} n={len(flag['map'][f]):4d}   unseen {pct(flag['plain'][f])} | {pct(flag['map'][f])}    nearest right {pct(own['plain'][f])} | {pct(own['map'][f])}")
    print("\nQ3. ORACLE (looks at the test wording, a diagnostic and not a fix): false alarm after removing the top k shift directions of that wording, learned-map reader")
    for f in ("S4", "S7", "S5"):
        print(f"    {f}:  " + "   ".join(f"k={k}: top {pct(oracle[(f, 'top')][k])}" + (f" / random {pct(oracle[(f, 'random')][k])}" if k else "") for k in (0, 1, 2, 4, 8)))
    print("\nQ4. AUROC of 'known skill in this wording' (0) against 'never-taught skill in the same wording' (1); 0.5 is chance")
    print("    wording  " + "   ".join(f"{n[:34]:34s}" for n in next(iter(auroc.values()))))
    for f in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"):
        if f in auroc:
            print(f"    {f:7s}  " + "   ".join(f"{np.mean(v):34.3f}" for v in auroc[f].values()))
    print("\nQ5. Length")
    fam_l = np.array([r[0] for r in length["rows"]])
    w, bc, fl = (np.array([r[i] for r in length["rows"]], dtype=float) for i in (1, 2, 3))
    for f in sorted(set(fam_l)):
        s = fam_l == f
        print(f"    {f:15s} words median {np.median(w[s]):5.1f}   false alarm {pct(fl[s])}")
    for label, mask in (("taught wordings S1-S3", np.isin(fam_l, TAUGHT_W)), ("held-out wordings S4-S8", np.isin(fam_l, HELD))):
        rho_c = spearmanr(w[mask], bc[mask])[0]
        terc = np.quantile(w[mask], [1 / 3, 2 / 3])
        parts = [fl[mask & (w <= terc[0])], fl[mask & (w > terc[0]) & (w <= terc[1])], fl[mask & (w > terc[1])]]
        print(f"    {label}: Spearman(words, best cosine) {rho_c:+.2f};  false alarm by length third (short/middle/long): " + " / ".join(pct(p) for p in parts))


if __name__ == "__main__":
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (150, 154)
    main(range(lo, hi + 1))
