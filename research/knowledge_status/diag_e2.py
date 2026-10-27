"""Side check for Task 11 (not part of the suite): WHERE do the false "unseen" alarms and the misses of the headline reader come from? Status machine, q = 0.05, standard condition,
reader R5_R4_qwen_L26, dev seeds 100 to 129 (cached embeddings). Nothing here changes a setting; it only splits the numbers of E2_TABLE.md.

    PYTHONPATH=<repo root> python diag_e2.py > results/E2_DIAG.txt

1. False unseen by wording family and by skill, for the taught skills (the taught wordings S1 to S3 are the in-distribution reference).
2. Which untaught skills are found, and which taught group absorbs the ones that are missed.
3. Other content split: everyday sentences, the chatter around S6 requests, traps, never-taught content.
4. How far the wordings sit from the taught prototypes (mean best cosine to the own group's prototypes) and the median conformal p-value per family: a systematic shift shows as a median p far below 0.5.
5. Local evidence of Solid parts by wording against the Thin skill (why the Thin-vs-Solid AUROC of E2 is lower than the count ratio suggests).
"""

import sys
from collections import defaultdict

import numpy as np

import assignments
import evaluate as EV
import run_e2 as R
import skills
import status as ST


def main(seeds=range(100, 130), rep_id="R5_R4_qwen_L26") -> None:
    fam_fu, skill_fu = defaultdict(list), defaultdict(list)
    found, absorbed = defaultdict(list), defaultdict(lambda: defaultdict(int))
    other_rates = defaultdict(list)
    cos_by_fam, p_by_fam = defaultdict(list), defaultdict(list)
    ev_solid_fam, ev_thin = defaultdict(list), []
    for seed in seeds:
        world = R.build_world(seed, "dev", R.SIZES)
        P, index = R.prepare(world, rep_id)
        V = lambda texts: P[[index[t] for t in texts]]
        groups = {s: ST.SkillGroup(s, V([e.span for e in world.teach[s]]), len(world.teach[s]), None, world.comp[s]["accuracy"]) for s in world.taught}
        other = ST.SkillGroup("other", V([i.text for i in world.teach_none]), len(world.teach_none), None, 1.0)
        cal_vec, cal_grp, _, _ = R.calibration_arrays(world, V)
        m = ST.StatusMachine(groups, other, cal_vec, cal_grp, q=R.HEADLINE_Q, seed=seed)
        cases, tvec = world.test_cases, V(world.test_texts)
        out = m.assess_many(tvec)
        protos = {n: g.protos / np.linalg.norm(g.protos, axis=1, keepdims=True) for n, g in groups.items()}
        for k, (a, c) in enumerate(zip(out, cases)):
            flagged = a.status == "unseen"
            if c.kind == "arith" and c.status in ("solid", "thin"):
                fam_fu[c.family].append(flagged)
                if c.family in assignments.TEST_FAMILIES:
                    skill_fu[(c.skill, c.status)].append(flagged)
                cos_by_fam[c.family].append(float((tvec[k] @ protos[c.skill].T).max()))
                p_by_fam[c.family].append(a.p_value)
                if not flagged and a.status in ("solid", "thin"):
                    (ev_solid_fam[c.family] if c.status == "solid" else ev_thin).append(a.evidence)
            elif c.kind == "arith" and c.status == "unseen":
                found[c.skill].append(flagged)
                if not flagged:
                    absorbed[c.skill][a.skill or "other"] += 1
            elif c.kind in ("other", "trap", "unseen_content"):
                other_rates[(c.kind, c.family)].append(flagged)
        print(f"seed {seed} done", file=sys.stderr, flush=True)
    pct = lambda v: f"{100 * np.mean(v):5.1f}%"
    print(f"Diagnosis of E2, {rep_id}, status machine q = {R.HEADLINE_Q}, standard condition, seeds {min(seeds)}-{max(seeds)}\n")
    print("1. False unseen of taught skills (solid or thin truth), by wording family")
    for fam in sorted(fam_fu):
        print(f"   {fam:3s} n={len(fam_fu[fam]):5d}  {pct(fam_fu[fam])}" + ("   (held-out wording)" if fam in assignments.TEST_FAMILIES else "   (taught wording)"))
    print("   by skill and taught state, held-out wordings only:")
    for key in sorted(skill_fu):
        print(f"   {key[0]:4s} {key[1]:5s} n={len(skill_fu[key]):5d}  {pct(skill_fu[key])}")
    print("\n2. Untaught skills: found as unseen, and (when missed) which group took the part")
    for s in skills.SKILLS:
        if found[s]:
            miss = absorbed[s]
            tot = sum(miss.values())
            print(f"   {s:4s} n={len(found[s]):5d} found {pct(found[s])}   missed parts taken by: " + (", ".join(f"{k} {v / tot:.0%}" for k, v in sorted(miss.items(), key=lambda kv: -kv[1])) if tot else "none"))
    print("\n3. Other content called unseen")
    for key in sorted(other_rates):
        print(f"   {key[0]:15s} {key[1]:7s} n={len(other_rates[key]):5d}  {pct(other_rates[key])}")
    print("\n4. Distance of the wordings from the taught prototypes (taught skills): mean best cosine to the own group's prototypes, and the median conformal p-value (0.5 is what in-distribution parts give)")
    for fam in sorted(cos_by_fam):
        print(f"   {fam:3s} mean best cosine {np.mean(cos_by_fam[fam]):.3f}   median p {np.median(p_by_fam[fam]):.3f}")
    print("\n5. Local evidence of parts a skill covers (taught count: Solid 2000, Thin 20)")
    for fam in sorted(ev_solid_fam):
        v = ev_solid_fam[fam]
        print(f"   Solid, {fam:3s} n={len(v):5d} median {np.median(v):7.1f}   10th percentile {np.percentile(v, 10):7.1f}")
    print(f"   Thin  all wordings n={len(ev_thin):5d} median {np.median(ev_thin):7.1f}   90th percentile {np.percentile(ev_thin, 90):7.1f}")


if __name__ == "__main__":
    main()
