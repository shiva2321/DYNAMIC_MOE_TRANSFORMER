"""Side check for Task 9 (not part of the suite): where do the two misses of R5 on Qwen layer 26 (B1a 0.92, B1c 0.89) come from? Dev seed 100, cached embeddings.

    python diag_r5_failures.py > results/E1_DIAG_R5_FAILURES.txt

1. B1a by test wording family and by true class, probe C fixed at 0.1 (E1 picks it by cross-validation, so the overall number differs a little).
2. B1c by the untaught skill: how well a part of that skill is told apart from the parts of the taught skills, and which taught skill its parts sit nearest to.
   An untaught skill whose PARENTS are taught (mul with add taught, div with sub taught ...) is "Related" in the design, not "Unseen"; B1c scores it as untaught.
"""

import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import assignments
import ceiling
import learned_map
import reps
import skills

warnings.filterwarnings("ignore")
CACHE = Path(__file__).resolve().parent / "_cache" / "reps"


def main(seed: int = 100, base: str = "R4_qwen_L26") -> None:
    items = ceiling.build_items(seed, "dev", ceiling.SIZES)
    rep = reps.get_rep(base)
    texts = list(dict.fromkeys([i.text for i in items.all]))
    X = reps.cached_encode(rep, texts, CACHE)
    idx = {t: k for k, t in enumerate(texts)}
    emb = lambda g: X[[idx[i.text] for i in g]]
    teach, test = items.role("teach"), items.role("test")
    held = {i.text for i in items.all if i.role in ("anchor", "twin", "partner")}
    pool = [i for i in teach if i.text not in held]
    mapper = learned_map.make_mapper(seed)

    f = mapper(emb(pool), pool, skills.SKILLS)
    ytr, yte = np.array([i.label for i in teach]), np.array([i.label for i in test])
    fam = np.array([i.family for i in test])
    Ztr, Zte = f(emb(teach)), f(emb(test))
    sc = StandardScaler().fit(Ztr)
    pred = LogisticRegression(C=0.1, max_iter=300, tol=1e-3).fit(sc.transform(Ztr), ytr).predict(sc.transform(Zte))
    print(f"R5 on {base}, dev seed {seed}\n\nB1a by test wording family (probe C 0.1): overall {float((pred == yte).mean()):.3f}")
    for g in sorted(set(fam)):
        m = fam == g
        print(f"   {g:7s} n={int(m.sum()):4d}  accuracy {float((pred[m] == yte[m]).mean()):.2f}")
    print("   by class: " + ", ".join(f"{s} {float((pred[yte == s] == yte[yte == s]).mean()):.2f}" for s in sorted(set(yte))))
    wrong = Counter((t, p) for t, p in zip(yte, pred) if t != p)
    print("   most common mistakes (true -> said): " + ", ".join(f"{t}->{p} x{n}" for (t, p), n in wrong.most_common(6)))

    # B1c, replaying the draw of ceiling.run_ceiling
    arng = np.random.default_rng([seed, 0xE1C])
    test_sk = [i for i in test if i.label != "none"]
    Xt0, lab = emb(test_sk), np.array([i.label for i in test_sk])
    nones = [i for i in pool if i.label == "none"]
    auc: dict = {s: [] for s in skills.SKILLS}
    related_auc: dict = {True: [], False: []}
    nearest: dict = {s: Counter() for s in skills.SKILLS}
    for _ in range(ceiling.SIZES.assignments):
        asg = assignments.draw_assignment(arng)
        taught = [s for s in skills.SKILLS if asg[s] != "unseen"]
        rows, keys, used = [], [], []
        for s in taught:
            mine = [i for i in teach if i.label == s][: assignments.COUNTS[asg[s]]]
            rows.append(emb(mine))
            keys += [(s, i.family) for i in mine]
            used += [i for i in mine if i.text not in held]
        T0 = np.vstack(rows)
        g = mapper(np.vstack([emb(used), emb(nones)]), used + nones, taught)
        T, Xt = g(T0), g(Xt0)
        c = T.mean(axis=0)
        U = lambda z: z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        groups = sorted(set(keys))
        P = np.vstack([U(U(T - c)[[k for k, key in enumerate(keys) if key == grp]].mean(axis=0, keepdims=True)) for grp in groups])
        sims = U(Xt - c) @ P.T
        d = 1.0 - sims.max(axis=1)
        is_taught = np.array([asg[s] != "unseen" for s in lab])
        for u in [s for s in skills.SKILLS if asg[s] == "unseen"]:
            m = (lab == u) | is_taught
            auc[u].append(roc_auc_score(is_taught[m].astype(int), -d[m]))
            has_parent = bool(skills.PARENTS[u] & set(taught))
            related_auc[has_parent].append(auc[u][-1])
            near = [groups[k][0] for k in sims[lab == u].argmax(axis=1)]
            nearest[u].update(near)
    print("\nB1c by the untaught skill (AUROC of that skill's parts against the taught skills' parts, mean over the teaching assignments where it is untaught)")
    for s in skills.SKILLS:
        tot = sum(nearest[s].values())
        print(f"   {s}: AUROC {np.mean(auc[s]):.2f} (n={len(auc[s])})  nearest taught skill: " + ", ".join(f"{k} {v / tot:.0%}" for k, v in nearest[s].most_common(3)))
    print(f"   untaught skill whose parent skill is taught (Related in the design): AUROC {np.mean(related_auc[True]):.2f} (n={len(related_auc[True])}); no parent taught: {np.mean(related_auc[False]):.2f} (n={len(related_auc[False])})")


if __name__ == "__main__":
    main()
