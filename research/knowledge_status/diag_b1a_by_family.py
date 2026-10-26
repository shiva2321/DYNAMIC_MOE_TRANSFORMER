"""Side check for E1 (not part of the suite): B1a accuracy per test wording family, dev seed 100, from the cached embeddings. The probe's C is fixed at 0.1
(E1 itself picks it by cross-validation), so the overall number differs a little from the E1 table.

    python diag_b1a_by_family.py > results/E1_DIAG_B1A_BY_FAMILY.txt
"""

import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import ceiling
import reps

warnings.filterwarnings("ignore")
CACHE = Path(__file__).resolve().parent / "_cache" / "reps"


def main(seed: int = 100, rep_ids=("R0_hash", "R1_minilm384", "R4_qwen_L18")) -> None:
    items = ceiling.build_items(seed, "dev", ceiling.SIZES)
    teach, test = items.role("teach"), items.role("test")
    ytr, yte = np.array([i.label for i in teach]), np.array([i.label for i in test])
    fam = np.array([i.family for i in test])
    print(f"B1a accuracy by test wording family, dev seed {seed}, probe C fixed at 0.1 (S4 stories, S5 swapped order, S6 chatter, S7 typos, S8 register; the last three rows are the 'none' class)")
    for rep_id in rep_ids:
        rep = reps.get_rep(rep_id)
        texts = list(dict.fromkeys([i.text for i in items.all]))
        X = reps.cached_encode(rep, texts, CACHE)
        idx = {t: k for k, t in enumerate(texts)}
        Xtr, Xte = X[[idx[i.text] for i in teach]], X[[idx[i.text] for i in test]]
        sc = StandardScaler().fit(Xtr)
        pred = LogisticRegression(C=0.1, max_iter=300, tol=1e-3).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
        print(f"{rep_id}: overall {float((pred == yte).mean()):.3f}")
        for f in sorted(set(fam)):
            m = fam == f
            print(f"   {f:7s} n={int(m.sum()):4d}  accuracy {float((pred[m] == yte[m]).mean()):.2f}")
        print("   per class: " + ", ".join(f"{str(s)} {float((pred[yte == s] == yte[yte == s]).mean()):.2f}" for s in sorted(set(yte))))
        reps.clear_models()


if __name__ == "__main__":
    main()
