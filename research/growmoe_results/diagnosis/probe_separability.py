"""Study 2: can the gate's own density model tell the operators apart? (It sees only the input, never the label or the loss.)

The pool decides "known or new" and "which expert" from one thing: how dense the pool's stored inputs are around this input (a von Mises-Fisher kernel density, one
``VmfSignature`` per expert, 256 stored inputs each, unit-length vectors). If a regime is a different FUNCTION on the same inputs (an operator), the only handle the gate has
is whatever the regime tag contributes to the input vector. This probe builds one signature per operator (and one per operator x output slot) from training examples,
exactly as the pool builds them, and asks:
  1. how well does "highest density" identify the operator (chance 1/6) and the operator x slot group (chance 1/36)?
  2. how often does a signature of ANOTHER operator still accept the input (conformal p >= q), i.e. would a single expert absorb it without any novelty?
  3. geometry: cosine similarity between random inputs of the same operator, of different operators, and the cosine cost of changing only the operator or only one digit.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import diag_lib as D  # noqa: E402
from diag_lib import A, L  # noqa: E402
from growmoe.conformal import conformal_pvalues  # noqa: E402
from growmoe.signature import VmfSignature  # noqa: E402

Q = 0.02          # the false-novelty level the arithmetic pools ran with
N_FIT = 3000      # training examples streamed into each signature (it keeps 256 of them, like the pool)
N_TEST = 24000


def fit_signatures(xu, groups, ids, seed):
    sigs = {}
    for g in ids:
        idx = np.flatnonzero(groups == g)
        rng = np.random.default_rng([seed, int(g), 0x516])
        idx = rng.permutation(idx)[:N_FIT]
        s = VmfSignature(d=A.D_MODEL, max_protos=256, seed=seed * 1000 + int(g))
        for i in range(0, len(idx), 64):
            s.update(torch.from_numpy(xu[idx[i:i + 64]]))
        s.fit()
        sigs[g] = s
    return sigs


def score_matrix(sigs, ids, xt):
    """(n, G) log-density and conformal p-value of every test input under every signature."""
    logf, pval = [], []
    for g in ids:
        s = sigs[g]
        sc = s.score(torch.from_numpy(xt))
        logf.append((sc + s._log_c[s._kappa]).numpy())
        pval.append(conformal_pvalues(s.cal_scores, sc).numpy())
    return np.stack(logf, 1), np.stack(pval, 1)


def main():
    out = {}
    tr = L.build_train("FULL", 0, 8000)
    xtr = D.unit(tr.x())
    op_tr, pos_tr = tr.op[tr.pi], tr.pos
    test = L.build_test(40)
    sel = np.random.default_rng(5).choice(test.n_samples, size=min(N_TEST, test.n_samples), replace=False)
    xte = D.unit(test.x())[sel]
    op_te, pos_te = test.op[test.pi][sel], test.pos[sel]

    # ---- 1/2: operators
    ops = list(range(A.N_OPS))
    sigs = fit_signatures(xtr, op_tr, ops, 0)
    logf, pval = score_matrix(sigs, ops, xte)
    pred = logf.argmax(1)
    acc_op = float((pred == op_te).mean())
    conf = np.array([[float(((op_te == j) & (pred == i)).sum() / max((op_te == j).sum(), 1)) for j in ops] for i in ops])      # [predicted i | true j]
    accept = np.array([[float((pval[op_te == j, i] >= Q).mean()) for j in ops] for i in ops])                                # [signature i accepts | true operator j]
    out["operator"] = {"accuracy_argmax_density": acc_op, "chance": 1 / 6, "confusion_pred_given_true": conf.tolist(), "accept_rate_sig_i_given_true_j": accept.tolist(),
                       "own_accept_mean": float(np.mean(np.diag(accept))), "foreign_accept_mean": float((accept.sum() - np.trace(accept)) / (A.N_OPS * (A.N_OPS - 1)))}
    print(f"operator from highest density: {100 * acc_op:.1f} % (chance 16.7 %)")
    print("signature i accepts an input of operator j (rows i, columns j; own = diagonal):")
    print(np.round(100 * accept).astype(int))

    # ---- (operator x slot) groups
    grp_tr = op_tr * A.N_POS + pos_tr
    grp_te = op_te * A.N_POS + pos_te
    ids = sorted(int(g) for g in np.unique(grp_tr) if (grp_tr == g).sum() >= 200)
    sigs_g = fit_signatures(xtr, grp_tr, ids, 1)
    logf_g, pval_g = score_matrix(sigs_g, ids, xte)
    pred_g = np.array(ids)[logf_g.argmax(1)]
    known = np.isin(grp_te, ids)
    acc_g = float((pred_g[known] == grp_te[known]).mean())
    acc_g_op = float(((pred_g[known] // A.N_POS) == (grp_te[known] // A.N_POS)).mean())
    out["operator_x_slot"] = {"n_groups": len(ids), "accuracy_group": acc_g, "accuracy_operator_part": acc_g_op, "chance_group": 1 / len(ids)}
    print(f"operator x slot group from highest density: {100 * acc_g:.1f} % (chance {100 / len(ids):.1f} %); operator part right {100 * acc_g_op:.1f} %")

    # ---- 3: geometry of the input space the gate works in
    rng = np.random.default_rng(3)
    ia, ib = rng.integers(0, len(xtr), 40000), rng.integers(0, len(xtr), 40000)
    cos = (xtr[ia] * xtr[ib]).sum(1)
    same = op_tr[ia] == op_tr[ib]
    out["geometry"] = {"cos_same_operator": float(cos[same].mean()), "cos_different_operator": float(cos[~same].mean()),
                       "cos_change_only_operator": 1 - 2 / 12, "cos_change_one_digit": 1 - 2 / 12, "note": "12 active coordinates of equal size: changing the operator or one digit moves 2 of them"}
    print(f"mean cosine between random inputs: same operator {cos[same].mean():.3f}, different operator {cos[~same].mean():.3f}; "
          f"changing ONLY the operator -> cos {1 - 2 / 12:.3f}; changing ONLY one digit -> cos {1 - 2 / 12:.3f}")

    # ---- how much of the nearest-neighbour structure is operator-aligned (kNN in the same unit vectors)
    base = torch.from_numpy(xtr[np.random.default_rng(9).choice(len(xtr), 60000, replace=False)])
    base_op = op_tr[np.random.default_rng(9).choice(len(xtr), 60000, replace=False)]
    q = torch.from_numpy(xte[:4000])
    nn_idx = (q @ base.T).argmax(1).numpy()
    out["nearest_neighbour_same_operator"] = float((base_op[nn_idx] == op_te[:4000]).mean())
    print(f"nearest stored input has the same operator: {100 * out['nearest_neighbour_same_operator']:.1f} % (chance 16.7 %)")

    D.atomic_write_text((HERE / "separability.json"), json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
