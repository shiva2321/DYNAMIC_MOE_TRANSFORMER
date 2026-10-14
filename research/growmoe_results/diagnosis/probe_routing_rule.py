"""Study 2b: why does density routing drain the older operators' inputs to the newer experts?

Signatures are built the way the sequential pool builds them: expert 0 from three operators together (max, add, sub), experts 1-3 from one operator each (mod, div, mul), 256 stored inputs
per expert, kappa chosen by leave-one-out likelihood. Then test inputs of every operator are routed with the product's rules. Also printed: each signature's kappa and its vMF
log-normaliser, which the pool's rule ``accept_then_logf`` adds to the score before comparing experts.

    python probe_routing_rule.py
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

Q = 0.02


def build(xu, op, ops, seed):
    s = VmfSignature(d=A.D_MODEL, max_protos=256, seed=seed)
    idx = np.flatnonzero(np.isin(op, ops))
    idx = np.random.default_rng([seed, 5]).permutation(idx)[:3000 * len(ops)]
    for i in range(0, len(idx), 64):
        s.update(torch.from_numpy(xu[idx[i:i + 64]]))
    s.fit()
    return s


def main():
    tr = L.build_train("FULL", 0, 8000)
    xu = D.unit(tr.x())
    op = tr.op[tr.pi]
    names = ["e0 (max+add+sub)", "e1 (mod)", "e2 (div)", "e3 (mul)"]
    groups = [[A.OPS.index(o) for o in ("max", "add", "sub")], [A.OPS.index("mod")], [A.OPS.index("div")], [A.OPS.index("mul")]]
    sigs = [build(xu, op, g, 11 + i) for i, g in enumerate(groups)]
    print("signature   kappa   log C_d(kappa)   mean cosine of a stored input to its nearest other stored input")
    info = []
    for n, s in zip(names, sigs):
        print(f"{n:<18s} {s.kappa:6.0f}   {s._log_c[s._kappa]:9.1f}   {float(s.nn_similarity().mean()):.3f}")
        info.append({"name": n, "kappa": s.kappa, "log_norm": s._log_c[s._kappa], "nn_cos": float(s.nn_similarity().mean())})
    test = L.build_test(40)
    sel = np.random.default_rng(5).choice(test.n_samples, size=24000, replace=False)
    xt = torch.from_numpy(D.unit(test.x())[sel])
    opt = test.op[test.pi][sel]
    score = torch.stack([s.score(xt) for s in sigs], 1)                                   # log mean exp(kappa cos): no normaliser
    logf = torch.stack([s.score(xt) + s._log_c[s._kappa] for s in sigs], 1)               # what the pool compares
    p = torch.stack([conformal_pvalues(s.cal_scores, s.score(xt)) for s in sigs], 1)
    near = torch.stack([(xt @ s.protos.T).max(dim=1).values for s in sigs], 1)
    rules = {}
    elig = p.double() >= Q
    rules["pool rule: accepted experts, highest log-density (with each expert's normaliser)"] = torch.where(elig.any(1), logf.masked_fill(~elig, float("-inf")).argmax(1), logf.argmax(1))
    rules["highest log-density without the normaliser"] = score.argmax(1)
    rules["highest conformal p-value"] = p.argmax(1)
    rules["nearest stored input"] = near.argmax(1)
    rules["accepted experts, nearest stored input"] = torch.where(elig.any(1), near.masked_fill(~elig, -9).argmax(1), near.argmax(1))
    out = {"signatures": info, "rules": {}}
    own = {A.OPS.index("max"): 0, A.OPS.index("add"): 0, A.OPS.index("sub"): 0, A.OPS.index("mod"): 1, A.OPS.index("div"): 2, A.OPS.index("mul"): 3}
    print("\nshare of each operator's test inputs routed to its own expert (max, add, sub -> e0):")
    print(f"{'rule':<84s} " + " ".join(f"{o:>5s}" for o in ("max", "add", "sub", "mod", "div", "mul")))
    for name, r in rules.items():
        row = [float((r[opt == A.OPS.index(o)] == own[A.OPS.index(o)]).float().mean()) for o in ("max", "add", "sub", "mod", "div", "mul")]
        out["rules"][name] = row
        print(f"{name:<84s} " + " ".join(f"{100 * v:5.1f}" for v in row))
    D.atomic_write_text((HERE / "routing_rule.json"), json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
