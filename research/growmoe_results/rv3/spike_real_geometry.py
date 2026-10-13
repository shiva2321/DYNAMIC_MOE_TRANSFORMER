"""THROWAWAY spike (dev seed 100): does changing the GEOMETRY of the embedding space help the single-input gate on real text?   (entry E-024, addendum)

Same clean test as spike_real_text.py, on the cached 128-d embeddings, but the space is first transformed:
  raw                     the pool's own input (MiniLM -> random projection -> unit length)
  centered, all data      subtract the global mean, renormalise (an upper bound: it uses statistics of every category)
  whitened, all data      ZCA whitening with shrinkage, renormalise (upper bound)
  centered, early         the mean of three random categories' reference documents only (what a stream would know early)
  whitened, early         ZCA from those three categories only
Batches of 8 documents; foreign categories = 2 nearest + 2 random.

    python research/growmoe_results/rv3/spike_real_geometry.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                                                                      # noqa: E402

import spike_real_text as R                                                             # noqa: E402

R.N_NEAR, R.N_RAND = 2, 2


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def zca(x_fit, shrink=0.05):
    mu = x_fit.mean(0)
    xc = x_fit - mu
    w, u = np.linalg.eigh(xc.T @ xc / len(xc))
    return mu, u @ np.diag(1.0 / np.sqrt(w + shrink * w.mean())) @ u.T


def transforms(x, y):
    rng = np.random.default_rng(7)
    early = np.isin(y, rng.choice(np.unique(y), size=3, replace=False))
    mu_all = x.mean(0)
    mu_e = x[early].mean(0)
    _, w_all = zca(x)
    mu_w, w_e = zca(x[early])
    return {"raw": x, "centered, all data": unit(x - mu_all), "whitened, all data": unit((x - mu_all) @ w_all), "centered, early": unit(x - mu_e),
            "whitened, early": unit((x - mu_w) @ w_e)}


if __name__ == "__main__":
    for name, bs in (("20news", (8,)), ("banking77", (8,))):
        n_ref, n_test, _ = R.DATASETS[name]
        R.DATASETS[name] = (n_ref, n_test, bs)
        x, y = R.embed(name, n_ref + n_test)
        print(f"\n##### {name}", flush=True)
        for label, xt in transforms(x, y).items():
            res = R.run(name, 100, xt.astype(np.float32), y)
            rows = res["rows"]
            own = [r for r in rows if r["kind"] == "own"]
            frg = [r for r in rows if r["kind"] == "foreign"]
            au = R.T.auroc([r["persample"] for r in own], [r["persample"] for r in frg], higher_is_pos=True)
            blind = np.mean([r["persample"] < 0.30 for r in frg])
            hard = sorted(frg, key=lambda r: -r["cos"])[:max(1, len(frg) // 5)]
            print(f"{label:22s} per-sample: foreign inputs flagged {np.mean([r['persample'] for r in frg]):.3f}, AUROC {au:.3f}, pairs nearly invisible {100 * blind:4.1f} %   | batch (8 documents): "
                  f"false alarm @0.05 {np.mean([r['p_batch'] <= 0.05 for r in own]):.3f}, detection @0.001 {100 * np.mean([r['p_batch'] <= 0.001 for r in frg]):5.1f} %, hardest 20 % pairs {100 * np.mean([r['p_batch'] <= 0.001 for r in hard]):5.1f} %", flush=True)
