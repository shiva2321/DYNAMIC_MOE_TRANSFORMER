"""THROWAWAY spike on DEV seeds (100-102): does the batch-level test separate OVERLAPPING regimes on REAL text embeddings?   (entry E-024)

The synthetic streams build their regimes as shifted means, which is exactly what a distance-of-means test looks for. Here the regimes are real text categories with the natural overlap
of real language: 20 Newsgroups (20 categories; several are near neighbours, e.g. comp.sys.ibm.pc.hardware and comp.sys.mac.hardware), AG News (4 topics), and the 20 most frequent
intents of banking77 (many intents are paraphrases of each other's situation). Frozen all-MiniLM-L6-v2 -> fixed random projection to 128 -> unit length (the pool's own HybridFrontend).
Clean setting as in A1: one signature (<= 256 stored prototypes) per category from reference documents; test batches of held-out documents from the same category (should NOT be
rejected) or from another category (should be). Per-sample test = conformal p-values with Benjamini-Hochberg at q = 0.02 (the pool's gate); batch test = permutation test on the
distance of means against the stored prototypes (999 permutations). For each category the foreign categories are its 4 nearest by centre cosine plus 4 random ones.

    python research/growmoe_results/rv3/spike_real_text.py
"""

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np                                                                      # noqa: E402
import pandas as pd                                                                     # noqa: E402
import torch                                                                            # noqa: E402

import spike_set_tests as T                                                             # noqa: E402
from growmoe.conformal import benjamini_hochberg                                        # noqa: E402
from growmoe.signature import VmfSignature                                              # noqa: E402

torch.set_num_threads(3)
CACHE = Path(os.environ.get("RV3_CACHE", HERE / "_cache_real"))
HUB = Path.home() / ".cache" / "huggingface" / "hub"
SEEDS = (100, 101, 102)
N_NEAR, N_RAND = 4, 4                         # foreign categories per category: nearest by centre cosine, and random
D = 128
DATASETS = {                                   # name: (n_ref, n_test per category, batch sizes)
    "20news": (256, 256, (8, 16, 64)),
    "agnews": (256, 512, (8, 16, 64)),
    "banking77": (64, 64, (8, 16, 32)),
}


def load_texts(name: str):
    if name == "20news":
        snap = next((HUB / "datasets--SetFit--20_newsgroups" / "snapshots").iterdir())
        df = pd.read_json(snap / "train.jsonl", lines=True)
        return df["text"].astype(str).tolist(), df["label"].to_numpy(), None
    if name == "agnews":
        snap = next((HUB / "datasets--ag_news" / "snapshots").iterdir())
        df = pd.read_parquet(snap / "data" / "train-00000-of-00001.parquet")
        return df["text"].astype(str).tolist(), df["label"].to_numpy(), None
    snap = next((HUB / "datasets--banking77" / "snapshots").iterdir())
    df = pd.read_parquet(snap / "data" / "train-00000-of-00001.parquet")
    top = df["label"].value_counts().index[:20].tolist()
    keep = df["label"].isin(top).to_numpy()
    return df["text"][keep].astype(str).tolist(), df["label"].to_numpy()[keep], top


def embed(name: str, need: int):
    """One fixed pool of up to `need` documents per category, embedded once and cached on disk."""
    f = CACHE / f"{name}.npz"
    if f.exists():
        z = np.load(f)
        return z["x"], z["y"]
    from sentence_transformers import SentenceTransformer
    texts, y, _ = load_texts(name)
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:need] for c in np.unique(y)])
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    st.max_seq_length = 128
    emb = st.encode([texts[i] for i in idx], batch_size=128, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True).cpu().float()
    P = torch.from_numpy((np.random.default_rng([0, 0xE1B]).standard_normal((384, D)) / np.sqrt(D)).astype(np.float32))
    z = emb @ P
    x = (z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)).numpy()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, x=x, y=y[idx])
    return x, y[idx]


def auroc_low_is_foreign(own, foreign):
    return T.auroc(list(own), list(foreign))


def run(name: str, seed: int, x_all: np.ndarray, y_all: np.ndarray) -> dict:
    n_ref, n_test, sizes = DATASETS[name]
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    classes = [c for c in np.unique(y_all) if (y_all == c).sum() >= n_ref + n_test]
    ref, tst = {}, {}
    for c in classes:
        idx = rng.permutation(np.where(y_all == c)[0])
        ref[c], tst[c] = torch.from_numpy(x_all[idx[:n_ref]]), torch.from_numpy(x_all[idx[n_ref:n_ref + n_test]])
    mean = {c: torch.nn.functional.normalize(ref[c].mean(0), dim=0) for c in classes}
    sigs = {}
    for c in classes:
        s = VmfSignature(D, 256, seed=int(c))
        s.update(ref[c])
        s.fit()
        sigs[c] = s
    rows = []
    for c in classes:
        cos = {c2: float(mean[c] @ mean[c2]) for c2 in classes if c2 != c}
        near = sorted(cos, key=cos.get, reverse=True)[:N_NEAR]
        rest = [c2 for c2 in classes if c2 != c and c2 not in near]
        foreign = near + list(rng.choice(rest, size=min(N_RAND, len(rest)), replace=False))
        sig = sigs[c]
        for b in sizes:
            for kind, src in [("own", [c])] + [("foreign", [c2]) for c2 in foreign]:
                xs = tst[src[0]]
                for lo in range(0, len(xs) - b + 1, b):
                    xb = xs[lo:lo + b]
                    rows.append({"b": b, "kind": kind, "cls": int(c), "other": int(src[0]), "cos": cos.get(src[0], 1.0), "persample": float(benjamini_hochberg(sig.pvalue(xb), 0.02).float().mean()),
                                 "p_batch": T.meandist_p(sig.protos, xb, g)})
    return {"dataset": name, "seed": seed, "n_classes": len(classes), "rows": rows}


def summarize(res: list) -> str:
    out = []
    for name in DATASETS:
        rr = [r for r in res if r["dataset"] == name]
        if not rr:
            continue
        rows = [dict(row, seed=r["seed"]) for r in rr for row in r["rows"]]
        cos_f = [row["cos"] for row in rows if row["kind"] == "foreign"]
        out.append(f"\n=== {name}: {rr[0]['n_classes']} categories, seeds {[r['seed'] for r in rr]}; centre cosine of tested foreign pairs: min {min(cos_f):.2f}, median {np.median(cos_f):.2f}, max {max(cos_f):.2f} ===")
        out.append("batch   per-sample test: foreign inputs flagged (own / foreign), AUROC  | batch test: false alarm @0.05 / @0.001, detection @0.05 / @0.001, AUROC")
        for b in sorted({row["b"] for row in rows}):
            own = [row for row in rows if row["b"] == b and row["kind"] == "own"]
            frg = [row for row in rows if row["b"] == b and row["kind"] == "foreign"]
            fa = [np.mean([row["p_batch"] <= a for row in own]) for a in (0.05, 0.001)]
            de = [np.mean([row["p_batch"] <= a for row in frg]) for a in (0.05, 0.001)]
            au_p = T.auroc([row["persample"] for row in own], [row["persample"] for row in frg], higher_is_pos=True)
            au_b = T.auroc([row["p_batch"] for row in own], [row["p_batch"] for row in frg])
            out.append(f"b={b:3d}   {np.mean([row['persample'] for row in own]):.3f} / {np.mean([row['persample'] for row in frg]):.3f}, AUROC {au_p:.3f}   |   {fa[0]:.3f} / {fa[1]:.3f},  {de[0]:.3f} / {de[1]:.3f},  AUROC {au_b:.3f}"
                       f"   (own n={len(own)}, foreign n={len(frg)})")
            # the hard part: foreign batches the per-sample test barely notices
            blind = [row for row in frg if row["persample"] < 0.30]
            if blind:
                out.append(f"        pairs the per-sample test is nearly blind to (flags < 30 % of the inputs): {len(blind)} of {len(frg)} foreign batches ({100 * len(blind) / len(frg):.0f} %); "
                           f"batch test detects {100 * np.mean([row['p_batch'] <= 0.05 for row in blind]):.1f} % @0.05 and {100 * np.mean([row['p_batch'] <= 0.001 for row in blind]):.1f} % @0.001; "
                           f"their centre cosine median {np.median([row['cos'] for row in blind]):.2f}")
            hard = sorted(frg, key=lambda row: -row["cos"])[:max(1, len(frg) // 5)]
            out.append(f"        the 20 % most similar foreign pairs (centre cosine >= {min(row['cos'] for row in hard):.2f}): per-sample flags {np.mean([row['persample'] for row in hard]):.3f} of the inputs, "
                       f"batch test detects {100 * np.mean([row['p_batch'] <= 0.001 for row in hard]):.1f} % @0.001")
    return "\n".join(out)


if __name__ == "__main__":
    res, t0 = [], time.time()
    for name, (n_ref, n_test, sizes) in DATASETS.items():
        x_all, y_all = embed(name, n_ref + n_test)
        print(f"{name}: {len(x_all)} embedded documents, {len(np.unique(y_all))} categories ({time.time() - t0:.0f} s)", flush=True)
        for seed in SEEDS:
            res.append(run(name, seed, x_all, y_all))
            print(f"  {name} seed {seed} done ({time.time() - t0:.0f} s)", flush=True)
    (HERE / "spike_real_text.json").write_text(json.dumps(res), encoding="utf-8")
    text = summarize(res)
    (HERE / "spike_real_text_summary.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
