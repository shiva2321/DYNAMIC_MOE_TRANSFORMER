"""Spike C: does a better text/code front-end fix S1? Swap ONLY the text and code regimes (pydoc text, stdlib code) from
hashed n-grams to a frozen pretrained sentence encoder (all-MiniLM-L6-v2, local cache, offline), projected to d = 128.
Everything else (pool code, configs, seeds, other regimes) is unchanged. Throwaway diligence script."""
import sys
import time

from dil_common import *  # noqa

from growmoe.bench import run_stream
from growmoe.config import PhyloCfg
from growmoe.frontend import HashFrontend
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams import s1_types

SEEDS = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 10))
NAMES = ["digits", "wine", "breast_cancer", "sinusoid", "ar2", "pydoc_text", "stdlib_code"]


class HybridFrontend:
    """Numeric arrays -> the usual hash frontend. Lists of strings -> frozen MiniLM embedding -> fixed random projection to 128."""

    def __init__(self, d=128, seed=0):
        from sentence_transformers import SentenceTransformer
        self.d = d
        self.hash = HashFrontend(d=d, seed=seed)
        self.st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        rng = np.random.default_rng([seed, 0xE1B])
        self.P = torch.from_numpy((rng.standard_normal((384, d)) / np.sqrt(d)).astype(np.float32))
        self.cache = {}

    def __call__(self, items):
        if not isinstance(items, (np.ndarray, torch.Tensor)) and len(items) and isinstance(items[0], (str, bytes, bytearray)):
            texts = [s if isinstance(s, str) else bytes(s).decode("utf-8", "replace") for s in items]
            miss = [t for t in dict.fromkeys(texts) if t not in self.cache]
            if miss:
                emb = self.st.encode(miss, batch_size=128, convert_to_tensor=True, show_progress_bar=False,
                                     normalize_embeddings=True).cpu().float()
                for t, e in zip(miss, emb):
                    self.cache[t] = e
            e = torch.stack([self.cache[t] for t in texts])
            z = e @ self.P
            return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return self.hash(items)


def one_nn_regime_acc(stream):
    """Own-regime 1-NN accuracy: test sample -> regime of its most similar training sample (the T8 diagnostic)."""
    xs, rs = [], []
    seen = set()
    for seg in stream.segments:
        if seg.first_visit and seg.regime not in seen:
            seen.add(seg.regime)
            xs.append(seg.x)
            rs.append(torch.full((seg.x.shape[0],), seg.regime))
    X, R = torch.cat(xs), torch.cat(rs)
    out = {}
    for r, reg in stream.regimes.items():
        nn_idx = (reg.x_te @ X.T).argmax(dim=1)
        out[r] = float((R[nn_idx] == r).float().mean())
    return out


t0 = time.time()
fe_h = HybridFrontend(d=128, seed=0)
print(f"encoder loaded ({time.time()-t0:.0f}s)", flush=True)
acc = {k: [] for k in ("A_hash", "A_minilm", "kNN_hash", "kNN_minilm")}
per_regime = {k: [] for k in acc}
sep = {"hash": [], "minilm": []}
for sd in SEEDS:
    s_hash = s1_types(seed=sd, n_train=400, n_test=150)
    s_mini = s1_types(seed=sd, frontend=fe_h, n_train=400, n_test=150)
    sep["hash"].append(one_nn_regime_acc(s_hash))
    sep["minilm"].append(one_nn_regime_acc(s_mini))
    for tag, st in (("hash", s_hash), ("minilm", s_mini)):
        res = run_stream(lambda: PhyloMoE(PhyloCfg(seed=sd, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf")), st, seed=sd)
        acc[f"A_{tag}"].append(res.accuracy_stats["avg"])
        per_regime[f"A_{tag}"].append([res.final_regime_acc[r] for r in range(7)])
        hold = []

        def mk():
            p = KnnPool(st.n_classes, 128, k=5, capacity=None, seed=sd)
            hold.append(p)
            return p

        res = run_stream(mk, st, seed=sd)
        acc[f"kNN_{tag}"].append(res.accuracy_stats["avg"])
        per_regime[f"kNN_{tag}"].append([res.final_regime_acc[r] for r in range(7)])
    print(f"seed {sd} done ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== S1 final average accuracy (%), {len(SEEDS)} seeds, mean [95% CI] ===")
for k, v in acc.items():
    print(f"{k:12s} {fmt(v)}")
print("\n=== Per-regime final accuracy (%), mean over seeds ===")
print(f"{'regime':14s} " + " ".join(f"{k:>11s}" for k in per_regime))
for r in range(7):
    print(f"{NAMES[r]:14s} " + " ".join(f"{100*np.mean([row[r] for row in per_regime[k]]):11.1f}" for k in per_regime))
print("\n=== Own-regime 1-NN separability (%), mean over seeds ===")
print(f"{'regime':14s} {'hash':>8s} {'minilm':>8s}")
for r in range(7):
    print(f"{NAMES[r]:14s} {100*np.mean([d[r] for d in sep['hash']]):8.1f} {100*np.mean([d[r] for d in sep['minilm']]):8.1f}")
