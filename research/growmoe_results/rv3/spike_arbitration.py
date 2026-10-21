"""THROWAWAY spike on DEV seeds (100-102): is there anything to gain from SMARTER test-time arbitration?   (gatekeeper research, entry E-023)

The pool routes each test input with `accept_then_logf`: among experts whose conformal p-value is >= q take the one with the highest log-density. Literature (FLEX, TOOD,
confidence-based task prediction) suggests two upgrades: use the experts' own confidence, and mix several experts with calibrated weights instead of picking one.
The train-time owner bound measured earlier says the headroom is small (S1: 81.3 % owner-routed vs 78.2 % exact, about 1 point elsewhere); this spike measures what
the cheap upgrades actually capture. Variants, all on the same trained pool and test inputs:

  hard_default   accept_then_logf (checked against `pool.predict`)
  hard_conf      among accepting experts take the one with the highest max softmax probability
  hard_ent       ... the lowest predictive entropy
  soft_T         mixture of the accepting experts' probabilities, weights softmax(log-density / T), T in {0.5, 2, 8, 32}
  soft_top2_T    the same restricted to the top-2 accepting experts
  owner_oracle   each regime's inputs go to the expert most of that regime's inputs reach (a bound for routing among the existing experts)

    python research/growmoe_results/rv3/spike_arbitration.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rv2a"))
sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402

import run_rv2a as A                                                                    # noqa: E402
import spike_set_gate as G                                                              # noqa: E402
from growmoe.config import PhyloCfg                                                     # noqa: E402

torch.set_num_threads(3)
STREAMS, SEEDS = ("S1", "S2-30", "F-15", "OV-0.3"), (100, 101, 102)
STEPS = {"func": 5, "perm": 20, "types": 200}                                           # tuned A_vote budgets (tuning_rv2a.json)
TEMPS = (0.5, 2.0, 8.0, 32.0)


def run(name: str, seed: int, variant: str = "vote") -> dict:
    st = A.get_stream(name, seed, False)
    pool = G.SetGatePool(PhyloCfg(router=False, spawn_rule="probe", growth_rule="probe", seed=seed, d=st.d, n_classes=st.n_classes, inner_steps=STEPS[A.family(name)], q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf"),
                         set_cfg=G.VARIANTS[variant])
    for x, y, ctx in st.algo_batches(64, seed):
        pool.observe(x, y, ctx)
    regs = list(st.regimes.values())
    x = torch.cat([r.x_te for r in regs])
    y = torch.cat([r.y_te for r in regs])
    regime = torch.cat([torch.full((len(r.y_te),), i, dtype=torch.long) for i, r in enumerate(regs)])
    xn = pool._normalize(x.float())
    eids = list(pool.experts)
    P, LogF = pool._gate_matrices(xn, eids)
    with torch.no_grad():
        probs = torch.stack([pool.experts[e].module.proba(xn) for e in eids])           # (E, n, C)
    q = pool.cfg.q
    elig = P.double() >= q                                                              # (n, E)
    none = ~elig.any(1)
    elig_or_all = torch.where(none[:, None], torch.ones_like(elig), elig)
    rows = torch.arange(len(y))

    def acc_of_choice(idx):
        return float((probs[idx, rows].argmax(-1) == y).float().mean())

    def acc_of_mix(w):                                                                  # w: (n, E) weights
        mix = (w.T[:, :, None] * probs).sum(0)
        return float((mix.argmax(-1) == y).float().mean())

    out = {"stream": name, "seed": seed, "variant": variant, "experts": len(eids), "mean_eligible": float(elig.sum(1).float().mean()), "share_multi_eligible": float((elig.sum(1) > 1).float().mean())}
    default_idx = torch.where(elig.any(1), LogF.masked_fill(~elig, -float("inf")).argmax(1), LogF.argmax(1))
    ref = (pool.predict(x).probs.argmax(1) == y).float().mean()
    out["hard_default"] = acc_of_choice(default_idx)
    assert abs(out["hard_default"] - float(ref)) < 1e-9, "re-implementation of accept_then_logf differs from pool.predict"
    conf = probs.max(-1).values.T                                                       # (n, E)
    ent = -(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log()).sum(-1).T
    out["hard_conf"] = acc_of_choice(conf.masked_fill(~elig_or_all, -float("inf")).argmax(1))
    out["hard_ent"] = acc_of_choice((-ent).masked_fill(~elig_or_all, -float("inf")).argmax(1))
    for T in TEMPS:
        z = (LogF / T).masked_fill(~elig_or_all, -float("inf"))
        out[f"soft_{T}"] = acc_of_mix(torch.softmax(z, 1))
        top = z.topk(min(2, z.shape[1]), dim=1)
        w2 = torch.zeros_like(z).scatter_(1, top.indices, torch.softmax(top.values, 1))
        out[f"soft_top2_{T}"] = acc_of_mix(w2)
    owner = torch.stack([torch.mode(default_idx[regime == r]).values for r in range(len(regs))])
    out["owner_oracle"] = acc_of_choice(owner[regime])
    # set-level (chunk) routing: each regime's test inputs arrive in chunks of 64 from ONE source (a deployment assumption, like the batch vote); route the chunk jointly
    chunks = [c for r in range(len(regs)) for c in [idx[lo:lo + 64] for idx in [(regime == r).nonzero().squeeze(1)] for lo in range(0, len(idx), 64)] if len(c) >= 8]
    covered = torch.cat(chunks)

    def acc_chunks(choose):
        return sum(int((probs[choose(c), c].argmax(-1) == y[c]).sum()) for c in chunks) / len(covered)

    out["hard_chunked"] = float((probs[default_idx[covered], covered].argmax(-1) == y[covered]).float().mean())              # the per-sample rule on the same inputs
    out["chunk_accept_vote"] = acc_chunks(lambda c: int(torch.mode(default_idx[c]).values))                                   # majority of the per-sample choices
    out["chunk_logf"] = acc_chunks(lambda c: int(LogF[c].sum(0).argmax()))                                                    # largest summed log-density
    m_stored = torch.stack([pool.experts[e].meta["sig"].protos.mean(0) for e in eids])
    m_recent = torch.stack([torch.cat(list(pool.recent[e])).mean(0) if pool.recent.get(e) else pool.experts[e].meta["sig"].protos.mean(0) for e in eids])
    out["chunk_mean_stored"] = acc_chunks(lambda c: int((m_stored - xn[c].mean(0)).norm(dim=1).argmin()))                   # nearest stored-prototype mean
    out["chunk_mean_recent"] = acc_chunks(lambda c: int((m_recent - xn[c].mean(0)).norm(dim=1).argmin()))                   # nearest recent-training mean

    def group_acc(chunk_list, choose, veto=False):
        """Route each group jointly to choose(c) (None = per sample). veto=True: inputs the chosen expert's own per-sample gate rejects keep their per-sample choice."""
        ok = used = 0
        for c in chunk_list:
            e = choose(c)
            if e is None:
                idx = default_idx[c]
            else:
                idx = torch.full((len(c),), e, dtype=torch.long)
                if veto:
                    idx = torch.where(elig[c, e], idx, default_idx[c])
                used += 1
            ok += int((probs[idx, c].argmax(-1) == y[c]).sum())
        return ok / sum(len(c) for c in chunk_list), used / len(chunk_list)

    def mixed_chunks(n_major, n_minor):
        R = len(regs)
        return [torch.cat([(regime == r).nonzero().squeeze(1)[:n_major], (regime == (r + 1) % R).nonzero().squeeze(1)[n_major:n_major + n_minor]]) for r in range(R)]

    vote_of = lambda c: int(torch.mode(default_idx[c]).values)                          # noqa: E731  the group's majority of the per-sample choices
    out["vote_pure"], _ = group_acc(chunks, vote_of)
    out["vote_pure_veto"], _ = group_acc(chunks, vote_of, veto=True)
    for b in (8, 16, 32, 128):                                                          # how small may a group be before the vote loses its power?
        cb = [c for r in range(len(regs)) for c in [idx[lo:lo + b] for idx in [(regime == r).nonzero().squeeze(1)] for lo in range(0, len(idx), b)] if len(c) >= min(8, b)]
        out[f"vote_b{b}"], _ = group_acc(cb, vote_of)
        out[f"vote_b{b}_perSample"] = float(sum(int((probs[default_idx[c], c].argmax(-1) == y[c]).sum()) for c in cb) / sum(len(c) for c in cb))
    for tag, (n1, n2) in (("mixed", (32, 32)), ("mixed9010", (58, 6))):
        mc = mixed_chunks(n1, n2)
        out[f"{tag}_perSample"] = float(sum(int((probs[default_idx[c], c].argmax(-1) == y[c]).sum()) for c in mc) / sum(len(c) for c in mc))
        out[f"{tag}_vote"], _ = group_acc(mc, vote_of)
        out[f"{tag}_vote_veto"], _ = group_acc(mc, vote_of, veto=True)

    if pool.set_cfg is not None:                                                        # the set-level router: the batch goes to the nearest expert that is not rejected; else per sample
        def set_route_acc(chunk_list, veto=False):
            """veto=True: inputs the chosen owner's own per-sample gate rejects (p < q) keep their per-sample choice, so a minority inside the batch is not forced onto the owner."""
            ok = used = 0
            for c in chunk_list:
                verdict, owner = pool._set_decide(eids, xn[c])
                if verdict == "known":
                    e = eids.index(owner)
                    idx = torch.full((len(c),), e, dtype=torch.long)
                    if veto:
                        idx = torch.where(elig[c, e], idx, default_idx[c])
                else:
                    idx = default_idx[c]
                ok += int((probs[idx, c].argmax(-1) == y[c]).sum())
                used += verdict == "known"
            return ok / sum(len(c) for c in chunk_list), used / len(chunk_list)

        out["chunk_set_route"], out["used_set_route"] = set_route_acc(chunks)
        out["chunk_set_route_veto"], _ = set_route_acc(chunks, veto=True)
        for tag, (n1, n2) in (("mixed", (32, 32)), ("mixed9010", (58, 6))):
            mc = mixed_chunks(n1, n2)
            out[f"{tag}_hard"] = float(sum(int((probs[default_idx[c], c].argmax(-1) == y[c]).sum()) for c in mc) / sum(len(c) for c in mc))
            out[f"{tag}_set_route"], out[f"used_{tag}_set_route"] = set_route_acc(mc)
            out[f"{tag}_set_route_veto"], _ = set_route_acc(mc, veto=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default=",".join(STREAMS))
    ap.add_argument("--variant", default="vote")
    ap.add_argument("--tag", default="vote")
    args = ap.parse_args()
    streams = args.streams.split(",")
    rows = []
    for name in streams:
        for seed in SEEDS:
            rows.append(run(name, seed, args.variant))
            print(name, seed, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
    (HERE / f"spike_arbitration_{args.tag}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    variants = [k for k in rows[0] if k.startswith(("hard_", "soft_", "owner_", "chunk_", "mixed_", "mixed9010_", "vote_"))]
    print(f"\nmean accuracy % over seeds 100-102, pool trained with variant {args.variant} (difference to hard_default in points)")
    print("stream  E    multi-eligible %   " + "  ".join(f"{v:>14s}" for v in variants))
    for name in streams:
        rr = [r for r in rows if r["stream"] == name]
        cells = []
        for v in variants:
            d = [100 * (r[v] - r["hard_default"]) for r in rr]
            cells.append(f"{100 * np.mean([r[v] for r in rr]):6.1f} ({np.mean(d):+.1f})")
        print(f"{name:7s} {np.mean([r['experts'] for r in rr]):4.1f} {100 * np.mean([r['share_multi_eligible'] for r in rr]):10.1f}       " + "  ".join(f"{c:>14s}" for c in cells))
