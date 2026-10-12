"""REFERENCE ONLY (written by Claude while reviewing R12). Re-implement; do not copy numbers. Run from the repo root:
    python scratch/claude_ref_r12_checks.py det|growth|learn|paths|grads|control
It reproduces: cross-process determinism, what the growth policy really decides (test (f) construction), the learning effect with controls,
the predict-vs-observe path mismatch in the bootstrap state, and whether head parameters are really frozen."""
import sys, hashlib, warnings, math
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import torch
from growmoe.config import PhyloCfg, GrowthCfg
from growmoe.experts import MLPExpert
from growmoe.pools.phylo import PhyloMoE
from growmoe.stack import StackMoE, StackGrowthPolicy


def units(n, d, seed):
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, d))
    return torch.from_numpy(x / np.linalg.norm(x, axis=1, keepdims=True)).float()


def digest(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def check_det():
    d, seed = 32, 42
    head_cfg = PhyloCfg(d=d, hidden=32, n_classes=3, seed=seed, route_rule="accept_then_logf", q=0.02)
    stack = StackMoE(d=d, head_pool=PhyloMoE(head_cfg), seed=seed)
    stack.add_layer()
    stack.spawn_infant(0, seed=seed + 1)
    x = units(50, d, seed + 2)
    y = torch.from_numpy(np.random.default_rng(seed + 3).integers(0, 3, 50))
    for _ in range(3):
        stack.observe(x, y)
    p = stack.predict(x).probs
    infant_w = [q.detach().clone() for q in stack.layers[0].experts[1].parameters()] if "1" in stack.layers[0].experts else []
    print("probs digest", digest(p), "| infant weights digest", digest(torch.cat([w.flatten() for w in infant_w])) if infant_w else "n/a",
          "| global seed", torch.initial_seed())


def teacher_task(sd, d=32, c=4, hidden_t=16, depth_t=2, n_tr=400, n_val=100):
    teacher = MLPExpert(d_in=d, hidden=hidden_t, depth=depth_t, n_classes=c, seed=sd * 10)
    teacher.mark_seen(torch.arange(c))
    x_tr, x_val = units(n_tr, d, sd * 10 + 1), units(n_val, d, sd * 10 + 2)
    with torch.no_grad():
        return x_tr, teacher(x_tr).argmax(1), x_val, teacher(x_val).argmax(1)


def make_stack(sd, d=32, c=4, mode="trainable", sigma=None, no_promote=False):
    cfg = PhyloCfg(d=d, hidden=16, n_classes=c, seed=sd * 100 + 1, route_rule="accept_then_logf", q=0.02)
    st = StackMoE(d=d, head_pool=PhyloMoE(cfg), seed=sd * 100 + 2,
                  max_infant_samples=10**9 if no_promote else 300,
                  plateau_tol=-1.0 if no_promote else 0.02)
    li = st.add_layer()
    e = st.spawn_infant(li, seed=sd * 100 + 3)
    if mode == "frozen_identity":
        st.mark_mature(li, e)
    elif mode == "frozen_random":
        ex = st.layers[li].experts[e]
        g = torch.Generator().manual_seed(sd * 7 + 1)
        with torch.no_grad():
            for b in ex.blocks:
                b.w2.copy_(torch.randn(b.w2.shape, generator=g) * sigma)
        st.mark_mature(li, e)
    return st, li, e


def eval_loss(st, x_val, y_val):
    with torch.no_grad():
        h, _ = st._forward_layers(x_val)
        pr = st.head_pool.predict(h)
        return float(torch.nn.functional.nll_loss(torch.log(torch.clamp(pr.probs, min=1e-7)), y_val))


def run_arm(sd, mode, steps=10, sigma=None, no_promote=False):
    x_tr, y_tr, x_val, y_val = teacher_task(sd)
    st, li, e = make_stack(sd, mode=mode, sigma=sigma, no_promote=no_promote)
    rng = np.random.default_rng(sd)
    for _ in range(steps):
        idx = torch.from_numpy(rng.integers(0, len(x_tr), 64))
        st.observe(x_tr[idx], y_tr[idx])
    prom = [ev for ev in st.events if ev.kind == "promote_rep"]
    ex = st.layers[li].experts[e]
    w2 = torch.cat([b.w2.detach().flatten() for b in ex.blocks])
    return eval_loss(st, x_val, y_val), (prom[0].step if prom else None), float(w2.std()), float(w2.abs().max())


def tci(diffs):
    from scipy import stats
    a = np.array(diffs, dtype=float)
    m, se = a.mean(), a.std(ddof=1) / math.sqrt(len(a))
    t = stats.t.ppf(0.975, len(a) - 1)
    return m, m - t * se, m + t * se


def check_learn():
    for label, seeds in (("agent's seeds", [100, 101, 102, 103, 104]), ("other dev seeds", list(range(105, 115)))):
        fr, tr, rr, np_ = [], [], [], []
        steps_to_prom, wstd = [], []
        for sd in seeds:
            f, _, _, _ = run_arm(sd, "frozen_identity")
            t, prom_step, s, amax = run_arm(sd, "trainable")
            r, _, _, _ = run_arm(sd, "frozen_random", sigma=max(s, 1e-4))
            n, _, _, _ = run_arm(sd, "trainable", no_promote=True)
            fr.append(f); tr.append(t); rr.append(r); np_.append(n)
            steps_to_prom.append(prom_step); wstd.append(s)
        print(f"== {label}  (seeds {seeds[0]}..{seeds[-1]})  held-out NLL, lower is better")
        print(f"   frozen identity   {np.mean(fr):.4f}")
        print(f"   trainable (loop)  {np.mean(tr):.4f}   promotion step per seed {steps_to_prom}   trained w2 std {np.mean(wstd):.4f}")
        print(f"   frozen RANDOM res {np.mean(rr):.4f}   (same shape, w2 ~ N(0, trained std), never trained)")
        print(f"   trainable no-promotion {np.mean(np_):.4f}")
        for name, a, b in (("identity - trainable", fr, tr), ("random - trainable", rr, tr), ("identity - random", fr, rr), ("identity - trainable(no promo)", fr, np_)):
            m, lo, hi = tci([x - y for x, y in zip(a, b)])
            print(f"   paired {name:32s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")


def check_growth():
    d, c = 32, 4
    for sd in [100, 101, 102, 103, 104]:
        rng = np.random.default_rng(sd)
        x_tr, x_va = units(400, d, sd * 10 + 1), units(100, d, sd * 10 + 2)
        t1 = MLPExpert(d_in=d, hidden=8, depth=1, n_classes=c, seed=sd * 10 + 3); t1.mark_seen(torch.arange(c))
        y1, v1 = t1(x_tr).argmax(1), t1(x_va).argmax(1)
        s1 = MLPExpert(d_in=d, hidden=8, depth=1, n_classes=c, seed=sd * 10 + 4); s1.mark_seen(torch.arange(c))
        o1 = torch.optim.Adam(s1.parameters(), lr=3e-3)
        for _ in range(150):
            idx = torch.from_numpy(rng.integers(0, len(x_tr), 64)); s1.train_step(x_tr[idx], y1[idx], o1)
        d1 = StackGrowthPolicy(GrowthCfg(train_steps=20, n_boot=100, min_gain=0.03)).step(s1, x_tr, y1, x_va, v1, rng=rng)
        t3 = MLPExpert(d_in=d, hidden=16, depth=3, n_classes=c, seed=sd * 10 + 5); t3.mark_seen(torch.arange(c))
        y3, v3 = t3(x_tr).argmax(1), t3(x_va).argmax(1)
        s3 = MLPExpert(d_in=d, hidden=4, depth=1, n_classes=c, seed=sd * 10 + 6); s3.mark_seen(torch.arange(c))
        o3 = torch.optim.Adam(s3.parameters(), lr=3e-3)
        for _ in range(80):
            idx = torch.from_numpy(rng.integers(0, len(x_tr), 64)); s3.train_step(x_tr[idx], y3[idx], o3)
        d3 = StackGrowthPolicy(GrowthCfg(train_steps=20, n_boot=100, min_gain=0.03)).step(s3, x_tr, y3, x_va, v3, rng=rng)
        g = lambda dec: {k: round(v["gain"], 3) for k, v in (dec.tried or {}).items()}
        print(f"seed {sd}: depth-1 -> kind={d1.kind} reason={d1.reason} | depth-3 -> kind={d3.kind} reason={d3.reason} gains={g(d3)}")


def check_paths():
    d = 32
    cfg = PhyloCfg(d=d, hidden=32, n_classes=4, seed=5, route_rule="accept_then_logf", q=0.02)
    st = StackMoE(d=d, head_pool=PhyloMoE(cfg), seed=6)
    st.add_layer()
    e1 = st.spawn_infant(0, seed=7)
    x = units(64, d, 8); y = torch.randint(0, 4, (64,))
    for _ in range(3):
        st.observe(x, y)
    st.layers[0].mark_mature(e1)                      # now one scorable mature expert
    e2 = st.spawn_infant(0, seed=9)                  # new unscored infant next to a scorable expert
    xt = units(40, d, 10)
    pred_paths = st.predict(xt).meta["paths"]
    obs_paths = st.observe(xt, None).meta["paths"]
    same = pred_paths == obs_paths
    print("predict==observe paths with a fresh unscored infant next to a scorable expert:", same)
    if not same:
        n_diff = sum(1 for a, b in zip(pred_paths, obs_paths) if a != b)
        print("   samples whose path differs:", n_diff, "of", len(pred_paths), "| example predict", pred_paths[0], "observe", obs_paths[0])


def check_grads():
    d = 32
    cfg = PhyloCfg(d=d, hidden=32, n_classes=4, seed=5, route_rule="accept_then_logf", q=0.02)
    hp = PhyloMoE(cfg)
    print("head pool is nn.Module:", isinstance(hp, torch.nn.Module))
    st = StackMoE(d=d, head_pool=hp, seed=6)
    st.add_layer(); st.spawn_infant(0, seed=7)
    x = units(64, d, 8); y = torch.randint(0, 4, (64,))
    st.observe(x, y); st.observe(x, y)
    n_with_grad = sum(1 for rec in hp.experts.values() for p in rec.module.parameters() if p.grad is not None)
    n_total = sum(1 for rec in hp.experts.values() for p in rec.module.parameters())
    print(f"head expert parameters with a populated .grad after observe: {n_with_grad}/{n_total}")
    print("head parameters all requires_grad=True after observe:", all(p.requires_grad for rec in hp.experts.values() for p in rec.module.parameters()))




def check_control():
    """Valid capacity-without-learning control: the layer's ROUTED expert (expert 0) gets random, never-trained residual weights."""
    for label, seeds in (("agent seeds", [100, 101, 102, 103, 104]), ("other dev seeds", list(range(105, 115)))):
        for sigma in (0.0087, 0.05, 0.2):
            res = {"identity": [], "trainable": [], "random_routed": []}
            for sd in seeds:
                for mode in res:
                    x_tr, y_tr, x_val, y_val = teacher_task(sd)
                    st, li, e = make_stack(sd, mode="frozen_identity" if mode != "trainable" else "trainable")
                    if mode == "random_routed":
                        ex = st.layers[li].experts[0]
                        g = torch.Generator().manual_seed(sd * 13 + 5)
                        with torch.no_grad():
                            for b in ex.blocks:
                                b.w2.copy_(torch.randn(b.w2.shape, generator=g) * sigma)
                    rng = np.random.default_rng(sd)
                    for _ in range(10):
                        idx = torch.from_numpy(rng.integers(0, len(x_tr), 64))
                        st.observe(x_tr[idx], y_tr[idx])
                    res[mode].append(eval_loss(st, x_val, y_val))
            m_it = tci([a - b for a, b in zip(res["identity"], res["trainable"])])
            m_ir = tci([a - b for a, b in zip(res["identity"], res["random_routed"])])
            m_rt = tci([a - b for a, b in zip(res["random_routed"], res["trainable"])])
            print(f"{label:16s} sigma={sigma}: identity-trainable {m_it[0]:+.4f} | identity-random {m_ir[0]:+.4f} [{m_ir[1]:+.4f},{m_ir[2]:+.4f}] | random-trainable {m_rt[0]:+.4f} [{m_rt[1]:+.4f},{m_rt[2]:+.4f}]")


if __name__ == "__main__":
    {"det": check_det, "growth": check_growth, "learn": check_learn, "paths": check_paths, "grads": check_grads, "control": check_control}[sys.argv[1]]()
