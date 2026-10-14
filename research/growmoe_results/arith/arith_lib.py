"""Learners and evaluation for the arithmetic benchmark (data format: growmoe/arith.py). Nothing in here knows any arithmetic.

Every learner sees the same examples ((problem, output slot) -> symbol) and answers a problem by answering all its slots; a problem is right only if every slot is right.
Dense networks, the transformer, the lookup and the oracle-routed experts run on the GPU when there is one; GrowMoE (PhyloMoE) runs on the CPU.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from growmoe import arith as A  # noqa: E402
from growmoe.accounting import pool_report  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LENGTHS = {"FULL": (1, 2, 3, 4, 5), "HOLE": (1, 2, 4, 5), "SHORT": (1, 2, 3)}
OPS_ALL = tuple(range(A.N_OPS))
OP_ORDER = tuple(A.OPS.index(o) for o in ("max", "add", "sub", "mod", "div", "mul"))      # operator-sequential curriculum: easy to hard
TEST_SEED = 987654                                                                         # the test grid is the SAME for every learner, regime and seed


# ------------------------------------------------------------------------------------------------ data
def build_train(regime: str, seed: int, n_per_op: int) -> A.ProblemSet:
    lens = LENGTHS[regime]
    rng = np.random.default_rng([seed, 0xA17, sum(map(ord, regime))])
    return A.ProblemSet(**A.sample_problems(rng, n_per_op, OPS_ALL, lens, lens, "train"))


def build_test(per_cell: int) -> A.ProblemSet:
    """Held-out problems, ``per_cell`` for every (operator, length of A, length of B): 6 x 5 x 5 cells, ordered op-major then la then lb."""
    rng = np.random.default_rng([TEST_SEED, per_cell])
    cols = {k: [] for k in ("op", "a", "b", "la", "lb")}
    for oi in OPS_ALL:
        for la in range(1, 6):
            for lb in range(1, 6):
                p = A.sample_problems(rng, per_cell, (oi,), (la,), (lb,), "test")
                for k in cols:
                    cols[k].append(p[k])
    return A.ProblemSet(**{k: np.concatenate(v) for k, v in cols.items()})


def build_seen(train: A.ProblemSet, per_op: int, seed: int) -> A.ProblemSet:
    rng = np.random.default_rng([seed, 0x5EE])
    idx = np.concatenate([rng.choice(np.flatnonzero(train.op == oi), size=min(per_op, int((train.op == oi).sum())), replace=False) for oi in OPS_ALL])
    return train.subset(np.isin(np.arange(len(train)), idx))


def op_blocks(train: A.ProblemSet, seed: int):
    """Sample indices of the training examples, one block per operator in curriculum order, shuffled inside a block."""
    rng = np.random.default_rng([seed, 0xB10C])
    for oi in OP_ORDER:
        idx = np.flatnonzero(train.op[train.pi] == oi)
        yield oi, rng.permutation(idx)


# ------------------------------------------------------------------------------------------------ learners
class Learner:
    name = "learner"
    cpu = False                                                          # True: runs on the CPU (do not start many at once)

    def fit(self, tr: A.ProblemSet, seed: int) -> dict:
        raise NotImplementedError

    def fit_sequential(self, tr: A.ProblemSet, seed: int, after_block=None) -> dict:
        raise NotImplementedError(f"{self.name} has no sequential mode")

    def predict(self, ps: A.ProblemSet) -> dict:
        """-> {"pred": (n_samples,) class per example, optional "expert", "unknown", "p_nov"}"""
        raise NotImplementedError


def _mlp(width: int, depth: int) -> nn.Module:
    layers, d = [], A.D_MODEL
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.GELU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, A.N_CLASS))


def _train_net(net, x, y, epochs, lr, bs, gen=None, replay=None):
    """Adam with a cosine decay; ``x`` and ``y`` already on DEV. ``replay`` = (x, y) mixed into every batch (same size) when given."""
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    n = x.shape[0]
    steps = epochs * math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=x.device, generator=gen)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            xb, yb = x[b], y[b]
            if replay is not None and replay[0].shape[0] > 0:
                r = torch.randint(0, replay[0].shape[0], (len(b),), device=x.device)
                xb, yb = torch.cat([xb, replay[0][r]]), torch.cat([yb, replay[1][r]])
            loss = F.cross_entropy(net(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    net.eval()


@torch.no_grad()
def _logits(net, x: np.ndarray, chunk=65536) -> np.ndarray:
    out = []
    for i in range(0, len(x), chunk):
        out.append(net(torch.from_numpy(x[i:i + chunk]).to(DEV)).argmax(-1).cpu().numpy())
    return np.concatenate(out)


class DenseMLP(Learner):
    """One dense network on everything (the standard supervised baseline). ``replay`` > 0 in the sequential mode keeps that many old examples to mix in."""

    def __init__(self, width=1024, depth=3, epochs=40, lr=2e-3, bs=512, replay=0, name=None):
        self.width, self.depth, self.epochs, self.lr, self.bs, self.replay = width, depth, epochs, lr, bs, replay
        self.name = name or f"MLP-w{width}d{depth}"

    def fit(self, tr, seed):
        torch.manual_seed(seed)
        self.net = _mlp(self.width, self.depth).to(DEV)
        t0 = time.time()
        _train_net(self.net, torch.from_numpy(tr.x()).to(DEV), torch.from_numpy(tr.y).long().to(DEV), self.epochs, self.lr, self.bs)
        return {"seconds": time.time() - t0, "params": sum(p.numel() for p in self.net.parameters()), "stored_vectors": 0}

    def fit_sequential(self, tr, seed, after_block=None):
        torch.manual_seed(seed)
        self.net = _mlp(self.width, self.depth).to(DEV)
        t0 = time.time()
        keep_x, keep_y = [], []
        for k, (oi, idx) in enumerate(op_blocks(tr, seed)):
            x, y = torch.from_numpy(tr.x(idx)).to(DEV), torch.from_numpy(tr.y[idx]).long().to(DEV)
            rep = None
            if self.replay and keep_x:
                rep = (torch.cat(keep_x), torch.cat(keep_y))
            _train_net(self.net, x, y, self.epochs, self.lr, self.bs, replay=rep)             # each operator gets as many epochs over ITS data as in the mixed protocol
            if self.replay:                                              # reservoir-like: a uniform random subset of everything seen so far, capped
                keep_x.append(x), keep_y.append(y)
                allx, ally = torch.cat(keep_x), torch.cat(keep_y)
                if allx.shape[0] > self.replay:
                    sel = torch.randperm(allx.shape[0], device=DEV)[:self.replay]
                    allx, ally = allx[sel], ally[sel]
                keep_x, keep_y = [allx], [ally]
            if after_block:
                after_block(k, oi)
        return {"seconds": time.time() - t0, "params": sum(p.numel() for p in self.net.parameters()),
                "stored_vectors": int(sum(t.shape[0] for t in keep_x))}

    def predict(self, ps):
        return {"pred": _logits(self.net, ps.x())}


class OracleExperts(Learner):
    """One small network per operator (``group="op"``) or per (operator, output slot) (``"oppos"``), TOLD which one to use (the routing ceiling for expert pools)."""

    def __init__(self, group="op", width=512, depth=2, epochs=60, lr=2e-3, bs=256):
        self.group, self.width, self.depth, self.epochs, self.lr, self.bs = group, width, depth, epochs, lr, bs
        self.name = f"Oracle-{group}-experts"

    def _gid(self, ps, idx=None):
        op, pos = (ps.op[ps.pi], ps.pos) if idx is None else (ps.op[ps.pi[idx]], ps.pos[idx])
        return op if self.group == "op" else op * A.N_POS + pos

    def fit(self, tr, seed):
        torch.manual_seed(seed)
        t0 = time.time()
        self.nets = {}
        gid = self._gid(tr)
        x_all, y_all = torch.from_numpy(tr.x()).to(DEV), torch.from_numpy(tr.y).long().to(DEV)
        for g in np.unique(gid):
            sel = torch.from_numpy(np.flatnonzero(gid == g)).to(DEV)
            net = _mlp(self.width, self.depth).to(DEV)
            _train_net(net, x_all[sel], y_all[sel], self.epochs, self.lr, self.bs)
            self.nets[int(g)] = net
        return {"seconds": time.time() - t0, "params": sum(p.numel() for n in self.nets.values() for p in n.parameters()), "stored_vectors": 0, "experts": len(self.nets)}

    def fit_sequential(self, tr, seed, after_block=None):
        torch.manual_seed(seed)
        t0 = time.time()
        self.nets = {}
        gid = self._gid(tr)
        x_all, y_all = torch.from_numpy(tr.x()).to(DEV), torch.from_numpy(tr.y).long().to(DEV)
        for k, (oi, idx) in enumerate(op_blocks(tr, seed)):               # each block only ever touches its own experts: no interference by construction
            for g in np.unique(gid[idx]):
                sel = torch.from_numpy(idx[gid[idx] == g]).to(DEV)
                net = _mlp(self.width, self.depth).to(DEV)
                _train_net(net, x_all[sel], y_all[sel], self.epochs, self.lr, self.bs)
                self.nets[int(g)] = net
            if after_block:
                after_block(k, oi)
        return {"seconds": time.time() - t0, "params": sum(p.numel() for n in self.nets.values() for p in n.parameters()), "stored_vectors": 0, "experts": len(self.nets)}

    def predict(self, ps):
        gid = self._gid(ps)
        x = ps.x()
        pred = np.zeros(len(gid), dtype=np.int64)
        for g, net in self.nets.items():
            m = np.flatnonzero(gid == g)
            if len(m):
                pred[m] = _logits(net, x[m])
        return {"pred": pred}


class Knn(Learner):
    """Nearest stored example (cosine), a pure lookup: right on what it has seen, a guess on the rest."""
    name = "kNN-1"

    def fit(self, tr, seed):
        t0 = time.time()
        x = torch.from_numpy(tr.x()).to(DEV)
        self.x = F.normalize(x, dim=1)
        self.y = torch.from_numpy(tr.y).long().to(DEV)
        return {"seconds": time.time() - t0, "params": 0, "stored_vectors": int(x.shape[0])}

    @torch.no_grad()
    def predict(self, ps):
        x = F.normalize(torch.from_numpy(ps.x()).to(DEV), dim=1)
        out = []
        for i in range(0, x.shape[0], 512):
            out.append(self.y[(x[i:i + 512] @ self.x.T).argmax(1)].cpu().numpy())
        return {"pred": np.concatenate(out)}


class TinyGPT(nn.Module):
    def __init__(self, vocab=19, d=128, heads=4, layers=3, ff=512, max_len=24):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, A.N_CLASS)

    def forward(self, seq):
        L = seq.shape[1]
        h = self.tok(seq) + self.pos(torch.arange(L, device=seq.device))[None]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=seq.device)
        return self.head(self.ln(self.blocks(h, mask=mask, is_causal=True)))


class Transformer(Learner):
    """A small decoder-only transformer on the digit tokens (the standard sequence-model way to learn arithmetic from examples); answers by greedy decoding.
    It writes the answer most-significant digit first, exactly as the slots are ordered for everyone else; no reversal trick, no scratchpad."""

    def __init__(self, epochs=100, lr=1.5e-3, bs=256, layers=3, d=128, name="Transformer"):
        self.epochs, self.lr, self.bs, self.layers, self.d, self.name = epochs, lr, bs, layers, d, name

    def fit(self, tr, seed):
        torch.manual_seed(seed)
        self.net = TinyGPT(d=self.d, layers=self.layers, ff=4 * self.d).to(DEV)
        inp, out = A.sequence_tokens(tr)
        seq = torch.from_numpy(np.concatenate([inp, out], axis=1)).long().to(DEV)         # (n, 24)
        valid = torch.from_numpy(A.valid_output_tokens(tr)).to(DEV)                       # only the operator's own slots count for the loss
        n = seq.shape[0]
        steps = self.epochs * math.ceil(n / self.bs)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps, pct_start=0.05)
        t0 = time.time()
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=DEV)
            for i in range(0, n, self.bs):
                b = perm[i:i + self.bs]
                logits = self.net(seq[b, :-1])[:, 12:23]                                     # positions 12..22 predict output tokens 0..10
                loss = (F.cross_entropy(logits.reshape(-1, A.N_CLASS), seq[b, 13:].reshape(-1), reduction="none") * valid[b].reshape(-1)).sum() / valid[b].sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                sched.step()
        self.net.eval()
        return {"seconds": time.time() - t0, "params": sum(p.numel() for p in self.net.parameters()), "stored_vectors": 0}

    @torch.no_grad()
    def predict(self, ps):
        inp, _ = A.sequence_tokens(ps)
        gen = []
        for i in range(0, len(ps), 4096):
            seq = torch.from_numpy(inp[i:i + 4096]).long().to(DEV)
            for _ in range(11):
                nxt = self.net(seq)[:, -1].argmax(-1, keepdim=True)
                seq = torch.cat([seq, nxt], dim=1)
            gen.append(seq[:, 13:].cpu().numpy())                                             # (m, 11): sign, digit 9 ... digit 0
        gen = np.concatenate(gen)
        slot = np.where(ps.pos == A.SIGN_POS, 0, 10 - ps.pos)                                 # token index of each example's slot
        return {"pred": gen[ps.pi, slot]}


class Pool(Learner):
    """GrowMoE (PhyloMoE) with its product defaults (``set_gate`` = the stage-3 batch gate). Sees the examples in batches of 64, one pass."""
    cpu = True

    def __init__(self, steps=5, set_gate=False, name=None, **cfg_kw):
        self.steps, self.set_gate, self.cfg_kw = steps, set_gate, cfg_kw
        self.name = name or ("GrowMoE+stage3" if set_gate else "GrowMoE")

    def _new(self, seed):
        from growmoe.config import PhyloCfg
        from growmoe.pools.phylo import PhyloMoE
        torch.set_num_threads(3)
        cfg = PhyloCfg(seed=seed, d=A.D_MODEL, n_classes=A.N_CLASS, inner_steps=self.steps, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf",
                       set_gate=self.set_gate, **self.cfg_kw)
        self.pool = PhyloMoE(cfg)

    def _feed(self, tr, idx):
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            if len(b):
                self.pool.observe(torch.from_numpy(tr.x(b)), torch.from_numpy(tr.y[b]).long())

    def _info(self, t0):
        rep = pool_report(self.pool, A.D_MODEL)
        s = self.pool.summary()
        return {"seconds": time.time() - t0, "params": rep.params, "stored_vectors": rep.stored_vectors, "memory_bytes": rep.memory_bytes, "experts": rep.n_experts,
                "events": {k: sum(1 for e in self.pool.events if e.kind == k) for k in sorted({e.kind for e in self.pool.events})},
                "set_stats": s.get("set_stats"), "predict_macs": rep.predict_macs}

    def fit(self, tr, seed):
        self._new(seed)
        t0 = time.time()
        self._feed(tr, np.random.default_rng([seed, 0x5F1]).permutation(tr.n_samples))
        return self._info(t0)

    def fit_sequential(self, tr, seed, after_block=None):
        self._new(seed)
        t0 = time.time()
        for k, (oi, idx) in enumerate(op_blocks(tr, seed)):
            self._feed(tr, idx)
            if after_block:
                after_block(k, oi)
        return self._info(t0)

    @torch.no_grad()
    def predict(self, ps):
        x = ps.x()
        pred, exp, unk, pn = [], [], [], []
        for i in range(0, len(x), 4096):
            o = self.pool.predict(torch.from_numpy(x[i:i + 4096]))
            pred.append(o.probs.argmax(-1).numpy())
            exp.append(o.expert.numpy())
            unk.append(o.unknown.numpy())
            pn.append(o.p_nov.numpy())
        return {"pred": np.concatenate(pred), "expert": np.concatenate(exp), "unknown": np.concatenate(unk), "p_nov": np.concatenate(pn)}


def _pool_predict_grouped(self, ps: A.ProblemSet) -> dict:
    """The SAME trained pool, read out differently: the output slots of one problem are a group (they share the operands and the operator) and are answered together by the
    expert most of them chose on their own (stage 3b group vote with its per-input veto). No retraining."""
    x = ps.x()
    cfg = self.pool.cfg
    saved = (cfg.group_vote, cfg.group_min)
    cfg.group_vote, cfg.group_min = True, 5
    pred, expert = np.zeros(len(x), dtype=np.int64), np.zeros(len(x), dtype=np.int64)
    try:
        with torch.no_grad():
            for lo in range(0, len(ps), 600):
                idx = np.flatnonzero((ps.pi >= lo) & (ps.pi < lo + 600))
                o = self.pool.predict_grouped(torch.from_numpy(x[idx]), torch.from_numpy(ps.pi[idx]).long())
                pred[idx], expert[idx] = o.probs.argmax(-1).numpy(), o.expert.numpy()
    finally:
        cfg.group_vote, cfg.group_min = saved
    return {"pred": pred, "expert": expert}


Pool.predict_grouped = _pool_predict_grouped


class SmallNet(Learner):
    """Continual baselines from growmoe.baselines on the same examples (a plain network that just keeps training, or one with a replay buffer)."""
    cpu = True

    def __init__(self, kind="replay", steps=20, hidden=128, replay_size=2048):
        self.kind, self.steps, self.hidden, self.replay_size = kind, steps, hidden, replay_size
        self.name = f"{'Replay' if kind == 'replay' else 'FineTune'}Net-w{hidden}"

    def fit_sequential(self, tr, seed, after_block=None):
        from growmoe.baselines import FineTune, ReplayNet
        from growmoe.config import CommonCfg
        torch.set_num_threads(3)
        cfg = CommonCfg(seed=seed, d=A.D_MODEL, n_classes=A.N_CLASS, inner_steps=self.steps, hidden=self.hidden)
        self.m = ReplayNet(cfg, replay_size=self.replay_size) if self.kind == "replay" else FineTune(cfg)
        t0 = time.time()
        for k, (oi, idx) in enumerate(op_blocks(tr, seed)):
            for i in range(0, len(idx), 64):
                b = idx[i:i + 64]
                self.m.observe(torch.from_numpy(tr.x(b)), torch.from_numpy(tr.y[b]).long())
            if after_block:
                after_block(k, oi)
        return {"seconds": time.time() - t0, "params": sum(p.numel() for p in self.m.net.parameters()), "stored_vectors": int(getattr(getattr(self.m, "buffer", None), "n", 0))}

    @torch.no_grad()
    def predict(self, ps):
        x = ps.x()
        return {"pred": np.concatenate([self.m.predict(torch.from_numpy(x[i:i + 4096])).probs.argmax(-1).numpy() for i in range(0, len(x), 4096)])}


# ------------------------------------------------------------------------------------------------ evaluation
def cell_tables(ps: A.ProblemSet, pred: np.ndarray, per_cell: int):
    """(6, 5, 5) exact-match and digit accuracy per (operator, length of A, length of B) for a test grid built by ``build_test``."""
    ex, dg = ps.exact(pred).astype(np.float64), ps.digit_acc(pred)
    return ex.reshape(A.N_OPS, 5, 5, per_cell).mean(-1), dg.reshape(A.N_OPS, 5, 5, per_cell).mean(-1)


def sample_cells(ps: A.ProblemSet) -> np.ndarray:
    """Cell id (operator, la, lb) of every example of a test grid."""
    return (ps.op[ps.pi] * 25 + (ps.la[ps.pi] - 1) * 5 + (ps.lb[ps.pi] - 1)).astype(np.int64)


def evaluate(learner: Learner, test: A.ProblemSet, per_cell: int, seen: A.ProblemSet = None):
    """Score one trained learner on the held-out grid (and on a sample of its own training problems). Returns (summary dict, raw per-example arrays for the npz)."""
    t0 = time.time()
    out = learner.predict(test)
    ex, dg = cell_tables(test, out["pred"], per_cell)
    res = {"grid_exact": ex.tolist(), "grid_digit": dg.tolist(), "predict_seconds": time.time() - t0}
    raw = {"correct": (out["pred"] == test.y)}
    for k in ("expert", "unknown", "p_nov"):
        if k in out:
            raw[k] = out[k]
    if hasattr(learner, "predict_grouped"):
        g = learner.predict_grouped(test)
        gex, gdg = cell_tables(test, g["pred"], per_cell)
        res["grid_exact_group"], res["grid_digit_group"] = gex.tolist(), gdg.tolist()
        raw["g_correct"], raw["g_expert"] = (g["pred"] == test.y), g["expert"]
    if "unknown" in out:
        cells = sample_cells(test)
        res["grid_unknown"] = (np.bincount(cells, weights=out["unknown"].astype(np.float64), minlength=A.N_OPS * 25) / np.bincount(cells, minlength=A.N_OPS * 25)).reshape(A.N_OPS, 5, 5).tolist()
    if seen is not None:
        so = learner.predict(seen)
        e, d = seen.exact(so["pred"]), seen.digit_acc(so["pred"])
        res["seen_exact_by_op"] = [float(e[seen.op == oi].mean()) for oi in OPS_ALL]
        res["seen_digit_by_op"] = [float(d[seen.op == oi].mean()) for oi in OPS_ALL]
        if "unknown" in so:
            res["seen_unknown_rate"] = float(so["unknown"].mean())
    return res, raw
