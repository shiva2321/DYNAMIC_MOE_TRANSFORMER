"""
real_pilot_v4: Architecture v7 Non-Token Concept-Chunk Cognitive Core Pilot.

Implements the pre-registered dual-track v7 protocol (`results/real_pilot_v4/PREREG.md`)
after approved research upgrades (I1-I4, I8, I9 Teacher-Informed Track 2, I11 Muon guard):
  - Track 1 (Strictly From-Scratch): Symbolic/Text-Chunked BeliefMeaningTask
  - Track 2 (Real Streams `wiki -> story -> code -> math -> forum`, Banking77 overlap-trap,
             I4 verified edit families + disjoint leave-one-family-out, MixedPromptSwitch)
    * Compares From-Scratch baselines AND separately labeled Offline-Teacher-Informed (`_I9`) arms
  - Zero next-byte cross-entropy / perplexity in core objectives or headline claims
  - Runs at most 4 `--mode dev --seed 100` trials logged to `tuning_log.json`,
    followed by ONE frozen held-out `--mode test --seeds 0 1 2` execution.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmoe.belief_training import (
    BeliefMeaningTask,
    BudgetedSurfaceDict,
    ConceptBeliefStore,
    K_CONCEPTS,
    Q_ATTR,
    Q_PROP,
    Q_VAL,
)
from dmoe.continual import LatentQuarantineGate
from dmoe.null_space import NullSpaceProjector
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.spgc import MuonBONGOptimizer, classify_v7_spgc_param_groups
from experiments.real_pilot_data import DOMAINS, RealCorpora, SEQ_LEN
from experiments.v7_edit_families import (
    HELDOUT_FAMILIES,
    TRAIN_FAMILIES,
    _CORPUS_FLIPS,
    _CORPUS_PARAS,
    generate_verified_triple,
)

# Expose legacy alias names purely for read-only diagnostics
_FLIP_RULES = _CORPUS_FLIPS
_PARAPHRASE_RULES = _CORPUS_PARAS

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_ROOT = ROOT / "results" / "real_pilot_v4"

DEFAULT_V7_CFG: Dict[str, Any] = dict(
    d_model=192,
    num_heads=6,
    d_ff=256,
    d_ff_shared=256,
    d_ff_spec=256,
    initial_layers=6,
    router_layer=3,
    initial_experts=1,
    chunk_size=4,
    use_v7_core=True,
    use_multiscale_chunker=True,
    router_mode="causal_context",
    d_route=96,
    router_num_keys=8,
    enable_memory=True,
    auto_write_ltm=True,
    num_ltm_slots=32,
    enable_three_level_attn=True,
    local_window=8,
    chunk_stride=4,
    enable_world_model=True,
    wm_rollout_horizon=3,
    causal_attn=True,
    use_rope=True,
    enable_layer_stm=True,
    hybrid_mode=True,
    layer_stm_every=2,
    enable_full_stack_adapters=True,
    adapter_rank=16,
    enable_thinking=True,
    max_thinking_steps=2,
    skip_threshold=0.10,
    theta_halt=0.20,
    abstain_threshold=0.60,
    auto_lifecycle=True,
    max_experts=6,
    min_mature_steps=160,
    z_nuance_thresh=2.0,
    d_sem_nuance_thresh=2.0,
    nuance_cluster_min=20,
    nuance_cluster_cos=0.72,
    detach_decoder=True,
    lora_decoder_rank=16,
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_texts_fixed(texts: List[bytes], length: int, device: torch.device) -> torch.Tensor:
    out = torch.full((len(texts), length), 32, dtype=torch.long)
    for i, b in enumerate(texts):
        n = min(len(b), length)
        if n > 0:
            out[i, :n] = torch.tensor(list(b[:n]), dtype=torch.long)
    return out.to(device)


def roc_auc_pairs(pos_scores: List[float], neg_scores: List[float]) -> float:
    if not pos_scores or not neg_scores:
        return 0.5
    tot = 0.0
    for p in pos_scores:
        for q in neg_scores:
            if p > q:
                tot += 1.0
            elif p == q:
                tot += 0.5
    return float(tot / (len(pos_scores) * len(neg_scores)))


def make_surface_jitter(raw: bytes) -> Optional[bytes]:
    from experiments.v7_edit_families import _make_corpus_uniform

    res = _make_corpus_uniform(raw, random.Random(0), seq_len=SEQ_LEN)
    return res[1] if res is not None else None


def make_minimal_nuance_flip(raw: bytes) -> Optional[bytes]:
    from experiments.v7_edit_families import _make_corpus_uniform

    res = _make_corpus_uniform(raw, random.Random(0), seq_len=SEQ_LEN)
    return res[2] if res is not None else None


# --------------------------------------------------------------------------------------------------
# Offline Cached Teacher-Geometry Encoder (I9 — Strictly Track 2 Training Only)
# --------------------------------------------------------------------------------------------------
class OfflineTeacherGramEncoder:
    """Computes offline pairwise similarity Gram matrices on training batches using local all-MiniLM-L6-v2."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.enc = AutoModel.from_pretrained(model_name, local_files_only=True).to(DEV)
        self.enc.eval()
        for p in self.enc.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        outs = []
        for i in range(0, len(texts), batch_size):
            batch_s = texts[i : i + batch_size]
            tk = self.tok(batch_s, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEV)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                h = self.enc(**tk).last_hidden_state
            mask = tk["attention_mask"].unsqueeze(-1).float()
            pooled = (h.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            outs.append(F.normalize(pooled, dim=-1))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def batch_gram(self, x_bytes: torch.Tensor) -> torch.Tensor:
        strs = [bytes(row.detach().cpu().numpy().astype(np.uint8).tobytes()).decode("utf-8", "ignore").strip() for row in x_bytes]
        emb = self.encode_texts(strs, batch_size=len(strs))
        return emb @ emb.t()


# --------------------------------------------------------------------------------------------------
# Track 1 helpers: Structured Semantic Byte Chunks (6 x 4B = 24B) + Minimal Paraphrase Variants
# --------------------------------------------------------------------------------------------------
def encode_belief_chunks(x_sym: torch.Tensor, paraphrase: bool = False) -> torch.Tensor:
    """Convert `[op, a, b, c1, c2, q]` -> fixed 24-byte UTF-8 chunks `[B, 24]`."""
    rows = x_sym.detach().cpu().tolist()
    B = len(rows)
    out = torch.empty((B, 24), dtype=torch.long)
    for i, row in enumerate(rows):
        op, a_tok, b_tok, c1_tok, c2_tok, q_tok = row
        a = a_tok - 4
        b = b_tok - 4
        c1 = c1_tok - 4
        c2 = c2_tok - 4
        q = q_tok - 20
        if paraphrase:
            s = f"P{op:02d} X{a:02d} Y{b:02d} M{c1:02d} N{c2:02d} R{q:02d} "
        else:
            s = f"O{op:02d} A{a:02d} B{b:02d} C{c1:02d} D{c2:02d} Q{q:02d} "
        out[i] = torch.tensor(list(s.encode("ascii")), dtype=torch.long)
    return out


# --------------------------------------------------------------------------------------------------
# Track 2 Benchmark Container: Banking77, Verified Edit Families (A-D train / E-F heldout), Switch
# --------------------------------------------------------------------------------------------------
class NonTokenRealBenchmark:
    def __init__(self, corpora: RealCorpora, eval_split: str):
        from datasets import load_dataset

        self.eval_split = eval_split
        self.corpora = corpora

        tr = load_dataset("banking77", split="train")
        rng = random.Random(0)
        if eval_split == "test":
            te = load_dataset("banking77", split="test")
            tr_idx = rng.sample(range(len(tr)), 4000)
            self.b77_tr_txt = [tr[i]["text"].encode("utf-8", "ignore") for i in tr_idx]
            self.b77_tr_y = torch.tensor([tr[i]["label"] for i in tr_idx], dtype=torch.long)
            self.b77_te_txt = [t.encode("utf-8", "ignore") for t in te["text"]]
            self.b77_te_y = torch.tensor(te["label"], dtype=torch.long)
        else:
            idx = list(range(len(tr)))
            rng.shuffle(idx)
            tr_i, te_i = idx[:2000], idx[2000:3200]
            self.b77_tr_txt = [tr[i]["text"].encode("utf-8", "ignore") for i in tr_i]
            self.b77_tr_y = torch.tensor([tr[i]["label"] for i in tr_i], dtype=torch.long)
            self.b77_te_txt = [tr[i]["text"].encode("utf-8", "ignore") for i in te_i]
            self.b77_te_y = torch.tensor([tr[i]["label"] for i in te_i], dtype=torch.long)

        self.b77_tr_by_lbl: Dict[int, List[int]] = {}
        for i, lbl in enumerate(self.b77_tr_y.tolist()):
            self.b77_tr_by_lbl.setdefault(lbl, []).append(i)
        self.b77_tr_labels_valid = [k for k, v in self.b77_tr_by_lbl.items() if len(v) >= 2]

        self.same_pairs, self.diff_pairs = self._build_overlap_trap_pairs(self.b77_te_txt, self.b77_te_y)
        self.tr_same_pairs, self.tr_diff_pairs = self._build_overlap_trap_pairs(self.b77_tr_txt, self.b77_tr_y)

        # Pre-generate & verify TRAIN edit triples (Families A-D strictly, balanced!)
        rng_fam = random.Random(42 if eval_split == "dev" else 99)
        self.train_edit_triples: Dict[str, List[Tuple[bytes, bytes, bytes]]] = {}
        self.train_family_hist: Dict[str, int] = {}
        for d in DOMAINS:
            pool = corpora.train[d]
            max_off = len(pool) - SEQ_LEN - 1
            items: List[Tuple[bytes, bytes, bytes]] = []
            for k_i in range(256):
                off = int((k_i * 997) % max_off)
                raw = pool[off : off + SEQ_LEN].tobytes()
                o_b, p_b, f_b, tag = generate_verified_triple(raw, rng_fam, allowed_families=TRAIN_FAMILIES, seq_len=SEQ_LEN)
                self.train_family_hist[tag] = self.train_family_hist.get(tag, 0) + 1
                items.append((o_b, p_b, f_b))
            self.train_edit_triples[d] = items

        # Build evaluation NuanceFlip triples per domain (balanced TRAIN_FAMILIES on eval split windows)
        self.nuance_pairs: Dict[str, List[Tuple[bytes, bytes, bytes]]] = {}
        for d in DOMAINS:
            triples: List[Tuple[bytes, bytes, bytes]] = []
            for w in corpora.test_windows[d][:48]:
                raw = bytes(w[:SEQ_LEN].numpy().astype(np.uint8).tobytes())
                o_b, p_b, f_b, _ = generate_verified_triple(raw, rng_fam, allowed_families=TRAIN_FAMILIES, seq_len=SEQ_LEN)
                triples.append((o_b, p_b, f_b))
            self.nuance_pairs[d] = triples

        # Build strictly HELD-OUT ANTI-SHORTCUT edit triples (Families E-F, NEVER seen in training!)
        self.heldout_family_pairs: List[Tuple[bytes, bytes, bytes]] = []
        for d in DOMAINS:
            for w in corpora.test_windows[d][48:68]:
                raw = bytes(w[:SEQ_LEN].numpy().astype(np.uint8).tobytes())
                o_b, p_b, f_b, tag = generate_verified_triple(raw, rng_fam, allowed_families=HELDOUT_FAMILIES, seq_len=SEQ_LEN)
                assert tag in HELDOUT_FAMILIES
                self.heldout_family_pairs.append((o_b, p_b, f_b))

        # MixedPromptSwitch pairs ([64B domA | 64B domB]) from held-out eval windows
        sw_list: List[torch.Tensor] = []
        for i in range(32):
            sw_list.append(torch.cat([corpora.test_windows["story"][i, :64], corpora.test_windows["code"][i, :64]], dim=0))
            sw_list.append(torch.cat([corpora.test_windows["wiki"][i, :64], corpora.test_windows["math"][i, :64]], dim=0))
        self.mixed_switch_seqs = torch.stack(sw_list, dim=0)

    @staticmethod
    def _build_overlap_trap_pairs(texts: List[bytes], labels: torch.Tensor) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        words = [set(t.decode("utf-8", "ignore").lower().split()) for t in texts]
        by_lbl: Dict[int, List[int]] = {}
        for i, lbl in enumerate(labels.tolist()):
            by_lbl.setdefault(lbl, []).append(i)
        same_cands, diff_cands = [], []
        for _, idxs in by_lbl.items():
            for a_i in range(min(8, len(idxs))):
                for b_i in range(a_i + 1, min(8, len(idxs))):
                    u, v = idxs[a_i], idxs[b_i]
                    jac = len(words[u] & words[v]) / max(1, len(words[u] | words[v]))
                    same_cands.append((jac, u, v))
        same_cands.sort(key=lambda z: z[0])
        same_pairs = [(u, v) for _, u, v in same_cands[:200]]

        for i in range(0, min(400, len(texts))):
            for j in range(i + 1, min(400, len(texts))):
                if labels[i] != labels[j]:
                    jac = len(words[i] & words[j]) / max(1, len(words[i] | words[j]))
                    if jac >= 0.25:
                        diff_cands.append((jac, i, j))
        diff_cands.sort(key=lambda z: -z[0])
        diff_pairs = [(u, v) for _, u, v in diff_cands[:200]]
        return same_pairs, diff_pairs

    def sample_stream_contrastive_batch(
        self, domain: str, gen_np: np.random.Generator, rng_py: random.Random
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Equal-budget 16-sequence sensory batch:
          - 6 items (0..5): 3 verified paraphrase pairs from domain train families (A-D)
          - 4 items (6..9): 2 verified nuance-flip contradiction pairs from domain train families (A-D)
          - 6 items (10..15): Banking77 train split (1 low-overlap same-intent + 2 overlap-trap diff-intent)
        """
        tr_triples = self.train_edit_triples[domain]
        n_tr = len(tr_triples)
        seqs_b: List[bytes] = []
        eq_labels: List[int] = []
        contra_a: List[int] = []
        contra_b: List[int] = []
        hard_a: List[int] = []
        hard_b: List[int] = []
        nuance_t: List[float] = []

        for p_i in range(2):
            o_b, p_b, _ = tr_triples[int(gen_np.integers(0, n_tr))]
            seqs_b.extend([o_b, p_b])
            eq_labels.extend([p_i, p_i])

        next_lbl = 10
        for _ in range(2):
            o_b, _, f_b = tr_triples[int(gen_np.integers(0, n_tr))]
            ia, ib = len(seqs_b), len(seqs_b) + 1
            seqs_b.extend([o_b, f_b])
            eq_labels.extend([next_lbl, next_lbl + 1])
            contra_a.append(ia)
            contra_b.append(ib)
            hard_a.append(ia)
            hard_b.append(ib)
            nuance_t.append(1.0)
            next_lbl += 2

        # 8 Banking77 items: sample broadly across all 77 train classes + overlap hard-negs!
        c1, c2 = rng_py.sample(self.b77_tr_labels_valid, 2)
        u_s1, v_s1 = rng_py.sample(self.b77_tr_by_lbl[c1], 2)
        if self.tr_same_pairs and (int(gen_np.integers(0, 2)) == 0):
            u_s2, v_s2 = rng_py.choice(self.tr_same_pairs)
        else:
            u_s2, v_s2 = rng_py.sample(self.b77_tr_by_lbl[c2], 2)
        if self.tr_diff_pairs:
            u_d1, v_d1 = rng_py.choice(self.tr_diff_pairs)
            u_d2, v_d2 = rng_py.choice(self.tr_diff_pairs)
        else:
            c3, c4 = rng_py.sample(self.b77_tr_labels_valid, 2)
            u_d1, v_d1 = rng_py.choice(self.b77_tr_by_lbl[c3]), rng_py.choice(self.b77_tr_by_lbl[c4])
            u_d2, v_d2 = v_s1, u_d1
        base_b = len(seqs_b)
        seqs_b.extend([
            self.b77_tr_txt[u_s1],
            self.b77_tr_txt[v_s1],
            self.b77_tr_txt[u_s2],
            self.b77_tr_txt[v_s2],
            self.b77_tr_txt[u_d1],
            self.b77_tr_txt[v_d1],
            self.b77_tr_txt[u_d2],
            self.b77_tr_txt[v_d2],
        ])
        eq_labels.extend([
            int(self.b77_tr_y[u_s1].item()) + 100,
            int(self.b77_tr_y[v_s1].item()) + 100,
            int(self.b77_tr_y[u_s2].item()) + 100,
            int(self.b77_tr_y[v_s2].item()) + 100,
            int(self.b77_tr_y[u_d1].item()) + 100,
            int(self.b77_tr_y[v_d1].item()) + 100,
            int(self.b77_tr_y[u_d2].item()) + 100,
            int(self.b77_tr_y[v_d2].item()) + 100,
        ])
        contra_a.extend([base_b + 4, base_b + 6])
        contra_b.extend([base_b + 5, base_b + 7])
        hard_a.extend([base_b + 4, base_b + 6])
        hard_b.extend([base_b + 5, base_b + 7])
        nuance_t.extend([0.0, 0.0])

        x_b = encode_texts_fixed(seqs_b, SEQ_LEN, DEV)
        y_eq = torch.tensor(eq_labels, dtype=torch.long, device=DEV)
        return (
            x_b,
            y_eq,
            (torch.tensor(contra_a, dtype=torch.long, device=DEV), torch.tensor(contra_b, dtype=torch.long, device=DEV)),
            (torch.tensor(hard_a, dtype=torch.long, device=DEV), torch.tensor(hard_b, dtype=torch.long, device=DEV)),
            torch.tensor(nuance_t, dtype=torch.float32, device=DEV),
        )


def bag_of_bytes_features(texts: List[bytes], length: int = SEQ_LEN) -> torch.Tensor:
    feats = torch.zeros(len(texts), 256, dtype=torch.float32)
    for i, b in enumerate(texts):
        arr = torch.tensor(list(b[:length]), dtype=torch.long)
        if arr.numel() > 0:
            feats[i] = torch.bincount(arr, minlength=256).float()
    return F.normalize(feats, p=2, dim=-1)


# --------------------------------------------------------------------------------------------------
# Model Builders
# --------------------------------------------------------------------------------------------------
def _base_arm_name(arm: str) -> str:
    return arm[:-3] if arm.endswith("_I9") else arm


def build_v7_arm(
    arm: str, cfg: Dict[str, Any], seed: int, steps_per_dom: int = 200
) -> Tuple[RealSelfStabilizingDMoE, Optional[NullSpaceProjector], Any, Optional[LatentQuarantineGate]]:
    set_all_seeds(seed)
    base_arm = _base_arm_name(arm)
    c = dict(cfg)
    c["initial_experts"] = 1
    c["min_mature_steps"] = min(int(cfg["min_mature_steps"]), max(20, int(steps_per_dom * 0.85)))
    if base_arm in ("Dense_v7_Matched",):
        c["auto_lifecycle"] = False
        c["max_experts"] = 1
        c["d_ff_shared"] = 512
        c["d_ff_spec"] = 256
    elif base_arm == "Naive_JEPA_M17":
        c["auto_lifecycle"] = False
        c["max_experts"] = 1

    model = RealSelfStabilizingDMoE(**c).to(DEV)
    if base_arm == "Cog_v7_FrozenRandomRouter" and model.context_router is not None:
        for p in model.context_router.parameters():
            p.requires_grad_(False)
    if base_arm == "Cog_v7_SequenceTop1" and model.context_router is not None:
        model.context_router.sequence_level = True
        model.context_router.tau_skip = -1.0
        model.context_router.tau_top2 = 2.0
        model.context_router.delta_mix = 0.0
        model.z_nuance_thresh = 1e9
        model.d_sem_nuance_thresh = 1e9

    if base_arm in ("Cog_v7", "Cog_v7_Mixed", "Cog_v7_SequenceTop1"):
        gpm = NullSpaceProjector(
            model,
            energy_threshold=0.97,
            max_rank_ratio=0.85,
            max_step_rank_ratio=0.24,
            step_rank_decay=0.84,
            defer_cov_commit=True,
            track_embeddings=True,
            freeze_shared_norms_on_update=True,
            freeze_unprojected_shared=True,
        )
        model.attach_null_space_projector(gpm)
        groups = classify_v7_spgc_param_groups(
            model,
            shared_mode="acquire",
            specialist_mode="acquire",
            lr_muon=8e-3,
            lr_adam=3e-3,
            lr_consolidate=2e-3,
        )
        opt = MuonBONGOptimizer(groups, mode="acquire", bong_prior_pull=0.02, grad_clip=1.5)
        gate = LatentQuarantineGate(latent_dim=c["d_model"], seed=seed, max_bytes=16384, store_samples=False) if base_arm != "Cog_v7_Mixed" else None
        return model, gpm, opt, gate
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2.5e-3, weight_decay=0.01)
        return model, None, opt, None


# --------------------------------------------------------------------------------------------------
# Track 2 Evaluation Probes (Strictly Zero Perplexity / Zero BPB!)
# --------------------------------------------------------------------------------------------------
def ridge_probe(ftr: torch.Tensor, ytr: torch.Tensor, fte: torch.Tensor, yte: torch.Tensor, n_cls: int) -> float:
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6
    ftr_n, fte_n = (ftr - mu) / sd, (fte - mu) / sd
    n = ftr_n.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    va, tr = perm[: n // 5], perm[n // 5 :]

    def fit(F_: torch.Tensor, y_: torch.Tensor, lam: float) -> torch.Tensor:
        Y = F.one_hot(y_, n_cls).float()
        A = F_.T @ F_ + lam * torch.eye(F_.size(1))
        return torch.linalg.solve(A, F_.T @ Y)

    best_lam, best_acc = 1.0, -1.0
    for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0):
        W = fit(ftr_n[tr], ytr[tr], lam)
        acc = float(((ftr_n[va] @ W).argmax(-1) == ytr[va]).float().mean())
        if acc > best_acc:
            best_lam, best_acc = lam, acc
    W = fit(ftr_n, ytr, best_lam)
    return float(((fte_n @ W).argmax(-1) == yte).float().mean())


@torch.no_grad()
def eval_banking77_v7(model: Optional[RealSelfStabilizingDMoE], bench: NonTokenRealBenchmark) -> Dict[str, Any]:
    def feat_extract(texts: List[bytes]) -> torch.Tensor:
        if model is None:
            return bag_of_bytes_features(texts, SEQ_LEN)
        model.eval()
        feats = []
        for s in range(0, len(texts), 64):
            xb = encode_texts_fixed(texts[s : s + 64], SEQ_LEN, DEV)
            out = model(xb, update_stats=False)
            feats.append(out["z_expr_pool"].float().cpu())
        return torch.cat(feats, dim=0)

    ftr = feat_extract(bench.b77_tr_txt)
    fte = feat_extract(bench.b77_te_txt)
    acc = ridge_probe(ftr, bench.b77_tr_y, fte, bench.b77_te_y, 77)
    fn = F.normalize(fte, p=2, dim=-1)
    pos = [float((fn[u] @ fn[v]).item()) for u, v in bench.same_pairs]
    neg = [float((fn[u] @ fn[v]).item()) for u, v in bench.diff_pairs]
    auc = roc_auc_pairs(pos, neg)
    return {"intent_77_acc": acc, "same_vs_overlap_diff_auc": auc}


def _ctx_eval_vec(o: Dict[str, torch.Tensor]) -> torch.Tensor:
    zc = o["z_ctx"].float()
    T = zc.size(1)
    tail = F.normalize(zc[:, max(0, T // 2) :].mean(dim=1), dim=-1)
    return F.normalize(0.5 * o["z_ctx_pool"].float() + 0.5 * tail, dim=-1)


@torch.no_grad()
def eval_triples_nuance(
    model: Optional[RealSelfStabilizingDMoE], triples: List[Tuple[bytes, bytes, bytes]]
) -> Dict[str, float]:
    if not triples:
        return {"auc": 0.5, "selectivity": 1.0, "clean_trig": 0.0, "flip_trig": 0.0, "esc_gain": 0.0, "esc_margin_gain": 0.0}
    if model is None:
        jit_d = [1.0 - float((bag_of_bytes_features([a], SEQ_LEN) @ bag_of_bytes_features([j], SEQ_LEN).t()).item()) for a, j, _ in triples]
        flp_d = [1.0 - float((bag_of_bytes_features([a], SEQ_LEN) @ bag_of_bytes_features([f], SEQ_LEN).t()).item()) for a, _, f in triples]
        return {"auc": roc_auc_pairs(flp_d, jit_d), "selectivity": 1.0, "clean_trig": 0.0, "flip_trig": 0.0, "esc_gain": 0.0, "esc_margin_gain": 0.0}

    model.eval()
    xa = encode_texts_fixed([t[0] for t in triples], SEQ_LEN, DEV)
    xj = encode_texts_fixed([t[1] for t in triples], SEQ_LEN, DEV)
    xf = encode_texts_fixed([t[2] for t in triples], SEQ_LEN, DEV)
    oa = model(xa, update_stats=False)
    oj = model(xj, update_stats=False)
    of = model(xf, update_stats=False)
    za, zj, zf = _ctx_eval_vec(oa), _ctx_eval_vec(oj), _ctx_eval_vec(of)
    jit_dist = (1.0 - (za * zj).sum(dim=-1)).tolist()
    flip_dist = (1.0 - (za * zf).sum(dim=-1)).tolist()

    clean_mask = oa["nuance_mask"][:, oa["nuance_mask"].size(1) // 2 :]
    flip_mask = of["nuance_mask"][:, of["nuance_mask"].size(1) // 2 :]
    clean_trig = float(clean_mask.float().mean().item())
    flip_trig = float(flip_mask.float().mean().item())

    # Strictly matched-depth Thinker vs Think0 comparison (fixes depth mismatch in smoke_v7_b)
    oa0 = model(xa, update_stats=False, force_thinking_steps=0)
    oj0 = model(xj, update_stats=False, force_thinking_steps=0)
    of0 = model(xf, update_stats=False, force_thinking_steps=0)
    za0, zj0, zf0 = _ctx_eval_vec(oa0), _ctx_eval_vec(oj0), _ctx_eval_vec(of0)
    flip_sep_full = float((1.0 - (za * zf).sum(dim=-1)).mean().item())
    jit_sep_full = float((1.0 - (za * zj).sum(dim=-1)).mean().item())
    flip_sep_t0 = float((1.0 - (za0 * zf0).sum(dim=-1)).mean().item())
    jit_sep_t0 = float((1.0 - (za0 * zj0).sum(dim=-1)).mean().item())

    return {
        "auc": roc_auc_pairs(flip_dist, jit_dist),
        "selectivity": (flip_trig + 1e-4) / (clean_trig + 1e-4),
        "clean_trig": clean_trig,
        "flip_trig": flip_trig,
        "esc_gain": flip_sep_full - flip_sep_t0,
        "esc_margin_gain": (flip_sep_full - jit_sep_full) - (flip_sep_t0 - jit_sep_t0),
    }


@torch.no_grad()
def eval_domain_non_token(
    model: Optional[RealSelfStabilizingDMoE],
    bench: NonTokenRealBenchmark,
    domain: str,
) -> Dict[str, Any]:
    wins = bench.corpora.test_windows[domain][:64, :SEQ_LEN].to(DEV)
    n = wins.size(0)
    triples = bench.nuance_pairs.get(domain, [])

    if model is None:
        pre_b = [bytes(w[:64].cpu().numpy().astype(np.uint8).tobytes()) for w in wins]
        fut_b = [bytes(w[64:].cpu().numpy().astype(np.uint8).tobytes()) for w in wins]
        fp = bag_of_bytes_features(pre_b, 64)
        ff = bag_of_bytes_features(fut_b, 64)
        ranks = (fp @ ff.t()).argsort(dim=-1, descending=True)
        r1 = float((ranks[:, 0] == torch.arange(n)).float().mean())
        r5 = float((ranks[:, :5] == torch.arange(n).unsqueeze(1)).any(dim=-1).float().mean())
        nd = eval_triples_nuance(None, triples)
        return {
            "concept_r1": r1,
            "concept_r5": r5,
            "nuance_flip_auc": nd["auc"],
            "nuance_selectivity": nd["selectivity"],
            "nuance_clean_trig": nd["clean_trig"],
            "nuance_flip_trig": nd["flip_trig"],
            "nuance_escalation_gain": nd["esc_gain"],
            "nuance_esc_margin_gain": nd["esc_margin_gain"],
        }

    model.eval()
    pref = wins[:, :64]
    fut64 = wins[:, 64:]
    hard_flips = [encode_texts_fixed([fl[64:]], 64, DEV)[0] for _, _, fl in triples[:16]]
    cand_x = torch.cat([fut64, torch.stack(hard_flips, dim=0)], dim=0) if hard_flips else fut64

    o_pref = model(pref, update_stats=False)
    if model.world_model is not None:
        wm = model.world_model.imagine(o_pref["latent"], horizon=3)
        h_k = min(3, wm["imagined_latents"].size(2))
        z_pred = wm["imagined_latents"][:, -1, :h_k, :].mean(dim=1)
        z_tgt_raw = model.world_model.encode_target(cand_x)
        D_t = z_tgt_raw.size(-1)
        z_tgt = F.layer_norm(z_tgt_raw.detach(), (D_t,)).mean(dim=1)
    else:
        z_pred = o_pref["latent"][:, -1, :]
        o_cand = model(cand_x, update_stats=False)
        z_tgt = o_cand["latent"].mean(dim=1)
    sims = F.normalize(z_pred.float(), dim=-1) @ F.normalize(z_tgt.float(), dim=-1).t()
    top_r = sims.argsort(dim=-1, descending=True)
    gt = torch.arange(n, device=DEV)
    r1 = float((top_r[:, 0] == gt).float().mean().item())
    r5 = float((top_r[:, :5] == gt.unsqueeze(1)).any(dim=-1).float().mean().item())

    nd = eval_triples_nuance(model, triples)
    return {
        "concept_r1": r1,
        "concept_r5": r5,
        "nuance_flip_auc": nd["auc"],
        "nuance_selectivity": nd["selectivity"],
        "nuance_clean_trig": nd["clean_trig"],
        "nuance_flip_trig": nd["flip_trig"],
        "nuance_escalation_gain": nd["esc_gain"],
        "nuance_esc_margin_gain": nd["esc_margin_gain"],
    }


@torch.no_grad()
def eval_mixed_prompt_switch(model: RealSelfStabilizingDMoE, bench: NonTokenRealBenchmark) -> Dict[str, Any]:
    model.eval()
    seqs = bench.mixed_switch_seqs.to(DEV)
    out = model(seqs, update_stats=False)
    top_ids = out["top_ids"]
    k_used = out["k_used"]
    T = top_ids.size(1)
    half = T // 2
    masked = torch.where(k_used > 0, top_ids[..., 0], -torch.ones_like(top_ids[..., 0]))
    switched = 0
    for i in range(seqs.size(0)):
        a = [int(v) for v in masked[i, :half].tolist() if v >= 0]
        b = [int(v) for v in masked[i, half:].tolist() if v >= 0]
        if a and b:
            if Counter(a).most_common(1)[0][0] != Counter(b).most_common(1)[0][0]:
                switched += 1
    k_hist = {f"k{kv}": float((k_used == kv).float().mean().item()) for kv in (0, 1, 2)}
    return {"mid_seq_switch_rate": float(switched / max(1, seqs.size(0))), "k_dist": k_hist}


# --------------------------------------------------------------------------------------------------
# Track 1 Runner: Strictly From-Scratch Compositional Generalization & Guard Rails (I11)
# --------------------------------------------------------------------------------------------------
def run_track1_arm(arm: str, cfg: Dict[str, Any], seed: int, steps_scale: float = 1.0) -> Dict[str, Any]:
    set_all_seeds(seed)
    base_arm = _base_arm_name(arm)
    task = BeliefMeaningTask(seed=seed)
    gen = torch.Generator().manual_seed(seed * 97 + 1)

    if base_arm == "Upkeeper_M09":
        upk = BudgetedSurfaceDict(max_total_bytes=16384, min_sources_for_flip=2, min_flip_count=3, flip_threshold_ratio=0.45)
        for ph, n_st, src in [("p1", int(800 * steps_scale), "s0"), ("pb", int(20 * steps_scale), "burst"), ("p2", int(300 * steps_scale), "s1"), ("p3", int(300 * steps_scale), "s2")]:
            for st_i in range(n_st):
                xs, y = task.sample_stream_batch(ph, 64, gen)
                xb = encode_belief_chunks(xs, paraphrase=(ph == "p1" and (st_i % 2 == 1)))
                src_str = f"{src}_{st_i % 2 if ph in ('p1','p2','p3') else 0}"
                upk.update(xb, y, source_id=src_str)

        def eval_upk(pair_split: str, para: bool = False) -> float:
            hits, tot = 0, 0
            for q_tok in (Q_VAL, Q_ATTR, Q_PROP):
                xs, ys = task.sample_eval_subset(q_tok, "p3", [0, 1, 2], pair_split, None, None, 128, gen)
                xb = encode_belief_chunks(xs, paraphrase=para)
                preds = upk.predict_log_probs(xb).argmax(dim=-1)
                hits += int((preds == ys).sum().item())
                tot += ys.numel()
            return hits / max(1, tot)

        return {
            "seen_acc": eval_upk("seen"),
            "heldout_unseen_acc": eval_upk("heldout"),
            "paraphrase_acc": eval_upk("heldout", para=True),
            "burst_immunity_acc": eval_upk("seen"),
            "abstain_auroc": 0.5,
            "rollout_planning_delta": 0.0,
            "quarantine_bytes": int(upk.storage_bytes()),
            "muon_grok_drift": 0.0,
        }

    n_p1 = max(720, int(750 * steps_scale))
    n_pb = max(8, int(20 * steps_scale))
    n_p2 = max(160, int(200 * steps_scale))
    n_p3 = max(160, int(200 * steps_scale))
    # Prevent Track-1 lifecycle from freezing domain-0 trunk prematurely before p1 grokking completes
    t1_cfg = dict(cfg)
    t1_cfg["min_mature_steps"] = n_p1 + 50
    model, gpm, opt, gate = build_v7_arm(base_arm, t1_cfg, seed, steps_per_dom=n_p1 + n_p2 + n_p3 + 50)
    cls_head = nn.Linear(t1_cfg["d_model"], K_CONCEPTS).to(DEV)
    if isinstance(opt, MuonBONGOptimizer):
        groups = classify_v7_spgc_param_groups(model, shared_mode="acquire", specialist_mode="acquire", lr_muon=8e-3, lr_adam=3e-3, lr_consolidate=2e-3, wd_trunk=0.10, wd_fast=0.02)
        groups.append({"name": "track1_cls", "params": list(cls_head.parameters()), "use_muon": False, "role": "fast", "mode": "acquire", "lr": 5e-3, "lr_consolidate": 3e-3, "weight_decay": 0.02, "prec_scale": 0.25})
        opt = MuonBONGOptimizer(groups, mode="acquire", bong_prior_pull=0.02)
    else:
        opt = torch.optim.AdamW(list(p for p in model.parameters() if p.requires_grad) + list(cls_head.parameters()), lr=3e-3, weight_decay=0.05)

    c_store = ConceptBeliefStore(d_ctx=t1_cfg["d_model"], max_total_bytes=16384) if base_arm.startswith("Cog_v7") else None

    def step_t1(xs: torch.Tensor, y: torch.Tensor, source_id: str, ph: str) -> None:
        xb_raw = encode_belief_chunks(xs, paraphrase=False).to(DEV)
        y_dev = y.to(DEV)
        b_sz = xs.size(0)
        if ph == "p1":
            xb_para = encode_belief_chunks(xs, paraphrase=True).to(DEV)
            xb = torch.cat([xb_raw, xb_para], dim=0)
            y_train = torch.cat([y_dev, y_dev], dim=0)
        else:
            xb = xb_raw
            y_train = y_dev
            if gate is not None:
                with torch.no_grad():
                    model.eval()
                    o_scr = model(xb, update_stats=False)
                    lg_scr = cls_head(o_scr["latent"][:, -1, :])
                    xb_act, y_act, flips = gate.triage_batch(xb, y_train, lg_scr, o_scr["z_expr_pool"], source_id=source_id, skip_familiar=False, enable_quarantine=True)
                if xb_act is None or y_act is None:
                    return
                xb, y_train = xb_act, y_act
                if flips > 0 and isinstance(opt, MuonBONGOptimizer):
                    opt.temper_precision(roles=("fast",), factor=0.25)

        model.train()
        saved = gpm.snapshot_monitored_weights() if (gpm is not None and gpm.p_null) else None
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(xb, update_stats=True)
            lg = cls_head(out["latent"][:, -1, :])
            l_cls = F.cross_entropy(lg, y_train)
            if base_arm == "Naive_JEPA_M17":
                loss = l_cls + 0.05 * (out["world_model_loss"]["loss"] if out.get("world_model_loss") is not None else 0.0)
            else:
                # Positive paraphrase invariance ONLY (zero instance-repulsion InfoNCE) + d_ctx/abstain compactness for OOD abstention
                l_para = (1.0 - (out["z_expr_pool"][:b_sz].float() * out["z_expr_pool"][b_sz:].float()).sum(dim=-1)).mean() if ph == "p1" else 0.0
                wm_l = out["world_model_loss"]["loss"] if out.get("world_model_loss") is not None else 0.0
                l_dctx_in = F.relu(out["d_ctx"].float() - 0.18).pow(2).mean()
                l_ab_in = (
                    F.binary_cross_entropy_with_logits(out["abstain_logits"].float().squeeze(-1), torch.zeros_like(out["d_ctx"].float()))
                    if out.get("abstain_logits") is not None and ph != "pb"
                    else 0.0
                )
                loss = l_cls + 0.05 * l_para + 0.03 * wm_l + 0.05 * l_dctx_in + 0.03 * l_ab_in
        opt.zero_grad()
        loss.backward()
        if gpm is not None and gpm.p_null:
            gpm.project_gradients()
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(cls_head.parameters()), 1.5)
        opt.step()
        if saved is not None:
            gpm.project_parameter_deltas(saved)
        if gpm is not None and getattr(gpm, "defer_cov_commit", False):
            gpm.commit_pending()

        if c_store is not None and model.concept_quantizer is not None and ph != "pb":
            with torch.no_grad():
                n_first = min(xs.size(0), y_train.size(0))
                cids, _ = model.concept_quantizer.assign(out["z_expr_pool"][:n_first])
                c_store.observe_batch(cids, xs[:n_first, 5], out["z_ctx_pool"][:n_first], y_train[:n_first], lg[:n_first], source_id=source_id)

    @torch.no_grad()
    def eval_t1(phase: str, pair_split: str, para: bool = False) -> float:
        model.eval()
        hits, tot = 0, 0
        for q_tok in (Q_VAL, Q_ATTR, Q_PROP):
            xs, ys = task.sample_eval_subset(q_tok, phase, [0, 1, 2], pair_split, None, None, 128, gen)
            xb = encode_belief_chunks(xs, paraphrase=para).to(DEV)
            o = model(xb, update_stats=False)
            pr = cls_head(o["latent"][:, -1, :]).argmax(-1).cpu()
            hits += int((pr == ys).sum().item())
            tot += ys.numel()
        return hits / max(1, tot)

    for st in range(n_p1):
        xs, y = task.sample_stream_batch("p1", 64, gen)
        step_t1(xs, y, f"s_{st % 2}", "p1")
    mid_held = eval_t1("p1", "heldout")

    if gpm is not None:
        gpm.update_basis_for_domain(0)
        if isinstance(opt, MuonBONGOptimizer):
            opt.set_group_mode("trunk", "consolidate")

    for st in range(n_pb):
        xs, y = task.sample_stream_batch("pb", 64, gen)
        step_t1(xs, y, "adversarial_burst", "pb")
    with torch.no_grad():
        model.eval()
        xs_s, ys_s = task.sample_eval_subset(Q_ATTR, "p1", [0, 1, 2], "seen", None, None, 128, gen)
        o_s = model(encode_belief_chunks(xs_s).to(DEV), update_stats=False)
        burst_acc = float((cls_head(o_s["latent"][:, -1, :]).argmax(-1).cpu() == ys_s).float().mean())

    for st in range(n_p2):
        xs, y = task.sample_stream_batch("p2", 64, gen)
        step_t1(xs, y, f"s2_{st % 2}", "p2")
    for st in range(n_p3):
        xs, y = task.sample_stream_batch("p3", 64, gen)
        step_t1(xs, y, f"s3_{st % 2}", "p3")

    with torch.no_grad():
        model.eval()
        xs_k, _ = task.sample_eval_subset(Q_VAL, "p3", [0, 1, 2], "heldout", None, None, 64, gen)
        xk_b = encode_belief_chunks(xs_k).to(DEV)
        xu_b = torch.randint(33, 122, (64, 24), device=DEV)
        ok = model(xk_b, update_stats=False)
        ou = model(xu_b, update_stats=False)
        sk = ok["nuance_score"].mean(dim=-1).tolist()
        su = ou["nuance_score"].mean(dim=-1).tolist()
        abstain_auroc = roc_auc_pairs(su, sk)

        o_t0 = model(xk_b, update_stats=False, force_thinking_steps=0)
        xs_pred_full = cls_head(ok["latent"][:, -1, :]).argmax(-1).cpu()
        xs_pred_t0 = cls_head(o_t0["latent"][:, -1, :]).argmax(-1).cpu()
        ys_k_exact, _, _ = task._compute_targets(xs_k[:, 0], xs_k[:, 1] - 4, xs_k[:, 2] - 4, xs_k[:, 3] - 4, xs_k[:, 4] - 4, xs_k[:, 5], "p3")
        rollout_delta = float((xs_pred_full == ys_k_exact).float().mean().item() - (xs_pred_t0 == ys_k_exact).float().mean().item())

    fin_held = eval_t1("p3", "heldout")
    return {
        "seen_acc": eval_t1("p3", "seen"),
        "heldout_unseen_acc": fin_held,
        "paraphrase_acc": eval_t1("p3", "heldout", para=True),
        "burst_immunity_acc": burst_acc,
        "abstain_auroc": abstain_auroc,
        "rollout_planning_delta": rollout_delta,
        "muon_grok_drift": mid_held - fin_held,
        "quarantine_bytes": int(gate.storage_bytes()) if gate is not None else 0,
    }


# --------------------------------------------------------------------------------------------------
# Track 2 Runner: Real Sensory Streams (Zero Perplexity / Zero BPB!)
# --------------------------------------------------------------------------------------------------
def run_track2_arm(
    arm: str,
    cfg: Dict[str, Any],
    bench: NonTokenRealBenchmark,
    seed: int,
    steps_per_dom: int,
    teacher: Optional[OfflineTeacherGramEncoder] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    base_arm = _base_arm_name(arm)
    use_teacher = arm.endswith("_I9") and teacher is not None

    if base_arm == "BagOfBytes_Null":
        per_dom = {d: eval_domain_non_token(None, bench, d) for d in DOMAINS}
        b77 = eval_banking77_v7(None, bench)
        hf = eval_triples_nuance(None, bench.heldout_family_pairs)
        return {
            "arm": arm,
            "teacher_informed": False,
            "train_seconds": time.perf_counter() - t0,
            "banking77": b77,
            "heldout_families": hf,
            "per_domain_final": per_dom,
            "mean_final_r1": float(np.mean([per_dom[d]["concept_r1"] for d in DOMAINS])),
            "mean_final_r5": float(np.mean([per_dom[d]["concept_r5"] for d in DOMAINS])),
            "mean_nuance_flip_auc": float(np.mean([per_dom[d]["nuance_flip_auc"] for d in DOMAINS])),
            "mean_nuance_selectivity": 1.0,
            "mean_escalation_gain": 0.0,
            "mean_esc_margin_gain": 0.0,
        }

    model, gpm, opt, gate = build_v7_arm(arm, cfg, seed, steps_per_dom=steps_per_dom)
    if base_arm == "Untrained_Cog_v7":
        per_dom = {d: eval_domain_non_token(model, bench, d) for d in DOMAINS}
        b77 = eval_banking77_v7(model, bench)
        hf = eval_triples_nuance(model, bench.heldout_family_pairs)
        sw = eval_mixed_prompt_switch(model, bench)
        return {
            "arm": arm,
            "teacher_informed": False,
            "train_seconds": time.perf_counter() - t0,
            "banking77": b77,
            "heldout_families": hf,
            "per_domain_final": per_dom,
            "mixed_switch": sw,
            "mean_final_r1": float(np.mean([per_dom[d]["concept_r1"] for d in DOMAINS])),
            "mean_final_r5": float(np.mean([per_dom[d]["concept_r5"] for d in DOMAINS])),
            "mean_nuance_flip_auc": float(np.mean([per_dom[d]["nuance_flip_auc"] for d in DOMAINS])),
            "mean_nuance_selectivity": float(np.mean([per_dom[d]["nuance_selectivity"] for d in DOMAINS])),
            "mean_escalation_gain": float(np.mean([per_dom[d]["nuance_escalation_gain"] for d in DOMAINS])),
            "mean_esc_margin_gain": float(np.mean([per_dom[d]["nuance_esc_margin_gain"] for d in DOMAINS])),
        }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEV)

    gen_np = np.random.default_rng(seed * 313 + 7)
    rng_py = random.Random(seed * 313 + 7)
    peak_r5: Dict[str, float] = {}
    peak_auc: Dict[str, float] = {}
    events_log: List[Dict[str, Any]] = []
    skipped_steps = 0
    backward_passes = 0
    gstep = 0
    mid_b77_acc: Optional[float] = None

    def train_one_step(dom: str, allow_quarantine: bool) -> None:
        nonlocal opt, gstep, skipped_steps, backward_passes
        gstep += 1
        x_b, y_eq, (ca, cb), (ha, hb), nt = bench.sample_stream_contrastive_batch(dom, gen_np, rng_py)
        t_gram = teacher.batch_gram(x_b) if use_teacher else None

        if gate is not None and allow_quarantine and gstep > 30:
            with torch.no_grad():
                model.eval()
                o_s = model(x_b, update_stats=False)
                err_s = o_s["nuance_score"].mean(dim=-1)
                xb_a, _, _ = gate.triage_continuous(
                    x_b,
                    err_s.unsqueeze(-1),
                    torch.zeros_like(err_s).unsqueeze(-1),
                    o_s["z_expr_pool"],
                    source_id=f"{dom}_{gstep % 2}",
                    skip_familiar=False,
                    enable_quarantine=True,
                )
            if xb_a is None or xb_a.size(0) < 4:
                skipped_steps += 1
                return

        if gpm is not None:
            gpm.is_tracking = (gstep % 2 == 1) or (len(gpm.covariances) == 0)
        saved = gpm.snapshot_monitored_weights() if (gpm is not None and gpm.p_null) else None

        model.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(x_b, update_stats=True)
            if base_arm == "Naive_JEPA_M17":
                loss_d = model.compute_v7_non_token_loss(
                    out, x=None, equiv_labels=None, contra_pair_idx=None, wm_weight=1.0, fsp_weight=0.0, sigreg_weight=0.0, equiv_weight=0.0, contra_weight=0.0, think_weight=0.0, erc_weight=0.0, balance_weight=0.0
                )
            else:
                loss_d = model.compute_v7_non_token_loss(
                    out,
                    x=x_b,
                    equiv_labels=y_eq,
                    contra_pair_idx=(ca, cb),
                    hard_neg_pair_idx=(ha, hb),
                    nuance_targets=nt,
                    teacher_gram=t_gram,
                    wm_weight=0.50,
                    fsp_weight=0.15,
                    sigreg_weight=0.25,
                    equiv_weight=1.65,
                    contra_weight=1.35,
                    think_weight=0.35,
                    erc_weight=0.20,
                    balance_weight=0.40,
                    teacher_weight=0.65 if use_teacher else 0.0,
                )

        evs = out.get("lifecycle_events", [])
        spawned_now = any(e.startswith("spawned_") for e in evs)
        for ev in evs:
            events_log.append({"step": gstep, "domain": dom, "event": ev})

        opt.zero_grad()
        loss_d["loss"].backward()
        backward_passes += 1
        if gpm is not None and gpm.p_null:
            gpm.project_gradients()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.5)
        opt.step()
        if saved is not None:
            gpm.project_parameter_deltas(saved)
        if gpm is not None and getattr(gpm, "defer_cov_commit", False):
            gpm.commit_pending()

        if spawned_now and isinstance(opt, MuonBONGOptimizer):
            groups = classify_v7_spgc_param_groups(
                model,
                shared_mode="consolidate",
                specialist_mode="acquire",
                lr_muon=8e-3,
                lr_adam=3e-3,
                lr_consolidate=2e-3,
            )
            opt.rebuild_preserving_state(groups)

    if base_arm == "Cog_v7_Mixed":
        tot = steps_per_dom * len(DOMAINS)
        for st_m in range(tot):
            dom = DOMAINS[int(gen_np.integers(0, len(DOMAINS)))]
            train_one_step(dom, allow_quarantine=False)
            if st_m == tot // 2:
                mid_b77_acc = eval_banking77_v7(model, bench)["intent_77_acc"]
    else:
        for d_idx, dom in enumerate(DOMAINS):
            for _ in range(steps_per_dom):
                train_one_step(dom, allow_quarantine=(base_arm != "Cog_v7_SequenceTop1"))
            if d_idx < len(DOMAINS) - 1:
                ev_now = eval_domain_non_token(model, bench, dom)
                peak_r5[dom] = ev_now["concept_r5"]
                peak_auc[dom] = ev_now["nuance_flip_auc"]
            if d_idx == 2:
                mid_b77_acc = eval_banking77_v7(model, bench)["intent_77_acc"]

    per_dom = {d: eval_domain_non_token(model, bench, d) for d in DOMAINS}
    b77 = eval_banking77_v7(model, bench)
    hf = eval_triples_nuance(model, bench.heldout_family_pairs)
    sw = eval_mixed_prompt_switch(model, bench)

    fg_r5 = float(np.mean([peak_r5[d] - per_dom[d]["concept_r5"] for d in DOMAINS[:-1]])) if peak_r5 else None
    fg_auc = float(np.mean([peak_auc[d] - per_dom[d]["nuance_flip_auc"] for d in DOMAINS[:-1]])) if peak_auc else None
    at_cap = 0
    mean_cap = 0.0
    if gpm is not None and gpm.bases:
        ratios = [b.size(1) / max(1, b.size(0)) for b in gpm.bases.values() if b is not None]
        at_cap = sum(1 for r in ratios if r >= 0.84)
        mean_cap = float(np.mean(ratios)) if ratios else 0.0

    # Isolated secondary verbalization check behind stop_gradient
    det_opt = torch.optim.AdamW(model.decoder.parameters(), lr=5e-3)
    det_bpb_end = None
    for _ in range(10):
        x_d, _, _, _, _ = bench.sample_stream_contrastive_batch("wiki", gen_np, rng_py)
        with torch.no_grad():
            o_d = model(x_d, update_stats=False)
        rlg = model.render_text_detached(o_d)
        dloss = F.cross_entropy(rlg.reshape(-1, 256), x_d.reshape(-1))
        det_opt.zero_grad()
        dloss.backward()
        det_opt.step()
        det_bpb_end = float(dloss.item() / math.log(2.0))

    moe_ref = model.layers[model._moe_ref_layer_idx].moe
    return {
        "arm": arm,
        "teacher_informed": use_teacher,
        "seed": seed,
        "train_seconds": time.perf_counter() - t0,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(DEV) / 1048576.0) if torch.cuda.is_available() else 0.0,
        "num_experts": len(moe_ref.experts),
        "backward_passes": backward_passes,
        "skipped_steps": skipped_steps,
        "banking77": b77,
        "b77_muon_drift": (mid_b77_acc - b77["intent_77_acc"]) if mid_b77_acc is not None else 0.0,
        "heldout_families": hf,
        "mixed_switch": sw,
        "per_domain_final": per_dom,
        "mean_final_r1": float(np.mean([per_dom[d]["concept_r1"] for d in DOMAINS])),
        "mean_final_r5": float(np.mean([per_dom[d]["concept_r5"] for d in DOMAINS])),
        "mean_nuance_flip_auc": float(np.mean([per_dom[d]["nuance_flip_auc"] for d in DOMAINS])),
        "mean_nuance_selectivity": float(np.mean([per_dom[d]["nuance_selectivity"] for d in DOMAINS])),
        "mean_escalation_gain": float(np.mean([per_dom[d]["nuance_escalation_gain"] for d in DOMAINS])),
        "mean_esc_margin_gain": float(np.mean([per_dom[d]["nuance_esc_margin_gain"] for d in DOMAINS])),
        "forgetting_r5": fg_r5,
        "forgetting_nuance_auc": fg_auc,
        "gpm_mean_cap_ratio": mean_cap,
        "gpm_at_cap_count": at_cap,
        "quarantine_bytes": int(gate.storage_bytes()) if gate is not None else 0,
        "nuance_buf_bytes": int(model.nuance_buffer.storage_bytes()) if model.nuance_buffer is not None else 0,
        "detached_renderer_diag_bpb": det_bpb_end,
        "events": events_log[-40:],
    }


def _agg(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not runs:
        return out
    flat_keys = [
        "train_seconds", "peak_vram_mb", "num_experts", "backward_passes", "skipped_steps",
        "mean_final_r1", "mean_final_r5", "mean_nuance_flip_auc", "mean_nuance_selectivity",
        "mean_escalation_gain", "mean_esc_margin_gain", "forgetting_r5", "forgetting_nuance_auc",
        "gpm_at_cap_count", "b77_muon_drift", "detached_renderer_diag_bpb",
    ]
    for k in flat_keys:
        vals = [r[k] for r in runs if r.get(k) is not None]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "vals": vals}
    out["teacher_informed"] = bool(runs[0].get("teacher_informed", False))
    out["b77_acc"] = {"mean": float(np.mean([r["banking77"]["intent_77_acc"] for r in runs]))}
    out["b77_auc"] = {"mean": float(np.mean([r["banking77"]["same_vs_overlap_diff_auc"] for r in runs]))}
    if all("heldout_families" in r for r in runs):
        out["heldout_fam_auc"] = {"mean": float(np.mean([r["heldout_families"]["auc"] for r in runs]))}
        out["heldout_fam_selectivity"] = {"mean": float(np.mean([r["heldout_families"]["selectivity"] for r in runs]))}
    if all("mixed_switch" in r for r in runs):
        out["mid_seq_switch_rate"] = {"mean": float(np.mean([r["mixed_switch"]["mid_seq_switch_rate"] for r in runs]))}
        out["k_dist"] = {kv: float(np.mean([r["mixed_switch"]["k_dist"][kv] for r in runs])) for kv in ("k0", "k1", "k2")}
    if all("track1" in r for r in runs):
        t1_keys = list(runs[0]["track1"].keys())
        out["track1"] = {tk: {"mean": float(np.mean([r["track1"][tk] for r in runs]))} for tk in t1_keys}
    return out


def _append_dev_tuning_log(out_name: str, summary: Dict[str, Any]) -> None:
    log_path = RESULTS_ROOT / "tuning_log.json"
    rows: List[Dict[str, Any]] = []
    if log_path.exists():
        try:
            rows = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            rows = []
    rows.append({
        "trial_id": out_name,
        "utc": summary["meta"]["utc"],
        "cfg": {k: summary["meta"]["cfg"][k] for k in ("min_mature_steps", "z_nuance_thresh", "d_sem_nuance_thresh", "nuance_cluster_min", "nuance_cluster_cos")},
        "steps_per_dom": summary["meta"]["steps_per_dom"],
        "arms": summary["arms"],
    })
    log_path.write_text(json.dumps(rows, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "dev", "test"], default="smoke")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out_name", type=str, default=None)
    ap.add_argument("--z_nuance", type=float, default=DEFAULT_V7_CFG["z_nuance_thresh"])
    ap.add_argument("--d_sem", type=float, default=DEFAULT_V7_CFG["d_sem_nuance_thresh"])
    ap.add_argument("--cluster_min", type=int, default=DEFAULT_V7_CFG["nuance_cluster_min"])
    ap.add_argument("--cluster_cos", type=float, default=DEFAULT_V7_CFG["nuance_cluster_cos"])
    ap.add_argument("--arms", type=str, nargs="+", default=None)
    args = ap.parse_args()

    cfg = dict(DEFAULT_V7_CFG)
    cfg["z_nuance_thresh"] = float(args.z_nuance)
    cfg["d_sem_nuance_thresh"] = float(args.d_sem)
    cfg["nuance_cluster_min"] = int(args.cluster_min)
    cfg["nuance_cluster_cos"] = float(args.cluster_cos)

    if args.mode == "smoke":
        eval_split = "dev"
        seeds = args.seeds or [100]
        steps = args.steps or 25
        t1_scale = 0.35
        arms = args.arms or ["BagOfBytes_Null", "Upkeeper_M09", "Dense_v7_Matched", "Cog_v7", "Cog_v7_I9"]
    elif args.mode == "dev":
        eval_split = "dev"
        seeds = args.seeds or [100]
        steps = args.steps or 200
        t1_scale = 0.8
        arms = args.arms or [
            "BagOfBytes_Null", "Untrained_Cog_v7", "Upkeeper_M09",
            "Naive_JEPA_M17", "Dense_v7_Matched", "Cog_v7_SequenceTop1", "Cog_v7",
            "Dense_v7_Matched_I9", "Cog_v7_I9", "Cog_v7_Mixed_I9",
        ]
    else:
        eval_split = "test"
        seeds = args.seeds or [0, 1, 2]
        steps = args.steps or 250
        t1_scale = 1.0
        arms = args.arms or [
            "BagOfBytes_Null", "Untrained_Cog_v7", "Upkeeper_M09",
            "Naive_JEPA_M17", "Dense_v7_Matched", "Cog_v7_FrozenRandomRouter",
            "Cog_v7_SequenceTop1", "Cog_v7",
            "Dense_v7_Matched_I9", "Cog_v7_I9", "Cog_v7_Mixed_I9",
        ]

    out_name = args.out_name or f"{args.mode}_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    out_dir = RESULTS_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    corpora = RealCorpora(seq_len=SEQ_LEN, eval_split=eval_split)
    bench = NonTokenRealBenchmark(corpora, eval_split=eval_split)
    print(f"[I4 Verified Train Edit Histogram ({eval_split})]: {bench.train_family_hist}", flush=True)

    teacher = OfflineTeacherGramEncoder() if any(a.endswith("_I9") for a in arms) else None

    meta = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "eval_split": eval_split,
        "seeds": seeds,
        "steps_per_dom": steps,
        "cfg": cfg,
        "train_family_hist": bench.train_family_hist,
        "device": str(DEV),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2))

    by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
    for s in seeds:
        for arm in arms:
            print(f"[v7] Running arm={arm} seed={s} split={eval_split} steps/dom={steps}...", flush=True)
            base_arm = _base_arm_name(arm)
            if base_arm == "Upkeeper_M09":
                res = {"arm": arm, "teacher_informed": False, "seed": s, "track1": run_track1_arm(arm, cfg, s, steps_scale=t1_scale), "banking77": {"intent_77_acc": 1/77, "same_vs_overlap_diff_auc": 0.5}}
            else:
                res = run_track2_arm(arm, cfg, bench, s, steps_per_dom=steps, teacher=teacher)
                if not arm.endswith("_I9") and base_arm not in ("BagOfBytes_Null", "Untrained_Cog_v7"):
                    res["track1"] = run_track1_arm(arm, cfg, s, steps_scale=t1_scale)
            by_arm[arm].append(res)
            (out_dir / f"seed{s}_{arm}.json").write_text(json.dumps(res, indent=2))

    summary = {"meta": meta, "arms": {a: _agg(by_arm[a]) for a in arms}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.mode == "dev":
        _append_dev_tuning_log(out_name, summary)

    print(f"\n=== real_pilot_v4 Summary ({out_name}) ===")
    for a, st in summary["arms"].items():
        b_auc = st.get("b77_auc", {}).get("mean")
        r5 = st.get("mean_final_r5", {}).get("mean")
        n_auc = st.get("mean_nuance_flip_auc", {}).get("mean")
        hf_auc = st.get("heldout_fam_auc", {}).get("mean")
        sel = st.get("mean_nuance_selectivity", {}).get("mean")
        emg = st.get("mean_esc_margin_gain", {}).get("mean")
        sw = st.get("mid_seq_switch_rate", {}).get("mean")
        fg = st.get("forgetting_r5", {}).get("mean")
        t1_h = st.get("track1", {}).get("heldout_unseen_acc", {}).get("mean")
        sec = st.get("train_seconds", {}).get("mean")
        ti = "*" if st.get("teacher_informed") else " "
        print(
            f" {ti}{a:24s} | b77={b_auc if b_auc is not None else -1:.4f} "
            f"r5={r5 if r5 is not None else -1:.4f} n_auc={n_auc if n_auc is not None else -1:.4f} "
            f"ho_fam={hf_auc if hf_auc is not None else -1:.4f} sel={sel if sel is not None else -1:.2f} "
            f"th_mg={emg if emg is not None else 0:+.3f} sw={sw if sw is not None else -1:.3f} "
            f"fg_r5={fg if fg is not None else -1:.4f} t1_h={t1_h if t1_h is not None else -1:.4f} "
            f"sec={sec if sec is not None else -1:.1f}"
        )


if __name__ == "__main__":
    main()
