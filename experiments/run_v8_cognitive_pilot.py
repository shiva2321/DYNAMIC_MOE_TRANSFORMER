"""
real_pilot_v5: Architecture v8 Consolidated Non-Token Cognitive Pilot.

Consolidates:
  - v3 SPGC + I10 meaning-consistency Fisher scaling (gentler trunk prec_scale=0.85, max_prec=25)
  - v6 decaying GPM + full-stack rank-16 private adapters + layer_stm_every=4
  - v7 dynamic token-level k in {0,1,2} causal-context routing + deep-target JEPA + FSP + Thinker RLP
  - v8 upgrades:
      * BYTE_PAD_ID=257 & whitespace content masking (fixing constant-padding collapse & R@5)
      * Order-sensitive Relational Pooling (`rel_pool_proj`) + unnormalized `z_expr_raw_pool`
      * Real 128B contiguous domain streams in every training batch + continual detached LoRA verbalizer
      * Single-pass inline quarantine + energy-only Thinker skipping wasted decoder dispatches (~90s faster)
      * I7 procedural structural pre-training (`P0`) with zero lexical overlap against HELDOUT_FAMILIES
      * I5 Causal Abstraction / Interchange-Intervention Training (`forward_interchange`) on Track 1
      * I6 chunked `extract_episode_kv` Episodic In-Context Rule Induction (`LongTermSlotMemory`)
      * I8 functional routing NMI across operation/intent roles
      * Same-script `Cog_v6_Ref` consolidated baseline for honest apples-to-apples comparison
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
    K_CONCEPTS,
    Q_ATTR,
    Q_PROP,
    Q_VAL,
)
from dmoe.null_space import NullSpaceProjector
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.spgc import LatentQuarantineGate, MuonBONGOptimizer, classify_v7_spgc_param_groups
from experiments.real_pilot_data import DOMAINS, RealCorpora, SEQ_LEN
from experiments.run_v7_cognitive_pilot import (
    OfflineTeacherGramEncoder,
    bag_of_bytes_features,
    encode_belief_chunks,
)
from experiments.v8_edit_families import (
    HELDOUT_FAMILIES,
    TRAIN_FAMILIES,
    generate_i7_procedural_batch,
    generate_v7_triple,
    generate_v8_train_triple,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_ROOT = ROOT / "results" / "real_pilot_v5"

DEFAULT_V8_CFG: Dict[str, Any] = dict(
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
    use_v8_core=True,
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
    layer_stm_every=4,
    enable_full_stack_adapters=True,
    adapter_rank=16,
    enable_thinking=True,
    max_thinking_steps=2,
    skip_threshold=0.10,
    theta_halt=0.20,
    halt_use_energy_only=True,
    use_marginal_drop_halt=True,
    skip_final_lookahead=True,
    abstain_threshold=0.60,
    auto_lifecycle=True,
    max_experts=6,
    min_mature_steps=160,
    z_nuance_thresh=2.15,
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


def encode_texts_v8(texts: List[bytes], length: int, device: torch.device, pad_id: int = 257) -> torch.Tensor:
    out = torch.full((len(texts), length), int(pad_id), dtype=torch.long)
    for i, b in enumerate(texts):
        n = min(len(b), length)
        if n > 0:
            out[i, :n] = torch.tensor(list(b[:n]), dtype=torch.long)
    return out.to(device)


def roc_auc_pairs(pos_scores: List[float], neg_scores: List[float]) -> float:
    if not pos_scores or not neg_scores:
        return 0.5
    p = np.asarray(pos_scores, dtype=np.float64)[:, None]
    q = np.asarray(neg_scores, dtype=np.float64)[None, :]
    return float(np.mean((p > q) + 0.5 * (p == q)))


def compute_nmi(labels_a: List[int], labels_b: List[int]) -> float:
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0
    n = float(len(labels_a))
    ca, cb = Counter(labels_a), Counter(labels_b)
    cab = Counter(zip(labels_a, labels_b))
    ha = -sum((v / n) * math.log(v / n) for v in ca.values() if v > 0)
    hb = -sum((v / n) * math.log(v / n) for v in cb.values() if v > 0)
    if ha + hb <= 1e-12:
        return 0.0
    mi = sum((v / n) * math.log((n * v) / (ca[ka] * cb[kb])) for (ka, kb), v in cab.items() if v > 0)
    return float(max(0.0, min(1.0, (2.0 * mi) / (ha + hb))))


class V8RealBenchmark:
    """Unified Track 2 benchmark with raw 128B continuations, enriched A-D train edits, disjoint E-F test edits & Banking77."""

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

        # Training triples strictly from TRAIN_FAMILIES A-D (zero overlap with E-F)
        rng_fam = random.Random(42 if eval_split == "dev" else 99)
        self.train_edit_triples: Dict[str, List[Tuple[bytes, bytes, bytes, str]]] = {}
        for d in DOMAINS:
            pool = corpora.train[d]
            max_off = len(pool) - SEQ_LEN - 1
            items: List[Tuple[bytes, bytes, bytes, str]] = []
            for k_i in range(320):
                off = int((k_i * 997) % max_off)
                raw = pool[off : off + SEQ_LEN].tobytes()
                o_b, p_b, f_b, tag = generate_v8_train_triple(raw, rng_fam, allowed_families=TRAIN_FAMILIES, seq_len=SEQ_LEN)
                items.append((o_b, p_b, f_b, tag))
            self.train_edit_triples[d] = items

        # Evaluation NuanceFlip triples per domain (legacy v7 A-D generator for strict continuity)
        self.nuance_pairs: Dict[str, List[Tuple[bytes, bytes, bytes]]] = {}
        for d in DOMAINS:
            triples: List[Tuple[bytes, bytes, bytes]] = []
            for w in corpora.test_windows[d][:48]:
                raw = bytes(w[:SEQ_LEN].numpy().astype(np.uint8).tobytes())
                o_b, p_b, f_b, _ = generate_v7_triple(raw, rng_fam, allowed_families=TRAIN_FAMILIES, seq_len=SEQ_LEN)
                triples.append((o_b, p_b, f_b))
            self.nuance_pairs[d] = triples

        # Untouched HELD-OUT ANTI-SHORTCUT edit triples (Families E-F, NEVER used during training!)
        self.heldout_family_pairs: List[Tuple[bytes, bytes, bytes]] = []
        for d in DOMAINS:
            for w in corpora.test_windows[d][48:68]:
                raw = bytes(w[:SEQ_LEN].numpy().astype(np.uint8).tobytes())
                o_b, p_b, f_b, tag = generate_v7_triple(raw, rng_fam, allowed_families=HELDOUT_FAMILIES, seq_len=SEQ_LEN)
                assert tag in HELDOUT_FAMILIES
                self.heldout_family_pairs.append((o_b, p_b, f_b))

        # MixedPromptSwitch ([64B domA | 64B domB])
        sw_list: List[torch.Tensor] = []
        for i in range(32):
            sw_list.append(torch.cat([corpora.test_windows["story"][i, :64], corpora.test_windows["code"][i, :64]], dim=0))
            sw_list.append(torch.cat([corpora.test_windows["wiki"][i, :64], corpora.test_windows["math"][i, :64]], dim=0))
        self.mixed_switch_seqs = torch.stack(sw_list, dim=0)

        # I8 functional role benchmark items (6 distinct semantic/syntactic roles for functional routing NMI)
        self.func_role_texts: List[bytes] = []
        self.func_role_labels: List[int] = []
        for w in corpora.test_windows["wiki"][68:96]:
            self.func_role_texts.append(bytes(w[:SEQ_LEN].numpy().astype(np.uint8).tobytes()))
            self.func_role_labels.append(0)
        for w in corpora.test_windows["code"][68:96]:
            self.func_role_texts.append(bytes(w[:SEQ_LEN].numpy().astype(np.uint8).tobytes()))
            self.func_role_labels.append(1)
        for fi, fam_name in enumerate(("code_alpha_var", "algebra_sympy", "unit_scale")):
            for w in corpora.test_windows["math"][68:96]:
                raw = bytes(w[:SEQ_LEN].numpy().astype(np.uint8).tobytes())
                o_b, _, _, _ = generate_v7_triple(raw, rng_fam, allowed_families=(fam_name,), seq_len=SEQ_LEN)
                self.func_role_texts.append(o_b)
                self.func_role_labels.append(2 + fi)
        for b_i in range(28):
            self.func_role_texts.append(self.b77_te_txt[b_i])
            self.func_role_labels.append(5)

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

    def sample_v8_stream_batch(
        self, domain: str, gen_np: np.random.Generator, rng_py: random.Random
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        16-sequence v8 Track 2 batch:
          - 0..3:   4 fresh contiguous 128B real corpus windows (`corpora.sample_train(domain)`)
          - 4..9:   3 verified A-D triples -> 1 paraphrase pair (4,5) + 2 nuance-flip pairs (6,7) & (8,9)
          - 10..15: 6 Banking77 train utterances (`BYTE_PAD_ID=257` padded!) -> 1 same pair + 2 hard-neg diff pairs
        Returns `(x_b, y_eq, contra_pair, hard_neg_pair, nuance_t, struct_t, gram_mask)`.
        """
        raw_win = self.corpora.sample_train(domain, 4, gen_np)[:, :SEQ_LEN].to(DEV)
        tr_triples = self.train_edit_triples[domain]
        n_tr = len(tr_triples)

        seqs_b: List[bytes] = []
        eq_labels: List[int] = [-100, -101, -102, -103]
        contra_a: List[int] = []
        contra_b: List[int] = []
        hard_a: List[int] = []
        hard_b: List[int] = []
        nuance_t: List[float] = []
        struct_t: List[int] = [0, 0, 0, 0]

        # 4..5: Verified paraphrase pair
        o_p, p_p, _, _ = tr_triples[int(gen_np.integers(0, n_tr))]
        seqs_b.extend([o_p, p_p])
        eq_labels.extend([1, 1])
        struct_t.extend([1, 1])

        # 6..9: 2 verified nuance-flip contradiction pairs
        nxt = 10
        for _ in range(2):
            o_f, _, f_f, _ = tr_triples[int(gen_np.integers(0, n_tr))]
            ia = 4 + len(seqs_b)
            ib = ia + 1
            seqs_b.extend([o_f, f_f])
            eq_labels.extend([nxt, nxt + 1])
            contra_a.append(ia)
            contra_b.append(ib)
            hard_a.append(ia)
            hard_b.append(ib)
            nuance_t.append(1.0)
            struct_t.extend([0, 2])
            nxt += 2

        # 10..15: 6 Banking77 train utterances (BYTE_PAD_ID=257 padded!)
        # 2 positive same-intent pairs + 1 overlap-matched hard-negative diff-intent pair
        # Note: keep topic diffs ONLY in hard_neg (not contra_a/contra_b, which stays pure for nuance flips!)
        c1, c2 = rng_py.sample(self.b77_tr_labels_valid, 2)
        u_s1, v_s1 = rng_py.sample(self.b77_tr_by_lbl[c1], 2)
        u_s2, v_s2 = rng_py.sample(self.b77_tr_by_lbl[c2], 2)
        if self.tr_diff_pairs:
            u_d1, v_d1 = rng_py.choice(self.tr_diff_pairs)
        else:
            c3, c4 = rng_py.sample(self.b77_tr_labels_valid, 2)
            u_d1, v_d1 = rng_py.choice(self.b77_tr_by_lbl[c3]), rng_py.choice(self.b77_tr_by_lbl[c4])

        base_b = 4 + len(seqs_b)  # 10
        b77_raw = [
            self.b77_tr_txt[u_s1],
            self.b77_tr_txt[v_s1],
            self.b77_tr_txt[u_s2],
            self.b77_tr_txt[v_s2],
            self.b77_tr_txt[u_d1],
            self.b77_tr_txt[v_d1],
        ]
        b77_lbls = [
            int(self.b77_tr_y[u_s1].item()) + 100,
            int(self.b77_tr_y[v_s1].item()) + 100,
            int(self.b77_tr_y[u_s2].item()) + 100,
            int(self.b77_tr_y[v_s2].item()) + 100,
            int(self.b77_tr_y[u_d1].item()) + 100,
            int(self.b77_tr_y[v_d1].item()) + 100,
        ]
        eq_labels.extend(b77_lbls)
        hard_a.append(base_b + 4)
        hard_b.append(base_b + 5)
        struct_t.extend([1, 1, 1, 1, 0, 0])

        x_edits = encode_texts_v8(seqs_b, SEQ_LEN, DEV, pad_id=32)
        x_b77 = encode_texts_v8(b77_raw, SEQ_LEN, DEV, pad_id=257)
        x_b = torch.cat([raw_win, x_edits, x_b77], dim=0)

        # Unlabeled I9 Gram focuses cleanly on real streams (`0..3` raw + `10..15` B77), excluding flip pairs
        gm = torch.zeros((16, 16), dtype=torch.bool, device=DEV)
        real_idx = torch.tensor([0, 1, 2, 3, 10, 11, 12, 13, 14, 15], dtype=torch.long, device=DEV)
        gm[real_idx.unsqueeze(1), real_idx.unsqueeze(0)] = True
        gm[torch.tensor(contra_a, device=DEV), torch.tensor(contra_b, device=DEV)] = False
        gm[torch.tensor(contra_b, device=DEV), torch.tensor(contra_a, device=DEV)] = False
        gm.fill_diagonal_(False)

        return (
            x_b,
            torch.tensor(eq_labels, dtype=torch.long, device=DEV),
            (torch.tensor(contra_a, dtype=torch.long, device=DEV), torch.tensor(contra_b, dtype=torch.long, device=DEV)),
            (torch.tensor(hard_a, dtype=torch.long, device=DEV), torch.tensor(hard_b, dtype=torch.long, device=DEV)),
            torch.tensor(nuance_t, dtype=torch.float32, device=DEV),
            torch.tensor(struct_t, dtype=torch.long, device=DEV),
            gm,
        )


def _base_arm_name(arm: str) -> str:
    return arm[:-3] if arm.endswith("_I9") else arm


def build_v8_arm(
    arm: str, cfg: Dict[str, Any], seed: int, steps_per_dom: int = 250
) -> Tuple[RealSelfStabilizingDMoE, Optional[NullSpaceProjector], Any, Optional[LatentQuarantineGate]]:
    set_all_seeds(seed)
    base_arm = _base_arm_name(arm)

    if base_arm == "Cog_v6_Ref":
        # Match actual winning Cog_v6_Tier2 config from real_pilot_v2/v3 (chunk_size=1, d_ff=512, lr=1e-3)!
        c6 = dict(
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            d_ff=512,
            initial_layers=cfg["initial_layers"],
            initial_experts=1,
            chunk_size=1,
            enable_thinking=True,
            max_thinking_steps=2,
            skip_threshold=0.15,
            theta_halt=0.22,
            halt_use_energy_only=True,
            use_marginal_drop_halt=True,
            min_surprise_drop=0.008,
            skip_final_lookahead=True,
            enable_memory=True,
            auto_write_ltm=False,
            num_ltm_slots=cfg["num_ltm_slots"],
            enable_three_level_attn=True,
            local_window=16,
            chunk_stride=8,
            enable_world_model=True,
            wm_rollout_horizon=3,
            align_wm_target_with_encoder=True,
            causal_attn=True,
            use_rope=True,
            enable_layer_stm=True,
            hybrid_mode=True,
            layer_stm_every=cfg["layer_stm_every"],
            enable_full_stack_adapters=True,
            adapter_rank=16,
            router_mode="causal_proto",
            num_router_protos=64,
            router_sim_threshold=0.55,
            route_during_train=True,
            use_trunk_routing=True,
            router_context_blend=0.20,
            extend_loss_margin=0.25,
            router_use_idf=True,
            router_center_multi=False,
            train_revisit_conf_margin=0.02,
            preserve_input_novel_on_extend=True,
            defer_on_novel_streak=True,
            pc_mode="adaptive_rel",
            pc_thresh=0.24,
            ltm_null_logit=1.5,
            auto_lifecycle=True,
            max_experts=6,
            min_mature_steps=min(40, max(15, int(steps_per_dom * 0.20))),
            novel_confirm_steps=3,
            novel_surprise_threshold=0.45,
            dormant_age=10**9,
            prune_age=10**9,
            detach_decoder=False,
        )
        m6 = RealSelfStabilizingDMoE(**c6).to(DEV)
        if m6.lifecycle is not None:
            m6.lifecycle.freeze_embed_after_first = True
        gpm6 = NullSpaceProjector(
            m6,
            energy_threshold=0.97,
            max_rank_ratio=0.85,
            max_step_rank_ratio=0.26,
            step_rank_decay=0.84,
            defer_cov_commit=True,
            track_embeddings=True,
            freeze_shared_norms_on_update=True,
            freeze_unprojected_shared=True,
        )
        m6.attach_null_space_projector(gpm6)
        opt6 = torch.optim.AdamW([p for p in m6.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)
        return m6, gpm6, opt6, None

    c = dict(cfg)
    c["initial_experts"] = 1
    c["min_mature_steps"] = min(int(cfg["min_mature_steps"]), max(20, int(steps_per_dom * 0.85)))
    if base_arm == "Cog_v8_NoI5_NoRel":
        c["use_v8_core"] = False
        c["use_v7_core"] = True
        c["halt_use_energy_only"] = False
    elif base_arm == "Dense_v8_Matched":
        c["auto_lifecycle"] = False
        c["max_experts"] = 1
        c["d_ff_shared"] = 512
        c["d_ff_spec"] = 256

    model = RealSelfStabilizingDMoE(**c).to(DEV)
    if base_arm == "Cog_v8_FrozenRandomRouter" and model.context_router is not None:
        for p in model.context_router.parameters():
            p.requires_grad_(False)
    if base_arm == "Cog_v8_SequenceTop1" and model.context_router is not None:
        model.context_router.sequence_level = True
        model.context_router.tau_skip = -1.0
        model.context_router.tau_top2 = 2.0
        model.context_router.delta_mix = 0.0
        model.z_nuance_thresh = 1e9
        model.d_sem_nuance_thresh = 1e9

    if base_arm in ("Cog_v8", "Cog_v8_Mixed", "Cog_v8_SequenceTop1", "Cog_v8_NoI5_NoRel"):
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
            exempt_readout_heads=True,
        )
        model.attach_null_space_projector(gpm)
        groups = classify_v7_spgc_param_groups(
            model,
            shared_mode="acquire",
            specialist_mode="acquire",
            lr_muon=8e-3,
            lr_adam=3e-3,
            lr_consolidate=2e-3,
            trunk_prec_scale=0.85,
        )
        opt = MuonBONGOptimizer(
            groups,
            mode="acquire",
            bong_prior_pull=0.02,
            bong_max_prec=100.0,
            grad_clip=4.5,
        )
        gate = (
            LatentQuarantineGate(latent_dim=c["d_model"], seed=seed, max_bytes=16384, store_samples=False)
            if base_arm != "Cog_v8_Mixed"
            else None
        )
        return model, gpm, opt, gate
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2.5e-3, weight_decay=0.01)
        return model, None, opt, None


# --------------------------------------------------------------------------------------------------
# Evaluation Helpers
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
    for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        W = fit(ftr_n[tr], ytr[tr], lam)
        acc = float(((ftr_n[va] @ W).argmax(-1) == ytr[va]).float().mean())
        if acc > best_acc:
            best_lam, best_acc = lam, acc
    W = fit(ftr_n, ytr, best_lam)
    return float(((fte_n @ W).argmax(-1) == yte).float().mean())


@torch.no_grad()
def eval_banking77_v8(model: Optional[RealSelfStabilizingDMoE], bench: V8RealBenchmark) -> Dict[str, Any]:
    def feat_extract(texts: List[bytes]) -> Tuple[torch.Tensor, torch.Tensor]:
        if model is None:
            fb = bag_of_bytes_features(texts, SEQ_LEN)
            return fb, fb
        model.eval()
        is_v78 = bool(getattr(model, "use_v7_core", False))
        # Length-bucketed forward pass eliminates wasted trailing pad chunks while keeping 100% raw bytes
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        raw_out: Optional[torch.Tensor] = None
        norm_out: Optional[torch.Tensor] = None
        for s in range(0, len(order), 64):
            idxs = order[s : s + 64]
            b_txt = [texts[j] for j in idxs]
            max_b = max((len(t) for t in b_txt), default=8)
            L_b = min(SEQ_LEN, max(8, ((max_b + 3) // 4) * 4)) if is_v78 else min(SEQ_LEN, max(16, max_b))
            xb = encode_texts_v8(b_txt, L_b, DEV, pad_id=257 if is_v78 else 32)
            out = model(xb, update_stats=False)
            if is_v78:
                r_vec = out.get("z_expr_raw_pool", out["z_expr_pool"]).float().cpu()
                n_vec = out.get("z_expr_contrast", out["z_expr_pool"]).float().cpu()
            else:
                r_vec = out["latent"].float().mean(dim=1).cpu()
                n_vec = F.normalize(r_vec, dim=-1)
            if raw_out is None:
                raw_out = torch.empty((len(texts), r_vec.size(-1)), dtype=torch.float32)
                norm_out = torch.empty((len(texts), n_vec.size(-1)), dtype=torch.float32)
            idx_t = torch.tensor(idxs, dtype=torch.long)
            raw_out[idx_t] = r_vec
            norm_out[idx_t] = n_vec
        assert raw_out is not None and norm_out is not None
        return raw_out, norm_out

    ftr_raw, _ = feat_extract(bench.b77_tr_txt)
    fte_raw, fte_norm = feat_extract(bench.b77_te_txt)
    acc = ridge_probe(ftr_raw, bench.b77_tr_y, fte_raw, bench.b77_te_y, 77)
    fn = F.normalize(fte_norm, p=2, dim=-1)
    pos = [float((fn[u] @ fn[v]).item()) for u, v in bench.same_pairs]
    neg = [float((fn[u] @ fn[v]).item()) for u, v in bench.diff_pairs]
    return {"intent_77_acc": acc, "same_vs_overlap_diff_auc": roc_auc_pairs(pos, neg)}


def _ctx_eval_vec_v8(o: Dict[str, torch.Tensor], use_rel: bool = True) -> torch.Tensor:
    if use_rel and "z_ctx_pool" in o:
        zc = o["z_ctx"].float()
        T = zc.size(1)
        cm = o.get("chunk_content_mask")
        if cm is not None and cm[:, max(0, T // 2) :].any(dim=-1).all():
            m_t = cm[:, max(0, T // 2) :].unsqueeze(-1).float()
            tail = F.normalize((zc[:, max(0, T // 2) :] * m_t).sum(dim=1) / m_t.sum(dim=1).clamp_min(1.0), dim=-1)
        else:
            tail = F.normalize(zc[:, max(0, T // 2) :].mean(dim=1), dim=-1)
        return F.normalize(0.65 * o["z_ctx_pool"].float() + 0.35 * tail, dim=-1)
    if "z_ctx" in o:
        zc = o["z_ctx"].float()
        T = zc.size(1)
        tail = F.normalize(zc[:, max(0, T // 2) :].mean(dim=1), dim=-1)
        return F.normalize(0.5 * o["z_ctx_pool"].float() + 0.5 * tail, dim=-1)
    lat = o["latent"].float()
    T = lat.size(1)
    return F.normalize(0.5 * lat.mean(dim=1) + 0.5 * lat[:, max(0, T // 2) :].mean(dim=1), dim=-1)


@torch.no_grad()
def eval_triples_nuance_v8(
    model: Optional[RealSelfStabilizingDMoE], triples: List[Tuple[bytes, bytes, bytes]]
) -> Dict[str, float]:
    if not triples:
        return {"auc": 0.5, "selectivity": 1.0, "clean_trig": 0.0, "flip_trig": 0.0, "esc_gain": 0.0, "esc_margin_gain": 0.0}
    if model is None:
        jit_d = [1.0 - float((bag_of_bytes_features([a], SEQ_LEN) @ bag_of_bytes_features([j], SEQ_LEN).t()).item()) for a, j, _ in triples]
        flp_d = [1.0 - float((bag_of_bytes_features([a], SEQ_LEN) @ bag_of_bytes_features([f], SEQ_LEN).t()).item()) for a, _, f in triples]
        return {"auc": roc_auc_pairs(flp_d, jit_d), "selectivity": 1.0, "clean_trig": 0.0, "flip_trig": 0.0, "esc_gain": 0.0, "esc_margin_gain": 0.0}

    model.eval()
    xa = encode_texts_v8([t[0] for t in triples], SEQ_LEN, DEV, pad_id=32)
    xj = encode_texts_v8([t[1] for t in triples], SEQ_LEN, DEV, pad_id=32)
    xf = encode_texts_v8([t[2] for t in triples], SEQ_LEN, DEV, pad_id=32)
    oa = model(xa, update_stats=False)
    oj = model(xj, update_stats=False)
    of = model(xf, update_stats=False)
    use_rel = bool(getattr(model, "use_v8_core", False))
    za, zj, zf = _ctx_eval_vec_v8(oa, use_rel), _ctx_eval_vec_v8(oj, use_rel), _ctx_eval_vec_v8(of, use_rel)
    jit_dist = (1.0 - (za * zj).sum(dim=-1)).tolist()
    flip_dist = (1.0 - (za * zf).sum(dim=-1)).tolist()

    if "nuance_mask" in oa:
        clean_mask = oa["nuance_mask"][:, oa["nuance_mask"].size(1) // 2 :]
        flip_mask = of["nuance_mask"][:, of["nuance_mask"].size(1) // 2 :]
        clean_trig = float(clean_mask.float().mean().item())
        flip_trig = float(flip_mask.float().mean().item())
    else:
        clean_trig, flip_trig = 0.0, 0.0

    oa0 = model(xa, update_stats=False, force_thinking_steps=0)
    oj0 = model(xj, update_stats=False, force_thinking_steps=0)
    of0 = model(xf, update_stats=False, force_thinking_steps=0)
    za0, zj0, zf0 = _ctx_eval_vec_v8(oa0, use_rel), _ctx_eval_vec_v8(oj0, use_rel), _ctx_eval_vec_v8(of0, use_rel)
    flip_sep_full = float((1.0 - (za * zf).sum(dim=-1)).mean().item())
    jit_sep_full = float((1.0 - (za * zj).sum(dim=-1)).mean().item())
    flip_sep_t0 = float((1.0 - (za0 * zf0).sum(dim=-1)).mean().item())
    jit_sep_t0 = float((1.0 - (za0 * zj0).sum(dim=-1)).mean().item())

    return {
        "auc": roc_auc_pairs(flip_dist, jit_dist),
        "selectivity": (flip_trig + 1e-4) / (clean_trig + 1e-4) if "nuance_mask" in oa else 1.0,
        "clean_trig": clean_trig,
        "flip_trig": flip_trig,
        "esc_gain": flip_sep_full - flip_sep_t0,
        "esc_margin_gain": (flip_sep_full - jit_sep_full) - (flip_sep_t0 - jit_sep_t0),
    }


@torch.no_grad()
def eval_domain_v8(model: Optional[RealSelfStabilizingDMoE], bench: V8RealBenchmark, domain: str) -> Dict[str, Any]:
    wins = bench.corpora.test_windows[domain][:64, :SEQ_LEN].to(DEV)
    n = wins.size(0)
    triples = bench.nuance_pairs.get(domain, [])
    pref = wins[:, :64]
    fut64 = wins[:, 64:]
    hard_flips = [encode_texts_v8([fl[64:]], 64, DEV, pad_id=32)[0] for _, _, fl in triples[:16]]
    cand_x = torch.cat([fut64, torch.stack(hard_flips, dim=0)], dim=0) if hard_flips else fut64

    if model is None:
        pre_b = [bytes(w.cpu().numpy().astype(np.uint8).tobytes()) for w in pref]
        fut_b = [bytes(w.cpu().numpy().astype(np.uint8).tobytes()) for w in cand_x]
        fp = bag_of_bytes_features(pre_b, 64)
        ff = bag_of_bytes_features(fut_b, 64)
        ranks = (fp @ ff.t()).argsort(dim=-1, descending=True)
        r1 = float((ranks[:, 0] == torch.arange(n)).float().mean())
        r5 = float((ranks[:, :5] == torch.arange(n).unsqueeze(1)).any(dim=-1).float().mean())
        nd = eval_triples_nuance_v8(None, triples)
        return {
            "concept_r1": r1,
            "concept_r5": r5,
            "nuance_flip_auc": nd["auc"],
            "nuance_selectivity": nd["selectivity"],
            "nuance_escalation_gain": nd["esc_gain"],
            "nuance_esc_margin_gain": nd["esc_margin_gain"],
        }

    model.eval()

    o_pref = model(pref, update_stats=False)
    if model.world_model is not None:
        wm = o_pref.get("world_model_imagine") or model.world_model.imagine(o_pref["latent"], horizon=3)
        h_k = min(3, wm["imagined_latents"].size(2))
        z_pred_raw = wm["imagined_latents"][:, -1, :h_k, :].float().mean(dim=1)
        if bool(getattr(model, "use_v8_core", False)) and o_pref.get("pos_chunks") is not None and hasattr(model, "_pool_pos_chunks"):
            cm_p = o_pref.get("chunk_content_mask")
            if cm_p is not None:
                cm_f = cm_p.unsqueeze(-1).float()
                h_ctx_p = (o_pref["latent"].float() * cm_f).sum(dim=1) / cm_f.sum(dim=1).clamp_min(1.0)
            else:
                h_ctx_p = o_pref["latent"].float().mean(dim=1)
            pos_pre_p = model._pool_pos_chunks(o_pref["pos_chunks"], mask=cm_p)
            z_pred = model.project_wm_retrieval(
                z_pred_raw.to(dtype=model.byte_embed.weight.dtype), h_ctx=h_ctx_p, pos_pre=pos_pre_p
            ).float()
        else:
            z_pred = (
                model.project_wm_retrieval(z_pred_raw.to(dtype=model.byte_embed.weight.dtype)).float()
                if hasattr(model, "project_wm_retrieval")
                else z_pred_raw
            )
        if bool(getattr(model, "use_v8_core", False)) and hasattr(model.world_model, "encode_target_summary"):
            z_tgt = model.world_model.encode_target_summary(cand_x)
        else:
            z_tgt_raw = model.world_model.encode_target(cand_x)
            D_t = z_tgt_raw.size(-1)
            z_tgt = F.layer_norm(z_tgt_raw.float().detach(), (D_t,)).mean(dim=1)
    else:
        z_pred = o_pref["latent"][:, -1, :].float()
        o_cand = model(cand_x, update_stats=False)
        z_tgt = o_cand["latent"].float().mean(dim=1)
    sims = F.normalize(z_pred, dim=-1) @ F.normalize(z_tgt, dim=-1).t()
    top_r = sims.argsort(dim=-1, descending=True)
    gt = torch.arange(n, device=DEV)
    r1 = float((top_r[:, 0] == gt).float().mean().item())
    r5 = float((top_r[:, :5] == gt.unsqueeze(1)).any(dim=-1).float().mean().item())

    nd = eval_triples_nuance_v8(model, triples)
    return {
        "concept_r1": r1,
        "concept_r5": r5,
        "nuance_flip_auc": nd["auc"],
        "nuance_selectivity": nd["selectivity"],
        "nuance_escalation_gain": nd["esc_gain"],
        "nuance_esc_margin_gain": nd["esc_margin_gain"],
    }


@torch.no_grad()
def eval_routing_diagnostics_v8(model: RealSelfStabilizingDMoE, bench: V8RealBenchmark) -> Dict[str, Any]:
    model.eval()
    seqs = bench.mixed_switch_seqs.to(DEV)
    out = model(seqs, update_stats=False)
    is_v78 = bool(getattr(model, "use_v7_core", False))
    if is_v78 and "top_ids" in out:
        top_ids = out["top_ids"]
        k_used = out["k_used"]
        half = top_ids.size(1) // 2
        masked = torch.where(k_used > 0, top_ids[..., 0], -torch.ones_like(top_ids[..., 0]))
        switched = 0
        for i in range(seqs.size(0)):
            a = [int(v) for v in masked[i, :half].tolist() if v >= 0]
            b = [int(v) for v in masked[i, half:].tolist() if v >= 0]
            if a and b and Counter(a).most_common(1)[0][0] != Counter(b).most_common(1)[0][0]:
                switched += 1
        k_hist = {f"k{kv}": float((k_used == kv).float().mean().item()) for kv in (0, 1, 2)}
        sw_rate = float(switched / max(1, seqs.size(0)))
    else:
        sw_rate = 0.0
        k_hist = {"k0": 0.0, "k1": 1.0, "k2": 0.0}

    # I8 Functional routing NMI across 6 held-out semantic/functional roles
    rf_preds: List[int] = []
    for s in range(0, len(bench.func_role_texts), 64):
        xb_r = encode_texts_v8(bench.func_role_texts[s : s + 64], SEQ_LEN, DEV, pad_id=257 if is_v78 else 32)
        o_r = model(xb_r, update_stats=False)
        rf_preds.extend(o_r["assigned_experts"].detach().cpu().tolist())
    f_nmi = compute_nmi(bench.func_role_labels, rf_preds)
    return {"mid_seq_switch_rate": sw_rate, "k_dist": k_hist, "func_routing_nmi": f_nmi}


# --------------------------------------------------------------------------------------------------
# Track 1 Runner: Strictly From-Scratch with I5 Causal IIT + I6 Episodic Rule Induction
# --------------------------------------------------------------------------------------------------
def _sample_episodic_induction_batch(
    task: BeliefMeaningTask, n_episodes: int, gen: torch.Generator, n_shots: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create I6 zero-weight-update episodic rule-induction tasks with a fresh random `phi_ep` permutation.
    Returns `(x_sup_ep [E, 28], x_query [E, 24], y_query [E])`. Zero target leakage into `x_query`!
    """
    phi_ep = torch.randperm(K_CONCEPTS, generator=gen)
    inv_ep = torch.argsort(phi_ep)
    g_ep = torch.randperm(K_CONCEPTS, generator=gen)

    x_sup_list: List[torch.Tensor] = []
    x_q_rows: List[torch.Tensor] = []
    y_q_list: List[int] = []

    for _ in range(n_episodes):
        # Support shots on a fixed operation `op=0`, mapping pairs via `phi_ep -> g_ep`
        a_s = torch.randint(0, K_CONCEPTS, (n_shots,), generator=gen)
        b_s = torch.randint(0, K_CONCEPTS, (n_shots,), generator=gen)
        m_s = inv_ep[(phi_ep[a_s] + 3 * phi_ep[b_s]) % K_CONCEPTS]
        y_s = g_ep[m_s]
        c_z = torch.zeros(n_shots, dtype=torch.long)
        q_a = torch.full((n_shots,), Q_ATTR, dtype=torch.long)
        xs_s = torch.stack([torch.zeros(n_shots, dtype=torch.long), a_s + 4, b_s + 4, c_z + 4, c_z + 4, q_a], dim=-1)
        xb_s = encode_belief_chunks(xs_s)  # [4, 24]
        sup_seq = []
        for si in range(n_shots):
            ans_b = torch.tensor(list(f"Y{int(y_s[si].item()):02d} ".encode("ascii")), dtype=torch.long)
            sup_seq.append(torch.cat([xb_s[si], ans_b], dim=0))  # [28]
        # Stack per-shot `[n_shots, 28]` so every support query shares causal RoPE positions 0..5!
        x_sup_list.append(torch.stack(sup_seq, dim=0))  # [n_shots, 28]

        # Held-out query within the same novel episode (zero answer inside `x_query`!)
        # Re-query a shuffled composition drawn from support pairs to test true in-episode binding lookup/induction
        pick = int(torch.randint(0, n_shots, (1,), generator=gen).item())
        qa, qb, qy = int(a_s[pick].item()), int(b_s[pick].item()), int(y_s[pick].item())
        xs_q = torch.tensor([[0, qa + 4, qb + 4, 4, 4, Q_ATTR]], dtype=torch.long)
        x_q_rows.append(encode_belief_chunks(xs_q)[0])
        y_q_list.append(qy)

    return torch.stack(x_sup_list, dim=0), torch.stack(x_q_rows, dim=0), torch.tensor(y_q_list, dtype=torch.long)


def run_track1_arm_v8(arm: str, cfg: Dict[str, Any], seed: int, steps_scale: float = 1.0) -> Dict[str, Any]:
    set_all_seeds(seed)
    base_arm = _base_arm_name(arm)
    task = BeliefMeaningTask(seed=seed)
    gen = torch.Generator().manual_seed(seed * 97 + 1)

    if base_arm == "Upkeeper_M09":
        upk = BudgetedSurfaceDict(max_total_bytes=16384, min_sources_for_flip=2, min_flip_count=3, flip_threshold_ratio=0.45)
        for ph, n_st, src in [("p1", int(800 * steps_scale), "s0"), ("pb", int(20 * steps_scale), "burst"), ("p2", int(200 * steps_scale), "s1"), ("p3", int(200 * steps_scale), "s2")]:
            for st_i in range(n_st):
                xs, y = task.sample_stream_batch(ph, 64, gen)
                xb = encode_belief_chunks(xs, paraphrase=(ph == "p1" and (st_i % 2 == 1)))
                upk.update(xb, y, source_id=f"{src}_{st_i % 2 if ph in ('p1','p2','p3') else 0}")

        def eval_upk(pair_split: str, para: bool = False) -> float:
            hits, tot = 0, 0
            for q_tok in (Q_VAL, Q_ATTR, Q_PROP):
                xs, ys = task.sample_eval_subset(q_tok, "p3", [0, 1, 2], pair_split, None, None, 128, gen)
                preds = upk.predict_log_probs(encode_belief_chunks(xs, paraphrase=para)).argmax(dim=-1)
                hits += int((preds == ys).sum().item())
                tot += ys.numel()
            return hits / max(1, tot)

        return {
            "seen_acc": eval_upk("seen"),
            "heldout_unseen_acc": eval_upk("heldout"),
            "paraphrase_acc": eval_upk("heldout", para=True),
            "iit_causal_acc": 1.0 / K_CONCEPTS,
            "episodic_fewshot_acc": 1.0 / K_CONCEPTS,
            "burst_immunity_acc": eval_upk("seen"),
            "abstain_auroc": 0.5,
            "rollout_planning_delta": 0.0,
            "quarantine_bytes": int(upk.storage_bytes()),
            "muon_grok_drift": 0.0,
        }

    n_p1 = max(220, int(1300 * steps_scale))
    n_pb = max(8, int(20 * steps_scale))
    n_p2 = max(40, int(180 * steps_scale))
    n_p3 = max(40, int(180 * steps_scale))
    t1_cfg = dict(cfg)
    t1_cfg["min_mature_steps"] = n_p1 + 50
    model, gpm, opt, gate = build_v8_arm(base_arm, t1_cfg, seed, steps_per_dom=n_p1 + n_p2 + n_p3 + 50)
    cls_head = nn.Linear(t1_cfg["d_model"], K_CONCEPTS).to(DEV)
    use_iit = bool(getattr(model, "use_v8_core", False) and base_arm not in ("Cog_v8_NoI5_NoRel",))

    if isinstance(opt, MuonBONGOptimizer):
        groups = classify_v7_spgc_param_groups(
            model,
            shared_mode="acquire",
            specialist_mode="acquire",
            lr_muon=8e-3,
            lr_adam=3e-3,
            lr_consolidate=2e-3,
            wd_trunk=0.10,
            wd_fast=0.02,
            trunk_prec_scale=0.85,
        )
        groups.append({
            "name": "track1_cls",
            "params": list(cls_head.parameters()),
            "use_muon": False,
            "role": "fast",
            "mode": "acquire",
            "lr": 5e-3,
            "lr_consolidate": 3e-3,
            "weight_decay": 0.02,
            "prec_scale": 0.25,
        })
        opt = MuonBONGOptimizer(groups, mode="acquire", bong_prior_pull=0.02, bong_max_prec=100.0)
    else:
        opt = torch.optim.AdamW(
            list(p for p in model.parameters() if p.requires_grad) + list(cls_head.parameters()),
            lr=3e-3,
            weight_decay=0.05,
        )

    D_h = t1_cfg["d_model"]

    def step_t1(xs: torch.Tensor, y: torch.Tensor, source_id: str, ph: str, step_i: int) -> None:
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

        # Register established clean p1 signatures softly with QuarantineGate so burst flips fire from step 0!
        if ph == "p1" and gate is not None and step_i >= (n_p1 // 2) and (step_i % 2 == 0):
            with torch.no_grad():
                model.eval()
                o_reg = model(xb_raw, update_stats=False)
                lg_reg = cls_head(o_reg["latent"][:, -1, :])
                gate.triage_batch(
                    xb_raw.detach(),
                    y_dev.detach(),
                    lg_reg.detach(),
                    o_reg["z_expr_pool"].detach(),
                    source_id=source_id,
                    skip_familiar=False,
                    enable_quarantine=False,
                )

        if ph != "p1" and gate is not None:
            with torch.no_grad():
                model.eval()
                o_scr = model(xb, update_stats=False)
                lg_scr = cls_head(o_scr["latent"][:, -1, :])
                xb_act, y_act, flips = gate.triage_batch(
                    xb.detach(),
                    y_train.detach(),
                    lg_scr.detach(),
                    o_scr["z_expr_pool"].detach(),
                    source_id=source_id,
                    skip_familiar=False,
                    enable_quarantine=True,
                )
            if xb_act is None or y_act is None:
                return
            xb, y_train = xb_act, y_act
            if flips > 0 and isinstance(opt, MuonBONGOptimizer):
                opt.temper_precision(roles=("fast",), factor=0.25)

        model.train()
        saved = gpm.snapshot_monitored_weights() if (gpm is not None and gpm.p_null) else None
        fisher_sc = 1.0
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(xb, update_stats=True)
            lg = cls_head(out["latent"][:, -1, :])

            l_cls = F.cross_entropy(lg, y_train)
            l_para = (
                (1.0 - (out["z_expr_pool"][:b_sz].float() * out["z_expr_pool"][b_sz:].float()).sum(dim=-1)).mean()
                if ph == "p1"
                else 0.0
            )
            wm_l = out["world_model_loss"]["loss"] if out.get("world_model_loss") is not None else 0.0
            l_ab_cal = 0.0
            if out.get("abstain_logits") is not None and ph == "p1":
                ab_in = out["abstain_logits"][:b_sz, -1, 0].float()
                l_ab_cal = F.binary_cross_entropy_with_logits(ab_in, torch.zeros_like(ab_in))
                if step_i % 4 == 0:
                    xu_ood = torch.randint(33, 122, (16, 24), device=DEV)
                    ou_ood = model(xu_ood, update_stats=False)
                    if ou_ood.get("abstain_logits") is not None:
                        ab_ood = ou_ood["abstain_logits"][:, -1, 0].float()
                        l_ab_cal = l_ab_cal + F.binary_cross_entropy_with_logits(ab_ood, torch.ones_like(ab_ood))

            loss = l_cls + 0.06 * l_para + 0.03 * wm_l + 0.08 * l_ab_cal

            # I5 Causal Abstraction Interchange-Intervention Training (simultaneous m+ctx swap in p1 -> task-10100 0.844/0.773!)
            if use_iit and ph == "p1" and step_i >= (n_p1 // 3) and (step_i % 2 == 0) and b_sz >= 64:
                xs_b, xs_d = xs[:32], xs[32:64]
                x_b_in, x_d_in = xb_raw[:32], xb_raw[32:64]
                o_cf_m = model.forward_interchange(
                    x_b_in, x_d_in, intervene_chunks=(0, 3), dim_slice=(0, D_h // 2), force_thinking_steps=1
                )
                y_cf_m, _, _ = task._compute_targets(
                    xs_d[:, 0], xs_d[:, 1] - 4, xs_d[:, 2] - 4, xs_b[:, 3] - 4, xs_b[:, 4] - 4, xs_b[:, 5], "p1"
                )
                o_cf_c = model.forward_interchange(
                    x_b_in, x_d_in, intervene_chunks=(3, 5), dim_slice=(D_h // 2, (3 * D_h) // 4), force_thinking_steps=1
                )
                y_cf_c, _, _ = task._compute_targets(
                    xs_b[:, 0], xs_b[:, 1] - 4, xs_b[:, 2] - 4, xs_d[:, 3] - 4, xs_d[:, 4] - 4, xs_b[:, 5], "p1"
                )
                loss = loss + 0.30 * F.cross_entropy(cls_head(o_cf_m["latent"][:, -1, :]), y_cf_m.to(DEV))
                loss = loss + 0.30 * F.cross_entropy(cls_head(o_cf_c["latent"][:, -1, :]), y_cf_c.to(DEV))

            # I6 Episodic In-Context Memory Training (`extract_episode_kv` top_k=6 per aligned shot -> post-Layer-0 LTM)
            if use_iit and ph == "p1" and (step_i % 3 == 1) and step_i >= (n_p1 // 5):
                x_sup_ep, x_q_ep, y_q_ep = _sample_episodic_induction_batch(task, 16, gen)
                e_sz, n_sh, l_sh = x_sup_ep.shape
                ep_kv_flat = model.extract_episode_kv(x_sup_ep.view(e_sz * n_sh, l_sh).to(DEV), top_k=6, min_nll=0.0)
                if ep_kv_flat is not None:
                    k_sh = ep_kv_flat["ep_keys"].size(1)
                    d_kv = ep_kv_flat["ep_keys"].size(-1)
                    ep_kv = {
                        "ep_keys": ep_kv_flat["ep_keys"].view(e_sz, n_sh * k_sh, d_kv),
                        "ep_vals": ep_kv_flat["ep_vals"].view(e_sz, n_sh * k_sh, d_kv),
                        "ep_mask": ep_kv_flat["ep_mask"].view(e_sz, n_sh * k_sh),
                    }
                    o_ep = model(x_q_ep.to(DEV), update_stats=False, episode_memory=ep_kv)
                    loss = loss + 0.40 * F.cross_entropy(cls_head(o_ep["latent"][:, -1, :]), y_q_ep.to(DEV))

            # I10 consistency fisher scale (keep 1.0 during pb so burst never weakens consolidated cls precision!)
            acc_batch = float((lg[:b_sz].detach().argmax(-1) == y_train[:b_sz]).float().mean().item())
            fisher_sc = 1.0 if ph == "pb" else (0.25 + 1.15 * acc_batch)

        opt.zero_grad()
        loss.backward()
        if gpm is not None and gpm.p_null:
            gpm.project_gradients()
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(cls_head.parameters()), 1.5)
        if isinstance(opt, MuonBONGOptimizer):
            opt.step(fisher_scale=fisher_sc)
        else:
            opt.step()
        if saved is not None:
            gpm.project_parameter_deltas(saved)
        if gpm is not None and getattr(gpm, "defer_cov_commit", False):
            gpm.commit_pending()

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
        step_t1(xs, y, f"s_{st % 2}", "p1", st)
    mid_held = eval_t1("p1", "heldout")

    if gpm is not None:
        gpm.update_basis_for_domain(0)
        if isinstance(opt, MuonBONGOptimizer):
            opt.set_group_mode("trunk", "consolidate")

    for st in range(n_pb):
        xs, y = task.sample_stream_batch("pb", 64, gen)
        step_t1(xs, y, "adversarial_burst", "pb", st)
    with torch.no_grad():
        model.eval()
        xs_s, ys_s = task.sample_eval_subset(Q_ATTR, "p1", [0, 1, 2], "seen", None, None, 128, gen)
        o_s = model(encode_belief_chunks(xs_s).to(DEV), update_stats=False)
        burst_acc = float((cls_head(o_s["latent"][:, -1, :]).argmax(-1).cpu() == ys_s).float().mean())

    for st in range(n_p2):
        xs, y = task.sample_stream_batch("p2", 64, gen)
        step_t1(xs, y, f"s2_{st % 2}", "p2", st)
    for st in range(n_p3):
        xs, y = task.sample_stream_batch("p3", 64, gen)
        step_t1(xs, y, f"s3_{st % 2}", "p3", st)

    with torch.no_grad():
        model.eval()
        xs_k, _ = task.sample_eval_subset(Q_VAL, "p3", [0, 1, 2], "heldout", None, None, 64, gen)
        xs_k2, _ = task.sample_eval_subset(Q_ATTR, "p3", [0, 1, 2], "heldout", None, None, 64, gen)
        xk_b = encode_belief_chunks(xs_k).to(DEV)
        xu_b = torch.randint(33, 122, (64, 24), device=DEV)
        ok = model(xk_b, update_stats=False)
        ou = model(xu_b, update_stats=False)
        if ok.get("abstain_logits") is not None and ou.get("abstain_logits") is not None:
            sk = (0.5 * ok["abstain_logits"][:, -1, 0].sigmoid() + 0.5 * (ok["nuance_score"].mean(dim=-1) / 3.0)).tolist()
            su = (0.5 * ou["abstain_logits"][:, -1, 0].sigmoid() + 0.5 * (ou["nuance_score"].mean(dim=-1) / 3.0)).tolist()
        else:
            sk = ok["nuance_score"].mean(dim=-1).tolist()
            su = ou["nuance_score"].mean(dim=-1).tolist()
        abstain_auroc = roc_auc_pairs(su, sk)

        # I5 Causal IIT verification (swap meaning prefix subspace [0:D//2] from xs_iit_d into xs_iit_b on seen test pairs)
        xs_iit_b, _ = task.sample_eval_subset(Q_VAL, "p3", [0, 1, 2], "seen", None, None, 64, gen)
        xs_iit_d, _ = task.sample_eval_subset(Q_ATTR, "p3", [0, 1, 2], "seen", None, None, 64, gen)
        xk_iit_b = encode_belief_chunks(xs_iit_b).to(DEV)
        xd_iit_b = encode_belief_chunks(xs_iit_d).to(DEV)
        if hasattr(model, "forward_interchange") and getattr(model, "use_v7_core", False):
            o_iit_eval = model.forward_interchange(xk_iit_b, xd_iit_b, intervene_chunks=(0, 3), dim_slice=(0, D_h // 2))
            y_cf_gt, _, _ = task._compute_targets(
                xs_iit_d[:, 0], xs_iit_d[:, 1] - 4, xs_iit_d[:, 2] - 4, xs_iit_b[:, 3] - 4, xs_iit_b[:, 4] - 4, xs_iit_b[:, 5], "p3"
            )
            iit_acc = float((cls_head(o_iit_eval["latent"][:, -1, :]).argmax(-1).cpu() == y_cf_gt).float().mean().item())
        else:
            iit_acc = 1.0 / K_CONCEPTS

        # Held-out I6 Episodic Few-Shot Induction (novel swapped phi_ep, ZERO weight updates!)
        x_sup_t, x_q_t, y_q_t = _sample_episodic_induction_batch(task, 64, gen)
        e_sz_t, n_sh_t, l_sh_t = x_sup_t.shape
        ep_kv_flat_t = model.extract_episode_kv(x_sup_t.view(e_sz_t * n_sh_t, l_sh_t).to(DEV), top_k=6, min_nll=0.0)
        if ep_kv_flat_t is not None:
            k_sh_t = ep_kv_flat_t["ep_keys"].size(1)
            d_kv_t = ep_kv_flat_t["ep_keys"].size(-1)
            ep_kv_t = {
                "ep_keys": ep_kv_flat_t["ep_keys"].view(e_sz_t, n_sh_t * k_sh_t, d_kv_t),
                "ep_vals": ep_kv_flat_t["ep_vals"].view(e_sz_t, n_sh_t * k_sh_t, d_kv_t),
                "ep_mask": ep_kv_flat_t["ep_mask"].view(e_sz_t, n_sh_t * k_sh_t),
            }
            o_ep_t = model(x_q_t.to(DEV), update_stats=False, episode_memory=ep_kv_t)
            ep_acc = float((cls_head(o_ep_t["latent"][:, -1, :]).argmax(-1).cpu() == y_q_t).float().mean().item())
        else:
            ep_acc = 1.0 / K_CONCEPTS

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
        "iit_causal_acc": iit_acc,
        "episodic_fewshot_acc": ep_acc,
        "burst_immunity_acc": burst_acc,
        "abstain_auroc": abstain_auroc,
        "rollout_planning_delta": rollout_delta,
        "muon_grok_drift": mid_held - fin_held,
        "quarantine_bytes": int(gate.storage_bytes()) if gate is not None else 0,
    }


# --------------------------------------------------------------------------------------------------
# Track 2 Runner (Consolidated v8 + Cog_v6_Ref Same-Script Comparison)
# --------------------------------------------------------------------------------------------------
def run_track2_arm_v8(
    arm: str,
    cfg: Dict[str, Any],
    bench: V8RealBenchmark,
    seed: int,
    steps_per_dom: int,
    teacher: Optional[OfflineTeacherGramEncoder] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    base_arm = _base_arm_name(arm)
    use_teacher = arm.endswith("_I9") and teacher is not None

    if base_arm == "BagOfBytes_Null":
        per_dom = {d: eval_domain_v8(None, bench, d) for d in DOMAINS}
        b77 = eval_banking77_v8(None, bench)
        hf = eval_triples_nuance_v8(None, bench.heldout_family_pairs)
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

    model, gpm, opt, gate = build_v8_arm(arm, cfg, seed, steps_per_dom=steps_per_dom)
    if base_arm == "Untrained_Cog_v8":
        per_dom = {d: eval_domain_v8(model, bench, d) for d in DOMAINS}
        b77 = eval_banking77_v8(model, bench)
        hf = eval_triples_nuance_v8(model, bench.heldout_family_pairs)
        rd = eval_routing_diagnostics_v8(model, bench)
        return {
            "arm": arm,
            "teacher_informed": False,
            "train_seconds": time.perf_counter() - t0,
            "banking77": b77,
            "heldout_families": hf,
            "routing_diag": rd,
            "per_domain_final": per_dom,
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
    dec_opt = (
        torch.optim.AdamW(model.decoder.parameters(), lr=4e-3, weight_decay=0.01)
        if getattr(model, "use_v7_core", False)
        else None
    )

    # P0: I7 Procedural Structural Pre-Training (zero HELDOUT overlap)
    if getattr(model, "use_v8_core", False) and base_arm not in ("Cog_v8_NoI5_NoRel",):
        n_i7 = max(12, int(steps_per_dom * 0.35))
        model.train()
        for _ in range(n_i7):
            x_i7, eq_i7, c_i7, nt_i7, st_i7 = generate_i7_procedural_batch(16, rng_py, seq_len=SEQ_LEN, device=DEV)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                o_i7 = model(x_i7, update_stats=False)
                l_i7 = model.compute_v8_non_token_loss(
                    o_i7,
                    x=x_i7,
                    equiv_labels=eq_i7,
                    contra_pair_idx=c_i7,
                    nuance_targets=nt_i7,
                    struct_targets=st_i7,
                    wm_weight=0.25,
                    fsp_weight=0.10,
                    equiv_weight=1.25,
                    contra_weight=1.25,
                    think_weight=0.25,
                    struct_weight=0.35,
                )
            opt.zero_grad()
            l_i7["loss"].backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.5)
            if isinstance(opt, MuonBONGOptimizer):
                opt.step(fisher_scale=0.5)
            else:
                opt.step()

    peak_r5: Dict[str, float] = {}
    peak_auc: Dict[str, float] = {}
    events_log: List[Dict[str, Any]] = []
    skipped_steps = 0
    backward_passes = 0
    gstep = 0
    mid_b77_acc: Optional[float] = None
    is_v6 = (base_arm == "Cog_v6_Ref")

    def train_one_step(dom: str, allow_quarantine: bool) -> None:
        nonlocal opt, dec_opt, gstep, skipped_steps, backward_passes
        gstep += 1

        if is_v6:
            w = bench.corpora.sample_train(dom, 16, gen_np)[:, : SEQ_LEN + 1].to(DEV)
            x6, y6 = w[:, :-1], w[:, 1:]
            if gpm is not None:
                gpm.is_tracking = (gstep % 4 == 1) or (len(gpm.covariances) == 0)
            saved = gpm.snapshot_monitored_weights() if (gpm is not None and gpm.p_null) else None
            model.train()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model(x6, update_stats=True, targets=y6)
                evs = out.get("lifecycle_events", [])
                spawned_now = any(e.startswith("spawned_") for e in evs)
                for ev in evs:
                    events_log.append({"step": gstep, "domain": dom, "event": ev})
                if spawned_now:
                    if gpm is not None and getattr(gpm, "defer_cov_commit", False):
                        gpm.discard_pending()
                    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)
                    out = model(x6, update_stats=True, targets=y6)
                    saved = gpm.snapshot_monitored_weights() if (gpm is not None and gpm.p_null) else None
                elif model.lifecycle is not None and model.lifecycle.should_defer_update:
                    if gpm is not None and getattr(gpm, "defer_cov_commit", False):
                        gpm.discard_pending()
                    skipped_steps += 1
                    return
                loss_d = model.compute_total_loss(out, y6, wm_weight=0.15)
            opt.zero_grad()
            loss_d["loss"].backward()
            backward_passes += 1
            if gpm is not None and gpm.p_null:
                gpm.project_gradients()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            if saved is not None:
                gpm.project_parameter_deltas(saved)
            if gpm is not None and getattr(gpm, "defer_cov_commit", False):
                gpm.commit_pending()
            return

        x_b, y_eq, (ca, cb), (ha, hb), nt, st_y, gm_mask = bench.sample_v8_stream_batch(dom, gen_np, rng_py)
        t_gram = teacher.batch_gram(x_b) if use_teacher else None

        if gpm is not None:
            gpm.is_tracking = (gstep % 4 == 1) or (len(gpm.covariances) == 0)
        saved = gpm.snapshot_monitored_weights() if (gpm is not None and gpm.p_null) else None

        model.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(x_b, update_stats=True)
            # Single-pass inline quarantine (eliminates v7's redundant extra forward pass!)
            if gate is not None and allow_quarantine and gstep > 30:
                err_s = out["nuance_score"].detach().mean(dim=-1)
                xb_a, _, _ = gate.triage_continuous(
                    x_b.detach(),
                    err_s.unsqueeze(-1),
                    torch.zeros_like(err_s).unsqueeze(-1),
                    out["z_expr_pool"].detach(),
                    source_id=f"{dom}_{gstep % 2}",
                    skip_familiar=False,
                    enable_quarantine=True,
                )
                if xb_a is None or xb_a.size(0) < 4:
                    if gpm is not None and getattr(gpm, "defer_cov_commit", False):
                        gpm.discard_pending()
                    skipped_steps += 1
                    return

            loss_d = model.compute_v8_non_token_loss(
                out,
                x=x_b,
                equiv_labels=y_eq,
                contra_pair_idx=(ca, cb),
                hard_neg_pair_idx=(ha, hb),
                nuance_targets=nt,
                teacher_gram=t_gram,
                gram_mask=gm_mask,
                struct_targets=st_y,
                wm_weight=0.75,
                fsp_weight=0.22,
                sigreg_weight=0.25,
                equiv_weight=1.75,
                contra_weight=1.40,
                think_weight=0.35,
                erc_weight=0.20,
                balance_weight=0.40,
                teacher_weight=0.70 if use_teacher else 0.0,
                struct_weight=0.22,
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
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 4.5)
        if isinstance(opt, MuonBONGOptimizer):
            wm_err_v = float(loss_d["wm_loss"].item()) if loss_d.get("wm_loss") is not None else 0.5
            f_scale = float(np.clip(1.35 - 0.30 * wm_err_v, 0.25, 1.35))
            opt.step(fisher_scale=f_scale)
        else:
            opt.step()
        if saved is not None:
            gpm.project_parameter_deltas(saved)
        if gpm is not None and getattr(gpm, "defer_cov_commit", False):
            gpm.commit_pending()

        # Train detached LoRA verbalizer strictly behind `.detach()` every 4 steps (0 core grad!)
        if dec_opt is not None and (gstep % 4 == 0) and x_b.size(1) > model.chunk_size:
            dec_req = [p for p in model.decoder.parameters() if p.requires_grad]
            if dec_req:
                o_raw_det = {
                    "latent": out["latent"][:4].detach(),
                    "top_ids": out["top_ids"][:4].detach(),
                    "top_w": out["top_w"][:4].detach(),
                    "num_experts": out.get("num_experts", 1),
                }
                rlg = model.render_text_detached(o_raw_det)[:, :-1, :, :]
                tgt_det = x_b[:4, model.chunk_size :]
                dloss = F.cross_entropy(rlg.reshape(-1, 256), tgt_det.reshape(-1), ignore_index=257)
                dec_opt.zero_grad()
                dloss.backward()
                dec_opt.step()

        if spawned_now:
            if isinstance(opt, MuonBONGOptimizer):
                groups = classify_v7_spgc_param_groups(
                    model,
                    shared_mode="consolidate",
                    specialist_mode="acquire",
                    lr_muon=8e-3,
                    lr_adam=3e-3,
                    lr_consolidate=2e-3,
                    trunk_prec_scale=0.85,
                )
                opt.rebuild_preserving_state(groups)
            if dec_opt is not None:
                dec_req_new = [p for p in model.decoder.parameters() if p.requires_grad]
                if dec_req_new:
                    dec_opt = torch.optim.AdamW(dec_req_new, lr=4e-3, weight_decay=0.01)

    if base_arm == "Cog_v8_Mixed":
        tot = steps_per_dom * len(DOMAINS)
        for st_m in range(tot):
            dom = DOMAINS[int(gen_np.integers(0, len(DOMAINS)))]
            train_one_step(dom, allow_quarantine=False)
            if st_m == tot // 2:
                mid_b77_acc = eval_banking77_v8(model, bench)["intent_77_acc"]
    else:
        for d_idx, dom in enumerate(DOMAINS):
            for _ in range(steps_per_dom):
                train_one_step(dom, allow_quarantine=(base_arm not in ("Cog_v8_SequenceTop1", "Cog_v6_Ref")))
            if d_idx < len(DOMAINS) - 1:
                ev_now = eval_domain_v8(model, bench, dom)
                peak_r5[dom] = ev_now["concept_r5"]
                peak_auc[dom] = ev_now["nuance_flip_auc"]
            if d_idx == 2:
                mid_b77_acc = eval_banking77_v8(model, bench)["intent_77_acc"]

    per_dom = {d: eval_domain_v8(model, bench, d) for d in DOMAINS}
    b77 = eval_banking77_v8(model, bench)
    hf = eval_triples_nuance_v8(model, bench.heldout_family_pairs)
    rd = eval_routing_diagnostics_v8(model, bench)

    fg_r5 = float(np.mean([peak_r5[d] - per_dom[d]["concept_r5"] for d in DOMAINS[:-1]])) if peak_r5 else None
    fg_auc = float(np.mean([peak_auc[d] - per_dom[d]["nuance_flip_auc"] for d in DOMAINS[:-1]])) if peak_auc else None
    at_cap = 0
    if gpm is not None and gpm.bases:
        ratios = [b.size(1) / max(1, b.size(0)) for b in gpm.bases.values() if b is not None]
        at_cap = sum(1 for r in ratios if r >= 0.84)

    # Fair held-out BPB check on `wiki/story/code/math/forum` (detached verbalizer for v8/Dense; native for v6)
    with torch.no_grad():
        model.eval()
        bpbs = []
        for dom in DOMAINS:
            if is_v6:
                w_full = bench.corpora.test_windows[dom][:64, : SEQ_LEN + 1].to(DEV)
                x_ev, y_ev = w_full[:, :-1], w_full[:, 1:]
                o_ev = model(x_ev, update_stats=False)
                ce_ev = F.cross_entropy(o_ev["output"].reshape(-1, 256), y_ev.reshape(-1))
            else:
                w_eval = bench.corpora.test_windows[dom][:64, :SEQ_LEN].to(DEV)
                o_ev = model(w_eval, update_stats=False)
                lg_next = o_ev["output"][:, :-1, :, :]
                y_next = w_eval[:, model.chunk_size :]
                ce_ev = F.cross_entropy(lg_next.reshape(-1, 256), y_next.reshape(-1))
            bpbs.append(float(ce_ev.item() / math.log(2.0)))
        diag_bpb = float(np.mean(bpbs))

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
        "routing_diag": rd,
        "per_domain_final": per_dom,
        "mean_final_r1": float(np.mean([per_dom[d]["concept_r1"] for d in DOMAINS])),
        "mean_final_r5": float(np.mean([per_dom[d]["concept_r5"] for d in DOMAINS])),
        "mean_nuance_flip_auc": float(np.mean([per_dom[d]["nuance_flip_auc"] for d in DOMAINS])),
        "mean_nuance_selectivity": float(np.mean([per_dom[d]["nuance_selectivity"] for d in DOMAINS])),
        "mean_escalation_gain": float(np.mean([per_dom[d]["nuance_escalation_gain"] for d in DOMAINS])),
        "mean_esc_margin_gain": float(np.mean([per_dom[d]["nuance_esc_margin_gain"] for d in DOMAINS])),
        "forgetting_r5": fg_r5,
        "forgetting_nuance_auc": fg_auc,
        "gpm_at_cap_count": at_cap,
        "quarantine_bytes": int(gate.storage_bytes()) if gate is not None else 0,
        "detached_renderer_diag_bpb": diag_bpb,
        "events": events_log[-30:],
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
    if all("routing_diag" in r for r in runs):
        out["mid_seq_switch_rate"] = {"mean": float(np.mean([r["routing_diag"]["mid_seq_switch_rate"] for r in runs]))}
        out["func_routing_nmi"] = {"mean": float(np.mean([r["routing_diag"]["func_routing_nmi"] for r in runs]))}
        out["k_dist"] = {kv: float(np.mean([r["routing_diag"]["k_dist"][kv] for r in runs])) for kv in ("k0", "k1", "k2")}
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
    if len(rows) >= 4:
        raise RuntimeError(f"Protocol violation: dev tuning budget exhausted ({len(rows)} >= 4 trials recorded in tuning_log.json)")
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
    ap.add_argument("--z_nuance", type=float, default=DEFAULT_V8_CFG["z_nuance_thresh"])
    ap.add_argument("--d_sem", type=float, default=DEFAULT_V8_CFG["d_sem_nuance_thresh"])
    ap.add_argument("--cluster_min", type=int, default=DEFAULT_V8_CFG["nuance_cluster_min"])
    ap.add_argument("--cluster_cos", type=float, default=DEFAULT_V8_CFG["nuance_cluster_cos"])
    ap.add_argument("--arms", type=str, nargs="+", default=None)
    args = ap.parse_args()

    cfg = dict(DEFAULT_V8_CFG)
    cfg["z_nuance_thresh"] = float(args.z_nuance)
    cfg["d_sem_nuance_thresh"] = float(args.d_sem)
    cfg["nuance_cluster_min"] = int(args.cluster_min)
    cfg["nuance_cluster_cos"] = float(args.cluster_cos)

    if args.mode == "smoke":
        eval_split = "dev"
        seeds = args.seeds or [0]
        steps_per_dom = args.steps or 25
        default_arms = ["BagOfBytes_Null", "Untrained_Cog_v8", "Upkeeper_M09", "Dense_v8_Matched", "Cog_v6_Ref", "Cog_v8", "Cog_v8_I9"]
        out_name = args.out_name or "smoke_v8"
    elif args.mode == "dev":
        eval_split = "dev"
        seeds = args.seeds or [100]
        steps_per_dom = args.steps or 200
        default_arms = ["BagOfBytes_Null", "Upkeeper_M09", "Dense_v8_Matched", "Cog_v6_Ref", "Cog_v8_NoI5_NoRel", "Cog_v8", "Cog_v8_I9", "Cog_v8_Mixed_I9"]
        out_name = args.out_name or f"dev_{int(time.time())}"
    else:
        eval_split = "test"
        seeds = args.seeds or [0, 1, 2]
        steps_per_dom = args.steps or 250
        default_arms = [
            "BagOfBytes_Null",
            "Untrained_Cog_v8",
            "Upkeeper_M09",
            "Dense_v8_Matched",
            "Cog_v6_Ref",
            "Cog_v8_FrozenRandomRouter",
            "Cog_v8_SequenceTop1",
            "Cog_v8_NoI5_NoRel",
            "Cog_v8",
            "Cog_v8_I9",
            "Cog_v8_Mixed_I9",
        ]
        out_name = args.out_name or "test_final"

    arms = args.arms or default_arms
    out_dir = RESULTS_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    corpora = RealCorpora(seq_len=SEQ_LEN, eval_split=eval_split)
    bench = V8RealBenchmark(corpora, eval_split=eval_split)
    teacher: Optional[OfflineTeacherGramEncoder] = None
    if any(a.endswith("_I9") for a in arms):
        teacher = OfflineTeacherGramEncoder()

    t1_arms = {"Upkeeper_M09", "Dense_v8_Matched", "Cog_v8_FrozenRandomRouter", "Cog_v8_SequenceTop1", "Cog_v8_NoI5_NoRel", "Cog_v8"}
    results_by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
    t1_scale = 0.1 if args.mode == "smoke" else 1.0

    for seed in seeds:
        for arm in arms:
            if arm == "Upkeeper_M09":
                t1 = run_track1_arm_v8(arm, cfg, seed, steps_scale=t1_scale)
                res = {
                    "arm": arm,
                    "teacher_informed": False,
                    "seed": seed,
                    "banking77": {"intent_77_acc": 1.0 / 77.0, "same_vs_overlap_diff_auc": 0.5},
                    "track1": t1,
                }
            else:
                res = run_track2_arm_v8(arm, cfg, bench, seed, steps_per_dom=steps_per_dom, teacher=teacher)
                if arm in t1_arms:
                    res["track1"] = run_track1_arm_v8(arm, cfg, seed, steps_scale=t1_scale)
            results_by_arm[arm].append(res)
            (out_dir / f"seed{seed}_{arm}.json").write_text(json.dumps(res, indent=2))
            t1_str = (
                f" | T1 unseen={res['track1']['heldout_unseen_acc']:.3f} iit={res['track1']['iit_causal_acc']:.3f} ep={res['track1']['episodic_fewshot_acc']:.3f} burst={res['track1']['burst_immunity_acc']:.3f}"
                if "track1" in res
                else ""
            )
            print(
                f"[{out_name} s{seed} {arm}] "
                f"b77_acc={res['banking77']['intent_77_acc']:.3f} b77_auc={res['banking77']['same_vs_overlap_diff_auc']:.3f} "
                f"r5={res.get('mean_final_r5', 0):.3f} nf={res.get('mean_nuance_flip_auc', 0):.3f} "
                f"hf={res.get('heldout_families', {}).get('auc', 0):.3f} sw={res.get('routing_diag', {}).get('mid_seq_switch_rate', 0):.3f} "
                f"fnmi={res.get('routing_diag', {}).get('func_routing_nmi', 0):.3f} sec={res.get('train_seconds', 0):.1f}{t1_str}",
                flush=True,
            )

    summary = {
        "meta": {
            "mode": args.mode,
            "eval_split": eval_split,
            "seeds": seeds,
            "steps_per_dom": steps_per_dom,
            "utc": datetime.now(timezone.utc).isoformat(),
            "device": str(DEV),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cfg": cfg,
        },
        "arms": {a: _agg(results_by_arm[a]) for a in arms},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.mode == "dev":
        _append_dev_tuning_log(out_name, summary)
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
