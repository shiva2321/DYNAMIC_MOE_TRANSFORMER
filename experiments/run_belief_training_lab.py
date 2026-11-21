"""
Experiment Runner for `belief_training_lab_v1` (Belief & Meaning Training Algorithms).

Protocol (research-protocol.md & results/belief_training_lab/PREREG_v1.md):
  - DEV_SEEDS = [100, 101, 102, 103, 104] for --mode tune ONLY -> tuning_log_v1.json
  - TEST_SEEDS = [0..19] (n = 20) for ONE frozen --mode test -> results_v1.json
  - Strict causality gate (<= 1e-5), FLOP accounting (torch.utils.flop_counter.FlopCounterMode),
    clocked wall-clock timing, honest byte accounting, paired 95% bootstrap CIs,
    and script-evaluated boolean falsifiers F1..F8.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmoe.belief_training import (
    BONGPrecisionOptimizer,
    BeliefMeaningTask,
    BudgetedSurfaceDict,
    ConceptBeliefBackbone,
    ConceptBeliefStore,
    K_CONCEPTS,
    MuPCProspectiveNetwork,
    NullSpaceGuard,
    Q_ATTR,
    Q_PROP,
    Q_VAL,
    SparseConceptQuantizer,
    continual_backprop_recycle,
    equivalence_and_contradiction_losses,
    sigreg_loss,
)

RESULTS_DIR = ROOT / "results" / "belief_training_lab"
DEV_SEEDS = [100, 101, 102, 103, 104]
TEST_SEEDS = list(range(20))

ALL_ARMS = [
    "B0_SurfaceDict",
    "B1_Dense_SGD",
    "B2_Dense_Replay",
    "B3_Dense_NullSpace",
    "B4_JointOracle",
    "Null_Untrained",
    "Null_ShuffledTargets",
    "Null_BlindZeroInput",
    "Idea1_ConceptWCS",
    "Idea1_WakeOnly",
    "Idea1_NoCtxSplit",
    "Idea2_EquivContra",
    "Idea2_ShuffledEquiv",
    "Idea2_NoSIGReg",
    "Idea3_MuPC_BONG_Pure",
    "Idea3_Backbone_BONG",
    "Idea3_MuPC_Adam",
    "Idea_Triad_All3",
]


@dataclass
class LabConfig:
    batch_size: int = 128
    steps_p1: int = 3600
    steps_pb: int = 10
    steps_p2: int = 800
    steps_p3: int = 600
    confirm_frac: float = 0.25
    lr_dense: float = 6e-3
    lr_dense_post: float = 4e-3
    wd_p1: float = 0.22
    wd_bong_p1: float = 0.10
    wd_post: float = 0.01
    lr_idea2: float = 6e-3
    lr_bong_bb: float = 1.2e-2
    lr_mupc_bong: float = 2e-3
    lr_mupc_adam: float = 3e-3
    wd_mupc: float = 0.02
    lr_chunk: float = 1e-2
    lr_bong_chunk: float = 3e-2
    lambda_equiv: float = 0.35
    lambda_contra: float = 0.20
    lambda_sigreg: float = 0.03
    chunk_size: int = 128
    chunk_steps: int = 6
    sleep_period: int = 100
    sleep_steps: int = 4
    elastic_pull: float = 0.001
    rho_thresh: float = 0.15
    tau_pc_skip: float = 0.10
    max_store_bytes: int = 16384
    num_atoms: int = 64
    k_train: int = 10
    eta_z: float = 0.10
    beta_out: float = 0.10


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git(args: List[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=str(ROOT), capture_output=True, text=True, timeout=20, check=True
        ).stdout.strip()
    except Exception:
        return "UNKNOWN"


def build_run_meta(exp_id: str, seeds: List[int], tuning_seeds: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "exp_id": exp_id,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "seeds": [int(s) for s in seeds],
        "tuning_seeds": [int(s) for s in tuning_seeds],
        "config": config,
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class FlopTracker:
    """
    Measure exact per-operation-type FLOPs using torch.utils.flop_counter.FlopCounterMode
    once per step-kind and accumulate per-seed FLOPs from actual executed step counts.
    """

    def __init__(self) -> None:
        self.per_kind_flops: Dict[str, int] = {}
        self.counts: Dict[str, int] = {}

    def record(self, kind: str, fn: Any, count_multiplier: int = 1) -> Any:
        if kind not in self.per_kind_flops:
            with FlopCounterMode(display=False) as fc:
                res = fn()
            self.per_kind_flops[kind] = int(fc.get_total_flops())
            self.counts[kind] = self.counts.get(kind, 0) + count_multiplier
            return res
        self.counts[kind] = self.counts.get(kind, 0) + count_multiplier
        return fn()

    def total_flops(self, prefix: Optional[str] = None) -> int:
        tot = 0
        for k, cnt in self.counts.items():
            if prefix is None or k.startswith(prefix):
                tot += int(cnt) * int(self.per_kind_flops.get(k, 0))
        return tot


def check_causality_gate(device: torch.device) -> Dict[str, float]:
    """Gate G1: verify perturbing future tokens never changes earlier causal latents."""
    torch.manual_seed(0)
    bb = ConceptBeliefBackbone(use_concept_readout=True).to(device).eval()
    x = torch.tensor([[0, 5, 9, 6, 12, Q_VAL], [2, 11, 4, 15, 8, Q_ATTR]], dtype=torch.long, device=device)
    with torch.no_grad():
        o0 = bb(x, use_fast_adapter=False)
        x_f1 = x.clone()
        x_f1[:, 3:] = torch.tensor([16, 17, Q_PROP], device=device)
        o1 = bb(x_f1, use_fast_adapter=False)
        x_f2 = x.clone()
        x_f2[:, 5] = Q_PROP
        o2 = bb(x_f2, use_fast_adapter=False)
    return {
        "z_expr_future_diff": float((o0["z_expr"] - o1["z_expr"]).abs().max().item()),
        "z_ctx_future_diff": float((o0["z_ctx"] - o2["z_ctx"]).abs().max().item()),
    }


@torch.no_grad()
def eval_model(
    predict_fn: Any,
    task: BeliefMeaningTask,
    truth_stage: str,
    device: torch.device,
    eval_seed_base: int,
    n_samples: int = 256,
) -> Dict[str, float]:
    """Evaluate all pre-registered subsets under torch.no_grad() on a separate eval generator."""
    gen = torch.Generator().manual_seed(int(eval_seed_base))

    def _score(
        q_tok: int,
        op_list: List[int],
        pair_split: str,
        concepts: Optional[List[int]],
        ctx_f: Optional[int],
    ) -> Tuple[float, float]:
        x, y = task.sample_eval_subset(q_tok, truth_stage, op_list, pair_split, concepts, ctx_f, n_samples, gen)
        xd, yd = x.to(device), y.to(device)
        lp = predict_fn(xd)
        acc = float((lp.argmax(dim=-1) == yd).float().mean().item())
        nll = float(F.nll_loss(lp, yd).item())
        return acc, nll

    out: Dict[str, float] = {}
    if truth_stage == "p1":
        val_h, _ = _score(Q_VAL, [0, 1, 2, 3], "heldout", None, None)
        attr_s, _ = _score(Q_ATTR, [0, 1, 2], "seen", None, None)
        prop_s, _ = _score(Q_PROP, [0, 1, 2], "seen", None, None)
        out["p1_val_heldout"] = val_h
        out["p1_seen_mean"] = (val_h + attr_s + prop_s) / 3.0
        return out

    if truth_stage == "pb":
        # Note: against true p1 truth!
        gen_pb = torch.Generator().manual_seed(int(eval_seed_base) + 17)
        accs = []
        for qt, ops, split in [(Q_VAL, [0, 1, 2, 3], "seen"), (Q_ATTR, [0, 1, 2], "seen"), (Q_PROP, [0, 1, 2], "seen")]:
            x, y = task.sample_eval_subset(qt, "p1", ops, split, None, None, n_samples, gen_pb)
            lp = predict_fn(x.to(device))
            accs.append(float((lp.argmax(dim=-1) == y.to(device)).float().mean().item()))
        out["pb_retention"] = sum(accs) / len(accs)
        return out

    if truth_stage == "p2":
        split_new, nll_sn = _score(Q_ATTR, [0, 1], "seen", task.s_split, 1)
        split_old, nll_so = _score(Q_ATTR, [0, 1], "seen", task.s_split, 0)
        split_tr_o, _ = _score(Q_ATTR, [2, 3], "seen", task.s_split, None)
        split_tr_h, _ = _score(Q_ATTR, [0, 1, 2, 3], "heldout", task.s_split, None)
        unmod_a, nll_ua = _score(Q_ATTR, [0, 1, 2], "seen", task.s_rev + task.s_static, None)
        ret_v, nll_v = _score(Q_VAL, [0, 1, 2, 3], "seen", None, None)
        ret_p, nll_p = _score(Q_PROP, [0, 1, 2], "seen", None, None)
        out["p2_split_new"] = split_new
        out["p2_split_old"] = split_old
        out["p2_split_both_min"] = min(split_new, split_old)
        out["p2_split_transfer"] = 0.5 * (split_tr_o + split_tr_h)
        out["p2_unmodified"] = (unmod_a + ret_v + ret_p) / 3.0
        out["p2_nll"] = (nll_sn + nll_so + nll_ua + nll_v + nll_p) / 5.0
        return out

    # Phase 3 full evaluation against Phase 3 world truth
    rev_direct, nll_rd = _score(Q_PROP, [0], "seen", task.s_rev, None)
    rev_equiv, nll_re = _score(Q_PROP, [1, 2, 3], "seen", task.s_rev, None)
    rev_held, nll_rh = _score(Q_PROP, [0, 1, 2, 3], "heldout", task.s_rev, None)

    val_all, nll_va = _score(Q_VAL, [0, 1, 2, 3], "seen", None, None)
    val_held, nll_vh = _score(Q_VAL, [0, 1, 2, 3], "heldout", None, None)
    sp_new, nll_sn = _score(Q_ATTR, [0, 1], "seen", task.s_split, 1)
    sp_old, nll_so = _score(Q_ATTR, [0, 1], "seen", task.s_split, 0)
    sp_tr, nll_st = _score(Q_ATTR, [2, 3], "heldout", task.s_split, None)
    attr_rest, nll_ar = _score(Q_ATTR, [0, 1, 2, 3], "seen", task.s_rev + task.s_static, None)
    prop_rest, nll_pr = _score(Q_PROP, [0, 1, 2, 3], "seen", task.s_split + task.s_static, None)

    retention = (val_all + val_held + sp_new + sp_old + attr_rest + prop_rest) / 6.0
    overall_acc = (
        val_all + val_held + sp_new + sp_old + sp_tr + attr_rest + prop_rest + rev_direct + rev_equiv + rev_held
    ) / 10.0
    overall_nll = (
        nll_va + nll_vh + nll_sn + nll_so + nll_st + nll_ar + nll_pr + nll_rd + nll_re + nll_rh
    ) / 10.0

    out["p3_rev_direct"] = rev_direct
    out["p3_rev_equiv_transfer"] = rev_equiv
    out["p3_rev_heldout"] = rev_held
    out["p3_retention_all"] = retention
    out["p3_overall_acc"] = overall_acc
    out["p3_overall_nll"] = overall_nll
    return out


def _backbone_predict(model: ConceptBeliefBackbone, blind: bool = False, use_fast: bool = False):
    def _fn(x: torch.Tensor) -> torch.Tensor:
        xin = torch.zeros_like(x) if blind else x
        out = model(xin, use_fast_adapter=use_fast)
        return F.log_softmax(out["logits"].float(), dim=-1)
    return _fn


def _make_adamw_groups(
    model: ConceptBeliefBackbone,
    wd_trunk: float,
    wd_ctx: float,
    extra_params: Optional[List[nn.Parameter]] = None,
) -> List[Dict[str, Any]]:
    """Apply heavy grokking decay to slow trunk while keeping dormant context/fast heads alive."""
    low_ids = {id(p) for m in (model.ctx_proj, model.fast_adapter) for p in m.parameters()}
    trunk_p = [p for p in model.parameters() if id(p) not in low_ids]
    ctx_p = [p for p in model.parameters() if id(p) in low_ids]
    if extra_params:
        trunk_p.extend(extra_params)
    return [
        {"params": trunk_p, "weight_decay": float(wd_trunk)},
        {"params": ctx_p, "weight_decay": float(wd_ctx)},
    ]


def run_arm_seed(
    arm: str,
    seed: int,
    cfg: LabConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Run one complete 4-phase continual stream (or JointOracle/Null) for one arm and one seed."""
    set_all_seeds(seed * 1009 + 7)
    task = BeliefMeaningTask(seed=seed)
    stream_gen = torch.Generator().manual_seed(seed * 313 + 17)
    eval_base = seed * 977 + 3
    flops = FlopTracker()
    backprop_calls_p1 = 0
    backprop_calls_post = 0
    post_batches = 0
    skipped_post_batches = 0
    aux_bytes = 0
    store_bytes = 0
    pb_burst_flips = 0.0

    # --- Non-neural dictionary baseline ---
    if arm == "B0_SurfaceDict":
        d = BudgetedSurfaceDict(max_total_bytes=cfg.max_store_bytes)
        t_train = 0.0
        t0 = time.perf_counter()
        for step in range(cfg.steps_p1):
            x, y = task.sample_stream_batch("p1", cfg.batch_size, stream_gen)
            d.update(x, y, source_id=f"src_{step % 3}")
        t_train += time.perf_counter() - t0
        p1_res = eval_model(d.predict_log_probs, task, "p1", device, eval_base + 1)

        t0 = time.perf_counter()
        for _ in range(cfg.steps_pb):
            x, y = task.sample_stream_batch("pb", cfg.batch_size, stream_gen)
            d.update(x, y, source_id="noisy_single")
        t_train += time.perf_counter() - t0
        pb_res = eval_model(d.predict_log_probs, task, "pb", device, eval_base + 2)

        t0 = time.perf_counter()
        for step in range(cfg.steps_p2):
            x, y = task.sample_stream_batch("p2", cfg.batch_size, stream_gen, cfg.confirm_frac)
            d.update(x, y, source_id=f"src_{step % 3}")
        t_train += time.perf_counter() - t0
        p2_res = eval_model(d.predict_log_probs, task, "p2", device, eval_base + 3)

        t0 = time.perf_counter()
        for step in range(cfg.steps_p3):
            x, y = task.sample_stream_batch("p3", cfg.batch_size, stream_gen, cfg.confirm_frac)
            d.update(x, y, source_id=f"src_{step % 3}")
        t_train += time.perf_counter() - t0
        p3_res = eval_model(d.predict_log_probs, task, "p3", device, eval_base + 4)

        return {
            **p1_res, **pb_res, **p2_res, **p3_res,
            "pb_burst_flips": 0.0,
            "flops_p1": 0.0,
            "flops_post_p1": 0.0,
            "flops_total": 0.0,
            "backprop_calls_p1": 0.0,
            "backprop_calls_post_p1": 0.0,
            "skip_rate_post_p1": 1.0,
            "wall_clock_sec": float(t_train),
            "param_count": 0.0,
            "store_bytes": float(d.storage_bytes()),
            "aux_memory_bytes": float(d.storage_bytes()),
        }

    # --- Strictly backprop-free MuPC arms (Idea 3a) ---
    if arm in ("Idea3_MuPC_BONG_Pure", "Idea3_MuPC_Adam"):
        net = MuPCProspectiveNetwork(
            d_emb=20, d_pc=120, num_layers=3,
            k_train=cfg.k_train, eta_z=cfg.eta_z, beta_out=cfg.beta_out,
        ).to(device)
        if arm == "Idea3_MuPC_BONG_Pure":
            opt = BONGPrecisionOptimizer(
                net.ordered_params(),
                lr=cfg.lr_mupc_bong,
                init_prec=2e-3,
                gamma=0.996,
                damping=5e-3,
                min_prec=1e-5,
                grad_clip=0.5,
                prior_wd=0.005,
            )
        else:
            opt = torch.optim.AdamW(net.ordered_params(), lr=cfg.lr_mupc_adam, weight_decay=cfg.wd_mupc)

        def _pc_predict(x_in: torch.Tensor) -> torch.Tensor:
            return F.log_softmax(net(x_in)["logits"].float(), dim=-1)

        ema_loss = 2.77
        t_train = 0.0
        _sync_device(device)
        t0 = time.perf_counter()
        for _ in range(cfg.steps_p1):
            x, y = task.sample_stream_batch("p1", cfg.batch_size, stream_gen)
            xd, yd = x.to(device), y.to(device)
            res = flops.record("p1_pc_upd", lambda: net.local_update_step(xd, yd, opt))
            ema_loss = 0.95 * ema_loss + 0.05 * res["ce_loss"]
        _sync_device(device)
        t_train += time.perf_counter() - t0
        p1_res = eval_model(_pc_predict, task, "p1", device, eval_base + 1)

        # Post-P1: surprise-gated equilibration (0 backprop!)
        for phase, n_steps in [("pb", cfg.steps_pb), ("p2", cfg.steps_p2), ("p3", cfg.steps_p3)]:
            _sync_device(device)
            t0 = time.perf_counter()
            for _ in range(n_steps):
                x, y = task.sample_stream_batch(phase, cfg.batch_size, stream_gen, cfg.confirm_frac)
                xd, yd = x.to(device), y.to(device)
                post_batches += 1
                res = flops.record(
                    "post_pc_try",
                    lambda: net.local_update_step(xd, yd, opt, skip_threshold=cfg.tau_pc_skip, ema_ref_loss=ema_loss),
                )
                if res["skipped"] > 0.5:
                    skipped_post_batches += 1
                elif phase != "pb":
                    ema_loss = 0.97 * ema_loss + 0.03 * res["ce_loss"]
            _sync_device(device)
            t_train += time.perf_counter() - t0
            if phase == "pb":
                pb_res = eval_model(_pc_predict, task, "pb", device, eval_base + 2)
            elif phase == "p2":
                p2_res = eval_model(_pc_predict, task, "p2", device, eval_base + 3)

        p3_res = eval_model(_pc_predict, task, "p3", device, eval_base + 4)
        aux_b = opt.precision_bytes() if isinstance(opt, BONGPrecisionOptimizer) else sum(p.numel() * 8 for p in net.parameters())
        return {
            **p1_res, **pb_res, **p2_res, **p3_res,
            "pb_burst_flips": 0.0,
            "flops_p1": float(flops.total_flops("p1_")),
            "flops_post_p1": float(flops.total_flops("post_")),
            "flops_total": float(flops.total_flops()),
            "backprop_calls_p1": 0.0,
            "backprop_calls_post_p1": 0.0,
            "skip_rate_post_p1": float(skipped_post_batches / max(1, post_batches)),
            "wall_clock_sec": float(t_train),
            "param_count": float(sum(p.numel() for p in net.parameters())),
            "store_bytes": 0.0,
            "aux_memory_bytes": float(aux_b),
        }

    # --- Shared RoPE Transformer Backbone arms ---
    model = ConceptBeliefBackbone(use_concept_readout=True).to(device)

    if arm == "Null_Untrained":
        p1_res = eval_model(_backbone_predict(model), task, "p1", device, eval_base + 1)
        pb_res = eval_model(_backbone_predict(model), task, "pb", device, eval_base + 2)
        p2_res = eval_model(_backbone_predict(model), task, "p2", device, eval_base + 3)
        p3_res = eval_model(_backbone_predict(model), task, "p3", device, eval_base + 4)
        return {
            **p1_res, **pb_res, **p2_res, **p3_res,
            "pb_burst_flips": 0.0, "flops_p1": 0.0, "flops_post_p1": 0.0, "flops_total": 0.0,
            "backprop_calls_p1": 0.0, "backprop_calls_post_p1": 0.0, "skip_rate_post_p1": 1.0,
            "wall_clock_sec": 0.0, "param_count": float(sum(p.numel() for p in model.parameters())),
            "store_bytes": 0.0, "aux_memory_bytes": 0.0,
        }

    if arm == "B4_JointOracle":
        opt = torch.optim.AdamW(_make_adamw_groups(model, cfg.wd_p1, cfg.wd_post), lr=cfg.lr_dense)
        total_steps = cfg.steps_p1 + cfg.steps_p2 + cfg.steps_p3
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=5e-4)
        _sync_device(device)
        t0 = time.perf_counter()
        for step in range(total_steps):
            # Upweight S_split ATTR consistently alongside p3 truth so JointOracle sees balanced split + rev
            if step % 3 == 0:
                x, y = task.sample_stream_batch("p2", cfg.batch_size, stream_gen, confirm_frac=0.35)
                m_c = task.val_table[x[:, 0], x[:, 1] - 4, x[:, 2] - 4]
                rev_hit = (x[:, 5] == Q_PROP) & task.s_rev_mask[m_c]
                if rev_hit.any():
                    y[rev_hit] = task.h_rev[m_c[rev_hit]]
            else:
                x, y = task.sample_stream_batch("joint_oracle", cfg.batch_size, stream_gen)
            xd, yd = x.to(device), y.to(device)

            def _step():
                opt.zero_grad()
                out = model(xd)
                loss = F.cross_entropy(out["logits"], yd)
                loss.backward()
                opt.step()

            flops.record("oracle_step", _step)
            sched.step()
            backprop_calls_p1 += 1
        _sync_device(device)
        wall = time.perf_counter() - t0
        p1_res = eval_model(_backbone_predict(model), task, "p1", device, eval_base + 1)
        pb_res = eval_model(_backbone_predict(model), task, "pb", device, eval_base + 2)
        p2_res = eval_model(_backbone_predict(model), task, "p2", device, eval_base + 3)
        p3_res = eval_model(_backbone_predict(model), task, "p3", device, eval_base + 4)
        return {
            **p1_res, **pb_res, **p2_res, **p3_res,
            "pb_burst_flips": 0.0,
            "flops_p1": float(flops.total_flops()),
            "flops_post_p1": 0.0,
            "flops_total": float(flops.total_flops()),
            "backprop_calls_p1": float(backprop_calls_p1),
            "backprop_calls_post_p1": 0.0,
            "skip_rate_post_p1": 0.0,
            "wall_clock_sec": float(wall),
            "param_count": float(sum(p.numel() for p in model.parameters())),
            "store_bytes": 0.0,
            "aux_memory_bytes": 0.0,
        }

    # --- Idea 1 / Triad Wake-Chunk-Sleep arms (plus Null_ShuffledTargets & Null_BlindZeroInput) ---
    is_wcs = arm in ("Idea1_ConceptWCS", "Idea1_WakeOnly", "Idea1_NoCtxSplit", "Idea_Triad_All3", "Null_ShuffledTargets", "Null_BlindZeroInput")
    if is_wcs:
        use_equiv = arm in ("Idea_Triad_All3", "Null_ShuffledTargets", "Null_BlindZeroInput")
        use_sleep = arm != "Idea1_WakeOnly"
        enable_split = arm != "Idea1_NoCtxSplit"
        use_bong_chunk = arm in ("Idea_Triad_All3", "Null_ShuffledTargets", "Null_BlindZeroInput")
        shuffle_all_targets = arm == "Null_ShuffledTargets"
        blind_eval = arm == "Null_BlindZeroInput"

        quantizer = SparseConceptQuantizer(d_model=model.d_model, num_atoms=cfg.num_atoms).to(device)
        store = ConceptBeliefStore(
            d_ctx=model.d_ctx,
            max_total_bytes=cfg.max_store_bytes,
            enable_ctx_split=enable_split,
            enable_mech_p=True,
        )
        ns = NullSpaceGuard(model, energy_thresh=0.96, relax=0.02)

        slow_params = list(model.parameters()) + list(quantizer.parameters())
        opt_slow = torch.optim.AdamW(
            _make_adamw_groups(model, cfg.wd_p1, cfg.wd_post, extra_params=list(quantizer.parameters())),
            lr=cfg.lr_dense,
        )
        opt_q = None

        fast_mods = [model.fast_adapter, model.ctx_proj, model.attr_head, model.prop_head]
        fast_params = [p for m in fast_mods for p in m.parameters()]
        if use_bong_chunk:
            opt_fast: Any = BONGPrecisionOptimizer(
                fast_params, lr=cfg.lr_bong_chunk, init_prec=0.02, gamma=0.985, damping=0.01, prior_wd=cfg.wd_post
            )
        else:
            opt_fast = torch.optim.AdamW(fast_params, lr=cfg.lr_chunk, weight_decay=cfg.wd_post)

        # Anchors & FFN utilities for Elastic TTT + Continual Backprop
        anchor_fast: List[torch.Tensor] = []
        ffn_utils = [torch.ones(blk.ffn[0].out_features, device=device) * 0.2 for blk in model.blocks]

        # Bounded buffers (explicitly byte-counted!)
        chunk_x_buf: List[torch.Tensor] = []
        chunk_y_buf: List[torch.Tensor] = []
        chunk_items = 0
        max_mem_items = 256
        mem_x: List[torch.Tensor] = []
        mem_y: List[torch.Tensor] = []
        mem_seen = 0
        recent_buf: deque = deque(maxlen=32)
        ema_loss = 2.77
        chunks_since_sleep = 0

        def _wcs_predict(x_in: torch.Tensor) -> torch.Tensor:
            xin = torch.zeros_like(x_in) if blind_eval else x_in
            out = model(xin, use_fast_adapter=True)
            cids, _ = quantizer.assign(out["z_expr"])
            return store.fuse_logits(cids, xin[:, 5], out["z_ctx"], out["logits"])

        t_train = 0.0
        _sync_device(device)
        t0 = time.perf_counter()

        # Phase 1: self-trained concept + backbone grounding
        for step in range(cfg.steps_p1):
            x, y = task.sample_stream_batch("p1", cfg.batch_size, stream_gen)
            if shuffle_all_targets:
                y = torch.randint(0, K_CONCEPTS, (y.size(0),), generator=stream_gen)
            xd, yd = x.to(device), y.to(device)

            def _p1_step():
                opt_slow.zero_grad()
                if opt_q is not None:
                    opt_q.zero_grad()
                out = model(xd, use_fast_adapter=False)
                ce = F.cross_entropy(out["logits"], yd)
                qloss, _ = quantizer.quantize_loss(out["z_expr"])
                sl = sigreg_loss(out["z_expr"])
                loss = ce + 0.12 * qloss + cfg.lambda_sigreg * sl
                if use_equiv:
                    eq_w = cfg.lambda_equiv * min(1.0, max(0.25, float(step + 1) / 1600.0))
                    le, lc = equivalence_and_contradiction_losses(out["z_expr"], out["z_ctx"], xd, yd, shuffle_equiv=False)
                    loss = loss + eq_w * le + cfg.lambda_contra * lc
                loss.backward()
                opt_slow.step()
                if opt_q is not None:
                    opt_q.step()
                return out, float(ce.item())

            out_p1, ce_v = flops.record("p1_wcs_step", _p1_step)
            backprop_calls_p1 += 1
            ema_loss = 0.96 * ema_loss + 0.04 * ce_v
            if (step + 1) % 40 == 0:
                quantizer.revive_dead_atoms(out_p1["z_expr"])
            if step >= int(0.60 * cfg.steps_p1) and (step % 4 == 0):
                for j in range(min(4, x.size(0))):
                    mem_seen += 1
                    if len(mem_x) < max_mem_items:
                        mem_x.append(x[j])
                        mem_y.append(y[j])
                    else:
                        idx_r = int(torch.randint(0, mem_seen, (1,), generator=stream_gen).item())
                        if idx_r < max_mem_items:
                            mem_x[idx_r] = x[j]
                            mem_y[idx_r] = y[j]
            if step >= int(0.88 * cfg.steps_p1):
                with torch.no_grad():
                    cids, _ = quantizer.assign(out_p1["z_expr"])
                    store.observe_batch(cids, xd[:, 5], out_p1["z_ctx"], yd, out_p1["logits"], source_id=f"src_{step % 3}")

        with torch.no_grad():
            anchor_fast = [p.detach().clone() for p in fast_params]
            ns.consolidate()
            ns.tracking = False
        for pg in opt_slow.param_groups:
            pg["lr"] = min(cfg.lr_dense_post, 8e-4)
            pg["weight_decay"] = cfg.wd_post
        _sync_device(device)
        t_train += time.perf_counter() - t0
        p1_res = eval_model(_wcs_predict, task, "p1", device, eval_base + 1)

        # Post-P1 Wake-Chunk-Sleep Stream (Phases B -> 2 -> 3, ZERO phase boundaries!)
        post_step_global = 0
        for phase, n_steps in [("pb", cfg.steps_pb), ("p2", cfg.steps_p2), ("p3", cfg.steps_p3)]:
            flips_before = store.flips_executed
            _sync_device(device)
            t0 = time.perf_counter()
            for step in range(n_steps):
                post_step_global += 1
                post_batches += 1
                x, y = task.sample_stream_batch(phase, cfg.batch_size, stream_gen, cfg.confirm_frac)
                if shuffle_all_targets:
                    y = torch.randint(0, K_CONCEPTS, (y.size(0),), generator=stream_gen)
                src = "noisy_single" if phase == "pb" else f"src_{step % 3}"
                xd, yd = x.to(device), y.to(device)

                # Wake: forward ONLY + concept store triage (0 backprop!)
                def _wake_fwd():
                    with torch.no_grad():
                        out = model(xd, use_fast_adapter=True)
                        cids, _ = quantizer.assign(out["z_expr"])
                        return out, cids

                out_w, cids = flops.record("post_wake_fwd", _wake_fwd)
                obs = store.observe_batch(cids, xd[:, 5], out_w["z_ctx"], yd, out_w["logits"], source_id=src)
                if obs["flips_here"] > 0:
                    # Prune any superseded reservoir entries that contradict settled pure store flips
                    with torch.no_grad():
                        if mem_x:
                            mx_chk = torch.stack(mem_x, dim=0).to(device)
                            my_chk = torch.stack(mem_y, dim=0)
                            mcids = flops.record(
                                "post_prune_chk",
                                lambda: quantizer.assign(model.encode_trunk(mx_chk)["z_expr"])[0],
                            ).cpu().tolist()
                            kept_x, kept_y = [], []
                            for mx_i, my_i, cid_i in zip(mem_x, my_chk.tolist(), mcids):
                                k_i = (int(cid_i), int(mx_i[5].item()))
                                ent_i = store.primary.get(k_i)
                                if ent_i is not None and ent_i["count"] >= 2 and float(ent_i.get("purity", 1.0)) >= 0.72 and int(ent_i["val"]) != int(my_i):
                                    continue
                                kept_x.append(mx_i)
                                kept_y.append(torch.tensor(my_i, dtype=torch.long))
                            mem_x, mem_y = kept_x, kept_y
                    if isinstance(opt_fast, BONGPrecisionOptimizer):
                        for pr in opt_fast.precisions:
                            pr.mul_(0.25)

                with torch.no_grad():
                    ce_vec = F.cross_entropy(out_w["logits"], yd, reduction="none")
                    reducible = (ce_vec - ema_loss) > cfg.rho_thresh
                    pick_mask = obs["needs_chunk_mask"] & (reducible | (ce_vec > 0.50))
                    # Record confirmed low-error non-burst samples into balanced reservoir
                    if src != "noisy_single":
                        ok_mask = (~obs["needs_chunk_mask"]) & (ce_vec < 0.5)
                        if int(ok_mask.sum().item()) > 0:
                            ok_cpu = ok_mask.cpu()
                            for j in range(min(2, int(ok_cpu.sum().item()))):
                                mem_seen += 1
                                idx_r = mem_seen - 1 if len(mem_x) < max_mem_items else int(torch.randint(0, mem_seen, (1,), generator=stream_gen).item())
                                if idx_r < max_mem_items:
                                    if idx_r < len(mem_x):
                                        mem_x[idx_r] = x[ok_cpu][j]
                                        mem_y[idx_r] = y[ok_cpu][j]
                                    else:
                                        mem_x.append(x[ok_cpu][j])
                                        mem_y.append(y[ok_cpu][j])
                        ema_loss = 0.98 * ema_loss + 0.02 * float(ce_vec.mean().item())

                n_pick = int(pick_mask.sum().item())
                if n_pick < 4:
                    skipped_post_batches += 1
                else:
                    pm_cpu = pick_mask.cpu()
                    px = x[pm_cpu]
                    py = y[pm_cpu]
                    # Pair surprising Q_ATTR concepts with matching-atom confirmed Q_ATTR items from the same batch
                    if src != "noisy_single":
                        pm_attr = pick_mask & (xd[:, 5] == Q_ATTR)
                        if bool(pm_attr.any().item()):
                            pm_cids = cids[pm_attr].unique()
                            ok_pair = (~pick_mask) & (xd[:, 5] == Q_ATTR) & torch.isin(cids, pm_cids)
                            if not bool(ok_pair.any().item()):
                                ok_pair = (~pick_mask) & (xd[:, 5] == Q_ATTR)
                            if int(ok_pair.sum().item()) > 0:
                                ok_p_cpu = ok_pair.cpu()
                                n_pair = min(px.size(0), int(ok_p_cpu.sum().item()))
                                px = torch.cat([px, x[ok_p_cpu][:n_pair]], dim=0)
                                py = torch.cat([py, y[ok_p_cpu][:n_pair]], dim=0)
                    chunk_x_buf.append(px)
                    chunk_y_buf.append(py)
                    chunk_items += int(px.size(0))
                    recent_buf.append((px[:6], py[:6]))

                # Tier 2: Cached-Trunk Chunked Elastic Fast Update (LaCT + Elastic TTT)
                if chunk_items >= cfg.chunk_size:
                    cx = torch.cat(chunk_x_buf, dim=0)[: cfg.chunk_size].to(device)
                    cy = torch.cat(chunk_y_buf, dim=0)[: cfg.chunk_size].to(device)
                    chunk_x_buf.clear()
                    chunk_y_buf.clear()
                    chunk_items = 0
                    if mem_x:
                        mxa_cpu = torch.stack(mem_x, dim=0)
                        mya_cpu = torch.stack(mem_y, dim=0)
                        ri = torch.randint(0, mxa_cpu.size(0), (min(32, mxa_cpu.size(0)),))
                        cx = torch.cat([cx[:96], mxa_cpu[ri].to(device)], dim=0)
                        cy = torch.cat([cy[:96], mya_cpu[ri].to(device)], dim=0)

                    def _chunk_step():
                        with torch.no_grad():
                            trunk = model.encode_trunk(cx)
                            ze_base = trunk["z_expr"].detach()
                            ctx_f = trunk["ctx_feat"].detach()
                        qt = cx[:, 5]
                        upd_mask = (qt == Q_ATTR) | (qt == Q_PROP)
                        if not upd_mask.any():
                            return
                        for _ in range(cfg.chunk_steps):
                            if not isinstance(opt_fast, BONGPrecisionOptimizer):
                                opt_fast.zero_grad()
                            out_c = model.forward_fast_heads(qt, ze_base, ctx_f, use_fast_adapter=True)
                            loss_c = F.cross_entropy(out_c["logits"][upd_mask], cy[upd_mask])
                            _, lc = equivalence_and_contradiction_losses(
                                ze_base[upd_mask], out_c["z_ctx"][upd_mask], cx[upd_mask], cy[upd_mask], shuffle_equiv=False
                            )
                            loss_c = loss_c + cfg.lambda_contra * lc
                            el = sum((p - a).pow(2).sum() for p, a in zip(fast_params, anchor_fast))
                            loss_c = loss_c + cfg.elastic_pull * el
                            loss_c.backward()
                            torch.nn.utils.clip_grad_norm_(fast_params, 1.0)
                            opt_fast.step()
                            if isinstance(opt_fast, BONGPrecisionOptimizer):
                                opt_fast.zero_grad()

                    flops.record("post_chunk_step", _chunk_step)
                    backprop_calls_post += cfg.chunk_steps
                    chunks_since_sleep += 1
                    if store.rev_event_open:
                        flops.record(
                            "post_mech_p",
                            lambda: store.run_mechanism_p_sweep(
                                quantizer.scaled_atoms().to(device), model, device, tau_derive=0.75, use_fast_adapter=True
                            ),
                        )

                # Tier 3: Periodic Sleep Consolidation with Balanced Reservoir Replay
                if use_sleep and (post_step_global % cfg.sleep_period == 0) and chunks_since_sleep > 0 and len(mem_x) > 0:
                    chunks_since_sleep = 0
                    mx_all = torch.stack(mem_x, dim=0).to(device)
                    my_all = torch.stack(mem_y, dim=0).to(device)
                    if len(recent_buf) > 0:
                        rx_all = torch.cat([item[0] for item in recent_buf], dim=0).to(device)
                        ry_all = torch.cat([item[1] for item in recent_buf], dim=0).to(device)
                    else:
                        rx_all, ry_all = mx_all, my_all

                    def _sleep_step():
                        ns.tracking = True
                        for _ in range(cfg.sleep_steps):
                            idx_m = torch.randint(0, mx_all.size(0), (min(42, mx_all.size(0)),), device=device)
                            idx_r = torch.randint(0, rx_all.size(0), (min(22, rx_all.size(0)),), device=device)
                            bx = torch.cat([mx_all[idx_m], rx_all[idx_r]], dim=0)
                            by = torch.cat([my_all[idx_m], ry_all[idx_r]], dim=0)
                            opt_slow.zero_grad()
                            out_s = model(bx, use_fast_adapter=True)
                            ql_s, _ = quantizer.quantize_loss(out_s["z_expr"])
                            loss_s = F.cross_entropy(out_s["logits"], by) + 0.15 * ql_s + cfg.lambda_sigreg * sigreg_loss(out_s["z_expr"])
                            if use_equiv:
                                le, lc = equivalence_and_contradiction_losses(
                                    out_s["z_expr"], out_s["z_ctx"], bx, by, shuffle_equiv=False
                                )
                                loss_s = loss_s + cfg.lambda_equiv * le + cfg.lambda_contra * lc
                            loss_s.backward()
                            ns.project_grads()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            opt_slow.step()
                        ns.tracking = False

                    flops.record("post_sleep_step", _sleep_step)
                    backprop_calls_post += cfg.sleep_steps
                    with torch.no_grad():
                        for a_p, f_p in zip(anchor_fast, fast_params):
                            a_p.mul_(0.85).add_(0.15 * f_p.detach())
                        for blk_idx, blk in enumerate(model.blocks):
                            dummy_act = F.gelu(blk.ffn[0](blk.norm2(model(mx_all[:16])["hidden"])))
                            continual_backprop_recycle(
                                blk.ffn[0],
                                blk.ffn[2],
                                ffn_utils[blk_idx],
                                dummy_act,
                                replace_rate=0.02,
                                maturity_threshold=0.005,
                                bong_opt=None,
                            )
                        ns.consolidate()
                        if store.rev_event_open:
                            store.run_mechanism_p_sweep(
                                quantizer.scaled_atoms().to(device), model, device, tau_derive=0.75, use_fast_adapter=True
                            )

            _sync_device(device)
            t_train += time.perf_counter() - t0
            if phase == "pb":
                pb_burst_flips = float(store.flips_executed - flips_before)
                pb_res = eval_model(_wcs_predict, task, "pb", device, eval_base + 2)
            elif phase == "p2":
                p2_res = eval_model(_wcs_predict, task, "p2", device, eval_base + 3)

        p3_res = eval_model(_wcs_predict, task, "p3", device, eval_base + 4)
        ns.remove()

        store_bytes = store.storage_bytes()
        aux_bytes = (
            store_bytes
            + ((len(mem_x) + sum(len(item[0]) for item in recent_buf)) * 28)
            + (chunk_items * 28)
            + sum(p.numel() * 4 for p in anchor_fast)
            + sum(c.numel() * 4 for c in ns.covs.values())
            + (opt_fast.precision_bytes() if isinstance(opt_fast, BONGPrecisionOptimizer) else 0)
        )
        return {
            **p1_res, **pb_res, **p2_res, **p3_res,
            "pb_burst_flips": pb_burst_flips,
            "flops_p1": float(flops.total_flops("p1_")),
            "flops_post_p1": float(flops.total_flops("post_")),
            "flops_total": float(flops.total_flops()),
            "backprop_calls_p1": float(backprop_calls_p1),
            "backprop_calls_post_p1": float(backprop_calls_post),
            "skip_rate_post_p1": float(skipped_post_batches / max(1, post_batches)),
            "wall_clock_sec": float(t_train),
            "param_count": float(sum(p.numel() for p in slow_params)),
            "store_bytes": float(store_bytes),
            "aux_memory_bytes": float(aux_bytes),
            "splits_executed": float(store.splits_executed),
            "p_demotions": float(store.p_demotions),
        }

    # --- Standard Backbone Baseline & Idea 2 / Idea 3b arms ---
    is_idea2 = arm in ("Idea2_EquivContra", "Idea2_ShuffledEquiv", "Idea2_NoSIGReg")
    is_bong_bb = arm == "Idea3_Backbone_BONG"
    is_replay = arm == "B2_Dense_Replay"
    is_nullspace = arm == "B3_Dense_NullSpace"

    ns_guard = NullSpaceGuard(model, energy_thresh=0.96, relax=0.10) if is_nullspace else None
    if is_bong_bb:
        low_ids = {id(p) for m in (model.ctx_proj, model.fast_adapter) for p in m.parameters()}
        opt_bb: Any = BONGPrecisionOptimizer(
            list(model.parameters()),
            lr=cfg.lr_bong_bb,
            init_prec=0.02,
            gamma=0.996,
            damping=0.01,
            prior_wd=cfg.wd_bong_p1,
            low_wd_param_ids=low_ids,
        )
        ffn_utils_bb = [torch.ones(blk.ffn[0].out_features, device=device) * 0.2 for blk in model.blocks]
    else:
        lr_use = cfg.lr_idea2 if is_idea2 else cfg.lr_dense
        opt_bb = torch.optim.AdamW(_make_adamw_groups(model, cfg.wd_p1, cfg.wd_post), lr=lr_use)

    max_rep = cfg.max_store_bytes // 28  # 585 items <= 16384B
    rep_x: List[torch.Tensor] = []
    rep_y: List[torch.Tensor] = []
    rep_seen = 0
    ema_loss_bb = 2.77

    _sync_device(device)
    t0 = time.perf_counter()

    def _compute_bb_loss(xb: torch.Tensor, yb: torch.Tensor, eq_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out = model(xb, use_fast_adapter=False)
        loss = F.cross_entropy(out["logits"], yb)
        if is_idea2:
            le, lc = equivalence_and_contradiction_losses(
                out["z_expr"], out["z_ctx"], xb, yb, shuffle_equiv=(arm == "Idea2_ShuffledEquiv")
            )
            loss = loss + (cfg.lambda_equiv * eq_scale) * le + cfg.lambda_contra * lc
            if arm != "Idea2_NoSIGReg":
                loss = loss + cfg.lambda_sigreg * sigreg_loss(out["z_expr"])
        return loss, out["hidden"]

    # Phase 1
    t_train = 0.0
    _sync_device(device)
    t0 = time.perf_counter()
    for step in range(cfg.steps_p1):
        x, y = task.sample_stream_batch("p1", cfg.batch_size, stream_gen)
        xd, yd = x.to(device), y.to(device)
        eq_s = min(1.0, max(0.25, float(step + 1) / 1600.0))

        def _p1_bb():
            opt_bb.zero_grad()
            loss, _ = _compute_bb_loss(xd, yd, eq_scale=eq_s)
            loss.backward()
            opt_bb.step()
            return float(loss.item())

        lv = flops.record("p1_bb_step", _p1_bb)
        backprop_calls_p1 += 1
        ema_loss_bb = 0.96 * ema_loss_bb + 0.04 * lv
        if is_replay and step % 2 == 0:
            for j in range(min(4, x.size(0))):
                rep_seen += 1
                if len(rep_x) < max_rep:
                    rep_x.append(x[j])
                    rep_y.append(y[j])
                else:
                    idx_r = int(torch.randint(0, rep_seen, (1,), generator=stream_gen).item())
                    if idx_r < max_rep:
                        rep_x[idx_r] = x[j]
                        rep_y[idx_r] = y[j]

    if ns_guard is not None:
        ns_guard.consolidate()
    _sync_device(device)
    t_train += time.perf_counter() - t0
    p1_res = eval_model(_backbone_predict(model), task, "p1", device, eval_base + 1)

    # Phases B -> 2 -> 3
    if not is_bong_bb:
        for pg in opt_bb.param_groups:
            pg["lr"] = cfg.lr_dense_post if not is_idea2 else cfg.lr_idea2 * 0.5
            pg["weight_decay"] = cfg.wd_post
    else:
        opt_bb.prior_wd = cfg.wd_post
    post_global = 0
    for phase, n_steps in [("pb", cfg.steps_pb), ("p2", cfg.steps_p2), ("p3", cfg.steps_p3)]:
        _sync_device(device)
        t0 = time.perf_counter()
        for step in range(n_steps):
            post_global += 1
            post_batches += 1
            x, y = task.sample_stream_batch(phase, cfg.batch_size, stream_gen, cfg.confirm_frac)
            xd, yd = x.to(device), y.to(device)

            if is_bong_bb:
                # Surprise check before backward
                def _bong_fwd():
                    with torch.no_grad():
                        return float(F.cross_entropy(model(xd)["logits"], yd).item())

                ce_now = flops.record("post_bong_fwd", _bong_fwd)
                if (ce_now - ema_loss_bb) < 0.05 and ce_now < 0.35:
                    skipped_post_batches += 1
                    continue

                def _bong_bwd():
                    opt_bb.zero_grad()
                    loss, hid = _compute_bb_loss(xd, yd)
                    loss.backward()
                    opt_bb.step()
                    return hid

                hid_out = flops.record("post_bong_bwd", _bong_bwd)
                backprop_calls_post += 1
                if phase != "pb":
                    ema_loss_bb = 0.97 * ema_loss_bb + 0.03 * ce_now
                if post_global % 20 == 0:
                    with torch.no_grad():
                        for b_idx, blk in enumerate(model.blocks):
                            act_v = F.gelu(blk.ffn[0](blk.norm2(hid_out)))
                            continual_backprop_recycle(
                                blk.ffn[0],
                                blk.ffn[2],
                                ffn_utils_bb[b_idx],
                                act_v,
                                replace_rate=0.02,
                                maturity_threshold=0.005,
                                bong_opt=opt_bb,
                            )
            else:
                if is_replay and len(rep_x) >= 16:
                    r_pick = torch.randint(0, len(rep_x), (16,), generator=stream_gen)
                    rx = torch.stack([rep_x[int(k)] for k in r_pick], dim=0).to(device)
                    ry = torch.stack([rep_y[int(k)] for k in r_pick], dim=0).to(device)
                    xb = torch.cat([xd, rx], dim=0)
                    yb = torch.cat([yd, ry], dim=0)
                else:
                    xb, yb = xd, yd

                def _bb_post():
                    opt_bb.zero_grad()
                    loss, _ = _compute_bb_loss(xb, yb)
                    loss.backward()
                    if ns_guard is not None:
                        ns_guard.project_grads()
                    opt_bb.step()

                kind = "post_rep_step" if (is_replay and len(rep_x) >= 16) else "post_bb_step"
                flops.record(kind, _bb_post)
                backprop_calls_post += 1

                if is_replay and step % 4 == 0:
                    for j in range(min(2, x.size(0))):
                        rep_seen += 1
                        if len(rep_x) < max_rep:
                            rep_x.append(x[j])
                            rep_y.append(y[j])
                        else:
                            idx_r = int(torch.randint(0, rep_seen, (1,), generator=stream_gen).item())
                            if idx_r < max_rep:
                                rep_x[idx_r] = x[j]
                                rep_y[idx_r] = y[j]

                if ns_guard is not None and post_global % 25 == 0:
                    ns_guard.consolidate()

        _sync_device(device)
        t_train += time.perf_counter() - t0
        if phase == "pb":
            pb_res = eval_model(_backbone_predict(model), task, "pb", device, eval_base + 2)
        elif phase == "p2":
            p2_res = eval_model(_backbone_predict(model), task, "p2", device, eval_base + 3)

    p3_res = eval_model(_backbone_predict(model), task, "p3", device, eval_base + 4)

    if ns_guard is not None:
        aux_bytes = sum(c.numel() * 4 for c in ns_guard.covs.values()) + sum(p.numel() * 4 for p in ns_guard.p_null.values())
        ns_guard.remove()
    elif is_bong_bb:
        aux_bytes = opt_bb.precision_bytes()
    elif is_replay:
        aux_bytes = len(rep_x) * 28

    return {
        **p1_res, **pb_res, **p2_res, **p3_res,
        "pb_burst_flips": 0.0,
        "flops_p1": float(flops.total_flops("p1_")),
        "flops_post_p1": float(flops.total_flops("post_")),
        "flops_total": float(flops.total_flops()),
        "backprop_calls_p1": float(backprop_calls_p1),
        "backprop_calls_post_p1": float(backprop_calls_post),
        "skip_rate_post_p1": float(skipped_post_batches / max(1, post_batches)),
        "wall_clock_sec": float(t_train),
        "param_count": float(sum(p.numel() for p in model.parameters())),
        "store_bytes": float(len(rep_x) * 28) if is_replay else 0.0,
        "aux_memory_bytes": float(aux_bytes),
    }


def bootstrap_ci_diff(a: List[float], b: List[float], seed: int = 1234, n_boot: int = 2000) -> Dict[str, float]:
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    mean_d = sum(diffs) / max(1, n)
    gen = torch.Generator().manual_seed(seed)
    dt = torch.tensor(diffs, dtype=torch.float32)
    idx = torch.randint(0, n, (n_boot, n), generator=gen)
    boot_means = dt[idx].mean(dim=1).sort().values
    lo = float(boot_means[int(0.025 * n_boot)].item())
    hi = float(boot_means[int(0.975 * n_boot)].item())
    return {"mean_diff": mean_d, "ci95_low": lo, "ci95_high": hi}


def summarize_list(vals: List[float]) -> Dict[str, Any]:
    n = len(vals)
    mean = sum(vals) / max(1, n)
    var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    ci_half = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": sd,
        "ci95_half": ci_half,
        "min": min(vals) if n else 0.0,
        "max": max(vals) if n else 0.0,
        "n": n,
        "per_seed": vals,
    }


def aggregate_arm(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    keys = sorted(rows[0].keys())
    return {k: summarize_list([r[k] for r in rows]) for k in keys}


def evaluate_falsifiers(agg: Dict[str, Any]) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    """Compute pre-registered boolean falsifiers F1..F8 and paired CIs strictly from measured outputs."""
    paired = {
        "f1_idea1_vs_no_split_p2_min": bootstrap_ci_diff(
            agg["Idea1_ConceptWCS"]["p2_split_both_min"]["per_seed"],
            agg["Idea1_NoCtxSplit"]["p2_split_both_min"]["per_seed"],
        ),
        "f1_idea1_vs_dense_p2_min": bootstrap_ci_diff(
            agg["Idea1_ConceptWCS"]["p2_split_both_min"]["per_seed"],
            agg["B1_Dense_SGD"]["p2_split_both_min"]["per_seed"],
        ),
        "f3_idea2_vs_dense_equiv": bootstrap_ci_diff(
            agg["Idea2_EquivContra"]["p3_rev_equiv_transfer"]["per_seed"],
            agg["B1_Dense_SGD"]["p3_rev_equiv_transfer"]["per_seed"],
        ),
        "f3_idea2_vs_shuffled_equiv": bootstrap_ci_diff(
            agg["Idea2_EquivContra"]["p3_rev_equiv_transfer"]["per_seed"],
            agg["Idea2_ShuffledEquiv"]["p3_rev_equiv_transfer"]["per_seed"],
        ),
        "f4_mupc_bong_vs_adam_overall": bootstrap_ci_diff(
            agg["Idea3_MuPC_BONG_Pure"]["p3_overall_acc"]["per_seed"],
            agg["Idea3_MuPC_Adam"]["p3_overall_acc"]["per_seed"],
        ),
        "f5_bb_bong_vs_dense_retention": bootstrap_ci_diff(
            agg["Idea3_Backbone_BONG"]["p3_retention_all"]["per_seed"],
            agg["B1_Dense_SGD"]["p3_retention_all"]["per_seed"],
        ),
        "f5_bb_bong_vs_dense_pb": bootstrap_ci_diff(
            agg["Idea3_Backbone_BONG"]["pb_retention"]["per_seed"],
            agg["B1_Dense_SGD"]["pb_retention"]["per_seed"],
        ),
        "f6_triad_vs_replay_overall": bootstrap_ci_diff(
            agg["Idea_Triad_All3"]["p3_overall_acc"]["per_seed"],
            agg["B2_Dense_Replay"]["p3_overall_acc"]["per_seed"],
        ),
        "f6_triad_vs_dense_overall": bootstrap_ci_diff(
            agg["Idea_Triad_All3"]["p3_overall_acc"]["per_seed"],
            agg["B1_Dense_SGD"]["p3_overall_acc"]["per_seed"],
        ),
    }

    best_single = max(
        agg["B0_SurfaceDict"]["p3_overall_acc"]["mean"],
        agg["B1_Dense_SGD"]["p3_overall_acc"]["mean"],
        agg["B2_Dense_Replay"]["p3_overall_acc"]["mean"],
    )
    nulls = ("Null_Untrained", "Null_ShuffledTargets", "Null_BlindZeroInput")

    falsifiers = {
        "F1_idea1_context_split": (
            agg["Idea1_ConceptWCS"]["p2_split_both_min"]["mean"] >= 0.75
            and paired["f1_idea1_vs_no_split_p2_min"]["mean_diff"] >= 0.15
            and paired["f1_idea1_vs_no_split_p2_min"]["ci95_low"] > 0.0
            and paired["f1_idea1_vs_dense_p2_min"]["mean_diff"] >= 0.20
            and paired["f1_idea1_vs_dense_p2_min"]["ci95_low"] > 0.0
        ),
        "F2_idea1_burst_and_compute": (
            agg["Idea1_ConceptWCS"]["pb_retention"]["mean"] >= 0.85
            and agg["Idea1_ConceptWCS"]["pb_burst_flips"]["mean"] <= 0.1
            and agg["Idea1_ConceptWCS"]["flops_post_p1"]["mean"] <= 0.60 * agg["B2_Dense_Replay"]["flops_post_p1"]["mean"]
            and agg["Idea1_ConceptWCS"]["p3_overall_acc"]["mean"] >= agg["B2_Dense_Replay"]["p3_overall_acc"]["mean"] - 0.03
        ),
        "F3_idea2_equiv_transfer_and_x47_control": (
            agg["Idea2_EquivContra"]["p3_rev_equiv_transfer"]["mean"] >= 0.70
            and paired["f3_idea2_vs_dense_equiv"]["mean_diff"] >= 0.15
            and paired["f3_idea2_vs_dense_equiv"]["ci95_low"] > 0.0
            and paired["f3_idea2_vs_shuffled_equiv"]["mean_diff"] >= 0.15
            and paired["f3_idea2_vs_shuffled_equiv"]["ci95_low"] > 0.0
        ),
        "F4_idea3a_mupc_bong_viable": (
            agg["Idea3_MuPC_BONG_Pure"]["backprop_calls_p1"]["max"] == 0.0
            and agg["Idea3_MuPC_BONG_Pure"]["backprop_calls_post_p1"]["max"] == 0.0
            and agg["Idea3_MuPC_BONG_Pure"]["p1_val_heldout"]["mean"] >= 0.65
            and agg["Idea3_MuPC_BONG_Pure"]["p3_overall_acc"]["mean"] >= 0.55
            and (
                paired["f4_mupc_bong_vs_adam_overall"]["ci95_low"] > 0.0
                or paired["f4_mupc_bong_vs_adam_overall"]["mean_diff"] >= 0.05
            )
        ),
        "F5_idea3b_backbone_bong": (
            paired["f5_bb_bong_vs_dense_retention"]["mean_diff"] >= 0.10
            and paired["f5_bb_bong_vs_dense_retention"]["ci95_low"] > 0.0
            and paired["f5_bb_bong_vs_dense_pb"]["mean_diff"] >= 0.10
        ),
        "F6_triad_synergy_and_beats_replay": (
            agg["Idea_Triad_All3"]["p3_overall_acc"]["mean"] >= best_single + 0.05
            and paired["f6_triad_vs_replay_overall"]["ci95_low"] > 0.0
            and agg["Idea_Triad_All3"]["p3_rev_equiv_transfer"]["mean"] >= 0.75
            and agg["Idea_Triad_All3"]["p2_split_both_min"]["mean"] >= 0.75
            and agg["Idea_Triad_All3"]["flops_post_p1"]["mean"] < agg["B4_JointOracle"]["flops_total"]["mean"]
        ),
        "F7_calibration_honesty": (
            agg["Idea_Triad_All3"]["p3_overall_nll"]["mean"] <= agg["B1_Dense_SGD"]["p3_overall_nll"]["mean"]
            and agg["Idea_Triad_All3"]["p3_overall_nll"]["mean"] <= 1.2
        ),
        "F8_null_controls_fail": all(
            agg[nm]["p1_val_heldout"]["mean"] < 0.15
            and agg[nm]["p2_split_both_min"]["mean"] < 0.15
            and agg[nm]["p3_rev_equiv_transfer"]["mean"] < 0.15
            and agg[nm]["p3_overall_acc"]["mean"] < 0.15
            for nm in nulls
        ),
    }
    return falsifiers, paired


def run_suite(
    arms: List[str],
    seeds: List[int],
    cfg: LabConfig,
    device: torch.device,
) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    for arm in arms:
        rows = []
        for s in seeds:
            rows.append(run_arm_seed(arm, s, cfg, device))
        agg[arm] = aggregate_arm(rows)
        a = agg[arm]
        print(
            f"[{arm:21s}] p1_vh={a['p1_val_heldout']['mean']:.3f} pb_ret={a['pb_retention']['mean']:.3f} "
            f"p2_min={a['p2_split_both_min']['mean']:.3f} p3_eq={a['p3_rev_equiv_transfer']['mean']:.3f} "
            f"p3_ret={a['p3_retention_all']['mean']:.3f} all={a['p3_overall_acc']['mean']:.3f} "
            f"nll={a['p3_overall_nll']['mean']:.3f} flops_post={a['flops_post_p1']['mean']:.2e} "
            f"t={a['wall_clock_sec']['mean']:.2f}s",
            flush=True,
        )
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tune", "test"], default="tune")
    parser.add_argument("--arms", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps_p1", type=int, default=3600)
    parser.add_argument("--steps_p2", type=int, default=800)
    parser.add_argument("--steps_p3", type=int, default=600)
    parser.add_argument("--lr_dense", type=float, default=6e-3)
    parser.add_argument("--lr_dense_post", type=float, default=4e-3)
    parser.add_argument("--wd_p1", type=float, default=0.22)
    parser.add_argument("--wd_bong_p1", type=float, default=0.10)
    parser.add_argument("--wd_post", type=float, default=0.01)
    parser.add_argument("--lr_idea2", type=float, default=6e-3)
    parser.add_argument("--lr_bong_bb", type=float, default=1.2e-2)
    parser.add_argument("--lr_mupc_bong", type=float, default=2e-3)
    parser.add_argument("--lr_mupc_adam", type=float, default=3e-3)
    parser.add_argument("--wd_mupc", type=float, default=0.02)
    parser.add_argument("--lr_chunk", type=float, default=1e-2)
    parser.add_argument("--lr_bong_chunk", type=float, default=3e-2)
    parser.add_argument("--lambda_equiv", type=float, default=0.35)
    parser.add_argument("--lambda_contra", type=float, default=0.20)
    parser.add_argument("--lambda_sigreg", type=float, default=0.03)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--chunk_steps", type=int, default=6)
    parser.add_argument("--sleep_period", type=int, default=100)
    parser.add_argument("--sleep_steps", type=int, default=4)
    parser.add_argument("--elastic_pull", type=float, default=0.001)
    parser.add_argument("--rho_thresh", type=float, default=0.15)
    parser.add_argument("--tau_pc_skip", type=float, default=0.10)
    parser.add_argument("--num_atoms", type=int, default=64)
    parser.add_argument("--k_train", type=int, default=10)
    parser.add_argument("--eta_z", type=float, default=0.10)
    parser.add_argument("--beta_out", type=float, default=0.10)
    parser.add_argument("--out_name", type=str, default="results_v2.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = LabConfig(
        batch_size=args.batch_size,
        steps_p1=args.steps_p1,
        steps_p2=args.steps_p2,
        steps_p3=args.steps_p3,
        lr_dense=args.lr_dense,
        lr_dense_post=args.lr_dense_post,
        wd_p1=args.wd_p1,
        wd_bong_p1=args.wd_bong_p1,
        wd_post=args.wd_post,
        lr_idea2=args.lr_idea2,
        lr_bong_bb=args.lr_bong_bb,
        lr_mupc_bong=args.lr_mupc_bong,
        lr_mupc_adam=args.lr_mupc_adam,
        wd_mupc=args.wd_mupc,
        lr_chunk=args.lr_chunk,
        lr_bong_chunk=args.lr_bong_chunk,
        lambda_equiv=args.lambda_equiv,
        lambda_contra=args.lambda_contra,
        lambda_sigreg=args.lambda_sigreg,
        chunk_size=args.chunk_size,
        chunk_steps=args.chunk_steps,
        sleep_period=args.sleep_period,
        sleep_steps=args.sleep_steps,
        elastic_pull=args.elastic_pull,
        rho_thresh=args.rho_thresh,
        tau_pc_skip=args.tau_pc_skip,
        num_atoms=args.num_atoms,
        k_train=args.k_train,
        eta_z=args.eta_z,
        beta_out=args.beta_out,
    )

    arm_list = ALL_ARMS if args.arms == "all" else [a.strip() for a in args.arms.split(",") if a.strip()]
    causality = check_causality_gate(device)
    print(f"belief_training_lab_v1 | device={device} | mode={args.mode} | causality={causality}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "tune":
        agg = run_suite(arm_list, DEV_SEEDS, cfg, device)
        dev_falsifiers, dev_paired = ({}, {})
        if set(ALL_ARMS).issubset(set(agg.keys())):
            dev_falsifiers, dev_paired = evaluate_falsifiers(agg)
        log_path = RESULTS_DIR / "tuning_log_v1.json"
        existing: List[Dict[str, Any]] = []
        if log_path.exists():
            existing = json.loads(log_path.read_text(encoding="utf-8"))
        existing.append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": asdict(cfg),
            "causality": causality,
            "arms": agg,
            "dev_falsifiers": dev_falsifiers,
            "dev_paired": dev_paired,
        })
        log_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        print(f"Appended DEV tuning entry #{len(existing)} to {log_path}")
        for fk, fv in dev_falsifiers.items():
            print(f"  DEV {fk}: {fv}", flush=True)
    else:
        agg = run_suite(ALL_ARMS, TEST_SEEDS, cfg, device)
        falsifiers, paired = evaluate_falsifiers(agg)
        payload = {
            "_meta": build_run_meta("belief_training_lab_v2", TEST_SEEDS, DEV_SEEDS, asdict(cfg)),
            "causality": causality,
            "arms": agg,
            "paired_comparisons": paired,
            "falsifiers": falsifiers,
            "all_passed": all(falsifiers.values()),
        }
        out_path = RESULTS_DIR / args.out_name
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved frozen TEST results to {out_path} | all_passed={payload['all_passed']}")
        for fk, fv in falsifiers.items():
            print(f"  {fk}: {fv}")


if __name__ == "__main__":
    main()
