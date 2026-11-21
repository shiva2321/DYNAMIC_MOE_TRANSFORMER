"""
Experiment Runner for `belief_training_v3` (SPGC: Selective Precision-Gated Consolidation).

Protocol (research-protocol.md):
  - DEV_SEEDS = [100, 101, 102, 103, 104] ONLY for --mode tune (logged to tuning_log_v1.json).
  - Freeze results/belief_training_v3/PREREG_v1.md before ANY TEST_SEEDS = [0..19] execution.
  - Exact PyTorch FLOPs keyed per exact active sub-batch size (no hidden screen or optimizer FLOPs),
    CUDA-synchronized wall-clock seconds, exact auxiliary/store byte accounting, and real ablations
    comparing SPGC against B0..B4 baselines, 3 Nulls, and v2's Idea3_Backbone_BONG.
"""

from __future__ import annotations

import argparse
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
    K_CONCEPTS,
    Q_ATTR,
    Q_PROP,
    Q_VAL,
)
from dmoe.spgc import (
    LatentQuarantineGate,
    MuonBONGOptimizer,
    classify_spgc_param_groups,
)

RESULTS_DIR = ROOT / "results" / "belief_training_v3"
DEV_SEEDS = [100, 101, 102, 103, 104]
TEST_SEEDS = list(range(20))

V3_ARMS = [
    "B0_SurfaceDict",
    "B1_Dense_SGD",
    "B2_Dense_Replay",
    "B4_JointOracle",
    "Null_Untrained",
    "Null_ShuffledTargets",
    "Null_BlindZeroInput",
    "V2_Backbone_BONG",
    "SPGC_Full",
    "SPGC_NoMuon",
    "SPGC_NoSkip",
    "SPGC_NoQuarantine",
    "SPGC_NoBONG",
    "SPGC_NoReplay",
]


@dataclass
class V3Config:
    batch_size: int = 128
    steps_p1: int = 3600
    steps_pb: int = 10
    steps_p2: int = 800
    steps_p3: int = 600
    confirm_frac: float = 0.25

    # Baselines (matched from frozen PREREG_v2)
    lr_dense: float = 6e-3
    lr_dense_post: float = 4e-3
    wd_p1: float = 0.22
    wd_bong_p1: float = 0.10
    wd_post: float = 0.01
    lr_bong_bb: float = 1.2e-2

    # SPGC parameters (tuned ONLY on DEV_SEEDS)
    spgc_lr_muon: float = 1.5e-2
    spgc_lr_adam: float = 6e-3
    spgc_lr_consolidate: float = 1.0e-2
    spgc_wd_trunk_p1: float = 0.20
    spgc_wd_fast: float = 0.01
    spgc_wd_post: float = 0.01
    spgc_p1_min_steps: int = 850
    spgc_p1_skip_ce: float = 0.12
    spgc_p1_refresh_period: int = 64
    spgc_min_flush_p1: int = 32
    spgc_min_flush_post: int = 16
    spgc_tau_skip: float = 0.12
    spgc_conf_established: float = 0.45
    spgc_min_flip_count: int = 6
    spgc_flip_ratio: float = 0.35
    spgc_min_sources: int = 2
    spgc_rev_grace_steps: int = 24
    spgc_replay_k: int = 16
    spgc_bong_gamma: float = 0.996
    spgc_bong_damping: float = 0.01
    spgc_bong_init_prec: float = 0.02
    spgc_bong_min_prec: float = 0.005
    spgc_bong_max_prec: float = 100.0
    spgc_bong_prior_pull: float = 0.0
    spgc_grokfast_alpha: float = 0.0
    spgc_grokfast_lamb: float = 0.0
    spgc_use_fast_adapter: bool = True
    max_store_bytes: int = 16384


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


class ExactFlopTracker:
    """Measure exact FLOPs per (kind, sub_batch_size) once and sum real counts."""

    def __init__(self) -> None:
        self.per_kind_flops: Dict[str, int] = {}
        self.counts: Dict[str, int] = {}

    def record(self, kind: str, fn: Any) -> Any:
        if kind not in self.per_kind_flops:
            with FlopCounterMode(display=False) as fc:
                res = fn()
            self.per_kind_flops[kind] = int(fc.get_total_flops())
            self.counts[kind] = self.counts.get(kind, 0) + 1
            return res
        self.counts[kind] = self.counts.get(kind, 0) + 1
        return fn()

    def total_flops(self, prefix: Optional[str] = None) -> int:
        tot = 0
        for k, cnt in self.counts.items():
            if prefix is None or k.startswith(prefix):
                tot += int(cnt) * int(self.per_kind_flops.get(k, 0))
        return tot


def check_causality_gate(device: torch.device) -> Dict[str, float]:
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
) -> List[Dict[str, Any]]:
    low_ids = {id(p) for m in (model.ctx_proj, model.fast_adapter) for p in m.parameters()}
    trunk_p = [p for p in model.parameters() if id(p) not in low_ids]
    ctx_p = [p for p in model.parameters() if id(p) in low_ids]
    return [
        {"params": trunk_p, "weight_decay": float(wd_trunk)},
        {"params": ctx_p, "weight_decay": float(wd_ctx)},
    ]


def run_arm_seed(
    arm: str,
    seed: int,
    cfg: V3Config,
    device: torch.device,
) -> Dict[str, Any]:
    set_all_seeds(seed * 1009 + 7)
    task = BeliefMeaningTask(seed=seed)
    stream_gen = torch.Generator().manual_seed(seed * 313 + 17)
    eval_base = seed * 977 + 3
    flops = ExactFlopTracker()
    bp_p1 = 0
    bp_post = 0
    p1_batches = 0
    skipped_p1_batches = 0
    post_batches = 0
    skipped_post_batches = 0

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

        sb = d.storage_bytes()
        return {
            **p1_res, **pb_res, **p2_res, **p3_res,
            "pb_burst_flips": 0.0,
            "flops_p1": 0.0,
            "flops_post_p1": 0.0,
            "flops_total": 0.0,
            "backprop_calls_p1": 0.0,
            "backprop_calls_post_p1": 0.0,
            "skip_rate_p1": 1.0,
            "skip_rate_post_p1": 1.0,
            "wall_clock_sec": float(t_train),
            "param_count": 0.0,
            "store_bytes": float(sb),
            "aux_memory_bytes": float(sb),
        }

    model = ConceptBeliefBackbone(use_concept_readout=True).to(device)
    param_count = float(sum(p.numel() for p in model.parameters()))

    if arm == "Null_Untrained":
        p1_res = eval_model(_backbone_predict(model), task, "p1", device, eval_base + 1)
        pb_res = eval_model(_backbone_predict(model), task, "pb", device, eval_base + 2)
        p2_res = eval_model(_backbone_predict(model), task, "p2", device, eval_base + 3)
        p3_res = eval_model(_backbone_predict(model), task, "p3", device, eval_base + 4)
        return {
            **p1_res, **pb_res, **p2_res, **p3_res,
            "pb_burst_flips": 0.0,
            "flops_p1": 0.0,
            "flops_post_p1": 0.0,
            "flops_total": 0.0,
            "backprop_calls_p1": 0.0,
            "backprop_calls_post_p1": 0.0,
            "skip_rate_p1": 1.0,
            "skip_rate_post_p1": 1.0,
            "wall_clock_sec": 0.0,
            "param_count": param_count,
            "store_bytes": 0.0,
            "aux_memory_bytes": 0.0,
        }

    if arm == "B4_JointOracle":
        opt = torch.optim.AdamW(_make_adamw_groups(model, cfg.wd_p1, cfg.wd_post), lr=cfg.lr_dense)
        tot_steps = cfg.steps_p1 + cfg.steps_p2 + cfg.steps_p3
        _sync_device(device)
        t0 = time.perf_counter()
        for _ in range(tot_steps):
            x, y = task.sample_stream_batch("joint_oracle", cfg.batch_size, stream_gen)
            xd, yd = x.to(device), y.to(device)

            def _jo_step():
                opt.zero_grad()
                loss = F.cross_entropy(model(xd)["logits"], yd)
                loss.backward()
                opt.step()

            flops.record("p1_joint", _jo_step)
            bp_p1 += 1
        _sync_device(device)
        t_train = time.perf_counter() - t0
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
            "backprop_calls_p1": float(bp_p1),
            "backprop_calls_post_p1": 0.0,
            "skip_rate_p1": 0.0,
            "skip_rate_post_p1": 0.0,
            "wall_clock_sec": float(t_train),
            "param_count": param_count,
            "store_bytes": 0.0,
            "aux_memory_bytes": 0.0,
        }

    # --- SPGC family (and Null controls using SPGC loop) ---
    is_spgc = arm.startswith("SPGC_") or arm in ("Null_ShuffledTargets", "Null_BlindZeroInput")
    if is_spgc:
        is_shuffled = arm == "Null_ShuffledTargets"
        is_blind = arm == "Null_BlindZeroInput"
        use_muon = arm != "SPGC_NoMuon"
        use_skip = arm != "SPGC_NoSkip"
        use_quarantine = arm != "SPGC_NoQuarantine"
        use_bong = arm != "SPGC_NoBONG"
        use_replay = arm != "SPGC_NoReplay"

        groups = classify_spgc_param_groups(
            model,
            wd_trunk=cfg.spgc_wd_trunk_p1,
            wd_fast=cfg.spgc_wd_fast,
            lr_muon=cfg.spgc_lr_muon,
            lr_adam=cfg.spgc_lr_adam,
            lr_fast=cfg.spgc_lr_consolidate,
        )
        opt_spgc = MuonBONGOptimizer(
            groups,
            grokfast_alpha=cfg.spgc_grokfast_alpha,
            grokfast_lamb=cfg.spgc_grokfast_lamb,
            bong_gamma=cfg.spgc_bong_gamma,
            bong_damping=cfg.spgc_bong_damping,
            bong_init_prec=cfg.spgc_bong_init_prec,
            bong_min_prec=cfg.spgc_bong_min_prec,
            bong_max_prec=cfg.spgc_bong_max_prec,
            bong_prior_pull=cfg.spgc_bong_prior_pull,
            mode="acquire",
            track_fisher=True,
            disable_muon=(not use_muon),
            disable_bong=(not use_bong),
        )
        gate = LatentQuarantineGate(
            latent_dim=model.d_model,
            seed=seed,
            max_bytes=cfg.max_store_bytes,
            conf_established=cfg.spgc_conf_established,
            tau_skip=cfg.spgc_tau_skip,
            min_flip_count=cfg.spgc_min_flip_count,
            flip_ratio=cfg.spgc_flip_ratio,
            min_sources=cfg.spgc_min_sources,
            rev_grace_steps=cfg.spgc_rev_grace_steps,
            bytes_per_sample=32,
        )

        # Phase 1 (Acquire)
        t_train = 0.0
        ema_ce_p1 = 2.77
        p1_screening_active = False
        _sync_device(device)
        t0 = time.perf_counter()
        for step in range(cfg.steps_p1):
            p1_batches += 1
            x, y = task.sample_stream_batch("p1", cfg.batch_size, stream_gen)
            if is_shuffled:
                y = torch.randint(0, K_CONCEPTS, y.shape, generator=stream_gen, dtype=y.dtype)
            xd = torch.zeros_like(x, device=device) if is_blind else x.to(device)
            yd = y.to(device)
            src = f"src_{step % 3}"

            # Activate selective screening cleanly once initial memorization/grokking settles
            if use_skip and (step >= cfg.spgc_p1_min_steps) and (ema_ce_p1 < cfg.spgc_p1_skip_ce):
                p1_screening_active = True

            if p1_screening_active and (step % cfg.spgc_p1_refresh_period != 0):
                def _p1_screen():
                    with torch.no_grad():
                        return model(xd, use_fast_adapter=False)

                out_s = flops.record(f"p1_screen_{xd.size(0)}", _p1_screen)
                ce_s = float(F.cross_entropy(out_s["logits"], yd).item())
                ema_ce_p1 = 0.95 * ema_ce_p1 + 0.05 * ce_s

                x_act, y_act, _ = gate.triage_batch(
                    xd,
                    yd,
                    out_s["logits"],
                    out_s["z_expr"],
                    source_id=src,
                    q_tag=xd[:, 5],
                    allow_core_write=use_replay,
                    skip_familiar=True,
                    enable_quarantine=False,
                )
                is_last_p1 = (step == cfg.steps_p1 - 1)
                bx, by = gate.enqueue_or_flush(
                    x_act,
                    y_act,
                    min_flush=cfg.spgc_min_flush_p1,
                    force_flush=is_last_p1,
                    device=device,
                )
                if bx is None:
                    skipped_p1_batches += 1
                    continue

                def _p1_comp_bwd():
                    opt_spgc.zero_grad()
                    o = model(bx, use_fast_adapter=False)
                    loss = F.cross_entropy(o["logits"], by)
                    loss.backward()
                    opt_spgc.step()

                flops.record(f"p1_comp_{bx.size(0)}", _p1_comp_bwd)
                bp_p1 += 1
            else:
                def _p1_full():
                    opt_spgc.zero_grad()
                    o = model(xd, use_fast_adapter=False)
                    loss = F.cross_entropy(o["logits"], yd)
                    loss.backward()
                    opt_spgc.step()
                    return o, float(loss.item())

                out_f, lv = flops.record(f"p1_full_{xd.size(0)}", _p1_full)
                bp_p1 += 1
                ema_ce_p1 = 0.95 * ema_ce_p1 + 0.05 * lv
                # Establish Phase-1 priors & fill compact core reservoir once initial accuracy warms up
                if step >= max(400, cfg.spgc_p1_min_steps // 2) and (step % 8 == 0):
                    gate.triage_batch(
                        xd,
                        yd,
                        out_f["logits"],
                        out_f["z_expr"],
                        source_id=src,
                        q_tag=xd[:, 5],
                        allow_core_write=use_replay,
                        skip_familiar=False,
                        enable_quarantine=False,
                    )

        # Flush any queued P1 surprises cleanly before transition
        bx_rem, by_rem = gate.enqueue_or_flush(None, None, min_flush=1, force_flush=True, device=device)
        if bx_rem is not None:
            def _p1_tail():
                opt_spgc.zero_grad()
                loss = F.cross_entropy(model(bx_rem, use_fast_adapter=False)["logits"], by_rem)
                loss.backward()
                opt_spgc.step()
            flops.record(f"p1_comp_{bx_rem.size(0)}", _p1_tail)
            bp_p1 += 1

        _sync_device(device)
        t_train += time.perf_counter() - t0
        p1_res = eval_model(_backbone_predict(model, blind=is_blind, use_fast=False), task, "p1", device, eval_base + 1)

        # Calibrate diagonal Fisher curvature on verified core memories before locking trunk
        _sync_device(device)
        t0 = time.perf_counter()
        if use_bong and len(gate.core_x) >= 32:
            for _ in range(4):
                rx_c, ry_c = gate.sample_core_replay(32, device)
                if rx_c is not None:
                    def _fcal():
                        opt_spgc.zero_grad()
                        lg_c = model(rx_c, use_fast_adapter=False)["logits"]
                        # Predictive Fisher using sampled predicted classes (accurate even when training loss ~ 0)
                        y_hat = torch.distributions.Categorical(logits=lg_c.detach()).sample()
                        F.cross_entropy(lg_c, y_hat).backward()
                        opt_spgc.accumulate_fisher_sample(grads_scale=0.5)
                        opt_spgc.zero_grad()
                    flops.record(f"post_fcal_{rx_c.size(0)}", _fcal)
        opt_spgc.set_mode("consolidate", snapshot_anchor=True, new_weight_decay=cfg.spgc_wd_post)
        _sync_device(device)
        t_train += time.perf_counter() - t0

        # Continual phases PB -> P2 -> P3
        pb_res: Dict[str, float] = {}
        p2_res: Dict[str, float] = {}
        pb_flips = 0.0
        use_fa = bool(cfg.spgc_use_fast_adapter)

        for phase, n_steps in [("pb", cfg.steps_pb), ("p2", cfg.steps_p2), ("p3", cfg.steps_p3)]:
            _sync_device(device)
            t0 = time.perf_counter()
            flips_before = gate.flips_executed
            for step in range(n_steps):
                post_batches += 1
                x, y = task.sample_stream_batch(phase, cfg.batch_size, stream_gen, cfg.confirm_frac)
                if is_shuffled:
                    y = torch.randint(0, K_CONCEPTS, y.shape, generator=stream_gen, dtype=y.dtype)
                xd = torch.zeros_like(x, device=device) if is_blind else x.to(device)
                yd = y.to(device)
                src = "noisy_single" if phase == "pb" else f"src_{step % 3}"

                def _post_screen():
                    with torch.no_grad():
                        return model(xd, use_fast_adapter=use_fa)

                out_s = flops.record(f"post_screen_{xd.size(0)}", _post_screen)
                x_act, y_act, flips_here = gate.triage_batch(
                    xd,
                    yd,
                    out_s["logits"],
                    out_s["z_expr"],
                    source_id=src,
                    q_tag=xd[:, 5],
                    allow_core_write=(use_replay and phase != "pb"),
                    skip_familiar=use_skip,
                    enable_quarantine=use_quarantine,
                )
                is_last_step = (step == n_steps - 1) and (phase != "pb")
                min_f = 1 if (not use_skip or flips_here > 0 or step < 20) else cfg.spgc_min_flush_post
                bx, by = gate.enqueue_or_flush(
                    x_act,
                    y_act,
                    min_flush=min_f,
                    force_flush=is_last_step,
                    device=device,
                )
                if bx is None:
                    skipped_post_batches += 1
                    continue

                if flips_here > 0 and use_bong:
                    opt_spgc.temper_precision(roles=("fast",), factor=0.25)

                if use_replay:
                    rx, ry = gate.sample_core_replay(cfg.spgc_replay_k, device)
                    if rx is not None:
                        xb = torch.cat([bx, rx], dim=0)
                        yb = torch.cat([by, ry], dim=0)
                    else:
                        xb, yb = bx, by
                else:
                    xb, yb = bx, by

                def _post_bwd():
                    opt_spgc.zero_grad()
                    o = model(xb, use_fast_adapter=use_fa)
                    loss = F.cross_entropy(o["logits"], yb)
                    loss.backward()
                    opt_spgc.step()

                flops.record(f"post_bwd_{xb.size(0)}", _post_bwd)
                bp_post += 1

            _sync_device(device)
            t_train += time.perf_counter() - t0
            if phase == "pb":
                pb_flips = float(gate.flips_executed - flips_before)
                pb_res = eval_model(_backbone_predict(model, blind=is_blind, use_fast=use_fa), task, "pb", device, eval_base + 2)
            elif phase == "p2":
                p2_res = eval_model(_backbone_predict(model, blind=is_blind, use_fast=use_fa), task, "p2", device, eval_base + 3)

        p3_res = eval_model(_backbone_predict(model, blind=is_blind, use_fast=use_fa), task, "p3", device, eval_base + 4)
        sb = float(gate.storage_bytes())
        aux_b = float(gate.storage_bytes() + opt_spgc.precision_bytes())
        return {
            **p1_res, **pb_res, **p2_res, **p3_res,
            "pb_burst_flips": pb_flips,
            "flops_p1": float(flops.total_flops("p1_")),
            "flops_post_p1": float(flops.total_flops("post_")),
            "flops_total": float(flops.total_flops()),
            "backprop_calls_p1": float(bp_p1),
            "backprop_calls_post_p1": float(bp_post),
            "skip_rate_p1": float(skipped_p1_batches / max(1, p1_batches)),
            "skip_rate_post_p1": float(skipped_post_batches / max(1, post_batches)),
            "wall_clock_sec": float(t_train),
            "param_count": param_count,
            "store_bytes": sb,
            "aux_memory_bytes": aux_b,
        }

    # --- Reference Baselines: B1_Dense_SGD, B2_Dense_Replay, V2_Backbone_BONG ---
    is_v2_bong = arm == "V2_Backbone_BONG"
    is_replay = arm == "B2_Dense_Replay"
    if is_v2_bong:
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
    else:
        opt_bb = torch.optim.AdamW(_make_adamw_groups(model, cfg.wd_p1, cfg.wd_post), lr=cfg.lr_dense)

    max_rep = cfg.max_store_bytes // 28
    rep_x: List[torch.Tensor] = []
    rep_y: List[torch.Tensor] = []
    rep_seen = 0
    ema_loss_bb = 2.77

    t_train = 0.0
    _sync_device(device)
    t0 = time.perf_counter()
    for step in range(cfg.steps_p1):
        p1_batches += 1
        x, y = task.sample_stream_batch("p1", cfg.batch_size, stream_gen)
        xd, yd = x.to(device), y.to(device)

        def _p1_bb():
            opt_bb.zero_grad()
            loss = F.cross_entropy(model(xd, use_fast_adapter=False)["logits"], yd)
            loss.backward()
            opt_bb.step()
            return float(loss.item())

        lv = flops.record("p1_bb", _p1_bb)
        bp_p1 += 1
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

    _sync_device(device)
    t_train += time.perf_counter() - t0
    p1_res = eval_model(_backbone_predict(model), task, "p1", device, eval_base + 1)

    if not is_v2_bong:
        for pg in opt_bb.param_groups:
            pg["lr"] = cfg.lr_dense_post
            pg["weight_decay"] = cfg.wd_post
    else:
        opt_bb.prior_wd = cfg.wd_post

    pb_res, p2_res = {}, {}
    for phase, n_steps in [("pb", cfg.steps_pb), ("p2", cfg.steps_p2), ("p3", cfg.steps_p3)]:
        _sync_device(device)
        t0 = time.perf_counter()
        for step in range(n_steps):
            post_batches += 1
            x, y = task.sample_stream_batch(phase, cfg.batch_size, stream_gen, cfg.confirm_frac)
            xd, yd = x.to(device), y.to(device)

            if is_v2_bong:
                def _bong_fwd():
                    with torch.no_grad():
                        return float(F.cross_entropy(model(xd)["logits"], yd).item())

                ce_now = flops.record("post_bong_fwd", _bong_fwd)
                if (ce_now - ema_loss_bb) < 0.05 and ce_now < 0.35:
                    skipped_post_batches += 1
                    continue

                def _bong_bwd():
                    opt_bb.zero_grad()
                    loss = F.cross_entropy(model(xd)["logits"], yd)
                    loss.backward()
                    opt_bb.step()

                flops.record("post_bong_bwd", _bong_bwd)
                bp_post += 1
                if phase != "pb":
                    ema_loss_bb = 0.97 * ema_loss_bb + 0.03 * ce_now
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
                    loss = F.cross_entropy(model(xb)["logits"], yb)
                    loss.backward()
                    opt_bb.step()

                kind = "post_rep" if (is_replay and len(rep_x) >= 16) else "post_bb"
                flops.record(kind, _bb_post)
                bp_post += 1
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

        _sync_device(device)
        t_train += time.perf_counter() - t0
        if phase == "pb":
            pb_res = eval_model(_backbone_predict(model), task, "pb", device, eval_base + 2)
        elif phase == "p2":
            p2_res = eval_model(_backbone_predict(model), task, "p2", device, eval_base + 3)

    p3_res = eval_model(_backbone_predict(model), task, "p3", device, eval_base + 4)
    aux_b = float(opt_bb.precision_bytes()) if is_v2_bong else float(len(rep_x) * 28)
    sb = float(len(rep_x) * 28) if is_replay else 0.0
    return {
        **p1_res, **pb_res, **p2_res, **p3_res,
        "pb_burst_flips": 0.0,
        "flops_p1": float(flops.total_flops("p1_")),
        "flops_post_p1": float(flops.total_flops("post_")),
        "flops_total": float(flops.total_flops()),
        "backprop_calls_p1": float(bp_p1),
        "backprop_calls_post_p1": float(bp_post),
        "skip_rate_p1": 0.0,
        "skip_rate_post_p1": float(skipped_post_batches / max(1, post_batches)),
        "wall_clock_sec": float(t_train),
        "param_count": param_count,
        "store_bytes": sb,
        "aux_memory_bytes": aux_b,
    }


def summarize_metric(vals: List[float]) -> Dict[str, Any]:
    n = len(vals)
    mean = sum(vals) / max(1, n)
    var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    ci95 = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": sd,
        "ci95_half": ci95,
        "min": min(vals) if n else 0.0,
        "max": max(vals) if n else 0.0,
        "n": n,
        "per_seed": vals,
    }


def bootstrap_ci_diff(a: List[float], b: List[float], n_boot: int = 2000, seed: int = 12345) -> Dict[str, float]:
    n = len(a)
    diffs = torch.tensor([x - y for x, y in zip(a, b)], dtype=torch.float64)
    mean_d = float(diffs.mean().item())
    if n <= 1:
        return {"mean_diff": mean_d, "ci95_low": mean_d, "ci95_high": mean_d}
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    boot_means = diffs[idx].mean(dim=1)
    q = torch.quantile(boot_means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {"mean_diff": mean_d, "ci95_low": float(q[0].item()), "ci95_high": float(q[1].item())}


def evaluate_v3_falsifiers(agg: Dict[str, Any]) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    paired = {
        "spgc_vs_replay_overall": bootstrap_ci_diff(
            agg["SPGC_Full"]["p3_overall_acc"]["per_seed"],
            agg["B2_Dense_Replay"]["p3_overall_acc"]["per_seed"],
        ),
        "spgc_vs_sgd_overall": bootstrap_ci_diff(
            agg["SPGC_Full"]["p3_overall_acc"]["per_seed"],
            agg["B1_Dense_SGD"]["p3_overall_acc"]["per_seed"],
        ),
        "spgc_vs_replay_pb_ret": bootstrap_ci_diff(
            agg["SPGC_Full"]["pb_retention"]["per_seed"],
            agg["B2_Dense_Replay"]["pb_retention"]["per_seed"],
        ),
        "spgc_vs_v2bong_pb_ret": bootstrap_ci_diff(
            agg["SPGC_Full"]["pb_retention"]["per_seed"],
            agg["V2_Backbone_BONG"]["pb_retention"]["per_seed"],
        ),
        "spgc_vs_replay_equiv": bootstrap_ci_diff(
            agg["SPGC_Full"]["p3_rev_equiv_transfer"]["per_seed"],
            agg["B2_Dense_Replay"]["p3_rev_equiv_transfer"]["per_seed"],
        ),
        "spgc_vs_nomuon_p1": bootstrap_ci_diff(
            agg["SPGC_Full"]["p1_val_heldout"]["per_seed"],
            agg["SPGC_NoMuon"]["p1_val_heldout"]["per_seed"],
        ),
        "spgc_vs_noskip_flops": bootstrap_ci_diff(
            agg["SPGC_Full"]["flops_total"]["per_seed"],
            agg["SPGC_NoSkip"]["flops_total"]["per_seed"],
        ),
        "spgc_vs_noquar_pb": bootstrap_ci_diff(
            agg["SPGC_Full"]["pb_retention"]["per_seed"],
            agg["SPGC_NoQuarantine"]["pb_retention"]["per_seed"],
        ),
        "spgc_vs_nobong_equiv": bootstrap_ci_diff(
            agg["SPGC_Full"]["p3_rev_equiv_transfer"]["per_seed"],
            agg["SPGC_NoBONG"]["p3_rev_equiv_transfer"]["per_seed"],
        ),
    }

    nulls = ("Null_Untrained", "Null_ShuffledTargets", "Null_BlindZeroInput")
    falsifiers = {
        "V1_beats_or_matches_replay_with_less_total_flops_and_time": (
            agg["SPGC_Full"]["p3_overall_acc"]["mean"] >= agg["B2_Dense_Replay"]["p3_overall_acc"]["mean"]
            and agg["SPGC_Full"]["flops_total"]["mean"] <= 0.80 * agg["B2_Dense_Replay"]["flops_total"]["mean"]
            and agg["SPGC_Full"]["wall_clock_sec"]["mean"] <= agg["B2_Dense_Replay"]["wall_clock_sec"]["mean"]
        ),
        "V2_burst_immunity_over_replay_and_v2bong": (
            agg["SPGC_Full"]["pb_retention"]["mean"] >= 0.80
            and agg["SPGC_Full"]["pb_burst_flips"]["mean"] <= 0.10
            and paired["spgc_vs_replay_pb_ret"]["ci95_low"] > 0.50
            and paired["spgc_vs_v2bong_pb_ret"]["ci95_low"] > 0.40
        ),
        "V3_context_split_and_equiv_transfer_intact": (
            agg["SPGC_Full"]["p2_split_both_min"]["mean"] >= 0.85
            and agg["SPGC_Full"]["p3_rev_equiv_transfer"]["mean"] >= 0.75
            and paired["spgc_vs_replay_equiv"]["ci95_low"] > 0.05
        ),
        "V4_calibrated_probabilities": (
            agg["SPGC_Full"]["p3_overall_nll"]["mean"] <= agg["B2_Dense_Replay"]["p3_overall_nll"]["mean"]
            and agg["SPGC_Full"]["p3_overall_nll"]["mean"] <= 0.90
        ),
        "V5_budget_honesty": agg["SPGC_Full"]["store_bytes"]["max"] <= 16384.0,
        "V6_null_controls_fail": all(
            agg[nm]["p1_val_heldout"]["mean"] < 0.15
            and agg[nm]["p2_split_both_min"]["mean"] <= 0.30
            and agg[nm]["p3_rev_equiv_transfer"]["mean"] <= 0.30
            and agg[nm]["p3_overall_acc"]["mean"] < 0.15
            for nm in nulls
        ),
    }
    return falsifiers, paired


def run_suite(arms: List[str], seeds: List[int], cfg: V3Config, device: torch.device) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    for arm in arms:
        rows = [run_arm_seed(arm, s, cfg, device) for s in seeds]
        keys = sorted(rows[0].keys())
        agg[arm] = {k: summarize_metric([r[k] for r in rows]) for k in keys}
        print(
            f"[{arm:<20}] "
            f"p1_vh={agg[arm]['p1_val_heldout']['mean']:.3f} "
            f"pb_ret={agg[arm]['pb_retention']['mean']:.3f} "
            f"p2_min={agg[arm]['p2_split_both_min']['mean']:.3f} "
            f"p3_eq={agg[arm]['p3_rev_equiv_transfer']['mean']:.3f} "
            f"p3_ret={agg[arm]['p3_retention_all']['mean']:.3f} "
            f"all={agg[arm]['p3_overall_acc']['mean']:.3f} "
            f"nll={agg[arm]['p3_overall_nll']['mean']:.3f} "
            f"fl_tot={agg[arm]['flops_total']['mean']:.2e} "
            f"bp={agg[arm]['backprop_calls_p1']['mean']:.0f}+{agg[arm]['backprop_calls_post_p1']['mean']:.0f} "
            f"t={agg[arm]['wall_clock_sec']['mean']:.2f}s",
            flush=True,
        )
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="belief_training_v3 SPGC suite")
    parser.add_argument("--mode", choices=["tune", "test"], default="tune")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_name", default="results_v1.json")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg = V3Config()
    causality = check_causality_gate(device)
    print(f"belief_training_v3 | device={device} | mode={args.mode} | causality={causality}", flush=True)
    assert max(causality.values()) <= 1e-5, f"Causality failure: {causality}"

    arms = args.arms if args.arms else list(V3_ARMS)
    if args.mode == "tune":
        agg = run_suite(arms, DEV_SEEDS, cfg, device)
        log_path = RESULTS_DIR / "tuning_log_v1.json"
        existing = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
        entry = {
            "meta": build_run_meta("belief_training_v3_dev", DEV_SEEDS, DEV_SEEDS, asdict(cfg)),
            "note": args.note,
            "arms": agg,
        }
        if set(arms) >= set(V3_ARMS):
            f_res, p_res = evaluate_v3_falsifiers(agg)
            entry["falsifiers"] = f_res
            entry["paired_comparisons"] = p_res
        existing.append(entry)
        log_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        print(f"Logged DEV entry #{len(existing)} to {log_path}")
    else:
        agg = run_suite(arms, TEST_SEEDS, cfg, device)
        falsifiers, paired = evaluate_v3_falsifiers(agg) if set(arms) >= set(V3_ARMS) else ({}, {})
        out_obj = {
            "_meta": build_run_meta("belief_training_v3", TEST_SEEDS, DEV_SEEDS, asdict(cfg)),
            "causality": causality,
            "arms": agg,
            "paired_comparisons": paired,
            "falsifiers": falsifiers,
            "all_passed": all(falsifiers.values()) if falsifiers else False,
        }
        out_path = RESULTS_DIR / args.out_name
        out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
        print(f"Saved frozen TEST results to {out_path} | all_passed={out_obj['all_passed']}")
        for fk, fv in falsifiers.items():
            print(f"  {fk}: {fv}")


if __name__ == "__main__":
    main()
