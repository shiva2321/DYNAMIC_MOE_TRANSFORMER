"""
Direct Integration Benchmark on `RealSelfStabilizingDMoE` (`Stage 3B-Mem` Continual Stream).

Compares on the real Dynamic Morphic MoE Transformer architecture (`dmoe/real_model.py`):
  1. `RealDMoE_AdamW_Base`: standard dense AdamW (no upkeeper / no selective skip)
  2. `RealDMoE_Stage3_Ref`: exact Stage 3 baseline (`AdamW` + surface-bigram `CalibratedBayesianUpkeeper` <= 4096B)
  3. `RealDMoE_SPGC`: our unified SPGC (`MuonBONGOptimizer` + token-level selective compaction +
     latent SimHash `LatentQuarantineGate` <= 4096B with ZERO surface bigram keys and ZERO inference logit mixing)

Measures accuracy, NLL, burst immunity, exact PyTorch FLOPs (`ExactFlopTracker`), backward call count,
wall-clock training seconds, and storage bytes on DEV (`[100..104]`) or TEST (`[0..19]`).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmoe.data.stage3_tasks import (
    ALPHABET_SIZE,
    ALPHABET_START,
    build_compositional_belief_tables,
    generate_3b_pair_eval_batch,
)
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.spgc import LatentQuarantineGate, MuonBONGOptimizer
from dmoe.upkeeper import CalibratedBayesianUpkeeper

RESULTS_DIR = ROOT / "results" / "belief_training_v3"
DEV_SEEDS = [100, 101, 102, 103, 104]
TEST_SEEDS = list(range(20))


class ExactFlopTracker:
    def __init__(self) -> None:
        self.per_kind: Dict[str, int] = {}
        self.counts: Dict[str, int] = {}

    def record(self, kind: str, fn: Any) -> Any:
        if kind not in self.per_kind:
            with FlopCounterMode(display=False) as fc:
                res = fn()
            self.per_kind[kind] = int(fc.get_total_flops())
            self.counts[kind] = self.counts.get(kind, 0) + 1
            return res
        self.counts[kind] = self.counts.get(kind, 0) + 1
        return fn()

    def total(self) -> int:
        return sum(cnt * self.per_kind.get(k, 0) for k, cnt in self.counts.items())


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _eval_split(
    model: RealSelfStabilizingDMoE,
    upk: Optional[CalibratedBayesianUpkeeper],
    tbl: torch.Tensor,
    pairs: List[Tuple[int, int]],
    eval_seed: int,
    device: torch.device,
) -> Tuple[float, float]:
    x, y, mk = generate_3b_pair_eval_batch(tbl, pairs, batch_size=64, seq_len=16, seed=eval_seed)
    x, y, mk = x.to(device), y.to(device), mk.to(device)
    with torch.no_grad():
        model.eval()
        out = model(x, update_stats=False)
        lg = out["output"].reshape(x.size(0), x.size(1), -1)
        lp = upk.fuse_logits(x, lg) if upk is not None else F.log_softmax(lg, dim=-1)
        acc = float(((lp.argmax(-1) == y) & mk).sum().item() / max(1, int(mk.sum().item())))
        nll = float(F.nll_loss(lp[mk], y[mk]).item())
    return acc, nll


def run_real_dmoe_seed(
    arm: str,
    seed: int,
    device: torch.device,
    init_steps: int = 2200,
    rev_steps: int = 800,
) -> Dict[str, float]:
    set_all_seeds(seed)
    spec = build_compositional_belief_tables(seed=seed)
    init_t, burst_t, rev_t = spec["init_table"], spec["burst_table"], spec["rev_table"]
    seen_mask, seen_pairs = spec["seen_mask"], spec["seen_pairs"]
    held_pairs, exc_pairs = spec["heldout_pairs"], spec["exc_pairs"]
    P = (seen_mask.float() / seen_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)).to(device)

    model = RealSelfStabilizingDMoE(
        d_model=64,
        num_heads=4,
        d_ff=256,
        initial_layers=1,
        initial_experts=1,
        chunk_size=1,
        enable_thinking=False,
        enable_memory=False,
        enable_world_model=False,
        causal_attn=True,
        use_rope=True,
        enable_three_level_attn=False,
        local_window=2,
    ).to(device)
    flops = ExactFlopTracker()
    bs = 64
    lr = 1.2e-2
    wd = 1.5
    gen = torch.Generator(device=device).manual_seed(seed * 131 + 7)
    is_spgc = arm == "RealDMoE_SPGC"
    is_ref = arm == "RealDMoE_Stage3_Ref"

    upk = CalibratedBayesianUpkeeper(max_total_bytes=4096, min_sources_for_flip=2, min_flip_count=3) if is_ref else None
    gate: Optional[LatentQuarantineGate] = None
    opt_spgc: Optional[MuonBONGOptimizer] = None
    opt_adam: Optional[torch.optim.Optimizer] = None

    def _build_adam_opt(bb_scale: float = 1.0, phase_lr: float = lr) -> torch.optim.AdamW:
        pgs = []
        for nm, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_head = "decoder" in nm or "moe" in nm or "out_head" in nm or "ffns" in nm
            sc = 1.0 if is_head else bb_scale
            if sc <= 0.0:
                continue
            pgs.append({"params": [p], "lr": phase_lr * sc, "base_lr": phase_lr * sc, "weight_decay": wd if p.ndim >= 2 else 0.0})
        return torch.optim.AdamW(pgs, betas=(0.9, 0.98))

    if is_spgc:
        trunk_muon, trunk_vec, fast_muon, fast_vec = [], [], [], []
        for nm, p in model.named_parameters():
            if not p.requires_grad:
                continue
            ln = nm.lower()
            is_fast = ("moe" in ln) or ("decoder" in ln)
            is_embed = ("byte_embed" in ln) or ("pos_embed" in ln) or ("chunk_proj" in ln) or ("decoder" in ln) or ("norm" in ln)
            if p.ndim == 2 and not is_embed:
                (fast_muon if is_fast else trunk_muon).append(p)
            else:
                (fast_vec if is_fast else trunk_vec).append(p)
        groups = []
        if trunk_muon:
            groups.append({"name": "trunk_muon", "params": trunk_muon, "use_muon": True, "role": "trunk", "lr": 0.012, "base_lr": 0.012, "lr_consolidate": 0.008, "weight_decay": 1.5, "prec_scale": 2.0})
        if trunk_vec:
            groups.append({"name": "trunk_vec", "params": trunk_vec, "use_muon": False, "role": "trunk", "lr": 0.012, "base_lr": 0.012, "lr_consolidate": 0.006, "weight_decay": 1.5, "prec_scale": 2.0})
        if fast_muon:
            groups.append({"name": "fast_muon", "params": fast_muon, "use_muon": True, "role": "fast", "lr": 0.012, "base_lr": 0.012, "lr_consolidate": 0.012, "weight_decay": 1.5, "prec_scale": 0.15})
        if fast_vec:
            groups.append({"name": "fast_vec", "params": fast_vec, "use_muon": False, "role": "fast", "lr": 0.012, "base_lr": 0.012, "lr_consolidate": 0.012, "weight_decay": 1.5, "prec_scale": 0.15})
        opt_spgc = MuonBONGOptimizer(
            groups,
            bong_gamma=0.996,
            bong_damping=0.02,
            bong_init_prec=0.02,
            bong_min_prec=0.005,
            bong_prior_pull=0.0,
            grad_clip=1.0,
            mode="acquire",
            track_fisher=True,
        )
        gate = LatentQuarantineGate(
            latent_dim=64,
            seed=seed,
            max_bytes=4096,
            n_bits=7,
            conf_established=0.45,
            tau_skip=0.10,
            min_flip_count=5,
            flip_ratio=0.30,
            min_sources=2,
            rev_grace_steps=60,
            bytes_per_sample=0,
            store_samples=False,
        )
    else:
        opt_adam = _build_adam_opt(1.0, lr)

    bp_calls = 0
    total_steps = 0
    skipped_steps = 0
    wall_t = 0.0
    ema_loss = 3.0
    consolidated = False
    rev_stabilized = False

    def _run_phase(tbl: torch.Tensor, steps: int, src_fn: Callable[[int], str], is_burst: bool, is_init: bool) -> int:
        nonlocal opt_adam, bp_calls, total_steps, skipped_steps, wall_t, ema_loss, consolidated, rev_stabilized
        tb_dev = tbl.to(device)
        flips_phase = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        q_seq_buf_x: List[torch.Tensor] = []
        q_seq_buf_y: List[torch.Tensor] = []
        q_seq_buf_m: List[torch.Tensor] = []

        for st in range(steps):
            total_steps += 1
            xi = torch.zeros(bs, 16, dtype=torch.long, device=device)
            xi[:, 0] = torch.randint(0, ALPHABET_SIZE, (bs,), device=device, generator=gen)
            for t in range(1, 16):
                xi[:, t] = torch.multinomial(P[xi[:, t - 1]], num_samples=1, generator=gen).squeeze(-1)
            xt = xi + ALPHABET_START
            yf = torch.cat([xt[:, :1], tb_dev[xi[:, :-1], xi[:, 1:]] + ALPHABET_START], dim=1)
            src = src_fn(st)

            if is_spgc:
                assert gate is not None and opt_spgc is not None
                if is_init:
                    prog_i = max(0.0, (st - int(0.65 * steps)) / max(1, steps - 1 - int(0.65 * steps)))
                    dec_i = (5e-4 / 0.012) + (1.0 - 5e-4 / 0.012) * 0.5 * (1.0 + math.cos(math.pi * prog_i))
                    for pg in opt_spgc.param_groups:
                        pg["lr"] = pg["base_lr"] * dec_i

                do_screen = (not is_init) or (st >= 900 and ema_loss < 0.15)
                if do_screen:
                    def _scr():
                        model.eval()
                        with torch.no_grad():
                            return model(xt, update_stats=False)

                    out_s = flops.record(f"scr_{xt.size(0)}", _scr)
                    lg_s = out_s["output"].reshape(bs, 16, -1)[:, 1:, :]
                    z_s = out_s["latent"][:, :-1, :].reshape(-1, 64)
                    y_s = yf[:, 1:].reshape(-1)

                    if not is_init:
                        x_tok = xt.unsqueeze(1).expand(-1, 15, -1).reshape(-1, 16)
                        x_act_tok, _, fl_h, act_mask = gate.triage_batch(
                            x_tok,
                            y_s,
                            lg_s.reshape(-1, 256),
                            z_s,
                            source_id=src,
                            allow_core_write=False,
                            skip_familiar=True,
                            enable_quarantine=True,
                            return_active_mask=True,
                        )
                        flips_phase += fl_h
                        if x_act_tok is None or not bool(act_mask.any().item()):
                            skipped_steps += 1
                            continue
                        if fl_h > 0:
                            opt_spgc.temper_precision(roles=("fast",), factor=0.25)
                        tok_active = act_mask.view(bs, 15)
                    else:
                        probs = F.softmax(lg_s.reshape(-1, 256).detach(), dim=-1)
                        confs, preds = probs.max(dim=-1)
                        ces = F.cross_entropy(lg_s.reshape(-1, 256).detach(), y_s, reduction="none")
                        tok_active = (~((preds == y_s) & (ces <= 0.08) & (confs >= 0.72))).view(bs, 15)
                        # Drop fully trivial batches cleanly when residual surprise rate is negligible
                        if float(tok_active.float().mean().item()) < 0.015:
                            skipped_steps += 1
                            continue

                    seq_active = tok_active.any(dim=1)
                    if not bool(seq_active.any().item()):
                        skipped_steps += 1
                        continue
                    xt_sel = xt[seq_active]
                    yf_sel = yf[seq_active]
                    mk_sel = tok_active[seq_active]

                    # Accumulate sparse residual sequences until >= 24 sequences ready (avoids tiny batch launches)
                    n_buf = sum(t_x.size(0) for t_x in q_seq_buf_x)
                    is_end = (st == steps - 1) and (not is_burst)
                    if (n_buf + xt_sel.size(0) < 24) and (not is_end) and (not is_burst or st < steps - 1):
                        q_seq_buf_x.append(xt_sel)
                        q_seq_buf_y.append(yf_sel)
                        q_seq_buf_m.append(mk_sel)
                        skipped_steps += 1
                        continue
                    if q_seq_buf_x:
                        xt_u = torch.cat(q_seq_buf_x + [xt_sel], dim=0)[:64]
                        yf_u = torch.cat(q_seq_buf_y + [yf_sel], dim=0)[:64]
                        mask_u = torch.cat(q_seq_buf_m + [mk_sel], dim=0)[:64]
                        q_seq_buf_x.clear()
                        q_seq_buf_y.clear()
                        q_seq_buf_m.clear()
                    else:
                        xt_u, yf_u, mask_u = xt_sel, yf_sel, mk_sel
                else:
                    xt_u, yf_u = xt, yf
                    mask_u = torch.ones(xt_u.size(0), 15, dtype=torch.bool, device=device)

                def _bwd_spgc():
                    model.train()
                    out_m = model(xt_u, update_stats=False)
                    lg = out_m["output"].reshape(xt_u.size(0), 16, -1)[:, 1:, :]
                    ce = F.cross_entropy(lg.reshape(-1, 256), yf_u[:, 1:].reshape(-1), reduction="none")
                    loss = (ce * mask_u.reshape(-1).float()).sum() / mask_u.sum().clamp_min(1)
                    opt_spgc.zero_grad()
                    loss.backward()
                    opt_spgc.step()
                    return float(loss.item()), out_m

                lv, out_m = flops.record(f"bwd_{xt_u.size(0)}", _bwd_spgc)
                bp_calls += 1
                ema_loss = 0.95 * ema_loss + 0.05 * lv
                if is_init and st >= 650 and st % 4 == 0:
                    x_tok_i = xt_u.unsqueeze(1).expand(-1, 15, -1).reshape(-1, 16)
                    gate.triage_batch(
                        x_tok_i[:256],
                        yf_u[:, 1:].reshape(-1)[:256],
                        out_m["output"].reshape(xt_u.size(0), 16, -1)[:, 1:, :].reshape(-1, 256)[:256],
                        out_m["latent"][:, :-1, :].reshape(-1, 64)[:256],
                        source_id=src,
                        allow_core_write=False,
                        skip_familiar=False,
                        enable_quarantine=False,
                    )
            else:
                prev_flips = upk.flips_executed if upk is not None else 0
                if upk is not None:
                    gy, um = upk.gate_targets(xt, yf, source_id=src)
                    lm = um[:, 1:]
                    if not bool(lm.any().item()) or float(lm.float().mean().item()) < 0.25:
                        skipped_steps += 1
                        continue
                    if consolidated and (not rev_stabilized) and (upk.flips_executed > prev_flips):
                        opt_adam = _build_adam_opt(bb_scale=0.05, phase_lr=lr)
                        rev_stabilized = True
                else:
                    gy = yf
                    lm = torch.ones(bs, 15, dtype=torch.bool, device=device)

                lr_lo = 5e-4 if not is_burst else 5e-3
                d_start = 0.70 if is_init else (1.0 if is_burst else 0.65)
                prog = max(0.0, (st - int(d_start * steps)) / max(1, steps - 1 - int(d_start * steps)))
                dec = (lr_lo / lr) + (1.0 - lr_lo / lr) * 0.5 * (1.0 + math.cos(math.pi * prog))
                assert opt_adam is not None
                for pg in opt_adam.param_groups:
                    pg["lr"] = pg["base_lr"] * dec

                def _bwd_adam():
                    model.train()
                    out_m = model(xt, update_stats=False)
                    lg = out_m["output"].reshape(bs, 16, -1)[:, 1:, :]
                    ce = F.cross_entropy(lg.reshape(-1, 256), gy[:, 1:].reshape(-1), reduction="none")
                    loss = (ce * lm.reshape(-1).float()).sum() / lm.sum().clamp_min(1)
                    opt_adam.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt_adam.step()
                    return float(loss.item())

                lv = flops.record(f"bwd_{bs}", _bwd_adam)
                bp_calls += 1
                ema_loss = 0.95 * ema_loss + 0.05 * lv
                if is_init and st >= 1500 and ema_loss < 0.05:
                    consolidated = True

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_t += time.perf_counter() - t0
        return flips_phase

    # Phase 1: Init
    _run_phase(init_t, init_steps, lambda s: f"init_src_{s % 2}", is_burst=False, is_init=True)
    if is_spgc and opt_spgc is not None:
        opt_spgc.set_mode("consolidate", snapshot_anchor=False, new_weight_decay=0.05)

    i_s, _ = _eval_split(model, upk, init_t, seen_pairs, seed * 100 + 1, device)
    i_e, _ = _eval_split(model, upk, init_t, exc_pairs, seed * 100 + 2, device)
    i_h, i_nll = _eval_split(model, upk, init_t, held_pairs, seed * 100 + 3, device)

    # Phase 2: 5-step single-source burst
    f0 = upk.flips_executed if upk is not None else 0
    fl_spgc = _run_phase(burst_t, 5, lambda s: "noisy_single", is_burst=True, is_init=False)
    b_flips = float(fl_spgc if is_spgc else ((upk.flips_executed - f0) if upk is not None else 0))
    b_s, _ = _eval_split(model, upk, init_t, seen_pairs, seed * 100 + 1, device)
    b_e, _ = _eval_split(model, upk, init_t, exc_pairs, seed * 100 + 2, device)
    b_h, _ = _eval_split(model, upk, init_t, held_pairs, seed * 100 + 3, device)

    # Phase 3: Persistent multi-source revision
    _run_phase(rev_t, rev_steps, lambda s: f"rev_src_{s % 3}", is_burst=False, is_init=False)
    r_s, _ = _eval_split(model, upk, rev_t, seen_pairs, seed * 100 + 4, device)
    r_e, _ = _eval_split(model, upk, rev_t, exc_pairs, seed * 100 + 5, device)
    r_h, r_nll = _eval_split(model, upk, rev_t, held_pairs, seed * 100 + 6, device)

    sb = gate.storage_bytes() if gate is not None else (upk.memory_footprint_bytes() if upk is not None else 0)
    return {
        "init_seen": i_s, "init_exc": i_e, "init_heldout": i_h, "init_nll": i_nll,
        "burst_flips": b_flips, "burst_seen": b_s, "burst_exc": b_e, "burst_heldout": b_h,
        "rev_seen": r_s, "rev_exc": r_e, "rev_heldout": r_h, "rev_nll": r_nll,
        "flops_total": float(flops.total()),
        "bp_calls": float(bp_calls),
        "skip_rate": float(skipped_steps / max(1, total_steps)),
        "wall_clock_sec": float(wall_t),
        "storage_bytes": float(sb),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify SPGC on RealSelfStabilizingDMoE (Stage 3B-Mem)")
    parser.add_argument("--mode", choices=["tune", "test"], default="tune")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_name", default="real_dmoe_spgc_v1.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    seeds = DEV_SEEDS if args.mode == "tune" else TEST_SEEDS
    arms = ["RealDMoE_AdamW_Base", "RealDMoE_Stage3_Ref", "RealDMoE_SPGC"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}
    for arm in arms:
        rows = [run_real_dmoe_seed(arm, s, device) for s in seeds]
        agg = {}
        for k in rows[0]:
            vals = [r[k] for r in rows]
            mean = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) if len(vals) > 1 else 0.0
            agg[k] = {"mean": mean, "std": sd, "ci95_half": 1.96 * sd / math.sqrt(len(vals)), "per_seed": vals}
        summary[arm] = agg
        print(
            f"[{arm:<22}] init_ho={agg['init_heldout']['mean']:.3f} burst_s={agg['burst_seen']['mean']:.3f} "
            f"flips={agg['burst_flips']['mean']:.2f} rev_s={agg['rev_seen']['mean']:.3f} "
            f"rev_e={agg['rev_exc']['mean']:.3f} rev_ho={agg['rev_heldout']['mean']:.3f} "
            f"nll={agg['rev_nll']['mean']:.3f} fl={agg['flops_total']['mean']:.2e} "
            f"bp={agg['bp_calls']['mean']:.0f} t={agg['wall_clock_sec']['mean']:.2f}s B={agg['storage_bytes']['mean']:.0f}",
            flush=True,
        )

    out_path = RESULTS_DIR / args.out_name
    meta = {
        "exp_id": f"real_dmoe_spgc_{args.mode}",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, check=False).stdout.strip() or "UNKNOWN",
        "seeds": [int(s) for s in seeds],
        "config": {"init_steps": 2200, "rev_steps": 800, "d_model": 64, "num_heads": 4},
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.write_text(json.dumps({"_meta": meta, "mode": args.mode, "seeds": seeds, "arms": summary}, indent=2), encoding="utf-8")
    print(f"Saved RealSelfStabilizingDMoE verification to {out_path}")


if __name__ == "__main__":
    main()
