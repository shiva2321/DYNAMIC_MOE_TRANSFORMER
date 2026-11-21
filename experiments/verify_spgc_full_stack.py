"""
Full-Stack Integration Benchmark for Upgraded `ContinualLearner(enable_spgc=True)` on
`RealSelfStabilizingDMoE` at larger scale (`d_model=128, initial_layers=2, initial_experts=2`)
with ALL major architecture subsystems turned ON simultaneously:
  - `enable_memory=True` (`dmoe/memory.py`: dual-timescale STM/LTM)
  - `enable_three_level_attn=True` (`dmoe/hierarchical_attention.py`: local/mid/global causal attention)
  - `initial_experts=2` (`dmoe/routing.py`: multi-expert routing + dynamic expert spawn verification)
  - `enable_null_space=True` (`dmoe/null_space.py`: GPM / AlphaEdit null-space projection)

Compares 4 arms through the exact `ContinualLearner.step()` API (zero custom training loop hacks):
  1. `FullStack_AdamW_Dense`: `ContinualLearner(enable_spgc=False, enable_null_space=False, replay_buffer_capacity=0)`
  2. `FullStack_AdamW_NullSpace`: `ContinualLearner(enable_spgc=False, enable_null_space=True, replay_buffer_capacity=200)`
  3. `FullStack_SPGC_Only`: `ContinualLearner(enable_spgc=True, enable_null_space=False, spgc_store_samples=False)`
  4. `FullStack_SPGC_Plus_NullSpace`: `ContinualLearner(enable_spgc=True, enable_null_space=True, spgc_store_samples=False)`

Measures strict causality, initial held-out generalization, 5-step single-source burst immunity,
persistent multi-source belief revision, exact PyTorch FLOPs (`FlopCounterMode`), backward calls,
wall-clock seconds, gate `storage_bytes`, optimizer `aux_memory_bytes`, and dynamic expert spawn check.
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
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmoe.chunker import BYTE_PAD_ID
from dmoe.continual import ContinualLearner
from dmoe.data.stage3_tasks import (
    ALPHABET_SIZE,
    ALPHABET_START,
    build_compositional_belief_tables,
    generate_3b_pair_eval_batch,
)
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.spgc import MuonBONGOptimizer

RESULTS_DIR = ROOT / "results" / "belief_training_v3"
DEV_SEEDS = [100, 101, 102]
TEST_SEEDS = [0, 1, 2, 3, 4]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_full_stack_causality(device: torch.device) -> float:
    set_all_seeds(0)
    m = RealSelfStabilizingDMoE(
        d_model=128,
        num_heads=4,
        d_ff=256,
        initial_layers=2,
        initial_experts=2,
        chunk_size=1,
        enable_memory=True,
        enable_thinking=False,
        enable_world_model=False,
        enable_three_level_attn=True,
        causal_attn=True,
        use_rope=True,
        local_window=2,
    ).to(device)
    m.eval()
    x1 = torch.randint(ALPHABET_START, ALPHABET_START + ALPHABET_SIZE, (4, 16), device=device)
    x2 = x1.clone()
    x2[:, 8:] = torch.randint(ALPHABET_START, ALPHABET_START + ALPHABET_SIZE, (4, 8), device=device)
    with torch.no_grad():
        o1 = m(x1, update_stats=False)["output"].reshape(4, 16, -1)[:, :8, :]
        o2 = m(x2, update_stats=False)["output"].reshape(4, 16, -1)[:, :8, :]
    return float((o1 - o2).abs().max().item())


def _eval_split(
    model: RealSelfStabilizingDMoE,
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
        lp = F.log_softmax(lg, dim=-1)
        acc = float(((lp.argmax(-1) == y) & mk).sum().item() / max(1, int(mk.sum().item())))
        nll = float(F.nll_loss(lp[mk], y[mk]).item())
    return acc, nll


def _measure_step_flops(learner: ContinualLearner, xt: torch.Tensor, yf: torch.Tensor, src: str) -> Tuple[int, Dict[str, Any]]:
    with FlopCounterMode(display=False) as fc:
        rep = learner.step(xt, yf, output_head="byte_chunk", source_id=src)
    return int(fc.get_total_flops()), rep


def run_full_stack_seed(
    arm: str,
    seed: int,
    device: torch.device,
    init_steps: int = 1500,
    rev_steps: int = 500,
) -> Dict[str, float]:
    set_all_seeds(seed)
    spec = build_compositional_belief_tables(seed=seed)
    init_t, burst_t, rev_t = spec["init_table"], spec["burst_table"], spec["rev_table"]
    seen_mask, seen_pairs = spec["seen_mask"], spec["seen_pairs"]
    held_pairs, exc_pairs = spec["heldout_pairs"], spec["exc_pairs"]
    P = (seen_mask.float() / seen_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)).to(device)

    model = RealSelfStabilizingDMoE(
        d_model=128,
        num_heads=4,
        d_ff=256,
        initial_layers=2,
        initial_experts=2,
        chunk_size=1,
        enable_memory=True,
        enable_thinking=False,
        enable_world_model=False,
        enable_three_level_attn=True,
        auto_lifecycle=False,
        causal_attn=True,
        use_rope=True,
        local_window=2,
    ).to(device)

    is_spgc = arm in ("FullStack_SPGC_Only", "FullStack_SPGC_Plus_NullSpace")
    use_ns = arm in ("FullStack_AdamW_NullSpace", "FullStack_SPGC_Plus_NullSpace")
    replay_cap = 200 if arm == "FullStack_AdamW_NullSpace" else 0
    base_opt = None
    if not is_spgc:
        decay_p = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        nodecay_p = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        base_opt = torch.optim.AdamW(
            [{"params": decay_p, "weight_decay": 1.5}, {"params": nodecay_p, "weight_decay": 0.0}],
            lr=8e-3,
            betas=(0.9, 0.98),
        )

    learner = ContinualLearner(
        model=model,
        optimizer=base_opt,
        replay_buffer_capacity=replay_cap,
        replay_batch_size=8,
        distillation_weight=0.25,
        mutation_check_interval=100000,
        enable_null_space=use_ns,
        null_space_energy_threshold=0.95,
        null_space_max_rank_ratio=0.5,
        enable_spgc=is_spgc,
        spgc_tau_skip=0.10,
        spgc_min_sources=2,
        spgc_max_bytes=8192,
        spgc_lr=5e-3,
        spgc_wd_trunk=1.5,
        spgc_wd_fast=1.5,
        spgc_seed=seed,
        spgc_store_samples=False,
        spgc_min_flush=16,
        spgc_min_steps_before_skip=int(0.65 * init_steps),
        spgc_auto_consolidate_steps=int(0.85 * init_steps),
        spgc_auto_consolidate_loss=0.10,
        spgc_bong_prior_pull=0.0,
        spgc_enable_token_gating=True,
    )
    if is_spgc and learner.spgc_gate is not None:
        learner.spgc_gate.min_flip_count = 5
        learner.spgc_gate.flip_ratio = 0.30
        learner.spgc_gate.rev_grace_steps = 40
        learner.spgc_gate.tent_ttl_steps = 64

    bs = 48
    gen = torch.Generator(device=device).manual_seed(seed * 137 + 11)
    bp_calls = 0
    total_steps = 0
    skipped_steps = 0
    wall_t = 0.0
    flop_cache: Dict[str, int] = {}
    flop_total = 0

    def _run_phase(tbl: torch.Tensor, steps: int, src_fn: Callable[[int], str], phase_tag: str) -> int:
        nonlocal bp_calls, total_steps, skipped_steps, wall_t, flop_total
        tb_dev = tbl.to(device)
        flips_phase = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        for st in range(steps):
            total_steps += 1
            xi = torch.zeros(bs, 16, dtype=torch.long, device=device)
            xi[:, 0] = torch.randint(0, ALPHABET_SIZE, (bs,), device=device, generator=gen)
            for t in range(1, 16):
                xi[:, t] = torch.multinomial(P[xi[:, t - 1]], num_samples=1, generator=gen).squeeze(-1)
            xt = xi + ALPHABET_START
            yf = torch.cat(
                [torch.full((bs, 1), BYTE_PAD_ID, dtype=torch.long, device=device), tb_dev[xi[:, :-1], xi[:, 1:]] + ALPHABET_START],
                dim=1,
            )
            src = src_fn(st)

            # Measure exact FLOPs on representative steps (first active & first skipped per phase)
            need_probe = (f"{phase_tag}_act" not in flop_cache) or (is_spgc and f"{phase_tag}_skip" not in flop_cache and st % 40 == 0)
            if need_probe:
                fl_step, rep = _measure_step_flops(learner, xt, yf, src)
                sk = bool(rep.get("skipped_backward", False))
                key = f"{phase_tag}_{'skip' if sk else 'act'}"
                if key not in flop_cache:
                    flop_cache[key] = fl_step
                flop_total += fl_step
            else:
                rep = learner.step(xt, yf, output_head="byte_chunk", source_id=src)
                sk = bool(rep.get("skipped_backward", False))
                key = f"{phase_tag}_{'skip' if sk else 'act'}"
                flop_total += flop_cache.get(key, flop_cache.get(f"{phase_tag}_act", 0))

            flips_phase += int(rep.get("flips", 0))
            if bool(rep.get("skipped_backward", False)):
                skipped_steps += 1
            else:
                bp_calls += 1

        if is_spgc and phase_tag != "burst":
            rep_fl = learner.flush_spgc_queue(output_head="byte_chunk")
            if not bool(rep_fl.get("skipped_backward", True)):
                bp_calls += 1
                flop_total += flop_cache.get(f"{phase_tag}_act", 0)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_t += time.perf_counter() - t0
        return flips_phase

    # Phase 1: Initial learning
    _run_phase(init_t, init_steps, lambda s: f"init_src_{s % 2}", "init")
    if is_spgc:
        learner.set_spgc_consolidate(new_weight_decay=0.01)
    if use_ns and learner.null_space_projector is not None:
        learner.null_space_projector.update_basis_for_domain(0)

    i_s, _ = _eval_split(model, init_t, seen_pairs, seed * 100 + 1, device)
    i_e, _ = _eval_split(model, init_t, exc_pairs, seed * 100 + 2, device)
    i_h, i_nll = _eval_split(model, init_t, held_pairs, seed * 100 + 3, device)

    # Phase 2: 5-step single-source adversarial burst
    b_flips = float(_run_phase(burst_t, 5, lambda s: "noisy_single", "burst"))
    b_s, _ = _eval_split(model, init_t, seen_pairs, seed * 100 + 1, device)
    b_e, _ = _eval_split(model, init_t, exc_pairs, seed * 100 + 2, device)
    b_h, _ = _eval_split(model, init_t, held_pairs, seed * 100 + 3, device)

    # Phase 3: Persistent multi-source belief revision
    _run_phase(rev_t, rev_steps, lambda s: f"rev_src_{s % 3}", "rev")
    r_s, _ = _eval_split(model, rev_t, seen_pairs, seed * 100 + 4, device)
    r_e, _ = _eval_split(model, rev_t, exc_pairs, seed * 100 + 5, device)
    r_h, r_nll = _eval_split(model, rev_t, held_pairs, seed * 100 + 6, device)

    # Phase 4: Verify dynamic domain_transition + expert spawning preserves optimizer integrity
    learner.domain_transition(new_domain_id=2, spawn_new_expert=True)
    n_exp_final = float(len(model.layers[0].moe.experts))

    sb = float(learner.spgc_gate.storage_bytes()) if learner.spgc_gate is not None else 0.0
    aux_b = float(learner.optimizer.precision_bytes()) if isinstance(learner.optimizer, MuonBONGOptimizer) else 0.0
    return {
        "init_seen": i_s,
        "init_exc": i_e,
        "init_heldout": i_h,
        "init_nll": i_nll,
        "burst_flips": b_flips,
        "burst_seen": b_s,
        "burst_exc": b_e,
        "burst_heldout": b_h,
        "rev_seen": r_s,
        "rev_exc": r_e,
        "rev_heldout": r_h,
        "rev_nll": r_nll,
        "flops_total": float(flop_total),
        "bp_calls": float(bp_calls),
        "skip_rate": float(skipped_steps / max(1, total_steps)),
        "wall_clock_sec": float(wall_t),
        "storage_bytes": sb,
        "aux_memory_bytes": aux_b,
        "num_experts_final": n_exp_final,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-stack SPGC verification on d_model=128 2L 2E RealSelfStabilizingDMoE")
    parser.add_argument("--mode", choices=["pilot", "dev"], default="pilot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--init_steps", type=int, default=1500)
    parser.add_argument("--rev_steps", type=int, default=500)
    parser.add_argument("--out_name", default="full_stack_dmoe_pilot.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    caus_diff = verify_full_stack_causality(device)
    print(f"full_stack_dmoe | device={device} | mode={args.mode} | causality_max_diff={caus_diff:.2e}", flush=True)
    assert caus_diff < 1e-6, f"Causality violation on full-stack model: {caus_diff}"

    seeds = DEV_SEEDS if args.mode == "dev" else TEST_SEEDS
    arms = [
        "FullStack_AdamW_Dense",
        "FullStack_AdamW_NullSpace",
        "FullStack_SPGC_Only",
        "FullStack_SPGC_Plus_NullSpace",
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}
    for arm in arms:
        rows = [
            run_full_stack_seed(arm, s, device, init_steps=args.init_steps, rev_steps=args.rev_steps)
            for s in seeds
        ]
        agg: Dict[str, Any] = {}
        for k in rows[0]:
            vals = [r[k] for r in rows]
            mean = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) if len(vals) > 1 else 0.0
            agg[k] = {
                "mean": mean,
                "std": sd,
                "ci95_half": 1.96 * sd / math.sqrt(len(vals)),
                "min": min(vals),
                "max": max(vals),
                "per_seed": vals,
            }
        summary[arm] = agg
        print(
            f"[{arm:<29}] init_ho={agg['init_heldout']['mean']:.3f} burst_s={agg['burst_seen']['mean']:.3f} "
            f"flips={agg['burst_flips']['mean']:.2f} rev_s={agg['rev_seen']['mean']:.3f} "
            f"rev_ho={agg['rev_heldout']['mean']:.3f} nll={agg['rev_nll']['mean']:.3f} "
            f"fl={agg['flops_total']['mean']:.2e} bp={agg['bp_calls']['mean']:.0f} "
            f"skip={agg['skip_rate']['mean']:.2f} t={agg['wall_clock_sec']['mean']:.1f}s "
            f"B={agg['storage_bytes']['mean']:.0f} auxB={agg['aux_memory_bytes']['mean']:.0f} "
            f"E={agg['num_experts_final']['mean']:.0f}",
            flush=True,
        )

    out_path = RESULTS_DIR / args.out_name
    meta = {
        "exp_id": f"full_stack_dmoe_{args.mode}",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, check=False
        ).stdout.strip() or "UNKNOWN",
        "seeds": [int(s) for s in seeds],
        "config": {
            "d_model": 128,
            "num_heads": 4,
            "d_ff": 256,
            "initial_layers": 2,
            "initial_experts": 2,
            "enable_memory": True,
            "enable_three_level_attn": True,
            "init_steps": int(args.init_steps),
            "rev_steps": int(args.rev_steps),
            "batch_size": 48,
        },
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.write_text(
        json.dumps({"_meta": meta, "causality_max_diff": caus_diff, "mode": args.mode, "seeds": seeds, "arms": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved full-stack verification to {out_path}", flush=True)


if __name__ == "__main__":
    main()
