"""
Benchmark Runner for Round Z1: Spherical Residual Stream (nGPT Family).

Evaluates:
- base (RoPE + pre-RMSNorm + AdamW)
- base_qknorm (base + QK-norm)
- base_scaled (isolating control: base + learned scales matching nGPT, no normalizations)
- ngpt (unit-norm stream + unit-norm weights + learned scales + weight decay 0.0)
- ngpt_stream_only (unit-norm stream, unconstrained weights)
- base_muon (base + Muon on 2D weights)
- ngpt_muon (ngpt + Muon on 2D weights + weight renormalization)
- ngpt_salience (ngpt + explicit scalar salience channel; Atlas P-1)

Follows Track Z protocol:
1. Verify PREREG file precedes results.
2. Per-arm hyperparameter tuning on dev seeds (100, 101) with edge detection.
3. Test evaluation across seeds 0-9 (10 seeds) for 1000 steps.
4. Paired speedup and statistical tests against base and base_scaled.
5. Geometry health diagnostics.
6. Write results.json and render tables.txt; diff validate.
"""

import os
import sys
import time
import json
import math
import random
import contextlib
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from geomlab.protocol import (
    check_prereg,
    paired_stats,
    bonferroni,
    ci95,
    write_results,
    render_tables,
    validate_report,
)
from geomlab.diagnostics import (
    effective_rank,
    isotropy,
    intrinsic_dim_twonn,
    hubness_skew,
    norm_stats,
)
from geomlab.spherical_stream import (
    SphericalStreamLM,
    BaselineLM,
    SphericalStreamWithSalienceLM,
    normalize_weights_,
    compute_speedup,
    l2_norm,
)
from ishmn.muon import Muon


# -------------------------------------------------------------------------------------------------
# 1. Corpus Loading & Tokenization (Char-level Python stdlib)
# -------------------------------------------------------------------------------------------------

def load_stdlib_splits() -> Tuple[str, str, Dict[str, int], Dict[int, str]]:
    """Loads Python stdlib files for train and held-out validation."""
    lib_dir = Path(sys.base_prefix) / "Lib"
    train_files = ["ast.py", "dis.py", "inspect.py", "tokenize.py", "difflib.py", "pydoc.py", "tarfile.py", "zipfile.py"]
    val_files = ["heapq.py", "calendar.py", "bisect.py", "queue.py"]

    train_text = ""
    for f in train_files:
        p = lib_dir / f
        if p.exists():
            train_text += p.read_text(encoding="utf-8", errors="ignore") + "\n"

    val_text = ""
    for f in val_files:
        p = lib_dir / f
        if p.exists():
            val_text += p.read_text(encoding="utf-8", errors="ignore") + "\n"

    # Vocabulary: printable ASCII (32..126) + newline + tab
    valid_chars = set(chr(c) for c in range(32, 127)) | {"\n", "\t"}
    chars = sorted(list(valid_chars))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}

    return train_text, val_text, char2idx, idx2char


def build_batches(text: str, char2idx: Dict[str, int], seq_len: int, batch_size: int, n_batches: int, seed: int = 42):
    """Generates fixed evaluation or training batches."""
    rng = np.random.default_rng(seed)
    indices = [char2idx[c] for c in text if c in char2idx]
    arr = np.array(indices, dtype=np.int64)
    max_start = len(arr) - seq_len - 1

    batches = []
    for _ in range(n_batches):
        starts = rng.integers(0, max_start, size=batch_size)
        x = np.stack([arr[s : s + seq_len] for s in starts])
        y = np.stack([arr[s + 1 : s + seq_len + 1] for s in starts])
        batches.append((torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)))
    return batches


# -------------------------------------------------------------------------------------------------
# 2. Synthetic MQAR Recall Generator
# -------------------------------------------------------------------------------------------------

def evaluate_mqar_recall(model: nn.Module, vocab_size: int, device: torch.device, n_pairs: int = 4, seq_len: int = 64, n_eval: int = 200, seed: int = 777, use_bf16: bool = True) -> float:
    """
    Evaluates multi-query associative recall on a small synthetic benchmark.
    Tests model's ability to retrieve associated values given query keys.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    correct = 0
    total = 0

    key_vocab = list(range(10, 40))
    val_vocab = list(range(40, 70))
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_bf16 and device.type == "cuda") else contextlib.nullcontext()

    with torch.no_grad():
        for _ in range(n_eval):
            # Select unique keys and values
            keys = rng.choice(key_vocab, size=n_pairs, replace=False)
            vals = rng.choice(val_vocab, size=n_pairs, replace=False)
            mapping = dict(zip(keys, vals))

            # Build sequence: [k1, v1, k2, v2, ..., query_k]
            tokens = []
            for k, v in zip(keys, vals):
                tokens.extend([int(k), int(v)])
            # Pad with 0s
            pad_len = seq_len - (2 * n_pairs + 1)
            tokens.extend([0] * pad_len)
            target_key = int(rng.choice(keys))
            tokens.append(target_key)
            expected_val = mapping[target_key]

            x = torch.tensor([tokens], dtype=torch.long, device=device)
            with autocast_ctx:
                logits = model(x)
            pred = int(logits[0, -1].argmax().item())
            if pred == expected_val:
                correct += 1
            total += 1

    return float(correct / total) if total > 0 else 0.0


# -------------------------------------------------------------------------------------------------
# 3. Model & Optimizer Factory
# -------------------------------------------------------------------------------------------------

def create_model(arm: str, vocab_size: int, d_model: int = 256, n_layers: int = 6, n_heads: int = 4, d_mlp: int = 768, max_seq_len: int = 256) -> nn.Module:
    """Instantiates model architecture according to arm name."""
    if arm == "base":
        return BaselineLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp, max_seq_len=max_seq_len, mode="base")
    elif arm == "base_qknorm":
        return BaselineLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp, max_seq_len=max_seq_len, mode="base_qknorm")
    elif arm == "base_scaled":
        return BaselineLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp, max_seq_len=max_seq_len, mode="base_scaled")
    elif arm == "ngpt":
        return SphericalStreamLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp, max_seq_len=max_seq_len, mode="ngpt")
    elif arm == "ngpt_stream_only":
        return SphericalStreamLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp, max_seq_len=max_seq_len, mode="ngpt_stream_only")
    elif arm == "base_muon":
        return BaselineLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp, max_seq_len=max_seq_len, mode="base")
    elif arm == "ngpt_muon":
        return SphericalStreamLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp, max_seq_len=max_seq_len, mode="ngpt")
    elif arm == "ngpt_salience":
        return SphericalStreamWithSalienceLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp, max_seq_len=max_seq_len)
    else:
        raise ValueError(f"Unknown arm: {arm}")


def create_optimizer(model: nn.Module, arm: str, lr: float, weight_decay: float = 0.0) -> Tuple[Any, Optional[Any]]:
    """Creates optimizer for the arm. Returns (primary_opt, aux_opt_if_muon)."""
    if "muon" in arm:
        # Separate 2D linear weights (>128 elements) for Muon, and 1D / scales / embeddings for AdamW
        muon_params = []
        adamw_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2 and p.numel() > 128 and "embed" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        opt_muon = Muon(muon_params, lr=lr, momentum=0.95, weight_decay=weight_decay)
        opt_adamw = torch.optim.AdamW(adamw_params, lr=1e-3, weight_decay=0.0)
        return opt_muon, opt_adamw
    else:
        # Standard AdamW
        wd = 0.0 if arm in ("ngpt", "ngpt_muon", "ngpt_salience") else weight_decay
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        return opt, None


def optimizer_step(model: nn.Module, arm: str, opt_primary: Any, opt_aux: Optional[Any] = None):
    """Executes optimizer step and applies post-step weight normalization if required."""
    opt_primary.step()
    if opt_aux is not None:
        opt_aux.step()

    if arm in ("ngpt", "ngpt_muon", "ngpt_salience"):
        normalize_weights_(model)


# -------------------------------------------------------------------------------------------------
# 4. Training & Evaluation Engine
# -------------------------------------------------------------------------------------------------

def evaluate_model(model: nn.Module, val_batches: List[Tuple[torch.Tensor, torch.Tensor]], device: torch.device, use_bf16: bool = True) -> float:
    """Computes mean cross-entropy loss over fixed validation batches."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_bf16 and device.type == "cuda") else contextlib.nullcontext()
    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device), y.to(device)
            with autocast_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()
    return total_loss / total_tokens if total_tokens > 0 else 0.0


def train_and_eval(
    arm: str,
    lr: float,
    weight_decay: float,
    train_text: str,
    char2idx: Dict[str, int],
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    seed: int,
    total_steps: int = 1000,
    eval_interval: int = 100,
    batch_size: int = 32,
    seq_len: int = 256,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Runs training for total_steps and logs validation checkpoints."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    vocab_size = len(char2idx)
    model = create_model(arm, vocab_size=vocab_size, max_seq_len=seq_len).to(device)
    if arm in ("ngpt", "ngpt_muon", "ngpt_salience"):
        normalize_weights_(model)

    opt_primary, opt_aux = create_optimizer(model, arm, lr=lr, weight_decay=weight_decay)

    # Pre-build train batch stream for speed and exact reproducibility
    train_batches = build_batches(train_text, char2idx, seq_len=seq_len, batch_size=batch_size, n_batches=total_steps, seed=seed)

    eval_steps = []
    val_losses = []
    step_times = []

    start_time = time.perf_counter()
    model.train()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_bf16 and device.type == "cuda") else contextlib.nullcontext()

    for step in range(1, total_steps + 1):
        x, y = train_batches[step - 1]
        x, y = x.to(device), y.to(device)

        opt_primary.zero_grad()
        if opt_aux is not None:
            opt_aux.zero_grad()

        with autocast_ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[{arm} seed {seed}] Loss diverged at step {step}")
            return {
                "diverged": True,
                "step": step,
                "final_loss": 999.0,
                "eval_steps": eval_steps,
                "val_losses": val_losses,
                "step_times": step_times,
            }

        loss.backward()
        optimizer_step(model, arm, opt_primary, opt_aux)

        if step % eval_interval == 0:
            val_loss = evaluate_model(model, val_batches, device, use_bf16=use_bf16)
            elapsed = time.perf_counter() - start_time
            eval_steps.append(step)
            val_losses.append(val_loss)
            step_times.append(elapsed)
            model.train()

    total_time = time.perf_counter() - start_time

    # Diagnostics on final activations
    model.eval()
    with torch.no_grad():
        sample_x, _ = val_batches[0]
        sample_x = sample_x.to(device)
        if hasattr(model, "forward_with_intermediates"):
            intermediates = model.forward_with_intermediates(sample_x)
            final_h = intermediates[-1]
        elif hasattr(model, "forward_with_salience"):
            _, (final_h, _) = model.forward_with_salience(sample_x)
        else:
            final_h = sample_x

        # Compute geometry diagnostics in float32 for exact SVD and distance metrics
        final_h = final_h.float()
        erank_val = effective_rank(final_h)
        iso_val = isotropy(final_h)
        twonn_val = intrinsic_dim_twonn(final_h)
        hub_val = hubness_skew(final_h)
        nstats = norm_stats(final_h)

    # MQAR recall
    mqar_score = evaluate_mqar_recall(model, vocab_size=vocab_size, device=device, seed=seed)

    return {
        "diverged": False,
        "final_loss": val_losses[-1],
        "eval_steps": eval_steps,
        "val_losses": val_losses,
        "step_times": step_times,
        "total_time": total_time,
        "ms_per_step": (total_time / total_steps) * 1000.0,
        "mqar_recall": mqar_score,
        "effective_rank": erank_val,
        "uniformity_loss": iso_val["uniformity_loss"],
        "mean_cosine": iso_val["mean_pairwise_cosine"],
        "twonn_id": twonn_val,
        "hubness_skew": hub_val,
        "norm_mean": nstats["mean_norm"],
        "norm_std": nstats["std_norm"],
        "norm_max": nstats["max_norm"],
    }


# -------------------------------------------------------------------------------------------------
# 5. Main Execution: Tuning & Test Runs
# -------------------------------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Track Z Round Z1: Spherical Residual Stream Benchmark (nGPT Family)")
    print("=" * 80)

    # 1. PREREG check
    prereg_file = Path("research/geomlab_results/z1/PREREG.md")
    if not prereg_file.exists():
        raise FileNotFoundError(f"PREREG file does not exist: {prereg_file}")
    print(f"PREREG check: PREREG.md exists (mtime {prereg_file.stat().st_mtime})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load corpus
    train_text, val_text, char2idx, idx2char = load_stdlib_splits()
    print(f"Corpus loaded: Train {len(train_text)} chars | Val {len(val_text)} chars | Vocab {len(char2idx)} tokens")

    # 16 fixed held-out validation batches
    val_batches = build_batches(val_text, char2idx, seq_len=256, batch_size=32, n_batches=16, seed=999)

    parser = argparse.ArgumentParser(description="Z1 Spherical Residual Stream Benchmark")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16 mixed precision on CUDA")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false", help="Disable bfloat16")
    parser.add_argument("--triad", action="store_true", default=True, help="Run only Core Decision Triad (base, base_scaled, ngpt)")
    parser.add_argument("--all-arms", dest="triad", action="store_false", help="Run all 8 arms")
    parser.add_argument("--checkpoint", type=str, default="research/geomlab_results/z1/checkpoint_z1_bf16.json", help="Checkpoint file path")
    args, _ = parser.parse_known_args()

    use_bf16 = args.bf16
    checkpoint_file = Path(args.checkpoint)
    print(f"Execution mode: {'bfloat16 mixed precision' if use_bf16 else 'float32 standard'} | Triad mode: {args.triad}")

    if args.triad:
        arms = ["base", "base_scaled", "ngpt"]
    else:
        arms = [
            "base",
            "base_qknorm",
            "base_scaled",
            "ngpt",
            "ngpt_stream_only",
            "base_muon",
            "ngpt_muon",
            "ngpt_salience",
        ]

    dev_seeds = [100, 101]
    test_seeds = list(range(10))

    # Pre-registered hyperparameter grids
    lr_grid_adamw = [1e-4, 3e-4, 1e-3, 3e-3]
    lr_grid_muon = [0.005, 0.01, 0.02, 0.04]
    wd_grid_base = [0.0, 0.01, 0.1]

    # Step 1: Tuning on dev seeds (or resume from checkpoint)
    ckpt_data = {}
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                ckpt_data = json.load(f)
            print(f"Loaded existing checkpoint from {checkpoint_file}")
        except Exception as e:
            print(f"Warning: could not load checkpoint: {e}")
    elif Path("research/geomlab_results/z1/checkpoint_z1.json").exists():
        try:
            with open("research/geomlab_results/z1/checkpoint_z1.json", "r", encoding="utf-8") as f:
                old_ckpt = json.load(f)
                ckpt_data = {
                    "best_configs": {a: old_ckpt["best_configs"][a] for a in arms if a in old_ckpt.get("best_configs", {})},
                    "base_runs": [],
                    "raw_results": {a: [] for a in arms},
                    "peak_vram_mb": 0.0,
                }
            print(f"Initialized from tuned hyperparameters: {list(ckpt_data['best_configs'].keys())}")
        except Exception as e:
            print(f"Warning: could not read baseline checkpoint: {e}")

    best_configs = ckpt_data.get("best_configs", {})

    if best_configs and all(arm in best_configs for arm in arms):
        print("\n--- Hyperparameters loaded from checkpoint (skipping dev seed tuning) ---")
        for arm in arms:
            cfg = best_configs[arm]
            print(f"  {arm} selected: LR = {cfg['lr']}, WD = {cfg['weight_decay']} (dev score: {cfg.get('dev_score', 0.0):.4f})")
    else:
        print("\n--- Tuning Hyperparameters on Dev Seeds (100, 101) ---")
        best_configs = {}

        for arm in arms:
            print(f"Tuning {arm}...")
            is_muon = "muon" in arm
            lr_candidates = lr_grid_muon if is_muon else lr_grid_adamw
            wd_candidates = [0.0] if arm in ("ngpt", "ngpt_muon", "ngpt_salience") else wd_grid_base

            # Phase 1: Tune weight decay for baselines at default LR
            best_wd = 0.0
            if arm not in ("ngpt", "ngpt_muon", "ngpt_salience"):
                def_lr = 0.01 if is_muon else 1e-3
                wd_scores = {}
                for wd in wd_candidates:
                    scores = [
                        train_and_eval(arm, def_lr, wd, train_text, char2idx, val_batches, device, seed, total_steps=200, eval_interval=100)["final_loss"]
                        for seed in dev_seeds
                    ]
                    wd_scores[wd] = float(np.mean(scores))
                best_wd = min(wd_scores, key=wd_scores.get)
                print(f"  {arm} best weight decay: {best_wd}")

            # Phase 2: Sweep LR candidates at best_wd
            lr_scores = {}
            for lr in lr_candidates:
                scores = [
                    train_and_eval(arm, lr, best_wd, train_text, char2idx, val_batches, device, seed, total_steps=200, eval_interval=100)["final_loss"]
                    for seed in dev_seeds
                ]
                lr_scores[lr] = float(np.mean(scores))

            best_lr = min(lr_scores, key=lr_scores.get)
            best_score = lr_scores[best_lr]

            # Phase 3: Edge detection check
            lr_idx = lr_candidates.index(best_lr)
            if lr_idx == 0:
                ext_lr = best_lr / 3.0
                print(f"  [Edge check] Best LR {best_lr} is at lower edge; testing {ext_lr}...")
                ext_scores = [
                    train_and_eval(arm, ext_lr, best_wd, train_text, char2idx, val_batches, device, seed, total_steps=200, eval_interval=100)["final_loss"]
                    for seed in dev_seeds
                ]
                if np.mean(ext_scores) < best_score:
                    best_score = float(np.mean(ext_scores))
                    best_lr = ext_lr
            elif lr_idx == len(lr_candidates) - 1:
                ext_lr = best_lr * 3.0
                print(f"  [Edge check] Best LR {best_lr} is at upper edge; testing {ext_lr}...")
                ext_scores = [
                    train_and_eval(arm, ext_lr, best_wd, train_text, char2idx, val_batches, device, seed, total_steps=200, eval_interval=100)["final_loss"]
                    for seed in dev_seeds
                ]
                if np.mean(ext_scores) < best_score:
                    best_score = float(np.mean(ext_scores))
                    best_lr = ext_lr

            best_configs[arm] = {"lr": best_lr, "weight_decay": best_wd, "dev_score": best_score}
            print(f"  {arm} selected: LR = {best_lr}, WD = {best_wd} (dev score: {best_score:.4f})")

    def _save_checkpoint(best_cfg, b_runs, r_results, peak_vram):
        tmp_path = checkpoint_file.with_suffix(".tmp")
        payload = {
            "best_configs": best_cfg,
            "base_runs": b_runs,
            "raw_results": r_results,
            "peak_vram_mb": peak_vram,
        }
        with open(tmp_path, "w", encoding="utf-8") as f_out:
            json.dump(payload, f_out, indent=2)
        tmp_path.replace(checkpoint_file)

    # Step 2: Test Evaluation on Seeds 0-9
    print("\n--- Evaluating Test Seeds (0..9) ---")
    base_runs = ckpt_data.get("base_runs", [])
    raw_results = ckpt_data.get("raw_results", {arm: [] for arm in arms})
    peak_vram_mb = ckpt_data.get("peak_vram_mb", 0.0)

    # Ensure all arms are in raw_results dict
    for arm in arms:
        if arm not in raw_results:
            raw_results[arm] = []

    # First run base across all test seeds to establish reference target losses
    print("Running base arm to establish baseline target losses...")
    for seed in test_seeds:
        existing = [r for r in raw_results["base"] if r["seed"] == seed]
        if existing and len(base_runs) > seed:
            print(f"  base seed {seed}: loaded from checkpoint (Val Loss = {existing[0]['final_loss']:.4f})")
            continue
        cfg = best_configs["base"]
        res = train_and_eval(
            arm="base",
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            train_text=train_text,
            char2idx=char2idx,
            val_batches=val_batches,
            device=device,
            seed=seed,
            total_steps=1000,
            eval_interval=100,
            use_bf16=use_bf16,
        )
        base_runs.append(res)
        res_entry = {
            "seed": seed,
            "final_loss": res["final_loss"],
            "token_speedup": 1.0,
            "wallclock_speedup": 1.0,
            "ms_per_step": res["ms_per_step"],
            "mqar_recall": res["mqar_recall"],
            "effective_rank": res["effective_rank"],
            "uniformity_loss": res["uniformity_loss"],
            "twonn_id": res["twonn_id"],
            "hubness_skew": res["hubness_skew"],
            "norm_mean": res["norm_mean"],
            "norm_std": res["norm_std"],
        }
        raw_results["base"].append(res_entry)
        if torch.cuda.is_available():
            peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() / (1024 * 1024))
        _save_checkpoint(best_configs, base_runs, raw_results, peak_vram_mb)
        print(f"  base seed {seed}: Val Loss = {res['final_loss']:.4f}, Step Time = {res['ms_per_step']:.1f} ms")

    # Now run remaining arms and compute speedup relative to base on same seed
    for arm in arms:
        if arm == "base":
            continue
        print(f"Running {arm} across test seeds...")
        cfg = best_configs[arm]
        for seed_idx, seed in enumerate(test_seeds):
            existing = [r for r in raw_results[arm] if r["seed"] == seed]
            if existing:
                print(f"  {arm} seed {seed}: loaded from checkpoint (Loss = {existing[0]['final_loss']:.4f})")
                continue
            res = train_and_eval(
                arm=arm,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                train_text=train_text,
                char2idx=char2idx,
                val_batches=val_batches,
                device=device,
                seed=seed,
                total_steps=1000,
                eval_interval=100,
                use_bf16=use_bf16,
            )
            base_res = base_runs[seed_idx]
            sp_res = compute_speedup(
                eval_steps=res["eval_steps"],
                val_losses_base=base_res["val_losses"],
                val_losses_arm=res["val_losses"],
                step_times_base=base_res["step_times"],
                step_times_arm=res["step_times"],
                tokens_per_step=32 * 256,
            )
            res_entry = {
                "seed": seed,
                "final_loss": res["final_loss"],
                "token_speedup": sp_res["token_speedup"],
                "wallclock_speedup": sp_res["wallclock_speedup"],
                "ms_per_step": res["ms_per_step"],
                "mqar_recall": res["mqar_recall"],
                "effective_rank": res["effective_rank"],
                "uniformity_loss": res["uniformity_loss"],
                "twonn_id": res["twonn_id"],
                "hubness_skew": res["hubness_skew"],
                "norm_mean": res["norm_mean"],
                "norm_std": res["norm_std"],
            }
            raw_results[arm].append(res_entry)
            if torch.cuda.is_available():
                peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() / (1024 * 1024))
            _save_checkpoint(best_configs, base_runs, raw_results, peak_vram_mb)
            print(f"  {arm} seed {seed}: Loss = {res['final_loss']:.4f}, Sp_tok = {sp_res['token_speedup']:.2f}x, Sp_wall = {sp_res['wallclock_speedup']:.2f}x")

    # Step 3: Statistical Analysis & Aggregation
    def _metric_stat(vals):
        arr = np.asarray(vals, dtype=float)
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ci_h = ci95(arr)
        return {
            "mean": mean_v,
            "std": std_v,
            "ci_half": ci_h,
            "values": [float(x) for x in vals],
        }

    summary = {}
    for arm in arms:
        runs = raw_results[arm]
        summary[arm] = {
            "config": best_configs[arm],
            "final_loss": _metric_stat([r["final_loss"] for r in runs]),
            "token_speedup": _metric_stat([r["token_speedup"] for r in runs]),
            "wallclock_speedup": _metric_stat([r["wallclock_speedup"] for r in runs]),
            "ms_per_step": _metric_stat([r["ms_per_step"] for r in runs]),
            "mqar_recall": _metric_stat([r["mqar_recall"] for r in runs]),
            "effective_rank": _metric_stat([r["effective_rank"] for r in runs]),
            "uniformity_loss": _metric_stat([r["uniformity_loss"] for r in runs]),
            "twonn_id": _metric_stat([r["twonn_id"] for r in runs]),
            "hubness_skew": _metric_stat([r["hubness_skew"] for r in runs]),
            "norm_mean": _metric_stat([r["norm_mean"] for r in runs]),
            "norm_std": _metric_stat([r["norm_std"] for r in runs]),
        }

    # Paired comparisons vs base
    base_losses = [r["final_loss"] for r in raw_results["base"]]
    paired_vs_base = {}
    for arm in arms:
        if arm == "base":
            continue
        arm_losses = [r["final_loss"] for r in raw_results[arm]]
        p_stats = paired_stats(arm_losses, base_losses)  # diff = arm - base
        paired_vs_base[arm] = p_stats

    # Paired comparison: ngpt vs base_scaled (isolating control)
    scaled_losses = [r["final_loss"] for r in raw_results["base_scaled"]]
    ngpt_losses = [r["final_loss"] for r in raw_results["ngpt"]]
    ngpt_vs_scaled = paired_stats(ngpt_losses, scaled_losses)

    # Bonferroni corrections
    p_vals = [paired_vs_base[a]["p_value_ttest"] for a in paired_vs_base]
    bonf_res = bonferroni(p_vals)
    for idx, a in enumerate(paired_vs_base):
        paired_vs_base[a]["p_bonferroni"] = bonf_res["adjusted_p_values"][idx]

    # Save results.json
    results_data = {
        "metadata": {
            "date": "2026-10-06",
            "track": "Track Z",
            "round": "Z1",
            "device": str(device),
            "peak_vram_mb": peak_vram_mb,
            "test_seeds": test_seeds,
            "dev_seeds": dev_seeds,
        },
        "configs": best_configs,
        "summary": summary,
        "paired_vs_base": paired_vs_base,
        "ngpt_vs_scaled": ngpt_vs_scaled,
        "raw_results": raw_results,
    }

    out_json = Path("research/geomlab_results/z1/results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved raw results to {out_json}")

    # Verify PREREG precedes results.json
    check_prereg("research/geomlab_results/z1/PREREG.md", out_json)
    print("PREREG integrity check passed: PREREG.md strictly precedes results.json.")

    # Step 4: Render Tables
    table1_lines = [
        "Table 1: Primary Comparison across 10 Seeds (mean +- 95% CI)",
        "Arm                 LR       WD    Final Val Loss      Token Speedup    Wall Speedup     ms/step     MQAR Recall",
        "----------------------------------------------------------------------------------------------------------------",
    ]
    for arm in arms:
        cfg = best_configs[arm]
        s = summary[arm]
        loss_str = f"{s['final_loss']['mean']:.4f} +- {s['final_loss']['ci_half']:.4f}"
        tok_str = f"{s['token_speedup']['mean']:.2f}x +- {s['token_speedup']['ci_half']:.2f}"
        wall_str = f"{s['wallclock_speedup']['mean']:.2f}x +- {s['wallclock_speedup']['ci_half']:.2f}"
        step_str = f"{s['ms_per_step']['mean']:.1f} +- {s['ms_per_step']['ci_half']:.1f}"
        mqar_str = f"{s['mqar_recall']['mean'] * 100:.1f}% +- {s['mqar_recall']['ci_half'] * 100:.1f}%"
        lr_str = f"{cfg['lr']:.4f}" if cfg['lr'] < 0.01 else f"{cfg['lr']:.3f}"
        table1_lines.append(
            f"{arm:<18} {lr_str:<8} {cfg['weight_decay']:<5} {loss_str:<18} {tok_str:<16} {wall_str:<16} {step_str:<11} {mqar_str}"
        )

    table2_lines = [
        "\nTable 2: Paired Statistical Comparison vs Base (10 Seeds)",
        "Arm                 Mean Diff (Arm - Base)      95% CI             p-val (paired t)   p-Bonferroni   Cohen's d_z",
        "----------------------------------------------------------------------------------------------------------------",
    ]
    for arm in arms:
        if arm == "base":
            continue
        p = paired_vs_base[arm]
        diff_str = f"{p['mean_diff']:+.4f}"
        ci_str = f"[{p['mean_diff'] - p['ci95_diff']:+.4f}, {p['mean_diff'] + p['ci95_diff']:+.4f}]"
        table2_lines.append(
            f"{arm:<18} {diff_str:<26} {ci_str:<18} {p['p_value_ttest']:<18.4e} {p['p_bonferroni']:<14.4e} {p['cohen_dz']:+.2f}"
        )

    table3_lines = [
        "\nTable 3: Geometry Health & Norm Drift across 10 Seeds (mean +- 95% CI)",
        "Arm                 Stream Norm (Mean+-Std)    Effective Rank      Uniformity Loss    TwoNN Dim          Hubness Skew",
        "-----------------------------------------------------------------------------------------------------------------",
    ]
    for arm in arms:
        s = summary[arm]
        norm_str = f"{s['norm_mean']['mean']:.3f} +- {s['norm_std']['mean']:.4f}"
        erank_str = f"{s['effective_rank']['mean']:.2f} +- {s['effective_rank']['ci_half']:.2f}"
        unif_str = f"{s['uniformity_loss']['mean']:.3f} +- {s['uniformity_loss']['ci_half']:.3f}"
        twonn_str = f"{s['twonn_id']['mean']:.2f} +- {s['twonn_id']['ci_half']:.2f}"
        hub_str = f"{s['hubness_skew']['mean']:.2f} +- {s['hubness_skew']['ci_half']:.2f}"
        table3_lines.append(
            f"{arm:<18} {norm_str:<26} {erank_str:<19} {unif_str:<18} {twonn_str:<18} {hub_str}"
        )

    all_tables_text = "\n".join(table1_lines + table2_lines + table3_lines) + "\n"
    out_tables = Path("research/geomlab_results/z1/tables.txt")
    out_tables.write_text(all_tables_text, encoding="utf-8")
    print(f"Saved rendered tables to {out_tables}")

    print("\n" + all_tables_text)

    # Validate report tables against rendered tables
    validate_report("research/geomlab_results/z1/tables.txt", all_tables_text)
    print("Report validation successful: rendered tables verified.")


if __name__ == "__main__":
    main()
