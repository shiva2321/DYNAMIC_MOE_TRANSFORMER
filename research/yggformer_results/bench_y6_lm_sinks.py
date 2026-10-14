"""
bench_y6_lm_sinks.py - Benchmark runner for Round Y6 (E8/E9 Small LM and Attention Sinks).
Arms:
1. euclid
2. euclid_gate (Qiu et al. 2025 sigmoid gate)
3. hyp
4. hyp_phase
5. hyp_phase_bias
6. hyp_phase_bias_null (null_key=True)

Data: Local Python stdlib text (ast.py, dis.py, inspect.py, tokenize.py; test: heapq.py, calendar.py).
Evaluates: Perplexity (+- 95% CI), first-token sink mass, max activation norm, operating point.
ASCII output only.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, ".")

import json
import time
import math
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from yggformer.config import YggCfg
from yggformer.model import YggFormer
from yggformer.params import match_params, count_parameters
from yggformer.report import operating_point


def load_stdlib_corpus() -> Tuple[str, str, Dict[str, int], Dict[int, str]]:
    """Loads Python stdlib files for training and held-out evaluation."""
    lib_dir = Path(sys.base_prefix) / "Lib"
    train_files = ["ast.py", "dis.py", "inspect.py", "tokenize.py", "difflib.py"]
    test_files = ["heapq.py", "calendar.py"]

    train_text = ""
    for f in train_files:
        p = lib_dir / f
        if p.exists():
            train_text += p.read_text(encoding="utf-8", errors="ignore") + "\n"

    test_text = ""
    for f in test_files:
        p = lib_dir / f
        if p.exists():
            test_text += p.read_text(encoding="utf-8", errors="ignore") + "\n"

    # Filter to printable ASCII
    chars = sorted(list(set(c for c in train_text + test_text if ord(c) < 128)))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}

    return train_text, test_text, char2idx, idx2char


def get_batches(text: str, char2idx: Dict[str, int], seq_len: int, batch_size: int, n_batches: int, seed: int = 42):
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


def measure_sink_and_norm(model: YggFormer, x: torch.Tensor) -> Tuple[float, float]:
    """Measures first-token (or null-key) attention sink mass and maximum activation norm."""
    B, T = x.shape
    sink_masses = []
    max_norm = 0.0

    with torch.no_grad():
        positions = torch.arange(T, device=x.device)
        h = model.tok_emb(x) + model.pos_emb(positions)

        for block in model.blocks:
            norm_in = block.norm1(h)
            max_norm = max(max_norm, norm_in.norm(dim=-1).max().item())

            attn = block.attn
            H = attn.n_heads
            DH = attn.head_dim
            qkv = attn.qkv(norm_in).view(B, T, 3, H, DH).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            if attn.null_k is not None:
                k = torch.cat([attn.null_k.expand(B, -1, 1, -1), k], dim=2)
                v = torch.cat([attn.null_v.expand(B, -1, 1, -1), v], dim=2)
                causal_mask = torch.ones(T, T + 1, dtype=torch.bool, device=x.device)
                causal_mask[:, 1:] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
                causal_mask = causal_mask.view(1, 1, T, T + 1)
                is_null = True
            else:
                causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).view(1, 1, T, T)
                is_null = False

            if attn.geometry in {"euclid", "euclid_gate"}:
                logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(DH)
            elif attn.geometry == "euclid_temp":
                inv_tau = (1.0 / (F.softplus(attn.g_tau) + 1e-3)).view(1, H, 1, 1)
                logits = torch.matmul(q * inv_tau, k.transpose(-1, -2)) / math.sqrt(DH)
            else:
                c = (F.softplus(attn.g_c) + 1e-3).view(1, H, 1, 1)
                tau = (F.softplus(attn.g_tau) + 1e-3).view(1, H, 1, 1)
                q_scaled = q * attn.q_scale
                k_scaled = k * attn.k_scale
                if attn.r_max is not None:
                    from geometry.attention import clip_radius, augment
                    q_scaled = clip_radius(q_scaled, attn.r_max)
                    k_scaled = clip_radius(k_scaled, attn.r_max)
                else:
                    from geometry.attention import augment

                th_q, th_k = None, None
                if attn.phase is not None:
                    ph = attn.phase(norm_in).view(B, T, 2, H, attn.n_phase).permute(2, 0, 3, 1, 4)
                    th_q, th_k = ph[0], ph[1]
                    if attn.null_k is not None:
                        th_k = torch.cat([torch.zeros(B, H, 1, attn.n_phase, device=x.device), th_k], dim=2)
                
                kb = None
                if attn.bias_proj is not None:
                    kb = attn.bias_proj(norm_in).permute(0, 2, 1)
                    if attn.null_k is not None:
                        kb = torch.cat([attn.null_beta.expand(B, -1, 1), kb], dim=2)

                kappa = F.softplus(attn.g_kappa).view(1, H, 1, 1) if attn.phase is not None else 1.0
                q_aug, k_aug, _, _ = augment(q_scaled, k_scaled, k_scaled, c=c, tau=tau, theta_q=th_q, theta_k=th_k, kappa=kappa, key_bias=kb)
                logits = torch.matmul(q_aug, k_aug.transpose(-1, -2))

            logits = logits.masked_fill(~causal_mask, float("-inf"))
            weights = F.softmax(logits, dim=-1)

            if is_null:
                sink = weights[:, :, :, 0].mean().item()
            else:
                sink = weights[:, :, 1:, 0].mean().item()
            sink_masses.append(sink)

            h = block(h)

        max_norm = max(max_norm, model.final_norm(h).norm(dim=-1).max().item())

    mean_sink = float(np.mean(sink_masses)) if sink_masses else 0.0
    return mean_sink, max_norm



def train_and_eval_lm(
    arm: str,
    lr: float,
    seed: int,
    device: torch.device,
    vocab_size: int,
    train_batches: List,
    test_batches: List,
    d_model: int = 64,
    head_dim: int = 16,
    n_layers: int = 2,
    seq_len: int = 64,
) -> Tuple[float, float, float, Dict, float]:
    """Returns (perplexity, sink_mass, max_norm, op_report, peak_mem_mb)."""
    torch.manual_seed(seed)
    n_heads = d_model // head_dim

    is_null = (arm == "hyp_phase_bias_null")
    base_geom = "hyp_phase_bias" if is_null else arm

    ref_cfg = YggCfg(
        vocab=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        head_geometry="euclid",
        ffn_mult=2.0,
        max_len=seq_len + 16,
        seed=seed,
    )

    target_cfg = YggCfg(
        vocab=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        head_geometry=base_geom,
        ffn_mult=2.0,
        max_len=seq_len + 16,
        null_key=is_null,
        seed=seed,
    )

    matched_cfg = match_params(ref_cfg, target_cfg)
    model = YggFormer(matched_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    torch.cuda.reset_peak_memory_stats(device)
    model.train()

    # Training
    for x, y in train_batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, is_causal=True)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Evaluation on held-out test
    model.eval()
    test_losses = []
    with torch.no_grad():
        for x, y in test_batches:
            x, y = x.to(device), y.to(device)
            logits = model(x, is_causal=True)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            test_losses.append(loss.item())

    mean_loss = float(np.mean(test_losses))
    perplexity = math.exp(mean_loss)

    # Sink mass and activation norm on sample test batch
    sample_x = test_batches[0][0].to(device)
    sink_mass, max_norm = measure_sink_and_norm(model, sample_x)

    # Operating point (computed for all arms)
    op_rep = operating_point(model, sample_x)

    if seed == 0:
        os.makedirs("research/yggformer_results", exist_ok=True)
        torch.save({"model_state": model.state_dict(), "cfg": matched_cfg}, f"research/yggformer_results/y6_model_{arm}.pt")

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    del model, optimizer
    torch.cuda.empty_cache()

    return perplexity, sink_mass, max_norm, op_rep, peak_mem_mb


def main():
    print("=" * 70)
    print("STARTING ROUND Y6: E8/E9 SMALL LM & ATTENTION SINKS")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_text, test_text, char2idx, idx2char = load_stdlib_corpus()
    vocab_size = len(char2idx)
    print(f"Loaded stdlib corpus: Train {len(train_text)} chars | Test {len(test_text)} chars | Vocab {vocab_size} chars")

    seq_len = 64
    batch_size = 32
    n_train_steps = 1500
    n_tune_steps = 300
    n_test_steps = 30

    train_batches = get_batches(train_text, char2idx, seq_len=seq_len, batch_size=batch_size, n_batches=n_train_steps, seed=100)
    test_batches = get_batches(test_text, char2idx, seq_len=seq_len, batch_size=batch_size, n_batches=n_test_steps, seed=200)

    arms = ["euclid", "euclid_gate", "euclid_temp", "hyp", "hyp_phase", "hyp_phase_bias", "hyp_phase_bias_null"]
    lr_grid = [3e-3, 1e-2, 3e-2, 1e-1]

    # Tune LR on dev seed 100
    print("\n--- TUNING LR ON DEV SEED 100 ---", flush=True)
    best_lrs = {}
    for arm in arms:
        best_lr = lr_grid[0]
        best_ppl = float("inf")
        for lr in lr_grid:
            ppl, _, _, _, _ = train_and_eval_lm(arm, lr=lr, seed=100, device=device, vocab_size=vocab_size, train_batches=train_batches[:n_tune_steps], test_batches=test_batches)
            print(f"  Arm {arm:<20} LR {lr:1.0e} -> PPL {ppl:.2f}", flush=True)
            if ppl < best_ppl:
                best_ppl = ppl
                best_lr = lr
        best_lrs[arm] = best_lr
        print(f"  >> Chosen LR for {arm}: {best_lr:1.0e} (Dev PPL: {best_ppl:.2f})", flush=True)

    # Run 10 test seeds (0-9)
    print("\n--- RUNNING 10 TEST SEEDS (0-9) ---", flush=True)
    test_seeds = list(range(10))
    results = {"metadata": {"arms": arms, "best_lrs": best_lrs, "vocab_size": vocab_size, "n_train_steps": n_train_steps}, "arms": {}}

    t0 = time.time()
    for arm in arms:
        lr = best_lrs[arm]
        ppls, sinks, norms, peaks = [], [], [], []
        op_reps = []

        print(f"Running 10 test seeds for {arm}...", end="", flush=True)
        for seed in test_seeds:
            tr_b = get_batches(train_text, char2idx, seq_len=seq_len, batch_size=batch_size, n_batches=n_train_steps, seed=seed * 10)
            te_b = get_batches(test_text, char2idx, seq_len=seq_len, batch_size=batch_size, n_batches=n_test_steps, seed=seed * 10 + 5)
            ppl, sink, norm, op_rep, peak_mb = train_and_eval_lm(arm, lr=lr, seed=seed, device=device, vocab_size=vocab_size, train_batches=tr_b, test_batches=te_b)
            ppls.append(ppl)
            sinks.append(sink)
            norms.append(norm)
            peaks.append(peak_mb)
            op_reps.append(op_rep)
            print(".", end="", flush=True)
        print(" done.", flush=True)

        ppl_mean = float(np.mean(ppls))
        ppl_std = float(np.std(ppls, ddof=1))
        ppl_ci = 2.262 * (ppl_std / math.sqrt(len(ppls)))

        sink_mean = float(np.mean(sinks))
        norm_mean = float(np.mean(norms))
        peak_mean = float(np.mean(peaks))

        results["arms"][arm] = {
            "ppls": ppls,
            "ppl_mean": ppl_mean,
            "ppl_ci": ppl_ci,
            "sink_mass_mean": sink_mean,
            "max_norm_mean": norm_mean,
            "peak_mb": peak_mean,
            "operating_point": op_reps[0],
        }

        print(f"  Arm {arm:<20} | PPL: {ppl_mean:5.2f} +- {ppl_ci:4.2f} | Sink Mass: {sink_mean*100:4.1f}% | Max Norm: {norm_mean:5.2f} | Peak: {peak_mean:4.1f}MB", flush=True)

    # Statistical significance vs euclid
    import scipy.stats as sp_stats
    euclid_ppls = results["arms"]["euclid"]["ppls"]
    n_comparisons = len(arms) - 1
    bonferroni_alpha = 0.05 / n_comparisons

    print("\n--- STATISTICAL SIGNIFICANCE VS EUCLID (10 SEEDS) ---", flush=True)
    for arm in arms:
        if arm == "euclid":
            continue
        arm_ppls = results["arms"][arm]["ppls"]
        res = sp_stats.ttest_rel(arm_ppls, euclid_ppls)
        pval = float(res.pvalue)
        diff_mean = float(np.mean(arm_ppls) - np.mean(euclid_ppls))
        sig_bonferroni = pval < bonferroni_alpha
        results["arms"][arm]["p_val_vs_euclid"] = pval
        results["arms"][arm]["diff_vs_euclid"] = diff_mean
        results["arms"][arm]["sig_bonferroni"] = sig_bonferroni
        print(f"  {arm:<20} | Diff: {diff_mean:+5.2f} | p-val: {pval:.4e} | Bonferroni sig (alpha={bonferroni_alpha:.4f}): {sig_bonferroni}", flush=True)

    elapsed = time.time() - t0
    results["metadata"]["wall_clock_s"] = elapsed
    print(f"\nE8/E9 LM benchmark completed in {elapsed:.1f}s", flush=True)

    os.makedirs("research/yggformer_results", exist_ok=True)
    with open("research/yggformer_results/e6_lm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to research/yggformer_results/e6_lm_results.json")


if __name__ == "__main__":
    main()
