"""
10-Seed Evaluation of UnifiedNoveltyDetector on Real UTF-8 Data Streams.

Evaluates two real-data regimes across 10 seeds:
Regime A (Structured Real English Operational Stream):
  1. In-Distribution Real Text (English operational statements with punctuation jitter -> FPR)
  2. Coarse Real Domain Shift (real Python source code byte windows from dmoe/*.py)
  3. Near-Twin Real Semantic Flip (identical English vocabulary & syntax, inverted policy outcome)
  4. Local 4-Byte Nuance in Real Text (28/32 bytes familiar English, 4-byte 'NOT!' span flip)

Regime B (Open-Text Real Document Corpus Split from research_base/**/*.md):
  - Strictly disjoint line split: 70% Train / 15% Calibrate / 15% Unseen Held-Out Test
  - Self-supervised next-byte prediction (x = line[:32], y = line[1:33])
  1. Unseen Held-Out Markdown Prose (brand-new sentences -> Open-Text FPR)
  2. Real Python Code Shift (dmoe/*.py lines)
  3. Near-Twin Continuation Shift (real English prose input with shifted byte continuation)
  4. Local 4-Byte Span Anomaly in Unseen Prose (28/32 bytes unseen English prose, 4-byte non-ASCII/code spike)
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from dmoe.novelty import UnifiedNoveltyDetector
from dmoe.real_model import RealSelfStabilizingDMoE

SEQ_LEN = 32

REAL_ENGLISH_CLAUSES: List[Tuple[str, str]] = [
    ("service health is normal today: ", "allow incoming client traffic.  "),
    ("database replica sync is ready: ", "route read queries to replica.  "),
    ("authentication token is valid:  ", "grant access to user session.   "),
    ("memory usage is below limit:    ", "keep worker pool size stable.   "),
    ("network latency is very low:    ", "enable fast streaming channel.  "),
    ("tls certificate is verified:    ", "accept secure handshake now.    "),
    ("disk capacity is sufficient:    ", "continue writing audit logs.    "),
    ("cache hit ratio is optimal:     ", "serve responses from memory.    "),
    ("queue depth is within bounds:   ", "process jobs at normal rate.    "),
    ("checksum validation succeeded:  ", "commit payload to storage.      "),
    ("api rate limit is respected:    ", "forward request to upstream.    "),
    ("cluster quorum is established:  ", "elect primary leader node.      "),
]

NEAR_TWIN_FLIP_CLAUSES: List[Tuple[str, str]] = [
    ("service health is normal today: ", "grant access to user session.   "),
    ("database replica sync is ready: ", "commit payload to storage.      "),
    ("authentication token is valid:  ", "allow incoming client traffic.  "),
    ("memory usage is below limit:    ", "elect primary leader node.      "),
    ("network latency is very low:    ", "continue writing audit logs.    "),
    ("tls certificate is verified:    ", "serve responses from memory.    "),
    ("disk capacity is sufficient:    ", "route read queries to replica.  "),
    ("cache hit ratio is optimal:     ", "keep worker pool size stable.   "),
    ("queue depth is within bounds:   ", "accept secure handshake now.    "),
    ("checksum validation succeeded:  ", "forward request to upstream.    "),
    ("api rate limit is respected:    ", "enable fast streaming channel.  "),
    ("cluster quorum is established:  ", "process jobs at normal rate.    "),
]


def str_to_bytes_tensor(s: str, length: int = SEQ_LEN) -> torch.Tensor:
    raw = s.encode("utf-8", errors="ignore")[:length]
    if len(raw) < length:
        raw = raw + b" " * (length - len(raw))
    return torch.tensor(list(raw), dtype=torch.long)


def load_real_code_windows(root_dir: str, length: int = SEQ_LEN) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Load real Python code byte windows from dmoe/*.py."""
    windows: List[Tuple[torch.Tensor, torch.Tensor]] = []
    py_files = sorted(glob.glob(os.path.join(root_dir, "dmoe", "*.py")))
    for path in py_files:
        with open(path, "rb") as f:
            data = f.read()
        for line in data.splitlines():
            s = line.strip()
            if len(s) > length and any(tok in s for tok in (b"def ", b"self.", b"torch.", b"return ", b"==")):
                x = torch.tensor(list(s[:length]), dtype=torch.long)
                y = torch.tensor(list(s[1 : length + 1]), dtype=torch.long)
                windows.append((x, y))
    return windows


def load_real_prose_windows(root_dir: str, length: int = SEQ_LEN) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Load real English prose lines from research_base/**/*.md (excluding code/table lines)."""
    windows: List[Tuple[torch.Tensor, torch.Tensor]] = []
    md_files = sorted(glob.glob(os.path.join(root_dir, "research_base", "**", "*.md"), recursive=True))
    for path in md_files:
        with open(path, "rb") as f:
            data = f.read()
        for line in data.splitlines():
            s = line.strip()
            # Filter for natural English prose lines (alphabetic start, no markdown table/code prefixes)
            if len(s) > length + 2 and s[0:1].isalpha() and not any(c in s for c in (b"|", b"`", b"_", b"/", b"=")):
                # Keep clean ASCII range
                if all(32 <= b < 127 for b in s[: length + 1]):
                    x = torch.tensor(list(s[:length]), dtype=torch.long)
                    y = torch.tensor(list(s[1 : length + 1]), dtype=torch.long)
                    windows.append((x, y))
    return windows


def sample_english_batch(
    clauses: List[Tuple[str, str]],
    batch_size: int,
    seed: int,
    device: torch.device,
    held_out_variation: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(clauses), (batch_size,), generator=g)
    xs, ys = [], []
    for i in range(batch_size):
        c_idx = int(idx[i].item())
        inp_s, out_s = clauses[c_idx]
        x_t = str_to_bytes_tensor(inp_s, SEQ_LEN)
        y_t = str_to_bytes_tensor(out_s, SEQ_LEN)
        if held_out_variation and i % 2 == 0:
            x_t[-1] = ord(".")
        xs.append(x_t)
        ys.append(y_t)
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def sample_window_batch(
    windows: List[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(windows), (batch_size,), generator=g)
    xs = [windows[int(i.item())][0].clone() for i in idx]
    ys = [windows[int(i.item())][1].clone() for i in idx]
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def sample_local_nuance_batch(
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xb, yb = sample_english_batch(REAL_ENGLISH_CLAUSES, batch_size, seed, device, held_out_variation=False)
    nuance_span = torch.tensor([ord("N"), ord("O"), ord("T"), ord("!")], dtype=torch.long, device=device)
    xb[:, 18:22] = nuance_span.unsqueeze(0).expand(batch_size, 4)
    yb[:, 0:5] = torch.tensor([ord("d"), ord("e"), ord("n"), ord("y"), ord(" ")], dtype=torch.long, device=device)
    return xb, yb


def run_structured_seed(
    seed: int,
    train_steps: int,
    code_windows: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    torch.manual_seed(seed)
    model = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=2,
        d_ff=64,
        initial_layers=1,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=2,
        auto_lifecycle=False,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3)

    model.train()
    for st in range(train_steps):
        xb, yb = sample_english_batch(REAL_ENGLISH_CLAUSES, 24, seed=seed * 1000 + st, device=device)
        opt.zero_grad()
        out = model(xb, force_thinking_steps=2)
        loss = model.thinker.compute_thinking_loss(out["thinking_out"], yb)["loss"]
        loss.backward()
        opt.step()

    model.eval()
    det = UnifiedNoveltyDetector(d_model=32, tail_fraction=0.15)
    with torch.no_grad():
        for st in range(20):
            xb, yb = sample_english_batch(
                REAL_ENGLISH_CLAUSES, 16, seed=seed * 2000 + st, device=device, held_out_variation=(st % 3 == 0)
            )
            out = model(xb, force_thinking_steps=2)
            seq_sig = F.normalize(model.encode(xb).mean(dim=1), p=2, dim=-1)
            tok_surp = out["surprise_trace"][-1]
            tok_nll = F.cross_entropy(
                out["output"].reshape(-1, 256), yb.reshape(-1), reduction="none"
            ).view(xb.size(0), xb.size(1))
            ex = model.layers[0].moe.experts[0].exemplars
            det.evaluate(
                expert_idx=0,
                seq_sig=seq_sig,
                expert_centroid=ex if ex.numel() > 0 else None,
                token_latents=out["anchor_latent"],
                token_surprise=tok_surp,
                token_nll=tok_nll,
                expert_is_converged=True,
                update_baseline=True,
            )

    streams = ("in_dist_real", "coarse_real_code", "near_twin_real", "local_nuance_real")
    old_hits = {s: 0 for s in streams}
    new_hits = {s: 0 for s in streams}
    n_eval = 10

    with torch.no_grad():
        ex = model.layers[0].moe.experts[0].exemplars
        cent = F.normalize(ex.mean(dim=0, keepdim=True).float(), p=2, dim=-1).to(device)

        for st in range(n_eval):
            for s_name in streams:
                if s_name == "in_dist_real":
                    xb, yb = sample_english_batch(
                        REAL_ENGLISH_CLAUSES, 16, seed=seed * 3000 + st, device=device, held_out_variation=True
                    )
                elif s_name == "coarse_real_code":
                    xb, yb = sample_window_batch(code_windows, 16, seed=seed * 4000 + st, device=device)
                elif s_name == "near_twin_real":
                    xb, yb = sample_english_batch(NEAR_TWIN_FLIP_CLAUSES, 16, seed=seed * 5000 + st, device=device)
                else:
                    xb, yb = sample_local_nuance_batch(16, seed=seed * 6000 + st, device=device)

                out = model(xb, force_thinking_steps=2)
                seq_sig = F.normalize(model.encode(xb).mean(dim=1), p=2, dim=-1)
                tok_surp = out["surprise_trace"][-1]
                tok_nll = F.cross_entropy(
                    out["output"].reshape(-1, 256), yb.reshape(-1), reduction="none"
                ).view(xb.size(0), xb.size(1))

                seq_dist = float((1.0 - (seq_sig.float() @ cent.T).squeeze(-1)).mean().item())
                mean_surp = float(tok_surp.mean().item())
                if seq_dist >= 0.45 or mean_surp >= 0.42:
                    old_hits[s_name] += 1

                verdict = det.evaluate(
                    expert_idx=0,
                    seq_sig=seq_sig,
                    expert_centroid=ex,
                    token_latents=out["anchor_latent"],
                    token_surprise=tok_surp,
                    token_nll=tok_nll,
                    expert_is_converged=True,
                    update_baseline=False,
                )
                if verdict.is_novel:
                    new_hits[s_name] += 1

    return {
        "Old_MeanOnly": {s: old_hits[s] / n_eval for s in streams},
        "New_Unified": {s: new_hits[s] / n_eval for s in streams},
    }


def run_open_corpus_seed(
    seed: int,
    train_steps: int,
    prose_windows: List[Tuple[torch.Tensor, torch.Tensor]],
    code_windows: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate on a strictly disjoint 70/15/15 split of real Markdown prose lines from research_base/.
    """
    torch.manual_seed(seed)
    g_split = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(prose_windows), generator=g_split).tolist()
    shuffled_prose = [prose_windows[i] for i in perm]
    n_total = len(shuffled_prose)
    n_train = int(0.70 * n_total)
    n_cal = int(0.85 * n_total)
    train_win = shuffled_prose[:n_train]
    cal_win = shuffled_prose[n_train:n_cal]
    test_win = shuffled_prose[n_cal:]

    model = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=2,
        d_ff=64,
        initial_layers=1,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=2,
        auto_lifecycle=False,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3)

    model.train()
    for st in range(train_steps):
        xb, yb = sample_window_batch(train_win, 24, seed=seed * 1000 + st, device=device)
        opt.zero_grad()
        out = model(xb, force_thinking_steps=2)
        loss = model.thinker.compute_thinking_loss(out["thinking_out"], yb)["loss"]
        loss.backward()
        opt.step()

    model.eval()
    det = UnifiedNoveltyDetector(d_model=32, tail_fraction=0.15)
    with torch.no_grad():
        for st in range(20):
            xb, yb = sample_window_batch(cal_win, 16, seed=seed * 2000 + st, device=device)
            out = model(xb, force_thinking_steps=2)
            seq_sig = F.normalize(model.encode(xb).mean(dim=1), p=2, dim=-1)
            tok_surp = out["surprise_trace"][-1]
            tok_nll = F.cross_entropy(
                out["output"].reshape(-1, 256), yb.reshape(-1), reduction="none"
            ).view(xb.size(0), xb.size(1))
            ex = model.layers[0].moe.experts[0].exemplars
            det.evaluate(
                expert_idx=0,
                seq_sig=seq_sig,
                expert_centroid=ex if ex.numel() > 0 else None,
                token_latents=out["anchor_latent"],
                token_surprise=tok_surp,
                token_nll=tok_nll,
                expert_is_converged=True,
                update_baseline=True,
            )

    streams = ("in_dist_real", "coarse_real_code", "near_twin_real", "local_nuance_real", "local_nuance_ascii")
    old_hits = {s: 0 for s in streams}
    new_hits = {s: 0 for s in streams}
    n_eval = 10

    with torch.no_grad():
        ex = model.layers[0].moe.experts[0].exemplars
        cent = F.normalize(ex.mean(dim=0, keepdim=True).float(), p=2, dim=-1).to(device)

        for st in range(n_eval):
            for s_name in streams:
                if s_name == "in_dist_real":
                    xb, yb = sample_window_batch(test_win, 16, seed=seed * 3000 + st, device=device)
                elif s_name == "coarse_real_code":
                    xb, yb = sample_window_batch(code_windows, 16, seed=seed * 4000 + st, device=device)
                elif s_name == "near_twin_real":
                    # Same unseen English prose input, but continuation rule shifted (ROT-13 / cipher flip)
                    xb, yb = sample_window_batch(test_win, 16, seed=seed * 5000 + st, device=device)
                    yb = ((yb - 32 + 13) % 95) + 32
                elif s_name == "local_nuance_real":
                    # Unseen English prose with a 4-byte localized binary/symbol anomaly at [20:24]
                    xb, yb = sample_window_batch(test_win, 16, seed=seed * 6000 + st, device=device)
                    xb[:, 20:24] = torch.tensor([180, 181, 182, 183], dtype=torch.long, device=device)
                    yb[:, 19:23] = torch.tensor([180, 181, 182, 183], dtype=torch.long, device=device)
                else:
                    # Unseen English prose with a 4-byte printable-ASCII contradiction ('NOT!') at [20:24]
                    xb, yb = sample_window_batch(test_win, 16, seed=seed * 7000 + st, device=device)
                    ascii_flip = torch.tensor([ord("N"), ord("O"), ord("T"), ord("!")], dtype=torch.long, device=device)
                    xb[:, 20:24] = ascii_flip
                    yb[:, 19:23] = torch.tensor([ord("Z"), ord("Q"), ord("X"), ord("!")], dtype=torch.long, device=device)

                out = model(xb, force_thinking_steps=2)
                seq_sig = F.normalize(model.encode(xb).mean(dim=1), p=2, dim=-1)
                tok_surp = out["surprise_trace"][-1]
                tok_nll = F.cross_entropy(
                    out["output"].reshape(-1, 256), yb.reshape(-1), reduction="none"
                ).view(xb.size(0), xb.size(1))

                seq_dist = float((1.0 - (seq_sig.float() @ cent.T).squeeze(-1)).mean().item())
                mean_surp = float(tok_surp.mean().item())
                if seq_dist >= 0.45 or mean_surp >= 0.42:
                    old_hits[s_name] += 1

                verdict = det.evaluate(
                    expert_idx=0,
                    seq_sig=seq_sig,
                    expert_centroid=ex,
                    token_latents=out["anchor_latent"],
                    token_surprise=tok_surp,
                    token_nll=tok_nll,
                    expert_is_converged=True,
                    update_baseline=False,
                )
                if verdict.is_novel:
                    new_hits[s_name] += 1

    return {
        "Old_MeanOnly": {s: old_hits[s] / n_eval for s in streams},
        "New_Unified": {s: new_hits[s] / n_eval for s in streams},
    }


def summarize_runs(runs: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary = {}
    keys = list(runs[0]["New_Unified"].keys())
    for method in ("Old_MeanOnly", "New_Unified"):
        summary[method] = {}
        for stream in keys:
            vals = [r[method][stream] for r in runs]
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
            summary[method][stream] = {"mean": m, "std": sd}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--out", type=str, default="results/dmoe_v4_unified_novelty/real_stream_results.json")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code_windows = load_real_code_windows(root_dir, SEQ_LEN)
    prose_windows = load_real_prose_windows(root_dir, SEQ_LEN)
    print(f"Loaded {len(prose_windows)} real Markdown prose windows and {len(code_windows)} real Python code windows.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    structured_seeds: List[Dict[str, Dict[str, float]]] = []
    open_corpus_seeds: List[Dict[str, Dict[str, float]]] = []

    for s in range(args.seeds):
        res_a = run_structured_seed(seed=200 + s, train_steps=args.train_steps, code_windows=code_windows, device=device)
        res_b = run_open_corpus_seed(
            seed=300 + s,
            train_steps=args.train_steps,
            prose_windows=prose_windows,
            code_windows=code_windows,
            device=device,
        )
        structured_seeds.append(res_a)
        open_corpus_seeds.append(res_b)
        na = res_a["New_Unified"]
        nb = res_b["New_Unified"]
        print(
            f"[Seed {s+1:02d}/{args.seeds:02d}] "
            f"Structured_New(FPR={na['in_dist_real']:.2f}, Code={na['coarse_real_code']:.2f}, Twin={na['near_twin_real']:.2f}, Nuance={na['local_nuance_real']:.2f}) | "
            f"OpenCorpus_New(FPR={nb['in_dist_real']:.2f}, Code={nb['coarse_real_code']:.2f}, Twin={nb['near_twin_real']:.2f}, Nuance={nb['local_nuance_real']:.2f})",
            flush=True,
        )

    sum_a = summarize_runs(structured_seeds)
    sum_b = summarize_runs(open_corpus_seeds)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "num_prose_windows": len(prose_windows),
                "num_code_windows": len(code_windows),
                "structured_summary": sum_a,
                "open_corpus_summary": sum_b,
            },
            f,
            indent=2,
        )

    print("\n=== REGIME A: STRUCTURED REAL ENGLISH OPERATIONAL STREAM (10 SEEDS) ===")
    for method, stats in sum_a.items():
        print(
            f"{method:14s} | FPR(RealInDist)={stats['in_dist_real']['mean']:.3f}+/-{stats['in_dist_real']['std']:.3f} "
            f"| RealCode={stats['coarse_real_code']['mean']:.3f} "
            f"| NearTwin={stats['near_twin_real']['mean']:.3f} "
            f"| LocalNuance={stats['local_nuance_real']['mean']:.3f}+/-{stats['local_nuance_real']['std']:.3f}"
        )

    print("\n=== REGIME B: OPEN-TEXT REAL MARKDOWN CORPUS 70/15/15 SPLIT (10 SEEDS) ===")
    for method, stats in sum_b.items():
        print(
            f"{method:14s} | FPR(UnseenProse)={stats['in_dist_real']['mean']:.3f}+/-{stats['in_dist_real']['std']:.3f} "
            f"| RealCode={stats['coarse_real_code']['mean']:.3f} "
            f"| NearTwin={stats['near_twin_real']['mean']:.3f} "
            f"| LocalNuance(Byte)={stats['local_nuance_real']['mean']:.3f} "
            f"| LocalNuance(ASCII)={stats['local_nuance_ascii']['mean']:.3f}+/-{stats['local_nuance_ascii']['std']:.3f}"
        )


if __name__ == "__main__":
    main()
