"""
bench_e4_precision.py - Accurate E4 numerics and precision benchmark.
Evaluates float32, float16, and bfloat16 on trained models at operating point u ~ 1-3
with and without safe_radius clipping, measuring:
1. Max logit error vs float32 reference.
2. Test perplexity.
3. NaN / Inf rate under stress conditions (large radius excursions).
Updates e3_results.json["e4_numerics"]. ASCII output only.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import math
import torch
import torch.nn.functional as F
import numpy as np

from yggformer.config import YggCfg
from yggformer.model import YggFormer
from geometry.attention import safe_radius
from research.yggformer_results.bench_y6_lm_sinks import load_stdlib_corpus, get_batches


def run_e4():
    print("=" * 70)
    print("RUNNING E4 PRECISION & SAFE RADIUS BENCHMARK (AT OPERATING POINT u ~ 1-3)")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = "research/yggformer_results/y6_model_hyp.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = saved["cfg"]

    _, test_text, char2idx, _ = load_stdlib_corpus()
    test_batches = get_batches(test_text, char2idx, seq_len=64, batch_size=8, n_batches=10, seed=42)

    # Base float32 reference model
    ref_model = YggFormer(cfg).to(device)
    ref_model.load_state_dict(saved["model_state"])
    for block in ref_model.blocks:
        block.attn.compute_dtype = torch.float32
        block.attn.r_max = None
    ref_model.eval()

    # Collect reference logits on test batches
    ref_logits_list = []
    with torch.no_grad():
        for x, _ in test_batches:
            x = x.to(device)
            logits = ref_model(x, is_causal=True)
            ref_logits_list.append(logits.float().cpu())

    dtypes = ["float32", "float16", "bfloat16"]
    clip_options = [True, False]
    e4_results = {}

    for dt_str in dtypes:
        dt = getattr(torch, dt_str)
        sr = safe_radius(dt_str)

        for clip in clip_options:
            clip_label = "clipped" if clip else "unclipped"
            r_max = sr if clip else 1000.0
            exp_key = f"{dt_str}_{clip_label}"

            model = YggFormer(cfg).to(device)
            model.load_state_dict(saved["model_state"])
            for block in model.blocks:
                block.attn.compute_dtype = dt
                block.attn.r_max = r_max
            model.eval()

            logit_errors = []
            test_losses = []
            nan_count = 0

            with torch.no_grad():
                for b_idx, (x, y) in enumerate(test_batches):
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x, is_causal=True)

                    if not torch.isfinite(logits).all():
                        nan_count += 1
                        continue

                    # Logit discrepancy vs fp32 reference
                    ref_log = ref_logits_list[b_idx].to(device)
                    diff = (logits.float() - ref_log).abs().max().item()
                    logit_errors.append(diff)

                    loss = F.cross_entropy(logits.view(-1, cfg.vocab), y.view(-1))
                    test_losses.append(loss.item())

            # Stress test: large radius excursion u ~ 8-15
            stress_x = torch.randint(0, cfg.vocab, (4, 64), device=device)
            # Inject high-norm embedding
            with torch.no_grad():
                stress_emb = model.tok_emb(stress_x) * 10.0 # large norm
                stress_logits = model(inputs_embeds=stress_emb, is_causal=True)
                stress_nan = not torch.isfinite(stress_logits).all()

            mean_err = float(np.mean(logit_errors)) if logit_errors else float("nan")
            max_err = float(np.max(logit_errors)) if logit_errors else float("nan")
            mean_loss = float(np.mean(test_losses)) if test_losses else float("nan")
            ppl = math.exp(mean_loss) if not math.isnan(mean_loss) else float("nan")
            nan_rate = float(nan_count / len(test_batches))
            if stress_nan and not clip:
                stress_status = "overflows_on_stress"
            elif stress_nan and clip:
                stress_status = "overflow_detected"
            else:
                stress_status = "stable_under_stress"

            e4_results[exp_key] = {
                "dtype": dt_str,
                "clipped": clip,
                "r_max": r_max,
                "mean_logit_error": mean_err,
                "max_logit_error": max_err,
                "test_ppl": ppl,
                "nan_rate": nan_rate,
                "stress_status": stress_status,
            }

            print(f"  {exp_key:<22} | Logit Err (mean/max): {mean_err:7.4f} / {max_err:7.4f} | PPL: {ppl:5.2f} | Stress: {stress_status}")

            del model
            torch.cuda.empty_cache()

    # Update e3_results.json
    e3_path = "research/yggformer_results/e3_results.json"
    if os.path.exists(e3_path):
        with open(e3_path, "r") as f:
            e3_data = json.load(f)
        e3_data["e4_numerics"] = e4_results
        with open(e3_path, "w") as f:
            json.dump(e3_data, f, indent=2)
        print(f"\nSuccessfully updated e4_numerics in {e3_path}")


if __name__ == "__main__":
    run_e4()
